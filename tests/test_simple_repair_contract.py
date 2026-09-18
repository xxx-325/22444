"""Regression coverage for the bounded simple-candidate repair path."""

import copy
import json
import unittest
from unittest.mock import patch

from dialogue_benchmark import cli
from dialogue_benchmark.llm import (
    ModelStageError,
    parse_text_response,
    repair_simple_candidate,
    review_candidates,
)
from tests.simple_test_helpers import maybe_relevance_response


class ScriptedClient:
    """A transport fake that records the exact request sent at each stage."""

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.usage = []

    def ask(self, prompt, payload):
        self.calls.append((prompt, copy.deepcopy(payload)))
        self.usage.append({"status": "completed"})
        relevance = maybe_relevance_response(prompt, payload)
        if relevance is not None:
            return relevance
        response = next(self.responses)
        if isinstance(response, BaseException):
            self.usage[-1]["status"] = "failed"
            raise response
        if callable(response):
            response = response(prompt, payload)
        return parse_text_response(response) if isinstance(response, str) else copy.deepcopy(response)


def atomicity_review(assignments):
    return """REVIEW q1
review_contract: simple_atomicity_v1
point_atomicity: %s
END_REVIEW""" % assignments


def completeness_review():
    return """REVIEW q1
review_contract: simple_v1
completeness: complete
missing: none
END_REVIEW"""


def evidence_review(assignments):
    return """REVIEW q1
review_contract: simple_v1
point_evidence: %s
END_REVIEW""" % assignments


class SimpleRepairContractTests(unittest.TestCase):
    def setUp(self):
        self.scope = {
            "cutoff": 3,
            "dialogue": [
                {"id": "m1", "kind": "message", "role": "assistant", "order": 1,
                 "stage_id": "s1", "text": "端口由 18080 改为 18081。"},
                {"id": "guard", "kind": "message", "role": "assistant", "order": 2,
                 "stage_id": "s1", "text": "guard-only review material."},
                {"id": "unrelated", "kind": "message", "role": "assistant", "order": 3,
                 "stage_id": "s2", "text": "unrelated full-session material."},
            ],
            "events": [], "versions": [], "stages": [{"id": "s1"}, {"id": "s2"}],
            "review_guard_sources": ["guard"],
            "review_guard_complete": True,
            "model_request_chars": 60000,
            "max_context_chars": 60000,
        }
        self.facts = [{
            "id": "f1", "qa_mode": "general",
            "statement": "端口由 18080 改为 18081。", "sources": ["m1"],
        }]
        self.candidate = {
            "id": "general-g1-q1", "candidate_id": "general-g1-q1",
            "model_id": "provider-q1", "qa_mode": "general", "type": "single-hop",
            "origin_qa_mode": "general", "evidence_group_id": "g1",
            "question": "端口如何变化？",
            "answer_points": [{"text": "端口由 18080 改为 18081。", "sources": ["m1"]}],
            "forbidden_points": [{
                "text": "端口保持 18080，并且旧配置继续使用。", "sources": ["m1"],
            }],
        }
        # This is the exact QA request projection, not a full scope or review guard.
        self.original_prompt = "ORIGINAL QA INSTRUCTION"
        self.original_payload = {
            "facts": [{"text": "端口由 18080 改为 18081。", "materials": ["资料1"]}],
            "materials": [{
                "reference": "资料1", "source_kind": "code",
                "text": "config.py::load_config\nreturn endpoint 18081",
            }],
            "relations": ["资料1 is the selected material"],
            "focus": {"text": "端口配置决定", "sources": ["资料1"]},
        }
        self.generation_context = {
            "prompt": self.original_prompt,
            "payload": copy.deepcopy(self.original_payload),
            "ref_to_source": {"资料1": "m1"},
        }

    def _run_atomic_repair(self, repair_response, post_atomicity,
                           candidate=None, context=None):
        client = ScriptedClient([
            atomicity_review("A1=single;F1=compound"),
            repair_response,
            atomicity_review(post_atomicity),
            completeness_review(),
            evidence_review("A1=supported@资料1;F1=contradicted@资料1;F2=contradicted@资料1"),
        ])
        result = review_candidates(
            self.scope, self.facts, [candidate or self.candidate], client,
            qa_mode="general", review_mode="simple", allow_repair=True,
            generation_context=context or self.generation_context)
        return result, client

    def test_only_forbidden_compound_is_repaired_and_fully_re_reviewed(self):
        repair_response = """QA q1
QUESTION: 端口如何变化？
ANSWER_POINT: 端口由 18080 改为 18081。 || SOURCES: 资料1
FORBIDDEN_POINT: 端口保持 18080。 || SOURCES: 资料1
FORBIDDEN_POINT: 旧配置继续使用。 || SOURCES: 资料1
END_QA"""
        result, client = self._run_atomic_repair(
            repair_response, "A1=single;F1=single;F2=single")

        self.assertEqual(result["questions"][0]["status"], "approved")
        self.assertEqual([receipt["stage"] for receipt in client.usage], [
            "review_relevance", "review_atomicity", "repair",
            "review_relevance", "review_atomicity", "review_completeness",
            "review_evidence",
        ])
        self.assertEqual(len(result.get("revisions", [])), 1)
        revision = result["revisions"][0]
        self.assertEqual(revision["before"]["id"], self.candidate["id"])
        self.assertEqual(revision["original_review"]["reason"],
                         "semantic_answer_atomicity_failed")
        self.assertEqual(revision["after"]["id"], self.candidate["id"])
        self.assertEqual(revision["after"]["type"], self.candidate["type"])
        self.assertEqual(len(revision["after"]["forbidden_points"]), 2)
        self.assertEqual(revision["review_result"]["questions"][0]["status"],
                         "approved")

        repair_prompt, repair_payload = client.calls[2]
        self.assertTrue(repair_prompt.startswith(self.original_prompt))
        self.assertIn("Correct exactly the supplied review_issue once", repair_prompt)
        for key, value in self.original_payload.items():
            self.assertEqual(repair_payload[key], value)
        self.assertEqual(repair_payload["original_candidate"]["question"],
                         self.candidate["question"])
        self.assertEqual(repair_payload["original_candidate"]["forbidden_points"][0][
            "review_id"], "F1")
        self.assertEqual(repair_payload["original_candidate"]["forbidden_points"][0][
            "sources"], ["资料1"])
        self.assertIn("F1", repair_payload["review_issue"])
        payload_text = json.dumps(repair_payload, ensure_ascii=False)
        self.assertNotIn("guard-only review material", payload_text)
        self.assertNotIn("unrelated full-session material", payload_text)
        self.assertNotIn("m1", payload_text)

    def test_answer_compound_is_repaired_without_changing_id_or_type(self):
        candidate = copy.deepcopy(self.candidate)
        candidate["answer_points"] = [{
            "text": "端口由 18080 改为 18081，并且旧配置继续使用。",
            "sources": ["m1"],
        }]
        candidate["forbidden_points"] = []
        repair_response = """QA q1
QUESTION: 端口如何变化？
ANSWER_POINT: 端口由 18080 改为 18081。 || SOURCES: 资料1
ANSWER_POINT: 旧配置继续使用。 || SOURCES: 资料1
END_QA"""
        # The repaired question is deliberately still one target, and the fake
        # evidence review covers both retained atomic answer points.
        client = ScriptedClient([
            atomicity_review("A1=compound"), repair_response,
            atomicity_review("A1=single;A2=single"), completeness_review(),
            evidence_review("A1=supported@资料1;A2=supported@资料1"),
        ])
        result = review_candidates(
            self.scope, self.facts, [candidate], client,
            qa_mode="general", review_mode="simple", allow_repair=True,
            generation_context=self.generation_context)

        self.assertEqual(result["questions"][0]["status"], "approved")
        revised = result["questions"][0]
        self.assertEqual(revised["id"], candidate["id"])
        self.assertEqual(revised["type"], candidate["type"])
        self.assertEqual(revised["candidate_id"], candidate["candidate_id"])
        self.assertEqual([item["stage"] for item in client.usage], [
            "review_relevance", "review_atomicity", "repair",
            "review_relevance", "review_atomicity", "review_completeness",
            "review_evidence",
        ])

    def test_atomic_repair_must_keep_question_and_skips_re_review_when_changed(self):
        changed_question = "请解释为什么端口改动一定修复了业务问题？"
        response = """QA q1
QUESTION: %s
ANSWER_POINT: 端口由 18080 改为 18081。 || SOURCES: 资料1
FORBIDDEN_POINT: 端口保持 18080。 || SOURCES: 资料1
FORBIDDEN_POINT: 旧配置继续使用。 || SOURCES: 资料1
END_QA""" % changed_question
        client = ScriptedClient([
            atomicity_review("A1=single;F1=compound"), response,
        ])
        result = review_candidates(
            self.scope, self.facts, [self.candidate], client,
            qa_mode="general", review_mode="simple", allow_repair=True,
            generation_context=self.generation_context)

        self.assertEqual([receipt["stage"] for receipt in client.usage], [
            "review_relevance", "review_atomicity", "repair",
        ])
        self.assertEqual(len(result.get("revisions", [])), 1)
        revision = result["revisions"][0]
        self.assertEqual(revision["before"]["question"], self.candidate["question"])
        self.assertTrue(any(
            item.get("reason") == "repair_changed_question"
            for item in revision.get("invalid", [])
            if isinstance(item, dict)))
        self.assertNotIn("after", revision)
        self.assertFalse(any(
            item.get("status") == "approved" for item in result.get("questions", [])))

    def test_repair_budget_is_shared_and_no_second_repair_is_attempted(self):
        second = copy.deepcopy(self.candidate)
        second["id"] = second["candidate_id"] = "general-g1-q2"
        client = ScriptedClient([
            atomicity_review("A1=single;F1=compound"),
            # The first repair remains compound; the no-repair re-review must reject it.
            """QA q1
QUESTION: 端口如何变化？
ANSWER_POINT: 端口由 18080 改为 18081。 || SOURCES: 资料1
FORBIDDEN_POINT: 端口保持 18080，并且旧配置继续使用。 || SOURCES: 资料1
END_QA""",
            atomicity_review("A1=single;F1=compound"),
            atomicity_review("A1=single;F1=compound"),
        ])
        result = review_candidates(
            self.scope, self.facts, [self.candidate, second], client,
            qa_mode="general", review_mode="simple", allow_repair=True,
            generation_context=self.generation_context,
            repair_state={"remaining": 1})

        self.assertEqual([receipt["stage"] for receipt in client.usage], [
            "review_relevance", "review_atomicity", "repair",
            "review_relevance", "review_atomicity",
            "review_relevance", "review_atomicity",
        ])
        self.assertEqual(sum(receipt["stage"] == "repair"
                             for receipt in client.usage), 1)
        self.assertEqual(len(result.get("revisions", [])), 1)
        self.assertFalse(any(
            item.get("status") == "approved" for item in result.get("questions", [])))
        self.assertTrue(any(
            isinstance(item, dict)
            and isinstance(item.get("question"), dict)
            and item["question"].get("id") == second["id"]
            for item in result.get("rejected", [])))

    def test_whitelisted_local_reference_can_be_fixed_but_invalid_source_cannot(self):
        leaking = copy.deepcopy(self.candidate)
        leaking["question"] = "资料1中的端口如何变化？"
        repair_response = """QA q1
QUESTION: 端口如何变化？
ANSWER_POINT: 端口由 18080 改为 18081。 || SOURCES: 资料1
END_QA"""
        client = ScriptedClient([
            repair_response, atomicity_review("A1=single"),
            completeness_review(), evidence_review("A1=supported@资料1"),
        ])
        revision, revised = repair_simple_candidate(
            self.scope, self.facts, leaking,
            {"reason": "local_reference_in_public_text", "failed_checks": [],
             "question": leaking},
            client, qa_mode="general", generation_context=self.generation_context)
        self.assertIsNotNone(revised)
        self.assertEqual(revised["questions"][0]["status"], "approved")
        self.assertEqual(revised["questions"][0]["id"], leaking["id"])
        self.assertNotIn("资料1", revised["questions"][0]["question"])
        self.assertNotIn("m1", json.dumps(client.calls[0][1], ensure_ascii=False))

        invalid = copy.deepcopy(self.candidate)
        invalid["answer_points"] = [{"text": "端口由 18080 改为 18081。", "sources": ["m1"]}]
        invalid["forbidden_points"] = []
        invalid_client = ScriptedClient(["""QA q1
QUESTION: 端口如何变化？
ANSWER_POINT: 端口由 18080 改为 18081。 || SOURCES: 资料999
END_QA"""])
        bad_revision, bad_result = repair_simple_candidate(
            self.scope, self.facts, invalid,
            {"reason": "local_reference_in_public_text", "failed_checks": [],
             "question": invalid},
            invalid_client, qa_mode="general", generation_context=self.generation_context)
        self.assertIsNone(bad_result)
        self.assertTrue(bad_revision.get("invalid"))
        self.assertNotIn("after", bad_revision)

    def test_api_credential_and_budget_failures_retain_unapproved_candidate(self):
        for error in (ModelStageError("http_error", http_status=503),
                      ModelStageError("credential_guard")):
            with self.subTest(error=error.code):
                client = ScriptedClient([
                    atomicity_review("A1=single;F1=compound"), error,
                ])
                result = review_candidates(
                    self.scope, self.facts, [self.candidate], client,
                    qa_mode="general", review_mode="simple", allow_repair=True,
                    generation_context=self.generation_context)
                self.assertFalse(any(
                    item.get("status") == "approved"
                    for item in result.get("questions", [])))
                self.assertEqual(len(result.get("revisions", [])), 1)
                revision = result["revisions"][0]
                self.assertEqual(revision["before"]["id"], self.candidate["id"])
                self.assertEqual(revision["error"]["error_code"], error.code)
                self.assertTrue(any(
                    isinstance(item.get("question"), dict)
                    and item["question"].get("id") == self.candidate["id"]
                    for item in result.get("rejected", [])
                    if isinstance(item, dict)))

        huge_context = copy.deepcopy(self.generation_context)
        huge_context["payload"]["materials"][0]["text"] = "x" * 10000
        small_scope = dict(self.scope, model_request_chars=5000)
        client = ScriptedClient([atomicity_review("A1=single;F1=compound")])
        result = review_candidates(
            small_scope, self.facts, [self.candidate], client,
            qa_mode="general", review_mode="simple", allow_repair=True,
            generation_context=huge_context)
        self.assertFalse(any(
            item.get("status") == "approved" for item in result.get("questions", [])))
        self.assertEqual(len(result.get("revisions", [])), 1)
        self.assertEqual(result["revisions"][0]["error"]["error_code"],
                         "request_budget")
        self.assertTrue(any(
            isinstance(item.get("question"), dict)
            and item["question"].get("id") == self.candidate["id"]
            for item in result.get("rejected", []) if isinstance(item, dict)))
        self.assertEqual([receipt["stage"] for receipt in client.usage],
                         ["review_relevance", "review_atomicity"])

    def test_cli_validation_repair_still_runs_code_static_postcheck(self):
        """A public-text repair cannot bypass the final code evidence gate."""
        original = copy.deepcopy(self.candidate)
        original.update(qa_mode="code", type="failure_diagnosis")
        rejected = {
            "question": original,
            "reason": "local_reference_in_public_text",
        }
        repaired_question = copy.deepcopy(original)
        repaired_question["question"] = "端口如何变化？"
        repaired_question["status"] = "approved"
        revised = {
            "questions": [repaired_question], "rejected": [],
            "stage_errors": [], "stage_status": {"review": "completed"},
        }
        revision = {
            "candidate_id": original["id"], "before": copy.deepcopy(original),
            "after": copy.deepcopy(repaired_question),
        }
        generation = {
            "generated": {
                "questions": [], "all_candidates": [copy.deepcopy(original)],
                "rejected": [rejected], "stage_errors": [],
                "stage_status": {"qa": "completed"}, "raw_generated": 0,
            },
            "active_group": {
                "id": "g1", "qa_mode": "code", "facts": self.facts,
                "scope": self.scope, "eligible_types": ("failure_diagnosis",),
            },
            "static_precheck": {"status": "supported", "reason": "fixture"},
            "expanded_static_precheck": None, "expansion_audits": [],
            "generation_attempt_count": 1, "generation_request_count": 2,
            "expansion_rounds": 0, "expansion_stop_reason": "candidate_generated",
            "repair_context": self.generation_context,
        }
        static_calls = []

        def static_check(_group, _index, _target_type, candidate=None):
            static_calls.append(copy.deepcopy(candidate))
            return {"status": "supported", "reason": "fixture",
                    "fact_ids": ["f1"], "source_ids": ["m1"]}

        class FakeChatClient:
            def __init__(self, *unused):
                self.usage = []

        with patch.object(cli, "ChatClient", FakeChatClient), \
                patch.object(cli, "generate_simple_target", return_value=generation), \
                patch.object(cli, "repair_simple_validation_rejection",
                             return_value=(revision, revised)), \
                patch.object(cli, "static_candidate_labels",
                             side_effect=lambda _g, _q, _i, target: {"type": target}), \
                patch.object(cli, "static_code_evidence_check", side_effect=static_check):
            result = cli._run_qa_tasks(
                [(0, "code", generation["active_group"])],
                "https://example.invalid", "model", "KEY", 1,
                review_mode="simple", evidence_indexes={"code": {}},
                expansion_budget=0)

        self.assertEqual(len(result["questions"]), 1)
        self.assertEqual(result["questions"][0]["status"], "approved")
        self.assertEqual(result["questions"][0]["static_code_evidence"]["status"],
                         "supported")
        self.assertTrue(any(
            isinstance(candidate, dict)
            and candidate.get("id") == repaired_question["id"]
            for candidate in static_calls))


if __name__ == "__main__":
    unittest.main()
