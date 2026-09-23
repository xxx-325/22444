"""Offline regressions for the simple atomicity and evidence contracts."""

import copy
import unittest

from dialogue_benchmark.llm import (
    SIMPLE_ATOMICITY_PROMPT,
    SIMPLE_QA_PROMPT,
    parse_text_response,
    review_candidates,
)
from tests.simple_test_helpers import maybe_relevance_response


class ScriptedClient:
    """Return scripted model documents and retain safe request projections."""

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.usage = []

    def ask(self, prompt, payload):
        # Target routing is exercised separately in test_memory_types.
        if "review_contract: target_v1" in prompt:
            return {"reviews": [{"id": "q1", "review_contract": "target_v1",
                                 "target_alignment": "aligned"}]}
        self.calls.append((prompt, copy.deepcopy(payload)))
        self.usage.append({"status": "completed"})
        relevance = maybe_relevance_response(prompt, payload)
        if relevance is not None:
            return relevance
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        if callable(response):
            response = response(prompt, payload)
        if isinstance(response, str):
            return parse_text_response(response)
        return copy.deepcopy(response)


def atomicity_review(point_evidence):
    return """REVIEW q1
review_contract: simple_atomicity_v1
point_atomicity: %s
END_REVIEW""" % point_evidence


def completeness_review(value="complete", missing="none"):
    return """REVIEW q1
review_contract: simple_v1
completeness: %s
missing: %s
END_REVIEW""" % (value, missing)


def evidence_review(point_evidence):
    return """REVIEW q1
review_contract: simple_v1
point_evidence: %s
END_REVIEW""" % point_evidence


class SimpleAtomicityTests(unittest.TestCase):
    def setUp(self):
        self.scope = {
            "cutoff": 4,
            "dialogue": [
                {"id": "m1", "kind": "message", "role": "user", "order": 1,
                 "stage_id": "s1", "text": "端口由 18080 改为 18081。"},
                {"id": "m2", "kind": "message", "role": "assistant", "order": 2,
                 "stage_id": "s1", "text": "端口不可用时最多重试两次并返回失败。"},
                {"id": "m3", "kind": "message", "role": "assistant", "order": 3,
                 "stage_id": "s1", "text": "审核需要保留反证。"},
            ],
            "events": [],
            "versions": [],
            "stages": [{"id": "s1"}],
            "review_guard_sources": ["m3"],
            "review_guard_complete": True,
            "model_request_chars": 60000,
            "max_context_chars": 60000,
        }
        self.facts = [
            {"id": "f1", "statement": "端口由 18080 改为 18081。", "sources": ["m1"]},
            {"id": "f2", "statement": "端口不可用时最多重试两次并返回失败。", "sources": ["m2"]},
        ]
        self.candidate = {
            "id": "general_g1_q1",
            "candidate_id": "general_g1_q1",
            "model_id": "q1",
            "qa_mode": "general",
            "type": "constraint_followthrough",
            "question": "端口如何变化，端口不可用时如何处理？",
            "answer_points": [
                {"text": "端口由 18080 改为 18081。", "sources": ["m1"]},
                {"text": "端口不可用时最多重试两次并返回失败。", "sources": ["m2"]},
            ],
            "forbidden_points": [
                {"text": "端口保持 18080。", "sources": ["m1"]},
            ],
        }

    def _run(self, responses, candidate=None, allow_repair=False):
        client = ScriptedClient(responses)
        result = review_candidates(
            self.scope, self.facts, [candidate or self.candidate], client,
            qa_mode="general", review_mode="simple",
            allow_repair=allow_repair)
        return result, client

    def test_shared_rule_splits_independently_true_members_not_ranges_or_paths(self):
        qa_prompt = " ".join(SIMPLE_QA_PROMPT.split())
        self.assertIn("每行只写一个可以单独判真的结论", qa_prompt)
        self.assertIn("同一个对象从旧值改为新值是一条", qa_prompt)
        self.assertIn("一个条件导致一个结果是一条", qa_prompt)
        self.assertIn("不同对象/动作必须分行", qa_prompt)

        atomicity_prompt = " ".join(SIMPLE_ATOMICITY_PROMPT.split())
        self.assertIn("each member can be true independently", atomicity_prompt)
        self.assertIn("Do not split a continuous range or path", atomicity_prompt)

    def test_port_transition_and_conditional_result_follow_model_atomicity(self):
        candidate = copy.deepcopy(self.candidate)
        candidate["answer_points"] = [
            {"text": "端口由 18080 改为 18081。", "sources": ["m1"]},
            {"text": "在端口不可用时，客户端会重试两次并返回失败。",
             "sources": ["m2"]},
        ]
        result, client = self._run([
            atomicity_review("A1=single;A2=single;F1=single"),
            completeness_review(),
            evidence_review(
                "A1=supported@资料1;A2=supported@资料2;F1=contradicted@资料1"),
        ], candidate=candidate)

        self.assertEqual(result["questions"][0]["status"], "approved")
        self.assertEqual(
            [receipt["stage"] for receipt in client.usage],
            ["review_relevance", "review_atomicity", "review_completeness",
             "review_evidence"],
        )
        atomic_prompt, atomic_payload = client.calls[1]
        self.assertIn("simple_atomicity_v1", atomic_prompt)
        self.assertIn("point_atomicity", atomic_prompt)
        self.assertEqual(
            set(atomic_payload["candidates"][0]),
            {"id", "immutable_question", "answer_points", "forbidden_points"},
        )
        self.assertEqual(
            [point["review_id"] for point in atomic_payload["candidates"][0][
                "answer_points"]],
            ["A1", "A2"],
        )
        self.assertEqual(
            [point["review_id"] for point in atomic_payload["candidates"][0][
                "forbidden_points"]],
            ["F1"],
        )
        self.assertNotIn("sources", atomic_payload["candidates"][0]["answer_points"][0])
        self.assertNotIn("facts", atomic_payload)
        self.assertNotIn("materials", atomic_payload)

    def test_compound_port_and_retry_point_rejects_before_follow_ups(self):
        candidate = copy.deepcopy(self.candidate)
        candidate["answer_points"] = [
            {"text": "端口由 18080 改为 18081，失败时最多重试两次。",
             "sources": ["m1", "m2"]},
        ]
        result, client = self._run(
            [atomicity_review("A1=compound;F1=single")],
            candidate=candidate, allow_repair=True)

        self.assertFalse(result["questions"])
        self.assertEqual(result["rejected"][0]["reason"],
                         "semantic_answer_atomicity_failed")
        self.assertEqual(result["rejected"][0]["failed_checks"],
                         ["atomic_points_correct"])
        self.assertEqual([receipt["stage"] for receipt in client.usage],
                         ["review_relevance", "review_atomicity"])
        self.assertFalse(result.get("revisions"))

    def test_uncertain_invalid_and_transport_atomicity_never_publish(self):
        cases = [
            (
                "uncertain",
                atomicity_review("A1=uncertain;A2=single;F1=single"),
                "uncertain_atomicity",
            ),
            (
                "missing-id",
                atomicity_review("A1=single;A2=single"),
                "invalid_atomicity_review",
            ),
            ("transport", RuntimeError("synthetic atomicity failure"),
             "review_atomicity_failed"),
        ]
        for label, response, error in cases:
            with self.subTest(label=label):
                result, client = self._run([response])
                self.assertFalse(any(
                    item.get("status") == "approved"
                    for item in result.get("questions", [])
                ))
                self.assertEqual(len(client.calls), 2)
                self.assertEqual([receipt["stage"] for receipt in client.usage],
                                 ["review_relevance", "review_atomicity"])
                if label == "transport":
                    self.assertEqual(result["stage_status"]["review"], "failed")
                    self.assertEqual(result["stage_errors"][0]["stage"],
                                     "review_atomicity")
                    self.assertEqual(result["questions"][0]["review_error"], error)
                else:
                    self.assertEqual(result["questions"][0]["status"],
                                     "needs_review")
                    self.assertEqual(result["questions"][0]["review_error"], error)

    def test_forbidden_point_statuses_use_reverse_evidence_semantics(self):
        candidate = copy.deepcopy(self.candidate)
        candidate["answer_points"] = candidate["answer_points"][:1]
        statuses = {
            "contradicted@资料1": "approved",
            "supported@资料1": "rejected",
            "insufficient": "needs_review",
            "stale@资料1": "rejected",
        }
        for forbidden_status, expected in statuses.items():
            with self.subTest(forbidden_status=forbidden_status):
                result, _client = self._run([
                    atomicity_review("A1=single;F1=single"),
                    completeness_review(),
                    evidence_review(
                        "A1=supported@资料1;F1=%s" % forbidden_status),
                ], candidate=candidate)
                if expected == "approved":
                    self.assertEqual(result["questions"][0]["status"], "approved")
                elif expected == "needs_review":
                    self.assertEqual(result["questions"][0]["status"], "needs_review")
                    self.assertEqual(result["questions"][0]["review_error"],
                                     "insufficient_evidence")
                else:
                    self.assertFalse(result["questions"])
                    failed_checks = result["rejected"][0]["failed_checks"]
                    self.assertIn("evidence_supported", failed_checks)
                    if forbidden_status.startswith("stale"):
                        self.assertIn("version_consistent", failed_checks)


if __name__ == "__main__":
    unittest.main()
