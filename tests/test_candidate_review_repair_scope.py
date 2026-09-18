"""Integration regression for re-scoping a repaired simple candidate."""

import copy
import json
import unittest

from dialogue_benchmark.llm import parse_text_response, review_candidates
from tests.simple_test_helpers import maybe_relevance_response


class _ScriptedClient:
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
        return parse_text_response(response) if isinstance(response, str) else response


def _atomicity(points):
    return ("REVIEW q1\nreview_contract: simple_atomicity_v1\n"
            "point_atomicity: %s\nEND_REVIEW" % points)


def _completeness():
    return ("REVIEW q1\nreview_contract: simple_v1\n"
            "completeness: complete\nmissing: none\nEND_REVIEW")


def _evidence(points):
    return ("REVIEW q1\nreview_contract: simple_v1\n"
            "point_evidence: %s\nEND_REVIEW" % points)


class CandidateReviewRepairScopeTests(unittest.TestCase):
    def test_repair_recomputes_scope_for_changed_candidate_before_one_full_rereview(self):
        scope = {
            "cutoff": 3,
            "dialogue": [
                {"id": "source-a", "order": 1, "kind": "message", "role": "assistant",
                 "text": "pkg/alpha.py::load_config 记录了旧值。"},
                {"id": "source-b", "order": 2, "kind": "message", "role": "assistant",
                 "text": "pkg/beta.py::parse 记录了新值。"},
            ],
            "events": [], "versions": [], "edges": [], "historical_edges": [],
            "stages": [], "review_guard_complete": True,
            "model_request_chars": 60000, "max_context_chars": 60000,
        }
        facts = [
            {"id": "f-a", "qa_mode": "general",
             "statement": "pkg/alpha.py::load_config 记录了变化", "sources": ["source-a"]},
            {"id": "f-b", "qa_mode": "general",
             "statement": "pkg/beta.py::parse 记录了变化", "sources": ["source-b"]},
        ]
        candidate = {
            "id": "q1", "candidate_id": "q1", "model_id": "model-q1",
            "qa_mode": "general", "type": "single-hop", "fact_ids": ["f-a"],
            "question": "记录的一个变化是什么？",
            "answer_points": [{
                "text": "pkg/alpha.py::load_config 从旧值改为新值，并且随后校验。",
                "sources": ["source-a"],
            }],
            "forbidden_points": [],
        }
        group = {
            "id": "g1", "qa_mode": "general",
            "allowed_types": ("single-hop",), "eligible_types": ("single-hop",),
            "facts": copy.deepcopy(facts), "scope": copy.deepcopy(scope),
            "review_guard_complete": True,
        }
        original_context = {
            "prompt": "ORIGINAL QA INSTRUCTION",
            "payload": {
                "facts": [{"text": "原始安全事实"}],
                "materials": [
                    {"reference": "资料1", "source_kind": "conversation",
                     "text": "pkg/alpha.py::load_config"},
                    {"reference": "资料2", "source_kind": "conversation",
                     "text": "pkg/beta.py::parse"},
                ],
                "relations": [],
                "focus": {"text": "记录的一个变化", "sources": ["资料1"]},
            },
            "ref_to_source": {"资料1": "source-a", "资料2": "source-b"},
        }
        calls = []

        def resolve(review_candidate):
            sources = tuple(review_candidate["answer_points"][0]["sources"])
            calls.append((sources, copy.deepcopy(review_candidate)))
            if sources == ("source-a",):
                return copy.deepcopy(scope), {"complete": True, "reason": "initial"}
            narrowed = copy.deepcopy(scope)
            narrowed["dialogue"] = [record for record in narrowed["dialogue"]
                                     if record["id"] == "source-b"]
            narrowed["review_guard_sources"] = ["source-b"]
            return narrowed, {"complete": True, "reason": "recomputed"}

        repair_response = """QA q1
QUESTION: 记录的一个变化是什么？
ANSWER_POINT: pkg/beta.py::parse 从旧值改为新值。 || SOURCES: 资料2
ANSWER_POINT: pkg/beta.py::parse 随后校验。 || SOURCES: 资料2
END_QA"""
        client = _ScriptedClient([
            _atomicity("A1=compound"), repair_response,
            _atomicity("A1=single;A2=single"), _completeness(),
            _evidence("A1=supported@资料1;A2=supported@资料1"),
        ])

        result = review_candidates(
            scope, facts, [candidate], client, qa_mode="general",
            review_mode="simple", allow_repair=True,
            generation_context=original_context,
            review_scope_resolver=resolve,
        )
        self.assertEqual([item[0] for item in calls],
                         [("source-a",), ("source-b",)])
        self.assertEqual([receipt["stage"] for receipt in client.usage], [
            "review_relevance", "review_atomicity", "repair",
            "review_relevance", "review_atomicity", "review_completeness",
            "review_evidence",
        ])
        self.assertEqual(result["questions"][0]["status"], "approved")
        self.assertEqual(len(result.get("revisions", [])), 1)
        self.assertEqual(result["revisions"][0]["after"]["id"], candidate["id"])
        self.assertEqual(result["revisions"][0]["after"]["fact_ids"], ["f-b"])
        evidence_payload = client.calls[-1][1]
        materials_text = json.dumps(evidence_payload["materials"], ensure_ascii=False)
        self.assertIn("pkg/beta.py::parse", materials_text)
        self.assertNotIn("pkg/alpha.py::load_config", materials_text)

    def test_failed_scope_resolution_cannot_fallback_to_a_complete_original_guard(self):
        scope = {
            "cutoff": 1,
            "dialogue": [{
                "id": "source-a", "order": 1, "kind": "message",
                "role": "assistant", "text": "pkg/alpha.py::load_config 记录了变化。",
            }],
            "events": [], "versions": [], "edges": [], "historical_edges": [],
            "stages": [], "review_guard_complete": True,
            "model_request_chars": 60000, "max_context_chars": 60000,
        }
        facts = [{
            "id": "f-a", "qa_mode": "general",
            "statement": "pkg/alpha.py::load_config 记录了变化",
            "sources": ["source-a"],
        }]
        candidate = {
            "id": "q1", "candidate_id": "q1", "model_id": "model-q1",
            "qa_mode": "general", "type": "single-hop", "fact_ids": ["f-a"],
            "question": "pkg/alpha.py::load_config 的记录是什么？",
            "answer_points": [{"text": "记录了变化。", "sources": ["source-a"]}],
            "forbidden_points": [],
        }
        client = _ScriptedClient([_atomicity("A1=single"), _completeness()])

        result = review_candidates(
            scope, facts, [candidate], client, qa_mode="general",
            review_mode="simple", allow_repair=False,
            review_scope_resolver=lambda _candidate: (
                None, {"complete": False, "reason": "candidate_guard_over_budget"}),
        )

        self.assertEqual(result["questions"][0]["status"], "needs_review")
        self.assertEqual(result["questions"][0]["review_error"],
                         "incomplete_review_guard")
        self.assertEqual([receipt["stage"] for receipt in client.usage], [
            "review_relevance", "review_atomicity", "review_completeness",
        ])
        self.assertEqual(result["stage_status"].get("review_evidence"), "skipped")


if __name__ == "__main__":
    unittest.main()
