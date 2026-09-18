"""Offline regressions for the shared one-expansion generation helper."""

import copy
import unittest

from dialogue_benchmark.cli import generate_simple_target
from dialogue_benchmark.fact_index import build_evidence_index
from dialogue_benchmark.llm import parse_text_response
from tests.simple_test_helpers import maybe_focus_response


class ScriptedClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.usage = []

    def ask(self, prompt, payload):
        self.calls.append((prompt, copy.deepcopy(payload)))
        self.usage.append({"stage": "qa", "status": "completed"})
        focus = maybe_focus_response(prompt, payload)
        if focus is not None:
            return focus
        return parse_text_response(next(self.responses))


class SimpleExpansionOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.scope = {
            "cutoff": 2,
            "dialogue": [
                {"id": "e-base", "order": 1, "kind": "message",
                 "source_kind": "conversation", "call_id": "c1",
                 "role": "assistant", "text": "run runner.py"},
                {"id": "e-extra", "order": 2, "kind": "message",
                 "source_kind": "conversation", "call_id": "c1",
                 "role": "assistant", "text":
                 "runner.py outcome was validated after the request."},
            ],
            "events": [], "versions": [], "edges": [],
            "historical_edges": [], "stages": [],
        }
        self.facts = [
            {"id": "f-base", "qa_mode": "general", "statement":
             "runner.py currently handles the request.",
             "sources": ["e-base"]},
            {"id": "f-extra", "qa_mode": "general", "statement":
             "runner.py outcome was validated after the request.",
             "sources": ["e-extra"]},
        ]
        self.index = build_evidence_index(
            self.facts, [self.scope], "general", 60000)
        # This suite exercises the bounded orchestration loop.  Seed one
        # already-authorized relation explicitly so general-source visibility
        # remains testable without coupling the fixture to call/result record
        # kinds (which general QA cannot cite).
        self.index["expansion_candidates"]["f-base"] = [{
            "missing_kind": "outcome", "relation": "call_result",
            "fact_ids": ["f-extra"], "source_ids": ["e-extra"],
            "order": 2, "distance": 1,
        }]
        self.group = {
            "id": "g-base",
            "qa_mode": "general",
            "facts": [copy.deepcopy(self.facts[0])],
            "scope": dict(self.index["universe"], evidence_group={"id": "g-base"}),
            "relation_path": {},
            "expansion_pointer": {
                "attempted": False,
                "candidates": [{
                    "missing_kind": "outcome",
                    "relation": "explicit_result",
                    "fact_ids": ["f-extra"],
                    "source_ids": ["e-extra"],
                    "distance": 1,
                    "base_fact_id": "f-base",
                }],
            },
        }

    def test_cli_helper_retries_once_after_object_directed_no_qa(self):
        client = ScriptedClient([
            "NO_QA\nMISSING_KIND: outcome\nMISSING_OBJECT: runner.py",
            "NO_QA\nMISSING_KIND: outcome\nMISSING_OBJECT: runner.py",
        ])

        result = generate_simple_target(
            self.group, self.index, "single-hop", client, "general",
            expansion_budget=1)

        self.assertEqual(result["generation_request_count"], 4)
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(len(result["expansion_audits"]), 1)
        self.assertEqual(result["expansion_audits"][0]["status"], "expanded")
        self.assertEqual(result["expansion_rounds"], 1)
        self.assertEqual(result["expansion_stop_reason"],
                         "expansion_budget_exhausted")
        self.assertEqual(result["generated"]["missing_object"], "runner.py")
        self.assertFalse(result["generated"]["questions"])
        self.assertNotIn("expansion_pending", result)
        # The retry is the fourth request now: each generation attempt is
        # focus followed by QA, and the expanded QA payload must include the
        # newly selected material.
        self.assertGreaterEqual(len(client.calls[3][1]["materials"]), 2)

    def test_cli_helper_does_not_retry_for_unmatched_object(self):
        client = ScriptedClient([
            "NO_QA\nMISSING_KIND: outcome\nMISSING_OBJECT: other.py",
        ])

        result = generate_simple_target(
            self.group, self.index, "single-hop", client, "general",
            expansion_budget=1)

        self.assertEqual(result["generation_request_count"], 2)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(len(result["expansion_audits"]), 1)
        self.assertNotEqual(result["expansion_audits"][0]["status"], "expanded")
        self.assertEqual(result["expansion_stop_reason"],
                         "no_matching_explicit_relation")
        self.assertFalse(result["generated"]["questions"])
        self.assertNotIn("expansion_pending", result)

    def test_cli_helper_uses_one_extra_request_for_a_successful_candidate(self):
        response = """QA q1
QUESTION: runner.py 的结果是什么？
ANSWER_POINT: runner.py outcome was validated after the request. || SOURCES: 资料2
END_QA"""
        client = ScriptedClient([
            "NO_QA\nMISSING_KIND: outcome\nMISSING_OBJECT: runner.py",
            response,
        ])

        result = generate_simple_target(
            self.group, self.index, "single-hop", client, "general",
            expansion_budget=1)

        self.assertEqual(result["generation_request_count"], 4)
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(len(result["generated"]["questions"]), 1)
        self.assertEqual(result["active_group"]["facts"][-1]["id"], "f-extra")
        self.assertEqual(result["expansion_rounds"], 1)
        self.assertEqual(result["expansion_stop_reason"],
                         "candidate_generated")
        self.assertEqual(result["expansion_audits"][0]["status"], "expanded")
        self.assertEqual(
            {fact["id"] for fact in result["generated"]["facts"]},
            {"f-base", "f-extra"},
        )


if __name__ == "__main__":
    unittest.main()
