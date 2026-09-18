"""Offline contracts for the simple NO_QA evidence-extension protocol."""

import copy
import unittest

from dialogue_benchmark.llm import (
    GENERAL_FACT_PROMPT,
    SIMPLE_QA_PROMPT,
    generate_from_facts,
    parse_text_response,
)
from dialogue_benchmark.protocol import MISSING_KINDS


class ScriptedClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def ask(self, prompt, payload):
        self.calls.append((prompt, copy.deepcopy(payload)))
        return parse_text_response(self.response)


class MissingObjectContractTests(unittest.TestCase):
    def setUp(self):
        self.scope = {
            "cutoff": 2,
            "dialogue": [
                {"id": "e1", "kind": "message", "role": "assistant",
                 "order": 1, "text": "runner.py uses the recorded port."},
            ],
            "events": [], "versions": [], "stages": [],
            "model_request_chars": 60000,
            "max_context_chars": 60000,
        }
        self.facts = [{
            "id": "f1",
            "statement": "runner.py uses the recorded port.",
            "sources": ["e1"],
        }]

    def test_no_qa_returns_each_declared_kind_and_exact_object(self):
        self.assertEqual(
            MISSING_KINDS,
            {"earlier_state", "later_state", "reason", "outcome", "dependency"},
        )
        for kind in sorted(MISSING_KINDS):
            with self.subTest(kind=kind):
                parsed = parse_text_response(
                    "NO_QA\nMISSING_KIND: %s\nMISSING_OBJECT: runner.py" % kind)
                self.assertEqual(parsed, {
                    "questions": [],
                    "missing_kind": kind,
                    "missing_object": "runner.py",
                })

    def test_no_qa_extension_is_strict_and_rejects_legacy_or_ambiguous_forms(self):
        invalid = [
            "NO_QA\nMISSING_KIND: reason",
            "NO_QA\nMISSING_KIND: old_state\nMISSING_OBJECT: runner.py",
            "NO_QA\nMISSING_KIND: reason\nMISSING_OBJECT:",
            "NO_QA\nMISSING_KIND: reason\nMISSING_OBJECT: runner.py\nwhy",
            "NO_QA\nMISSING_KIND: reason\nMISSING_OBJECT: runner.py\nEND",
        ]
        for response in invalid:
            with self.subTest(response=response):
                with self.assertRaises(ValueError):
                    parse_text_response(response)

    def test_simple_prompt_names_five_kinds_and_requires_one_object(self):
        for prompt in (SIMPLE_QA_PROMPT,):
            for kind in sorted(MISSING_KINDS):
                self.assertIn(kind, prompt)
            self.assertIn("MISSING_OBJECT:", prompt)
            self.assertIn("原文中精确的路径或符号", prompt)

        # General extraction may use document records, but its contract must
        # not silently turn tool output into conversation facts.
        self.assertIn("explicitly supplied document blocks", GENERAL_FACT_PROMPT)
        self.assertIn("Do not infer code or repository\nfacts from tool records",
                      GENERAL_FACT_PROMPT)

    def test_generation_propagates_kind_and_object_without_model_annotations(self):
        client = ScriptedClient(
            "NO_QA\nMISSING_KIND: dependency\nMISSING_OBJECT: runner.py")
        result = generate_from_facts(
            self.scope, self.facts, client, qa_mode="general",
            allowed_types=("single-hop",), target_type="single-hop")

        self.assertFalse(result["questions"])
        self.assertEqual(result["missing_kind"], "dependency")
        self.assertEqual(result["missing_object"], "runner.py")
        self.assertEqual(result["stage_status"]["qa"], "completed")
        self.assertNotIn("type", result)
        self.assertNotIn("difficulty", result)
        self.assertEqual(len(client.calls), 1)


if __name__ == "__main__":
    unittest.main()
