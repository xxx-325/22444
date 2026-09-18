"""Review mode wiring without provider requests."""

import unittest
from unittest.mock import patch

from dialogue_benchmark import cli


class ReviewModeCliTests(unittest.TestCase):
    def test_simple_default_and_split_comparison_mode(self):
        parser = cli._build_parser()
        args = parser.parse_args(["input.json", "--output", "unused"])
        self.assertEqual(args.review_mode, "simple")
        args = parser.parse_args([
            "input.json", "--output", "unused", "--review-mode", "split"])
        self.assertEqual(args.review_mode, "split")
        with self.assertRaises(SystemExit):
            parser.parse_args([
                "input.json", "--output", "unused", "--review-mode", "unchecked"])

    def test_mode_reaches_review_and_raw_candidates_keep_group(self):
        class FakeClient:
            def __init__(self, *args):
                self.usage = []

        generated = {
            "questions": [{"id": "q1", "question": "Which prior decision applies?"}],
            "all_candidates": [{"id": "q1"}, {"id": "q2"}],
            "stage_status": {"qa": "completed"},
        }
        tasks = [(0, "general", {
            "id": "general-group-1", "scope": {}, "facts": [],
            "allowed_types": ("single-hop",),
        })]
        with patch.object(cli, "ChatClient", FakeClient), \
                patch.object(cli, "generate_from_facts", return_value=generated), \
                patch.object(cli, "review_candidates", return_value={
                    "questions": [], "stage_status": {"review": "completed"},
                }) as review:
            result = cli._run_qa_tasks(
                tasks, "https://example.invalid", "test-model", "KEY", 1,
                review_mode="single")
        self.assertEqual(review.call_args.kwargs["review_mode"], "single")
        self.assertTrue(all(
            q["evidence_group_id"] == "general-group-1"
            for q in result["all_candidates"]))


if __name__ == "__main__":
    unittest.main()
