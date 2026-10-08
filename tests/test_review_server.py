import json
import tempfile
import unittest
from pathlib import Path

from dialogue_benchmark.review_server import (PAGE, append_decision, materialize_public,
                                               queue_payload)


class ReviewServerTests(unittest.TestCase):
    def test_queue_payload_reads_candidates_and_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "qa-review-queue.json").write_text(json.dumps({
                "items": [{"candidate_id": "q1", "status": "needs_review"}],
                "stage_errors": [{"error_code": "protocol_error"}],
            }), encoding="utf-8")
            append_decision(root, {"candidate_id": "q1", "action": "approve"})
            payload = queue_payload(root)
            self.assertEqual(payload["items"][0]["candidate_id"], "q1")
            self.assertEqual(payload["decisions"][0]["action"], "approve")
            self.assertEqual(payload["stage_errors"][0]["error_code"], "protocol_error")

    def test_decision_validation_and_append_only_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "action"):
                append_decision(root, {"candidate_id": "q1", "action": "publish"})
            first = append_decision(root, {"candidate_id": "q1", "action": "defer"})
            second = append_decision(root, {"candidate_id": "q1", "action": "reject"})
            rows = [json.loads(line) for line in
                    (root / "review-decisions.jsonl").read_text().splitlines()]
            self.assertEqual([row["action"] for row in rows], ["defer", "reject"])
            self.assertEqual(first["candidate_id"], second["candidate_id"])

    def test_approval_materializes_public_question_and_rejection_removes_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "qa-review-queue.json").write_text(json.dumps({
                "items": [{
                    "candidate_id": "q1",
                    "id": "q1",
                    "qa_mode": "memory",
                    "question": "Which rule applies?",
                    "answer_points": [{"text": "Use the confirmed rule."}],
                    "status": "needs_review",
                }],
                "stage_errors": [],
            }), encoding="utf-8")
            (root / "qa-public.json").write_text(json.dumps({
                "status": "needs_review", "questions": [], "counts": {},
            }), encoding="utf-8")
            append_decision(root, {"candidate_id": "q1", "action": "approve"})
            self.assertEqual(len(json.loads(
                (root / "qa-public.json").read_text())["questions"]), 1)
            append_decision(root, {"candidate_id": "q1", "action": "reject"})
            self.assertEqual(json.loads(
                (root / "qa-public.json").read_text())["questions"], [])
            self.assertEqual(materialize_public(root)["status"], "completed_no_questions")

    def test_page_is_local_and_has_non_blocking_actions(self):
        self.assertIn("/api/decision", PAGE)
        self.assertIn("生成流程不会等待", PAGE)
        self.assertNotIn("http://", PAGE)


if __name__ == "__main__":
    unittest.main()
