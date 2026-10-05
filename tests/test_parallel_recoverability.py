import threading
import time
import unittest

from dialogue_benchmark import cli


class ParallelRecoverabilityTests(unittest.TestCase):
    def test_probes_overlap_and_keep_publication_order(self):
        questions = [
            {"id": "q1", "qa_mode": "code", "status": "approved",
             "question": "q1", "answer_points": [], "forbidden_points": []},
            {"id": "q2", "qa_mode": "code", "status": "approved",
             "question": "q2", "answer_points": [], "forbidden_points": []},
        ]
        threads = set()

        def probe(item):
            threads.add(threading.current_thread().name)
            time.sleep(0.02)
            return {"status": "uncertain", "reason": item["id"]}

        view = cli._publication_view(
            questions, {"code": 2}, recoverability_check=probe,
            recoverability_workers=2)
        self.assertEqual([item["id"] for item in view["questions"]], ["q1", "q2"])
        self.assertGreaterEqual(len(threads), 2)

    def test_duplicate_candidate_id_is_probed_once(self):
        questions = [
            {"id": "same", "qa_mode": "code", "status": "approved",
             "question": "first", "answer_points": [], "forbidden_points": []},
            {"id": "same", "qa_mode": "code", "status": "approved",
             "question": "second", "answer_points": [], "forbidden_points": []},
        ]
        calls = []

        def probe(item):
            calls.append(item["id"])
            return {"status": "uncertain", "reason": "not enough"}

        cli._publication_view(
            questions, {"code": 2}, recoverability_check=probe,
            recoverability_workers=2)
        self.assertEqual(calls, ["same"])


if __name__ == "__main__":
    unittest.main()
