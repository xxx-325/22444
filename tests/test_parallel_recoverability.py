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

    def test_probe_results_usage_and_errors_follow_candidate_order(self):
        questions = [
            {"id": candidate_id, "qa_mode": "code", "status": "approved",
             "question": candidate_id, "answer_points": [], "forbidden_points": []}
            for candidate_id in ("q1", "q2", "q3", "q1")
        ]
        state = {"results": {}, "errors": []}
        q3_finished = threading.Event()
        q2_finished = threading.Event()
        lock = threading.Lock()

        def probe(item):
            candidate_id = item["id"]
            if candidate_id == "q1":
                self.assertTrue(q2_finished.wait(5))
            elif candidate_id == "q2":
                self.assertTrue(q3_finished.wait(5))
            result = {
                "status": "uncertain", "reason": candidate_id,
                "usage": [{"candidate_id": candidate_id, "step": step}
                          for step in range(2 if candidate_id == "q1" else 1)],
            }
            with lock:
                state["results"][candidate_id] = result
                if candidate_id != "q1":
                    state["errors"].append({"candidate_id": candidate_id, "error": candidate_id})
            if candidate_id == "q3":
                q3_finished.set()
            elif candidate_id == "q2":
                q2_finished.set()
            return result

        cli._publication_view(
            questions, {"code": 3}, recoverability_check=probe,
            recoverability_workers=3)
        self.assertEqual(list(state["results"]), ["q3", "q2", "q1"])
        ordered = cli._ordered_recoverability_state(state, questions)
        self.assertEqual(list(ordered["results"]), ["q1", "q2", "q3"])
        self.assertEqual(ordered["usage"], [
            {"candidate_id": "q1", "step": 0}, {"candidate_id": "q1", "step": 1},
            {"candidate_id": "q2", "step": 0}, {"candidate_id": "q3", "step": 0},
        ])
        self.assertEqual([error["candidate_id"] for error in ordered["errors"]], ["q2", "q3"])
        self.assertEqual(list(state["results"]), ["q3", "q2", "q1"])


if __name__ == "__main__":
    unittest.main()
