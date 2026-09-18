import unittest
from copy import deepcopy

from dialogue_benchmark.cli import _publication_view
from dialogue_benchmark.selection import replenish, build_audit


def question(qid, mode="code", status="approved", text=None):
    return {"id": qid, "candidate_id": qid, "qa_mode": mode, "status": status,
            "question": text or qid, "evidence_group_id": qid,
            "answer_points": [{"text": text or qid, "sources": ["e1"]}],
            "forbidden_points": []}


class ReplenishmentTests(unittest.TestCase):
    def tasks(self, n=6):
        return [(i * 2 + j, mode, {"id": mode + str(i)})
                for i in range(n) for j, mode in enumerate(("general", "code"))]

    def test_new_groups_until_each_track_reaches_target(self):
        seen, checkpoints = [], []
        def run(tasks):
            self.assertLessEqual(len(tasks), 2)
            seen.extend(t[2]["id"] for t in tasks)
            qs = [question(t[2]["id"], t[1],
                           "needs_review" if t[2]["id"] == "code0" else "approved") for t in tasks]
            return {"questions": qs, "all_candidates": qs}
        result = replenish(self.tasks(), {"general": 1, "code": 3},
                           {"general": 6, "code": 6}, 2, run, _publication_view,
                           lambda r, p: checkpoints.append(deepcopy(r)))
        self.assertEqual(result["counts"], {"general": 1, "code": 3})
        self.assertNotIn("general1", seen)
        self.assertEqual(len(seen), len(set(seen)))
        self.assertGreater(len(checkpoints), 1)
        self.assertEqual(result["progress"]["stop_reasons"]["code"], "target_reached")

    def test_pending_duplicate_and_credential_do_not_count(self):
        qs = [question("a", status="needs_review", text="same"),
              question("b", text="same"), question("c", text="password=example-secret"),
              question("d", text="same"), question("e")]
        tasks = [(i, "code", {"id": str(i)}) for i in range(5)]
        def run(batch):
            items = [qs[t[0]] for t in batch]
            return {"questions": items, "all_candidates": items}
        result = replenish(tasks, {"code": 2}, {"code": 5}, 1, run, _publication_view)
        self.assertEqual([q["id"] for q in result["questions"]], ["b", "e"])
        self.assertEqual(result["progress"]["attempted"]["code"], 5)
        excluded = {s["candidate_id"]: s for s in result["selection"]}
        self.assertEqual(excluded["a"]["duplicate_of"], "b")
        self.assertEqual(excluded["c"]["selection_status"], "safety_blocked")
        self.assertEqual(len(result["all_candidates"]), 5)

    def test_budget_and_pool_exhaustion_and_global_blocker(self):
        def empty(batch):
            return {"questions": [], "stage_errors": [{"error_code": "timeout"}]}
        result = replenish(self.tasks(3), {"general": 20, "code": 20},
                           {"general": 1, "code": 8}, 2, empty, _publication_view)
        self.assertEqual(result["progress"]["stop_reasons"],
                         {"general": "budget_exhausted", "code": "pool_exhausted"})
        self.assertEqual(result["progress"]["missing"], {"general": 20, "code": 20})
        called = []
        def blocked(batch):
            called.append(batch)
            return {"questions": [question("ok", "general")],
                    "stage_errors": [{"http_status": 401}]}
        result = replenish(self.tasks(), {"general": 20, "code": 20},
                           {"general": 6, "code": 6}, 2, blocked, _publication_view)
        self.assertEqual(len(called), 1)
        self.assertEqual(len(result["questions"]), 1)
        self.assertEqual(result["progress"]["stop_reasons"]["code"], "global_blocker")

    def test_over_quota_preserves_approved_and_audit_revisions(self):
        qs = [question("a"), question("b")]
        invalid = question("bad")
        view = _publication_view(qs, {"code": 1})
        audit = build_audit(qs + [invalid], qs,
                            [{"question": invalid, "reason": "invalid_answer_evidence"}],
                            view["selection"], view["questions"],
                            [{"candidate_id": "b", "before": qs[1], "after": dict(qs[1], question="revised")}])
        by_id = {r["candidate_id"]: r for r in audit}
        self.assertEqual(by_id["b"]["review_status"], "approved")
        self.assertEqual(by_id["b"]["selection_status"], "over_quota")
        self.assertEqual(by_id["b"]["revisions"][0]["after"]["question"], "revised")
        self.assertEqual(by_id["bad"]["review_status"], "rejected")
        self.assertEqual(by_id["bad"]["original"]["id"], "bad")

    def test_redaction_before_global_dedup(self):
        qs = [question("a", text="Open /Users/alice/proj/a.py"),
              question("b", text="Open /Users/bob/proj/a.py")]
        view = _publication_view(qs, {"code": 2})
        self.assertEqual(view["counts"]["code"], 1)
        self.assertEqual(view["selection"][0]["selection_status"], "duplicate")
        self.assertIn("~/proj/a.py", view["questions"][0]["question"])

    def test_after_batch_records_dedup_usage_decisions_and_nonfatal_errors(self):
        tasks = [(0, "code", {"id": "g1"})]

        def run(batch):
            item = question("q1")
            return {"questions": [item], "all_candidates": [item]}

        def after_batch(merged, batch_result, batch_number):
            return {
                "usage": [{"request_count": 1, "stage": "duplicate_review"}],
                "decisions": [{"candidate_id": "q2", "duplicate_of": "q1"}],
                "errors": [{"error_code": "dedup_error", "batch": batch_number}],
            }

        result = replenish(
            tasks, {"code": 1}, {"code": 1}, 1, run, _publication_view,
            after_batch=after_batch)
        self.assertEqual(len(result["duplicate_decisions"]), 1)
        self.assertEqual(len(result["dedup_errors"]), 1)
        self.assertEqual(result["progress"]["requests"], 1)
        receipt = result["progress"]["batches"][0]
        self.assertEqual(receipt["duplicate_decisions"], 1)
        self.assertEqual(receipt["dedup_errors"][0]["error_code"], "dedup_error")

    def test_publication_applies_reviewed_duplicate_without_mutating_outcomes(self):
        left = question("left", text="same")
        right = question("right", text="different")
        decision = {
            "candidate_id": "right", "duplicate_of": "left",
            "selection_status": "duplicate", "reason": "reviewed_near_duplicate",
        }
        view = _publication_view(
            [left, right], {"code": 2}, duplicate_decisions=[decision])
        self.assertEqual([item["id"] for item in view["questions"]], ["left"])
        self.assertEqual(view["selection"][0]["candidate_id"], "right")
        self.assertEqual([left["id"], right["id"]], ["left", "right"])
