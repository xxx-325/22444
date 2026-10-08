import unittest

from dialogue_benchmark.selection import (
    difficulty_balance,
    difficulty_targets,
    select_approved,
    type_balance,
    balance_group_tasks,
)


def candidate(index, difficulty, mode="code", kind=None):
    return {
        "id": "q%d" % index,
        "qa_mode": mode,
        "status": "approved",
        "question": "question %d" % index,
        "difficulty": difficulty,
        "difficulty_origin": "static_graph_distance",
        "type": kind or "kind-%d" % index,
        "evidence_group_id": "group-%d" % index,
        "fact_ids": ["fact-%d" % index],
    }


class DifficultySelectionTests(unittest.TestCase):
    def test_non_difficulty_track_is_omitted_from_balance(self):
        self.assertEqual(difficulty_targets({"memory": 4}), {})
        self.assertEqual(difficulty_balance([{"qa_mode": "memory"}], {"memory": 1}), {})

    def test_group_balance_preserves_groups_and_track_slots(self):
        tasks = [(i, "code", {"id": str(i), "static_difficulty": level})
                 for i, level in enumerate(["easy", "easy", "easy", "medium", "hard"])]
        balanced = balance_group_tasks(tasks)
        self.assertEqual(sorted(t[2]["id"] for t in balanced), [str(i) for i in range(5)])
        self.assertEqual([t[1] for t in balanced], ["code"] * 5)
        self.assertEqual([t[2]["static_difficulty"] for t in balanced[:3]], ["easy", "medium", "hard"])

    def test_group_balance_keeps_memory_unchanged_and_unknown(self):
        tasks = [(0, "memory", {"id": "m"}), (1, "code", {"id": "u"}),
                 (2, "code", {"id": "e", "static_difficulty": "easy"})]
        balanced = balance_group_tasks(tasks)
        self.assertEqual(balanced[0], tasks[0])
        self.assertEqual({item[2]["id"] for item in balanced}, {"m", "u", "e"})
    def test_targets_are_balanced_and_deterministic(self):
        self.assertEqual(
            difficulty_targets({"code": 10})["code"],
            {"easy": 3, "medium": 4, "hard": 3},
        )

    def test_selection_prefers_unfilled_levels_without_rejecting_easy(self):
        questions = (
            [candidate(index, "easy") for index in range(6)]
            + [candidate(6 + index, "medium") for index in range(2)]
            + [candidate(8 + index, "hard") for index in range(2)]
        )
        kept, _, counts = select_approved(questions, {"code": 6})
        self.assertEqual(counts, {"code": 6})
        self.assertEqual(
            {level: sum(q.get("difficulty") == level for q in kept)
             for level in ("easy", "medium", "hard")},
            {"easy": 2, "medium": 2, "hard": 2},
        )

    def test_shortage_backfills_and_reports_soft_shortfall(self):
        kept, _, _ = select_approved(
            [candidate(index, "easy") for index in range(4)],
            {"code": 4},
        )
        balance = difficulty_balance(kept, {"code": 4})["code"]
        self.assertEqual(len(kept), 4)
        self.assertEqual(balance["actual"], {"easy": 4, "medium": 0, "hard": 0})
        self.assertEqual(balance["shortfall"], {"easy": 0, "medium": 2, "hard": 1})

    def test_type_balance_is_best_effort(self):
        questions = (
            [candidate(index, "easy", kind="alpha") for index in range(4)]
            + [candidate(4 + index, "medium", kind="beta") for index in range(4)]
            + [candidate(8 + index, "hard", kind="gamma") for index in range(4)]
        )
        kept, _, _ = select_approved(questions, {"code": 6})
        balance = type_balance(kept, {"code": 6})["code"]
        self.assertEqual(balance["actual"], {"alpha": 2, "beta": 2, "gamma": 2})

    def test_external_track_has_type_balance_but_no_graph_difficulty(self):
        questions = [
            candidate(index, "hard", mode="memory", kind="M1" if index < 2 else "M2")
            for index in range(4)
        ]
        for question in questions:
            question.pop("difficulty_origin")
        kept, _, _ = select_approved(questions, {"memory": 4})
        self.assertEqual(difficulty_balance(kept, {"memory": 4}), {})
        self.assertEqual(
            type_balance(kept, {"memory": 4})["memory"]["actual"],
            {"M1": 2, "M2": 2},
        )


if __name__ == "__main__":
    unittest.main()
