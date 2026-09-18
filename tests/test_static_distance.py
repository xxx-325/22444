import unittest

from dialogue_benchmark.fact_index import (
    build_evidence_index,
    static_candidate_labels,
)


class StaticDistanceTests(unittest.TestCase):
    def setUp(self):
        self.scope = {
            "cutoff": 10,
            "dialogue": [
                {"id": "m", "kind": "message", "role": "user", "order": 4,
                 "stage_id": "s", "text": "请按反馈验证变更。"},
                {"id": "e1", "kind": "patch", "order": 1, "text": "v1"},
                {"id": "e2", "kind": "patch", "order": 2, "text": "v2"},
                {"id": "e3", "kind": "patch", "order": 3, "text": "v3"},
                {"id": "e4", "kind": "patch", "order": 4, "text": "v4"},
                {"id": "u", "kind": "message", "role": "user", "order": 10,
                 "stage_id": "u", "text": "无关旁枝。"},
            ],
            "events": [],
            "versions": [
                {"id": "v1", "path": "a.py", "observed_at": 1,
                 "source": "e1", "previous": None},
                {"id": "v2", "path": "a.py", "observed_at": 2,
                 "source": "e2", "previous": "v1"},
                {"id": "v3", "path": "a.py", "observed_at": 3,
                 "source": "e3", "previous": "v2"},
                {"id": "v4", "path": "a.py", "observed_at": 4,
                 "source": "e4", "previous": "v3"},
            ],
            "stages": [{"id": "s"}, {"id": "u"}],
            "edges": [], "historical_edges": [],
            "full_range_covered": True,
        }
        # The facts share m but carry distant version citations. Co-recording
        # must not create a one-hop v1->v4 shortcut.
        self.facts = [
            {"id": "f1", "statement": "用户反馈旧方案需要修改",
             "sources": ["m", "v1"]},
            {"id": "f2", "statement": "新版后来通过验证",
             "sources": ["m", "v4"]},
        ]
        self.index = build_evidence_index(
            self.facts, [self.scope], "code", 32000)
        self.group = {
            "qa_mode": "code",
            "relation_path": {"seed_node": "v1"},
        }

    def labels_for(self, *sources):
        candidate = {
            "answer_points": [{"text": "answer", "sources": list(sources)}],
            "forbidden_points": [],
        }
        return static_candidate_labels(
            self.group, candidate, self.index, "history_tracking")

    def test_version_distance_maps_to_provisional_difficulty(self):
        expected = {
            "v1": (0, "easy"),
            "v2": (1, "easy"),
            "v3": (2, "medium"),
            "v4": (3, "hard"),
        }
        for source, (distance, difficulty) in expected.items():
            with self.subTest(source=source):
                labels = self.labels_for(source)
                self.assertEqual(labels["difficulty_distance"], distance)
                self.assertEqual(labels["difficulty"], difficulty)

    def test_shared_citation_does_not_collapse_version_lineage(self):
        direct = [edge for edge in self.index["direct_graph"].get("v1", [])
                  if edge[0] == "v4"]
        self.assertFalse(direct)
        self.assertEqual(self.labels_for("v4")["difficulty_distance"], 3)

    def test_unreachable_source_is_unknown(self):
        labels = self.labels_for("u")
        self.assertIsNone(labels["difficulty_distance"])
        self.assertEqual(labels["difficulty"], "unknown")
        self.assertFalse(labels["relation_path_complete"])

    def test_unreferenced_branch_does_not_raise_difficulty(self):
        labels = self.labels_for("v2")
        self.assertEqual(labels["difficulty_distance"], 1)
        self.assertEqual(labels["difficulty"], "easy")
        self.assertNotIn("v4", labels["static_evidence_path"]["necessary_nodes"])


if __name__ == "__main__":
    unittest.main()
