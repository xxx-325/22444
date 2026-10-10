import unittest

from dialogue_benchmark.anchor_graph import (
    anchor_difficulty,
    anchor_seeds,
    build_action_graph,
    combine_anchor_group,
    difficulty_basis,
    expand_anchor,
    keys,
)


class AnchorGraphTests(unittest.TestCase):
    def test_actions_are_compressed_and_receipts_are_noise(self):
        records = [
            {"id": "call-1", "kind": "call", "call_id": "c1", "order": 1,
             "path": "service.py", "text": "run tests"},
            {"id": "result-1", "kind": "result", "call_id": "c1", "order": 2,
             "path": "service.py", "text": "validation passed: SLA changed"},
            {"id": "patch-1", "kind": "patch", "call_id": "c1", "order": 3,
             "changes": {"service.py": {"content": "new"}}, "text": "updated"},
            {"id": "receipt", "kind": "result", "order": 4, "text": "OK"},
            {"id": "listing", "kind": "result", "order": 5,
             "text": "ls -la\nservice.py"},
        ]
        graph = build_action_graph(records)
        action = next(node for node in graph["nodes"] if node["id"] == "action:c1")
        self.assertEqual(action["record_ids"], ["call-1", "result-1", "patch-1"])
        self.assertEqual(len(graph["nodes"]), 1)
        self.assertEqual(
            {row["reason"] for row in graph["noise"]},
            {"success_receipt", "directory_listing"},
        )

    def test_graph_links_paths_and_public_user_corrections(self):
        records = [
            {"id": "user-1", "kind": "message", "role": "user", "order": 1,
             "text": "SLA policy uses 24 hours"},
            {"id": "call-1", "kind": "call", "call_id": "c1", "order": 2,
             "path": "policy.py", "text": "inspect SLA"},
            {"id": "result-1", "kind": "result", "call_id": "c1", "order": 3,
             "path": "policy.py", "text": "current value is 24 hours"},
            {"id": "user-2", "kind": "message", "role": "user", "order": 4,
             "text": "更正：SLA policy 改为 12 hours"},
        ]
        graph = build_action_graph(records)
        relations = {
            (edge["relation"], edge["from"], edge["to"])
            for edge in graph["edges"]
        }
        self.assertIn(("responds_to", "action:c1", "user-1"), relations)
        self.assertIn(("corrects", "user-2", "user-1"), relations)
        self.assertNotIn(("corrects", "action:c1", "user-1"), relations)

    def test_seed_and_expansion_keep_related_evidence_and_budget(self):
        records = [
            {"id": "source", "kind": "message", "order": 1, "path": "policy.py",
             "text": "SLA policy 24 hours"},
            {"id": "fix", "kind": "message", "order": 2, "path": "policy.py",
             "text": "SLA policy 改为 12 hours", "fix_of": "source"},
            {"id": "other", "kind": "message", "order": 3, "path": "other.py",
             "text": "unrelated policy"},
            {"id": "use", "kind": "message", "order": 4, "path": "consumer.py",
             "text": "consumer applies SLA policy"},
        ]
        graph = build_action_graph(records)
        event = {
            "id": "a1",
            "source_ids": ["source"],
            "used_by": ["use"],
            "focus": "SLA policy",
        }
        seeds = anchor_seeds(event, graph)
        self.assertEqual(seeds["anchor_id"], "a1")
        self.assertEqual(seeds["unresolved_ids"], [])
        expanded = expand_anchor(event, graph, budget_chars=8000)
        admitted_ids = {node["id"] for node in expanded["nodes"]}
        self.assertIn("source", admitted_ids)
        self.assertIn("fix", admitted_ids)
        self.assertNotIn("other", admitted_ids)
        self.assertEqual(expanded["anchor_id"], "a1")

        small = expand_anchor(event, graph, budget_chars=400)
        self.assertEqual(small["status"], "unrealizable:seed_over_budget")
        self.assertEqual(small["rejected"][0]["reason"], "seed_over_budget")

    def test_unresolved_seed_is_visible_to_group_review(self):
        records = [
            {"id": "source", "kind": "message", "text": "public rule"},
        ]
        graph = build_action_graph(records)
        event = {"id": "a1", "source_ids": ["missing"]}
        seeds = anchor_seeds(event, graph)
        self.assertEqual(seeds["unresolved_ids"], ["missing"])
        result = combine_anchor_group([event], records)
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["invalid_required_anchor_ids"], [])

    def test_group_distinguishes_required_and_supporting_without_type_rules(self):
        records = [
            {"id": "a", "kind": "message", "order": 1, "text": "rule A"},
            {"id": "b", "kind": "message", "order": 2, "text": "environment B"},
        ]
        result = combine_anchor_group(
            [
                {"id": "a1", "memory_kind": "M1", "source_ids": ["a"]},
                {"id": "a2", "memory_kind": "M5", "source_ids": ["b"]},
            ],
            records,
            required_anchor_ids=["a1"],
            combination_reason="A chooses the route and B determines whether it is allowed",
        )
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["required_anchor_ids"], ["a1"])
        self.assertEqual(result["supporting_anchor_ids"], ["a2"])
        self.assertEqual(result["memory_kinds"], ["M1"])
        self.assertIn("combination_reason", result)

    def test_difficulty_uses_required_evidence_not_memory_kind_count(self):
        self.assertEqual(anchor_difficulty(["f1"]), "easy")
        self.assertEqual(anchor_difficulty(["f1", "f2"]), "medium")
        self.assertEqual(anchor_difficulty(["f1", "f2"], revision=True), "hard")
        self.assertEqual(anchor_difficulty(["f1", "f2", "f3"]), "hard")
        self.assertEqual(anchor_difficulty(["f1"], cross_stage=True), "easy")
        self.assertEqual(anchor_difficulty([], complete=False), "unknown")
        basis = difficulty_basis(
            [{"id": "f1"}, {"id": "f1"}],
            info_nodes=["n1"],
            context_nodes=["n2"],
        )
        self.assertEqual(basis["difficulty"], "easy")
        self.assertEqual(basis["required_facts"], 1)
        self.assertEqual(basis["info_nodes"], 1)
        self.assertEqual(basis["context_nodes"], 1)

    def test_keys_are_stable_and_drop_common_prose(self):
        values = keys("Customer ticket update in service.py, SLA 24")
        self.assertIn("service.py", values)
        self.assertIn("sla", values)
        self.assertNotIn("customer", values)
        self.assertNotIn("ticket", values)


if __name__ == "__main__":
    unittest.main()
