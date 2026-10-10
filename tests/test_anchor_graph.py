import unittest

from dialogue_benchmark.anchor_graph import (
    anchor_difficulty,
    build_anchor_subgraph,
    combine_anchor_group,
    difficulty_basis,
    filter_anchor_noise,
    merge_action_records,
)


class AnchorGraphTests(unittest.TestCase):
    def test_call_result_patch_are_one_action_and_do_not_link_call_to_result(self):
        records = [
            {"id": "call-1", "kind": "call", "call_id": "c1", "order": 1,
             "path": "service.py", "text": "run tests"},
            {"id": "result-1", "kind": "result", "call_id": "c1", "order": 2,
             "path": "service.py", "text": "validation passed: SLA changed"},
            {"id": "patch-1", "kind": "patch", "call_id": "c1", "order": 3,
             "changes": {"service.py": {"content": "new"}}, "text": "updated"},
        ]
        nodes = merge_action_records(records)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0]["id"], "action:c1")
        self.assertEqual(nodes[0]["record_ids"], ["call-1", "result-1", "patch-1"])
        graph = build_anchor_subgraph(
            {"id": "a1", "memory_kind": "M3", "focus": "SLA",
             "source_ids": ["result-1"], "used_by": ["patch-1"]}, records)
        self.assertEqual(graph["status"], "ready")
        self.assertEqual(len(graph["nodes"]), 1)
        self.assertEqual(graph["edges"], [])

    def test_noise_removes_receipts_directory_listing_and_repeated_pass(self):
        usable, noise = filter_anchor_noise([
            {"id": "e1", "text": "OK"},
            {"id": "e2", "text": "tests passed"},
            {"id": "e3", "text": "ls -la\nservice.py"},
            {"id": "e4", "text": "public rule: keep SLA"},
            {"id": "e5", "text": "public rule: keep SLA"},
        ])
        self.assertEqual([row["id"] for row in usable], ["e4"])
        self.assertEqual([row["reason"] for row in noise], [
            "success_receipt", "success_receipt", "directory_listing", "duplicate"])

    def test_source_usage_context_seed_and_explicit_relations(self):
        records = [
            {"id": "fact", "kind": "message", "order": 1, "path": "policy.py",
             "text": "customer policy"},
            {"id": "fix", "kind": "message", "order": 2, "path": "policy.py",
             "fix_of": "fact", "text": "corrected policy"},
            {"id": "other", "kind": "message", "order": 3, "path": "other.py",
             "corrects": "fix", "text": "unrelated correction"},
            {"id": "use", "kind": "message", "order": 4, "path": "consumer.py",
             "text": "consumer applies policy"},
        ]
        graph = build_anchor_subgraph(
            {"id": "a1", "memory_kind": "M1", "source_ids": ["fact"],
             "used_by": ["use"], "context_ids": ["fix"]}, records)
        relations = {(edge["relation"], edge["from"], edge["to"])
                     for edge in graph["edges"]}
        self.assertIn(("same_path", "fact", "fix"), relations)
        self.assertIn(("fix_chain", "fix", "fact"), relations)
        # ``other`` is not a seed and has no shared path with the selected
        # policy action, so an explicit correction outside the selected chain
        # cannot silently enter the graph.
        self.assertNotIn("other", {node["id"] for node in graph["nodes"]})
        self.assertEqual(graph["unresolved_ids"], [])

    def test_budget_stops_expansion_and_reports_it(self):
        records = [
            {"id": "source", "kind": "message", "order": 1, "path": "a.py",
             "text": "source fact"},
            {"id": "context", "kind": "message", "order": 2, "path": "a.py",
             "text": "large context " + ("x" * 300)},
        ]
        graph = build_anchor_subgraph(
            {"id": "a1", "source_ids": ["source"], "context_ids": ["context"]},
            records, max_chars=220)
        self.assertTrue(graph["budget"]["over_budget"])
        self.assertIn("source", {node["id"] for node in graph["nodes"]})
        self.assertNotIn("context", {node["id"] for node in graph["nodes"]})

    def test_group_distinguishes_required_and_supporting_without_keyword_rules(self):
        records = [
            {"id": "a", "kind": "message", "order": 1, "text": "rule A"},
            {"id": "b", "kind": "message", "order": 2, "text": "environment B"},
        ]
        result = combine_anchor_group([
            {"id": "a1", "memory_kind": "M1", "source_ids": ["a"]},
            {"id": "a2", "memory_kind": "M5", "source_ids": ["b"]},
        ], records, required_anchor_ids=["a1"],
        combination_reason="A chooses the route and B determines whether it is allowed")
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["required_anchor_ids"], ["a1"])
        self.assertEqual(result["supporting_anchor_ids"], ["a2"])
        self.assertEqual(result["memory_kinds"], ["M1"])
        self.assertIn("combination_reason", result)

    def test_difficulty_uses_required_facts_and_revision_not_memory_kind_count(self):
        self.assertEqual(anchor_difficulty(["f1"]), "easy")
        self.assertEqual(anchor_difficulty(["f1", "f2"]), "medium")
        self.assertEqual(anchor_difficulty(["f1", "f2"], revision=True), "hard")
        self.assertEqual(anchor_difficulty(["f1", "f2", "f3"]), "hard")
        self.assertEqual(anchor_difficulty(["f1"], cross_stage=True), "hard")
        self.assertEqual(anchor_difficulty([], complete=False), "unknown")
        basis = difficulty_basis([{"id": "f1"}, {"id": "f1"}], info_nodes=["n1"],
                                 revision=False, cross_stage=False)
        self.assertEqual(basis["difficulty"], "easy")
        self.assertEqual(basis["required_facts"], 1)
        self.assertEqual(basis["info_nodes"], 1)


if __name__ == "__main__":
    unittest.main()
