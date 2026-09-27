import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dialogue_benchmark import cli
from dialogue_benchmark.external import (external_review_projection, filter_external_facts,
                                         load_external_scopes)


class ExternalSourceTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"id": "e1", "order": 1, "kind": "message", "role": "user",
             "text": "旧客户端必须保留空 note 的兼容规则", "source_kind": "conversation"},
            {"id": "e2", "order": 2, "kind": "message", "role": "assistant",
             "text": "我会在导出层检查这个约定", "source_kind": "conversation"},
            {"id": "e3", "order": 3, "kind": "result", "text": "下游测试发现其他空字段不能删除",
             "source_kind": "test"},
        ]

    def test_external_event_builds_scope_without_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "external-events.json"
            path.write_text(json.dumps({"version": 1, "events": [{
                "id": "x1", "kind": "compatibility_contract",
                "source_ids": ["e1"], "used_by": ["e3"], "qa_mode": "code",
            }]}), encoding="utf-8")
            loaded = load_external_scopes(path, self.records, 3, {"code"}, 32000)

        self.assertEqual(loaded["rejected"], [])
        self.assertEqual(len(loaded["scopes"]), 1)
        scope = loaded["scopes"][0]
        self.assertEqual(scope["external_event_id"], "x1")
        self.assertEqual(scope["evidence_group"]["target_types"],
                         ["compatibility_preservation"])
        self.assertEqual(scope["edges"], [])
        self.assertEqual(scope["versions"], [])
        self.assertEqual([row["id"] for row in scope["dialogue"]], ["e1", "e3"])

    def test_external_event_accepts_only_explicit_context_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "external-events.json"
            path.write_text(json.dumps({"version": 1, "events": [{
                "id": "x1", "kind": "compatibility_contract",
                "source_ids": ["e1"], "used_by": ["e3"],
                "context_ids": ["e2"], "qa_mode": "code",
            }]}), encoding="utf-8")
            loaded = load_external_scopes(path, self.records, 3, {"code"}, 32000)

        self.assertEqual([row["id"] for row in loaded["scopes"][0]["dialogue"]],
                         ["e1", "e2", "e3"])

    def test_external_review_keeps_declared_context_without_code_symbols(self):
        scope = {"external_event_id": "x1", "external_source_ids": ["e1"],
                 "external_usage_ids": ["e3"], "dialogue": self.records,
                 "cutoff": 3, "events": [], "versions": [], "model_request_chars": 32000}
        group = {"scope": scope, "facts": [{"id": "f1", "sources": ["e1"],
                                             "statement": "旧客户端必须保留空 note"}]}
        candidate = {"id": "q1", "question": "旧客户端的空 note 如何处理？",
                     "answer_points": [{"text": "保留空 note。", "sources": ["e1"]}]}
        projected, audit = external_review_projection(group, candidate)
        self.assertTrue(audit["complete"])
        self.assertEqual(projected["scope"]["review_guard_sources"], ["e1", "e2", "e3"])
        self.assertNotIn("review_guard_complete", scope)

        candidate["answer_points"][0]["sources"] = ["unprovided"]
        projected, audit = external_review_projection(group, candidate)
        self.assertIsNone(projected)
        self.assertFalse(audit["complete"])
        candidate["answer_points"][0]["sources"] = ["e1"]
        scope["model_request_chars"] = 10
        projected, audit = external_review_projection(group, candidate)
        self.assertIsNone(projected)
        self.assertEqual(audit["reason"], "candidate_guard_over_budget")

    def test_external_event_requires_public_source_and_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "external-events.json"
            path.write_text(json.dumps({"version": 1, "events": [
                {"id": "missing-use", "kind": "user_correction",
                 "source_ids": ["e1"], "used_by": []},
                {"id": "bad-source", "kind": "user_correction",
                 "source_ids": ["e99"], "used_by": ["e3"]},
            ]}), encoding="utf-8")
            loaded = load_external_scopes(path, self.records, 3, {"code", "general"})

        reasons = {row["id"]: row["reason"] for row in loaded["rejected"]}
        self.assertIn("missing-use", reasons)
        self.assertIn("bad-source", reasons)
        self.assertFalse(loaded["scopes"])

    def test_external_event_requires_usage_after_public_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "external-events.json"
            path.write_text(json.dumps({"version": 1, "events": [
                {"id": "same-order", "kind": "user_correction",
                 "source_ids": ["e1"], "used_by": ["e1"]},
                {"id": "earlier-use", "kind": "user_correction",
                 "source_ids": ["e2"], "used_by": ["e1"]},
            ]}), encoding="utf-8")
            loaded = load_external_scopes(path, self.records, 3, {"code", "general"})

        reasons = {row["id"]: row["reason"] for row in loaded["rejected"]}
        self.assertEqual(reasons["same-order"], "usage_not_after_source")
        self.assertEqual(reasons["earlier-use"], "usage_not_after_source")

    def test_only_facts_grounded_in_declared_external_sources_survive(self):
        scope = {"external_event_id": "x1", "external_source_ids": ["e1"]}
        facts = [
            {"id": "f1", "sources": ["e1"], "statement": "rule"},
            {"id": "f2", "sources": ["e3"], "statement": "test"},
        ]
        kept = filter_external_facts(facts, [scope])
        self.assertEqual([fact["id"] for fact in kept], ["f1"])
        self.assertEqual(kept[0]["external_event_ids"], ["x1"])

    def test_external_cli_skips_graph_in_static_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            dialogue = directory / "dialogue.json"
            dialogue.write_text(json.dumps({"version": 1, "records": [
                {"kind": "message", "role": "user", "text": "old rule"},
                {"kind": "message", "role": "assistant", "text": "noted"},
                {"kind": "result", "text": "test observed"},
            ]}), encoding="utf-8")
            events = directory / "external-events.json"
            events.write_text(json.dumps({"version": 1, "events": [{
                "id": "x1", "kind": "verification_result",
                "source_ids": ["e1"], "used_by": ["e3"], "qa_mode": "code",
            }]}), encoding="utf-8")
            output = directory / "run"
            with patch.object(cli, "build_graph", side_effect=AssertionError("graph built")):
                status = cli.main([
                    str(dialogue), "--output", str(output), "--qa-mode", "code",
                    "--qa-source", "external", "--external-events", str(events),
                ])

            self.assertEqual(status, 0)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["qa_source"], "external")
            self.assertEqual(manifest["external_event_count"], 1)
            graph = json.loads((output / "graph.json").read_text())
            self.assertEqual(graph["mode"], "external")


if __name__ == "__main__":
    unittest.main()
