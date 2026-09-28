import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dialogue_benchmark import cli
from dialogue_benchmark.external import (external_review_projection, filter_external_facts,
                                         load_external_scopes, external_usage_review)


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

    def test_all_memory_classes_keep_their_provenance(self):
        kinds = ["user_correction", "environment_observation", "perturbation_revealed",
                 "failure_avoidance", "environment_observation", "user_correction"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "external-events.json"
            path.write_text(json.dumps({"version": 1, "events": [
                {"id": "x" + str(i), "kind": kind, "memory_kind": "M" + str(i),
                 "source_ids": ["e1"], "used_by": ["e3"], "qa_mode": "code"}
                for i, kind in enumerate(kinds, 1)]}))
            result = load_external_scopes(path, self.records, 3, {"code"})
        self.assertEqual(result["rejected"], [])
        self.assertEqual({s["memory_kind"] for s in result["scopes"]},
                         {"M" + str(i) for i in range(1, 7)})
        m4 = next(s for s in result["scopes"] if s["memory_kind"] == "M4")
        self.assertEqual(m4["evidence_group"]["target_types"], ["failure_avoidance"])

    def test_one_external_event_can_supply_both_independent_tracks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "external-events.json"
            path.write_text(json.dumps({"version": 1, "events": [{
                "id": "x1", "kind": "compatibility_contract", "memory_kind": "M1",
                "source_ids": ["e1"], "used_by": ["e3"], "qa_mode": "both"}]}))
            for enabled in ({"code", "general"}, {"general"}):
                result = load_external_scopes(path, self.records, 3, enabled)
                self.assertEqual({s["track"] for s in result["scopes"]}, enabled)
                self.assertEqual(len(result["events"]), 1)
            result = load_external_scopes(path, self.records, 3, {"code", "general"},
                                          max_groups={"general": 0, "code": 1})
            self.assertEqual([s["track"] for s in result["scopes"]], ["code"])
            self.assertEqual(len(result["events"]), 1)

    def test_usage_requires_a_cited_public_action_or_result(self):
        scope = {"external_usage_ids": ["e3"]}
        for value, expected in [("not_applied", "not_applied"), ("uncertain", "uncertain"),
                                ("applied@资料1", "uncertain"),
                                ("applied@资料3,unknown", "uncertain"),
                                ("applied@资料3", "applied"), ([], "uncertain")]:
            with self.subTest(value=value):
                original = {"reviews": [{"usage": value, "usage_reason": "Observed result",
                                         "point_evidence": "A1=supported@e1"}]}
                clean, decision = external_usage_review(
                    original, scope, {"资料1": "e1", "资料3": "e3"})
                self.assertEqual(decision["status"], expected)
                self.assertEqual(clean["reviews"], [{"point_evidence": "A1=supported@e1"}])
                self.assertIn("usage", original["reviews"][0])

    def test_external_usage_determines_publication_without_changing_answer_evidence(self):
        from dialogue_benchmark.llm import review_candidates
        from tests.test_simple_review import SimpleReviewTests, TextClient
        fixture = SimpleReviewTests()
        fixture.setUp()
        fixture.scope.update(external_event_id="x1", external_source_ids=["m1"],
                             external_usage_ids=["m2"])
        for usage, status in [("not_applied", "needs_review"),
                              ("applied@资料2", "approved")]:
            with self.subTest(usage=usage):
                client = TextClient([fixture.atomicity(),
                    "REVIEW q1\nreview_contract: simple_v1\ncompleteness: complete\nEND_REVIEW",
                    "REVIEW q1\nreview_contract: simple_v1\n"
                    "point_evidence: A1=supported@资料1;F1=contradicted@资料1\n"
                    "usage: " + usage + "\nusage_reason: Public result confirms use\nEND_REVIEW"])
                result = review_candidates(fixture.scope, fixture.facts, [fixture.candidate],
                                           client, qa_mode="general", review_mode="simple",
                                           allow_repair=False)
                self.assertEqual(result["stage_errors"], [])
                self.assertEqual(result["questions"][0]["status"], status)
                self.assertIn("historical_use", client.payloads[-1])

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
