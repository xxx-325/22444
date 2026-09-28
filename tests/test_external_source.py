import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dialogue_benchmark import cli
from dialogue_benchmark.external import (external_review_projection, filter_external_facts,
                                         load_external_scopes, external_usage_review)
from dialogue_benchmark.fact_index import build_evidence_index, static_evidence_check


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
                "id": "x1", "kind": "compatibility_contract", "memory_kind": "M1",
                "source_ids": ["e1"], "used_by": ["e3"], "qa_mode": "code",
            }]}), encoding="utf-8")
            loaded = load_external_scopes(path, self.records, 3, 32000)

        self.assertEqual(loaded["rejected"], [])
        self.assertEqual(len(loaded["scopes"]), 1)
        scope = loaded["scopes"][0]
        self.assertEqual(scope["external_event_id"], "x1")
        self.assertEqual(scope["evidence_group"]["target_types"],
                         ["M1"])
        self.assertEqual(scope["edges"], [])
        self.assertEqual(scope["versions"], [])
        self.assertEqual([row["id"] for row in scope["dialogue"]], ["e1", "e2", "e3"])

    def test_external_event_keeps_explicit_context_and_later_public_messages(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "external-events.json"
            path.write_text(json.dumps({"version": 1, "events": [{
                "id": "x1", "kind": "compatibility_contract", "memory_kind": "M1",
                "source_ids": ["e1"], "used_by": ["e3"],
                "context_ids": ["e2"], "qa_mode": "code",
            }]}), encoding="utf-8")
            loaded = load_external_scopes(path, self.records, 3, 32000)

        self.assertEqual([row["id"] for row in loaded["scopes"][0]["dialogue"]],
                         ["e1", "e2", "e3"])

    def test_task_rules_and_later_scoped_correction_share_one_closed_group(self):
        records = [*self.records,
                   {"id": "e4", "order": 4, "kind": "message", "role": "user",
                    "text": "仅客户 A 的 note 改为省略，其他客户仍保留空值。"},
                   {"id": "e5", "order": 5, "kind": "message", "role": "user",
                    "text": "客户 A 的其他空字段继续保留。"},
                   {"id": "e6", "order": 6, "kind": "message", "role": "user",
                    "text": "无关的页面样式任务。"}]
        events = [
            {"id": "old", "kind": "compatibility_contract", "memory_kind": "M1",
             "task_id": "export", "source_ids": ["e1"], "used_by": ["e3"]},
            {"id": "new", "kind": "user_correction", "memory_kind": "M6",
             "task_id": "export-update", "source_ids": ["e4"], "supersedes": ["old"]},
            {"id": "exception", "kind": "compatibility_contract", "memory_kind": "M1",
             "task_id": "export-update", "source_ids": ["e5"]},
            {"id": "unrelated", "kind": "external_observation", "memory_kind": "M2",
             "task_id": "styling", "source_ids": ["e6"]}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.json"
            path.write_text(json.dumps({"version": 1, "events": events}))
            loaded = load_external_scopes(path, records, 6, max_groups=1)
        self.assertEqual(len(loaded["scopes"]), 1)
        scope = loaded["scopes"][0]
        self.assertEqual(scope["external_event_ids"], ["old", "new", "exception"])
        self.assertEqual(scope["external_source_ids"], ["e1", "e4", "e5"])
        self.assertEqual(scope["memory_kinds"], ["M1", "M6"])
        self.assertEqual([r["id"] for r in scope["dialogue"]], ["e1", "e2", "e3", "e4", "e5", "e6"])
        self.assertEqual(loaded["rejected"], [{"id": "unrelated", "reason": "external_group_budget"}])

    def test_scope_and_review_share_later_public_correction_without_a_planned_event_link(self):
        records = [*self.records,
            dict(id="later-code", order=4, kind="message", role="assistant",
                 text="Should the rule become optional?"),
            dict(id="later-user", order=5, kind="message", role="user",
                 text="Yes. Apply it only when explicitly requested."),
            dict(id="unrelated-tool", order=6, kind="result", text="Unrelated output"),
            dict(id="future", order=7, kind="message", role="user", text="After cutoff")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.json"
            path.write_text(json.dumps({"version": 1, "events": [dict(
                id="rule", kind="compatibility_contract", memory_kind="M1",
                task_id="first", source_ids=["e1"], used_by=["e3"])]}))
            scope, = load_external_scopes(path, records, 6)["scopes"]
        self.assertEqual([row["id"] for row in scope["dialogue"]],
                         ["e1", "e2", "e3", "later-code", "later-user"])
        self.assertEqual(scope["external_source_ids"], ["e1"])
        candidate = dict(id="q1", question="Which rule must future work follow?",
                         answer_points=[dict(text="Always keep nulls.", sources=["e1"])])
        group, audit = external_review_projection(dict(scope=scope, facts=[]), candidate)
        self.assertTrue(audit["complete"])
        self.assertEqual([row["id"] for row in group["scope"]["dialogue"]],
                         ["e1", "e2", "e3", "later-code", "later-user"])
        self.assertEqual(scope["dialogue"], group["scope"]["dialogue"])
        from dialogue_benchmark.llm import _evidence_review_request
        prompt, payload, refs = _evidence_review_request(
            group["scope"], group["scope"]["review_guard_sources"], [], candidate)
        self.assertIn("later-user", refs.values())
        self.assertIn("only when explicitly requested", str(payload))
        self.assertIn("earlier confirmed rule is stale", prompt)
        self.assertNotIn("Unrelated output", str(payload))
        self.assertNotIn("After cutoff", str(payload))

    def test_confirmation_and_observation_have_distinct_evidence(self):
        scope = {"dialogue": self.records, "external_source_ids": ["e1", "e2"],
                 "external_usage_ids": ["e2", "e3"]}
        for value, expected in [("confirmed@资料1", "confirmed"),
                                ("confirmed@资料2", "uncertain"),
                                ("applied@资料2", "uncertain"),
                                ("applied@资料3", "applied")]:
            with self.subTest(value=value):
                _, decision = external_usage_review(
                    {"reviews": [{"usage": value, "usage_reason": "Concrete evidence"}]},
                    scope, {"资料1": "e1", "资料2": "e2", "资料3": "e3"})
                self.assertEqual(decision["status"], expected)

    def test_invocation_alone_does_not_prove_application(self):
        scope = {"dialogue": [dict(id="c", kind="call", text="export --timeout 7"),
                               dict(id="r", kind="result", text="Permission denied")],
                 "external_usage_ids": ["c", "r"]}
        _, decision = external_usage_review(
            {"reviews": [{"usage": "applied@资料1", "usage_reason": "The command used the option"}]},
            scope, {"资料1": "c", "资料2": "r"})
        self.assertEqual(decision["status"], "uncertain")

    def test_later_public_user_correction_can_supply_confirmation(self):
        correction = dict(id="later", kind="message", role="user",
                          text="Only client A omits empty notes; other clients retain them.")
        scope = {"external_source_ids": ["e1"], "external_usage_ids": ["e3"],
                 "dialogue": [*self.records, correction]}
        _, decision = external_usage_review(
            {"reviews": [{"usage": "confirmed@correction", "usage_reason": "User narrowed the same rule."}]},
            scope, {"correction": "later"})
        self.assertEqual(decision["status"], "confirmed")
        self.assertEqual(decision["sources"], ["later"])
        scope["dialogue"] = self.records
        _, absent = external_usage_review(
            {"reviews": [{"usage": "confirmed@correction", "usage_reason": "Not in scope."}]},
            scope, {"correction": "later"})
        self.assertEqual(absent["status"], "uncertain")

    def test_all_memory_classes_keep_their_provenance(self):
        kinds = ["user_correction", "environment_observation", "perturbation_revealed",
                 "failure_avoidance", "environment_observation", "user_correction"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "external-events.json"
            path.write_text(json.dumps({"version": 1, "events": [
                {"id": "x" + str(i), "kind": kind, "memory_kind": "M" + str(i),
                 "source_ids": ["e1"], "used_by": ["e3"], "qa_mode": "code"}
                for i, kind in enumerate(kinds, 1)]}))
            result = load_external_scopes(path, self.records, 3)
        self.assertEqual(result["rejected"], [])
        self.assertEqual({s["memory_kind"] for s in result["scopes"]},
                         {"M" + str(i) for i in range(1, 7)})
        m4 = next(s for s in result["scopes"] if s["memory_kind"] == "M4")
        self.assertEqual(m4["evidence_group"]["target_types"], ["M4"])

    def test_one_external_event_is_processed_once_without_track_routing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "external-events.json"
            for routing in ({}, {"qa_mode": "both"}, {"qa_mode": "code"}, {"qa_mode": "general"}):
                path.write_text(json.dumps({"version": 1, "events": [{
                    "id": "x1", "kind": "compatibility_contract", "memory_kind": "M1",
                    "source_ids": ["e1"], "used_by": ["e3"], **routing}]}))
                result = load_external_scopes(path, self.records, 3)
                self.assertEqual([s["track"] for s in result["scopes"]], ["memory"])
                self.assertEqual(len(result["events"]), 1)
            result = load_external_scopes(path, self.records, 3, max_groups=0)
            self.assertEqual(result["scopes"], [])
            self.assertEqual(result["rejected"][0]["reason"], "external_group_budget")

    def test_missing_or_unknown_memory_type_is_not_guessed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "external-events.json"
            for memory_kind in (None, "M7", "external_state_application"):
                path.write_text(json.dumps({"version": 1, "events": [{
                    "id": "x1", "kind": "external_observation", "memory_kind": memory_kind,
                    "source_ids": ["e1"], "used_by": ["e3"]}]}))
                result = load_external_scopes(path, self.records, 3)
                self.assertEqual(result["scopes"], [])
                self.assertEqual(result["rejected"][0]["reason"], "invalid_memory_kind")

    def test_failure_avoidance_reaches_precheck_with_closed_answer_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "external-events.json"
            path.write_text(json.dumps({"version": 1, "events": [{
                "id": "x4", "kind": "failure_avoidance", "memory_kind": "M4",
                "source_ids": ["e1"], "used_by": ["e3"], "qa_mode": "both"}]}))
            loaded = load_external_scopes(path, self.records, 3)
        facts = [{"id": "f1", "sources": ["e1"], "statement": self.records[0]["text"]}]
        for scope in loaded["scopes"]:
            with self.subTest(track=scope["track"]):
                index = build_evidence_index(facts, [scope], scope["track"])
                group = {"scope": scope, "facts": facts, "qa_mode": scope["track"]}
                for candidate in (None, {"answer_points": [{
                        "text": facts[0]["statement"], "sources": ["e1"]}]}):
                    check = static_evidence_check(group, index, "M4", candidate)
                    self.assertEqual(check["status"], "supported")
                bad = {"answer_points": [{"text": "Unprovided result", "sources": ["e99"]}]}
                check = static_evidence_check(group, index, "M4", bad)
                self.assertEqual(check["reason"], "answer_source_out_of_scope")
                check = static_evidence_check(group, index, "M6")
                self.assertEqual(check["reason"], "external_type_mismatch")

    def test_m6_decision_and_correction_keep_distinct_answer_targets(self):
        variants = [
            ("external_observation", "M6"),
            ("user_correction", "M6"),
        ]
        for kind, target in variants:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "external-events.json"
                path.write_text(json.dumps({"version": 1, "events": [{
                    "id": "x6", "kind": kind, "memory_kind": "M6",
                    "source_ids": ["e1"], "used_by": ["e3"], "qa_mode": "both"}]}))
                loaded = load_external_scopes(path, self.records, 3)
                self.assertEqual(loaded["rejected"], [])
                self.assertEqual(len(loaded["scopes"]), 1)
                for scope in loaded["scopes"]:
                    self.assertEqual(scope["external_kind"], kind)
                    self.assertEqual(scope["memory_kind"], "M6")
                    self.assertEqual(scope["evidence_group"]["target_types"], [target])
                    facts = [{"id": "f1", "sources": ["e1"],
                              "statement": self.records[0]["text"]}]
                    index = build_evidence_index(facts, [scope], scope["track"])
                    group = {"scope": scope, "facts": facts, "qa_mode": scope["track"]}
                    self.assertEqual(static_evidence_check(group, index, target)["status"],
                                     "supported")

    def test_usage_requires_a_cited_public_result(self):
        scope = {"external_usage_ids": ["e3"], "dialogue": self.records}
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
                              ("applied@资料2", "needs_review"),
                              ("confirmed@资料1", "approved")]:
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

    def test_positive_usage_without_citation_is_a_format_failure(self):
        for value in ('applied', 'confirmed'):
            with self.subTest(value=value):
                _, decision = external_usage_review(
                    {"reviews": [{"usage": value, "usage_reason": "资料3 shows use"}]},
                    {"dialogue": self.records, "external_usage_ids": ["e3"]}, {"资料3": "e3"})
                self.assertEqual(decision["status"], "uncertain")
                self.assertEqual(decision["reason"], "invalid_usage_evidence")
                self.assertEqual(decision["sources"], [])

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

    def test_public_disclosure_does_not_require_prior_application(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "external-events.json"
            path.write_text(json.dumps({"version": 1, "events": [
                {"id": "missing-use", "kind": "user_correction", "memory_kind": "M6",
                 "source_ids": ["e1"], "used_by": []},
                {"id": "bad-source", "kind": "user_correction", "memory_kind": "M6",
                 "source_ids": ["e99"], "used_by": ["e3"]},
            ]}), encoding="utf-8")
            loaded = load_external_scopes(path, self.records, 3)

        reasons = {row["id"]: row["reason"] for row in loaded["rejected"]}
        self.assertIn("bad-source", reasons)
        self.assertEqual(len(loaded["scopes"]), 1)
        self.assertEqual(loaded["scopes"][0]["external_source_ids"], ["e1"])
        self.assertEqual(loaded["scopes"][0]["external_usage_ids"], [])

    def test_external_event_requires_usage_after_public_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "external-events.json"
            path.write_text(json.dumps({"version": 1, "events": [
                {"id": "same-order", "kind": "user_correction", "memory_kind": "M6",
                 "source_ids": ["e1"], "used_by": ["e1"]},
                {"id": "earlier-use", "kind": "user_correction", "memory_kind": "M6",
                 "source_ids": ["e2"], "used_by": ["e1"]},
            ]}), encoding="utf-8")
            loaded = load_external_scopes(path, self.records, 3)

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
                "id": "x1", "kind": "verification_result", "memory_kind": "M5",
                "source_ids": ["e1"], "used_by": ["e3"], "qa_mode": "code",
            }]}), encoding="utf-8")
            output = directory / "run"
            with patch.object(cli, "build_graph", side_effect=AssertionError("graph built")):
                status = cli.main([
                    str(dialogue), "--output", str(output), "--qa-count", "5",
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
