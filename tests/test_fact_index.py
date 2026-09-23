import unittest

from dialogue_benchmark.fact_index import (
    build_evidence_groups,
    build_evidence_index,
    coverage_report,
    expand_evidence_group_once,
    merge_scopes,
    static_candidate_labels,
)


class FactIndexTests(unittest.TestCase):
    @staticmethod
    def _version_scope(count=4):
        return {
            "cutoff": count,
            "dialogue": [
                {"id": "e%d" % index, "order": index, "kind": "patch",
                 "text": "config.py version %d" % index}
                for index in range(1, count + 1)
            ],
            "events": [],
            "versions": [
                {"id": "v%d" % index, "path": "config.py",
                 "observed_at": index, "source": "e%d" % index,
                 "previous": "v%d" % (index - 1) if index > 1 else None,
                 "status": "known"}
                for index in range(1, count + 1)
            ],
            "edges": [], "historical_edges": [],
        }

    def test_fact_with_old_and_new_citations_does_not_shortcut_lineage(self):
        fact = {"id": "f1", "qa_mode": "code",
                "statement": "config.py changed across recorded versions",
                "sources": ["v1", "v4"]}
        index = build_evidence_index([fact], [self._version_scope()], "code")
        group = {
            "qa_mode": "code",
            "relation_path": {"seed_node": "v1"},
        }
        candidate = {"answer_points": [{"text": "old and new behavior",
                                         "sources": ["v1", "v4"]}]}
        labels = static_candidate_labels(
            group, candidate, index, "correction_update")

        self.assertEqual(labels["difficulty_distance"], 3)
        self.assertEqual(labels["difficulty"], "hard")
        self.assertFalse(any(
            neighbor == "v4" and distance < 3
            for neighbor, distance, _ in index["direct_graph"]["v1"]
        ))

    def test_source_only_version_expansion_walks_current_lineage_frontier(self):
        fact = {"id": "f4", "qa_mode": "code",
                "statement": "config.py must preserve the ValueError behavior",
                "sources": ["v4"]}
        index = build_evidence_index([fact], [self._version_scope()], "code")
        groups = build_evidence_groups(
            [fact], [self._version_scope()], "code", {"constraint_followthrough"}, 1,
            evidence_index=index)
        self.assertEqual(len(groups), 1)

        expanded, audit = expand_evidence_group_once(
            groups[0], index, "earlier_state", "config.py")

        self.assertEqual(audit["status"], "expanded")
        self.assertEqual(audit["relation"], "version_previous")
        self.assertTrue(expanded["scope"]["generation_extra_sources"])
        self.assertEqual(len(expanded["facts"]), 1)
        repeated, repeated_audit = expand_evidence_group_once(
            expanded, index, "earlier_state", "config.py")
        self.assertEqual(repeated_audit["status"], "expanded")
        self.assertIn("v2", repeated["scope"]["generation_extra_sources"])
        repeated_again, repeated_again_audit = expand_evidence_group_once(
            repeated, index, "earlier_state", "config.py")
        self.assertEqual(repeated_again_audit["status"], "expanded")
        self.assertIn("v1", repeated_again["scope"]["generation_extra_sources"])
        exhausted, exhausted_audit = expand_evidence_group_once(
            repeated_again, index, "earlier_state", "config.py")
        self.assertIsNone(exhausted)
        self.assertEqual(exhausted_audit["reason"], "no_new_evidence")

    def test_expansion_skips_duplicate_pointer_and_finds_new_evidence(self):
        fact = {"id": "f4", "qa_mode": "code",
                "statement": "config.py must preserve the ValueError behavior",
                "sources": ["v4"]}
        index = build_evidence_index([fact], [self._version_scope()], "code")
        group = build_evidence_groups(
            [fact], [self._version_scope()], "code", {"constraint_followthrough"}, 1,
            evidence_index=index)[0]
        group["expansion_pointer"] = {
            "attempted": False,
            "candidates": [
                {"missing_kind": "earlier_state", "relation": "version_previous",
                 "fact_ids": ["f4"], "source_ids": ["v4"], "distance": 0},
                {"missing_kind": "earlier_state", "relation": "version_previous",
                 "fact_ids": [], "source_ids": ["v3"], "distance": 1},
            ],
        }

        expanded, audit = expand_evidence_group_once(
            group, index, "earlier_state", "config.py")

        self.assertEqual(audit["status"], "expanded")
        self.assertIn("v3", expanded["scope"]["generation_extra_sources"])

    def test_call_result_expansion_over_budget_is_audited_not_generated(self):
        scope = {
            "cutoff": 2,
            "dialogue": [
                {"id": "call", "order": 1, "kind": "call", "call_id": "c1",
                 "text": "run check"},
                {"id": "result", "order": 2, "kind": "result", "call_id": "c1",
                 "text": "check result " + "x" * 19990},
            ],
            "events": [], "versions": [], "edges": [], "historical_edges": [],
        }
        fact = {"id": "fc", "qa_mode": "code", "statement": "check must be called",
                "sources": ["call"]}
        index = build_evidence_index([fact], [scope], "code", 6000)
        group = build_evidence_groups(
            [fact], [scope], "code", {"constraint_followthrough"}, 1, 2000, 6000,
            evidence_index=index)[0]

        expanded, audit = expand_evidence_group_once(
            group, index, "outcome", "check", target_chars=2000,
            max_chars=6000)

        self.assertIsNone(expanded)
        self.assertEqual(audit["status"], "over_budget")
        self.assertEqual(audit["reason"], "all_matching_expansions_over_budget")

    def test_final_projection_dedup_continues_to_new_groups(self):
        facts = [{"id": "f%d" % i, "qa_mode": "general",
                  "statement": "config.py requires setting%d" % i, "sources": ["e1"]}
                 for i in range(1, 4)]
        facts.append({"id": "f4", "qa_mode": "general",
                      "statement": "必须兼容 runner.py requires a timeout", "sources": ["e2"]})
        scope = {"cutoff": 2, "dialogue": [
            {"id": "e1", "kind": "message", "order": 1, "stage_id": "s1", "text": "config.py requirements"},
            {"id": "e2", "kind": "message", "order": 2, "stage_id": "s2", "text": "runner.py timeout"}],
            "events": [], "versions": [], "edges": [], "stages": []}
        groups = build_evidence_groups(facts, [scope], "general", {"constraint_followthrough"}, 2)
        signatures = [frozenset(f["id"] for f in g["facts"]) for g in groups]
        self.assertEqual(len(signatures), len(set(signatures)))
        self.assertEqual(len(groups), 2)
        self.assertTrue(any("f4" in key for key in signatures))

    def test_unsent_groups_are_only_in_pool_coverage(self):
        facts = [{"id": "f1", "qa_mode": "code", "sources": ["e1"]},
                 {"id": "f2", "qa_mode": "code", "sources": ["e2"]}]
        groups = [{"id": "g%d" % i, "qa_mode": "code", "facts": [fact],
                   "scope": {"dialogue": [{"id": "e%d" % i}]}}
                  for i, fact in enumerate(facts, 1)]
        report = coverage_report([], [], {}, facts, groups, [], [], ["g1"])["code"]
        self.assertEqual(report["facts"]["group_coverage"], .5)
        self.assertEqual(report["candidate_pool"]["facts"]["group_coverage"], 1)
        self.assertEqual(report["groups"]["attempted"], 1)

    def test_cross_chunk_general_facts_form_small_multi_hop_group(self):
        scopes = [
            {
                "cutoff": 100,
                "dialogue": [
                    {"id": "e1", "order": 1, "kind": "message", "role": "user",
                     "stage_id": "stage-1", "text": "配置读取需要继续支持 yaml。"},
                    {"id": "near-1", "order": 2, "kind": "message", "role": "assistant",
                     "stage_id": "stage-1", "text": "一段无关讨论。"},
                    {"id": "noise", "order": 40, "kind": "message", "role": "assistant",
                     "stage_id": "stage-1", "text": "另一段无关讨论。"},
                    {"id": "near-2", "order": 80, "kind": "message", "role": "assistant",
                     "stage_id": "stage-1", "text": "稍后的无关讨论。"},
                ],
                "stages": [{"id": "stage-1", "record_ids": ["e1", "near-1", "noise", "near-2"],
                            "start_order": 1, "end_order": 80}],
                "events": [], "versions": [], "edges": [], "historical_edges": [],
            },
            {
                "cutoff": 100,
                "dialogue": [
                    {"id": "e90", "order": 90, "kind": "message", "role": "user",
                     "stage_id": "stage-9", "text": "yaml 配置报错时不要自动退回默认值。"},
                ],
                "stages": [{"id": "stage-9", "record_ids": ["e90"],
                            "start_order": 90, "end_order": 90}],
                "events": [], "versions": [], "edges": [], "historical_edges": [],
            },
        ]
        facts = [
            {"id": "f1", "statement": "用户要求配置读取支持 yaml", "sources": ["e1"]},
            {"id": "f2", "statement": "用户要求 yaml 报错时不回退", "sources": ["e90"]},
        ]
        groups = build_evidence_groups(
            facts, scopes, "general", {"constraint_followthrough"}, 4, 8000, 16000)
        cross = [group for group in groups if len(group["facts"]) == 2]
        self.assertTrue(cross)
        self.assertEqual(cross[0]["max_questions"], 2)
        self.assertEqual(cross[0]["scope"]["evidence_group"]["stage_count"], 2)
        self.assertEqual(cross[0]["scope"]["evidence_group"]["target_types"], ["constraint_followthrough"])
        self.assertNotIn("noise", {record["id"] for record in cross[0]["scope"]["dialogue"]})

    def test_coverage_separates_scope_fact_group_and_question_coverage(self):
        records = [{"id": "e1", "kind": "message"}, {"id": "e2", "kind": "result"},
                   {"id": "e3", "kind": "message"}]
        scopes = {"code": [{"dialogue": records[:2]},
                           {"dialogue": records[:2], "facts_failed": True}]}
        facts = [{"id": "f1", "qa_mode": "code", "sources": ["e1"]},
                 {"id": "f2", "qa_mode": "code", "sources": ["e2"]}]
        groups = [{"id": "g1", "qa_mode": "code", "facts": facts[:1],
                   "scope": {"dialogue": records[:1]}}]
        report = coverage_report(records, [{"id": "v1"}], scopes, facts, groups,
                                 [{"track": "code", "phase": "qa_review", "generated": 2,
                                   "approved": 1, "needs_review": 1}],
                                 [{"qa_mode": "code", "evidence_group_id": "g1"}])["code"]
        self.assertEqual(report["sources"]["eligible"], 4)
        self.assertEqual(report["sources"]["in_scopes"], 2)
        self.assertEqual(report["sources"]["uncovered_ids"], ["e3", "v1"])
        self.assertEqual(report["facts"]["unused_ids"], ["f2"])
        self.assertEqual(report["facts"]["group_coverage"], .5)
        self.assertEqual(report["chunks"]["facts_completed"], 1)
        self.assertEqual(report["groups"]["generated"], 1)
        self.assertEqual(report["groups"]["published"], 1)
        self.assertIn("objects", report)
        self.assertIn("relations", report)
        self.assertIn("versions", report)

    def test_code_failure_change_and_validation_can_form_three_fact_group(self):
        scopes = [{
            "cutoff": 30,
            "dialogue": [
                {"id": "e1", "order": 1, "kind": "result",
                 "text": "query_bridge.py dtype mismatch error"},
                {"id": "e10", "order": 10, "kind": "patch", "success": True,
                 "changes": {"query_bridge.py": {"type": "update",
                                                  "unified_diff": "+cast_cache"}}},
                {"id": "e20", "order": 20, "kind": "result",
                 "text": "query_bridge.py pytest passed"},
            ],
            "events": [
                {"id": "e1", "order": 1, "kind": "result", "affected_paths": []},
                {"id": "e10", "order": 10, "kind": "patch",
                 "affected_paths": ["query_bridge.py"]},
                {"id": "e20", "order": 20, "kind": "result", "affected_paths": []},
            ],
            "versions": [], "edges": [], "historical_edges": [],
        }]
        facts = [
            {"id": "f1", "statement": "query_bridge.py 曾出现 dtype mismatch 错误",
             "sources": ["e1"]},
            {"id": "f2", "statement": "query_bridge.py 后来新增 cast_cache 修复",
             "sources": ["e10"]},
            {"id": "f3", "statement": "query_bridge.py 修复后 pytest 通过",
             "sources": ["e20"]},
        ]
        groups = build_evidence_groups(
            facts, scopes, "code", {"failure_avoidance"}, 8, 10000, 20000)
        self.assertTrue(any(len(group["facts"]) == 3 for group in groups))

    def test_merge_scope_records_full_coverage_and_deduplicates_overlap(self):
        record = {"id": "e1", "order": 1, "kind": "message", "role": "user", "text": "x"}
        scope = {"cutoff": 1, "dialogue": [record], "events": [], "versions": [],
                 "edges": [], "historical_edges": [], "over_budget": False}
        merged = merge_scopes([scope, scope], "general")
        self.assertEqual(len(merged["dialogue"]), 1)
        self.assertTrue(merged["full_range_covered"])

    def test_merge_scope_keeps_distinct_graph_edges(self):
        scope = {
            "cutoff": 3,
            "dialogue": [], "events": [], "versions": [],
            "edges": [
                {"from": "a.py::run", "to": "b.py::load", "kind": "call_reference",
                 "source": "e1"},
                {"from": "a.py::run", "to": "c.py::load", "kind": "call_reference",
                 "source": "e2"},
            ],
            "historical_edges": [], "over_budget": False,
        }
        merged = merge_scopes([scope], "code")
        self.assertEqual(len(merged["edges"]), 2)

    def test_same_file_without_shared_object_does_not_create_code_group(self):
        scope = {
            "cutoff": 20,
            "dialogue": [
                {"id": "e1", "order": 1, "kind": "patch",
                 "changes": {"service.py": {"type": "update"}}},
                {"id": "e2", "order": 2, "kind": "patch",
                 "changes": {"service.py": {"type": "update"}}},
            ],
            "events": [
                {"id": "e1", "order": 1, "kind": "patch",
                 "affected_paths": ["service.py"]},
                {"id": "e2", "order": 2, "kind": "patch",
                 "affected_paths": ["service.py"]},
            ],
            "versions": [], "edges": [], "historical_edges": [],
        }
        facts = [
            {"id": "f1", "statement": "service.py 的 csv_parser 读取 CSV",
             "sources": ["e1"]},
            {"id": "f2", "statement": "service.py 的 auth_token 校验 JWT",
             "sources": ["e2"]},
        ]
        groups = build_evidence_groups(
            facts, [scope], "code", {"compatibility_preservation"}, 4, 8000, 16000)
        self.assertFalse(any(len(group["facts"]) > 1 for group in groups))

    def test_weak_common_symbol_does_not_link_unrelated_files(self):
        scope = {
            "cutoff": 20,
            "dialogue": [
                {"id": "e1", "order": 1, "kind": "patch",
                 "changes": {"payments.py": {"type": "update"}}},
                {"id": "e2", "order": 20, "kind": "patch",
                 "changes": {"search.py": {"type": "update"}}},
            ],
            "events": [
                {"id": "e1", "order": 1, "kind": "patch", "affected_paths": ["payments.py"]},
                {"id": "e2", "order": 20, "kind": "patch", "affected_paths": ["search.py"]},
            ],
            "versions": [], "edges": [], "historical_edges": [],
        }
        facts = [
            {"id": "f1", "statement": "支付模块 logger 输出乱码", "sources": ["e1"]},
            {"id": "f2", "statement": "搜索模块 logger 改成异步", "sources": ["e2"]},
        ]
        groups = build_evidence_groups(
            facts, [scope], "code", {"compatibility_preservation"}, 4, 8000, 16000)
        self.assertFalse(any(len(group["facts"]) > 1 for group in groups))

    def test_verification_reuse_requires_result_not_full_range(self):
        fact = {"id": "f1", "statement": "兼容性测试通过", "sources": ["e1"]}
        scope = {
            "cutoff": 1,
            "dialogue": [{"id": "e1", "order": 1, "kind": "message", "role": "user",
                          "stage_id": "stage-1", "text": "必须保留兼容性。"}],
            "stages": [{"id": "stage-1", "record_ids": ["e1"],
                        "start_order": 1, "end_order": 1}],
            "events": [], "versions": [], "edges": [], "historical_edges": [],
            "over_budget": False,
        }
        groups = build_evidence_groups(
            [fact], [scope], "general", {"verification_reuse"}, 2, 8000, 16000)
        self.assertEqual(len(groups), 1)
        self.assertFalse(groups[0]["scope"].get("full_range_required", False))

        incomplete = dict(scope, over_budget=True)
        groups = build_evidence_groups(
            [fact], [incomplete], "general", {"verification_reuse"}, 2, 8000, 16000)
        self.assertTrue(groups)
        failed = dict(scope, facts_failed=True)
        merged_failed = merge_scopes([failed], "general")
        self.assertFalse(merged_failed["full_range_covered"])

    def test_single_specific_chinese_term_links_labeled_changes(self):
        scopes = [{
            "cutoff": 20,
            "dialogue": [
                {"id": "e1", "order": 1, "kind": "message", "role": "user",
                 "stage_id": "stage-1", "text": "用户反馈`缓存失效`时返回友好提示。"},
                {"id": "e2", "order": 20, "kind": "message", "role": "user",
                 "stage_id": "stage-2", "text": "后来`缓存失效`改为抛出明确错误。"},
            ],
            "stages": [
                {"id": "stage-1", "record_ids": ["e1"], "start_order": 1, "end_order": 1},
                {"id": "stage-2", "record_ids": ["e2"], "start_order": 20, "end_order": 20},
            ],
            "events": [], "versions": [], "edges": [], "historical_edges": [],
        }]
        facts = [
            {"id": "f1", "statement": "用户反馈`缓存失效`时返回友好提示", "sources": ["e1"]},
            {"id": "f2", "statement": "用户要求后来`缓存失效`改为抛出明确错误", "sources": ["e2"]},
        ]
        # An old/new behavior change is a correction only when the dialogue
        # explicitly replaces the earlier rule; it does not establish a
        # compatibility promise by itself.
        groups = build_evidence_groups(
            facts, scopes, "general", {"correction_update"}, 4, 8000, 16000)
        self.assertTrue(any({"f1", "f2"} ==
                            {fact["id"] for fact in group["facts"]}
                            for group in groups))

    def test_version_ancestry_supplies_compatibility_links(self):
        scope = {
            "cutoff": 20,
            "dialogue": [
                {"id": "e1", "order": 1, "kind": "observation", "text": "config.py"},
                {"id": "e2", "order": 20, "kind": "patch", "text": "config.py"},
            ],
            "events": [
                {"id": "e1", "order": 1, "kind": "observation", "affected_paths": ["config.py"]},
                {"id": "e2", "order": 20, "kind": "patch", "affected_paths": ["config.py"]},
            ],
            "versions": [
                {"id": "v1", "path": "config.py", "observed_at": 1,
                 "source": "e1", "previous": None, "status": "known"},
                {"id": "v2", "path": "config.py", "observed_at": 20,
                 "source": "e2", "previous": "v1", "status": "known"},
            ],
            "edges": [], "historical_edges": [],
        }
        facts = [
            {"id": "f1", "statement": "必须兼容 config.py 的 load_config 在缺失输入时返回 None",
             "sources": ["v1"]},
            {"id": "f2", "statement": "必须兼容 config.py 的 load_config 在缺失输入时抛出 ValueError",
             "sources": ["v2"]},
        ]
        groups = build_evidence_groups(
            facts, [scope], "code", {"compatibility_preservation"}, 4, 8000, 16000)
        self.assertTrue(any({"f1", "f2"} ==
                            {fact["id"] for fact in group["facts"]}
                            for group in groups))

    def test_explicit_cross_file_graph_edge_links_facts(self):
        scope = {
            "cutoff": 20,
            "dialogue": [
                {"id": "e1", "order": 1, "kind": "patch",
                 "changes": {"main.py": {"type": "update"}}},
                {"id": "e2", "order": 20, "kind": "patch",
                 "changes": {"config.py": {"type": "update"}}},
            ],
            "events": [
                {"id": "e1", "order": 1, "kind": "patch", "affected_paths": ["main.py"]},
                {"id": "e2", "order": 20, "kind": "patch", "affected_paths": ["config.py"]},
            ],
            "versions": [],
            "edges": [{"from": "main.py::main", "to": "config.py::load_config",
                       "kind": "call_reference", "source": "e1"}],
            "historical_edges": [],
        }
        facts = [
            {"id": "f1", "statement": "必须兼容 main.py 的入口在输入为空时触发条件分支",
             "sources": ["e1"]},
            {"id": "f2", "statement": "必须兼容 config.py 的加载函数在缺失时抛出异常",
             "sources": ["e2"]},
        ]
        groups = build_evidence_groups(
            facts, [scope], "code", {"compatibility_preservation"}, 4, 8000, 16000)
        self.assertTrue(any({"f1", "f2"} ==
                            {fact["id"] for fact in group["facts"]}
                            for group in groups))

    def test_structural_history_pair_survives_neighbor_top_k(self):
        scope = {
            "cutoff": 90, "dialogue": [], "events": [], "versions": [],
            "edges": [], "historical_edges": [],
        }
        facts = []
        for index in range(1, 10):
            order = index * 10
            event_id, version_id = "e%d" % index, "v%d" % index
            scope["dialogue"].append({"id": event_id, "order": order,
                                       "kind": "patch", "text": "config.py"})
            scope["events"].append({"id": event_id, "order": order,
                                     "kind": "patch", "affected_paths": ["config.py"]})
            scope["versions"].append({
                "id": version_id, "path": "config.py", "observed_at": order,
                "source": event_id,
                "previous": "v%d" % (index - 1) if index > 1 else None,
                "status": "known",
            })
            facts.append({"id": "f%d" % index,
                          "statement": "必须兼容 config.py 的 load_config 在条件%d时返回结果" % index,
                          "sources": [version_id]})
        groups = build_evidence_groups(
            facts, [scope], "code", {"compatibility_preservation"}, 100, 100000, 200000)
        self.assertTrue(any({"f1", "f9"} ==
                            {fact["id"] for fact in group["facts"]}
                            for group in groups))

    def test_unrelated_version_branches_do_not_link_by_transition_flag(self):
        scope = {
            "cutoff": 40, "dialogue": [], "events": [], "versions": [],
            "edges": [], "historical_edges": [],
        }
        # Both branches patch the same file, but neither descends from the
        # other. A transition marker alone must not make them one history QA.
        for event_id, version_id, order, previous in (
                ("ea1", "va1", 10, None),
                ("ea2", "va2", 20, "va1"),
                ("eb1", "vb1", 30, None),
                ("eb2", "vb2", 40, "vb1")):
            scope["dialogue"].append({"id": event_id, "order": order,
                                       "kind": "patch", "text": "config.py"})
            scope["events"].append({"id": event_id, "order": order,
                                     "kind": "patch", "affected_paths": ["config.py"]})
            scope["versions"].append({
                "id": version_id, "path": "config.py", "observed_at": order,
                "source": event_id, "previous": previous, "status": "known",
            })
        facts = [
            {"id": "fa", "statement": "config.py 分支 A 修改读取行为",
             "sources": ["va2"]},
            {"id": "fb", "statement": "config.py 分支 B 修改写入行为",
             "sources": ["vb2"]},
        ]
        groups = build_evidence_groups(
            facts, [scope], "code", {"compatibility_preservation"}, 4, 8000, 16000)
        self.assertFalse(any({"fa", "fb"} ==
                             {fact["id"] for fact in group["facts"]}
                             for group in groups))

    def test_same_version_lineage_different_symbols_does_not_link(self):
        scope = {
            "cutoff": 20,
            "dialogue": [
                {"id": "e1", "order": 1, "kind": "patch", "text": "a.py"},
                {"id": "e2", "order": 20, "kind": "patch", "text": "a.py"},
            ],
            "events": [
                {"id": "e1", "order": 1, "kind": "patch", "affected_paths": ["a.py"]},
                {"id": "e2", "order": 20, "kind": "patch", "affected_paths": ["a.py"]},
            ],
            "versions": [
                {"id": "v1", "path": "a.py", "observed_at": 1,
                 "source": "e1", "previous": None, "status": "known"},
                {"id": "v2", "path": "a.py", "observed_at": 20,
                 "source": "e2", "previous": "v1", "status": "known"},
            ],
            "edges": [], "historical_edges": [],
        }
        facts = [
            {"id": "f1", "statement": "必须兼容 a.py 的 alpha 负责支付校验",
             "sources": ["v1"]},
            {"id": "f2", "statement": "a.py 的 beta 负责日志格式",
             "sources": ["v2"]},
        ]
        groups = build_evidence_groups(
            facts, [scope], "code", {"compatibility_preservation"}, 4, 8000, 16000)
        self.assertFalse(any({"f1", "f2"} ==
                             {fact["id"] for fact in group["facts"]}
                             for group in groups))

    def test_sibling_versions_do_not_count_as_a_transition(self):
        scope = {
            "cutoff": 30,
            "dialogue": [
                {"id": "e1", "order": 1, "kind": "patch", "text": "a.py"},
                {"id": "e2", "order": 20, "kind": "patch", "text": "a.py"},
                {"id": "e3", "order": 30, "kind": "patch", "text": "a.py"},
            ],
            "events": [
                {"id": "e1", "order": 1, "kind": "patch", "affected_paths": ["a.py"]},
                {"id": "e2", "order": 20, "kind": "patch", "affected_paths": ["a.py"]},
                {"id": "e3", "order": 30, "kind": "patch", "affected_paths": ["a.py"]},
            ],
            "versions": [
                {"id": "v1", "path": "a.py", "observed_at": 1,
                 "source": "e1", "previous": None, "status": "known"},
                {"id": "v2", "path": "a.py", "observed_at": 20,
                 "source": "e2", "previous": "v1", "status": "known"},
                {"id": "v3", "path": "a.py", "observed_at": 30,
                 "source": "e3", "previous": "v1", "status": "known"},
            ],
            "edges": [], "historical_edges": [],
        }
        facts = [
            {"id": "f2", "statement": "a.py 的 load_config 返回 None",
             "sources": ["v2"]},
            {"id": "f3", "statement": "a.py 的 load_config 抛出异常",
             "sources": ["v3"]},
        ]
        groups = build_evidence_groups(
            facts, [scope], "code", {"compatibility_preservation"}, 4, 8000, 16000)
        self.assertFalse(any({"f2", "f3"} ==
                             {fact["id"] for fact in group["facts"]}
                             for group in groups))

    def test_path_only_correction_update_cue_does_not_link_unrelated_symbols(self):
        scope = {
            "cutoff": 20,
            "dialogue": [
                {"id": "e1", "order": 1, "kind": "message", "text": "a.py"},
                {"id": "e2", "order": 20, "kind": "message", "text": "a.py"},
            ],
            "events": [], "versions": [], "edges": [], "historical_edges": [],
        }
        facts = [
            {"id": "f1", "statement": "之前 a.py 的 alpha 负责支付校验",
             "sources": ["e1"]},
            {"id": "f2", "statement": "后来 a.py 的 beta 负责日志格式",
             "sources": ["e2"]},
        ]
        groups = build_evidence_groups(
            facts, [scope], "code", {"compatibility_preservation"}, 4, 8000, 16000)
        self.assertFalse(any({"f1", "f2"} ==
                             {fact["id"] for fact in group["facts"]}
                             for group in groups))

    def test_review_guard_includes_call_and_result_in_both_directions(self):
        scope = {
            "cutoff": 2,
            "dialogue": [
                {"id": "call", "order": 1, "kind": "call", "call_id": "c1",
                 "text": "run check_config"},
                {"id": "result", "order": 2, "kind": "result", "call_id": "c1",
                 "text": "check_config failed"},
            ],
            "events": [], "versions": [], "edges": [], "historical_edges": [],
        }
        for fact in (
                {"id": "fc", "statement": "check_config 必须被调用", "sources": ["call"]},
                {"id": "fr", "statement": "check_config 必须返回失败", "sources": ["result"]}):
            groups = build_evidence_groups(
                [fact], [scope], "code", {"constraint_followthrough"}, 1, 8000, 16000)
            self.assertEqual(len(groups), 1)
            self.assertTrue(groups[0]["review_guard_complete"])
            self.assertEqual(set(groups[0]["review_guard_sources"]), {"call", "result"})
            self.assertEqual({item["id"] for item in groups[0]["scope"]["dialogue"]},
                             {"call", "result"})

    def test_review_guard_follows_version_lineage_forward(self):
        scope = {
            "cutoff": 20,
            "dialogue": [
                {"id": "e1", "order": 1, "kind": "observation", "text": "config.py"},
                {"id": "e2", "order": 20, "kind": "patch", "text": "config.py"},
            ],
            "events": [],
            "versions": [
                {"id": "v1", "path": "config.py", "observed_at": 1,
                 "source": "e1", "previous": None, "status": "known"},
                {"id": "v2", "path": "config.py", "observed_at": 20,
                 "source": "e2", "previous": "v1", "status": "known"},
            ],
            "edges": [], "historical_edges": [],
        }
        facts = [
            {"id": "f1", "statement": "必须兼容 config.py 的 load_config 返回 None",
             "sources": ["v1"]},
            {"id": "f2", "statement": "必须兼容 config.py 的 load_config 后来抛出 ValueError",
             "sources": ["v2"]},
        ]
        groups = build_evidence_groups(
            facts, [scope], "code", {"constraint_followthrough"}, 1, 8000, 16000)
        self.assertEqual(len(groups), 1)
        self.assertTrue(groups[0]["review_guard_complete"])
        self.assertTrue(groups[0]["facts"])
        self.assertEqual(set(groups[0]["review_guard_sources"]),
                         {"v1", "v2", "e1", "e2"})

    def test_review_guard_follows_source_lineage_without_later_fact(self):
        scope = {
            "cutoff": 20,
            "dialogue": [
                {"id": "e1", "order": 1, "kind": "observation", "text": "config.py"},
                {"id": "e2", "order": 20, "kind": "patch", "text": "config.py"},
                {"id": "e3", "order": 30, "kind": "patch", "text": "config.py"},
            ],
            "events": [],
            "versions": [
                {"id": "v1", "path": "config.py", "observed_at": 1,
                 "source": "e1", "previous": None, "status": "known"},
                {"id": "v2", "path": "config.py", "observed_at": 20,
                 "source": "e2", "previous": "v1", "status": "known"},
                {"id": "v3", "path": "config.py", "observed_at": 30,
                 "source": "e3", "previous": "v2", "status": "known"},
            ],
            "edges": [], "historical_edges": [],
        }
        for source in ("v1", "e1"):
            with self.subTest(source=source):
                fact = {"id": "f1", "statement": "必须兼容 config.py 的 load_config 返回 None",
                        "sources": [source]}
                groups = build_evidence_groups(
                    [fact], [scope], "code", {"constraint_followthrough"}, 1, 8000, 16000)
                self.assertEqual(len(groups), 1)
                self.assertTrue(groups[0]["review_guard_complete"])
                self.assertEqual(set(groups[0]["review_guard_sources"]),
                                 {"v1", "v2", "e1", "e2"})
                self.assertNotIn("v3", groups[0]["review_guard_sources"])
                self.assertNotIn("e3", groups[0]["review_guard_sources"])

    def test_review_guard_follows_explicit_lineage_even_for_different_symbols(self):
        scope = {
            "cutoff": 20,
            "dialogue": [
                {"id": "e1", "order": 1, "kind": "patch", "text": "a.py"},
                {"id": "e2", "order": 20, "kind": "patch", "text": "a.py"},
            ],
            "events": [],
            "versions": [
                {"id": "v1", "path": "a.py", "observed_at": 1,
                 "source": "e1", "previous": None, "status": "known"},
                {"id": "v2", "path": "a.py", "observed_at": 20,
                 "source": "e2", "previous": "v1", "status": "known"},
            ],
            "edges": [], "historical_edges": [],
        }
        facts = [
            {"id": "fa", "statement": "必须兼容 a.py 的 alpha 负责支付校验", "sources": ["v1"]},
            {"id": "fb", "statement": "a.py 的 beta 后来修改日志格式", "sources": ["v2"]},
        ]
        groups = build_evidence_groups(
            facts, [scope], "code", {"constraint_followthrough"}, 1, 8000, 16000)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]["facts"]), 1)
        self.assertEqual(set(groups[0]["review_guard_sources"]),
                         {"v1", "v2", "e1", "e2"})

    def test_review_guard_does_not_expand_same_path_without_explicit_relation(self):
        scope = {
            "cutoff": 20,
            "dialogue": [
                {"id": "e1", "order": 1, "kind": "patch",
                 "changes": {"a.py": {"type": "update"}}},
                {"id": "e2", "order": 20, "kind": "patch",
                 "changes": {"a.py": {"type": "update"}}},
            ],
            "events": [], "versions": [], "edges": [], "historical_edges": [],
        }
        facts = [
            {"id": "fa", "statement": "必须兼容 a.py 的 alpha 负责支付校验", "sources": ["e1"]},
            {"id": "fb", "statement": "a.py 的 beta 后来修改日志格式", "sources": ["e2"]},
        ]
        groups = build_evidence_groups(
            facts, [scope], "code", {"constraint_followthrough"}, 1, 8000, 16000)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]["review_guard_sources"]), 1)

    def test_over_budget_review_guard_keeps_group_as_incomplete(self):
        scope = {
            "cutoff": 2,
            "dialogue": [
                {"id": "call", "order": 1, "kind": "call", "call_id": "c1",
                 "text": "run check_config"},
                {"id": "result", "order": 2, "kind": "result", "call_id": "c1",
                 "text": "x" * 20000},
            ],
            "events": [], "versions": [], "edges": [], "historical_edges": [],
        }
        fact = {"id": "f1", "statement": "check_config 必须被调用", "sources": ["call"]}
        groups = build_evidence_groups(
            [fact], [scope], "code", {"constraint_followthrough"}, 1, 2000, 6000)
        self.assertEqual(len(groups), 1)
        self.assertFalse(groups[0]["review_guard_complete"])
        self.assertEqual(groups[0]["scope"]["review_guard_reason"], "over_budget")
        self.assertEqual(groups[0]["review_guard_sources"], ["call"])
        self.assertGreater(groups[0]["scope"]["review_guard_omitted_count"], 0)


if __name__ == "__main__":
    unittest.main()
