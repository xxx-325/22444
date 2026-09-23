import unittest

from dialogue_benchmark.fact_index import (
    build_evidence_groups,
    build_evidence_index,
    static_evidence_check,
)


class CodeEvidenceRequirementTests(unittest.TestCase):
    @staticmethod
    def _scope(dialogue, versions=None, edges=None):
        return {
            "cutoff": max((item.get("order", 0) for item in dialogue), default=0),
            "dialogue": dialogue,
            "events": [],
            "versions": versions or [],
            "edges": edges or [],
            "historical_edges": [],
            "over_budget": False,
        }

    def test_same_source_repeated_rg_facts_are_one_piece_of_evidence(self):
        scope = self._scope([{
            "id": "rg-result", "order": 1, "kind": "result",
            "text": "rg: load_config defaults to None",
        }])
        facts = [
            {"id": "f%d" % index,
             "statement": "load_config defaults to None%s" % (" " * index),
             "sources": ["rg-result"]}
            for index in range(6)
        ]

        index = build_evidence_index(facts, [scope], "code")

        self.assertEqual(len(index["infos"]), 1)
        self.assertEqual(index["fact_aliases"]["f5"], "f0")
        self.assertEqual(index["merged_fact_ids"]["f0"],
                         ["f1", "f2", "f3", "f4", "f5"])
        self.assertNotIn("merged_fact_ids", index["infos"][0]["fact"])

    def test_dedup_preserves_case_and_internal_code_whitespace(self):
        scope = self._scope([{
            "id": "source", "order": 1, "kind": "result", "text": "code facts",
        }])
        facts = [
            {"id": "upper", "statement": "return 'X'", "sources": ["source"]},
            {"id": "lower", "statement": "return 'x'", "sources": ["source"]},
            {"id": "double", "statement": "value = 'a  b'", "sources": ["source"]},
            {"id": "single", "statement": "value = 'a b'", "sources": ["source"]},
        ]

        index = build_evidence_index(facts, [scope], "code")

        self.assertEqual(len(index["infos"]), 4)
        self.assertEqual({info["fact"]["id"] for info in index["infos"]},
                         {"upper", "lower", "double", "single"})

    def test_call_result_edge_is_not_a_business_logic_relation(self):
        scope = self._scope([
            {"id": "call", "order": 1, "kind": "call", "call_id": "c1",
             "text": "lookup_config when key is absent"},
            {"id": "result", "order": 2, "kind": "result", "call_id": "c1",
             "text": "lookup_config returned None"},
        ])
        facts = [
            {"id": "condition", "statement": "当 lookup_config 的 key 缺失时进入分支",
             "sources": ["call"]},
            {"id": "outcome", "statement": "lookup_config 返回 None",
             "sources": ["result"]},
        ]
        index = build_evidence_index(facts, [scope], "code")
        group = {
            "qa_mode": "code", "facts": facts, "scope": index["universe"],
            "review_guard_complete": True,
        }

        check = static_evidence_check(
            group, index, "compatibility_preservation")

        self.assertEqual(check["status"], "insufficient")
        self.assertEqual(check["reason"], "missing_type_evidence")

    def test_only_new_version_is_insufficient_for_history(self):
        dialogue = [
            {"id": "e1", "order": 1, "kind": "observation", "text": "config.py"},
            {"id": "e2", "order": 2, "kind": "patch", "text": "config.py"},
        ]
        versions = [
            {"id": "v1", "path": "config.py", "observed_at": 1,
             "source": "e1", "previous": None, "status": "known"},
            {"id": "v2", "path": "config.py", "observed_at": 2,
             "source": "e2", "previous": "v1", "status": "known"},
        ]
        scope = self._scope(dialogue, versions)
        fact = {"id": "new", "statement": "config.py 的 load_config 现在返回 None",
                "sources": ["v2"]}
        index = build_evidence_index([fact], [scope], "code")
        group = {"qa_mode": "code", "facts": [fact], "scope": index["universe"],
                 "review_guard_complete": True}

        check = static_evidence_check(group, index, "correction_update")

        self.assertEqual(check["status"], "insufficient")
        self.assertEqual(check["reason"], "correction_missing_old")

    def test_single_error_without_failure_condition_is_not_failure_avoidance(self):
        scope = self._scope([{
            "id": "failed", "order": 1, "kind": "result",
            "text": "load_config failed with ValueError",
        }])
        fact = {"id": "error", "statement": "load_config 实际运行报错 ValueError",
                "sources": ["failed"]}
        index = build_evidence_index([fact], [scope], "code")
        group = {"qa_mode": "code", "facts": [fact], "scope": index["universe"],
                 "review_guard_complete": True}

        check = static_evidence_check(group, index, "failure_avoidance")

        self.assertEqual(check["status"], "insufficient")
        self.assertEqual(check["reason"], "missing_type_evidence")

    def test_one_message_can_contain_a_complete_old_new_transition(self):
        scope = self._scope([{
            "id": "decision", "order": 4, "kind": "message",
            "text": "之前 load_config 返回 None，后来改为抛出 ValueError。",
        }])
        fact = {"id": "transition",
                "statement": "之前 load_config 返回 None，后来改为抛出 ValueError",
                "sources": ["decision"]}
        index = build_evidence_index([fact], [scope], "code")
        groups = build_evidence_groups(
            [fact], [scope], "code", {"correction_update"}, 1,
            evidence_index=index, static_selection=True)

        self.assertEqual(len(groups), 1)
        self.assertEqual(
            groups[0]["static_evidence_requirements"]["correction_update"]["status"],
            "supported")

    def test_historical_constraint_is_valid_constraint_followthrough(self):
        scope = self._scope([{
            "id": "constraint", "order": 1, "kind": "message",
            "text": "此前用户要求 load_config 必须拒绝空路径。",
        }])
        fact = {"id": "constraint-fact",
                "statement": "此前用户要求 load_config 必须拒绝空路径",
                "sources": ["constraint"]}
        index = build_evidence_index([fact], [scope], "code")
        groups = build_evidence_groups(
            [fact], [scope], "code", {"constraint_followthrough"}, 1,
            evidence_index=index, static_selection=True)

        self.assertEqual(len(groups), 1)
        self.assertEqual(
            groups[0]["static_evidence_requirements"]["constraint_followthrough"]["status"],
            "supported")

    def test_explicit_cross_file_dependency_alone_is_not_compatibility(self):
        scope = self._scope(
            [
                {"id": "main", "order": 1, "kind": "observation",
                 "changes": {"main.py": {"content": "run"}}},
                {"id": "config", "order": 2, "kind": "observation",
                 "changes": {"config.py": {"content": "load"}}},
            ],
            edges=[{"from": "main.py::run", "to": "config.py::load_config",
                    "kind": "call_reference", "source": "main"}],
        )
        facts = [
            {"id": "condition",
             "statement": "main.py 的 run 当输入为空时进入配置分支",
             "sources": ["main"]},
            {"id": "outcome",
             "statement": "config.py 的 load_config 返回默认配置",
             "sources": ["config"]},
        ]
        index = build_evidence_index(facts, [scope], "code")
        groups = build_evidence_groups(
            facts, [scope], "code", {"compatibility_preservation"}, 1,
            evidence_index=index, static_selection=True)

        self.assertFalse(groups)

    def test_post_generation_checks_only_actual_answer_citations(self):
        dialogue = [
            {"id": "e1", "order": 1, "kind": "observation", "text": "config.py"},
            {"id": "e2", "order": 2, "kind": "patch", "text": "config.py"},
        ]
        versions = [
            {"id": "v1", "path": "config.py", "observed_at": 1,
             "source": "e1", "previous": None, "status": "known"},
            {"id": "v2", "path": "config.py", "observed_at": 2,
             "source": "e2", "previous": "v1", "status": "known"},
        ]
        scope = self._scope(dialogue, versions)
        facts = [
            {"id": "old", "statement": "之前 load_config 返回 None", "sources": ["v1"]},
            {"id": "new", "statement": "用户纠正 load_config 规则，现在改为抛出 ValueError", "sources": ["v2"]},
        ]
        index = build_evidence_index(facts, [scope], "code")
        group = {"qa_mode": "code", "facts": facts, "scope": index["universe"],
                 "review_guard_complete": True}
        candidate = {"answer_points": [{"text": "现在会抛出 ValueError",
                                          "sources": ["v2"]}]}

        check = static_evidence_check(
            group, index, "correction_update", candidate)

        self.assertEqual(check["status"], "insufficient")

        complete_candidate = {
            "answer_points": [{"text": "load_config 的行为发生了变化",
                               "sources": ["v1", "v2"]}],
        }
        complete_check = static_evidence_check(
            group, index, "correction_update", complete_candidate)
        self.assertEqual(complete_check["status"], "supported")

    def test_multi_source_transition_fact_requires_all_answer_citations(self):
        dialogue = [
            {"id": "e1", "order": 1, "kind": "observation", "text": "config.py"},
            {"id": "e2", "order": 2, "kind": "patch", "text": "config.py"},
        ]
        versions = [
            {"id": "v1", "path": "config.py", "observed_at": 1,
             "source": "e1", "previous": None, "status": "known"},
            {"id": "v2", "path": "config.py", "observed_at": 2,
             "source": "e2", "previous": "v1", "status": "known"},
        ]
        scope = self._scope(dialogue, versions)
        fact = {
            "id": "transition",
            "statement": "load_config previously returned None and now raises ValueError",
            "sources": ["v1", "v2"],
        }
        index = build_evidence_index([fact], [scope], "code")
        group = {"qa_mode": "code", "facts": [fact], "scope": index["universe"],
                 "review_guard_complete": True}
        partial_candidate = {
            "answer_points": [{"text": "load_config now raises ValueError",
                               "sources": ["v2"]}],
        }

        partial = static_evidence_check(
            group, index, "correction_update", partial_candidate)
        complete = static_evidence_check(
            group, index, "correction_update", {
                "answer_points": [{"text": "load_config changed its behavior",
                                   "sources": ["v1", "v2"]}],
            })

        self.assertEqual(partial["status"], "insufficient")
        self.assertEqual(partial["reason"], "missing_type_evidence")
        self.assertEqual(complete["status"], "supported")

    def test_failure_source_gate_does_not_claim_semantic_target_approval(self):
        scope = self._scope(
            [
                {"id": "failed", "order": 1, "kind": "result",
                 "text": "load_config failed with ValueError",
                 "changes": {"loader.py": {"content": "failure"}}},
                {"id": "fixed", "order": 2, "kind": "patch",
                 "text": "validate_config fix validates empty input",
                 "changes": {"validation.py": {"content": "fix"}}},
            ],
            edges=[{"from": "loader.py::load_config",
                    "to": "validation.py::validate_config",
                    "kind": "call_reference", "source": "fixed"}],
        )
        facts = [
            {"id": "error", "statement": "loader.py 的 load_config 实际运行报错 ValueError",
             "sources": ["failed"]},
            {"id": "fix", "statement": "后来 validation.py 修复 validate_config 并验证空输入",
             "sources": ["fixed"]},
        ]
        index = build_evidence_index(facts, [scope], "code")
        group = {"qa_mode": "code", "facts": facts, "scope": index["universe"],
                 "review_guard_complete": True}
        candidate = {"answer_points": [{"text": "出现了 ValueError 报错",
                                          "sources": ["failed"]}]}

        check = static_evidence_check(
            group, index, "failure_avoidance", candidate)

        self.assertEqual(check["status"], "insufficient")
        self.assertEqual(check["reason"], "missing_type_evidence")

    def test_post_generation_behavior_answer_cannot_cite_only_outcome(self):
        scope = self._scope(
            [
                {"id": "condition-source", "order": 1, "kind": "observation",
                 "changes": {"main.py": {"content": "branch"}}},
                {"id": "outcome-source", "order": 2, "kind": "observation",
                 "changes": {"config.py": {"content": "load"}}},
            ],
            edges=[{"from": "main.py::run", "to": "config.py::load_config",
                    "kind": "call_reference", "source": "condition-source"}],
        )
        facts = [
            {"id": "condition", "statement": "main.py 的 run 当输入为空时进入分支",
             "sources": ["condition-source"]},
            {"id": "outcome", "statement": "config.py 的 load_config 返回默认配置",
             "sources": ["outcome-source"]},
        ]
        index = build_evidence_index(facts, [scope], "code")
        group = {"qa_mode": "code", "facts": facts, "scope": index["universe"],
                 "review_guard_complete": True}
        candidate = {"answer_points": [{"text": "load_config 返回默认配置",
                                          "sources": ["outcome-source"]}]}

        check = static_evidence_check(
            group, index, "compatibility_preservation", candidate)

        self.assertEqual(check["status"], "insufficient")
        self.assertEqual(check["reason"], "missing_type_evidence")

    def test_general_qa_grouping_is_unchanged(self):
        scope = self._scope([{
            "id": "message", "order": 1, "kind": "message", "stage_id": "s1",
            "text": "用户要求 export 输出简洁答案。",
        }])
        fact = {"id": "preference", "statement": "用户要求 export 输出简洁答案",
                "sources": ["message"]}

        groups = build_evidence_groups(
            [fact], [scope], "general", {"constraint_followthrough"}, 1,
            static_selection=True)

        self.assertEqual(len(groups), 1)
        self.assertIn("static_evidence_requirements", groups[0])

    def test_planned_test_is_not_verification_reuse(self):
        scope = self._scope([{
            "id": "plan", "order": 1, "kind": "message",
            "text": "计划运行 export 的 EU 输入测试。",
        }])
        fact = {"id": "plan-fact", "statement": "计划运行 export 的 EU 输入测试",
                "sources": ["plan"]}
        index = build_evidence_index([fact], [scope], "general")
        self.assertEqual(build_evidence_groups(
            [fact], [scope], "general", {"verification_reuse"}, 1,
            evidence_index=index, static_selection=True), [])

    def test_code_only_environment_text_is_not_external_state(self):
        scope = self._scope([{
            "id": "source", "order": 1, "kind": "observation",
            "source_kind": "code", "text": "production export path",
        }])
        fact = {"id": "code-fact", "statement": "production 环境只支持 EU 编码",
                "sources": ["source"]}
        index = build_evidence_index([fact], [scope], "code")
        self.assertEqual(build_evidence_groups(
            [fact], [scope], "code", {"external_state_application"}, 1,
            evidence_index=index, static_selection=True), [])


if __name__ == "__main__":
    unittest.main()
