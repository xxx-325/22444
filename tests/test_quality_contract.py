import unittest

from dialogue_benchmark.quality import (
    CODE_QA_TYPES,
    GENERAL_QA_TYPES,
    validate_candidates,
    validate_facts,
)


class DualTrackQualityTests(unittest.TestCase):
    def setUp(self):
        self.scope = {
            "cutoff": 9,
            "dialogue": [
                {"id": "m1", "kind": "message", "role": "user", "text": "先支持 yaml"},
                {"id": "m2", "kind": "message", "role": "assistant", "text": "已记录"},
                {"id": "tool1", "kind": "result", "text": "tests passed"},
            ],
            "events": [{"id": "tool1", "order": 3, "kind": "result"}],
            "versions": [],
        }
        self.general_fact = {"id": "fg", "statement": "用户要求支持 yaml", "sources": ["m1"]}
        self.code_fact = {"id": "fc", "statement": "运行曾报错，修复后测试结果为通过", "sources": ["tool1"]}

    def _question(self, **updates):
        question = {
            "id": "q1",
            "qa_mode": "general",
            "type": "single-hop",
            "difficulty": "easy",
            "difficulty_reason": "单条消息直接给出",
            "memory_requirement": "找回用户已确认的约束",
            "use_case": "开发者继续实现配置解析时确认需要支持的格式，避免遗漏兼容要求",
            "answer_target": "用户要求支持的配置格式",
            "fact_ids": ["fg"],
            "question": "用户要求配置支持什么格式？",
            "answer_points": [{"text": "用户要求支持 yaml 格式。", "sources": ["m1"]}],
            "forbidden_points": [],
        }
        question.update(updates)
        return question

    def test_public_type_sets_are_disjoint(self):
        self.assertEqual(len(GENERAL_QA_TYPES), 5)
        self.assertEqual(len(CODE_QA_TYPES), 4)
        self.assertTrue(GENERAL_QA_TYPES.isdisjoint(CODE_QA_TYPES))

    def test_general_candidate_is_normalized(self):
        accepted, rejected = validate_candidates(
            {"questions": [self._question()]}, [self.general_fact], self.scope,
            qa_mode="general", allowed_types={"single-hop"})
        self.assertFalse(rejected)
        self.assertEqual(accepted[0]["qa_mode"], "general")
        self.assertEqual(accepted[0]["type"], "single-hop")
        self.assertEqual(accepted[0]["use_case"], self._question()["use_case"])

    def test_open_domain_records_external_knowledge_separately(self):
        question = self._question(type="open-domain")
        accepted, rejected = validate_candidates(
            {"questions": [question]}, [self.general_fact], self.scope,
            qa_mode="general", allowed_types={"open-domain"})
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]["reason"], "missing_external_knowledge")
        question["external_knowledge"] = "YAML is a structured data serialization format."
        accepted, rejected = validate_candidates(
            {"questions": [question]}, [self.general_fact], self.scope,
            qa_mode="general", allowed_types={"open-domain"})
        self.assertFalse(rejected)
        self.assertEqual(accepted[0]["external_knowledge"], question["external_knowledge"])

    def test_general_cannot_cite_tool_or_code_fields(self):
        tool_fact = {"id": "ft", "statement": "工具通过", "sources": ["tool1"]}
        question = self._question(fact_ids=["ft"],
                                  answer_points=[{"text": "工具通过。", "sources": ["tool1"]}])
        accepted, rejected = validate_candidates(
            {"questions": [question]}, [tool_fact], self.scope, qa_mode="general")
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]["reason"], "fact_source_not_allowed_for_mode")

        question = self._question(category="history_tracking", track="history_core")
        accepted, rejected = validate_candidates(
            {"questions": [question]}, [self.general_fact], self.scope, qa_mode="general")
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]["reason"], "code_fields_on_general_question")

    def test_code_candidate_uses_new_fields_and_legacy_alias(self):
        question = {
            "id": "qc",
            "qa_mode": "code",
            "type": "failure_diagnosis",
            "category": "failure_diagnosis",
            "track": "history_core",
            "difficulty": "medium",
            "difficulty_reason": "需要结合测试结果",
            "memory_requirement": "找回历史测试反馈",
            "use_case": "开发者选择回归测试时找回历史测试结论",
            "answer_target": "历史测试结果",
            "fact_ids": ["fc"],
            "question": "测试结果是什么？",
            "answer_points": [{"text": "测试结果为通过。", "sources": ["tool1"]}],
            "forbidden_points": [],
        }
        accepted, rejected = validate_candidates(
            {"questions": [question]}, [self.code_fact], self.scope, qa_mode="code")
        self.assertFalse(rejected)
        self.assertEqual(accepted[0]["category"], "failure_diagnosis")

        legacy = dict(question)
        legacy.pop("qa_mode")
        legacy.pop("type")
        accepted, rejected = validate_candidates(
            {"questions": [legacy]}, [self.code_fact], self.scope)
        self.assertFalse(rejected)
        self.assertEqual(accepted[0]["qa_mode"], "code")
        self.assertEqual(accepted[0]["type"], "failure_diagnosis")

    def test_surface_usefulness_is_left_to_semantic_review(self):
        question = {
            "id": "surface",
            "qa_mode": "code",
            "type": "history_tracking",
            "category": "history_tracking",
            "track": "history_core",
            "difficulty": "easy",
            "difficulty_reason": "直接回忆补丁",
            "memory_requirement": "找回补丁细节",
            "use_case": "开发者查找补丁",
            "answer_target": "函数签名的历史变化",
            "fact_ids": ["fc"],
            "question": "这个函数签名经历了什么格式变化？",
            "answer_points": [{"text": "函数签名改成多行格式。", "sources": ["tool1"]}],
            "forbidden_points": [],
        }
        accepted, rejected = validate_candidates(
            {"questions": [question]}, [self.code_fact], self.scope, qa_mode="code")
        self.assertFalse(rejected)
        self.assertEqual(accepted[0]["status"], "awaiting_semantic_review")

    def test_bad_question_does_not_discard_good_question(self):
        good = self._question(id="good")
        bad = self._question(id="bad", fact_ids=["missing"])
        accepted, rejected = validate_candidates(
            {"questions": [good, bad]}, [self.general_fact], self.scope,
            qa_mode="general")
        self.assertEqual([item["id"] for item in accepted], ["good"])
        self.assertEqual(rejected[0]["reason"], "unknown_fact")

    def test_point_evidence_must_close_over_selected_facts(self):
        other_fact = {"id": "other", "statement": "助手已记录", "sources": ["m2"]}
        question = self._question(
            answer_points=[{"text": "助手已记录。", "sources": ["m2"]}])
        accepted, rejected = validate_candidates(
            {"questions": [question]}, [self.general_fact, other_fact], self.scope,
            qa_mode="general")
        self.assertFalse(accepted)

    def test_compound_answer_point_reaches_review_with_warning(self):
        question = self._question(
            answer_points=[{"text": "支持 yaml 并且保留旧客户端兼容性。",
                            "sources": ["m1", "m2"]}])
        accepted, rejected = validate_candidates(
            {"questions": [question]}, [self.general_fact], self.scope,
            qa_mode="general")
        self.assertFalse(rejected)
        self.assertEqual(accepted[0]["pre_review_warnings"], ["compound_answer_point"])

    def test_internal_source_id_is_rejected_from_public_text(self):
        question = self._question(
            question="e48 之后用户要求什么？",
            answer_points=[{"text": "用户要求支持 yaml。", "sources": ["m1"]}])
        accepted, rejected = validate_candidates(
            {"questions": [question]}, [self.general_fact], self.scope,
            qa_mode="general")
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]["reason"], "internal_id_in_public_text")

    def test_atomic_duplicate_and_conflicting_points_are_rejected(self):
        question = self._question(
            answer_points=[{"text": "同一结论", "sources": ["m1"]},
                           {"text": "同一结论", "sources": ["m1"]}])
        accepted, rejected = validate_candidates(
            {"questions": [question]}, [self.general_fact], self.scope,
            qa_mode="general")
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]["reason"], "duplicate_answer_point")

        question = self._question(
            forbidden_points=[{"text": "用户要求支持 yaml 格式。", "sources": ["m1"]}])
        accepted, rejected = validate_candidates(
            {"questions": [question]}, [self.general_fact], self.scope,
            qa_mode="general")
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]["reason"], "conflicting_answer_and_forbidden_point")

    def test_fact_validation_can_restrict_to_general_messages(self):
        facts, rejected = validate_facts(
            {"facts": [self.general_fact, self.code_fact]}, self.scope,
            return_rejected=True, qa_mode="general")
        self.assertEqual([fact["id"] for fact in facts], ["fg"])
        self.assertEqual(rejected[0]["reason"], "unknown_or_missing_evidence")

    def test_both_source_scope_is_available_for_shared_fact_extraction(self):
        facts = {"facts": [self.general_fact, self.code_fact]}
        accepted, rejected = validate_facts(facts, self.scope,
                                            return_rejected=True, qa_mode="both")
        self.assertEqual({fact["id"] for fact in accepted}, {"fg", "fc"})
        self.assertFalse(rejected)


if __name__ == "__main__":
    unittest.main()
