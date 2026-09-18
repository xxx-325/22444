import copy
import unittest

from dialogue_benchmark.llm import review_candidates


class SplitReviewTests(unittest.TestCase):
    def setUp(self):
        self.scope = {
            "cutoff": 2,
            "dialogue": [
                {"id": "m1", "kind": "message", "role": "user", "order": 1,
                 "stage_id": "s1", "text": "配置必须支持 yaml"},
                {"id": "m2", "kind": "message", "role": "assistant", "order": 2,
                 "stage_id": "s1", "text": "后续仍需验证兼容性"},
            ],
            "events": [], "versions": [], "stages": [{"id": "s1"}],
            "review_guard_sources": ["m2"], "review_guard_complete": True,
            "review_guard_reason": "complete", "max_context_chars": 60000,
        }
        self.facts = [
            {"id": "f1", "statement": "用户要求配置支持 yaml", "sources": ["m1"]},
            {"id": "f2", "statement": "兼容性仍需验证", "sources": ["m2"]},
        ]
        self.candidate = {
            "id": "general_g1_q1", "candidate_id": "general_g1_q1", "model_id": "q1",
            "qa_mode": "general", "type": "single-hop", "difficulty": "easy",
            "difficulty_reason": "单条用户约束直接给出", "memory_requirement": "恢复配置约束",
            "use_case": "开发者实现配置解析时决定必须支持的格式",
            "answer_target": "配置必须支持的格式", "fact_ids": ["f1"],
            "question": "配置必须支持什么格式？",
            "answer_points": [{"text": "配置必须支持 yaml。", "sources": ["m1"]}],
            "forbidden_points": [], "track": None,
        }

    def evidence_review(self, **updates):
        review = {
            "id": "q1", "review_contract": "structured_v2",
            "unambiguous": True, "difficulty_justified": True,
            "not_answer_leaking": True, "natural_wording": True,
            "practical_useful": True, "type_correct": True,
            "external_knowledge_separated": True, "external_knowledge_necessary": True,
            "full_range_checked": True, "recommended_type": "single-hop",
            "recommended_track": "none", "type_basis": "单条用户消息",
            "necessary_source_ids": "m1", "necessary_stage_ids": "none",
            "history_only_fact": "none", "history_source_ids": "none",
            "useful_task": "实现配置解析", "useful_decision": "选择必须支持的格式",
            "answer_effect": "决定解析器兼容范围",
            "point_evidence": "A1=supported@m1", "causal_bridge": "none",
            "reason": "题面与证据边界成立",
        }
        review.update(updates)
        return {"reviews": [review]}

    def structure_review(self, **updates):
        review = {
            "id": "q1", "review_contract": "structured_v2",
            "answer_requirements": "R1=必须支持的格式",
            "requirement_coverage": "R1=A1", "point_claims": "A1.1=支持 yaml",
            "reason": "答案结构完整",
        }
        review.update(updates)
        return {"reviews": [review]}

    class Client:
        def __init__(self, responses):
            self.responses = iter(responses)
            self.payloads = []
            self.usage = []

        def ask(self, prompt, data):
            self.payloads.append((prompt, copy.deepcopy(data)))
            self.usage.append({"status": "completed"})
            return next(self.responses)

    def test_split_review_partitions_inputs_and_approves(self):
        client = self.Client([self.structure_review(), self.evidence_review()])
        result = review_candidates(
            self.scope, self.facts, [self.candidate], client,
            qa_mode="general", review_mode="split", allow_repair=False)
        self.assertEqual(result["questions"][0]["status"], "approved")
        structure_payload = client.payloads[0][1]
        self.assertNotIn("scope", structure_payload)
        self.assertNotIn("facts", structure_payload)
        self.assertNotIn("use_case", structure_payload["candidates"][0])
        structure_candidate = structure_payload["candidates"][0]
        self.assertEqual(structure_candidate["immutable_question"], self.candidate["question"])
        self.assertEqual(structure_candidate["answer_points"][0]["review_id"], "A1")
        self.assertNotIn("sources", structure_candidate["answer_points"][0])
        evidence_candidate = client.payloads[1][1]["candidates"][0]
        self.assertEqual(evidence_candidate["answer_points"][0]["immutable_text"],
                         self.candidate["answer_points"][0]["text"])
        self.assertEqual([fact["id"] for fact in client.payloads[1][1]["facts"]],
                         ["f1", "f2"])
        self.assertIn("m2", {row["id"] for row in client.payloads[1][1]["scope"]["dialogue"]})
        self.assertEqual([item["stage"] for item in client.usage],
                         ["review_structure", "review_evidence"])

    def test_missing_answer_is_semantic_rejection_not_forbidden_coverage(self):
        structure = self.structure_review(
            answer_requirements="R1=格式;R2=默认行为",
            requirement_coverage="R1=A1;R2=MISSING")
        client = self.Client([structure])
        result = review_candidates(
            self.scope, self.facts, [self.candidate], client,
            qa_mode="general", review_mode="split", allow_repair=False)
        self.assertFalse(result["questions"])
        self.assertEqual(result["rejected"][0]["reason"],
                         "semantic_answer_structure_failed")
        self.assertIn("answer_complete", result["rejected"][0]["failed_checks"])

    def test_boolean_mapping_conflict_needs_review(self):
        structure = self.structure_review(
            answer_complete=True,
            answer_requirements="R1=格式;R2=默认行为",
            requirement_coverage="R1=A1;R2=MISSING")
        client = self.Client([structure])
        result = review_candidates(
            self.scope, self.facts, [self.candidate], client,
            qa_mode="general", review_mode="split", allow_repair=False)
        self.assertEqual(result["questions"][0]["status"], "needs_review")
        self.assertIn("answer_complete", result["questions"][0]["review_conflicts"])

    def test_independent_claims_are_rejected_but_transition_is_one_claim(self):
        compound_candidate = copy.deepcopy(self.candidate)
        compound_candidate["answer_points"][0]["text"] = (
            "配置必须支持 yaml，并且默认严格解析。")
        compound = self.structure_review(
            atomic_points_correct=False,
            point_claims="A1.1=支持 yaml;A1.2=默认严格解析")
        result = review_candidates(
            self.scope, self.facts, [compound_candidate],
            self.Client([compound]),
            qa_mode="general", review_mode="split", allow_repair=False)
        self.assertFalse(result["questions"])
        transition_candidate = copy.deepcopy(self.candidate)
        transition_candidate["answer_points"][0]["text"] = "旧行为变为新行为。"
        transition = self.structure_review(point_claims="A1.1=旧行为变为新行为")
        result = review_candidates(
            self.scope, self.facts, [transition_candidate],
            self.Client([transition, self.evidence_review()]),
            qa_mode="general", review_mode="split", allow_repair=False)
        self.assertEqual(result["questions"][0]["status"], "approved")

    def test_incomplete_guard_blocks_without_model_call(self):
        scope = dict(self.scope, review_guard_complete=False,
                     review_guard_reason="over_budget")
        client = self.Client([])
        result = review_candidates(
            scope, self.facts, [self.candidate], client,
            qa_mode="general", review_mode="split")
        self.assertEqual(result["questions"][0]["status"], "needs_review")
        self.assertEqual(result["questions"][0]["review_error"],
                         "incomplete_review_guard")
        self.assertFalse(client.payloads)

    def test_answer_review_identity_and_scope_cannot_override_question_lane(self):
        wrong_id = self.evidence_review()
        wrong_id["reviews"][0]["id"] = "another-candidate"
        result = review_candidates(
            self.scope, self.facts, [self.candidate],
            self.Client([self.structure_review(), wrong_id]),
            qa_mode="general", review_mode="split", allow_repair=False)
        self.assertEqual(result["questions"][0]["status"], "needs_review")
        self.assertEqual(result["questions"][0]["review_error"],
                         "evidence_review_unknown_id")

        override = self.evidence_review(requirement_coverage="R1=MISSING")
        result = review_candidates(
            self.scope, self.facts, [self.candidate],
            self.Client([self.structure_review(), override]),
            qa_mode="general", review_mode="split", allow_repair=False)
        self.assertEqual(result["questions"][0]["review_error"],
                         "evidence_review_scope_conflict")

    def test_answer_transport_failure_preserves_completed_question_review(self):
        structure = self.structure_review()

        class Client(self.Client):
            def ask(self, prompt, data):
                if self.payloads:
                    raise ValueError("answer transport failed")
                return super().ask(prompt, data)

        result = review_candidates(
            self.scope, self.facts, [self.candidate], Client([structure]),
            qa_mode="general", review_mode="split", allow_repair=False)
        self.assertEqual(result["questions"][0]["status"], "needs_review")
        self.assertEqual(result["questions"][0]["structure_review"]["answer_complete"], True)
        self.assertEqual(result["questions"][0]["review_error"], "review_evidence_failed")

    def test_candidate_declared_unknown_source_is_not_an_allowed_review_source(self):
        candidate = copy.deepcopy(self.candidate)
        candidate["answer_points"][0]["sources"] = ["unknown"]
        evidence = self.evidence_review(point_evidence="A1=supported@unknown")
        result = review_candidates(
            self.scope, self.facts, [candidate],
            self.Client([self.structure_review(), evidence]),
            qa_mode="general", review_mode="split", allow_repair=False)
        self.assertEqual(result["questions"][0]["status"], "needs_review")
        self.assertIn("point_evidence_out_of_scope",
                      result["questions"][0]["missing_review_fields"])

    def test_one_supported_missing_answer_repair_preserves_original_and_rechecks_both_lanes(self):
        candidate = copy.deepcopy(self.candidate)
        candidate.update(
            fact_ids=["f1", "f2"], answer_target="配置格式与后续兼容性动作",
            question="配置必须支持什么格式，后续还要做什么？")
        evidence_review = self.evidence_review(
            necessary_source_ids="m1,m2", answer_effect="决定格式和后续验证")
        incomplete = self.structure_review(
            answer_requirements="R1=配置格式;R2=后续动作",
            requirement_coverage="R1=A1;R2=MISSING")
        repaired = copy.deepcopy(candidate)
        repaired["id"] = "q1"
        repaired["answer_points"].append(
            {"text": "后续仍需验证兼容性。", "sources": ["m2"]})
        complete = self.structure_review(
            answer_requirements="R1=配置格式;R2=后续动作",
            requirement_coverage="R1=A1;R2=A2",
            point_claims="A1.1=支持 yaml;A2.1=验证兼容性")
        complete_evidence = self.evidence_review(
            necessary_source_ids="m1,m2", answer_effect="决定格式和后续验证",
            point_evidence="A1=supported@m1;A2=supported@m2")
        client = self.Client([
            incomplete, evidence_review, {"questions": [repaired]},
            complete, complete_evidence,
        ])
        result = review_candidates(
            self.scope, self.facts, [candidate], client,
            qa_mode="general", review_mode="split", allow_repair=True)
        self.assertEqual(result["questions"][0]["status"], "approved")
        self.assertEqual(len(result["revisions"]), 1)
        self.assertEqual(len(result["revisions"][0]["before"]["answer_points"]), 1)
        self.assertEqual(len(result["revisions"][0]["after"]["answer_points"]), 2)
        self.assertEqual([item["stage"] for item in client.usage],
                         ["review_structure", "review_evidence", "repair",
                          "review_structure", "review_evidence"])

    def test_wrong_multi_hop_label_is_repairable_as_single_hop(self):
        candidate = copy.deepcopy(self.candidate)
        candidate["type"] = "multi-hop"
        negative = self.evidence_review(
            type_correct=False, recommended_type="single-hop",
            necessary_stage_ids="none", reason="只需一条来源，应为 single-hop")
        repaired = copy.deepcopy(candidate)
        repaired.update(id="q1", type="single-hop")
        client = self.Client([
            self.structure_review(), negative, {"questions": [repaired]},
            self.structure_review(), self.evidence_review(),
        ])
        result = review_candidates(
            self.scope, self.facts, [candidate], client,
            qa_mode="general", review_mode="split", allow_repair=True)
        self.assertEqual(result["questions"][0]["status"], "approved")
        self.assertEqual(result["questions"][0]["type"], "single-hop")
        self.assertEqual([item["stage"] for item in client.usage],
                         ["review_structure", "review_evidence", "repair",
                          "review_structure", "review_evidence"])

    def test_false_history_label_is_repairable_as_inference_control(self):
        scope = copy.deepcopy(self.scope)
        scope["qa_mode"] = "code"
        facts = [{"id": "f1", "statement": "当前配置使用 yaml", "sources": ["m1"]}]
        candidate = copy.deepcopy(self.candidate)
        candidate.update(
            qa_mode="code", type="fact_recall", category="fact_recall",
            track="history_core", answer_target="当前配置格式")

        def evidence_review(history_correct, track):
            return {"reviews": [{
                "id": "q1", "review_contract": "structured_v2",
                "unambiguous": True, "difficulty_justified": True,
                "not_answer_leaking": True, "natural_wording": True,
                "practical_useful": True, "type_correct": history_correct,
                "history_requirement_correct": history_correct,
                "current_snapshot_alone_sufficient": True,
                "history_evidence_required": False,
                "recommended_type": "fact_recall", "recommended_track": track,
                "type_basis": "当前快照直接给出配置",
                "necessary_source_ids": "m1", "necessary_stage_ids": "none",
                "history_only_fact": "none", "history_source_ids": "none",
                "useful_task": "恢复当前配置", "useful_decision": "选择解析格式",
                "answer_effect": "决定当前解析器输入",
                "point_evidence": "A1=supported@m1", "causal_bridge": "none",
                "reason": "当前快照足够",
            }]}

        repaired = copy.deepcopy(candidate)
        repaired.update(id="q1", track="inference_control")
        client = self.Client([
            self.structure_review(), evidence_review(False, "inference_control"),
            {"questions": [repaired]}, self.structure_review(),
            evidence_review(True, "inference_control"),
        ])
        result = review_candidates(
            scope, facts, [candidate], client,
            qa_mode="code", review_mode="split", allow_repair=True)
        self.assertEqual(result["questions"][0]["status"], "approved")
        self.assertEqual(result["questions"][0]["track"], "inference_control")
        self.assertEqual([item["stage"] for item in client.usage],
                         ["review_structure", "review_evidence", "repair",
                          "review_structure", "review_evidence"])


if __name__ == "__main__":
    unittest.main()
