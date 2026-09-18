"""Regression tests for single-lane and split-lane candidate review."""

import copy
import unittest

from dialogue_benchmark.llm import review_candidates
from dialogue_benchmark.quality import apply_review


class ReviewRegressionTests(unittest.TestCase):
    def setUp(self):
        self.scope = {
            "qa_mode": "general",
            "cutoff": 1,
            "model_request_chars": 32000,
            "max_context_chars": 32000,
            "dialogue": [{
                "id": "m1", "order": 1, "kind": "message", "role": "user",
                "text": "项目配置约定使用 yaml。",
            }],
            "events": [], "versions": [], "edges": [], "historical_edges": [],
            "evidence_group": {
                "id": "general-group-1", "target_types": ["single-hop"],
                "stage_count": 1,
            },
        }
        self.facts = [{
            "id": "f1", "qa_mode": "general", "statement": "项目配置约定使用 yaml。",
            "sources": ["m1"],
        }]
        self.candidate = {
            "id": "q1", "candidate_id": "q1", "qa_mode": "general",
            "type": "single-hop", "difficulty": "easy",
            "difficulty_reason": "直接找回一条约定",
            "memory_requirement": "恢复配置格式约定",
            "use_case": "继续配置项目时选择正确格式",
            "fact_ids": ["f1"],
            "question": "项目配置约定使用什么格式？",
            "answer_points": [{"text": "项目配置约定使用 yaml。", "sources": ["m1"]}],
            "forbidden_points": [],
        }

    def _decision(self, **updates):
        decision = {
            "id": "q1",
            "review_contract": "structured_v2",
            "evidence_supported": True,
            "version_consistent": True,
            "unambiguous": True,
            "category_correct": True,
            "difficulty_justified": True,
            "not_answer_leaking": True,
            "natural_wording": True,
            "practical_useful": True,
            "answer_complete": True,
            "atomic_points_correct": True,
            "type_correct": True,
            "type_basis": "问题只要求一条直接约定",
            "useful_task": "恢复配置约定",
            "useful_decision": "选择配置格式",
            "answer_effect": "避免使用错误格式",
            "recommended_type": "single-hop",
            "necessary_source_ids": "m1",
            "answer_requirements": "R1=配置格式",
            "requirement_coverage": "R1=A1",
            "point_claims": "A1.1=项目配置约定使用 yaml",
            "point_evidence": "A1=supported@m1",
            "reason": "结构化核验通过。",
        }
        decision.update(updates)
        return decision

    def _question_decision(self, **updates):
        decision = self._decision()
        decision.pop("category_correct")
        decision.pop("evidence_supported")
        decision.pop("version_consistent")
        decision.pop("answer_complete")
        decision.pop("atomic_points_correct")
        decision.pop("point_claims")
        decision.pop("answer_requirements")
        decision.pop("requirement_coverage")
        decision.pop("causal_support", None)
        decision.update(
            causal_bridge="none", recommended_track="none",
            necessary_stage_ids="none", history_only_fact="none",
            history_source_ids="none", external_knowledge_separated=True,
            external_knowledge_necessary=True, full_range_checked=True)
        decision.update(updates)
        return decision

    def _answer_decision(self, **updates):
        source = self._decision()
        decision = {key: source[key] for key in (
            "id", "review_contract", "answer_requirements",
            "requirement_coverage", "point_claims", "reason")}
        decision.update(updates)
        return decision

    class _Client:
        def __init__(self, responses):
            self.responses = list(responses)
            self.calls = []

        def ask(self, prompt, data):
            self.calls.append((prompt, copy.deepcopy(data)))
            return self.responses.pop(0)

    def test_single_mode_positive_approves_structured_candidate(self):
        client = self._Client([{"reviews": [self._decision()]}])
        result = review_candidates(
            self.scope, self.facts, [self.candidate], client,
            qa_mode="general", allow_repair=False, review_mode="single")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(result["questions"][0]["status"], "approved")

    def test_split_mode_positive_runs_two_lanes_and_approves(self):
        # Split review is shape-only first, then evidence/semantic review.
        client = self._Client([{"reviews": [self._answer_decision()]},
                               {"reviews": [self._question_decision()]}])
        result = review_candidates(
            self.scope, self.facts, [self.candidate], client,
            qa_mode="general", allow_repair=False, review_mode="split")
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(result["questions"][0]["status"], "approved")

    def test_split_question_failure_does_not_reach_answer_lane(self):
        decision = self._question_decision(type_correct=False, recommended_type="temporal")
        client = self._Client([{"reviews": [self._answer_decision()]},
                               {"reviews": [decision]}])
        result = review_candidates(
            self.scope, self.facts, [self.candidate], client,
            qa_mode="general", allow_repair=False, review_mode="split")
        self.assertEqual(len(client.calls), 2)
        self.assertFalse(any(q.get("status") == "approved" for q in result["questions"]))
        self.assertTrue(result["rejected"])
        self.assertEqual(result["rejected"][0]["reason"],
                         "semantic_review_failed")

    def test_structured_answer_out_of_scope_evidence_fails_closed(self):
        decision = self._decision(point_evidence="A1=supported@m2")
        kept, rejected = apply_review(
            [self.candidate], {"reviews": [decision]}, require_structured=True,
            allowed_sources={"m1"})
        self.assertFalse(rejected)
        self.assertEqual(kept[0]["status"], "needs_review")
        self.assertIn("point_evidence_out_of_scope", kept[0]["missing_review_fields"])

    def test_forbidden_point_must_be_contradicted_not_supported(self):
        candidate = copy.deepcopy(self.candidate)
        candidate["forbidden_points"] = [{
            "text": "项目配置约定使用 json。", "sources": ["m1"]}]
        valid = self._decision(
            point_claims="A1.1=项目配置约定使用 yaml;F1.1=项目配置约定使用 json",
            point_evidence="A1=supported@m1;F1=contradicted@m1")
        for key in ("evidence_supported", "version_consistent", "answer_complete",
                    "atomic_points_correct"):
            valid.pop(key)
        kept, rejected = apply_review(
            [candidate], {"reviews": [valid]}, require_structured=True,
            allowed_sources={"m1"})
        self.assertFalse(rejected)
        self.assertEqual(kept[0]["status"], "approved")

        invalid = self._decision(
            point_claims="A1.1=项目配置约定使用 yaml;F1.1=项目配置约定使用 json",
            point_evidence="A1=supported@m1;F1=supported@m1")
        for key in ("evidence_supported", "version_consistent", "answer_complete",
                    "atomic_points_correct"):
            invalid.pop(key)
        kept, rejected = apply_review(
            [candidate], {"reviews": [invalid]}, require_structured=True,
            allowed_sources={"m1"})
        self.assertFalse(kept)
        self.assertEqual(rejected[0]["reason"], "semantic_review_failed")
        self.assertIn("evidence_supported", rejected[0]["failed_checks"])

    def test_claims_cannot_import_missing_answer_text_from_evidence(self):
        decision = self._decision(point_claims="A1.1=证据里的另一个事实")
        kept, rejected = apply_review(
            [self.candidate], {"reviews": [decision]}, require_structured=True,
            allowed_sources={"m1"})
        self.assertFalse(rejected)
        self.assertEqual(kept[0]["status"], "needs_review")
        self.assertIn("point_claim_not_verbatim_span",
                      kept[0]["missing_review_fields"])

    def test_allow_repair_false_keeps_one_request_for_failed_review(self):
        decision = self._decision(natural_wording=False)
        client = self._Client([{"reviews": [decision]}])
        result = review_candidates(
            self.scope, self.facts, [self.candidate], client,
            qa_mode="general", allow_repair=False, review_mode="single")
        self.assertEqual(len(client.calls), 1)
        self.assertFalse(result.get("revisions"))
        self.assertEqual(result["rejected"][0]["reason"], "semantic_review_failed")


if __name__ == "__main__":
    unittest.main()
