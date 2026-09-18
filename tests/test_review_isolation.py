import copy
import unittest

from dialogue_benchmark.llm import parse_text_response, review_candidates


class ReviewIsolationTests(unittest.TestCase):
    def setUp(self):
        self.scope = {
            "cutoff": 2,
            "dialogue": [
                {"id": "m1", "kind": "message", "role": "user", "order": 1,
                 "stage_id": "s1", "text": "配置必须支持 yaml。"},
                {"id": "m2", "kind": "message", "role": "assistant", "order": 2,
                 "stage_id": "s1", "text": "后续仍需验证旧格式兼容性。"},
            ],
            "events": [],
            "versions": [],
            "stages": [{"id": "s1"}],
            "review_guard_sources": ["m2"],
            "review_guard_complete": True,
            "review_guard_reason": "complete",
            "max_context_chars": 60000,
        }
        self.facts = [
            {"id": "f1", "statement": "配置必须支持 yaml", "sources": ["m1"]},
            {"id": "f2", "statement": "后续仍需验证旧格式兼容性", "sources": ["m2"]},
        ]
        self.candidate = {
            "id": "general_g1_q1",
            "candidate_id": "general_g1_q1",
            "model_id": "q1",
            "qa_mode": "general",
            "type": "single-hop",
            "difficulty": "easy",
            "difficulty_reason": "同一讨论阶段直接给出两项约束",
            "memory_requirement": "恢复格式约束及后续验证动作",
            "use_case": "开发者实现配置解析时确定格式与兼容性验证范围",
            "answer_target": "配置格式与后续验证动作",
            "external_knowledge": "",
            "fact_ids": ["f1", "f2"],
            "question": "配置必须支持什么格式，后续还需执行什么验证？",
            "answer_points": [
                {"text": "配置必须支持 yaml。", "sources": ["m1"]},
                {"text": "后续仍需验证旧格式兼容性。", "sources": ["m2"]},
            ],
            "forbidden_points": [],
        }

    class TextClient:
        def __init__(self, responses):
            self.responses = iter(responses)
            self.payloads = []
            self.prompts = []
            self.usage = []

        def ask(self, prompt, data):
            self.prompts.append(prompt)
            self.payloads.append(copy.deepcopy(data))
            self.usage.append({"status": "completed"})
            return parse_text_response(next(self.responses))

    @staticmethod
    def structure_response(missing=False):
        coverage = "R1=A1;R2=MISSING" if missing else "R1=A1;R2=A2"
        claims = ("A1.1=配置必须支持 yaml" if missing else
                  "A1.1=配置必须支持 yaml;A2.1=后续仍需验证旧格式兼容性")
        return f"""REVIEW general_g1_q1
review_contract: structured_v2
answer_requirements: R1=必须支持的格式;R2=后续验证动作
requirement_coverage: {coverage}
point_claims: {claims}
reason: 逐项核对题面义务与候选答案点
END_REVIEW"""

    @staticmethod
    def evidence_response():
        return """REVIEW general_g1_q1
review_contract: structured_v2
unambiguous: true
difficulty_justified: true
not_answer_leaking: true
natural_wording: true
practical_useful: true
type_correct: true
external_knowledge_separated: true
external_knowledge_necessary: true
full_range_checked: true
recommended_type: single-hop
recommended_track: none
type_basis: 同一讨论阶段的两条直接约束
necessary_source_ids: m1,m2
necessary_stage_ids: s1
history_only_fact: none
history_source_ids: none
useful_task: 实现配置解析
useful_decision: 确定支持格式和兼容性验证范围
answer_effect: 决定解析器实现与验证步骤
point_evidence: A1=supported@m1;A2=supported@m2
causal_bridge: none
reason: 题面、用途、类型和候选答案均受可见证据支持
END_REVIEW"""

    def test_review_requests_isolate_structure_from_evidence(self):
        client = self.TextClient([
            self.structure_response(),
            self.evidence_response(),
        ])

        result = review_candidates(
            self.scope, self.facts, [self.candidate], client,
            qa_mode="general", review_mode="split", allow_repair=False)

        self.assertEqual(result["questions"][0]["status"], "approved")
        self.assertEqual(len(client.payloads), 2)

        structure_payload = client.payloads[0]
        for forbidden in ("facts", "scope", "raw", "use_case", "answer_target"):
            self.assertNotIn(forbidden, structure_payload)
        structure_candidate = structure_payload["candidates"][0]
        self.assertEqual(structure_candidate["immutable_question"],
                         self.candidate["question"])
        for forbidden in ("use_case", "answer_target", "fact_ids", "sources"):
            self.assertNotIn(forbidden, structure_candidate)
        self.assertEqual(
            set(structure_candidate),
            {"id", "candidate_id", "model_id", "immutable_question",
             "answer_points", "forbidden_points"},
        )
        for point in structure_candidate["answer_points"]:
            self.assertEqual(set(point), {"review_id", "immutable_text"})

        evidence_payload = client.payloads[1]
        self.assertEqual([fact["id"] for fact in evidence_payload["facts"]],
                         ["f1", "f2"])
        self.assertEqual(evidence_payload["candidates"][0]["use_case"],
                         self.candidate["use_case"])
        evidence_ids = {record["id"]
                        for record in evidence_payload["scope"]["dialogue"]}
        self.assertIn("m2", evidence_ids)
        self.assertEqual([item["stage"] for item in client.usage],
                         ["review_structure", "review_evidence"])

    def test_missing_answer_cannot_be_completed_from_hidden_facts(self):
        candidate = copy.deepcopy(self.candidate)
        candidate["answer_points"] = candidate["answer_points"][:1]
        client = self.TextClient([self.structure_response(missing=True)])

        result = review_candidates(
            self.scope, self.facts, [candidate], client,
            qa_mode="general", review_mode="split", allow_repair=False)

        self.assertFalse(result["questions"])
        self.assertEqual(len(client.payloads), 1)
        self.assertNotIn("facts", client.payloads[0])
        self.assertEqual(result["rejected"][0]["reason"],
                         "semantic_answer_structure_failed")
        self.assertIn("answer_complete", result["rejected"][0]["failed_checks"])
        self.assertEqual(client.usage[0]["stage"], "review_structure")

    def test_one_obligation_may_require_multiple_existing_answer_points(self):
        candidate = copy.deepcopy(self.candidate)
        candidate["question"] = "配置与后续验证要求是什么？"
        structure = """REVIEW general_g1_q1
review_contract: structured_v2
answer_requirements: R1=配置与后续验证要求
requirement_coverage: R1=A1,A2
point_claims: A1.1=配置必须支持 yaml;A2.1=后续仍需验证旧格式兼容性
reason: 一个归一义务由两个现有原子答案点共同覆盖
END_REVIEW"""
        client = self.TextClient([structure, self.evidence_response()])
        result = review_candidates(
            self.scope, self.facts, [candidate], client,
            qa_mode="general", review_mode="split", allow_repair=False)
        self.assertEqual(result["questions"][0]["status"], "approved")
        self.assertEqual(len(client.payloads), 2)


if __name__ == "__main__":
    unittest.main()
