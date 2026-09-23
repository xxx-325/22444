import copy
import unittest

from dialogue_benchmark.llm import (
    generate_from_facts,
    parse_text_response,
    review_candidates,
    simple_evidence_payload,
)
from dialogue_benchmark.quality import apply_simple_relevance_review
from tests.simple_test_helpers import maybe_focus_response, maybe_relevance_response


class TextClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.payloads = []
        self.prompts = []
        self.usage = []

    def ask(self, prompt, data):
        # Target routing is exercised separately in test_memory_types.
        if "review_contract: target_v1" in prompt:
            return {"reviews": [{"id": "q1", "review_contract": "target_v1",
                                 "target_alignment": "aligned"}]}
        self.prompts.append(prompt)
        self.payloads.append(copy.deepcopy(data))
        self.usage.append({"status": "completed"})
        focus = maybe_focus_response(prompt, data)
        if focus is not None:
            return focus
        relevance = maybe_relevance_response(prompt, data)
        if relevance is not None:
            return relevance
        response = next(self.responses)
        return parse_text_response(response) if isinstance(response, str) else response


class RelevanceScriptedClient:
    """Answer each simple stage while keeping relevance's response explicit."""

    def __init__(self):
        self.calls = []
        self.stages = []

    def ask(self, prompt, payload):
        # Target routing is exercised separately in test_memory_types.
        if "review_contract: target_v1" in prompt:
            return {"reviews": [{"id": "q1", "review_contract": "target_v1",
                                 "target_alignment": "aligned"}]}
        self.calls.append((prompt, copy.deepcopy(payload)))
        if "simple_relevance_v1" in prompt:
            self.stages.append("review_relevance")
            response = """REVIEW q1
review_contract: simple_relevance_v1
point_relevance: A1=direct;A2=extra;A3=extra
END_REVIEW"""
        elif "simple_atomicity_v1" in prompt:
            self.stages.append("review_atomicity")
            response = """REVIEW q1
review_contract: simple_atomicity_v1
point_atomicity: A1=single
END_REVIEW"""
        elif "fully answer every requested subject" in prompt:
            self.stages.append("review_completeness")
            response = """REVIEW q1
review_contract: simple_v1
completeness: complete
END_REVIEW"""
        else:
            self.stages.append("review_evidence")
            response = """REVIEW q1
review_contract: simple_v1
point_evidence: A1=supported@资料1
END_REVIEW"""
        return parse_text_response(response)


class SimpleReviewTests(unittest.TestCase):
    def setUp(self):
        self.scope = {
            "cutoff": 2,
            "dialogue": [
                {"id": "m1", "kind": "message", "role": "user", "order": 1,
                 "stage_id": "s1", "text": "配置必须支持 yaml。"},
                {"id": "m2", "kind": "message", "role": "assistant", "order": 2,
                 "stage_id": "s2", "text": "旧格式不再受支持。"},
            ],
            "events": [], "versions": [],
            "stages": [{"id": "s1"}, {"id": "s2"}],
            "review_guard_sources": ["m2"],
            "review_guard_complete": True,
            "model_request_chars": 60000,
            "max_context_chars": 60000,
        }
        self.facts = [
            {"id": "f1", "statement": "配置必须支持 yaml", "sources": ["m1"]},
        ]
        self.candidate = {
            "id": "general_g1_q1", "candidate_id": "general_g1_q1",
            "model_id": "q1", "qa_mode": "general", "type": "constraint_followthrough",
            "fact_ids": ["f1"], "question": "配置必须支持什么格式？",
            "answer_points": [{"text": "配置必须支持 yaml。", "sources": ["m1"]}],
            "forbidden_points": [
                {"text": "配置只支持 JSON。", "sources": ["m1"]}],
        }
        self.generation_context = {
            "prompt": "ORIGINAL QA INSTRUCTION",
            "payload": {
                "facts": [{"text": "配置必须支持 yaml", "materials": ["资料1"]}],
                "materials": [
                    {"reference": "资料1", "text": "配置必须支持 yaml。"},
                    {"reference": "资料2", "text": "旧格式不再受支持。"},
                ],
                "relations": [],
            },
            "ref_to_source": {"资料1": "m1", "资料2": "m2"},
        }

    @staticmethod
    def atomicity(points="A1=single;F1=single", review_id="q1"):
        return """REVIEW %s
review_contract: simple_atomicity_v1
point_atomicity: %s
END_REVIEW""" % (review_id, points)

    def test_simple_generation_binds_program_target_and_missing_kind(self):
        response = """QA q1
QUESTION: 配置必须支持什么格式？
ANSWER_POINT: 配置必须支持 yaml。 || SOURCES: 资料1
END_QA"""
        client = TextClient([response])
        result = generate_from_facts(
            self.scope, self.facts, client, qa_mode="general",
            target_type="constraint_followthrough")
        self.assertEqual(result["questions"][0]["type"], "constraint_followthrough")
        self.assertEqual(result["questions"][0]["fact_ids"], ["f1"])
        self.assertNotIn("difficulty", result["questions"][0])
        self.assertNotIn("scope", client.payloads[0])
        self.assertEqual(client.payloads[0]["material_references"][0]["reference"], "资料1")
        self.assertNotIn("id", client.payloads[0]["material_references"][0])
        self.assertEqual(client.payloads[1]["facts"], [{
            "text": "配置必须支持 yaml", "materials": ["资料1"]}])
        self.assertNotIn("constraint_followthrough", client.prompts[1])
        self.assertNotIn("TARGET_TYPE", client.prompts[1])

        noisy = response.replace(
            "QA q1", "QA q1\nTYPE: correction_update\nDIFFICULTY: hard\nTRACK: history_core\nFACT_IDS: f3,f4")
        result = generate_from_facts(
            self.scope, self.facts, TextClient([noisy]), qa_mode="general",
            target_type="constraint_followthrough")
        self.assertEqual(result["questions"][0]["type"], "constraint_followthrough")
        self.assertNotIn("difficulty", result["questions"][0])
        self.assertNotIn("track", result["questions"][0])
        self.assertEqual(result["questions"][0]["fact_ids"], ["f1"])
        self.assertNotIn("FACT_IDS:", client.prompts[1])

        bad_source = response.replace("SOURCES: 资料1", "SOURCES: 资料999")
        result = generate_from_facts(
            self.scope, self.facts, TextClient([bad_source]), qa_mode="general",
            target_type="constraint_followthrough")
        self.assertFalse(result["questions"])
        self.assertEqual(result["rejected"][0]["reason"],
                         "invalid_answer_evidence")

        internal_id = response.replace(
            "配置必须支持什么格式？", "配置在 e228 时必须支持什么格式？")
        result = generate_from_facts(
            self.scope, self.facts, TextClient([internal_id]), qa_mode="general",
            target_type="constraint_followthrough")
        self.assertFalse(result["questions"])
        self.assertEqual(result["rejected"][0]["reason"],
                         "internal_id_in_public_text")

        local_reference = response.replace(
            "配置必须支持什么格式？", "资料1中的配置必须支持什么格式？")
        result = generate_from_facts(
            self.scope, self.facts, TextClient([local_reference]), qa_mode="general",
            target_type="constraint_followthrough")
        self.assertFalse(result["questions"])
        self.assertEqual(result["rejected"][0]["reason"],
                         "local_reference_in_public_text")

        raw_source_id = response.replace(
            "配置必须支持什么格式？", "m1 中的配置必须支持什么格式？")
        result = generate_from_facts(
            self.scope, self.facts, TextClient([raw_source_id]), qa_mode="general",
            target_type="constraint_followthrough")
        self.assertFalse(result["questions"])
        self.assertEqual(result["rejected"][0]["reason"],
                         "internal_id_in_public_text")

        no_qa = TextClient([
            "NO_QA\nMISSING_KIND: reason\nMISSING_OBJECT: yaml",
        ])
        result = generate_from_facts(
            self.scope, self.facts, no_qa, qa_mode="general",
            target_type="constraint_followthrough")
        self.assertEqual(result["missing_kind"], "reason")
        self.assertEqual(result["missing_object"], "yaml")
        self.assertEqual(result["questions"], [])
        with self.assertRaises(ValueError):
            parse_text_response("NO_QA\nMISSING_KIND: arbitrary")

        never_called = TextClient([])
        result = generate_from_facts(
            self.scope, self.facts, never_called, qa_mode="general",
            allowed_types={"correction_update"}, target_type="constraint_followthrough")
        self.assertEqual(result["stage_status"]["qa"], "failed")
        self.assertFalse(never_called.payloads)

    def test_simple_focused_calls_are_isolated_and_approve(self):
        client = TextClient([
            self.atomicity(),
            """REVIEW q1
review_contract: simple_v1
completeness: complete
END_REVIEW""",
            """REVIEW q1
review_contract: simple_v1
point_evidence: A1=supported@资料1;F1=contradicted@资料1
END_REVIEW""",
        ])
        result = review_candidates(
            self.scope, self.facts, [self.candidate], client,
            qa_mode="general", review_mode="simple", allow_repair=False)
        self.assertEqual(result["questions"][0]["status"], "approved")
        self.assertEqual(result["questions"][0]["quality_status"], "approved")
        self.assertEqual([item["stage"] for item in client.usage],
                         ["review_relevance", "review_atomicity",
                          "review_completeness", "review_evidence"])
        relevance = client.payloads[0]
        self.assertNotIn("scope", relevance)
        self.assertNotIn("facts", relevance)
        self.assertEqual(set(relevance["candidates"][0]), {
            "id", "immutable_question", "answer_points", "forbidden_points"})
        self.assertEqual(relevance["candidates"][0]["id"], "q1")
        self.assertNotIn("sources", relevance["candidates"][0]["answer_points"][0])
        self.assertNotIn("scope", client.payloads[3])
        second = client.payloads[3]["candidates"][0]
        self.assertEqual(second["id"], "q1")
        self.assertNotIn("type", second)
        self.assertNotIn("difficulty", second)
        self.assertIn("sources", second["answer_points"][0])
        self.assertEqual(result["questions"][0]["completeness_review"]["returned_id"],
                         "q1")
        self.assertEqual(result["questions"][0]["evidence_review"]["returned_id"],
                         "q1")

    def test_local_review_id_does_not_follow_candidate_model_id(self):
        candidate = copy.deepcopy(self.candidate)
        candidate["model_id"] = "q2"
        approved = TextClient([
            self.atomicity(),
            """REVIEW q1
review_contract: simple_v1
completeness: complete
END_REVIEW""",
            """REVIEW q1
review_contract: simple_v1
point_evidence: A1=supported@资料1;F1=contradicted@资料1
END_REVIEW""",
        ])
        result = review_candidates(
            self.scope, self.facts, [candidate], approved,
            qa_mode="general", review_mode="simple", allow_repair=False)
        self.assertEqual(result["questions"][0]["status"], "approved")
        self.assertEqual(approved.payloads[0]["candidates"][0]["id"], "q1")
        self.assertEqual(approved.payloads[1]["candidates"][0]["id"], "q1")
        self.assertEqual(approved.payloads[3]["candidates"][0]["id"], "q1")

        wrong = TextClient([self.atomicity(review_id="q2")])
        result = review_candidates(
            self.scope, self.facts, [candidate], wrong,
            qa_mode="general", review_mode="simple", allow_repair=False)
        self.assertEqual(result["questions"][0]["status"], "needs_review")

    def test_relevance_filters_side_points_before_atomicity_for_thin_cli(self):
        candidate = {
            "id": "general_cli_q1", "candidate_id": "general_cli_q1",
            "model_id": "q1", "qa_mode": "general", "type": "constraint_followthrough",
            "fact_ids": ["f-cli"],
            "question": "薄 CLI 的 train 子命令接收哪种凭据？",
            "answer_points": [
                {"text": "train 子命令只接受前一阶段生成的哈希 receipt。",
                 "sources": ["m1"]},
                {"text": "CLI 将结果写入 output 目录。", "sources": ["m1"]},
                {"text": "receipt 缺失时 CLI 返回错误。", "sources": ["m1"]},
            ],
            "forbidden_points": [],
        }
        client = RelevanceScriptedClient()
        result = review_candidates(
            self.scope, [{"id": "f-cli", "statement": "CLI 约束",
                          "sources": ["m1"]}], [candidate], client,
            qa_mode="general", review_mode="simple", allow_repair=False)

        self.assertEqual(result["questions"][0]["status"], "approved")
        self.assertEqual(client.stages, [
            "review_relevance", "review_atomicity", "review_completeness",
            "review_evidence"])
        self.assertEqual(
            [point["review_id"] for point in
             client.calls[0][1]["candidates"][0]["answer_points"]],
            ["A1", "A2", "A3"])
        self.assertEqual(
            [point["review_id"] for point in
             client.calls[1][1]["candidates"][0]["answer_points"]],
            ["A1"])
        self.assertEqual(
            [point["text"] for point in result["questions"][0]["answer_points"]],
            ["train 子命令只接受前一阶段生成的哈希 receipt。"])
        self.assertEqual(
            result["questions"][0]["relevance_review"]["point_relevance"],
            "A1=direct;A2=extra;A3=extra")

    def test_static_relevance_drops_signature_only_steps_from_value_flow(self):
        candidate = {
            "id": "code-flow-q1", "qa_mode": "code",
            "type": "compatibility_preservation",
            "question": "timeout_seconds 如何一路传到 subprocess.run 并最终成为 timeout 参数？",
            "answer_points": [
                {"text": "run 方法新增 timeout_seconds: int = 3600 参数。",
                 "sources": ["m1"]},
                {"text": "run 调用 _run_command 时把 timeout_seconds 传入。",
                 "sources": ["m1"]},
                {"text": "_run_command 新增 timeout_seconds: int 形参。",
                 "sources": ["m1"]},
                {"text": "_default_runner 签名新增 timeout_seconds。",
                 "sources": ["m1"]},
                {"text": "_default_runner 在 subprocess.run 中设置 timeout=timeout_seconds。",
                 "sources": ["m1"]},
            ],
            "forbidden_points": [],
        }
        review = {"reviews": [{
            "id": "q1", "review_contract": "simple_relevance_v1",
            "point_relevance": "A1=direct;A2=direct;A3=direct;A4=direct;A5=direct",
        }]}

        kept, rejected = apply_simple_relevance_review([candidate], review)

        self.assertFalse(rejected)
        self.assertEqual(
            [point["text"] for point in kept[0]["answer_points"]],
            [candidate["answer_points"][1]["text"],
             candidate["answer_points"][4]["text"]])
        self.assertEqual(kept[0]["static_relevance_filtered"],
                         ["A1", "A3", "A4"])

    def test_simple_material_view_redacts_wrappers_but_preserves_raw_content(self):
        scope = {
            "dialogue": [
                {"id": "e1", "kind": "patch", "order": 1, "success": False,
                 "changes": {"src/a.py": {
                     "id": "business-id", "hash": "business-hash",
                     "status": "failed", "code": "e123 = 1"}}},
                {"id": "e1-fragment", "parent_id": "e1", "kind": "patch",
                 "order": 1, "success": False,
                 "changes": {"src/a.py": {
                     "id": "business-id", "hash": "business-hash",
                     "status": "failed", "code": "e123 = 1"}}},
            ],
            "events": [],
            "versions": [{"id": "v1", "path": "src/a.py", "source": "e1",
                          "status": "unknown", "sha256": "private-sha"}],
            "edges": [{"source": "node-7", "target": "node-8",
                       "relation": "references"}],
        }
        payload, mapping = simple_evidence_payload(
            scope, {"e1", "v1"},
            facts=[{"statement": "e1 中的操作失败", "sources": ["e1"]}])
        self.assertEqual(set(mapping.values()), {"e1", "v1"})
        self.assertNotIn("node-7", str(payload))
        self.assertNotIn("private-sha", str(payload))
        patch_records = payload["materials"][0]["original_records"]
        self.assertEqual(len(patch_records), 1)
        self.assertEqual(patch_records[0]["operation_outcome"], "操作失败")
        self.assertEqual(
            patch_records[0]["changes"]["src/a.py"]["hash"], "business-hash")
        version_records = payload["materials"][1]["original_records"]
        self.assertEqual(version_records[0]["content_state"],
                         "内容未知，不能视为完整代码")
        self.assertNotIn("e1", payload["facts"][0]["text"])

    def test_metadata_bearing_fact_is_omitted_but_raw_business_hash_is_preserved(self):
        scope = {
            "dialogue": [{
                "id": "e1", "kind": "result", "order": 1,
                "content": {"sha256": "business-sha", "status": "expected"},
            }],
            "events": [],
            "versions": [
                {"id": "v4", "path": "src/a.py", "source": "e1",
                 "previous": None, "status": "known", "content": "old"},
                {"id": "v8", "path": "src/a.py", "source": "e1",
                 "previous": "v4", "status": "known", "content": "new",
                 "sha256": "wrapper-sha"},
            ],
        }
        payload, _ = simple_evidence_payload(
            scope, {"e1"}, facts=[{
                "statement": "previous `v8` / `v4`, sha256 wrapper-sha",
                "sources": ["e1"],
            }])
        self.assertEqual(payload["facts"], [])
        raw = [record for material in payload["materials"]
               for record in material.get("original_records", [])]
        self.assertTrue(any(record.get("content", {}).get("sha256") == "business-sha"
                            for record in raw if isinstance(record.get("content"), dict)))
        self.assertNotIn("wrapper-sha", str(payload))

    def test_missing_early_reject_and_insufficient_needs_review(self):
        missing = TextClient([self.atomicity(), """REVIEW q1
review_contract: simple_v1
completeness: missing
missing: 还需说明兼容性
END_REVIEW"""])
        result = review_candidates(
            self.scope, self.facts, [self.candidate], missing,
            qa_mode="general", review_mode="simple", allow_repair=False)
        self.assertEqual(result["rejected"][0]["reason"],
                         "semantic_answer_incomplete")
        self.assertEqual(len(missing.payloads), 3)

        insufficient = TextClient([
            self.atomicity(),
            """REVIEW q1
review_contract: simple_v1
completeness: complete
missing: none
END_REVIEW""",
            """REVIEW q1
review_contract: simple_v1
point_evidence: A1=insufficient;F1=contradicted@资料1
END_REVIEW""",
        ])
        result = review_candidates(
            self.scope, self.facts, [self.candidate], insufficient,
            qa_mode="general", review_mode="simple", allow_repair=False)
        self.assertEqual(result["questions"][0]["status"], "needs_review")
        self.assertEqual(result["questions"][0]["review_error"],
                         "insufficient_evidence")

    def test_unknown_point_or_source_never_approves(self):
        for evidence in (
                "A1=supported@资料1;F1=contradicted@资料1;A2=supported@资料1",
                "A1=supported@资料999;F1=contradicted@资料1"):
            client = TextClient([
                self.atomicity(),
                """REVIEW q1
review_contract: simple_v1
completeness: complete
END_REVIEW""",
                """REVIEW q1
review_contract: simple_v1
point_evidence: %s
END_REVIEW""" % evidence,
            ])
            result = review_candidates(
                self.scope, self.facts, [self.candidate], client,
                qa_mode="general", review_mode="simple", allow_repair=False)
            self.assertEqual(result["questions"][0]["status"], "needs_review")
            self.assertEqual(result["questions"][0]["review_error"],
                             "invalid_evidence_review")

    def test_one_supported_missing_answer_repair_is_rechecked(self):
        candidate = copy.deepcopy(self.candidate)
        candidate["question"] = "配置必须支持什么格式，并保留什么兼容性？"
        candidate["forbidden_points"] = []
        client = TextClient([
            self.atomicity("A1=single"),
            """REVIEW q1
review_contract: simple_v1
completeness: missing
missing: 兼容性要求
END_REVIEW""",
            """REVIEW q1
review_contract: simple_v1
point_evidence: A1=supported@资料1
END_REVIEW""",
            """QA q1
QUESTION: 配置必须支持什么格式，并保留什么兼容性？
ANSWER_POINT: 配置必须支持 yaml。 || SOURCES: 资料1
ANSWER_POINT: 还必须保留旧格式兼容性。 || SOURCES: 资料2
END_QA""",
            self.atomicity("A1=single;A2=single"),
            """REVIEW q1
review_contract: simple_v1
completeness: complete
END_REVIEW""",
            """REVIEW q1
review_contract: simple_v1
point_evidence: A1=supported@资料1;A2=supported@资料2
END_REVIEW""",
        ])
        facts = self.facts + [{
            "id": "f2", "statement": "还必须保留旧格式兼容性", "sources": ["m2"]}]
        result = review_candidates(
            self.scope, facts, [candidate], client,
            qa_mode="general", review_mode="simple", allow_repair=True,
            generation_context=self.generation_context)
        self.assertEqual(result["questions"][0]["status"], "approved")
        self.assertTrue(result["questions"][0]["repair_attempted"])
        self.assertEqual(len(result["questions"][0]["answer_points"]), 2)
        self.assertEqual(result["questions"][0]["fact_ids"], ["f1", "f2"])
        self.assertEqual([item["stage"] for item in client.usage], [
            "review_relevance", "review_atomicity", "review_completeness",
            "review_evidence", "repair", "review_relevance",
            "review_atomicity", "review_completeness", "review_evidence"])


if __name__ == "__main__":
    unittest.main()
