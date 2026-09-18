"""Offline contracts for the content-level focus stage and temporal wording."""

import copy
import json
import unittest

from dialogue_benchmark.cli import generate_simple_target
from dialogue_benchmark.fact_index import build_evidence_index
from dialogue_benchmark.llm import (
    SIMPLE_QA_PROMPT,
    generate_from_facts,
    parse_text_response,
)
from dialogue_benchmark.quality import validate_simple_candidates


class ScriptedClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.usage = []

    def ask(self, prompt, payload):
        self.calls.append((prompt, copy.deepcopy(payload)))
        self.usage.append({"status": "completed"})
        response = next(self.responses)
        return parse_text_response(response) if isinstance(response, str) else response


class FailingClient(ScriptedClient):
    def ask(self, prompt, payload):
        self.calls.append((prompt, copy.deepcopy(payload)))
        self.usage.append({"status": "failed"})
        raise RuntimeError("synthetic focus transport failure")


class SimpleContentFocusTests(unittest.TestCase):
    def setUp(self):
        self.scope = {
            "cutoff": 3,
            "dialogue": [
                {"id": "selected-source", "order": 1, "kind": "message",
                 "role": "assistant", "text": "端口决定由 18080 改为 18081。"},
                {"id": "side-source", "order": 2, "kind": "message",
                 "role": "assistant", "text": "旁支讨论了另一个缓存方案。"},
                {"id": "extra-source", "order": 3, "kind": "message",
                 "role": "assistant", "text": "补充记录了端口决定的后续验证。"},
            ],
            "events": [], "versions": [], "stages": [],
            "generation_extra_sources": ["extra-source"],
            "evidence_group": {"target_types": ["single-hop"]},
            "model_request_chars": 60000,
            "max_context_chars": 60000,
        }
        self.facts = [{
            "id": "f-selected", "qa_mode": "general",
            "statement": "端口决定由 18080 改为 18081。",
            "sources": ["selected-source"],
        }]
        self.qa_response = """QA q1
QUESTION: 端口决定改成了什么？
ANSWER_POINT: 端口由 18080 改为 18081。 || SOURCES: 资料1
END_QA"""

    def test_focus_payload_is_safe_and_qa_uses_focus_sources_and_extras(self):
        client = ScriptedClient([
            "FOCUS: 围绕端口决定的变化确认后续可复用的配置选择\nSOURCES: 资料1",
            self.qa_response,
        ])
        result = generate_from_facts(
            self.scope, self.facts, client, qa_mode="general",
            allowed_types={"single-hop"}, target_type="single-hop",
            generation_mode="simple")

        self.assertEqual(result["stage_status"]["focus"], "completed")
        self.assertEqual(result["stage_status"]["qa"], "completed")
        self.assertEqual(len(client.calls), 2)
        focus_payload = client.calls[0][1]
        qa_payload = client.calls[1][1]
        self.assertIn("facts", focus_payload)
        self.assertIn("material_references", focus_payload)
        self.assertNotIn("original_records", focus_payload)
        focus_text = json.dumps(focus_payload, ensure_ascii=False)
        self.assertNotIn("selected-source", focus_text)
        self.assertNotIn("side-source", focus_text)
        self.assertNotIn("extra-source", focus_text)
        self.assertNotIn("f-selected", focus_text)

        self.assertIn("focus", qa_payload)
        self.assertEqual(qa_payload["focus"]["text"],
                         "围绕端口决定的变化确认后续可复用的配置选择")
        qa_text = json.dumps(qa_payload, ensure_ascii=False)
        self.assertIn("端口决定由 18080 改为 18081", qa_text)
        self.assertIn("端口决定的后续验证", qa_text)
        self.assertNotIn("旁支讨论了另一个缓存方案", qa_text)
        self.assertNotIn("selected-source", qa_text)
        self.assertNotIn("side-source", qa_text)
        self.assertNotIn("extra-source", qa_text)
        self.assertNotIn("f-selected", qa_text)

    def test_invalid_or_missing_focus_never_falls_back_to_qa_and_is_checkpointed(self):
        invalid_responses = (
            "FOCUS: 选择配置变化\nSOURCES: 资料999",
            "FOCUS: \nSOURCES: 资料1",
            "FOCUS: 资料3对应的事实\nSOURCES: 资料1",
            self.qa_response,
        )
        for response in invalid_responses:
            with self.subTest(response=response.splitlines()[0]):
                checkpoints = {}

                def checkpoint(name, data):
                    checkpoints[name] = copy.deepcopy(data)

                client = ScriptedClient([response])
                result = generate_from_facts(
                    self.scope, self.facts, client, qa_mode="general",
                    allowed_types={"single-hop"}, target_type="single-hop",
                    generation_mode="simple", checkpoint=checkpoint)

                self.assertEqual(result["stage_status"]["focus"], "failed")
                self.assertFalse(result["questions"])
                self.assertEqual(len(client.calls), 1)
                self.assertTrue(any("focus" in name for name in checkpoints))

        client = FailingClient([])
        result = generate_from_facts(
            self.scope, self.facts, client, qa_mode="general",
            allowed_types={"single-hop"}, target_type="single-hop",
            generation_mode="simple")
        self.assertEqual(result["stage_status"]["focus"], "failed")
        self.assertFalse(result["questions"])
        self.assertEqual(len(client.calls), 1)

    def test_focus_missing_object_rejects_local_reference_without_qa(self):
        client = ScriptedClient([
            "NO_QA\nMISSING_KIND: dependency\nMISSING_OBJECT: 资料1",
        ])
        result = generate_from_facts(
            self.scope, self.facts, client, qa_mode="general",
            allowed_types={"single-hop"}, target_type="single-hop",
            generation_mode="simple")

        self.assertEqual(result["stage_status"]["focus"], "failed")
        self.assertEqual(result["stage_status"]["qa"], "not_submitted")
        self.assertEqual(result["stage_errors"][0]["error_code"],
                         "invalid_missing_object")
        self.assertFalse(result["questions"])
        self.assertEqual(len(client.calls), 1)

    def test_selected_focus_reference_is_renumbered_for_qa_without_text_leak(self):
        scope = copy.deepcopy(self.scope)
        scope["dialogue"] = [
            {"id": "source-1", "order": 1, "kind": "message",
             "role": "assistant", "text": "第一个事实。"},
            {"id": "source-2", "order": 2, "kind": "message",
             "role": "assistant", "text": "第二个事实。"},
            {"id": "source-3", "order": 3, "kind": "message",
             "role": "assistant", "text": "第三个事实，是实际需要回答的选择。"},
        ]
        scope.pop("generation_extra_sources", None)
        facts = [
            {"id": "f1", "statement": "第一个事实。", "sources": ["source-1"]},
            {"id": "f2", "statement": "第二个事实。", "sources": ["source-2"]},
            {"id": "f3", "statement": "第三个事实，是实际需要回答的选择。",
             "sources": ["source-3"]},
        ]
        client = ScriptedClient([
            "FOCUS: 确认第三个事实对应的可复用选择\nSOURCES: 资料3",
            """QA q1
QUESTION: 第三个事实对应什么选择？
ANSWER_POINT: 第三个事实对应实际需要回答的选择。 || SOURCES: 资料1
END_QA""",
        ])
        result = generate_from_facts(
            scope, facts, client, qa_mode="general",
            allowed_types={"single-hop"}, target_type="single-hop",
            generation_mode="simple")

        self.assertEqual(len(client.calls), 2)
        qa_payload = client.calls[1][1]
        self.assertEqual(qa_payload["focus"]["sources"], ["资料1"])
        self.assertNotIn("资料3", qa_payload["focus"]["text"])
        material_text = json.dumps(qa_payload["materials"], ensure_ascii=False)
        self.assertIn("第三个事实", material_text)
        self.assertNotIn("第一个事实", material_text)
        self.assertNotIn("第二个事实", material_text)
        self.assertEqual(result["questions"][0]["answer_points"][0]["sources"],
                         ["source-3"])

    def test_focus_missing_evidence_expands_then_reselects_focus_with_one_attempt(self):
        scope = {
            "cutoff": 2,
            "dialogue": [
                {"id": "e-base", "order": 1, "kind": "message",
                 "role": "assistant", "text": "run runner.py"},
                {"id": "e-extra", "order": 2, "kind": "message",
                 "role": "assistant",
                 "text": "runner.py outcome was validated after the request."},
            ],
            "events": [], "versions": [], "edges": [],
            "historical_edges": [], "stages": [],
        }
        facts = [
            {"id": "f-base", "qa_mode": "general",
             "statement": "runner.py currently handles the request.",
             "sources": ["e-base"]},
            {"id": "f-extra", "qa_mode": "general",
             "statement": "runner.py outcome was validated after the request.",
             "sources": ["e-extra"]},
        ]
        index = build_evidence_index(facts, [scope], "general", 60000)
        index["expansion_candidates"]["f-base"] = [{
            "missing_kind": "outcome", "relation": "call_result",
            "fact_ids": ["f-extra"], "source_ids": ["e-extra"],
            "order": 2, "distance": 1,
        }]
        group = {
            "id": "g-base", "qa_mode": "general",
            "facts": [copy.deepcopy(facts[0])],
            "scope": dict(index["universe"], evidence_group={"id": "g-base"}),
            "relation_path": {},
            "expansion_pointer": {
                "attempted": False,
                "candidates": [{
                    "missing_kind": "outcome", "relation": "explicit_result",
                    "fact_ids": ["f-extra"], "source_ids": ["e-extra"],
                    "distance": 1, "base_fact_id": "f-base",
                }],
            },
        }
        client = ScriptedClient([
            "NO_QA\nMISSING_KIND: outcome\nMISSING_OBJECT: runner.py",
            "FOCUS: 围绕 runner.py 的记录结果确认验证状态\nSOURCES: 资料1,资料2",
            """QA q1
QUESTION: runner.py 的结果是什么？
ANSWER_POINT: runner.py 的结果在请求后得到验证。 || SOURCES: 资料2
END_QA""",
        ])

        result = generate_simple_target(
            group, index, "single-hop", client, "general", expansion_budget=1)

        self.assertEqual(len(client.calls), 3)
        self.assertEqual(result["generation_request_count"], 3)
        self.assertEqual(result["generation_attempt_count"], 2)
        self.assertEqual(result["expansion_rounds"], 1)
        self.assertEqual(result["expansion_stop_reason"], "candidate_generated")
        self.assertEqual(len(result["generated"]["questions"]), 1)
        self.assertEqual(
            {fact["id"] for fact in result["active_group"]["facts"]},
            {"f-base", "f-extra"},
        )
        self.assertIn("runner.py outcome was validated", json.dumps(
            client.calls[1][1], ensure_ascii=False))
        self.assertIn("focus", client.calls[2][1])


class SimpleTemporalReferenceTests(unittest.TestCase):
    def setUp(self):
        self.scope = {
            "cutoff": 1,
            "dialogue": [{"id": "m1", "order": 1, "kind": "message",
                           "role": "assistant", "text": "之前记录了配置要求。"}],
        }
        self.facts = [{"id": "f1", "statement": "之前记录了配置要求。",
                       "sources": ["m1"]}]

    @staticmethod
    def candidate(question):
        return {"id": "q1", "question": question,
                "answer_points": [{"text": "配置要求是 yaml。", "sources": ["m1"]}],
                "forbidden_points": []}

    def test_bare_recentness_words_are_rejected_without_rewriting_candidate(self):
        for word in ("上次", "上一次", "最近一次", "最后一次"):
            question = "%s配置使用了什么格式？" % word
            with self.subTest(word=word):
                accepted, rejected = validate_simple_candidates(
                    {"questions": [self.candidate(question)]}, self.facts,
                    self.scope, qa_mode="general")
                self.assertFalse(accepted)
                self.assertEqual(rejected[0]["reason"],
                                 "unsupported_temporal_reference")
                self.assertEqual(rejected[0]["question"]["question"], question)

    def test_explicit_event_and_quoted_code_literal_are_not_rejected(self):
        questions = (
            "之前配置要求使用 JSON，后来改为 yaml；发生了什么变化？",
            "代码 `value = '上次'` 返回的字面量是什么？",
        )
        for question in questions:
            with self.subTest(question=question):
                accepted, rejected = validate_simple_candidates(
                    {"questions": [self.candidate(question)]}, self.facts,
                    self.scope, qa_mode="general")
                self.assertEqual(len(accepted), 1)
                self.assertFalse(rejected)

    def test_temporal_gate_checks_answer_and_forbidden_text_but_ignores_literals(self):
        for field in ("answer_points", "forbidden_points"):
            with self.subTest(field=field):
                candidate = self.candidate("配置使用了什么格式？")
                candidate[field] = [{
                    "text": "上次记录的配置要求。",
                    "sources": ["m1"],
                }]
                accepted, rejected = validate_simple_candidates(
                    {"questions": [candidate]}, self.facts,
                    self.scope, qa_mode="general")
                self.assertFalse(accepted)
                self.assertEqual(rejected[0]["reason"],
                                 "unsupported_temporal_reference")

        literal = self.candidate("配置使用了什么格式？")
        literal["answer_points"] = [{
            "text": "代码 `value = '上次'` 的字面量。",
            "sources": ["m1"],
        }]
        accepted, rejected = validate_simple_candidates(
            {"questions": [literal]}, self.facts,
            self.scope, qa_mode="general")
        self.assertEqual(len(accepted), 1)
        self.assertFalse(rejected)

    def test_prompt_preserves_recorded_transition_without_inventing_reason(self):
        prompt = " ".join(SIMPLE_QA_PROMPT.split())
        self.assertIn("同一个对象从旧值改为新值是一条", prompt)
        self.assertIn("一个条件导致一个结果是一条", prompt)
        self.assertIn("A relation never proves cause", prompt)


if __name__ == "__main__":
    unittest.main()
