import copy
import io
import json
import urllib.error
import unittest
from unittest.mock import patch

from dialogue_benchmark.fact_index import build_evidence_groups, merge_scopes
from dialogue_benchmark.llm import (ChatClient, FACT_FORMAT, GENERAL_FACT_PROMPT,
    ModelStageError, evidence_projection, extract_facts, generate_from_facts,
    parse_text_response, request_size, review_candidates, stage_error)
from dialogue_benchmark.quality import validate_candidates, validate_facts


class ProjectionContractTests(unittest.TestCase):
    def setUp(self):
        self.scope = {
            "seed": "src/service.py",
            "cutoff": 4,
            "qa_mode": "code",
            "dialogue": [
                {"id": "m1", "order": 1, "kind": "message", "role": "user",
                 "text": "旧行为"},
                {"id": "m2", "order": 2, "kind": "message", "role": "assistant",
                 "text": "中间无关内容"},
                {"id": "m3", "order": 3, "kind": "message", "role": "user",
                 "text": "新行为"},
            ],
            "events": [
                {"id": "e1", "order": 1, "kind": "result",
                 "affected_paths": ["src/service.py"]},
                {"id": "e3", "order": 3, "kind": "patch",
                 "affected_paths": ["src/service.py"]},
            ],
            "versions": [
                {"id": "v1", "observed_at": 1, "path": "src/service.py",
                 "source": "e1", "previous": None},
                {"id": "v2", "observed_at": 3, "path": "src/service.py",
                 "source": "e3", "previous": "v1"},
            ],
            "edges": [{"from": "src/service.py", "to": "src/service.py::run",
                       "source": "e3"}],
            "historical_edges": [],
            "nodes": [{"id": "src/service.py"}],
            "subgraph_events": [{"id": "internal-search-state"}],
            "adaptive": {"layers": [{"depth": 99}], "selected_hops": 3},
            "candidate_seed_event": "e3",
            "model_request_chars": 16000,
            "max_context_chars": 16000,
        }

    def test_projection_drops_authoring_bookkeeping_and_keeps_lossless_sources(self):
        projected = evidence_projection(self.scope, {"e3"}, padding_records=0)
        self.assertEqual([record["id"] for record in projected["dialogue"]], ["m3"])
        self.assertEqual([event["id"] for event in projected["events"]], ["e3"])
        self.assertEqual([version["id"] for version in projected["versions"]], ["v2"])
        self.assertNotIn("nodes", projected)
        self.assertNotIn("subgraph_events", projected)
        self.assertNotIn("adaptive", projected)

    def test_parent_id_alias_is_valid_and_projects_fragment(self):
        scope = {
            "dialogue": [{"id": "m1#fragment-1", "parent_id": "m1", "order": 1,
                          "kind": "message", "role": "user", "text": "完整片段"}],
            "events": [], "versions": [], "edges": [], "historical_edges": [],
        }
        fact = {"id": "f1", "statement": "保留完整片段", "sources": ["m1"]}
        facts, rejected = validate_facts({"facts": [fact]}, scope,
                                         return_rejected=True, qa_mode="general")
        self.assertEqual([item["id"] for item in facts], ["f1"])
        self.assertFalse(rejected)
        projected = evidence_projection(scope, {"m1"}, padding_records=0)
        self.assertEqual([item["id"] for item in projected["dialogue"]],
                         ["m1#fragment-1"])

    def test_review_projection_keeps_the_full_small_fact_group(self):
        facts = [
            {"id": "f1", "statement": "候选相关", "sources": ["m1"]},
            {"id": "f2", "statement": "不相关", "sources": ["m3"]},
        ]
        candidate = {
            "id": "q1", "qa_mode": "general", "type": "single-hop",
            "difficulty": "easy", "difficulty_reason": "direct",
            "memory_requirement": "恢复约束", "fact_ids": ["f1"],
            "answer_target": "旧行为",
            "question": "旧行为是什么？",
            "answer_points": [{"text": "旧行为", "sources": ["m1"]}],
            "forbidden_points": [],
        }
        payloads = []

        class Client:
            def ask(self, prompt, data):
                payloads.append(data)
                return {"reviews": []}

        review_candidates(self.scope, facts, [candidate], Client(), qa_mode="general")
        self.assertEqual(len(payloads), 1)
        payload = payloads[0]
        self.assertEqual([fact["id"] for fact in payload["facts"]], ["f1", "f2"])
        self.assertEqual([record["id"] for record in payload["scope"]["dialogue"]],
                         ["m1", "m2", "m3"])

    def test_focus_projection_failure_stops_before_qa(self):
        fact = {"id": "f1", "statement": "事实", "sources": ["m1"]}

        class Client:
            def ask(self, prompt, data):
                raise AssertionError("projection should fail before transport")

        with patch("dialogue_benchmark.llm.simple_focus_payload",
                   side_effect=ValueError("synthetic projection failure")):
            result = generate_from_facts(self.scope, [fact], Client(), qa_mode="general",
                                         allowed_types={"single-hop"})
        self.assertEqual(result["facts"], [fact])
        self.assertEqual(result["stage_status"]["focus"], "failed")
        self.assertEqual(result["stage_status"]["qa"], "not_submitted")
        self.assertEqual(result["stage_errors"][0]["stage"], "focus")

    def test_explicitly_incomplete_scope_blocks_adversarial_group(self):
        scope = copy.deepcopy(self.scope)
        scope["full_range_covered"] = False
        merged = merge_scopes([scope], "general", model_request_chars=32000)
        self.assertFalse(merged["full_range_covered"])
        fact = {"id": "f1", "statement": "用户要求旧行为", "sources": ["m1"]}
        groups = build_evidence_groups([fact], [scope], "general", {"adversarial"},
                                       max_groups=2, target_chars=8000, max_chars=16000)
        self.assertFalse(groups)

    def test_general_fact_prompt_defines_parseable_protocol(self):
        prompt = GENERAL_FACT_PROMPT
        example = prompt[prompt.index("FACT f1"):prompt.index("END_FACT") + len("END_FACT")]
        document = parse_text_response(example)
        self.assertEqual(document["facts"][0]["id"], "f1")
        self.assertIn(FACT_FORMAT, prompt)

    def test_no_qa_is_completed_empty_generation(self):
        class Client:
            def ask(self, prompt, data):
                return parse_text_response("NO_QA")
        fact = {"id": "f1", "statement": "旧行为", "sources": ["m1"]}
        result = generate_from_facts(self.scope, [fact], Client(), qa_mode="general",
                                     target_type="single-hop")
        self.assertEqual(result["stage_status"]["qa"], "completed")
        self.assertEqual(result["questions"], [])
        self.assertEqual(result["stage_errors"], [])

    def test_review_trims_only_optional_context_and_keeps_fact_sources(self):
        scope = copy.deepcopy(self.scope)
        scope["dialogue"][1]["text"] = "optional noise" * 4000
        scope["model_request_chars"] = 9000
        facts = [{"id": "f1", "statement": "变更", "sources": ["m1", "m3"]}]
        candidate = {"id": "q1", "fact_ids": ["f1"],
                     "answer_points": [{"text": "旧行为", "sources": ["m1"]}],
                     "forbidden_points": []}
        payloads = []
        class Client:
            def ask(self, prompt, data):
                self_size = request_size(prompt, data)
                assert self_size <= 9000
                payloads.append(data)
                return {"reviews": []}
        result = review_candidates(scope, facts, [candidate], Client())
        self.assertEqual(result["stage_status"]["review"], "completed")
        self.assertEqual([r["id"] for r in payloads[0]["scope"]["dialogue"]], ["m1", "m3"])

    def test_review_does_not_drop_oversized_required_evidence(self):
        scope = copy.deepcopy(self.scope)
        scope["dialogue"][0]["text"] = "required evidence" * 4000
        facts = [{"id": "f1", "statement": "旧行为", "sources": ["m1"]}]
        candidate = {"id": "q1", "fact_ids": ["f1"],
                     "answer_points": [{"text": "旧行为", "sources": ["m1"]}],
                     "forbidden_points": []}
        class Client:
            def ask(self, prompt, data):
                raise AssertionError("Must fail before network")
        saved = {}
        result = review_candidates(scope, facts, [candidate], Client(),
                                   checkpoint=lambda name, data: saved.update({name: data}))
        self.assertEqual(result["questions"][0]["status"], "needs_review")
        self.assertEqual(saved["review_single-error.json"]["error_code"], "request_budget")


class TransportDiagnosticsTests(unittest.TestCase):
    def test_split_review_uses_complete_text_blocks_through_http_envelope(self):
        candidate_id = "general_g1_q1"
        question = """REVIEW general_g1_q1
review_contract: structured_v2
unambiguous: true
difficulty_justified: true
not_answer_leaking: true
natural_wording: true
practical_useful: true
type_correct: true
history_requirement_correct: true
current_snapshot_alone_sufficient: true
history_evidence_required: false
external_knowledge_separated: true
external_knowledge_necessary: true
full_range_checked: true
recommended_type: single-hop
recommended_track: none
type_basis: 单条消息直接给出
necessary_source_ids: m1
necessary_stage_ids: none
history_only_fact: none
history_source_ids: none
useful_task: 恢复配置约定
useful_decision: 选择配置格式
answer_effect: 决定解析器输入
point_evidence: A1=supported@m1
causal_bridge: none
reason: 题目合同完整
END_REVIEW"""
        answer = """REVIEW general_g1_q1
review_contract: structured_v2
answer_requirements: R1=配置格式
requirement_coverage: R1=A1
point_claims: A1.1=配置使用 yaml
reason: 答案合同完整
END_REVIEW"""
        envelopes = [
            {"usage": {"total_tokens": 20}, "choices": [{"finish_reason": "stop",
             "message": {"content": content}}]}
            for content in (answer, question)
        ]
        scope = {
            "dialogue": [{"id": "m1", "kind": "message", "role": "user",
                          "order": 1, "stage_id": "s1", "text": "配置使用 yaml"}],
            "events": [], "versions": [], "stages": [{"id": "s1"}],
            "model_request_chars": 20000, "max_context_chars": 20000,
        }
        facts = [{"id": "f1", "statement": "配置使用 yaml", "sources": ["m1"]}]
        candidate = {
            "id": candidate_id, "candidate_id": candidate_id, "model_id": "q1",
            "qa_mode": "general", "type": "single-hop", "difficulty": "easy",
            "difficulty_reason": "直接事实", "memory_requirement": "恢复配置约定",
            "use_case": "实现解析器时选择格式", "answer_target": "配置格式",
            "fact_ids": ["f1"], "question": "配置使用什么格式？",
            "answer_points": [{"text": "配置使用 yaml。", "sources": ["m1"]}],
            "forbidden_points": [], "track": None,
        }
        with patch.dict("os.environ", {"BENCHMARK_API_KEY": "test-placeholder-key"}), \
                patch("dialogue_benchmark.llm.urllib.request.build_opener") as opener:
            opener.return_value.open.side_effect = [
                io.BytesIO(json.dumps(item).encode()) for item in envelopes]
            client = ChatClient("https://example.invalid/chat/completions", "test")
            result = review_candidates(
                scope, facts, [candidate], client, qa_mode="general",
                review_mode="split", allow_repair=False)
        self.assertEqual(result["questions"][0]["status"], "approved")
        self.assertTrue(result["questions"][0]["review"]["answer_complete"])
        self.assertTrue(result["questions"][0]["review"]["atomic_points_correct"])
        self.assertEqual(len(client.responses), 2)
        self.assertEqual([item["stage"] for item in client.usage],
                         ["review_structure", "review_evidence"])
        for call in opener.return_value.open.call_args_list:
            sent = json.loads(call.args[0].data)["messages"][1]["content"]
            prompt = sent.split("\nDATA:\n", 1)[0]
            self.assertIn("REVIEW " + candidate_id, prompt)
            self.assertEqual(prompt.count("END_REVIEW"), 2)

    def test_malformed_visible_output_is_checkpointed_without_reasoning(self):
        response = {"usage": {"total_tokens": 123}, "choices": [{"finish_reason": "stop",
                    "message": {"content": "ordinary prose", "reasoning_content": "private reasoning"}}]}
        saved = {}
        with patch.dict("os.environ", {"BENCHMARK_API_KEY": "test-placeholder-key"}), \
                patch("dialogue_benchmark.llm.urllib.request.build_opener") as opener:
            opener.return_value.open.return_value = io.BytesIO(json.dumps(response).encode())
            client = ChatClient("https://example.invalid/chat/completions", "test")
            result = extract_facts({"dialogue": []}, client, qa_mode="general",
                                   checkpoint=lambda name, data: saved.update({name: data}))
        self.assertEqual(result["stage_errors"][0]["error_code"], "protocol_error")
        self.assertEqual(client.usage[0]["total_tokens"], 123)
        self.assertEqual(client.usage[0]["request_count"], 1)
        self.assertEqual(client.usage[0]["status"], "protocol_error")
        self.assertEqual(saved["facts-response.json"], {"content": "ordinary prose"})
        self.assertNotIn("private reasoning", json.dumps(saved))

    def test_transport_errors_are_distinct_and_do_not_log_bodies(self):
        cases = [
            (urllib.error.HTTPError("private-url", 401, "private-body", {}, None), "http_error"),
            (urllib.error.URLError("private-network-detail"), "connection_error"),
            (TimeoutError("private-timeout"), "timeout"),
        ]
        for error, expected in cases:
            with self.subTest(expected=expected), \
                    patch.dict("os.environ", {"BENCHMARK_API_KEY": "test-placeholder-key"}), \
                    patch("dialogue_benchmark.llm.urllib.request.build_opener") as opener:
                opener.return_value.open.side_effect = error
                client = ChatClient("https://example.invalid/chat/completions", "test")
                with self.assertRaises(ModelStageError) as caught:
                    client.ask("test", {"dialogue": []})
                diagnostic = stage_error("facts", caught.exception)
                self.assertEqual(diagnostic["error_code"], expected)
                self.assertNotIn("private", json.dumps(diagnostic))

    def test_transport_budget_matches_sent_serialization(self):
        data = {"dialogue": [{"text": "quoted \\\" value", "id": "e1"}],
                "model_request_chars": 10000}
        response = {"choices": [{"finish_reason": "stop", "message": {"content": "NO_FACTS"}}]}
        with patch.dict("os.environ", {"BENCHMARK_API_KEY": "test-placeholder-key"}), \
                patch("dialogue_benchmark.llm.urllib.request.build_opener") as opener:
            opener.return_value.open.return_value = io.BytesIO(json.dumps(response).encode())
            client = ChatClient("https://example.invalid/chat/completions", "test")
            client.ask("test", data)
            sent = json.loads(opener.return_value.open.call_args[0][0].data)
        self.assertEqual(sent["messages"][1]["content"],
                         "test\nDATA:\n" + json.dumps(data, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    unittest.main()
