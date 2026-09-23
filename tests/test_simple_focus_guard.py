"""Focused regressions for simple generation focus and review-guard closure."""

import copy
import json
import unittest

from dialogue_benchmark.fact_index import (
    _review_guard_closure,
    build_evidence_index,
    build_evidence_groups,
)
from dialogue_benchmark.llm import (
    SIMPLE_COMPLETENESS_PROMPT,
    SIMPLE_QA_PROMPT,
    SIMPLE_RELEVANCE_PROMPT,
    SIMPLE_REPAIR_PROMPT,
    _simple_focus_issue,
    generate_from_facts,
    parse_text_response,
    review_candidates,
)
from tests.simple_test_helpers import maybe_focus_response, maybe_relevance_response


class _NoQaClient:
    def __init__(self):
        self.calls = []

    def ask(self, prompt, payload):
        # Target routing is exercised separately in test_memory_types.
        if "review_contract: target_v1" in prompt:
            return {"reviews": [{"id": "q1", "review_contract": "target_v1",
                                 "target_alignment": "aligned"}]}
        self.calls.append((prompt, copy.deepcopy(payload)))
        focus = maybe_focus_response(prompt, payload)
        if focus is not None:
            return focus
        return parse_text_response(
            "NO_QA\nMISSING_KIND: outcome\nMISSING_OBJECT: runner.py")


class SimpleFocusPromptTests(unittest.TestCase):
    def setUp(self):
        self.scope = {
            "cutoff": 3,
            "dialogue": [
                {"id": "selected-source", "order": 1, "kind": "message",
                 "role": "assistant", "text": "端口决定由 18080 改为 18081。"},
                {"id": "side-source", "order": 2, "kind": "message",
                 "role": "assistant", "text": "旁支讨论了另一个缓存方案。"},
            ],
            "events": [],
            "versions": [],
            "stages": [],
            "evidence_group": {"target_types": ["constraint_followthrough"]},
            "model_request_chars": 60000,
            "max_context_chars": 60000,
        }
        self.fact = {
            "id": "f-selected",
            "qa_mode": "general",
            "statement": "端口决定由 18080 改为 18081。",
            "sources": ["selected-source"],
        }

    def test_simple_focus_and_qa_payloads_are_local_to_selected_facts(self):
        client = _NoQaClient()
        result = generate_from_facts(
            self.scope, [self.fact], client, qa_mode="general",
            allowed_types=("constraint_followthrough",), target_type="constraint_followthrough",
        )

        self.assertEqual(result["stage_status"]["focus"], "completed")
        self.assertEqual(result["stage_status"]["qa"], "completed")
        self.assertEqual(len(client.calls), 2)
        focus_prompt, focus_payload = client.calls[0]
        qa_prompt, qa_payload = client.calls[1]
        self.assertIn("The focus states one specific task or decision",
                      " ".join(focus_prompt.split()))
        self.assertIn("focus", qa_payload)
        self.assertNotIn("relationship_target", qa_payload)
        self.assertNotIn("FOCUS:", qa_prompt)

        focus_text = json.dumps(focus_payload, ensure_ascii=False)
        qa_text = json.dumps(qa_payload, ensure_ascii=False)
        self.assertNotIn("selected-source", focus_text)
        self.assertNotIn("side-source", focus_text)
        self.assertNotIn("f-selected", focus_text)
        self.assertNotIn("selected-source", qa_text)
        self.assertNotIn("side-source", qa_text)
        self.assertNotIn("f-selected", qa_text)
        self.assertEqual(len(focus_payload["material_references"]), 1)
        self.assertEqual(len(qa_payload["materials"]), 1)

    def test_simple_prompt_requires_grounded_relationships_and_concrete_forbidden_claims(self):
        prompt = " ".join(SIMPLE_QA_PROMPT.split())
        self.assertIn("A relation never proves cause, importance, or correctness", prompt)
        self.assertIn("Time order alone is not causation", prompt)
        self.assertIn("补丁应用成功不代表运行正确", prompt)
        self.assertIn("答案可以多行", prompt)
        self.assertIn("不同对象/动作必须分行", prompt)
        self.assertIn("材料直接否定一个具体错误结论", prompt)
        self.assertIn("链条末端的行为或结果", prompt)

    def test_terminal_behavior_is_kept_and_checked(self):
        relevance = " ".join(SIMPLE_RELEVANCE_PROMPT.split())
        completeness = " ".join(SIMPLE_COMPLETENESS_PROMPT.split())
        self.assertIn("terminal behavior, failure, or outcome", relevance)
        self.assertIn("requested terminal behavior or outcome", completeness)
        self.assertIn("If the question ends at consumption by a named API argument",
                      relevance)
        self.assertIn("one continuous value -> comparison -> one error/no-error",
                      " ".join(SIMPLE_REPAIR_PROMPT.split()))
        self.assertIn("Never leave that signature-plus-transfer sentence joined",
                      " ".join(SIMPLE_REPAIR_PROMPT.split()))

    def test_one_turn_environment_housekeeping_is_not_a_general_focus(self):
        issue = _simple_focus_issue(
            {"text": "本轮是否允许创建或安装 `.venv-online` 环境"},
            "general", "constraint_followthrough")
        self.assertIsNotNone(issue)
        self.assertIn("只对本轮有效", issue)

    def test_plan_approval_status_is_not_a_general_focus(self):
        issue = _simple_focus_issue(
            {"text": "开始修改前需要确认用户对完整实施计划的批准状态"},
            "general", "constraint_followthrough")
        self.assertIsNotNone(issue)
        self.assertIn("不选“是否已批准/已确认”", issue)

    def test_persistent_compatibility_constraint_remains_eligible(self):
        issue = _simple_focus_issue(
            {"text": "新适配器是否必须保持与旧版 CLI 的参数兼容"},
            "general", "constraint_followthrough")
        self.assertIsNone(issue)

    def test_unseen_upstream_cli_consumer_is_not_a_behavior_focus(self):
        issue = _simple_focus_issue(
            {"text": "config_paths 如何通过 --config 传递到上游 CLI 的配置消费点"},
            "code", "compatibility_preservation")
        self.assertIsNone(issue)

    def test_two_config_consumers_are_not_one_behavior_focus(self):
        issue = _simple_focus_issue(
            {"text": "说明 --config 的插入位置与模板加载的路径序列"},
            "code", "compatibility_preservation")
        self.assertIsNotNone(issue)
        self.assertIn("不要把两个消费方向合成一题", issue)

        issue = _simple_focus_issue(
            {"text": "--config 的展开顺序与模板加载传入的路径序列"},
            "code", "compatibility_preservation")
        self.assertIsNotNone(issue)

    def test_config_insertion_and_loading_are_not_one_causal_focus(self):
        issue = _simple_focus_issue(
            {"text": "--config 参数的插入位置与顺序，以及该顺序如何影响分层配置的加载"},
            "code", "compatibility_preservation")
        self.assertIsNotNone(issue)
        self.assertIn("不要把两个并行消费方向写成因果链", issue)

    def test_validation_and_transfer_are_not_one_behavior_focus(self):
        issue = _simple_focus_issue(
            {"text": "config_paths 经唯一性校验后插入命令并传给模板加载器"},
            "code", "compatibility_preservation")
        self.assertIsNotNone(issue)
        self.assertIn("只选一个值或条件", issue)

    def test_value_flow_focus_does_not_center_parameter_declarations(self):
        issue = _simple_focus_issue(
            {"text": "run 方法新增的 timeout_seconds 参数如何经 _run_command 传递"},
            "code", "compatibility_preservation")
        self.assertIsNotNone(issue)
        self.assertIn("具体的值传递链", issue)

    def test_config_paths_do_not_claim_to_choose_a_fixed_insertion_index(self):
        issue = _simple_focus_issue(
            {"text": "config_paths 路径序列如何决定 --config 参数的插入位置"},
            "code", "compatibility_preservation")
        self.assertIsNone(issue)


class ReviewGuardFocusTests(unittest.TestCase):
    @staticmethod
    def _infos(scope, facts):
        index = build_evidence_index(facts, [scope], "code", 60000)
        return index, index["info_by_id"]

    def test_same_object_versions_are_kept_but_sibling_symbol_is_not_a_new_anchor(self):
        scope = {
            "cutoff": 4,
            "dialogue": [
                {"id": "e1", "order": 1, "kind": "patch", "text": "a.py load_config old"},
                {"id": "e2", "order": 2, "kind": "patch", "text": "a.py load_config base"},
                {"id": "e3", "order": 3, "kind": "patch", "text": "a.py load_config later"},
                {"id": "e4", "order": 4, "kind": "patch", "text": "a.py save_cache later"},
            ],
            "events": [],
            "versions": [
                {"id": "v1", "path": "a.py", "observed_at": 1,
                 "source": "e1", "previous": None, "status": "known"},
                {"id": "v2", "path": "a.py", "observed_at": 2,
                 "source": "e2", "previous": "v1", "status": "known"},
                {"id": "v3", "path": "a.py", "observed_at": 3,
                 "source": "e3", "previous": "v2", "status": "known"},
                {"id": "v4", "path": "a.py", "observed_at": 4,
                 "source": "e4", "previous": "v3", "status": "known"},
            ],
            "edges": [], "historical_edges": [],
        }
        facts = [
            {"id": "a-old", "statement": "之前 a.py::load_config 返回 None",
             "sources": ["v1"]},
            {"id": "a-base", "statement": "当前 a.py::load_config 返回默认配置",
             "sources": ["v2"]},
            {"id": "a-later", "statement": "后来 a.py::load_config 增加空值校验",
             "sources": ["v3"]},
            {"id": "b-sibling", "statement": "后来 a.py::save_cache 修改日志格式",
             "sources": ["v4"]},
        ]
        index, info_by_id = self._infos(scope, facts)

        closure = _review_guard_closure(
            [info_by_id["a-base"]], index["infos"])

        self.assertEqual(
            {item["fact"]["id"] for item in closure},
            {"a-old", "a-base", "a-later"},
        )
        self.assertNotIn("b-sibling", {item["fact"]["id"] for item in closure})

    def test_dependency_anchor_keeps_its_updates_but_cannot_bridge_to_next_file(self):
        scope = {
            "cutoff": 5,
            "dialogue": [
                {"id": "main-source", "order": 1, "kind": "observation",
                 "changes": {"main.py": {"content": "run"}}},
                {"id": "adapter-old-source", "order": 2, "kind": "observation",
                 "changes": {"adapter.py": {"content": "dispatch old"}}},
                {"id": "adapter-new-source", "order": 3, "kind": "patch",
                 "changes": {"adapter.py": {"content": "dispatch new"}}},
                {"id": "config-source", "order": 4, "kind": "observation",
                 "changes": {"config.py": {"content": "load"}}},
            ],
            "events": [],
            "versions": [
                {"id": "adapter-v1", "path": "adapter.py", "observed_at": 2,
                 "source": "adapter-old-source", "previous": None, "status": "known"},
                {"id": "adapter-v2", "path": "adapter.py", "observed_at": 3,
                 "source": "adapter-new-source", "previous": "adapter-v1", "status": "known"},
            ],
            "edges": [
                {"from": "main.py::run", "to": "adapter.py::dispatch",
                 "kind": "call_reference", "source": "main-source"},
                {"from": "adapter.py::dispatch", "to": "config.py::load",
                 "kind": "call_reference", "source": "adapter-new-source"},
            ],
            "historical_edges": [],
        }
        facts = [
            {"id": "main", "statement": "main.py::run 在空输入时进入处理分支",
             "sources": ["main-source"]},
            {"id": "adapter-old", "statement": "adapter.py::dispatch 之前返回默认配置",
             "sources": ["adapter-v1"]},
            {"id": "adapter-new", "statement": "adapter.py::dispatch 后来增加校验",
             "sources": ["adapter-v2"]},
            {"id": "config", "statement": "config.py::load 后来读取配置文件",
             "sources": ["config-source"]},
        ]
        index, info_by_id = self._infos(scope, facts)

        closure = _review_guard_closure(
            [info_by_id["main"]], index["infos"])
        closure_ids = {item["fact"]["id"] for item in closure}

        self.assertTrue({"main", "adapter-old", "adapter-new"} <= closure_ids)
        self.assertNotIn("config", closure_ids)

    def test_mixed_dependency_fact_cannot_make_its_second_path_a_new_anchor(self):
        scope = {
            "cutoff": 3,
            "dialogue": [
                {"id": "main-source", "order": 1, "kind": "observation",
                 "changes": {"main.py": {"content": "run"}}},
                {"id": "mixed-source", "order": 2, "kind": "patch",
                 "changes": {
                     "adapter.py": {"content": "dispatch"},
                     "config.py": {"content": "load"},
                 }},
                {"id": "config-later-source", "order": 3, "kind": "patch",
                 "changes": {"config.py": {"content": "load later"}}},
            ],
            "events": [], "versions": [],
            "edges": [{
                "from": "main.py::run", "to": "adapter.py::dispatch",
                "kind": "call_reference", "source": "main-source",
            }],
            "historical_edges": [],
        }
        facts = [
            {"id": "main", "statement": "main.py::run 进入处理分支",
             "sources": ["main-source"]},
            {"id": "mixed", "statement": (
                "adapter.py::dispatch 与 config.py::load 一起被修改"),
             "sources": ["mixed-source"]},
            {"id": "config-later", "statement":
             "config.py::load 后来修改读取方式", "sources": ["config-later-source"]},
        ]
        index, info_by_id = self._infos(scope, facts)

        closure = _review_guard_closure(
            [info_by_id["main"]], index["infos"])
        closure_ids = {item["fact"]["id"] for item in closure}

        self.assertIn("main", closure_ids)
        self.assertNotIn("config-later", closure_ids)

    def test_call_result_closure_is_retained_without_using_other_facts_as_anchors(self):
        scope = {
            "cutoff": 3,
            "dialogue": [
                {"id": "call", "order": 1, "kind": "call", "call_id": "c1",
                 "text": "adapter.py::dispatch()"},
                {"id": "result", "order": 2, "kind": "result", "call_id": "c1",
                 "text": "adapter.py::dispatch returned failure"},
                {"id": "other", "order": 3, "kind": "result", "call_id": "c2",
                 "text": "config.py::load returned success"},
            ],
            "events": [], "versions": [], "edges": [], "historical_edges": [],
        }
        facts = [
            {"id": "call-fact", "statement": "adapter.py::dispatch 被调用",
             "sources": ["call"]},
            {"id": "result-fact", "statement": "adapter.py::dispatch 返回失败",
             "sources": ["result"]},
            {"id": "other-fact", "statement": "config.py::load 返回成功",
             "sources": ["other"]},
        ]
        index, info_by_id = self._infos(scope, facts)

        closure = _review_guard_closure(
            [info_by_id["call-fact"]], index["infos"])
        closure_ids = {item["fact"]["id"] for item in closure}

        self.assertEqual(closure_ids, {"call-fact", "result-fact"})

    def test_over_budget_guard_remains_incomplete_for_downstream_review(self):
        scope = {
            "cutoff": 2,
            "dialogue": [
                {"id": "call", "order": 1, "kind": "call", "call_id": "c1",
                 "text": "run check_config"},
                {"id": "result", "order": 2, "kind": "result", "call_id": "c1",
                 "text": "check_config result " + "x" * 20000},
            ],
            "events": [], "versions": [], "edges": [], "historical_edges": [],
        }
        fact = {"id": "call-fact", "statement": "check_config 必须被调用",
                "sources": ["call"]}
        groups = build_evidence_groups(
            [fact], [scope], "code", {"constraint_followthrough"}, 1,
            target_chars=2000, max_chars=6000)

        self.assertEqual(len(groups), 1)
        self.assertFalse(groups[0]["review_guard_complete"])
        self.assertEqual(groups[0]["scope"]["review_guard_reason"], "over_budget")
        self.assertGreater(groups[0]["scope"]["review_guard_omitted_count"], 0)


class SimpleReviewGuardOrderingTests(unittest.TestCase):
    class Client:
        def __init__(self, responses):
            self.responses = iter(responses)
            self.calls = []
            self.usage = []

        def ask(self, prompt, payload):
            # Target routing is exercised separately in test_memory_types.
            if "review_contract: target_v1" in prompt:
                return {"reviews": [{"id": "q1", "review_contract": "target_v1",
                                     "target_alignment": "aligned"}]}
            self.calls.append((prompt, copy.deepcopy(payload)))
            self.usage.append({"status": "completed"})
            relevance = maybe_relevance_response(prompt, payload)
            if relevance is not None:
                return relevance
            response = next(self.responses)
            return parse_text_response(response) if isinstance(response, str) else response

    @staticmethod
    def _scope(complete=False):
        return {
            "cutoff": 1,
            "dialogue": [{
                "id": "m1", "kind": "message", "role": "user", "order": 1,
                "text": "配置必须支持 yaml。",
            }],
            "events": [], "versions": [], "stages": [],
            "review_guard_complete": complete,
            "review_guard_reason": "over_budget",
            "model_request_chars": 60000,
            "max_context_chars": 60000,
        }

    @staticmethod
    def _facts():
        return [{"id": "f1", "statement": "配置必须支持 yaml", "sources": ["m1"]}]

    @staticmethod
    def _candidate(qa_mode="general", question_type="constraint_followthrough"):
        return {
            "id": "q1", "candidate_id": "q1", "model_id": "q1",
            "qa_mode": qa_mode, "type": question_type, "fact_ids": ["f1"],
            "question": "配置必须支持什么格式？",
            "answer_points": [{"text": "配置必须支持 yaml。", "sources": ["m1"]}],
            "forbidden_points": [{"text": "配置只支持 JSON。", "sources": ["m1"]}],
        }

    @staticmethod
    def _atomicity(value="single"):
        return ("REVIEW q1\nreview_contract: simple_atomicity_v1\n"
                "point_atomicity: A1=%s;F1=single\nEND_REVIEW" % value)

    @staticmethod
    def _completeness():
        return ("REVIEW q1\nreview_contract: simple_v1\n"
                "completeness: complete\nEND_REVIEW")

    def test_compound_atomicity_is_rejected_even_when_guard_is_incomplete(self):
        client = self.Client([self._atomicity("compound")])
        result = review_candidates(
            self._scope(), self._facts(), [self._candidate()], client,
            qa_mode="general", review_mode="simple", allow_repair=False)

        self.assertFalse(result["questions"])
        self.assertEqual(result["rejected"][0]["reason"],
                         "semantic_answer_atomicity_failed")
        self.assertEqual(len(client.calls), 2)
        self.assertEqual([item["stage"] for item in client.usage],
                         ["review_relevance", "review_atomicity"])

    def test_structurally_valid_candidate_stays_pending_before_evidence_when_guard_is_incomplete(self):
        for qa_mode, question_type, prefix in (
                ("general", "constraint_followthrough", []),
                ("code", "constraint_followthrough", [])):
            with self.subTest(qa_mode=qa_mode):
                responses = prefix + [self._atomicity(), self._completeness()]
                client = self.Client(responses)
                result = review_candidates(
                    self._scope(), self._facts(),
                    [self._candidate(qa_mode, question_type)], client,
                    qa_mode=qa_mode, review_mode="simple", allow_repair=False)

                self.assertEqual(result["questions"][0]["status"], "needs_review")
                self.assertEqual(result["questions"][0]["review_error"],
                                 "incomplete_review_guard")
                self.assertEqual(len(client.calls), 3 + len(prefix))
                self.assertEqual([item["stage"] for item in client.usage],
                                 ["review_relevance", "review_atomicity", "review_completeness"])
                self.assertEqual(result["stage_status"].get("review_evidence"),
                                 "skipped")
                self.assertEqual(result["stage_status"]["review"], "blocked")


if __name__ == "__main__":
    unittest.main()
