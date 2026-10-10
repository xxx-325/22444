"""Offline checks of the unified external QA-to-requirement contract."""

import contextlib
import io
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dialogue_benchmark import cli
from dialogue_benchmark.external import external_review_projection, filter_external_facts, load_external_scopes
from dialogue_benchmark.fact_index import build_evidence_index, static_candidate_labels, static_evidence_check
from dialogue_benchmark.llm import extract_facts, generate_from_facts, review_candidates, parse_text_response
from dialogue_benchmark.protocol import MEMORY_TYPES, MEMORY_TYPE_GUIDANCE, MEMORY_TASK_GUIDANCE
from dialogue_benchmark.task_eval.artifacts import qa_inputs, save, read
from dialogue_benchmark.task_eval.prompts import task_direction
from tests.test_memory_types import Client
from viewer.build_data import build


class ExternalClient(Client):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def ask(self, prompt, payload):
        if "Extract only externally supplied" in prompt:
            self.calls.append((prompt, payload))
            self.usage.append({"status": "completed"})
            return {"facts": [{"id": "f1", "statement": "导出必须保留空值。", "sources": ["资料1"]}]}
        response = super().ask(prompt, payload)
        if "point_evidence:" in prompt:
            response["reviews"][0].update(usage="confirmed@资料1", usage_reason="用户确认导出必须保留空值。")
        return response


class OrderChoiceClient(Client):
    def ask(self, prompt, payload):
        response = super().ask(prompt, payload)
        if "WORKFLOW:" in prompt:
            response["workflow"]["text"] = "续办 Northwind 2025-04-18-A 班次：读取订单 → 处理待办处置 → 交付汇总"
        elif "FOCUS:" in prompt:
            response["focus"]["text"] = "找回 Northwind 2025-04-18-A 班次 ORD-NW-101 的已选处置及适用条件"
        elif "QUESTION: 自然问题" in prompt:
            return parse_text_response(
                "QA q1\nQUESTION: Northwind 2025-04-18-A 班次的 ORD-NW-101 应按哪项已确认决定继续交付？\n"
                "ANSWER_POINT: ORD-NW-101 在该班次按 op-5 的 release/resolved 决定继续处理。"
                " || SOURCES: 资料3\nEND_QA")
        elif "point_evidence:" in prompt:
            response["reviews"][0].update(
                point_evidence="A1=supported@资料3", usage="confirmed@资料3",
                usage_reason="仓库确认托盘数后，客户已更新该班次该订单的处置决定。")
        return response


class ExternalMemoryPipelineTests(unittest.TestCase):
    def test_external_fact_prompt_separates_customer_choice_from_api_specs(self):
        choice = "Maple 本次晚班作业选择每批 2 条，批次前缀 MP。"
        scope = self.scope("M6")
        scope["dialogue"][0]["text"] = (
            "新增 dispatch(records, batch_size=10, prefix='B')，正整数校验失败时抛 ValueError。\n"
            + choice + "请据此运行本次作业。")

        class ChoiceClient(Client):
            def ask(self, prompt, payload):
                self.calls.append((prompt, payload))
                self.usage.append({"status": "completed"})
                return parse_text_response("FACT f1\nSOURCES: 资料1\nSOURCE_KIND: conversation\nTEXT: "
                                           + choice + "\nEND_FACT")

        client = ChoiceClient()
        result = extract_facts(scope, client, qa_mode="memory", external_only=True)
        self.assertEqual(result["stage_errors"], [])
        fact, = filter_external_facts(result["facts"], [scope])
        self.assertEqual(fact["statement"], choice)
        self.assertEqual(fact["sources"], ["e1"])
        self.assertEqual(len(client.calls), 1)
        material, = client.calls[0][1]["materials"]
        self.assertEqual(material["original_records"][0]["text"], scope["dialogue"][0]["text"])
        prompt = " ".join(client.calls[0][0].split())
        self.assertIn("API requirements and corrections are not external business facts by themselves", prompt)
        self.assertIn("customer's actual choice, authorization, business agreement, or external state", prompt)
        self.assertIn("including its object and applicable scope", prompt)
        self.assertIn("Generic API behavior and sample data are context, not separate fact targets", prompt)
        self.assertIn("customer selection for one named job or cycle remains a fact", prompt)
        self.assertIn("do not infer a permanent policy", prompt)
        self.assertIn("Keep technical feasibility, customer authorization and observed completion separate", prompt)
        self.assertIn("does not establish dispatch or execution", prompt)
        self.assertIn("Later stages assess future usefulness and final-repository recoverability", prompt)

    def test_external_fact_extraction_indexes_optional_tool_output_and_keeps_public_correction(self):
        scope = self.scope("M1")
        scope['dialogue'].extend([
            dict(id='e3', kind='result', order=3,
                 text='EXPORT_IMPLEMENTATION_ONLY ' * 20000),
            dict(id='e4', kind='message', role='user', order=4,
                 text='客户 A 的 note 例外改为省略，其他空字段继续保留。'),
        ])
        scope['external_context_source_ids'] = ['e3', 'e4']
        scope['model_request_chars'] = 48000
        client = ExternalClient()
        result = extract_facts(scope, client, qa_mode='memory', external_only=True)
        self.assertEqual(result['stage_errors'], [])
        self.assertEqual(result['stage_status']['facts'], 'completed')
        self.assertEqual(len(client.calls), 1)
        payload = str(client.calls[0][1])
        self.assertNotIn('EXPORT_IMPLEMENTATION_ONLY', payload)
        self.assertIn('客户 A 的 note 例外改为省略', payload)
        self.assertEqual(result['facts'][0]['sources'], ['e1'])

    def test_memory_focus_keeps_order_choices_without_neighboring_api_contracts(self):
        scope, facts = self.order_choice_scope()
        client = OrderChoiceClient()
        saved = {}
        generated = generate_from_facts(scope, facts, client, qa_mode="memory", target_type="M2",
            checkpoint=lambda name, value: saved.update({name: value}))
        self.assertEqual(generated["stage_errors"], [])
        self.assertEqual(generated["generation_request_count"], 3)
        focus_request = saved["focus-input.json"]
        focus = focus_request["payload"]
        self.assertEqual(set(focus), {"facts", "material_references", "relations", "workflow"})
        self.assertEqual([fact["text"] for fact in focus["facts"]],
                         [fact["statement"] for fact in facts])
        self.assertEqual([fact["materials"] for fact in focus["facts"]], [["资料1"], ["资料1"]])
        self.assertEqual(focus["facts"], saved["qa-input.json"]["payload"]["facts"])
        self.assertEqual(focus_request["ref_to_source"], saved["workflow-input.json"]["ref_to_source"])
        self.assertEqual(focus_request["ref_to_source"], saved["qa-input.json"]["ref_to_source"])
        self.assertEqual([item["reference"] for item in focus["material_references"]],
                         ["资料1", "资料2", "资料3"])
        for item in focus["material_references"]:
            self.assertFalse({"text", "original_records", "changes"} & set(item))
        for text in ("active_only", "append_disposition", "render_summary", "2025-04-18T16:10:00Z"):
            self.assertNotIn(text, str(focus))
            for name in ("workflow-input.json", "qa-input.json"):
                self.assertIn(text, str(saved[name]["payload"]["materials"]))
        for prompt, _ in client.calls[:3]:
            self.assertIn("外部事实、实际已选决定或状态", prompt)
            self.assertIn("即使写成参数或代码", prompt)
        self.assertIn("facts 是候选集合，不要求全部覆盖", saved["focus-input.json"]["prompt"])
        self.assertIn("未选中的 facts 不追加为答案", saved["qa-input.json"]["prompt"])
        question, = generated["questions"]
        self.assertIn("ORD-NW-101", question["question"])
        self.assertNotIn("ORD-NW-102", question["question"])
        self.assertEqual(len(question["answer_points"]), 1)

        reviewed = review_candidates(scope, facts, generated["questions"], client,
            qa_mode="memory", review_mode="simple", allow_repair=False)
        self.assertEqual(reviewed["stage_errors"], [])
        self.assertEqual(reviewed["questions"][0]["status"], "approved")
        self.assertEqual(reviewed["questions"][0]["question"], question["question"])
        self.assertEqual(reviewed["questions"][0]["answer_points"], question["answer_points"])
        self.assertEqual(reviewed["questions"][0]["answer_points"][0]["sources"], ["e3"])
        target_prompt, target_payload = client.calls[3]
        self.assertIn("focus fixes the selected object, scope, and one decision within facts", target_prompt)
        self.assertIn("Other facts may remain unused", target_prompt)
        self.assertIn("states remain historical targets even when written as parameters or code", target_prompt)
        self.assertNotIn("materials", target_payload)
        evidence_payload = client.calls[-1][1]
        self.assertEqual(evidence_payload["facts"], focus["facts"])
        self.assertIn("2025-04-18T16:10:00Z", str(evidence_payload["materials"]))
        self.assertIn("render_summary", str(evidence_payload["materials"]))
        self.assertEqual(
            [record["text"] for material in evidence_payload["materials"]
             for record in material["original_records"]],
            [record["text"] for record in scope["dialogue"]])
        self.assertEqual([receipt["stage"] for receipt in client.usage], [
            "workflow", "focus", "qa", "review_target", "review_relevance", "review_atomicity",
            "review_completeness", "review_evidence"])

    def test_memory_authoring_can_decline_candidates_without_generating_qa(self):
        for stop_at in ("WORKFLOW:", "FOCUS:"):
            with self.subTest(stop_at=stop_at):
                class NoCandidateClient(Client):
                    def ask(self, prompt, payload):
                        if stop_at in prompt:
                            self.calls.append((prompt, payload))
                            self.usage.append({"status": "completed"})
                            return {"questions": []}
                        return super().ask(prompt, payload)

                scope, facts = self.order_choice_scope()
                client = NoCandidateClient()
                generated = generate_from_facts(scope, facts, client,
                    qa_mode="memory", target_type="M1")
                self.assertEqual(generated["questions"], [])
                self.assertEqual(generated["stage_errors"], [])
                expected = ["workflow"] if stop_at == "WORKFLOW:" else ["workflow", "focus"]
                self.assertEqual([receipt["stage"] for receipt in client.usage], expected)
                self.assertEqual(generated["generation_request_count"], len(expected))

    def test_memory_authoring_preserves_cycle_and_hypothetical_approval(self):
        client = Client()
        generate_from_facts(self.scope("M6"),
            [{"id": "f1", "statement": "本周期仅批准 P1。", "sources": ["e1"]}],
            client, qa_mode="memory", target_type="M6")
        workflow_prompt, focus_prompt, qa_prompt = [prompt for prompt, _ in client.calls]
        self.assertIn("在给定历史已确认的客户、对象、周期和条件内", workflow_prompt)
        self.assertIn("尚未完成、可验收的后续业务工作", workflow_prompt)
        self.assertIn("所需历史决定必须已经能从材料找回", workflow_prompt)
        self.assertIn("不以取得未记录的新批准或新确认为前提", workflow_prompt)
        self.assertIn("假设中的后续批准不是已经发生的更新或局部纠正", workflow_prompt)
        for prompt in (focus_prompt, qa_prompt):
            self.assertIn("Preserve the publicly confirmed customer, object, cycle, and applicability", prompt)
            self.assertIn("A hypothetical later approval is not an actual update or scoped correction", prompt)
            self.assertIn("Do not extend a cycle-limited authorization to other cycles", prompt)
        self.assertEqual([receipt["stage"] for receipt in client.usage], ["workflow", "focus", "qa"])

    def test_m6_focus_and_qa_retrieve_confirmed_history_not_new_approval(self):
        client = Client()
        generate_from_facts(self.scope("M6"),
            [{"id": "f1", "statement": "本周期仅批准 P1。", "sources": ["e1"]}],
            client, qa_mode="memory", target_type="M6")
        _, focus_prompt, qa_prompt = [prompt for prompt, _ in client.calls]
        self.assertIn("从给定历史中找回已经确认", focus_prompt)
        self.assertIn("不是向用户再次取得确认或询问未记录的新状态", focus_prompt)
        self.assertIn("FOCUS: 客户、对象和适用范围内需要找回的一项已确认决定或状态", focus_prompt)
        self.assertIn("focus 从 facts 中选择并固定本题的一个追问对象和决定", qa_prompt)
        self.assertIn("不是必答清单", qa_prompt)
        self.assertIn("只有原文记载的实际纠正才能改变答案", qa_prompt)
        for prompt in (focus_prompt, qa_prompt):
            self.assertIn("within its confirmed scope", prompt)
            self.assertIn("Apply a scoped correction only if one is explicitly recorded", prompt)
            self.assertNotIn("including any scoped correction", prompt)
            self.assertNotIn("需要确认的历史", prompt)
            self.assertNotIn("尚需从历史确认", prompt)
        self.assertEqual([receipt["stage"] for receipt in client.usage], ["workflow", "focus", "qa"])

    @unittest.skipUnless(importlib.util.find_spec("simulator"), "Add the simulator checkout to PYTHONPATH")
    def test_simulator_route_and_disclosed_update_reach_requirement_input(self):
        from simulator.openhands.commit_scenario import expand_tasks
        from simulator.openhands.prepare_scenario import prepare_scenario
        from simulator.openhands.memory_episode import scenario_external_events
        from dialogue_benchmark.openhands_input import normalize_openhands
        from dialogue_benchmark.external import load_external_scopes
        from dialogue_benchmark.task_eval.history import prepare_history
        from dialogue_benchmark.task_eval.selection import select_task

        tasks = [dict(kind="commit", reference=ref, base=base, title="Delivery", body="Extend delivery")
                 for ref, base in (("first", "base"), ("second", "first"))]
        route = {"baseline": "The project supports generic deliveries without customer-specific rules.",
                 "stages": [dict(commit=t["reference"], requirement=t["body"],
                                  history="Private planned choice, disclose after an applicable attempt") for t in tasks]}
        old = dict(id="old", type="M1", text="Keep explicit nulls", scope="Harbor",
                   trigger="Code chooses field selection", behavior="Retain null fields in delivery")
        new = dict(id="new", type="M6", text="Harbor omits note nulls only", scope="Harbor",
                   trigger="Code extends delivery", behavior="Omit only null note fields", supersedes=["old"])
        unreleased = dict(old, id="private", text="UNRELEASED-POLICY")
        drafts = [dict(request=request, facts=facts, extensions=[], repository_edits=[])
                  for request, facts in (("Add customer deliveries", [old, unreleased]),
                                         ("Add delivery reports", [new]))]
        public = [dict(id="goal", kind="user", text="Add customer deliveries"),
                  dict(id="u1", kind="user", text=old["text"]),
                  dict(id="goal2", kind="user", text="Add delivery reports"),
                  dict(id="u2", kind="user", text=new["text"])]
        public = [dict(row, schema="model-visible-dialogue-v1", sequence=i, timestamp=i)
                  for i, row in enumerate(public, 1)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("simulator.openhands.prepare_scenario.source_context", return_value={"patch": "fixture"}), \
                    patch("simulator.openhands.prepare_scenario.call_scenario",
                          side_effect=[drafts[0], {"allowed": True}, drafts[1], {"allowed": True}]):
                scenario = prepare_scenario(None, tasks, None, root / "scenario", development_plan=route)
            checkpoint = dict(tasks=expand_tasks(tasks, (scenario, "fixture-digest")),
                              state=dict(messages=[dict(id="goal", task_id="task-1"),
                                                   dict(id="goal2", task_id="task-2")]))
            sidecar = scenario_external_events(checkpoint, public)
            self.assertNotIn("UNRELEASED-POLICY", str(sidecar))
            self.assertEqual([r["fact_id"] for r in sidecar["rejected"]], ["private"])
            save(root / "external-events.json", sidecar)
            records = normalize_openhands(list(enumerate(public, 1)))
            loaded = load_external_scopes(root / "external-events.json", records, len(records))
            self.assertEqual(loaded["rejected"], [])
            scope, = loaded["scopes"]
            self.assertEqual(scope["external_event_ids"], ["external-old", "external-new"])
            saved = {}
            class RoutedClient(Client):
                def ask(self, prompt, payload):
                    response = super().ask(prompt, payload)
                    for key in ("workflow", "focus"):
                        if key in response:
                            response[key]["sources"] = ["资料2"]
                    return response
            client = RoutedClient()
            generated = generate_from_facts(scope, [dict(id="f1", statement=old["text"], sources=["e2"])],
                client, qa_mode="memory", target_type="M6", checkpoint=lambda name, value: saved.update({name: value}))
            self.assertEqual([u["stage"] for u in client.usage], ["workflow", "focus", "qa"])
            for name in ("workflow-input.json", "qa-input.json"):
                self.assertIn(new["text"], str(saved[name]["payload"]))
                self.assertNotIn("UNRELEASED-POLICY", str(saved[name]))
            self.assertIn("material_references", saved["focus-input.json"]["payload"])
            self.assertNotIn(new["text"], str(saved["focus-input.json"]["payload"]))
            self.assertNotIn("UNRELEASED-POLICY", str(saved["focus-input.json"]))
            question = dict(generated["questions"][0], status="approved")
            save(root / "qa/manifest.json", {"qa_source": "external"})
            save(root / "qa/qa-public.json", {"questions": [question]})
            save(root / "qa/normalized.json", records)
            save(root / "qa/stages/group-raw-candidates.json", {"questions": [question]})
            save(root / "qa/stages/group-qa-input.json", saved["qa-input.json"])
            item, = qa_inputs(root / "qa")
            history = prepare_history(item["public_records"], item["generation_input"], qa_source_ids={"e2"})
            self.assertIn(new["text"], str(history["initial_events"]))
            self.assertEqual({e["id"] for e in history["initial_events"]}, {"goal", "u1", "goal2", "u2"})

            class Budget:
                requests = 0
                tokens = 0

                def call(self, prompt, payload, config, output):
                    self.payload = payload
                    return {"reviews": [dict(decision="pending", reason="Offline contract check", request="none")]}
            budget = Budget()
            select_task(question, history, root, {}, root / "selection", budget,
                        workflow=item["development_workflow"])
            self.assertEqual(budget.payload["development_workflow"], saved["qa-input.json"]["payload"]["workflow"]["text"])

    def test_task_input_keeps_reviewed_sources_and_original_draft_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = dict(id="q1", type="M1", status="approved", answer_points=[
                dict(text="Keep nulls", sources=["e1"])])
            corrected = dict(original, answer_points=[dict(text="Omit note nulls for Harbor only", sources=["e3"])])
            save(root / "qa-public.json", {"questions": [corrected]})
            save(root / "qa-audit.json", {"questions": [corrected]})
            save(root / "manifest.json", {"qa_source": "external"})
            save(root / "stages/group-raw-candidates.json", {"questions": [original]})
            save(root / "normalized.json", [
                {"id": "e1", "kind": "message", "role": "user", "text": "Keep nulls"},
                {"id": "e3", "kind": "message", "role": "user", "text": "Harbor note null exception"},
            ])
            save(root / "stages/group-qa-input.json", {
                "ref_to_source": {"资料1": "e1", "资料3": "e3"},
                "payload": {"scope": {"dialogue": [
                    {"id": "e1"}, {"id": "e3"}]},
                    "workflow": {"text": "Batch delivery", "sources": ["资料3"]}},
            })
            item, = qa_inputs(root)
        self.assertEqual(item["original_candidate"]["answer_points"][0]["sources"], ["e1"])
        self.assertEqual(item["reviewed_candidate"]["answer_points"][0]["sources"], ["e3"])

    def test_task_input_uses_the_recorded_event_instead_of_shared_sources(self):
        events = {"version": 1, "events": [
            dict(id="verification", source_ids=["e1"], task_id="task-1"),
            dict(id="absence", source_ids=["e3"], task_id="task-2"),
        ]}
        for lineage, expected in (
                ({"event_ids": ["absence"]}, ["absence"]),
                (None, ["verification"])):
            with self.subTest(lineage=lineage), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                # e1 is shared conversation context; the question came from "absence".
                question = dict(id="q1", type="M1", status="approved",
                                answer_points=[dict(text="Dispatcher marks absence", sources=["e1", "e3"])])
                if lineage is not None:
                    question["external_lineage"] = lineage
                else:
                    question["answer_points"][0]["sources"] = ["e1"]
                save(root / "qa-public.json", {"questions": [question]})
                save(root / "qa-audit.json", {"questions": [question]})
                save(root / "manifest.json", {"qa_source": "external"})
                save(root / "external-events.json", events)
                save(root / "stages/group-raw-candidates.json", {"questions": [question]})
                save(root / "normalized.json", [
                    {"id": "e1", "kind": "message", "role": "user", "text": "Customer context"},
                    {"id": "e3", "kind": "message", "role": "user", "text": "Dispatcher marks absence"},
                ])
                save(root / "stages/group-qa-input.json", {
                    "ref_to_source": {"资料1": "e1", "资料3": "e3"},
                    "payload": {"scope": {"dialogue": [{"id": "e1"}, {"id": "e3"}]}}})
                item, = qa_inputs(root)
                self.assertEqual(item["external_lineage"]["event_ids"], expected)

    def test_task_input_rejects_a_lineage_event_missing_from_the_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            question = dict(id="q1", type="M1", status="approved",
                            external_lineage={"event_ids": ["missing"]},
                            answer_points=[dict(text="Keep nulls", sources=["e1"])])
            save(root / "qa-public.json", {"questions": [question]})
            save(root / "qa-audit.json", {"questions": [question]})
            save(root / "manifest.json", {"qa_source": "external"})
            save(root / "external-events.json", {"version": 1, "events": [dict(id="x", source_ids=["e1"])]})
            save(root / "stages/group-raw-candidates.json", {"questions": [question]})
            save(root / "normalized.json", [{"id": "e1", "kind": "message", "role": "user", "text": "Keep nulls"}])
            save(root / "stages/group-qa-input.json", {
                "ref_to_source": {"资料1": "e1"}, "payload": {"scope": {"dialogue": [{"id": "e1"}]}}})
            diagnostics = []
            self.assertEqual(qa_inputs(root, diagnostics=diagnostics), [])
            self.assertEqual(diagnostics[0]["reason"], "unknown_external_event")

    def test_task_input_drops_workflow_from_a_disjoint_focus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            question = dict(id="q1", type="M6", status="approved",
                            answer_points=[dict(text="Keep the permit zone", sources=["e2"])])
            save(root / "qa-public.json", {"questions": [question]})
            save(root / "manifest.json", {"qa_source": "external"})
            save(root / "stages/group-raw-candidates.json", {"questions": [question]})
            save(root / "normalized.json", [
                {"id": "e2", "kind": "message", "role": "user", "text": "Keep the permit zone"},
            ])
            save(root / "stages/group-qa-input.json", {
                "ref_to_source": {"资料2": "e2", "资料10": "e10"},
                "payload": {"scope": {"dialogue": [{"id": "e2"}]},
                            "focus": {"text": "permit zone", "sources": ["资料2"]},
                            "workflow": {"text": "Close the fleet audit", "sources": ["资料10"]}},
            })
            item, = qa_inputs(root)
            self.assertNotIn("development_workflow", item)

    def test_workflow_decline_or_invalid_sources_stop_before_qa(self):
        scope = self.scope("M1")
        facts = [{"id": "f1", "statement": "导出必须保留空值。", "sources": ["e1"]}]
        for response in ("NO_QA", "WORKFLOW: 批量导出\nSOURCES: 资料99"):
            with self.subTest(response=response):
                client = Client()
                with patch.object(client, "ask", return_value=parse_text_response(response)) as ask:
                    result = generate_from_facts(scope, facts, client, qa_mode="memory", target_type="M1")
                self.assertEqual(ask.call_count, 1)
                self.assertEqual(result["questions"], [])
                self.assertEqual(result["stage_status"]["qa"], "not_submitted")

    def test_correction_references_reach_focus_and_text_reaches_workflow_and_qa(self):
        scope = self.scope("M1")
        scope["dialogue"].append({"id": "e3", "kind": "message", "role": "user", "order": 3,
                                  "text": "客户 A 的 note 例外改为省略，其他空字段继续保留。"})
        scope["cutoff"] = 3
        scope["external_source_ids"].append("e3")
        facts = [{"id": "f1", "statement": "导出必须保留空值。", "sources": ["e1"]}]
        client = Client()
        saved = {}
        result = generate_from_facts(scope, facts, client, qa_mode="memory", target_type="M1",
                                     checkpoint=lambda name, value: saved.update({name: value}))
        self.assertEqual(result["stage_errors"], [])
        self.assertEqual([u["stage"] for u in client.usage], ["workflow", "focus", "qa"])
        for name in ("workflow-input.json", "focus-input.json", "qa-input.json"):
            request = saved[name]
            self.assertIn("e3", request["ref_to_source"].values())
        for name in ("workflow-input.json", "qa-input.json"):
            request = saved[name]
            self.assertIn("客户 A 的 note 例外", str(request["payload"]["materials"]))
        focus_payload = saved["focus-input.json"]["payload"]
        self.assertNotIn("materials", focus_payload)
        self.assertNotIn("客户 A 的 note 例外", str(focus_payload))
        self.assertEqual(focus_payload["facts"], saved["qa-input.json"]["payload"]["facts"])
        self.assertEqual(saved["focus-input.json"]["payload"]["workflow"]["text"],
                         saved["qa-input.json"]["payload"]["workflow"]["text"])

    def unlinked_correction_scope(self, max_chars=32000, correction_suffix=""):
        records = [
            dict(id="e1", kind="message", role="user", order=1,
                 text="对账时 amount 差异不超过 0.01 视为未变化。"),
            dict(id="e59", kind="result", order=59, text="原对账样例通过。"),
            dict(id="e61", kind="message", role="user", order=61,
                 text="容差仅在显式传入 tolerances 时生效；默认 None 时精确比较。" + correction_suffix),
            dict(id="e62", kind="message", role="user", order=62, text="另一个任务需要新增页面主题。"),
            dict(id="e70", kind="result", order=70, text="UNRELATED_TOOL_DETAIL"),
            dict(id="e91", kind="message", role="user", order=91,
                 text="确认：未传 tolerances 就按精确比较，之前的容差只适用于显式开启的情况。"),
            dict(id="e93", kind="message", role="user", order=93, text="AFTER_CUTOFF_RULE"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.json"
            save(path, {"version": 1, "events": [dict(
                id="initial-rule", kind="compatibility_contract", memory_kind="M6",
                source_ids=["e1"], used_by=["e59"])]})
            return load_external_scopes(path, records, 92, max_chars=max_chars)["scopes"][0]

    def test_unlinked_corrections_reach_generation_repair_and_static_source_closure(self):
        scope = self.unlinked_correction_scope()
        facts = [dict(id="f1", statement="amount 差异不超过 0.01 视为未变化。", sources=["e1"])]
        question = "客户对账时，之前约定的容差规则在什么情况下适用？"

        class CorrectionClient(ExternalClient):
            def ask(self, prompt, payload):
                if "Extract only externally supplied" in prompt:
                    self.calls.append((prompt, payload))
                    self.usage.append({"status": "completed"})
                    return {"facts": [dict(id="f1", statement=facts[0]["statement"], sources=["资料1"])]}
                if "WORKFLOW:" in prompt or "FOCUS:" in prompt or "QUESTION: 自然问题" in prompt:
                    self.calls.append((prompt, payload))
                    self.usage.append({"status": "completed"})
                    if "WORKFLOW:" in prompt:
                        return {"workflow": dict(text="扩展客户对账：比较差异 → 汇总报告", sources=["资料1"])}
                    if "FOCUS:" in prompt:
                        return {"focus": dict(text="确认客户对账容差的适用条件", sources=["资料1", "资料3", "资料4"])}
                    if "review_issue" in payload:
                        return parse_text_response("QA q1\nQUESTION: " + question + "\n"
                            "ANSWER_POINT: 只有显式传入 tolerances 时才启用容差。 || SOURCES: 资料1,资料3,资料4\n"
                            "ANSWER_POINT: tolerances 默认 None 时精确比较。 || SOURCES: 资料3,资料4\nEND_QA")
                    return parse_text_response("QA q1\nQUESTION: " + question + "\n"
                        "ANSWER_POINT: 只有显式传入 tolerances 时才启用容差，默认 None 时精确比较。"
                        " || SOURCES: 资料1,资料3,资料4\nEND_QA")
                if "simple_atomicity_v1" in prompt:
                    self.calls.append((prompt, payload))
                    self.usage.append({"status": "completed"})
                    count = len(payload["candidates"][0]["answer_points"])
                    return {"reviews": [dict(id="q1", review_contract="simple_atomicity_v1",
                        point_atomicity="A1=compound" if count == 1 else "A1=single;A2=single")]}
                if "simple_relevance_v1" in prompt:
                    from tests.simple_test_helpers import relevance_response_for
                    self.calls.append((prompt, payload))
                    self.usage.append({"status": "completed"})
                    return relevance_response_for(payload)
                response = super().ask(prompt, payload)
                if "point_evidence:" in prompt:
                    response["reviews"][0].update(
                        point_evidence="A1=supported@资料1,资料3,资料4;A2=supported@资料3,资料4",
                        usage="confirmed@资料1,资料3,资料4",
                        usage_reason="用户在原约定后确认了显式启用容差的条件。")
                return response

        client = CorrectionClient()
        extracted = extract_facts(scope, client, qa_mode="memory", external_only=True)
        self.assertEqual(extracted["stage_errors"], [])
        extraction_payload = client.calls[0][1]
        self.assertEqual(len(extraction_payload["materials"]), 1)
        self.assertNotIn("默认 None", str(extraction_payload))
        self.assertNotIn("页面主题", str(extraction_payload))
        self.assertEqual(filter_external_facts([
            *facts, dict(id="other", statement="新增页面主题", sources=["e62"])
        ], [scope]), facts)

        saved = {}
        generated = generate_from_facts(scope, facts, client, qa_mode="memory", target_type="M6",
            checkpoint=lambda name, value: saved.update({name: value}))
        self.assertEqual(generated["stage_errors"], [])
        group = dict(scope=scope, facts=facts, qa_mode="memory")
        index = build_evidence_index(facts, [scope], "memory")
        self.assertEqual(static_evidence_check(group, index, "M6", generated["questions"][0])["status"], "supported")

        def resolve(candidate):
            projected, audit = external_review_projection(group, candidate)
            return (projected["scope"] if projected else None), audit

        reviewed = review_candidates(scope, facts, generated["questions"], client,
            qa_mode="memory", review_mode="simple", generation_context=generated["_repair_context"],
            review_scope_resolver=resolve, checkpoint=lambda name, value: saved.update({name: value}))
        self.assertEqual(reviewed["stage_errors"], [])
        self.assertEqual(reviewed["questions"][0]["status"], "approved")
        for name in ("workflow-input.json", "focus-input.json", "qa-input.json", "repair-input.json"):
            request = saved[name]
            self.assertTrue({"e1", "e61", "e91"} <= set(request["ref_to_source"].values()))
            self.assertNotIn("e70", request["ref_to_source"].values())
            self.assertNotIn("e93", request["ref_to_source"].values())
            self.assertEqual(len(request["payload"]["facts"]), 1)
            self.assertNotIn("页面主题", str(request["payload"]["facts"]))
        self.assertEqual(saved["qa-input.json"]["payload"]["materials"],
                         saved["repair-input.json"]["payload"]["materials"])
        self.assertNotIn("默认 None", str(saved["focus-input.json"]["payload"]))
        for name in ("workflow-input.json", "qa-input.json", "repair-input.json"):
            self.assertIn("默认 None", str(saved[name]["payload"]["materials"]))
        evidence_payload = next(payload for prompt, payload in client.calls if "point_evidence:" in prompt)
        self.assertIn("默认 None", str(evidence_payload["materials"]))

    def test_public_correction_context_is_not_dropped_to_fit_generation_budget(self):
        scope = self.unlinked_correction_scope(max_chars=3000, correction_suffix="有效条件。" * 2000)
        facts = [dict(id="f1", statement="amount 容差为 0.01。", sources=["e1"])]
        client = Client()
        result = generate_from_facts(scope, facts, client, qa_mode="memory", target_type="M6")
        self.assertEqual(client.calls, [])
        self.assertEqual(result["stage_errors"][0]["error_code"], "request_budget")
        self.assertEqual(result["stage_status"]["workflow"], "failed")

    def test_memory_authoring_indexes_optional_tools_but_keeps_public_sources(self):
        scope = self.scope("M1")
        scope["dialogue"][0]["text"] += " 客户已确认的适用条件。" * 500
        observation = "实际下游观察：保留空值的导出已被接收。"
        correction = "客户 A 的 note 例外改为省略，其他空字段继续保留。" + "有效条件。" * 900
        scope["dialogue"].extend([
            dict(id="e3", kind="call", order=3, text="WRITE_TEST_DETAIL\n" * 5000),
            dict(id="e4", kind="result", order=4, text="保留空值。\n" * 10000),
            dict(id="e5", kind="result", order=5, text=observation),
            dict(id="e6", kind="message", role="user", order=6, text=correction),
        ])
        scope["cutoff"] = 6
        scope["external_source_ids"].append("e5")
        scope["external_usage_ids"].extend(["e3", "e4"])
        facts = [dict(id="f1", statement="导出必须保留空值。", sources=["e1"]),
                 dict(id="f2", statement=observation, sources=["e5"])]
        client, saved = Client(), {}
        result = generate_from_facts(scope, facts, client, qa_mode="memory", target_type="M1",
            checkpoint=lambda name, value: saved.update({name: value}))
        self.assertEqual(result["stage_errors"], [])
        for name in ("workflow-input.json", "qa-input.json"):
            request = saved[name]
            by_source = {request["ref_to_source"][item["reference"]]: item
                         for item in request["payload"]["materials"]}
            for source in ("e1", "e5", "e6"):
                text = next(row["text"] for row in scope["dialogue"] if row["id"] == source)
                self.assertEqual(by_source[source]["original_records"][0]["text"], text)
            for source in ("e3", "e4"):
                self.assertNotIn("original_records", by_source[source])
                self.assertIn("未提供原文", by_source[source]["evidence_note"])
        self.assertEqual(saved["workflow-input.json"]["ref_to_source"],
                         saved["qa-input.json"]["ref_to_source"])
        self.assertEqual(saved["focus-input.json"]["ref_to_source"],
                         saved["qa-input.json"]["ref_to_source"])
        self.assertNotIn("WRITE_TEST_DETAIL", str(result["_repair_context"]))
        from dialogue_benchmark.llm import _target_review_request
        _, target_payload = _target_review_request(scope, set(saved["qa-input.json"]["ref_to_source"].values()),
            facts, result["questions"][0], "memory")
        self.assertNotIn("WRITE_TEST_DETAIL", str(target_payload))
        self.assertNotIn(correction, str(target_payload))
        self.assertIn("focus", target_payload)

        # Usage indexing is an authoring projection, not a source deletion.
        scope["model_request_chars"] = 200000
        projected, audit = external_review_projection(dict(scope=scope, facts=facts), result["questions"][0])
        self.assertTrue(audit["complete"])
        self.assertEqual(projected["scope"]["dialogue"], scope["dialogue"])
        from dialogue_benchmark.llm import _evidence_review_request
        _, payload, refs = _evidence_review_request(projected["scope"],
            projected["scope"]["review_guard_sources"], facts, result["questions"][0])
        reviewed_tools = next(item for item in payload["materials"] if refs[item["reference"]] == "e4")
        self.assertIn("original_records", reviewed_tools)
        self.assertNotIn("evidence_note", reviewed_tools)
        self.assertIn(correction, str(payload))

    def external_correction_group(self):
        scope = self.unlinked_correction_scope()
        facts = [dict(id="f1", statement="amount 差异不超过 0.01 视为未变化。", sources=["e1"])]
        group = dict(id="g1", scope=scope, facts=facts, qa_mode="memory", allowed_types=("M6",))
        return group, build_evidence_index(facts, [scope], "memory")

    def test_correction_only_citations_reach_semantic_review_through_cli(self):
        group, index = self.external_correction_group()
        candidate = dict(id="q1", type="M6", qa_mode="memory", forbidden_points=[],
            question="客户对账时，之前约定的容差何时才适用？",
            answer_points=[dict(text="只有显式传入 tolerances 时才启用容差。", sources=["e61", "e91"])])
        check = static_evidence_check(group, index, "M6", candidate)
        self.assertEqual(check["status"], "unknown")
        self.assertEqual(check["reason"], "external_context_requires_review")
        self.assertEqual(check["fact_ids"], [])

        class CorrectionReviewClient(ExternalClient):
            def ask(self, prompt, payload):
                response = super().ask(prompt, payload)
                if "point_evidence:" in prompt:
                    response["reviews"][0].update(
                        point_evidence="A1=supported@资料3,资料4",
                        usage="confirmed@资料1,资料3,资料4",
                        usage_reason="用户在原约定后确认了显式启用容差的条件。")
                return response

        client = CorrectionReviewClient()
        generated = dict(questions=[candidate], all_candidates=[candidate],
            stage_status={"qa": "completed"}, stage_errors=[], rejected=[], raw_generated=1)
        with patch.object(cli, "ChatClient", return_value=client), \
                patch.object(cli, "generate_from_facts", return_value=generated):
            result = cli._run_qa_tasks([(0, "memory", group)], "https://example.invalid",
                "offline-test", "KEY", 1, review_mode="simple", evidence_indexes={"memory": index})
        self.assertEqual(result["stage_errors"], [])
        self.assertEqual(result["rejected"], [])
        self.assertEqual(result["questions"][0]["status"], "approved")
        self.assertEqual(result["questions"][0]["static_evidence_status"], "unknown")
        self.assertEqual([receipt["stage"] for receipt in client.usage], [
            "review_relevance", "review_atomicity", "review_completeness", "review_evidence"])

    def test_answer_from_an_earlier_undeclared_message_is_rejected_before_review(self):
        group, index = self.external_correction_group()
        # Anchor the event on the later confirmation; e1 is now an earlier,
        # undeclared message of another rule.
        group = dict(group, scope=dict(group["scope"], external_source_ids=["e91"],
                                       external_usage_ids=[], external_declared_context_ids=[]))
        candidate = dict(id="q1", type="M6", qa_mode="memory", forbidden_points=[],
            question="客户对账时，之前约定的容差是多少？",
            answer_points=[dict(text="amount 差异不超过 0.01 视为未变化。", sources=["e1"])])
        client = ExternalClient()
        generated = dict(questions=[candidate], all_candidates=[candidate],
            stage_status={"qa": "completed"}, stage_errors=[], rejected=[], raw_generated=1)
        with patch.object(cli, "ChatClient", return_value=client), \
                patch.object(cli, "generate_from_facts", return_value=generated):
            result = cli._run_qa_tasks([(0, "memory", group)], "https://example.invalid",
                "offline-test", "KEY", 1, review_mode="simple", evidence_indexes={"memory": index})
        self.assertEqual(result["questions"], [])
        self.assertEqual([item["reason"] for item in result["rejected"]], ["external_anchor_not_cited"])
        self.assertEqual(client.usage, [])

    def test_review_repair_that_drops_the_event_anchor_is_rejected(self):
        group, index = self.external_correction_group()
        group = dict(group, scope=dict(group["scope"], external_source_ids=["e91"],
                                       external_usage_ids=[], external_declared_context_ids=[]))
        candidate = dict(id="q1", type="M6", qa_mode="memory", forbidden_points=[],
            question="客户对账时，未传 tolerances 时如何比较？",
            answer_points=[dict(text="未传 tolerances 就按精确比较。", sources=["e91"])])
        repaired = dict(candidate, status="approved",
            answer_points=[dict(text="amount 差异不超过 0.01 视为未变化。", sources=["e1"])])
        generated = dict(questions=[candidate], all_candidates=[candidate],
            stage_status={"qa": "completed"}, stage_errors=[], rejected=[], raw_generated=1)
        reviewed = dict(questions=[repaired], rejected=[], stage_errors=[], review_warnings=[],
                        stage_status={"review": "completed"}, revisions=[])
        with patch.object(cli, "ChatClient", return_value=ExternalClient()), \
                patch.object(cli, "generate_from_facts", return_value=generated), \
                patch.object(cli, "review_candidates", return_value=reviewed):
            result = cli._run_qa_tasks([(0, "memory", group)], "https://example.invalid",
                "offline-test", "KEY", 1, review_mode="simple", evidence_indexes={"memory": index})
        self.assertEqual(result["questions"], [])
        self.assertEqual([item["reason"] for item in result["rejected"]], ["external_anchor_not_cited"])

    def test_external_anchor_gate_accepts_declared_context_and_later_corrections(self):
        from dialogue_benchmark.external import external_anchor_reason
        scope = dict(external_event_id="x", external_source_ids=["e5"], external_usage_ids=["e7"],
                     external_declared_context_ids=["e2"], dialogue=[
                         dict(id="e1", kind="message", role="user", order=1),
                         dict(id="e2", kind="message", role="user", order=2),
                         dict(id="e5", kind="message", role="user", order=5),
                         dict(id="e7", kind="result", order=7),
                         dict(id="e8", kind="message", role="user", order=8),
                         dict(id="e9", kind="message", role="assistant", order=9)])
        def reason(*sources):
            return external_anchor_reason(
                {"answer_points": [{"text": "x", "sources": list(sources)}]}, scope)
        for allowed in ("e5#fragment-1", "e7", "e2", "e8"):
            self.assertIsNone(reason("e1", allowed), allowed)
        self.assertEqual(reason("e1"), "external_anchor_not_cited")
        self.assertEqual(reason("e9"), "external_anchor_not_cited")
        self.assertIsNone(external_anchor_reason(
            {"answer_points": [{"text": "x", "sources": ["e1"]}]}, dict(scope, external_event_id=None)))

    def test_external_context_outside_the_scope_is_still_rejected(self):
        group, index = self.external_correction_group()
        for source in ("e70", "e93", "unprovided"):
            with self.subTest(source=source):
                candidate = dict(answer_points=[dict(text="A rule.", sources=[source])])
                check = static_evidence_check(group, index, "M6", candidate)
                self.assertEqual(check["status"], "insufficient")
                self.assertEqual(check["reason"], "answer_source_out_of_scope")

    def test_unrelated_context_is_not_static_external_evidence(self):
        group, index = self.external_correction_group()
        candidate = dict(id="q1", type="M6", qa_mode="memory", forbidden_points=[],
            question="客户的新页面需要哪种主题？",
            answer_points=[dict(text="新增页面主题。", sources=["e62"])])
        check = static_evidence_check(group, index, "M6", candidate)
        self.assertEqual(check["status"], "insufficient")
        self.assertEqual(check["reason"], "answer_source_out_of_scope")
        self.assertEqual(check["fact_ids"], [])
        client = Client(alignment="drifted")
        reviewed = review_candidates(group["scope"], group["facts"], [candidate], client,
            qa_mode="memory", review_mode="simple", allow_repair=False)
        self.assertTrue(reviewed["questions"])
        self.assertEqual(reviewed["questions"][0]["status"], "needs_review")
        self.assertTrue(reviewed["questions"][0]["target_review_conflict"])
        self.assertGreater(len(client.usage), 1)

    def scope(self, kind):
        return {"cutoff": 2, "track": "memory", "memory_kind": kind,
                "external_event_id": "x1", "external_kind": "external_observation",
                "external_source_ids": ["e1"], "external_usage_ids": ["e2"],
                "generation_extra_sources": ["e2"],
                "review_guard_sources": ["e1", "e2"], "review_guard_complete": True,
                "dialogue": [
                    {"id": "e1", "kind": "message", "role": "user", "order": 1,
                     "text": "扩展导出时必须保留空值。"},
                    {"id": "e2", "kind": "message", "role": "assistant", "order": 2,
                     "text": "下游验证确认保留了空值。"}],
                "events": [], "versions": [], "model_request_chars": 32000,
                "evidence_group": {"id": "external-x1"}}

    def order_choice_scope(self):
        scope = self.scope("M2")
        scope["dialogue"][0]["text"] = (
            "load_orders 的 active_only 默认 False；append_disposition 校验 action 和 resolver。\n"
            "Northwind 2025-04-18-A 班次实际选择：\n"
            "append_disposition(orders, 'ORD-NW-101', action='hold', resolver='op-4', resolved=False)，托盘数未确认；\n"
            "append_disposition(orders, 'ORD-NW-102', action='release', resolver='op-4', resolved=True)，托盘数已确认。")
        scope["dialogue"][1]["text"] = "本班次处置已记录：ORD-NW-101 hold/unresolved；ORD-NW-102 release/resolved。"
        scope["dialogue"].append(dict(id="e3", kind="message", role="user", order=3,
            text="仓库于 2025-04-18T16:10:00Z 确认 ORD-NW-101 托盘数；Northwind 2025-04-18-A 班次改由 "
                 "op-5 选择 release/resolved，授权 W-NW-101。render_summary 默认 show_latest=False，旧调用保持兼容。"))
        scope["cutoff"] = 3
        scope["generation_extra_sources"].append("e3")
        scope["review_guard_sources"].append("e3")
        facts = [dict(id="f1", sources=["e1"],
                      statement="Northwind 2025-04-18-A 班次 ORD-NW-101 已选 hold/unresolved，由 op-4 处理，托盘数未确认。"),
                 dict(id="f2", sources=["e1"],
                      statement="Northwind 2025-04-18-A 班次 ORD-NW-102 已选 release/resolved，由 op-4 处理，托盘数已确认。")]
        return scope, facts

    def test_six_static_types_reach_generation_review_and_requirement_direction(self):
        self.assertEqual(MEMORY_TYPES, set(MEMORY_TASK_GUIDANCE))
        for kind in sorted(MEMORY_TYPES):
            with self.subTest(kind=kind):
                scope = self.scope(kind)
                facts = [{"id": "f1", "statement": "导出必须保留空值。", "sources": ["e1"]}]
                client = ExternalClient()
                result = generate_from_facts(scope, facts, client, qa_mode="memory", target_type=kind)
                question = result["questions"][0]
                group = {"scope": scope, "facts": facts, "qa_mode": "memory"}
                with patch("dialogue_benchmark.fact_index._build_direct_evidence_graph",
                           side_effect=AssertionError("No graph for external events")):
                    index = build_evidence_index(facts, [scope], "memory")
                    question.update(static_candidate_labels(group, question, index, kind))
                reviewed = review_candidates(scope, facts, [question], client,
                                             qa_mode="memory", review_mode="simple", allow_repair=False)
                self.assertEqual(reviewed["questions"][0]["status"], "approved")
                self.assertEqual(question["type"], "unknown")
                self.assertEqual(question["type_origin"], "static_public_evidence")
                self.assertEqual(cli._public_question(question)["difficulty_origin"], "static_evidence_complexity")
                self.assertNotIn("category", question)
                for prompt, payload in client.calls[1:3]:
                    self.assertIn(MEMORY_TYPE_GUIDANCE[kind], prompt)
                    self.assertIn("future business workflow", prompt)
                    self.assertNotIn(kind, prompt)
                qa_prompt = next(prompt for prompt, payload in client.calls
                                 if 'QUESTION: 自然问题' in prompt)
                self.assertIn('不写成开发需求', qa_prompt)
                self.assertIn('不追加样例输出或计数计算题', qa_prompt)
                self.assertIn(MEMORY_TASK_GUIDANCE[kind], task_direction(kind))

    def test_trivia_target_is_rejected_before_more_review_calls(self):
        client = Client(alignment="drifted")
        candidate = {"id": "q1", "qa_mode": "memory", "type": "M2",
                     "question": "这次运行一共输出多少字节？",
                     "answer_points": [{"text": "99 字节。", "sources": ["e1"]}],
                     "forbidden_points": []}
        result = review_candidates(self.scope("M2"), [], [candidate], client,
                                   qa_mode="memory", review_mode="simple", allow_repair=False)
        self.assertTrue(result["questions"])
        self.assertTrue(result["questions"][0]["target_review_conflict"])
        self.assertGreater(len(client.calls), 1)
        self.assertIn("byte total from one run", client.calls[0][0])
        self.assertIn("external size limit can be useful", client.calls[0][0])

    def test_external_target_repair_keeps_facts_and_correction_context(self):
        scope = self.scope("M1")
        scope["dialogue"].append({"id": "e3", "kind": "message", "role": "user", "order": 3,
                                  "text": "导出失败时可以重试。"})
        facts = [{"id": "f1", "statement": "导出必须保留空值。", "sources": ["e1"]}]
        client = ExternalClient()
        generated = generate_from_facts(scope, facts, client, qa_mode="memory", target_type="M1")
        context = generated["_repair_context"]
        client.alignment = "drifted"
        result = review_candidates(scope, facts, generated["questions"], client,
                                   qa_mode="memory", review_mode="simple", generation_context=context)
        self.assertEqual(result["questions"][0]["status"], "approved")
        self.assertNotIn("revisions", result)

    def test_customer_named_api_focus_can_be_rejected_without_replacement_qa(self):
        class InvalidFocusClient(OrderChoiceClient):
            def ask(self, prompt, payload):
                if "review_issue" in payload:
                    self.calls.append((prompt, payload))
                    self.usage.append({"status": "completed"})
                    return {"questions": []}
                return super().ask(prompt, payload)

        for allow_repair in (False, True):
            with self.subTest(allow_repair=allow_repair):
                scope, facts = self.order_choice_scope()
                client = InvalidFocusClient()
                generated = generate_from_facts(scope, facts, client, qa_mode="memory", target_type="M2")
                api_focus = "找回 Northwind 班次的 active_only 过滤、append_disposition 校验及 render_summary 接口契约"
                context = generated["_repair_context"]
                context["payload"]["focus"]["text"] = api_focus
                candidate = dict(generated["questions"][0],
                    question="Northwind 班次的 load_orders、append_disposition 和 render_summary 有哪些接口契约？",
                    answer_points=[dict(text="load_orders 的 active_only 默认 False。", sources=["e1"])],
                    _generation_focus=dict(text=api_focus, sources=["e1"]))
                client.alignment = "drifted"
                reviewed = review_candidates(scope, facts, [candidate], client, qa_mode="memory",
                    review_mode="simple", allow_repair=allow_repair, generation_context=context)
                self.assertEqual(reviewed["stage_errors"], [])
                target_prompt, target_payload = client.calls[3]
                self.assertIn("Choose drifted for a surrounding API contract", target_prompt)
                self.assertIn("even if it names the same customer", target_prompt)
                self.assertIn("allowed external candidate set", target_prompt)
                self.assertIn("focus fixes the selected object, scope, and one decision within facts", target_prompt)
                self.assertEqual(target_payload["focus"]["text"], api_focus)
                self.assertNotIn("materials", target_payload)
                if not allow_repair:
                    self.assertTrue(reviewed["questions"])
                    self.assertTrue(reviewed["questions"][0]["target_review_conflict"])
                    self.assertGreater(len(client.usage[3:]), 1)
                    continue
                self.assertTrue(reviewed["questions"])
                self.assertTrue(reviewed["questions"][0]["target_review_conflict"])
                self.assertNotIn("revisions", reviewed)

    def test_cli_extracts_once_publishes_one_pool_and_joins_original_task_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save(root / "dialogue.json", {"version": 1, "records": self.scope("M1")["dialogue"]})
            event = {"id": "x1", "kind": "compatibility_contract", "memory_kind": "M1",
                     "source_ids": ["e1"], "used_by": ["e2"], "qa_mode": "both"}
            save(root / "events.json", {"version": 1, "events": [event, event]})
            output = root / "run"
            with patch.object(cli, "ChatClient", ExternalClient), \
                 patch.object(cli, "build_graph", side_effect=AssertionError("graph built")), \
                 patch("dialogue_benchmark.repository_probe.probe_candidate", return_value={"status": "not_recoverable"}):
                status = cli.main([str(root / "dialogue.json"), "--output", str(output),
                    "--qa-source", "external", "--external-events", str(root / "events.json"),
                    "--qa-count", "1", "--group-budget", "10", "--parallel-workers", "1",
                    "--repository", str(root), "--allow-network", "--endpoint", "https://example.invalid",
                    "--model", "offline-test"])
            self.assertEqual(status, 0)
            manifest = read(output / "manifest.json")
            self.assertEqual(manifest["external_event_count"], 1)
            self.assertEqual(manifest["chunks"]["unique"], 1)
            self.assertEqual(manifest["progress"]["targets"], {"memory": 1})
            self.assertEqual(manifest["progress"]["stop_reasons"], {"memory": "target_reached"})
            self.assertEqual(set(manifest["coverage"]), {"memory"})
            self.assertEqual(manifest["questions"]["type"], {"unknown": 1})
            self.assertNotIn("general_count", manifest)
            self.assertFalse((output / "general-qa.json").exists())
            self.assertFalse((output / "code-qa.json").exists())
            items = qa_inputs(output)
            self.assertEqual(len(items), 1)
            # The published type is inferred from the answer evidence.  If
            # the evidence does not justify a memory class, retain the item
            # as unknown instead of copying the event's controller label.
            self.assertEqual(items[0]["qa"]["type"], "unknown")
            self.assertEqual(items[0]["original_candidate"]["type"], "M1")
            self.assertTrue(Path(items[0]["generation_input"]).is_file())
            self.assertEqual(items[0]["development_workflow"], "增加批量导出：读取记录 → 导出 → 汇总结果")
            view = build(output)
            self.assertEqual(view["targets"], {"memory": 1})
            self.assertEqual(view["questions"][0]["type"], "unknown")
            self.assertEqual(view["meta"]["expansion_mode"], "external_events")

    def test_empty_external_pool_stays_reviewable_without_model_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save(root / "dialogue.json", {"version": 1, "records": self.scope("M2")["dialogue"]})
            save(root / "events.json", {"version": 1, "events": []})
            with patch.object(cli, "ChatClient", side_effect=AssertionError("No eligible event")):
                status = cli.main([str(root / "dialogue.json"), "--output", str(root / "run"),
                    "--qa-source", "external", "--external-events", str(root / "events.json"),
                    "--qa-count", "2", "--allow-network", "--endpoint", "https://example.invalid",
                    "--model", "offline-test"])
            self.assertEqual(status, 0)
            view = build(root / "run")
            self.assertEqual(view["targets"], {"memory": 2})
            self.assertEqual(view["questions"], [])
            self.assertEqual(view["groups"], [])

    def test_external_count_options_do_not_silently_accept_graph_quotas(self):
        parser = cli._build_parser()
        with tempfile.TemporaryDirectory() as directory:
            events = Path(directory) / "events.json"
            save(events, {"version": 1, "events": []})
            base = ["input", "--output", "out", "--qa-source", "external", "--external-events", str(events)]
            options = cli._parse_options(parser.parse_args(base + ["--qa-count", "20"]), parser)
            self.assertEqual(options["memory_group_budget"], 80)
            for flags in (["--qa-mode", "both"], ["--code-count", "3"], ["--general-count", "3"]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    cli._parse_options(parser.parse_args(base + flags), parser)
