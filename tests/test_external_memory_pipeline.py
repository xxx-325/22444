"""Offline checks of the unified external QA-to-requirement contract."""

import contextlib
import io
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dialogue_benchmark import cli
from dialogue_benchmark.fact_index import build_evidence_index, static_candidate_labels
from dialogue_benchmark.llm import generate_from_facts, review_candidates, parse_text_response
from dialogue_benchmark.protocol import MEMORY_TYPES, MEMORY_TYPE_GUIDANCE, MEMORY_TASK_GUIDANCE
from dialogue_benchmark.task_eval.artifacts import qa_inputs, save, read
from dialogue_benchmark.task_eval.prompts import task_direction
from tests.test_memory_types import Client
from viewer.build_data import build


class ExternalClient(Client):
    def __init__(self, *args):
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


class ExternalMemoryPipelineTests(unittest.TestCase):
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
        route = {"stages": [dict(commit=t["reference"], requirement=t["body"],
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
            for name in ("workflow-input.json", "focus-input.json", "qa-input.json"):
                self.assertIn(new["text"], str(saved[name]["payload"]))
                self.assertNotIn("UNRELEASED-POLICY", str(saved[name]))
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
            save(root / "stages/group-qa-input.json", {"payload": {"workflow": {"text": "Batch delivery"}}})
            item, = qa_inputs(root)
        self.assertEqual(item["original_candidate"]["answer_points"][0]["sources"], ["e1"])
        self.assertEqual(item["reviewed_candidate"]["answer_points"][0]["sources"], ["e3"])

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

    def test_corrections_reach_workflow_focus_and_qa_even_with_stale_fact(self):
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
            self.assertIn("客户 A 的 note 例外", str(request["payload"]["materials"]))
        self.assertEqual(saved["focus-input.json"]["payload"]["workflow"]["text"],
                         saved["qa-input.json"]["payload"]["workflow"]["text"])

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
                self.assertEqual(question["type"], kind)
                self.assertEqual(question["type_origin"], "external_event")
                self.assertNotIn("difficulty", cli._public_question(question))
                self.assertNotIn("category", question)
                for prompt, payload in client.calls[1:4]:
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
        self.assertEqual(result["questions"], [])
        self.assertEqual(result["rejected"][0]["reason"], "answer_target_mismatch")
        self.assertEqual(len(client.calls), 1)
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
        self.assertEqual(len(result["revisions"]), 1)
        repair, = [payload for _, payload in client.calls if "review_issue" in payload]
        self.assertEqual(repair["facts"], context["payload"]["facts"])
        self.assertEqual(repair["materials"], context["payload"]["materials"])
        self.assertIn("external rules selected in facts", repair["review_issue"])
        self.assertIn("even if focus included them", repair["review_issue"])

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
            self.assertEqual(manifest["questions"]["type"], {"M1": 1})
            self.assertNotIn("general_count", manifest)
            self.assertFalse((output / "general-qa.json").exists())
            self.assertFalse((output / "code-qa.json").exists())
            items = qa_inputs(output)
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["qa"]["type"], "M1")
            self.assertEqual(items[0]["original_candidate"]["type"], "M1")
            self.assertTrue(Path(items[0]["generation_input"]).is_file())
            self.assertEqual(items[0]["development_workflow"], "增加批量导出：读取记录 → 导出 → 汇总结果")
            view = build(output)
            self.assertEqual(view["targets"], {"memory": 1})
            self.assertEqual(view["questions"][0]["type"], "M1")
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
