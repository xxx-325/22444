"""Offline checks of the unified external QA-to-requirement contract."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dialogue_benchmark import cli
from dialogue_benchmark.fact_index import build_evidence_index, static_candidate_labels
from dialogue_benchmark.llm import generate_from_facts, review_candidates
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
            response["reviews"][0].update(usage="applied@资料2", usage_reason="下游验证确认保留了空值。")
        return response


class ExternalMemoryPipelineTests(unittest.TestCase):
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
                for prompt, payload in client.calls[:3]:
                    self.assertIn(MEMORY_TYPE_GUIDANCE[kind], prompt)
                    self.assertIn("future implementation choice", prompt)
                    self.assertNotIn(kind, prompt)
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
