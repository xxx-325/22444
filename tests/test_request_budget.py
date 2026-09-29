"""Offline regressions for request budgets outside model-visible payloads."""

import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dialogue_benchmark import llm
from dialogue_benchmark.task_eval.artifacts import read
from dialogue_benchmark.task_eval.runtime import ask_model, write_tests


class RequestBudgetTests(unittest.TestCase):
    def setUp(self):
        key = patch.dict("os.environ", {"BENCHMARK_API_KEY": "test-only"})
        key.start()
        self.addCleanup(key.stop)
        transport = patch("dialogue_benchmark.llm.urllib.request.build_opener")
        self.opener = transport.start().return_value
        self.addCleanup(transport.stop)
        self.client = llm.ChatClient("https://example.invalid", "test")
        self.scope = {
            "dialogue": [{"id": "e1", "kind": "message", "role": "user", "order": 1,
                          "text": "导出必须保留空值。" + "背景。" * 23000}],
            "events": [], "versions": [],
            "model_request_chars": 96000, "max_context_chars": 32000,
        }
        self.facts = [{"id": "f1", "statement": "导出必须保留空值。", "sources": ["e1"]}]
        self.payload, _ = llm.simple_evidence_payload(self.scope, {"e1"}, self.facts)

    def responses(self, *contents):
        self.opener.open.side_effect = [io.BytesIO(json.dumps({
            "choices": [{"finish_reason": "stop", "message": {"content": content}}],
        }).encode()) for content in contents]

    def assert_private_budget(self):
        for call in self.opener.open.call_args_list:
            body = json.loads(call.args[0].data)
            text = body["messages"][1]["content"]
            for field in ("model_request_chars", "max_context_chars", "request_budget"):
                self.assertNotIn(field, text)

    def test_default_transport_limit_remains_60000(self):
        size = llm.request_size("test", self.payload)
        self.assertGreater(size, 60000)
        self.assertLess(size, 96000)
        with self.assertRaises(llm.ModelStageError) as caught:
            self.client.ask("test", self.payload)
        self.assertEqual(caught.exception.code, "request_budget")
        self.assertEqual(caught.exception.details, {"request_chars": size, "limit_chars": 60000})
        self.opener.open.assert_not_called()
        self.assertEqual(self.client.usage, [])

    def test_explicit_transport_limit_is_enforced_without_changing_content(self):
        self.responses("NO_QA")
        self.assertEqual(self.client.ask("test", self.payload, request_budget=96000),
                         {"questions": []})
        sent = json.loads(self.opener.open.call_args.args[0].data)
        self.assertEqual(sent["messages"][1]["content"],
                         "test\nDATA:\n" + json.dumps(self.payload, ensure_ascii=False,
                                                       separators=(",", ":")))
        self.assert_private_budget()
        oversized = dict(self.payload, context="背景。" * 10000)
        with self.assertRaises(llm.ModelStageError) as caught:
            self.client.ask("test", oversized, request_budget=96000)
        self.assertEqual(caught.exception.code, "request_budget")
        self.assertEqual(caught.exception.details["limit_chars"], 96000)
        self.assertGreater(caught.exception.details["request_chars"], 96000)
        self.opener.open.assert_called_once()
        self.assertEqual(len(self.client.usage), 1)

    def test_task_host_calls_use_the_same_serialized_request_limit(self):
        config = {"judge": {"base_url": "https://example.invalid", "model": "test",
                            "key_env": "BENCHMARK_API_KEY"}}
        size = llm.request_size("test", self.payload)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for limit in (None, size - 1):
                settings = dict(config, **({"model_request_chars": limit} if limit else {}))
                with self.subTest(limit=limit), self.assertRaises(llm.ModelStageError) as caught:
                    ask_model("test", self.payload, settings, root / str(limit))
                self.assertEqual(caught.exception.details,
                                 {"request_chars": size, "limit_chars": limit or 60000})
            self.opener.open.assert_not_called()
            self.responses("NO_QA")
            self.assertEqual(ask_model("test", self.payload, dict(config, model_request_chars=size),
                                       root / "allowed"), {"questions": []})
            self.assertEqual(read(root / "allowed/usage.json")[0]["request_chars"], size)
            self.assertEqual(read(root / "allowed/input.json")["payload"], self.payload)
        self.assert_private_budget()
        sent = json.loads(self.opener.open.call_args.args[0].data)
        self.assertNotIn("max_tokens", sent)
        self.assertNotIn("max_output_tokens", sent)

    def test_task_writer_keeps_full_60584_byte_snapshot_under_configured_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, spec = root / "baseline", root / "spec"
            (baseline / "tests").mkdir(parents=True)
            spec.mkdir()
            source = "VALUE = 1\n"
            regression = "def test_existing(): assert True\n"
            regression += "#" + "x" * (60584 - len(source) - len(regression) - 2) + "\n"
            (baseline / "entry.py").write_text(source)
            (baseline / "tests/test_existing.py").write_text(regression)
            (spec / "task.md").write_text("Add the requested feature")
            (spec / "acceptance.md").write_text("Preserve existing behavior")
            config = {"judge": {"base_url": "https://example.invalid", "model": "test",
                                "key_env": "BENCHMARK_API_KEY"}}
            budget = SimpleNamespace(call=ask_model)
            self.assertEqual(sum(path.stat().st_size for path in baseline.rglob("*.py")), 60584)
            self.assertIsNone(write_tests(spec, baseline, config, root / "default", budget))
            self.opener.open.assert_not_called()

            config["model_request_chars"] = 96000
            self.responses("FILE test_acceptance.py\ndef test_feature(): assert True\nEND_FILE\n"
                           "FILE acceptance.md\nExecutable checks\nEND_FILE")
            result = write_tests(spec, baseline, config, root / "author", budget)
            self.assertEqual(result["status"], "finished")
            self.opener.open.assert_called_once()
            payload = read(root / "author/input.json")["payload"]
            self.assertEqual(payload["repository"],
                             {"entry.py": source, "tests/test_existing.py": regression})
            size = read(root / "author/usage.json")[0]["request_chars"]
            self.assertGreater(size, 60000)
            self.assertLess(size, 96000)
            self.assertEqual((spec / "regression/tests/test_existing.py").read_text(), regression)
            self.assertEqual((baseline / "tests/test_existing.py").read_text(), regression)
            self.assertIn("/workspace/candidate/tests", (spec / "commands/existing_suite.sh").read_text())

            # The snapshot fits, but requirements and prompt still count toward the limit.
            (spec / "task.md").write_text("Additional requirements " + "x" * 36000)
            self.assertIsNone(write_tests(spec, baseline, config, root / "oversized", budget))
            self.opener.open.assert_called_once()
        self.assert_private_budget()

    def test_unconfigured_workflow_still_fails_before_transport(self):
        scope = {key: value for key, value in self.scope.items()
                 if key not in {"model_request_chars", "max_context_chars"}}
        result = llm.generate_from_facts(scope, self.facts, self.client,
                                         qa_mode="memory", target_type="M6")
        error, = result["stage_errors"]
        self.assertEqual(error["stage"], "workflow")
        self.assertEqual(error["error_code"], "request_budget")
        self.assertEqual(error["limit_chars"], 60000)
        self.opener.open.assert_not_called()

    def test_scope_budget_reaches_extraction_generation_and_review(self):
        self.responses(
            "FACT f1\nSOURCES: 资料1\nSOURCE_KIND: conversation\nTEXT: 导出必须保留空值。\nEND_FACT",
            "WORKFLOW: 扩展导出：读取记录 → 导出 → 汇总结果\nSOURCES: 资料1",
            "FOCUS: 确认扩展导出时仍适用的空值限制\nSOURCES: 资料1",
            "QA q1\nQUESTION: 扩展导出时要继续遵守哪项限制？\n"
            "ANSWER_POINT: 导出必须保留空值。 || SOURCES: 资料1\nEND_QA",
            "REVIEW q1\nreview_contract: target_v1\ntarget_alignment: aligned\nEND_REVIEW",
            "REVIEW q1\nreview_contract: simple_relevance_v1\npoint_relevance: A1=direct\nEND_REVIEW",
            "REVIEW q1\nreview_contract: simple_atomicity_v1\npoint_atomicity: A1=single\nEND_REVIEW",
            "REVIEW q1\nreview_contract: simple_v1\ncompleteness: complete\nEND_REVIEW",
            "REVIEW q1\nreview_contract: simple_v1\npoint_evidence: A1=supported@资料1\nEND_REVIEW",
        )
        extracted = llm.extract_facts(self.scope, self.client, qa_mode="memory", external_only=True)
        self.assertEqual(extracted["stage_errors"], [])
        self.assertEqual(len(extracted["facts"]), 1)
        generated = llm.generate_from_facts(self.scope, extracted["facts"], self.client,
                                            qa_mode="memory", target_type="M6")
        self.assertEqual(generated["stage_errors"], [])
        self.assertEqual(len(generated["questions"]), 1)
        reviewed = llm.review_candidates(self.scope, extracted["facts"], generated["questions"],
                                         self.client, qa_mode="memory", review_mode="simple",
                                         allow_repair=False)
        self.assertEqual(reviewed["stage_errors"], [])
        self.assertEqual(reviewed["questions"][0]["status"], "approved")
        stages = [item["stage"] for item in self.client.usage]
        self.assertEqual(stages, ["facts", "workflow", "focus", "qa", "review_target",
                                  "review_relevance", "review_atomicity",
                                  "review_completeness", "review_evidence"])
        for receipt in self.client.usage:
            self.assertEqual(receipt["status"], "completed")
            self.assertLessEqual(receipt["request_chars"], 96000)
            if receipt["stage"] in {"facts", "workflow", "qa", "review_evidence"}:
                self.assertGreater(receipt["request_chars"], 60000)
        self.assert_private_budget()


if __name__ == "__main__":
    unittest.main()
