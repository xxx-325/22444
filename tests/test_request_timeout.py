import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dialogue_benchmark import cli, llm
from dialogue_benchmark.repository_probe import probe_candidate
from dialogue_benchmark.task_eval.runtime import ask_model, configure


class RequestTimeoutTests(unittest.TestCase):
    def test_transport_uses_configured_seconds_without_output_cap_or_retry(self):
        for kwargs, expected in (({}, 90), ({"timeout": 1800}, 1800), ({"timeout": 0.5}, 0.5)):
            with self.subTest(expected=expected), \
                    patch.dict("os.environ", {"BENCHMARK_API_KEY": "test-only"}), \
                    patch("dialogue_benchmark.llm.urllib.request.build_opener") as opener:
                opener.return_value.open.return_value = io.BytesIO(json.dumps({
                    "choices": [{"finish_reason": "stop", "message": {"content": "NO_QA"}}]
                }).encode())
                llm.ChatClient("https://example.invalid", "test", **kwargs).ask("test", {})
                opener.return_value.open.assert_called_once()
                self.assertEqual(opener.return_value.open.call_args.kwargs, {"timeout": expected})
                sent = json.loads(opener.return_value.open.call_args.args[0].data)
                self.assertNotIn("max_tokens", sent)
                self.assertNotIn("max_output_tokens", sent)

    def test_invalid_transport_timeouts_fail_before_network_access(self):
        for value in (0, -1, float("nan"), float("inf"), -float("inf"), None, True, "1800"):
            with self.subTest(value=value), \
                    patch("dialogue_benchmark.llm.urllib.request.build_opener") as opener:
                with self.assertRaisesRegex(ValueError, "positive finite number"):
                    llm.ChatClient("https://example.invalid", "test", timeout=value)
                opener.assert_not_called()

    def test_reasoning_effort_reaches_provider_without_an_output_cap(self):
        for effort in (None, "low", "high", "max"):
            with self.subTest(effort=effort), \
                    patch.dict("os.environ", {"BENCHMARK_API_KEY": "test-only"}), \
                    patch("dialogue_benchmark.llm.urllib.request.build_opener") as opener:
                opener.return_value.open.return_value = io.BytesIO(json.dumps({
                    "choices": [{"finish_reason": "stop", "message": {"content": "NO_QA"}}]
                }).encode())
                client = llm.ChatClient("https://example.invalid", "test", reasoning_effort=effort)
                client.ask("test", {})
                sent = json.loads(opener.return_value.open.call_args.args[0].data)
                self.assertEqual(sent.get("reasoning_effort"), effort)
                self.assertEqual(client.usage[0].get("reasoning_effort"), effort)
                self.assertNotIn("max_tokens", sent)
                self.assertNotIn("max_completion_tokens", sent)
                if effort is None:
                    self.assertNotIn("reasoning_effort", sent)

    def test_host_author_and_agent_config_preserve_separate_efforts(self):
        original = {"image": "sdk", "execution_image": "executor", "execution_backend": "ssh_sandbox",
                    "code": dict(model="solver", key_env="TEST_KEY", reasoning_effort="high"),
                    "judge": dict(model="judge", key_env="TEST_KEY", reasoning_effort="max",
                                  base_url="https://example.invalid")}
        config = self.configured(original, control=True)
        self.assertEqual(config["code"]["reasoning_effort"], "high")
        self.assertEqual(config["judge"]["reasoning_effort"], "max")
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict("os.environ", {"TEST_KEY": "test-only"}), \
                patch("dialogue_benchmark.llm.urllib.request.build_opener") as opener:
            opener.return_value.open.return_value = io.BytesIO(json.dumps({
                "choices": [{"finish_reason": "stop", "message": {"content": "NO_QA"}}]
            }).encode())
            ask_model("test", {}, config, directory)
            sent = json.loads(opener.return_value.open.call_args.args[0].data)
            self.assertEqual(sent["reasoning_effort"], "max")
        original["judge"]["reasoning_effort"] = "invalid"
        with self.assertRaisesRegex(ValueError, "reasoning_effort"):
            self.configured(original, control=True)

    def test_cli_timeout_default_override_and_invalid_values(self):
        parser = cli._build_parser()
        for flags, expected in (([], 90), (["--request-timeout", "1800"], 1800)):
            args = parser.parse_args(["input.json", "--output", "unused"] + flags)
            cli._parse_options(args, parser)
            self.assertEqual(args.request_timeout, expected)
        for value in ("0", "-1", "nan", "inf", "invalid"):
            with self.subTest(value=value), patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                args = parser.parse_args(["input.json", "--output", "unused", "--request-timeout", value])
                cli._parse_options(args, parser)

    def test_parallel_entrypoints_forward_timeout(self):
        for options, expected in (({}, 90), ({"request_timeout": 1800}, 1800)):
            for function, tasks in ((llm.generate_parallel, [{}]),
                                    (llm.generate_tasks, [{"scope": {}}])):
                with self.subTest(function=function.__name__, expected=expected), \
                        patch.object(llm, "ChatClient", return_value=SimpleNamespace(usage=[])) as client, \
                        patch.object(llm, "generate", return_value={"facts": [], "questions": []}):
                    function(tasks, "https://example.invalid", "test", **options)
                self.assertEqual(client.call_args.args[3], expected)

    def test_repository_probe_forwards_timeout(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch("dialogue_benchmark.repository_probe._repository_entries", return_value=[]), \
                patch("dialogue_benchmark.repository_probe.ChatClient") as client:
            client.return_value.usage = []
            client.return_value.responses = []
            client.return_value.ask.return_value = {"probe": {
                "decision": "uncertain", "reason": "No evidence", "query": "none", "evidence": "none"}}
            probe_candidate({"question": "Which rule applies?"}, directory, "https://example.invalid",
                            "test", "KEY", Path(directory) / "probe", request_timeout=1800)
            self.assertEqual(client.call_args.args[3], 1800)
            client.return_value.ask.assert_called_once()

    def test_host_model_calls_inherit_judge_timeout(self):
        for settings, expected in (({}, 90), ({"request_timeout": 1800}, 1800)):
            config = {"judge": dict(base_url="https://example.invalid", model="test",
                                    key_env="BENCHMARK_API_KEY", **settings)}
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory, \
                    patch.dict("os.environ", {"BENCHMARK_API_KEY": "test-only"}), \
                    patch("dialogue_benchmark.llm.urllib.request.build_opener") as opener:
                opener.return_value.open.return_value = io.BytesIO(json.dumps({
                    "choices": [{"finish_reason": "stop", "message": {"content": "NO_QA"}}]
                }).encode())
                ask_model("test", {}, config, directory)
                self.assertEqual(opener.return_value.open.call_args.kwargs, {"timeout": expected})

    def configured(self, original, control):
        with patch.dict("sys.modules", {"simulator.episode": SimpleNamespace(load_environment=lambda _: None)}), \
                patch("sys.path", []), patch.dict("os.environ", {"TEST_KEY": "test-only"}), \
                patch("dialogue_benchmark.task_eval.runtime.read",
                      return_value=original if control else {"config": original}):
            return configure("/simulator", "/checkpoint.json", "/provider.env",
                             control_config="/control.json" if control else None)

    def test_task_config_preserves_role_timeouts_and_absent_defaults(self):
        for control in (False, True):
            for settings in ({}, {"request_timeout": 1800}):
                original = {"image": "sdk", "execution_image": "executor", "execution_backend": "ssh_sandbox",
                            "code": dict(model="solver", key_env="TEST_KEY", **settings),
                            "judge": dict(model="judge", key_env="TEST_KEY", **settings)}
                with self.subTest(control=control, settings=settings):
                    config = self.configured(original, control)
                    for role in ("code", "judge"):
                        if settings:
                            self.assertEqual(config[role]["request_timeout"], 1800)
                        else:
                            self.assertNotIn("request_timeout", config[role])
                        self.assertIsNone(config[role]["max_output_tokens"])
                        self.assertNotIn("max_output_tokens", original[role])

    def test_task_config_rejects_invalid_role_timeouts(self):
        for role in ("code", "judge"):
            for value in (0, -1, float("nan"), float("inf"), None, True, "1800"):
                original = {"image": "sdk", "execution_image": "executor", "execution_backend": "ssh_sandbox",
                            "code": dict(model="solver", key_env="TEST_KEY"),
                            "judge": dict(model="judge", key_env="TEST_KEY")}
                original[role]["request_timeout"] = value
                with self.subTest(role=role, value=value), self.assertRaisesRegex(ValueError, "positive finite number"):
                    self.configured(original, control=True)
