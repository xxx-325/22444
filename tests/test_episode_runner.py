import json
import hashlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from dialogue_benchmark.task_eval.artifacts import read, save
from run_episode import main


class EpisodeRunnerTests(unittest.TestCase):
    def test_qa_only_retains_answers_and_evidence_without_starting_tasks(self):
        for questions in ([], [{"id": "q1", "qa_mode": "memory", "type": "M1",
                                "question": "Which customer rule applies?"}]):
            with self.subTest(questions=questions), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source"
                candidate = source / "workspace/candidate"
                candidate.mkdir(parents=True)
                (candidate / "a.py").write_text("value = True\n")
                (source / "session.jsonl").write_text(json.dumps({"kind": "user", "content": "rule"}) + "\n")
                config = {"judge": {"base_url": "https://example.invalid", "model": "test", "key_env": "KEY"}}
                def generated(args):
                    output = Path(args[args.index("--output") + 1])
                    save(output / "qa-public.json", {"status": "completed", "questions": questions})
                    save(output / "manifest.json", {"qa_mode": "memory", "usage": [
                        dict(prompt_tokens=10, completion_tokens=5)]})
                    return 0
                with patch("run_episode.configure", return_value=config), \
                     patch("run_episode.generate_qa", side_effect=generated), \
                     patch("run_episode.run_tasks") as tasks, \
                     patch("run_episode.preflight_openhands_runtime"), \
                     patch("run_episode.compact_run") as compact, \
                     patch("render_run.build", return_value={}):
                    self.assertEqual(main(["--source-run", str(source), "--simulator-path", str(root),
                        "--env-file", str(root / ".env"), "--output", str(root / "run"), "--qa-only"]), 0)
                tasks.assert_not_called()
                compact.assert_not_called()
                self.assertFalse((root / "run/tasks").exists())
                self.assertEqual(read(root / "run/qa/qa-public.json")["questions"], questions)
                self.assertEqual(read(root / "run/usage.json")["total_tokens"], 15)
                state = read(root / "run/pipeline.json")
                self.assertEqual(state["status"], "completed")
                self.assertEqual(state["stop_reason"], "qa_only" if questions else "no_eligible_qa")
                page = (root / "run/report.html").read_text()
                self.assertIn("记忆召回 QA", page)
                self.assertNotIn('href="tasks/', page)
                self.assertNotIn("需求生成尚未完成", page)
                self.assertTrue((root / "run/qa-viewer/index.html").is_file())

    def test_cleanup_failure_keeps_the_usage_receipt_saved_before_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            candidate = source / "workspace/candidate"
            candidate.mkdir(parents=True)
            (candidate / "a.py").write_text("x = 1\n")
            (source / "session.jsonl").write_text(json.dumps({"kind": "user", "content": "rule"}) + "\n")
            config = {"judge": {"base_url": "https://example.invalid", "model": "test", "key_env": "KEY"}}
            def generated(args):
                save(Path(args[args.index("--output") + 1]) / "qa-public.json", {"questions": [{"id": "q1"}]})
                return 0
            def tasks(args):
                save(Path(args[args.index("--output") + 1]) / "manifest.json", {"tasks": []})
                return 0
            receipt = dict(requests=3, prompt_tokens=100, completion_tokens=20, complete=True)
            with patch("run_episode.configure", return_value=config), \
                 patch("run_episode.generate_qa", side_effect=generated), \
                 patch("run_episode.run_tasks", side_effect=tasks), patch("run_episode.render"), \
                 patch("run_episode.preflight_openhands_runtime"), \
                 patch("run_episode.episode_usage", return_value=receipt) as usage, \
                 patch("run_episode.compact_run", side_effect=RuntimeError("cleanup failed")):
                with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                    main(["--source-run", str(source), "--simulator-path", str(root),
                          "--env-file", str(root / ".env"), "--output", str(root / "run")])
            usage.assert_called_once()
            self.assertEqual(read(root / "run/usage.json"), receipt)

    def test_external_mode_uses_sidecar_and_probes_the_pinned_final_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            candidate = source / "workspace/candidate"
            candidate.mkdir(parents=True)
            (candidate / "a.py").write_text("final_code = True\n")
            (source / "session.jsonl").write_text(json.dumps({"kind": "user", "content": "Customer rule"}) + "\n")
            events = source / "external-events.json"
            save(events, {"version": 1, "events": []})
            config = {"judge": {"base_url": "https://example.invalid", "model": "test", "key_env": "KEY",
                                "request_timeout": 1800, "reasoning_effort": "max"}}
            def generated(args):
                output = Path(args[args.index("--output") + 1])
                save(output / "qa-public.json", {"questions": [{"id": "q1"}]})
                return 0
            with patch("run_episode.configure", return_value=config), \
                 patch("run_episode.generate_qa", side_effect=generated) as qa, \
                 patch("run_episode.run_tasks", return_value=0) as tasks, \
                 patch("run_episode.preflight_openhands_runtime"), \
                 patch("run_episode.render"), patch("run_episode.compact_run"):
                self.assertEqual(main(["--source-run", str(source), "--simulator-path", str(root),
                                       "--env-file", str(root / ".env"), "--output", str(root / "run"),
                                       "--qa-source", "external", "--external-events", str(events),
                                       "--model-request-chars", "96000"]), 0)
            args = qa.call_args.args[0]
            self.assertEqual(args[args.index("--qa-source") + 1], "external")
            self.assertEqual(args[args.index("--qa-count") + 1], "40")
            self.assertEqual(args[args.index("--model-request-chars") + 1], "96000")
            task_args = tasks.call_args.args[0]
            self.assertEqual(task_args[task_args.index("--model-request-chars") + 1], "96000")
            self.assertEqual(read(root / "run/pipeline.json")["parameters"]["model_request_chars"], 96000)
            self.assertEqual(args[args.index("--request-timeout") + 1], "1800")
            self.assertEqual(args[args.index("--reasoning-effort") + 1], "max")
            self.assertNotIn("--qa-mode", args)
            self.assertNotIn("--general-count", args)
            self.assertNotIn("--code-count", args)
            self.assertNotIn("--adaptive-subgraphs", args)
            self.assertEqual(args[args.index("--external-events") + 1], str(events))
            self.assertEqual(args[args.index("--repository") + 1], str((root / "run/baseline").resolve()))

    def test_all_stages_use_converted_input_and_same_frozen_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            candidate = source / "workspace/candidate"
            candidate.mkdir(parents=True)
            (candidate / "a.py").write_text("final_dialogue_code = True\n")
            (source / "session.jsonl").write_text(json.dumps({"kind": "user", "content": "Keep behavior"}) + "\n")
            config = {"judge": {"base_url": "https://example.invalid", "model": "test", "key_env": "TEST_KEY"}}
            order = []
            def qa(args):
                order.append("qa")
                self.assertEqual(args[args.index("--request-timeout") + 1], "90")
                self.assertEqual(read(Path(args[0]))["version"], 1)
                self.assertEqual(args[args.index("--general-count") + 1], "40")
                self.assertEqual(args[args.index("--code-count") + 1], "40")
                self.assertNotIn("--repository", args)
                self.assertNotIn("--external-events", args)
                self.assertIn("--adaptive-subgraphs", args)
                output = Path(args[args.index("--output") + 1])
                save(output / "manifest.json", {
                    "input_sha256": hashlib.sha256(Path(args[0]).read_bytes()).hexdigest()})
                save(output / "qa-public.json", {"questions": [{"id": "q1"}]})
                return 0
            def tasks(args):
                order.append("resume" if "--resume" in args else "tasks")
                base = Path(args[args.index("--baseline") + 1])
                self.assertTrue((base / ".git").is_dir())
                self.assertEqual((base / "a.py").read_text(), (candidate / "a.py").read_text())
                save(Path(args[args.index("--output") + 1]) / "manifest.json", {"tasks": []})
                return 0
            with patch("run_episode.configure", return_value=config), \
                 patch("run_episode.generate_qa", side_effect=qa), \
                 patch("run_episode.run_tasks", side_effect=tasks), \
                 patch("run_episode.preflight_openhands_runtime"), \
                 patch("run_episode.render") as render, patch("run_episode.compact_run"):
                self.assertEqual(main(["--source-run", str(source), "--simulator-path", str(root),
                                       "--env-file", str(root / ".env"), "--output", str(root / "run")]), 0)
                self.assertEqual(main(["--source-run", str(source), "--simulator-path", str(root),
                                       "--env-file", str(root / ".env"), "--output", str(root / "run"),
                                       "--resume-tasks"]), 0)
            self.assertEqual(order, ["qa", "tasks", "resume"])
            self.assertEqual(render.call_count, 6)
            self.assertEqual(read(root / "run/pipeline.json")["status"], "completed")
            self.assertFalse((candidate / ".git").exists())

    def test_resume_rejects_changed_inputs_before_overwriting_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            (source / "workspace/candidate").mkdir(parents=True)
            (source / "session.jsonl").write_text('{"kind":"user","content":"rule"}\n')
            output = root / "run"
            save(output / "qa/manifest.json", {})
            save(output / "qa/qa-public.json", {"questions": []})
            saved = {"parameters": {"task_count": 99}, "status": "failed"}
            save(output / "pipeline.json", saved)
            with patch("run_episode.configure") as configure:
                with self.assertRaisesRegex(ValueError, "same inputs and parameters"):
                    main(["--source-run", str(source), "--simulator-path", str(root),
                          "--env-file", str(root / ".env"), "--output", str(output), "--resume"])
            configure.assert_not_called()
            self.assertEqual(read(output / "pipeline.json"), saved)

    def test_no_eligible_qa_saves_an_empty_task_report_without_starting_agents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            candidate = source / "workspace/candidate"
            candidate.mkdir(parents=True)
            (candidate / "a.py").write_text("value = True\n")
            (source / "session.jsonl").write_text(json.dumps({"kind": "user", "content": "rule"}) + "\n")
            config = {"judge": {"base_url": "https://example.invalid", "model": "test", "key_env": "KEY"}}
            def generated(args):
                output = Path(args[args.index("--output") + 1])
                save(output / "qa-public.json", {"questions": []})
                return 0
            with patch("run_episode.configure", return_value=config), \
                 patch("run_episode.generate_qa", side_effect=generated), \
                 patch("run_episode.run_tasks") as tasks, patch("run_episode.render"), \
                 patch("run_episode.preflight_openhands_runtime"):
                status = main(["--source-run", str(source), "--simulator-path", str(root),
                               "--env-file", str(root / ".env"), "--output", str(root / "run")])
            self.assertEqual(status, 0)
            tasks.assert_not_called()
            self.assertEqual(read(root / "run/tasks/manifest.json")["stop_reason"], "no_eligible_qa")
            self.assertTrue((root / "run/tasks/report.md").is_file())
            self.assertEqual(read(root / "run/pipeline.json")["status"], "completed")

    def test_failed_qa_is_not_reported_as_an_empty_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            candidate = source / "workspace/candidate"
            candidate.mkdir(parents=True)
            (candidate / "a.py").write_text("value = True\n")
            (source / "session.jsonl").write_text(json.dumps({"kind": "user", "content": "rule"}) + "\n")
            config = {"judge": {"base_url": "https://example.invalid", "model": "test", "key_env": "KEY"}}
            def generated(args):
                output = Path(args[args.index("--output") + 1])
                save(output / "qa-public.json", {"status": "failed", "questions": []})
                save(output / "qa-audit.json", {"stage_errors": [{"error_code": "request_budget"}]})
                return 0
            receipt = dict(requests=0, prompt_tokens=0, completion_tokens=0, complete=True)
            with patch("run_episode.configure", return_value=config), \
                 patch("run_episode.generate_qa", side_effect=generated), \
                 patch("run_episode.run_tasks") as tasks, patch("run_episode.render"), \
                 patch("run_episode.preflight_openhands_runtime"), \
                 patch("run_episode.episode_usage", return_value=receipt):
                with self.assertRaisesRegex(RuntimeError, "QA generation failed"):
                    main(["--source-run", str(source), "--simulator-path", str(root),
                          "--env-file", str(root / ".env"), "--output", str(root / "run")])
            tasks.assert_not_called()
            state = read(root / "run/pipeline.json")
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["stop_reason"], "qa_generation_failed")
            self.assertEqual(read(root / "run/usage.json"), receipt)
            self.assertFalse((root / "run/tasks").exists())

    def run_external_fact_episode(self, content, *, expected_error=None, request_chars=32000, http_error=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        source = root / "source"
        candidate = source / "workspace/candidate"
        candidate.mkdir(parents=True)
        (candidate / "a.py").write_text("value = True\n")
        (source / "session.jsonl").write_text(json.dumps({
            "kind": "user", "text": "Maple 本次晚班作业选择每批 2 条，批次前缀 MP。"}) + "\n")
        save(source / "external-events.json", {"version": 1, "events": [{
            "id": "x1", "kind": "compatibility_contract", "memory_kind": "M1", "source_ids": ["e1"]}]})
        config = {"judge": {"base_url": "https://example.invalid", "model": "offline-test",
                            "key_env": "BENCHMARK_OFFLINE_TEST_KEY"}}
        envelope = {"choices": [{"finish_reason": "stop", "message": {"content": content}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}
        with patch("run_episode.configure", return_value=config), \
             patch("dialogue_benchmark.llm.urllib.request.build_opener") as transport, \
             patch.dict(os.environ, {"BENCHMARK_OFFLINE_TEST_KEY": "offline-fixture-token"}), \
             patch("run_episode.run_tasks") as tasks, patch("run_episode.render"), \
             patch("run_episode.preflight_openhands_runtime"), \
             patch("dialogue_benchmark.repository_probe.probe_candidate") as probe:
            transport.return_value.open.return_value = io.BytesIO(json.dumps(envelope).encode())
            transport.return_value.open.side_effect = http_error
            args = ["--source-run", str(source), "--simulator-path", str(root),
                    "--env-file", str(root / ".env"), "--output", str(root / "run"),
                    "--qa-source", "external", "--external-events", str(source / "external-events.json"),
                    "--qa-count", "1", "--parallel-workers", "1",
                    "--model-request-chars", str(request_chars)]
            if expected_error:
                with self.assertRaisesRegex(RuntimeError, "QA generation failed"):
                    main(args)
            else:
                self.assertEqual(main(args), 0)
            tasks.assert_not_called()
            probe.assert_not_called()
            self.assertEqual(transport.return_value.open.call_count, 0 if expected_error == "request_budget" else 1)
        return root / "run"

    def test_successful_empty_external_facts_complete_without_tasks(self):
        output = self.run_external_fact_episode("NO_FACTS")
        public = read(output / "qa/qa-public.json")
        self.assertEqual(public["status"], "completed_no_questions")
        self.assertEqual(public["questions"], [])
        self.assertEqual(read(output / "qa/facts.json"), [])
        extraction = read(output / "qa/fact-extraction.json")
        self.assertEqual(extraction["stage_errors"], [])
        self.assertEqual(extraction["stage_status"][0]["facts"], "completed")
        self.assertEqual(extraction["usage"][0]["status"], "completed")
        audit = read(output / "qa/qa-audit.json")
        self.assertEqual(audit["stage_errors"], [])
        skipped, = [row for row in audit["stage_status"] if row.get("reason") == "no_external_fact"]
        self.assertEqual(skipped["grouping"], "skipped")
        self.assertEqual(audit["progress"]["stop_reasons"], {"memory": "pool_exhausted"})
        self.assertEqual(audit["progress"]["requests"], 1)
        self.assertEqual(read(output / "qa/manifest.json")["failures"]["total"], 0)
        state = read(output / "pipeline.json")
        self.assertEqual((state["status"], state["stop_reason"]), ("completed", "no_eligible_qa"))
        self.assertEqual(read(output / "tasks/manifest.json")["tasks"], [])
        self.assertTrue((output / "tasks/report.md").is_file())

    def test_external_fact_http_protocol_and_budget_errors_remain_failed(self):
        for error, content, request_chars in (("http_error", "NO_FACTS", 32000),
                                              ("protocol_error", "untagged response", 32000),
                                              ("request_budget", "NO_FACTS", 1000)):
            with self.subTest(error=error):
                http_error = HTTPError("https://example.invalid", 503, "Offline fixture", None, None) \
                    if error == "http_error" else None
                output = self.run_external_fact_episode(content, expected_error=error,
                    request_chars=request_chars, http_error=http_error)
                self.assertEqual(read(output / "qa/qa-public.json")["status"], "failed")
                errors = read(output / "qa/stage-errors.json")
                self.assertEqual([row["error_code"] for row in errors], [error])
                self.assertEqual(read(output / "qa/manifest.json")["failures"]["total"], 1)
                extraction = read(output / "qa/fact-extraction.json")
                self.assertEqual(extraction["stage_status"][0]["facts"], "failed")
                state = read(output / "pipeline.json")
                self.assertEqual((state["status"], state["stop_reason"]), ("failed", "qa_generation_failed"))
                self.assertFalse((output / "tasks").exists())
