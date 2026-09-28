import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dialogue_benchmark.task_eval.artifacts import read, save
from run_episode import main


class EpisodeRunnerTests(unittest.TestCase):
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
            config = {"judge": {"base_url": "https://example.invalid", "model": "test", "key_env": "KEY"}}
            def generated(args):
                output = Path(args[args.index("--output") + 1])
                save(output / "qa-public.json", {"questions": [{"id": "q1"}]})
                return 0
            with patch("run_episode.configure", return_value=config), \
                 patch("run_episode.generate_qa", side_effect=generated) as qa, \
                 patch("run_episode.run_tasks", return_value=0), \
                 patch("run_episode.render"), patch("run_episode.compact_run"):
                self.assertEqual(main(["--source-run", str(source), "--simulator-path", str(root),
                                       "--env-file", str(root / ".env"), "--output", str(root / "run"),
                                       "--qa-source", "external", "--external-events", str(events)]), 0)
            args = qa.call_args.args[0]
            self.assertEqual(args[args.index("--qa-source") + 1], "external")
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
                self.assertEqual(read(Path(args[0]))["version"], 1)
                self.assertEqual(args[args.index("--general-count") + 1], "40")
                output = Path(args[args.index("--output") + 1])
                save(output / "manifest.json", {
                    "input_sha256": hashlib.sha256(Path(args[0]).read_bytes()).hexdigest()})
                save(output / "qa-public.json", {"questions": [{"id": "q1"}]})
                return 0
            def tasks(args):
                order.append("recovery" if "--recover-checkpoints" in args else "tasks")
                base = Path(args[args.index("--baseline") + 1])
                self.assertTrue((base / ".git").is_dir())
                self.assertEqual((base / "a.py").read_text(), (candidate / "a.py").read_text())
                return 0
            with patch("run_episode.configure", return_value=config), \
                 patch("run_episode.generate_qa", side_effect=qa), \
                 patch("run_episode.run_tasks", side_effect=tasks), \
                 patch("run_episode.render") as render:
                self.assertEqual(main(["--source-run", str(source), "--simulator-path", str(root),
                                       "--env-file", str(root / ".env"), "--output", str(root / "run")]), 0)
                self.assertEqual(main(["--source-run", str(source), "--simulator-path", str(root),
                                       "--env-file", str(root / ".env"), "--output", str(root / "run"),
                                       "--resume-tasks"]), 0)
            self.assertEqual(order, ["qa", "tasks", "tasks"])
            self.assertEqual(render.call_count, 4)
            self.assertEqual(read(root / "run/pipeline.json")["status"], "completed")
            self.assertFalse((candidate / ".git").exists())

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
                 patch("run_episode.run_tasks") as tasks, patch("run_episode.render"):
                status = main(["--source-run", str(source), "--simulator-path", str(root),
                               "--env-file", str(root / ".env"), "--output", str(root / "run")])
            self.assertEqual(status, 0)
            tasks.assert_not_called()
            self.assertEqual(read(root / "run/tasks/manifest.json")["stop_reason"], "no_eligible_qa")
            self.assertTrue((root / "run/tasks/report.md").is_file())
            self.assertEqual(read(root / "run/pipeline.json")["status"], "completed")
