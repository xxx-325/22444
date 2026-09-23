import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dialogue_benchmark.task_eval.artifacts import read, save
from run_episode import main


class EpisodeRunnerTests(unittest.TestCase):
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
                save(output / "qa-public.json", {"questions": []})
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
