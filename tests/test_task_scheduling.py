import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dialogue_benchmark.task_eval.artifacts import read
from dialogue_benchmark.task_eval.run import main


class TaskSchedulingTests(unittest.TestCase):
    def test_failed_requirement_does_not_consume_completed_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source/workspace/candidate"
            source.mkdir(parents=True)
            (source / "a.py").write_text("value = 1\n")
            items = [{"qa": {"type": "constraint_followthrough", "id": "q%d" % n}, "original_candidate": {"evidence_group_id": "g%d" % n}}
                     for n in range(5)]
            def construct(item, *args):
                return None if item["qa"]["id"] == "q0" else {"accepted": True}
            with patch("dialogue_benchmark.task_eval.run.qa_inputs", return_value=items), \
                 patch("dialogue_benchmark.task_eval.run.configure", return_value={}), \
                 patch("dialogue_benchmark.task_eval.run.construct", side_effect=construct) as author, \
                 patch("dialogue_benchmark.task_eval.run.evaluate", return_value={}) as evaluate:
                main(["--simulator-path", str(root), "--source-run", str(root / "source"),
                      "--qa-run", str(root), "--env-file", str(root / ".env"),
                      "--output", str(root / "output"), "--count", "2", "--workers", "2"])
            manifest = read(root / "output/manifest.json")
            self.assertEqual(author.call_count, 3)
            self.assertEqual(evaluate.call_count, 2)
            self.assertEqual(manifest["completed"], 2)
            self.assertEqual(manifest["stop_reason"], "target_met")
            self.assertEqual(manifest["shortfall"], 0)
            self.assertTrue((root / "output/report.html").is_file())
