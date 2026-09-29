"""Offline checks for repeated evaluation of admitted, unchanged tasks."""

import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dialogue_benchmark.task_eval.artifacts import fingerprint, qa_fingerprint, read, save
from dialogue_benchmark.task_eval.repeat import main
from dialogue_benchmark.task_eval.versions import pin_baseline


class FrozenTaskRepeatTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.baseline = self.root / "source/baseline"
        self.baseline.mkdir(parents=True)
        (self.baseline / "feature.py").write_text("value = 1\n")
        pin_baseline(self.baseline)
        self.task = self.root / "source/tasks/task-01"
        self.spec = self.task / "frozen"
        self.spec.mkdir(parents=True)
        (self.spec / "task.md").write_text("Implement the public feature.")
        (self.spec / "acceptance.md").write_text("The feature must work.")
        (self.spec / "memory-use.md").write_text("Private author notes.")
        save(self.spec / "acceptance.json", [{"id": "a1", "basis": ["task"],
             "requirement": "The feature works", "tests": ["test_feature::test_value"]}])
        self.history = {"oracle_answer": "Original frozen oracle.", "contracts": [], "events": []}
        save(self.spec / "history.json", self.history)
        self.qa = {"id": "q1", "type": "M6", "answer_points": ["QA fallback answer."]}
        save(self.task / "author-reference/qa.json", self.qa)
        (self.task / "author-reference/secret.txt").write_text("Do not expose author files.")
        reference = self.task / "construction-00/reference-solver/workspace/candidate"
        reference.mkdir(parents=True)
        (reference / "solution.py").write_text("Do not expose reference code.")
        self.receipt = {"baseline_sha256": fingerprint(self.baseline),
                        "spec_sha256": fingerprint(self.spec), "qa_id": "q1",
                        "accepted_attempt": 0, "validation": {"VERDICT": "accept"}}
        save(self.task / "frozen.json", self.receipt)
        self.config = {"image": "sdk", "execution_image": "executor", "execution_backend": "ssh_sandbox",
                       "code": {"model": "original-solver", "key_env": "REPEAT_TEST_KEY", "max_output_tokens": 99},
                       "judge": {"model": "original-judge", "key_env": "REPEAT_TEST_KEY"}}
        self.manifest = {"baseline": str(self.baseline), "baseline_sha256": fingerprint(self.baseline),
                         "config": self.config, "execution": {"agent_requests": 7, "agent_tokens": 12345,
                                                              "agent_seconds": 42},
                         "tasks": [{"task": "task-01", "status": "error"}]}
        save(self.task.parent / "manifest.json", self.manifest)
        self.output = self.root / "repeat"
        self.calls = []
        patches = [
            patch.dict("sys.modules", {"simulator.episode": SimpleNamespace(load_environment=lambda _: None)}),
            patch("sys.path", list(sys.path)), patch.dict(os.environ, {"REPEAT_TEST_KEY": "offline-test"}),
            patch("dialogue_benchmark.task_eval.run.run_agent", side_effect=self.agent),
            patch("dialogue_benchmark.task_eval.run.run_checks", return_value=self.checks()),
            patch("dialogue_benchmark.task_eval.run.construct"),
        ]
        for mocked in patches:
            mocked.start()
            self.addCleanup(mocked.stop)

    def args(self):
        return ["--source-task", str(self.task), "--simulator-path", str(self.root / "simulator"),
                "--env-file", str(self.root / "provider.env"), "--output", str(self.output)]

    def checks(self, status="passed"):
        return {"status": status, "cases": [{"id": "test_feature::test_value", "status": status}]}

    def agent(self, root, config, role, message, **options):
        self.assertEqual(role, "code")
        self.assertNotIn("reference", options)
        self.assertEqual(options["history"], self.history)
        self.assertEqual(options["max_requests"], 7)
        self.assertEqual(options["max_tokens"], 12345)
        self.assertEqual(options["max_seconds"], 42)
        self.assertEqual(config["code"]["model"], "original-solver")
        self.assertIsNone(config["code"]["max_output_tokens"])
        self.assertIsNone(config["judge"]["max_output_tokens"])
        candidate = root / "workspace/candidate"
        self.assertEqual([path.name for path in candidate.iterdir()], ["feature.py"])
        self.assertEqual((candidate / "feature.py").read_text(), "value = 1\n")
        self.assertIn("Implement the public feature.", message)
        for private in ("Private author notes.", "Do not expose", "QA fallback answer."):
            self.assertNotIn(private, message)
        self.calls.append((root, message))
        (candidate / "feature.py").write_text("value = %d\n" % (len(self.calls) + 1))
        result = {"status": "finished", "metrics": {"total_tokens": 10, "usage_complete": True}}
        save(root / "result.json", result)
        return result

    def test_two_fresh_pairs_use_same_hash_and_reverse_order(self):
        original = fingerprint(self.task)
        self.assertEqual(main(self.args()), 0)
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(["Original frozen oracle." in message for _, message in self.calls],
                         [False, True, True, False])
        self.assertEqual(len({root for root, _ in self.calls}), 4)
        self.assertEqual(fingerprint(self.task), original)
        self.assertEqual(fingerprint(self.output / "baseline"), fingerprint(self.baseline))
        manifest = read(self.output / "manifest.json")
        self.assertEqual(manifest["completed"], 2)
        self.assertEqual(manifest["spec_sha256"], self.receipt["spec_sha256"])
        self.assertEqual(manifest["status"], "completed")
        for pair in ("pair-01", "pair-02"):
            root = self.output / pair
            self.assertEqual(fingerprint(root / "frozen"), self.receipt["spec_sha256"])
            self.assertEqual(read(root / "frozen.json"), self.receipt)
            self.assertEqual(list((root / "author-reference").iterdir()), [root / "author-reference/qa.json"])
            self.assertFalse((root / "construction-00").exists())
        first = read(self.output / "pair-01/manifest.json")
        self.assertEqual(first, manifest)
        self.assertEqual(read(self.output / "pair-02/manifest.json"), manifest)
        self.assertEqual(read(self.output / "source.json")["frozen"], self.receipt)
        self.assertTrue((self.output / "report.html").exists())
        self.assertIn("Pass-rate difference (with − without): 0.0 percentage points.",
                      (self.output / "report.md").read_text())

    def test_failed_trials_are_not_filtered_or_replaced(self):
        with patch("dialogue_benchmark.task_eval.run.run_checks", return_value=self.checks("failed")):
            self.assertEqual(main(self.args()), 0)
        manifest = read(self.output / "manifest.json")
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(manifest["completed"], 2)
        self.assertEqual([trial["result"] for row in manifest["tasks"]
                          for trial in row["comparison"].values()], ["failed"] * 4)
        self.assertIn("0/2/0", (self.output / "report.md").read_text())

    def test_completed_uncertain_trials_are_reported_separately_from_failures(self):
        with patch("dialogue_benchmark.task_eval.run.run_checks", return_value=self.checks("uncertain")):
            self.assertEqual(main(self.args()), 0)
        manifest = read(self.output / "manifest.json")
        self.assertEqual(manifest["completed"], 2)
        self.assertEqual([trial["result"] for row in manifest["tasks"]
                          for trial in row["comparison"].values()], ["uncertain"] * 4)
        self.assertEqual([row["paired_differences"]["completion_difference"] for row in manifest["tasks"]], [0, 0])
        self.assertIn("0/0/2", (self.output / "report.md").read_text())

    def test_execution_error_keeps_partial_pair_and_attempts_second_pair(self):
        with patch("dialogue_benchmark.task_eval.run.run_checks", side_effect=[
                self.checks(), RuntimeError("Check runner unavailable"), self.checks("failed"), self.checks()]):
            self.assertEqual(main(self.args()), 1)
        manifest = read(self.output / "manifest.json")
        self.assertEqual(len(self.calls), 4)
        first, second = manifest["tasks"]
        self.assertEqual(first["status"], "error")
        self.assertEqual(list(first["comparison"]), ["without_memory"])
        self.assertEqual(second["status"], "evaluated")
        self.assertEqual(list(second["comparison"]), ["with_memory", "without_memory"])
        self.assertTrue((self.output / "pair-01/failure.json").exists())
        self.assertTrue((self.output / "pair-01/trial-2/result.json").exists())
        self.assertIsNone(first["paired_differences"]["completion_difference"])
        self.assertEqual(second["paired_differences"]["completion_difference"], -1)
        report = (self.output / "report.md").read_text()
        self.assertIn("| without_memory | 2 | 2/0/0 | 100.0% |", report)
        self.assertIn("| with_memory | 1 | 0/1/0 | 0.0% |", report)
        self.assertNotIn("Pass-rate difference", report)

    def test_interruption_retains_completed_condition_and_pending_pair(self):
        with patch("dialogue_benchmark.task_eval.run.run_checks", side_effect=[self.checks(), KeyboardInterrupt()]):
            with self.assertRaises(KeyboardInterrupt):
                main(self.args())
        manifest = read(self.output / "manifest.json")
        self.assertEqual(manifest["status"], "interrupted")
        self.assertEqual([row["status"] for row in manifest["tasks"]], ["interrupted", "pending"])
        self.assertEqual(list(manifest["tasks"][0]["comparison"]), ["without_memory"])
        self.assertEqual(read(self.output / "pair-01/manifest.json"), manifest)
        self.assertEqual(len(self.calls), 2)
        self.assertIsNone(read(self.output / "pair-01/paired-differences.json")["completion_difference"])
        self.assertIsNone(manifest["tasks"][0]["paired_differences"]["completion_difference"])
        self.assertNotIn("Pass-rate difference", (self.output / "report.md").read_text())

    def test_changed_source_hash_is_rejected_before_any_solver(self):
        for path in (self.spec / "task.md", self.baseline / "feature.py"):
            with self.subTest(path=path):
                original = path.read_bytes()
                path.write_text("Changed input")
                with self.assertRaises(ValueError):
                    main(self.args())
                self.assertFalse(self.output.exists())
                self.assertFalse(self.calls)
                path.write_bytes(original)

    def test_source_qa_change_stops_before_second_pair(self):
        def checks(*args, **kwargs):
            if len(self.calls) == 2:
                save(self.task / "author-reference/qa.json", dict(self.qa, answer_points=["Changed oracle"]))
            return self.checks()
        with patch("dialogue_benchmark.task_eval.run.run_checks", side_effect=checks):
            with self.assertRaisesRegex(ValueError, "Source inputs changed"):
                main(self.args())
        self.assertEqual(len(self.calls), 2)
        manifest = read(self.output / "manifest.json")
        self.assertEqual(manifest["status"], "error")
        self.assertEqual(manifest["tasks"][1]["status"], "pending")

    def test_same_id_answer_edit_is_rejected_before_repetition(self):
        save(self.task / "frozen.json", dict(self.receipt, qa_sha256=qa_fingerprint(self.qa)))
        save(self.task / "author-reference/qa.json", dict(self.qa, answer_points=["Different answer"]))
        with self.assertRaisesRegex(ValueError, "Frozen QA changed"):
            main(self.args())
        self.assertFalse(self.calls)
        self.assertFalse(self.output.exists())

    def test_existing_output_is_retained(self):
        self.output.mkdir()
        marker = self.output / "existing.txt"
        marker.write_text("Keep existing output")
        with self.assertRaises(SystemExit):
            main(self.args())
        self.assertEqual(marker.read_text(), "Keep existing output")
        self.assertFalse(self.calls)


if __name__ == "__main__":
    unittest.main()
