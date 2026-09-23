"""Offline orchestration contracts for task preflight and paired evaluation."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from dialogue_benchmark.task_eval.artifacts import read, save
from dialogue_benchmark.task_eval.checks import run_checks
from dialogue_benchmark.task_eval.run import construct
from dialogue_benchmark.task_eval.runtime import review_task
from dialogue_benchmark.task_eval.versions import pin_baseline


class TaskPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.baseline = self.base / "baseline"
        self.baseline.mkdir()
        (self.baseline / "a.py").write_text("value = 1\n")
        pin_baseline(self.baseline)
        input_path = self.base / "qa-input.json"
        save(input_path, {"payload": "Actual evidence"})
        self.item = {"qa": {"type": "constraint_followthrough", "id": "q1", "answer_points": [{"text": "Historical behavior"}]},
                     "generation_input": str(input_path)}
        self.root = self.base / "task"
        self.with_coverage = True
        self.validator_status = "ConversationExecutionStatus.FINISHED"
        self.reference_status = "ConversationExecutionStatus.FINISHED"

    def fake_agent(self, root, config, role, message, **kwargs):
        if root.name == "author":
            checks = root / "workspace/checks"
            checks.mkdir()
            for name, text in (("task.md", "Preserve pending data after replacement"),
                               ("acceptance.md", "| a1 | Pending data remains readable | task | test: test_acceptance::test_feature |"),
                               ("test_acceptance.py", "Original test")):
                (checks / name).write_text(text)
        elif root.name == "reference-solver":
            save(root / "trajectory.json", [{"id": "reference-action"}])
            return {"status": self.reference_status}
        elif root.name == "validator":
            checks = root / "workspace/checks"
            checks.mkdir()
            (checks / "validation.txt").write_text(
                "BASELINE: unmet\nREFERENCE: pass\nTESTS: executable\n"
                "MUTATIONS: caught\nCOVERAGE: complete\nVERDICT: accept\n")
            if self.with_coverage:
                (checks / "coverage.md").write_text("Combine writing and replacement")
                (checks / "test_interactions.py").write_text("Combined-condition test")
            return {"status": self.validator_status}
        return {"status": "ConversationExecutionStatus.FINISHED"}

    def execute(self, results):
        results = [dict(r, cases=[{"id": "test_acceptance::test_feature", "status": r["status"]}]) for r in results]
        with patch("dialogue_benchmark.task_eval.run.run_agent", side_effect=self.fake_agent), \
             patch("dialogue_benchmark.task_eval.run.review_task", return_value={"status": "clean", "issue": "none"}), \
             patch("dialogue_benchmark.task_eval.run.run_checks", side_effect=results) as checks:
            receipt = construct(self.item, self.root, self.baseline, {"execution_image": "image"}, 0, {})
        return receipt, checks

    def test_extra_checks_are_run_before_freeze_and_used_for_receipt(self):
        receipt, checks = self.execute([{"status": "failed"}, {"status": "passed"},
                                       {"status": "failed", "tests": 2}, {"status": "passed", "tests": 2}])
        self.assertEqual(checks.call_count, 4)
        final_spec = checks.call_args_list[2].args[1]
        self.assertTrue((final_spec / "test_interactions.py").exists())
        self.assertTrue((self.root / "frozen/test_interactions.py").exists())
        self.assertEqual(receipt["reference_checks"]["tests"], 2)
        self.assertFalse((self.root / "frozen/checkpoints.json").exists())
        self.assertEqual(read(self.root / "frozen/acceptance.json")[0]["id"], "a1")

    def test_reference_failing_new_combination_is_not_frozen(self):
        receipt, _ = self.execute([{"status": "failed"}, {"status": "passed"},
                                   {"status": "failed"}, {"status": "failed"}])
        self.assertIsNone(receipt)
        self.assertFalse((self.root / "frozen").exists())
        self.assertFalse(read(self.root / "construction.json")[0]["accepted"])

    def test_model_complete_without_saved_coverage_does_not_pass(self):
        self.with_coverage = False
        receipt, checks = self.execute([{"status": "failed"}, {"status": "passed"}])
        self.assertIsNone(receipt)
        self.assertEqual(checks.call_count, 2)

    def test_interrupted_validator_cannot_admit_with_earlier_accept_report(self):
        self.validator_status = "ConversationExecutionStatus.STUCK"
        receipt, _ = self.execute([{"status": "failed"}, {"status": "passed"},
                                   {"status": "failed"}, {"status": "passed"}])
        self.assertIsNone(receipt)
        self.assertFalse((self.root / "frozen").exists())

    def test_interrupted_reference_does_not_produce_checkpoints(self):
        self.reference_status = "ConversationExecutionStatus.STUCK"
        receipt, _ = self.execute([{"status": "failed"}, {"status": "passed"},
                                   {"status": "failed"}, {"status": "passed"}])
        self.assertIsNone(receipt)
        self.assertNotIn("checkpoint_extraction", read(self.root / "construction.json")[0])

    def test_leaking_task_is_rejected_before_reference_solver(self):
        with patch("dialogue_benchmark.task_eval.run.run_agent", side_effect=self.fake_agent) as agent, \
             patch("dialogue_benchmark.task_eval.run.review_task", return_value={"status": "leaked", "issue": "Internal fix given"}), \
             patch("dialogue_benchmark.task_eval.run.run_checks") as checks:
            receipt = construct(self.item, self.root, self.baseline, {"execution_image": "image"}, 0, {})
        self.assertIsNone(receipt)
        self.assertEqual(agent.call_count, 1)
        checks.assert_not_called()
        self.assertTrue((self.root / "author-reference/previous-00/task.md").exists())

    def test_public_review_is_small_and_does_not_resolve_conflicts_as_clean(self):
        config = {"judge": {"base_url": "https://example.com/v1", "model": "test", "key_env": "KEY"}}
        with patch("dialogue_benchmark.llm.ChatClient") as client:
            client.return_value.usage = []
            client.return_value.responses = []
            client.return_value.ask.return_value = {"reviews": [{"leakage": "clean", "issue": "Gives internal fix"}]}
            result = review_task("Public task", "Historical answer", config, self.base / "review")
        self.assertEqual(result["status"], "uncertain")
        payload = client.return_value.ask.call_args.args[1]
        self.assertEqual(set(payload), {"public_task", "historical_answer"})

    def test_test_runner_executes_frozen_combinations_with_terminal_parity(self):
        spec, output = self.base / "spec", self.base / "checks-run"
        spec.mkdir()
        for name in ("test_acceptance.py", "test_interactions.py"):
            (spec / name).write_text("test content")
        # A report retained in the source must not become this execution's receipt.
        (spec / "receipt.xml").write_text("<testsuite><testcase /></testsuite>")
        class Sandbox:
            name = "test-sandbox"
            def __init__(self, directory, workspace, image, role, identity, reference):
                self.workspace = workspace
            def prepare(self):
                (self.workspace / "experiments").mkdir()
            def unpause(self):
                pass
            def pause(self):
                pass
        with patch.dict("sys.modules", {"simulator.openhands.sandbox": SimpleNamespace(ExecutionSandbox=Sandbox)}), \
             patch("dialogue_benchmark.task_eval.checks.release_completed_execution") as release, \
             patch("dialogue_benchmark.task_eval.checks.subprocess.run",
                   return_value=SimpleNamespace(returncode=124, stdout="", stderr="timeout")) as run:
            result = run_checks(self.baseline, spec, output, "image")
        command = run.call_args.args[0]
        self.assertIn("-t", command)
        self.assertIn("--rootdir=/workspace/checks", command)
        self.assertFalse(any("PYTHONPATH=" in part for part in command))
        self.assertIn("/workspace/checks/test_interactions.py", command)
        self.assertIn("--junitxml=/workspace/experiments/receipt.xml", command)
        self.assertEqual(result["status"], "error")
        release.assert_not_called()


if __name__ == "__main__":
    unittest.main()
