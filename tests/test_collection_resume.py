import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dialogue_benchmark.collection import episode_usage, run_collection
from dialogue_benchmark.task_eval.artifacts import read, save


class CollectionResumeTests(unittest.TestCase):
    def test_episode_usage_retains_compacted_ledgers_without_double_counting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save(root / "qa/manifest.json", {"usage": [
                {"attempts": 1, "prompt_tokens": 2, "completion_tokens": 1}]})
            save(root / "tasks/task-01/usage.json", [
                {"attempts": 2, "prompt_tokens": 5, "completion_tokens": 3}])
            save(root / "usage.json", episode_usage(root))
            (root / "tasks/task-01/usage.json").unlink()
            resumed = episode_usage(root)
            self.assertEqual(resumed["requests"], 3)
            self.assertEqual(resumed["total_tokens"], 11)
            self.assertTrue(resumed["complete"])

    def _plan(self, root):
        save(root / "runtime.json", {"user": {"model": "test"}})
        save(root / "prepared.json", {
            "repository": str(root), "base": "base-sha", "tasks": [],
            "development_plan": str(root / "development-plan.json")})
        plan = {
            "runtime_config": "runtime.json", "max_total_requests": 100,
            "max_total_tokens": 100000,
            "projects": [{"id": "project", "prepared_config": "prepared.json",
                          "scenarios": [{"id": "scenario", "memory_kinds": ["M1"]}]}],
        }
        save(root / "plan.json", plan)
        return plan

    def _command(self, command, cwd, log):
        target = Path(command[command.index("--output") + 1])
        name = target.name
        budget = {"attempts": 1, "prompt_tokens": 2, "completion_tokens": 1}
        if name == "scenario":
            save(target / "frozen/report.json", {"status": "candidate_pass",
                                                  "design": {"memory_kinds": ["M1"]}})
            save(target / "frozen/scenario.json", {})
            save(target / "budget.json", budget)
        elif name == "requirements":
            save(target / "report.json", {"status": "candidate_pass"})
            save(target / "budget.json", budget)
            if getattr(self, "interrupt_requirements", False):
                self.interrupt_requirements = False
                raise KeyboardInterrupt()
        elif name == "dialogue":
            package = target.with_name("dialogue-package")
            save(package / "manifest.json", {"schema": "memory-episode-v1"})
            save(package / "external-events.json", {"events": []})
            save(package / "private/review.json", {"budget": budget})
        elif name == "external":
            save(target / "pipeline.json", {"status": "completed", "stop_reason": "no_eligible_qa"})
            save(target / "qa/qa-public.json", {"questions": []})
            save(target / "usage.json", dict(budget, complete=True))
        else:
            self.fail("unexpected stage: " + name)
        return 0

    def _invoke(self, root, resume=False):
        with patch("dialogue_benchmark.collection._command", side_effect=self._command), \
             patch("dialogue_benchmark.collection.subprocess.check_output", return_value="root\n"), \
             patch("dialogue_benchmark.episode_input.load_episode_manifest",
                   return_value={}):
            return run_collection(root / "plan.json", root / "run", root, root / ".env",
                                  resume=resume)

    def test_resume_archives_interrupted_attempt_and_reuses_completed_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._plan(root)
            self.interrupt_requirements = True
            with self.assertRaises(KeyboardInterrupt):
                self._invoke(root)
            self.assertTrue((root / "run/project/scenario/scenario/frozen/report.json").is_file())
            result = self._invoke(root, resume=True)
            self.assertEqual(result["status"], "completed")
            self.assertTrue((root / "run/attempts/project/scenario/requirements/attempt-1").is_dir())
            self.assertEqual(sum(s.get("resume_count", 0) for s in result["stages"]), 0)

    def test_resume_rejects_changed_runtime_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._plan(root)
            self.interrupt_requirements = True
            with self.assertRaises(KeyboardInterrupt):
                self._invoke(root)
            save(root / "runtime.json", {"user": {"model": "changed"}})
            with self.assertRaisesRegex(ValueError, "same plan, runtime"):
                self._invoke(root, resume=True)


if __name__ == "__main__":
    unittest.main()
