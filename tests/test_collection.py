import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dialogue_benchmark.collection import episode_usage, run_collection, validate_plan
from dialogue_benchmark.task_eval.artifacts import read, save


class CollectionTests(unittest.TestCase):
    def test_costs_include_failed_calls_without_counting_aggregate_ledgers_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save(root / "qa/manifest.json", {"usage": [dict(prompt_tokens=10, completion_tokens=2)]})
            save(root / "tasks/task-01/selection/1/usage.json", [dict(prompt_tokens=20, completion_tokens=3)])
            save(root / "tasks/task-01/selection-budget.json", dict(requests=1, prompt_tokens=20, completion_tokens=3))
            save(root / "tasks/task-01/trial-1/private/budget.json", dict(
                attempts=2, calls=1, prompt_tokens=30, completion_tokens=4, usage_missing=True))
            save(root / "tasks/task-01/trial-1/workspace/candidate/usage.json", [{}])
            result = episode_usage(root)
            self.assertEqual(result["requests"], 4)
            self.assertEqual(result["total_tokens"], 69)
            self.assertFalse(result["complete"])
            self.assertEqual(len(result["receipts"]), 3)

    def plan(self, root, budget=100):
        save(root / "runtime.json", {"user": {"max_output_tokens": 12}})
        plan = dict(runtime_config="runtime.json", max_total_requests=budget, max_total_tokens=10000,
                    projects=[dict(id="planner", brief="Plan dependencies", increments=2,
                                   scenarios=[dict(id="first", memory_kinds=["M1"]), dict(id="second", memory_kinds=["M6"])])])
        save(root / "input.json", plan)
        return plan

    def command(self, command, cwd, log):
        target = Path(command[command.index("--output") + 1])
        self.commands.append(command)
        name = target.name
        budget = dict(attempts=1, prompt_tokens=10, completion_tokens=2)
        if name == "project":
            cfg = read(Path(command[command.index("--config") + 1]))
            self.assertIsNone(cfg["user"]["max_output_tokens"])
            save(target / "project.json", dict(status="completed"))
            save(target / "config.json", dict(repository=str(target), base="base-sha", tasks=[{"commit": "next-sha"}]))
            save(target / "private/budget.json", budget)
        elif name == "scenario":
            save(target / "frozen/report.json", dict(status="candidate_pass", design={"memory_kinds": ["M1"]}))
            save(target / "frozen/scenario.json", {})
            save(target / "budget.json", budget)
        elif name == "requirements":
            rejected = self.reject_first and target.parent.name == "first"
            save(target / "report.json", dict(status="rejected" if rejected else "candidate_pass"))
            save(target / "budget.json", budget)
        elif name == "dialogue":
            package = target.with_name("dialogue-package")
            save(package / "manifest.json", dict(schema="memory-episode-v1"))
            save(package / "external-events.json", dict(events=[{"memory_kind": "M1"}]))
            save(package / "private/review.json", dict(budget=budget))
        elif name == "evaluation":
            self.assertIn("--episode-manifest", command)
            self.assertIn("external", command)
            save(target / "pipeline.json", dict(status="completed", stop_reason="no_eligible_qa"))
            save(target / "qa/qa-public.json", dict(questions=[]))
            save(target / "usage.json", dict(requests=1, prompt_tokens=10, completion_tokens=2, complete=True))
        else:
            self.fail("Unexpected stage")
        return 0

    def invoke(self, root):
        self.commands = []
        with patch("dialogue_benchmark.collection._command", side_effect=self.command), \
             patch("dialogue_benchmark.collection.subprocess.check_output", return_value="root-sha\n"), \
             patch("dialogue_benchmark.episode_input.load_episode_manifest", side_effect=lambda p: {
                 "external_events": p.parent / "external-events.json"}):
            return run_collection(root / "input.json", root / "run", root, root / ".env")

    def test_rejected_scenario_is_retained_and_next_scenario_starts_from_same_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.plan(root)
            self.reject_first = True
            result = self.invoke(root)
            project = result["projects"][0]
            self.assertEqual(project["lineage"], ["root-sha"])
            self.assertEqual([s["status"] for s in project["scenarios"]],
                             ["requirements_rejected", "no_eligible_qa"])
            self.assertEqual(result["usage"]["requests"], 7)
            self.assertEqual(project["scenarios"][1]["paired_tasks"], 0)
            for scenario in ("first", "second"):
                self.assertEqual(read(root / "run/planner" / scenario / "config.json")["base"], "base-sha")

    def test_budget_stops_new_stages_without_retrying_or_dropping_prior_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.plan(root, budget=2)
            self.reject_first = False
            with self.assertRaisesRegex(RuntimeError, "collection_budget_exhausted"):
                self.invoke(root)
            result = read(root / "run/collection.json")
            self.assertEqual(result["status"], "stopped")
            self.assertEqual(len(self.commands), 2)
            self.assertEqual(result["usage"]["requests"], 2)
            self.assertTrue((root / "run/planner/project/config.json").is_file())

    def test_duplicate_or_escaping_targets_are_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.plan(Path(directory))
            plan["projects"][0]["id"] = "../project"
            with self.assertRaises(ValueError):
                validate_plan(plan)
