import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dialogue_benchmark.collection import (EVALUATION_DEFAULTS, _aggregate_status,
                                            _completed_pair, _transient_stage_failure,
                                            _recover_receipt,
                                            episode_usage,
                                            run_collection, validate_plan)
from dialogue_benchmark.task_eval.artifacts import read, save


class CollectionTests(unittest.TestCase):
    def test_receipt_left_in_atomic_temp_file_is_recovered(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "pipeline.json"
            temporary = receipt.with_suffix(".json.tmp")
            save(temporary, {"status": "completed"})
            self.assertEqual(_recover_receipt(receipt)["status"], "completed")
            self.assertTrue(receipt.is_file())

    def test_only_evaluated_terminal_pairs_count(self):
        comparison = {
            "without_memory": {"result": "passed"},
            "with_memory": {"result": "failed"},
        }
        self.assertTrue(_completed_pair({"status": "evaluated", "comparison": comparison}))
        self.assertFalse(_completed_pair({"status": "completed", "comparison": comparison}))
        self.assertFalse(_completed_pair({
            "status": "evaluated",
            "comparison": {**comparison, "with_memory": {"result": "uncertain"}},
        }))
        self.assertFalse(_completed_pair({
            "status": "evaluated",
            "comparison": {**comparison, "with_memory": {
                "result": "failed", "execution_status": "interrupted"}},
        }))

    def test_only_gateway_failures_are_retryable(self):
        self.assertTrue(_transient_stage_failure(
            "ConversationRunError: BadGatewayError - Relay request failed"))
        self.assertFalse(_transient_stage_failure("SyntaxError: invalid input"))
        self.assertFalse(_transient_stage_failure("Unauthorized: invalid api key"))

    def test_empty_child_statuses_are_not_success(self):
        self.assertEqual(_aggregate_status([]), "no_scenarios")
        self.assertEqual(_aggregate_status(["project_rejected"]), "partial_failure")
        self.assertEqual(_aggregate_status(["completed", "below_target"]), "below_target")
        self.assertEqual(_aggregate_status(["completed", "evaluation_failed"]), "partial_failure")
        self.assertEqual(_aggregate_status(["qa_only"]), "completed")
        self.assertEqual(_aggregate_status(["qa_only", "completed"]), "completed")

    def test_quality_mapping_is_accepted(self):
        plan = {"projects": [dict(id="planner", brief="Plan dependencies",
                                  scenarios=[dict(id="first")])],
                "runtime_config": "runtime.json",
                "max_total_requests": 1, "max_total_tokens": 1,
                "dialogue_quality": {"min_external_events": 1}}
        self.assertIsNone(validate_plan(plan))
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            validate_plan({**plan, "dialogue_quality": {}})
        for bad in ({"unknown": 1}, {"min_external_events": -1},
                    {"require_external_event_closure": 1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_plan({**plan, "dialogue_quality": bad})

    def test_quality_mapping_is_forwarded_to_scenario(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.plan(root)
            self.reject_first = False
            self.quality_mapping = {
                "min_external_events": 1,
                "min_distinct_external_foci": 1,
                "require_external_event_closure": True,
            }
            plan["dialogue_quality"] = self.quality_mapping
            save(root / "input.json", plan)
            self.invoke(root)

    def test_volume_targets_are_reported_without_blocking_a_complete_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.plan(root)
            self.reject_first = False
            self.paired = True
            plan["evaluation"] = {"qa_count": 8, "task_count": 3, "task_budget": 4}
            save(root / "input.json", plan)
            result = self.invoke(root)
            evaluation = result["projects"][0]["scenarios"][0]["evaluations"]["external"]
            self.assertEqual(result["status"], "completed")
            self.assertEqual(evaluation["shortfall"], {"published_qa": 7, "paired_tasks": 2})
            report = (root / "run/collection.md").read_text()
            self.assertIn("QA target", report)
            self.assertIn("QA shortfall", report)

    def test_empty_qa_or_incomplete_pair_cannot_complete_a_route(self):
        cases = ("empty_qa", "no_pair", "uncertain_pair")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                plan = self.plan(root)
                self.reject_first = False
                if case == "empty_qa":
                    self.empty_qa = True
                elif case == "no_pair":
                    self.nonempty_qa = True
                else:
                    self.uncertain_pair = True
                    self.nonempty_qa = True
                save(root / "input.json", plan)
                result = self.invoke(root)
                self.assertNotEqual(result["status"], "completed")
                self.assertNotEqual(result["projects"][0]["scenarios"][0]["status"], "completed")
                for name in ("empty_qa", "nonempty_qa", "uncertain_pair"):
                    if hasattr(self, name):
                        delattr(self, name)

    def test_nonempty_graph_qa_only_route_can_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.plan(root)
            self.reject_first = False
            plan["qa_sources"] = ["graph", "external"]
            self.paired = True
            save(root / "input.json", plan)
            result = self.invoke(root)
            scene = result["projects"][0]["scenarios"][0]
            self.assertEqual(result["status"], "completed")
            self.assertEqual(scene["evaluations"]["graph"]["status"], "qa_only")

    def test_empty_graph_qa_cannot_complete_a_two_route_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.plan(root)
            self.reject_first = False
            self.paired = True
            self.empty_qa_source = "graph"
            plan["qa_sources"] = ["graph", "external"]
            save(root / "input.json", plan)
            result = self.invoke(root)
            scene = result["projects"][0]["scenarios"][0]
            self.assertNotEqual(result["status"], "completed")
            self.assertNotEqual(scene["status"], "completed")
            self.assertEqual(scene["evaluations"]["graph"]["published_qa"], 0)

    def test_default_task_workers_allow_independent_tasks_to_run_in_parallel(self):
        self.assertEqual(EVALUATION_DEFAULTS["parallel_workers"], 6)
        self.assertEqual(EVALUATION_DEFAULTS["task_workers"], 2)

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

    def test_started_qa_without_final_usage_is_not_reported_as_complete_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "qa").mkdir()
            self.assertFalse(episode_usage(root)["complete"])

    def test_interrupted_stage_still_reads_its_primary_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.plan(root)
            def interrupted(command, cwd, log):
                target = Path(command[command.index("--output") + 1])
                save(target / "private/budget.json", dict(attempts=3, prompt_tokens=60,
                                                          completion_tokens=9))
                raise KeyboardInterrupt()
            with patch("dialogue_benchmark.collection._command", side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    run_collection(root / "input.json", root / "run", root, root / ".env")
            result = read(root / "run/collection.json")
            self.assertEqual(result["usage"]["requests"], 3)
            self.assertEqual(result["usage"]["total_tokens"], 69)
            self.assertEqual(result["projects"][0]["status"], "blocked")

    def test_incomplete_usage_is_a_warning_and_does_not_stop_following_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.plan(root)
            self.reject_first = False
            self.paired = True
            self.incomplete_usage = True
            result = self.invoke(root)
            self.assertEqual(result["status"], "completed_with_warnings")
            self.assertEqual(result["projects"][0]["scenarios"][0]["status"], "completed")
            self.assertIn({"code": "usage_incomplete"}, result["warnings"])

    def test_missing_receipt_retries_the_stage_before_marking_it_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.plan(root)
            self.reject_first = False
            self.paired = True
            self.missing_receipt_attempts = 0
            result = self.invoke(root)
            dialogue_calls = [
                command for command in self.commands if Path(
                    command[command.index("--output") + 1]).name == "dialogue"
            ]
            self.assertEqual(len(dialogue_calls), 3)
            self.assertEqual(result["projects"][0]["scenarios"][0]["status"], "completed")
            self.assertEqual(result["status"], "completed")

    def command(self, command, cwd, log):
        target = Path(command[command.index("--output") + 1])
        self.commands.append(command)
        name = target.name
        budget = dict(attempts=1, prompt_tokens=10, completion_tokens=2)
        if name == "project":
            cfg = read(Path(command[command.index("--config") + 1]))
            self.assertIsNone(cfg["user"]["max_output_tokens"])
            save(target / "project.json", dict(status="completed"))
            save(target / "config.json", dict(repository=str(target), base="base-sha", tasks=[{"commit": "next-sha"}],
                                              development_plan=str(target / "development-plan.json")))
            save(target / "private/budget.json", budget)
        elif name == "scenario":
            cfg = read(Path(command[command.index("--config") + 1]))
            self.assertEqual(Path(cfg["development_plan"]).name, "development-plan.json")
            if hasattr(self, "quality_mapping"):
                self.assertEqual(cfg["dialogue_quality"], self.quality_mapping)
            save(target / "frozen/report.json", dict(status="candidate_pass", design={"memory_kinds": ["M1"]}))
            save(target / "frozen/scenario.json", {})
            save(target / "budget.json", budget)
        elif name == "requirements":
            rejected = self.reject_first and target.parent.name == "first"
            save(target / "report.json", dict(status="rejected" if rejected else "candidate_pass"))
            save(target / "budget.json", budget)
        elif name == "dialogue":
            package = target.with_name("dialogue-package")
            if hasattr(self, "missing_receipt_attempts") and self.missing_receipt_attempts == 0:
                self.missing_receipt_attempts += 1
                save(package / "private/review.json", dict(budget=budget))
                return 2
            save(package / "manifest.json", dict(schema="memory-episode-v1"))
            save(package / "external-events.json", dict(events=[{"memory_kind": "M1"}]))
            review_budget = dict(budget)
            if getattr(self, "incomplete_usage", False):
                review_budget["usage_missing"] = True
                review_budget["pending"] = {"request-1": 0}
            save(package / "private/review.json", dict(budget=review_budget))
        elif name in ("external", "graph"):
            self.assertIn("--episode-manifest", command)
            source = command[command.index("--qa-source") + 1]
            self.assertEqual(source, name)
            self.assertEqual(command[command.index("--model-request-chars") + 1], "96000")
            if source == "external":
                self.assertEqual(command[command.index("--qa-count") + 1], "8")
                self.assertNotIn("--general-count", command)
                self.assertNotIn("--code-count", command)
            else:
                self.assertEqual(command[command.index("--general-count") + 1], "50")
                self.assertEqual(command[command.index("--code-count") + 1], "50")
                self.assertNotIn("--qa-count", command)
                self.assertNotIn("--group-budget", command)
            if source == getattr(self, "fail_route", None):
                save(target / "pipeline.json", dict(status="failed"))
                save(target / "usage.json", dict(requests=1, prompt_tokens=10, completion_tokens=2, complete=True))
                return 1
            paired = getattr(self, "paired", False)
            qa_only = "--qa-only" in command
            nonempty_qa = paired or qa_only or getattr(self, "nonempty_qa", False)
            save(target / "pipeline.json", dict(status="completed", **(
                {"stop_reason": "qa_only"} if qa_only else {} if paired else {"stop_reason": "no_eligible_qa"})))
            empty_qa = (getattr(self, "empty_qa", False)
                        or getattr(self, "empty_qa_source", None) == source)
            save(target / "qa/qa-public.json", dict(questions=[{"id": "q1"}] if nonempty_qa and not empty_qa else []))
            if not qa_only:
                if getattr(self, "uncertain_pair", False):
                    tasks = [dict(task="task-01", status="evaluated",
                                  comparison={"without_memory": {"result": "failed", "trial": "trial-1"},
                                              "with_memory": {"result": "uncertain", "trial": "trial-2"}})]
                else:
                    tasks = [dict(task="task-01", status="evaluated",
                                  comparison={"without_memory": {"result": "failed", "metrics": {}, "trial": "trial-1"},
                                              "with_memory": {"result": "passed", "metrics": {}, "trial": "trial-2"}})] if paired else []
                save(target / "tasks/manifest.json", dict(tasks=tasks))
            save(target / "usage.json", dict(requests=1, prompt_tokens=10, completion_tokens=2, complete=True))
        else:
            self.fail("Unexpected stage")
        return 0

    def invoke(self, root):
        self.commands = []
        with patch("dialogue_benchmark.collection._command", side_effect=self.command), \
             patch("dialogue_benchmark.collection.subprocess.check_output", return_value="root-sha\n"), \
             patch("dialogue_benchmark.episode_input.load_episode_manifest", side_effect=lambda p: {
                 "external_events": p.parent / "external-events.json"} if not getattr(self, "empty", False) else {}):
            return run_collection(root / "input.json", root / "run", root, root / ".env")

    def test_relative_python_is_resolved_before_stage_changes_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.plan(root)
            commands = []
            def capture_and_stop(command, cwd, log):
                commands.append(command)
                raise KeyboardInterrupt()

            with patch("dialogue_benchmark.collection._command",
                       side_effect=capture_and_stop), \
                 patch("dialogue_benchmark.collection.subprocess.check_output",
                       return_value="root-sha\n"), \
                 patch("dialogue_benchmark.episode_input.load_episode_manifest",
                       side_effect=lambda p: {"external_events": p.parent / "external-events.json"}):
                with self.assertRaises(KeyboardInterrupt):
                    run_collection(root / "input.json", root / "run", root, root / ".env",
                                   python=Path(".venv/bin/python"))
            self.assertTrue(Path(commands[0][0]).is_absolute())
            self.assertTrue(commands[0][0].endswith("/.venv/bin/python"))

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
            self.assertEqual(project["scenarios"][1]["evaluations"]["external"]["paired_tasks"], 0)
            for scenario in ("first", "second"):
                self.assertEqual(read(root / "run/planner" / scenario / "config.json")["base"], "base-sha")
                self.assertEqual(read(root / "run/planner" / scenario / "config.json")["development_plan"],
                                 str((root / "run/planner/project/development-plan.json").resolve()))

    def test_budget_stops_new_stages_without_retrying_or_dropping_prior_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.plan(root, budget=2)
            self.reject_first = False
            with self.assertRaisesRegex(RuntimeError, "collection_budget_exhausted"):
                self.invoke(root)
            result = read(root / "run/collection.json")
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(len(self.commands), 2)
            self.assertEqual(result["usage"]["requests"], 2)
            self.assertTrue((root / "run/planner/project/config.json").is_file())

    def test_reused_project_keeps_lineage_but_uses_current_runtime_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.plan(root)
            plan["projects"][0].pop("brief")
            plan["projects"][0]["prepared_config"] = "prepared.json"
            save(root / "input.json", plan)
            save(root / "prepared.json", dict(repository=str(root), base="base-sha",
                tasks=[{"commit": "next-sha"}], development_plan="development-plan.json",
                judge=dict(model="old", reasoning_effort="low", candidate_pythonpath="/workspace/candidate")))
            save(root / "runtime.json", dict(
                user=dict(model="current-user", reasoning_effort="max"),
                judge=dict(model="current-judge", reasoning_effort="max", max_output_tokens=12)))
            self.reject_first = False
            result = self.invoke(root)
            self.assertFalse(any("simulator.openhands.prepare_project" in c for c in self.commands))
            for scenario in ("first", "second"):
                config = read(root / "run/planner" / scenario / "config.json")
                self.assertEqual(config["base"], "base-sha")
                self.assertEqual(config["judge"]["model"], "current-judge")
                self.assertEqual(config["judge"]["reasoning_effort"], "max")
                self.assertEqual(config["judge"]["candidate_pythonpath"], "/workspace/candidate")
                self.assertIsNone(config["judge"]["max_output_tokens"])
            self.assertEqual(result["projects"][0]["lineage"], ["root-sha"])

    def test_pairs_are_aggregated_with_distinct_scenario_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.plan(root)
            self.reject_first = False
            self.paired = True
            result = self.invoke(root)
            scenes = result["projects"][0]["scenarios"]
            self.assertEqual(sum(s["evaluations"]["external"]["paired_tasks"] for s in scenes), 2)
            self.assertEqual(result["usage"]["requests"], 9)
            report = (root / "run/report.md").read_text()
            self.assertIn("planner/first/evaluation/external/tasks/task-01", report)
            self.assertIn("planner/second/evaluation/external/tasks/task-01", report)
            self.assertEqual(scenes[0]["public_memory_counts"]["M1"], 1)

    def test_duplicate_or_escaping_targets_are_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.plan(Path(directory))
            plan["projects"][0]["id"] = "../project"
            with self.assertRaises(ValueError):
                validate_plan(plan)

    def test_qa_only_is_forwarded_without_creating_an_empty_pair_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.plan(root)
            plan["evaluation"] = {"qa_only": True}
            save(root / "input.json", plan)
            self.reject_first = False
            result = self.invoke(root)
            evaluation = [c for c in self.commands if "--episode-manifest" in c]
            self.assertEqual(len(evaluation), 2)
            self.assertTrue(all("--qa-only" in c for c in evaluation))
            self.assertTrue(all(c[c.index("--qa-only") + 1] != "True" for c in evaluation))
            self.assertTrue(result["qa_only"])
            self.assertEqual([s["evaluations"]["external"]["published_qa"]
                              for s in result["projects"][0]["scenarios"]], [1, 1])
            self.assertEqual(result["projects"][0]["status"], "completed")
            self.assertEqual([s["status"] for s in result["projects"][0]["scenarios"]],
                             ["qa_only", "qa_only"])
            self.assertFalse(any((root / "run").glob("planner/*/evaluation/tasks")))
            report = (root / "run/collection.md").read_text()
            self.assertIn("Not scheduled", report)
            self.assertNotIn("(report.md)", report)
            self.assertFalse((root / "run/report.md").exists())

    def test_qa_only_requires_an_actual_boolean(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.plan(Path(directory))
            for value in (True, False):
                plan["evaluation"] = {"qa_only": value}
                validate_plan(plan)
            for value in ("true", 1, None):
                plan["evaluation"] = {"qa_only": value}
                with self.assertRaisesRegex(ValueError, "qa_only must be a boolean"):
                    validate_plan(plan)

    def test_valid_empty_history_skips_evaluation_and_continues(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.plan(root)
            self.reject_first = False
            self.empty = True
            result = self.invoke(root)
            self.assertEqual(result["usage"]["requests"], 7)
            self.assertEqual([s["status"] for s in result["projects"][0]["scenarios"]],
                             ["no_external_history", "no_external_history"])
            self.assertFalse(any("run_episode.py" in " ".join(c) for c in self.commands))

    def test_both_routes_share_one_dialogue_but_keep_separate_qa_and_task_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.plan(root)
            plan["qa_sources"] = ["graph", "external"]
            save(root / "input.json", plan)
            self.reject_first = False
            self.paired = True
            result = self.invoke(root)
            dialogue_commands = [c for c in self.commands if "simulator" in c]
            self.assertEqual(len(dialogue_commands), 2)
            evaluations = [c for c in self.commands if "--episode-manifest" in c]
            self.assertEqual(len(evaluations), 4)
            for index in (0, 2):
                graph, external = evaluations[index:index + 2]
                self.assertEqual(graph[graph.index("--episode-manifest") + 1],
                                 external[external.index("--episode-manifest") + 1])
                self.assertIn("--qa-only", graph)
                self.assertNotIn("--qa-only", external)
            for scene in result["projects"][0]["scenarios"]:
                self.assertEqual(set(scene["evaluations"]), {"graph", "external"})
                self.assertEqual(scene["evaluations"]["graph"]["published_qa"], 1)
                self.assertEqual(scene["evaluations"]["graph"]["paired_tasks"], 0)
                self.assertEqual(scene["evaluations"]["external"]["paired_tasks"], 1)
                self.assertFalse((root / "run/planner" / scene["id"] / "evaluation/graph/tasks").exists())
            self.assertEqual(result["usage"]["requests"], 11)
            report = (root / "run/collection.md").read_text()
            self.assertIn("| graph |", report)
            self.assertIn("| external |", report)

    def test_graph_route_runs_even_without_external_sidecar_events(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.plan(root)
            plan["qa_sources"] = ["external", "graph"]
            save(root / "input.json", plan)
            self.reject_first = False
            self.empty = True
            result = self.invoke(root)
            evaluated = [c for c in self.commands if "--episode-manifest" in c]
            self.assertEqual(len(evaluated), 2)
            self.assertTrue(all(c[c.index("--qa-source") + 1] == "graph" for c in evaluated))
            for scene in result["projects"][0]["scenarios"]:
                self.assertEqual(scene["evaluations"]["external"]["status"], "no_external_history")
                self.assertEqual(scene["evaluations"]["graph"]["status"], "qa_only")

    def test_failure_in_one_route_does_not_suppress_the_other(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.plan(root)
            plan["qa_sources"] = ["external", "graph"]
            save(root / "input.json", plan)
            self.reject_first = False
            self.fail_route = "external"
            result = self.invoke(root)
            for scene in result["projects"][0]["scenarios"]:
                self.assertEqual(scene["status"], "partial_failure")
                self.assertEqual(scene["evaluations"]["external"]["status"], "evaluation_failed")
                self.assertEqual(scene["evaluations"]["graph"]["status"], "qa_only")
            self.assertEqual(result["projects"][0]["status"], "partial_failure")
            self.assertEqual(result["status"], "completed_with_warnings")

    def test_invalid_route_selection_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.plan(Path(directory))
            for sources in ([], "graph", ["unknown"], ["graph", "graph"]):
                with self.subTest(sources=sources):
                    plan["qa_sources"] = sources
                    with self.assertRaisesRegex(ValueError, "qa_sources"):
                        validate_plan(plan)
