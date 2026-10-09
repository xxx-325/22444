import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from dialogue_benchmark.long_pipeline import (_manifest_from_collection, _qa_stage,
                                               _route_command, _task_stage, _write_receipt, build_config)
from dialogue_benchmark.pipeline_runner import PipelineRunner
from dialogue_benchmark.task_eval.artifacts import save


class LongPipelineTests(unittest.TestCase):
    def test_stage_receipt_hashes_input_and_all_outputs_without_patches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.json"
            source.write_text("input", encoding="utf-8")
            artifacts = [root / "output/one.json", root / "output/two.json"]
            for artifact in artifacts:
                save(artifact, {"name": artifact.name})
            receipt_path = root / "output/stage-receipt.json"
            _write_receipt(receipt_path, artifacts, input_path=source,
                           result="completed_with_warnings", warnings=["qa_shortfall"],
                           retryable=False)
            receipt = json.loads(receipt_path.read_text())
            self.assertEqual(receipt["input_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(receipt["warnings"], ["qa_shortfall"])
            self.assertFalse(receipt["retryable"])
            previous_hash = receipt["output_sha256"]
            save(artifacts[1], {"name": "changed"})
            _write_receipt(receipt_path, artifacts, input_path=source)
            self.assertNotEqual(json.loads(receipt_path.read_text())["output_sha256"], previous_hash)

    def save_qa(self, directory, *, route="external", provisional=False, empty=False,
                public_status=None):
        questions = [] if empty else [{
            "id": "q1", "type": "M1" if route == "external" else "constraint_followthrough",
            "status": "needs_review" if provisional else "approved",
            "question": "Which agreement applies to the customer export?",
            "answer_points": [{"text": "Preserve explicitly empty fields.", "sources": ["e1"]}],
        }]
        save(directory / "manifest.json", {"qa_source": route})
        save(directory / "qa-public.json", {"status": public_status,
                                             "questions": [] if provisional else questions})
        save(directory / "qa-candidates.json", {"questions": questions})
        save(directory / "stages/group-raw-candidates.json", {"questions": questions})
        save(directory / "stages/group-qa-input.json", {
            "payload": {"scope": {"dialogue": [{"id": "e1", "text": "Preserve explicitly empty fields."}]}}})

    def test_manifest_uses_exported_complete_round_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "dialogue-package/manifest.json"
            save(manifest, {"placeholder": True})
            loaded = {"manifest": {"quality": {
                "passed": True, "profile": {"min_user_code_rounds": 30},
                "checks": {"complete_user_code_rounds": {"actual": 30}},
            }}}
            with patch("dialogue_benchmark.long_pipeline.load_episode_manifest", return_value=loaded):
                self.assertEqual(_manifest_from_collection(root), (manifest, []))

    def test_route_command_forwards_parallel_workers(self):
        args = SimpleNamespace(
            python=Path(os.sys.executable), simulator_path=Path("sim"), env_file=Path(".env"),
        )
        plan = {"evaluation": {"parallel_workers": 7}}
        for route in ("graph", "external"):
            command = _route_command(route, Path("manifest.json"), Path("output"), args, plan)
            index = command.index("--parallel-workers")
            self.assertEqual(command[index + 1], "7")

    def test_qa_stage_uses_configured_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "repo"
            output = root / "qa"
            input_root.mkdir()
            manifest = root / "episode.json"
            plan = root / "plan.json"
            save(manifest, {"placeholder": True})
            save(plan, {"qa_sources": ["graph"], "evaluation": {"parallel_workers": 4}})
            save(input_root / "manifest-path.json", {"manifest": str(manifest), "plan": str(plan)})
            calls = []

            def fake_run(command, cwd, stdout, stderr):
                calls.append(command)
                route_dir = Path(command[command.index("--output") + 1])
                save(route_dir / "pipeline.json", {"status": "completed"})
                save(route_dir / "qa/manifest.json", {"input_sha256": "test"})
                save(route_dir / "qa/qa-public.json", {"questions": []})
                return 0

            args = SimpleNamespace(
                input=input_root, output=output, python=Path(os.sys.executable),
                simulator_path=root / "sim", env_file=root / ".env",
            )
            with patch("dialogue_benchmark.long_pipeline._run_logged", side_effect=fake_run):
                self.assertEqual(_qa_stage(args), 0)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][calls[0].index("--qa-source") + 1], "graph")
            summary = json.loads((output / "qa-summary.json").read_text())
            self.assertEqual(set(summary["routes"]), {"graph"})

    def test_qa_stage_archives_incomplete_route_before_fresh_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "repo"
            output = root / "qa"
            input_root.mkdir()
            manifest = root / "episode.json"
            plan = root / "plan.json"
            save(manifest, {"placeholder": True})
            save(plan, {"evaluation": {"parallel_workers": 4}})
            save(input_root / "manifest-path.json", {"manifest": str(manifest), "plan": str(plan)})
            stale = output / "graph/qa/stale.txt"
            stale.parent.mkdir(parents=True)
            stale.write_text("partial", encoding="utf-8")
            args = SimpleNamespace(
                input=input_root, output=output, python=Path(os.sys.executable),
                simulator_path=root / "sim", env_file=root / ".env",
            )

            def fake_run(command, cwd, stdout, stderr):
                route_dir = Path(command[command.index("--output") + 1])
                save(route_dir / "pipeline.json", {"status": "completed"})
                save(route_dir / "qa/manifest.json", {"input_sha256": "test"})
                save(route_dir / "qa/qa-public.json", {
                    "questions": [{"status": "approved"}],
                })
                return 0

            with patch("dialogue_benchmark.long_pipeline._run_logged", side_effect=fake_run) as run:
                self.assertEqual(_qa_stage(args), 0)
            graph_command = next(call.args[0] for call in run.call_args_list
                                 if call.args[0][call.args[0].index("--qa-source") + 1] == "graph")
            self.assertNotIn("--resume-tasks", graph_command)
            self.assertFalse(stale.exists())
            self.assertTrue(any(path.name.startswith("qa")
                                for path in (output / "graph/.incomplete").iterdir()))

    def test_qa_stage_keeps_partial_batch_as_retryable_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "repo"
            output = root / "qa"
            input_root.mkdir()
            manifest = root / "episode.json"
            plan = root / "plan.json"
            save(manifest, {"placeholder": True})
            save(plan, {"qa_sources": ["graph"], "evaluation": {"parallel_workers": 4}})
            save(input_root / "manifest-path.json", {"manifest": str(manifest), "plan": str(plan)})
            save(output / "graph/qa/batch-001.json", {"batch": 1})
            args = SimpleNamespace(
                input=input_root, output=output, python=Path(os.sys.executable),
                simulator_path=root / "sim", env_file=root / ".env",
            )
            with patch("dialogue_benchmark.long_pipeline._run_logged") as run:
                self.assertEqual(_qa_stage(args), 1)
            run.assert_not_called()
            self.assertTrue((output / "graph/qa/batch-001.json").is_file())
            self.assertEqual(json.loads((output / "graph/pipeline.json").read_text())["status"],
                             "retryable")

    def test_task_stage_archives_tasks_without_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            input_root = root / "qa"
            external = input_root / "external"
            output = root / "task"
            repo.mkdir(parents=True)
            save(repo / "manifest-path.json", {"manifest": str(root / "episode.json"),
                                                "plan": str(root / "plan.json")})
            save(root / "episode.json", {"placeholder": True})
            save(root / "plan.json", {"evaluation": {"parallel_workers": 4}})
            save(external / "pipeline.json", {"status": "completed"})
            save(external / "qa/qa-public.json", {"questions": [{"status": "approved"}]})
            save(external / "qa/manifest.json", {"input_sha256": "test"})
            stale = output / "tasks/partial.json"
            stale.parent.mkdir(parents=True)
            stale.write_text("partial", encoding="utf-8")
            args = SimpleNamespace(
                input=input_root, output=output, python=Path(os.sys.executable),
                simulator_path=root / "sim", env_file=root / ".env",
            )

            def fake_run(command, cwd, stdout, stderr):
                save(output / "tasks/manifest.json", {"status": "completed", "tasks": []})
                return 0

            with patch("dialogue_benchmark.long_pipeline.qa_inputs",
                       return_value=[{"qa": {"id": "q1", "status": "approved"}}]), \
                 patch("dialogue_benchmark.long_pipeline._run_logged", side_effect=fake_run) as run:
                self.assertEqual(_task_stage(args), 0)
            self.assertIn("--resume-tasks", run.call_args.args[0])
            self.assertFalse(stale.exists())
            self.assertTrue(any(path.name.startswith("tasks")
                                for path in (output / ".incomplete").iterdir()))

    def test_task_stage_continues_with_provisional_external_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            input_root = root / "qa"
            external = input_root / "external"
            output = root / "task"
            repo.mkdir(parents=True)
            save(repo / "manifest-path.json", {"manifest": str(root / "episode.json"),
                                                "plan": str(root / "plan.json")})
            save(root / "episode.json", {"placeholder": True})
            save(root / "plan.json", {"evaluation": {"qa_count": 12}})
            save(external / "pipeline.json", {"status": "completed_with_warnings"})
            self.save_qa(external / "qa", provisional=True)
            args = SimpleNamespace(
                input=input_root, output=output, python=Path(os.sys.executable),
                simulator_path=root / "sim", env_file=root / ".env",
            )

            def fake_run(command, cwd, stdout, stderr):
                self.assertIn("--allow-provisional", command)
                save(output / "tasks/manifest.json", {"status": "completed", "tasks": []})
                return 0

            with patch("dialogue_benchmark.long_pipeline._run_logged", side_effect=fake_run):
                self.assertEqual(_task_stage(args), 0)

    def test_failed_qa_aggregate_keeps_source_validated_approved_task_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save(root / "repo/manifest-path.json", {"manifest": str(root / "episode.json"),
                                                     "plan": str(root / "plan.json")})
            save(root / "plan.json", {"evaluation": {"qa_count": 1, "general_count": 1,
                                                       "code_count": 0, "task_count": 1}})
            args = SimpleNamespace(input=root / "repo", output=root / "qa", python=Path(os.sys.executable),
                                   simulator_path=root / "sim", env_file=root / ".env")
            def fake_qa(command, cwd, stdout, stderr):
                target = Path(command[command.index("--output") + 1])
                route = command[command.index("--qa-source") + 1]
                save(target / "pipeline.json", {"status": "completed_with_warnings"})
                self.save_qa(target / "qa", route=route, public_status="failed")
                return 0
            with patch("dialogue_benchmark.long_pipeline._run_logged", side_effect=fake_qa):
                self.assertEqual(_qa_stage(args), 0)
            receipt = json.loads((root / "qa/stage-receipt.json").read_text())
            self.assertEqual(receipt["result"], "completed_with_warnings")
            args.input, args.output = root / "qa", root / "task"
            def fake_tasks(command, cwd, stdout, stderr):
                save(root / "task/tasks/manifest.json", {"status": "complete", "completed": 1,
                                                          "tasks": []})
                return 0
            with patch("dialogue_benchmark.long_pipeline._run_logged", side_effect=fake_tasks) as run:
                self.assertEqual(_task_stage(args), 0)
            run.assert_called_once()

    def test_empty_qa_outputs_produce_warning_and_empty_requirement_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save(root / "repo/manifest-path.json", {"manifest": str(root / "episode.json"),
                                                     "plan": str(root / "plan.json")})
            save(root / "plan.json", {"evaluation": {"task_count": 3}})
            args = SimpleNamespace(input=root / "repo", output=root / "qa", python=Path(os.sys.executable),
                                   simulator_path=root / "sim", env_file=root / ".env")
            calls = []
            def fake_run(command, cwd, stdout, stderr):
                calls.append(command)
                target = Path(command[command.index("--output") + 1])
                route = command[command.index("--qa-source") + 1]
                save(target / "pipeline.json", {"status": "completed_with_warnings"})
                self.save_qa(target / "qa", route=route, empty=True, public_status="failed")
                return 0
            with patch("dialogue_benchmark.long_pipeline._run_logged", side_effect=fake_run):
                self.assertEqual(_qa_stage(args), 0)
                self.assertEqual(_qa_stage(args), 0)
            self.assertEqual(len(calls), 2)
            receipt = json.loads((root / "qa/stage-receipt.json").read_text())
            self.assertEqual(receipt["result"], "completed_with_warnings")
            self.assertTrue(receipt["input_sha256"])
            self.assertTrue(receipt["output_sha256"])
            self.assertIn("external_not_usable_for_tasks", receipt["warnings"])
            self.assertFalse(receipt["retryable"])
            summary = json.loads((root / "qa/qa-summary.json").read_text())
            self.assertTrue(summary["routes"]["graph"]["output_ready"])
            self.assertFalse(summary["routes"]["graph"]["usable_for_tasks"])
            self.assertTrue(summary["routes"]["external"]["output_ready"])
            self.assertFalse(summary["routes"]["external"]["usable_for_tasks"])
            args.input, args.output = root / "qa", root / "task"
            with patch("dialogue_benchmark.long_pipeline._run_logged") as run:
                self.assertEqual(_task_stage(args), 0)
            run.assert_not_called()
            summary = json.loads((root / "task/task-stage.json").read_text())
            self.assertEqual(summary["completed"], 0)
            self.assertEqual(summary["shortfall"], 3)
            self.assertTrue((root / "task/tasks/report.md").is_file())
            self.assertTrue((root / "task/tasks/report.html").is_file())

    def test_task_stage_honors_qa_usable_for_tasks_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            input_root = root / "qa"
            external = input_root / "external"
            output = root / "task"
            repo.mkdir(parents=True)
            save(repo / "manifest-path.json", {"manifest": str(root / "episode.json"),
                                                "plan": str(root / "plan.json")})
            save(root / "episode.json", {"placeholder": True})
            save(root / "plan.json", {"evaluation": {"task_count": 1}})
            save(external / "pipeline.json", {"status": "completed"})
            self.save_qa(external / "qa")
            save(input_root / "qa-summary.json", {"routes": {
                "external": {"output_ready": True, "usable_for_tasks": False}
            }})
            args = SimpleNamespace(input=input_root, output=output)
            with patch("dialogue_benchmark.long_pipeline._run_logged") as run:
                self.assertEqual(_task_stage(args), 0)
            run.assert_not_called()
            summary = json.loads((output / "task-stage.json").read_text())
            self.assertEqual(summary["reason"], "external_qa_not_qualified")

    def test_unavailable_qa_preserves_existing_requirements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save(root / "repo/manifest-path.json", {"manifest": str(root / "episode.json"),
                                                     "plan": str(root / "plan.json")})
            save(root / "plan.json", {"evaluation": {"task_count": 3}})
            self.save_qa(root / "qa/external/qa", empty=True)
            save(root / "qa/external/pipeline.json", {"status": "completed"})
            existing = {"status": "complete", "completed": 1,
                        "tasks": [{"task": "task-01", "status": "evaluated"}]}
            path = root / "task/tasks/manifest.json"
            save(path, existing)
            before = path.read_bytes()
            args = SimpleNamespace(input=root / "qa", output=root / "task")
            self.assertEqual(_task_stage(args), 0)
            self.assertEqual(path.read_bytes(), before)
            summary = json.loads((root / "task/task-stage.json").read_text())
            self.assertEqual(summary["completed"], 1)
            self.assertEqual(summary["shortfall"], 2)

    def test_unavailable_qa_records_shortfall_for_existing_empty_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save(root / "repo/manifest-path.json", {"manifest": str(root / "episode.json"),
                                                     "plan": str(root / "plan.json")})
            save(root / "plan.json", {"evaluation": {"task_count": 3}})
            self.save_qa(root / "qa/external/qa", empty=True)
            save(root / "qa/external/pipeline.json", {"status": "completed_with_warnings"})
            path = root / "task/tasks/manifest.json"
            save(path, {"status": "skipped", "tasks": []})
            args = SimpleNamespace(input=root / "qa", output=root / "task")
            self.assertEqual(_task_stage(args), 0)
            manifest = json.loads(path.read_text())
            self.assertEqual(manifest["status"], "completed_with_warnings")
            self.assertEqual(manifest["completed"], 0)
            self.assertEqual(manifest["shortfall"], 3)

    def test_three_case_warning_dry_run_continues_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "plan.json"
            save(source, {"evaluation": {"task_count": 3}})
            config = {
                "cases": [{"id": name, "source": str(source)}
                          for name in ("support", "migration", "release")],
                "stages": {stage: {"command": ["fake", stage],
                                   "receipt": "{output}/stage-receipt.json"}
                           for stage in ("repo", "qa", "task")},
            }
            calls = []
            route_calls = []
            def fake_route(command, cwd, stdout, stderr):
                target = Path(command[command.index("--output") + 1])
                route = command[command.index("--qa-source") + 1]
                name = next(name for name in ("support", "migration", "release") if name in target.parts)
                route_calls.append((name, route))
                if name == "release" or (name == "migration" and route == "graph"):
                    save(target / "pipeline.json", {"status": "failed"})
                    return 1
                save(target / "pipeline.json", {"status": "completed_with_warnings"})
                self.save_qa(target / "qa", route=route, empty=True, public_status="failed")
                return 0
            def fake_worker(command, cwd, env, stdout, stderr):
                name, stage = env["PIPELINE_CASE_ID"], env["PIPELINE_STAGE"]
                calls.append((name, stage))
                output = Path(env["PIPELINE_OUTPUT"])
                output.mkdir(parents=True, exist_ok=True)
                if stage == "repo":
                    info = output / "manifest-path.json"
                    save(info, {"manifest": str(root / "episode.json"), "plan": str(source)})
                    quality = output / "quality-summary.json"
                    save(quality, {"status": "completed_with_warnings", "warnings": ["external_events_shortfall"],
                                   "checks": {"external_event_closure": {"passed": True}}})
                    _write_receipt(output / "stage-receipt.json", [info, quality],
                                   result="completed_with_warnings")
                    return
                args = SimpleNamespace(input=Path(env["PIPELINE_INPUT"]), output=output,
                                       python=Path(os.sys.executable), simulator_path=root / "sim",
                                       env_file=root / ".env")
                code = (_qa_stage if stage == "qa" else _task_stage)(args)
                if code:
                    raise RuntimeError("Local QA route failed")
            with patch("dialogue_benchmark.long_pipeline._run_logged", side_effect=fake_route):
                runner = PipelineRunner(config, root / "run", command_runner=fake_worker, max_attempts=1)
                report = runner.run()
                self.assertEqual(report["status"], "completed_with_warnings")
                state = json.loads((root / "run/pipeline-state.json").read_text())
                for name in ("support", "migration"):
                    self.assertEqual(state["cases"][name]["stages"]["task"]["status"], "completed")
                    self.assertTrue((root / "run/cases" / name / "task/tasks/report.md").is_file())
                self.assertNotEqual(state["cases"]["release"]["stages"]["qa"]["status"], "completed")
                summary = json.loads((root / "run/cases/migration/qa/qa-summary.json").read_text())
                self.assertFalse(summary["routes"]["graph"]["output_ready"])
                self.assertTrue(summary["routes"]["external"]["output_ready"])
                self.assertEqual(len(route_calls), 6)
                completed_calls = [call for call in calls if call[0] != "release"]
                calls.clear()
                route_calls.clear()
                PipelineRunner(config, root / "run", resume=True, command_runner=fake_worker,
                               max_attempts=1).run()
                self.assertEqual([call for call in calls if call[0] != "release"], [])
                self.assertEqual(route_calls, [])
                self.assertEqual(len(completed_calls), 6)

    def test_config_expands_three_absolute_inputs(self):
        source = Path(__file__).parents[1] / "examples/collection-five/long-dialogue-three.json"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            config = build_config(source, output, simulator_path=output / "sim",
                                  env_file=output / ".env", python=Path(os.sys.executable))
            self.assertEqual([case["id"] for case in config["cases"]],
                             ["release-operations", "document-pipeline", "dependency-planner"])
            for case in config["cases"]:
                self.assertTrue(Path(case["source"]).is_absolute())
                plan = json.loads(Path(case["source"]).read_text())
                prepared = Path(plan["projects"][0]["prepared_config"])
                self.assertTrue(prepared.is_absolute())
                self.assertEqual(prepared.name, "config.json")

    def test_config_builds_fresh_business_cases_from_briefs(self):
        source = Path(__file__).parents[1] / "examples/collection-three-business.json"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            config = build_config(source, output, simulator_path=output / "sim",
                                  env_file=output / ".env", python=Path(os.sys.executable))
            self.assertEqual(
                [case["id"] for case in config["cases"]],
                [
                    "customer-service-platform",
                    "data-migration-reconciliation",
                    "release-change-orchestrator",
                ],
            )
            for case in config["cases"]:
                plan = json.loads(Path(case["source"]).read_text(encoding="utf-8"))
                project = plan["projects"][0]
                self.assertIn("brief", project)
                self.assertNotIn("prepared_config", project)
                self.assertEqual(project["increments"], 0)
                task_command = case["stages"]["task"]["command"]
                self.assertEqual(
                    task_command[task_command.index("--task-slot-directory") + 1],
                    str((output / ".task-slots").resolve()),
                )
                self.assertEqual(task_command[task_command.index("--task-slots") + 1], "3")
                repo_command = case["stages"]["repo"]["command"]
                self.assertEqual(repo_command[repo_command.index("--task-slot-directory") + 1],
                                 task_command[task_command.index("--task-slot-directory") + 1])
                self.assertEqual(repo_command[repo_command.index("--task-slots") + 1], "3")

    def test_real_subprocess_receipts_gate_stage_completion_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "worker.py"
            script.write_text(
                "import hashlib, json, os, pathlib\n"
                "out=pathlib.Path(os.environ['PIPELINE_OUTPUT']); out.mkdir(parents=True, exist_ok=True)\n"
                "artifact=out/'artifact.txt'; artifact.write_text(os.environ['PIPELINE_STAGE'])\n"
                "sha=hashlib.sha256(artifact.read_bytes()).hexdigest()\n"
                "(out/'receipt.json').write_text(json.dumps({'status':'completed','sha256':sha,'artifacts':[{'path':'artifact.txt','sha256':sha}]}))\n",
                encoding="utf-8")
            command = [os.sys.executable, str(script)]
            config = {"cases": [{"id": "case", "source": str(root / "source.json")}],
                      "stages": {stage: {"command": command, "receipt": "{output}/receipt.json"}
                                 for stage in ("repo", "qa", "task")}}
            (root / "source.json").write_text("{}", encoding="utf-8")
            report = PipelineRunner(config, root / "run").run()
            self.assertEqual(report["status"], "completed")
            self.assertEqual((root / "run/cases/case/task/artifact.txt").read_text(), "task")
            attempts = json.loads((root / "run/pipeline-state.json").read_text())
            self.assertEqual(attempts["cases"]["case"]["stages"]["repo"]["attempts"], 1)
            resumed = PipelineRunner(config, root / "run", resume=True).run()
            self.assertEqual(resumed["status"], "completed")
            state = json.loads((root / "run/pipeline-state.json").read_text())
            self.assertEqual(state["cases"]["case"]["stages"]["repo"]["attempts"], 1)


if __name__ == "__main__":
    unittest.main()
