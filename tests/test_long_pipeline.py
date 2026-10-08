import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from dialogue_benchmark.long_pipeline import (_qa_stage, _route_command, _task_stage,
                                               build_config)
from dialogue_benchmark.pipeline_runner import PipelineRunner
from dialogue_benchmark.task_eval.artifacts import save


class LongPipelineTests(unittest.TestCase):
    def test_route_command_forwards_parallel_workers(self):
        args = SimpleNamespace(
            python=Path(os.sys.executable), simulator_path=Path("sim"), env_file=Path(".env"),
        )
        plan = {"evaluation": {"parallel_workers": 7}}
        for route in ("graph", "external"):
            command = _route_command(route, Path("manifest.json"), Path("output"), args, plan)
            index = command.index("--parallel-workers")
            self.assertEqual(command[index + 1], "7")

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

            with patch("dialogue_benchmark.long_pipeline._run_logged", side_effect=fake_run) as run:
                self.assertEqual(_task_stage(args), 0)
            self.assertIn("--resume-tasks", run.call_args.args[0])
            self.assertFalse(stale.exists())
            self.assertTrue(any(path.name.startswith("tasks")
                                for path in (output / ".incomplete").iterdir()))

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
