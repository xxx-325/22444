import json
import os
from pathlib import Path
import stat
import tempfile
import unittest

from dialogue_benchmark.long_pipeline import build_config
from dialogue_benchmark.pipeline_runner import PipelineRunner


class LongPipelineTests(unittest.TestCase):
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
