import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from dialogue_benchmark.pipeline_runner import PipelineRunner, StageCommandError


def _config(case_ids):
    command = ["stage-worker", "{stage}", "{case_id}", "{input}", "{output}"]
    return {
        "stages": {stage: {"command": command} for stage in ("repo", "qa", "task")},
        "cases": [{"id": case_id, "source": "source/%s" % case_id}
                  for case_id in case_ids],
    }


class PipelineRunnerTests(unittest.TestCase):
    def test_three_stage_workers_pipeline_cases_and_write_handoffs(self):
        calls = []
        lock = threading.Lock()
        active = 0
        maximum = 0

        def run(command, cwd, env, stdout, stderr):
            nonlocal active, maximum
            stage, case_id = command[1], command[2]
            with lock:
                active += 1
                maximum = max(maximum, active)
                calls.append((stage, case_id))
            time.sleep(0.05)
            Path(env["PIPELINE_OUTPUT"]).mkdir(parents=True, exist_ok=True)
            Path(env["PIPELINE_OUTPUT"], "artifact.txt").write_text(
                "%s:%s" % (stage, case_id), encoding="utf-8")
            stdout.write_text("ok", encoding="utf-8")
            stderr.write_text("", encoding="utf-8")
            with lock:
                active -= 1

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = PipelineRunner(
                _config(("a", "b", "c")), root / "run",
                command_runner=run,
                poll_interval=0.001,
            ).run()
            self.assertEqual(report["status"], "completed")
            self.assertEqual(maximum, 3)
            for case_id in ("a", "b", "c"):
                states = report["cases"][case_id]["stages"]
                self.assertTrue(all(item["status"] == "completed"
                                    for item in states.values()))
                for stage in ("repo", "qa", "task"):
                    handoff = root / "run" / "cases" / case_id / stage / ".pipeline-handoff.json"
                    self.assertTrue(handoff.exists())
            for case_id in ("a", "b", "c"):
                positions = {stage: calls.index((stage, case_id))
                             for stage in ("repo", "qa", "task")}
                self.assertLess(positions["repo"], positions["qa"])
                self.assertLess(positions["qa"], positions["task"])

    def test_resume_does_not_repeat_completed_work(self):
        calls = []

        def run(command, cwd, env, stdout, stderr):
            calls.append(tuple(command[:3]))
            Path(env["PIPELINE_OUTPUT"]).mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            PipelineRunner(_config(("a",)), output, command_runner=run).run()
            first = list(calls)
            PipelineRunner(_config(("a",)), output, resume=True,
                           command_runner=run).run()
            self.assertEqual(calls, first)
            state = json.loads((output / "pipeline-state.json").read_text())
            self.assertEqual(state["cases"]["a"]["stages"]["repo"]["attempts"], 1)

    def test_one_failed_case_is_skipped_downstream(self):
        calls = []

        def run(command, cwd, env, stdout, stderr):
            stage, case_id = command[1], command[2]
            calls.append((stage, case_id))
            if case_id == "bad" and stage == "repo":
                raise StageCommandError("fixture failure")
            Path(env["PIPELINE_OUTPUT"]).mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as directory:
            report = PipelineRunner(
                _config(("bad", "good")), Path(directory) / "run",
                max_attempts=1, command_runner=run,
            ).run()
        bad = report["cases"]["bad"]["stages"]
        good = report["cases"]["good"]["stages"]
        self.assertEqual(report["status"], "completed_with_warnings")
        self.assertEqual(bad["repo"]["status"], "failed")
        self.assertEqual(bad["qa"]["status"], "skipped")
        self.assertEqual(bad["task"]["status"], "skipped")
        self.assertTrue(all(good[stage]["status"] == "completed"
                            for stage in ("repo", "qa", "task")))
        self.assertNotIn(("qa", "bad"), calls)
        self.assertNotIn(("task", "bad"), calls)

    def test_config_rejects_duplicate_case_ids(self):
        with self.assertRaises(ValueError):
            PipelineRunner(_config(("same", "same")), Path("unused"))

    def test_receipt_is_required_when_stage_spec_declares_one(self):
        config = _config(("a",))
        for spec in config["stages"].values():
            spec["receipt"] = {"path": "{output}/receipt.json"}

        def run(command, cwd, env, stdout, stderr):
            output = Path(env["PIPELINE_OUTPUT"])
            output.mkdir(parents=True, exist_ok=True)
            Path(env["PIPELINE_HANDOFF"]).parent.joinpath("artifact.txt").write_text(
                env["PIPELINE_STAGE"], encoding="utf-8")
            (output / "receipt.json").write_text(
                json.dumps({"status": "completed", "sha256": "worker"}),
                encoding="utf-8")

        with tempfile.TemporaryDirectory() as directory:
            report = PipelineRunner(config, Path(directory) / "run",
                                    command_runner=run).run()
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["cases"]["a"]["stages"]["task"]["status"], "completed")

    def test_missing_receipt_is_needs_review_and_skips_dependents(self):
        config = _config(("a",))
        config["stages"]["repo"]["receipt"] = "{output}/receipt.json"

        def run(command, cwd, env, stdout, stderr):
            Path(env["PIPELINE_OUTPUT"]).mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as directory:
            report = PipelineRunner(config, Path(directory) / "run",
                                    max_attempts=1, command_runner=run).run()
        stages = report["cases"]["a"]["stages"]
        self.assertEqual(stages["repo"]["status"], "needs_review")
        self.assertEqual(stages["qa"]["status"], "skipped")
        self.assertEqual(stages["task"]["status"], "skipped")

    def test_resume_tampered_handoff_is_needs_review(self):
        calls = []

        def run(command, cwd, env, stdout, stderr):
            calls.append(tuple(command[:3]))
            Path(env["PIPELINE_OUTPUT"]).mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            PipelineRunner(_config(("a",)), output, command_runner=run).run()
            handoff = output / "cases" / "a" / "repo" / ".pipeline-handoff.json"
            handoff.write_text(handoff.read_text(encoding="utf-8") + "tampered", encoding="utf-8")
            report = PipelineRunner(_config(("a",)), output, resume=True,
                                    command_runner=run).run()
        stages = report["cases"]["a"]["stages"]
        self.assertEqual(stages["repo"]["status"], "needs_review")
        self.assertEqual(stages["qa"]["status"], "completed")
        self.assertEqual(stages["task"]["status"], "completed")
        self.assertEqual(len(calls), 3)


if __name__ == "__main__":
    unittest.main()
