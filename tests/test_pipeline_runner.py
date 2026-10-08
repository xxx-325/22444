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
    def test_three_repo_stages_start_together_and_write_handoffs(self):
        calls = []
        lock = threading.Lock()
        repo_started = threading.Barrier(3)
        active = 0
        maximum = 0

        def run(command, cwd, env, stdout, stderr):
            nonlocal active, maximum
            stage, case_id = command[1], command[2]
            with lock:
                active += 1
                maximum = max(maximum, active)
                calls.append((stage, case_id))
            if stage == "repo":
                repo_started.wait(timeout=2)
            time.sleep(0.01)
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
            self.assertEqual(len(calls), 9)
            self.assertEqual({stage for stage, _ in calls[:3]}, {"repo"})
            for case_id in ("a", "b", "c"):
                states = report["cases"][case_id]["stages"]
                self.assertTrue(all(item["status"] == "completed"
                                    for item in states.values()))
                self.assertTrue(all(item["attempts"] == 1
                                    for item in states.values()))
                for stage in ("repo", "qa", "task"):
                    handoff = root / "run" / "cases" / case_id / stage / ".pipeline-handoff.json"
                    self.assertTrue(handoff.exists())
            for case_id in ("a", "b", "c"):
                positions = {stage: calls.index((stage, case_id))
                             for stage in ("repo", "qa", "task")}
                self.assertLess(positions["repo"], positions["qa"])
                self.assertLess(positions["qa"], positions["task"])

    def test_ready_qa_starts_while_other_repo_stages_are_running(self):
        repo_started = threading.Barrier(3)
        qa_started = threading.Event()
        slow_repo_finished = []
        lock = threading.Lock()

        def run(command, cwd, env, stdout, stderr):
            stage, case_id = command[1], command[2]
            if stage == "repo":
                repo_started.wait(timeout=2)
                if case_id != "fast":
                    if not qa_started.wait(timeout=2):
                        raise StageCommandError("ready QA did not start")
                    with lock:
                        slow_repo_finished.append(case_id)
            elif stage == "qa" and case_id == "fast":
                with lock:
                    self.assertEqual(slow_repo_finished, [])
                qa_started.set()

        with tempfile.TemporaryDirectory() as directory:
            report = PipelineRunner(
                _config(("fast", "slow-a", "slow-b")), Path(directory) / "run",
                max_attempts=1, command_runner=run, poll_interval=0.001,
            ).run()
        self.assertEqual(report["status"], "completed")
        self.assertTrue(qa_started.is_set())
        self.assertCountEqual(slow_repo_finished, ["slow-a", "slow-b"])

    def test_shared_workers_respect_max_inflight_without_duplicate_claims(self):
        for limit in (1, 2, 3):
            with self.subTest(max_inflight=limit):
                calls = []
                lock = threading.Lock()
                active = 0
                maximum = 0

                def run(command, cwd, env, stdout, stderr):
                    nonlocal active, maximum
                    with lock:
                        active += 1
                        maximum = max(maximum, active)
                        calls.append((command[1], command[2]))
                    time.sleep(0.01)
                    with lock:
                        active -= 1

                with tempfile.TemporaryDirectory() as directory:
                    report = PipelineRunner(
                        _config(("a", "b", "c", "d", "e")),
                        Path(directory) / "run", max_inflight=limit,
                        command_runner=run, poll_interval=0.001,
                    ).run()
                self.assertEqual(report["status"], "completed")
                self.assertEqual(maximum, limit)
                self.assertEqual(len(calls), 15)
                self.assertEqual(len(set(calls)), 15)

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

    def test_resume_retries_unexhausted_failed_stage_and_unskips_dependents(self):
        calls = []
        fail_once = {"repo": True}

        def run(command, cwd, env, stdout, stderr):
            stage = command[1]
            calls.append(stage)
            if stage == "repo" and fail_once["repo"]:
                fail_once["repo"] = False
                raise StageCommandError("temporary failure")
            Path(env["PIPELINE_OUTPUT"]).mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            first = PipelineRunner(
                _config(("a",)), output, max_attempts=1,
                command_runner=run, poll_interval=0.001,
            ).run()
            self.assertEqual(first["cases"]["a"]["stages"]["repo"]["status"], "failed")
            resumed = PipelineRunner(
                _config(("a",)), output, max_attempts=2, resume=True,
                command_runner=run, poll_interval=0.001,
            ).run()
        self.assertEqual(resumed["status"], "completed")
        self.assertEqual(calls, ["repo", "repo", "qa", "task"])

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

    def test_completed_receipt_warning_reaches_pipeline_report(self):
        config = _config(("a",))
        def run(command, cwd, env, stdout, stderr):
            output = Path(env["PIPELINE_OUTPUT"])
            output.mkdir(parents=True, exist_ok=True)
            artifact = output / "artifact.txt"
            artifact.write_text(env["PIPELINE_STAGE"], encoding="utf-8")
            import hashlib
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            (output / "receipt.json").write_text(json.dumps({
                "status": "completed", "result": "completed_with_warnings",
                "sha256": digest,
                "artifacts": [{"path": "artifact.txt", "sha256": digest}],
            }), encoding="utf-8")
        for stage in config["stages"].values():
            stage["receipt"] = "{output}/receipt.json"
        with tempfile.TemporaryDirectory() as directory:
            report = PipelineRunner(config, Path(directory) / "run",
                                    command_runner=run).run()
        self.assertEqual(report["status"], "completed_with_warnings")
        self.assertEqual(
            report["cases"]["a"]["stages"]["repo"]["result"],
            "completed_with_warnings",
        )

    def test_resume_tampered_handoff_rebuilds_downstream_chain(self):
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
        self.assertEqual(stages["repo"]["status"], "completed")
        self.assertEqual(stages["qa"]["status"], "completed")
        self.assertEqual(stages["task"]["status"], "completed")
        self.assertEqual(len(calls), 6)


if __name__ == "__main__":
    unittest.main()
