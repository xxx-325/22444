import json
import hashlib
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
        "cases": [{"id": case_id, "source": str(Path(__file__).resolve())}
                  for case_id in case_ids],
    }


def _completed_receipt(output, *, result="completed"):
    artifact = output / "artifact.txt"
    artifact.write_text("valid output", encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (output / "receipt.json").write_text(json.dumps({
        "status": "completed", "result": result, "sha256": digest,
        "artifacts": [{"path": "artifact.txt", "sha256": digest}],
    }), encoding="utf-8")


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
            output = Path(env["PIPELINE_OUTPUT"])
            output.joinpath("artifact.txt").write_text(env["PIPELINE_STAGE"], encoding="utf-8")
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

    def test_resume_tampered_handoff_rejects_without_repeating_work(self):
        calls = []

        def run(command, cwd, env, stdout, stderr):
            calls.append(tuple(command[:3]))
            Path(env["PIPELINE_OUTPUT"]).mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            PipelineRunner(_config(("a",)), output, command_runner=run).run()
            handoff = output / "cases" / "a" / "repo" / ".pipeline-handoff.json"
            handoff.write_text(handoff.read_text(encoding="utf-8") + "tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pipeline handoff is invalid: a/repo"):
                PipelineRunner(_config(("a",)), output, resume=True,
                               command_runner=run).run()
        self.assertEqual(len(calls), 3)

    def test_resume_changed_source_at_same_path_rejects_without_calls_or_state_write(self):
        calls = []

        def run(command, cwd, env, stdout, stderr):
            calls.append(command[1])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_text('{"brief": "first"}', encoding="utf-8")
            config = _config(("a",))
            config["cases"][0]["source"] = str(source)
            output = root / "run"
            PipelineRunner(config, output, command_runner=run).run()
            original_state = (output / "pipeline-state.json").read_bytes()
            handoff = json.loads((output / "cases/a/repo/.pipeline-handoff.json").read_text())
            self.assertTrue(handoff["input_sha256"])
            source.write_text('{"brief": "second"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pipeline input changed: a/repo"):
                PipelineRunner(config, output, resume=True, command_runner=run).run()
            self.assertEqual((output / "pipeline-state.json").read_bytes(), original_state)
        self.assertEqual(calls, ["repo", "qa", "task"])

    def test_resume_missing_source_rejects_without_repeating_work(self):
        calls = []

        def run(command, cwd, env, stdout, stderr):
            calls.append(command[1])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_text("{}", encoding="utf-8")
            config = _config(("a",))
            config["cases"][0]["source"] = str(source)
            output = root / "run"
            PipelineRunner(config, output, command_runner=run).run()
            source.unlink()
            with self.assertRaisesRegex(ValueError, "pipeline input is missing or unreadable"):
                PipelineRunner(config, output, resume=True, command_runner=run).run()
        self.assertEqual(calls, ["repo", "qa", "task"])

    def test_resume_changed_upstream_content_rejects_before_downstream_reuse(self):
        calls = []

        def run(command, cwd, env, stdout, stderr):
            calls.append(command[1])
            Path(env["PIPELINE_OUTPUT"], "artifact.txt").write_text(command[1], encoding="utf-8")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            config = _config(("a",))
            PipelineRunner(config, output, command_runner=run).run()
            (output / "cases/a/repo/artifact.txt").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pipeline input changed: a/qa"):
                PipelineRunner(config, output, resume=True, command_runner=run).run()
        self.assertEqual(calls, ["repo", "qa", "task"])

    def test_resume_validates_receipt_artifact_contents(self):
        import hashlib
        calls = []
        config = _config(("a",))
        for spec in config["stages"].values():
            spec["receipt"] = "{output}/receipt.json"

        def run(command, cwd, env, stdout, stderr):
            calls.append(command[1])
            output = Path(env["PIPELINE_OUTPUT"])
            artifact = output / "artifact.txt"
            artifact.write_text(command[1], encoding="utf-8")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            (output / "receipt.json").write_text(json.dumps({
                "status": "completed", "sha256": digest,
                "artifacts": [{"path": "artifact.txt", "sha256": digest}],
            }), encoding="utf-8")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            PipelineRunner(config, output, command_runner=run).run()
            (output / "cases/a/task/artifact.txt").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pipeline cached artifact is invalid: a/task"):
                PipelineRunner(config, output, resume=True, command_runner=run).run()
        self.assertEqual(calls, ["repo", "qa", "task"])

    def test_resume_recovers_handoff_written_before_running_state_was_saved(self):
        calls = []

        def run(command, cwd, env, stdout, stderr):
            calls.append(command[1])

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            config = _config(("a",))
            PipelineRunner(config, output, command_runner=run).run()
            state_path = output / "pipeline-state.json"
            state = json.loads(state_path.read_text())
            record = state["cases"]["a"]["stages"]["repo"]
            record["status"] = "running"
            record.pop("handoff")
            record.pop("handoff_sha256")
            state_path.write_text(json.dumps(state), encoding="utf-8")
            resumed = PipelineRunner(config, output, resume=True, command_runner=run).run()
            self.assertEqual(resumed["cases"]["a"]["stages"]["repo"]["status"], "completed")
        self.assertEqual(calls, ["repo", "qa", "task"])

    def test_command_failure_with_verified_completed_output_continues_without_retry(self):
        for exception in (StageCommandError("command exited with status 1"),
                          RuntimeError("usage ledger temporarily unavailable")):
            with self.subTest(error=type(exception).__name__), tempfile.TemporaryDirectory() as directory:
                calls = []
                config = _config(("a",))
                config["stages"]["repo"]["receipt"] = "{output}/receipt.json"

                def run(command, cwd, env, stdout, stderr):
                    calls.append(command[1])
                    if command[1] == "repo":
                        _completed_receipt(Path(env["PIPELINE_OUTPUT"]))
                        raise exception

                report = PipelineRunner(config, Path(directory) / "run", command_runner=run).run()
                record = report["cases"]["a"]["stages"]["repo"]
                self.assertEqual(calls, ["repo", "qa", "task"])
                self.assertEqual(record["status"], "completed")
                self.assertEqual(record["result"], "completed_with_warnings")
                self.assertEqual(record["failures"][0]["detail"], str(exception))
                self.assertEqual(report["status"], "completed_with_warnings")

    def test_failed_command_output_does_not_hide_hard_errors_or_noncompletion(self):
        for result, error, tamper in (
                ("routes_failed", "command exited with status 1", False),
                ("skipped", "command exited with status 1", False),
                ("completed", "collection_authentication_failed", False),
                ("completed", "source closure failed", False),
                ("completed", "command exited with status 1", True)):
            with self.subTest(result=result, error=error, tamper=tamper), tempfile.TemporaryDirectory() as directory:
                calls = []
                config = _config(("a",))
                config["stages"]["repo"]["receipt"] = "{output}/receipt.json"

                def run(command, cwd, env, stdout, stderr):
                    calls.append(command[1])
                    output = Path(env["PIPELINE_OUTPUT"])
                    _completed_receipt(output, result=result)
                    if tamper:
                        (output / "artifact.txt").write_text("tampered", encoding="utf-8")
                    raise StageCommandError(error)

                report = PipelineRunner(config, Path(directory) / "run", command_runner=run,
                                        max_attempts=1).run()
                self.assertEqual(calls, ["repo"])
                self.assertEqual(report["cases"]["a"]["stages"]["repo"]["status"], "failed")

    def test_resume_recovers_verified_receipt_and_reopens_dependency_skipped_stages(self):
        for prior_status in ("running", "failed", "needs_review"):
            with self.subTest(status=prior_status), tempfile.TemporaryDirectory() as directory:
                calls = []
                config = _config(("a",))
                config["stages"]["repo"]["receipt"] = "{output}/receipt.json"

                def run(command, cwd, env, stdout, stderr):
                    calls.append(command[1])
                    if command[1] == "repo":
                        raise StageCommandError("temporary report failure")

                output = Path(directory) / "run"
                PipelineRunner(config, output, command_runner=run, max_attempts=1).run()
                _completed_receipt(output / "cases/a/repo", result="completed_with_warnings")
                state_path = output / "pipeline-state.json"
                state = json.loads(state_path.read_text())
                state["cases"]["a"]["stages"]["repo"]["status"] = prior_status
                state_path.write_text(json.dumps(state), encoding="utf-8")
                report = PipelineRunner(config, output, resume=True, command_runner=run,
                                        max_attempts=1).run()
                self.assertEqual(calls, ["repo", "qa", "task"])
                self.assertEqual(report["cases"]["a"]["stages"]["repo"]["attempts"], 1)
                self.assertTrue(all(row["status"] == "completed" for row in
                                    report["cases"]["a"]["stages"].values()))

    def test_resume_promotes_warning_handoff_after_failed_state(self):
        with tempfile.TemporaryDirectory() as directory:
            config = _config(("a",))
            config["stages"]["repo"]["receipt"] = "{output}/receipt.json"
            calls = []

            def run(command, cwd, env, stdout, stderr):
                calls.append(command[1])
                if command[1] == "repo":
                    _completed_receipt(Path(env["PIPELINE_OUTPUT"]),
                                        result="completed_with_warnings")

            output = Path(directory) / "run"
            first = PipelineRunner(config, output, command_runner=run).run()
            self.assertEqual(calls, ["repo", "qa", "task"])
            state_path = output / "pipeline-state.json"
            state = json.loads(state_path.read_text())
            state["cases"]["a"]["stages"]["repo"]["status"] = "failed"
            state_path.write_text(json.dumps(state), encoding="utf-8")

            def should_not_run(*args):
                raise AssertionError("a valid warning handoff should be reused")

            report = PipelineRunner(config, output, resume=True,
                                    command_runner=should_not_run).run()
            self.assertEqual(report["status"], "completed_with_warnings")
            self.assertEqual(report["cases"]["a"]["stages"]["repo"]["status"], "completed")
            self.assertEqual(report["cases"]["a"]["stages"]["repo"]["result"],
                             "completed_with_warnings")

    def test_resume_does_not_recover_receipt_after_authentication_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            config = _config(("a",))
            config["stages"]["repo"]["receipt"] = "{output}/receipt.json"
            calls = []

            def run(command, cwd, env, stdout, stderr):
                calls.append(command[1])
                raise StageCommandError("collection_authentication_failed")

            output = Path(directory) / "run"
            PipelineRunner(config, output, command_runner=run, max_attempts=1).run()
            _completed_receipt(output / "cases/a/repo")
            report = PipelineRunner(config, output, resume=True, command_runner=run,
                                    max_attempts=1).run()
            self.assertEqual(calls, ["repo"])
            self.assertEqual(report["cases"]["a"]["stages"]["repo"]["status"], "failed")

    def test_stdout_hard_error_does_not_get_hidden_by_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            config = _config(("a",))
            config["stages"]["repo"]["receipt"] = "{output}/receipt.json"

            def run(command, cwd, env, stdout, stderr):
                _completed_receipt(Path(env["PIPELINE_OUTPUT"]))
                stdout.write_text("HTTP 401 invalid_api_key", encoding="utf-8")
                raise StageCommandError("command exited with status 1")

            report = PipelineRunner(config, Path(directory) / "run",
                                    command_runner=run, max_attempts=1).run()
            stages = report["cases"]["a"]["stages"]
            self.assertEqual(stages["repo"]["status"], "failed")
            self.assertEqual(stages["qa"]["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
