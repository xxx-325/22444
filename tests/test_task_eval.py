import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from dialogue_benchmark.llm import _ask_stage
from dialogue_benchmark.task_eval.artifacts import qa_inputs, group_qa_inputs, generation_request, save
from dialogue_benchmark.task_eval.checks import pytest_result, _test_write_violations
from dialogue_benchmark.task_eval.metrics import compare_trials, measure
from dialogue_benchmark.task_eval.runtime import (configure, preflight_openhands_runtime,
                                                   readable_reference, release_completed_execution,
                                                   run_agent, shared_task_slot)
from dialogue_benchmark.task_eval.run import (admission, freeze, implementation_pollution,
                                               main as run_tasks,
                                               reference_solver_answer,
                                               solver_input, unchanged, validated_spec,
                                               _static_repository_recovery,
                                               external_clarification_history,
                                               repository_design_probe)


class TaskEvaluationTests(unittest.TestCase):
    def test_empty_external_groups_save_shortfall_before_provider_or_runtime(self):
        for question_count in (0,):
            with self.subTest(question_count=question_count), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                qa_run = root / "qa"
                questions = [{"id": "q%d" % index, "type": "M1", "status": "approved",
                              "question": "Question %d" % index,
                              "answer_points": [{"text": "Rule %d" % index, "sources": ["e%d" % index]}]}
                             for index in range(question_count)]
                save(qa_run / "qa-public.json", {"status": "approved", "questions": questions})
                save(qa_run / "manifest.json", {"qa_source": "external"})
                save(qa_run / "stages/group-raw-candidates.json", {"questions": questions})
                save(qa_run / "stages/group-qa-input.json", {"payload": {"scope": {
                    "dialogue": [{"id": "e%d" % index} for index in range(question_count)]}}})
                args = ["--simulator-path", str(root), "--source-run", str(root / "source"),
                        "--qa-run", str(qa_run), "--env-file", str(root / ".env"),
                        "--output", str(root / "tasks"), "--count", "3"]
                with patch("dialogue_benchmark.task_eval.run.configure") as provider, \
                     patch("dialogue_benchmark.task_eval.run.preflight_openhands_runtime") as runtime:
                    self.assertEqual(run_tasks(args), 0)
                    provider.assert_not_called()
                    runtime.assert_not_called()
                    manifest = json.loads((root / "tasks/manifest.json").read_text())
                    self.assertEqual((manifest["status"], manifest["completed"], manifest["shortfall"]),
                                     ("incomplete", 0, 3))
                    self.assertEqual(manifest["stop_reason"],
                                     "no_related_external_qa" if question_count else "no_eligible_qa")
                    self.assertTrue((root / "tasks/qa-grouping.json").is_file())
                    self.assertTrue((root / "tasks/report.md").is_file())
                    manifest["tasks"] = [{"task": "task-01", "status": "evaluated"}]
                    save(root / "tasks/manifest.json", manifest)
                    with self.assertRaisesRegex(ValueError, "Resume task selection changed"):
                        run_tasks(args + ["--resume"])
                    self.assertEqual(json.loads((root / "tasks/manifest.json").read_text()), manifest)

    def test_task_slots_are_shared_between_processes_and_released_after_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = {
                "DIALOGUE_TASK_SLOT_DIR": str(root / "slots"),
                "DIALOGUE_TASK_SLOT_LIMIT": "1",
            }
            code = (
                "import sys\n"
                "from dialogue_benchmark.task_eval.runtime import shared_task_slot\n"
                "with shared_task_slot():\n"
                "    print('acquired', flush=True)\n"
                "    sys.stdin.readline()\n"
            )
            process = subprocess.Popen(
                [sys.executable, "-c", code], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                env={**os.environ, **environment},
            )
            try:
                self.assertEqual(process.stdout.readline().strip(), "acquired")
                import fcntl
                with (root / "slots/0.lock").open("a") as lock:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                process.terminate()
                process.wait(timeout=10)
                with patch.dict(os.environ, environment), shared_task_slot():
                    with (root / "slots/0.lock").open("a") as lock:
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=10)

    def test_task_slot_is_released_on_local_failure(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "DIALOGUE_TASK_SLOT_DIR": directory, "DIALOGUE_TASK_SLOT_LIMIT": "1",
        }):
            with self.assertRaisesRegex(RuntimeError, "local"):
                with shared_task_slot():
                    raise RuntimeError("local")
            with shared_task_slot():
                pass

    def test_external_related_questions_group_with_all_answers_and_private_requests(self):
        members = []
        for index, source in enumerate(("m1", "m2"), 1):
            members.append({"qa": {"id": "q%d" % index, "question": "Question %d" % index,
                                    "answer_points": [{"text": "Answer %d" % index}]},
                            "qa_source": "external", "generation_input": "/tmp/input-%d" % index,
                            "source_ids": [source], "associations": ["task:checkout"],
                            "provisional": False})
        with patch("dialogue_benchmark.task_eval.artifacts.read",
                   side_effect=lambda path: {"payload": path}):
            grouped = group_qa_inputs(members, 2)
            self.assertEqual(len(grouped), 1)
            self.assertEqual(grouped[0]["qa_ids"], ["q1", "q2"])
            self.assertEqual([point["text"] for point in grouped[0]["qa"]["answer_points"]],
                             ["Answer 1", "Answer 2"])
            request = generation_request(grouped[0])
            self.assertEqual([row["qa_id"] for row in request["questions"]], ["q1", "q2"])

    def test_external_unrelated_questions_are_not_padded_into_group(self):
        members = [{"qa": {"id": "q%d" % index, "question": "Question %d" % index},
                    "qa_source": "external", "generation_input": "/tmp/input-%d" % index,
                    "source_ids": ["m%d" % index], "associations": ["task:%d" % index]}
                   for index in (1, 2)]
        diagnostics = []
        self.assertEqual(group_qa_inputs(members, 2, diagnostics=diagnostics), members)
        self.assertEqual(diagnostics, [])

    def test_external_questions_sharing_only_source_are_not_grouped(self):
        members = [{"qa": {"id": "q%d" % index, "question": "Question %d" % index},
                    "qa_source": "external", "generation_input": "/tmp/input-%d" % index,
                    "source_ids": ["same-source"], "associations": []}
                   for index in (1, 2)]
        diagnostics = []
        self.assertEqual(group_qa_inputs(members, 2, diagnostics=diagnostics), members)
        self.assertEqual(diagnostics, [])

    def test_external_group_keeps_business_lineage_without_public_ids(self):
        members = []
        for index in (1, 2):
            members.append({"qa": {"id": "q%d" % index,
                                    "question": "Question %d" % index,
                                    "answer_points": [{"text": "Answer %d" % index}]},
                            "qa_source": "external", "generation_input": "/tmp/input-%d" % index,
                            "source_ids": ["source-%d" % index],
                            "associations": ["behavior:handoff"],
                            "external_lineage": {
                                "source_ids": ["source-%d" % index],
                                "event_ids": ["event-%d" % index],
                                "business_behavior": ["handoff"],
                                "impact": ["retry policy"],
                            }})
        grouped = group_qa_inputs(members, 2)
        self.assertEqual(grouped[0]["external_lineage"]["business_behavior"], ["handoff"])
        self.assertEqual(grouped[0]["external_lineage"]["event_ids"], ["event-1", "event-2"])
        self.assertNotIn("source-1", grouped[0]["qa"]["question"])

    def test_single_external_question_is_a_valid_optional_pool(self):
        member = {"qa": {"id": "q1", "question": "Question 1"},
                  "qa_source": "external", "generation_input": "/tmp/input-1",
                  "source_ids": ["m1"], "associations": ["task:checkout"]}
        diagnostics = []
        self.assertEqual(group_qa_inputs([member], 1, diagnostics=diagnostics), [member])
        self.assertEqual(diagnostics, [])

    def test_optional_pool_limit_keeps_related_remainder(self):
        members = [{"qa": {"id": "q%d" % index, "question": "Question %d" % index},
                    "qa_source": "external", "associations": ["task:checkout"]}
                   for index in range(5)]
        grouped = group_qa_inputs(members)
        self.assertEqual(grouped[0]["qa_ids"], ["q0", "q1", "q2", "q3"])
        self.assertEqual(grouped[1], members[4])

    def test_openhands_preflight_uses_current_interpreter_without_starting_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            simulator = Path(directory)
            completed = SimpleNamespace(returncode=0, stdout="", stderr="")
            with patch("dialogue_benchmark.task_eval.runtime.subprocess.run",
                       return_value=completed) as probe:
                result = preflight_openhands_runtime(simulator)
            self.assertEqual(result["python"], sys.executable)
            command = probe.call_args.args[0]
            self.assertEqual(command[:2], [sys.executable, "-c"])
            self.assertIn("simulator.openhands.container", command[2])
            self.assertIn("simulator.openhands.worker", command[2])
            self.assertEqual(probe.call_args.kwargs["timeout"], 30)
            self.assertEqual(probe.call_args.kwargs["check"], False)
            self.assertEqual(probe.call_args.kwargs["env"]["OPENHANDS_SUPPRESS_BANNER"], "1")
            self.assertEqual(probe.call_args.kwargs["env"]["DO_NOT_TRACK"], "1")

    def test_openhands_preflight_reports_missing_runtime_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            simulator = Path(directory)
            interpreter = simulator / ".venv-openhands/bin/python"
            interpreter.parent.mkdir(parents=True)
            interpreter.write_text("#!/bin/sh\n")
            interpreter.chmod(0o755)
            failed = SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="simulator.openhands.container: ModuleNotFoundError: No module named 'httpx'",
            )
            with patch("dialogue_benchmark.task_eval.runtime.subprocess.run",
                       return_value=failed):
                with self.assertRaisesRegex(RuntimeError, "httpx"):
                    preflight_openhands_runtime(simulator, python_executable=interpreter)

    def test_openhands_preflight_reports_missing_interpreter(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "interpreter does not exist"):
                preflight_openhands_runtime(directory, python_executable=Path(directory) / "missing-python")

    def test_generated_acceptance_tests_keep_candidate_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test_acceptance.py").write_text(
                "def test_bad(candidate_root):\n"
                "    (candidate_root / 'input.json').write_text('{}')\n",
                encoding="utf-8")
            self.assertEqual(len(_test_write_violations(root)), 1)

    def test_generated_acceptance_tests_use_tmp_path_for_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test_acceptance.py").write_text(
                "def test_ok(candidate_root, tmp_path):\n"
                "    (tmp_path / 'input.json').write_text('{}')\n",
                encoding="utf-8")
            self.assertEqual(_test_write_violations(root), [])

    def test_read_only_precheck_respects_open_mode_and_function_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test_acceptance.py").write_text(
                "def test_read(candidate_root):\n"
                "    path = candidate_root / 'data.json'\n"
                "    with path.open() as stream: stream.read()\n"
                "    with path.open(mode='rb') as stream: stream.read()\n"
                "def test_output(tmp_path):\n"
                "    path = tmp_path / 'output.json'\n"
                "    path.write_text('{}')\n"
                "def test_write(candidate_root):\n"
                "    path = candidate_root / 'data.json'\n"
                "    path.open(mode='a')\n")
            violations = _test_write_violations(root)
            self.assertEqual(len(violations), 1, violations)
            self.assertIn(":10:", violations[0])

    def test_author_uses_remaining_runtime_budget(self):
        for limit, turns in ((100, 1), (12, 1)):
            with self.subTest(limit=limit), tempfile.TemporaryDirectory() as directory:
                workers = []
                class Budget:
                    def __init__(self, config, journal):
                        self.config = config
                        self.deadline = time.monotonic() + 60
                        self.data = {"attempts": 0, "prompt_tokens": 0, "completion_tokens": 0}
                class Worker:
                    def __init__(self, *args, **kwargs):
                        self.budget = kwargs["budget"]
                        self.messages = []
                        self.closed = False
                        workers.append(self)
                    def start(self): pass
                    def close(self): self.closed = True
                    def events(self): return []
                    def turn(self, message):
                        self.messages.append(message)
                        self.budget.data["attempts"] += 1
                        self.budget.data["prompt_tokens"] += 5
                        return {"status": "finished"}
                with patch.dict("sys.modules", {
                        "simulator.openhands.budget": SimpleNamespace(Budget=Budget),
                        "simulator.openhands.container": SimpleNamespace(SDKContainer=Worker)}), \
                     patch("dialogue_benchmark.task_eval.retention.release_agent"), \
                     patch("dialogue_benchmark.task_eval.retention.save_trace"):
                    result = run_agent(Path(directory), {"judge": {}, "image": "test"}, "judge", "Draft",
                                       max_seconds=37, max_tokens=limit)
                self.assertEqual(len(workers), 1)
                self.assertEqual(len(workers[0].messages), turns)
                self.assertTrue(workers[0].closed)
                self.assertEqual(workers[0].budget.config["max_seconds"], 37)

    def test_task_agents_do_not_inherit_per_response_output_limit(self):
        original = {"image": "sdk", "execution_image": "executor", "execution_backend": "ssh_sandbox",
                    "code": {"model": "solver", "key_env": "TEST_PROVIDER_KEY", "max_output_tokens": 8192},
                    "judge": {"model": "judge", "key_env": "TEST_PROVIDER_KEY", "max_output_tokens": 4096}}
        with patch.dict("sys.modules", {"simulator.episode": SimpleNamespace(load_environment=lambda _: None)}), \
             patch("sys.path", []), \
             patch.dict("os.environ", {"TEST_PROVIDER_KEY": "test-only"}), \
             patch("dialogue_benchmark.task_eval.runtime.read", return_value={"config": original}):
            config = configure("/simulator", "/checkpoint.json", "/provider.env")
        self.assertIsNone(config["code"]["max_output_tokens"])
        self.assertIsNone(config["judge"]["max_output_tokens"])
        self.assertEqual(original["code"]["max_output_tokens"], 8192)
        self.assertEqual(config["code"]["model"], "solver")

    def test_expired_agent_retains_trajectory_and_reports_resource_limit(self):
        class Budget:
            def __init__(self, config, journal):
                self.deadline = config["max_seconds"]
                self.data = {"attempts": 1, "prompt_tokens": 1, "completion_tokens": 1}
        class Worker:
            def __init__(self, *args, **kwargs):
                pass
            def start(self):
                pass
            def turn(self, message):
                raise TimeoutError("Execution deadline reached")
            def close(self):
                pass
            def events(self):
                return [{"id": "read-code", "kind": "ActionEvent", "tool_name": "terminal",
                         "action": {"command": "cat a.py"}}]
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict("sys.modules", {
                 "simulator.openhands.budget": SimpleNamespace(Budget=Budget),
                 "simulator.openhands.container": SimpleNamespace(SDKContainer=Worker)}), \
             patch("dialogue_benchmark.task_eval.runtime.time.monotonic", return_value=1201):
            root = Path(directory)
            outcome = run_agent(root, {"code": {}, "image": "test"}, "code", "Implement")
            self.assertEqual(outcome["status"], "error")
            self.assertEqual(outcome["error_code"], "runtime_budget_exhausted")
            self.assertEqual(json.loads((root / "trajectory.json").read_text())[0]["id"], "read-code")

    def test_execution_release_keeps_host_artifacts_and_selects_check_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = root / "private/execution/environment.json"
            record.parent.mkdir(parents=True)
            record.write_text(json.dumps({"container_id": "a" * 64, "status": "ready"}))
            snapshot = root / "snapshot.txt"
            snapshot.write_text("saved code and evidence")
            with patch("dialogue_benchmark.task_eval.retention.release_agent") as release:
                release_completed_execution(record)
            release.assert_called_once_with(root)
            self.assertEqual(snapshot.read_text(), "saved code and evidence")
            self.assertTrue(record.exists())

    def test_incomplete_execution_setup_is_not_stopped(self):
        with tempfile.TemporaryDirectory() as directory:
            record = Path(directory) / "environment.json"
            record.write_text(json.dumps({"container_id": "a" * 64, "status": "creating"}))
            with patch("dialogue_benchmark.task_eval.retention.release_agent") as stop:
                release_completed_execution(record)
            stop.assert_not_called()

    def test_private_inputs_are_exported_readably_without_changing_originals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "private"
            source.mkdir(mode=0o700)
            (source / "qa.json").write_text('{"question": "Historical question"}')
            (source / "qa.json").chmod(0o600)
            (source / "check.sh").write_text("#!/bin/sh\nexit 0\n")
            (source / "check.sh").chmod(0o700)
            target = readable_reference(source, root / "export")
            self.assertEqual((target / "qa.json").read_bytes(), (source / "qa.json").read_bytes())
            self.assertEqual((target / "qa.json").stat().st_mode & 0o777, 0o644)
            self.assertEqual(target.stat().st_mode & 0o777, 0o755)
            self.assertEqual((source / "qa.json").stat().st_mode & 0o777, 0o600)
            self.assertEqual(source.stat().st_mode & 0o777, 0o700)
            self.assertEqual((target / "check.sh").stat().st_mode & 0o777, 0o755)
            self.assertEqual((source / "check.sh").stat().st_mode & 0o777, 0o700)

    def test_only_answer_is_added_to_solver_context(self):
        normal = solver_input("Implement behavior B.")
        memory = solver_input("Implement behavior B.", "Old behavior A failed under C.")
        self.assertTrue(memory.startswith(normal))
        self.assertNotIn("checkpoint", normal)
        self.assertNotIn("acceptance.md", memory)
        self.assertIn("Old behavior", memory[len(normal):])

    def test_external_reference_solver_receives_qa_answer_without_history_contract(self):
        item = {"qa_source": "external", "qa": {
            "answer_points": [{"text": "Maple keeps explicit null values except note."}]}}
        self.assertEqual(reference_solver_answer(item),
                         "- Maple keeps explicit null values except note.")

    def test_graph_reference_solver_keeps_contract_context(self):
        item = {"qa_source": "graph", "qa": {
            "answer_points": [{"text": "Historical rule."}]}}
        history = {"contracts": [], "oracle_answer": "Contract context."}
        with patch("dialogue_benchmark.task_eval.run.historical_context",
                   return_value="Contract context."):
            self.assertEqual(reference_solver_answer(item, history),
                             "- Historical rule.\nContract context.")

    def test_frozen_content_change_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, spec = root / "base", root / "spec"
            baseline.mkdir()
            spec.mkdir()
            (baseline / "a.py").write_text("x = 1\n")
            for name in ("task.md", "acceptance.md"):
                (spec / name).write_text("1. Explicit action\n")
            (spec / "acceptance.json").write_text('[{"id": "a1"}]')
            (spec / "fixture.txt").write_text("Fixture used by the generated test")
            receipt = freeze(spec, root / "frozen", baseline)
            self.assertEqual((root / "frozen/fixture.txt").read_text(),
                             "Fixture used by the generated test")
            self.assertTrue(unchanged(receipt, root / "frozen", baseline))
            (root / "frozen/task.md").write_text("Changed task")
            self.assertFalse(unchanged(receipt, root / "frozen", baseline))

    def test_empty_and_skipped_suites_are_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "junit.xml"
            for body in ("", "<testcase><skipped /></testcase>"):
                path.write_text("<testsuite>" + body + "</testsuite>")
                self.assertEqual(pytest_result(0, path)["status"], "unavailable")
            path.write_text("<testsuite><testcase /></testsuite>")
            self.assertEqual(pytest_result(0, path)["status"], "passed")

    def test_model_accept_cannot_override_test_failure(self):
        decision = {"BASELINE": "unmet", "REFERENCE": "pass", "VERDICT": "accept",
                    "TESTS": "executable", "MUTATIONS": "caught", "COVERAGE": "complete"}
        self.assertFalse(admission(decision, {"status": "failed"}, {"status": "failed"}))
        self.assertFalse(admission(decision, {"status": "passed"}, {"status": "passed"}))
        self.assertTrue(admission(decision, {"status": "failed"}, {"status": "passed"}))
        decision["TESTS"] = "unavailable"
        self.assertTrue(admission(decision, {"status": "unavailable"}, {"status": "unavailable"}))

    def test_inspection_acceptance_controls_history_admission(self):
        decision = {"BASELINE": "unmet", "REFERENCE": "pass", "VERDICT": "accept",
                    "TESTS": "executable", "MUTATIONS": "caught", "COVERAGE": "complete"}
        passed = {"status": "passed", "rows": []}
        failed = {"status": "failed", "rows": []}
        uncertain = {"status": "uncertain", "rows": []}
        # A public inspection can expose a missing baseline artifact even when
        # every automatic regression passes.
        self.assertTrue(admission(decision, {"status": "passed"}, {"status": "passed"},
                                  failed, passed))
        # A baseline that already satisfies the inspection is not an unmet task.
        self.assertFalse(admission(decision, {"status": "passed"}, {"status": "passed"},
                                   passed, passed))
        # Missing inspection evidence and inspector/runtime errors stay closed.
        self.assertFalse(admission(decision, {"status": "unavailable"}, {"status": "passed"},
                                   uncertain, passed))
        self.assertFalse(admission(decision, {"status": "passed"}, {"status": "error"},
                                   failed, uncertain))
        # Pure inspection tasks use the same rule when no pytest cases exist.
        self.assertTrue(admission(dict(decision, TESTS="unavailable"),
                                  {"status": "unavailable"}, {"status": "unavailable"},
                                  failed, passed))
        self.assertFalse(admission(dict(decision, TESTS="unavailable", MUTATIONS="unverified"),
                                   {"status": "unavailable"}, {"status": "unavailable"},
                                   failed, passed))

    def test_unresolved_coverage_and_skipped_tests_prevent_admission(self):
        decision = {"BASELINE": "unmet", "REFERENCE": "pass", "VERDICT": "accept",
                    "TESTS": "executable", "MUTATIONS": "caught", "COVERAGE": "complete"}
        for field, value in (("COVERAGE", "gaps"), ("COVERAGE", "uncertain"),
                             ("COVERAGE", None), ("MUTATIONS", "missed")):
            self.assertFalse(admission(dict(decision, **{field: value}),
                                       {"status": "failed"}, {"status": "passed"}))
        self.assertFalse(admission(decision, {"status": "failed"}, {"status": "passed", "skipped": 1}))
        self.assertFalse(admission(decision, {"status": "error"}, {"status": "passed"}))

    def test_validator_checks_are_preserved_without_replacing_author_tests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, checks = root / "spec", root / "checks"
            spec.mkdir()
            checks.mkdir()
            (spec / "test_acceptance.py").write_text("Original tests")
            self.assertIsNone(validated_spec(spec, checks, root / "uncovered"))
            (checks / "coverage.md").write_text("Replacement combined with a pending write")
            (checks / "test_interactions.py").write_text("Combination regression")
            (checks / "test_acceptance.py").write_text("Do not replace original tests")
            combined = validated_spec(spec, checks, root / "combined")
            self.assertEqual((combined / "test_acceptance.py").read_text(), "Original tests")
            self.assertEqual((combined / "test_interactions.py").read_text(), "Combination regression")

    def test_validator_requires_and_freezes_one_command_per_inspect_row(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, checks = root / "spec", root / "checks"
            spec.mkdir()
            checks.mkdir()
            (spec / "acceptance.md").write_text(
                "| ID | Requirement | Basis | Check |\n"
                "|---|---|---|---|\n"
                "| a1 | Export meaning | task | inspect: run export and verify output |\n")
            (checks / "coverage.md").write_text("a1: complete\n")
            self.assertIsNone(validated_spec(spec, checks, root / "missing-command"))
            commands = checks / "commands"
            commands.mkdir()
            (commands / "inspect-a1.sh").write_text("#!/bin/sh\nexit 0\n")
            frozen = validated_spec(spec, checks, root / "frozen")
            self.assertIsNotNone(frozen)
            self.assertEqual((frozen / "commands/inspect-a1.sh").read_text(),
                             "#!/bin/sh\nexit 0\n")

    def test_timeout_and_invalid_junit_are_errors_not_unavailable_tests(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.xml"
            self.assertEqual(pytest_result(124, path)["status"], "error")
            path.write_text("broken xml")
            self.assertEqual(pytest_result(0, path)["status"], "error")
            path.write_text("<testsuite><testcase /></testsuite>")
            self.assertEqual(pytest_result(3, path)["status"], "error")

    def test_only_two_successes_get_cost_differences(self):
        pair = {"without_memory": {"result": "passed", "metrics": {"tool_calls": 10}},
                "with_memory": {"result": "failed", "metrics": {"tool_calls": 1}}}
        self.assertIsNone(compare_trials(pair)["cost_differences"]["tool_calls"])
        self.assertEqual(compare_trials(pair)["raw_cost_differences"]["tool_calls"], -9)
        pair["without_memory"]["metrics"]["usage_complete"] = True
        pair["with_memory"].update(result="passed", metrics={"tool_calls": 1, "usage_complete": True})
        self.assertEqual(compare_trials(pair)["cost_differences"]["tool_calls"], -9)
        self.assertEqual(compare_trials(pair)["comparable_cost_differences"]["tool_calls"], -9)
        self.assertEqual(compare_trials(pair)["pair_class"], "both_passed")

    def test_pair_class_keeps_capability_and_failure_categories(self):
        base = {"metrics": {"usage_complete": True}}
        self.assertEqual(compare_trials({
            "without_memory": dict(base, result="failed"),
            "with_memory": dict(base, result="passed")})["pair_class"],
                         "memory_capability_gain")
        self.assertEqual(compare_trials({
            "without_memory": dict(base, result="passed"),
            "with_memory": dict(base, result="failed")})["pair_class"],
                         "memory_harm_candidate")
        self.assertEqual(compare_trials({
            "without_memory": dict(base, result="failed"),
            "with_memory": dict(base, result="uncertain")})["pair_class"],
                         "both_failed")

    def test_evaluator_failure_is_not_a_solver_outcome(self):
        base = {"metrics": {"usage_complete": True}}
        judge_error = dict(base, result="uncertain", judge_status="error")
        pair = compare_trials({"without_memory": dict(base, result="failed"),
                               "with_memory": judge_error})
        self.assertEqual(pair["pair_class"], "evaluation_failed")
        self.assertIsNone(pair["completion_difference"])
        self.assertEqual(pair["evaluation_failures"], {"with_memory": "judge_error"})
        checks_error = dict(base, result="uncertain", checks={"status": "error"})
        self.assertEqual(compare_trials({"without_memory": checks_error,
                                         "with_memory": dict(base, result="passed")})["pair_class"],
                         "evaluation_failed")
        # A deterministic pass or fail survives an unrelated Judge error.
        self.assertEqual(compare_trials({
            "without_memory": dict(base, result="failed", judge_status="error"),
            "with_memory": dict(base, result="passed")})["pair_class"], "memory_capability_gain")

    def test_report_scores_only_approved_tasks_with_usable_evaluation(self):
        from dialogue_benchmark.task_eval.report import write_report
        base = {"metrics": {"usage_complete": True}, "trial": "trial-1"}
        def task(name, without, with_memory, **extra):
            return dict(task=name, status="evaluated", comparison={
                "without_memory": dict(base, **without), "with_memory": dict(base, **with_memory)}, **extra)
        manifest = {"tasks": [
            task("task-01", {"result": "failed"}, {"result": "passed"}),
            task("task-02", {"result": "passed"}, {"result": "passed"}, provisional=True),
            task("task-03", {"result": "uncertain", "judge_status": "error"}, {"result": "passed"}),
        ]}
        with tempfile.TemporaryDirectory() as directory:
            write_report(Path(directory), manifest)
            report = (Path(directory) / "report.md").read_text()
        self.assertIn("Scored tasks: 1. Excluded from pass rates: 2.", report)
        self.assertIn("- task-02: provisional QA", report)
        self.assertIn("- task-03: evaluation failed: without_memory=judge_error", report)
        self.assertIn("| without_memory | 1 | 0/1/0 | 0.0% |", report)
        self.assertIn("| with_memory | 1 | 1/0/0 | 100.0% |", report)
        self.assertIn("Pass-rate difference (with − without): 100.0 percentage points.", report)

    def test_repository_recovery_does_not_treat_common_number_as_leak(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.py").write_text("DEFAULT_TIMEOUT = 600\n")
            self.assertFalse(_static_repository_recovery(
                root, "The operation waits 600 seconds and does not read history."))

    def test_external_responder_history_does_not_replace_solver_history(self):
        item = {
            "qa_source": "external",
            "qa": {"answer_points": [{"text": "SMS is disabled.", "sources": ["e1"]}]},
            "public_records": [{"id": "e1", "original_id": "e1", "order": 3,
                                "kind": "message", "role": "user",
                                "text": "The provider disabled SMS."}],
        }
        history = external_clarification_history(item)
        self.assertEqual(history["contracts"][0]["sources"], ["e1"])
        self.assertEqual(history["events"][0]["text"], "The provider disabled SMS.")

    def test_external_responder_history_uses_reviewed_sources_of_published_qa(self):
        records = [{"id": "e1", "original_id": "e1", "order": 3, "kind": "message",
                    "role": "user", "text": "The provider disabled SMS."},
                   {"id": "e2", "original_id": "e2", "order": 5, "kind": "message",
                    "role": "user", "text": "Email stays enabled."}]
        def member(qa_id, text, source):
            # Published QA keeps answer text only; sources live in the reviewed candidate.
            return {"qa_source": "external", "public_records": records,
                    "qa": {"id": qa_id, "answer_points": [{"text": text}]},
                    "reviewed_candidate": {"id": qa_id,
                                           "answer_points": [{"text": text, "sources": [source]}]}}
        single = member("q1", "SMS is disabled.", "e1")
        history = external_clarification_history(single)
        self.assertEqual(history["contracts"][0]["sources"], ["e1"])
        grouped = dict(single, qa={"id": "group", "answer_points": [{"text": "SMS is disabled."},
                                                                  {"text": "Email stays enabled."}]},
                       qa_members=[single, member("q2", "Email stays enabled.", "e2")])
        history = external_clarification_history(grouped)
        self.assertEqual([row["sources"] for row in history["contracts"]], [["e1"], ["e2"]])

    def test_repository_probe_error_stays_needs_review_and_m1_requires_probe(self):
        validation = {"BASELINE": "unmet", "REFERENCE": "pass", "VERDICT": "accept",
                      "TESTS": "unavailable", "MUTATIONS": "unavailable", "COVERAGE": "complete"}
        self.assertFalse(admission(validation, {"status": "unavailable"},
                                   {"status": "unavailable"}, task_type="M1"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            baseline.mkdir()
            spec = root / "spec"
            spec.mkdir()
            (spec / "task.md").write_text("Implement the task.")
            with patch("dialogue_benchmark.task_eval.run.run_agent",
                       return_value={"status": "error", "error_type": "timeout"}), \
                 patch("dialogue_benchmark.task_eval.run.run_checks",
                       return_value={"status": "error"}):
                probe = repository_design_probe(
                    root / "task", baseline, spec,
                    {"execution_image": "image", "code": {}},
                    {"max_requests": 1, "max_tokens": 10, "max_seconds": 1})
            self.assertEqual(probe["status"], "needs_review")

    def test_raw_token_delta_requires_complete_usage_but_other_costs_survive_failure(self):
        pair = {
            "without_memory": {"result": "failed", "metrics": {
                "tool_calls": 8, "file_view_calls": 4, "shell_read_or_search_calls": 3,
                "total_tokens": 100, "usage_complete": True}},
            "with_memory": {"result": "passed", "metrics": {
                "tool_calls": 5, "file_view_calls": 2, "shell_read_or_search_calls": 1,
                "total_tokens": 60, "usage_complete": False}},
        }
        compared = compare_trials(pair)
        self.assertEqual(compared["raw_cost_differences"]["tool_calls"], -3)
        self.assertEqual(compared["raw_cost_differences"]["file_view_calls"], -2)
        self.assertEqual(compared["raw_cost_differences"]["shell_read_or_search_calls"], -2)
        self.assertIsNone(compared["raw_cost_differences"]["total_tokens"])
        self.assertIsNone(compared["comparable_cost_differences"]["tool_calls"])

    def test_implementation_pollution_finds_frozen_inputs(self):
        self.assertEqual(
            implementation_pollution([
                "src/app.py", "tests/test_app.py", "fixtures/input.json",
                "docs/acceptance.md", "src/testdata/input.json", "README.md"]),
            ["docs/acceptance.md", "fixtures/input.json", "src/testdata/input.json",
             "tests/test_app.py"])
        self.assertEqual(implementation_pollution(["src/testdata_parser.py"]), [])

    def test_completion_difference_waits_for_both_outcomes(self):
        self.assertIsNone(compare_trials({})["completion_difference"])
        for condition, other in (("without_memory", "with_memory"), ("with_memory", "without_memory")):
            for outcome in ("passed", "failed", "uncertain"):
                with self.subTest(condition=condition, outcome=outcome):
                    pair = {condition: {"result": outcome}}
                    self.assertIsNone(compare_trials(pair)["completion_difference"])
                    pair[other] = {"result": "pending"}
                    self.assertIsNone(compare_trials(pair)["completion_difference"])

    def test_completed_outcomes_keep_their_existing_completion_difference(self):
        for left, right, expected in (
            ("passed", "passed", 0), ("passed", "failed", -1), ("passed", "uncertain", -1),
            ("failed", "passed", 1), ("failed", "failed", 0), ("failed", "uncertain", 0),
            ("uncertain", "passed", 1), ("uncertain", "failed", 0), ("uncertain", "uncertain", 0)):
            with self.subTest(left=left, right=right):
                pair = {"without_memory": {"result": left}, "with_memory": {"result": right}}
                self.assertEqual(compare_trials(pair)["completion_difference"], expected)

    def test_actions_and_usage_are_not_double_counted(self):
        events = [{"kind": "ActionEvent", "tool_call_id": "c", "tool_name": "file_editor",
                   "action": {"command": "view"}},
                  {"kind": "ObservationEvent", "tool_call_id": "c", "tool_name": "file_editor",
                   "observation": {"content": [{"text": "hello"}]}}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "provider.jsonl"
            usage = {"prompt_tokens": 10, "completion_tokens": 4}
            path.write_text(json.dumps({"kind": "response", "usage": usage}) + "\n")
            result = measure(events, path)
            self.assertEqual(result["tool_calls"], 1)
            self.assertEqual(result["file_view_chars"], 5)
            self.assertEqual(result["total_tokens"], 14)
            self.assertIsNone(result["reasoning_chars"])
            path.write_text(json.dumps({"kind": "response"}) + "\n")
            self.assertFalse(measure(events, path)["usage_complete"])

    def test_shell_read_observations_are_separate_from_file_views(self):
        events = [{"kind": "ActionEvent", "tool_call_id": "c", "tool_name": "terminal",
                   "action": {"command": "cd /workspace/candidate && cat src/a.py"}},
                  {"kind": "ObservationEvent", "tool_call_id": "c", "tool_name": "terminal",
                   "observation": {"content": [{"text": "some source"}]}}]
        with tempfile.TemporaryDirectory() as directory:
            result = measure(events, Path(directory) / "missing.jsonl")
        self.assertEqual(result["file_view_calls"], 0)
        self.assertEqual(result["shell_read_or_search_calls"], 1)
        self.assertEqual(result["shell_read_or_search_output_chars"], 11)
        self.assertFalse(result["file_read_count_complete"])

    def test_qa_input_is_joined_not_reconstructed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "stages").mkdir()
            question = {"type": "constraint_followthrough", "id": "g1_q1", "status": "approved", "question": "Question"}
            (root / "qa-public.json").write_text(json.dumps({"questions": [question]}))
            self.assertEqual(qa_inputs(root), [])
            (root / "stages/group-raw-candidates.json").write_text(json.dumps({"questions": [question]}))
            (root / "stages/group-qa-input.json").write_text(json.dumps({"payload": "exact input"}))
            result = qa_inputs(root)
            self.assertEqual(len(result), 1)
            self.assertEqual(json.loads(Path(result[0]["generation_input"]).read_text())["payload"], "exact input")

    def test_provisional_qa_can_feed_tasks_without_becoming_approved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "stages").mkdir()
            question = {
                "type": "constraint_followthrough", "id": "g1_q1",
                "status": "needs_review", "question": "Question",
            }
            (root / "qa-public.json").write_text(json.dumps({"questions": []}))
            (root / "qa-candidates.json").write_text(
                json.dumps({"questions": [question]}))
            (root / "stages/group-raw-candidates.json").write_text(
                json.dumps({"questions": [question]}))
            (root / "stages/group-qa-input.json").write_text(
                json.dumps({"payload": "exact input"}))
            self.assertEqual(qa_inputs(root), [])
            result = qa_inputs(root, include_provisional=True)
            self.assertEqual(len(result), 1)
            self.assertTrue(result[0]["provisional"])
            self.assertEqual(result[0]["qa"]["status"], "needs_review")

    def test_plain_message_normalized_input_is_public_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "stages").mkdir()
            question = {"type": "constraint_followthrough", "id": "g1_q1",
                        "status": "approved", "question": "Question"}
            (root / "qa-public.json").write_text(json.dumps({"questions": [question]}))
            (root / "stages/group-raw-candidates.json").write_text(
                json.dumps({"questions": [question]}))
            (root / "stages/group-qa-input.json").write_text(json.dumps({"payload": "exact"}))
            (root / "normalized.json").write_text(json.dumps([{
                "kind": "message", "role": "user", "id": "e1", "text": "public"
            }]))
            result = qa_inputs(root)
            self.assertEqual(result[0]["public_records"][0]["original_id"], "e1")
            self.assertEqual(result[0]["public_records"][0]["input_schema"],
                             "model-visible-dialogue-v1")

    def test_unresolved_graph_type_does_not_block_other_task_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "stages").mkdir()
            question = {"type": None, "type_status": "unresolved", "id": "g1_q1",
                        "status": "approved", "question": "Which agreement applies?"}
            (root / "qa-public.json").write_text(json.dumps({"questions": [question]}))
            (root / "stages/group-raw-candidates.json").write_text(
                json.dumps({"questions": [question]}))
            (root / "stages/group-qa-input.json").write_text(json.dumps({"payload": "exact"}))
            self.assertEqual(qa_inputs(root)[0]["qa"]["type_status"], "unresolved")
            (root / "manifest.json").write_text(json.dumps({"qa_source": "external"}))
            # The external source still enforces its source contract; this
            # deliberately incomplete fixture is rejected for missing
            # answer evidence, not because the type is unresolved.
            self.assertEqual(qa_inputs(root), [])

    def test_evidence_contract_is_bound_without_inventing_judgments(self):
        class Client:
            def ask(self, prompt, data):
                # Target routing is exercised separately in test_memory_types.
                if "review_contract: target_v1" in prompt:
                    return {"reviews": [{"id": "q1", "review_contract": "target_v1",
                                         "target_alignment": "aligned"}]}
                return {"reviews": [{"id": "q1", "point_evidence": "A1=insufficient"}]}
        document = _ask_stage(Client(), "review_contract: simple_v1", {}, "review_evidence")
        self.assertEqual(document["reviews"][0]["review_contract"], "simple_v1")
        self.assertEqual(document["reviews"][0]["point_evidence"], "A1=insufficient")
        document = _ask_stage(Client(), "review_contract: structured_v2", {}, "review_evidence")
        self.assertNotIn("review_contract", document["reviews"][0])


if __name__ == "__main__":
    unittest.main()
