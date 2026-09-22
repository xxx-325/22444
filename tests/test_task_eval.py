import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from dialogue_benchmark.llm import _ask_stage
from dialogue_benchmark.task_eval.artifacts import qa_inputs
from dialogue_benchmark.task_eval.checks import pytest_result
from dialogue_benchmark.task_eval.metrics import checkpoint_summary, measure
from dialogue_benchmark.task_eval.runtime import configure, readable_reference, release_completed_execution, run_agent
from dialogue_benchmark.task_eval.run import (admission, freeze, solver_input,
                                              trial_status, unchanged, validated_spec)


class TaskEvaluationTests(unittest.TestCase):
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

    def test_frozen_content_change_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, spec = root / "base", root / "spec"
            baseline.mkdir()
            spec.mkdir()
            (baseline / "a.py").write_text("x = 1\n")
            for name in ("task.md", "checkpoints.md", "acceptance.md"):
                (spec / name).write_text("1. Explicit action\n")
            (spec / "checkpoints.json").write_text('{"checkpoints": []}')
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

    def test_unresolved_coverage_and_skipped_tests_prevent_admission(self):
        decision = {"BASELINE": "unmet", "REFERENCE": "pass", "VERDICT": "accept",
                    "TESTS": "executable", "MUTATIONS": "caught", "COVERAGE": "complete"}
        for field, value in (("COVERAGE", "gaps"), ("COVERAGE", "uncertain"),
                             ("COVERAGE", None), ("MUTATIONS", "missed")):
            self.assertFalse(admission(dict(decision, **{field: value}),
                                       {"status": "failed"}, {"status": "passed"}))
        self.assertFalse(admission(decision, {"status": "failed"}, {"status": "passed", "skipped": 1}))
        self.assertFalse(admission(decision, {"status": "error"}, {"status": "passed"}))

    def test_execution_error_cannot_become_judge_fallback_pass(self):
        done = {"status": "ConversationExecutionStatus.FINISHED"}
        self.assertEqual(trial_status("RESULT: passed", {"status": "error"}, done, "partial"), "uncertain")
        self.assertEqual(trial_status("RESULT: passed", {"status": "unavailable"}, done, "executable"), "uncertain")
        self.assertEqual(trial_status("RESULT: passed", {"status": "passed"}, {"status": "error"}, "executable"), "uncertain")
        self.assertEqual(trial_status("RESULT: passed", {"status": "unavailable"}, done, "unavailable"), "passed")
        self.assertEqual(trial_status("RESULT: passed", {"status": "failed"}, done, "executable"), "failed")

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

    def test_timeout_and_invalid_junit_are_errors_not_unavailable_tests(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.xml"
            self.assertEqual(pytest_result(124, path)["status"], "error")
            path.write_text("broken xml")
            self.assertEqual(pytest_result(0, path)["status"], "error")
            path.write_text("<testsuite><testcase /></testsuite>")
            self.assertEqual(pytest_result(3, path)["status"], "error")

    def test_skipped_checkpoints_are_not_correctness_failures(self):
        result = checkpoint_summary([{"index": 1, "status": "skipped", "evidence": "Not seen"},
                                     {"index": 2, "status": "alternative", "evidence": "Event a"}], 2)
        self.assertTrue(result["complete"])
        self.assertEqual(result["action_coverage"], 0)
        self.assertNotIn("passed", result)
        self.assertFalse(checkpoint_summary([{"index": 1, "status": "observed"},
                                            {"index": 1, "status": "observed"}], 2)["complete"])

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
            question = {"id": "g1_q1", "status": "approved", "question": "Question"}
            (root / "qa-public.json").write_text(json.dumps({"questions": [question]}))
            self.assertEqual(qa_inputs(root), [])
            (root / "stages/group-raw-candidates.json").write_text(json.dumps({"questions": [question]}))
            (root / "stages/group-qa-input.json").write_text(json.dumps({"payload": "exact input"}))
            result = qa_inputs(root)
            self.assertEqual(len(result), 1)
            self.assertEqual(json.loads(Path(result[0]["generation_input"]).read_text())["payload"], "exact input")

    def test_evidence_contract_is_bound_without_inventing_judgments(self):
        class Client:
            def ask(self, prompt, data):
                return {"reviews": [{"id": "q1", "point_evidence": "A1=insufficient"}]}
        document = _ask_stage(Client(), "review_contract: simple_v1", {}, "review_evidence")
        self.assertEqual(document["reviews"][0]["review_contract"], "simple_v1")
        self.assertEqual(document["reviews"][0]["point_evidence"], "A1=insufficient")
        document = _ask_stage(Client(), "review_contract: structured_v2", {}, "review_evidence")
        self.assertNotIn("review_contract", document["reviews"][0])


if __name__ == "__main__":
    unittest.main()
