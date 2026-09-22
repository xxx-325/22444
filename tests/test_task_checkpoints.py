"""Reference-derived checkpoints and solver-only trajectory comparisons."""

from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from dialogue_benchmark.task_eval.artifacts import read, save, fingerprint
from dialogue_benchmark.task_eval.checkpoints import extract_checkpoints, match_checkpoints, project_trajectory, trajectory_payload
from dialogue_benchmark.task_eval.metrics import checkpoint_summary, compare_checkpoints
from dialogue_benchmark.task_eval.report import write_report
from dialogue_benchmark.task_eval.run import evaluate, freeze, recover_checkpoint_failures


def trace(command="cat src/stream.py"):
    return [{"id": "solver-action", "kind": "ActionEvent", "tool_call_id": "call1",
             "tool_name": "terminal", "thought": "Private reasoning",
             "action": {"command": command}},
            {"id": "solver-result", "kind": "ObservationEvent", "tool_call_id": "call1",
             "observation": {"content": [{"text": "def close(): flush()"}],
                             "is_error": False, "metadata": {"exit_code": 0}}}]


def summary(*statuses):
    return checkpoint_summary([{"index": i, "status": status, "checkpoint": "Inspect object %s" % i,
                                "evidence": "Actual solver action", "sources": [{"action_id": "solver-action"}]}
                               for i, status in enumerate(statuses, 1)], len(statuses))


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoints = [{"index": 1, "text": "Inspect close behavior",
                             "reference_evidence": [{"action_id": "reference-action"}]}]

    def test_projection_preserves_real_results_without_reasoning_or_editor_duplicates(self):
        events = trace()
        events[0]["action"]["old_str"] = "Duplicate old code"
        events[1]["observation"]["old_content"] = "Full duplicate file"
        steps, sources = project_trajectory(events)
        self.assertNotIn("Private reasoning", str(steps))
        self.assertNotIn("Duplicate", str(steps))
        self.assertNotIn("Full duplicate", str(steps))
        self.assertEqual(steps[0]["results"][0]["exit_code"], 0)
        self.assertEqual(sources["1"], {"action_id": "solver-action", "observation_ids": ["solver-result"]})

    def test_projection_marks_omitted_output_and_does_not_support_unreturned_calls(self):
        events = trace()
        events[1]["observation"]["content"] = "x" * 6000
        steps, _ = project_trajectory(events)
        self.assertIn("[Middle output omitted]", steps[0]["results"][0]["text"])
        _, sources = project_trajectory(events[:1])
        self.assertEqual(sources, {})

    def test_projection_redacts_credential_assignments_without_changing_original_trace(self):
        from dialogue_benchmark.security import credential_detected
        events = trace("python check.py")
        original = 'token = normalize(value)\npassword = "sample-password"'
        events[1]["observation"]["content"] = original
        steps, sources = project_trajectory(events)
        self.assertIn("<credential assignment omitted>", steps[0]["results"][0]["text"])
        self.assertFalse(credential_detected(json.dumps(steps)))
        self.assertEqual(events[1]["observation"]["content"], original)
        self.assertEqual(sources["1"]["action_id"], "solver-action")

    def test_empty_password_prompt_survives_serialized_credential_guard(self):
        from dialogue_benchmark.security import credential_detected
        events = trace("cat tests/test_prompt.py")
        original = 'assert output == "Password: "'
        events[1]["observation"]["content"] = original
        self.assertTrue(credential_detected(json.dumps(original)))
        steps, _ = project_trajectory(events)
        self.assertFalse(credential_detected(json.dumps(steps)))
        self.assertEqual(events[1]["observation"]["content"], original)
        events[1]["observation"]["content"] = "Bearer abcdefghijklmnop"
        steps, _ = project_trajectory(events)
        self.assertTrue(credential_detected(json.dumps(steps)))

    def test_long_trajectory_fits_budget_without_losing_actions_or_source_ids(self):
        events = []
        for i in range(80):
            pair = trace("cat src/file%d.py" % i)
            for event in pair:
                event["id"] += str(i)
                event["tool_call_id"] += str(i)
            pair[1]["observation"]["content"] = "start" + "x" * 6000 + "end"
            events.extend(pair)
        payload, sources = trajectory_payload({"task": "Requirement"}, events, "Extract")
        self.assertLessEqual(len(json.dumps(payload, ensure_ascii=False)) + len("Extract"), 54000)
        self.assertEqual(len(payload["steps"]), 80)
        self.assertEqual(set(sources), {str(i) for i in range(1, 81)})
        self.assertEqual(payload["steps"][-1]["action"]["command"], "cat src/file79.py")
        text = payload["steps"][-1]["results"][0]["text"]
        self.assertIn("[Middle output omitted]", text)
        self.assertTrue(text.startswith("start") and text.endswith("end"))
        self.assertEqual(len(events[-1]["observation"]["content"]), 6008)

    def test_extraction_numbers_checkpoints_statically_and_binds_actual_reference_evidence(self):
        facts = [{"statement": "Inspect close behavior", "sources": ["1"]}] * 2
        with patch("dialogue_benchmark.task_eval.checkpoints.ask_model", return_value={"facts": facts}) as ask:
            result = extract_checkpoints("New requirement", trace(), {}, self.root)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["checkpoints"]), 1)
        self.assertEqual(result["checkpoints"][0]["index"], 1)
        self.assertEqual(result["checkpoints"][0]["reference_evidence"][0]["action_id"], "solver-action")
        self.assertEqual(set(ask.call_args.args[1]), {"task", "steps"})

    def test_fabricated_reference_steps_cannot_be_frozen(self):
        with patch("dialogue_benchmark.task_eval.checkpoints.ask_model",
                   return_value={"facts": [{"statement": "Invented action", "sources": ["999"]}]}):
            result = extract_checkpoints("Task", trace(), {}, self.root)
        self.assertEqual(result["status"], "failed")
        self.assertNotIn("checkpoints", result)

    def test_matching_only_receives_this_solver_trace_and_unadorned_checkpoint_text(self):
        response = {"reviews": [{"id": "1", "status": "observed", "sources": "1",
                                  "evidence": "Read the close function"}]}
        with patch("dialogue_benchmark.task_eval.checkpoints.ask_model", return_value=response) as ask:
            result = match_checkpoints(self.checkpoints, trace(), {}, self.root, trajectory_complete=True)
        payload = ask.call_args.args[1]
        self.assertEqual(set(payload), {"checkpoints", "steps"})
        self.assertNotIn("reference-action", str(payload))
        self.assertEqual(result["action_coverage"], 1)
        self.assertEqual(result["rows"][0]["sources"][0]["action_id"], "solver-action")

    def test_judge_or_other_trial_evidence_is_uncertain_not_observed(self):
        response = {"reviews": [{"id": "1", "status": "observed", "sources": "judge-action",
                                  "evidence": "Judge ran a test"}]}
        with patch("dialogue_benchmark.task_eval.checkpoints.ask_model", return_value=response):
            result = match_checkpoints(self.checkpoints, trace(), {}, self.root, trajectory_complete=True)
        self.assertEqual(result["counts"]["uncertain"], 1)
        self.assertIsNone(result["action_coverage"])

    def test_missing_row_only_affects_that_checkpoint(self):
        checkpoints = self.checkpoints + [{"index": 2, "text": "Reproduce the failure"}]
        with patch("dialogue_benchmark.task_eval.checkpoints.ask_model",
                   return_value={"reviews": [{"id": "1", "status": "skipped", "sources": "none",
                                               "evidence": "Not present in the complete trace"}]}):
            result = match_checkpoints(checkpoints, trace(), {}, self.root, trajectory_complete=True)
        self.assertEqual(result["counts"]["skipped"], 1)
        self.assertEqual(result["counts"]["uncertain"], 1)
        self.assertFalse(result["resolved"])

    def test_incomplete_trajectory_or_failed_matching_cannot_imply_steps_saved(self):
        with patch("dialogue_benchmark.task_eval.checkpoints.ask_model") as ask:
            result = match_checkpoints(self.checkpoints, trace(), {}, self.root, trajectory_complete=False)
        ask.assert_not_called()
        self.assertEqual(result["counts"]["uncertain"], 1)
        with patch("dialogue_benchmark.task_eval.checkpoints.ask_model", side_effect=TimeoutError):
            result = match_checkpoints(self.checkpoints, trace(), {}, self.root, trajectory_complete=True)
        self.assertIsNone(result["action_coverage"])
        self.assertIn("error", read(self.root / "result.json"))

    def test_pair_comparison_requires_correctness_and_resolved_matching(self):
        pair = {"without_memory": {"result": "passed", "checkpoints": summary("observed", "observed", "skipped")},
                "with_memory": {"result": "passed", "checkpoints": summary("skipped", "alternative", "observed")}}
        result = compare_checkpoints(pair)
        self.assertTrue(result["eligible"])
        self.assertAlmostEqual(result["coverage_reduction"], 1 / 3)
        self.assertEqual(result["skipped_in_memory"], [1])
        self.assertEqual(result["alternative_in_memory"], [2])
        self.assertEqual(result["additional_in_memory"], [3])
        pair["with_memory"]["result"] = "failed"
        self.assertFalse(compare_checkpoints(pair)["eligible"])
        pair["with_memory"].update(result="passed", checkpoints=summary("skipped", "uncertain", "observed"))
        self.assertFalse(compare_checkpoints(pair)["eligible"])
        self.assertIsNone(summary()["action_coverage"])

    def test_report_shows_counts_individual_actions_and_cited_evidence(self):
        pair = {"without_memory": {"trial": "trial-1", "result": "passed", "checkpoints": summary("observed", "observed")},
                "with_memory": {"trial": "trial-2", "result": "passed", "checkpoints": summary("skipped", "observed")}}
        write_report(self.root, {"tasks": [{"task": "task-01", "status": "evaluated", "comparison": pair}]})
        report = (self.root / "report.md").read_text()
        for text in ("2/2 (100%)", "1/2 (50%)", "50.0 percentage points", "Inspect object 1", "solver-action"):
            self.assertIn(text, report)
        pair["with_memory"]["result"] = "failed"
        write_report(self.root, {"tasks": [{"task": "task-01", "status": "evaluated", "comparison": pair}]})
        report = (self.root / "report.md").read_text()
        self.assertNotIn("coverage reduction:", report)
        self.assertIn("task_not_passed", report)

    def test_evaluation_keeps_correctness_when_checkpoint_request_fails(self):
        from dialogue_benchmark.task_eval.versions import pin_baseline
        baseline, spec, task_root = self.root / "baseline", self.root / "spec", self.root / "task"
        baseline.mkdir()
        spec.mkdir()
        (baseline / "a.py").write_text("x = 1\n")
        pin_baseline(baseline)
        for name in ("task.md", "acceptance.md", "checkpoints.md"):
            (spec / name).write_text("Private checkpoint" if name == "checkpoints.md" else "Public requirement")
        save(spec / "checkpoints.json", {"checkpoints": self.checkpoints})
        receipt = freeze(spec, task_root / "frozen", baseline)
        receipt["validation"] = {"TESTS": "executable"}

        def agent(root, config, role, message, **kwargs):
            if role == "code":
                self.assertNotIn("Private checkpoint", message)
                self.assertNotIn("Inspect close", message)
                save(root / "trajectory.json", trace())
            else:
                checks = root / "workspace/checks"
                checks.mkdir()
                (checks / "verdict.txt").write_text("RESULT: passed\n")
            return {"status": "finished", "metrics": {}}

        with patch("dialogue_benchmark.task_eval.run.run_agent", side_effect=agent), \
             patch("dialogue_benchmark.task_eval.run.run_checks", return_value={"status": "passed"}), \
             patch("dialogue_benchmark.task_eval.checkpoints.ask_model", side_effect=TimeoutError):
            result = evaluate({"qa": {"answer_points": ["Old experience"]}}, task_root, baseline,
                              receipt, {"execution_image": "unused"}, {}, 0)
        self.assertEqual([trial["result"] for trial in result.values()], ["passed", "passed"])
        self.assertFalse(read(task_root / "checkpoint-comparison.json")["eligible"])

    def test_recovery_reuses_validated_code_and_rechecks_before_scored_trials(self):
        from dialogue_benchmark.task_eval.versions import pin_baseline
        baseline = self.root / "baseline"
        baseline.mkdir()
        (baseline / "a.py").write_text("x = 1\n")
        version = pin_baseline(baseline)
        task = self.root / "task-01"
        run = task / "construction-00"
        candidate = run / "reference-solver/workspace/candidate"
        candidate.mkdir(parents=True)
        (candidate / "a.py").write_text("x = 2\n")
        save(run / "reference-solver/trajectory.json", trace())
        spec = run / "validated-spec"
        spec.mkdir()
        for name in ("task.md", "acceptance.md"):
            (spec / name).write_text("Public requirement")
        qa = {"id": "q1", "answer_points": ["Historical answer"]}
        save(task / "author-reference/qa.json", qa)
        validation = {"BASELINE": "unmet", "REFERENCE": "pass", "TESTS": "executable",
                      "MUTATIONS": "caught", "COVERAGE": "complete", "VERDICT": "accept"}
        save(task / "construction.json", [{"attempt": 0, "validation_accepted": True,
             "validation": validation, "reason": "checkpoint_extraction_failed",
             "reference_version": {"candidate_sha256": fingerprint(candidate)}}])
        save(self.root / "manifest.json", {"baseline": str(baseline), "baseline_version": version,
             "qa_run": "unused", "target": 1, "tasks": [{"task": "task-01", "status": "not_admitted", "qa_id": "q1"}]})
        extracted = {"status": "completed", "checkpoints": self.checkpoints}
        with patch("dialogue_benchmark.task_eval.run.qa_inputs", return_value=[{"qa": qa}]), \
             patch("dialogue_benchmark.task_eval.run.run_checks", side_effect=[{"status": "failed"}, {"status": "passed"}]) as checks, \
             patch("dialogue_benchmark.task_eval.run.extract_checkpoints", return_value=extracted), \
             patch("dialogue_benchmark.task_eval.run.evaluate", side_effect=lambda *a: (save(task / "comparison.json", {}) or {})) as evaluate, \
             patch("dialogue_benchmark.task_eval.run.construct") as author:
            recover_checkpoint_failures(self.root, {"execution_image": "unused"}, {})
            recover_checkpoint_failures(self.root, {"execution_image": "unused"}, {})
        author.assert_not_called()
        self.assertEqual(checks.call_count, 2)
        self.assertEqual(evaluate.call_count, 1)
        self.assertTrue((task / "frozen/checkpoints.json").exists())
        self.assertEqual(read(self.root / "manifest.json")["completed"], 1)

    def test_matching_recovery_keeps_solver_run_metrics_and_original_review(self):
        from dialogue_benchmark.task_eval.versions import pin_baseline
        baseline = self.root / "baseline"
        baseline.mkdir()
        version = pin_baseline(baseline)
        spec = self.root / "spec"
        spec.mkdir()
        for name in ("task.md", "acceptance.md", "checkpoints.md"):
            (spec / name).write_text("Frozen content")
        save(spec / "checkpoints.json", {"checkpoints": self.checkpoints})
        task = self.root / "task-01"
        save(task / "frozen.json", freeze(spec, task / "frozen", baseline))
        trial = task / "trial-1"
        save(trial / "trajectory.json", trace())
        failed = {"error": {"error_code": "request_budget"}}
        save(trial / "checkpoint-review/result.json", failed)
        comparison = {"without_memory": {"trial": "trial-1", "result": "passed",
            "solver_status": "finished", "metrics": {"tool_calls": 7}, "checkpoints": failed}}
        save(task / "comparison.json", comparison)
        save(self.root / "manifest.json", {"baseline": str(baseline), "baseline_version": version,
            "qa_run": "unused", "tasks": [{"task": "task-01", "status": "evaluated"}]})
        with patch("dialogue_benchmark.task_eval.run.qa_inputs", return_value=[]), \
             patch("dialogue_benchmark.task_eval.run.match_checkpoints", return_value=summary("observed")) as match, \
             patch("dialogue_benchmark.task_eval.run.run_agent") as agent:
            recover_checkpoint_failures(self.root, {}, {})
        agent.assert_not_called()
        self.assertEqual(match.call_count, 1)
        result = read(task / "comparison.json")["without_memory"]
        self.assertEqual(result["metrics"], {"tool_calls": 7})
        self.assertEqual(result["result"], "passed")
        self.assertEqual(read(trial / "checkpoint-review/result.json"), failed)
        self.assertEqual(read(trial / "trajectory.json"), trace())


if __name__ == "__main__":
    unittest.main()
