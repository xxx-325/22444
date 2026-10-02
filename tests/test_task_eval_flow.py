"""Offline orchestration contracts for task preflight and paired evaluation."""

import runpy
import shlex
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch
from types import SimpleNamespace

from dialogue_benchmark.llm import parse_text_response
from dialogue_benchmark.task_eval.artifacts import copy_tree, read, save, fingerprint, qa_fingerprint
from dialogue_benchmark.task_eval.checks import acceptance_items, assess_acceptance, check_history_mutations, run_checks
from dialogue_benchmark.task_eval.run import construct, evaluate, freeze, inspect_acceptance, prepare_test_reference
from dialogue_benchmark.task_eval.runtime import review_checks, review_task, write_history_mutation
from dialogue_benchmark.task_eval.versions import export_change, pin_baseline


@contextmanager
def local_command_checks():
    """Run frozen shell checks locally while preserving real Git patch operations."""
    real_run = subprocess.run
    workspaces = []

    def sandbox(directory, workspace, image, role, identity, reference):
        workspaces.append(workspace)
        return SimpleNamespace(name="local-checks", record={}, prepare=lambda: None,
                               unpause=lambda: None, pause=lambda: None)

    def execute(argv, **kwargs):
        if argv[0] != "docker":
            return real_run(argv, **kwargs)
        workspace = workspaces[-1]
        script = workspace / "checks/commands" / Path(argv[-1]).name
        return real_run(["bash", str(script)], cwd=workspace / "candidate", **kwargs)

    with patch.dict("sys.modules", {"simulator.openhands.sandbox": SimpleNamespace(ExecutionSandbox=sandbox)}), \
         patch("dialogue_benchmark.task_eval.checks.release_completed_execution"), \
         patch("dialogue_benchmark.task_eval.checks.subprocess.run", side_effect=execute):
        yield


class TaskPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.baseline = self.base / "baseline"
        self.baseline.mkdir()
        (self.baseline / "a.py").write_text("value = 1\n")
        pin_baseline(self.baseline)
        input_path = self.base / "qa-input.json"
        save(input_path, {"payload": "Actual evidence"})
        self.item = {"qa": {"type": "constraint_followthrough", "id": "q1", "answer_points": [{"text": "Historical behavior"}]},
                     "generation_input": str(input_path)}
        self.root = self.base / "task"
        self.with_coverage = True
        self.validator_status = "ConversationExecutionStatus.FINISHED"
        self.reference_status = "ConversationExecutionStatus.FINISHED"
        for target, kwargs in (("select_task", {"return_value": {"status": "candidate"}}),
                               ("write_draft", {"side_effect": self.fake_draft})):
            mocked = patch("dialogue_benchmark.task_eval.run." + target, **kwargs)
            mocked.start()
            self.addCleanup(mocked.stop)

    def fake_draft(self, selection, config, output, spec, budget, feedback=""):
        spec.mkdir(parents=True, exist_ok=True)
        for name, text in (("task.md", "Preserve pending data after replacement"),
                           ("memory-use.md", "A historical constraint changes the output."),
                           ("acceptance.md", "| a1 | Pending data remains readable | task | test: test_acceptance::test_feature |")):
            (spec / name).write_text(text)

    def fake_agent(self, root, config, role, message, **kwargs):
        if root.name == "author":
            checks = root / "workspace/checks"
            checks.mkdir(exist_ok=True)
            for name, text in (("task.md", "Preserve pending data after replacement"),
                               ("memory-use.md", "A historical constraint changes the output."),
                               ("acceptance.md", "| a1 | Pending data remains readable | task | test: test_acceptance::test_feature |"),
                               ("test_acceptance.py", "Original test")):
                (checks / name).write_text(text)
        elif root.name == "reference-solver":
            save(root / "trajectory.json", [{"id": "reference-action"}])
            return {"status": self.reference_status}
        elif root.name == "validator":
            checks = root / "workspace/checks"
            checks.mkdir()
            (checks / "validation.txt").write_text(
                "BASELINE: unmet\nREFERENCE: pass\nTESTS: executable\n"
                "MUTATIONS: caught\nCOVERAGE: complete\nVERDICT: accept\n")
            if self.with_coverage:
                (checks / "coverage.md").write_text("Combine writing and replacement")
                (checks / "test_interactions.py").write_text("Combined-condition test")
            return {"status": self.validator_status}
        return {"status": "ConversationExecutionStatus.FINISHED"}

    def execute(self, results, **options):
        results = [dict(r, cases=[{"id": "test_acceptance::test_feature", "status": r["status"]}]) for r in results]
        with patch("dialogue_benchmark.task_eval.run.run_agent", side_effect=self.fake_agent), \
             patch("dialogue_benchmark.task_eval.run.review_task", return_value={"status": "clean", "issue": "none"}), \
             patch("dialogue_benchmark.task_eval.run.run_checks", side_effect=results) as checks:
            receipt = construct(self.item, self.root, self.baseline, {"execution_image": "image"}, 0, {}, **options)
        return receipt, checks

    def test_paired_evaluation_inspects_each_actual_delivery_with_the_shared_judge(self):
        spec = self.base / "spec"
        spec.mkdir()
        (spec / "task.md").write_text("Deliver the order status report.")
        (spec / "acceptance.md").write_text(
            "| a1 | Report the recorded status | task | inspect: Read report.txt; one must be delivered |\n")
        save(spec / "acceptance.json", acceptance_items(spec))
        receipt = freeze(spec, self.root / "frozen", self.baseline)

        def agent(root, config, role, message, **options):
            if role == "code":
                text = "one: failed\n" if root.name == "trial-1" else "one: delivered\n"
                (root / "workspace/candidate/report.txt").write_text(text)
                return {"status": "finished", "metrics": {}}
            self.assertEqual(role, "judge")
            text = (root / "workspace/candidate/report.txt").read_text()
            status = "passed" if text == "one: delivered\n" else "failed"
            checks = root / "workspace/checks"
            checks.mkdir()
            (checks / "acceptance-review.txt").write_text(
                "REVIEW a1\nstatus: %s\nevidence: /workspace/candidate/report.txt:1\nEND_REVIEW" % status)
            return {"status": "finished"}

        with patch("dialogue_benchmark.task_eval.run.run_agent", side_effect=agent), \
             patch("dialogue_benchmark.task_eval.run.run_checks", return_value={"status": "passed", "cases": []}), \
             patch("dialogue_benchmark.task_eval.run.inspect_acceptance", wraps=inspect_acceptance) as inspector:
            result = evaluate(self.item, self.root, self.baseline, receipt, {"execution_image": "image"}, {}, 0)
        self.assertEqual(inspector.call_count, 2)
        self.assertEqual(result["without_memory"]["result"], "failed")
        self.assertEqual(result["with_memory"]["result"], "passed")
        self.assertFalse(result["with_memory"]["history_available"])

    def test_changed_qa_is_rejected_before_evaluation(self):
        receipt = {"qa_sha256": qa_fingerprint(self.item["qa"])}
        self.item["qa"]["answer_points"] = ["Different historical answer"]
        with patch("dialogue_benchmark.task_eval.run.run_agent") as agent:
            with self.assertRaisesRegex(ValueError, "Frozen QA changed"):
                evaluate(self.item, self.root, self.baseline, receipt, {}, {}, 0)
        agent.assert_not_called()

    def test_external_rejected_draft_returns_specific_feedback_to_selector(self):
        self.item["qa_source"] = "external"
        def draft(selection, config, output, spec, budget, feedback=""):
            self.fake_draft(selection, config, output, spec, budget, feedback)
            save(spec / "oracle-answer.json", {"answer": "- Historical behavior"})
            with (spec / "acceptance.md").open("a") as handle:
                handle.write("\n| a2 | Known history | answer | inspect: Check saved historical behavior |\n")
        with patch("dialogue_benchmark.task_eval.run.write_draft", side_effect=draft), \
             patch("dialogue_benchmark.task_eval.run.select_task", side_effect=[
                 {"status": "candidate"}, {"status": "stop", "reason": "No grounded replacement"}]) as selector, \
             patch("dialogue_benchmark.task_eval.run.review_task", return_value={
                 "status": "ineligible", "issue": "All required rules already public"}), \
             patch("dialogue_benchmark.task_eval.run.run_agent") as agent:
            result = construct(self.item, self.root, self.baseline, {}, 1, {})
        self.assertIsNone(result)
        self.assertEqual(selector.call_count, 2)
        self.assertEqual(selector.call_args.kwargs["feedback"]["review"]["issue"],
                         "All required rules already public")
        self.assertIn("Preserve pending", selector.call_args.kwargs["feedback"]["public_task"])
        self.assertEqual(read(self.root / "construction.json")[0]["reason"], "task_ineligible")
        agent.assert_not_called()

    def test_external_admission_replays_mutation_instead_of_trusting_validator_label(self):
        self.item["qa_source"] = "external"
        self.item["qa"]["answer_points"] = ["Preserve other null values."]
        config = {"execution_image": "image"}

        def draft(selection, config, output, spec, budget, feedback=""):
            spec.mkdir(parents=True)
            (spec / "task.md").write_text("Add a batch exporter under the agreed rules.")
            (spec / "memory-use.md").write_text("Keep other null values.")
            save(spec / "oracle-answer.json", {"answer": "- Preserve other null values."})
            (spec / "acceptance.md").write_text(
                "| a1 | Export records | task | test: test_export::test_feature |\n"
                "| a2 | Keep other nulls | answer | test: test_export::test_null |\n")

        def checks(candidate, *args, **kwargs):
            source = (candidate / "a.py").read_text()
            cases = [{"id": "test_export::test_" + name,
                      "status": "passed" if marker in source else "failed"}
                     for name, marker in (("feature", "feature = True"), ("null", "keep_nulls = True"))]
            return {"status": "passed" if all(c["status"] == "passed" for c in cases) else "failed",
                    "cases": cases, "skipped": 0}

        def agent(root, config, role, message, **options):
            if root.name == "author":
                (root / "workspace/checks/test_acceptance.py").write_text("# Fixed tests\n")
            elif root.name == "reference-solver":
                self.assertIn("Preserve other null values.", message)
                (root / "workspace/candidate/a.py").write_text("feature = True\nkeep_nulls = True\n")
            elif root.name == "validator":
                folder = root / "workspace/checks"
                folder.mkdir()
                (folder / "coverage.md").write_text("Both requirements have tests.")
                (folder / "validation.txt").write_text(
                    "BASELINE: unmet\nREFERENCE: pass\nTESTS: executable\n"
                    "MUTATIONS: caught\nCOVERAGE: complete\nVERDICT: accept\n")
                if saved_patch:
                    (folder / "mutations.txt").write_text("REVIEW m1\nacceptance: a2\nEND_REVIEW")
                    (folder / "m1.patch").write_text(
                        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
                        "@@ -1,2 +1,2 @@\n feature = True\n-keep_nulls = True\n+keep_nulls = False\n")
            return {"status": "finished", "metrics": {}}

        for saved_patch in (False, True):
            root = self.base / ("external-" + str(saved_patch))
            with patch("dialogue_benchmark.task_eval.run.write_draft", side_effect=draft), \
                 patch("dialogue_benchmark.task_eval.run.run_agent", side_effect=agent), \
                 patch("dialogue_benchmark.task_eval.run.review_task", return_value={"status": "clean"}), \
                 patch("dialogue_benchmark.task_eval.run.run_checks", side_effect=checks), \
                 patch("dialogue_benchmark.task_eval.checks.run_checks", side_effect=checks):
                receipt = construct(self.item, root, self.baseline, config, 0, {})
            self.assertEqual(receipt is not None, saved_patch)
            record = read(root / "construction.json")[0]
            self.assertEqual(record["history_mutations"]["status"], "caught" if saved_patch else "unverified")
            self.assertEqual(record["validation_accepted"], saved_patch)

    def test_failed_exploration_retains_attempted_usage(self):
        self.item["qa"]["question"] = "Which historical report rule applies?"
        for metrics, requests, complete in (
                ({"attempted_requests": 0, "usage_complete": False}, 0, True),
                ({"attempted_requests": 2, "prompt_tokens": 30,
                  "completion_tokens": 5, "usage_complete": True}, 2, True),
                ({"attempted_requests": 1, "usage_complete": False}, 1, False)):
            with self.subTest(metrics=metrics):
                root = self.base / ("exploration-%d" % requests)
                with patch("dialogue_benchmark.task_eval.run.explore_repository",
                           return_value=("", {"status": "error", "metrics": metrics})), \
                     patch("dialogue_benchmark.task_eval.run.select_task",
                           return_value={"status": "pending", "reason": "No candidate"}):
                    construct(self.item, root, self.baseline, {}, 0, {})
                budget = read(root / "selection-budget.json")
                self.assertEqual(budget["requests"], requests)
                self.assertEqual(budget["usage_complete"], complete)
                self.assertEqual(budget["total_tokens"], 35 if requests == 2 else 0)

    def test_construction_inspects_mutant_and_charges_the_same_preflight_budget(self):
        rule = "Keep other null values."
        self.item["qa"]["answer_points"] = [{"text": rule}]
        self.item["public_records"] = [dict(id="e1", original_id="choice", order=1, kind="message", text=rule)]
        save(Path(self.item["generation_input"]), {"ref_to_source": {"资料1": "e1"}})

        def draft(selection, config, output, spec, budget, feedback=""):
            self.fake_draft(selection, config, output, spec, budget, feedback)
            (spec / "history-contract.txt").write_text(
                "REVIEW h1\nstatement: " + rule + "\nscope: All exports\nsources: choice\n"
                "supersedes: none\nbehavior: Preserve other nulls\nactive: yes\nrepository: external\nEND_REVIEW")
            (spec / "acceptance.md").write_text(
                "| a1 | Export works | task | inspect: Read a.py; feature must be True |\n"
                "| a2 | Preserve other nulls | h1 | test: test_export::test_null |\n")

        def review(task, answer, config, output, *, evidence, budget):
            return {"status": "clean", "task_review": {"leakage": "clean"}, "history_rows": [
                dict(id=row["id"], applicable="yes", public="none", answer="sufficient",
                     historical_sources=row["sources"], answer_quote=rule) for row in evidence["contracts"]]}

        def coverage(spec, baseline, candidate, changed, results, config, output, budget):
            stages.append("coverage")
            self.assertFalse((output.parent / "validator").exists())
            rows = [dict(id=identity, coverage="complete", evidence=evidence) for identity, evidence in (
                ("a1", "acceptance.md specifies reading a.py and requiring feature=True."),
                ("a2", "test_export::test_null checks the historical rule."),
                ("tests", "The historical test adds no extra requirement."))]
            with patch.object(budget, "call", return_value={"reviews": rows}) as call:
                result = review_checks(spec, baseline, candidate, changed, results, config, output, budget)
            prompt, payload = call.call_args.args[:2]
            self.assertIn("实际检查及证据将在下一阶段完成", prompt)
            self.assertIn("不能仅因没有同名测试或 pytest 结果就判 gaps", prompt)
            self.assertIn("缺少具体检查动作或预期含义，仍判 gaps", prompt)
            self.assertIn("inspect: Read a.py; feature must be True", payload["criteria_and_tests"]["acceptance.md"])
            self.assertNotIn("acceptance-review.txt", payload["criteria_and_tests"])
            self.assertEqual(payload["executed_checks"]["reference"]["cases"],
                             [{"id": "test_export::test_null", "status": "passed"}])
            return result

        def checks(candidate, *args, **kwargs):
            namespace = {}
            exec((candidate / "a.py").read_text(), namespace)
            status = "passed" if namespace.get("keep_other_nulls") else "failed"
            return {"status": status, "cases": [{"id": "test_export::test_null", "status": status}]}

        def agent(root, config, role, message, **options):
            if root.name == "author":
                (root / "workspace/checks/test_acceptance.py").write_text("# Fixed historical check\n")
            elif root.name == "reference-solver":
                (root / "workspace/candidate/a.py").write_text("feature = True\nkeep_other_nulls = True\n")
            elif root.name == "validator":
                stages.append("validator")
                folder = root / "workspace/checks"
                folder.mkdir()
                self.assertIn("验收项 ID（如 a6，不是 Markdown 行号）", message)
                (folder / "mutations.txt").write_text("REVIEW m1\nacceptance: %s\nEND_REVIEW" % acceptance_id)
                (folder / "m1.patch").write_text(
                    "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
                    "@@ -1,2 +1,2 @@\n feature = True\n-keep_other_nulls = True\n+keep_other_nulls = False\n")
                if with_reference_inspection:
                    (folder / "acceptance-review.txt").write_text(
                        "REVIEW a1\nstatus: passed\nevidence: /reference/implementation/a.py:1,2\nEND_REVIEW")
                return {"status": "finished", "metrics": {
                    "attempted_requests": 1, "prompt_tokens": 10, "completion_tokens": 5, "usage_complete": True}}
            elif root.name == "judge":
                stages.append("mutant_inspection")
                self.assertEqual(options["max_requests"], 7)
                self.assertEqual(options["max_tokens"], 185)
                self.assertEqual((root / "workspace/candidate/a.py").read_text(),
                                 "feature = True\nkeep_other_nulls = False\n")
                folder = root / "workspace/checks"
                folder.mkdir()
                (folder / "acceptance-review.txt").write_text(
                    "REVIEW a1\nstatus: passed\nevidence: /workspace/candidate/a.py:1,2\nEND_REVIEW")
                return {"status": "finished", "metrics": {
                    "attempted_requests": 2, "prompt_tokens": 40, "completion_tokens": 10, "usage_complete": True}}
            return {"status": "finished"}

        for name, with_reference_inspection, acceptance_id in (
                ("inspected", True, "a2"), ("missing-inspection", False, "a2"), ("unknown-acceptance", True, "8")):
            with self.subTest(mode=name):
                root = self.root / name
                known_acceptance = acceptance_id == "a2"
                stages = []
                with patch("dialogue_benchmark.task_eval.run.write_draft", side_effect=draft), \
                     patch("dialogue_benchmark.task_eval.run.review_task", side_effect=review), \
                     patch("dialogue_benchmark.task_eval.run.review_sources", return_value={"support": "supported"}), \
                     patch("dialogue_benchmark.task_eval.run.review_checks", side_effect=coverage), \
                     patch("dialogue_benchmark.task_eval.run.run_agent", side_effect=agent), \
                     patch("dialogue_benchmark.task_eval.run.run_checks", side_effect=checks), \
                     patch("dialogue_benchmark.task_eval.checks.run_checks", side_effect=checks):
                    receipt = construct(self.item, root, self.baseline, {"execution_image": "image"}, 0,
                                        {"max_requests": 8, "max_tokens": 200})
                self.assertEqual(stages, ["coverage", "validator"] + (["mutant_inspection"] if known_acceptance else []))
                self.assertEqual(receipt is not None, with_reference_inspection and known_acceptance)
                self.assertEqual((root / "frozen.json").exists(), with_reference_inspection and known_acceptance)
                record = read(root / "construction.json")[0]
                self.assertEqual(record["checks_review"]["status"], "complete")
                self.assertEqual(record["history_mutations"]["status"], "caught" if known_acceptance else "unverified")
                if not known_acceptance:
                    self.assertEqual(record["reason"], "historical_mutation_not_verified")
                    self.assertIn("Unknown acceptance IDs: '8'", record["validation_evidence"])
                self.assertEqual(record["reference_acceptance"]["status"],
                                 "passed" if with_reference_inspection else "uncertain")
                self.assertEqual(record["validation_accepted"], with_reference_inspection and known_acceptance)
                self.assertEqual(record["preflight_budget"]["requests"], 3 if known_acceptance else 1)
                self.assertEqual(record["preflight_budget"]["total_tokens"], 65 if known_acceptance else 15)
                self.assertEqual(record["preflight_budget"], read(root / "construction-00/preflight/selection-budget.json"))

    def test_history_admission_checks_missing_baseline_inspection_after_mutant(self):
        rule = "The approved report status is DELIVERED."
        self.item["qa"]["answer_points"] = [{"text": rule}]
        self.item["public_records"] = [dict(id="e1", original_id="choice", order=1,
                                             kind="message", text=rule)]
        save(Path(self.item["generation_input"]), {"ref_to_source": {"资料1": "e1"}})
        inspect_calls = []

        def draft(selection, config, output, spec, budget, feedback=""):
            self.fake_draft(selection, config, output, spec, budget, feedback)
            (spec / "history-contract.txt").write_text(
                "REVIEW h1\nstatement: " + rule + "\nscope: All reports\nsources: choice\n"
                "supersedes: none\nbehavior: Report the approved status\nactive: yes\n"
                "repository: external\nEND_REVIEW")
            (spec / "acceptance.md").write_text(
                "| a1 | Deliver the report | task | inspect: Read report.txt; it must be ready |\n"
                "| a2 | Apply the approved status | h1 | test: test_acceptance::test_status |\n")

        def review(task, answer, config, output, *, evidence, budget):
            return {"status": "clean", "task_review": {"leakage": "clean"}, "history_rows": [
                dict(id=row["id"], applicable="yes", public="none", answer="sufficient",
                     historical_sources=row["sources"], answer_quote=rule)
                for row in evidence["contracts"]]}

        def coverage(spec, baseline, candidate, changed, results, config, output, budget):
            output.mkdir(parents=True)
            (output / "coverage.md").write_text("The test and inspect checks cover the two acceptance rows.")
            return {"status": "complete"}

        def checks(candidate, *args, **kwargs):
            mutant = "history-mutations" in str(candidate)
            return {"status": "failed" if mutant else "passed",
                    "cases": [{"id": "test_acceptance::test_status",
                               "status": "failed" if mutant else "passed"}]}

        def agent(root, config, role, message, **options):
            if root.name == "reference-solver":
                (root / "workspace/candidate/report.txt").write_text("READY\nDELIVERED\n")
            elif root.name == "validator":
                folder = root / "workspace/checks"
                folder.mkdir()
                (folder / "mutations.txt").write_text("REVIEW m1\nacceptance: a2\nEND_REVIEW")
                (folder / "m1.patch").write_text(
                    "--- a/report.txt\n+++ b/report.txt\n@@ -1,2 +1,2 @@\n"
                    " READY\n-DELIVERED\n+UNKNOWN\n")
                (folder / "acceptance-review.txt").write_text(
                    "REVIEW a1\nstatus: passed\n"
                    "evidence: /reference/implementation/report.txt:1\nEND_REVIEW")
            return {"status": "finished", "metrics": {
                "attempted_requests": 1, "prompt_tokens": 10,
                "completion_tokens": 5, "usage_complete": True}}

        def inspect(candidate, spec, items, checks_result, output, config, agent_options, *, budget=None):
            inspect_calls.append(output.name)
            review_path = output / "workspace/checks/acceptance-review.txt"
            review_path.parent.mkdir(parents=True, exist_ok=True)
            status = "failed" if candidate == self.baseline else "passed"
            evidence_path = output / "workspace/checks/inspection.txt"
            evidence_path.write_text("INSPECTED\n")
            review_path.write_text(
                "REVIEW a1\nstatus: %s\nevidence: /workspace/checks/inspection.txt:1\nEND_REVIEW" % status)
            return {"status": "finished", "metrics": {}}, review_path, {
                "/workspace/candidate": candidate,
                "/workspace/checks": output / "workspace/checks"}

        with patch("dialogue_benchmark.task_eval.run.write_draft", side_effect=draft), \
             patch("dialogue_benchmark.task_eval.run.review_task", side_effect=review), \
             patch("dialogue_benchmark.task_eval.run.review_sources", return_value={"support": "supported"}), \
             patch("dialogue_benchmark.task_eval.run.review_checks", side_effect=coverage), \
             patch("dialogue_benchmark.task_eval.run.run_agent", side_effect=agent), \
             patch("dialogue_benchmark.task_eval.run.run_checks", side_effect=checks), \
             patch("dialogue_benchmark.task_eval.checks.run_checks", side_effect=checks), \
             patch("dialogue_benchmark.task_eval.run.inspect_acceptance", side_effect=inspect):
            receipt = construct(self.item, self.root, self.baseline, {"execution_image": "image"}, 0,
                                {"max_requests": 8, "max_tokens": 200})

        self.assertIsNotNone(receipt)
        self.assertEqual(inspect_calls, ["m1", "baseline-inspection"])
        record = read(self.root / "construction.json")[0]
        self.assertEqual(record["baseline_acceptance"]["status"], "failed")
        self.assertEqual(record["reference_acceptance"]["status"], "passed")
        self.assertEqual(record["validation"]["BASELINE"], "unmet")
        self.assertEqual(record["validation"]["REFERENCE"], "pass")
        self.assertEqual(record["history_mutations"]["status"], "caught")

    def test_reuse_preparation_revalidates_without_regenerating_tests(self):
        self.validator_status = "ConversationExecutionStatus.STUCK"
        self.execute([{"status": "failed"}, {"status": "passed"}])
        original = self.root
        prior = original / "construction-00"
        save(prior / "author/result.json", {"status": "finished", "metrics": {"total_tokens": 100}})
        save(original / "selection/result.json", {"status": "candidate"})
        save(original.parent / "manifest.json", {"baseline_sha256": fingerprint(self.baseline)})
        original_test = (prior / "author/workspace/checks/test_acceptance.py").read_bytes()
        self.root = self.base / "retry/task"
        self.validator_status = "ConversationExecutionStatus.FINISHED"
        with patch("dialogue_benchmark.task_eval.run.write_draft") as draft, \
             patch("dialogue_benchmark.task_eval.run.select_task") as select:
            receipt, checks = self.execute([{"status": "failed"}, {"status": "passed"},
                                            {"status": "failed"}, {"status": "passed"}],
                                           reuse_preparation=prior)
        self.assertIsNotNone(receipt)
        draft.assert_not_called()
        select.assert_not_called()
        self.assertEqual(checks.call_count, 4)
        self.assertEqual((self.root / "frozen/test_acceptance.py").read_bytes(), original_test)
        record = read(self.root / "construction.json")[0]
        self.assertEqual(record["author_reused_from"], str(prior.resolve()))
        self.assertEqual(record["construction_budget"]["total_tokens"], 0)
        self.root = self.base / "repair/task"
        feedback = self.base / "review.md"
        feedback.write_text("Remove an unsupported exception-class assertion.")
        def repair(spec, baseline, config, output, budget, message):
            self.assertIn(feedback.read_text(), message)
            (spec / "test_acceptance.py").write_text("Repaired test")
            return {"status": "finished", "method": "model_file_generation"}
        with patch("dialogue_benchmark.task_eval.run.repair_tests", side_effect=repair) as repair_call:
            receipt, _ = self.execute([{"status": "failed"}, {"status": "passed"},
                                       {"status": "failed"}, {"status": "passed"}],
                                      reuse_preparation=prior, preparation_feedback=feedback)
        repair_call.assert_called_once()
        self.assertIsNotNone(receipt)
        self.assertEqual((self.root / "frozen/test_acceptance.py").read_text(), "Repaired test")
        self.assertEqual((prior / "author/workspace/checks/test_acceptance.py").read_bytes(), original_test)
        self.assertNotIn("author_reused_from", read(self.root / "construction.json")[0])
        self.root = self.base / "changed/task"
        self.item["qa"]["id"] = "different-question"
        with self.assertRaisesRegex(ValueError, "does not match"):
            construct(self.item, self.root, self.baseline, {}, 0, {}, reuse_preparation=prior)

    def test_extra_checks_are_run_before_freeze_and_used_for_receipt(self):
        receipt, checks = self.execute([{"status": "failed"}, {"status": "passed"},
                                       {"status": "failed", "tests": 2}, {"status": "passed", "tests": 2}])
        self.assertEqual(checks.call_count, 4)
        final_spec = checks.call_args_list[2].args[1]
        self.assertTrue((final_spec / "test_interactions.py").exists())
        self.assertTrue((self.root / "frozen/test_interactions.py").exists())
        self.assertEqual(receipt["reference_checks"]["tests"], 2)
        self.assertFalse((self.root / "frozen/checkpoints.json").exists())
        self.assertEqual(read(self.root / "frozen/acceptance.json")[0]["id"], "a1")

    def test_test_author_gets_criteria_and_previous_tests_without_raw_history(self):
        spec = self.base / "checks"
        spec.mkdir()
        reference = self.base / "reference"
        previous = reference / "previous-00"
        previous.mkdir(parents=True)
        for directory in (spec, previous):
            save(directory / "history.json", {"events": ["raw dialogue"]})
            (directory / "history-review.md").write_text("Raw dialogue")
            (directory / "history-contract.txt").write_text("Confirmed rule")
            (directory / "test_acceptance.py").write_text("Previous test")
        save(reference / "qa-input.json", {"payload": "Raw generation evidence"})
        output = prepare_test_reference(spec, reference, self.base / "test-reference")
        self.assertEqual(sorted(p.name for p in output.iterdir()), ["previous-00"])
        self.assertTrue((previous / "history.json").is_file())
        for directory in (spec, output / "previous-00"):
            self.assertFalse((directory / "history.json").exists())
            self.assertFalse((directory / "history-review.md").exists())
            self.assertEqual((directory / "history-contract.txt").read_text(), "Confirmed rule")
            self.assertEqual((directory / "test_acceptance.py").read_text(), "Previous test")

    def test_reference_failing_new_combination_is_not_frozen(self):
        receipt, _ = self.execute([{"status": "failed"}, {"status": "passed"},
                                   {"status": "failed"}, {"status": "failed"}])
        self.assertIsNone(receipt)
        self.assertFalse((self.root / "frozen").exists())
        self.assertFalse(read(self.root / "construction.json")[0]["accepted"])

    def test_model_complete_without_saved_coverage_does_not_pass(self):
        self.with_coverage = False
        receipt, checks = self.execute([{"status": "failed"}, {"status": "passed"}])
        self.assertIsNone(receipt)
        self.assertEqual(checks.call_count, 2)

    def test_baseline_collection_error_is_repaired_before_reference_execution(self):
        checks = [dict(status="error", errors=1, cases=[dict(
            id="test_acceptance::collection", status="error", detail="No module named new_api")])]
        checks.extend(dict(status=status, cases=[dict(id="test_acceptance::test_feature", status=status)])
                      for status in ("failed", "passed", "failed", "passed"))
        def repair(spec, baseline, config, output, budget, feedback):
            self.assertEqual((spec / "test_acceptance.py").read_text(), "Original test")
            from dialogue_benchmark.task_eval.run import run_agent
            return run_agent(output, config, "judge", feedback)
        with patch("dialogue_benchmark.task_eval.run.run_agent", side_effect=self.fake_agent) as agent, \
             patch("dialogue_benchmark.task_eval.run.repair_tests", side_effect=repair), \
             patch("dialogue_benchmark.task_eval.run.write_draft", side_effect=self.fake_draft) as draft, \
             patch("dialogue_benchmark.task_eval.run.review_task", return_value={"status": "clean"}), \
             patch("dialogue_benchmark.task_eval.run.run_checks", side_effect=checks):
            receipt = construct(self.item, self.root, self.baseline, {"execution_image": "image"}, 1, {})
        self.assertIsNotNone(receipt)
        self.assertEqual(draft.call_count, 1)
        calls = agent.call_args_list
        self.assertEqual([call.args[0].name for call in calls],
                         ["author", "author", "reference-solver", "validator"])
        self.assertIn("No module named new_api", calls[1].args[3])
        self.assertTrue((self.root / "author-reference/previous-00/test_acceptance.py").is_file())
        records = read(self.root / "construction.json")
        self.assertEqual(records[0]["reason"], "baseline_check_error")
        self.assertTrue(records[1]["accepted"])

    def test_failed_reference_returns_to_repair_before_validation(self):
        results = [dict(status=status, cases=[dict(id="test_acceptance::test_feature",
                    status=status, detail="unrecognized arguments: -q" if index == 1 else "")])
                   for index, status in enumerate(("failed", "failed", "failed", "passed",
                                                   "failed", "passed"))]
        results[1]["cases"][:0] = [dict(id="regression::test_ok", status="passed",
                                     detail="passed-only-noise" * 2000)]
        def agent(root, *args, **kwargs):
            if root.name == "reference-solver":
                if root.parent.name == "construction-01":
                    self.assertIn("unrecognized arguments: -q", args[2])
                (root / "workspace/candidate/a.py").write_text("value = 2\n")
            return self.fake_agent(root, *args, **kwargs)
        def repair(spec, baseline, config, output, budget, feedback):
            self.assertIn("unrecognized arguments: -q", feedback)
            self.assertNotIn("passed-only-noise", feedback)
            self.assertIn("+value = 2", feedback)
            self.assertEqual((spec / "test_acceptance.py").read_text(), "Original test")
            return agent(output, config, "judge", feedback)
        with patch("dialogue_benchmark.task_eval.run.run_agent", side_effect=agent) as worker, \
             patch("dialogue_benchmark.task_eval.run.repair_tests", side_effect=repair), \
             patch("dialogue_benchmark.task_eval.run.review_task", return_value={"status": "clean"}), \
             patch("dialogue_benchmark.task_eval.run.run_checks", side_effect=results):
            receipt = construct(self.item, self.root, self.baseline, {"execution_image": "image"}, 1, {})
        self.assertIsNotNone(receipt)
        self.assertEqual([call.args[0].parent.name for call in worker.call_args_list
                          if call.args[0].name == "validator"], ["construction-01"])
        records = read(self.root / "construction.json")
        self.assertEqual(records[0]["reason"], "reference_check_failed")
        self.assertFalse(records[0]["accepted"])
        self.assertTrue(records[1]["accepted"])

    def test_failed_reference_at_revision_limit_never_runs_validator(self):
        receipt, checks = self.execute([{"status": "failed"}, {"status": "error"}])
        self.assertIsNone(receipt)
        self.assertEqual(checks.call_count, 2)
        self.assertFalse((self.root / "construction-00/validator").exists())
        self.assertEqual(read(self.root / "construction.json")[0]["reason"], "reference_check_error")

    def check_cumulative_test_repairs(self, mode):
        from dialogue_benchmark.task_eval import prompts

        (self.baseline / "tests").mkdir()
        (self.baseline / "tests/test_existing.py").write_text("def test_existing(): pass\n")
        (self.baseline / "delivery.txt").write_text("old")
        original = ("from pathlib import Path\n\n"
                    "def test_feature(candidate_root):\n"
                    "    path = Path(__file__).parent / 'delivery.txt'\n"
                    "    assert path.read_text() == EXPECTED\n")
        repaired_path = original.replace("Path(__file__).parent", "candidate_root")
        complete = "EXPECTED = 'ready'\n" + repaired_path
        calls, repair_inputs, executions = [], [], []

        def model_call(budget, prompt, payload, config, output):
            calls.append(prompt)
            self.assertIn(prompts.TEST_EXECUTION, prompt)
            fixture = output / "workspace/checks/conftest.py"
            self.assertIn("def candidate_root():", fixture.read_text())
            if prompt == prompts.TEST_FILES:
                files = {"test_acceptance.py": original,
                         "acceptance.md": payload["requirements"]["acceptance.md"]}
            else:
                self.assertEqual(prompt, prompts.TEST_REPAIR)
                files = dict(payload["files"])
                repair_inputs.append(files["test_acceptance.py"])
                files["test_acceptance.py"] = repaired_path if len(repair_inputs) == 1 else complete
            return {"files": [{"name": name, "content": content} for name, content in files.items()]}

        def checks(candidate, spec, output, *args, **kwargs):
            source = (spec / "test_acceptance.py").read_text()
            executions.append((output.parent.name, output.name, source))
            try:
                runpy.run_path(str(spec / "test_acceptance.py"))["test_feature"](candidate)
                status, detail = "passed", ""
            except AssertionError:
                status, detail = "failed", "Delivery is not ready"
            except (FileNotFoundError, NameError) as error:
                status, detail = "error", str(error)
            return {"status": status, "cases": [
                {"id": "test_acceptance::test_feature", "status": status, "detail": detail}]}

        def agent(root, *args, **kwargs):
            self.assertNotEqual(root.name, "author")
            result = self.fake_agent(root, *args, **kwargs)
            if root.name == "reference-solver":
                (root / "workspace/candidate/delivery.txt").write_text("ready")
            return result

        options = {}
        with patch("dialogue_benchmark.task_eval.selection.SelectionBudget.call", new=model_call), \
             patch("dialogue_benchmark.task_eval.run.run_agent", side_effect=agent) as worker, \
             patch("dialogue_benchmark.task_eval.run.run_checks", side_effect=checks), \
             patch("dialogue_benchmark.task_eval.run.review_task", return_value={"status": "clean"}) as review:
            if mode != "fresh":
                self.assertIsNone(construct(self.item, self.root, self.baseline,
                                           {"execution_image": "image"}, 0, {}))
                prior = self.root / "construction-00"
                save(self.root / "selection/result.json", {"status": "candidate"})
                save(self.root.parent / "manifest.json", {"baseline_sha256": fingerprint(self.baseline)})
                prior_hash = fingerprint(prior)
                options["reuse_preparation"] = prior
                self.root = self.base / "retry/task"
                if mode == "feedback":
                    feedback = self.base / "feedback.md"
                    feedback.write_text("Read delivery.txt through candidate_root.")
                    options["preparation_feedback"] = feedback
                review.reset_mock()
                executions.clear()
            receipt = construct(self.item, self.root, self.baseline,
                                {"execution_image": "image"}, 2, {}, **options)
            review.assert_called_once()
        self.assertIsNotNone(receipt)
        self.assertEqual(calls.count(prompts.TEST_FILES), 1)
        self.assertEqual(calls.count(prompts.TEST_REPAIR), 2)
        self.assertEqual(repair_inputs, [original, repaired_path])
        self.assertEqual((self.root / "frozen/test_acceptance.py").read_text(), complete)
        self.assertEqual([call.args[0].name for call in worker.call_args_list],
                         ["reference-solver", "validator"])
        expected_errors = 1 if mode == "feedback" else 2
        self.assertEqual([record.get("reason") for record in read(self.root / "construction.json")][:-1],
                         ["baseline_check_error"] * expected_errors)
        first = self.root / "construction-00/qualified-draft"
        frozen = self.root / "frozen"
        for name in ("task.md", "memory-use.md"):
            self.assertEqual((first / name).read_bytes(), (frozen / name).read_bytes())
        self.assertEqual([{key: row[key] for key in ("id", "requirement", "basis")}
                          for row in acceptance_items(first)],
                         [{key: row[key] for key in ("id", "requirement", "basis")}
                          for row in acceptance_items(frozen)])
        if mode != "fresh":
            self.assertEqual(fingerprint(prior), prior_hash)

    def test_fresh_construction_repairs_previous_complete_tests_twice(self):
        self.check_cumulative_test_repairs("fresh")

    def test_reused_construction_repairs_previous_complete_tests_twice(self):
        self.check_cumulative_test_repairs("reuse")

    def test_preparation_feedback_repairs_previous_complete_tests_twice(self):
        self.check_cumulative_test_repairs("feedback")

    def test_custom_command_retry_keeps_agent_and_previous_suite(self):
        def agent(root, *args, **kwargs):
            if root.name == "author" and root.parent.name == "construction-01":
                spec = root / "workspace/checks"
                self.assertEqual((spec / "test_acceptance.py").read_text(), "Original test")
                self.assertEqual((spec / "commands/custom.sh").read_text(), "exit 2\n")
            result = self.fake_agent(root, *args, **kwargs)
            if root.name == "author":
                commands = root / "workspace/checks/commands"
                commands.mkdir(exist_ok=True)
                (commands / "custom.sh").write_text("exit 2\n" if root.parent.name == "construction-00"
                                                     else "exit 0\n")
            return result

        results = [dict(status=status, cases=[dict(id="test_acceptance::test_feature", status=status)])
                   for status in ("error", "failed", "passed", "failed", "passed")]
        with patch("dialogue_benchmark.task_eval.run.run_agent", side_effect=agent) as worker, \
             patch("dialogue_benchmark.task_eval.selection.SelectionBudget.call") as model, \
             patch("dialogue_benchmark.task_eval.run.review_task", return_value={"status": "clean"}), \
             patch("dialogue_benchmark.task_eval.run.run_checks", side_effect=results):
            receipt = construct(self.item, self.root, self.baseline, {"execution_image": "image"}, 1, {})
        self.assertIsNotNone(receipt)
        model.assert_not_called()
        self.assertEqual([call.args[0].name for call in worker.call_args_list],
                         ["author", "author", "reference-solver", "validator"])
        self.assertEqual((self.root / "frozen/commands/custom.sh").read_text(), "exit 0\n")

    def test_interrupted_validator_cannot_admit_with_earlier_accept_report(self):
        self.validator_status = "ConversationExecutionStatus.STUCK"
        receipt, _ = self.execute([{"status": "failed"}, {"status": "passed"},
                                   {"status": "failed"}, {"status": "passed"}])
        self.assertIsNone(receipt)
        self.assertFalse((self.root / "frozen").exists())
        self.assertEqual(read(self.root / "construction.json")[0]["reason"], "validator_incomplete")

    def test_interrupted_reference_does_not_produce_checkpoints(self):
        self.reference_status = "ConversationExecutionStatus.STUCK"
        receipt, _ = self.execute([{"status": "failed"}, {"status": "passed"},
                                   {"status": "failed"}, {"status": "passed"}])
        self.assertIsNone(receipt)
        self.assertNotIn("checkpoint_extraction", read(self.root / "construction.json")[0])

    def test_validator_runtime_failure_does_not_reauthor_the_task(self):
        original = self.fake_agent
        def agent(root, *args, **kwargs):
            if root.name == "validator":
                return {"status": "error", "error_code": "token_budget_exhausted",
                        "metrics": {"attempted_requests": 12}}
            return original(root, *args, **kwargs)
        results = [dict(status=status, cases=[]) for status in ("failed", "passed")]
        with patch("dialogue_benchmark.task_eval.run.run_agent", side_effect=agent) as worker, \
             patch("dialogue_benchmark.task_eval.run.review_task", return_value={"status": "clean"}), \
             patch("dialogue_benchmark.task_eval.run.run_checks", side_effect=results):
            receipt = construct(self.item, self.root, self.baseline, {"execution_image": "image"}, 3, {})
        self.assertIsNone(receipt)
        self.assertEqual([call.args[0].name for call in worker.call_args_list],
                         ["author", "reference-solver", "validator"])
        records = read(self.root / "construction.json")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["validator_error"]["error_code"], "token_budget_exhausted")
        self.assertEqual(records[0]["validator_metrics"]["attempted_requests"], 12)

    def test_leaking_task_is_rejected_before_reference_solver(self):
        with patch("dialogue_benchmark.task_eval.run.run_agent", side_effect=self.fake_agent) as agent, \
             patch("dialogue_benchmark.task_eval.run.review_task", return_value={"status": "leaked", "issue": "Internal fix given"}), \
             patch("dialogue_benchmark.task_eval.run.run_checks") as checks:
            receipt = construct(self.item, self.root, self.baseline, {"execution_image": "image"}, 0, {})
        self.assertIsNone(receipt)
        self.assertEqual(agent.call_count, 0)
        checks.assert_not_called()
        self.assertTrue((self.root / "construction-00/qualified-draft/task.md").exists())

    def test_selection_only_qualifies_once_without_starting_agents(self):
        with patch("dialogue_benchmark.task_eval.run.run_agent") as agent, \
             patch("dialogue_benchmark.task_eval.run.review_task", return_value={"status": "clean"}) as review, \
             patch("dialogue_benchmark.task_eval.run.run_checks") as checks:
            receipt = construct(self.item, self.root, self.baseline, {}, 0, {}, selection_only=True)
        self.assertEqual(receipt["status"], "qualified")
        self.assertEqual(review.call_count, 1)
        self.assertIsNotNone(review.call_args.kwargs["budget"])
        agent.assert_not_called()
        checks.assert_not_called()

    def test_author_budget_failure_keeps_cause_without_identical_retries(self):
        with patch("dialogue_benchmark.task_eval.run.review_task", return_value={"status": "clean"}), \
             patch("dialogue_benchmark.task_eval.run.run_agent", return_value={
                "status": "error", "error_code": "token_budget_exhausted", "error_type": "RuntimeError"}) as agent:
            receipt = construct(self.item, self.root, self.baseline, {}, 5, {})
        self.assertIsNone(receipt)
        self.assertEqual(agent.call_count, 1)
        records = read(self.root / "construction.json")
        self.assertEqual(records[0]["reason"], "token_budget_exhausted")
        self.assertEqual(records[0]["author_error"]["error_code"], "token_budget_exhausted")

    def test_ineligible_draft_never_enters_test_construction(self):
        stages = []
        def author(root, *args, **kwargs):
            checks = root / "workspace/checks"
            checks.mkdir()
            for name, text in (("task.md", "Add documentation"),
                               ("memory-use.md", "Keep rendering unchanged"),
                               ("acceptance.md", "| a1 | Add page | task | inspect: Check page |")):
                (checks / name).write_text(text)
            stages.append("tests")
            return {"status": "finished"}
        with patch("dialogue_benchmark.task_eval.run.run_agent", side_effect=author), \
             patch("dialogue_benchmark.task_eval.run.review_task", return_value={"status": "ineligible", "issue": "Scope expanded"}), \
             patch("dialogue_benchmark.task_eval.run.run_checks") as checks:
            self.assertIsNone(construct(self.item, self.root, self.baseline, {}, 2, {}))
        self.assertEqual(stages, [])
        checks.assert_not_called()
        self.assertEqual(len(read(self.root / "construction.json")), 1)

    def test_second_phase_cannot_change_qualified_requirement(self):
        original = self.fake_agent
        def author(root, *args, **kwargs):
            result = original(root, *args, **kwargs)
            if root.name == "author":
                (root / "workspace/checks/task.md").write_text("An expanded requirement")
            return result
        with patch("dialogue_benchmark.task_eval.run.run_agent", side_effect=author), \
             patch("dialogue_benchmark.task_eval.run.review_task", return_value={"status": "clean", "issue": "none"}), \
             patch("dialogue_benchmark.task_eval.run.run_checks") as checks:
            self.assertIsNone(construct(self.item, self.root, self.baseline, {}, 0, {}))
        checks.assert_not_called()
        record = read(self.root / "construction.json")[0]
        self.assertEqual(record["reason"], "qualified_draft_changed")
        self.assertEqual(record["changed_qualified_files"], ["task.md"])

    def test_stuck_author_is_not_retried_or_reported_as_changed_draft(self):
        with patch("dialogue_benchmark.task_eval.run.run_agent", return_value={"status": "STUCK"}) as agent, \
             patch("dialogue_benchmark.task_eval.run.review_task", return_value={"status": "clean"}), \
             patch("dialogue_benchmark.task_eval.run.run_checks") as checks:
            self.assertIsNone(construct(self.item, self.root, self.baseline, {}, 5, {}))
        self.assertEqual(agent.call_count, 1)
        checks.assert_not_called()
        record = read(self.root / "construction.json")[0]
        self.assertEqual(record["reason"], "author_incomplete")
        self.assertEqual(record["changed_qualified_files"], [])

    def test_public_review_is_small_and_does_not_resolve_conflicts_as_clean(self):
        config = {"judge": {"base_url": "https://example.com/v1", "model": "test", "key_env": "KEY"}}
        with patch("dialogue_benchmark.llm.ChatClient") as client:
            client.return_value.usage = []
            client.return_value.responses = []
            client.return_value.ask.return_value = {"reviews": [{"leakage": "clean", "issue": "Gives internal fix"}]}
            result = review_task("Public task", "Historical answer", config, self.base / "review")
        self.assertEqual(result["status"], "uncertain")
        payload = client.return_value.ask.call_args.args[1]
        self.assertEqual(set(payload), {"public_task", "historical_answer"})

    def test_existing_api_delivery_uses_history_qualification_and_frozen_checks(self):
        workflow = "Read input.json → apply Maple's confirmed status rules → deliver report.json"
        task = ("Complete Maple's pending report for input.json with its confirmed status rules. "
                "Reuse the existing write_report API and preserve its behavior. Deliver report.json "
                "as an ordered JSON array of records with order_id and canonical status fields.")
        answer = "For Maple, code 7 means DELIVERED and code 9 means FAILED."
        evidence = {"development_workflow": workflow,
                    "history_targets": [{"id": "h1", "statement": answer, "sources": ["event1"]}],
                    "repository_queries": [{"id": "query1", "result": {
                        "lines": [{"text": "def write_report(records, rules, destination):"}]}}]}
        response = parse_text_response(
            "H h1 | yes | none | sufficient | event1 | none | " + answer + "\nTASK | clean")
        with patch("dialogue_benchmark.task_eval.runtime.ask_model", return_value=response) as ask:
            qualified = review_task(task, answer, {}, self.base / "delivery-review", evidence=evidence)
        self.assertEqual(ask.call_count, 1)
        self.assertEqual(qualified["status"], "clean")
        self.assertEqual(ask.call_args.args[1]["development_workflow"], workflow)
        self.assertNotIn(answer, str(ask.call_args.args[1]["public_repository"]))
        response = parse_text_response(
            "H h1 | yes | full | not_applicable | event1 | task | none\nTASK | clean")
        with patch("dialogue_benchmark.task_eval.runtime.ask_model", return_value=response):
            leaked = review_task(task + answer, answer, {}, self.base / "disclosed-review", evidence=evidence)
        self.assertEqual(leaked["status"], "ineligible")
        self.assertEqual(leaked["issue"], "no_memory_gap")

        source = ("import json\n\ndef write_report(records, rules, destination):\n"
                  "    result = [dict(order_id=row['order_id'], status=rules[row['status']]) for row in records]\n"
                  "    destination.write_text(json.dumps(result))\n")
        (self.baseline / "a.py").write_text(source)
        records = [{"order_id": "one", "status": 9}, {"order_id": "two", "status": 7}]
        save(self.baseline / "input.json", records)
        candidates = [self.baseline]
        for name, rules in (("completed", {7: "DELIVERED", 9: "FAILED"}),
                            ("wrong-rule", {7: "FAILED", 9: "DELIVERED"})):
            candidate = self.base / name
            copy_tree(self.baseline, candidate)
            runpy.run_path(str(candidate / "a.py"))["write_report"](
                read(candidate / "input.json"), rules, candidate / "report.json")
            self.assertEqual((candidate / "a.py").read_text(), source)
            candidates.append(candidate)
        spec = self.base / "delivery-spec"
        (spec / "commands").mkdir(parents=True)
        (spec / "task.md").write_text(task)
        (spec / "acceptance.md").write_text(
            "| a1 | Deliver all records in input order | task | command: records |\n"
            "| a2 | Apply Maple's status rules | h1 | command: statuses |\n")
        conditions = {"records": "[row['order_id'] for row in records] == ['one', 'two']",
                      "statuses": "[row['status'] for row in records] == ['FAILED', 'DELIVERED']"}
        for name, condition in conditions.items():
            check = ("import json, sys; records = json.load(open('report.json')); "
                     "sys.exit(0 if " + condition + " else 1)")
            (spec / "commands" / (name + ".sh")).write_text(
                "test -f report.json || exit 1\n" + shlex.quote(sys.executable) + " -c " + shlex.quote(check) + "\n")
        items = acceptance_items(spec, {"contracts": [{"id": "h1", "active": True}]})
        with local_command_checks():
            results = [run_checks(candidate, spec, self.base / ("delivery-checks-%d" % index), "image")
                       for index, candidate in enumerate(candidates)]
        self.assertEqual([result["status"] for result in results], ["failed", "passed", "failed"])
        self.assertEqual(assess_acceptance(items, results[1])["status"], "passed")
        wrong = assess_acceptance(items, results[2])
        self.assertEqual([(row["id"], row["status"]) for row in wrong["rows"]],
                         [("a1", "passed"), ("a2", "failed")])

    def test_report_mutation_replays_patch_and_fails_only_historical_requirement(self):
        report = "one: FAILED\ntwo: DELIVERED\n"
        for index, name in enumerate(("docs/report.md", "report.txt", "report.rst", "external-report.txt")):
            with self.subTest(report=name):
                root = self.base / ("report-mutation-%d" % index)
                spec, candidate = root / "spec", root / "candidate"
                (spec / "commands").mkdir(parents=True)
                copy_tree(self.baseline, candidate)
                (candidate / name).parent.mkdir(parents=True, exist_ok=True)
                (candidate / name).write_text(report)
                (spec / "task.md").write_text("Deliver Maple's order status report at " + name)
                (spec / "memory-use.md").write_text("Private historical agreement")
                if name == "external-report.txt":
                    basis, history = "answer", None
                    save(spec / "oracle-answer.json", {"answer": "Maple: one failed; two delivered."})
                else:
                    basis = "h1"
                    history = {"contracts": [{"id": "h1", "active": True, "repository": "external"}]}
                    save(spec / "history.json", history)
                    (spec / "history-contract.txt").write_text("Maple: one failed; two delivered.")
                (spec / "acceptance.md").write_text(
                    "| a1 | Report all orders in order | task | command: records |\n"
                    "| a2 | Apply Maple's confirmed statuses | " + basis + " | command: statuses |\n")
                conditions = {"records": "[row[0] for row in rows] == ['one', 'two']",
                              "statuses": "[row[1] for row in rows] == ['FAILED', 'DELIVERED']"}
                for check_name, condition in conditions.items():
                    script = ("from pathlib import Path; import sys; "
                              "rows = [line.split(': ') for line in Path(" + repr(name) + ").read_text().splitlines()]; "
                              "sys.exit(0 if " + condition + " else 1)")
                    (spec / "commands" / (check_name + ".sh")).write_text(
                        shlex.quote(sys.executable) + " -c " + shlex.quote(script) + "\n")
                items = acceptance_items(spec, history)
                save(spec / "acceptance.json", items)
                # Accidental copies of evaluation material are never mutation targets.
                for private in ("qa-input.json", "memory-use.md", "history.json", "task.md",
                                "docs/history-contract.txt", "tests/test_report.py"):
                    (candidate / private).parent.mkdir(parents=True, exist_ok=True)
                    (candidate / private).write_text("Private material or tests\n")
                version = export_change(self.baseline, candidate, root / "reference-version")
                before_reference, before_spec = fingerprint(candidate), fingerprint(spec)
                files = [{"name": "mutations.txt", "content":
                          "REVIEW m1\nacceptance: a2\nfile: " + name + "\nEND_REVIEW"},
                         {"name": "before.txt", "content": "one: FAILED"},
                         {"name": "after.txt", "content": "one: DELIVERED"}]
                budget = SimpleNamespace(call=Mock(return_value={"files": files}))
                authored = write_history_mutation(spec, candidate, version["changed_files"], {},
                                                  root / "validator", budget)
                self.assertEqual(authored["status"], "finished", authored)
                self.assertEqual(budget.call.call_count, 1)
                self.assertEqual(budget.call.call_args.args[1]["reference_sources"], {name: report})
                self.assertTrue(read(root / "validator/version/version.json")["replay_verified"])
                with local_command_checks():
                    correct = run_checks(candidate, spec, root / "reference-checks", "image")
                    result = check_history_mutations(candidate, spec, root / "validator/workspace/checks",
                                                     root / "replay", "image")
                self.assertEqual(assess_acceptance(items, correct)["status"], "passed")
                self.assertEqual(result["status"], "caught", result)
                variant = result["variants"][0]
                self.assertTrue(variant["patch_applied"])
                self.assertEqual([(row["id"], row["status"]) for row in variant["acceptance"]["rows"]],
                                 [("a1", "passed"), ("a2", "failed")])
                self.assertIn("one: DELIVERED", (root / "replay/m1/candidate" / name).read_text())
                self.assertEqual(fingerprint(candidate), before_reference)
                self.assertEqual(fingerprint(spec), before_spec)

    def test_test_runner_executes_frozen_combinations_with_terminal_parity(self):
        spec, output = self.base / "spec", self.base / "checks-run"
        spec.mkdir()
        for name in ("test_acceptance.py", "test_interactions.py"):
            (spec / name).write_text("test content")
        (spec / "tests").mkdir()
        (spec / "tests/test_feature.py").write_text("test content")
        # A report retained in the source must not become this execution's receipt.
        (spec / "receipt.xml").write_text("<testsuite><testcase /></testsuite>")
        class Sandbox:
            name = "test-sandbox"
            def __init__(self, directory, workspace, image, role, identity, reference):
                self.workspace = workspace
            def prepare(self):
                (self.workspace / "experiments").mkdir()
            def unpause(self):
                pass
            def pause(self):
                pass
        with patch.dict("sys.modules", {"simulator.openhands.sandbox": SimpleNamespace(ExecutionSandbox=Sandbox)}), \
             patch("dialogue_benchmark.task_eval.checks.release_completed_execution") as release, \
             patch("dialogue_benchmark.task_eval.checks.subprocess.run",
                   return_value=SimpleNamespace(returncode=124, stdout="", stderr="timeout")) as run:
            result = run_checks(self.baseline, spec, output, "image")
        command = run.call_args.args[0]
        self.assertIn("-t", command)
        self.assertEqual(command[command.index("-w") + 1], "/workspace/checks/validation-candidate")
        self.assertIn("--rootdir=/workspace/checks", command)
        self.assertFalse(any("PYTHONPATH=" in part for part in command))
        self.assertIn("/workspace/checks/test_interactions.py", command)
        self.assertIn("/workspace/checks/tests/test_feature.py", command)
        self.assertIn("--junitxml=/workspace/experiments/receipt.xml", command)
        self.assertEqual(result["status"], "error")
        release.assert_not_called()


if __name__ == "__main__":
    unittest.main()
