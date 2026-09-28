"""Executable contracts for evidence-based historical task evaluation."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dialogue_benchmark.task_eval.artifacts import save
from dialogue_benchmark.task_eval.checks import (
    acceptance_items, assess_acceptance, check_history_mutations, pytest_result)
from dialogue_benchmark.task_eval.history import answer_quote_supported, historical_question, qualified_oracle_complete
from dialogue_benchmark.task_eval.metrics import measure


class FrozenAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.items = [
            {"id": "a1", "basis": ["task"], "requirement": "Export works", "tests": ["test_export::test_batch"]},
            {"id": "a2", "basis": ["h1"], "requirement": "Preserve other nulls", "tests": ["test_export::test_null"]}]

    def checks(self, second="passed"):
        return {"status": "failed" if second == "failed" else "passed", "cases": [
            {"id": "test_export::test_batch", "status": "passed"},
            {"id": "test_export::test_null", "status": second}]}

    def test_test_evidence_passes_without_judge_or_history_mention(self):
        result = assess_acceptance(self.items, self.checks(), self.root / "missing-review")
        self.assertEqual(result["status"], "passed")

    def test_repeating_answer_cannot_override_wrong_behavior(self):
        review = self.root / "review.txt"
        review.write_text("REVIEW a2\nstatus: passed\nevidence: I repeated the answer\nEND_REVIEW")
        result = assess_acceptance(self.items, self.checks("failed"), review)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["rows"][0]["status"], "passed")

    def test_missing_duplicate_skipped_results_are_not_proof(self):
        for cases in ([], self.checks()["cases"] * 2, self.checks("skipped")["cases"]):
            self.assertEqual(assess_acceptance(self.items, {"status": "passed", "cases": cases})["status"], "uncertain")

    def test_junit_keeps_individual_failure_details(self):
        path = self.root / "receipt.xml"
        path.write_text('<testsuite><testcase classname="test_export" name="test_null">'
                        '<failure message="Wrong null">Expected retained field</failure></testcase></testsuite>')
        result = pytest_result(1, path)
        self.assertEqual(result["cases"][0]["id"], "test_export::test_null")
        self.assertIn("Expected retained field", result["cases"][0]["detail"])

    def test_file_node_ids_match_junit_modules_without_basename_aliases(self):
        items = [dict(self.items[0], tests=["tests/test_export.py::test_batch"])]
        checks = {"status": "passed", "cases": [
            {"id": "tests.test_export::test_batch", "status": "passed"}]}
        self.assertEqual(assess_acceptance(items, checks)["status"], "passed")
        checks["cases"][0]["id"] = "other.test_export::test_batch"
        self.assertEqual(assess_acceptance(items, checks)["status"], "uncertain")
        checks["cases"] = [
            {"id": "tests.test_export::test_batch", "status": "passed"},
            {"id": "tests/test_export.py::test_batch", "status": "passed"}]
        self.assertEqual(assess_acceptance(items, checks)["status"], "uncertain")

    def test_manual_evidence_must_resolve_and_missing_feature_is_not_inapplicable(self):
        items = [dict(self.items[0], tests=[])]
        code = self.root / "output.txt"
        code.write_text("real output\n")
        review = self.root / "review.txt"
        for status, location, expected in (
            ("passed", "/workspace/checks/output.txt:1", "passed"),
            ("failed", "/workspace/checks/output.txt:1", "failed"),
            ("passed", "/workspace/checks/output.txt:2", "uncertain"),
            ("passed", "/workspace/checks/missing:1", "uncertain"),
            ("not_applicable", "/workspace/checks/output.txt:1", "uncertain")):
            review.write_text(f"REVIEW a1\nstatus: {status}\nevidence: {location}\nEND_REVIEW")
            result = assess_acceptance(items, {"status": "unavailable"}, review,
                                       {"/workspace/checks": self.root})
            self.assertEqual(result["status"], expected)

    def test_pytest_class_nodes_match_only_the_same_junit_class(self):
        table = self.root / "acceptance.md"
        table.write_text("| a1 | Export | task | test: tests/test_export.py::TestExport::test_batch |\n")
        items = acceptance_items(self.root)
        checks = {"status": "passed", "cases": [
            {"id": "tests.test_export.TestExport::test_batch", "status": "passed"}]}
        self.assertEqual(assess_acceptance(items, checks)["status"], "passed")
        checks["cases"][0]["id"] = "tests.test_export.TestOther::test_batch"
        self.assertEqual(assess_acceptance(items, checks)["status"], "uncertain")
        checks["cases"] = [
            {"id": "tests.test_export.TestExport::test_batch", "status": "passed"},
            {"id": "tests/test_export.py::TestExport::test_batch", "status": "passed"}]
        self.assertEqual(assess_acceptance(items, checks)["status"], "uncertain")

    def test_acceptance_requires_every_active_rule_and_supports_commands(self):
        (self.root / "commands").mkdir()
        (self.root / "commands/export.sh").write_text("exit 0\n")
        table = self.root / "acceptance.md"
        table.write_text("| a1 | Export | task | command: export |\n")
        history = {"contracts": [{"id": "h1", "active": True}]}
        with self.assertRaises(ValueError):
            acceptance_items(self.root, history)
        table.write_text(table.read_text() + "| a2 | Retain null | h1 | test: test_export::test_null |\n")
        self.assertEqual(acceptance_items(self.root, history)[0]["tests"], ["command::export"])

    def test_oracle_requires_actual_answer_excerpt_for_all_active_rules(self):
        history = {"oracle_answer": "Retain other nulls", "contracts": [
            {"id": "h1", "active": True, "repository": "external", "sources": ["event1"]}]}
        for quote, state, expected in (("Retain other nulls", "sufficient", True),
                                       ("Private criterion", "sufficient", False),
                                       ("Retain other nulls", "insufficient", False),
                                       ("Retain other nulls", "uncertain", False)):
            review = {"status": "clean", "task_review": {"leakage": "clean"}, "history_rows": [
                {"id": "h1", "applicable": "yes", "public": "none", "answer": state,
                 "historical_sources": ["event1"], "answer_quote": quote}]}
            self.assertEqual(qualified_oracle_complete(history, review), expected)

    def test_one_requirement_can_reference_test_and_command(self):
        (self.root / "commands").mkdir()
        (self.root / "commands/export.sh").write_text("exit 0\n")
        table = self.root / "acceptance.md"
        table.write_text("| a1 | Export | task | test: test_export::test_batch; command: export |\n")
        items = acceptance_items(self.root)
        self.assertEqual(items[0]["tests"], ["test_export::test_batch", "command::export"])
        cases = [{"id": identity, "status": "passed"} for identity in items[0]["tests"]]
        self.assertEqual(assess_acceptance(items, {"status": "passed", "cases": cases})["status"], "passed")
        cases[1]["status"] = "failed"
        self.assertEqual(assess_acceptance(items, {"status": "failed", "cases": cases})["status"], "failed")
        table.write_text("| a1 | Export | task | test: test_export::test_batch; command: absent |\n")
        with self.assertRaisesRegex(ValueError, "Missing frozen command"):
            acceptance_items(self.root)

    def test_comma_between_tests_and_command_keeps_both_checks(self):
        (self.root / "commands").mkdir()
        (self.root / "commands/export.sh").write_text("exit 0\n")
        (self.root / "acceptance.md").write_text(
            "| a1 | Export | task | test: test_export::test_batch, test_export::test_null, command: export |\n")
        items = acceptance_items(self.root)
        self.assertEqual(items[0]["tests"],
                         ["test_export::test_batch", "test_export::test_null", "command::export"])
        cases = [{"id": identity, "status": "passed"} for identity in items[0]["tests"]]
        self.assertEqual(assess_acceptance(items, {"status": "passed", "cases": cases})["status"], "passed")
        cases[-1]["status"] = "failed"
        self.assertEqual(assess_acceptance(items, {"status": "failed", "cases": cases})["status"], "failed")

    def test_answer_quote_cannot_change_a_high_overlap_rule(self):
        answer = "- Maple 的 note 字段为 null 时必须省略。\n- 其他字段保留显式 null。"
        self.assertTrue(answer_quote_supported(
            "Maple 的 note 字段为 null 时必须省略；其他字段保留显式 null。", answer))
        self.assertFalse(answer_quote_supported("Maple 的 note 字段为 null 时不能省略。", answer))
        self.assertFalse(answer_quote_supported("其他字段不保留显式 null。", answer))

    def test_answer_quote_accepts_escaped_bullets_without_changing_claims(self):
        answer = "- 按实际字节计量。\n- 客户上限为 384。"
        self.assertTrue(answer_quote_supported(r"按实际字节计量。\n- 客户上限为 384。", answer))
        self.assertTrue(answer_quote_supported(r"按实际字节计量。\r\n- 客户上限为 384。", answer))
        self.assertFalse(answer_quote_supported(r"按实际字节计量。\n- 客户上限为 512。", answer))

    def test_completion_does_not_trigger_historical_responder(self):
        self.assertIsNone(historical_question("Done. All tests pass. Previous requirements preserved."))
        self.assertEqual(historical_question("HISTORY_QUESTION: Which CRM keeps null?"), "Which CRM keeps null?")

    def test_development_calls_exclude_control_but_keep_failed_calls(self):
        events = [{"kind": "ActionEvent", "tool_name": name, "action": {}} for name in (
            "think", "finish", "terminal", "file_editor")]
        result = measure(events, self.root / "missing")
        self.assertEqual(result["tool_calls"], 2)
        self.assertEqual(result["raw_action_count"], 4)

    def test_mutation_is_replayed_and_must_preserve_new_function(self):
        spec, candidate, validator = [self.root / name for name in ("spec", "candidate", "validator")]
        for path in (spec, candidate, validator):
            path.mkdir()
        save(spec / "acceptance.json", self.items)
        save(spec / "history.json", {"contracts": [{"id": "h1", "active": True, "repository": "external"}]})
        (candidate / "export.py").write_text("feature = True\nkeep_other_nulls = True\n")
        (validator / "mutations.txt").write_text("REVIEW m1\nacceptance: a2\nEND_REVIEW")
        (validator / "m1.patch").write_text(
            "diff --git a/export.py b/export.py\n--- a/export.py\n+++ b/export.py\n"
            "@@ -1,2 +1,2 @@\n feature = True\n-keep_other_nulls = True\n+keep_other_nulls = False\n")
        def checks(code, *args, **kwargs):
            namespace = {}
            exec((code / "export.py").read_text(), namespace)
            result = self.checks("passed" if namespace["keep_other_nulls"] else "failed")
            result["cases"][0]["status"] = "passed" if namespace["feature"] else "failed"
            return result
        with patch("dialogue_benchmark.task_eval.checks.run_checks", side_effect=checks):
            result = check_history_mutations(candidate, spec, validator, self.root / "mutations", "image")
        self.assertEqual(result["status"], "caught")
        self.assertIn("keep_other_nulls = True", (candidate / "export.py").read_text())
        self.assertTrue((self.root / "mutations/m1/result.json").exists())
        # A mixed public/history row cannot establish a specifically historical failure.
        mixed = [dict(row) for row in self.items]
        mixed[1]["basis"] = ["task", "h1"]
        save(spec / "acceptance.json", mixed)
        with patch("dialogue_benchmark.task_eval.checks.run_checks", side_effect=checks) as runner:
            result = check_history_mutations(candidate, spec, validator, self.root / "mixed", "image")
        self.assertEqual(result["status"], "unverified")
        runner.assert_not_called()
        save(spec / "acceptance.json", self.items)
        (validator / "m1.patch").write_text(
            "diff --git a/export.py b/export.py\n--- a/export.py\n+++ b/export.py\n"
            "@@ -1,2 +1,2 @@\n-feature = True\n-keep_other_nulls = True\n+feature = False\n+keep_other_nulls = False\n")
        with patch("dialogue_benchmark.task_eval.checks.run_checks", side_effect=checks):
            result = check_history_mutations(candidate, spec, validator, self.root / "broken", "image")
        self.assertEqual(result["status"], "unverified")


if __name__ == "__main__":
    unittest.main()
