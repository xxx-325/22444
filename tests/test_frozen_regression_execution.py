"""Offline execution of frozen repository tests and compact review evidence."""

import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dialogue_benchmark.task_eval.artifacts import copy_tree, fingerprint, save
from dialogue_benchmark.task_eval.checks import assess_acceptance, run_checks
from dialogue_benchmark.task_eval.runtime import review_checks, write_tests


class FrozenRegressionExecutionTests(unittest.TestCase):
    def test_frozen_tests_use_candidate_implementation_and_repository_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, spec = root / "baseline", root / "spec"
            (baseline / "tests").mkdir(parents=True)
            spec.mkdir()
            (baseline / "app.py").write_text("def transform(value): return value * 3\n")
            (baseline / "data.json").write_text('{"value": 2, "expected": 6}\n')
            (baseline / "data.csv").write_text("value,expected\n3,9\n")
            (baseline / "tests/test_json.py").write_text(
                "from pathlib import Path\nimport json\nimport app\n\n"
                "def test_json_data():\n"
                "    row = json.loads((Path(__file__).parents[1] / 'data.json').read_text())\n"
                "    assert app.transform(row['value']) == row['expected']\n")
            (baseline / "tests/csv_test.py").write_text(
                "from pathlib import Path\nimport csv\nimport app\n\n"
                "def test_csv_data():\n"
                "    with (Path(__file__).parents[1] / 'data.csv').open() as stream:\n"
                "        row = next(csv.DictReader(stream))\n"
                "    assert app.transform(int(row['value'])) == int(row['expected'])\n")
            (spec / "task.md").write_text("Preserve the transformation and data files.")
            (spec / "acceptance.md").write_text("Existing behavior must continue to work.")
            response = {"files": [
                {"name": "test_acceptance.py", "content": "def test_feature(): assert True\n"},
                {"name": "acceptance.md", "content": (
                    "| a1 | Preserve behavior | task | command: existing_suite |\n")} ]}
            authored = write_tests(spec, baseline, {}, root / "author",
                                   SimpleNamespace(call=lambda *args: response))
            self.assertEqual(authored["status"], "finished")
            # Only the frozen tests may overlay the candidate, never old source.
            (spec / "regression/app.py").write_text((baseline / "app.py").read_text())
            spec_fingerprint = fingerprint(spec)
            real_run = subprocess.run
            workspaces = []

            def sandbox(private, workspace, image, role, identity, reference):
                workspaces.append(workspace)
                return SimpleNamespace(
                    name="local-checks", record={},
                    prepare=lambda: (workspace / "experiments").mkdir(),
                    unpause=lambda: None, pause=lambda: None)

            def execute(argv, **kwargs):
                workspace = workspaces[-1]
                command = [part.replace("/workspace", str(workspace))
                           for part in argv[argv.index("local-checks") + 1:]]
                env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                           PYTHONDONTWRITEBYTECODE="1")
                env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
                if command[0] == "python":
                    command[0] = sys.executable
                elif command[0] == "bash":
                    command = ["bash", "-c", Path(command[1]).read_text().replace(
                        "/workspace", str(workspace)).replace(
                        "python -m pytest", shlex.quote(sys.executable) + " -m pytest")]
                return real_run(command, cwd=workspace / "candidate", env=env, **kwargs)

            for name, implementation, expected in (
                    ("correct", "return value + value + value", "passed"),
                    ("broken", "return value * 2", "failed")):
                with self.subTest(candidate=name):
                    candidate = root / name
                    copy_tree(baseline, candidate)
                    (candidate / "app.py").write_text("def transform(value): " + implementation + "\n")
                    # A solver cannot weaken or skip the frozen original tests.
                    (candidate / "tests/test_json.py").write_text("def test_json_data(): pass\n")
                    (candidate / "tests/csv_test.py").unlink()
                    (candidate / "tests/conftest.py").write_text("import pytest\npytest.skip('skip all', allow_module_level=True)\n")
                    candidate_fingerprint = fingerprint(candidate)
                    with patch.dict("sys.modules", {"simulator.openhands.sandbox": SimpleNamespace(
                            ExecutionSandbox=sandbox)}), \
                         patch("dialogue_benchmark.task_eval.checks.release_completed_execution"), \
                         patch("dialogue_benchmark.task_eval.checks.subprocess.run", side_effect=execute):
                        result = run_checks(candidate, spec, root / (name + "-checks"), "image")
                    self.assertEqual(result["status"], expected, result)
                    cases = {case["id"]: case["status"] for case in result["cases"]}
                    self.assertEqual(cases, {
                        "test_acceptance::test_feature": "passed",
                        "regression.tests.test_json::test_json_data": expected,
                        "regression.tests.csv_test::test_csv_data": expected,
                        "command::existing_suite": expected})
                    item = {"id": "a1", "tests": ["regression/tests/test_json.py::test_json_data"]}
                    self.assertEqual(assess_acceptance([item], result)["status"], expected)
                    self.assertEqual(fingerprint(candidate), candidate_fingerprint)
                    self.assertEqual(fingerprint(spec), spec_fingerprint)
                    self.assertEqual((workspaces[-1] / "candidate/app.py").read_text(),
                                     (candidate / "app.py").read_text())


class ReviewSourceDiffTests(unittest.TestCase):
    def test_review_excludes_candidate_tests_only_when_frozen_directory_exists(self):
        for frozen in ("absent", "file", "directory"):
            with self.subTest(frozen=frozen), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                spec, baseline, candidate = (root / name for name in ("spec", "base", "candidate"))
                spec.mkdir()
                save(spec / "acceptance.json", [{"id": "h1"}])
                scoring_files = {
                    "acceptance.md": "The report must show admission before recovery.\n",
                    "test_acceptance.py": (
                        "def test_report(candidate):\n"
                        "    assert 'admitted' in (candidate / 'report.md').read_text()\n"),
                    "commands/existing_suite.sh": "python -m pytest /workspace/candidate/tests\n"}
                if frozen == "directory":
                    scoring_files["regression/tests/test_existing.py"] = (
                        "def test_existing(): assert True\n")
                elif frozen == "file":
                    (spec / "regression").mkdir()
                    (spec / "regression/tests").write_text("Not a frozen test directory.\n")
                for name, content in scoring_files.items():
                    path = spec / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content)
                changes = {
                    "app.py": ("VALUE = 1\n", "VALUE = 2\n"),
                    "tests/test_existing.py": ("def test_existing(): assert True\n",
                                               "def test_existing(): pass\n"),
                    "tests/test_stage_03.py": (None, "def test_stage_03(): assert True\n"),
                    "tests/test_removed.py": ("def test_removed(): assert True\n", None),
                    "src/tests/helper.py": ("VALUE = 1\n", "VALUE = 2\n"),
                    "tests_extra.py": ("VALUE = 1\n", "VALUE = 2\n")}
                for name, versions in changes.items():
                    for checkout, content in zip((baseline, candidate), versions):
                        if content is not None:
                            path = checkout / name
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_text(content)
                checks = {"reference": {"status": "passed", "tests": 2, "passed": 2,
                                       "failed": 0, "errors": 0, "skipped": 0,
                                       "cases": [{"id": "test_acceptance::test_report", "status": "passed"},
                                                 {"id": "command::existing_suite", "status": "passed"}]}}
                rows = [{"id": "h1", "coverage": "unsupported",
                         "evidence": "The report substring assertion does not verify event order."},
                        {"id": "tests", "coverage": "complete", "evidence": "Existing suite retained."}]
                payloads = []

                def review(prompt, payload, config, output):
                    payloads.append(payload)
                    return {"reviews": rows}

                result = review_checks(spec, baseline, candidate, list(changes), checks, {},
                                       root / "review", SimpleNamespace(call=review))
                self.assertEqual(result, {"status": "revise", "rows": rows})
                payload = payloads[0]
                expected_sources = ({"app.py", "src/tests/helper.py", "tests_extra.py"}
                                    if frozen == "directory" else set(changes))
                self.assertEqual(set(payload["changed_sources"]), expected_sources)
                self.assertIn("+VALUE = 2\n", payload["changed_sources"]["app.py"])
                self.assertEqual(payload["criteria_and_tests"], scoring_files)
                self.assertEqual(payload["executed_checks"], checks)
                if frozen == "directory":
                    self.assertIn("/workspace/candidate/tests", payload["execution_context"])
                    self.assertIn("frozen regression/tests", payload["execution_context"])
                    self.assertIn("not part of the scored suite", payload["execution_context"])
                else:
                    self.assertNotIn("execution_context", payload)

    def test_review_keeps_changes_and_complete_checks_without_duplicating_background(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, baseline, candidate = (root / name for name in ("spec", "base", "candidate"))
            for path in (spec, baseline, candidate):
                path.mkdir()
            save(spec / "acceptance.json", [{"id": "a1"}])
            (spec / "acceptance.md").write_text("Every record must be exported.\n")
            (spec / "test_acceptance.py").write_text("def test_every_record(): assert True\n")
            background = "".join("# unchanged background %d\n" % index for index in range(100))
            (baseline / "app.py").write_text(background + "def value(): return 1\n")
            (candidate / "app.py").write_text(background + "def value(): return 2\n")
            (baseline / "removed.py").write_text("OLD = 1")
            (candidate / "added.py").write_text("NEW = 2")
            for path in (baseline, candidate):
                (path / "unchanged.py").write_text("UNCHANGED = 3\n")
            checks = {"reference": {"status": "passed", "tests": 1, "passed": 1, "failed": 0,
                                   "errors": 0, "skipped": 0,
                                   "cases": [{"id": "test_acceptance::test_every_record", "status": "passed"}]}}
            payloads = []

            def review(prompt, payload, config, output):
                payloads.append(payload)
                return {"reviews": [{"id": identity, "coverage": "complete", "evidence": "Complete checks"}
                                    for identity in ("a1", "tests")]}

            result = review_checks(spec, baseline, candidate,
                                   ["app.py", "removed.py", "added.py", "unchanged.py"],
                                   checks, {}, root / "review", SimpleNamespace(call=review))
            self.assertEqual(result["status"], "complete")
            payload = payloads[0]
            sources = payload["changed_sources"]
            self.assertEqual(set(sources), {"app.py", "removed.py", "added.py"})
            self.assertIn("-def value(): return 1\n", sources["app.py"])
            self.assertIn("+def value(): return 2\n", sources["app.py"])
            self.assertNotIn("unchanged background 0", sources["app.py"])
            self.assertEqual(sources["app.py"].count("unchanged background 99"), 1)
            self.assertIn("--- /dev/null\n", sources["added.py"])
            self.assertIn("+NEW = 2\n\\ No newline at end of file\n", sources["added.py"])
            self.assertIn("+++ /dev/null\n", sources["removed.py"])
            self.assertIn("-OLD = 1\n\\ No newline at end of file\n", sources["removed.py"])
            self.assertEqual(payload["executed_checks"], checks)
            self.assertEqual(payload["criteria_and_tests"], {
                name: (spec / name).read_text() for name in ("acceptance.md", "test_acceptance.py")})


if __name__ == "__main__":
    unittest.main()
