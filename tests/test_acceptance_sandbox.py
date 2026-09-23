"""Opt-in real sandbox smoke test; no model calls or memory-benefit claims."""

import os
from pathlib import Path
import unittest

from dialogue_benchmark.task_eval.artifacts import save
from dialogue_benchmark.task_eval.checks import acceptance_items, assess_acceptance, run_checks


@unittest.skipUnless(os.environ.get("QA_TASK_SANDBOX_IMAGE"), "Requires the simulator's pinned execution image")
class AcceptanceSandboxTests(unittest.TestCase):
    def test_fixed_checks_identify_correct_and_history_blind_implementations(self):
        root = Path(os.environ["QA_TASK_SMOKE_OUTPUT"])
        root.mkdir(parents=True, exist_ok=False)
        spec = root / "spec"
        spec.mkdir()
        (spec / "acceptance.md").write_text(
            "| a1 | Batch export | task | test: test_acceptance::test_batch |\n"
            "| a2 | CRM note-only exception | h1 | test: test_acceptance::test_exception |\n"
            "| a3 | Documentation | task | command: documentation |\n")
        (spec / "test_acceptance.py").write_text(
            "from crm import export\n\n"
            "def test_batch():\n    assert export([{'id': 1}, {'id': 2}], 'crm_v1') == [{'id': 1}, {'id': 2}]\n\n"
            "def test_exception():\n"
            "    row = {'id': 1, 'note': None, 'other': None}\n"
            "    assert export([row], 'crm_v1') == [{'id': 1, 'other': None}]\n"
            "    assert export([row], 'crm_v2') == [row]\n")
        (spec / "commands").mkdir()
        (spec / "commands/documentation.sh").write_text("test -f GUIDE.md\n")
        items = acceptance_items(spec, {"contracts": [{"id": "h1", "active": True}]})
        implementations = {
            "baseline": "def export(rows, target):\n    return []\n",
            "correct": "def export(rows, target):\n    return [{k: v for k, v in r.items() if not (target == 'crm_v1' and k == 'note' and v is None)} for r in rows]\n",
            "forgot_history": "def export(rows, target):\n    return [{k: v for k, v in r.items() if v is not None} for r in rows]\n"}
        results = {}
        for name, code in implementations.items():
            candidate = root / name / "candidate"
            (candidate / "lib").mkdir(parents=True)
            (candidate / "lib/crm.py").write_text(code)
            (candidate / "GUIDE.md").write_text("Batch CRM export\n")
            checks = run_checks(candidate, spec, root / name / "checks", os.environ["QA_TASK_SANDBOX_IMAGE"],
                                candidate_pythonpath="/workspace/candidate/lib")
            results[name] = assess_acceptance(items, checks)
        save(root / "results.json", results)
        self.assertEqual(results["baseline"]["status"], "failed")
        self.assertEqual(results["correct"]["status"], "passed")
        self.assertEqual(results["forgot_history"]["status"], "failed")
        self.assertEqual(results["forgot_history"]["rows"][0]["status"], "passed")
        self.assertEqual(results["forgot_history"]["rows"][1]["status"], "failed")
        self.assertEqual(results["correct"]["rows"][2]["status"], "passed")
