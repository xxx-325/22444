"""Optional business fact pools, scoped public rules, and bounded solver turns."""

from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dialogue_benchmark.task_eval.artifacts import public_global_agreements, read
from dialogue_benchmark.task_eval.prompts import SELECT_TASK, EXTERNAL_ACCEPTANCE
from dialogue_benchmark.task_eval.runtime import run_agent
from dialogue_benchmark.task_eval.selection import SelectionBudget, write_draft


class BusinessTaskTests(unittest.TestCase):
    def test_global_index_preserves_explicit_scope_and_public_source(self):
        records = [
            {"id": "local", "role": "user", "text": "Maple reports use English headings."},
            {"id": "local-tasks", "role": "user", "text": "All tasks for Maple use English headings."},
            {"id": "global", "role": "user", "text": "Across the project, code comments use English."},
            {"id": "guess", "role": "assistant", "text": "All tasks might use English."},
        ]
        rules = public_global_agreements(records)
        self.assertEqual([rule["sources"] for rule in rules], [["global"]])
        self.assertEqual(rules[0]["text"], records[2]["text"])
        self.assertIn(records[2]["text"], rules[0]["context"])

    def test_global_index_retains_public_revision_as_separate_fact(self):
        records = [
            {"id": "old", "role": "user", "text": "All tasks write comments in English."},
            {"id": "new", "role": "user", "text": "Project-wide comment language is now Chinese."},
        ]
        self.assertEqual([row["sources"] for row in public_global_agreements(records)],
                         [["old"], ["new"]])

    def test_optional_facts_do_not_all_become_injected_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected = "Maple retains original item order."
            global_rule = "All tasks write comments in English."
            unrelated = "Birch uses a different filename."
            selection = dict(qa_source="external", public={"public_goal": "Deliver Maple exports"},
                             historical_answer="\n".join("- " + line for line in
                                                         [selected, unrelated, global_rule]),
                             global_agreements=[{"text": global_rule, "context": global_rule}])
            responses = [{"files": [{"name": "task.md", "content": "Deliver Maple exports."}]},
                         {"files": [{"name": "memory-use.md", "content": "Order and global comments apply."},
                                    {"name": "acceptance.md", "content": "| a1 | Deliver | task | inspect: Deliver |"},
                                    {"name": "applicable-answer.txt", "content": selected + "\n" + global_rule}]}]
            budget = SelectionBudget(root, {})
            with patch.object(budget, "call", side_effect=responses) as call:
                write_draft(selection, {}, root / "draft", root / "spec", budget)
            self.assertEqual(call.call_count, 2)
            self.assertNotIn(global_rule, str(call.call_args_list[0].args[1]))
            answer = read(root / "spec/oracle-answer.json")["answer"]
            self.assertEqual(answer, "- " + selected + "\n- " + global_rule)
            self.assertNotIn(unrelated, answer)
            self.assertIn("不要求每个问题都影响交付", SELECT_TASK)
            self.assertIn("可采用一题或多题", EXTERNAL_ACCEPTANCE)

    def test_private_author_cannot_invent_an_applicable_fact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection = dict(qa_source="external", public={}, historical_answer="- Public historical fact.")
            responses = [{"files": [{"name": "task.md", "content": "Deliver report."}]},
                         {"files": [{"name": "memory-use.md", "content": "Use invented fact."},
                                    {"name": "acceptance.md", "content": "table"},
                                    {"name": "applicable-answer.txt", "content": "Invented fact."}]}]
            budget = SelectionBudget(root, {})
            with patch.object(budget, "call", side_effect=responses):
                with self.assertRaisesRegex(ValueError, "quote supplied public facts"):
                    write_draft(selection, {}, root / "draft", root / "spec", budget)

    def test_solver_two_rounds_share_budget_and_keep_first_and_failure_cost(self):
        for fail_second, max_requests, expected in [(False, 8, 2), (True, 8, 2), (False, 1, 1)]:
            with self.subTest(fail_second=fail_second, max_requests=max_requests), tempfile.TemporaryDirectory() as directory:
                workers = []
                class Budget:
                    def __init__(self, config, journal):
                        self.deadline = time.monotonic() + 60
                        self.data = dict(attempts=0, prompt_tokens=0, completion_tokens=0)
                class Worker:
                    def __init__(self, *args, **kwargs):
                        self.budget = kwargs["budget"]
                        self.messages = []
                        workers.append(self)
                    def start(self): pass
                    def close(self): pass
                    def events(self): return []
                    def turn(self, message):
                        self.messages.append(message)
                        self.budget.data["attempts"] += 1
                        self.budget.data["prompt_tokens"] += 5
                        if fail_second and len(self.messages) == 2:
                            raise RuntimeError("local second turn failure")
                        return {"status": "finished"}
                checkpoints = []
                with patch.dict("sys.modules", {
                        "simulator.openhands.budget": SimpleNamespace(Budget=Budget),
                        "simulator.openhands.container": SimpleNamespace(SDKContainer=Worker)}), \
                     patch("dialogue_benchmark.task_eval.retention.release_agent"), \
                     patch("dialogue_benchmark.task_eval.retention.save_trace"):
                    result = run_agent(Path(directory), {"code": {}, "image": "test"}, "code", "Deliver report",
                                       max_requests=max_requests, max_rounds=2,
                                       on_round=lambda number, row: checkpoints.append((number, row)))
                self.assertEqual(len(workers), 1)
                self.assertEqual(len(workers[0].messages), expected)
                self.assertEqual(len(result["rounds"]), expected)
                self.assertEqual(result["rounds"][0]["metrics"]["attempted_requests"], 1)
                self.assertEqual(result["metrics"]["attempted_requests"], expected)
                self.assertEqual(checkpoints[-1][0], expected)
                if expected == 2:
                    self.assertNotIn("reference", workers[0].messages[1])
                    self.assertNotIn("acceptance", workers[0].messages[1])
                if fail_second:
                    self.assertEqual(result["rounds"][-1]["status"], "error")
