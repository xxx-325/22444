import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dialogue_benchmark.task_eval.artifacts import read, save
from dialogue_benchmark.task_eval.history import (
    answer_clarification, freeze_contract, historical_context, prepare_history, read_history_review)
from dialogue_benchmark.task_eval.runtime import run_agent, configure
from dialogue_benchmark.task_eval.run import evaluate, freeze
from dialogue_benchmark.task_eval.versions import pin_baseline
import test_task_eval_flow


CONTRACT = """REVIEW h1
statement: Preserve blank values for all tenants.
scope: All tenants before the correction; non-EU tenants afterwards.
sources: old
supersedes: none
behavior: Preserve explicit blank values.
repository: external
END_REVIEW
REVIEW h2
statement: Reject blank values for EU tenants.
scope: EU tenants after the correction only.
sources: correction
supersedes: h1
behavior: EU blanks are rejected; other tenants keep the original rule.
repository: external
END_REVIEW
"""


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.history = {"cutoff_event_id": "correction", "events": [
            {"id": "old", "order": 1, "text": "Preserve all blanks."},
            {"id": "correction", "order": 9, "text": "EU tenants must reject blanks."}]}
        (self.root / "history-contract.txt").write_text(CONTRACT)

    def test_scoped_replacement_and_oracle_are_frozen_without_collapsing_history(self):
        history = freeze_contract(self.root, self.history, "EU changed; other tenants unchanged.")
        self.assertEqual(len(history["contracts"]), 2)
        self.assertEqual(history["contracts"][1]["supersedes"], ["h1"])
        self.assertIn("non-EU", history["contracts"][0]["scope"])
        self.assertNotIn("h1", historical_context(history))
        self.assertEqual(history["oracle_sufficiency"], "not_established_by_reference")
        for invalid in (CONTRACT.replace("sources: correction", "sources: private"),
                        CONTRACT.replace("sources: correction", "sources: old"),
                        CONTRACT.replace("supersedes: h1", "supersedes: missing")):
            (self.root / "history-contract.txt").write_text(invalid)
            with self.assertRaises(ValueError):
                freeze_contract(self.root, self.history, "answer")

    def test_full_public_updates_available_but_seed_evidence_stays_explicit(self):
        save(self.root / "input.json", {"payload": {}, "ref_to_source": {"资料1": "e1"}})
        records = [dict(e, id="e%d" % (i + 1), original_id=e["id"], kind="message")
                   for i, e in enumerate(self.history["events"])]
        result = prepare_history(records, self.root / "input.json")
        self.assertEqual(result["selected_event_ids"], ["old"])
        self.assertEqual(result["events"][-1]["id"], "correction")

    def test_missing_or_unfinished_judge_does_not_imply_application(self):
        history = freeze_contract(self.root, self.history, "answer")
        path = self.root / "review.txt"
        path.write_text("REVIEW h1\nstatus: applied\nevidence: none\nEND_REVIEW\n"
                        "REVIEW h2\nstatus: applied\nevidence: test rejects EU blanks\nEND_REVIEW")
        self.assertEqual(read_history_review(path, history, True)["counts"]["insufficient"], 1)
        self.assertEqual(read_history_review(path, history, False)["counts"]["insufficient"], 2)

    def test_responder_cannot_invent_source_or_see_oracle_and_criteria(self):
        history = freeze_contract(self.root, self.history, "PRIVATE_ORACLE")
        decision = {"status": "answer", "sources": "correction", "reply": "EU rejects blanks.",
                    "kind": "historical_reask"}
        with patch("dialogue_benchmark.task_eval.runtime.ask_model", return_value={"reviews": [decision]}) as ask:
            answer_clarification("What is the EU rule?", history, [], {}, self.root / "reply")
            payload = ask.call_args.args[1]
            self.assertNotIn("PRIVATE_ORACLE", str(payload))
            self.assertNotIn("behavior", str(payload))
            decision["sources"] = "private"
            with self.assertRaises(ValueError):
                answer_clarification("Question", history, [], {}, self.root / "reply")

    def test_same_worker_continues_only_for_grounded_answer(self):
        history = freeze_contract(self.root, self.history, "answer")
        workers = []
        class Budget:
            def __init__(self, config, journal):
                self.deadline = time.monotonic() + 60
                self.data = {"attempts": 0, "prompt_tokens": 0, "completion_tokens": 0}
        class Worker:
            def __init__(self, *args, **kwargs):
                self.messages = []
                workers.append(self)
            def start(self): pass
            def close(self): pass
            def turn(self, text):
                self.messages.append(text)
                return {"status": "finished"}
            def events(self):
                return [{"id": "finish%d" % len(self.messages), "kind": "ActionEvent",
                         "tool_name": "finish", "action": {"message": "Question" if len(self.messages) == 1 else "Done"}}]
        decisions = [{"status": "answer", "sources": ["correction"], "reply": "EU rejects blanks.",
                      "kind": "historical_reask"},
                     {"status": "no_question", "sources": [], "reply": "none", "kind": "none"}]
        with patch.dict("sys.modules", {
            "simulator.openhands.budget": SimpleNamespace(Budget=Budget),
            "simulator.openhands.container": SimpleNamespace(SDKContainer=Worker)}), \
             patch("dialogue_benchmark.task_eval.history.answer_clarification", side_effect=decisions), \
             patch("dialogue_benchmark.task_eval.retention.save_trace"), \
             patch("dialogue_benchmark.task_eval.retention.release_agent"):
            result = run_agent(self.root / "trial", {"code": {}, "image": "test"}, "code", "Implement",
                               history=history)
        self.assertEqual(len(workers), 1)
        self.assertEqual(workers[0].messages, ["Implement", "EU rejects blanks."])
        self.assertEqual(result["clarification_status"], "no_question")
        self.assertTrue(result["clarifications"][0]["delivered"])
        self.assertFalse(result["clarifications"][1]["delivered"])
        self.assertEqual(result["responder_cost"]["requests"], 2)

    def test_independent_config_needs_no_private_checkpoint(self):
        config = {"image": "sdk", "execution_image": "runtime", "execution_backend": "local",
                  "code": {"model": "code", "key_env": "TEST_KEY"},
                  "judge": {"model": "judge", "key_env": "TEST_KEY"}}
        save(self.root / "config.json", config)
        with patch.dict("sys.modules", {"simulator.episode": SimpleNamespace(load_environment=lambda _: None)}), \
             patch.dict("os.environ", {"TEST_KEY": "test"}):
            result = configure(self.root, "missing", "unused", control_config=self.root / "config.json")
            self.assertIsNone(result["code"]["max_output_tokens"])
            config["code"]["api_key"] = "not-allowed"
            save(self.root / "config.json", config)
            with self.assertRaises(ValueError):
                configure(self.root, "missing", "unused", control_config=self.root / "config.json")

    def test_two_arms_share_history_responder_but_only_oracle_gets_answer(self):
        baseline, task = self.root / "base", self.root / "task"
        baseline.mkdir()
        pin_baseline(baseline)
        spec = self.root / "spec"
        spec.mkdir()
        (spec / "history-contract.txt").write_text(CONTRACT)
        history = freeze_contract(spec, self.history, "ORACLE_ONLY")
        for name in ("task.md", "acceptance.md", "checkpoints.md"):
            (spec / name).write_text("Implement the previously agreed tenant behavior")
        save(spec / "checkpoints.json", {"status": "unavailable", "checkpoints": []})
        receipt = freeze(spec, task / "frozen", baseline)
        receipt["validation"] = {"TESTS": "executable"}
        prompts_seen = []
        def agent(root, config, role, message, **kwargs):
            if role == "code":
                self.assertEqual(kwargs["history"], history)
                prompts_seen.append(message)
                save(root / "trajectory.json", [])
                return {"status": "finished", "metrics": {}, "clarification_status": "no_question",
                        "clarifications": []}
            reference = kwargs["reference"]
            self.assertNotIn("ORACLE_ONLY", str(read(reference / "spec/history.json")))
            checks = root / "workspace/checks"
            checks.mkdir()
            (checks / "verdict.txt").write_text("RESULT: passed")
            # A passing functional result does not excuse a historical violation.
            status = "violated" if root.parent.name == "trial-1" else "applied"
            (checks / "history-review.txt").write_text(
                "REVIEW h1\nstatus: applied\nevidence: code retains non-EU values\nEND_REVIEW\n"
                "REVIEW h2\nstatus: %s\nevidence: tests exercise EU values\nEND_REVIEW" % status)
            return {"status": "finished"}
        with patch("dialogue_benchmark.task_eval.run.run_agent", side_effect=agent), \
             patch("dialogue_benchmark.task_eval.run.run_checks", return_value={"status": "passed"}):
            result = evaluate({"qa": {}}, task, baseline, receipt, {"execution_image": "fake"}, {}, 0)
        self.assertNotIn("ORACLE_ONLY", prompts_seen[0])
        self.assertIn("ORACLE_ONLY", prompts_seen[1])
        self.assertEqual(result["without_memory"]["result"], "failed")
        self.assertEqual(result["with_memory"]["result"], "passed")
        self.assertEqual(result["with_memory"]["information_condition"], "oracle_history")
        self.assertIsNone(result["with_memory"]["checkpoints"]["action_coverage"])
        from dialogue_benchmark.task_eval.report import write_report
        write_report(task, {"tasks": [{"task": "task-01", "status": "evaluated", "comparison": result}]})
        report = (task / "report.md").read_text()
        self.assertIn("Cross-session re-asks", report)
        self.assertIn("oracle_history", report)
        self.assertIn("Applied / violated", report)
        self.assertIn("Aggregate execution costs", report)


class HistoryConstructionTests(unittest.TestCase):
    def test_informed_reference_admits_without_unassisted_checkpoint(self):
        fixture = test_task_eval_flow.TaskPreflightTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        save(Path(fixture.item["generation_input"]), {"payload": {}, "ref_to_source": {"资料1": "e1"}})
        fixture.item["public_records"] = [
            {"id": "e1", "original_id": "old", "order": 1, "kind": "message", "text": "Preserve blank"},
            {"id": "e2", "original_id": "correction", "order": 2, "kind": "message", "text": "EU rejects blank"}]
        original = fixture.fake_agent
        def agent(root, config, role, message, **kwargs):
            result = original(root, config, role, message, **kwargs)
            if root.name == "author":
                (root / "workspace/checks/history-contract.txt").write_text(CONTRACT)
            if root.name == "validator":
                path = root / "workspace/checks/validation.txt"
                path.write_text(path.read_text() + "HISTORY: supported\n")
            if root.name == "reference-solver":
                self.assertIn("Historical behavior", message)
                self.assertIn("EU tenants", message)
                self.assertNotIn("behavior:", message)
            return result
        fixture.fake_agent = agent
        receipt, _ = fixture.execute([{"status": "failed"}, {"status": "passed"},
                                      {"status": "failed"}, {"status": "passed"}])
        self.assertIsNotNone(receipt)
        self.assertEqual(receipt["comparison"], "without_memory_vs_oracle_history")
        self.assertEqual(read(fixture.root / "frozen/checkpoints.json")["status"], "unavailable")
        self.assertEqual(read(fixture.root / "frozen/history.json")["oracle_answer"], "- Historical behavior")
