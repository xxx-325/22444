import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from dialogue_benchmark.llm import parse_text_response
from dialogue_benchmark.task_eval.artifacts import read, save
from dialogue_benchmark.task_eval import retention
from dialogue_benchmark.task_eval.history import (
    answer_clarification, freeze_contract, freeze_targets, historical_context, prepare_history,
    read_history_review, qualified_oracle_complete, validate_contract_targets, review_history, review_sources)
from dialogue_benchmark.task_eval.runtime import run_agent, configure, repair_tests, write_tests, review_checks, write_history_mutation
from dialogue_benchmark.task_eval.run import evaluate, freeze
from dialogue_benchmark.task_eval.versions import pin_baseline
import test_task_eval_flow


CONTRACT = """REVIEW h1
statement: Preserve blank values for all tenants.
scope: All tenants before the correction; non-EU tenants afterwards.
sources: old
supersedes: none
behavior: Preserve explicit blank values.
active: yes
repository: external
END_REVIEW
REVIEW h2
statement: Reject blank values for EU tenants.
scope: EU tenants after the correction only.
sources: correction
supersedes: h1
behavior: EU blanks are rejected; other tenants keep the original rule.
active: yes
repository: external
END_REVIEW
"""


def qualification(history):
    """A complete per-target decision with quotes from the actual supplied answer."""
    return {"status": "clean", "issue": "none", "task_review": {"id": "task", "leakage": "clean"},
            "history_rows": [dict(id=rule["id"], applicable="yes" if rule["active"] else "no",
                                  public="full" if rule["repository"] == "recoverable" else "none",
                                  answer="sufficient" if rule["repository"] == "external" else "not_applicable",
                                  historical_sources=rule["sources"], answer_quote=history["oracle_answer"])
                             for rule in history["contracts"]]}


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
        self.assertEqual(result["qa_source_ids"], ["old"])
        self.assertEqual(result["events"][-1]["id"], "correction")

    def test_readable_history_retains_sources_and_later_corrections(self):
        self.history["events"].extend([
            {"id": "unrelated", "order": 10, "kind": "tool_result", "text": "BULK_TOOL_LOG"},
            {"id": "later", "order": 11, "kind": "message", "role": "user",
             "text": "EU rule also covers archived accounts."}])
        frozen = freeze_contract(self.root, self.history, "EU changed; other tenants unchanged.")
        review = (self.root / "history-review.md").read_text()
        for event in self.history["events"]:
            if event["id"] == "unrelated":
                self.assertNotIn(event["text"], review)
            else:
                self.assertIn(event["id"], review)
                self.assertIn(event["text"], review)
        self.assertIn("Supersedes: h1", review)
        self.assertIn("EU changed; other tenants unchanged.", review)
        self.assertEqual(read(self.root / "history.json")["events"], frozen["events"])
        projection = review_history(frozen)
        self.assertEqual([e["id"] for e in projection["events"]], ["old", "correction", "later"])
        self.assertEqual(len(frozen["events"]), 4)
        self.assertEqual(projection["contracts"], [
            {key: row[key] for key in ("id", "statement", "scope", "sources", "supersedes", "active")}
            for row in frozen["contracts"]])

    def test_fragment_answer_source_resolves_to_public_event_identity(self):
        save(self.root / "input.json", {"payload": {}})
        records = [{"id": "e1", "original_id": "original-event", "order": 1,
                    "kind": "message", "role": "user", "text": "External rule"}]
        result = prepare_history(records, self.root / "input.json", qa_source_ids={"e1#fragment-2"})
        self.assertEqual(result["qa_source_ids"], ["original-event"])
        self.assertEqual(result["initial_events"][0]["id"], "original-event")

    def test_generation_materials_keep_public_correction_closure(self):
        save(self.root / "input.json", {
            "payload": {"materials": [
                {"reference": "资料1", "text": "Original rule"},
                {"reference": "资料2", "text": "PRIVATE_MATERIAL_TEXT"},
                {"reference": "资料3", "text": "PRIVATE_PLAN"}]},
            "ref_to_source": {"资料1": "e1#fragment-2", "资料2": "e3",
                              "资料3": "private", "资料4": "e4"}})
        records = [
            {"id": "e1", "original_id": "old", "order": 1, "kind": "message",
             "role": "user", "text": "All tenants preserve blanks."},
            {"id": "e2", "original_id": "tool", "order": 2, "kind": "result",
             "role": "tool", "text": "Unrelated bulk output"},
            {"id": "e3", "original_id": "correction", "order": 3, "kind": "message",
             "role": "user", "text": "EU tenants must reject blanks."},
            {"id": "e4", "original_id": "unrelated", "order": 4, "kind": "message",
             "role": "user", "text": "Update the page style."}]
        for sources, expected_cited in (({"e1#fragment-2"}, ["old"]),
                ({"e1#fragment-2", "e3"}, ["old", "correction"])):
            with self.subTest(sources=sources):
                result = prepare_history(records, self.root / "input.json", qa_source_ids=sources)
                self.assertEqual(result["qa_source_ids"], expected_cited)
                self.assertEqual(result["selected_event_ids"], ["old", "correction"])
                self.assertEqual([e["id"] for e in result["initial_events"]], ["old", "correction"])
                self.assertEqual(result["initial_events"][1]["text"], records[2]["text"])
                self.assertNotIn("PRIVATE_", str(result))

    def test_source_review_checks_original_history_without_rechecking_answer_quotes(self):
        history = freeze_contract(self.root, self.history, "EU rejects blanks; other tenants preserve blanks.")
        self.assertTrue(all(event["text"] not in history["oracle_answer"] for event in history["events"]))
        rows = [dict(id=cid, support="supported", issue="none") for cid in ("h1", "h2")]
        with patch.object(SimpleNamespace(), "call", create=True, return_value={"reviews": rows}) as call:
            result = review_sources(history, {}, self.root / "source-review", SimpleNamespace(call=call))
        call.assert_called_once()
        self.assertEqual(set(call.call_args.args[1]), {"events", "contracts", "cutoff_event_id"})
        self.assertEqual(call.call_args.args[1]["events"], history["events"])
        self.assertEqual(result, {"support": "supported", "rows": rows})
        self.assertFalse((self.root / "source-review/oracle-review.txt").exists())

    def test_source_review_preserves_uncertain_replacement_and_old_scope(self):
        history = freeze_contract(self.root, self.history, "EU rejects blanks; other tenants preserve blanks.")
        for issue in ("The replacement direction contradicts the later correction.",
                      "The old tenant scope was not confirmed after the correction."):
            for support in ("unsupported", "uncertain"):
                with self.subTest(issue=issue, support=support):
                    rows = [dict(id="h1", support=support, issue=issue),
                            dict(id="h2", support="supported", issue="none")]
                    result = review_sources(history, {}, self.root / "source-review",
                                            SimpleNamespace(call=lambda *args: {"reviews": rows}))
                    self.assertEqual(result["support"], support)
                    self.assertEqual(result["rows"][0]["issue"], issue)

    def test_source_review_missing_duplicate_or_failed_response_cannot_pass(self):
        history = freeze_contract(self.root, self.history, "EU rejects blanks; other tenants preserve blanks.")
        row = dict(id="h1", support="supported", issue="none")
        for rows in ([], [row], [row, row], [dict(row, support="yes"), dict(row, id="h2")],
                     [dict(row, issue=""), dict(row, id="h2")],
                     [dict(row, support="uncertain"), dict(row, id="h2")]):
            with self.subTest(rows=rows):
                result = review_sources(history, {}, self.root / "source-review",
                                        SimpleNamespace(call=lambda *args: {"reviews": rows}))
                self.assertEqual(result["support"], "uncertain")
                self.assertNotIn("oracle_complete", result)

    def test_mixed_contract_checks_oracle_only_for_external_rule(self):
        (self.root / "history-contract.txt").write_text(CONTRACT.replace(
            "repository: external", "repository: recoverable", 1))
        history = freeze_contract(self.root, self.history, "EU rejects blanks.")
        review = qualification(history)
        self.assertTrue(qualified_oracle_complete(history, review))
        review["history_rows"][1]["answer"] = "insufficient"
        self.assertFalse(qualified_oracle_complete(history, review))
        (self.root / "history-contract.txt").write_text(CONTRACT.replace("external", "recoverable"))
        with self.assertRaises(ValueError):
            freeze_contract(self.root, self.history, "answer")

    def test_incomplete_or_uncertain_qualification_cannot_supply_oracle_coverage(self):
        history = freeze_contract(self.root, self.history, "EU rejects blanks; other tenants preserve blanks.")
        complete = qualification(history)
        for review in ({"status": "clean"}, dict(complete, status="uncertain"),
                       dict(complete, history_rows=complete["history_rows"][:1]),
                       dict(complete, history_rows=[complete["history_rows"][0]] * 2)):
            self.assertFalse(qualified_oracle_complete(history, review))
        for patch_row in ({"answer": "uncertain"}, {"answer": "insufficient"},
                          {"answer_quote": "Preserve all blanks."},
                          {"historical_sources": []}, {"historical_sources": ["unknown"]}):
            review = qualification(history)
            review["history_rows"][0].update(patch_row)
            self.assertFalse(qualified_oracle_complete(history, review))

    def test_source_aliases_are_exact_and_prefixes_are_rejected(self):
        self.history["source_aliases"] = {"source1": "old", "source2": "correction"}
        (self.root / "history-contract.txt").write_text(CONTRACT.replace("sources: old", "sources: source1"))
        self.assertEqual(freeze_contract(self.root, self.history, "answer")["contracts"][0]["sources"], ["old"])
        (self.root / "history-contract.txt").write_text(CONTRACT.replace("sources: old", "sources: ol"))
        with self.assertRaises(ValueError):
            freeze_contract(self.root, self.history, "answer")

    def test_history_targets_are_frozen_before_repository_and_answer_labels(self):
        self.history["source_aliases"] = {"source1": "old", "source2": "correction"}
        targets = freeze_targets([{"id": "h1", "statement": "EU rejects blanks",
                                   "scope": "EU tenants", "behavior": "reject",
                                   "sources": "source2", "supersedes": "none"}], self.history)
        self.assertEqual(targets["targets"][0]["sources"], ["correction"])
        rows = [{"id": "h1", "statement": "EU rejects blanks", "scope": "EU tenants",
                 "behavior": "reject", "sources": "correction", "supersedes": "none",
                 "active": "yes", "repository": "recoverable"}]
        validate_contract_targets(rows, targets)
        rows[0]["statement"] = "EU accepts blanks"
        with self.assertRaises(ValueError):
            validate_contract_targets(rows, targets)

    def test_freezing_targets_does_not_require_an_external_gap(self):
        targets = freeze_targets([{"id": "h1", "statement": "Preserve all blanks.",
                                   "scope": "all tenants", "behavior": "preserve",
                                   "sources": "old", "supersedes": "none"}], self.history)
        (self.root / "history-contract.txt").write_text(
            "REVIEW h1\nstatement: Preserve all blanks.\nscope: all tenants\n"
            "sources: old\nsupersedes: none\nbehavior: preserve\nactive: yes\n"
            "repository: recoverable\nEND_REVIEW\n")
        frozen = freeze_contract(self.root, self.history, "answer", require_external=False, targets=targets)
        self.assertEqual(frozen["contracts"][0]["repository"], "recoverable")

    def test_focus_keeps_user_correction_and_plan_without_unrelated_tools(self):
        save(self.root / "input.json", {"ref_to_source": {"r1": "e1"}})
        records = [dict(id="e%d" % i, original_id="uuid%d" % i, order=i, kind=kind,
                        role=role, text=text) for i, kind, role, text in (
            (1, "message", "assistant", "The plan"),
            (2, "tool_result", "tool", "Unrelated bulk output"),
            (3, "message", "user", "Follow that plan, except EU."))]
        result = prepare_history(records, self.root / "input.json")
        self.assertEqual([e["id"] for e in result["initial_events"]], ["uuid1", "uuid3"])
        self.assertEqual(len(result["events"]), 3)

    def test_history_uses_same_acceptance_without_subjective_override(self):
        history = freeze_contract(self.root, self.history, "answer")
        result = read_history_review({"rows": [
            {"id": "a1", "basis": ["h1"], "status": "passed", "evidence": "test"},
            {"id": "a2", "basis": ["h2"], "status": "failed", "evidence": "wrong output"}]}, history)
        self.assertEqual(result["counts"]["applied"], 1)
        self.assertEqual(result["counts"]["violated"], 1)

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
                         "tool_name": "finish", "action": {"message": "HISTORY_QUESTION: What is the EU rule?" if len(self.messages) == 1 else "Done"}}]
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
        self.assertEqual(len(result["clarifications"]), 1)
        self.assertEqual(result["responder_cost"]["requests"], 1)

    def test_responder_resolves_frozen_rule_citations_to_public_events(self):
        history = freeze_contract(self.root, self.history, "PRIVATE_ORACLE")
        decision = {"status": "answer", "sources": "h2,correction", "reply": "EU rejects blanks.",
                    "kind": "historical_reask"}
        with patch("dialogue_benchmark.task_eval.runtime.ask_model", return_value={"reviews": [decision]}) as ask:
            result = answer_clarification("What is the EU rule?", history, [], {}, self.root / "reply")
            self.assertEqual(result["sources"], ["correction"])
            self.assertEqual(result["reply"], decision["reply"])
            self.assertIn("correction,old", ask.call_args.args[0])
            decision.update(kind="same_session_repeat", sources="h2")
            repeat = answer_clarification("Repeat that rule", history,
                [{"delivered": True, "sources": ["correction"]}], {}, self.root / "repeat")
            self.assertEqual(repeat["sources"], ["correction"])
            for invalid in ("h3", "h2,unknown", "h"):
                decision.update(kind="historical_reask", sources=invalid)
                with self.subTest(source=invalid), self.assertRaises(ValueError):
                    answer_clarification("Question", history, [], {}, self.root / "invalid")

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

    def test_finished_at_budget_limit_never_calls_responder(self):
        history = freeze_contract(self.root, self.history, "answer")
        class Budget:
            def __init__(self, config, journal):
                self.deadline = time.monotonic() - 1
                self.data = {"attempts": 1, "prompt_tokens": 1, "completion_tokens": 1}
        class Worker:
            def __init__(self, *args, **kwargs): pass
            def start(self): pass
            def close(self): pass
            def turn(self, message): return {"status": "finished"}
            def events(self):
                return [{"kind": "ActionEvent", "tool_name": "finish", "action": {"message": "Done"}}]
        with patch.dict("sys.modules", {
                "simulator.openhands.budget": SimpleNamespace(Budget=Budget),
                "simulator.openhands.container": SimpleNamespace(SDKContainer=Worker)}), \
             patch("dialogue_benchmark.task_eval.history.answer_clarification") as responder, \
             patch("dialogue_benchmark.task_eval.retention.save_trace"), \
             patch("dialogue_benchmark.task_eval.retention.release_agent"):
            result = run_agent(self.root / "limit", {"code": {}, "image": "test"}, "code", "Implement",
                               history=history, max_requests=1)
        responder.assert_not_called()
        self.assertEqual(result["status"], "finished")
        self.assertEqual(result["clarification_status"], "no_question")
        self.assertEqual(result["clarifications"], [])

    def test_two_arms_share_history_responder_but_only_oracle_gets_answer(self):
        baseline, task = self.root / "base", self.root / "task"
        baseline.mkdir()
        pin_baseline(baseline)
        spec = self.root / "spec"
        spec.mkdir()
        (spec / "history-contract.txt").write_text(CONTRACT)
        history = freeze_contract(spec, self.history, "ORACLE_ONLY")
        for name in ("task.md", "acceptance.md"):
            (spec / name).write_text("Implement the previously agreed tenant behavior")
        for name in ("history-review.md", "memory-use.md"):
            (spec / name).write_text("The injected answer was ORACLE_ONLY")
        save(spec / "acceptance.json", [
            {"id": "a1", "requirement": "Preserve blanks", "basis": ["h1"], "tests": ["test::non_eu"]},
            {"id": "a2", "requirement": "Reject EU blanks", "basis": ["h2"], "tests": ["test::eu"]}])
        receipt = freeze(spec, task / "frozen", baseline)
        receipt["validation"] = {"TESTS": "executable"}
        prompts_seen = []
        def agent(root, config, role, message, **kwargs):
            if role == "code":
                self.assertEqual(kwargs["history"], history)
                prompts_seen.append(message)
                save(root / "trajectory.json", [])
                return {"status": "finished", "metrics": {}, "clarification_status": "budget_exhausted",
                        "clarifications": [{"question": "EU rule?", "kind": "historical_reask", "delivered": True}]}
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
             patch("dialogue_benchmark.task_eval.run.run_checks", side_effect=[
                 {"status": "failed", "cases": [{"id": "test::non_eu", "status": "passed"}, {"id": "test::eu", "status": "failed"}]},
                 {"status": "passed", "cases": [{"id": "test::non_eu", "status": "passed"}, {"id": "test::eu", "status": "passed"}]}]):
            result = evaluate({"qa": {}}, task, baseline, receipt, {"execution_image": "fake"}, {}, 0)
        self.assertNotIn("ORACLE_ONLY", prompts_seen[0])
        self.assertIn("ORACLE_ONLY", prompts_seen[1])
        for trial in ("trial-1", "trial-2"):
            projected = task / trial / "judge-reference/spec"
            self.assertFalse((projected / "history-review.md").exists())
            self.assertFalse((projected / "memory-use.md").exists())
            self.assertNotIn("ORACLE_ONLY", str(read(projected / "history.json")))
        self.assertTrue((task / "frozen/history-review.md").exists())
        self.assertEqual(result["without_memory"]["result"], "failed")
        self.assertEqual(result["with_memory"]["result"], "passed")
        self.assertEqual(result["with_memory"]["history_question_count"], 1)
        self.assertEqual(result["with_memory"]["information_condition"], "oracle_history")
        self.assertNotIn("checkpoints", result["with_memory"])
        from dialogue_benchmark.task_eval.report import write_report
        write_report(task, {"tasks": [{"task": "task-01", "status": "evaluated", "comparison": result}]})
        report = (task / "report.md").read_text()
        self.assertIn("History questions", report)
        self.assertIn("oracle_history", report)
        self.assertIn("Frozen acceptance", report)
        self.assertIn("Aggregate execution costs", report)


class HistoryConstructionTests(unittest.TestCase):
    def construct_test_retries(self, *, before_retry=None, author_change=None, revisions=2):
        fixture = test_task_eval_flow.TaskPreflightTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        save(Path(fixture.item["generation_input"]), {"ref_to_source": {"资料1": "e1"}})
        fixture.item["public_records"] = [
            {"id": "e1", "original_id": "old", "order": 1, "kind": "message", "text": "Preserve all blanks"},
            {"id": "e2", "original_id": "correction", "order": 2, "kind": "message", "text": "EU rejects blanks"}]
        fixture.item["qa"]["answer_points"] = [{"text": "EU rejects blanks; other tenants preserve blanks."}]
        public_history = prepare_history(fixture.item["public_records"], fixture.item["generation_input"])
        targets = freeze_targets(parse_text_response(CONTRACT)["reviews"], public_history)
        histories = []

        def draft(selection, config, output, spec, budget, feedback=""):
            fixture.fake_draft(selection, config, output, spec, budget, feedback)
            (spec / "history-contract.txt").write_text(CONTRACT)
            path = spec / "acceptance.md"
            path.write_text(path.read_text() +
                            "\n| a2 | Other blanks retained | h1 | test: test_acceptance::test_other |\n"
                            "| a3 | EU blanks rejected | h2 | test: test_acceptance::test_eu |\n")

        def task_review(task, answer, config, output, *, evidence, budget):
            result = qualification({"oracle_answer": answer, "contracts": evidence["contracts"]})
            save(output / "result.json", result)
            return result

        def agent(root, *args, **kwargs):
            if root.name == "author":
                spec = root / "workspace/checks"
                (spec / "test_acceptance.py").write_text("# Tests for " + root.parent.name + "\n")
                if root.parent.name == "construction-01" and author_change:
                    author_change(spec)
                return {"status": "finished"}
            return fixture.fake_agent(root, *args, **kwargs)

        def repair(spec, baseline, config, output, budget, feedback):
            from dialogue_benchmark.task_eval.run import run_agent
            return run_agent(output, config, "judge", feedback)

        def checks(candidate, spec, output, *args, **kwargs):
            status = "failed" if candidate == fixture.baseline else "passed"
            if output.parent.name == "construction-01" and output.name == "final-reference-checks":
                status = "failed"
            return {"status": status, "cases": [
                {"id": "test_acceptance::test_" + name, "status": status}
                for name in ("feature", "other", "eu")]}

        def source_review(history, *args):
            histories.append(history)
            return {"support": "supported"}

        def coverage_review(spec, baseline, candidate, changed, results, config, output, budget):
            output.mkdir(parents=True)
            (output / "coverage.md").write_text("Each acceptance row is covered.")
            first = output.parent.name == "construction-00"
            if first and before_retry:
                before_retry(fixture, output.parent / "qualified-draft", histories[-1])
            return {"status": "revise" if first else "complete", "rows": []}

        def mutation_files(spec, candidate, changed, config, output, budget):
            (output / "workspace").mkdir(parents=True)
            return fixture.fake_agent(output, config, "judge", "Generate mutation files")

        with patch("dialogue_benchmark.task_eval.run.select_task", return_value={
                "status": "candidate", "history_targets": targets}), \
             patch("dialogue_benchmark.task_eval.run.write_draft", side_effect=draft) as draft_call, \
             patch("dialogue_benchmark.task_eval.run.review_task", side_effect=task_review) as review_call, \
             patch("dialogue_benchmark.task_eval.run.run_agent", side_effect=agent) as agent_call, \
             patch("dialogue_benchmark.task_eval.run.repair_tests", side_effect=repair), \
             patch("dialogue_benchmark.task_eval.run.run_checks", side_effect=checks) as check_call, \
             patch("dialogue_benchmark.task_eval.run.review_sources", side_effect=source_review) as source_call, \
             patch("dialogue_benchmark.task_eval.run.review_checks", side_effect=coverage_review) as coverage_call, \
             patch("dialogue_benchmark.task_eval.run.write_history_mutation", side_effect=mutation_files) as mutation_call, \
             patch("dialogue_benchmark.task_eval.run.check_history_mutations", return_value={"status": "caught"}) as replay_call:
            from dialogue_benchmark.task_eval.run import construct
            receipt = construct(fixture.item, fixture.root, fixture.baseline,
                                {"execution_image": "image"}, revisions, {})
        calls = SimpleNamespace(draft=draft_call, review=review_call, agent=agent_call,
                                checks=check_call, sources=source_call, coverage=coverage_call,
                                mutations=mutation_call, replay=replay_call)
        return fixture, receipt, calls

    def test_fixed_draft_retries_reuse_qualification_but_repeat_required_checks(self):
        fixture, receipt, calls = self.construct_test_retries()
        self.assertIsNotNone(receipt)
        self.assertEqual(receipt["accepted_attempt"], 2)
        calls.draft.assert_called_once()
        calls.review.assert_called_once()
        self.assertEqual([call.args[0].name for call in calls.agent.call_args_list],
                         ["author", "reference-solver"] * 3)
        self.assertEqual(calls.sources.call_count, 3)
        self.assertEqual(calls.coverage.call_count, 3)
        self.assertEqual(calls.mutations.call_count, 2)
        self.assertEqual(calls.replay.call_count, 2)
        self.assertEqual([(call.args[2].parent.name, call.args[2].name)
                          for call in calls.checks.call_args_list], [
            ("construction-00", "baseline-checks"), ("construction-00", "reference-checks"),
            ("construction-01", "baseline-checks"), ("construction-01", "reference-checks"),
            ("construction-01", "final-baseline-checks"), ("construction-01", "final-reference-checks"),
            ("construction-02", "baseline-checks"), ("construction-02", "reference-checks"),
            ("construction-02", "final-baseline-checks"), ("construction-02", "final-reference-checks")])
        records = read(fixture.root / "construction.json")
        self.assertEqual([record["accepted"] for record in records], [False, False, True])
        self.assertEqual(records[0]["reason"], "checks_revise")
        self.assertEqual(records[1]["final_reference_checks"]["status"], "failed")
        first = fixture.root / "construction-00"
        for attempt in (1, 2):
            current = fixture.root / ("construction-%02d" % attempt)
            reused_review = read(current / "task-review/result.json")
            self.assertEqual(reused_review.pop("reused_from"), "construction-00/task-review")
            self.assertEqual(reused_review.pop("usage"), [])
            self.assertEqual(reused_review, read(first / "task-review/result.json"))
            for name in ("task.md", "memory-use.md", "history-contract.txt", "history.json"):
                self.assertEqual((current / "qualified-draft" / name).read_bytes(),
                                 (first / "qualified-draft" / name).read_bytes())
        self.assertIn("construction-02", (fixture.root / "frozen/test_acceptance.py").read_text())

    def test_changed_fixed_draft_or_history_cannot_reuse_clean_qualification(self):
        for change in ("task", "contract", "requirements", "answer", "source"):
            with self.subTest(change=change):
                def before_retry(fixture, draft, history):
                    if change == "answer":
                        fixture.item["qa"]["answer_points"] = [{"text": "EU rejects blanks."}]
                    elif change == "source":
                        path = draft / "history-contract.txt"
                        path.write_text(path.read_text().replace("sources: old", "sources: correction"))
                    else:
                        name = {"task": "task.md", "contract": "history-contract.txt",
                                "requirements": "acceptance.md"}[change]
                        path = draft / name
                        text = path.read_text()
                        path.write_text(text.replace("Pending data remains readable", "Pending data is deleted")
                                        if change == "requirements" else text + "\nChanged after qualification.\n")
                fixture, receipt, calls = self.construct_test_retries(before_retry=before_retry)
                self.assertIsNone(receipt)
                self.assertFalse((fixture.root / "frozen").exists())
                calls.review.assert_called_once()
                calls.sources.assert_called_once()
                calls.coverage.assert_called_once()
                self.assertEqual([call.args[0].name for call in calls.agent.call_args_list],
                                 ["author", "reference-solver"])
                records = read(fixture.root / "construction.json")
                self.assertEqual(len(records), 2)
                self.assertFalse(records[-1]["accepted"])

    def test_later_test_author_cannot_change_qualified_task_or_requirements(self):
        for name in ("task.md", "history-contract.txt", "acceptance.md"):
            with self.subTest(name=name):
                def change(spec):
                    path = spec / name
                    text = path.read_text()
                    path.write_text(text.replace("Pending data remains readable", "Pending data is deleted")
                                    if name == "acceptance.md" else text + "\nChanged during test repair.\n")
                fixture, receipt, calls = self.construct_test_retries(author_change=change, revisions=1)
                self.assertIsNone(receipt)
                self.assertFalse((fixture.root / "frozen").exists())
                calls.review.assert_called_once()
                calls.sources.assert_called_once()
                calls.coverage.assert_called_once()
                self.assertEqual([call.args[0].name for call in calls.agent.call_args_list],
                                 ["author", "reference-solver", "author"])
                records = read(fixture.root / "construction.json")
                self.assertEqual(records[-1]["reason"],
                                 "invalid_acceptance" if name == "acceptance.md" else "qualified_draft_changed")
                self.assertFalse(records[-1]["accepted"])

    def test_oracle_and_replayed_mutation_are_both_admission_gates(self):
        for coverage, checks_coverage, mutation, support, accepted in (
                ("complete", "complete", "caught", "supported", True),
                ("missing", "complete", "caught", "supported", False),
                ("fallback", "complete", "caught", "supported", False),
                ("uncertain", "complete", "caught", "supported", False),
                ("changed_answer", "complete", "caught", "supported", False),
                ("changed_task", "complete", "caught", "supported", False),
                ("changed_contract", "complete", "caught", "supported", False),
                ("complete", "complete", "unverified", "supported", False),
                ("complete", "unsupported", "caught", "supported", False),
                ("complete", "complete", "caught", "unsupported", False),
                ("complete", "complete", "caught", "uncertain", False)):
            with self.subTest(coverage=coverage, checks_coverage=checks_coverage,
                              mutation=mutation, support=support):
                fixture = test_task_eval_flow.TaskPreflightTests()
                fixture.setUp()
                self.addCleanup(fixture.doCleanups)
                save(Path(fixture.item["generation_input"]), {"ref_to_source": {"资料1": "e1"}})
                fixture.item["public_records"] = [
                    {"id": "e1", "original_id": "old", "order": 1, "kind": "message", "text": "Preserve all blanks"},
                    {"id": "e2", "original_id": "correction", "order": 2, "kind": "message", "text": "EU rejects blanks"}]
                fixture.item["qa"]["answer_points"] = [{"text": "EU rejects blanks; other tenants preserve blanks."}]
                if coverage == "missing":
                    fixture.item["qa"]["answer_points"] = [{"text": "other tenants preserve blanks."}]
                public_history = prepare_history(fixture.item["public_records"], fixture.item["generation_input"])
                targets = freeze_targets(parse_text_response(CONTRACT)["reviews"], public_history)
                original = fixture.fake_agent
                def draft(selection, config, output, spec, budget, feedback=""):
                    fixture.fake_draft(selection, config, output, spec, budget, feedback)
                    (spec / "history-contract.txt").write_text(CONTRACT)
                    with (spec / "acceptance.md").open("a") as handle:
                        handle.write("\n| a2 | Other blanks retained | h1 | test: test_acceptance::test_other |\n"
                                     "| a3 | EU blanks rejected | h2 | test: test_acceptance::test_eu |\n")
                def agent(root, *args, **kwargs):
                    if root.name == "author":
                        (root / "workspace/checks/test_acceptance.py").write_text("Original test")
                        if coverage == "changed_answer":
                            fixture.item["qa"]["answer_points"] = [{"text": "EU rejects blanks."}]
                        elif coverage in {"changed_task", "changed_contract"}:
                            name = "task.md" if coverage == "changed_task" else "history-contract.txt"
                            path = root / "workspace/checks" / name
                            path.write_text(path.read_text() + "Changed after qualification.\n")
                        return {"status": "finished"}
                    result = original(root, *args, **kwargs)
                    checks = root / "workspace/checks"
                    if root.name == "author":
                        (checks / "history-contract.txt").write_text(CONTRACT)
                        with (checks / "acceptance.md").open("a") as handle:
                            handle.write("\n| a2 | Other blanks retained | h1 | test: test_acceptance::test_other |\n"
                                         "| a3 | EU blanks rejected | h2 | test: test_acceptance::test_eu |\n")
                    elif root.name == "validator":
                        reference = kwargs["reference"] / "spec"
                        self.assertFalse((reference / "history.json").exists())
                        self.assertFalse((reference / "history-review.md").exists())
                        self.assertTrue((reference / "history-contract.txt").is_file())
                    return result
                def task_review(task, answer, config, output, *, evidence, budget):
                    if coverage == "fallback":
                        return {"status": "clean", "issue": "none"}
                    result = qualification({"oracle_answer": answer, "contracts": evidence["contracts"]})
                    for row, quote in zip(result["history_rows"],
                                          ("other tenants preserve blanks", "EU rejects blanks")):
                        row["answer_quote"] = quote
                    if coverage == "uncertain":
                        result["status"] = "uncertain"
                    return result
                def checks(candidate, *args, **kwargs):
                    status = "failed" if candidate == fixture.baseline else "passed"
                    return {"status": status, "cases": [{"id": "test_acceptance::test_" + name, "status": status}
                                                         for name in ("feature", "other", "eu")]}
                def coverage_review(spec, baseline, candidate, changed, results, config, output, budget):
                    output.mkdir(parents=True)
                    (output / "coverage.md").write_text("Each acceptance row is covered.")
                    return {"status": "complete" if checks_coverage == "complete" else "revise", "rows": []}
                def mutation_files(spec, candidate, changed, config, output, budget):
                    (output / "workspace").mkdir(parents=True)
                    return agent(output, config, "judge", "Generate mutation files",
                                 reference=output.parent / "validator-reference")
                with patch("dialogue_benchmark.task_eval.run.run_agent", side_effect=agent), \
                     patch("dialogue_benchmark.task_eval.run.write_draft", side_effect=draft), \
                     patch("dialogue_benchmark.task_eval.run.select_task", return_value={
                         "status": "candidate", "history_targets": targets}), \
                     patch("dialogue_benchmark.task_eval.run.review_task", side_effect=task_review), \
                     patch("dialogue_benchmark.task_eval.run.review_sources", return_value={"support": support}) as sources, \
                     patch("dialogue_benchmark.task_eval.run.review_checks", side_effect=coverage_review), \
                     patch("dialogue_benchmark.task_eval.run.write_history_mutation", side_effect=mutation_files), \
                     patch("dialogue_benchmark.task_eval.run.run_checks", side_effect=checks), \
                     patch("dialogue_benchmark.task_eval.run.check_history_mutations", return_value={"status": mutation}) as replay:
                    from dialogue_benchmark.task_eval.run import construct
                    receipt = construct(fixture.item, fixture.root, fixture.baseline,
                                        {"execution_image": "image"}, 0, {})
                self.assertEqual(receipt is not None, accepted)
                self.assertEqual(sources.call_count, int(coverage == "complete"))
                self.assertEqual(replay.call_count, int(coverage == "complete" and checks_coverage == "complete"
                                                       and support == "supported"))
                record = read(fixture.root / "construction.json")[-1]
                if coverage == "changed_answer":
                    self.assertEqual(record["reason"], "qualified_history_not_verified")
                if coverage in {"changed_task", "changed_contract"}:
                    self.assertEqual(record["reason"], "qualified_draft_changed")
                if support != "supported":
                    self.assertEqual(record["reason"], "history_source_not_verified")
                if accepted:
                    self.assertTrue(record["oracle_complete"])
                    self.assertFalse((fixture.root / "frozen/checkpoints.json").exists())
                    self.assertFalse((fixture.root / "frozen/test_interactions.py").exists())
                    self.assertEqual(receipt["oracle_sufficiency"], "validated_against_external_rules")


class CheckReviewTests(unittest.TestCase):
    def test_initial_test_writer_uses_snapshot_and_freezes_real_regressions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, baseline = root / "spec", root / "baseline"
            spec.mkdir()
            (baseline / "tests").mkdir(parents=True)
            (baseline / "entry.py").write_text("VALUE = 1\n")
            original = "def test_existing(): assert True\n"
            (baseline / "tests/test_existing.py").write_text(original)
            business_inputs = {"orders.json": '[{"id": "one"}]\n',
                               "replies.csv": 'id,status\none,Delivered, signed\n'}
            for name, content in business_inputs.items():
                (baseline / name).write_text(content)
            (baseline / ".private.json").write_text("Not part of the snapshot input")
            (spec / "task.md").write_text("Add the new entry")
            (spec / "acceptance.md").write_text("Initial checks")
            calls = []
            def call(prompt, payload, config, output):
                calls.append(payload)
                return {"files": [
                    {"name": "test_acceptance.py", "content": "def test_feature(): assert True\n"},
                    {"name": "acceptance.md", "content": "Executable checks\n"}]}
            result = write_tests(spec, baseline, {}, root / "author", SimpleNamespace(call=call))
            self.assertEqual(result["status"], "finished")
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["repository"]["entry.py"], "VALUE = 1\n")
            for name, content in business_inputs.items():
                self.assertEqual(calls[0]["repository"][name], content)
            self.assertNotIn(".private.json", calls[0]["repository"])
            self.assertEqual((spec / "regression/tests/test_existing.py").read_text(), original)
            self.assertEqual((baseline / "tests/test_existing.py").read_text(), original)
            self.assertIn("/workspace/candidate/tests", (spec / "commands/existing_suite.sh").read_text())
            self.assertEqual((spec / "task.md").read_text(), "Add the new entry")
            (baseline / "large.py").write_text("# context\n" * 10000)
            self.assertIsNone(write_tests(spec, baseline, {}, root / "large", SimpleNamespace(call=call)))
            self.assertEqual(len(calls), 1)

    def test_initial_test_writer_preserves_a_contradiction_without_making_tests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, baseline = root / "spec", root / "baseline"
            spec.mkdir()
            (baseline / "tests").mkdir(parents=True)
            (baseline / "entry.py").write_text("pass\n")
            (spec / "task.md").write_text("Conflicting requirements")
            (spec / "acceptance.md").write_text("Original criteria")
            result = write_tests(spec, baseline, {}, root / "author", SimpleNamespace(call=lambda *a: {
                "files": [{"name": "NO_TASK.md", "content": "The required argument contradicts the task."}]}))
            self.assertEqual(result["status"], "finished")
            self.assertTrue((spec / "NO_TASK.md").exists())
            self.assertFalse((spec / "test_acceptance.py").exists())
            self.assertEqual((spec / "acceptance.md").read_text(), "Original criteria")

    def test_finite_repair_cannot_edit_task_or_frozen_regressions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = root / "spec"
            spec.mkdir()
            (spec / "task.md").write_text("Preserve order")
            (spec / "acceptance.md").write_text("Original references")
            (spec / "test_feature.py").write_text("def test_feature(): pass")
            files = [{"name": "acceptance.md", "content": "Updated references"},
                     {"name": "test_feature.py", "content": "def test_feature(): assert True"}]
            baseline = root / "baseline"
            baseline.mkdir()
            (baseline / "job.json").write_text('{"responses": [{"note": "Keep this note"}]}')
            calls = []
            def call(prompt, payload, *args):
                calls.append(payload)
                return {"files": files}
            result = repair_tests(spec, baseline, {}, root / "call", SimpleNamespace(call=call), "Fix an assertion")
            self.assertEqual(result["status"], "finished")
            self.assertIn('"note"', calls[0]["repository"]["job.json"])
            self.assertEqual((spec / "task.md").read_text(), "Preserve order")
            files.append({"name": "task.md", "content": "Weaker requirement"})
            result = repair_tests(spec, baseline, {}, root / "invalid", SimpleNamespace(call=call), "Fix an assertion")
            self.assertEqual(result["status"], "error")
            self.assertEqual((spec / "task.md").read_text(), "Preserve order")

    def test_model_source_mutation_exports_replayable_patch_without_changing_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, candidate = root / "spec", root / "candidate"
            spec.mkdir()
            candidate.mkdir()
            (candidate / "entry.py").write_text("limit = 384\n")
            for name in ("task.md", "history-contract.txt", "memory-use.md"):
                (spec / name).write_text("Apply the known receiver limit")
            save(spec / "acceptance.json", [{"id": "a2", "basis": ["h1"]}])
            files = [{"name": "mutations.txt", "content": "REVIEW m1\nacceptance: a2\nfile: entry.py\nEND_REVIEW"},
                     {"name": "before.txt", "content": "limit = 384"},
                     {"name": "after.txt", "content": "limit = 500"}]
            call = Mock(return_value={"files": files})
            result = write_history_mutation(spec, candidate, ["entry.py"], {}, root / "review",
                SimpleNamespace(call=call))
            self.assertEqual(call.call_args.args[1]["memory_use"], "Apply the known receiver limit")
            self.assertEqual(result["status"], "finished")
            self.assertEqual(result["method"], "model_file_generation")
            self.assertEqual((candidate / "entry.py").read_text(), "limit = 384\n")
            self.assertTrue(read(root / "review/version/version.json")["replay_verified"])
            self.assertIn("+limit = 500", (root / "review/workspace/checks/m1.patch").read_text())
            files[2]["content"] = files[1]["content"]
            result = write_history_mutation(spec, candidate, ["entry.py"], {}, root / "unchanged",
                SimpleNamespace(call=lambda *args: {"files": files}))
            self.assertEqual(result["status"], "error")
            files[2]["content"] = "limit = 500"
            files[0]["content"] = files[0]["content"].replace("entry.py", "../outside.py")
            result = write_history_mutation(spec, candidate, ["entry.py"], {}, root / "invalid",
                SimpleNamespace(call=lambda *args: {"files": files}))
            self.assertEqual(result["status"], "error")
            self.assertFalse((root / "outside.py").exists())

    def test_mutation_file_protocol_can_replace_one_inline_json_fragment(self):
        response = parse_text_response(
            'FILE mutations.txt\nREVIEW m1\nacceptance: a2\nfile: rules.json\nEND_REVIEW\nEND_FILE\n'
            'FILE before.txt\n"not delivered": "FAILED"\nEND_FILE\n'
            'FILE after.txt\n"not delivered": "DELIVERED"\nEND_FILE\n')
        self.assertTrue(response["files"][1]["content"].endswith("\n"))
        original = '{\n  "not delivered": "FAILED",\n  "ok": "DELIVERED"\n}\n'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, candidate = root / "spec", root / "candidate"
            spec.mkdir()
            candidate.mkdir()
            for name in ("task.md", "history-contract.txt", "memory-use.md"):
                (spec / name).write_text("Apply the customer status mapping")
            save(spec / "acceptance.json", [{"id": "a2", "basis": ["h1"]}])
            path = candidate / "rules.json"
            path.write_text(original)
            budget = SimpleNamespace(call=lambda *args: response)
            result = write_history_mutation(spec, candidate, ["rules.json"], {}, root / "review", budget)
            self.assertEqual(result["status"], "finished", result)
            self.assertTrue(read(root / "review/version/version.json")["replay_verified"])
            patch_text = (root / "review/workspace/checks/m1.patch").read_text()
            self.assertIn('+  "not delivered": "DELIVERED",\n', patch_text)
            self.assertEqual(path.read_text(), original)

            path.write_text('{"one": {"not delivered": "FAILED"}, '
                            '"two": {"not delivered": "FAILED"}}\n')
            result = write_history_mutation(spec, candidate, ["rules.json"], {}, root / "ambiguous", budget)
            self.assertEqual(result["status"], "error")
            self.assertFalse((root / "ambiguous/workspace/checks/m1.patch").exists())

    def test_review_uses_saved_tests_and_changed_sources_and_requires_all_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, baseline, candidate = [root / name for name in ("spec", "base", "candidate")]
            for path in (spec, baseline, candidate):
                path.mkdir()
            save(spec / "acceptance.json", [{"id": "a1"}])
            save(spec / "history.json", {"events": ["raw history"]})
            (spec / "history-review.md").write_text("Raw history")
            (spec / "task.md").write_text("Export selected records")
            (spec / "test_acceptance.py").write_text("Tests")
            (baseline / "entry.py").write_text("Old implementation")
            (baseline / "replies.csv").write_text("id,status\none,Delivered, signed\n")
            (baseline / ".private.json").write_text("Excluded")
            (candidate / "entry.py").write_text("New implementation")
            (candidate / "unrelated.py").write_text("Unrelated")
            for decision, expected in (("complete", "complete"), ("unsupported", "revise")):
                rows = [dict(id=identity, coverage=decision, evidence="test_acceptance.py")
                        for identity in ("a1", "tests")]
                with patch("dialogue_benchmark.task_eval.selection.ask_model", return_value={"reviews": rows}) as call:
                    result = review_checks(spec, baseline, candidate, ["entry.py"], {"reference": {
                        "status": "passed", "cases": [{"id": "test::case", "status": "passed", "detail": "Large log"}]}},
                                           {}, root / "review", SimpleNamespace(call=call))
                self.assertEqual(result["status"], expected)
                payload = call.call_args.args[1]
                self.assertNotIn("history.json", payload["criteria_and_tests"])
                self.assertNotIn("history-review.md", payload["criteria_and_tests"])
                self.assertEqual(set(payload["changed_sources"]), {"entry.py"})
                self.assertEqual(payload["business_inputs"], {"replies.csv": "id,status\none,Delivered, signed\n"})
                self.assertIn("-Old implementation\n", payload["changed_sources"]["entry.py"])
                self.assertIn("+New implementation\n", payload["changed_sources"]["entry.py"])
                self.assertEqual(payload["executed_checks"], {"reference": {"status": "passed",
                                  "cases": [{"id": "test::case", "status": "passed"}]}})
            result = review_checks(spec, baseline, candidate, [], {}, {}, root / "missing",
                SimpleNamespace(call=lambda *args: {"reviews": [rows[0]]}))
            self.assertEqual(result["status"], "uncertain")
