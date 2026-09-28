"""Controlled selection returns to the host before any autonomous execution."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dialogue_benchmark.task_eval.artifacts import read, save
from dialogue_benchmark.task_eval.selection import (SelectionBudget, extract_history_targets,
                                                     query_evidence, repository_overview,
                                                     select_task, write_draft, write_private_draft)
from dialogue_benchmark.task_eval.run import construct
from dialogue_benchmark.task_eval.history import oracle_coverage


class SelectionTests(unittest.TestCase):
    def test_repository_overview_is_static_and_compact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text("# Demo\nA parameter tool.\n", encoding="utf-8")
            (root / "src").mkdir()
            (root / "tests").mkdir()
            (root / "src" / "main.py").write_text("secret implementation\n", encoding="utf-8")
            overview = repository_overview(root)
        self.assertIn("A parameter tool.", overview["purpose"])
        self.assertIn("src/", overview["relevant_paths"])
        self.assertIn("tests/", overview["test_areas"])
        self.assertNotIn("secret implementation", str(overview))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        (self.repo / "api.py").write_text("\n".join("line%d" % i for i in range(100)))
        self.history = {"source_aliases": {"source1": "event1", "source2": "event2"},
                        "initial_events": [{"id": "event1", "source": "source1", "text": "Maple rule"}],
                        "events": [{"id": "event1", "order": 1, "text": "Maple rule"},
                                   {"id": "event2", "order": 2, "text": "x" * 7000}]}

    def run_selection(self, decisions, options=None):
        def ask(prompt, payload, config, output):
            save(output / "usage.json", [{"prompt_tokens": 10, "completion_tokens": 2}])
            return {"reviews": [next(iterator)]}
        iterator = iter(decisions)
        budget = SelectionBudget(self.root, options or {})
        with patch("dialogue_benchmark.task_eval.selection.ask_model", side_effect=ask):
            result = select_task({}, self.history, self.repo, {}, self.root / "selection", budget)
        return result, budget

    def query(self, **kwargs):
        if kwargs["op"] == "read":
            value = kwargs.get("path") if kwargs["target"] == "repo" else kwargs.get("source")
            path, text = (value, "-") if kwargs["target"] == "repo" else ("-", value)
        else:
            path = kwargs.get("path", "-") if kwargs["target"] == "repo" else "-"
            text = kwargs.get("text", "-")
        query = "|".join(str(value) for value in (
            kwargs["op"], kwargs["target"], path, text, kwargs.get("offset", 0)))
        return dict(decision="need_evidence", reason="Check rule availability", sources="qa",
                    request=query)

    def test_query_then_candidate_and_cost(self):
        result, budget = self.run_selection([
            self.query(op="read", target="repo", path="api.py"),
            dict(decision="candidate", reason="New feature uses historical rule", sources="qa,query1")])
        self.assertEqual(result["status"], "candidate")
        self.assertEqual(result["query_count"], 1)
        self.assertEqual(budget.requests, 2)
        self.assertEqual(budget.remaining()["max_tokens"], 1500000 - 24)

    def test_qa_workflow_is_direction_not_a_new_source_of_history(self):
        direction = "增加批量交付恢复：确认状态 → 恢复交付 → 汇总结果"
        payloads = []
        def ask(prompt, payload, config, output):
            payloads.append(payload.copy())
            save(output / "usage.json", [{"prompt_tokens": 10, "completion_tokens": 2}])
            return {"reviews": [dict(decision="pending", reason="Need repository evidence",
                                      sources="qa", request="none")]}
        with patch("dialogue_benchmark.task_eval.selection.ask_model", side_effect=ask):
            result = select_task({}, self.history, self.repo, {}, self.root / "selection",
                                 SelectionBudget(self.root, {}), workflow=direction)
        self.assertEqual(payloads[0]["development_workflow"], direction)
        self.assertEqual(result["evidence"]["development_workflow"], direction)
        self.assertEqual(result["evidence"]["history_sources"],
                         [{"source": "source1", "role": None, "answer_source": False}])

    def test_duplicate_default_offset_stops_pending(self):
        result, _ = self.run_selection([
            self.query(op="read", target="repo", path="api.py"),
            self.query(op="read", target="repo", path="api.py", offset=0)])
        self.assertEqual(result["status"], "pending")
        self.assertIn("duplicate", result["reason"])
        self.assertEqual(result["query_count"], 1)

    def test_pages_are_distinct_and_empty_lookup_is_not_a_verdict(self):
        result, _ = self.run_selection([
            self.query(op="lookup", target="repo", path=".", text="absent"),
            self.query(op="read", target="repo", path="api.py"),
            self.query(op="read", target="repo", path="api.py", offset=80),
            dict(decision="pending", reason="Still uncertain", sources="query1")])
        self.assertEqual(result["status"], "pending")
        self.assertEqual(result["query_count"], 3)
        self.assertEqual(result["evidence"]["queries"][1]["result"]["next_offset"], 80)

    def test_invalid_queries_and_unknown_citations(self):
        decisions = [self.query(op="shell", target="repo", path=".", text="pytest"),
                     self.query(op="read", target="repo", path="../secret"),
                     self.query(op="read", target="history", source="all"),
                     dict(decision="stop", reason="Claim", sources="invented"),
                     dict(decision="stop", reason="Unread source", sources="source1")]
        for decision in decisions:
            with self.subTest(decision=decision):
                result, _ = self.run_selection([decision])
                self.assertEqual(result["status"], "pending")
                self.assertEqual(result["query_count"], 0)

    def test_selection_reads_multitopic_history_only_when_requested(self):
        self.history["initial_events"][0]["text"] += " Unrelated rendering correction"
        self.history["qa_source_ids"] = ["event1"]
        calls = []
        responses = [self.query(op="read", target="history", source="source1"),
                     dict(decision="pending", reason="Need repository evidence", sources="source1")]
        budget = SelectionBudget(self.root, {})
        def call(prompt, payload, config, output):
            calls.append(dict(payload, queries=list(payload["queries"])))
            return {"reviews": [responses.pop(0)]}
        with patch.object(budget, "call", side_effect=call):
            result = select_task({"question": "Maple rule?"}, self.history,
                                 self.repo, {}, self.root / "on-demand", budget)
        self.assertNotIn("Unrelated rendering correction", str(calls[0]))
        self.assertEqual(calls[0]["history_sources"][0]["source"], "source1")
        self.assertTrue(calls[0]["history_sources"][0]["answer_source"])
        self.assertEqual(calls[1]["queries"][0]["result"]["text"], "Maple rule")
        self.assertEqual(result["reason"], "Need repository evidence")

    def test_history_query_is_not_repository_evidence(self):
        result, _ = self.run_selection([
            self.query(op="read", target="history", source="source2"),
            dict(decision="candidate", reason="Claim", sources="query1")])
        self.assertEqual(result["status"], "pending")

    def test_budget_stops_before_next_request(self):
        result, budget = self.run_selection([self.query(op="read", target="repo", path="api.py")],
                                             {"max_requests": 1})
        self.assertEqual(result["status"], "pending")
        self.assertEqual(budget.requests, 1)

    def test_missing_usage_and_request_errors_preserve_pending(self):
        for error in (TimeoutError("timeout"), None):
            with self.subTest(error=error):
                budget = SelectionBudget(self.root, {})
                with patch("dialogue_benchmark.task_eval.selection.ask_model", side_effect=error,
                           return_value={"reviews": [{"decision": "stop", "reason": "Claim", "sources": "source1"}]}):
                    result = select_task({}, self.history, self.repo, {}, self.root / "failure", budget)
                self.assertEqual(result["status"], "pending")
                self.assertFalse(budget.usage_complete)

    def test_pending_does_not_start_openhands_or_review(self):
        source = self.root / "input.json"
        save(source, {})
        item = {"qa": {"type": "constraint_followthrough", "id": "q1"}, "generation_input": str(source)}
        with patch("dialogue_benchmark.task_eval.run.select_task", return_value={"status": "pending", "reason": "Need fact"}), \
             patch("dialogue_benchmark.task_eval.run.run_agent") as agent, \
             patch("dialogue_benchmark.task_eval.run.review_task") as review:
            self.assertIsNone(construct(item, self.root / "task", self.repo, {}, 0, {}))
        agent.assert_not_called()
        review.assert_not_called()

    def test_exact_history_pagination(self):
        page = query_evidence(dict(op="read", target="history", source="source2"), self.repo, self.history)
        self.assertEqual(page["source"], "event2")
        self.assertEqual(page["next_offset"], 6000)

    def test_public_rule_still_checked_but_answer_only_covers_gap(self):
        history = {"public_task": "Preserve order.", "oracle_answer": "Maple omits null note.",
                   "contracts": [{"id": name, "active": True, "repository": "external"}
                                 for name in ("h1", "h2")]}
        path = self.root / "oracle.txt"
        path.write_text("REVIEW h1\ncoverage: provided\nquote: Preserve order.\nEND_REVIEW\n"
                        "REVIEW h2\ncoverage: complete\nquote: Maple omits null note.\nEND_REVIEW")
        self.assertTrue(oracle_coverage(path, history))
        history["oracle_answer"] = "Maple exports."
        self.assertFalse(oracle_coverage(path, history))
        path.write_text("REVIEW h1\ncoverage: provided\nquote: Preserve order.\nEND_REVIEW\n"
                        "REVIEW h2\ncoverage: provided\nquote: Preserve order.\nEND_REVIEW")
        self.assertFalse(oracle_coverage(path, history))

    def test_tool_free_draft_parses_all_files_before_writing(self):
        selection = {"decision": {}, "evidence": {}}
        budget = SelectionBudget(self.root, {})
        files = [dict(name=name, content="contents\n") for name in
                 ("task.md", "memory-use.md", "history-contract.txt", "acceptance.md")]
        with patch.object(budget, "call", return_value={"files": files}):
            write_draft(selection, {}, self.root / "request", self.root / "spec", budget)
        self.assertEqual((self.root / "spec/task.md").read_text(), "contents\n")
        with patch.object(budget, "call", return_value={"files": files[:-1]}):
            with self.assertRaises(ValueError):
                write_draft(selection, {}, self.root / "request2", self.root / "bad-spec", budget)
        self.assertFalse((self.root / "bad-spec").exists())

    def test_split_draft_keeps_public_request_free_of_history_answer(self):
        selection = {
            "public": {"public_goal": "Add Maple batch export",
                       "agreement_object": "Maple must omit null note",
                       "agreement_scope": "new exports must omit null note"},
            "public_repository_evidence": [{"id": "query1", "result": {"matches": []}}],
            "repository_exploration": "Suggested future interface: require caller_rule_map.",
            "history_targets": {"targets": [{"id": "h1", "statement": "omit null note",
                                               "scope": "Maple", "behavior": "omit", "sources": ["event1"],
                                               "supersedes": []}]},
            "public_history": {"events": [{"id": "event1", "text": "User confirmed omit null note"}]},
            "historical_answer": "Omit note=null; preserve other null fields.",
        }
        calls = []
        budget = SelectionBudget(self.root, {})
        def call(prompt, payload, config, output):
            calls.append((prompt, payload))
            if len(calls) == 1:
                return {"task": "Add batch export.\n"}
            return {"use": "h1 changes null handling.", "acceptance": [
                {"id": "a1", "basis": "task", "requirement": "Export", "check": "inspect: export"},
                {"id": "a2", "basis": "h1", "requirement": "Omit null note", "check": "inspect: omit"},
            ]}
        with patch.object(budget, "call", side_effect=call):
            write_draft(selection, {}, self.root / "draft", self.root / "spec", budget)
        self.assertEqual(len(calls), 2)
        public_payload = calls[0][1]
        self.assertNotIn("historical_answer", public_payload)
        self.assertNotIn("history_targets", public_payload)
        self.assertNotIn("omit null note", str(public_payload))
        self.assertEqual(public_payload["public_goal"], "Add Maple batch export")
        self.assertNotIn("agreement_object", public_payload)
        self.assertNotIn("agreement_scope", public_payload)
        self.assertNotIn("repository_exploration", public_payload)
        self.assertNotIn("caller_rule_map", str(public_payload))
        self.assertEqual(public_payload["repository_evidence"], selection["public_repository_evidence"])
        private_payload = calls[1][1]
        self.assertIn("Omit note=null", private_payload["historical_answer"])
        self.assertIn("omit null note", str(private_payload["history_targets"]))

    def test_candidate_with_query_is_executed_before_confirmation(self):
        result, budget = self.run_selection([
            self.query(op="read", target="repo", path="api.py") | {
                "decision": "candidate", "reason": "Need one repository fact",
                "sources": "qa"},
            dict(decision="candidate", reason="Confirmed", sources="qa,query1")])
        self.assertEqual(result["status"], "candidate")
        self.assertEqual(result["query_count"], 1)
        self.assertEqual(budget.requests, 2)

    def test_private_acceptance_uses_reviewed_applicability_and_retains_history(self):
        from dialogue_benchmark.llm import parse_text_response
        from dialogue_benchmark.task_eval.checks import acceptance_items
        spec = self.root / "spec"
        spec.mkdir()
        task = "Add the customer classifier, without changing the generic API.\n"
        (spec / "task.md").write_text(task)
        targets = [{"id": "h1", "statement": "Customer mapping", "scope": "Customer records",
                    "behavior": "Classify integer states", "sources": ["event1"], "supersedes": []},
                   {"id": "h2", "statement": "Temporary caller conversion", "scope": "Earlier run",
                    "behavior": "Convert locally", "sources": ["event2"], "supersedes": []}]
        selection = {"history_targets": {"targets": targets}, "public_history": self.history,
                     "historical_answer": "Customer mapping"}
        review = {"history_rows": [{"id": "h1", "applicable": "yes", "public": "none"},
                                   {"id": "h2", "applicable": "no", "public": "none"}]}
        response = {"use": "Use the customer mapping.", "acceptance": [
            {"id": "a1", "basis": "task", "requirement": "New API", "check": "inspect: call API"},
            {"id": "a2", "basis": "h1", "requirement": "Mapping", "check": "inspect: classify"}]}
        budget = SelectionBudget(self.root, {})
        with patch.object(budget, "call", return_value=response) as call:
            write_private_draft(selection, {}, self.root / "reviewed-private", spec, budget,
                                history_review=review)
        payload = call.call_args.args[1]
        self.assertEqual([t["id"] for t in payload["history_targets"]], ["h1"])
        self.assertEqual([e["id"] for e in payload["history_sources"]], ["event1"])
        contracts = parse_text_response((spec / "history-contract.txt").read_text())["reviews"]
        self.assertEqual([(r["id"], r["active"]) for r in contracts], [("h1", "yes"), ("h2", "no")])
        self.assertEqual((spec / "task.md").read_text(), task)
        self.assertEqual([r["basis"] for r in acceptance_items(spec, {"contracts": [
            {"id": r["id"], "active": r["active"] == "yes"} for r in contracts]})], [["task"], ["h1"]])

    def test_history_index_can_identify_a_requested_read(self):
        request = self.query(op="read", target="history", source="source1")
        request["sources"] = "qa,source1"
        result, _ = self.run_selection([
            request, dict(decision="pending", reason="Need repository evidence", sources="source1")])
        self.assertEqual(result["query_count"], 1)
        self.assertEqual(result["reason"], "Need repository evidence")

    def test_history_target_extraction_omits_answer_points(self):
        calls = []
        budget = SelectionBudget(self.root, {})
        def call(prompt, payload, config, output):
            calls.append(payload)
            return {"reviews": [{"id": "h1", "statement": "Maple omits null note",
                                  "scope": "Maple exports", "behavior": "omit",
                                  "sources": "source1", "supersedes": "none"}]}
        with patch.object(budget, "call", side_effect=call):
            result = extract_history_targets(
                {"question": "What does Maple export preserve?", "type": "constraint_followthrough",
                 "answer_points": [{"text": "secret answer point"}]}, self.history,
                {"public_goal": "Add batch export", "agreement_object": "Maple",
                 "agreement_scope": "future exports"}, {}, self.root / "targets", budget)
        self.assertEqual(result["status"], "candidate")
        self.assertNotIn("answer_points", calls[0])
        self.assertNotIn("secret answer point", str(calls[0]))

    def test_history_target_extraction_starts_from_qa_sources(self):
        calls = []
        budget = SelectionBudget(self.root, {})
        history = {
            "qa_source_ids": ["event1"],
            "initial_events": [
                {"id": "event1", "source": "source1", "role": "user", "text": "Maple rule"},
                {"id": "event2", "source": "source2", "role": "user", "text": "Unrelated follow-up"},
            ],
            "source_aliases": {"source1": "event1", "source2": "event2"},
            "events": [{"id": "event1", "order": 1, "text": "Maple rule"},
                       {"id": "event2", "order": 2, "text": "Unrelated follow-up"}],
        }
        def call(prompt, payload, config, output):
            calls.append(payload)
            return {"reviews": [{"id": "h1", "statement": "Maple rule",
                                  "scope": "Maple", "behavior": "keep",
                                  "sources": "source1", "supersedes": "none"}]}
        with patch.object(budget, "call", side_effect=call):
            result = extract_history_targets(
                {"question": "What rule applies?", "type": "constraint_followthrough",
                 "answer_points": []}, history, {"public_goal": "Add export",
                 "agreement_object": "Maple", "agreement_scope": "future exports"},
                {}, self.root / "targets-filtered", budget)
        self.assertEqual(result["status"], "candidate")
        self.assertIn("Maple rule", str(calls[0]["history"]))
        self.assertNotIn("Unrelated follow-up", str(calls[0]["history"]))

    def test_history_targets_keep_applicable_cycle_limited_authorization(self):
        statement = "Maple approved only P1 orders for the May delivery cycle."
        self.history["initial_events"][0]["text"] = statement
        self.history["events"][0]["text"] = statement
        row = {"id": "h1", "statement": statement, "scope": "Maple May delivery cycle",
               "behavior": "Select P1 orders", "sources": "source1", "supersedes": "none"}
        budget = SelectionBudget(self.root, {})
        with patch.object(budget, "call", return_value={"reviews": [row]}) as call:
            result = extract_history_targets(
                {"question": "Which orders are approved for Maple's May delivery cycle?", "type": "M6"},
                self.history, {"public_goal": "Add delivery reconciliation"},
                {}, self.root / "cycle-targets", budget)
        prompt = call.call_args.args[0]
        self.assertIn("普通的一次执行命令不要作为约定", prompt)
        self.assertIn("仍适用于本任务的、已确认的周期限定授权或状态必须保留", prompt)
        self.assertIn("不得扩展其周期或范围", prompt)
        self.assertEqual(result["status"], "candidate")
        self.assertEqual(result["targets"][0]["statement"], statement)
        self.assertEqual(result["targets"][0]["scope"], row["scope"])

    def test_history_target_extraction_preserves_saved_public_corrections(self):
        from dialogue_benchmark.task_eval.history import prepare_history
        records = [
            {"id": "e1", "original_id": "old", "order": 1, "kind": "message",
             "role": "user", "text": "Preserve blank notes for all tenants."},
            {"id": "e2", "original_id": "correction", "order": 2, "kind": "message",
             "role": "user", "text": "For EU tenants only, reject blank notes."},
            {"id": "e3", "original_id": "unrelated", "order": 3, "kind": "message",
             "role": "user", "text": "Use a blue page background."}]
        save(self.root / "input.json", {"payload": {"materials": [
            {"reference": "资料1"}, {"reference": "资料2"}]},
            "ref_to_source": {"资料1": "e1", "资料2": "e2"}})
        history = prepare_history(records, self.root / "input.json", qa_source_ids={"e1"})
        rows = [
            {"id": "h1", "statement": records[0]["text"], "scope": "non-EU tenants",
             "behavior": "preserve blanks", "sources": "source1", "supersedes": "none"},
            {"id": "h2", "statement": records[1]["text"], "scope": "EU tenants",
             "behavior": "reject blanks", "sources": "source2", "supersedes": "h1"}]
        budget = SelectionBudget(self.root, {})
        with patch.object(budget, "call", return_value={"reviews": rows}) as call:
            result = extract_history_targets(
                {"question": "Which blank-note rules apply to batch exports?", "type": "M6"},
                history, {"public_goal": "Add batch export"}, {}, self.root / "closed-targets", budget)
        self.assertEqual(result["status"], "candidate")
        self.assertEqual([row["source"] for row in call.call_args.args[1]["history"]],
                         ["source1", "source2"])
        self.assertEqual(result["targets"][1]["supersedes"], ["h1"])
        self.assertNotIn("blue page", str(call.call_args.args[1]))

    def test_file_protocol_preserves_nested_history_and_layout(self):
        from dialogue_benchmark.llm import parse_text_response
        self.assertEqual(parse_text_response("NO_TARGETS"), {"reviews": []})
        self.assertEqual(parse_text_response("TASK\nAdd export\nEND_TASK"),
                         {"task": "Add export\n"})
        self.assertEqual(parse_text_response(
            "USE\nhistorical use\nEND_USE\nACCEPT\n"
            "ACCEPT a1 | task | export | inspect: run\nEND_ACCEPT"),
                         {"use": "historical use", "acceptance": [
                             {"id": "a1", "basis": "task", "requirement": "export",
                              "check": "inspect: run"}]})
        content = "REVIEW h1\nstatement: rule\nEND_REVIEW\n\n  indented\n"
        self.assertEqual(parse_text_response("FILE history-contract.txt\n" + content + "END_FILE"),
                         {"files": [{"name": "history-contract.txt", "content": content}]})
        self.assertEqual(parse_text_response("FILE: history-contract.txt\n" + content + "END_FILE"),
                         {"files": [{"name": "history-contract.txt", "content": content}]})
        adjacent = ("FILE memory-use.md\nprivate\nFILE history-contract.txt\n"
                    "REVIEW h1\nstatement: rule\nEND_REVIEW\nFILE acceptance.md\n"
                    "| ID | Requirement | Basis | Check |\nEND_FILE")
        self.assertEqual([row["name"] for row in parse_text_response(adjacent)["files"]],
                         ["memory-use.md", "history-contract.txt", "acceptance.md"])
        with self.assertRaises(ValueError):
            parse_text_response("FILE task.md\nincomplete")

    def test_history_qualify_uses_short_rows(self):
        from dialogue_benchmark.llm import parse_text_response
        self.assertEqual(parse_text_response(
            "H h1 | yes | partial | sufficient | source1 | query1 | Exact fact\n"
            "TASK | clean"),
            {"history_reviews": [{"id": "h1", "applicable": "yes",
                                   "public": "partial", "answer": "sufficient",
                                   "historical_sources": "source1",
             "public_sources": "query1",
             "answer_quote": "Exact fact", "issue": "none"}],
             "task_review": {"id": "task", "leakage": "clean", "issue": "none"}})
        self.assertEqual(parse_text_response(
            "h1 | yes | partial | sufficient | source1 | query1 | Exact fact\n"
            "TASK | clean"),
            {"history_reviews": [{"id": "h1", "applicable": "yes",
                                   "public": "partial", "answer": "sufficient",
                                   "historical_sources": "source1",
                                   "public_sources": "query1",
             "answer_quote": "Exact fact", "issue": "none"}],
             "task_review": {"id": "task", "leakage": "clean", "issue": "none"}})
        self.assertEqual(parse_text_response(
            "h1 | applicable=yes | public=partial | answer=sufficient | "
            "historical_source=source1 | public_source=query1 | "
            "answer_quote=Exact fact\nTASK | clean"),
            {"history_reviews": [{"id": "h1", "applicable": "yes",
                                   "public": "partial", "answer": "sufficient",
                                   "historical_sources": "source1",
                                   "public_sources": "query1",
                                   "answer_quote": "Exact fact", "issue": "none"}],
             "task_review": {"id": "task", "leakage": "clean", "issue": "none"}})

    def test_history_review_keeps_multiline_answer_quotes_grounded(self):
        from dialogue_benchmark.llm import parse_text_response
        from dialogue_benchmark.task_eval.runtime import review_task
        answer = "- External code 7 means committed.\n- External code 4 means pending."
        text = "H h1 | yes | none | sufficient | event1 | none | " + answer + "\nTASK | clean"
        parsed = parse_text_response(text)
        self.assertEqual(parsed["history_reviews"][0]["answer_quote"], answer)
        evidence = {"history_targets": [{"id": "h1", "sources": ["event1"]}],
                    "repository_queries": [], "contracts": [],
                    "sources": [{"id": "event1", "text": "confirmed"}]}
        for quote, status in ((text, "clean"),
                              (text.replace("4 means pending", "4 means committed"), "uncertain")):
            with self.subTest(quote=quote), patch("dialogue_benchmark.task_eval.runtime.ask_model",
                                                 return_value=parse_text_response(quote)):
                result = review_task("New task", answer, {}, self.root / "multiline-review",
                                     evidence=evidence)
                self.assertEqual(result["status"], status)
        for invalid in (text + "\n- Extra quote", text.replace("\nTASK", "\nExplanation\nTASK")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_text_response(invalid)

    def test_history_review_accepts_labeled_choices_but_not_prose(self):
        from dialogue_benchmark.llm import parse_text_response
        from dialogue_benchmark.task_eval.runtime import review_task
        evidence = {"history_targets": [{"id": "h1", "sources": ["event1"]}],
                    "repository_queries": [], "sources": []}
        for value, status in (("sufficient", "clean"),
                              ("sufficient because I believe it", "uncertain")):
            parsed = parse_text_response(
                "H h1 | applicable yes | public none | answer " + value
                + " | event1 | none | Exact fact\nTASK | clean")
            with self.subTest(value=value), patch("dialogue_benchmark.task_eval.runtime.ask_model",
                                                 return_value=parsed):
                result = review_task("New task", "Exact fact", {}, self.root / "labeled-review",
                                     evidence=evidence)
                self.assertEqual(result["status"], status)

    def test_selection_uses_short_query_row(self):
        from dialogue_benchmark.llm import parse_text_response
        parsed = parse_text_response(
            "DECISION: need_evidence\nREASON: Need the entry point\n"
            "SOURCES: source1\nQUERY: read|repo|api.py|-|0\nEND")
        self.assertEqual(parsed["reviews"][0]["request"], "read|repo|api.py|-|0")
        self.assertEqual(parsed["reviews"][0]["decision"], "need_evidence")

    def test_selection_rejects_two_queries_in_one_decision(self):
        from dialogue_benchmark.llm import parse_text_response
        with self.assertRaises(ValueError):
            parse_text_response(
                "DECISION: need_evidence\nREASON: Check both files\n"
                "SOURCES: source1\nQUERY: lookup|repo|.|one|0\n"
                "QUERY: lookup|repo|.|two|0\nEND")

    def test_selection_eof_requires_every_field_and_no_extra_prose(self):
        from dialogue_benchmark.llm import parse_text_response
        complete = ("DECISION: need_evidence\nREASON: Need the entry point\n"
                    "SOURCES: qa\nQUERY: read|repo|api.py|-|0")
        self.assertEqual(parse_text_response(complete), parse_text_response(complete + "\nEND"))
        for response in (complete.rsplit("\n", 1)[0], complete + "\nHere is my analysis",
                         complete.replace("need_evidence", "candidate")):
            with self.subTest(response=response), self.assertRaises(ValueError):
                parse_text_response(response)

    def test_clean_review_requires_quote_from_injected_answer(self):
        from dialogue_benchmark.task_eval.runtime import review_task
        for quote, status in (("Exact historical fact", "clean"), ("Fact only in source", "uncertain"),
                              ("none", "uncertain")):
            with self.subTest(quote=quote), patch("dialogue_benchmark.task_eval.runtime.ask_model",
                    return_value={"reviews": [{"leakage": "clean", "issue": "none",
                                               "memory_gap": "Required fact", "answer_quote": quote}]}):
                result = review_task("New task", "Exact historical fact", {}, self.root / "review")
                self.assertEqual(result["status"], status)

    def test_history_review_derives_status_from_each_fixed_target(self):
        from dialogue_benchmark.task_eval.runtime import review_task
        evidence = {"history_targets": [{"id": "h1", "sources": ["event1"]}],
                    "repository_queries": [{"id": "query1"}],
                    "contracts": [], "sources": [{"id": "event1", "text": "confirmed"}]}
        response = {"history_reviews": [
            {"id": "h1", "applicable": "yes", "public": "none", "answer": "sufficient",
             "historical_sources": "event1", "public_sources": "none",
             "answer_quote": "Exact historical fact", "issue": "none"},
        ], "task_review": {"id": "task", "leakage": "clean", "issue": "none"}}
        with patch("dialogue_benchmark.task_eval.runtime.ask_model", return_value=response):
            result = review_task("New task", "Exact historical fact", {}, self.root / "history-review",
                                 evidence=evidence)
        self.assertEqual(result["status"], "clean")
        self.assertEqual(result["history_rows"][0]["id"], "h1")

    def test_history_review_prompt_renders_exact_frozen_row_prefixes(self):
        from dialogue_benchmark.llm import parse_text_response
        from dialogue_benchmark.task_eval.runtime import review_task
        evidence = {"history_targets": [{"id": target, "sources": ["event1"]}
                                         for target in ("h2", "h7")]}
        response = parse_text_response(
            "H h2 | yes | none | sufficient | event1 | none | Exact fact\n"
            "H h7 | yes | none | sufficient | event1 | none | Exact fact\nTASK | clean")
        with patch("dialogue_benchmark.task_eval.runtime.ask_model", return_value=response) as ask:
            result = review_task("New task", "Exact fact", {}, self.root / "fixed-prefixes",
                                 evidence=evidence)
        self.assertEqual(result["status"], "clean")
        prompt = ask.call_args.args[0]
        rows = [line for line in prompt.splitlines() if line.startswith("H ")]
        self.assertEqual([line.split("|", 1)[0].strip() for line in rows], ["H h2", "H h7"])
        self.assertTrue(all(len(line.split("|")) == 7 for line in rows))
        self.assertIn("中间保留一个空格", prompt)
        self.assertNotIn("H h1", prompt)
        self.assertNotIn("HISTORY_QUALIFY_ROWS", prompt)

    def test_history_review_rejects_duplicate_unknown_and_concatenated_ids(self):
        from dialogue_benchmark.llm import parse_text_response
        from dialogue_benchmark.task_eval.runtime import review_task
        evidence = {"history_targets": [{"id": target, "sources": ["event1"]}
                                         for target in ("h1", "h2")]}
        for prefixes in (("H h1", "H h1"), ("H h1", "H h9"), ("Hh1", "Hh2")):
            with self.subTest(prefixes=prefixes):
                response = parse_text_response("\n".join(
                    prefix + " | yes | none | sufficient | event1 | none | Exact fact"
                    for prefix in prefixes) + "\nTASK | clean")
                self.assertEqual([row["id"] for row in response["history_reviews"]],
                                 [prefix.removeprefix("H ") for prefix in prefixes])
                with patch("dialogue_benchmark.task_eval.runtime.ask_model", return_value=response) as ask:
                    result = review_task("New task", "Exact fact", {}, self.root / "invalid-ids",
                                         evidence=evidence)
                self.assertEqual(ask.call_count, 1)
                self.assertEqual(result["status"], "uncertain")
                self.assertEqual(result["issue"], "history_review_failed")
                self.assertEqual(result["error"]["error_code"], "validation_error")

    def test_history_review_clear_ids_do_not_override_incomplete_answer(self):
        from dialogue_benchmark.llm import parse_text_response
        from dialogue_benchmark.task_eval.runtime import review_task
        evidence = {"history_targets": [{"id": target, "sources": ["event1"]}
                                         for target in ("h1", "h2", "h3")],
                    "repository_queries": [{"id": "query1"}]}
        response = parse_text_response(
            "H h1 | yes | partial | sufficient | event1 | query1 | Exact fact\n"
            "H h2 | yes | full | not_applicable | event1 | query1 | none\n"
            "H h3 | yes | partial | insufficient | event1 | query1 | none\nTASK | clean")
        with patch("dialogue_benchmark.task_eval.runtime.ask_model", return_value=response) as ask:
            result = review_task("New task", "Exact fact", {}, self.root / "incomplete-answer",
                                 evidence=evidence)
        self.assertEqual(ask.call_count, 1)
        self.assertEqual(result["status"], "uncertain")
        self.assertEqual(result["issue"], "historical_answer_incomplete")

    def test_history_review_does_not_accept_one_quote_for_missing_target(self):
        from dialogue_benchmark.task_eval.runtime import review_task
        evidence = {"history_targets": [{"id": "h1", "sources": ["event1"]},
                                         {"id": "h2", "sources": ["event1"]}],
                    "repository_queries": [], "contracts": [],
                    "sources": [{"id": "event1", "text": "confirmed"}]}
        response = {"history_reviews": [
            {"id": "h1", "applicable": "yes", "public": "none", "answer": "sufficient",
             "historical_sources": "event1", "public_sources": "none",
             "answer_quote": "Exact historical fact", "issue": "none"},
            {"id": "h2", "applicable": "yes", "public": "none", "answer": "insufficient",
             "historical_sources": "event1", "public_sources": "none",
             "answer_quote": "none", "issue": "missing"},
        ], "task_review": {"id": "task", "leakage": "clean", "issue": "none"}}
        with patch("dialogue_benchmark.task_eval.runtime.ask_model", return_value=response):
            result = review_task("New task", "Exact historical fact", {}, self.root / "history-review-2",
                                 evidence=evidence)
        self.assertEqual(result["status"], "uncertain")
        self.assertEqual(result["issue"], "historical_answer_incomplete")

    def test_history_review_separates_private_rules_from_public_evidence(self):
        from dialogue_benchmark.task_eval.runtime import review_task
        evidence = {"history_targets": [{"id": "h1", "sources": ["event1"]}],
                    "repository_exploration": "api.py exposes configurable null handling.",
                    "contracts": [{"statement": "Private rule"}],
                    "sources": [{"id": "event1", "text": "Private raw history"}]}
        response = {"history_reviews": [
            {"id": "h1", "applicable": "yes", "public": "partial", "answer": "sufficient",
             "historical_sources": "event1", "public_sources": "repository_exploration",
             "answer_quote": "Exact rule"},
        ], "task_review": {"id": "task", "leakage": "clean", "issue": "none"}}
        with patch("dialogue_benchmark.task_eval.runtime.ask_model", return_value=response) as ask:
            result = review_task("Use the previous agreement", "Exact rule", {}, self.root / "separated",
                                 evidence=evidence)
        self.assertEqual(result["status"], "clean")
        payload = ask.call_args.args[1]
        self.assertEqual(set(payload), {"public_task", "public_repository",
                                       "private_history_targets", "injected_answer"})
        self.assertEqual(payload["public_repository"], [{"id": "repository_exploration",
                                                       "result": evidence["repository_exploration"]}])
        response["history_reviews"][0].update(public="full", public_sources="none")
        with patch("dialogue_benchmark.task_eval.runtime.ask_model", return_value=response):
            result = review_task("Use the previous agreement", "Exact rule", {}, self.root / "missing-public",
                                 evidence=evidence)
        self.assertEqual(result["status"], "uncertain")
        self.assertIn("public_evidence_missing", result["issue"])


if __name__ == "__main__":
    unittest.main()
