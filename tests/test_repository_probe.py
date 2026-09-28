import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dialogue_benchmark import cli
from dialogue_benchmark.llm import parse_text_response
from dialogue_benchmark.repository_probe import PROBE_PROMPT, _repository_entries, probe_candidate, repository_anchors


class FakeProbeClient:
    responses = []

    def __init__(self, *unused, **kwargs):
        self.usage = []

    def ask(self, prompt, payload):
        self.usage.append({"request_count": 1, "prompt_tokens": 10,
                           "completion_tokens": 3})
        value = self.responses.pop(0)
        return parse_text_response(value) if isinstance(value, str) else value


class RepositoryProbeTests(unittest.TestCase):
    def test_probe_file_map_includes_readable_nested_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "src/api.py").write_text("value = 1\n")
            (root / "README.md").write_text("Demo\n")
            self.assertEqual(_repository_entries(root), ["README.md", "src/api.py"])

    def test_protocol_is_small_and_rejects_answer_style_fields(self):
        self.assertIn("observations 中 id 为 query1 的结果支持判断时，输出 EVIDENCE: query1", PROBE_PROMPT)
        self.assertNotIn("END_PROBE", PROBE_PROMPT)
        result = parse_text_response(
            "PROBE: need_evidence\nREASON: read the entry\n"
            "QUERY: read|repo|config.py|-|0\nEVIDENCE: none")
        self.assertEqual(result["probe"]["decision"], "need_evidence")
        with self.assertRaises(ValueError):
            parse_text_response(
                "PROBE: recoverable\nREASON: yes\nQUERY: none\n"
                "EVIDENCE: q1\nANSWER: leaked")

    def test_protocol_requires_each_field_exactly_once_and_nonempty(self):
        lines = ["PROBE: need_evidence", "REASON: inspect entry",
                 "QUERY: read|repo|api.py|-|0", "EVIDENCE: query1"]
        for index, line in enumerate(lines):
            invalid = [lines[:index] + lines[index + 1:],
                       lines + [line],
                       lines[:index] + [line.split(":", 1)[0] + ": "] + lines[index + 1:]]
            for record in invalid:
                with self.subTest(response=record), self.assertRaises(ValueError):
                    parse_text_response("\n".join(record))
        for extra in ("EXTRA: text", "unlabelled text", "END_PROBE"):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                parse_text_response("\n".join(lines + [extra]))

    def test_protocol_requires_valid_decision_and_matching_query(self):
        for decision, query in (
                ("unknown", "none"), ("need_evidence", "none"),
                ("recoverable", "read|repo|api.py|-|0"),
                ("history_required", "read|repo|api.py|-|0"),
                ("uncertain", "read|repo|api.py|-|0")):
            with self.subTest(decision=decision, query=query), self.assertRaises(ValueError):
                parse_text_response("PROBE: %s\nREASON: inspect entry\nQUERY: %s\nEVIDENCE: none"
                                    % (decision, query))

    def test_other_protocols_still_require_their_end_markers(self):
        for response in ("TASK\nAdd export", "QA q1\nQUESTION: Which limit applies?",
                         "REVIEW q1\nreason: supported"):
            with self.subTest(response=response), self.assertRaises(ValueError):
                parse_text_response(response)

    def test_incomplete_probe_response_stays_uncertain_without_retry(self):
        responses = [
            "PROBE: need_evidence\nREASON: inspect entry\n"
            "QUERY: read|repo|api.py|-|0\nEVIDENCE: none",
            "PROBE: need_evidence\nREASON: inspect the remaining implementation\n"
            "QUERY: lookup|repo|.|export_records|0",
        ]
        envelopes = [io.BytesIO(json.dumps({"choices": [
            {"finish_reason": "stop", "message": {"content": response}}
        ]}).encode()) for response in responses]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            (root / "api.py").write_text("def export_records(rows): return rows\n")
            output = Path(directory) / "probe"
            with patch.dict("os.environ", {"BENCHMARK_API_KEY": "test-placeholder-key"}), \
                    patch("dialogue_benchmark.llm.urllib.request.build_opener") as opener:
                opener.return_value.open.side_effect = envelopes
                result = probe_candidate({"question": "Which customer selection applies?"}, root,
                    "https://example.invalid", "m", "BENCHMARK_API_KEY", output)
            self.assertEqual(opener.return_value.open.call_count, 2)
            self.assertEqual(result["status"], "uncertain")
            self.assertEqual(result["reason"], "probe_error:ModelStageError")
            self.assertEqual(result["error_code"], "protocol_error")
            self.assertEqual(result["query_count"], 1)
            self.assertEqual((output / "step-002/response-text.txt").read_text(), responses[1])
            self.assertFalse((output / "step-002/query.json").exists())

    def test_provider_truncation_stays_uncertain_with_four_complete_fields(self):
        response = ("PROBE: need_evidence\nREASON: inspect entry\n"
                    "QUERY: read|repo|api.py|-|0\nEVIDENCE: none")
        envelope = io.BytesIO(json.dumps({"choices": [
            {"finish_reason": "length", "message": {"content": response}}
        ]}).encode())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            (root / "api.py").write_text("def export_records(rows): return rows\n")
            output = Path(directory) / "probe"
            with patch.dict("os.environ", {"BENCHMARK_API_KEY": "test-placeholder-key"}), \
                    patch("dialogue_benchmark.llm.urllib.request.build_opener") as opener:
                opener.return_value.open.return_value = envelope
                result = probe_candidate({"question": "Which customer selection applies?"}, root,
                    "https://example.invalid", "m", "BENCHMARK_API_KEY", output)
            self.assertEqual(opener.return_value.open.call_count, 1)
            self.assertEqual(result["status"], "uncertain")
            self.assertEqual(result["error_code"], "incomplete_response")
            self.assertEqual(result["query_count"], 0)
            self.assertFalse((output / "step-001/query.json").exists())
            self.assertFalse((output / "step-001/response-text.txt").exists())

    def test_four_field_probe_continues_from_existing_query_evidence(self):
        responses = [
            "PROBE: need_evidence\nREASON: inspect implementation\n"
            "QUERY: read|repo|fulfilment.py|-|0\nEVIDENCE: none",
            "PROBE: need_evidence\n"
            "REASON: 当前只看到 fulfilment.py 前半部分，尚未看到批准决定、write_partner_batches 或测试中的范围限定说明。\n"
            "QUERY: read|repo|tests/test_stage_00.py|-|0\nEVIDENCE: query1",
            "PROBE: history_required\nREASON: approval is not recorded in the repository\n"
            "QUERY: none\nEVIDENCE: query1,query2",
        ]
        envelopes = [io.BytesIO(json.dumps({"choices": [
            {"finish_reason": "stop", "message": {"content": response}}
        ]}).encode()) for response in responses]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            (root / "tests").mkdir(parents=True)
            (root / "fulfilment.py").write_text("def write_partner_batches(rows): return rows\n")
            (root / "tests/test_stage_00.py").write_text("def test_empty(): assert not []\n")
            output = Path(directory) / "probe"
            with patch.dict("os.environ", {"BENCHMARK_API_KEY": "test-placeholder-key"}), \
                    patch("dialogue_benchmark.llm.urllib.request.build_opener") as opener:
                opener.return_value.open.side_effect = envelopes
                result = probe_candidate({"question": "Which customer selection applies?"}, root,
                    "https://example.invalid", "m", "BENCHMARK_API_KEY", output, max_steps=2)
            self.assertEqual(opener.return_value.open.call_count, 3)
            self.assertEqual(result["status"], "history_required")
            self.assertEqual(result["evidence"], ["query1", "query2"])
            self.assertEqual(result["query_count"], 2)
            self.assertTrue(all(item["status"] == "completed" for item in result["usage"]))
            self.assertTrue((output / "step-002/query.json").exists())
            self.assertFalse((output / "step-003/query.json").exists())

    def test_probe_evidence_requires_existing_query_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            (root / "api.py").write_text("def export_records(rows): return rows\n")
            for decision in ("need_evidence", "recoverable", "history_required"):
                for evidence in ("api.py lines 1-80", "query2"):
                    with self.subTest(decision=decision, evidence=evidence):
                        query = "lookup|repo|.|export_records|0" if decision == "need_evidence" else "none"
                        FakeProbeClient.responses = [
                            "PROBE: need_evidence\nREASON: inspect entry\n"
                            "QUERY: read|repo|api.py|-|0\nEVIDENCE: none",
                            "PROBE: %s\nREASON: check selection\nQUERY: %s\nEVIDENCE: %s"
                            % (decision, query, evidence),
                        ]
                        with patch("dialogue_benchmark.repository_probe.ChatClient", FakeProbeClient):
                            result = probe_candidate({"question": "Which customer selection applies?"}, root,
                                "https://example.invalid", "m", "KEY", Path(directory) / "probe")
                        self.assertEqual(result["status"], "uncertain")
                        self.assertEqual(result["reason"], "unknown_probe_evidence")
                        self.assertEqual(result["query_count"], 1)
                        self.assertEqual(len(result["usage"]), 2)

    def test_probe_reads_repo_and_checks_answer_claims(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            (root / "config.py").write_text("def load_config():\n    raise ValueError('missing')\n")
            output = Path(directory) / "probe"
            FakeProbeClient.responses = [
                "PROBE: need_evidence\nREASON: inspect entry\n"
                "QUERY: read|repo|config.py|-|0\nEVIDENCE: none",
                "PROBE: recoverable\nREASON: the file states the behavior\n"
                "QUERY: none\nEVIDENCE: query1",
            ]
            question = {"question": "config.py 的 load_config 如何处理缺失配置？",
                        "answer_points": [{"text": "load_config 遇到缺失配置会抛出 ValueError"}]}
            with patch("dialogue_benchmark.repository_probe.ChatClient", FakeProbeClient):
                result = probe_candidate(question, root, "https://example.invalid", "m", "KEY", output)
            self.assertEqual(result["status"], "recoverable")
            payload = json.dumps(json.loads((output / "step-001/input.json").read_text()),
                                 ensure_ascii=False)
            self.assertIn("load_config 遇到缺失配置会抛出 ValueError", payload)
            self.assertEqual(result["query_count"], 1)

    def test_mock_probe_receives_scope_evidence_beyond_first_read_page(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            (root / "tests").mkdir(parents=True)
            scope_line = "    approved = {'P1': []}"
            (root / "tests/test_scope.py").write_text(
                "# unrelated setup\n" * 96 + "def test_approved_scope():\n"
                + scope_line + "\n    assert set(approved) == {'P1'}\n")
            output = Path(directory) / "probe"
            FakeProbeClient.responses = [
                "PROBE: need_evidence\nREASON: locate the approval test\n"
                "QUERY: lookup|repo|.|approved_scope|0\nEVIDENCE: none",
                "PROBE: need_evidence\nREASON: inspect the test file\n"
                "QUERY: read|repo|tests/test_scope.py|-|0\nEVIDENCE: query1",
                "PROBE: need_evidence\nREASON: continue to the scope test\n"
                "QUERY: read|repo|tests/test_scope.py|-|80\nEVIDENCE: query1,query2",
                "PROBE: recoverable\nREASON: the scope test explicitly selects P1\n"
                "QUERY: none\nEVIDENCE: query3",
            ]
            question = {"question": "Which partner does the approval test select?",
                        "answer_points": [{"text": "The approval test selects only P1."}]}
            with patch("dialogue_benchmark.repository_probe.ChatClient", FakeProbeClient):
                result = probe_candidate(question, root, "https://example.invalid", "m", "KEY",
                                         output, max_steps=3)
            lookup_input = json.loads((output / "step-002/input.json").read_text())["payload"]
            lookup = lookup_input["observations"][0]
            self.assertEqual(lookup["result"]["matches"], [
                {"match": "tests/test_scope.py:97:def test_approved_scope():"}])
            page_input = json.loads((output / "step-003/input.json").read_text())["payload"]
            first_page = page_input["observations"][1]
            self.assertEqual(first_page["id"], "query2")
            self.assertEqual(first_page["result"]["next_offset"], 80)
            self.assertEqual(first_page["result"]["lines"][-1]["line"], 80)
            self.assertNotIn(scope_line, [row["text"] for row in first_page["result"]["lines"]])
            final_input = json.loads((output / "step-004/input.json").read_text())["payload"]
            scope_page = final_input["observations"][2]
            self.assertEqual(scope_page["query"]["offset"], first_page["result"]["next_offset"])
            self.assertIn({"line": 98, "text": scope_line}, scope_page["result"]["lines"])
            self.assertIsNone(scope_page["result"]["next_offset"])
            self.assertEqual(final_input["remaining_queries"], 0)
            self.assertEqual(result["status"], "recoverable")
            self.assertEqual(result["evidence"], [scope_page["id"]])
            self.assertEqual(result["query_count"], 3)
            self.assertEqual(len(result["usage"]), 4)

    def test_duplicate_query_does_not_claim_recoverable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            (root / "api.py").write_text("return None\n")
            FakeProbeClient.responses = [
                "PROBE: need_evidence\nREASON: inspect\n"
                "QUERY: read|repo|api.py|-|0\nEVIDENCE: none",
                "PROBE: need_evidence\nREASON: inspect again\n"
                "QUERY: read|repo|api.py|-|0\nEVIDENCE: query1",
            ]
            with patch("dialogue_benchmark.repository_probe.ChatClient", FakeProbeClient):
                result = probe_candidate({"question": "api.py 的返回行为是什么？"}, root,
                                         "https://example.invalid", "m", "KEY",
                                         Path(directory) / "probe")
            self.assertEqual(result["status"], "uncertain")
            self.assertIn("invalid_probe_query", result["reason"])

    def test_last_query_result_is_available_to_the_final_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            (root / "api.py").write_text("def send(rows, *, limit): pass\n")
            output = Path(directory) / "probe"
            FakeProbeClient.responses = [
                "PROBE: need_evidence\nREASON: inspect configuration\n"
                "QUERY: read|repo|api.py|-|0\nEVIDENCE: none",
                "PROBE: history_required\nREASON: the caller supplies the external limit\n"
                "QUERY: none\nEVIDENCE: query1",
            ]
            with patch("dialogue_benchmark.repository_probe.ChatClient", FakeProbeClient):
                result = probe_candidate({"question": "Which customer limit applies?"}, root,
                                         "https://example.invalid", "m", "KEY", output,
                                         max_steps=1)
            self.assertEqual(result["status"], "history_required")
            self.assertEqual(result["query_count"], 1)
            final_input = json.loads((output / "step-002/input.json").read_text())["payload"]
            self.assertEqual(final_input["remaining_queries"], 0)
            self.assertEqual(final_input["observations"][0]["result"]["lines"][0]["text"],
                             "def send(rows, *, limit): pass")

    def test_final_decision_cannot_execute_an_extra_query(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            (root / "api.py").write_text("def send(rows, *, limit): pass\n")
            FakeProbeClient.responses = [
                "PROBE: need_evidence\nREASON: inspect\n"
                "QUERY: read|repo|api.py|-|0\nEVIDENCE: none",
                "PROBE: need_evidence\nREASON: search further\n"
                "QUERY: lookup|repo|.|limit|0\nEVIDENCE: query1",
            ]
            with patch("dialogue_benchmark.repository_probe.ChatClient", FakeProbeClient):
                result = probe_candidate({"question": "Which customer limit applies?"}, root,
                                         "https://example.invalid", "m", "KEY",
                                         Path(directory) / "probe", max_steps=1)
            self.assertEqual(result["status"], "uncertain")
            self.assertEqual(result["reason"], "probe_steps_exhausted")
            self.assertEqual(result["query_count"], 1)

    def test_terminal_probe_with_followup_query_is_rejected_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            (root / "api.py").write_text("return None\n")
            for decision in ("recoverable", "history_required", "uncertain"):
                with self.subTest(decision=decision):
                    FakeProbeClient.responses = [{"probe": {
                        "decision": decision, "reason": "inspect the file",
                        "query": "read|repo|api.py|-|0", "evidence": "none"}}]
                    with patch("dialogue_benchmark.repository_probe.ChatClient", FakeProbeClient):
                        result = probe_candidate({"question": "api.py 的行为是什么？"}, root,
                            "https://example.invalid", "m", "KEY", Path(directory) / "probe")
                    self.assertEqual(result["status"], "uncertain")
                    self.assertEqual(result["reason"], "invalid_probe_response")
                    self.assertEqual(result["query_count"], 0)
                    self.assertEqual(len(result["usage"]), 1)

    def test_invalid_queries_stay_uncertain_without_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            (root / "api.py").write_text("return None\n")
            for query in ("read|repo|api.py|-", "read|repo|api.py|-|bad",
                          "read|repo|api.py|-|-1", "execute|repo|api.py|-|0",
                          "read|history|-|source1|0", "read|repo|../api.py|-|0"):
                with self.subTest(query=query):
                    FakeProbeClient.responses = [
                        "PROBE: need_evidence\nREASON: inspect entry\nQUERY: %s\nEVIDENCE: none"
                        % query]
                    with patch("dialogue_benchmark.repository_probe.ChatClient", FakeProbeClient):
                        result = probe_candidate({"question": "Which customer selection applies?"}, root,
                            "https://example.invalid", "m", "KEY", Path(directory) / "probe")
                    self.assertEqual(result["status"], "uncertain")
                    self.assertEqual(result["reason"], "invalid_probe_query:ValueError")
                    self.assertEqual(result["query_count"], 0)
                    self.assertEqual(len(result["usage"]), 1)

    def test_terminal_decisions_without_queries_require_uncertainty(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            (root / "api.py").write_text("return None\n")
            for decision, expected_reason in (
                    ("recoverable", "recoverable_without_repository_evidence"),
                    ("history_required", "history_required_without_repository_evidence"),
                    ("uncertain", "not enough context")):
                with self.subTest(decision=decision):
                    FakeProbeClient.responses = [
                        "PROBE: %s\nREASON: not enough context\nQUERY: none\nEVIDENCE: none"
                        % decision]
                    with patch("dialogue_benchmark.repository_probe.ChatClient", FakeProbeClient):
                        result = probe_candidate({"question": "api.py 的行为是什么？"}, root,
                            "https://example.invalid", "m", "KEY", Path(directory) / "probe")
                    self.assertEqual(result["status"], "uncertain")
                    self.assertEqual(result["reason"], expected_reason)
                    self.assertEqual(result["query_count"], 0)
                    self.assertEqual(len(result["usage"]), 1)

    def test_definitive_decisions_without_valid_cited_evidence_stay_uncertain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            (root / "api.py").write_text("return None\n")
            for decision in ("recoverable", "history_required"):
                for evidence in (None, "", "none", "query2", "api.py lines 1-80"):
                    with self.subTest(decision=decision, evidence=evidence):
                        terminal = "PROBE: %s\nREASON: behavior checked\nQUERY: none" % decision
                        if evidence is not None:
                            terminal += "\nEVIDENCE: " + evidence
                        FakeProbeClient.responses = [
                            "PROBE: need_evidence\nREASON: inspect entry\n"
                            "QUERY: read|repo|api.py|-|0\nEVIDENCE: none", terminal]
                        with patch("dialogue_benchmark.repository_probe.ChatClient", FakeProbeClient):
                            result = probe_candidate({"question": "api.py 的行为是什么？"}, root,
                                "https://example.invalid", "m", "KEY", Path(directory) / "probe")
                        self.assertEqual(result["status"], "uncertain")
                        self.assertEqual(result["query_count"], 1)
                        self.assertEqual(len(result["usage"]), 2)

    def test_queries_without_content_cannot_support_definitive_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            (root / "api.py").write_text("return None\n")
            for decision in ("recoverable", "history_required"):
                for query in ("lookup|repo|.|missing_symbol|0", "lookup|repo|.|api.py|0",
                              "read|repo|api.py|-|1"):
                    with self.subTest(decision=decision, query=query):
                        FakeProbeClient.responses = [
                            "PROBE: need_evidence\nREASON: inspect entry\n"
                            "QUERY: read|repo|api.py|-|0\nEVIDENCE: none",
                            "PROBE: need_evidence\nREASON: search further\n"
                            "QUERY: %s\nEVIDENCE: query1" % query,
                            "PROBE: %s\nREASON: behavior checked\nQUERY: none\nEVIDENCE: query2"
                            % decision,
                        ]
                        with patch("dialogue_benchmark.repository_probe.ChatClient", FakeProbeClient):
                            result = probe_candidate({"question": "api.py 的 missing_symbol 行为是什么？"}, root,
                                "https://example.invalid", "m", "KEY", Path(directory) / "probe")
                        self.assertEqual(result["status"], "uncertain")
                        self.assertEqual(result["reason"], decision + "_without_repository_evidence")
                        self.assertEqual(result["query_count"], 2)
                        self.assertEqual(len(result["usage"]), 3)

    def test_lookup_content_supports_definitive_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            (root / "api.py").write_text("def send(rows, *, limit): pass\n")
            for decision, question, reason in (
                    ("recoverable", "How does send obtain its limit?", "the caller supplies it"),
                    ("history_required", "Which customer limit applies?", "the limit must come from outside")):
                with self.subTest(decision=decision):
                    FakeProbeClient.responses = [
                        "PROBE: need_evidence\nREASON: search\n"
                        "QUERY: lookup|repo|.|limit|0\nEVIDENCE: none",
                        "PROBE: %s\nREASON: %s\nQUERY: none\nEVIDENCE: query1"
                        % (decision, reason),
                    ]
                    with patch("dialogue_benchmark.repository_probe.ChatClient", FakeProbeClient):
                        result = probe_candidate({"question": question}, root,
                            "https://example.invalid", "m", "KEY", Path(directory) / "probe")
                    self.assertEqual(result["status"], decision)
                    self.assertEqual(result["reason"], reason)
                    self.assertEqual(result["evidence"], ["query1"])
                    self.assertEqual(result["observations"][0]["result"]["matches"], [
                        {"match": "api.py:1:def send(rows, *, limit): pass"}])
                    self.assertEqual(result["query_count"], 1)
                    self.assertEqual(len(result["usage"]), 2)

    def test_recoverable_candidate_is_filtered_before_quota(self):
        question = {"id": "q1", "qa_mode": "code", "status": "approved",
                    "question": "config.py 如何处理缺失配置？", "answer_points": [],
                    "forbidden_points": []}
        view = cli._publication_view(
            [question], {"code": 1},
            recoverability_check=lambda item: {"status": "recoverable", "evidence": ["query1"]})
        self.assertEqual(view["questions"], [])
        self.assertEqual(view["publication_rejected"][0]["reason"],
                         "repository_recoverable")
        audit = cli.build_audit([question], [question], view["publication_rejected"],
                                [], [], [])
        self.assertEqual(audit[0]["selection_status"], "filtered_recoverable")
        self.assertEqual(audit[0]["review_status"], "approved")

    def test_external_mode_does_not_publish_uncertain_recoverability(self):
        question = {"id": "q1", "qa_mode": "code", "status": "approved",
                    "question": "config.py 如何处理缺失配置？", "answer_points": [],
                    "forbidden_points": []}
        view = cli._publication_view(
            [question], {"code": 1}, strict_external=True,
            recoverability_check=lambda item: {"status": "uncertain", "reason": "probe_error"})
        self.assertEqual(view["questions"], [])
        self.assertEqual(view["publication_rejected"][0]["reason"],
                         "repository_recoverability_uncertain")

    def test_probe_does_not_receive_credentials_or_private_paths(self):
        seen = []
        unsafe = {"id": "q1", "qa_mode": "code", "status": "approved",
                  "question": "config.py 如何处理？",
                  "answer_points": [{"text": "token=secret-value"}],
                  "forbidden_points": []}
        safe = {"id": "q2", "qa_mode": "code", "status": "approved",
                "question": "/Users/alice/work/config.py 如何处理？",
                "answer_points": [{"text": "ordinary"}],
                "forbidden_points": []}
        view = cli._publication_view(
            [unsafe, safe], {"code": 1}, workspaces=("/Users/alice/work",),
            recoverability_check=lambda item: seen.append(item) or {
                "status": "uncertain", "reason": "not enough"})
        self.assertEqual(len(view["questions"]), 1)
        self.assertNotIn("secret-value", json.dumps(seen))
        self.assertNotIn("/Users/alice", json.dumps(seen))

    def test_anchor_extraction_includes_claims_to_verify(self):
        anchors = repository_anchors("src/config.py 的 load_config 应处理 ValueError")
        self.assertIn("src/config.py", anchors["paths"])
        self.assertIn("load_config", anchors["terms"])
        self.assertIn("ValueError", anchors["errors"])
        claim_anchors = repository_anchors("如何处理缺失配置？", ["ParameterSource.UNSET"])
        self.assertIn("ParameterSource", claim_anchors["terms"])

    def test_cli_runs_probe_before_applying_quota(self):
        example = Path(__file__).resolve().parents[1] / "examples" / "dialogue.json"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            output = Path(directory) / "run"

            class FakeClient:
                def __init__(self, *unused):
                    self.usage = []

            def extract(scope, client, qa_mode, checkpoint=None):
                source = scope["dialogue"][0]["id"]
                return {"facts": [{"id": "f1", "qa_mode": qa_mode,
                                   "statement": "config.py preserves the recorded behavior",
                                   "sources": [source]}],
                        "questions": [], "rejected": [], "stage_errors": [],
                        "stage_status": {"facts": "completed"}}

            def generate(scope, facts, client, max_questions, qa_mode, target_type, **kwargs):
                source = facts[0]["sources"][0]
                question = {"id": "q1", "candidate_id": "q1", "qa_mode": qa_mode,
                            "question": "config.py 的 load_config 如何处理缺失配置？",
                            "answer_points": [{"text": "recorded behavior", "sources": [source]}],
                            "forbidden_points": []}
                return {"questions": [question], "all_candidates": [question],
                        "rejected": [], "stage_errors": [], "raw_generated": 1,
                        "stage_status": {"qa": "completed"}}

            def review(scope, facts, candidates, client, **kwargs):
                return {"questions": [dict(item, status="approved") for item in candidates],
                        "rejected": [], "stage_errors": [], "revisions": [],
                        "stage_status": {"review": "completed"}}

            with patch.object(cli, "ChatClient", FakeClient), \
                    patch.object(cli, "extract_facts", extract), \
                    patch.object(cli, "generate_from_facts", generate), \
                    patch.object(cli, "review_candidates", review), \
                    patch.object(cli, "build_evidence_groups", return_value=[{
                        "id": "code-group-1", "qa_mode": "code",
                        "scope": {"dialogue": [{"id": "e1", "order": 1,
                                                   "kind": "message", "role": "user",
                                                   "text": "config.py"}],
                                  "events": [], "versions": [], "edges": [],
                                  "historical_edges": [], "stages": []},
                        "facts": [{"id": "f1", "qa_mode": "code",
                                   "statement": "config.py preserves the recorded behavior",
                                   "sources": ["e1"]}],
                        "allowed_types": ("constraint_followthrough",),
                        "eligible_types": ("constraint_followthrough",),
                    }]), \
                    patch.object(cli, "static_evidence_check", return_value={
                        "status": "supported", "reason": "recorded_type_evidence",
                        "fact_ids": ["f1"], "source_ids": ["e1"],
                    }), \
                    patch("dialogue_benchmark.repository_probe.probe_candidate",
                          return_value={"status": "recoverable", "evidence": ["query1"],
                                         "reason": "current source is sufficient", "usage": []}) as probe:
                status = cli.main([
                    str(example), "--output", str(output), "--qa-mode", "code",
                    "--code-types", "constraint_followthrough", "--code-count", "1",
                    "--code-group-budget", "1", "--parallel-workers", "1",
                    "--allow-network", "--endpoint", "https://example.invalid",
                    "--model", "model", "--repository", str(root),
                    "--request-timeout", "1800",
                ])
            public = json.loads((output / "qa-public.json").read_text())
            audit = json.loads((output / "qa-audit.json").read_text())
            self.assertEqual(status, 0)
            self.assertEqual(public["counts"]["code"], 0)
            self.assertEqual(audit["recoverability"]["filtered"], 1)
            self.assertEqual(probe.call_args.kwargs["request_timeout"], 1800)
            self.assertEqual(audit["candidate_records"][0]["selection_status"],
                             "filtered_recoverable")


if __name__ == "__main__":
    unittest.main()
