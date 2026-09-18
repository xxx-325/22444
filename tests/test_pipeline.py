import copy
import json
import tempfile
import unittest
from pathlib import Path

from dialogue_benchmark.cli import main
from dialogue_benchmark.graph import build_graph, graph_at, query_scope
from dialogue_benchmark.chunking import split_scope
from dialogue_benchmark.llm import (evidence_projection, generate, outbound_guard,
                                    parse_json_response, parse_text_response)
from dialogue_benchmark.normalize import load_dialogue, normalize_codex, normalize_path
from dialogue_benchmark.patches import apply_diff
from dialogue_benchmark.quality import apply_review, validate_candidates
from dialogue_benchmark.subgraph import adaptive_subgraphs, build_event_index


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "dialogue.json"


class PatchTests(unittest.TestCase):
    def test_replace(self):
        self.assertEqual(apply_diff("a\nb\n", "@@ -1,2 +1,2 @@\n a\n-b\n+c\n"), "a\nc\n")

    def test_insert_empty(self):
        self.assertEqual(apply_diff("", "@@ -0,0 +1 @@\n+x\n"), "x\n")

    def test_delete(self):
        self.assertEqual(apply_diff("x\n", "@@ -1 +0,0 @@\n-x\n"), "")

    def test_no_newline(self):
        self.assertEqual(apply_diff("a", "@@ -1 +1 @@\n-a\n\\ No newline at end of file\n+b\n"), "b\n")

    def test_bad_context(self):
        with self.assertRaises(ValueError):
            apply_diff("a\n", "@@ -1 +1 @@\n-b\n+c\n")

    def test_bad_new_offset(self):
        with self.assertRaises(ValueError):
            apply_diff("a\n", "@@ -1 +9 @@\n-a\n+c\n")


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.records = load_dialogue(EXAMPLE)

    def test_versions_and_call_pair(self):
        graph = build_graph(self.records)
        self.assertEqual(len(graph["snapshots"]), 6)
        self.assertEqual(len(graph["versions"]), 4)
        self.assertEqual(graph["events"][4]["call_source"], "e4")
        self.assertEqual(graph["snapshots"][1]["files"]["config.py"], "v1")
        self.assertEqual(graph["snapshots"][2]["files"]["config.py"], "v3")

    def test_codex_records_receive_source_kinds(self):
        from dialogue_benchmark.normalize import normalize_codex
        rows = [
            (1, {"type": "response_item", "payload": {
                "type": "message", "role": "user", "content": [{"text": "规范"}],
                "source_kind": "document"}}),
            (2, {"type": "response_item", "payload": {
                "type": "function_call", "call_id": "c1", "name": "cat", "arguments": "{}"}}),
        ]
        result = normalize_codex(rows)
        self.assertEqual([item["source_kind"] for item in result], ["document", "tool"])

    def test_relative_import_and_cutoff(self):
        graph = build_graph(self.records)
        current = graph_at(graph, 2)
        self.assertTrue(any(e["to"] == "config.py::load_config" and e["kind"] == "call_reference"
                            for e in current["edges"]))
        scope = query_scope(graph, self.records, "config.py", 3)
        self.assertTrue(all(r["order"] <= 3 for r in scope["dialogue"]))
        self.assertNotIn("e6", {v["source"] for v in scope["versions"]})

    def test_failed_patch_does_not_modify(self):
        self.records[2]["success"] = False
        graph = build_graph(self.records[:3])
        self.assertEqual(graph["snapshots"][-1]["files"]["config.py"], "v1")

    def test_bad_patch_invalidates(self):
        self.records[2]["changes"]["config.py"]["unified_diff"] = "broken"
        graph = build_graph(self.records[:3])
        self.assertEqual(graph["versions"][-1]["status"], "unknown")
        self.assertNotIn("config.py", {n["id"] for n in graph_at(graph, 3)["nodes"]})

    def test_missing_base_not_invented(self):
        graph = build_graph(self.records[2:3])
        self.assertEqual(graph["versions"][0]["status"], "unknown")

    def test_delete_invalidates_edges(self):
        self.records[2]["changes"] = {"config.py": {"type": "delete"}}
        graph = build_graph(self.records[:3])
        self.assertFalse(any(e["to"] == "config.py::load_config" for e in graph_at(graph, 3)["edges"]))
        self.assertTrue(any(e["to"] == "config.py::load_config" for e in graph_at(graph, 2)["edges"]))

    def test_move(self):
        self.records[2]["changes"]["config.py"]["move_path"] = "settings.py"
        graph = build_graph(self.records[:3])
        nodes = {n["id"] for n in graph_at(graph, 3)["nodes"]}
        self.assertIn("settings.py", nodes)
        self.assertNotIn("config.py", nodes)
        scope = query_scope(graph, self.records[:3], "settings.py", 3)
        self.assertIn("e2", {v["source"] for v in scope["versions"]})

    def test_removed_call_remains_historical_only(self):
        self.records[2]["changes"] = {"main.py": {"type": "update", "unified_diff":
            "@@ -3,3 +3,2 @@\n def main(exists):\n-    value = load_config(exists)\n-    return 'missing' if value is None else 'ready'\n+    return 'ready'\n"}}
        graph = build_graph(self.records[:3])
        scope = query_scope(graph, self.records[:3], "main.py::main", 3, hops=1)
        self.assertTrue(any(e["to"] == "config.py::load_config" for e in scope["historical_edges"]))
        self.assertFalse(any(e["to"] == "config.py::load_config" for e in scope["edges"]))

    def test_scope_budget_no_silent_truncation(self):
        graph = build_graph(self.records)
        scope = query_scope(graph, self.records, "config.py", 6, max_chars=1)
        self.assertTrue(scope["over_budget"])
        self.assertTrue(scope["dialogue"])

    def test_scope_chunks_are_bounded_and_ordered(self):
        graph = build_graph(self.records)
        scope = query_scope(graph, self.records, "config.py", 6)
        chunks = split_scope(scope, max_chars=900, overlap_records=1)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0]["dialogue"][0]["order"], 1)
        self.assertTrue(all(chunk["dialogue"] for chunk in chunks))
        self.assertTrue(all(chunk["chunk_window"][0] <= chunk["chunk_window"][1]
                             for chunk in chunks))
        self.assertEqual(len({tuple(chunk["chunk_window"]) for chunk in chunks}), len(chunks))

    def test_chunk_preserves_multiple_payload_fields(self):
        text = "line\n" * 500
        scope = {"cutoff": 1, "dialogue": [{
            "id": "e1", "order": 1, "kind": "patch", "text": text,
            "changes": {"a.py": {"unified_diff": text, "content": text}},
        }], "events": [], "versions": [], "edges": [], "historical_edges": []}
        chunks = split_scope(scope, max_chars=1800, overlap_records=0)
        text_parts = [r.get("text", "") for c in chunks for r in c["dialogue"]]
        diff_parts = [r.get("changes", {}).get("a.py", {}).get("unified_diff", "")
                      for c in chunks for r in c["dialogue"]]
        content_parts = [r.get("changes", {}).get("a.py", {}).get("content", "")
                         for c in chunks for r in c["dialogue"]]
        self.assertEqual("".join(text_parts), text)
        self.assertEqual("".join(diff_parts), text)
        self.assertEqual("".join(content_parts), text)

    def test_chunk_does_not_repeat_object_index(self):
        scope = {
            "cutoff": 2,
            "dialogue": [
                {"id": "m1", "order": 1, "kind": "message", "role": "user",
                 "text": "change config"},
                {"id": "m2", "order": 2, "kind": "message", "role": "assistant",
                 "text": "done"},
            ],
            "object_index": [
                {"object_id": "config", "object_kind": "file", "normalized_name": "config.py",
                 "message_ids": ["m1"]},
                {"object_id": "unrelated", "object_kind": "file", "normalized_name": "other.py",
                 "message_ids": ["m99"]},
            ],
            "events": [], "versions": [], "edges": [], "historical_edges": [],
        }
        chunks = split_scope(scope, max_chars=900, overlap_records=0)
        self.assertTrue(chunks)
        self.assertTrue(all("object_index" not in chunk for chunk in chunks))
        self.assertTrue(all(chunk["context_chars"] <= 900 for chunk in chunks))

    def test_oversized_multiline_record_is_losslessly_fragmented(self):
        text = "".join("line-%04d value\n" % index for index in range(800))
        scope = {
            "cutoff": 1,
            "dialogue": [{"id": "e1", "order": 1, "kind": "message",
                          "role": "user", "text": text}],
            "events": [], "versions": [], "edges": [], "historical_edges": [],
        }
        chunks = split_scope(scope, max_chars=3000, overlap_records=0)
        fragments = [record for chunk in chunks for record in chunk["dialogue"]]
        self.assertGreater(len(fragments), 1)
        self.assertEqual("".join(record["text"] for record in fragments), text)
        self.assertTrue(all(record["parent_id"] == "e1" for record in fragments))
        self.assertTrue(all(record["fragment"]["complete_record"] is False
                            for record in fragments))
        self.assertTrue(all(not chunk["over_budget"] for chunk in chunks))

    def test_single_oversized_line_is_marked_not_truncated(self):
        text = "x" * 5000
        scope = {
            "cutoff": 1,
            "dialogue": [{"id": "e1", "order": 1, "kind": "result", "text": text}],
            "events": [], "versions": [], "edges": [], "historical_edges": [],
        }
        chunks = split_scope(scope, max_chars=1200, overlap_records=0)
        self.assertEqual(chunks[0]["dialogue"][0]["text"], text)
        self.assertTrue(chunks[0]["over_budget"])

    def test_no_seed_adaptive_search_enumerates_full_event_range(self):
        graph = build_graph(self.records)
        index = build_event_index(self.records, graph)
        scopes, metadata = adaptive_subgraphs(
            graph, self.records, self.records[-1]["order"], seed=None,
            max_chars=24000, beam_width=2, max_depth=3, max_candidates=3)
        self.assertEqual(metadata["enumerated_seed_count"], len(index))
        self.assertLessEqual(len(metadata["seeds"]), len(index))
        self.assertIn(min(index, key=lambda item: item["order"])["id"], metadata["seeds"])
        self.assertIn(max(index, key=lambda item: item["order"])["id"], metadata["seeds"])
        roots = {scope["candidate_seed_event"] for scope in scopes}
        self.assertIn(min(index, key=lambda item: item["order"])["id"], roots)
        self.assertIn(max(index, key=lambda item: item["order"])["id"], roots)

    def test_useful_oversized_adaptive_scope_is_left_for_chunking(self):
        graph = build_graph(self.records)
        scopes, _ = adaptive_subgraphs(
            graph, self.records, self.records[-1]["order"], seed="config.py",
            max_chars=1, beam_width=3, max_depth=3, max_candidates=2)
        self.assertTrue(scopes)
        self.assertTrue(scopes[0]["over_budget"])
        self.assertTrue(split_scope(scopes[0], max_chars=900))

    def test_adaptive_scope_keeps_cross_file_reference_edge(self):
        graph = build_graph(self.records)
        scopes, _ = adaptive_subgraphs(
            graph, self.records, self.records[-1]["order"], seed="main.py",
            max_chars=24000, beam_width=2, max_depth=2, max_candidates=2)
        self.assertTrue(scopes)
        self.assertTrue(any(edge.get("kind") == "call_reference"
                            and edge.get("to") == "config.py::load_config"
                            for scope in scopes for edge in scope.get("edges", [])))

    def test_codex_ignores_reasoning_and_deduplicated_messages(self):
        rows = [(1, {"type": "session_meta", "payload": {"cwd": "/private/project"}}),
                (2, {"type": "response_item", "payload": {"type": "reasoning", "text": "hidden"}}),
                (3, {"type": "event_msg", "payload": {"type": "agent_message", "message": "duplicate"}}),
                (4, {"type": "response_item", "payload": {"type": "message", "role": "assistant",
                                                           "content": [{"text": "visible"}]}})]
        result = normalize_codex(rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "visible")
        self.assertEqual(normalize_path("/private/project/a.py", result[0]["workspace"]), "a.py")


class QualityTests(unittest.TestCase):
    def test_document_source_kind_is_preserved_for_general_scope(self):
        records = [
            {"kind": "message", "role": "user", "text": "项目规范如下", "source_kind": "conversation"},
            {"kind": "message", "role": "assistant", "text": "Memory Index 必须只读", "source_kind": "document"},
        ]
        from dialogue_benchmark.general import build_general_scope
        scope = build_general_scope([
            dict(item, id="e%d" % index, order=index)
            for index, item in enumerate(records, 1)
        ], 2)
        self.assertEqual([item["source_kind"] for item in scope["dialogue"]],
                         ["conversation", "document"])

    def test_tagged_text_response(self):
        parsed = parse_text_response("""FACT f1
SOURCES: e1
TEXT: 旧版本曾使用少量 Memory
END_FACT
""")
        self.assertEqual(parsed["facts"][0]["sources"], ["e1"])
        self.assertEqual(parsed["facts"][0]["statement"], "旧版本曾使用少量 Memory")

    def test_empty_forbidden_marker_is_treated_as_empty_list(self):
        parsed = parse_text_response("""QA q1
TYPE: single-hop
DIFFICULTY: easy
DIFFICULTY_REASON: direct
MEMORY_REQUIREMENT: explicit choice
ANSWER_TARGET: selected configuration format
FACT_IDS: f1
QUESTION: 用户选择了什么？
ANSWER_POINT: 选择 yaml || SOURCES: e1
FORBIDDEN_POINT: 无 || SOURCES:
END_QA
""")
        self.assertEqual(parsed["questions"][0]["forbidden_points"], [])

    def test_fenced_json_response(self):
        self.assertEqual(parse_json_response('```json\n{"reviews": []}\n```'), {"reviews": []})
        with self.assertRaises(ValueError):
            parse_json_response('Some prose {"reviews": []}')

    def setUp(self):
        records = load_dialogue(EXAMPLE)
        self.scope = query_scope(build_graph(records), records, "config.py", 6)
        self.fact = {"id": "f1", "statement": "旧版返回值后来改为异常的 Explicit synthetic fact", "sources": ["e2"]}
        self.question = {"id": "q1", "question": "What changed?", "category": "history_tracking",
                         "difficulty": "medium", "difficulty_reason": "Compare versions",
                         "track": "history_core",
                         "memory_requirement": "Old return behavior", "fact_ids": ["f1"],
                         "use_case": "Developer selects compatibility tests using the previous return behavior",
                         "answer_target": "Old and new return behavior",
                         "answer_points": [{"text": "Return became exception", "sources": ["e2", "e3"]}],
                         "forbidden_points": []}

    def test_unknown_source_rejected(self):
        self.question["answer_points"][0]["sources"] = ["future"]
        accepted, rejected = validate_candidates({"questions": [self.question]}, [self.fact], self.scope)
        self.assertFalse(accepted)
        self.assertEqual(len(rejected), 1)

    def test_one_generation_request_enforces_two_candidate_limit(self):
        from dialogue_benchmark.llm import generate_from_facts
        calls = []
        questions = [dict(self.question, id="q%d" % i) for i in range(3)]
        class Client:
            def ask(self, prompt, data):
                calls.append(data)
                return {"questions": questions}
        result = generate_from_facts(self.scope, [self.fact], Client(), max_questions=2,
                                     generation_mode="legacy")
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(result["questions"]), 2)
        self.assertEqual(result["rejected"][0]["reason"], "question_limit_exceeded")

    def test_graph_sources_are_valid_fact_evidence(self):
        scope = copy.deepcopy(self.scope)
        scope["events"] = [{"id": "event-1", "order": 3}]
        scope["versions"] = [{"id": "version-1", "observed_at": 3}]
        from dialogue_benchmark.quality import validate_facts
        facts, rejected = validate_facts(
            {"facts": [{"id": "f-event", "statement": "event fact", "sources": ["event-1"]},
                       {"id": "f-version", "statement": "version fact", "sources": ["version-1"]}]},
            scope, return_rejected=True)
        self.assertEqual(len(facts), 2)
        self.assertFalse(rejected)

    def test_prevalidation_candidates_keep_stable_identity_and_sources(self):
        from dialogue_benchmark.llm import generate_from_facts
        good = dict(self.question, id="q1")
        bad = dict(self.question, id="q1", difficulty="unsupported")
        saved = {}
        class Client:
            def ask(self, prompt, data):
                return {"questions": [good, bad]}
        scope = dict(self.scope, evidence_group={"id": "code-group-2"})
        result = generate_from_facts(scope, [self.fact], Client(), max_questions=3,
                                     candidate_prefix="code_g2_",
                                     generation_mode="legacy",
                                     checkpoint=lambda n, d: saved.update({n: copy.deepcopy(d)}))
        self.assertEqual([q["id"] for q in result["all_candidates"]], ["code_g2_q1", "code_g2_q2"])
        self.assertEqual(result["rejected"][0]["question"]["id"], "code_g2_q2")
        self.assertEqual(result["all_candidates"][1]["model_id"], "q1")
        self.assertEqual(result["all_candidates"][1]["evidence_group_id"], "code-group-2")
        self.assertEqual(result["all_candidates"][0]["answer_points"], self.question["answer_points"])
        self.assertEqual(len(saved["raw-candidates.json"]["questions"]), 2)

    def test_evidence_projection_keeps_sources_and_shrinks_context(self):
        projected = evidence_projection(self.scope, {"e2"}, padding_records=0)
        self.assertEqual([record["id"] for record in projected["dialogue"]], ["e2"])
        self.assertLess(projected["context_chars"], self.scope["context_chars"])

    def test_missing_review_fails_closed(self):
        kept, rejected = apply_review([self.question], {"reviews": []})
        self.assertEqual(kept[0]["status"], "needs_review")
        self.assertEqual(kept[0]["review_error"], "review_unmatched")
        self.assertFalse(rejected)

    def test_review_aliases_and_omitted_id_match_only_unique_candidate(self):
        checks = ("evidence_supported", "version_consistent", "unambiguous", "category_correct",
                  "difficulty_justified", "not_answer_leaking", "natural_wording",
                  "history_requirement_correct", "type_correct", "practical_useful",
                  "answer_complete", "atomic_points_correct")
        decision = dict.fromkeys(checks, True)
        decision.update(reason="Evidence checked", current_snapshot_alone_sufficient=False,
                        history_evidence_required=True)
        question = dict(self.question, id="code_g3_q1", candidate_id="code_g3_q1", model_id="q1")
        for review_id in ("q1", "code_g3_q1", None, ""):
            kept, rejected = apply_review([question], {"reviews": [dict(decision, id=review_id)]})
            self.assertFalse(rejected)
            self.assertEqual(kept[0]["status"], "approved")
            self.assertEqual(kept[0]["review"]["id"], "code_g3_q1")

    def test_unknown_ambiguous_and_duplicate_review_ids_never_approve(self):
        first = dict(self.question, id="code_g1_q1", model_id="q1")
        second = dict(self.question, id="code_g2_q1", model_id="q1")
        for candidates, reviews, reason in (
                ([first], [{"id": "not-a-candidate"}], "unknown_review_id"),
                ([first, second], [{"id": "q1"}], "unknown_review_id"),
                ([first, second], [{"id": None}], "unknown_review_id"),
                ([first], [{"id": "q1"}, {"id": "code_g1_q1"}], "duplicate_review_id")):
            kept, rejected = apply_review(candidates, {"reviews": reviews})
            self.assertEqual(len(kept), len(candidates))
            self.assertTrue(all(q["status"] == "needs_review" for q in kept))
            self.assertIn(reason, [r["reason"] for r in rejected])

    def test_review_failure_is_isolated_per_candidate(self):
        from dialogue_benchmark.llm import review_candidates, parse_text_response
        payloads = []
        class Client:
            def ask(self, prompt, data):
                payloads.append(data)
                if len(payloads) == 1:
                    raise ValueError("synthetic transport failure")
                return parse_text_response("REVIEW\nevidence_supported: false\nreason: Unsupported\nEND_REVIEW")
        result = review_candidates(self.scope, [self.fact],
                                   [self.question, dict(self.question, id="q2")], Client())
        self.assertEqual(len(payloads), 2)
        self.assertTrue(all(len(p["candidates"]) == 1 for p in payloads))
        self.assertEqual(result["questions"][0]["status"], "needs_review")
        self.assertEqual(result["questions"][1]["id"], "q2")
        self.assertEqual(result["questions"][1]["status"], "needs_review")

    def test_credential_guard(self):
        with self.assertRaises(ValueError):
            outbound_guard('api_key="secret-value-123456789"', "different")
        with self.assertRaises(ValueError):
            outbound_guard("hello configured-secret", "configured-secret")
        with self.assertRaises(ValueError):
            outbound_guard("密码是 hunter2", "different")
        # Empty values, environment references, and documentation placeholders
        # are code examples rather than credentials.
        outbound_guard("token=None API_KEY=$API_KEY password=<placeholder>", "different")

    def test_fake_llm_pipeline(self):
        decision = {"id": "q1", "reason": "Synthetic test review"}
        for key in ("evidence_supported", "version_consistent", "unambiguous", "category_correct",
                    "difficulty_justified", "not_answer_leaking", "natural_wording",
                    "history_requirement_correct", "current_snapshot_alone_sufficient",
                    "history_evidence_required", "type_correct", "practical_useful",
                    "answer_complete", "atomic_points_correct"):
            decision[key] = True
        responses = iter([{"facts": [self.fact]}, {"questions": [self.question]}, {"reviews": [decision]}])
        decision["current_snapshot_alone_sufficient"] = False
        decision.update(
            review_contract="structured_v2", recommended_type="history_tracking",
            recommended_track="history_core", type_basis="历史变更来源",
            necessary_source_ids="e2,e3", necessary_stage_ids="none",
            history_only_fact="旧返回值后来变成异常", history_source_ids="e2,e3",
            useful_task="选择兼容性测试", useful_decision="是否保留旧返回行为",
            answer_effect="决定回归测试覆盖", answer_requirements="R1=返回行为变化",
            requirement_coverage="R1=A1", point_claims="A1.1=Return became exception",
            point_evidence="A1=supported@e2,e3")

        class FakeClient:
            def ask(self, prompt, data):
                return next(responses)

        result = generate(self.scope, FakeClient(), generation_mode="legacy",
                          review_mode="single")
        self.assertEqual(len(result["questions"]), 1)
        self.assertEqual(result["questions"][0]["status"], "approved")

    def test_later_stages_receive_projected_evidence(self):
        responses = iter([{"facts": [self.fact]}, {"questions": [self.question]}, {"reviews": []}])
        payloads = []

        class CapturingClient:
            def ask(self, prompt, data):
                payloads.append(data)
                return next(responses)

        generate(self.scope, CapturingClient(), generation_mode="legacy",
                 review_mode="single")
        self.assertEqual(len(payloads), 3)
        self.assertLessEqual(len(payloads[1]["scope"]["dialogue"]),
                             len(payloads[0]["dialogue"]))
        self.assertLessEqual(len(payloads[2]["scope"]["dialogue"]),
                             len(payloads[0]["dialogue"]))
        # Answer points may cite a directly related source beyond the fact's
        # original citation; review must keep both plus bounded neighbors.
        self.assertTrue({"e2", "e3"}.issubset(
            {record["id"] for record in payloads[2]["scope"]["dialogue"]}))

    def test_qa_failure_preserves_facts(self):
        class FailingQAClient:
            def __init__(self):
                self.calls = 0

            def ask(self, prompt, data):
                self.calls += 1
                if self.calls == 1:
                    return {"facts": [self.fact]}
                raise ValueError("synthetic QA failure")

        client = FailingQAClient()
        client.fact = self.fact
        result = generate(self.scope, client, generation_mode="legacy",
                          review_mode="single")
        self.assertEqual(len(result["facts"]), 1)
        self.assertFalse(result["questions"])
        self.assertEqual(result["stage_status"]["qa"], "failed")

    def test_review_failure_preserves_candidates(self):
        class FailingReviewClient:
            def __init__(self, fact, question):
                self.calls = 0
                self.fact = fact
                self.question = question

            def ask(self, prompt, data):
                self.calls += 1
                if self.calls == 1:
                    return {"facts": [self.fact]}
                if self.calls == 2:
                    return {"questions": [self.question]}
                raise ValueError("synthetic review failure")

        client = FailingReviewClient(self.fact, self.question)
        result = generate(self.scope, client, generation_mode="legacy",
                          review_mode="single")
        self.assertEqual(len(result["questions"]), 1)
        self.assertEqual(result["questions"][0]["status"], "needs_review")
        self.assertEqual(result["stage_status"]["review"], "failed")

    def test_history_diagnostics_are_boolean_not_approval_flags(self):
        checks = ("evidence_supported", "version_consistent", "unambiguous", "category_correct",
                  "difficulty_justified", "not_answer_leaking", "natural_wording",
                  "history_requirement_correct", "type_correct", "practical_useful",
                  "answer_complete", "atomic_points_correct")
        for track, sufficient in (("history_core", False), ("inference_control", True)):
            question = dict(self.question, track=track)
            decision = dict.fromkeys(checks, True)
            decision.update(id="q1", reason="Model decision", current_snapshot_alone_sufficient=sufficient,
                            history_evidence_required=not sufficient)
            kept, _ = apply_review([question], {"reviews": [decision]})
            self.assertEqual(len(kept), 1)
            del decision["history_evidence_required"]
            kept, _ = apply_review([question], {"reviews": [decision]})
            self.assertEqual(kept[0]["status"], "needs_review")

    def test_cli_static_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            args = [str(EXAMPLE), "--output", str(output), "--seed", "config.py"]
            self.assertEqual(main(args), 0)
            result = json.loads((output / "qa.json").read_text())
            self.assertEqual(result["questions"], [])
            self.assertEqual(result["status"], "static_only")
            self.assertEqual(main(args), 1)


if __name__ == "__main__":
    unittest.main()
