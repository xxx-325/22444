"""Offline contracts for the model-facing evidence reading view.

The model should receive source-local references and source material, while
the original evidence IDs and authoring metadata remain in a private sidecar.
These tests deliberately use synthetic IDs and strings; no real transcript or
provider response is needed.
"""

import copy
import io
import json
import unittest
from unittest.mock import patch

from dialogue_benchmark.llm import (
    ChatClient,
    SIMPLE_QA_PROMPT,
    _UNKNOWN_LOCAL_REFERENCE,
    _restore_local_sources,
    evidence_projection,
    generate_from_facts,
    request_size,
    review_candidates,
    simple_evidence_payload,
)
from dialogue_benchmark.quality import validate_simple_candidates
from tests.simple_test_helpers import maybe_focus_response, maybe_relevance_response


class CapturingClient:
    """Return fixed protocol documents and retain only safe request payloads."""

    def __init__(self):
        self.calls = []
        self.stages = []

    def ask(self, prompt, data):
        self.calls.append((prompt, copy.deepcopy(data)))
        index = len(self.calls)
        self.stages.append(("focus", "qa", "review_code_distinctiveness",
                            "review_relevance", "review_atomicity",
                            "review_completeness", "review_evidence")
                           [min(index - 1, 6)])
        focus = maybe_focus_response(prompt, data)
        if focus is not None:
            return focus
        relevance = maybe_relevance_response(prompt, data)
        if relevance is not None:
            return relevance
        if index == 2:
            return {
                "questions": [{
                    "id": "q1",
                    "question": "load_config 的超时测试应保留什么行为？",
                    "answer_points": [{
                        "text": "保留 timeout 后不发布半成品目录的行为。",
                        "sources": ["资料1"],
                    }],
                    "forbidden_points": [],
                }],
            }
        if index == 3:
            return {"reviews": [{
                "id": "q1", "review_contract": "code_distinctiveness_v1",
                "answer_basis": "A", "target_alignment": "aligned",
            }]}
        if index == 5:
            return {"reviews": [{
                "id": "q1", "review_contract": "simple_atomicity_v1",
                "point_atomicity": "A1=single",
            }]}
        if index == 6:
            return {"reviews": [{
                "id": "q1", "review_contract": "simple_v1",
                "completeness": "complete",
            }]}
        return {"reviews": [{
            "id": "q1", "review_contract": "simple_v1",
            "point_evidence": "A1=supported@资料1",
        }]}


class ModelEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.scope = {
            "cutoff": 8,
            "qa_mode": "code",
            "model_request_chars": 50000,
            "max_context_chars": 50000,
            "seed": "packages/memorax/config.py",
            "evidence_group": {
                "id": "code-group-secret-36",
                "target_types": ["history_tracking"],
                "budget_hash": "budget-hash-private-36",
                "constraints": {"max_questions": 1, "target_type": "history_tracking"},
            },
            "projection_hash": "projection-private-hash",
            "source_index": {"m-1": {"order": 1}},
            "fact_index": {"fact-secret-777": {"sources": ["m-1"]}},
            "dialogue": [
                {
                    "id": "m-1",
                    "kind": "message",
                    "role": "user",
                    "order": 1,
                    "stage_id": "stage-private-1",
                    "text": (
                        "实际代码位于 packages/memorax/config.py，函数 "
                        "load_config；测试原文是 pytest -q "
                        "tests/test_config.py::test_timeout。"
                    ),
                },
                {
                    "id": "call-2",
                    "kind": "call",
                    "role": "assistant",
                    "order": 2,
                    "call_id": "rpc-private-2",
                    "content": "pytest -q tests/test_config.py::test_timeout",
                },
                {
                    "id": "result-3",
                    "kind": "result",
                    "role": "tool",
                    "order": 3,
                    "call_id": "rpc-private-2",
                    "content": "stdout: status=200 business_id=job-42 hash=abc123",
                },
                {
                    "id": "m-4",
                    "kind": "message",
                    "role": "assistant",
                    "order": 4,
                    "stage_id": "stage-private-2",
                    "text": "后来更新仍需作为反证检查，不能只看当前快照。",
                },
            ],
            "events": [
                {
                    "id": "event-old",
                    "kind": "patch",
                    "order": 5,
                    "affected_paths": ["packages/memorax/config.py"],
                    "success": True,
                    "changes": {
                        "packages/memorax/config.py": {
                            "unified_diff": "- return None\n+ return timeout_result",
                        },
                    },
                },
                {
                    "id": "event-new",
                    "kind": "patch",
                    "order": 6,
                    "affected_paths": ["packages/memorax/config.py"],
                    "success": True,
                    "changes": {
                        "packages/memorax/config.py": {
                            "unified_diff": "- publish(tmp)\n+ publish(final)",
                        },
                    },
                },
            ],
            "versions": [
                {
                    "id": "version-old",
                    "observed_at": 5,
                    "path": "packages/memorax/config.py",
                    "source": "event-old",
                    "previous": None,
                },
                {
                    "id": "version-new",
                    "observed_at": 7,
                    "path": "packages/memorax/config.py",
                    "source": "event-new",
                    "previous": "version-old",
                },
            ],
            "edges": [{
                "from": "packages/memorax/config.py::load_config",
                "to": "packages/memorax/config.py::publish",
                "source": "event-new",
            }],
            "historical_edges": [],
        }
        self.facts = [{
            "id": "fact-secret-777",
            "statement": (
                "load_config 在 m-1 中约定超时目录行为，event-new 对应后续更新；"
                "必须检查后来更新。"
            ),
            "sources": ["m-1", "event-new", "version-new", "m-4"],
        }]
        self.source_ids = {
            "m-1", "call-2", "result-3", "m-4",
            "event-old", "event-new", "version-old", "version-new",
        }

    def test_model_payload_filters_wrappers_but_preserves_source_text(self):
        payload, ref_to_source = simple_evidence_payload(
        self.scope, self.source_ids, facts=self.facts)
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)

        def strings(value):
            if isinstance(value, dict):
                for item in value.values():
                    yield from strings(item)
            elif isinstance(value, list):
                for item in value:
                    yield from strings(item)
            elif isinstance(value, str):
                yield value

        source_text = "\n".join(strings(payload))

        # Authoring IDs, group/constraint metadata, and budget bookkeeping stay
        # in the sidecar or local caller state, never in the model payload.
        for forbidden in (
                "code-group-secret-36", "fact-secret-777", "event-old", "event-new",
                "version-old", "version-new", "stage-private-1", "stage-private-2",
                "budget-hash-private-36", "projection-private-hash", "model_request_chars",
                "max_context_chars", "source_index", "fact_index", "constraints"):
            self.assertNotIn(forbidden, serialized)
        self.assertNotIn("evidence_group", payload)
        self.assertNotIn("id", payload)

        # Source material is not summarized away: paths, function names, exact
        # test invocation, stdout, business status, and diff text survive.
        for literal in (
                "packages/memorax/config.py", "load_config",
                "pytest -q tests/test_config.py::test_timeout",
                "stdout: status=200 business_id=job-42 hash=abc123",
                "- publish(tmp)\n+ publish(final)",
                "后来更新仍需作为反证检查"):
            self.assertIn(literal, source_text)

        self.assertTrue(ref_to_source)
        self.assertEqual(set(ref_to_source), {
            item["reference"] for item in payload["materials"]})
        self.assertTrue(all(reference.startswith("资料")
                            for reference in ref_to_source))

    def test_local_source_round_trip_and_unknown_reference_fail_closed(self):
        payload, ref_to_source = simple_evidence_payload(
            self.scope, self.source_ids, facts=self.facts)
        by_source = {source: reference for reference, source in ref_to_source.items()}
        self.assertIn("m-1", by_source)
        response = {"questions": [{
            "id": "q1",
            "answer_points": [{"text": "原文", "sources": [by_source["m-1"]]}],
            "forbidden_points": [],
        }]}
        restored = _restore_local_sources(response, ref_to_source)
        self.assertEqual(
            restored["questions"][0]["answer_points"][0]["sources"], ["m-1"])

        unknown = copy.deepcopy(response)
        unknown["questions"][0]["question"] = "原文？"
        unknown["questions"][0]["answer_points"][0]["sources"] = ["资料999"]
        restored_unknown = _restore_local_sources(unknown, ref_to_source)
        self.assertEqual(
            restored_unknown["questions"][0]["answer_points"][0]["sources"],
            [_UNKNOWN_LOCAL_REFERENCE])
        accepted, rejected = validate_simple_candidates(
            restored_unknown, self.facts, self.scope, qa_mode="code")
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]["reason"], "invalid_answer_evidence")

    def test_explicit_relations_are_readable_without_chronology_causality(self):
        payload, ref_to_source = simple_evidence_payload(
            self.scope, self.source_ids, facts=self.facts)
        relations = payload["relations"]
        relation_text = "\n".join(relations)
        old_ref = next(ref for ref, source in ref_to_source.items()
                       if source == "version-old")
        new_ref = next(ref for ref, source in ref_to_source.items()
                       if source == "version-new")
        call_ref = next(ref for ref, source in ref_to_source.items()
                        if source == "call-2")
        result_ref = next(ref for ref, source in ref_to_source.items()
                          if source == "result-3")
        self.assertIn("后续版本", relation_text)
        self.assertIn(old_ref, relation_text)
        self.assertIn(new_ref, relation_text)
        self.assertIn("工具调用", relation_text)
        self.assertIn(call_ref, relation_text)
        self.assertIn(result_ref, relation_text)
        self.assertNotIn("因为时间先后", relation_text)
        self.assertNotIn("相邻所以", relation_text)

    def test_model_view_is_smaller_than_legacy_projection_offline(self):
        prompt = SIMPLE_QA_PROMPT
        old = {"facts": self.facts,
               "scope": evidence_projection(self.scope, self.source_ids,
                                              padding_records=0)}
        new, _ = simple_evidence_payload(self.scope, self.source_ids, facts=self.facts)
        old_chars = request_size(prompt, old)
        new_chars = request_size(prompt, new)
        self.assertLess(new_chars, old_chars)
        self.assertGreater(old_chars - new_chars, 0)

    def test_simple_generation_and_review_use_model_view_for_all_five_review_calls(self):
        client = CapturingClient()
        generated = generate_from_facts(
            self.scope, self.facts, client, qa_mode="code",
            allowed_types={"history_tracking"}, target_type="history_tracking",
            generation_mode="simple")
        self.assertEqual(generated["stage_status"]["qa"], "completed")
        self.assertEqual(len(client.calls), 2)
        reviewed = review_candidates(
            self.scope, self.facts, generated["questions"], client,
            qa_mode="code", review_mode="simple", allow_repair=False)
        self.assertEqual(reviewed["stage_status"]["review"], "completed")
        self.assertEqual(len(client.calls), 7)
        self.assertEqual(client.stages,
                         ["focus", "qa", "review_code_distinctiveness",
                          "review_relevance", "review_atomicity",
                          "review_completeness", "review_evidence"])
        distinctive_prompt, distinctive_request = client.calls[2]
        self.assertIn("code_distinctiveness_v1", distinctive_prompt)
        self.assertIn("answer_basis: A|B|C|D", distinctive_prompt)
        self.assertIn("materials", distinctive_request)
        self.assertIn("relations", distinctive_request)
        atomic_prompt = client.calls[4][0]
        self.assertIn(
            "point_atomicity: A1=<single|compound|uncertain>",
            atomic_prompt)
        self.assertNotIn("A2=<single|compound|uncertain>", atomic_prompt)
        evidence_prompt = client.calls[6][0]
        self.assertIn(
            "point_evidence: A1=STATUS",
            evidence_prompt)
        self.assertNotIn(
            "A2=STATUS", evidence_prompt)
        self.assertIn("A2=supported@资料1,资料2", evidence_prompt)
        self.assertIn("A2=insufficient", evidence_prompt)
        self.assertNotIn("[@资料", evidence_prompt)
        for _, request in client.calls:
            serialized = json.dumps(request, ensure_ascii=False, sort_keys=True)
            self.assertNotIn("code-group-secret-36", serialized)
            self.assertNotIn("fact-secret-777", serialized)
            self.assertNotIn("event-old", serialized)
            self.assertNotIn("version-new", serialized)
            self.assertNotIn("model_request_chars", serialized)
            self.assertNotIn("max_context_chars", serialized)
        evidence_request = client.calls[6][1]
        point = evidence_request["candidates"][0]["answer_points"][0]
        self.assertEqual(point["sources"], ["资料1"])
        self.assertIn("后来更新仍需作为反证检查",
                      json.dumps(evidence_request, ensure_ascii=False))

    def test_chat_client_serializes_only_the_model_view(self):
        payload, _ = simple_evidence_payload(
            self.scope, self.source_ids, facts=self.facts)
        response = {
            "choices": [{"finish_reason": "stop",
                         "message": {"content": "NO_QA"}}],
        }
        with patch.dict("os.environ", {"BENCHMARK_API_KEY": "test-placeholder-key"}), \
                patch("dialogue_benchmark.llm.urllib.request.build_opener") as opener:
            opener.return_value.open.return_value = io.BytesIO(
                json.dumps(response).encode())
            client = ChatClient("https://example.invalid/chat/completions", "deepseek-chat")
            client.ask("synthetic prompt", payload)
            request = opener.return_value.open.call_args[0][0]
        body = json.loads(request.data)
        sent_text = body["messages"][1]["content"]
        sent = json.loads(sent_text.split("\nDATA:\n", 1)[1])
        sent_text = json.dumps(sent, ensure_ascii=False, sort_keys=True)
        self.assertNotIn("code-group-secret-36", sent_text)
        self.assertNotIn("fact-secret-777", sent_text)
        self.assertNotIn("event-old", sent_text)
        self.assertNotIn("version-new", sent_text)
        self.assertNotIn("budget-hash-private-36", sent_text)
        self.assertNotIn("model_request_chars", sent_text)
        self.assertNotIn("max_context_chars", sent_text)
        self.assertIn("packages/memorax/config.py", sent_text)
        self.assertIn("pytest -q tests/test_config.py::test_timeout", sent_text)

    def test_legacy_generation_retains_legacy_payload_contract(self):
        class LegacyClient:
            def __init__(self):
                self.payloads = []

            def ask(self, prompt, data):
                self.payloads.append(copy.deepcopy(data))
                return {"questions": []}

        client = LegacyClient()
        result = generate_from_facts(
            self.scope, self.facts, client, qa_mode="code",
            allowed_types={"history_tracking"}, generation_mode="legacy",
            max_questions=1)
        self.assertEqual(result["stage_status"]["qa"], "completed")
        self.assertEqual(len(client.payloads), 1)
        legacy_scope = client.payloads[0]["scope"]
        self.assertIn("dialogue", legacy_scope)
        self.assertTrue(any(record.get("id") == "m-1"
                            for record in legacy_scope["dialogue"]))
        self.assertIn("model_request_chars", legacy_scope)

    def test_candidate_review_projects_all_object_hits_from_large_material(self):
        filler = "unrelated implementation detail\n" * 300
        source = (filler + "def timeout_seconds():\n    return 30\n" + filler
                  + "subprocess.run(timeout=timeout_seconds)\n" + filler)
        scope = {
            "dialogue": [{"id": "large", "order": 1, "kind": "message",
                          "role": "assistant", "text": source}],
            "events": [], "versions": [], "edges": [],
            "historical_edges": [],
        }
        candidate = {
            "id": "q1", "question": "timeout_seconds 如何传到 subprocess.run？",
            "answer_points": [{
                "text": "subprocess.run 使用 timeout_seconds。",
                "sources": ["large"],
            }],
            "forbidden_points": [],
        }

        payload, mapping = simple_evidence_payload(
            scope, ["large"], candidate=candidate)
        rendered = json.dumps(payload, ensure_ascii=False)

        self.assertEqual(mapping, {"资料1": "large"})
        self.assertEqual(rendered.count("timeout_seconds"), 4)
        self.assertIn("subprocess.run", rendered)
        self.assertIn("与候选对象无关的原文已省略", rendered)
        self.assertLess(len(rendered), len(source))


if __name__ == "__main__":
    unittest.main()
