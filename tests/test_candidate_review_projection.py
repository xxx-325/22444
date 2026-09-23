"""Regression tests for candidate-scoped review evidence projections."""

import copy
import unittest

from dialogue_benchmark.fact_index import (
    build_evidence_index,
    candidate_review_projection,
)


class CandidateReviewProjectionTests(unittest.TestCase):
    @staticmethod
    def _scope():
        return {
            "cutoff": 20,
            "dialogue": [
                {"id": "base", "order": 1, "kind": "patch",
                 "changes": {"pkg/a.py": {"content": "load_config old"}},
                 "text": "pkg/a.py::load_config 之前返回 None。"},
                {"id": "a-later", "order": 2, "kind": "patch",
                 "changes": {"pkg/a.py": {"content": "load_config fixed"}},
                 "text": "后来 pkg/a.py::load_config 修正空值处理。"},
                {"id": "a-sibling", "order": 3, "kind": "patch",
                 "changes": {"pkg/a.py": {"content": "save_cache logging"}},
                 "text": "后来 pkg/a.py::save_cache 只修改日志格式。"},
                {"id": "confirm", "order": 4, "kind": "message",
                 "text": "用户确认 pkg/a.py::load_config 的修正保留。"},
                {"id": "call", "order": 5, "kind": "call", "call_id": "call-a",
                 "text": "调用 pkg/a.py::load_config。"},
                {"id": "result", "order": 6, "kind": "result", "call_id": "call-a",
                 "text": "pkg/a.py::load_config 返回成功。"},
                {"id": "mixed", "order": 7, "kind": "patch",
                 "changes": {
                     "pkg/a.py": {"content": "load_config and c"},
                     "pkg/c.py": {"content": "other"},
                 },
                 "text": "pkg/a.py::load_config 与 pkg/c.py::other 一起修改。"},
                {"id": "c-later", "order": 8, "kind": "patch",
                 "changes": {"pkg/c.py": {"content": "other fixed"}},
                 "text": "后来 pkg/c.py::other 修正。"},
                {"id": "b-base", "order": 9, "kind": "patch",
                 "changes": {"pkg/b.py": {"content": "parse old"}},
                 "text": "pkg/b.py::parse 之前返回空值。"},
                {"id": "b-later", "order": 10, "kind": "patch",
                 "changes": {"pkg/b.py": {"content": "parse fixed"}},
                 "text": "后来 pkg/b.py::parse 修正空值处理。"},
            ],
            "events": [],
            "versions": [
                {"id": "a-v1", "path": "pkg/a.py", "observed_at": 1,
                 "source": "base", "previous": None},
                {"id": "a-v2", "path": "pkg/a.py", "observed_at": 2,
                 "source": "a-later", "previous": "a-v1"},
                {"id": "b-v1", "path": "pkg/b.py", "observed_at": 9,
                 "source": "b-base", "previous": None},
                {"id": "b-v2", "path": "pkg/b.py", "observed_at": 10,
                 "source": "b-later", "previous": "b-v1"},
            ],
            "edges": [],
            "historical_edges": [],
            "model_request_chars": 16000,
            "max_context_chars": 16000,
        }

    @staticmethod
    def _facts():
        return [
            {"id": "f-base", "statement": "pkg/a.py::load_config 之前返回 None",
             "sources": ["base"]},
            {"id": "f-a-later", "statement": "后来 pkg/a.py::load_config 修正空值处理",
             "sources": ["a-later"]},
            {"id": "f-a-sibling", "statement": "后来 pkg/a.py::save_cache 只修改日志格式",
             "sources": ["a-sibling"]},
            {"id": "f-confirm", "statement": "用户确认 pkg/a.py::load_config 的修正保留",
             "sources": ["confirm"]},
            {"id": "f-call", "statement": "调用 pkg/a.py::load_config",
             "sources": ["call"]},
            {"id": "f-result", "statement": "pkg/a.py::load_config 返回成功",
             "sources": ["result"]},
            {"id": "f-mixed", "statement": "pkg/a.py::load_config 与 pkg/c.py::other 一起修改",
             "sources": ["mixed"]},
            {"id": "f-c-later", "statement": "后来 pkg/c.py::other 修正",
             "sources": ["c-later"]},
            {"id": "f-b-base", "statement": "pkg/b.py::parse 之前返回空值",
             "sources": ["b-base"]},
            {"id": "f-b-later", "statement": "后来 pkg/b.py::parse 修正空值处理",
             "sources": ["b-later"]},
        ]

    @staticmethod
    def _group(scope, facts):
        return {
            "id": "code-group-1",
            "qa_mode": "code",
            "allowed_types": ("correction_update",),
            "eligible_types": ("correction_update",),
            "facts": copy.deepcopy(facts),
            "scope": copy.deepcopy(scope),
            "review_guard_complete": True,
        }

    @staticmethod
    def _candidate(fact_id, source, extra_fact_ids=(), extra_sources=(),
                   object_name=None):
        object_name = object_name or ("pkg/a.py::load_config" if source == "base"
                                      else "pkg/b.py::parse")
        fact_ids = [fact_id, *extra_fact_ids]
        sources = [source, *extra_sources]
        return {
            "id": "q-" + fact_id,
            "candidate_id": "q-" + fact_id,
            "qa_mode": "code",
            "type": "correction_update",
            "fact_ids": fact_ids,
            "question": "%s 的记录变化是什么？" % object_name,
            "answer_points": [{"text": "%s 的记录变化" % object_name,
                                "sources": sources}],
            "forbidden_points": [],
        }

    def _project(self, candidate, target_chars=16000, max_chars=None):
        scope = self._scope()
        facts = self._facts()
        index = build_evidence_index(facts, [scope], "code", 16000)
        candidate_ids = set(candidate.get("fact_ids", []))
        group = self._group(
            scope, [fact for fact in facts if fact["id"] in candidate_ids])
        return candidate_review_projection(
            group, index, candidate, target_chars=target_chars, max_chars=max_chars)

    @staticmethod
    def _scope_ids(projected, field="dialogue"):
        return {item["id"] for item in projected["scope"].get(field, [])
                if isinstance(item, dict) and isinstance(item.get("id"), str)}

    def test_same_object_revision_confirmation_and_call_result_are_retained(self):
        projected, audit = self._project(self._candidate(
            "f-base", "base", extra_fact_ids=("f-call",),
            extra_sources=("call",)))

        self.assertTrue(audit["complete"])
        dialogue_ids = self._scope_ids(projected)
        version_ids = self._scope_ids(projected, "versions")
        self.assertTrue({"base", "a-later", "confirm", "call", "result"}
                        <= dialogue_ids)
        self.assertTrue({"a-v1", "a-v2"} <= version_ids)
        self.assertNotIn("a-sibling", dialogue_ids)

    def test_mixed_source_does_not_promote_unmentioned_file_to_an_anchor(self):
        projected, audit = self._project(self._candidate("f-base", "base"))

        self.assertTrue(audit["complete"])
        dialogue_ids = self._scope_ids(projected)
        self.assertIn("mixed", dialogue_ids)
        self.assertNotIn("c-later", dialogue_ids)
        self.assertNotIn("a-sibling", dialogue_ids)

    def test_candidate_change_recomputes_anchors_without_stale_a_facts(self):
        projected_a, audit_a = self._project(self._candidate("f-base", "base"))
        projected_b, audit_b = self._project(
            self._candidate("f-b-base", "b-base"))

        self.assertTrue(audit_a["complete"])
        self.assertTrue(audit_b["complete"])
        ids_a = self._scope_ids(projected_a)
        ids_b = self._scope_ids(projected_b)
        self.assertIn("a-later", ids_a)
        self.assertNotIn("a-later", ids_b)
        self.assertIn("b-later", ids_b)
        self.assertNotIn("b-later", ids_a)
        self.assertNotEqual(audit_a["base_fact_ids"], audit_b["base_fact_ids"])

    def test_over_budget_guard_cannot_report_complete(self):
        scope = self._scope()
        scope["dialogue"][1]["text"] += " x" * 20000
        facts = self._facts()
        index = build_evidence_index(facts, [scope], "code", 900)
        group = self._group(scope, [facts[0]])
        projected, audit = candidate_review_projection(
            group, index, self._candidate("f-base", "base"),
            target_chars=600, max_chars=900)

        self.assertFalse(audit["complete"])
        self.assertIn("over_budget", audit["reason"])
        self.assertIsNone(projected)

    def test_historical_failure_keeps_later_same_object_counterevidence(self):
        scope = {
            "cutoff": 4,
            "dialogue": [
                {"id": "failure", "order": 1, "kind": "result",
                 "text": "pkg/a.py::load_config 实际运行失败。"},
                {"id": "correction", "order": 2, "kind": "message",
                 "text": "此前测试结果误报，已撤回 pkg/a.py::load_config 的失败结论。"},
            ],
            "events": [], "versions": [], "edges": [], "historical_edges": [],
            "model_request_chars": 16000,
            "max_context_chars": 16000,
        }
        facts = [
            {"id": "f-failure", "statement": "pkg/a.py::load_config 实际运行失败",
             "sources": ["failure"]},
            {"id": "f-correction", "statement":
             "此前测试结果误报，已撤回 pkg/a.py::load_config 的失败结论",
             "sources": ["correction"]},
        ]
        index = build_evidence_index(facts, [scope], "code", 16000)
        candidate = {
            "id": "q-failure", "candidate_id": "q-failure",
            "qa_mode": "code", "type": "failure_avoidance",
            "fact_ids": ["f-failure"],
            "question": "当时在 pkg/a.py::load_config 失败中，失败机制是什么？",
            "answer_points": [{"text": "记录的失败机制", "sources": ["failure"]}],
            "forbidden_points": [],
        }
        projected, audit = candidate_review_projection(
            self._group(scope, [facts[0]]), index, candidate)

        self.assertTrue(audit["complete"])
        self.assertIn("correction", self._scope_ids(projected))
        self.assertIn("correction", projected["scope"]["review_guard_sources"])

    def test_cutoff_out_candidate_source_fails_but_mixed_fact_keeps_legal_source(self):
        scope = {
            "cutoff": 2,
            "dialogue": [
                {"id": "legal", "order": 1, "kind": "message",
                 "text": "pkg/a.py::load_config 的约束已记录。"},
                {"id": "future", "order": 3, "kind": "patch",
                 "text": "pkg/a.py::load_config 后来又修改。"},
            ],
            "events": [], "versions": [], "edges": [], "historical_edges": [],
            "model_request_chars": 16000,
            "max_context_chars": 16000,
        }
        fact = {"id": "f-cross", "statement": "pkg/a.py::load_config 的约束",
                "sources": ["legal", "future"]}
        index = build_evidence_index([fact], [scope], "code", 16000)
        group = self._group(scope, [fact])
        legal_candidate = self._candidate(
            "f-cross", "legal", object_name="pkg/a.py::load_config")
        projected, audit = candidate_review_projection(
            group, index, legal_candidate)

        self.assertTrue(audit["complete"])
        self.assertIn("legal", self._scope_ids(projected))
        self.assertNotIn("future", self._scope_ids(projected))

        late_candidate = copy.deepcopy(legal_candidate)
        late_candidate["answer_points"][0]["sources"] = ["future"]
        rejected, late_audit = candidate_review_projection(
            group, index, late_candidate)

        self.assertIsNone(rejected)
        self.assertFalse(late_audit["complete"])
        self.assertEqual(late_audit["reason"], "candidate_source_after_cutoff")

    def test_same_property_in_another_file_is_not_counterevidence(self):
        scope = {
            "cutoff": 3,
            "dialogue": [
                {"id": "a", "order": 1, "kind": "patch",
                 "changes": {"src/a.py": {"content": "timeout_seconds = 30"}},
                 "text": "src/a.py 使用 timeout_seconds。"},
                {"id": "b", "order": 2, "kind": "patch",
                 "changes": {"src/b.py": {"content": "timeout_seconds = 60"}},
                 "text": "后来 src/b.py 修改 timeout_seconds。"},
            ],
            "events": [], "versions": [], "edges": [],
            "historical_edges": [], "model_request_chars": 16000,
            "max_context_chars": 16000,
        }
        facts = [
            {"id": "f-a", "statement": "src/a.py 使用 timeout_seconds",
             "sources": ["a"]},
            {"id": "f-b", "statement": "后来 src/b.py 修改 timeout_seconds",
             "sources": ["b"]},
        ]
        index = build_evidence_index(facts, [scope], "code", 16000)
        candidate = {
            "id": "q1", "qa_mode": "code", "type": "compatibility_preservation",
            "question": "src/a.py 中 timeout_seconds 如何控制执行？",
            "answer_points": [{"text": "src/a.py 使用 timeout_seconds。",
                               "sources": ["a"]}],
            "forbidden_points": [],
        }
        projected, audit = candidate_review_projection(
            self._group(scope, [facts[0]]), index, candidate)

        self.assertTrue(audit["complete"])
        self.assertIn("a", self._scope_ids(projected))
        self.assertNotIn("b", self._scope_ids(projected))


if __name__ == "__main__":
    unittest.main()
