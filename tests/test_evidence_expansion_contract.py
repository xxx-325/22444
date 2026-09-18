"""Offline contracts for bounded, object-directed evidence expansion."""

import copy
import unittest
from unittest.mock import patch

import dialogue_benchmark.fact_index as fact_index
from dialogue_benchmark.fact_index import (
    build_evidence_index,
    expand_evidence_group_once,
)


class EvidenceExpansionContractTests(unittest.TestCase):
    @staticmethod
    def _scope():
        return {
            "cutoff": 10,
            "dialogue": [
                {"id": "e0", "order": 0, "kind": "message", "role": "user",
                 "text": "反馈要求 policy.py 约束 runner.py 的 timeout。"},
                {"id": "e1", "order": 1, "kind": "patch", "role": "assistant",
                 "text": "旧版 runner.py 的 _default_runner 使用端口 18080。",
                 "changes": {"runner.py": {"content": "port=18080"}}},
                {"id": "e2", "order": 2, "kind": "patch", "role": "assistant",
                 "call_id": "call-runner", "text":
                 "新版 runner.py 的 _default_runner 使用端口 18081。",
                 "changes": {"runner.py": {"content": "port=18081"}}},
                {"id": "e3", "order": 3, "kind": "call", "call_id": "call-runner",
                 "text": "run runner.py _default_runner"},
                {"id": "e4", "order": 4, "kind": "result", "call_id": "call-runner",
                 "text": "runner.py _default_runner returned timeout; validation passed."},
                {"id": "e5", "order": 5, "kind": "observation", "role": "assistant",
                 "text": "adapter.py calls runner.py _default_runner.",
                 "changes": {"adapter.py": {"content": "call runner"}}},
                {"id": "e6", "order": 6, "kind": "message", "source_kind": "document",
                 "text": "设计文档记录 runner.py 的公开约束。"},
                {"id": "e7", "order": 7, "kind": "result", "source_kind": "tool",
                 "text": "工具输出记录 runner.py 的当前值。"},
                {"id": "e8", "order": 8, "kind": "observation", "source_kind": "code",
                 "text": "代码观察记录 runner.py 的符号。",
                 "changes": {"runner.py": {"content": "observed"}}},
                {"id": "e9", "order": 9, "kind": "result", "source_kind": "test",
                 "text": "测试输出记录 runner.py 的结果。"},
            ],
            "events": [],
            "versions": [
                {"id": "v1", "path": "runner.py", "observed_at": 1,
                 "source": "e1", "previous": None, "status": "known"},
                {"id": "v2", "path": "runner.py", "observed_at": 2,
                 "source": "e2", "previous": "v1", "status": "known"},
            ],
            "edges": [
                {"from": "runner.py::_default_runner", "to": "policy.py::timeout",
                 "relation": "uses"},
                {"from": "adapter.py::handle", "to": "runner.py::_default_runner",
                 "relation": "calls"},
            ],
            "historical_edges": [],
            "stages": [],
        }

    @staticmethod
    def _facts():
        return [
            {"id": "f-current", "statement":
             "新版 runner.py 的 _default_runner 使用端口 18081。",
             "sources": ["e2"]},
            {"id": "f-earlier", "statement":
             "旧版 runner.py 的 _default_runner 使用端口 18080。",
             "sources": ["e1"]},
            {"id": "f-feedback", "statement":
             "反馈要求 policy.py 约束 runner.py 的 timeout。",
             "sources": ["e0"]},
            {"id": "f-outcome", "statement":
             "runner.py _default_runner returned timeout; validation passed.",
             "sources": ["e4"]},
            {"id": "f-dependency", "statement":
             "adapter.py calls runner.py _default_runner.",
             "sources": ["e5"]},
            {"id": "f-document", "statement":
             "设计文档记录 runner.py 的公开约束。",
             "sources": ["e6"]},
            {"id": "f-document-extra", "statement":
             "设计文档还记录 runner.py 的兼容边界。",
             "sources": ["e6"]},
            {"id": "f-tool", "statement":
             "工具输出记录 runner.py 的当前值。",
             "sources": ["e7"]},
            {"id": "f-code", "statement":
             "代码观察记录 runner.py 的符号。",
             "sources": ["e8"]},
            {"id": "f-test", "statement":
             "测试输出记录 runner.py 的结果。",
             "sources": ["e9"]},
        ]

    def _index(self, qa_mode="code"):
        scope = self._scope()
        return build_evidence_index(self._facts(), [scope], qa_mode, 60000)

    @staticmethod
    def _group(index, fact_id, *, qa_mode="code", pointer=None,
               generation_extra_sources=()):
        fact = copy.deepcopy(index["info_by_id"][fact_id]["fact"])
        scope = copy.deepcopy(index["universe"])
        scope["evidence_group"] = {"id": "g-%s" % fact_id}
        if generation_extra_sources:
            scope["generation_extra_sources"] = list(generation_extra_sources)
        group = {
            "id": "g-%s" % fact_id,
            "qa_mode": qa_mode,
            "facts": [fact],
            "scope": scope,
            "relation_path": {},
        }
        if pointer is not None:
            group["expansion_pointer"] = copy.deepcopy(pointer)
        return group

    @staticmethod
    def _entry(index, base_id, missing_kind, object_name=None):
        entries = [entry for entry in index["expansion_candidates"].get(base_id, [])
                   if entry.get("missing_kind") == missing_kind]
        if object_name is not None:
            entries = [entry for entry in entries
                       if object_name in fact_index._entry_objects(
                           dict(entry, base_fact_id=base_id), index)]
        if not entries:
            return None
        entry = copy.deepcopy(entries[0])
        entry["base_fact_id"] = base_id
        return entry

    def test_all_five_missing_kinds_expand_only_the_named_object(self):
        index = self._index()
        cases = [
            ("earlier_state", "f-current", "runner.py"),
            ("later_state", "f-earlier", "runner.py"),
            ("reason", "f-current", "policy.py"),
            ("outcome", "f-current", "runner.py"),
            ("dependency", "f-current", "adapter.py"),
        ]
        for missing_kind, base_id, object_name in cases:
            with self.subTest(missing_kind=missing_kind):
                entry = self._entry(index, base_id, missing_kind, object_name)
                self.assertIsNotNone(entry, "missing explicit %s candidate" % missing_kind)
                group = self._group(
                    index, base_id,
                    pointer={"attempted": False, "candidates": [entry]},
                )
                expanded, audit = expand_evidence_group_once(
                    group, index, missing_kind, object_name,
                    target_chars=60000, max_chars=60000)
                self.assertIsNotNone(expanded)
                self.assertEqual(audit["status"], "expanded")
                self.assertEqual(audit["missing_kind"], missing_kind)
                self.assertIn(base_id, [fact["id"] for fact in expanded["facts"]])

    def test_missing_pointer_is_rebuilt_from_the_shared_index(self):
        index = self._index()
        group = self._group(index, "f-current")

        expanded, audit = expand_evidence_group_once(
            group, index, "earlier_state", "runner.py",
            target_chars=60000, max_chars=60000)

        self.assertIsNotNone(expanded)
        self.assertEqual(audit["status"], "expanded")
        self.assertEqual(expanded["expansion_pointer"]["candidates"], [])

    def test_invalid_or_mismatched_object_does_not_create_pending_work(self):
        index = self._index()
        entry = self._entry(index, "f-current", "earlier_state", "runner.py")
        group = self._group(
            index, "f-current",
            pointer={"attempted": False, "candidates": [entry]},
        )
        before = copy.deepcopy(group)

        expanded, audit = expand_evidence_group_once(
            group, index, "earlier_state", "not-in-materials.py",
            target_chars=60000, max_chars=60000)

        self.assertIsNone(expanded)
        self.assertNotEqual(audit["status"], "expanded")
        self.assertEqual(group, before)
        self.assertNotIn("expansion_pending", audit)

        expanded, audit = expand_evidence_group_once(
            group, index, "not_a_kind", "runner.py",
            target_chars=60000, max_chars=60000)
        self.assertIsNone(expanded)
        self.assertNotIn("expansion_pending", audit)

    def test_over_budget_first_candidate_continues_to_next_fitting_candidate(self):
        index = self._index()
        first = self._entry(index, "f-current", "outcome", "runner.py")
        second = self._entry(index, "f-current", "dependency", "adapter.py")
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        first["relation"] = "over_budget_first"
        second["missing_kind"] = "outcome"
        second["relation"] = "fits_second"
        group = self._group(
            index, "f-current",
            pointer={"attempted": False, "candidates": [first, second]},
        )
        projected = copy.deepcopy(group["scope"])

        with patch.object(
                fact_index, "_group_expansion_pointer",
                return_value={"candidates": [first, second]}), \
                patch.object(
                    fact_index, "_project_group",
                    side_effect=[(None, None, None), (projected, 60000, 0)]) as project:
            expanded, audit = expand_evidence_group_once(
                group, index, "outcome", "runner.py",
                target_chars=60000, max_chars=60000)

        self.assertIsNotNone(expanded)
        self.assertEqual(project.call_count, 2)
        self.assertEqual(audit["status"], "expanded")
        self.assertEqual(audit["relation"], "fits_second")

    def test_expansion_preserves_original_facts_and_generation_extras(self):
        index = self._index()
        entry = self._entry(index, "f-current", "earlier_state", "runner.py")
        group = self._group(
            index, "f-current",
            pointer={"attempted": False, "candidates": [entry]},
            generation_extra_sources=("existing-extra",),
        )

        expanded, _ = expand_evidence_group_once(
            group, index, "earlier_state", "runner.py",
            target_chars=60000, max_chars=60000)

        self.assertIsNotNone(expanded)
        self.assertEqual(
            {fact["id"] for fact in expanded["facts"]}
            & {"f-current"},
            {"f-current"},
        )
        self.assertIn("e2", expanded["facts"][0]["sources"])
        extras = set(expanded["scope"].get("generation_extra_sources", []))
        self.assertIn("existing-extra", extras)
        self.assertTrue(set(group["scope"]["generation_extra_sources"]) <= extras)

    def test_expansion_is_one_shot_and_does_not_make_a_pending_state(self):
        index = self._index()
        entry = self._entry(index, "f-current", "earlier_state", "runner.py")
        group = self._group(
            index, "f-current",
            pointer={"attempted": False, "candidates": [entry]},
        )
        expanded, _ = expand_evidence_group_once(
            group, index, "earlier_state", "runner.py",
            target_chars=60000, max_chars=60000)
        self.assertIsNotNone(expanded)

        with patch.object(
                fact_index, "_group_expansion_pointer",
                return_value={"candidates": [entry]}):
            repeated, audit = expand_evidence_group_once(
                expanded, index, "earlier_state", "runner.py",
                target_chars=60000, max_chars=60000)
        self.assertIsNone(repeated)
        self.assertEqual(audit["reason"], "no_new_evidence")
        self.assertNotIn("expansion_pending", audit)

    def test_general_expansion_accepts_document_but_rejects_tool_code_and_test(self):
        index = self._index("general")
        base = self._group(index, "f-document", qa_mode="general")
        for fact_id, expected in (("f-tool", False), ("f-code", False),
                                  ("f-test", False), ("f-document-extra", True)):
            with self.subTest(fact_id=fact_id):
                entry = {
                    "missing_kind": "outcome",
                    "relation": "declared_source",
                    "fact_ids": [fact_id],
                    "source_ids": list(index["info_by_id"][fact_id]["fact"]["sources"]),
                    "distance": 1,
                    "base_fact_id": "f-document",
                }
                group = copy.deepcopy(base)
                group["expansion_pointer"] = {
                    "attempted": False, "candidates": [entry],
                }
                with patch.object(
                        fact_index, "_group_expansion_pointer",
                        return_value={"candidates": [entry]}):
                    expanded, audit = expand_evidence_group_once(
                        group, index, "outcome", "runner.py",
                        target_chars=60000, max_chars=60000)
                self.assertEqual(expanded is not None, expected)
                if not expected:
                    self.assertNotEqual(audit["status"], "expanded")

    def test_same_file_without_an_explicit_edge_is_not_a_dependency(self):
        scope = self._scope()
        scope["edges"] = []
        facts = [
            {"id": "f-left", "statement": "runner.py handles timeout in alpha.",
             "sources": ["e2"]},
            {"id": "f-right", "statement": "runner.py handles retries in beta.",
             "sources": ["e8"]},
        ]
        index = build_evidence_index(facts, [scope], "code", 60000)
        dependency_entries = [
            entry for entry in index["expansion_candidates"]["f-left"]
            if entry.get("missing_kind") == "dependency"
        ]
        self.assertEqual(dependency_entries, [])

    def test_unresolvable_fragment_candidate_is_skipped_before_next_valid_candidate(self):
        scope = {
            "cutoff": 4,
            "dialogue": [
                {"id": "e-base", "order": 1, "kind": "message",
                 "text": "runner.py current behavior."},
                # A child record may legitimately make its parent citation
                # resolvable, but a concrete fragment ID that is not present
                # must not be replaced with that parent alias.
                {"id": "frag-1", "parent_id": "parent-0", "order": 2,
                 "kind": "message", "text": "runner.py fragment evidence."},
                {"id": "e-valid", "order": 3, "kind": "message",
                 "text": "runner.py valid additional evidence."},
            ],
            "events": [], "versions": [], "edges": [],
            "historical_edges": [], "stages": [],
        }
        facts = [
            {"id": "f-base", "statement": "runner.py current behavior.",
             "sources": ["e-base"]},
            {"id": "f-invalid", "statement":
             "runner.py fragment evidence.",
             # One unresolved source is enough to invalidate this candidate;
             # the existing parent alias must not stand in for this fragment.
             "sources": ["parent-0#fragment-2", "e-valid"]},
            {"id": "f-valid", "statement":
             "runner.py valid additional evidence.",
             "sources": ["e-valid"]},
        ]
        index = build_evidence_index(facts, [scope], "code", 60000)
        group = self._group(
            index, "f-base",
            pointer={
                "attempted": False,
                "candidates": [
                    {"missing_kind": "outcome", "relation": "invalid_fragment",
                     "fact_ids": ["f-invalid"],
                     "source_ids": ["parent-0#fragment-2", "e-valid"],
                     "distance": 1},
                    {"missing_kind": "outcome", "relation": "valid_source",
                     "fact_ids": ["f-valid"], "source_ids": ["e-valid"],
                     "distance": 2},
                ],
            },
        )

        with patch.object(
                fact_index, "_group_expansion_pointer",
                return_value={"candidates": group["expansion_pointer"]["candidates"]}):
            expanded, audit = expand_evidence_group_once(
                group, index, "outcome", "runner.py",
                target_chars=60000, max_chars=60000)

        self.assertIsNotNone(expanded)
        self.assertEqual(audit["relation"], "valid_source")
        expanded_ids = {fact["id"] for fact in expanded["facts"]}
        self.assertIn("f-base", expanded_ids)
        self.assertIn("f-valid", expanded_ids)
        self.assertNotIn("f-invalid", expanded_ids)

    def test_object_and_source_filters_precede_the_candidate_cap(self):
        """A late exact object must survive noisy candidates in the index."""
        records = [
            {"id": "call", "order": 1, "kind": "call", "call_id": "c1",
             "text": "run target.py"},
        ]
        for index in range(13):
            records.append({
                "id": "result-noise-%02d" % index,
                "order": index + 2,
                "kind": "result",
                "call_id": "c1",
                "text": "noise.py result %d" % index,
            })
        records.append({
            "id": "result-target", "order": 20, "kind": "result",
            "call_id": "c1", "text": "target.py result final",
        })
        scope = {
            "cutoff": 30, "dialogue": records, "events": [],
            "versions": [], "edges": [], "historical_edges": [], "stages": [],
        }
        facts = [{
            "id": "f-base", "qa_mode": "code",
            "statement": "target.py call", "sources": ["call"],
        }]
        for index in range(13):
            facts.append({
                "id": "f-noise-%02d" % index, "qa_mode": "code",
                "statement": "noise.py result %d" % index,
                "sources": ["result-noise-%02d" % index],
            })
        facts.append({
            "id": "f-target", "qa_mode": "code",
            "statement": "target.py result final", "sources": ["result-target"],
        })
        index = build_evidence_index(facts, [scope], "code", 60000)
        entries = index["expansion_candidates"]["f-base"]
        self.assertGreaterEqual(len(entries), 14)
        self.assertTrue(any("f-target" in entry.get("fact_ids", [])
                            for entry in entries))

        group = {
            "id": "g-base", "qa_mode": "code", "facts": [facts[0]],
            "scope": index["universe"], "relation_path": {},
        }
        expanded, audit = expand_evidence_group_once(
            group, index, "outcome", "target.py",
            target_chars=60000, max_chars=60000,
        )

        self.assertIsNotNone(expanded)
        self.assertEqual(audit["status"], "expanded")
        self.assertIn("f-target", {fact["id"] for fact in expanded["facts"]})

    def test_qualified_object_combines_one_file_with_its_symbol(self):
        scope = {
            "cutoff": 2,
            "dialogue": [
                {"id": "call", "order": 1, "kind": "call", "call_id": "c1",
                 "text": "run config.py"},
                {"id": "result", "order": 2, "kind": "result", "call_id": "c1",
                 "text": "function load_config returned ok",
                 "changes": {"config.py": {"content": "def load_config"}}},
            ],
            "events": [], "versions": [], "edges": [],
            "historical_edges": [], "stages": [],
        }
        facts = [
            {"id": "f-base", "qa_mode": "code", "statement": "config.py call",
             "sources": ["call"]},
            {"id": "f-target", "qa_mode": "code",
             "statement": "function load_config returned ok", "sources": ["result"]},
        ]
        index = build_evidence_index(facts, [scope], "code", 60000)
        group = {
            "id": "g-base", "qa_mode": "code", "facts": [facts[0]],
            "scope": index["universe"], "relation_path": {},
        }

        expanded, audit = expand_evidence_group_once(
            group, index, "outcome", "config.py::load_config",
            target_chars=60000, max_chars=60000,
        )

        self.assertIsNotNone(expanded)
        self.assertEqual(audit["status"], "expanded")
        self.assertIn("f-target", {fact["id"] for fact in expanded["facts"]})

    def test_qualified_object_does_not_cross_combine_multiple_files(self):
        scope = {
            "cutoff": 2,
            "dialogue": [
                {"id": "call", "order": 1, "kind": "call", "call_id": "c1",
                 "text": "run foo.py"},
                {"id": "result", "order": 2, "kind": "result", "call_id": "c1",
                 "text": "foo.py and bar.py expose funcA and funcB",
                 "changes": {
                     "foo.py": {"content": "funcA"},
                     "bar.py": {"content": "funcB"},
                 }},
            ],
            "events": [], "versions": [], "edges": [],
            "historical_edges": [], "stages": [],
        }
        facts = [
            {"id": "f-base", "qa_mode": "code", "statement": "foo.py call",
             "sources": ["call"]},
            {"id": "f-cross", "qa_mode": "code",
             "statement": "foo.py and bar.py expose funcA and funcB",
             "sources": ["result"]},
        ]
        index = build_evidence_index(facts, [scope], "code", 60000)
        group = {
            "id": "g-base", "qa_mode": "code", "facts": [facts[0]],
            "scope": index["universe"], "relation_path": {},
        }

        expanded, audit = expand_evidence_group_once(
            group, index, "outcome", "foo.py::funcB",
            target_chars=60000, max_chars=60000,
        )

        self.assertIsNone(expanded)
        self.assertEqual(audit["reason"], "no_matching_explicit_relation")


if __name__ == "__main__":
    unittest.main()
