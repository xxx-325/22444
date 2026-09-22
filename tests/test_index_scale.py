import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dialogue_benchmark.cli import _run_fact_tasks
from dialogue_benchmark.fact_index import _ExpansionCandidates, _initial_neighbor_lookup, _project_group
from dialogue_benchmark.task_eval.artifacts import save
from dialogue_benchmark.llm import _metadata_token_present, simple_evidence_payload


class IndexScaleTests(unittest.TestCase):
    def test_fact_anchored_reading_uses_excerpts_but_full_range_keeps_original(self):
        body = "unrelated = 0\n" * 1000 + "def important_function():\n    return 2\n" + "unrelated = 0\n" * 1000
        scope = {"dialogue": [{"id": "e1", "order": 1, "kind": "observation", "path": "a.py", "content": body}]}
        facts = [{"id": "f1", "statement": "important_function changed behavior", "sources": ["e1"]}]
        import json
        excerpt, sources = simple_evidence_payload(scope, {"e1"}, facts=facts)
        complete, _ = simple_evidence_payload(dict(scope, full_range_required=True), {"e1"}, facts=facts)
        self.assertLess(len(json.dumps(excerpt)), len(body) // 2)
        self.assertIn("def important_function", json.dumps(excerpt))
        self.assertIn(body, [r.get("content") for m in complete["materials"] for r in m["original_records"]])
        self.assertEqual(set(sources.values()), {"e1"})
        self.assertEqual(scope["dialogue"][0]["content"], body)

    def test_absent_metadata_does_not_compile_thousands_of_regexes(self):
        with patch("dialogue_benchmark.llm.re.search") as search:
            for index in range(2000):
                self.assertFalse(_metadata_token_present("The historical behavior changed", "e%d" % index))
            search.assert_not_called()
        self.assertFalse(_metadata_token_present("some1value", "e1"))
        self.assertTrue(_metadata_token_present("See e1.", "e1"))

    def test_simple_budget_measures_reading_payload_not_internal_graph(self):
        scope = {"dialogue": [{"id": "e1", "order": 1, "kind": "message", "role": "user",
                               "text": "Keep the earlier compatibility requirement."}],
                 "events": [{"id": "e1", "order": 1, "kind": "message"}], "versions": [],
                 "historical_edges": [{"from": "module.py", "to": "module.py::f%d" % i,
                                       "source": "e1", "kind": "contains"} for i in range(1000)]}
        infos = [{"fact": {"id": "f1", "statement": "Keep earlier compatibility", "sources": ["e1"]}}]
        self.assertIsNone(_project_group(scope, infos, 16000, 32000)[0])
        projected, _, _ = _project_group(scope, infos, 16000, 32000, readable_budget=True)
        self.assertIsNotNone(projected)
        self.assertEqual(len(projected["historical_edges"]), 1000)

    def test_initial_neighbors_are_bounded_and_include_distant_object(self):
        infos = [{"fact": {"id": "f%d" % i, "sources": ["s%d" % i]},
                  "entities": {"shared"}, "paths": {"a.py"}, "graph_neighbors": set()}
                 for i in range(2000)]
        neighbors = _initial_neighbor_lookup(infos)(infos[0])
        self.assertLessEqual(len(neighbors), 128)
        self.assertIn(1999, neighbors)

    def test_expansion_is_computed_only_for_requested_fact(self):
        a, b = {"fact": {"id": "a"}}, {"fact": {"id": "b"}}
        index = {"infos": [a, b], "info_by_id": {"a": a, "b": b}}
        lazy = _ExpansionCandidates(index)
        with patch("dialogue_benchmark.fact_index._build_expansion_candidates", return_value={"a": []}) as build:
            self.assertEqual(len(lazy), 0)
            self.assertEqual(lazy.get("a"), [])
            self.assertEqual(lazy["a"], [])
            self.assertIsNone(lazy.get("missing"))
            build.assert_called_once_with(index, [a])
        self.assertNotIn("b", lazy)

    def test_reuse_saved_facts_and_failures_without_model_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save(root / "old/stages/code-chunk-0000-facts.json",
                 [{"id": "f1", "statement": "Behavior changed", "sources": ["e1"]}])
            save(root / "old/stages/code-chunk-0001-facts-error.json",
                 {"stage": "facts", "error_code": "credential_guard"})
            stage = root / "new/stages"
            stage.mkdir(parents=True)
            tasks = [(i, "code", {"scope_index": 0, "chunk_index": i}) for i in range(2)]
            with patch("dialogue_benchmark.cli.ChatClient") as client:
                result = _run_fact_tasks(tasks, "unused", "unused", "unused", 2, stage, root / "old")
                client.assert_not_called()
            self.assertEqual(result["facts"][0]["id"], "code_s0_c0_f1")
            self.assertEqual(result["stage_errors"][0]["error_code"], "credential_guard")
            self.assertEqual(result["usage"], [])
