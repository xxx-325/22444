"""Offline contracts for smoke's complete-pool, selected-group execution."""

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


EXAMPLE = (Path(__file__).resolve().parents[1] / "examples"
           / "smoke_generation.py")
SPEC = importlib.util.spec_from_file_location("smoke_generation", EXAMPLE)
smoke_generation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke_generation)


class SmokeSharedIndexContractTests(unittest.TestCase):
    @staticmethod
    def _scope(prefix, text):
        return {
            "cutoff": 1,
            "dialogue": [{"id": prefix, "order": 1, "kind": "message",
                          "role": "assistant", "text": text}],
            "events": [], "versions": [], "edges": [],
            "historical_edges": [], "stages": [],
        }

    def _bundle(self):
        selected_scope = self._scope("e-selected", "selected material")
        outside_scope = self._scope("e-outside", "outside material")
        return {
            "schema_version": 1,
            "cases": [],
            "groups": {
                "general-selected": {
                    "qa_mode": "general",
                    "eligible_types": ["single-hop"],
                    "facts": [{"id": "f-selected", "statement":
                               "selected material", "sources": ["e-selected"]}],
                    "scope": selected_scope,
                },
                "general-other": {
                    "qa_mode": "general",
                    "eligible_types": ["single-hop"],
                    "facts": [{"id": "f-outside", "statement":
                               "outside material", "sources": ["e-outside"]}],
                    "scope": outside_scope,
                },
            },
            "evidence_pool": {
                "general": {
                    "facts": [
                        {"id": "f-selected", "statement": "selected material",
                         "sources": ["e-selected"]},
                        {"id": "f-outside", "statement": "outside material",
                         "sources": ["e-outside"]},
                        {"id": "f-unresolved", "statement": "missing material",
                         "sources": ["missing-fragment"]},
                    ],
                    "scopes": [selected_scope, outside_scope],
                },
                "code": {"facts": [], "scopes": []},
            },
        }

    def test_complete_pool_index_includes_unselected_group_and_drops_unresolved_fact(self):
        indexes, metadata = smoke_generation.build_shared_evidence_indexes(
            self._bundle())

        general = indexes["general"]
        self.assertIn("f-selected", general["info_by_id"])
        self.assertIn("f-outside", general["info_by_id"])
        self.assertNotIn("f-unresolved", general["info_by_id"])
        self.assertIn(
            "e-outside",
            {record["id"] for record in general["universe"]["dialogue"]},
        )
        self.assertEqual(metadata["general"]["scope"], "saved_extraction_pool")
        self.assertEqual(metadata["general"]["_unresolved_fact_ids"],
                         {"f-unresolved"})

    def test_selected_execution_receives_one_track_index_built_from_complete_pool(self):
        bundle = self._bundle()
        indexes, metadata = smoke_generation.build_shared_evidence_indexes(bundle)
        selected = [("general", "general-selected",
                     copy.deepcopy(bundle["groups"]["general-selected"]))]
        observed = []

        def fake_run(*args):
            # _execute_groups passes the shared index and its metadata after
            # the per-group arguments; no provider call is made here.
            observed.append((args[1], args[9], args[10]))
            return {"group_id": args[1], "public_questions": []}, []

        with patch.object(smoke_generation, "_run_group", fake_run):
            summaries, usage = smoke_generation._execute_groups(
                selected, "endpoint", "model", "KEY", 1,
                Path("unused"), "simple", indexes, metadata)

        self.assertEqual([item["group_id"] for item in summaries],
                         ["general-selected"])
        self.assertEqual(usage, [])
        self.assertEqual(len(observed), 1)
        group_id, index, index_metadata = observed[0]
        self.assertEqual(group_id, "general-selected")
        self.assertIs(index, indexes["general"])
        self.assertIn("f-outside", index["info_by_id"])
        self.assertEqual(index_metadata["scope"], "saved_extraction_pool")

    def test_multiple_selected_groups_share_the_same_track_index_object(self):
        bundle = self._bundle()
        indexes, metadata = smoke_generation.build_shared_evidence_indexes(bundle)
        selected = [("general", group_id, copy.deepcopy(group))
                    for group_id, group in bundle["groups"].items()]
        observed = []

        def fake_run(*args):
            observed.append(args[9])
            return {"group_id": args[1], "public_questions": []}, []

        with patch.object(smoke_generation, "_run_group", fake_run):
            smoke_generation._execute_groups(
                selected, "endpoint", "model", "KEY", 1,
                Path("unused"), "simple", indexes, metadata)

        self.assertEqual(len(observed), 2)
        self.assertIs(observed[0], observed[1])
        self.assertIs(observed[0], indexes["general"])

    def test_smoke_review_receives_expanded_active_group_facts(self):
        bundle = self._bundle()
        indexes, _ = smoke_generation.build_shared_evidence_indexes(bundle)
        base_group = copy.deepcopy(bundle["groups"]["general-selected"])
        expanded_group = copy.deepcopy(base_group)
        expanded_group["facts"].append({
            "id": "f-outside", "statement": "outside material",
            "sources": ["e-outside"],
        })
        candidate = {
            "id": "q1", "qa_mode": "general", "type": "single-hop",
            "question": "outside material 是什么？",
            "answer_points": [{"text": "outside material", "sources": ["e-outside"]}],
            "forbidden_points": [],
        }
        generated = {
            "questions": [copy.deepcopy(candidate)],
            "all_candidates": [copy.deepcopy(candidate)],
            "facts": copy.deepcopy(base_group["facts"]),
            "stage_status": {"qa": "completed"},
        }
        generation = {
            "generated": generated,
            "active_group": expanded_group,
            "static_precheck": None,
            "expanded_static_precheck": None,
            "expansion_audits": [{"status": "expanded"}],
            "expansion_rounds": 1,
            "expansion_stop_reason": "candidate_generated",
            "generation_request_count": 2,
        }
        seen_review_facts = []

        class FakeClient:
            usage = []
            responses = []

        seen_budgets = []

        def fake_generation(*args, **kwargs):
            seen_budgets.append(kwargs.get("expansion_budget"))
            return generation

        def fake_review(scope, facts, candidates, client, **kwargs):
            seen_review_facts.append(copy.deepcopy(facts))
            return {"questions": [], "revisions": [], "rejected": [],
                    "stage_errors": [], "stage_status": {"review": "completed"}}

        with tempfile.TemporaryDirectory() as directory:
            target_dir = Path(directory) / "target"
            target_dir.mkdir()
            with patch.object(smoke_generation, "ChatClient", return_value=FakeClient()), \
                    patch.object(smoke_generation, "generate_simple_target",
                                 side_effect=fake_generation), \
                    patch.object(smoke_generation, "review_candidates",
                                 side_effect=fake_review):
                smoke_generation._run_target_type(
                    "general", "general-selected", base_group, "single-hop",
                    "endpoint", "model", "KEY", 1, "simple", target_dir,
                    indexes["general"], "saved_extraction_pool", 3)

        self.assertEqual(len(seen_review_facts), 1)
        self.assertEqual(
            {fact["id"] for fact in seen_review_facts[0]},
            {"f-selected", "f-outside"},
        )
        self.assertEqual(seen_budgets, [3])


if __name__ == "__main__":
    unittest.main()
