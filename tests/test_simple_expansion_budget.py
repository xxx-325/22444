"""Offline contracts for bounded, cumulative simple evidence expansion."""

import copy
import unittest
from unittest.mock import patch

from dialogue_benchmark import cli
from dialogue_benchmark.fact_index import build_evidence_index


class SimpleExpansionBudgetTests(unittest.TestCase):
    def setUp(self):
        self.scope = {
            "cutoff": 7,
            "dialogue": [{
                "id": "e-base", "order": 1, "kind": "message", "role": "user",
                "source_kind": "conversation", "text": "runner.py base",
            }],
            "events": [], "versions": [], "edges": [],
            "historical_edges": [], "stages": [],
            "model_request_chars": 60000,
            "max_context_chars": 60000,
        }
        self.base_fact = {
            "id": "f-base", "qa_mode": "general",
            "statement": "runner.py must retain the base behavior", "sources": ["e-base"],
        }
        self.index = build_evidence_index(
            [self.base_fact], [self.scope], "general", 60000)
        self.group = {
            "id": "g-base", "qa_mode": "general",
            "facts": [copy.deepcopy(self.base_fact)],
            "scope": copy.deepcopy(self.index["universe"]),
            "relation_path": {},
            "eligible_types": ("constraint_followthrough",),
        }

    @staticmethod
    def _missing(kind="outcome", object_name="runner.py"):
        return {
            "questions": [], "all_candidates": [], "rejected": [],
            "stage_errors": [], "raw_generated": 0,
            "stage_status": {"qa": "completed"},
            "missing_kind": kind, "missing_object": object_name,
        }

    @staticmethod
    def _candidate():
        question = {
            "id": "q1", "candidate_id": "q1", "qa_mode": "general",
            "question": "runner.py 的结果是什么？",
            "answer_points": [{"text": "runner.py base", "sources": ["e-base"]}],
            "forbidden_points": [],
        }
        return {
            "questions": [question], "all_candidates": [copy.deepcopy(question)],
            "rejected": [], "stage_errors": [], "raw_generated": 1,
            "stage_status": {"qa": "completed"},
        }

    def _run(self, responses, *, expansion_budget=3, expand=None):
        calls = []

        def generate(scope, facts, client, **kwargs):
            calls.append({
                "cutoff": scope.get("cutoff"),
                "fact_ids": tuple(fact.get("id") for fact in facts),
                "target_type": kwargs.get("target_type"),
                "allowed_types": tuple(kwargs.get("allowed_types", ())),
            })
            return copy.deepcopy(responses[len(calls) - 1])

        with patch.object(cli, "generate_from_facts", side_effect=generate), \
                patch.object(cli, "expand_evidence_group_once",
                             side_effect=expand) as expand_mock:
            result = cli.generate_simple_target(
                self.group, self.index, "constraint_followthrough", object(), "general",
                expansion_budget=expansion_budget,
            )
        return result, calls, expand_mock

    @staticmethod
    def _fact_expander(active_group, _index, _kind, _object_name, **_kwargs):
        round_number = sum(
            fact.get("id", "").startswith("f-extra-")
            for fact in active_group.get("facts", [])
        ) + 1
        expanded = copy.deepcopy(active_group)
        expanded["facts"].append({
            "id": "f-extra-%d" % round_number,
            "qa_mode": "general",
            "statement": "runner.py extra %d" % round_number,
            "sources": ["e-base"],
        })
        return expanded, {
            "status": "expanded", "reason": "explicit_relation",
            "relation": "test", "added_fact_ids": [
                "f-extra-%d" % round_number],
        }

    def test_budget_zero_disables_expansion(self):
        result, calls, expand_mock = self._run(
            [self._missing()], expansion_budget=0,
            expand=self._fact_expander,
        )

        self.assertEqual(len(calls), 1)
        expand_mock.assert_not_called()
        self.assertEqual(result["expansion_rounds"], 0)
        self.assertEqual(result["expansion_audits"], [])
        self.assertEqual(result["expansion_stop_reason"],
                         "expansion_budget_exhausted")

    def test_budget_one_allows_one_expansion_then_stops(self):
        result, calls, expand_mock = self._run(
            [self._missing(), self._missing()], expansion_budget=1,
            expand=self._fact_expander,
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual(expand_mock.call_count, 1)
        self.assertEqual(result["expansion_rounds"], 1)
        self.assertEqual(len(result["expansion_audits"]), 1)
        self.assertEqual(result["expansion_stop_reason"],
                         "expansion_budget_exhausted")
        self.assertEqual(
            {fact["id"] for fact in result["active_group"]["facts"]},
            {"f-base", "f-extra-1"},
        )

    def test_budget_three_repeats_same_kind_only_with_new_facts(self):
        result, calls, expand_mock = self._run(
            [self._missing(), self._missing(), self._missing(), self._candidate()],
            expand=self._fact_expander,
        )

        self.assertEqual(len(calls), 4)
        self.assertEqual(expand_mock.call_count, 3)
        self.assertEqual(result["expansion_rounds"], 3)
        self.assertEqual(result["expansion_stop_reason"], "candidate_generated")
        self.assertEqual(len(result["expansion_audits"]), 3)
        self.assertTrue(all(item["status"] == "expanded"
                            for item in result["expansion_audits"]))
        self.assertEqual(len(result["generated"]["questions"]), 1)
        self.assertEqual(
            {fact["id"] for fact in result["generated"]["facts"]},
            {"f-base", "f-extra-1", "f-extra-2", "f-extra-3"},
        )
        self.assertEqual({item["cutoff"] for item in calls}, {7})
        self.assertEqual({item["target_type"] for item in calls}, {"constraint_followthrough"})
        self.assertEqual({item["allowed_types"] for item in calls},
                         {("constraint_followthrough",)})
        self.assertEqual(
            [item["fact_ids"] for item in calls],
            [
                ("f-base",),
                ("f-base", "f-extra-1"),
                ("f-base", "f-extra-1", "f-extra-2"),
                ("f-base", "f-extra-1", "f-extra-2", "f-extra-3"),
            ],
        )

    def test_no_new_evidence_stops_without_a_repeated_generation(self):
        calls = []

        def expand(active_group, _index, _kind, _object_name, **_kwargs):
            calls.append(tuple(fact["id"] for fact in active_group["facts"]))
            if len(calls) == 1:
                return self._fact_expander(
                    active_group, _index, _kind, _object_name, **_kwargs)
            return None, {
                "status": "not_applicable", "reason": "no_new_evidence",
            }

        result, generation_calls, _ = self._run(
            [self._missing(), self._missing()], expand=expand,
        )

        self.assertEqual(len(generation_calls), 2)
        self.assertEqual(calls, [("f-base",), ("f-base", "f-extra-1")])
        self.assertEqual(result["expansion_rounds"], 1)
        self.assertEqual(result["expansion_stop_reason"], "no_new_evidence")
        self.assertEqual(
            [item.get("reason") for item in result["expansion_audits"]],
            ["explicit_relation", "no_new_evidence"],
        )

    def test_no_match_and_over_budget_stop_without_retry_generation(self):
        for reason, status in (
                ("no_matching_explicit_relation", "not_applicable"),
                ("all_matching_expansions_over_budget", "over_budget")):
            with self.subTest(reason=reason):
                def expand(_active, _index, _kind, _object_name,
                           _reason=reason, _status=status, **_kwargs):
                    return None, {"status": _status, "reason": _reason}

                result, calls, expand_mock = self._run(
                    [self._missing()], expand=expand,
                )
                self.assertEqual(len(calls), 1)
                self.assertEqual(expand_mock.call_count, 1)
                self.assertEqual(result["expansion_rounds"], 0)
                self.assertEqual(result["expansion_stop_reason"], reason)

    def test_generation_failure_stops_without_fabricating_pending_expansion(self):
        failed = {
            "questions": [], "all_candidates": [], "rejected": [],
            "stage_errors": [{"stage": "qa", "error_code": "bad_response"}],
            "raw_generated": 0, "stage_status": {"qa": "failed"},
        }
        result, calls, expand_mock = self._run([failed], expand=self._fact_expander)

        self.assertEqual(len(calls), 1)
        expand_mock.assert_not_called()
        self.assertEqual(result["expansion_rounds"], 0)
        self.assertEqual(result["expansion_audits"], [])
        self.assertEqual(result["expansion_stop_reason"], "generation_failed")

    def test_source_only_expansion_is_the_anchor_for_the_next_round(self):
        seen_extras = []

        def expand(active_group, _index, _kind, _object_name, **_kwargs):
            extras = list(active_group["scope"].get(
                "generation_extra_sources", []))
            seen_extras.append(extras)
            expanded = copy.deepcopy(active_group)
            expanded["scope"]["generation_extra_sources"] = (
                extras + ["anchor-%d" % (len(extras) + 1)])
            return expanded, {
                "status": "expanded", "reason": "explicit_relation",
                "relation": "source_frontier", "added_fact_ids": [],
                "added_source_ids": ["anchor-%d" % (len(extras) + 1)],
            }

        result, calls, expand_mock = self._run(
            [self._missing(), self._missing(), self._candidate()],
            expand=expand,
        )

        self.assertEqual(len(calls), 3)
        self.assertEqual(expand_mock.call_count, 2)
        self.assertEqual(seen_extras, [[], ["anchor-1"]])
        self.assertEqual(result["expansion_rounds"], 2)
        self.assertEqual(result["expansion_stop_reason"], "candidate_generated")
        self.assertEqual(
            result["active_group"]["scope"]["generation_extra_sources"],
            ["anchor-1", "anchor-2"],
        )

    def test_review_receives_final_active_group_facts_and_budget(self):
        expanded_group = copy.deepcopy(self.group)
        expanded_group["facts"].append({
            "id": "f-extra", "qa_mode": "general",
            "statement": "runner.py expanded evidence", "sources": ["e-base"],
        })
        candidate = self._candidate()
        generation = {
            "generated": {
                "questions": [copy.deepcopy(candidate["questions"][0])],
                "all_candidates": [copy.deepcopy(candidate["questions"][0])],
                "facts": copy.deepcopy(self.group["facts"]),
                "rejected": [], "stage_errors": [],
                "raw_generated": 1, "stage_status": {"qa": "completed"},
            },
            "active_group": expanded_group,
            "static_precheck": None, "expanded_static_precheck": None,
            "expansion_audits": [{"status": "expanded"}],
            "expansion_rounds": 1,
            "expansion_stop_reason": "candidate_generated",
            "generation_attempt_count": 1,
            "generation_request_count": 2,
        }
        seen_facts = []
        seen_budget = []

        class FakeClient:
            usage = []

        def fake_generation(*args, **kwargs):
            seen_budget.append(kwargs.get("expansion_budget"))
            return copy.deepcopy(generation)

        def fake_review(_scope, facts, candidates, _client, **_kwargs):
            seen_facts.append(copy.deepcopy(facts))
            return {
                "questions": [dict(candidates[0], status="approved")],
                "rejected": [], "revisions": [], "stage_errors": [],
                "stage_status": {"review": "completed"},
            }

        with patch.object(cli, "ChatClient", return_value=FakeClient()), \
                patch.object(cli, "generate_simple_target",
                             side_effect=fake_generation), \
                patch.object(cli, "review_candidates", side_effect=fake_review):
            result = cli._run_qa_tasks(
                [(0, "general", self.group)], "endpoint", "model", "KEY", 1,
                review_mode="simple", evidence_indexes={"general": self.index},
                expansion_budget=3,
            )

        self.assertEqual(seen_budget, [3])
        self.assertEqual(len(seen_facts), 1)
        self.assertEqual(
            {fact["id"] for fact in seen_facts[0]}, {"f-base", "f-extra"})
        self.assertEqual(len(result["questions"]), 1)

    def test_static_correction_gaps_reach_directional_expansion(self):
        for reason, direction in (("correction_missing_old", "earlier_state"),
                                  ("correction_missing_new", "later_state")):
            with patch.object(cli, "static_evidence_check", return_value={"status": "insufficient", "reason": reason}), \
                 patch.object(cli, "expand_evidence_group_once", return_value=(None, {"reason": "no_related_evidence"})) as expand:
                result = cli.generate_simple_target(self.group, self.index, "correction_update",
                                                    object(), "general", expansion_budget=2)
            self.assertEqual(expand.call_args.args[2], direction)
            self.assertTrue(expand.call_args.kwargs["static_direction"])
            self.assertEqual(result["expansion_stop_reason"], "no_related_evidence")


if __name__ == "__main__":
    unittest.main()
