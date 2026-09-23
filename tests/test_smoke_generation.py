"""Offline contracts for the opt-in generation smoke runner."""

import importlib.util
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from dialogue_benchmark import cli


EXAMPLE = (Path(__file__).resolve().parents[1] / "examples"
           / "smoke_generation.py")
SPEC = importlib.util.spec_from_file_location("smoke_generation", EXAMPLE)
smoke_generation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke_generation)
import compare_reviews


class SmokeGenerationTests(unittest.TestCase):
    def test_group_only_generation_bundle_does_not_require_review_cases(self):
        bundle = {
            "schema_version": 1,
            "cases": [],
            "groups": {"general-a": {
                "qa_mode": "general", "facts": [], "scope": {},
            }},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.json"
            path.write_text(json.dumps(bundle), encoding="utf-8")
            loaded = smoke_generation.load_generation_bundle(path)
        self.assertEqual(list(loaded["groups"]), ["general-a"])

    def test_explicit_group_cohort_keeps_requested_order_and_tracks(self):
        bundle = {"groups": {
            "code-a": {"qa_mode": "code"},
            "general-a": {"qa_mode": "general"},
            "code-b": {"qa_mode": "code"},
        }, "cases": []}
        selected = smoke_generation.select_groups(
            bundle, "code-b,general-a,code-a")
        self.assertEqual(
            [(track, group_id) for track, group_id, _ in selected],
            [("code", "code-b"), ("general", "general-a"),
             ("code", "code-a")])

    def test_parallel_cohort_runs_each_group_once_with_bounded_concurrency(self):
        selected = [("code", "g%d" % index, {}) for index in range(3)]
        barrier = threading.Barrier(2, timeout=2)
        lock = threading.Lock()
        active = 0
        maximum = 0
        called = []

        def fake_run(track, group_id, group, *unused):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
                called.append(group_id)
            if group_id in {"g0", "g1"}:
                barrier.wait()
            with lock:
                active -= 1
            return {"track": track, "group_id": group_id}, [group_id]

        with patch.object(smoke_generation, "_run_group", fake_run):
            summaries, usage = smoke_generation._execute_groups(
                selected, "endpoint", "model", "KEY", 1, Path("unused"),
                "simple", {"code": "shared-index"}, {"code": {}},
                parallel_workers=2)

        self.assertEqual(sorted(called), ["g0", "g1", "g2"])
        self.assertEqual(maximum, 2)
        self.assertEqual([item["group_id"] for item in summaries],
                         ["g0", "g1", "g2"])
        self.assertEqual(usage, ["g0", "g1", "g2"])

    def test_plan_accounts_for_repair_and_second_split_review(self):
        selected = [("general", "general-group", {}),
                    ("code", "code-group", {})]
        plan = smoke_generation._plan(selected)
        self.assertEqual(plan["expected_calls_upper_bound"], 12)
        self.assertEqual(plan["max_questions_per_group"], 1)
        self.assertTrue(plan["allow_repair"])

    def test_simple_plan_is_explicit_and_does_not_reserve_annotation(self):
        selected = [("general", "general-group", {"eligible_types": ("constraint_followthrough",)}),
                    ("code", "code-group", {"eligible_types": ("constraint_followthrough",)})]
        plan = smoke_generation._plan(selected, "simple")
        self.assertEqual(plan["review_mode"], "simple")
        self.assertEqual(plan["base_calls_per_group"], 6)
        self.assertEqual(plan["repair_calls_per_group_upper_bound"], 5)
        self.assertEqual(plan["base_calls_per_target_type_by_track"],
                         {"general": 5, "code": 6})
        self.assertEqual(
            plan["evidence_supplement_calls_per_group_upper_bound"], 1)
        self.assertEqual(plan["post_annotation_calls_per_group"], 0)
        self.assertEqual(plan["expected_calls_upper_bound"], 34)
        self.assertEqual(len(plan["target_type_units"]), 2)

    def test_simple_plan_splits_a_group_by_static_target_type(self):
        selected = [("general", "general-group", {
            "eligible_types": ("correction_update", "constraint_followthrough"),
        })]
        plan = smoke_generation._plan(selected, "simple")
        self.assertEqual(
            [(unit["group_id"], unit["target_type"])
             for unit in plan["target_type_units"]],
            [("general-group", "constraint_followthrough"), ("general-group", "correction_update")])
        self.assertEqual(plan["group_plans"][0]["units"], 2)
        self.assertEqual(plan["expected_calls_upper_bound"], 32)

    def test_simple_plan_does_not_fallback_to_non_static_group_types(self):
        selected = [("general", "general-group", {
            "allowed_types": ("constraint_followthrough",),
        })]
        plan = smoke_generation._plan(selected, "simple")
        self.assertEqual(plan["target_type_units"], [])
        self.assertEqual(plan["group_plans"][0]["target_types"], [])
        self.assertEqual(plan["expected_calls_upper_bound"], 0)

    def test_simple_plan_exposes_the_per_target_expansion_budget(self):
        selected = [("general", "general-group", {
            "eligible_types": ("constraint_followthrough",),
        })]
        no_expansion = smoke_generation._plan(selected, "simple", 0)
        one_expansion = smoke_generation._plan(selected, "simple", 1)
        default_expansion = smoke_generation._plan(selected, "simple")

        self.assertEqual(
            no_expansion["directed_expansion_calls_per_target_type_upper_bound"],
            0,
        )
        self.assertEqual(no_expansion["expected_calls_upper_bound"], 10)
        self.assertEqual(one_expansion["expected_calls_upper_bound"], 12)
        self.assertEqual(default_expansion["expected_calls_upper_bound"], 16)

    def test_cli_expansion_budget_defaults_to_three_allows_zero_and_rejects_negative(self):
        parser = cli._build_parser()
        default_args = parser.parse_args(["input.json", "--output", "out"])
        self.assertEqual(default_args.expansion_budget, 3)

        zero_args = parser.parse_args([
            "input.json", "--output", "out", "--expansion-budget", "0",
        ])
        self.assertEqual(zero_args.expansion_budget, 0)

        negative_args = parser.parse_args([
            "input.json", "--output", "out", "--expansion-budget", "-1",
        ])
        with self.assertRaises(SystemExit):
            cli._parse_options(negative_args, parser)

    def test_compare_both_keeps_legacy_modes_and_simple_is_opt_in(self):
        bundle = {"cases": [{}, {}]}
        legacy = compare_reviews._plan(bundle, "both")
        self.assertEqual(set(legacy["modes"]), {"single", "split"})
        simple = compare_reviews._plan(bundle, "simple")
        self.assertEqual(simple["modes"]["simple"]["expected_requests"], 8)
        self.assertEqual(simple["modes"]["simple"]["post_annotation_calls"], 0)

        code_bundle = {
            "cases": [{"group_id": "code-group", "candidate": {}}],
            "groups": {"code-group": {"qa_mode": "code"}},
        }
        code = compare_reviews._plan(code_bundle, "simple")
        self.assertEqual(code["modes"]["simple"]["expected_requests"], 5)

    def test_public_projection_excludes_non_approved_questions(self):
        questions = [{
            "id": "approved", "qa_mode": "general", "status": "approved",
            "type": "constraint_followthrough", "question": "safe question",
            "answer_points": [], "forbidden_points": [],
        }, {
            "id": "needs-review", "qa_mode": "general", "status": "needs_review",
            "type": "constraint_followthrough", "question": "private review item",
            "answer_points": [], "forbidden_points": [],
        }]
        public = smoke_generation._public_questions(questions)
        self.assertEqual([item["id"] for item in public], ["approved"])


if __name__ == "__main__":
    unittest.main()
