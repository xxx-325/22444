"""Offline CLI contracts for the statically targeted simple pipeline."""

import json
import tempfile
import unittest
import threading
from pathlib import Path
from unittest.mock import patch

from dialogue_benchmark import cli
from dialogue_benchmark.fact_index import build_evidence_index


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "dialogue.json"


class SimpleCliPipelineTests(unittest.TestCase):
    def test_generation_and_review_are_bounded_without_reclassification_gate(self):
        scope = {"cutoff": 1, "dialogue": [{"id": "e1", "order": 1,
                 "kind": "message", "role": "user", "text": "保留 yaml"}],
                 "events": [], "versions": [], "model_request_chars": 32000}
        fact = {"id": "f1", "statement": "保留 yaml", "sources": ["e1"]}
        index = build_evidence_index([fact], [scope], "general")
        tasks = [(i, "general", {"id": "g%d" % i, "qa_mode": "general",
                 "scope": scope, "facts": [fact], "allowed_types": (),
                 "type_selection": "post_generation"}) for i in range(2)]
        finished, calls = set(), []
        lock = threading.Lock()

        class FakeClient:
            def __init__(self, *args, **kwargs):
                self.usage = []

        def generate(group, *args, **kwargs):
            q = {"id": group["id"], "qa_mode": "general", "question": group["id"],
                 "answer_points": [{"text": "保留 yaml", "sources": ["e1"]}],
                 "forbidden_points": []}
            with lock:
                finished.add(group["id"])
                calls.append("generate")
            return {"generated": {"questions": [q], "all_candidates": [q],
                    "stage_status": {"qa": "completed"}}, "active_group": group,
                    "static_precheck": {}, "expanded_static_precheck": {},
                    "expansion_audits": [], "expansion_rounds": 0,
                    "expansion_stop_reason": "candidate_generated",
                    "generation_attempt_count": 1, "generation_request_count": 1}

        def review(scope, facts, candidates, *args, **kwargs):
            # Review may start as soon as this group's generation finishes;
            # another group is allowed to remain in flight.
            self.assertTrue(finished)
            self.assertGreaterEqual(len(list(checkpoints.glob("*generated-candidates.json"))), 1)
            calls.append("review")
            return {"questions": [dict(candidates[0], status="approved")],
                    "review_warnings": [{"reason": "annotation_unavailable"}]}

        with tempfile.TemporaryDirectory() as directory:
            checkpoints = Path(directory)
            with patch.object(cli, "ChatClient", FakeClient), \
                    patch.object(cli, "generate_simple_target", generate), \
                    patch.object(cli, "review_candidates", review), \
                    patch.object(cli, "static_candidate_types", return_value=([], {})):
                result = cli._run_qa_tasks(tasks, "https://example.invalid", "model", "KEY", 2,
                    checkpoint_dir=checkpoints, review_mode="simple",
                    evidence_indexes={"general": index})
        self.assertEqual(calls.count("generate"), 2)
        self.assertEqual(calls.count("review"), 2)
        self.assertEqual(len(result["questions"]), 2)
        self.assertEqual(len(result["review_warnings"]), 2)
        self.assertFalse(result["rejected"])
        for q in result["questions"]:
            self.assertEqual(q["status"], "approved")
            self.assertIsNone(q["type"])
            self.assertEqual(q["type_status"], "unresolved")

    def test_post_generation_group_uses_one_untyped_request_then_static_label(self):
        scope = {
            "cutoff": 1,
            "dialogue": [{"id": "e1", "order": 1, "kind": "message",
                          "role": "user", "stage_id": "s1",
                          "text": "保留 yaml 配置格式"}],
            "events": [], "versions": [], "edges": [], "historical_edges": [],
            "stages": [{"id": "s1", "record_ids": ["e1"]}],
        }
        fact = {"id": "f1", "qa_mode": "general",
                "statement": "配置必须保留 yaml 格式", "sources": ["e1"]}
        index = build_evidence_index([fact], [scope], "general")
        group = {
            "id": "general-group-1", "qa_mode": "general",
            "scope": scope, "facts": [fact],
            "allowed_types": ("constraint_followthrough", "verification_reuse"),
            "eligible_types": ("constraint_followthrough", "verification_reuse"),
            "type_selection": "post_generation",
        }
        calls = []

        class FakeClient:
            def __init__(self, *unused, **kwargs):
                self.usage = []

        def generate(scope, facts, client, max_questions, target_type,
                     generation_mode, **kwargs):
            calls.append((target_type, generation_mode, max_questions))
            self.assertIsNone(target_type)
            self.assertEqual(generation_mode, "untyped")
            question = {
                "id": "q1", "candidate_id": "q1", "qa_mode": "general",
                "question": "配置格式需要保留什么？",
                "answer_points": [{"text": "必须保留 yaml 格式。", "sources": ["e1"]}],
                "forbidden_points": [],
            }
            return {"questions": [question], "all_candidates": [question],
                    "stage_status": {"qa": "completed"}, "raw_generated": 1,
                    "generation_attempt_count": 1, "generation_request_count": 1,
                    "expansion_rounds": 0, "expansion_stop_reason": "candidate_generated",
                    "repair_context": None}

        def review(scope, facts, candidates, client, **kwargs):
            return {"questions": [dict(item, status="approved") for item in candidates],
                    "rejected": [], "stage_errors": [],
                    "stage_status": {"review": "completed"}, "revisions": []}

        with patch.object(cli, "ChatClient", FakeClient), \
                patch.object(cli, "generate_from_facts", generate), \
                patch.object(cli, "review_candidates", review), \
                patch.object(cli, "static_candidate_types", return_value=(
                    ["constraint_followthrough"], {
                        "constraint_followthrough": {"status": "supported"},
                    })), \
                patch.object(cli, "static_evidence_check", return_value={
                    "status": "supported", "reason": "recorded_type_evidence",
                    "fact_ids": ["f1"], "source_ids": ["e1"],
                }):
            result = cli._run_qa_tasks(
                [(0, "general", group)], "https://example.invalid", "model", "KEY", 1,
                review_mode="simple", evidence_indexes={"general": index})

        self.assertEqual(calls, [(None, "untyped", 1)])
        self.assertEqual(len(result["questions"]), 1)
        self.assertEqual(result["questions"][0]["type"],
                         "constraint_followthrough")
        self.assertEqual(result["questions"][0]["type_origin"],
                         "static_post_generation")
        self.assertEqual(result["type_attempts"][0]["type_selection"],
                         "post_generation")

    def test_one_group_attempts_each_eligible_type_sequentially(self):
        scope = {
            "cutoff": 1,
            "dialogue": [{"id": "e1", "order": 1, "kind": "message",
                          "role": "user", "stage_id": "s1", "text": "keep yaml"}],
            "events": [], "versions": [], "edges": [], "historical_edges": [],
            "stages": [{"id": "s1", "record_ids": ["e1"]}],
        }
        fact = {"id": "f1", "qa_mode": "general",
                "statement": "keep yaml as required; yaml 测试通过", "sources": ["e1"]}
        index = build_evidence_index([fact], [scope], "general")
        group = {
            "id": "general-group-1", "qa_mode": "general",
            "scope": scope, "facts": [fact],
            "allowed_types": ("constraint_followthrough", "verification_reuse"),
            "eligible_types": ("constraint_followthrough", "verification_reuse"),
            "relation_path": {"seed_node": "e1"},
            "expansion_pointer": {"attempted": False, "candidates": []},
        }
        calls = []

        class FakeClient:
            def __init__(self, *unused, **kwargs):
                self.usage = []

        def generate(scope, facts, client, max_questions, target_type, **kwargs):
            calls.append((target_type, max_questions))
            question = {
                "id": "q1", "candidate_id": "q1", "qa_mode": "general",
                "question": "question for " + target_type,
                "answer_points": [{"text": "keep yaml", "sources": ["e1"]}],
                "forbidden_points": [],
            }
            return {"questions": [question], "all_candidates": [question],
                    "stage_status": {"qa": "completed"}, "raw_generated": 1}

        def review(scope, facts, candidates, client, **kwargs):
            return {"questions": [dict(item, status="approved") for item in candidates],
                    "rejected": [], "stage_errors": [],
                    "stage_status": {"review": "completed"}, "revisions": []}

        with patch.object(cli, "ChatClient", FakeClient), \
                patch.object(cli, "generate_from_facts", generate), \
                patch.object(cli, "review_candidates", review):
            result = cli._run_qa_tasks(
                [(0, "general", group)], "https://example.invalid", "model", "KEY", 1,
                review_mode="simple", evidence_indexes={"general": index})

        self.assertEqual(calls, [("constraint_followthrough", 1), ("verification_reuse", 1)])
        self.assertEqual(len(result["questions"]), 2)
        self.assertEqual([item["target_type"] for item in result["type_attempts"]],
                         ["constraint_followthrough", "verification_reuse"])
        for question in result["questions"]:
            self.assertEqual(question["type_origin"], "static_target")
            self.assertEqual(question["difficulty_origin"], "static_graph_distance")
            self.assertEqual(question["difficulty_distance"], 0)

    def test_explicit_empty_eligible_types_does_not_fall_back(self):
        class FakeClient:
            def __init__(self, *unused, **kwargs):
                self.usage = []

        group = {"id": "g1", "scope": {}, "facts": [],
                 "allowed_types": ("constraint_followthrough",), "eligible_types": ()}
        index = {"qa_mode": "general"}
        with patch.object(cli, "ChatClient", FakeClient), \
                patch.object(cli, "generate_from_facts") as generate:
            result = cli._run_qa_tasks(
                [(0, "general", group)], "https://example.invalid", "model", "KEY", 1,
                review_mode="simple", evidence_indexes={"general": index})
        generate.assert_not_called()
        self.assertFalse(result["questions"])
        self.assertFalse(result["stage_errors"])

    def test_approved_unknown_difficulty_is_published_with_the_label(self):
        question = {
            "id": "q1", "qa_mode": "general", "type": "constraint_followthrough",
            "type_origin": "static_target", "question": "What was required?",
            "difficulty": "unknown", "difficulty_distance": None,
            "difficulty_origin": "static_graph_distance", "status": "approved",
            "answer_points": [{"text": "Keep yaml."}], "forbidden_points": [],
        }

        view = cli._publication_view([question], {"general": 1})

        self.assertEqual([item["id"] for item in view["questions"]], ["q1"])
        self.assertFalse(view["publication_rejected"])

    def test_main_simple_both_tracks_reuses_indexes_and_publishes_static_labels(self):
        calls = []
        timeouts = {}

        class FakeClient:
            def __init__(self, *unused, **kwargs):
                self.usage = []
                self.timeout = unused[3]

        def extract(scope, client, qa_mode, checkpoint=None):
            timeouts["facts"] = client.timeout
            source = scope["dialogue"][0]["id"]
            return {
                "facts": [{"id": "f1", "qa_mode": qa_mode,
                           "statement": qa_mode + " must retain the recorded behavior",
                           "sources": [source]}],
                "questions": [], "rejected": [], "stage_errors": [],
                "stage_status": {"facts": "completed"},
            }

        def generate(scope, facts, client, max_questions, qa_mode, target_type,
                     generation_mode, **kwargs):
            timeouts["generation"] = client.timeout
            self.assertEqual(generation_mode, "simple")
            self.assertEqual(max_questions, 1)
            calls.append((qa_mode, target_type))
            source = facts[0]["sources"][0]
            question = {
                "id": "q1", "candidate_id": "q1", "qa_mode": qa_mode,
                "question": "%s %s question" % (qa_mode, target_type),
                "answer_points": [{"text": "grounded answer", "sources": [source]}],
                "forbidden_points": [],
            }
            return {"questions": [question], "all_candidates": [question],
                    "rejected": [], "stage_errors": [], "raw_generated": 1,
                    "stage_status": {"qa": "completed"}}

        def review(scope, facts, candidates, client, review_mode, **kwargs):
            timeouts["review"] = client.timeout
            self.assertEqual(review_mode, "simple")
            return {"questions": [dict(item, status="approved") for item in candidates],
                    "rejected": [], "stage_errors": [], "revisions": [],
                    "stage_status": {"review": "completed"}}

        def duplicate_review(candidates, client, reviewed_pairs):
            timeouts["deduplication"] = client.timeout
            return {"decisions": [], "errors": [], "usage": [],
                    "reviewed_pairs": set(reviewed_pairs)}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with patch.object(cli, "ChatClient", FakeClient), \
                    patch.object(cli, "extract_facts", extract), \
                    patch.object(cli, "generate_from_facts", generate), \
                    patch.object(cli, "review_candidates", review), \
                    patch.object(cli, "static_evidence_check", return_value={
                        "status": "unknown", "reason": "synthetic_unknown",
                        "fact_ids": [], "source_ids": [],
                    }), \
                    patch.object(cli, "review_duplicate_clusters", duplicate_review):
                status = cli.main([
                    str(EXAMPLE), "--output", str(output), "--qa-mode", "both",
                    "--general-types", "constraint_followthrough", "--code-types", "constraint_followthrough",
                    "--general-count", "1", "--code-count", "1",
                    "--general-group-budget", "1", "--code-group-budget", "1",
                    "--parallel-workers", "1", "--allow-network",
                    "--endpoint", "https://example.invalid", "--model", "model",
                    "--request-timeout", "1800",
                ])
            public = json.loads((output / "qa-public.json").read_text())
            audit = json.loads((output / "qa-audit.json").read_text())

        self.assertEqual(status, 0)
        self.assertEqual(calls, [("general", "constraint_followthrough"), ("code", "constraint_followthrough")])
        self.assertEqual(timeouts, {stage: 1800 for stage in ("facts", "generation", "review", "deduplication")})
        self.assertEqual(public["counts"], {"general": 1, "code": 1})
        self.assertFalse(audit["stage_errors"])
        for question in public["questions"]:
            self.assertEqual(question["type_origin"], "static_target")
            self.assertEqual(question["difficulty_origin"], "static_graph_distance")
            self.assertEqual(question["difficulty_distance"], 0)


if __name__ == "__main__":
    unittest.main()
