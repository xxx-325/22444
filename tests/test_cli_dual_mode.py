import argparse
import json
import tempfile
import time
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from dialogue_benchmark import cli


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "dialogue.json"


class DualModeCliTests(unittest.TestCase):
    def test_global_auth_blocker_stops_subsequent_fact_batches(self):
        calls = []
        class FakeClient:
            def __init__(self, *args):
                self.usage = []
        def extract(scope, client, **kwargs):
            calls.append(scope["name"])
            return {"facts": [], "stage_errors": [{"http_status": 401}],
                    "stage_status": {"facts": "failed"}}
        tasks = [(i, "code", {"name": str(i)}) for i in range(6)]
        with patch.object(cli, "ChatClient", FakeClient), patch.object(cli, "extract_facts", extract):
            result = cli._run_fact_tasks(tasks, "https://example.invalid", "model", "KEY", 2)
        self.assertEqual(sorted(calls), ["0", "1"])
        self.assertEqual(len(result["scopes"]["code"]), 6)
        self.assertTrue(all(s["facts_failed"] for s in result["scopes"]["code"]))
        self.assertEqual(sum(s.get("facts") == "not_submitted" for s in result["stage_status"]), 4)

    def test_single_worker_keeps_different_questions_with_the_same_answer(self):
        calls = []
        class FakeClient:
            def __init__(self, *args):
                self.usage = []
        def generate(scope, facts, client, max_questions, **kwargs):
            calls.append(scope["name"])
            return {"questions": [{"id": "q1", "question": scope["name"],
                                   "answer_points": [{"text": "Use yaml", "sources": ["e1"]}]}],
                    "stage_status": {"qa": "completed"}}
        def review(scope, facts, candidates, client, **kwargs):
            return {"questions": [dict(q, status="approved") for q in candidates]}
        tasks = [(i, "code", {"id": "g%d" % i, "scope": {"name": str(i)},
                              "facts": [], "allowed_types": ()}) for i in range(3)]
        with patch.object(cli, "ChatClient", FakeClient), \
             patch.object(cli, "generate_from_facts", generate), \
             patch.object(cli, "review_candidates", review):
            result = cli._run_qa_tasks(tasks, "https://example.invalid", "model", "KEY", 1)
        self.assertEqual(calls, ["0", "1", "2"])
        self.assertEqual(len(result["questions"]), 3)
        self.assertEqual(len(result["all_candidates"]), 3)
        self.assertFalse(result["rejected"])

    def test_distinct_groups_parallel_once_with_bounded_candidate_limits(self):
        gate = threading.Barrier(2)
        calls = []
        class FakeClient:
            def __init__(self, *args):
                self.usage = []
        def generate(scope, facts, client, max_questions, **kwargs):
            calls.append((scope["name"], max_questions))
            gate.wait(timeout=3)
            return {"questions": [{"id": "q%d" % i, "question": scope["name"] + str(i)}
                                  for i in range(max_questions)],
                    "stage_status": {"qa": "completed"}}
        def review(scope, facts, candidates, client, **kwargs):
            for q in candidates:
                self.assertEqual(q["candidate_id"], q["id"])
                self.assertTrue(q["id"].endswith(q["model_id"]))
            return {"questions": [dict(q, status="approved") for q in candidates]}
        tasks = [(i, "code", {"id": "g%d" % i, "scope": {"name": str(i)},
                              "facts": [], "allowed_types": (), **options})
                 for i, options in enumerate(({}, {"max_questions": 2}))]
        with patch.object(cli, "ChatClient", FakeClient), \
             patch.object(cli, "generate_from_facts", generate), \
             patch.object(cli, "review_candidates", review):
            result = cli._run_qa_tasks(tasks, "https://example.invalid", "model", "KEY", 2)
        self.assertEqual(sorted(calls), [("0", 1), ("1", 2)])
        self.assertEqual(len(result["questions"]), 3)
        self.assertEqual(len(result["all_candidates"]), 3)

    def test_candidate_scope_budget_is_distinct_from_model_chunk_budget(self):
        args = cli._build_parser().parse_args([
            str(EXAMPLE), "--output", "unused",
        ])
        self.assertGreater(args.max_context_chars, args.chunk_chars)
        self.assertLessEqual(args.chunk_chars, args.model_request_chars)

    def test_independent_type_and_count_options(self):
        parser = cli._build_parser()
        args = parser.parse_args([
            str(EXAMPLE), "--output", "unused", "--qa-mode", "both",
            "--general-types", "single-hop,temporal",
            "--code-types", "history_tracking,failure_diagnosis",
            "--general-count", "3", "--code-count", "7",
        ])
        options = cli._parse_options(args, parser)
        self.assertEqual(options["general_types"], ("single-hop", "temporal"))
        self.assertEqual(options["code_types"],
                         ("history_tracking", "failure_diagnosis"))
        self.assertEqual(options["general_count"], 3)
        self.assertEqual(options["code_count"], 7)

    def test_legacy_limit_conflicts_with_new_code_limit(self):
        parser = cli._build_parser()
        args = parser.parse_args([
            str(EXAMPLE), "--output", "unused", "--code-count", "2",
            "--max-questions", "3",
        ])
        with self.assertRaises(SystemExit):
            cli._parse_options(args, parser)

    def test_general_only_rejects_code_seed_before_network(self):
        parser = cli._build_parser()
        args = parser.parse_args([
            str(EXAMPLE), "--output", "unused", "--qa-mode", "general",
            "--seed", "config.py",
        ])
        with self.assertRaises(SystemExit):
            cli._parse_options(args, parser)

    def test_general_static_run_needs_no_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            self.assertEqual(cli.main([
                str(EXAMPLE), "--output", str(output), "--qa-mode", "general",
                "--general-types", "single-hop,temporal",
            ]), 0)
            public = json.loads((output / "qa-public.json").read_text())
            scope = json.loads((output / "general-scope.json").read_text())
            self.assertEqual(public["status"], "static_only")
            self.assertEqual(public["qa_mode"], "general")
            self.assertTrue(scope["dialogue"])
            self.assertTrue(all(record["kind"] == "message"
                                for record in scope["dialogue"]))
            self.assertTrue((output / "general-qa.json").exists())
            self.assertTrue((output / "code-qa.json").exists())
            code = json.loads((output / "code-qa.json").read_text())
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(code["status"], "disabled")
            self.assertFalse(code["enabled"])
            self.assertEqual(public["tracks"]["code"]["status"], "disabled")
            self.assertEqual(manifest["tracks"]["code"]["status"], "disabled")
            self.assertGreater(manifest["stages"]["general"], 0)
            self.assertEqual(manifest["configured_limits"]["general"], 10)
            self.assertEqual(manifest["configured_limits"]["code"], 0)

    def test_global_limits_are_independent(self):
        result = {"questions": [], "rejected": []}
        result["questions"].extend(
            {"id": "g%d" % index, "qa_mode": "general", "status": "approved"} for index in range(4))
        result["questions"].extend(
            {"id": "c%d" % index, "qa_mode": "code", "status": "approved"} for index in range(5))
        cli._limit_questions(result, {"general": 2, "code": 3})
        self.assertEqual(result["counts"], {"general": 2, "code": 3})
        self.assertEqual(len(result["questions"]), 5)
        self.assertEqual(len(result["rejected"]), 0)
        self.assertEqual(len(result["selection"]), 4)
        self.assertEqual(
            [item["track"] for item in result["selection"]],
            ["general", "general", "code", "code"],
        )

    def test_public_view_redacts_home_prefix_only(self):
        candidate = {
            "id": "q1", "qa_mode": "code", "type": "history_tracking",
            "category": "history_tracking", "track": "history_core",
            "question": "继续检查 /Users/example/project/src/service.py 的行为",
            "difficulty": "medium", "answer_points": [], "forbidden_points": [],
        }
        rejected = []
        public = cli._safe_public_questions([candidate], rejected)
        self.assertEqual(public[0]["question"], "继续检查 ~/project/src/service.py 的行为")
        self.assertFalse(rejected)

    def test_public_path_guard_catches_generic_and_id_paths(self):
        candidates = [
            {"id": "q1", "qa_mode": "code", "question": "检查 /foo/bar.py"},
            {"id": "/Users/me/private/q2", "qa_mode": "general",
             "question": "这个决定是什么？"},
            {"id": "q3", "qa_mode": "general", "question": "访问 https://example.test/a"},
        ]
        rejected = []
        public = cli._safe_public_questions(candidates, rejected)
        self.assertEqual([item["id"] for item in public], ["q1", "~/private/q2", "q3"])
        self.assertFalse(rejected)

    def test_credential_candidate_does_not_consume_quota(self):
        unsafe = {"id": "q-private", "qa_mode": "code",
                  "question": "password=hunter2"}
        safe = {"id": "q-safe", "qa_mode": "code", "status": "approved",
                "question": "检查相同逻辑"}
        rejected = []
        candidates, path_stats = cli._filter_private_questions(
            [unsafe, safe], rejected)
        result = {"questions": candidates, "rejected": rejected}
        cli._add_path_stats(result, path_stats)
        cli._limit_questions(result, {"general": 1, "code": 1})
        self.assertEqual([item["id"] for item in result["questions"]], ["q-safe"])
        self.assertEqual(result["question_stats"]["path_rejected"], 1)
        self.assertEqual(result["question_stats"]["credential_detected"], 1)
        self.assertEqual(result["question_stats"]["limit_rejected"], 0)

    def test_path_projection_preserves_structure_and_urls(self):
        cases = [
            ("/tmp/report.json", "/tmp/report.json"),
            ("/var/tmp/report.json", "/var/tmp/report.json"),
            ("/Users/alice/project/src/a.py", "~/project/src/a.py"),
            ("/home/alice/project/src/a.py", "~/project/src/a.py"),
            (r"C:\Users\alice\project\src\a.py", r"~\project\src\a.py"),
            ("D:/Users/alice/project/a.py", "~/project/a.py"),
            ("https://example.test/Users/alice/a", "https://example.test/Users/alice/a"),
            ("https://example.test/home/alice/a", "https://example.test/home/alice/a"),
        ]
        for original, expected in cases:
            with self.subTest(original=original):
                self.assertEqual(cli._redact_public_text(original)[0], expected)
        self.assertEqual(cli._redact_public_text(
            "/Users/alice/project/src/a.py", ["/Users/alice/project"]),
            ("<workspace>/src/a.py", 1))
        self.assertEqual(cli._redact_public_text(
            "/home/alice/project-other/a.py", ["/home/alice/project"])[0],
            "~/project-other/a.py")

    def test_projection_covers_public_fields_without_mutating_audit(self):
        original = "/Users/alice/project/src/a.py"
        candidate = {field: original for field in (
            "question", "use_case", "difficulty_reason", "memory_requirement", "external_knowledge")}
        candidate.update(id="q1", qa_mode="general",
                         answer_points=[{"text": original, "sources": ["e1"]}],
                         forbidden_points=[{"text": original, "sources": ["e2"]}])
        projected, count = cli._project_public_question(candidate)
        self.assertEqual(count, 7)
        self.assertNotIn("/Users/alice", json.dumps(projected))
        self.assertEqual(candidate["answer_points"][0]["text"], original)
        self.assertEqual(projected["answer_points"][0]["sources"], ["e1"])

    def test_credentials_reject_whole_question_in_each_public_field(self):
        secrets = ["-----BEGIN OPENSSH PRIVATE KEY-----",
                   "Bearer synthetic-test-token", "sk-" + "x" * 24,
                   '"api_key": "synthetic-test-key"',
                   "API_KEY=synthetic-test-key", "token=synthetic-test-token",
                   "password=synthetic-password"]
        for secret in secrets:
            for field in ("question", "use_case", "answer_points", "forbidden_points"):
                candidate = {"id": "q1", "question": "What was decided?"}
                candidate[field] = [{"text": secret}] if field.endswith("points") else secret
                with self.subTest(secret_type=secret.split()[0], field=field):
                    rejected = []
                    self.assertEqual(cli._safe_public_questions([candidate], rejected), [])
                    self.assertEqual(rejected[0]["reason"], "credential_detected")

    def test_redacted_duplicates_do_not_consume_quota(self):
        candidates = [{"id": "q%d" % i, "qa_mode": "general", "question": question}
                      for i, question in enumerate([
                          "检查 /Users/alice/project/a.py", "检查 /home/bob/project/a.py",
                          "检查 /tmp/report.json"])]
        rejected = []
        safe, stats = cli._filter_private_questions(candidates, rejected)
        self.assertEqual(len(safe), 2)
        self.assertEqual(stats["path_redacted"], 2)
        self.assertEqual(stats["redaction_deduplicated"], 1)
        self.assertEqual(stats["total"], 0)
        self.assertEqual(rejected[0]["reason"], "duplicate_after_redaction")

    def test_status_distinguishes_disabled_and_partial_failure(self):
        self.assertEqual(cli._status({"questions": [], "rejected": [],
                                       "stage_errors": []}),
                         "completed_no_questions")
        self.assertEqual(cli._status({"questions": [], "rejected": [],
                                       "stage_errors": [{"stage": "facts"}]}),
                         "failed")
        self.assertEqual(cli._status({"questions": [{"status": "approved"}],
                                      "rejected": [],
                                      "stage_errors": [{"stage": "facts"}]}),
                         "needs_review")

    def test_qa_stats_separate_raw_and_deduplicated_candidates(self):
        class FakeClient:
            def __init__(self, *unused):
                self.usage = []

        def fake_generate(scope, facts, client, max_questions, qa_mode,
                          allowed_types, checkpoint=None, candidate_prefix=None,
                          **unused):
            question = {"id": "q1", "qa_mode": qa_mode,
                        "question": "same question",
                        "answer_points": [{"text": "same answer", "sources": ["e1"]}]}
            return {"questions": [question], "rejected": [],
                    "stage_errors": [], "stage_status": {"qa": "completed"}}

        def fake_review(scope, facts, candidates, client, qa_mode, checkpoint=None,
                        review_mode="split"):
            return {"questions": candidates, "rejected": [], "stage_errors": [],
                    "stage_status": {"review": "completed"}}

        group = {"scope": {}, "facts": [], "allowed_types": ("single-hop",)}
        with patch.object(cli, "ChatClient", FakeClient), \
                patch.object(cli, "generate_from_facts", fake_generate), \
                patch.object(cli, "review_candidates", fake_review):
            result = cli._run_qa_tasks(
                [(0, "general", group), (1, "general", group)],
                "https://example.invalid", "model", "KEY", 2,
                review_mode="single")
        self.assertEqual(result["question_stats"]["raw_generated"], 2)
        self.assertEqual(result["question_stats"]["deduplicated"], 1)
        self.assertEqual(result["question_stats"]["pre_review_unique"], 2)
        self.assertEqual(result["question_stats"]["post_review"], 1)

    def test_chunk_identity_ignores_scope_bookkeeping_but_not_track(self):
        first = {"cutoff": 4, "dialogue": [{"id": "e1"}], "events": [],
                 "versions": [], "scope_index": 1, "chunk_index": 2}
        second = dict(first, scope_index=9, chunk_index=7, score=99)
        self.assertEqual(cli._chunk_identity("code", first),
                         cli._chunk_identity("code", second))
        self.assertNotEqual(cli._chunk_identity("general", first),
                            cli._chunk_identity("code", first))

    def test_same_fact_in_different_tracks_keeps_track_local_ids(self):
        class FakeClient:
            def __init__(self, *unused):
                self.usage = []

        def fake_extract(scope, client, qa_mode, checkpoint=None):
            return {
                "facts": [{"id": "f1", "statement": "same", "sources": ["e1"]}],
                "questions": [],
                "rejected": [], "stage_errors": [],
                "stage_status": {"facts": "completed"},
            }

        tasks = [
            (0, "general", {"scope_index": 0, "chunk_index": 0}),
            (1, "code", {"scope_index": 0, "chunk_index": 0}),
        ]
        with patch.object(cli, "ChatClient", FakeClient), \
                patch.object(cli, "extract_facts", fake_extract):
            result = cli._run_fact_tasks(
                tasks, "https://example.invalid", "model", "KEY", 2)

        self.assertEqual(len(result["facts"]), 2)
        facts_by_mode = {fact["qa_mode"]: fact["id"] for fact in result["facts"]}
        self.assertNotEqual(facts_by_mode["general"], facts_by_mode["code"])

    def test_shared_executor_processes_all_tasks_with_one_worker(self):
        active = 0
        maximum = 0
        completed = []

        class FakeClient:
            def __init__(self, *unused):
                self.usage = []

        def fake_extract(scope, client, qa_mode, checkpoint=None):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            time.sleep(0.001)
            completed.append(scope["chunk_index"])
            active -= 1
            return {"facts": [], "questions": [], "rejected": [],
                    "stage_errors": [], "stage_status": {"facts": "completed"}}

        tasks = [
            (index, "general" if index % 2 == 0 else "code",
             {"scope_index": index, "chunk_index": index})
            for index in range(4)
        ]
        with patch.object(cli, "ChatClient", FakeClient), \
                patch.object(cli, "extract_facts", fake_extract):
            result = cli._run_fact_tasks(
                tasks, "https://example.invalid", "model", "KEY", 1)
        self.assertEqual(sorted(completed), [0, 1, 2, 3])
        self.assertEqual(maximum, 1)
        self.assertEqual(len(result["stage_status"]), 4)

    def test_one_fact_chunk_failure_does_not_discard_other_chunks(self):
        class FakeClient:
            def __init__(self, *unused):
                self.usage = []

        def fake_extract(scope, client, qa_mode, checkpoint=None):
            if scope["chunk_index"] == 0:
                raise RuntimeError("synthetic")
            return {"facts": [{"id": "f1", "statement": "kept",
                                "sources": ["e2"]}],
                    "questions": [], "rejected": [], "stage_errors": [],
                    "stage_status": {"facts": "completed"}}

        tasks = [(0, "general", {"scope_index": 0, "chunk_index": 0}),
                 (1, "general", {"scope_index": 0, "chunk_index": 1})]
        with patch.object(cli, "ChatClient", FakeClient), \
                patch.object(cli, "extract_facts", fake_extract):
            result = cli._run_fact_tasks(
                tasks, "https://example.invalid", "model", "KEY", 2)
        self.assertEqual([fact["statement"] for fact in result["facts"]], ["kept"])
        self.assertEqual(len(result["stage_errors"]), 1)
        self.assertEqual(len(result["stage_status"]), 2)

    def test_network_both_mode_writes_separate_and_combined_outputs(self):
        class FakeClient:
            def __init__(self, *unused):
                self.usage = [{"total_tokens": 1}]

        def fake_extract(scope, client, qa_mode, checkpoint=None):
            fact = {"id": "f1", "qa_mode": qa_mode,
                    "statement": qa_mode + " fact", "sources": ["e1"]}
            checkpoint("facts.json", [fact])
            return {"facts": [fact], "questions": [], "rejected": [],
                    "stage_errors": [], "stage_status": {"facts": "completed"}}

        def fake_generate(scope, facts, client, max_questions, qa_mode,
                          allowed_types, checkpoint=None, candidate_prefix=None,
                          **unused):
            question = {
                "id": "q1", "qa_mode": qa_mode,
                "type": "single-hop" if qa_mode == "general" else "history_tracking",
                "category": None if qa_mode == "general" else "history_tracking",
                "track": None if qa_mode == "general" else "history_core",
                "question": qa_mode + " question", "difficulty": "easy",
                "fact_ids": ["f1"],
                "answer_points": [{"text": qa_mode + " answer", "sources": ["e1"]}],
                "forbidden_points": [], "status": "approved",
            }
            checkpoint("candidates.json", [question])
            return {"facts": facts, "questions": [question], "rejected": [],
                    "stage_errors": [], "stage_status": {"qa": "completed"}}

        def fake_review(scope, facts, candidates, client, qa_mode, checkpoint=None,
                        review_mode="split"):
            checkpoint("review.json", {"reviews": []})
            return {"facts": facts,
                    "questions": [dict(question, status="approved")
                                  for question in candidates],
                    "rejected": [], "stage_errors": [],
                    "stage_status": {"review": "completed"}}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with patch.object(cli, "ChatClient", FakeClient), \
                    patch.object(cli, "extract_facts", fake_extract), \
                    patch.object(cli, "generate_from_facts", fake_generate), \
                    patch.object(cli, "review_candidates", fake_review):
                status = cli.main([
                    str(EXAMPLE), "--output", str(output), "--qa-mode", "both",
                    "--general-count", "1", "--code-count", "1",
                    "--parallel-workers", "2", "--allow-network",
                    "--review-mode", "single",
                    "--endpoint", "https://example.invalid", "--model", "model",
                ])
            self.assertEqual(status, 0)
            public = json.loads((output / "qa-public.json").read_text())
            general = json.loads((output / "general-qa.json").read_text())
            code = json.loads((output / "code-qa.json").read_text())
            self.assertEqual(public["counts"], {"general": 1, "code": 1})
            self.assertEqual([item["qa_mode"] for item in general["questions"]],
                             ["general"])
            self.assertEqual([item["qa_mode"] for item in code["questions"]], ["code"])
            self.assertEqual(public["status"], "approved")
            self.assertEqual(general["status"], "approved")
            self.assertEqual(code["status"], "approved")
            self.assertTrue(any((output / "stages").iterdir()))
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["questions"]["published"], 2)
            self.assertEqual(manifest["questions"]["generated"], 2)
            self.assertEqual(manifest["questions"]["status"], {"approved": 2})
            self.assertEqual(manifest["questions"]["raw_generated"], 2)
            self.assertEqual(manifest["questions"]["deduplicated"], 2)
            self.assertEqual(manifest["questions"]["limit_rejected"], 0)
            self.assertEqual(manifest["questions"]["path_rejected"], 0)
            # Code scope exploration follows the independent group budget,
            # so it may legitimately produce more chunks than the final
            # one-question-per-track publication quota.
            self.assertGreaterEqual(manifest["chunks"]["unique"], 2)

    def test_unexpected_review_failure_keeps_generated_question(self):
        class FakeClient:
            def __init__(self, *unused):
                self.usage = []

        def fake_generate(scope, facts, client, max_questions, qa_mode,
                          allowed_types, checkpoint=None, candidate_prefix=None,
                          **unused):
            return {
                "facts": facts,
                "questions": [{"id": "q1", "qa_mode": qa_mode,
                               "question": "What was decided?", "fact_ids": ["f1"]}],
                "rejected": [], "stage_errors": [],
                "stage_status": {"qa": "completed"},
            }

        def failing_review(*unused, **unused_kwargs):
            raise RuntimeError("synthetic")

        group = {"scope": {}, "facts": [{"id": "f1"}],
                 "allowed_types": ("single-hop",)}
        with patch.object(cli, "ChatClient", FakeClient), \
                patch.object(cli, "generate_from_facts", fake_generate), \
                patch.object(cli, "review_candidates", failing_review):
            result = cli._run_qa_tasks(
                [(0, "general", group)], "https://example.invalid",
                "model", "KEY", 1, review_mode="single")
        self.assertEqual(len(result["questions"]), 1)
        self.assertEqual(result["questions"][0]["status"], "needs_review")
        self.assertEqual(result["stage_errors"][0]["stage"], "review")

    def test_end_to_end_private_path_is_audit_only(self):
        class FakeClient:
            def __init__(self, *unused):
                self.usage = []

        def fake_extract(scope, client, qa_mode, checkpoint=None):
            return {
                "facts": [{"id": "f1", "qa_mode": qa_mode,
                           "statement": "用户要求保留 yaml 兼容性", "sources": ["e1"]}],
                "questions": [], "rejected": [], "stage_errors": [],
                "stage_status": {"facts": "completed"},
            }

        def fake_generate(scope, facts, client, max_questions, qa_mode,
                          allowed_types, checkpoint=None, candidate_prefix=None,
                          **unused):
            return {
                "facts": facts,
                "questions": [{
                    "id": "q1", "qa_mode": qa_mode, "type": "single-hop",
                    "question": "请检查 /Users/private/project/config.py 的约束",
                    "difficulty": "easy", "difficulty_reason": "direct",
                    "memory_requirement": "恢复约束", "fact_ids": [facts[0]["id"]],
                    "answer_target": "yaml 兼容约束",
                    "answer_points": [{"text": "保留 yaml 兼容性", "sources": ["e1"]}],
                    "forbidden_points": [], "status": "approved",
                }],
                "rejected": [], "stage_errors": [],
                "stage_status": {"qa": "completed"},
            }

        def fake_review(scope, facts, candidates, client, qa_mode, checkpoint=None,
                        review_mode="split"):
            return {"facts": facts,
                    "questions": [dict(item, status="approved") for item in candidates],
                    "rejected": [], "stage_errors": [],
                    "stage_status": {"review": "completed"}}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with patch.object(cli, "ChatClient", FakeClient), \
                    patch.object(cli, "extract_facts", fake_extract), \
                    patch.object(cli, "generate_from_facts", fake_generate), \
                    patch.object(cli, "review_candidates", fake_review):
                status = cli.main([
                    str(EXAMPLE), "--output", str(output), "--qa-mode", "general",
                    "--general-types", "single-hop", "--general-count", "1",
                    "--parallel-workers", "1", "--allow-network",
                    "--review-mode", "single",
                    "--endpoint", "https://example.invalid", "--model", "model",
                ])
            self.assertEqual(status, 0)
            public = json.loads((output / "qa-public.json").read_text())
            audit = json.loads((output / "qa-audit.json").read_text())
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(len(public["questions"]), 1)
            self.assertEqual(len(audit["questions"]), 1)
            self.assertIn("/Users/private/", audit["questions"][0]["question"])
            self.assertIn("~/project/config.py", public["questions"][0]["question"])
            self.assertEqual(manifest["questions"]["published"], 1)
            self.assertEqual(manifest["questions"]["path_rejected"], 0)
            self.assertEqual(manifest["questions"]["path_redacted"], 1)

    def test_no_evidence_error_is_assigned_to_empty_track(self):
        class FakeClient:
            def __init__(self, *unused):
                self.usage = []

        def fake_extract(scope, client, qa_mode, checkpoint=None):
            return {"facts": [], "questions": [], "rejected": [],
                    "stage_errors": [], "stage_status": {"facts": "completed"}}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with patch.object(cli, "_prepare_code_scopes",
                              return_value=([], None)), \
                    patch.object(cli, "ChatClient", FakeClient), \
                    patch.object(cli, "extract_facts", fake_extract):
                status = cli.main([
                    str(EXAMPLE), "--output", str(output), "--qa-mode", "both",
                    "--allow-network", "--endpoint", "https://example.invalid",
                    "--model", "model", "--parallel-workers", "1",
                ])
            self.assertEqual(status, 0)
            code = json.loads((output / "code-qa.json").read_text())
            general = json.loads((output / "general-qa.json").read_text())
            self.assertEqual(code["status"], "failed")
            self.assertEqual(general["status"], "completed_no_questions")
            self.assertEqual(code["stage_errors"][0]["error_type"], "no_evidence")
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["tracks"]["code"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
