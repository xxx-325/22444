import gzip
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from dialogue_benchmark.storage import load, save_projection
from dialogue_benchmark.task_eval.artifacts import copy_tree, fingerprint, save
from dialogue_benchmark.task_eval.retention import (
    compact_run, compact_clarifications, docker_inventory, pending_executions, save_trace)


class RetentionTests(unittest.TestCase):
    def test_projection_roundtrip_keeps_variants_with_the_same_source_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = {"id": "v1", "content": "first fragment"}
            b = {"id": "v1", "content": "second fragment"}
            groups = [{"scope": {"versions": [a, b]}}, {"scope": {"versions": [a]}}]
            save_projection(root / "groups.json", groups)
            self.assertEqual(load(root / "groups.json"), groups)
            self.assertEqual(len(load(root / "evidence-records.json")), 2)

    def make_agent(self, root):
        save(root / "workspace/candidate/a.py", "code")
        event = {"id": "a1", "kind": "ActionEvent", "thought": "Inspect the earlier failure",
                 "tool_name": "terminal", "action": {"command": "cat a.py"}}
        path = root / "private/agent/outbox/events.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(event) + "\n")
        provider = root / "private/agent/provider.jsonl"
        provider.write_text(json.dumps({"kind": "request", "body": "repeated full context"}) + "\n" +
                            json.dumps({"kind": "response", "usage": {"total_tokens": 12},
                                        "output": {"choices": [{"message": {"content": "Done"}}]}}) + "\n")
        save(root / "trajectory.json", [event])
        save(root / "version.json", {"candidate_sha256": fingerprint(root / "workspace/candidate")})
        save(root / "result.json", {"status": "finished"})

    def test_trace_preserves_model_outputs_usage_and_event_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_agent(root)
            output = save_trace(root)
            with gzip.open(output, "rt") as stream:
                rows = [json.loads(line) for line in stream]
            self.assertEqual(rows[0]["value"]["id"], "a1")
            self.assertIn("thought", rows[0]["value"])
            self.assertEqual(rows[1]["value"]["usage"]["total_tokens"], 12)
            self.assertNotIn("repeated full context", str(rows))

    def test_inventory_selects_only_recorded_containers_with_owned_mounts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            own, other = "a" * 64, "b" * 64
            save(root / "private/execution/environment.json", {"container_id": own})
            listing = "\n".join(json.dumps({"ID": cid[:12], "Names": "session-tool-" + cid})
                                 for cid in (own, other))
            details = [{"Id": own, "Name": "/session-tool-a", "State": {"Status": "exited"},
                        "Mounts": [{"Type": "bind", "Source": "/host_mnt" + str(root / "workspace")}],
                        "NetworkSettings": {"Networks": {"session-test": {}}}}]
            with patch("dialogue_benchmark.task_eval.retention.subprocess.run", side_effect=[
                SimpleNamespace(stdout=listing), SimpleNamespace(stdout=json.dumps(details))]) as command:
                result = docker_inventory(root)
            self.assertEqual([r["id"] for r in result["containers"]], [own])
            self.assertNotIn(other[:12], command.call_args.args[0])

    def test_compaction_preserves_final_codes_inputs_traces_and_failure_reasons(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save(root / "pipeline.json", {"status": "completed"})
            save(root / "baseline/a.py", "baseline")
            baseline_hash = fingerprint(root / "baseline")
            save(root / "baseline.json", {"content_sha256": baseline_hash})
            qa = root / "qa"
            save(qa / "qa-public.json", {"questions": [{"id": "q1"}]})
            save(qa / "candidates.json", [{"id": "q1"}])
            save(qa / "qa-audit.json", {"candidates": [{"id": "q1"}]})
            save(qa / "stages/one-raw-candidates.json", {"questions": [{"id": "q1"}]})
            save(qa / "stages/one-qa-input.json", {"prompt": "Original evidence"})
            save(qa / "stages/rejected-raw-candidates.json", {"questions": [{"id": "q2"}]})
            save(qa / "stages/rejected-qa-input.json", {"prompt": "Evidence for rejected question"})
            save(qa / "stages/unneeded.json", {})
            save(qa / "evidence-groups.json", [])
            save(qa / "scopes.json", {"repeated": "source"})
            save(qa / "batch-001-audit.json", {})
            task = root / "tasks/task-01"
            save(task / "frozen.json", {"accepted_attempt": 1})
            save(task / "construction.json", [{"attempt": 0, "reason": "incorrect test"},
                                               {"attempt": 1, "accepted": True}])
            bad = task / "construction-00/author/workspace/checks/task.md"
            bad.parent.mkdir(parents=True)
            bad.write_text("Earlier requirement")
            (bad.parent / "test_acceptance.py").write_text("assert incorrect_assumption")
            save(root / "tasks/manifest.json", {"tasks": [{"task": "task-01", "status": "evaluated",
                "comparison": {"without_memory": {"trial": "trial-1"}, "with_memory": {"trial": "trial-2"}}}]})
            accepted = task / "construction-01"
            save(task / "frozen/history.json", {"events": ["public history"]})
            save(task / "author-reference/history.json", {"events": ["public history"]})
            check_xml = task / "checkpoint-recovery/baseline-checks/workspace/experiments/receipt.xml"
            check_xml.parent.mkdir(parents=True)
            check_xml.write_text("<testsuite/>")
            for agent in [accepted / role for role in ("author", "reference-solver", "validator")] + [
                task / "trial-1", task / "trial-1/judge", task / "trial-2", task / "trial-2/judge"]:
                self.make_agent(agent)
            empty = {"containers": [], "volumes": [], "networks": []}
            with patch("dialogue_benchmark.task_eval.retention.docker_inventory", return_value=empty):
                with patch("dialogue_benchmark.task_eval.retention.release_docker", return_value={"errors": [{"detail": "busy"}]}):
                    deferred = compact_run(root)
                self.assertEqual(deferred["reason"], "resource_release_failed")
                self.assertTrue((task / "trial-1/private/agent").exists())
                self.assertTrue((qa / "stages/unneeded.json").exists())
                receipt = compact_run(root)
            self.assertEqual(receipt["retained_code_versions"], 3)
            self.assertEqual(fingerprint(root / "baseline"), baseline_hash)
            self.assertFalse((task / "construction-00").exists())
            self.assertFalse((task / "trial-1/judge/workspace/candidate").exists())
            self.assertTrue((task / "trial-1/workspace/candidate/a.py").exists())
            self.assertTrue((accepted / "reference-solver/workspace/candidate/a.py").exists())
            self.assertTrue((task / "trial-1/trace.jsonl.gz").exists())
            self.assertEqual(load(task / "trial-1/trajectory.json")[0]["id"], "a1")
            self.assertIn("Earlier requirement", (task / "construction-summary.json").read_text())
            self.assertIn("incorrect_assumption", (task / "construction-summary.json").read_text())
            self.assertFalse((task / "construction.json").exists())
            self.assertFalse((task / "author-reference/history.json").exists())
            self.assertTrue((task / "frozen/history.json").exists())
            self.assertTrue((qa / "stages/one-qa-input.json").exists())
            self.assertTrue((qa / "stages/rejected-qa-input.json").exists())
            self.assertFalse((qa / "stages/unneeded.json").exists())
            self.assertFalse((qa / "candidates.json").exists())
            self.assertTrue((qa / "qa-audit.json").exists())
            self.assertTrue((task / "checkpoint-recovery/baseline-checks/workspace").exists())
            self.assertTrue((task / "checkpoint-recovery/baseline-checks/workspace/experiments/receipt.xml").exists())
            self.assertEqual(compact_run(root), receipt)

    def test_compaction_is_deferred_before_touching_unfinished_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save(root / "pipeline.json", {"status": "completed"})
            agent = root / "tasks/task-01/trial-1"
            self.make_agent(agent)
            save(agent / "result.json", {"status": "error"})
            checks = root / "tasks/task-01/trial-2/checks"
            save(checks / "private/environment.json", {"status": "ready"})
            save(checks / "execution.json", {"exit_code": 124})
            with patch("dialogue_benchmark.task_eval.retention.release_docker") as release:
                receipt = compact_run(root)
            self.assertEqual(receipt["status"], "deferred")
            self.assertEqual(len(receipt["pending"]), 2)
            self.assertTrue((agent / "private/agent").exists())
            self.assertFalse((agent / "trace.jsonl.gz").exists())
            release.assert_not_called()
            save(agent / "result.json", {"status": "finished"})
            save(checks / "execution.json", {"exit_code": 1})
            self.assertEqual(pending_executions(root), [])
            save(root / "pipeline.json", {"status": "running"})
            with self.assertRaises(ValueError):
                compact_run(root)

    def test_clarification_audit_shares_context_and_preserves_exact_rounds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exchanges = [{"question": "Earlier rule?", "reply": "Keep blanks", "delivered": True},
                         {"question": "Done", "reply": "none", "delivered": False}]
            save(root / "clarifications.json", exchanges)
            for number in (1, 2):
                step = root / "clarification" / str(number)
                save(step / "input.json", {"prompt": "Answer actual questions", "payload": {
                    "message": exchanges[number - 1]["question"], "supplied_history": {"events": ["old rule"]},
                    "exchange": exchanges[:number - 1]}})
                save(step / "response-text.json", ["original answer %d" % number])
                save(step / "usage.json", [{"total_tokens": 17}])
            self.assertEqual(compact_clarifications(root), root / "clarification")
            audit = load(root / "clarification-audit.json")
            self.assertEqual(audit["context"]["supplied_history"], {"events": ["old rule"]})
            self.assertEqual(audit["rounds"][1]["input.json"]["exchange_prefix_count"], 1)
            self.assertNotIn("old rule", str(audit["rounds"]))
            self.assertEqual(audit["rounds"][1]["response-text.json"], ["original answer 2"])
            self.assertEqual(audit["rounds"][1]["usage.json"][0]["total_tokens"], 17)

    def test_trace_is_linked_not_embedded_in_html(self):
        from dialogue_benchmark.task_eval.report import write_report
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent = root / "task-01/trial-1"
            self.make_agent(agent)
            save_trace(agent)
            write_report(root, {"tasks": [{"task": "task-01", "status": "evaluated",
                "comparison": {"without_memory": {"trial": "trial-1", "result": "passed"}}}]})
            page = (root / "report.html").read_text()
            self.assertIn('href="task-01/trial-1/trace.jsonl.gz"', page)
            self.assertNotIn("Inspect the earlier failure", page)

    def test_new_copies_exclude_caches_without_changing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save(root / "source/.mypy_cache/cache.db", "cache")
            save(root / "source/a.py", "code")
            before = fingerprint(root / "source")
            copy_tree(root / "source", root / "copy")
            self.assertFalse((root / "copy/.mypy_cache").exists())
            self.assertEqual(fingerprint(root / "source"), before)
