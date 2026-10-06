import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from dialogue_benchmark.task_eval.artifacts import read, save
from dialogue_benchmark.task_eval.run import main
from dialogue_benchmark.task_eval.versions import pin_baseline


class TaskSchedulingTests(unittest.TestCase):
    def run_tasks(self, root, construct, *, count=1, workers=2, selection_only=False):
        source = root / "source/workspace/candidate"
        source.mkdir(parents=True)
        (source / "a.py").write_text("value = 1\n")
        items = [{"qa": {"type": "constraint_followthrough", "id": "q%d" % n}}
                 for n in range(5)]
        args = ["--simulator-path", str(root), "--source-run", str(root / "source"),
                "--qa-run", str(root), "--env-file", str(root / ".env"),
                "--output", str(root / "output"), "--count", str(count),
                "--workers", str(workers), "--task-budget", "5"]
        if selection_only:
            args.append("--selection-only")
        with patch("dialogue_benchmark.task_eval.run.qa_inputs", return_value=items), \
             patch("dialogue_benchmark.task_eval.run.configure", return_value={}), \
             patch("dialogue_benchmark.task_eval.run.preflight_openhands_runtime"), \
             patch("dialogue_benchmark.task_eval.run.construct", side_effect=construct), \
             patch("dialogue_benchmark.task_eval.run.evaluate", return_value={}), \
             patch("dialogue_benchmark.task_eval.run.write_report"):
            return main(args)

    def test_finished_task_refills_slot_before_other_task_completes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            second_started, third_started = threading.Event(), threading.Event()
            lock = threading.Lock()
            active = peak = 0
            submitted = []
            refilled = []

            def construct(item, task_root, *args, **kwargs):
                nonlocal active, peak
                name = item["qa"]["id"]
                with lock:
                    active += 1
                    peak = max(peak, active)
                    submitted.append(name)
                try:
                    if name == "q0":
                        second_started.wait(3)
                        save(task_root / "construction.json", [{"status": "stop"}])
                        return None
                    if name == "q1":
                        second_started.set()
                        refilled.append(third_started.wait(3))
                    if name == "q2":
                        third_started.set()
                    return {"accepted": True}
                finally:
                    with lock:
                        active -= 1

            self.run_tasks(root, construct, count=2, workers=2)
            self.assertEqual(refilled, [True])
            self.assertEqual(set(submitted), {"q0", "q1", "q2"})
            self.assertEqual(peak, 2)
            manifest = read(root / "output/manifest.json")
            self.assertEqual(manifest["completed"], 2)
            self.assertEqual(manifest["stop_reason"], "target_met")
            for row in manifest["tasks"]:
                progress = read(root / "output" / row["task"] / "progress.json")
                self.assertEqual(progress["stage"], "terminal")
                self.assertEqual(progress["status"], row["status"])
                self.assertIsInstance(progress["finished_at"], float)

    def test_task_returns_and_errors_save_terminal_progress(self):
        for status in ("not_admitted", "stop", "pending", "error", "qualified", "evaluated"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)

                def construct(item, task_root, *args, **kwargs):
                    save(task_root / "progress.json", {"stage": "author"})
                    if status == "error":
                        raise RuntimeError("worker exited")
                    if status in {"qualified", "evaluated"}:
                        return {"accepted": True}
                    if status != "not_admitted":
                        save(task_root / "construction.json", [{"status": status}])
                    return None

                self.run_tasks(root, construct, selection_only=status == "qualified")
                manifest = read(root / "output/manifest.json")
                for row in manifest["tasks"]:
                    progress = read(root / "output" / row["task"] / "progress.json")
                    self.assertEqual(progress["stage"], "terminal")
                    self.assertEqual(progress["status"], status)
                    if status == "error":
                        self.assertEqual(progress["error_type"], "RuntimeError")
                        self.assertEqual(progress["detail"], "worker exited")

    def test_interruption_saves_terminal_progress_before_propagating(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(KeyboardInterrupt):
                self.run_tasks(root, lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))
            progress = read(root / "output/task-01/progress.json")
            self.assertEqual(progress["stage"], "terminal")
            self.assertEqual(progress["status"], "interrupted")
            self.assertEqual(progress["error_type"], "KeyboardInterrupt")

    def test_explicit_baseline_records_actual_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "controlled-repo"
            baseline.mkdir()
            (baseline / "a.py").write_text("value = 1\n")
            pin_baseline(baseline)
            with patch("dialogue_benchmark.task_eval.run.qa_inputs", return_value=[
                    {"qa": {"type": "constraint_followthrough", "id": "q1"}}]), \
                 patch("dialogue_benchmark.task_eval.run.configure", return_value={}), \
                 patch("dialogue_benchmark.task_eval.run.preflight_openhands_runtime"), \
                 patch("dialogue_benchmark.task_eval.run.construct", return_value=None):
                main(["--simulator-path", str(root), "--source-run", str(root / "dialogue"),
                      "--qa-run", str(root), "--env-file", str(root / ".env"),
                      "--baseline", str(baseline), "--output", str(root / "output"), "--count", "1"])
            self.assertEqual(read(root / "output/baseline.json")["source"], str(baseline.resolve()))
            manifest = read(root / "output/manifest.json")
            self.assertTrue(manifest["evaluator_version"]["package_sha256"])
            self.assertEqual(manifest["execution"]["agent_requests"], 80)
            self.assertEqual(manifest["execution"]["agent_seconds"], 1200)
            self.assertEqual(manifest["config"]["model_request_chars"], 60000)

    def test_configured_request_limit_reaches_construction_and_saved_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "source/workspace/candidate"
            baseline.mkdir(parents=True)
            (baseline / "a.py").write_text("value = 1\n")
            with patch("dialogue_benchmark.task_eval.run.qa_inputs", return_value=[
                    {"qa": {"type": "constraint_followthrough", "id": "q1"}}]), \
                 patch("dialogue_benchmark.task_eval.run.configure", return_value={}), \
                 patch("dialogue_benchmark.task_eval.run.preflight_openhands_runtime"), \
                 patch("dialogue_benchmark.task_eval.run.construct", return_value=None) as construct:
                main(["--simulator-path", str(root), "--source-run", str(root / "source"),
                      "--qa-run", str(root), "--env-file", str(root / ".env"),
                      "--output", str(root / "output"), "--count", "1",
                      "--model-request-chars", "96000"])
            config = construct.call_args.args[3]
            self.assertEqual(config["model_request_chars"], 96000)
            self.assertEqual(read(root / "output/manifest.json")["config"], config)

    def test_failed_requirement_does_not_consume_completed_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source/workspace/candidate"
            source.mkdir(parents=True)
            (source / "a.py").write_text("value = 1\n")
            items = [{"qa": {"type": "constraint_followthrough", "id": "q%d" % n}, "original_candidate": {"evidence_group_id": "g%d" % n}}
                     for n in range(5)]
            def construct(item, *args, **kwargs):
                return None if item["qa"]["id"] == "q0" else {"accepted": True}
            with patch("dialogue_benchmark.task_eval.run.qa_inputs", return_value=items), \
                 patch("dialogue_benchmark.task_eval.run.configure", return_value={}), \
                 patch("dialogue_benchmark.task_eval.run.preflight_openhands_runtime"), \
                 patch("dialogue_benchmark.task_eval.run.construct", side_effect=construct) as author, \
                 patch("dialogue_benchmark.task_eval.run.evaluate", return_value={}) as evaluate:
                main(["--simulator-path", str(root), "--source-run", str(root / "source"),
                      "--qa-run", str(root), "--env-file", str(root / ".env"),
                      "--output", str(root / "output"), "--count", "2", "--workers", "2"])
            manifest = read(root / "output/manifest.json")
            self.assertEqual(author.call_count, 3)
            self.assertEqual(evaluate.call_count, 2)
            self.assertEqual(manifest["completed"], 2)
            self.assertEqual(manifest["stop_reason"], "target_met")
            self.assertEqual(manifest["shortfall"], 0)
            self.assertTrue((root / "output/report.html").is_file())
