import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dialogue_benchmark.task_eval.artifacts import fingerprint, read, save
from dialogue_benchmark.task_eval.run import evaluate, main, recover_orphan_tasks, validate_resume
from dialogue_benchmark.task_eval.versions import pin_baseline


class TaskResumeTests(unittest.TestCase):
    def test_resume_registers_orphan_with_saved_qa_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            orphan = output / "task-04/author-reference"
            orphan.mkdir(parents=True)
            save(orphan / "qa.json", {"id": "q1", "type": "M6"})
            manifest = {"tasks": []}
            selected = [{"qa": {"id": "q1", "type": "M6"}}]
            recover_orphan_tasks(manifest, selected, output)
            self.assertEqual(manifest["tasks"], [{
                "task": "task-04",
                "status": "pending",
                "qa_id": "q1",
                "type": "M6",
                "recovered_orphan": True,
            }])

    def test_resume_rejects_changed_config_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            baseline.mkdir()
            (baseline / "a.py").write_text("value = 1\n")
            pin_baseline(baseline)
            expected = {
                "source_run": "source", "qa_run": "qa", "baseline": str(baseline),
                "baseline_sha256": fingerprint(baseline), "baseline_version": {"base": 1},
                "config": {"model": "new"}, "execution": {"agent_requests": 1},
                "target": 1, "task_budget": 1, "selection_only": False,
                "selected_qa_ids": ["q1"], "config_sha256": "new-hash",
                "selected_inputs_sha256": ["input-hash"],
            }
            manifest = dict(expected, config={"model": "old"}, tasks=[])
            output = root / "output"
            output.mkdir()
            save(output / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "Resume inputs changed: config"):
                validate_resume(manifest, expected, [], output, baseline)
            self.assertEqual(read(output / "manifest.json"), manifest)

    def test_resume_marks_started_solver_without_result_as_uncertain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            baseline.mkdir()
            (baseline / "a.py").write_text("value = 1\n")
            pin_baseline(baseline)
            frozen = root / "task/frozen"
            frozen.mkdir(parents=True)
            (frozen / "task.md").write_text("Do the thing")
            (frozen / "acceptance.md").write_text("The thing works")
            save(frozen / "acceptance.json", [])
            receipt = {"baseline_sha256": fingerprint(baseline), "spec_sha256": fingerprint(frozen)}
            trial = root / "task/trial-1/workspace/candidate"
            trial.mkdir(parents=True)
            (trial / "a.py").write_text("value = 2\n")
            comparison = {"without_memory": {"result": "failed", "trial": "trial-1"},
                          "with_memory": {"result": "failed", "trial": "trial-2"}}
            with patch("dialogue_benchmark.task_eval.run.run_agent", return_value={"status": "error", "metrics": {}}), \
                 patch("dialogue_benchmark.task_eval.run.run_checks", return_value={"status": "failed", "cases": []}), \
                 patch("dialogue_benchmark.task_eval.run.inspect_acceptance", return_value=({"status": "finished"}, None, {})):
                result = evaluate({"qa": {"id": "q1", "answer_points": []}}, root / "task", baseline,
                                  receipt, {"execution_image": "image"}, {}, 0, resume=True)
            self.assertEqual(result["without_memory"]["result"], "uncertain")
            self.assertEqual(result["without_memory"]["execution_status"], "interrupted")
            self.assertFalse((root / "task/trial-1/private").exists())

    def test_main_resume_keeps_terminal_record_without_construct_or_evaluate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source/workspace/candidate"
            source.mkdir(parents=True)
            (source / "a.py").write_text("value = 1\n")
            baseline = root / "baseline"
            baseline.mkdir()
            (baseline / "a.py").write_text("value = 1\n")
            pin_baseline(baseline)
            output = root / "output"
            common = ["--simulator-path", str(root), "--source-run", str(root / "source"),
                      "--qa-run", str(root), "--env-file", str(root / ".env"),
                      "--baseline", str(baseline), "--output", str(output), "--count", "1"]
            item = {"qa": {"id": "q1", "type": "constraint_followthrough"}}
            with patch("dialogue_benchmark.task_eval.run.qa_inputs", return_value=[item]), \
                 patch("dialogue_benchmark.task_eval.run.configure", return_value={}), \
                 patch("dialogue_benchmark.task_eval.run.preflight_openhands_runtime"), \
                 patch("dialogue_benchmark.task_eval.run.construct", return_value={"accepted": True}), \
                 patch("dialogue_benchmark.task_eval.run.evaluate", return_value={"without_memory": {"result": "failed", "trial": "trial-1"}, "with_memory": {"result": "failed", "trial": "trial-2"}}):
                main(common)
            before = read(output / "manifest.json")
            with patch("dialogue_benchmark.task_eval.run.qa_inputs", return_value=[item]), \
                 patch("dialogue_benchmark.task_eval.run.configure", return_value={}), \
                 patch("dialogue_benchmark.task_eval.run.preflight_openhands_runtime"), \
                 patch("dialogue_benchmark.task_eval.run.construct", side_effect=AssertionError("reran construct")), \
                 patch("dialogue_benchmark.task_eval.run.evaluate", side_effect=AssertionError("reran evaluate")):
                main(common + ["--resume"])
            self.assertEqual(read(output / "manifest.json")["tasks"], before["tasks"])

    def test_resume_replaces_pending_construction_with_new_task_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source/workspace/candidate"
            source.mkdir(parents=True)
            (source / "a.py").write_text("value = 1\n")
            baseline = root / "baseline"
            baseline.mkdir()
            (baseline / "a.py").write_text("value = 1\n")
            pin_baseline(baseline)
            output = root / "output"
            common = ["--simulator-path", str(root), "--source-run", str(root / "source"),
                      "--qa-run", str(root), "--env-file", str(root / ".env"),
                      "--baseline", str(baseline), "--output", str(output), "--count", "1"]
            item = {"qa": {"id": "q1", "type": "constraint_followthrough"}}
            with patch("dialogue_benchmark.task_eval.run.qa_inputs", return_value=[item]), \
                 patch("dialogue_benchmark.task_eval.run.configure", return_value={}), \
                 patch("dialogue_benchmark.task_eval.run.preflight_openhands_runtime"), \
                 patch("dialogue_benchmark.task_eval.run.construct", return_value=None):
                main(common)
            manifest = read(output / "manifest.json")
            manifest["tasks"][0]["status"] = "pending"
            save(output / "manifest.json", manifest)
            roots = []
            def construct(item, root, *args, **kwargs):
                roots.append(Path(root).name)
                return None
            with patch("dialogue_benchmark.task_eval.run.qa_inputs", return_value=[item]), \
                 patch("dialogue_benchmark.task_eval.run.configure", return_value={}), \
                 patch("dialogue_benchmark.task_eval.run.preflight_openhands_runtime"), \
                 patch("dialogue_benchmark.task_eval.run.construct", side_effect=construct):
                main(common + ["--resume"])
            self.assertEqual(roots, ["task-02"])
            resumed = read(output / "manifest.json")
            self.assertEqual([row["task"] for row in resumed["tasks"]], ["task-01", "task-02"])
            self.assertEqual(resumed["tasks"][0]["status"], "interrupted")

    def test_resume_recovers_error_identity_and_avoids_all_existing_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            baseline.mkdir()
            (baseline / "a.py").write_text("value = 1\n")
            pin_baseline(baseline)
            output = root / "output"
            common = ["--simulator-path", str(root), "--source-run", str(root / "source"),
                      "--qa-run", str(root), "--env-file", str(root / ".env"),
                      "--baseline", str(baseline), "--output", str(output), "--count", "3",
                      "--workers", "1"]
            items = [{"qa": {"id": f"q{i}", "type": "constraint_followthrough"}} for i in range(3)]
            roots = []

            def construct(item, task_root, *args, **kwargs):
                task_root = Path(task_root)
                self.assertFalse(task_root.exists())
                roots.append(task_root.name)
                save(task_root / "author-reference/qa.json", item["qa"])
                raise RuntimeError("provider unavailable")

            with patch("dialogue_benchmark.task_eval.run.qa_inputs", return_value=items), \
                 patch("dialogue_benchmark.task_eval.run.configure", return_value={}), \
                 patch("dialogue_benchmark.task_eval.run.preflight_openhands_runtime"), \
                 patch("dialogue_benchmark.task_eval.run.construct", side_effect=construct):
                main(common)
                manifest = read(output / "manifest.json")
                self.assertEqual([row["qa_id"] for row in manifest["tasks"]], ["q0", "q1", "q2"])
                manifest["tasks"][0]["status"] = "pending"
                del manifest["tasks"][1]["qa_id"]
                manifest["tasks"].pop()
                save(output / "manifest.json", manifest)
                (output / "task-04").mkdir()
                roots.clear()
                main(common + ["--resume"])
            self.assertEqual(roots, ["task-05", "task-06", "task-07"])
            resumed = read(output / "manifest.json")
            self.assertEqual(resumed["tasks"][1]["qa_id"], "q1")
            self.assertTrue((output / "task-03/author-reference/qa.json").exists())

    def test_resume_does_not_repeat_rejected_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            baseline.mkdir()
            pin_baseline(baseline)
            output = root / "output"
            common = ["--simulator-path", str(root), "--source-run", str(root / "source"),
                      "--qa-run", str(root), "--env-file", str(root / ".env"),
                      "--baseline", str(baseline), "--output", str(output), "--count", "1"]
            item = {"qa": {"id": "q1", "type": "constraint_followthrough"}}
            with patch("dialogue_benchmark.task_eval.run.qa_inputs", return_value=[item]), \
                 patch("dialogue_benchmark.task_eval.run.configure", return_value={}), \
                 patch("dialogue_benchmark.task_eval.run.preflight_openhands_runtime"), \
                 patch("dialogue_benchmark.task_eval.run.construct", return_value=None) as construct:
                main(common)
                before = read(output / "manifest.json")
                construct.reset_mock()
                main(common + ["--resume"])
                construct.assert_not_called()
            self.assertEqual(read(output / "manifest.json")["tasks"], before["tasks"])


if __name__ == "__main__":
    unittest.main()
