import tempfile
import unittest
from pathlib import Path

from dialogue_benchmark.task_eval.artifacts import copy_tree, fingerprint
from dialogue_benchmark.task_eval.versions import export_change, git, pin_baseline


class TaskVersionTests(unittest.TestCase):
    def test_binary_deletion_new_file_and_mode_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "baseline"
            base.mkdir()
            (base / "delete.txt").write_text("old\n")
            (base / "binary.bin").write_bytes(b"\0\xffold")
            (base / "run.sh").write_text("exit 0\n")
            (base / ".gitignore").write_text("new.txt\n")
            version = pin_baseline(base)
            before = fingerprint(base)
            candidate = root / "candidate"
            copy_tree(base, candidate)
            self.assertFalse((candidate / ".git").exists())
            (candidate / "delete.txt").unlink()
            (candidate / "binary.bin").write_bytes(b"\0\xffnew")
            (candidate / "new.txt").write_text("new without newline")
            (candidate / "run.sh").chmod(0o755)
            receipt = export_change(base, candidate, root / "result")
            self.assertEqual(receipt["base_commit"], version["base_commit"])
            self.assertTrue(receipt["replay_verified"])
            self.assertEqual(set(receipt["changed_files"]),
                             {"delete.txt", "binary.bin", "new.txt", "run.sh"})
            self.assertEqual(fingerprint(base), before)
            self.assertEqual(git(base, "remote"), b"")
            self.assertEqual(git(base, "status", "--porcelain"), b"")

    def test_no_change_and_mode_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            base.mkdir()
            file = base / "a"
            file.write_text("hello")
            pin_baseline(base)
            self.assertEqual(export_change(base, base, root / "result")["changed_files"], [])
            before = fingerprint(base)
            file.chmod(0o755)
            self.assertNotEqual(before, fingerprint(base))
