import tempfile
import unittest
from pathlib import Path

from dialogue_benchmark.repository_files import list_files, search_text
from dialogue_benchmark.task_eval.selection import query_evidence


class RepositoryFilesTests(unittest.TestCase):
    def make_repo(self, parent):
        root = Path(parent) / ".cache" / "worktrees" / "repo"
        (root / ".git").mkdir(parents=True)
        (root / "pkg").mkdir()
        (root / "build").mkdir()
        (root / ".hidden").mkdir()
        (root / ".gitignore").write_text("build/\n*.log\n/top.txt\n")
        (root / "pkg/api.py").write_text("def absent():\n    return 'dispatcher'\n")
        (root / "pkg/top.txt").write_text("dispatcher nested\n")
        (root / "top.txt").write_text("dispatcher ignored\n")
        (root / "build/out.py").write_text("dispatcher\n")
        (root / "run.log").write_text("dispatcher\n")
        (root / ".hidden/x.py").write_text("dispatcher\n")
        (root / "pkg/data.bin").write_bytes(b"dispatcher\0binary")
        return root

    def test_listing_skips_hidden_and_ignored_entries_under_a_hidden_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_repo(directory)
            self.assertEqual(list_files(root),
                             ["pkg/api.py", "pkg/data.bin", "pkg/top.txt"])
            index = query_evidence({"target": "repo", "op": "read", "path": ".", "offset": 0},
                                   root, [])
            self.assertEqual(index["files"], ["pkg/api.py", "pkg/data.bin", "pkg/top.txt"])

    def test_literal_search_is_sorted_and_skips_binary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_repo(directory)
            self.assertEqual(search_text(root, root, "dispatcher"),
                             ["pkg/api.py:2:    return 'dispatcher'", "pkg/top.txt:1:dispatcher nested"])
            (root / "pkg/long.py").write_text("x" * 600 + "dispatcher\n")
            long_match, = [line for line in search_text(root, root / "pkg", "dispatcher")
                           if line.startswith("pkg/long.py")]
            self.assertTrue(long_match.endswith("[... omitted end of long line]"))


if __name__ == "__main__":
    unittest.main()
