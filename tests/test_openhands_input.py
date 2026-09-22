import json
import tempfile
import unittest
from pathlib import Path

from dialogue_benchmark.graph import build_graph
from dialogue_benchmark.normalize import load_dialogue
from dialogue_benchmark.openhands_input import normalize_openhands
from convert_session import convert


class OpenHandsInputTests(unittest.TestCase):
    def rows(self, error=False):
        return [(1, {"id": "c", "kind": "tool_call", "call_id": "call",
                     "tool_name": "file_editor", "action": {
                         "command": "str_replace", "path": "/workspace/candidate/a.py"}}),
                (2, {"id": "r", "kind": "tool_result", "call_id": "call",
                     "tool_name": "file_editor", "observation": {
                         "command": "str_replace", "path": "/workspace/candidate/a.py",
                         "is_error": error, "prev_exist": True,
                         "old_content": "value = 1\n", "new_content": "value = 2\n"}})]

    def test_success_preserves_pair_and_both_versions(self):
        records = normalize_openhands(self.rows())
        self.assertEqual([r["kind"] for r in records], ["call", "result", "observation", "patch"])
        graph = build_graph(records)
        self.assertEqual([v["content"] for v in graph["versions"]], ["value = 1\n", "value = 2\n"])
        self.assertEqual(graph["versions"][1]["previous"], graph["versions"][0]["id"])
        self.assertEqual(records[-1]["original_id"], "r:after")
        self.assertNotIn("old_content", records[1]["text"])
        self.assertNotIn("content", records[-1]["changes"]["/workspace/candidate/a.py"])

    def test_terminal_output_retains_real_line_boundaries(self):
        records = normalize_openhands([(1, {"kind": "tool_result", "tool_name": "terminal",
            "observation": {"content": [{"text": "one\ntwo\n"}]}})])
        self.assertIn("one\ntwo\n", records[0]["text"])
        self.assertNotIn("one\\ntwo", records[0]["text"])

    def test_failed_or_unmatched_edit_does_not_create_version(self):
        self.assertEqual(len(normalize_openhands(self.rows(True))), 2)
        self.assertEqual(len(normalize_openhands(self.rows()[1:])), 1)

    def test_view_output_does_not_claim_complete_file(self):
        rows = self.rows()
        for _, row in rows:
            row.get("action", row.get("observation"))["command"] = "view"
        self.assertEqual(len(normalize_openhands(rows)), 2)

    def test_native_list_and_jsonl_load(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "dialogue.json"
            p.write_text(json.dumps([{"role": "user", "content": "Hello"}]))
            self.assertEqual(load_dialogue(p)[0]["text"], "Hello")
            p = p.with_suffix(".jsonl")
            p.write_text("\n".join(json.dumps(r) for _, r in self.rows()))
            self.assertEqual(len(load_dialogue(p)), 4)

    def test_converter_preserves_native_provenance_and_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "session.jsonl"
            source.write_text("\n".join(json.dumps(r) for _, r in self.rows()))
            expected = load_dialogue(source)
            receipt = convert(source, root / "input")
            actual = load_dialogue(root / "input/dialogue.json")
            self.assertEqual(actual, expected)
            self.assertEqual(actual[-1]["source_line"], 2)
            self.assertEqual(receipt["records"], 4)
            self.assertEqual(build_graph(actual)["versions"], build_graph(expected)["versions"])
            with self.assertRaises(FileExistsError):
                convert(source, root / "input")
