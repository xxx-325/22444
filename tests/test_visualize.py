import json
import tempfile
import unittest
from pathlib import Path

from dialogue_benchmark.graph import build_graph
from dialogue_benchmark.normalize import load_dialogue
from dialogue_benchmark.visualize import build_payload, export_viewer


class ViewerTests(unittest.TestCase):
    def make_run(self, root):
        example = Path(__file__).resolve().parents[1] / "examples" / "dialogue.json"
        records = load_dialogue(example)
        records[0]["text"] = '</script><script>alert("untrusted")</script>'
        records[0]["workspace"] = "/private/example"
        records[1]["changes"]["config.py"]["content"] += '# /private/example/src/module.py\n'
        for name, value in [("normalized.json", records), ("graph.json", build_graph(records))]:
            (root / name).write_text(json.dumps(value))

    def test_export_escapes_and_is_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_run(root)
            output = root / "viewer.html"
            export_viewer(root, output)
            html = output.read_text()
            self.assertNotIn('</script><script>alert("untrusted")</script>', html)
            self.assertNotIn('/private/example', html)
            self.assertIn("connect-src 'none'", html)
            self.assertNotIn('/* VIEWER_', html)
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                export_viewer(root, output)

    def test_snapshot_graphs_do_not_include_future_nodes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_run(root)
            payload = build_payload(root)
            self.assertEqual(payload['states'][payload['snapshot_states']['1']]['nodes'], [])
            self.assertTrue(payload['states'][payload['snapshot_states']['2']]['nodes'])
            self.assertEqual(payload['snapshot_states']['4'], payload['snapshot_states']['5'])
            self.assertEqual(payload['candidates'], [])


if __name__ == '__main__':
    unittest.main()
