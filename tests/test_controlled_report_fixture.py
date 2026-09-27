import json
import unittest
from pathlib import Path

from examples.controlled_report_fixture import build_repo, write_dialogue, write_events


class ControlledReportFixtureTests(unittest.TestCase):
    def test_has_a_clean_git_baseline(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            repository = build_repo(Path(directory) / "repo")
            self.assertTrue((repository / "src/reporting/core.py").is_file())
            self.assertTrue((repository / ".git").is_dir())


    def test_external_event_has_public_source_and_later_use(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dialogue = root / "dialogue.json"
            events = root / "external-events.json"
            write_dialogue(dialogue)
            write_events(events)
            records = json.loads(dialogue.read_text(encoding="utf-8"))["records"]
            event = json.loads(events.read_text(encoding="utf-8"))["events"][0]
            positions = {row["id"]: index for index, row in enumerate(records)}
            self.assertLess(positions[event["source_ids"][0]], positions[event["used_by"][0]])
            self.assertTrue(any(row["id"] == event["source_ids"][0] and row["role"] == "user"
                                for row in records))
