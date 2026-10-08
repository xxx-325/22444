import json
import re
import unittest
from pathlib import Path

from dialogue_benchmark.collection import validate_plan


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "examples/collection-five/long-dialogue-three.json"
PREPARED = {
    "release-operations": "prepared-release.json",
    "document-pipeline": "prepared-document.json",
    "dependency-planner": "prepared-dependency.json",
}
_SHA = re.compile(r"^[0-9a-f]{40}$")


class LongDialogueCaseTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))

    def test_three_case_plan_has_the_round_gate_and_is_accepted(self):
        validate_plan(self.plan)
        self.assertEqual(self.plan["dialogue_quality"]["min_user_code_rounds"], 50)
        self.assertEqual(self.plan["dialogue_quality"]["min_visible_messages"], 100)
        self.assertEqual({project["id"] for project in self.plan["projects"]}, set(PREPARED))
        for project in self.plan["projects"]:
            self.assertEqual(project["prepared_config"], PREPARED[project["id"]])
            self.assertEqual(len(project["scenarios"]), 1)

    def test_prepared_configs_record_a_repository_base_and_commit_chain(self):
        for project in self.plan["projects"]:
            path = ROOT / "examples/collection-five" / project["prepared_config"]
            self.assertTrue(path.is_file(), path)
            prepared = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(prepared["projects"][0]["id"], project["id"])
            config_path = (path.parent / prepared["projects"][0]["prepared_config"]).resolve()
            if not config_path.is_file():
                self.skipTest("prepared run artifacts are not checked out")
            config = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertTrue(Path(config["repository"]).is_dir())
            self.assertRegex(config["base"], _SHA)
            self.assertGreaterEqual(len(config["tasks"]), 6)
            self.assertTrue(all(_SHA.fullmatch(task["commit"]) for task in config["tasks"]))

    def test_public_case_text_does_not_contain_controller_fields(self):
        forbidden = re.compile(r"(?:source_ids|used_by|private[_ -]?history|judge[_ -]?summary|qa[_ -]?answer|internal[_ -]?id)", re.I)
        for project in self.plan["projects"]:
            scenario = project["scenarios"][0]
            public_text = "\n".join((scenario.get("brief", ""), scenario.get("runtime_conditions", "")))
            self.assertIsNone(forbidden.search(public_text), public_text)


if __name__ == "__main__":
    unittest.main()
