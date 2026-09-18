import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("candidate_viewer", ROOT / "viewer" / "build_data.py")
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)


class CandidateViewerTests(unittest.TestCase):
    def test_audit_all_states_and_redaction_without_changing_raw(self):
        raw = {"id": "q1", "qa_mode": "code", "question": "See /Users/alice/project/a.py",
               "answer_points": [{"text": "password=example-credential", "sources": ["e1"]}]}
        artifacts = {
            "normalized.json": [{"id": "e1", "kind": "message", "role": "user", "order": 1, "text": "hello"}],
            "graph.json": {"events": [], "versions": []}, "current-graph.json": {},
            "facts.json": [], "evidence-groups.json": [],
            "qa-public.json": {"questions": []},
            "manifest.json": {"code_count": 20, "general_count": 20},
            "qa-audit.json": {"candidate_records": [{"candidate_id": "q1", "current": raw,
                "original": raw, "review_status": "approved", "selection_status": "safety_blocked",
                "revisions": [{"before": raw}], "rejections": []}]}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in artifacts.items():
                (root / name).write_text(json.dumps(value))
            payload = viewer.build(root)
            encoded = json.dumps(payload, ensure_ascii=False)
            self.assertNotIn("example-credential", encoded)
            self.assertNotIn("/Users/alice", encoded)
            self.assertIn("~/project/a.py", encoded)
            self.assertIn("未载入展示页", encoded)
            self.assertEqual(payload["candidate_records"][0]["review_status"], "approved")
            self.assertEqual(payload["targets"], {"general": 20, "code": 20})
            self.assertIn("example-credential", (root / "qa-audit.json").read_text())

    def test_legacy_missing_records_are_labeled_without_inventing_candidates(self):
        run = ROOT / "runs" / "validation-quality-final-20260909-v5"
        if not run.exists():
            self.skipTest("Optional local Lambda artifacts not present")
        payload = viewer.build(run)
        self.assertGreater(len(payload["candidate_records"]), len(payload["questions"]))
        self.assertTrue(all("旧运行" in r["record_note"] for r in payload["candidate_records"]))
        self.assertTrue(any(r["selection_status"] == "over_quota" for r in payload["candidate_records"]))
