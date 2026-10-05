import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dialogue_benchmark.episode_input import load_episode_manifest
from dialogue_benchmark.fact_index import build_evidence_groups
from dialogue_benchmark.general import build_general_scope
from dialogue_benchmark.graph import build_graph
from dialogue_benchmark.normalize import load_dialogue, resolve_source_events
from dialogue_benchmark.openhands_input import normalize_openhands
from dialogue_benchmark.task_eval.artifacts import fingerprint, save
from run_episode import main as run_episode


def public_rows():
    events = [
        {"id": "user", "kind": "user", "text": "Keep cache.py compatible."},
        {"id": "call", "kind": "tool_call", "call_id": "c", "tool_name": "file_editor",
         "action": {"command": "str_replace", "path": "/workspace/candidate/cache.py"}},
        {"id": "result", "kind": "tool_result", "call_id": "c", "tool_name": "file_editor",
         "text": "File updated successfully", "is_error": False,
         "observation": {"old_content": "secret old", "new_content": "secret new"}},
    ]
    return [(i, dict(row, schema="model-visible-dialogue-v1", sequence=i, timestamp=i))
            for i, row in enumerate(events, 1)]


class PublicEpisodeTests(unittest.TestCase):
    def test_private_metadata_is_not_public_evidence(self):
        records = normalize_openhands(public_rows())
        self.assertEqual(len(records), 3)
        self.assertEqual(records[-1]["text"], "File updated successfully")
        self.assertNotIn("secret", json.dumps(records))
        self.assertEqual(build_graph(records)["versions"], [])
        scope = build_general_scope(records, 3)
        self.assertEqual(len(scope["dialogue"]), 3)
        self.assertEqual({r["stage_id"] for r in scope["dialogue"]}, {"stage-1"})
        self.assertEqual(scope["dialogue"][-1]["source_kind"], "tool")
        self.assertEqual(resolve_source_events(records, ["result"]), {"e3"})

    def test_delivered_create_and_replace_recover_versions_without_private_metadata(self):
        events = [
            {"id": "c1", "kind": "tool_call", "call_id": "c1", "tool_name": "file_editor",
             "action": {"command": "create", "path": "/workspace/candidate/a.py", "file_text": "v = 1\n"}},
            {"id": "r1", "kind": "tool_result", "call_id": "c1", "tool_name": "file_editor",
             "text": "File updated successfully"},
            {"id": "c2", "kind": "tool_call", "call_id": "c2", "tool_name": "file_editor",
             "action": {"command": "str_replace", "path": "/workspace/candidate/a.py",
                        "old_str": "v = 1", "new_str": "v = 2"}},
            {"id": "r2", "kind": "tool_result", "call_id": "c2", "tool_name": "file_editor",
             "text": "File updated successfully", "observation": {"new_content": "PRIVATE"}},
        ]
        rows = [(i, dict(row, schema="model-visible-dialogue-v1", sequence=i))
                for i, row in enumerate(events, 1)]
        records = normalize_openhands(rows)
        graph = build_graph(records)
        self.assertEqual([v["content"] for v in graph["versions"]], ["v = 1\n", "v = 2\n"])
        self.assertEqual(graph["versions"][1]["previous"], graph["versions"][0]["id"])
        self.assertNotIn("PRIVATE", json.dumps(records))
        self.assertEqual(records[-1]["original_id"], "r2")
        self.assertEqual(resolve_source_events(records, ["r2"]), {"e5", "e6"})

    def test_public_edit_without_a_complete_base_stays_unknown(self):
        rows = public_rows()
        rows[1][1]["action"].update(old_str="v = 1", new_str="v = 2")
        records = normalize_openhands(rows)
        graph = build_graph(records)
        self.assertEqual(len(graph["versions"]), 1)
        self.assertEqual(graph["versions"][0]["status"], "unknown")
        self.assertIsNone(graph["versions"][0]["content"])
        self.assertEqual(graph["diagnostics"][0]["reason"], "Missing complete base")

    def test_failed_or_ambiguous_public_edit_does_not_establish_a_patch(self):
        for change in ({"is_error": True}, {"text": "Error: File updated successfully"},
                       {"text": "The model says File updated successfully"}):
            rows = public_rows()
            rows[1][1]["action"].update(old_str="v = 1", new_str="v = 2")
            rows[2][1].update(change)
            self.assertEqual(build_graph(normalize_openhands(rows))["versions"], [])

    def test_identity_sequence_and_call_boundary(self):
        for field, value in (("sequence", 7), ("id", "user"), ("call_id", "unknown")):
            rows = public_rows()
            rows[-1][1][field] = value
            with self.assertRaises(ValueError):
                normalize_openhands(rows)
        with self.assertRaises(ValueError):
            resolve_source_events(normalize_openhands(public_rows())[:2], ["result"])

    def test_seed_fact_is_not_lost_to_top_root_limit(self):
        records = [{"id": "e%d" % i, "kind": "message", "role": "user", "order": i,
                    "text": "Object%d required decision." % i} for i in range(1, 35)]
        scope = build_general_scope(records, 34)
        facts = [{"id": "f%d" % i, "statement": "Object%d required decision." % i,
                  "sources": ["e%d" % i]} for i in range(1, 35)]
        groups = build_evidence_groups(facts, [scope], "general", ["constraint_followthrough"], 1,
                                       seed_sources={"e19"})
        self.assertIn("f19", [f["id"] for f in groups[0]["facts"]])
        self.assertEqual(build_evidence_groups(facts, [scope], "general", ["constraint_followthrough"], 1,
                                              seed_sources={"missing"}), [])

    def test_package_hash_cutoff_and_path_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dialogue = root / "dialogue.jsonl"
            dialogue.write_text("\n".join(json.dumps(row) for _, row in public_rows()))
            (root / "snapshot").mkdir()
            (root / "snapshot/a.py").write_text("x = 1\n")
            save(root / "control-config.json", {})
            manifest = {"schema": "memory-episode-v1", "dialogue": {
                "path": dialogue.name, "sha256": hashlib.sha256(dialogue.read_bytes()).hexdigest(),
                "cutoff_event_id": "result"}, "snapshot": {"path": "snapshot",
                "sha256": fingerprint(root / "snapshot"),
                "hash_algorithm": "relative-path-executable-content-v1"},
                "control_config": {"path": "control-config.json"}}
            save(root / "manifest.json", manifest)
            self.assertEqual(load_episode_manifest(root / "manifest.json")["dialogue"], dialogue.resolve())
            config = {"judge": {"base_url": "https://example.invalid", "model": "test", "key_env": "KEY"}}
            def qa(args):
                self.assertEqual(args[args.index("--source-event") + 1], "result")
                output = Path(args[args.index("--output") + 1])
                save(output / "manifest.json", {"input_sha256": hashlib.sha256(Path(args[0]).read_bytes()).hexdigest()})
                save(output / "qa-public.json", {"questions": []})
                return 0
            def tasks(args):
                self.assertEqual(Path(args[args.index("--control-config") + 1]), (root / "control-config.json").resolve())
                self.assertEqual(fingerprint(Path(args[args.index("--baseline") + 1])), manifest["snapshot"]["sha256"])
                return 0
            with patch("run_episode.configure", return_value=config) as configure, \
                 patch("run_episode.generate_qa", side_effect=qa), \
                 patch("run_episode.run_tasks", side_effect=tasks), patch("run_episode.render"):
                self.assertEqual(run_episode(["--episode-manifest", str(root / "manifest.json"),
                    "--simulator-path", str(root), "--env-file", str(root / "unused"),
                    "--output", str(root / "run"), "--source-event", "result"]), 0)
                self.assertEqual(configure.call_args.kwargs["control_config"], (root / "control-config.json").resolve())
            save(root / "dialogue.json", {"schema": "model-visible-dialogue-v1",
                                         "events": [row for _, row in public_rows()]})
            self.assertEqual(load_dialogue(root / "dialogue.json"), load_dialogue(dialogue))
            for change in ({"cutoff_event_id": "call"}, {"path": "../elsewhere"}, {"sha256": "bad"}):
                broken = dict(manifest, dialogue=dict(manifest["dialogue"], **change))
                save(root / "manifest.json", broken)
                with self.assertRaises(ValueError):
                    load_episode_manifest(root / "manifest.json")

            events = root / "external-events.json"
            save(events, {"version": 1, "events": []})
            manifest["external_events"] = {"path": events.name,
                "sha256": hashlib.sha256(events.read_bytes()).hexdigest()}
            save(root / "manifest.json", manifest)
            self.assertEqual(load_episode_manifest(root / "manifest.json")["external_events"], events.resolve())
            with patch("run_episode.configure", return_value=config), \
                 patch("run_episode.generate_qa", side_effect=qa) as generate, \
                 patch("run_episode.render"):
                self.assertEqual(run_episode(["--episode-manifest", str(root / "manifest.json"),
                    "--simulator-path", str(root), "--env-file", str(root / "unused"),
                    "--output", str(root / "external-run"), "--source-event", "result",
                    "--qa-source", "external"]), 0)
                args = generate.call_args.args[0]
                self.assertEqual(args[args.index("--external-events") + 1], str(events.resolve()))
            different = root / "different-events.json"
            save(different, {"version": 1, "events": [{"id": "unrelated"}]})
            with patch("run_episode.configure") as configure, self.assertRaises(SystemExit):
                run_episode(["--episode-manifest", str(root / "manifest.json"),
                    "--simulator-path", str(root), "--env-file", str(root / "unused"),
                    "--output", str(root / "mismatch-run"), "--qa-source", "external",
                    "--external-events", str(different)])
            configure.assert_not_called()
            for change in ({"sha256": "wrong"}, {"path": "../external-events.json"}):
                save(root / "manifest.json", dict(manifest,
                    external_events=dict(manifest["external_events"], **change)))
                with self.assertRaises(ValueError):
                    load_episode_manifest(root / "manifest.json")

    def test_shared_file_does_not_alone_establish_public_relation(self):
        from dialogue_benchmark.fact_index import build_evidence_index, _relation
        records = [{"id": "e1", "order": 1, "kind": "message", "role": "user",
                    "text": "app.py timeout failed.", "input_schema": "model-visible-dialogue-v1"},
                   {"id": "e2", "order": 2, "kind": "message", "role": "user",
                    "text": "app.py colours changed.", "input_schema": "model-visible-dialogue-v1"}]
        facts = [{"id": "f1", "statement": records[0]["text"], "sources": ["e1"]},
                 {"id": "f2", "statement": records[1]["text"], "sources": ["e2"]}]
        index = build_evidence_index(facts, [build_general_scope(records, 2)], "general")
        self.assertFalse(_relation(*index["infos"], "general")[1])
