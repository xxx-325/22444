"""Resolve the shared episode package without exposing controller artifacts."""

import hashlib
import json
from pathlib import Path

from .normalize import load_dialogue
from .task_eval.artifacts import fingerprint


def load_episode_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "memory-episode-v1":
        raise ValueError("Unsupported episode schema")

    def artifact(section):
        value = Path(manifest[section]["path"])
        target = (path.parent / value).resolve()
        if value.is_absolute() or path.parent not in target.parents:
            raise ValueError("Episode path escapes package: " + section)
        return target

    dialogue, snapshot, config = (artifact(key) for key in
                                   ("dialogue", "snapshot", "control_config"))
    if hashlib.sha256(dialogue.read_bytes()).hexdigest() != manifest["dialogue"]["sha256"]:
        raise ValueError("Public dialogue hash mismatch")
    records = load_dialogue(dialogue)
    if (not records or any(r.get("input_schema") != "model-visible-dialogue-v1" for r in records)
            or records[-1]["original_id"] != manifest["dialogue"]["cutoff_event_id"]):
        raise ValueError("Public dialogue schema or cutoff mismatch")
    if (not snapshot.is_dir() or any(p.name == ".git" or p.is_symlink() for p in snapshot.rglob("*"))
            or manifest["snapshot"].get("hash_algorithm") != "relative-path-executable-content-v1"
            or fingerprint(snapshot) != manifest["snapshot"]["sha256"]):
        raise ValueError("Snapshot hash or export boundary mismatch")
    if not config.is_file():
        raise ValueError("Missing control config")
    return {"manifest": manifest, "dialogue": dialogue, "snapshot": snapshot,
            "control_config": config}
