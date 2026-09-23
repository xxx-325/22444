"""Private task artifacts and immutable source snapshots."""

import hashlib
import json
import re
import shutil
from pathlib import Path

from ..storage import load
from ..protocol import QA_TYPES


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.chmod(0o600)
    temp.replace(path)


def read(path):
    return load(path)


def copy_tree(source, target, *, include_caches=False):
    source, target = Path(source), Path(target)
    if any(p.is_symlink() for p in source.rglob("*")):
        raise ValueError("Snapshot contains a symlink")
    ignored = [".git", "__pycache__", ".pytest_cache", ".venv", "*.pyc"]
    if not include_caches:
        ignored += [".mypy_cache", ".ruff_cache"]
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(*ignored))


def fingerprint(root):
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and not any(x in {"__pycache__", ".pytest_cache", ".git"}
                                      for x in path.parts):
            digest.update(str(path.relative_to(root)).encode() + b"\0")
            digest.update(b"x" if path.stat().st_mode & 0o111 else b"-")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def write_diff(base, candidate, output):
    from .versions import export_change
    output = Path(output)
    receipt = export_change(base, candidate, output.parent)
    if output.name != receipt["patch"]:
        (output.parent / receipt["patch"]).rename(output)
        receipt["patch"] = output.name
        save(output.parent / "version.json", receipt)
    return receipt["changed_files"]


def qa_inputs(qa_run):
    """Join approved questions to the actual saved generation request, never reconstruct it."""
    qa_run = Path(qa_run)
    public = read(qa_run / "qa-public.json")["questions"]
    if any(question.get("type") not in QA_TYPES for question in public):
        raise ValueError("QA input must use the six memory-purpose types; regenerate older QA")
    normalized_path = qa_run / "normalized.json"
    normalized = read(normalized_path) if normalized_path.exists() else []
    public_records = (normalized if normalized and all(
        r.get("input_schema") == "model-visible-dialogue-v1" for r in normalized) else None)
    requests = {}
    for path in sorted((qa_run / "stages").glob("*raw-candidates*")):
        input_path = path.with_name(path.name.replace("raw-candidates", "qa-input"))
        if not input_path.exists():
            continue
        for question in read(path).get("questions", []):
            requests[question["id"]] = (input_path, question)
    result = []
    for question in public:
        if question.get("status") == "approved" and question["id"] in requests:
            path, original = requests[question["id"]]
            item = {"qa": question, "generation_input": str(path.resolve()),
                    "original_candidate": original}
            if public_records is not None:
                item["public_records"] = public_records
            result.append(item)
    return result


def labels(text):
    return {key: value.lower() for key, value in re.findall(
        r"^([A-Z_]+):\s*([a-z_]+)\s*$", text, re.M)}
