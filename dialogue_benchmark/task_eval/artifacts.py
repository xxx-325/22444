"""Private task artifacts and immutable source snapshots."""

import hashlib
import json
import re
import shutil
from pathlib import Path

from ..storage import load


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
            result.append({"qa": question, "generation_input": str(path.resolve()),
                           "original_candidate": original})
    return result


def labels(text):
    return {key: value.lower() for key, value in re.findall(
        r"^([A-Z_]+):\s*([a-z_]+)\s*$", text, re.M)}
