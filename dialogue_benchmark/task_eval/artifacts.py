"""Private task artifacts and immutable source snapshots."""

import hashlib
import json
import re
import shutil
from pathlib import Path

from ..storage import load
from ..protocol import QA_TYPES, MEMORY_TYPES


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.chmod(0o600)
    temp.replace(path)


def read(path):
    return load(path)


def qa_fingerprint(question):
    return hashlib.sha256(json.dumps(question, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def install_candidate_fixture(directory):
    """Provide the same repository location in authoring and scored executions."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "conftest.py").write_text(
        'from pathlib import Path\nimport pytest\n\n'
        '@pytest.fixture(scope="session")\ndef candidate_root():\n'
        '    return Path("/workspace/candidate")\n', encoding="utf-8")


def copy_tree(source, target, *, include_caches=False):
    source, target = Path(source), Path(target)
    ignored_artifact_dirs = {".tox", ".venv", "_build"}
    if any(p.is_symlink() and not any(part in ignored_artifact_dirs
                                      for part in p.relative_to(source).parts)
           for p in source.rglob("*")):
        raise ValueError("Snapshot contains a symlink")
    ignored = [".git", "__pycache__", ".pytest_cache", ".venv", ".tox", "_build", "*.pyc"]
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
    run_manifest = read(qa_run / "manifest.json") if (qa_run / "manifest.json").exists() else {}
    types = MEMORY_TYPES if run_manifest.get("qa_source") == "external" else QA_TYPES
    if any(question.get("type") not in types and not (
            question.get("type") is None and question.get("type_status") == "unresolved"
            and run_manifest.get("qa_source", "graph") == "graph") for question in public):
        raise ValueError("QA types do not match the source mode; regenerate QA")
    normalized_path = qa_run / "normalized.json"
    normalized = read(normalized_path) if normalized_path.exists() else []
    public_records = None
    if normalized and all(r.get("input_schema") == "model-visible-dialogue-v1" for r in normalized):
        public_records = normalized
    elif normalized and all(
            isinstance(r, dict) and r.get("kind") == "message"
            and r.get("role") in {"user", "assistant"}
            and isinstance(r.get("text"), str) and isinstance(r.get("id"), str)
            for r in normalized):
        # A plain user/assistant dialogue is still a public history.  Older
        # list inputs predate the envelope marker and therefore do not carry
        # original_id/input_schema; add only the local identity needed by
        # task_eval, never tool or controller records.
        public_records = [dict(r, original_id=r.get("original_id", r["id"]),
                               input_schema="model-visible-dialogue-v1")
                          for r in normalized]
    requests = {}
    for path in sorted((qa_run / "stages").glob("*raw-candidates*")):
        input_path = path.with_name(path.name.replace("raw-candidates", "qa-input"))
        if not input_path.exists():
            continue
        for question in read(path).get("questions", []):
            requests[question["id"]] = (input_path, question)
    audit_path = qa_run / "qa-audit.json"
    reviewed = {question["id"]: question
                for question in (read(audit_path).get("questions", []) if audit_path.exists() else [])
                if question.get("status") == "approved"}
    result = []
    for question in public:
        if question.get("status") == "approved" and question["id"] in requests:
            path, original = requests[question["id"]]
            item = {"qa": question, "generation_input": str(path.resolve()),
                    "original_candidate": original,
                    "reviewed_candidate": reviewed.get(question["id"], original),
                    "qa_source": run_manifest.get("qa_source", "graph")}
            workflow = (read(path).get("payload", {}).get("workflow", {})
                        if item["qa_source"] == "external" else {})
            if isinstance(workflow, dict) and isinstance(workflow.get("text"), str):
                item["development_workflow"] = workflow["text"]
            if public_records is not None:
                item["public_records"] = public_records
            result.append(item)
    return result


def labels(text):
    return {key: value.lower() for key, value in re.findall(
        r"^([A-Z_]+):\s*([a-z_]+)\s*$", text, re.M)}
