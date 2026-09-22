"""Retain final experiment evidence and release disposable run resources."""

import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from .artifacts import fingerprint, read, save
from ..storage import compress_file, save_projection


def save_trace(root):
    """Keep original events and model responses, without repeated request histories."""
    root = Path(root)
    target = root / "trace.jsonl.gz"
    if target.exists():
        return target
    events = root / "private/agent/outbox/events.jsonl"
    provider = root / "private/agent/provider.jsonl"
    if not events.exists():
        raise ValueError("Original agent events not saved: " + str(root))
    temp = target.with_suffix(".tmp")
    digest = hashlib.sha256()
    event_ids = set()
    with gzip.open(temp, "wb") as output:
        def emit(kind, value):
            block = (json.dumps({"kind": kind, "value": value}, ensure_ascii=False) + "\n").encode()
            digest.update(block)
            output.write(block)

        initial = root / "private/input.json"
        if initial.exists():
            emit("initial_input", read(initial))
        with events.open() as source:
            for line in source:
                if line.strip():
                    event = json.loads(line)
                    event_ids.add(event.get("id"))
                    emit("event", event)
        if provider.exists():
            with provider.open() as source:
                for line in source:
                    if line.strip():
                        row = json.loads(line)
                        if row.get("kind") != "request":
                            emit("provider", row)
    verified = hashlib.sha256()
    with gzip.open(temp, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            verified.update(block)
    if verified.digest() != digest.digest():
        raise ValueError("Trace compression verification failed")
    trajectory = root / "trajectory.json"
    if trajectory.exists() or trajectory.with_suffix(".json.gz").exists():
        if {e["id"] for e in read(trajectory)} - event_ids:
            raise ValueError("Tool trajectory IDs missing from full trace: " + str(root))
    temp.chmod(0o600)
    temp.replace(target)
    return target


def docker_inventory(root):
    """Resolve exact recorded container IDs; never select global Docker resources."""
    root = Path(root).resolve()
    ids, names = set(), set()
    for directory, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in {
            "workspace", "runtime", ".git", "reference-input", "implementation", "validated-spec"}]
        for name in set(files) & {"environment.json", "config.json"}:
            row = read(Path(directory) / name)
            for key in ("container_id", "control_container_id"):
                value = row.get(key, "")
                if re.fullmatch(r"[a-f0-9]{12,64}", value):
                    ids.add(value)
            if row.get("name", "").startswith("session-tool-"):
                names.add(row["name"])
    if not ids and not names:
        return {"containers": [], "volumes": [], "networks": []}
    listing = subprocess.run(["docker", "ps", "-a", "--format", "{{json .}}"],
                             capture_output=True, text=True, check=True, timeout=30)
    selected = []
    for line in listing.stdout.splitlines():
        item = json.loads(line)
        if item["Names"] in names or any(value.startswith(item["ID"]) for value in ids):
            selected.append(item["ID"])
    containers, volumes, networks = [], set(), set()
    for start in range(0, len(selected), 50):
        result = subprocess.run(["docker", "inspect", *selected[start:start + 50]],
                                capture_output=True, text=True, check=True, timeout=30)
        for row in json.loads(result.stdout):
            # Inspect only mount metadata. Container environment values stay private.
            mounts = row.get("Mounts", [])
            owned = any(m.get("Type") == "bind" and
                        Path(m["Source"].removeprefix("/host_mnt")).resolve().is_relative_to(root)
                        for m in mounts)
            if not owned or not row["Name"].lstrip("/").startswith(("session-tool-", "session-oh-")):
                raise ValueError("Container ownership does not match run: " + row["Id"][:12])
            containers.append({"id": row["Id"], "name": row["Name"].lstrip("/"),
                               "state": row["State"]["Status"]})
            volumes.update(m["Name"] for m in mounts if m.get("Type") == "volume")
            networks.update(n for n in row.get("NetworkSettings", {}).get("Networks", {})
                            if n.startswith("session-"))
    return {"containers": containers, "volumes": sorted(volumes), "networks": sorted(networks)}


def release_docker(inventory):
    """Remove recorded sandboxes, then their unused named volumes and networks."""
    result = {"removed_containers": [], "removed_volumes": [], "removed_networks": [], "errors": []}
    for kind, values, command in (
        ("containers", [c["id"] for c in inventory["containers"]], ["docker", "rm", "--force"]),
        ("volumes", inventory["volumes"], ["docker", "volume", "rm"]),
        ("networks", inventory["networks"], ["docker", "network", "rm"]),
    ):
        for start in range(0, len(values), 25):
            process = subprocess.run(command + values[start:start + 25], capture_output=True,
                                     text=True, timeout=60)
            result["removed_" + kind].extend(process.stdout.splitlines())
            if process.returncode:
                result["errors"].append({"kind": kind, "detail": process.stderr.strip()})
    return result


def release_agent(root):
    """Called only after a finished agent has exported its host-side evidence."""
    root = Path(root)
    try:
        inventory = docker_inventory(root)
        save(root / "resources.json", inventory)
        result = release_docker(inventory)
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        result = {"errors": [{"error_type": type(error).__name__, "detail": str(error)}]}
    save(root / "release.json", result)


def code_versions(root):
    tasks = read(root / "tasks/manifest.json")["tasks"]
    codes = []
    for task in tasks:
        task_root = root / "tasks" / task["task"]
        frozen = task_root / "frozen.json"
        if not frozen.exists():
            continue
        accepted = read(frozen)["accepted_attempt"]
        agents = [task_root / ("construction-%02d/reference-solver" % accepted)]
        agents += [task_root / trial["trial"] for trial in task.get("comparison", {}).values()]
        for agent in agents:
            version = read(agent / "version.json")
            code = agent / "workspace/candidate"
            if fingerprint(code) != version["candidate_sha256"]:
                raise ValueError("Saved final code differs from version receipt: " + str(code))
            codes.append((agent, version["candidate_sha256"]))
    if fingerprint(root / "baseline") != read(root / "baseline.json")["content_sha256"]:
        raise ValueError("Frozen baseline changed")
    return codes


def preserve_test_receipt(checks):
    checks = Path(checks)
    xml = checks / "workspace/experiments/receipt.xml"
    if xml.exists():
        shutil.copy2(xml, checks / "receipt.xml")
    return [checks / "private", checks / "workspace"]


def compact_run(root):
    """Compact a completed run in place after preserving every agreed artifact."""
    root = Path(root).resolve()
    if read(root / "pipeline.json").get("status") != "completed":
        raise ValueError("Only a completed episode can be compacted")
    if (root / "retention.json").exists() and read(root / "retention.json").get("status") == "completed":
        return read(root / "retention.json")
    codes = code_versions(root)
    print("Retention: verified %d final code versions" % len(codes), flush=True)
    inventory_path = root / "retained-docker.json"
    inventory = read(inventory_path) if inventory_path.exists() else docker_inventory(root)
    save(inventory_path, inventory)
    print("Retention: identified %d run-owned containers" % len(inventory["containers"]), flush=True)
    deletions = set()

    def drop(path):
        if path.exists():
            if path.is_symlink() or not path.resolve().is_relative_to(root) or path == root:
                raise ValueError("Cleanup target is outside the run: " + str(path))
            deletions.add(path)

    def agent_evidence(agent, keep_code=False):
        save_trace(agent)
        if (agent / "trajectory.json").exists():
            compress_file(agent / "trajectory.json")
        drop(agent / "private")
        workspace = agent / "workspace"
        if workspace.exists():
            for child in workspace.iterdir():
                if child.name == "candidate" and keep_code:
                    continue
                if child.name == "checks":
                    continue
                drop(child)

    qa = root / "qa"
    public = read(qa / "qa-public.json")["questions"]
    public_ids = {q["id"] for q in public}
    input_files, found = set(), set()
    for path in (qa / "stages").glob("*raw-candidates*"):
        candidates = read(path).get("questions", [])
        ids = {q["id"] for q in candidates} & public_ids
        if ids:
            source = path.with_name(path.name.replace("raw-candidates", "qa-input"))
            if not source.exists():
                raise ValueError("Published QA generation input is missing: " + str(source))
            input_files.update((path, source))
            found.update(ids)
    if found != public_ids:
        raise ValueError("Cannot identify every published QA generation input")
    for path in (qa / "stages").iterdir():
        if path not in input_files:
            drop(path)
    groups = read(qa / "evidence-groups.json")
    save_projection(qa / "evidence-groups.json", groups)
    if read(qa / "evidence-groups.json") != groups:
        raise ValueError("Evidence projection verification failed")
    for name in ("scope.json", "scopes.json", "general-scope.json"):
        drop(qa / name)
    for path in qa.glob("batch-*"):
        drop(path)
    candidates = qa / "candidates.json"
    audit = qa / "qa-audit.json"
    if candidates.exists() and audit.exists() and read(candidates) == read(audit).get("candidates"):
        drop(candidates)

    manifest = read(root / "tasks/manifest.json")
    preserved = [root / "baseline.json", root / "input/dialogue.json", qa / "qa-public.json",
                 qa / "qa-audit.json", qa / "facts.json", root / "tasks/manifest.json"]
    preserved += list((root / "tasks").glob("task-*/frozen.json"))
    preserved += list((root / "tasks").glob("task-*/comparison.json"))
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in preserved if p.is_file()}
    for task in manifest["tasks"]:
        task_root = root / "tasks" / task["task"]
        frozen = task_root / "frozen.json"
        accepted = read(frozen)["accepted_attempt"] if frozen.exists() else None
        summary_path = task_root / "construction-summary.json"
        attempts = read(summary_path if summary_path.exists() else task_root / "construction.json")
        summary = []
        for record in attempts:
            attempt = task_root / ("construction-%02d" % record["attempt"])
            item = dict(record)
            requirement = attempt / "author/workspace/checks/task.md"
            if requirement.exists():
                item["requirement"] = requirement.read_text()
            summary.append(item)
        save(task_root / "construction-summary.json", summary)
        if accepted is None:
            last = next((item["requirement"] for item in reversed(summary) if "requirement" in item), None)
            if last:
                (task_root / "rejected-task.md").write_text(last, encoding="utf-8")
        for attempt in task_root.glob("construction-*"):
            if not attempt.is_dir():
                continue
            if accepted is None or attempt.name != "construction-%02d" % accepted:
                drop(attempt)
                continue
            for role in ("author", "reference-solver", "validator"):
                agent_evidence(attempt / role, keep_code=role == "reference-solver")
            drop(attempt / "validator-reference")
            drop(attempt / "validated-spec")
            for checks in attempt.glob("*checks"):
                for path in preserve_test_receipt(checks):
                    drop(path)
        for prior in (task_root / "author-reference").glob("previous-*"):
            drop(prior)
        for trial in task.get("comparison", {}).values():
            trial_root = task_root / trial["trial"]
            agent_evidence(trial_root, keep_code=True)
            agent_evidence(trial_root / "judge")
            drop(trial_root / "judge-reference")
            checks = trial_root / "checks"
            for path in preserve_test_receipt(checks):
                drop(path)
        for checks in task_root.glob("checkpoint-recovery/*checks"):
            for path in preserve_test_receipt(checks):
                drop(path)
        for path in task_root.glob("*before-*"):
            drop(path)
    drop(root / "diagnostics")
    for path in root.glob("*.log"):
        drop(path)
    # Keep only the topmost deletion roots; persist exact targets before removal.
    targets = [p for p in sorted(deletions) if not any(a in deletions for a in p.parents)]
    receipt = {"status": "prepared", "retained_code_versions": len(codes),
               "published_questions": len(public), "published_inputs_found": len(found),
               "preserved_sha256": hashes,
               "removed_paths": [str(p.relative_to(root)) for p in targets]}
    save(root / "retention.json", receipt)
    print("Retention: traces and final inputs verified; releasing Docker resources", flush=True)
    receipt["docker"] = release_docker(inventory)
    save(root / "retention.json", receipt)
    print("Retention: removing %d temporary paths" % len(targets), flush=True)
    for path in targets:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    code_versions(root)
    for name, expected in hashes.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Retained result changed: " + name)
    receipt["status"] = "completed"
    save(root / "retention.json", receipt)
    return receipt
