"""Resolve frozen acceptance items and execute checks in the existing sandbox."""

from pathlib import Path
import re
import subprocess
import uuid
import xml.etree.ElementTree as ET

from .artifacts import copy_tree, save
from .runtime import release_completed_execution

TEST_FILES = ("test_acceptance.py", "test_interactions.py")


def acceptance_items(spec, history=None):
    """Read the frozen human-readable table, using exact test names for linkage."""
    rows = []
    contracts = {c["id"] for c in (history or {}).get("contracts", []) if c.get("active", True)}
    for line in (Path(spec) / "acceptance.md").read_text().splitlines():
        cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
        if len(cells) != 4 or not re.fullmatch(r"a\d+", cells[0]):
            continue
        identity, requirement, basis, check = cells
        sources = [s.strip() for s in basis.split(",")]
        if (not requirement or not check or any(s != "task" and s not in contracts for s in sources)
                or any(row["id"] == identity for row in rows)):
            raise ValueError("Invalid acceptance item: " + identity)
        tests = [t.strip().strip("`") for t in check[5:].split(",")] if check.startswith("test:") else []
        if check.startswith("command:"):
            name = check[8:].strip()
            if not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or not (Path(spec) / "commands" / (name + ".sh")).is_file():
                raise ValueError("Missing frozen command: " + name)
            tests = ["command::" + name]
        if not tests and not check.startswith("inspect:"):
            raise ValueError("Acceptance check must be test:, command:, or inspect: " + identity)
        if tests and any(not re.fullmatch(r"[\w.]+::[\w\[\].-]+", t) for t in tests):
            raise ValueError("Use exact JUnit classname::name: " + identity)
        rows.append({"id": identity, "requirement": requirement, "basis": sources,
                     "tests": tests, "check": check})
    if not rows or contracts - {s for row in rows for s in row["basis"]}:
        raise ValueError("Acceptance must cover the task and every active historical rule")
    if not any("task" in row["basis"] for row in rows):
        raise ValueError("Acceptance must include new functionality")
    return rows


def _artifact_evidence(value, roots):
    """Resolve a file and actual line range; semantic sufficiency remains reviewable."""
    match = re.fullmatch(r"(/[^\s:]+):(\d+)(?:-(\d+))?", value.strip())
    if not match:
        return False
    name, start, end = match.groups()
    for prefix, root in roots.items():
        if name.startswith(prefix + "/"):
            root = Path(root).resolve()
            path = (root / name[len(prefix) + 1:]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                return False
            try:
                count = len(path.read_text().splitlines())
            except (UnicodeError, OSError):
                return False
            return 1 <= int(start) <= int(end or start) <= count
    return False


def assess_acceptance(items, checks, review_path=None, roots=None):
    """One result per mandatory item, plus execution failures in frozen regressions."""
    from ..llm import parse_text_response
    reviews = []
    if review_path and Path(review_path).is_file():
        try:
            reviews = parse_text_response(Path(review_path).read_text()).get("reviews", [])
        except ValueError:
            pass
    cases = {}
    for case in checks.get("cases", []):
        cases.setdefault(case["id"], []).append(case)
    rows = []
    for item in items:
        status, evidence = "uncertain", "Required check not established"
        if item["tests"]:
            selected = [cases.get(name, []) for name in item["tests"]]
            if any(c["status"] == "failed" for group in selected for c in group):
                status = "failed"
            elif (checks["status"] in {"passed", "failed"} and all(len(group) == 1 and group[0]["status"] == "passed"
                                                       for group in selected)):
                status = "passed"
            evidence = ", ".join(item["tests"])
        else:
            matches = [r for r in reviews if r.get("id") == item["id"]]
            if len(matches) == 1:
                review = matches[0]
                if (review.get("status") in {"passed", "failed"}
                        and _artifact_evidence(review.get("evidence", ""), roots or {})):
                    status, evidence = review["status"], review["evidence"]
        rows.append(dict(item, status=status, evidence=evidence))
    status = ("failed" if checks["status"] == "failed" or any(r["status"] == "failed" for r in rows)
              else "passed" if rows and checks["status"] != "error" and all(r["status"] == "passed" for r in rows)
              else "uncertain")
    return {"status": status, "rows": rows}


def pytest_result(exit_code, xml_path):
    result = {"exit_code": exit_code, "status": "unavailable", "tests": 0,
              "passed": 0, "failed": 0, "errors": 0, "skipped": 0, "cases": []}
    if not Path(xml_path).exists():
        if exit_code != 0:
            result["status"] = "error"
        return result
    try:
        cases = ET.parse(xml_path).getroot().findall(".//testcase")
    except ET.ParseError:
        result["status"] = "error"
        return result
    result.update(tests=len(cases),
                  failed=sum(c.find("failure") is not None for c in cases),
                  errors=sum(c.find("error") is not None for c in cases),
                  skipped=sum(c.find("skipped") is not None for c in cases))
    result["passed"] = len(cases) - result["failed"] - result["errors"] - result["skipped"]
    for case in cases:
        status, detail = "passed", ""
        for tag, state in (("skipped", "skipped"), ("failure", "failed"), ("error", "error")):
            node = case.find(tag)
            if node is not None:
                status = state
                detail = (node.get("message", "") + "\n" + (node.text or "")).strip()
        result["cases"].append({"id": case.get("classname", "") + "::" + case.get("name", ""),
                                "name": case.get("name", ""), "status": status, "detail": detail})
    if result["errors"]:
        result["status"] = "error"
    elif result["failed"]:
        result["status"] = "failed"
    elif exit_code != 0:
        result["status"] = "error"
    elif exit_code == 0 and result["passed"]:
        result["status"] = "passed"
    return result


def run_checks(candidate, spec, output, image, *, candidate_pythonpath=None):
    from simulator.openhands.sandbox import ExecutionSandbox
    output, spec = Path(output), Path(spec)
    tests = [name for name in TEST_FILES if (spec / name).is_file()]
    scripts = sorted((spec / "commands").glob("*.sh"))
    if not tests and not scripts:
        result = {"status": "unavailable", "reason": "No generated test; use frozen judge criteria"}
    else:
        workspace = output / "workspace"
        copy_tree(candidate, workspace / "candidate")
        copy_tree(spec, workspace / "checks")
        sandbox = ExecutionSandbox(output / "private", workspace, image, "judge",
                                   uuid.uuid4().hex, reference=spec)
        sandbox.prepare()
        sandbox.unpause()
        # Match the agent's PTY-enabled terminal, including /dev/tty availability.
        # Source, tests, and dependencies use the same execution image and mounts.
        command = ["docker", "exec", "-t", "--user", "1000", "-w", "/workspace/candidate",
                   "-e", "PYTHONDONTWRITEBYTECODE=1",
                   sandbox.name, "python", "-m", "pytest", "-c", "/dev/null",
                   "--rootdir=/workspace/checks",
                   "-p", "no:cacheprovider", "-q",
                   *["/workspace/checks/" + name for name in tests],
                   "--junitxml=/workspace/experiments/receipt.xml"]
        if candidate_pythonpath:
            command[command.index(sandbox.name):command.index(sandbox.name)] = ["-e", "PYTHONPATH=" + candidate_pythonpath]
        executions = []
        def execute(argv):
            try:
                process = subprocess.run(argv, text=True, capture_output=True, timeout=180)
                row = {"exit_code": process.returncode, "stdout": process.stdout, "stderr": process.stderr}
            except subprocess.TimeoutExpired:
                row = {"exit_code": 124, "stdout": "", "stderr": "Check execution timed out; sandbox retained"}
            executions.append(dict(row, command=argv))
            return row
        try:
            run = execute(command) if tests else {"exit_code": 0}
            result = (pytest_result(run["exit_code"], workspace / "experiments/receipt.xml") if tests else
                      {"status": "passed", "cases": [], "tests": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0})
            for script in scripts:
                executed = execute(command[:command.index(sandbox.name) + 1] +
                                   ["sh", "/workspace/checks/commands/" + script.name])
                # Commands use 1 for a violated requirement, 2+ for execution failures.
                state = "passed" if executed["exit_code"] == 0 else "failed" if executed["exit_code"] == 1 else "error"
                result["cases"].append({"id": "command::" + script.stem, "name": script.stem,
                                        "status": state, "detail": executed["stdout"] + executed["stderr"]})
                result["tests"] += 1
                result[{"passed": "passed", "failed": "failed", "error": "errors"}[state]] += 1
            if result["failed"]:
                result["status"] = "failed"
            elif result["errors"]:
                result["status"] = "error"
        finally:
            sandbox.pause()
        save(output / "execution.json", {"executions": executions})
        result["test_files"] = tests
    save(output / "result.json", result)
    if (tests or scripts) and all(r["exit_code"] != 124 for r in executions):
        release_completed_execution(sandbox.record)
    return result


def check_history_mutations(candidate, spec, validator_checks, output, image, *, candidate_pythonpath=None):
    """Replay saved wrong implementations; functioning task rows must still pass."""
    from ..llm import parse_text_response
    from .artifacts import read
    from .versions import pin_baseline
    output, validator_checks = Path(output), Path(validator_checks)
    path = validator_checks / "mutations.txt"
    rows = parse_text_response(path.read_text()).get("reviews", []) if path.is_file() else []
    items = read(Path(spec) / "acceptance.json")
    results = []
    for row in rows:
        name = row.get("id", "")
        if not re.fullmatch(r"m\d+", name):
            continue
        patch = validator_checks / (name + ".patch")
        targets = [r for r in items if r["id"] == row.get("acceptance") and any(b != "task" for b in r["basis"])]
        if not patch.is_file() or not patch.stat().st_size or len(targets) != 1:
            continue
        root = output / name
        if root.exists():
            continue
        copy_tree(candidate, root / "candidate")
        pin_baseline(root / "candidate")
        # Patches are relative to the reference implementation, never the baseline.
        process = subprocess.run(["git", "apply", "--", str(patch.resolve())],
                                 cwd=root / "candidate", text=True, capture_output=True)
        receipt = {"id": name, "acceptance": row["acceptance"], "patch_applied": process.returncode == 0,
                   "stderr": process.stderr}
        if process.returncode == 0:
            checks = run_checks(root / "candidate", spec, root / "checks", image,
                                candidate_pythonpath=candidate_pythonpath)
            assessment = assess_acceptance(items, checks)
            functional = [r for r in assessment["rows"] if r["basis"] == ["task"]]
            target = next(r for r in assessment["rows"] if r["id"] == row["acceptance"])
            receipt.update(checks=checks, acceptance=assessment,
                           caught=bool(functional) and all(r["status"] == "passed" for r in functional)
                                  and target["status"] == "failed")
        results.append(receipt)
        save(root / "result.json", receipt)
    result = {"status": "caught" if results and all(r.get("caught") for r in results) else "unverified", "variants": results}
    save(output / "result.json", result)
    return result
