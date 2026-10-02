"""Resolve frozen acceptance items and execute checks in the existing sandbox."""

import ast
from pathlib import Path
import re
import shutil
import subprocess
import uuid
import xml.etree.ElementTree as ET

from .artifacts import copy_tree, read, save, install_candidate_fixture
from .runtime import release_completed_execution


def _test_write_violations(spec):
    """Find generated tests that write through the read-only candidate fixture.

    The candidate snapshot is intentionally mounted read-only during checks.
    Catch the common alias form before starting a sandbox so the author can
    repair the test with ``tmp_path`` instead of receiving one opaque
    filesystem error per test.
    """
    violations = []
    write_methods = {"write_text", "write_bytes", "mkdir", "touch", "unlink",
                     "rename", "replace", "rmdir"}
    for path in sorted(Path(spec).rglob("test_*.py")) + sorted(Path(spec).rglob("*_test.py")):
        if not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError):
            continue
        for function in (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)):
            nodes = list(ast.walk(function))
            aliases = {"candidate_root"}
            assignments = {}
            for node in nodes:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            assignments.setdefault(target.id, []).append(node.value)

            def candidate_path(node):
                if isinstance(node, ast.Name):
                    return node.id in aliases
                return (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
                        and candidate_path(node.left))

            # Reassigned aliases are ambiguous; the real sandbox still enforces read-only mounts.
            aliases.update(name for name, values in assignments.items()
                           if len(values) == 1 and candidate_path(values[0]))
            for node in nodes:
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and candidate_path(node.func.value)):
                    continue
                writes = node.func.attr in write_methods
                if node.func.attr == "open":
                    mode = next((arg.value for arg in node.keywords if arg.arg == "mode"),
                                node.args[0] if node.args else ast.Constant("r"))
                    writes = (isinstance(mode, ast.Constant) and isinstance(mode.value, str)
                              and any(flag in mode.value for flag in "wax+"))
                if writes:
                    violations.append(f"{path.name}:{node.lineno}: candidate_root is read-only; "
                                      f"move temporary output to tmp_path ({node.func.attr})")
    return sorted(set(violations))

def _test_identity(value):
    """Match a pytest file node ID to its JUnit module ID without fuzzy aliases."""
    parts = value.split("::")
    module = parts[0]
    if module.endswith(".py"):
        module = module[:-3].replace("/", ".")
    return ".".join([module, *parts[1:-1]]) + "::" + parts[-1]


def acceptance_items(spec, history=None):
    """Read the frozen human-readable table, using exact test names for linkage."""
    rows = []
    contracts = {c["id"] for c in (history or {}).get("contracts", []) if c.get("active", True)}
    answer_path = Path(spec) / "oracle-answer.json"
    if answer_path.is_file():
        answer = read(answer_path).get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Missing private historical answer")
        contracts.add("answer")
    for line in (Path(spec) / "acceptance.md").read_text().splitlines():
        cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
        # Models occasionally capitalize the otherwise stable a1/a2 labels.
        # The identity is still unambiguous, so accept case without changing
        # the four-column contract.
        if len(cells) != 4 or not re.fullmatch(r"a\d+", cells[0], re.IGNORECASE):
            continue
        identity, requirement, basis, check = cells
        identity = identity.lower()
        sources = [s.strip() for s in basis.split(",")]
        if (not requirement or not check or any(s != "task" and s not in contracts for s in sources)
                or ("answer" in sources and sources != ["answer"])
                or any(row["id"] == identity for row in rows)):
            raise ValueError("Invalid acceptance item: " + identity)
        tests = []
        if not check.startswith("inspect:"):
            for part in re.split(r";|,\s*(?=(?:test|command):)", check):
                part = part.strip()
                if part.startswith("test:"):
                    tests.extend(t.strip().strip("`") for t in part[5:].split(","))
                elif part.startswith("command:"):
                    name = part[8:].strip()
                    if not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or not (Path(spec) / "commands" / (name + ".sh")).is_file():
                        raise ValueError("Missing frozen command: " + name)
                    tests.append("command::" + name)
                else:
                    raise ValueError("Acceptance check must be test:, command:, or inspect: " + identity)
        if not tests and not check.startswith("inspect:"):
            raise ValueError("Acceptance check must be test:, command:, or inspect: " + identity)
        if tests and any(not re.fullmatch(r"[\w./-]+(?:::[\w.-]+)*::[\w\[\].-]+", t) for t in tests):
            raise ValueError("Use an exact pytest node or JUnit classname::name: " + identity)
        rows.append({"id": identity, "requirement": requirement, "basis": sources,
                     "tests": tests, "check": check})
    if not rows or contracts - {s for row in rows for s in row["basis"]}:
        raise ValueError("Acceptance must cover the task and every active historical rule")
    if not any("task" in row["basis"] for row in rows):
        raise ValueError("Acceptance must include new functionality")
    return rows


def _artifact_evidence(value, roots):
    """Resolve a file and actual lines; semantic sufficiency remains reviewable."""
    match = re.fullmatch(r"(/[^\s:]+):(\d+(?:-\d+|(?:,\d+)+)?)", value.strip())
    if not match:
        return False
    name, lines = match.groups()
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
            if "," in lines:
                return all(1 <= int(line) <= count for line in lines.split(","))
            start, _, end = lines.partition("-")
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
        cases.setdefault(_test_identity(case["id"]), []).append(case)
    rows = []
    for item in items:
        status, evidence = "uncertain", "Required check not established"
        if item["tests"]:
            selected = [cases.get(_test_identity(name), []) for name in item["tests"]]
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


def pytest_result(exit_code, xml_path, *, required_modules=()):
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
    if result["status"] != "error":
        executed = [case["id"].removeprefix("checks.") for case in result["cases"]
                    if case["status"] in {"passed", "failed"}]
        missing = [module for module in required_modules if not any(
            identity.startswith((module + "::", module + ".")) for identity in executed)]
        if missing:
            result.update(status="error", reason="required_test_module_not_executed",
                          detail="No executed pytest cases for: " + ", ".join(missing),
                          unexecuted_modules=missing)
    return result


def run_checks(candidate, spec, output, image, *, candidate_pythonpath=None):
    from simulator.openhands.sandbox import ExecutionSandbox
    output, spec = Path(output), Path(spec)
    tests = sorted({path.relative_to(spec).as_posix()
                    for pattern in ("test_*.py", "*_test.py")
                    for path in spec.rglob(pattern) if path.is_file()})
    scripts = sorted((spec / "commands").glob("*.sh"))
    write_violations = _test_write_violations(spec)
    if write_violations:
        result = {"status": "error", "reason": "generated_test_writes_read_only_candidate",
                  "detail": "\n".join(write_violations), "exit_code": 2,
                  "tests": len(tests), "passed": 0, "failed": 0, "errors": len(tests),
                  "skipped": 0, "cases": [], "test_files": tests}
        save(output / "result.json", result)
        return result
    if not tests and not scripts:
        result = {"status": "unavailable", "reason": "No generated test; use frozen judge criteria"}
    else:
        workspace = output / "workspace"
        copy_tree(candidate, workspace / "candidate")
        copy_tree(spec, workspace / "checks")
        regression = workspace / "checks/regression/tests"
        if regression.is_dir():
            # Restore only frozen tests, at their original repository paths.
            # Their imports and repository-relative data belong to the candidate.
            candidate_tests = workspace / "candidate/tests"
            if candidate_tests.is_dir():
                shutil.rmtree(candidate_tests)
            elif candidate_tests.exists():
                candidate_tests.unlink()
            copy_tree(regression, candidate_tests)
        # Run inherited tests in a disposable writable copy. The submitted
        # snapshot and candidate_root fixture remain read-only.
        validation = workspace / "checks/validation-candidate"
        copy_tree(workspace / "candidate", validation)
        validation_path = "/workspace/checks/validation-candidate"
        for script in scripts:
            execution_script = workspace / "checks/commands" / script.name
            execution_script.write_text(
                script.read_text().replace("/workspace/candidate", validation_path), encoding="utf-8")
        test_paths = [validation_path + "/" + name.removeprefix("regression/")
                      if name.startswith("regression/tests/") else "/workspace/checks/" + name
                      for name in tests]
        install_candidate_fixture(workspace / "checks")
        sandbox = ExecutionSandbox(output / "private", workspace, image, "judge",
                                   uuid.uuid4().hex, reference=spec)
        sandbox.prepare()
        sandbox.unpause()
        # Match the agent's PTY-enabled terminal, including /dev/tty availability.
        # Source, tests, and dependencies use the same execution image and mounts.
        command = ["docker", "exec", "-t", "--user", "1000", "-w", validation_path,
                   "-e", "PYTHONDONTWRITEBYTECODE=1",
                   sandbox.name, "python", "-m", "pytest", "-c", "/dev/null",
                   "--rootdir=/workspace" if regression.is_dir() else "--rootdir=/workspace/checks",
                   "-p", "no:cacheprovider", "-q",
                   *test_paths,
                   "--junitxml=/workspace/experiments/receipt.xml"]
        if candidate_pythonpath:
            pythonpath = candidate_pythonpath.replace("/workspace/candidate", validation_path)
            command[command.index(sandbox.name):command.index(sandbox.name)] = ["-e", "PYTHONPATH=" + pythonpath]
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
            result = (pytest_result(run["exit_code"], workspace / "experiments/receipt.xml",
                                   required_modules=("test_interactions",) if "test_interactions.py" in tests else ()) if tests else
                      {"status": "passed", "cases": [], "tests": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0})
            if regression.is_dir():
                # Keep acceptance IDs tied to the frozen spec despite relocation.
                for case in result["cases"]:
                    if case["id"].startswith("checks."):
                        case["id"] = case["id"].removeprefix("checks.")
                    if case["id"].startswith("candidate.tests."):
                        case["id"] = "regression.tests." + case["id"].removeprefix("candidate.tests.")
                    elif case["id"].startswith("validation-candidate.tests."):
                        case["id"] = "regression.tests." + case["id"].removeprefix("validation-candidate.tests.")
            for script in scripts:
                # Generated command checks are shell scripts.  Run them with
                # bash so arrays, pipefail, and parameter expansion work
                # consistently across the host and the execution image.
                executed = execute(command[:command.index(sandbox.name) + 1] +
                                   ["bash", "/workspace/checks/commands/" + script.name])
                # Commands use 1 for a violated requirement, 2+ for execution failures.
                state = "passed" if executed["exit_code"] == 0 else "failed" if executed["exit_code"] == 1 else "error"
                result["cases"].append({"id": "command::" + script.stem, "name": script.stem,
                                        "status": state, "detail": executed["stdout"] + executed["stderr"]})
                result["tests"] += 1
                result[{"passed": "passed", "failed": "failed", "error": "errors"}[state]] += 1
            if result["errors"]:
                result["status"] = "error"
            elif result["failed"] and result["status"] != "error":
                result["status"] = "failed"
        finally:
            sandbox.pause()
        save(output / "execution.json", {"executions": executions})
        result["test_files"] = tests
    save(output / "result.json", result)
    if (tests or scripts) and all(r["exit_code"] != 124 for r in executions):
        release_completed_execution(sandbox.record)
    return result


def check_history_mutations(candidate, spec, validator_checks, output, image, *, candidate_pythonpath=None,
                            inspector=None):
    """Replay saved wrong implementations; functioning task rows must still pass."""
    # Keep this helper usable by offline/unit callers that do not install the
    # simulator package.  The normal CLI supplies the sandbox dependency;
    # without it there is no executable mutation evidence to claim.
    from ..llm import parse_text_response
    from .artifacts import read
    from .versions import pin_baseline
    output, validator_checks = Path(output), Path(validator_checks)
    path = validator_checks / "mutations.txt"
    rows = parse_text_response(path.read_text()).get("reviews", []) if path.is_file() else []
    items = read(Path(spec) / "acceptance.json")
    unknown = {row.get("acceptance", "") for row in rows} - {item["id"] for item in items}
    if unknown:
        result = {"status": "unverified", "variants": [], "error_type": "ValueError",
                  "detail": "Unknown acceptance IDs: " + ", ".join(repr(identity) for identity in sorted(unknown))}
        save(output / "result.json", result)
        return result
    history_path = Path(spec) / "history.json"
    history = read(history_path) if history_path.is_file() else {"contracts": []}
    external = {c["id"] for c in history["contracts"] if c["active"] and c["repository"] == "external"}
    if (Path(spec) / "oracle-answer.json").is_file():
        external.add("answer")
    results = []
    for row in rows:
        name = row.get("id", "")
        if not re.fullmatch(r"m\d+", name):
            continue
        patch = validator_checks / (name + ".patch")
        targets = [r for r in items if r["id"] == row.get("acceptance")
                   and "task" not in r["basis"] and external.intersection(r["basis"])]
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
            review_path, roots = None, None
            if inspector and any(not item["tests"] for item in items) and checks["status"] != "error":
                review_path, roots = inspector(root / "candidate", spec, checks, root)
            assessment = assess_acceptance(items, checks, review_path, roots)
            functional = [r for r in assessment["rows"] if "task" in r["basis"]]
            target = next(r for r in assessment["rows"] if r["id"] == row["acceptance"])
            receipt.update(checks=checks, acceptance=assessment,
                           caught=bool(functional) and all(r["status"] == "passed" for r in functional)
                                  and target["status"] == "failed")
        results.append(receipt)
        save(root / "result.json", receipt)
    result = {"status": "caught" if results and all(r.get("caught") for r in results) else "unverified", "variants": results}
    save(output / "result.json", result)
    return result
