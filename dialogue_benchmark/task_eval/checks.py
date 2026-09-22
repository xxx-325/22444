"""Execute frozen pytest checks in a disposable copy using the existing sandbox."""

from pathlib import Path
import subprocess
import uuid
import xml.etree.ElementTree as ET

from .artifacts import copy_tree, save
from .runtime import release_completed_execution

TEST_FILES = ("test_acceptance.py", "test_interactions.py")


def pytest_result(exit_code, xml_path):
    result = {"exit_code": exit_code, "status": "unavailable", "tests": 0,
              "passed": 0, "failed": 0, "errors": 0, "skipped": 0}
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
    if result["errors"]:
        result["status"] = "error"
    elif result["failed"]:
        result["status"] = "failed"
    elif exit_code != 0:
        result["status"] = "error"
    elif exit_code == 0 and result["passed"]:
        result["status"] = "passed"
    return result


def run_checks(candidate, spec, output, image):
    from simulator.openhands.sandbox import ExecutionSandbox
    output, spec = Path(output), Path(spec)
    tests = [name for name in TEST_FILES if (spec / name).is_file()]
    if not tests:
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
                   "-e", "PYTHONPATH=/workspace/candidate/src", "-e", "PYTHONDONTWRITEBYTECODE=1",
                   sandbox.name, "python", "-m", "pytest", "-c", "/dev/null",
                   "-p", "no:cacheprovider", "-q",
                   *["/workspace/checks/" + name for name in tests],
                   "--junitxml=/workspace/experiments/receipt.xml"]
        try:
            process = subprocess.run(command, text=True, capture_output=True, timeout=180)
            run = {"exit_code": process.returncode, "stdout": process.stdout, "stderr": process.stderr}
        except subprocess.TimeoutExpired:
            run = {"exit_code": 124, "stdout": "", "stderr": "Check execution timed out; sandbox retained"}
        finally:
            sandbox.pause()
        save(output / "execution.json", dict(run, command=command))
        result = pytest_result(run["exit_code"], workspace / "experiments/receipt.xml")
        result["test_files"] = tests
    save(output / "result.json", result)
    if tests and run["exit_code"] != 124:
        release_completed_execution(sandbox.record)
    return result
