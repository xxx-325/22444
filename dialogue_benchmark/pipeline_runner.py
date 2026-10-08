"""Small filesystem-backed runner for the repo, QA, and task stages.

The runner deliberately knows nothing about model prompts or stage internals.
Each stage is an external command.  A completed stage writes a handoff marker
which is the only input the next stage needs.  This makes the three stages
independent workers while keeping one case's failure from stopping the others.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import threading
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence


SCHEMA = "dialogue-pipeline-v1"
STAGES = ("repo", "qa", "task")
DEPENDENCY = {"repo": None, "qa": "repo", "task": "qa"}
TERMINAL = {"completed", "failed", "skipped", "needs_review"}
_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=".pipeline-", delete=False) as out:
        json.dump(value, out, ensure_ascii=False, indent=2)
        out.write("\n")
        temporary = Path(out.name)
    os.replace(temporary, path)


def _replace_tokens(value: str, mapping: Dict[str, str]) -> str:
    """Replace only known tokens, preserving braces in shell/Python snippets."""
    result = value
    for key, replacement in mapping.items():
        result = result.replace("{" + key + "}", replacement)
    return result


def _receipt_path(spec: object, paths: Dict[str, str]) -> Optional[Path]:
    if spec is None or spec is False:
        return None
    if spec is True:
        value = paths["receipt"]
    elif isinstance(spec, str):
        value = spec
    elif isinstance(spec, dict):
        value = spec.get("path", paths["receipt"])
    else:
        raise ValueError("stage receipt must be a path or object")
    if not isinstance(value, str) or not value:
        raise ValueError("stage receipt path must be a non-empty string")
    path = Path(_replace_tokens(value, paths))
    return path if path.is_absolute() else Path(paths["output"]) / path


class StageCommandError(RuntimeError):
    """A stage command returned a non-zero status or could not start."""


class StageReceiptError(RuntimeError):
    """A successful command did not leave an accepted completion receipt."""


CommandRunner = Callable[[Sequence[str], Path, Dict[str, str], Path, Path], None]


def run_command(command: Sequence[str], cwd: Path, env: Dict[str, str],
                stdout_path: Path, stderr_path: Path) -> None:
    """Stream both command outputs directly to their attempt logs."""
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8") as stderr:
        try:
            completed = subprocess.run(
                list(command), cwd=str(cwd), env=env, text=True,
                stdout=stdout, stderr=stderr, check=False,
            )
        except OSError as error:
            stderr.write(str(error))
            raise StageCommandError(str(error)) from error
    if completed.returncode:
        raise StageCommandError(
            "command exited with status %d" % completed.returncode)


def _command(value: object) -> Sequence[str]:
    if isinstance(value, str):
        return tuple(shlex.split(value))
    if isinstance(value, (list, tuple)) and value and all(
            isinstance(part, str) for part in value):
        return tuple(value)
    raise ValueError("stage command must be a non-empty string or list")


class PipelineRunner:
    """Run three stage workers over independent cases.

    ``config`` has a ``cases`` list and a ``stages`` mapping.  A stage command
    may use ``{case_dir}``, ``{source}``, ``{input}``, ``{output}``,
    ``{handoff}``, and ``{stage}`` tokens.  Per-case stage definitions under
    ``case.stages`` override the global definition.  The command receives
    equivalent ``PIPELINE_*`` environment variables as well.
    """

    def __init__(self, config: Dict[str, object], output: Path,
                 *, max_inflight: int = 3, max_attempts: int = 2,
                 resume: bool = False,
                 command_runner: CommandRunner = run_command,
                 poll_interval: float = 0.05) -> None:
        self.config = self._validate_config(config)
        self.output = Path(output)
        self.max_inflight = max(1, min(3, int(max_inflight)))
        self.max_attempts = max(1, int(max_attempts))
        self.resume = resume
        self.command_runner = command_runner
        self.poll_interval = poll_interval
        self.state_path = self.output / "pipeline-state.json"
        self.case_configs = {
            case["id"]: case for case in self.config["cases"]  # type: ignore[index]
        }
        self._lock = threading.RLock()
        self._slots = threading.Semaphore(self.max_inflight)
        self.state = self._load_or_create_state()

    @staticmethod
    def _validate_config(config: Dict[str, object]) -> Dict[str, object]:
        if not isinstance(config, dict):
            raise ValueError("pipeline config must be an object")
        stages = config.get("stages")
        if not isinstance(stages, dict):
            raise ValueError("pipeline config requires stages")
        for stage in STAGES:
            if stage not in stages:
                raise ValueError("pipeline config is missing stage %s" % stage)
            value = stages[stage]
            if isinstance(value, dict):
                value = value.get("command")
            _command(value)
        raw_cases = config.get("cases")
        if not isinstance(raw_cases, list) or not raw_cases:
            raise ValueError("pipeline config requires a non-empty cases list")
        cases = []
        seen = set()
        for raw in raw_cases:
            if not isinstance(raw, dict) or not isinstance(raw.get("id"), str):
                raise ValueError("each case needs a string id")
            case_id = raw["id"]
            if not _CASE_ID.match(case_id) or case_id in seen:
                raise ValueError("case ids must be unique safe names")
            if not isinstance(raw.get("source", ""), str):
                raise ValueError("case source must be a string")
            overrides = raw.get("stages", {})
            if not isinstance(overrides, dict):
                raise ValueError("case stages must be an object")
            for stage, value in overrides.items():
                if stage not in STAGES:
                    raise ValueError("unknown case stage %s" % stage)
                if isinstance(value, dict):
                    value = value.get("command")
                _command(value)
            seen.add(case_id)
            cases.append(dict(raw))
        return {"stages": dict(stages), "cases": cases}

    def _config_identity(self) -> str:
        return _digest(self.config)

    def _new_state(self) -> Dict[str, object]:
        cases = {}
        for raw in self.config["cases"]:  # type: ignore[index]
            case_id = raw["id"]
            cases[case_id] = {
                "id": case_id,
                "source": raw.get("source", ""),
                "stages": {
                    stage: {"status": "pending", "attempts": 0}
                    for stage in STAGES
                },
            }
        return {
            "schema": SCHEMA,
            "config_digest": self._config_identity(),
            "created_at": _now(),
            "updated_at": _now(),
            "cases": cases,
        }

    def _load_or_create_state(self) -> Dict[str, object]:
        self.output.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            if not self.resume:
                raise FileExistsError(
                    "pipeline state exists; pass resume=True to continue")
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("schema") != SCHEMA:
                raise ValueError("unsupported pipeline state schema")
            if state.get("config_digest") != self._config_identity():
                raise ValueError("pipeline config differs from saved state")
            for case in state.get("cases", {}).values():
                for stage in STAGES:
                    record = case["stages"][stage]
                    if record.get("status") == "running":
                        record["status"] = "pending"
                        record["recovered"] = True
                    if record.get("status") == "completed":
                        handoff = record.get("handoff")
                        handoff_path = Path(handoff) if handoff else None
                        invalid_reason = None
                        if not handoff_path or not handoff_path.is_file():
                            invalid_reason = "missing_handoff"
                        elif not record.get("handoff_sha256"):
                            invalid_reason = "handoff_unverified"
                        elif record["handoff_sha256"] != _file_sha256(handoff_path):
                            invalid_reason = "handoff_changed"
                        else:
                            try:
                                saved_handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
                            except (OSError, ValueError):
                                saved_handoff = None
                            if not isinstance(saved_handoff, dict) or saved_handoff.get("status") != "completed":
                                invalid_reason = "invalid_handoff"
                        if invalid_reason:
                            record.update({"status": "needs_review", "reason": invalid_reason,
                                           "recovered": True, "finished_at": _now()})
                            continue
                        receipt = record.get("receipt")
                        receipt_sha = record.get("receipt_sha256")
                        if receipt and (not Path(receipt).is_file()
                                        or (receipt_sha and receipt_sha != _file_sha256(Path(receipt)))):
                            record.update({"status": "needs_review", "reason": "receipt_changed",
                                           "recovered": True, "finished_at": _now()})
            _atomic_json(self.state_path, state)
            return state
        state = self._new_state()
        _atomic_json(self.state_path, state)
        return state

    def _save(self) -> None:
        self.state["updated_at"] = _now()
        _atomic_json(self.state_path, self.state)

    def _case_path(self, case_id: str) -> Path:
        return self.output / "cases" / case_id

    def _stage_config(self, case: Dict[str, object], stage: str) -> Dict[str, object]:
        configured = self.case_configs[case["id"]]
        value = configured.get("stages", {}).get(stage, self.config["stages"][stage])  # type: ignore[index]
        if isinstance(value, dict):
            return dict(value)
        return {"command": value}

    def _paths(self, case: Dict[str, object], stage: str) -> Dict[str, str]:
        case_dir = self._case_path(case["id"])
        stage_dir = case_dir / stage
        dependency = DEPENDENCY[stage]
        if dependency:
            input_path = case_dir / dependency
            handoff = input_path / ".pipeline-handoff.json"
        else:
            input_path = Path(case.get("source", ""))
            handoff = stage_dir / ".pipeline-handoff.json"
        return {
            "case_id": case["id"],
            "case_dir": str(case_dir),
            "source": str(case.get("source", "")),
            "input": str(input_path),
            "output": str(stage_dir),
            "handoff": str(handoff),
            "manifest": str(handoff),
            "receipt": str(stage_dir / "receipt.json"),
            "stage": stage,
        }

    def _validate_receipt(self, stage_spec: Dict[str, object],
                          paths: Dict[str, str]) -> Optional[Dict[str, str]]:
        receipt_path = _receipt_path(stage_spec.get("receipt"), paths)
        if receipt_path is None:
            return None
        if not receipt_path.is_file():
            raise StageReceiptError("missing stage receipt: %s" % receipt_path)
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise StageReceiptError("invalid stage receipt: %s" % receipt_path) from error
        expected = stage_spec.get("receipt")
        expected_status = "completed"
        expected_sha = None
        if isinstance(expected, dict):
            expected_status = expected.get("status", expected_status)
            expected_sha = expected.get("sha256", expected.get("sha"))
        if not isinstance(receipt, dict) or receipt.get("status") != expected_status:
            raise StageReceiptError("stage receipt is not completed: %s" % receipt_path)
        receipt_sha = receipt.get("sha256", receipt.get("sha"))
        if not isinstance(receipt_sha, str) or not receipt_sha:
            raise StageReceiptError("stage receipt has no sha256: %s" % receipt_path)
        if expected_sha is not None and receipt_sha != expected_sha:
            raise StageReceiptError("stage receipt sha256 differs: %s" % receipt_path)
        artifacts = receipt.get("artifacts")
        if artifacts is None and receipt.get("artifact") is not None:
            artifacts = [receipt["artifact"]]
        if artifacts is not None:
            if not isinstance(artifacts, list) or not artifacts:
                raise StageReceiptError("stage receipt artifacts are empty: %s" % receipt_path)
            for artifact in artifacts:
                if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
                    raise StageReceiptError("stage receipt artifact is invalid: %s" % receipt_path)
                artifact_path = Path(artifact["path"])
                if not artifact_path.is_absolute():
                    artifact_path = receipt_path.parent / artifact_path
                artifact_sha = artifact.get("sha256", artifact.get("sha"))
                if not artifact_path.is_file() or artifact_sha != _file_sha256(artifact_path):
                    raise StageReceiptError("stage receipt artifact hash differs: %s" % artifact_path)
        return {"path": str(receipt_path), "sha256": _file_sha256(receipt_path)}

    def _claim(self, stage: str) -> Optional[Dict[str, object]]:
        with self._lock:
            for case in self.state["cases"].values():  # type: ignore[union-attr]
                record = case["stages"][stage]
                if record.get("status") != "pending":
                    continue
                dependency = DEPENDENCY[stage]
                if dependency:
                    dependency_status = case["stages"][dependency].get("status")
                    if dependency_status in {"failed", "skipped", "needs_review"}:
                        record.update({"status": "skipped", "reason":
                                       "dependency_%s_%s" % (dependency, dependency_status),
                                       "finished_at": _now()})
                        self._save()
                        continue
                    if dependency_status != "completed":
                        continue
                record["status"] = "running"
                record["attempts"] = int(record.get("attempts", 0)) + 1
                record["started_at"] = _now()
                self._save()
                return dict(case)
        return None

    def _stage_complete(self, stage: str) -> bool:
        with self._lock:
            return all(case["stages"][stage].get("status") in TERMINAL
                       for case in self.state["cases"].values())  # type: ignore[union-attr]

    def _execute(self, case: Dict[str, object], stage: str) -> None:
        case_id = case["id"]
        stage_dir = self._case_path(case_id) / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        paths = self._paths(case, stage)
        stage_spec = self._stage_config(case, stage)
        raw_command = stage_spec["command"]
        command = tuple(_replace_tokens(item, paths)
                        for item in _command(raw_command))
        cwd = Path(_replace_tokens(str(stage_spec.get("cwd", paths["case_dir"])), paths))
        if not cwd.is_dir():
            cwd = stage_dir
        environment = os.environ.copy()
        environment.update({
            "PIPELINE_CASE_ID": case_id,
            "PIPELINE_STAGE": stage,
            "PIPELINE_INPUT": paths["input"],
            "PIPELINE_OUTPUT": paths["output"],
            "PIPELINE_HANDOFF": paths["handoff"],
            "PIPELINE_MANIFEST": paths["manifest"],
        })
        attempt = self._attempt(case_id, stage)
        stdout_path = stage_dir / ("attempt-%02d.stdout.log" % attempt)
        stderr_path = stage_dir / ("attempt-%02d.stderr.log" % attempt)
        with self._slots:
            self.command_runner(command, cwd, environment, stdout_path, stderr_path)
        receipt = self._validate_receipt(stage_spec, paths)
        handoff = {
            "schema": SCHEMA,
            "case_id": case_id,
            "stage": stage,
            "status": "completed",
            "artifact_dir": str(stage_dir.relative_to(self.output)),
            "input": paths["input"],
            "command_digest": _digest(list(command)),
            "completed_at": _now(),
        }
        if receipt:
            handoff["receipt"] = receipt["path"]
            handoff["receipt_sha256"] = receipt["sha256"]
        _atomic_json(stage_dir / ".pipeline-handoff.json", handoff)
        handoff_sha256 = _file_sha256(stage_dir / ".pipeline-handoff.json")
        with self._lock:
            record = self.state["cases"][case_id]["stages"][stage]
            record.update({"status": "completed", "finished_at": handoff["completed_at"],
                           "handoff": str(stage_dir / ".pipeline-handoff.json"),
                           "handoff_sha256": handoff_sha256})
            if receipt:
                record.update({"receipt": receipt["path"],
                               "receipt_sha256": receipt["sha256"]})
            self._save()

    def _attempt(self, case_id: str, stage: str) -> int:
        with self._lock:
            return int(self.state["cases"][case_id]["stages"][stage].get("attempts", 1))

    def _worker(self, stage: str) -> None:
        while True:
            case = self._claim(stage)
            if case is None:
                if self._stage_complete(stage):
                    return
                time.sleep(self.poll_interval)
                continue
            case_id = case["id"]
            try:
                self._execute(case, stage)
            except StageReceiptError as error:
                with self._lock:
                    record = self.state["cases"][case_id]["stages"][stage]
                    record.update({"status": "needs_review", "last_error": str(error),
                                   "finished_at": _now()})
                    self._save()
            except Exception as error:
                with self._lock:
                    record = self.state["cases"][case_id]["stages"][stage]
                    attempts = int(record.get("attempts", 1))
                    if attempts < self.max_attempts:
                        record.update({"status": "pending", "last_error": str(error)})
                    else:
                        record.update({"status": "failed", "last_error": str(error),
                                       "finished_at": _now()})
                    self._save()

    def run(self) -> Dict[str, object]:
        """Run the three workers and return the persisted final report."""
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="pipeline") as pool:
            futures = [pool.submit(self._worker, stage) for stage in STAGES]
            for future in futures:
                future.result()
        with self._lock:
            case_values = list(self.state["cases"].values())
            has_warning = any(
                record.get("status") in {"failed", "skipped", "needs_review"}
                for case in case_values
                for record in case["stages"].values()
            )
            report = {
                "schema": SCHEMA,
                "status": "completed_with_warnings" if has_warning else "completed",
                "cases": self.state["cases"],
                "finished_at": _now(),
            }
            _atomic_json(self.output / "pipeline-report.json", report)
            return report


def load_config(path: Path) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run repo, QA, and task stage workers")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-inflight", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=2)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_inflight <= 0 or args.max_attempts <= 0:
        raise SystemExit("max-inflight and max-attempts must be positive")
    runner = PipelineRunner(load_config(args.config), args.output,
                            max_inflight=args.max_inflight,
                            max_attempts=args.max_attempts, resume=args.resume)
    report = runner.run()
    print(json.dumps({"status": report["status"], "output": str(args.output)},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
