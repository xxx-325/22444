"""Run the fixed three-case long-dialogue pipeline.

The adapter only composes the repository's existing collection, QA, and task
commands.  It does not make model requests itself; each worker owns its
normal command and writes a small receipt for :mod:`pipeline_runner`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable, Optional

from .collection import run_collection
from .episode_input import load_episode_manifest
from .pipeline_runner import PipelineRunner
from .task_eval.artifacts import has_eligible_qa, read, save, qa_inputs
from .task_eval.report import write_report
from .task_eval.runtime import preflight_openhands_runtime


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_receipt(path: Path, artifacts: Iterable[Path], *, result="completed") -> None:
    path = Path(path).resolve()
    paths = [Path(item).resolve() for item in artifacts if Path(item).is_file()]
    if not paths:
        raise RuntimeError("stage produced no artifacts")
    relative = [{"path": str(item.relative_to(path.parent)), "sha256": _sha256(item)}
                for item in paths]
    payload = {"status": "completed", "result": result,
               "sha256": _sha256(paths[0]), "artifacts": relative}
    save(path, payload)


def _manifest_from_collection(root: Path):
    candidates = sorted(root.rglob("dialogue-package/manifest.json"))
    if len(candidates) != 1:
        raise RuntimeError("expected one dialogue package manifest, found %d" % len(candidates))
    manifest = candidates[0]
    loaded = load_episode_manifest(manifest)
    quality = loaded["manifest"].get("quality", {})
    warnings = []
    if quality.get("passed") is not True:
        warnings.append("dialogue_quality_below_target")
    profile = quality.get("profile", {})
    checks = quality.get("checks", {})
    check_names = {
        "min_increments": "increments", "min_visible_messages": "visible_messages",
        "min_public_tool_events": "public_tool_events", "min_external_events": "external_events",
        "min_distinct_external_foci": "distinct_external_foci",
    }
    for minimum, check_name in check_names.items():
        required = profile.get(minimum)
        if required is None:
            continue
        check = checks.get(check_name, {})
        if check.get("passed") is not True or check.get("actual", 0) < required:
            warnings.append("%s_shortfall" % check_name)
    closure = checks.get("external_event_closure", {})
    if profile.get("require_external_event_closure") and closure.get("passed") is not True:
        raise RuntimeError("dialogue manifest external source closure failed")
    scope = checks.get("external_scope_policy", {})
    if profile.get("require_declared_external_scope") and scope.get("passed") is not True:
        raise RuntimeError("dialogue manifest external scope failed")
    for hard_check in ("forbidden_documents", "control_transitions"):
        check = checks.get(hard_check, {})
        if check and check.get("passed") is not True:
            raise RuntimeError("dialogue manifest failed %s" % hard_check)
    # Use the configured number of complete user/code exchanges.
    required_rounds = profile.get("min_user_code_rounds")
    if required_rounds:
        rounds = checks.get("complete_user_code_rounds", {})
        actual = rounds.get("actual")
        if actual is None:
            dialogue = [json.loads(line) for line in loaded["dialogue"].read_text().splitlines()]
            pending = actual = 0
            for row in dialogue:
                if row.get("kind") == "user" and row.get("text", "").strip():
                    pending += 1
                elif (
                    row.get("kind") == "assistant"
                    and row.get("phase", "final") == "final"
                    and row.get("text", "").strip()
                    and pending
                ):
                    pending -= 1
                    actual += 1
        if actual < required_rounds:
            warnings.append("user_code_rounds_shortfall")
    return manifest, sorted(set(warnings))


def _run_logged(command, cwd: Path, stdout: Path, stderr: Path) -> int:
    stdout.parent.mkdir(parents=True, exist_ok=True)
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        return subprocess.run(command, cwd=str(cwd), stdout=out, stderr=err, check=False).returncode


def _archive_incomplete(directory: Path) -> Optional[Path]:
    """Move an incomplete stage directory aside before a fresh attempt."""
    if not directory.exists():
        return None
    archive_root = directory.parent / ".incomplete"
    archive_root.mkdir(parents=True, exist_ok=True)
    target = archive_root / directory.name
    suffix = 1
    while target.exists():
        target = archive_root / (directory.name + "-%d" % suffix)
        suffix += 1
    shutil.move(str(directory), str(target))
    return target


def _repo_stage(args) -> int:
    plan = Path(args.input).resolve()
    output = Path(args.output).resolve()
    # Fail before collection retries if the selected simulator interpreter
    # cannot import its host-side OpenHands adapter.
    preflight_openhands_runtime(args.simulator_path, python_executable=args.python)
    collection = output / "collection"
    state = collection / "collection.json"
    run_collection(plan, collection, Path(args.simulator_path), Path(args.env_file),
                   Path(args.python), resume=state.is_file(), dialogue_only=True)
    manifest, quality_warnings = _manifest_from_collection(collection)
    external_information_plan = collection / "external-information-plan.json"
    save(output / "manifest-path.json", {"manifest": str(manifest),
                                         "plan": str(plan),
                                         "external_information_plan": str(external_information_plan)
                                         if external_information_plan.is_file() else None,
                                         "quality_warnings": quality_warnings})
    save(output / "quality-summary.json", {
        "status": "completed_with_warnings" if quality_warnings else "completed",
        "warnings": quality_warnings,
        "manifest": str(manifest),
    })
    receipt_artifacts = [manifest, collection / "collection.json",
                         output / "manifest-path.json", output / "quality-summary.json"]
    if external_information_plan.is_file():
        receipt_artifacts.append(external_information_plan)
    _write_receipt(output / "stage-receipt.json", receipt_artifacts,
                   result="completed_with_warnings" if quality_warnings else "completed")
    return 0


def _route_command(route: str, manifest: Path, output: Path, args, plan: dict):
    evaluation = plan.get("evaluation", {})
    command = [str(args.python), str(ROOT / "run_episode.py"), "--episode-manifest", str(manifest),
               "--qa-source", route, "--simulator-path", str(Path(args.simulator_path).resolve()),
               "--env-file", str(Path(args.env_file).resolve()), "--output", str(output), "--qa-only",
               "--parallel-workers", str(evaluation.get("parallel_workers", 10))]
    if evaluation.get("model_request_chars") is not None:
        command += ["--model-request-chars", str(evaluation["model_request_chars"])]
    if route == "external":
        command += ["--qa-count", str(evaluation.get("qa_count", 40))]
        if evaluation.get("group_budget") is not None:
            command += ["--group-budget", str(evaluation["group_budget"])]
    else:
        command += ["--general-count", str(evaluation.get("general_count", 40)),
                    "--code-count", str(evaluation.get("code_count", 40))]
    return command


def _qa_stage(args) -> int:
    input_root = Path(args.input).resolve()
    output = Path(args.output).resolve()
    manifest_info = read(input_root / "manifest-path.json")
    manifest = Path(manifest_info["manifest"]).resolve()
    plan = read(manifest_info["plan"])
    routes = ("graph", "external")
    output.mkdir(parents=True, exist_ok=True)

    def run(route):
        route_dir = output / route
        qa_dir = route_dir / "qa"
        ready = _qa_output_ready(qa_dir)
        pipeline = route_dir / "pipeline.json"
        if ready and pipeline.is_file():
            try:
                if read(pipeline).get("status") in {"completed", "completed_with_warnings"}:
                    return route, 0
            except (OSError, ValueError, TypeError, AttributeError):
                pass
        if not ready:
            _archive_incomplete(qa_dir)
        command = _route_command(route, manifest, route_dir, args, plan)
        if ready:
            command.append("--resume-tasks")
        return route, _run_logged(command, ROOT, route_dir / "stdout.log", route_dir / "stderr.log")

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = dict(pool.map(run, routes))
    records = {}
    artifacts = []
    for route in routes:
        route_dir = output / route
        pipeline = route_dir / "pipeline.json"
        if not pipeline.is_file():
            save(pipeline, {"status": "failed", "reason": "route_command_failed",
                            "route": route, "returncode": outcomes[route]})
        ready = _qa_output_ready(route_dir / "qa")
        public = read(route_dir / "qa/qa-public.json") if ready else {}
        try:
            eligible = has_eligible_qa(route_dir / "qa", include_provisional=True)
        except (OSError, KeyError, TypeError, ValueError):
            eligible = False
        approved = eligible and bool(public.get("questions")) and all(
            q.get("status") == "approved" for q in public["questions"])
        try:
            route_status = read(pipeline).get("status")
        except (OSError, KeyError, TypeError, ValueError, AttributeError):
            route_status, ready = "invalid_output", False
        ready = ready and route_status not in {"failed", "blocked", "interrupted"}
        evaluation = plan.get("evaluation", {})
        target = (evaluation.get("qa_count", 40) if route == "external" else
                  evaluation.get("general_count", 40) + evaluation.get("code_count", 40))
        records[route] = {"returncode": outcomes[route],
                          "status": route_status, "output_ready": ready,
                          "qualified": ready and approved,
                          "eligible": ready and eligible,
                          "provisional": ready and eligible and not approved,
                          "published_qa": len(public.get("questions", [])),
                          "target": target, "shortfall": max(0, target - len(public.get("questions", [])))}
        artifacts.append(pipeline)
    save(output / "qa-summary.json", {"manifest": str(manifest), "routes": records})
    artifacts.append(output / "qa-summary.json")
    if not any(item["output_ready"] for item in records.values()):
        _write_receipt(output / "stage-receipt.json", artifacts, result="routes_failed")
        return 1
    _write_receipt(output / "stage-receipt.json", artifacts,
                   result="completed_with_warnings" if any(
                       not item["qualified"] or item["shortfall"] or item["returncode"]
                       or item["status"] != "completed"
                       for item in records.values()) else "completed")
    return 0


def _qa_output_ready(qa_dir):
    try:
        manifest = read(qa_dir / "manifest.json")
        public = read(qa_dir / "qa-public.json")
        return (isinstance(manifest, dict) and isinstance(public, dict)
                and isinstance(public.get("questions"), list)
                and public.get("status") not in {"blocked", "interrupted"})
    except (OSError, ValueError, TypeError):
        return False


def _task_stage(args) -> int:
    input_root = Path(args.input).resolve()
    output = Path(args.output).resolve()
    info = read(input_root.parent / "repo" / "manifest-path.json")
    manifest = Path(info["manifest"]).resolve()
    plan = read(info["plan"])
    external = input_root / "external"
    public_path = external / "qa/qa-public.json"
    public = read(public_path) if public_path.is_file() else {}
    pipeline = read(external / "pipeline.json") if (external / "pipeline.json").is_file() else {}
    try:
        eligible_items = qa_inputs(external / "qa", include_provisional=True)
    except (OSError, KeyError, TypeError, ValueError):
        eligible_items = []
    all_approved = bool(public.get("questions")) and all(
        question.get("status") == "approved" for question in public["questions"]
    )
    allow_provisional = any(item.get("provisional") for item in eligible_items)
    qualified = (
        pipeline.get("status") not in {"failed", "blocked", "interrupted"}
        and bool(eligible_items)
    )
    output.mkdir(parents=True, exist_ok=True)
    task_manifest = output / "tasks" / "manifest.json"
    stage_summary = output / "task-stage.json"
    if not qualified:
        target = plan.get("evaluation", {}).get("task_count", 1)
        task_data = read(task_manifest) if task_manifest.is_file() else {}
        if not task_data.get("tasks"):
            task_data = {
                "status": "completed_with_warnings", "reason": "external_qa_not_qualified",
                "stop_reason": "external_qa_not_qualified", "tasks": [],
                "target": target, "completed": 0, "shortfall": target,
                "notes": ["No source-validated external QA was available for requirements."],
            }
            save(task_manifest, task_data)
        write_report(output / "tasks", task_data)
        save(stage_summary, {"status": "completed_with_warnings",
                             "reason": "external_qa_not_qualified", "target": target,
                             "completed": task_data.get("completed", 0),
                             "shortfall": max(0, target - task_data.get("completed", 0)),
                             "tasks": task_data.get("tasks", [])})
        _write_receipt(output / "stage-receipt.json", [task_manifest, stage_summary,
                       output / "tasks/report.md"], result="completed_with_warnings")
        return 0
    qa_dir = output / "qa"
    qa_ready = all((qa_dir / name).is_file() for name in ("manifest.json", "qa-public.json"))
    if not qa_ready:
        _archive_incomplete(qa_dir)
        shutil.copytree(external / "qa", output / "qa")
    tasks_dir = output / "tasks"
    if tasks_dir.exists() and not (tasks_dir / "manifest.json").is_file():
        _archive_incomplete(tasks_dir)
    evaluation = plan.get("evaluation", {})
    command = [str(args.python), str(ROOT / "run_episode.py"), "--episode-manifest", str(manifest),
               "--qa-source", "external", "--simulator-path", str(Path(args.simulator_path).resolve()),
               "--env-file", str(Path(args.env_file).resolve()), "--output", str(output),
               "--qa-count", str(evaluation.get("qa_count", 40)), "--resume-tasks",
               "--task-count", str(evaluation.get("task_count", 1)),
               "--task-budget", str(evaluation.get("task_budget", 2)),
               "--task-workers", str(evaluation.get("task_workers", 2)),
               "--revisions", str(evaluation.get("revisions", 3)),
               "--model-request-chars", str(evaluation.get("model_request_chars", 96000))]
    if allow_provisional and not all_approved:
        command.append("--allow-provisional")
    code = _run_logged(command, ROOT, output / "stdout.log", output / "stderr.log")
    if code or not task_manifest.is_file():
        return code or 1
    task_data = read(task_manifest)
    result = "completed" if task_data.get("status") == "complete" else "completed_with_warnings"
    save(stage_summary, {"status": result, "manifest": str(task_manifest),
                         "completed": task_data.get("completed", 0),
                         "shortfall": task_data.get("shortfall", 0),
                         "provisional_completed": task_data.get("provisional_completed", 0)})
    _write_receipt(output / "stage-receipt.json", [task_manifest, stage_summary], result=result)
    return 0


def _stage_main(args) -> int:
    if args.stage == "task" and args.task_slot_directory:
        os.environ["DIALOGUE_TASK_SLOT_DIR"] = str(args.task_slot_directory.resolve())
        os.environ["DIALOGUE_TASK_SLOT_LIMIT"] = str(args.task_slots)
    return {"repo": _repo_stage, "qa": _qa_stage, "task": _task_stage}[args.stage](args)


def build_config(config_path: Path, output: Path, *, simulator_path: Path,
                 env_file: Path, python: Path) -> dict:
    config_path = config_path.resolve()
    master = read(config_path)
    projects = master.get("projects")
    if not isinstance(projects, list) or len(projects) != 3:
        raise ValueError("long dialogue input must define exactly three projects")
    input_dir = output / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    python_path = python.absolute()
    cases = []
    for project in projects:
        one = dict(master)
        one["runtime_config"] = str((config_path.parent / master["runtime_config"]).resolve())
        one["dialogue_only"] = True
        if project.get("prepared_config"):
            prepared = Path(project["prepared_config"])
            prepared = (
                (config_path.parent / prepared).resolve()
                if not prepared.is_absolute()
                else prepared.resolve()
            )
            if not prepared.is_file():
                raise ValueError("prepared_config must be an existing prepared config: %s" % prepared)
            prepared_project = prepared
            try:
                prepared_payload = read(prepared)
            except (OSError, ValueError, TypeError):
                prepared_payload = {}
            prepared_projects = (
                prepared_payload.get("projects")
                if isinstance(prepared_payload, dict)
                else None
            )
            if isinstance(prepared_projects, list) and len(prepared_projects) == 1:
                entry = prepared_projects[0]
                if entry.get("id") == project["id"] and entry.get("prepared_config"):
                    candidate = Path(entry["prepared_config"])
                    prepared_project = (prepared.parent / candidate).resolve()
            if not prepared_project.is_file():
                raise ValueError("prepared project config is missing: %s" % prepared_project)
            one["projects"] = [{**project, "prepared_config": str(prepared_project)}]
        elif project.get("brief"):
            # A fresh case starts from a model-authored repository.  Keep this
            # as a one-project collection plan so the existing collection
            # stages (project -> scenario -> dialogue) remain the source of
            # truth while the outer PipelineRunner can run cases in parallel.
            one["projects"] = [{key: value for key, value in project.items()
                                if key != "prepared_config"}]
        else:
            raise ValueError("each project needs either brief or prepared_config")
        plan = input_dir / (project["id"] + ".json")
        save(plan, one)
        common = [str(python_path), "-m", "dialogue_benchmark.long_pipeline",
                  "--stage", "repo", "--input", "{source}", "--output", "{output}",
                  "--simulator-path", str(simulator_path.resolve()), "--env-file", str(env_file.resolve()),
                  "--python", str(python_path)]
        cases.append({"id": project["id"], "source": str(plan), "stages": {
            "repo": {"command": common, "cwd": str(ROOT), "receipt": "{output}/stage-receipt.json"},
            "qa": {"command": [str(python_path), "-m", "dialogue_benchmark.long_pipeline",
                                  "--stage", "qa", "--input", "{input}", "--output", "{output}",
                                  "--simulator-path", str(simulator_path.resolve()), "--env-file", str(env_file.resolve()),
                                  "--python", str(python_path)], "cwd": str(ROOT), "receipt": "{output}/stage-receipt.json"},
            "task": {"command": [str(python_path), "-m", "dialogue_benchmark.long_pipeline",
                                    "--stage", "task", "--input", "{input}", "--output", "{output}",
                                    "--simulator-path", str(simulator_path.resolve()), "--env-file", str(env_file.resolve()),
                                    "--task-slot-directory", str((output / ".task-slots").resolve()),
                                    "--task-slots", str(master.get("evaluation", {}).get("max_task_workers", 3)),
                                    "--python", str(python_path)], "cwd": str(ROOT), "receipt": "{output}/stage-receipt.json"},
        }})
    return {"cases": cases, "stages": {stage: {"command": [str(python_path), "-c", "pass"]}
                                       for stage in ("repo", "qa", "task")}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "examples/collection-three-business.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulator-path", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--stage", choices=("repo", "qa", "task"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--task-slot-directory", type=Path)
    parser.add_argument("--task-slots", type=int, default=3)
    args = parser.parse_args(argv)
    if args.task_slots < 1:
        parser.error("--task-slots must be positive")
    if args.stage:
        if args.input is None:
            parser.error("--stage requires --input")
        return _stage_main(args)
    output = args.output.resolve()
    config = build_config(args.config, output, simulator_path=args.simulator_path,
                          env_file=args.env_file, python=args.python)
    save(output / "pipeline-config.json", config)
    if args.max_attempts <= 0:
        parser.error("--max-attempts must be positive")
    report = PipelineRunner(config, output, resume=args.resume,
                            max_attempts=args.max_attempts).run()
    print(json.dumps({"status": report["status"], "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
