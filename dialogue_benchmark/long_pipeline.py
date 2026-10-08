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
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable

from .collection import run_collection
from .episode_input import load_episode_manifest
from .pipeline_runner import PipelineRunner
from .task_eval.artifacts import read, save


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_receipt(path: Path, artifacts: Iterable[Path], *, result="completed") -> None:
    paths = [Path(item).resolve() for item in artifacts if Path(item).is_file()]
    if not paths:
        raise RuntimeError("stage produced no artifacts")
    relative = [{"path": str(item.relative_to(path.parent)), "sha256": _sha256(item)}
                for item in paths]
    payload = {"status": "completed", "result": result,
               "sha256": _sha256(paths[0]), "artifacts": relative}
    save(path, payload)


def _manifest_from_collection(root: Path) -> Path:
    candidates = sorted(root.rglob("dialogue-package/manifest.json"))
    if len(candidates) != 1:
        raise RuntimeError("expected one dialogue package manifest, found %d" % len(candidates))
    manifest = candidates[0]
    loaded = load_episode_manifest(manifest)
    quality = loaded["manifest"].get("quality", {})
    if quality.get("passed") is not True:
        raise RuntimeError("dialogue manifest quality did not pass")
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
            raise RuntimeError("dialogue manifest failed %s" % minimum)
    closure = checks.get("external_event_closure", {})
    if profile.get("require_external_event_closure") and closure.get("passed") is not True:
        raise RuntimeError("dialogue manifest external source closure failed")
    scope = checks.get("external_scope_policy", {})
    if profile.get("require_declared_external_scope") and scope.get("passed") is not True:
        raise RuntimeError("dialogue manifest external scope failed")
    # Long cases require fifty source-grounded code rounds. Newer manifests
    # expose an explicit check; older manifests can still be counted from the
    # exported public rows without treating tool calls as rounds.
    required_rounds = profile.get("min_user_code_rounds")
    if required_rounds:
        rounds = checks.get("user_code_rounds", {})
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
            raise RuntimeError("dialogue manifest has %d/%d complete code rounds" %
                               (actual, required_rounds))
    return manifest


def _run_logged(command, cwd: Path, stdout: Path, stderr: Path) -> int:
    stdout.parent.mkdir(parents=True, exist_ok=True)
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        return subprocess.run(command, cwd=str(cwd), stdout=out, stderr=err, check=False).returncode


def _repo_stage(args) -> int:
    plan = Path(args.input).resolve()
    output = Path(args.output).resolve()
    collection = output / "collection"
    state = collection / "collection.json"
    run_collection(plan, collection, Path(args.simulator_path), Path(args.env_file),
                   Path(args.python), resume=state.is_file(), dialogue_only=True)
    manifest = _manifest_from_collection(collection)
    save(output / "manifest-path.json", {"manifest": str(manifest),
                                         "plan": str(plan)})
    _write_receipt(output / "stage-receipt.json", [manifest, collection / "collection.json",
                                                     output / "manifest-path.json"])
    return 0


def _route_command(route: str, manifest: Path, output: Path, args, plan: dict):
    evaluation = plan.get("evaluation", {})
    command = [str(args.python), str(ROOT / "run_episode.py"), "--episode-manifest", str(manifest),
               "--qa-source", route, "--simulator-path", str(Path(args.simulator_path).resolve()),
               "--env-file", str(Path(args.env_file).resolve()), "--output", str(output), "--qa-only"]
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
        ready = (route_dir / "qa" / "manifest.json").is_file()
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
        public = read(route_dir / "qa/qa-public.json") if (route_dir / "qa/qa-public.json").is_file() else {}
        records[route] = {"returncode": outcomes[route],
                          "status": read(pipeline).get("status"),
                          "qualified": outcomes[route] == 0 and bool(public.get("questions"))
                          and all(q.get("status") == "approved" for q in public["questions"])}
        artifacts.append(pipeline)
    save(output / "qa-summary.json", {"manifest": str(manifest), "routes": records})
    artifacts.append(output / "qa-summary.json")
    if not any(item["qualified"] for item in records.values()):
        _write_receipt(output / "stage-receipt.json", artifacts, result="routes_failed")
        return 1
    _write_receipt(output / "stage-receipt.json", artifacts,
                   result="completed_with_route_warning" if not all(item["qualified"] for item in records.values()) else "completed")
    return 0


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
    qualified = pipeline.get("status") not in {"failed", "interrupted"} and bool(public.get("questions")) and all(
        q.get("status") == "approved" for q in public["questions"])
    output.mkdir(parents=True, exist_ok=True)
    task_manifest = output / "tasks" / "manifest.json"
    stage_summary = output / "task-stage.json"
    if not qualified:
        save(stage_summary, {"status": "skipped", "reason": "external_qa_not_qualified",
                             "tasks": []})
        _write_receipt(output / "stage-receipt.json", [stage_summary], result="skipped")
        return 0
    if not (output / "qa").exists():
        shutil.copytree(external / "qa", output / "qa")
    evaluation = plan.get("evaluation", {})
    command = [str(args.python), str(ROOT / "run_episode.py"), "--episode-manifest", str(manifest),
               "--qa-source", "external", "--simulator-path", str(Path(args.simulator_path).resolve()),
               "--env-file", str(Path(args.env_file).resolve()), "--output", str(output),
               "--qa-count", str(evaluation.get("qa_count", 40)), "--resume-tasks",
               "--task-count", str(evaluation.get("task_count", 1)),
               "--task-budget", str(evaluation.get("task_budget", 2)),
               "--task-workers", str(evaluation.get("task_workers", 2)),
               "--revisions", str(evaluation.get("revisions", 3))]
    code = _run_logged(command, ROOT, output / "stdout.log", output / "stderr.log")
    if code or not task_manifest.is_file():
        return code or 1
    save(stage_summary, {"status": "completed", "manifest": str(task_manifest)})
    _write_receipt(output / "stage-receipt.json", [task_manifest, stage_summary], result="completed")
    return 0


def _stage_main(args) -> int:
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
        prepared = Path(project.get("prepared_config", ""))
        prepared = (config_path.parent / prepared).resolve() if not prepared.is_absolute() else prepared.resolve()
        if prepared.name != "config.json" or not prepared.is_file():
            raise ValueError("prepared_config must be an existing project/config.json: %s" % prepared)
        one = dict(master)
        one["runtime_config"] = str((config_path.parent / master["runtime_config"]).resolve())
        one["dialogue_only"] = True
        one["projects"] = [{**project, "prepared_config": str(prepared)}]
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
                                    "--python", str(python_path)], "cwd": str(ROOT), "receipt": "{output}/stage-receipt.json"},
        }})
    return {"cases": cases, "stages": {stage: {"command": [str(python_path), "-c", "pass"]}
                                       for stage in ("repo", "qa", "task")}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulator-path", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", choices=("repo", "qa", "task"))
    parser.add_argument("--input", type=Path)
    args = parser.parse_args(argv)
    if args.stage:
        if args.input is None:
            parser.error("--stage requires --input")
        return _stage_main(args)
    if args.config is None:
        parser.error("--config is required for the pipeline run")
    output = args.output.resolve()
    config = build_config(args.config, output, simulator_path=args.simulator_path,
                          env_file=args.env_file, python=args.python)
    save(output / "pipeline-config.json", config)
    report = PipelineRunner(config, output, resume=args.resume).run()
    print(json.dumps({"status": report["status"], "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
