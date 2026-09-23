"""Run conversion, QA extraction, and paired repository tasks with saved stage outputs."""

import argparse
import hashlib
from pathlib import Path

from convert_session import convert
from dialogue_benchmark.cli import main as generate_qa
from dialogue_benchmark.task_eval.artifacts import copy_tree, fingerprint, read, save
from dialogue_benchmark.task_eval.run import main as run_tasks
from dialogue_benchmark.task_eval.runtime import configure
from dialogue_benchmark.task_eval.versions import baseline_version, pin_baseline
from dialogue_benchmark.task_eval.retention import compact_run
from dialogue_benchmark.task_eval.report import write_report
from render_run import render
from dialogue_benchmark.episode_input import load_episode_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("simulator-path", "env-file", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-run", type=Path)
    source.add_argument("--episode-manifest", type=Path)
    parser.add_argument("--source-event", action="append", default=[])
    parser.add_argument("--source-object", action="append", default=[])
    parser.add_argument("--design-probe", action="store_true")
    parser.add_argument("--general-count", type=int, default=40)
    parser.add_argument("--code-count", type=int, default=40)
    parser.add_argument("--task-count", type=int, default=12)
    parser.add_argument("--task-budget", type=int, default=24)
    parser.add_argument("--parallel-workers", type=int, default=10)
    parser.add_argument("--task-workers", type=int, default=3)
    parser.add_argument("--revisions", type=int, default=5)
    parser.add_argument("--reuse-facts", type=Path)
    parser.add_argument("--resume-tasks", action="store_true",
                        help="Reuse completed QA and start repository tasks in an empty tasks directory")
    args = parser.parse_args(argv)
    package = load_episode_manifest(args.episode_manifest) if args.episode_manifest else None
    source_run = args.source_run or args.episode_manifest.resolve().parent
    dialogue_path = package["dialogue"] if package else source_run / "session.jsonl"
    snapshot_path = package["snapshot"] if package else source_run / "workspace/candidate"
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (root / "tasks").exists() or ((root / "qa").exists() and not args.resume_tasks):
        parser.error("Use an output without QA/task results; previous runs are retained")
    if args.resume_tasks and not all((root / "qa" / name).is_file()
                                     for name in ("manifest.json", "qa-public.json")):
        parser.error("Resuming tasks requires completed QA outputs")
    state = {"source_run": str(source_run.resolve()),
             "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
             "phase": "input", "status": "running"}
    def phase(name):
        state["phase"] = name
        save(root / "pipeline.json", state)
        print("Pipeline:", name, flush=True)

    try:
        phase("input")
        if not (root / "input").exists():
            convert(dialogue_path, root / "input")
        conversion = read(root / "input/conversion.json")
        if hashlib.sha256(dialogue_path.read_bytes()).hexdigest() != conversion["source_sha256"]:
            raise ValueError("Converted input belongs to a different source session")
        if hashlib.sha256((root / "input/dialogue.json").read_bytes()).hexdigest() != conversion["output_sha256"]:
            raise ValueError("Converted dialogue changed after conversion")
        if not (root / "baseline").exists():
            copy_tree(snapshot_path, root / "baseline")
            save(root / "baseline.json", pin_baseline(root / "baseline"))
        version = baseline_version(root / "baseline")
        if package and fingerprint(root / "baseline") != package["manifest"]["snapshot"]["sha256"]:
            raise ValueError("Saved baseline differs from episode snapshot")
        if version != {k: read(root / "baseline.json")[k] for k in version}:
            raise ValueError("Pinned baseline changed")
        config = configure(args.simulator_path, source_run / "private/checkpoint.json", args.env_file,
                           **({"control_config": package["control_config"]} if package else {}))
        model = config["judge"]
        endpoint = model["base_url"].rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        phase("qa_reused" if args.resume_tasks else "qa")
        qa_args = [
            str(root / "input/dialogue.json"), "--output", str(root / "qa"),
            "--qa-mode", "both", "--general-count", str(args.general_count),
            "--code-count", str(args.code_count), "--questions-per-group", "3",
            "--parallel-workers", str(args.parallel_workers), "--chunk-chars", "24000",
            "--adaptive-subgraphs", "--expansion-budget", "3", "--allow-network",
            "--endpoint", endpoint, "--model", model["model"], "--key-env", model["key_env"],
        ]
        if args.reuse_facts:
            qa_args += ["--reuse-facts", str(args.reuse_facts)]
        for event_id in args.source_event:
            qa_args += ["--source-event", event_id]
        for name in args.source_object:
            qa_args += ["--source-object", name]
        if args.resume_tasks:
            if read(root / "qa/manifest.json")["input_sha256"] != conversion["output_sha256"]:
                raise ValueError("Completed QA belongs to a different converted dialogue")
        else:
            status = generate_qa(qa_args)
            if status:
                raise RuntimeError("QA generation did not complete; see qa/error.json")
        render(root)
        phase("repository_tasks")
        task_args = [
            "--simulator-path", str(args.simulator_path), "--source-run", str(source_run),
            "--qa-run", str(root / "qa"), "--env-file", str(args.env_file),
            "--output", str(root / "tasks"), "--baseline", str(root / "baseline"),
            "--count", str(args.task_count), "--task-budget", str(args.task_budget),
            "--workers", str(args.task_workers), "--revisions", str(args.revisions),
        ]
        if package:
            task_args += ["--control-config", str(package["control_config"])]
        if args.design_probe:
            task_args += ["--design-probe"]
        status = run_tasks(task_args)
        if status == 0:
            phase("checkpoint_recovery")
            status = run_tasks(task_args + ["--recover-checkpoints"])
        render(root)
        state["status"] = "completed" if status == 0 else "failed"
        phase("complete")
        if status == 0 and (root / "tasks/manifest.json").exists():
            compact_run(root)
            write_report(root / "tasks", read(root / "tasks/manifest.json"))
            render(root)
        return status
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     error_type=type(error).__name__)
        save(root / "pipeline.json", state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
