"""Run conversion, QA extraction, and paired repository tasks with saved stage outputs."""

import argparse
import hashlib
from pathlib import Path

from convert_session import convert
from dialogue_benchmark.cli import main as generate_qa
from dialogue_benchmark.task_eval.artifacts import copy_tree, read, save
from dialogue_benchmark.task_eval.run import main as run_tasks
from dialogue_benchmark.task_eval.runtime import configure
from dialogue_benchmark.task_eval.versions import baseline_version, pin_baseline
from dialogue_benchmark.task_eval.retention import compact_run
from dialogue_benchmark.task_eval.report import write_report
from render_run import render


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-run", "simulator-path", "env-file", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
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
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (root / "tasks").exists() or ((root / "qa").exists() and not args.resume_tasks):
        parser.error("Use an output without QA/task results; previous runs are retained")
    if args.resume_tasks and not all((root / "qa" / name).is_file()
                                     for name in ("manifest.json", "qa-public.json")):
        parser.error("Resuming tasks requires completed QA outputs")
    state = {"source_run": str(args.source_run.resolve()),
             "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
             "phase": "input", "status": "running"}
    def phase(name):
        state["phase"] = name
        save(root / "pipeline.json", state)
        print("Pipeline:", name, flush=True)

    try:
        phase("input")
        if not (root / "input").exists():
            convert(args.source_run / "session.jsonl", root / "input")
        conversion = read(root / "input/conversion.json")
        if hashlib.sha256((args.source_run / "session.jsonl").read_bytes()).hexdigest() != conversion["source_sha256"]:
            raise ValueError("Converted input belongs to a different source session")
        if hashlib.sha256((root / "input/dialogue.json").read_bytes()).hexdigest() != conversion["output_sha256"]:
            raise ValueError("Converted dialogue changed after conversion")
        if not (root / "baseline").exists():
            copy_tree(args.source_run / "workspace/candidate", root / "baseline")
            save(root / "baseline.json", pin_baseline(root / "baseline"))
        version = baseline_version(root / "baseline")
        if version != {k: read(root / "baseline.json")[k] for k in version}:
            raise ValueError("Pinned baseline changed")
        config = configure(args.simulator_path, args.source_run / "private/checkpoint.json", args.env_file)
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
            "--simulator-path", str(args.simulator_path), "--source-run", str(args.source_run),
            "--qa-run", str(root / "qa"), "--env-file", str(args.env_file),
            "--output", str(root / "tasks"), "--baseline", str(root / "baseline"),
            "--count", str(args.task_count), "--task-budget", str(args.task_budget),
            "--workers", str(args.task_workers), "--revisions", str(args.revisions),
        ]
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
