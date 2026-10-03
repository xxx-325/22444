"""Run conversion, QA extraction, and paired repository tasks with saved stage outputs."""

import argparse
import hashlib
from pathlib import Path

from convert_session import convert
from dialogue_benchmark.cli import main as generate_qa
from dialogue_benchmark.llm import DEFAULT_REQUEST_TIMEOUT
from dialogue_benchmark.task_eval.artifacts import copy_tree, fingerprint, read, save
from dialogue_benchmark.task_eval.run import main as run_tasks
from dialogue_benchmark.task_eval.runtime import configure, preflight_openhands_runtime
from dialogue_benchmark.task_eval.versions import baseline_version, pin_baseline
from dialogue_benchmark.task_eval.retention import compact_run
from dialogue_benchmark.task_eval.report import write_report
from render_run import render
from dialogue_benchmark.episode_input import load_episode_manifest
from dialogue_benchmark.collection import episode_usage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("simulator-path", "env-file", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-run", type=Path)
    source.add_argument("--episode-manifest", type=Path)
    parser.add_argument("--qa-source", choices=("graph", "external"), default="graph")
    parser.add_argument("--external-events", type=Path)
    parser.add_argument("--source-event", action="append", default=[])
    parser.add_argument("--source-object", action="append", default=[])
    parser.add_argument("--design-probe", action="store_true")
    parser.add_argument("--qa-only", action="store_true",
                        help="Finish after QA extraction without creating or evaluating repository tasks")
    parser.add_argument("--qa-count", type=int, help="External QA target (default: 40)")
    parser.add_argument("--group-budget", type=int, help="External event exploration budget")
    parser.add_argument("--general-count", type=int, help="Graph general QA target (default: 40)")
    parser.add_argument("--code-count", type=int, help="Graph code QA target (default: 40)")
    parser.add_argument("--task-count", type=int, default=12)
    parser.add_argument("--task-budget", type=int, default=24)
    parser.add_argument("--parallel-workers", type=int, default=10)
    parser.add_argument("--model-request-chars", type=int, default=32000,
                        help="Maximum serialized QA and task-construction request size, including evidence and prompt")
    parser.add_argument("--task-workers", type=int, default=3)
    parser.add_argument("--revisions", type=int, default=5)
    parser.add_argument("--reuse-facts", type=Path)
    parser.add_argument("--resume-tasks", action="store_true",
                        help="Reuse completed QA and start repository tasks in an empty tasks directory")
    args = parser.parse_args(argv)
    if args.qa_only and args.resume_tasks:
        parser.error("--qa-only cannot be combined with --resume-tasks")
    if args.qa_source == "external" and (args.general_count is not None or args.code_count is not None):
        parser.error("External QA uses --qa-count, not separate general/code counts")
    if args.qa_source == "graph" and (args.qa_count is not None or args.group_budget is not None):
        parser.error("--qa-count and --group-budget require --qa-source external")
    for name in ("qa_count", "group_budget", "general_count", "code_count", "model_request_chars"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            parser.error("--%s must be positive" % name.replace("_", "-"))
    package = load_episode_manifest(args.episode_manifest) if args.episode_manifest else None
    packaged_events = package.get("external_events") if package else None
    if args.qa_source == "external" and args.external_events is None:
        args.external_events = packaged_events
    if (args.qa_source == "external") != (args.external_events is not None):
        parser.error("External QA requires an event file from the episode manifest or --external-events")
    if args.external_events is not None and not args.external_events.is_file():
        parser.error("External event file does not exist")
    if (args.external_events is not None and packaged_events is not None
            and args.external_events.read_bytes() != packaged_events.read_bytes()):
        parser.error("Explicit external events differ from the episode manifest")
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
    usage_saved = False
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
        # Fail before any QA model requests when the selected interpreter
        # cannot import the host-side OpenHands adapter.
        preflight_openhands_runtime(args.simulator_path)
        model = config["judge"]
        endpoint = model["base_url"].rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        phase("qa_reused" if args.resume_tasks else "qa")
        qa_args = [
            str(root / "input/dialogue.json"), "--output", str(root / "qa"),
            "--parallel-workers", str(args.parallel_workers), "--chunk-chars", "24000",
            "--model-request-chars", str(args.model_request_chars),
            "--allow-network",
            "--endpoint", endpoint, "--model", model["model"], "--key-env", model["key_env"],
            "--request-timeout", str(model.get("request_timeout", DEFAULT_REQUEST_TIMEOUT)),
        ]
        if args.reuse_facts:
            qa_args += ["--reuse-facts", str(args.reuse_facts)]
        if model.get("reasoning_effort"):
            qa_args += ["--reasoning-effort", model["reasoning_effort"]]
        if args.qa_source == "external":
            qa_args += ["--qa-source", "external", "--external-events", str(args.external_events),
                        "--qa-count", str(args.qa_count or 40),
                        "--repository", str(root / "baseline")]
            if args.group_budget is not None:
                qa_args += ["--group-budget", str(args.group_budget)]
        else:
            qa_args += ["--qa-mode", "both", "--general-count", str(args.general_count or 40),
                        "--code-count", str(args.code_count or 40), "--questions-per-group", "3",
                        "--adaptive-subgraphs", "--expansion-budget", "3"]
        for event_id in args.source_event:
            qa_args += ["--source-event", event_id]
        for name in args.source_object:
            qa_args += ["--source-object", name]
        if args.resume_tasks:
            if read(root / "qa/manifest.json")["input_sha256"] != conversion["output_sha256"]:
                raise ValueError("Completed QA belongs to a different converted dialogue")
            if read(root / "qa/manifest.json").get("qa_source", "graph") != args.qa_source:
                raise ValueError("Completed QA uses a different source mode")
            if args.qa_source == "external" and read(root / "qa/manifest.json").get("qa_mode") != "memory":
                raise ValueError("External QA must use the unified memory types; regenerate QA")
        else:
            status = generate_qa(qa_args)
            if status:
                raise RuntimeError("QA generation did not complete; see qa/error.json")
        render(root)
        qa_result = read(root / "qa/qa-public.json")
        if qa_result.get("status") == "failed":
            state["stop_reason"] = "qa_generation_failed"
            raise RuntimeError("QA generation failed; see qa/qa-audit.json")
        if args.qa_only or not qa_result["questions"]:
            if not args.qa_only:
                task_manifest = {"target": args.task_count, "tasks": [],
                                 "stop_reason": "no_eligible_qa"}
                save(root / "tasks/manifest.json", task_manifest)
                write_report(root / "tasks", task_manifest)
            state.update(status="completed", stop_reason=(
                "qa_only" if qa_result["questions"] else "no_eligible_qa"))
            save(root / "usage.json", episode_usage(root))
            usage_saved = True
            phase("complete")
            render(root)
            return 0
        phase("repository_tasks")
        task_args = [
            "--simulator-path", str(args.simulator_path), "--source-run", str(source_run),
            "--qa-run", str(root / "qa"), "--env-file", str(args.env_file),
            "--output", str(root / "tasks"), "--baseline", str(root / "baseline"),
            "--count", str(args.task_count), "--task-budget", str(args.task_budget),
            "--workers", str(args.task_workers), "--revisions", str(args.revisions),
            "--model-request-chars", str(args.model_request_chars),
        ]
        if package:
            task_args += ["--control-config", str(package["control_config"])]
        if args.design_probe:
            task_args += ["--design-probe"]
        status = run_tasks(task_args)
        render(root)
        state["status"] = "completed" if status == 0 else "failed"
        phase("complete")
        save(root / "usage.json", episode_usage(root))
        usage_saved = True
        if status == 0 and (root / "tasks/manifest.json").exists():
            compact_run(root)
            write_report(root / "tasks", read(root / "tasks/manifest.json"))
            render(root)
        return status
    except BaseException as error:
        if not usage_saved:
            save(root / "usage.json", episode_usage(root))
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     error_type=type(error).__name__)
        save(root / "pipeline.json", state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
