"""Repeat one admitted frozen task twice with fresh solvers and reversed order."""

import argparse
from pathlib import Path

from .artifacts import copy_tree, qa_fingerprint, read, save
from .metrics import compare_trials
from .report import write_report
from .run import evaluate, unchanged
from .runtime import configure
from .versions import baseline_version, pin_baseline, source_version


def source_inputs(task):
    """Read only the admitted inputs, independently of prior trial outcomes."""
    manifest = read(task.parent / "manifest.json")
    receipt = read(task / "frozen.json")
    qa = read(task / "author-reference/qa.json")
    if "qa_sha256" in receipt and receipt["qa_sha256"] != qa_fingerprint(qa):
        raise ValueError("Frozen QA changed before repetition")
    baseline = Path(manifest["baseline"]).resolve()
    version = baseline_version(baseline)
    if (receipt.get("validation", {}).get("VERDICT") != "accept"
            or not isinstance(receipt.get("accepted_attempt"), int)
            or receipt["accepted_attempt"] < 0 or receipt["qa_id"] != qa["id"]):
        raise ValueError("Source task has no matching admitted frozen receipt")
    if (manifest["baseline_sha256"] != receipt["baseline_sha256"]
            or not unchanged(receipt, task / "frozen", baseline)):
        raise ValueError("Source baseline or frozen specification changed")
    execution = {key: manifest["execution"][key] for key in
                 ("agent_requests", "agent_tokens", "agent_seconds")}
    if any(value <= 0 for value in execution.values()):
        raise ValueError("Source agent budgets must be positive")
    return {"source_task": str(task), "source_manifest": str(task.parent / "manifest.json"),
            "baseline": str(baseline), "baseline_version": version,
            "frozen": receipt, "qa": qa, "config": manifest["config"],
            "execution": execution}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("source-task", "simulator-path", "env-file", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args(argv)
    task, output = args.source_task.resolve(), args.output.resolve()
    source = source_inputs(task)
    baseline_source = Path(source["baseline"])
    if any(path == output or path in output.parents for path in (task, baseline_source)):
        parser.error("Output must be outside the source task and baseline")
    if output.exists():
        parser.error("Use a new output directory; previous experiments are retained")
    output.mkdir(parents=True)
    save(output / "source.json", source)
    manifest = {"source_task": str(task), "source_receipt": "source.json",
                "comparison": "Historical answer injection; no memory retriever",
                "baseline_sha256": source["frozen"]["baseline_sha256"],
                "spec_sha256": source["frozen"]["spec_sha256"],
                "execution": source["execution"], "target": 2,
                "status": "running", "stop_reason": "running", "tasks": [
                    {"task": "pair-%02d" % (index + 1), "status": "pending",
                     "qa_id": source["qa"]["id"], "type": source["qa"]["type"],
                     "order": (["without_memory", "with_memory"] if index == 0
                               else ["with_memory", "without_memory"])}
                    for index in range(2)]}

    def persist():
        manifest["completed"] = sum(row["status"] == "evaluated" for row in manifest["tasks"])
        manifest["shortfall"] = manifest["target"] - manifest["completed"]
        save(output / "manifest.json", manifest)
        write_report(output, manifest)

    def verify_source():
        if source_inputs(task) != source:
            raise ValueError("Source inputs changed during repetition")

    root = None
    persist()
    try:
        config = configure(args.simulator_path, output / "source.json", args.env_file)
        baseline = output / "baseline"
        copy_tree(baseline_source, baseline, include_caches=True)
        version = pin_baseline(baseline)
        if version["content_sha256"] != source["frozen"]["baseline_sha256"]:
            raise ValueError("Copied baseline differs from the frozen source")
        save(output / "baseline.json", dict(version, source=str(baseline_source)))
        manifest.update(config=config, baseline=str(baseline), baseline_version=version,
                        evaluator_version=source_version(Path(__file__).resolve().parents[2], "dialogue_benchmark"),
                        simulator_version=source_version(args.simulator_path, "simulator"))
        options = dict(max_requests=source["execution"]["agent_requests"],
                       max_tokens=source["execution"]["agent_tokens"],
                       max_seconds=source["execution"]["agent_seconds"])
        for index, row in enumerate(manifest["tasks"]):
            verify_source()
            root = output / row["task"]
            row["status"] = "running"
            persist()
            try:
                copy_tree(task / "frozen", root / "frozen", include_caches=True)
                save(root / "frozen.json", source["frozen"])
                save(root / "author-reference/qa.json", source["qa"])
                comparison = evaluate({"qa": source["qa"]}, root, baseline,
                                      source["frozen"], config, options, index)
                row.update(status="evaluated", comparison=comparison)
            except (Exception, KeyboardInterrupt) as error:
                row.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "error",
                           error_type=type(error).__name__, detail=str(error))
                save(root / "failure.json", {key: value for key, value in row.items()
                                              if key != "comparison"})
                if isinstance(error, KeyboardInterrupt):
                    raise
            finally:
                comparison_path = root / "comparison.json"
                if comparison_path.exists():
                    row["comparison"] = read(comparison_path)
                row["paired_differences"] = compare_trials(row.get("comparison", {}))
                persist()
                save(root / "manifest.json", manifest)
            verify_source()
        manifest.update(status="completed", stop_reason="repetitions_completed")
    except (Exception, KeyboardInterrupt) as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
        manifest.update(status=status, stop_reason=status,
                        error_type=type(error).__name__, detail=str(error))
        raise
    finally:
        persist()
        if root is not None:
            save(root / "manifest.json", manifest)
    print("Frozen task repetitions complete:", output, flush=True)
    return int(any(row["status"] == "error" for row in manifest["tasks"]))


if __name__ == "__main__":
    raise SystemExit(main())
