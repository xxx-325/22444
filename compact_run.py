"""Compact completed episode outputs using the documented retention policy."""

import argparse
from pathlib import Path

from dialogue_benchmark.task_eval.artifacts import read
from dialogue_benchmark.task_eval.retention import compact_run
from dialogue_benchmark.task_eval.report import write_report
from render_run import render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    receipt = compact_run(args.run)
    if receipt["status"] == "deferred":
        print("Compaction deferred:", receipt["reason"], ", ".join(receipt.get("pending", [])))
        return
    write_report(args.run / "tasks", read(args.run / "tasks/manifest.json"))
    render(args.run)
    print("Retained code versions:", receipt["retained_code_versions"])
    print("Published QA with original inputs:", receipt["published_inputs_found"])
    print("Removed paths:", len(receipt["removed_paths"]))
    print("Docker errors:", len(receipt.get("docker", {}).get("errors", [])))


if __name__ == "__main__":
    main()
