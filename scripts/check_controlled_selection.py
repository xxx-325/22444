"""Run a small, explicitly synthetic selection control without OpenHands."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dialogue_benchmark.task_eval.artifacts import save
from dialogue_benchmark.task_eval.run import construct
from dialogue_benchmark.task_eval.runtime import configure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "simulator-path", "control-config", "env-file"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = configure(args.simulator_path, None, args.env_file, control_config=args.control_config)
    results = []
    for label in ("external", "documented", "incomplete-answer"):
        root = args.output / label
        repo = root / "repository"
        repo.mkdir(parents=True)
        (repo / "export.py").write_text('def export_record(record):\n    return dict(record)\n')
        (repo / "README.md").write_text(
            "# Export library\nexport_record returns one record. No batch API exists yet.\n"
            "Future batch exports must preserve input order and keep the single-record API compatible.\n" +
            ("For Maple, omit note when its value is null, to preserve the destination note. "
             "Keep other null fields and retain non-null note values.\n" if label == "documented" else ""))
        history = "For customer Maple exports, omit note=null to avoid clearing the remote note. Keep other null fields and non-null notes. This customer agreement applies to future exports too."
        source = root / "qa-input.json"
        save(source, {"ref_to_source": {"source1": "e1"}})
        item = {"qa": {"id": "q1", "type": "constraint_followthrough",
                       "question": "What customer-specific export agreement must future exports preserve?",
                       "answer_points": [{"text": history if label != "incomplete-answer" else "Preserve customer compatibility."}]},
                "generation_input": str(source), "public_records": [
                    {"id": "e1", "original_id": "agreement", "kind": "message", "role": "user", "order": 1, "text": history}]}
        receipt = construct(item, root / "task", repo, config, 0, {}, selection_only=True)
        results.append({"fixture": label, "synthetic": True, "receipt": receipt})
        save(args.output / "results.json", results)


if __name__ == "__main__":
    main()
