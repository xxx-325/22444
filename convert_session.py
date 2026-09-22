"""Convert supported dialogue formats into a reusable, provenance-preserving input."""

import argparse
from collections import Counter
import hashlib
from pathlib import Path

from dialogue_benchmark.normalize import load_dialogue
from dialogue_benchmark.task_eval.artifacts import save


def convert(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    records = load_dialogue(source)
    if not records:
        raise ValueError("No supported dialogue records")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    normalized = output / "dialogue.json"
    save(normalized, {"version": 1, "records": records})
    if load_dialogue(normalized) != records:
        raise ValueError("Normalized input did not preserve source records")
    receipt = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output": normalized.name,
        "output_sha256": hashlib.sha256(normalized.read_bytes()).hexdigest(),
        "records": len(records),
        "kinds": dict(Counter(r["kind"] for r in records)),
        "visible_messages": dict(Counter(r["role"] for r in records if r["kind"] == "message")),
        "cutoff": records[-1]["order"],
    }
    save(output / "conversion.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New input directory")
    args = parser.parse_args(argv)
    receipt = convert(args.source, args.output)
    print("Converted %d records; visible messages: %s" %
          (receipt["records"], receipt["visible_messages"]))


if __name__ == "__main__":
    main()
