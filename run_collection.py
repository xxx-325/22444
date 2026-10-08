"""Prepare a fixed collection of projects, dialogues and paired memory tasks."""

import argparse
from pathlib import Path
import sys

from dialogue_benchmark.collection import run_collection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("config", "output", "simulator-path", "env-file"):
        parser.add_argument("--" + option, required=True, type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable),
                        help="Python environment with the existing simulator dependencies")
    parser.add_argument("--resume", action="store_true",
                        help="Continue the same plan and runtime, retaining all prior stage attempts")
    parser.add_argument("--dialogue-only", action="store_true",
                        help="Stop after writing each dialogue package")
    args = parser.parse_args(argv)
    run_collection(args.config, args.output, args.simulator_path, args.env_file, args.python,
                   resume=args.resume, dialogue_only=args.dialogue_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
