import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = """
import hashlib, json
from dialogue_benchmark import cli
from dialogue_benchmark.graph import build_graph
from dialogue_benchmark.normalize import load_dialogue
records = load_dialogue("examples/dialogue.json")
args = cli._build_parser().parse_args(["examples/dialogue.json", "--output", "unused"])
scopes, _ = cli._prepare_code_scopes(build_graph(records), records, records[-1]["order"], args, 4)
print(hashlib.sha256(json.dumps(scopes, ensure_ascii=False, sort_keys=True,
                                default=list).encode()).hexdigest())
"""


class SubgraphDeterminismTests(unittest.TestCase):
    def test_adaptive_code_scopes_do_not_depend_on_hash_seed(self):
        # Resume compares recomputed scopes with saved ones, so a new process
        # must project the same scopes.
        digests = set()
        for seed in ("1", "2", "3"):
            result = subprocess.run(
                [sys.executable, "-c", SCRIPT], cwd=ROOT, capture_output=True, text=True,
                env=dict(os.environ, PYTHONHASHSEED=seed), check=True)
            digests.add(result.stdout.strip())
        self.assertEqual(len(digests), 1)


if __name__ == "__main__":
    unittest.main()
