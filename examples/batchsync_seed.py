"""Create a dependency-free project and three consecutive reference changes.

These commits supply real code tasks to the dialogue simulator. This script
does not create conversation messages, customer facts, QA, or task answers.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from textwrap import dedent


BASE = {
    ".gitignore": "__pycache__/\n*.pyc\n.pytest_cache/\n",
    "README.md": """\
        # BatchSync

        BatchSync prepares dictionary records for local reports and data delivery.
        It is a Python library with no network or third-party dependencies.
        Python 3.11 or newer is supported. Run `python -m unittest discover -s tests -v`.

        `batchsync.render_report(records)` returns JSON with sorted keys and keeps
        record order. Local reports omit null values. Input records are unchanged.
        """,
    "batchsync/__init__.py": "from .report import render_report\n",
    "batchsync/report.py": '''\
        import json


        def render_report(records):
            """Render a local report, omitting fields whose value is None."""
            rows = [{key: value for key, value in row.items() if value is not None}
                    for row in records]
            return json.dumps(rows, ensure_ascii=False, sort_keys=True)
        ''',
    "tests/test_report.py": '''\
        import unittest

        from batchsync import render_report


        class ReportTests(unittest.TestCase):
            def test_nulls_are_omitted_without_mutating_input(self):
                rows = [{"z": 2, "a": None}]
                self.assertEqual(render_report(rows), '[{"z": 2}]')
                self.assertEqual(rows, [{"z": 2, "a": None}])

            def test_unicode_and_order(self):
                self.assertEqual(render_report([{"b": "é", "a": 0}, {}]),
                                 '[{"a": 0, "b": "é"}, {}]')
        ''',
}

PROJECTION = {
    "batchsync/__init__.py": ("from .report import render_report\n"
                              "from .projection import UNSET, project_record, render_records\n"),
    "batchsync/projection.py": '''\
        import json


        UNSET = object()


        def project_record(record, *, omit_null_fields=(), omit_all_nulls=False):
            """Omit unset fields and apply a caller-supplied null policy."""
            excluded = set(omit_null_fields)
            return {key: value for key, value in record.items()
                    if value is not UNSET
                    and not (value is None and (omit_all_nulls or key in excluded))}


        def render_records(records, *, omit_null_fields=(), omit_all_nulls=False):
            """Encode a delivery body without changing the input dictionaries."""
            fields = tuple(omit_null_fields)
            rows = [project_record(row, omit_null_fields=fields,
                                   omit_all_nulls=omit_all_nulls) for row in records]
            return json.dumps(rows, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":"))
        ''',
    "docs/projection.md": """\
        # Field projection

        `project_record(record, *, omit_null_fields=(), omit_all_nulls=False)`
        returns a new dictionary. `UNSET` is an in-process sentinel for missing
        data and is always omitted. None is an explicit null and is kept by
        default. Selected null fields, or all null fields, can be omitted.
        Non-null values are unaffected. The caller supplies the delivery policy.

        `render_records(records, *, omit_null_fields=(), omit_all_nulls=False)`
        returns compact JSON with sorted keys, Unicode preserved and records in
        input order. This delivery API does not change the local report format.
        """,
    "tests/test_projection.py": '''\
        import unittest

        from batchsync import UNSET, project_record, render_records, render_report


        class ProjectionTests(unittest.TestCase):
            def test_missing_and_null_are_distinct(self):
                self.assertEqual(project_record({"x": UNSET, "y": None}), {"y": None})

            def test_field_policy_is_selective(self):
                row = {"a": None, "b": None, "c": 0, "d": False, "e": ""}
                self.assertEqual(project_record(row, omit_null_fields=["a"]),
                                 {"b": None, "c": 0, "d": False, "e": ""})
                self.assertIn("a", row)

            def test_omit_all_does_not_drop_false_values(self):
                self.assertEqual(project_record({"x": None, "y": 0}, omit_all_nulls=True),
                                 {"y": 0})

            def test_deterministic_body_and_local_compatibility(self):
                self.assertEqual(render_records([{"b": None, "a": "é"}]),
                                 '[{"a":"é","b":null}]')
                self.assertEqual(render_report([{"b": None}]), '[{}]')

            def test_policy_iterator_applies_to_every_row(self):
                self.assertEqual(render_records([{"x": None}, {"x": None}],
                                                omit_null_fields=iter(["x"])), '[{},{}]')
        ''',
}

BATCHING = {
    "batchsync/__init__.py": PROJECTION["batchsync/__init__.py"]
    + "from .batching import make_batches\n",
    "batchsync/batching.py": '''\
        from .projection import render_records


        def make_batches(records, *, max_bytes, omit_null_fields=(), omit_all_nulls=False):
            """Pack consecutive records into UTF-8 JSON bodies within max_bytes."""
            if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 2:
                raise ValueError("max_bytes must be an integer of at least 2")
            fields = tuple(omit_null_fields)

            def render(rows):
                return render_records(rows, omit_null_fields=fields,
                                      omit_all_nulls=omit_all_nulls)

            bodies, current = [], []
            for row in records:
                single = render([row])
                if len(single.encode("utf-8")) > max_bytes:
                    raise ValueError("record exceeds byte limit")
                combined = render(current + [row])
                if len(combined.encode("utf-8")) > max_bytes:
                    bodies.append(render(current))
                    current = [row]
                else:
                    current.append(row)
            if current:
                bodies.append(render(current))
            return bodies
        ''',
    "docs/batching.md": """\
        # Batch preparation

        `make_batches(records, *, max_bytes, omit_null_fields=(), omit_all_nulls=False)`
        returns JSON body strings. The byte limit is provided by the caller;
        it includes UTF-8 encoded array brackets and commas. Records retain
        input order and are not split. Projection happens before sizing.
        Each body is filled until the next record would exceed the limit.

        Empty input produces no batches. Invalid limits or any individual record
        that cannot fit raise ValueError. The function returns no partial result
        on failure and never performs network operations.
        """,
    "tests/test_batching.py": '''\
        import json
        import unittest

        from batchsync import make_batches


        class BatchingTests(unittest.TestCase):
            def test_exact_boundary_and_order(self):
                rows = [{"i": 1}, {"i": 2}, {"i": 3}]
                bodies = make_batches(rows, max_bytes=17)
                self.assertEqual(bodies, ['[{"i":1},{"i":2}]', '[{"i":3}]'])
                self.assertEqual([row for body in bodies for row in json.loads(body)], rows)

            def test_counts_bytes_including_punctuation(self):
                self.assertEqual(make_batches([{"x": "é"}], max_bytes=12), ['[{"x":"é"}]'])
                with self.assertRaises(ValueError):
                    make_batches([{"x": "é"}], max_bytes=11)

            def test_projection_precedes_size(self):
                self.assertEqual(make_batches([{"x": None}], max_bytes=4,
                                              omit_null_fields=["x"]), ['[{}]'])

            def test_empty_and_invalid(self):
                self.assertEqual(make_batches([], max_bytes=2), [])
                for limit in (1, 0, True, 2.5):
                    with self.subTest(limit=limit), self.assertRaises(ValueError):
                        make_batches([], max_bytes=limit)

            def test_late_oversized_record_does_not_return_partial_result(self):
                with self.assertRaises(ValueError):
                    make_batches([{}, {"long": "value"}], max_bytes=4)
        ''',
}

RECEIPTS = {
    "batchsync/__init__.py": BATCHING["batchsync/__init__.py"]
    + "from .receipts import classify_receipts\n",
    "batchsync/receipts.py": '''\
        def classify_receipts(receipts, *, committed_states, pending_states):
            """Group record IDs by a caller-supplied acknowledgement vocabulary."""
            committed, pending = set(committed_states), set(pending_states)
            if committed & pending:
                raise ValueError("receipt state groups overlap")
            result = {"committed": [], "pending": [], "rejected": []}
            seen = set()
            for receipt in receipts:
                record_id, state = receipt["id"], receipt["state"]
                if record_id in seen:
                    raise ValueError("duplicate receipt id")
                seen.add(record_id)
                group = ("committed" if state in committed else
                         "pending" if state in pending else "rejected")
                result[group].append(record_id)
            return result
        ''',
    "docs/receipts.md": """\
        # Receipt classification

        `classify_receipts(receipts, *, committed_states, pending_states)` takes
        dictionaries containing `id` and `state`. It returns lists of record IDs
        under `committed`, `pending`, and `rejected`, preserving input order within
        each list. Unknown states are rejected; duplicate IDs and overlapping
        state groups raise ValueError. Input dictionaries are not changed.

        A transport acknowledgement is not inherently proof that data was
        committed. The caller supplies the destination's vocabulary. This module
        neither resends records nor tracks remote state.
        """,
    "tests/test_receipts.py": '''\
        import unittest

        from batchsync import classify_receipts


        class ReceiptTests(unittest.TestCase):
            def test_custom_vocabulary_and_order(self):
                rows = [{"id": "a", "state": "written"},
                        {"id": "b", "state": "queued"},
                        {"id": "c", "state": "invalid"},
                        {"id": "d", "state": "written"}]
                self.assertEqual(classify_receipts(rows, committed_states=["written"],
                                                  pending_states=["queued"]),
                                 {"committed": ["a", "d"], "pending": ["b"], "rejected": ["c"]})
                self.assertEqual(rows[0], {"id": "a", "state": "written"})

            def test_empty(self):
                self.assertEqual(classify_receipts([], committed_states=[], pending_states=[]),
                                 {"committed": [], "pending": [], "rejected": []})

            def test_duplicate_and_overlap(self):
                with self.assertRaises(ValueError):
                    classify_receipts([], committed_states=["same"], pending_states=["same"])
                with self.assertRaises(ValueError):
                    classify_receipts([{"id": 1, "state": "a"}, {"id": 1, "state": "b"}],
                                      committed_states=["a"], pending_states=["b"])
        ''',
}

STAGES = [
    ("base", "feat:(report) add deterministic local JSON reports", BASE),
    ("projection", "feat:(projection) support configurable null and unset fields", PROJECTION),
    ("batching", "feat:(batching) pack delivery bodies within a UTF-8 byte limit", BATCHING),
    ("receipts", "feat:(receipts) classify delivery acknowledgements", RECEIPTS),
]


def build_repo(target: Path) -> dict:
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=False)
    subprocess.run(["git", "init", "-q", "-b", "main", str(target)], check=True)
    commits = []
    for stage, title, files in STAGES:
        for name, content in files.items():
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(dedent(content), encoding="utf-8")
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                       cwd=target, check=True)
        subprocess.run(["git", "add", "."], cwd=target, check=True)
        subprocess.run(["git", "-c", "user.name=BatchSync seed", "-c",
                        "user.email=seed@example.invalid", "commit", "-qm", title],
                       cwd=target, check=True)
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=target, text=True).strip()
        commits.append({"stage": stage, "commit": sha})
    return {"repository": str(target), "python": sys.version.split()[0],
            "test_command": "python -m unittest discover -s tests -v",
            "base": commits[0]["commit"], "commits": commits[1:]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_repo(args.repository)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
