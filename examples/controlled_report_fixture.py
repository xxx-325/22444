"""Materialize the small controlled repository used by the source-strategy pilot.

This is a synthetic mechanism fixture, including its scripted messages. It is
not an automatically generated development conversation. The repository is a
generic local JSON reporter; an external customer's update protocol exists only
in the dialogue. QA and future tasks use the normal model-backed pipeline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


FILES = {
    "README.md": """# Reporting

A small Python library and CLI for local JSON reports. `render_json(rows)`
serializes dictionaries in input order with sorted keys. Local reports omit
values that are None. `report-json input.json` reads a JSON list and prints it.
`UNSET` is an in-process sentinel available to callers; it is not a JSON value.
Run tests with `PYTHONPATH=src python -m pytest tests`.
""",
    "pyproject.toml": """[build-system]\nrequires = [\"setuptools>=68\"]\nbuild-backend = \"setuptools.build_meta\"\n\n[project]\nname = \"controlled-report\"\nversion = \"0.1.0\"\n\n[project.scripts]\nreport-json = \"reporting.cli:main\"\n""",
    "src/reporting/__init__.py": "from .core import UNSET, normalize_row, render_json\n\n__all__ = [\"UNSET\", \"normalize_row\", \"render_json\"]\n",
    "src/reporting/core.py": """import json


UNSET = object()


def normalize_row(row):
    \"\"\"Prepare a row for the existing JSON report.\"\"\"
    return {key: value for key, value in row.items() if value is not None}


def render_json(rows):
    return json.dumps([normalize_row(row) for row in rows], sort_keys=True)
""",
    "src/reporting/cli.py": """import argparse
import json
from pathlib import Path

from .core import render_json


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(\"input\", type=Path)
    args = parser.parse_args(argv)
    rows = json.loads(args.input.read_text(encoding=\"utf-8\"))
    print(render_json(rows))
    return 0


if __name__ == \"__main__\":
    raise SystemExit(main())
""",
    "tests/test_core.py": """from reporting.core import render_json


def test_json_report_is_sorted_and_omits_none_values():
    assert render_json([{"b": 2, "a": None}]) == '[{\"b\": 2}]'
""",
}


def build_repo(target: Path) -> Path:
    target = target.resolve()
    if target.exists():
        raise FileExistsError(target)
    for name, content in FILES.items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(target), "-c", "user.name=fixture", "-c",
         "user.email=fixture@example.invalid", "commit", "-qm", "initial"],
        check=True,
    )
    return target


def write_dialogue(output: Path) -> None:
    records = [
        {"id": "e1", "kind": "message", "role": "user", "text":
         "我们维护一个小型报表工具，当前入口是 reporting/cli.py，后面要把记录交给客户的更新接口。"},
        {"id": "e2", "kind": "message", "role": "assistant", "text":
         "我会先检查 reporting/core.py 和现有测试，保持当前 JSON 行为。"},
        {"id": "e3", "kind": "message", "role": "user", "text":
         "Maple 客户确认了更新协议：note 为显式 null 时必须省略该字段，保留远端原备注；其他字段的显式 null 必须原样发送，表示清空。非空 note 正常发送，UNSET 表示未设置、应省略。这只适用于交给 Maple 的更新数据，不改变我们现有本地 JSON 报表。以后增加 Maple 批量更新入口也沿用这条约定。"},
        {"id": "e4", "kind": "message", "role": "assistant", "text":
         "收到。Maple 更新数据和现有本地报表是两个不同用途，我会按各自的约定处理。"},
        {"id": "e5", "kind": "message", "role": "user", "text":
         "对，准备 Maple 后续接入时沿用刚才确认的字段协议，本地报表继续保持现有行为。"},
        {"id": "e6", "kind": "message", "role": "assistant", "text":
         "明白，后续接入 Maple 时会使用这条已确认的约定。"},
    ]
    output.write_text(json.dumps({"version": 1, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")


def write_events(output: Path) -> None:
    value = {
        "version": 1,
        "events": [{
            "id": "controlled-maple-contract",
            "kind": "compatibility_contract",
            "memory_kind": "M1",
            "source_ids": ["e3"],
            "used_by": ["e5"],
            "context_ids": ["e4", "e6"],
            "qa_mode": "code",
        }],
    }
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--external-events", type=Path, required=True)
    args = parser.parse_args(argv)
    build_repo(args.repository)
    args.input.parent.mkdir(parents=True, exist_ok=True)
    args.external_events.parent.mkdir(parents=True, exist_ok=True)
    write_dialogue(args.input)
    write_events(args.external_events)
    print(json.dumps({"repository": str(args.repository.resolve()),
                      "dialogue": str(args.input.resolve()),
                      "external_events": str(args.external_events.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
