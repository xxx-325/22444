"""Export a private offline viewer from existing run artifacts only."""

import argparse
import json
import os
from pathlib import Path

from .graph import graph_at
from .llm import outbound_guard


def read_artifact(directory, name, default=None):
    path = directory / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def build_payload(directory):
    graph = read_artifact(directory, "graph.json")
    records = read_artifact(directory, "normalized.json")
    if not graph or not records:
        raise ValueError("graph.json and normalized.json are required")
    states = {}
    last_files = None
    state_key = None
    snapshot_states = {}
    for snapshot in graph["snapshots"]:
        if snapshot["files"] != last_files:
            state_key = str(snapshot["at"])
            states[state_key] = graph_at(graph, snapshot["at"])
            last_files = snapshot["files"]
        snapshot_states[str(snapshot["at"])] = state_key
    result = read_artifact(directory, "qa-reviewed.json")
    if result is None:
        result = read_artifact(directory, "qa.json")
    payload = {
        "run": directory.name, "records": records, "graph": graph,
        "states": states, "snapshot_states": snapshot_states,
        "facts": read_artifact(directory, "facts.json", []),
        "candidates": read_artifact(directory, "candidates.json", []),
        "result": result,
        "review": read_artifact(directory, "review-retry.json",
                                read_artifact(directory, "review.json")),
        "failure": read_artifact(directory, "failure.json"),
        "scope": read_artifact(directory, "scope.json"),
    }
    roots = sorted({r.get("workspace", "") for r in records if r.get("workspace")},
                   key=len, reverse=True)

    def clean(value):
        if isinstance(value, str):
            for root in roots:
                for variant in {root, root.replace("\\", "/"), root.replace("\\", "\\\\")}:
                    value = value.replace(variant.rstrip("/\\") + "\\\\", "")
                    value = value.replace(variant.rstrip("/\\") + "\\", "")
                    value = value.replace(variant.rstrip("/\\") + "/", "")
                    value = value.replace(variant, "<workspace>")
            return value
        if isinstance(value, dict):
            return {clean(k): clean(v) for k, v in value.items() if k != "workspace"}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    payload = clean(payload)
    outbound_guard(json.dumps(payload, ensure_ascii=False), "")
    return payload


def export_viewer(directory, output):
    payload = build_payload(directory)
    assets = Path(__file__).parent / "viewer"
    data = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c").replace(
        "\u2028", "\\u2028").replace("\u2029", "\\u2029")
    html = (assets / "index.html").read_text(encoding="utf-8")
    html = html.replace("/* VIEWER_CSS */", (assets / "style.css").read_text(encoding="utf-8"))
    html = html.replace("/* VIEWER_JS */", (assets / "app.js").read_text(encoding="utf-8"))
    html = html.replace("/* VIEWER_DATA */", data)
    with output.open("x", encoding="utf-8") as stream:
        os.chmod(output, 0o600)
        stream.write(html)
    return len(html.encode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Export an existing run as a private offline HTML viewer")
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    size = export_viewer(args.run, args.output)
    print("Viewer exported: %s (%d bytes)" % (args.output, size))


if __name__ == "__main__":
    main()
