"""Build a local, offline projection of saved benchmark artifacts.

This is a viewer adapter, not a graph builder or QA generator. It never reads
project source files, executes dialogue content, or makes model requests.
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dialogue_benchmark.cli import _USER_PATH
from dialogue_benchmark.security import credential_detected
from dialogue_benchmark.selection import build_audit
from dialogue_benchmark.storage import load as load_artifact


def load(run, name):
    return load_artifact(run / name)


def build(run):
    records = load(run, "normalized.json")
    graph = load(run, "graph.json")
    current = load(run, "current-graph.json")
    facts = load(run, "facts.json")
    raw_groups = load(run, "evidence-groups.json")
    audit = load(run, "qa-audit.json")
    public = load(run, "qa-public.json")["questions"]
    manifest = load(run, "manifest.json")
    workspaces = sorted({r.get("workspace") for r in records if r.get("workspace")}, key=len, reverse=True)
    changes = Counter()

    def clean(value):
        if isinstance(value, str):
            if credential_detected(value):
                changes["credential_fields_hidden"] += 1
                return "[此字段含疑似凭据，未载入展示页]"
            parts = re.split(r"(https?://[^\s<>]+)", value)
            for index in range(0, len(parts), 2):
                for root in workspaces:
                    # Keep project structure while preserving URL paths.
                    for spelling in (root, root.replace("\\", "\\\\")):
                        parts[index], n = re.subn(re.escape(spelling) + r"(?=[/\\\s\"'`]|$)", "<workspace>", parts[index])
                        changes["path_replacements"] += n
                parts[index], n = _USER_PATH.subn("~", parts[index])
                changes["path_replacements"] += n
            return "".join(parts)
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, dict):
            return {clean(k): clean(v) for k, v in value.items() if k not in {"workspace", "call_id", "sha256", "reasoning", "reasoning_content"}}
        return value

    nodes = {}
    edges = {}

    def node(item):
        nodes.setdefault(item["id"], item)

    def edge(a, b, kind, **extra):
        if a and b and a != b:
            edges[(a, b, kind)] = dict(source=a, target=b, kind=kind, **extra)

    event_by_id = {e["id"]: e for e in graph["events"]}
    for row in records:
        event = event_by_id.get(row["id"], {})
        label = {"message": "用户" if row.get("role") == "user" else "助手", "call": "工具调用", "result": "工具结果", "patch": "代码补丁"}.get(row["kind"], row["kind"])
        node(dict(id=row["id"], kind=row["kind"], lane="message" if row["kind"] == "message" else "patch" if row["kind"] == "patch" else "tool", label=label,
                  order=row["order"], source_kind=row.get("source_kind"), text=row.get("text") or json.dumps({k: v for k, v in row.items() if k in {"changes", "path", "content", "success", "name"}}, ensure_ascii=False, indent=2),
                  timestamp=row.get("timestamp"), original_source_line=row.get("original_source_line")))
        edge(event.get("call_source"), row["id"], "tool_result")

    for version in graph["versions"]:
        node(dict(id=version["id"], kind="version", lane="version", label=version["path"].split("/")[-1], path=version["path"], order=version["observed_at"],
                  previous=version.get("previous"), source=version["source"], text=version.get("content") or "该版本没有完整文件内容。", status=version.get("status")))
        node(dict(id=version["path"], kind="file", lane="object", label=version["path"].split("/")[-1], path=version["path"], order=0, text=version["path"]))
        edge(version["source"], version["id"], "version_source")
        edge(version.get("previous"), version["id"], "previous_version")
        edge(version["id"], version["path"], "file_version")
        for symbol in (version.get("code") or {}).get("symbols", []):
            sid = version["path"] + "::" + symbol["name"]
            node(dict(id=sid, kind="symbol", lane="object", label=symbol["name"], order=0, path=version["path"], text="此符号在对话暴露的代码版本中被静态解析到；不代表实际运行调用。"))
            edge(version["id"], sid, "symbol_version", line=symbol.get("line"))
    for item in current.get("edges", []):
        edge(item["from"], item["to"], item["kind"], provenance=item.get("source"))

    # Oversized source fragments live in group projections, not the full graph.
    # Preserve their saved contents and parent pointers instead of inventing files.
    for group in raw_groups:
        for field in ("dialogue", "events", "versions"):
            for row in group.get("scope", {}).get(field, []):
                if not row.get("id") or row["id"] in nodes:
                    continue
                node(dict(id=row["id"], kind="fragment", lane="tool", label="来源片段",
                          order=row.get("order", row.get("observed_at", 0)),
                          path=row.get("path"), parent_id=row.get("parent_id"),
                          text=row.get("text") or row.get("content") or json.dumps(row, ensure_ascii=False, indent=2)))
                edge(row.get("parent_id"), row["id"], "source_fragment")

    # Facts may have already been deduplicated by the pipeline. Preserve all IDs
    # that groups actually reference; do not reassign IDs for display.
    fact_map = {f["id"]: f for f in facts}
    for group in raw_groups:
        for fact in group["facts"]:
            fact_map.setdefault(fact["id"], fact)
    for fact in fact_map.values():
        orders = [nodes[s]["order"] for s in fact.get("sources", []) if s in nodes]
        node(dict(id=fact["id"], kind="fact", lane="general" if fact.get("qa_mode") == "general" else "code", label="事实", order=sum(orders) / len(orders) if orders else 0,
                  sources=fact.get("sources", []), qa_mode=fact.get("qa_mode"), text=fact["statement"], source_kind=fact.get("source_kind")))
        for source in fact.get("sources", []):
            edge(source, fact["id"], "fact_source")

    audit_by_id = {q["id"]: q for q in audit.get("questions", [])}
    candidate_records = audit.get("candidate_records")
    if candidate_records is None:
        # Legacy files cannot prove which pre-validation candidates were lost.
        known = {q["id"]: q for q in audit.get("candidates", [])
                 if isinstance(q, dict) and q.get("id")}
        for q in audit.get("questions", []) + [r.get("question") for r in audit.get("rejected", [])]:
            if isinstance(q, dict) and q.get("id"):
                known.setdefault(q["id"], q)
        legacy_rejected = [r for r in audit.get("rejected", [])
                           if r.get("reason") != "question_limit_exceeded"]
        legacy_selection = [{"candidate_id": r["question"]["id"],
                             "selection_status": "over_quota", "reason": "question_limit_exceeded"}
                            for r in audit.get("rejected", [])
                            if r.get("reason") == "question_limit_exceeded"
                            and isinstance(r.get("question"), dict) and r["question"].get("id")]
        candidate_records = build_audit(list(known.values()), audit.get("questions", []),
                                       legacy_rejected, legacy_selection, public)
        for row in candidate_records:
            row["record_note"] = "旧运行：校验前原始版本、修正链或未保存的候选可能缺失；不补造。"
    questions = []
    for published in public:
        raw = audit_by_id.get(published["id"], {})
        question = dict(published)
        question.update({key: raw.get(key) for key in ("evidence_group_id", "fact_ids", "use_case", "difficulty_reason", "review", "cutoff")})
        # Keep published wording while restoring per-point source links.
        for field in ("answer_points", "forbidden_points"):
            question[field] = [dict(point, sources=(raw.get(field, [])[i].get("sources", []) if i < len(raw.get(field, [])) else [])) for i, point in enumerate(published.get(field, []))]
        questions.append(question)

    groups = []
    for group in raw_groups:
        scope = group["scope"]
        direct = {s for f in group["facts"] for s in f.get("sources", [])}
        projection = set(direct)
        for key in ("dialogue", "events", "versions"):
            projection.update(r["id"] for r in scope.get(key, []))
        ids = [f["id"] for f in group["facts"]]
        groups.append(dict(id=group["id"], qa_mode=group["qa_mode"], fact_ids=ids,
                           source_ids=sorted(direct), projection_ids=sorted(projection),
                           metadata=scope.get("evidence_group", {}), allowed_types=group.get("allowed_types", []),
                           question_ids=[q["id"] for q in questions if q.get("evidence_group_id") == group["id"]]))

    unknown = sorted({p for e in edges.values() for p in (e["source"], e["target"]) if p not in nodes})
    for source in unknown:
        node(dict(id=source, kind="missing", lane="tool", label="缺失来源", order=0, text="运行产物中未找到对应原文；不补造内容。"))
    adaptive_file = run / "adaptive-subgraphs.json"
    adaptive = load(run, adaptive_file.name).get("meta", {}) if adaptive_file.exists() else {}
    output = clean(dict(meta=dict(run=run.name, title="Lambda Forge" if run.name == "validation-quality-final-20260909-v5" else "真实对话", records=len(records), visible_messages=sum(r["kind"] == "message" for r in records),
                       versions=len(graph["versions"]), facts=len(fact_map), groups=len(groups), questions=len(questions),
                       cutoff=manifest.get("cutoff"), source_kinds=dict(Counter(r.get("source_kind", "unknown") for r in records)),
                       expansion_mode="illustrative_graph_traversal", missing_sources=unknown),
                   nodes=list(nodes.values()), edges=list(edges.values()), groups=groups, questions=questions,
                   candidate_records=candidate_records,
                   progress=manifest.get("progress", {}),
                   targets={mode: manifest.get(mode + "_count") for mode in ("general", "code")},
                   snapshots=graph.get("snapshots", []), adaptive=adaptive,
                   rejections=dict(Counter(r.get("reason", "unknown") for r in audit.get("rejected", []))),
                   coverage=manifest.get("coverage", {})))
    output["meta"]["redactions"] = dict(changes)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("data.js"))
    args = parser.parse_args()
    data = build(args.run)
    # This generated local artifact deliberately has no fetch dependency, so it
    # also works from file://. Escape HTML terminators defensively.
    args.output.write_text("window.BENCHMARK_DATA = " + json.dumps(data, ensure_ascii=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029") + ";\n", encoding="utf-8")
    os.chmod(args.output, 0o600)
    print(json.dumps({"records": data["meta"]["records"], "nodes": len(data["nodes"]), "edges": len(data["edges"]), "groups": len(data["groups"]), "questions": len(data["questions"]), "redactions": data["meta"]["redactions"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
