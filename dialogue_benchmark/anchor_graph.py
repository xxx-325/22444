"""Small, deterministic evidence graphs for externally anchored QA.

This module is deliberately independent from the model-facing QA pipeline.  It
only joins records that are already named by an external anchor and follows
explicit source relationships.  It does not infer relationships from fuzzy
keywords and it does not assign a single type to a QA question.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from copy import deepcopy
from typing import Any, Iterable, Mapping, Sequence


_RECEIPT = re.compile(
    r"^(?:ok|done|completed|success(?:fully)?|all\s+tests?\s+passed|"
    r"tests?\s+passed|passed|no\s+changes?)(?:[.!\s]*)$", re.I)
_DIRECTORY = re.compile(
    r"^(?:ls(?:\s+-[\w-]+)?|find\s+[^\n]*|tree(?:\s+[^\n]*)?|"
    r"directory\s+listing:?)\s*(?:\n|$)", re.I)
_PASS_ONLY = re.compile(r"^(?:[✓✔]\s*)?(?:\d+\s+)?(?:passed|pass|ok)(?:\s+tests?)?[.!\s]*$", re.I)

_RELATION_FIELDS = {
    "fix_of": "fix_chain",
    "fixes": "fix_chain",
    "fixed": "fix_chain",
    "previous": "fix_chain",
    "previous_id": "fix_chain",
    "supersedes": "corrects",
    "corrects": "corrects",
    "corrected_from": "corrects",
    "replaces": "corrects",
}


def _as_ids(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [item for item in value if isinstance(item, str) and item]
    return []


def _record_id(record: Mapping[str, Any]) -> str | None:
    value = record.get("id")
    return value if isinstance(value, str) and value else None


def _call_id(record: Mapping[str, Any]) -> str | None:
    for key in ("call_id", "tool_call_id", "parent_call_id"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _paths(record: Mapping[str, Any]) -> set[str]:
    values: list[Any] = []
    for key in ("path", "file", "affected_path"):
        values.append(record.get(key))
    for key in ("paths", "affected_paths"):
        value = record.get(key)
        values.extend(value if isinstance(value, (list, tuple, set)) else [value])
    changes = record.get("changes")
    if isinstance(changes, Mapping):
        values.extend(changes.keys())
    return {value for value in values if isinstance(value, str) and value}


def _text(record: Mapping[str, Any]) -> str:
    value = record.get("text", record.get("content", ""))
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return "\n".join(str(part) for part in value if part).strip()
    return ""


def _node_source_ids(node: Mapping[str, Any]) -> set[str]:
    return set(_as_ids(node.get("record_ids"))) | set(_as_ids(node.get("source_ids")))


def _node_order(node: Mapping[str, Any]) -> int:
    value = node.get("order", node.get("first_order", 0))
    return value if isinstance(value, int) else 0


def _json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def noise_reason(record: Mapping[str, Any], *, duplicate: bool = False) -> str | None:
    """Return a reason for obvious non-evidence output, or ``None``.

    The check intentionally stays conservative.  A result with a substantive
    message such as ``validation passed: ...`` remains usable evidence.
    """
    text = _text(record)
    if duplicate:
        return "duplicate"
    if not text:
        return "empty"
    if _DIRECTORY.match(text):
        return "directory_listing"
    if _RECEIPT.match(text) or _PASS_ONLY.match(text):
        return "success_receipt"
    return None


def filter_anchor_noise(records: Iterable[Mapping[str, Any]]) -> tuple[list[dict], list[dict]]:
    """Split records into usable records and deterministic noise diagnostics."""
    usable: list[dict] = []
    noise: list[dict] = []
    seen: set[str] = set()
    for raw in records:
        record = deepcopy(dict(raw))
        normalized = re.sub(r"\s+", " ", _text(record)).strip().lower()
        reason = noise_reason(record, duplicate=bool(normalized and normalized in seen))
        if normalized:
            seen.add(normalized)
        if reason:
            noise.append({"id": _record_id(record), "reason": reason,
                          "record": record})
        else:
            usable.append(record)
    return usable, noise


def merge_action_records(records: Iterable[Mapping[str, Any]]) -> list[dict]:
    """Merge one tool call, result and patch into a single action node.

    A call/result pair is represented by one node, so the graph never creates
    a false relationship between the two halves of one tool invocation.
    Records without a call identity remain individual nodes.
    """
    rows = [deepcopy(dict(record)) for record in records]
    groups: dict[str, list[dict]] = defaultdict(list)
    standalone: list[dict] = []
    for record in rows:
        call_id = _call_id(record)
        if call_id:
            groups[call_id].append(record)
        else:
            standalone.append(record)

    merged: list[dict] = []
    for call_id, parts in groups.items():
        parts.sort(key=lambda item: (item.get("order", 0), item.get("id", "")))
        source_ids = [item["id"] for item in parts if isinstance(item.get("id"), str)]
        texts = []
        for item in parts:
            text = _text(item)
            if text and text not in texts:
                texts.append(text)
        paths = sorted(set().union(*(_paths(item) for item in parts)))
        node = {
            "id": "action:" + call_id,
            "kind": "action",
            "call_id": call_id,
            "order": min((item.get("order", 0) for item in parts), default=0),
            "record_ids": source_ids,
            "source_ids": source_ids,
            "paths": paths,
            "text": "\n".join(texts),
            "parts": parts,
        }
        node["first_order"] = node["order"]
        merged.append(node)

    for record in standalone:
        node = dict(record)
        node.setdefault("record_ids", [_record_id(node)] if _record_id(node) else [])
        node.setdefault("source_ids", list(node.get("record_ids", [])))
        node.setdefault("paths", sorted(_paths(node)))
        node.setdefault("text", _text(node))
        node["first_order"] = node.get("order", 0)
        merged.append(node)
    merged.sort(key=lambda item: (_node_order(item), item.get("id", "")))
    return merged


def _node_lookup(nodes: Sequence[Mapping[str, Any]]) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for node in nodes:
        copied = dict(node)
        node_id = copied.get("id")
        if isinstance(node_id, str):
            lookup[node_id] = copied
        for source_id in _node_source_ids(copied):
            lookup[source_id] = copied
    return lookup


def _anchor_ids(anchor: Mapping[str, Any], field: str) -> list[str]:
    return list(dict.fromkeys(_as_ids(anchor.get(field))))


def _explicit_edges(nodes: Sequence[Mapping[str, Any]]) -> list[dict]:
    lookup = _node_lookup(nodes)
    edges: set[tuple[str, str, str]] = set()
    for node in nodes:
        source = node.get("id")
        if not isinstance(source, str):
            continue
        paths = _paths(node)
        for other in nodes:
            target = other.get("id")
            if not isinstance(target, str) or source >= target:
                continue
            if paths.intersection(_paths(other)):
                edges.add((source, target, "same_path"))
        for field, relation in _RELATION_FIELDS.items():
            for ref in _as_ids(node.get(field)):
                target_node = lookup.get(ref)
                if not target_node or target_node.get("id") == source:
                    continue
                target = target_node.get("id")
                if isinstance(target, str):
                    edges.add((source, target, relation))
    return [{"from": left, "to": right, "relation": relation}
            for left, right, relation in sorted(edges)]


def _node_priority(node: Mapping[str, Any], seed_node_ids: set[str]) -> tuple[int, int, str]:
    node_id = str(node.get("id", ""))
    return (0 if node_id in seed_node_ids else 1, _node_order(node), node_id)


def budget_nodes(nodes: Sequence[Mapping[str, Any]], max_chars: int) -> dict:
    """Keep a stable prefix of nodes within a character budget."""
    if not isinstance(max_chars, int) or max_chars <= 0:
        raise ValueError("max_chars must be positive")
    kept: list[dict] = []
    size = 0
    for node in nodes:
        candidate = kept + [node]
        candidate_size = _json_size(candidate)
        if candidate_size > max_chars:
            continue
        kept.append(deepcopy(dict(node)))
        size = candidate_size
    return {"nodes": kept, "chars": size, "over_budget": len(kept) != len(nodes),
            "max_chars": max_chars}


def build_anchor_subgraph(anchor: Mapping[str, Any], records: Iterable[Mapping[str, Any]], *,
                          max_chars: int = 48000) -> dict:
    """Build a bounded graph from one externally supplied anchor.

    Source, usage and context references are explicit seeds.  Same-path and
    correction edges are followed only over already supplied records.
    """
    raw_records = [dict(record) for record in records]
    usable, noise = filter_anchor_noise(raw_records)
    nodes = merge_action_records(usable)
    lookup = _node_lookup(nodes)
    source_ids = _anchor_ids(anchor, "source_ids")
    used_by = _anchor_ids(anchor, "used_by")
    context_ids = _anchor_ids(anchor, "context_ids")
    requested = list(dict.fromkeys(source_ids + used_by + context_ids))
    selected: dict[str, dict] = {}
    unresolved: list[str] = []
    for ref in requested:
        node = lookup.get(ref)
        if node is None:
            unresolved.append(ref)
            continue
        selected[str(node["id"])] = node
    # Explicit same-path expansion is useful for action-level evidence and is
    # intentionally narrower than keyword or embedding expansion.
    selected_paths = set().union(*(_paths(node) for node in selected.values())) if selected else set()
    for node in nodes:
        if selected_paths.intersection(_paths(node)):
            selected.setdefault(str(node["id"]), node)
    selected_nodes = sorted(selected.values(),
                            key=lambda item: _node_priority(item,
                                {str(lookup[ref]["id"]) for ref in requested if ref in lookup}))
    bounded = budget_nodes(selected_nodes, max_chars)
    kept_ids = {str(node.get("id")) for node in bounded["nodes"]}
    edges = [edge for edge in _explicit_edges(bounded["nodes"])
             if edge["from"] in kept_ids and edge["to"] in kept_ids]
    seed_node_ids = {str(lookup[ref]["id"]) for ref in requested if ref in lookup}
    return {
        "anchor_id": anchor.get("id"),
        "memory_kind": anchor.get("memory_kind"),
        "focus": anchor.get("focus"),
        "source_ids": source_ids,
        "used_by": used_by,
        "context_ids": context_ids,
        "nodes": bounded["nodes"],
        "edges": edges,
        "noise": noise,
        "unresolved_ids": unresolved,
        "seed_node_ids": sorted(seed_node_ids),
        "budget": {"chars": bounded["chars"], "max_chars": max_chars,
                   "over_budget": bounded["over_budget"]},
        "status": "rejected" if not source_ids or unresolved else "ready",
    }


def combine_anchor_group(anchors: Iterable[Mapping[str, Any]], records: Iterable[Mapping[str, Any]], *,
                         required_anchor_ids: Iterable[str] | None = None,
                         combination_reason: str | None = None,
                         max_chars: int = 48000) -> dict:
    """Combine independently sourced anchor subgraphs without type inference."""
    anchor_rows = [dict(anchor) for anchor in anchors]
    by_id = {row.get("id"): row for row in anchor_rows if isinstance(row.get("id"), str)}
    anchor_ids = list(by_id)
    required = list(dict.fromkeys(required_anchor_ids if required_anchor_ids is not None else anchor_ids))
    invalid_required = [item for item in required if item not in by_id]
    supporting = [item for item in anchor_ids if item not in required]
    subgraphs = [build_anchor_subgraph(row, records, max_chars=max_chars) for row in anchor_rows]
    nodes_by_id: dict[str, dict] = {}
    edges: set[tuple[str, str, str]] = set()
    for graph in subgraphs:
        for node in graph["nodes"]:
            nodes_by_id[str(node["id"])] = node
        for edge in graph["edges"]:
            edges.add((edge["from"], edge["to"], edge["relation"]))
    combined = budget_nodes(sorted(nodes_by_id.values(), key=lambda item: (_node_order(item), item.get("id", ""))), max_chars)
    kept = {str(node.get("id")) for node in combined["nodes"]}
    required_kinds = sorted({by_id[item].get("memory_kind") for item in required
                             if item in by_id and isinstance(by_id[item].get("memory_kind"), str)})
    return {
        "anchor_ids": anchor_ids,
        "required_anchor_ids": required,
        "supporting_anchor_ids": supporting,
        "memory_kinds": required_kinds,
        "combination_reason": combination_reason,
        "subgraphs": subgraphs,
        "nodes": combined["nodes"],
        "edges": [{"from": left, "to": right, "relation": relation}
                  for left, right, relation in sorted(edges)
                  if left in kept and right in kept],
        "budget": {"chars": combined["chars"], "max_chars": max_chars,
                   "over_budget": combined["over_budget"]},
        "invalid_required_anchor_ids": invalid_required,
        "status": "ready" if required and not invalid_required and
                  all(graph["status"] == "ready" for graph in subgraphs
                      if graph["anchor_id"] in required) else "needs_review",
    }


def anchor_difficulty(required_facts: int | Sequence[Any], *, info_nodes: int | Sequence[Any] | None = None,
                      revision: bool | int = False, cross_stage: bool | int = False,
                      complete: bool = True) -> str:
    """Return evidence-complexity difficulty, independent of M1-M6 kinds."""
    if isinstance(required_facts, int):
        count = required_facts
    else:
        values = []
        for fact in required_facts or []:
            if isinstance(fact, Mapping):
                value = fact.get("id", fact.get("statement"))
            else:
                value = fact
            if value not in values:
                values.append(value)
        count = len([value for value in values if value is not None])
    if not complete or count <= 0:
        return "unknown"
    if revision or cross_stage or count >= 3:
        return "hard"
    if count == 2:
        return "medium"
    return "easy"


def difficulty_basis(required_facts: int | Sequence[Any], *, info_nodes: int | Sequence[Any] | None = None,
                     revision: bool | int = False, cross_stage: bool | int = False,
                     complete: bool = True) -> dict:
    """Return the difficulty and the static inputs used to compute it."""
    if isinstance(required_facts, int):
        count = required_facts
    else:
        values = []
        for fact in required_facts or []:
            value = fact.get("id", fact.get("statement")) if isinstance(fact, Mapping) else fact
            if value not in values:
                values.append(value)
        count = len([value for value in values if value is not None])
    node_count = info_nodes if isinstance(info_nodes, int) else len(info_nodes or [])
    return {"difficulty": anchor_difficulty(count, revision=revision,
                                             cross_stage=cross_stage, complete=complete),
            "required_facts": count, "info_nodes": node_count,
            "revision": bool(revision), "cross_stage": bool(cross_stage),
            "complete": bool(complete)}

