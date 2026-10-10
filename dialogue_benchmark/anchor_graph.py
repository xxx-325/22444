"""Pure deterministic evidence graphs for anchored QA.

The graph is a small control-side projection. It never calls a model and
never decides whether a historical fact is true; it only joins records that
were already supplied by the dialogue and external-event sidecars.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from copy import deepcopy
from typing import Any, Iterable, Mapping


_RELATIONS = {
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
_CORRECTION_WORDS = re.compile(
    r"改为|改成|不再|例外|除外|仅限|仅在|只适用|纠正|更正|撤销|确认|默认|而是|"
    r"\binstead\b|\bexcept\b|\bexception\b|\bonly\b|\bno longer\b|\breplace\b",
    re.I,
)
_RECEIPT = re.compile(
    r"^(?:ok|done|completed|success(?:fully)?|tests?\s+passed|"
    r"passed|[✓✔]\s*\d*\s*passed|file\s+(?:updated|created)\s+successfully"
    r"|file\s+updated\s+successfully\s+at:.+)[.!\s]*$",
    re.I,
)
_LISTING = re.compile(
    r"^(?:ls(?:\s+-[\w-]+)?|find\s+[^\n]*|tree(?:\s+[^\n]*)?)\s*(?:\n|$)",
    re.I,
)
_FAILURE = re.compile(r"traceback|failed|failure|error|exception|非零|失败|错误", re.I)
_STOPWORDS = {
    "customer", "ticket", "test", "tests", "file", "files", "code",
    "update", "updated", "run", "running", "实现", "测试", "问题", "功能",
    "需求", "代码", "文件", "用户", "客户", "工单",
    "export", "policy", "rule",
}
_ALLOWED_EDGES = {
    "M1": {"responds_to", "same_value", "corrects", "fix_chain"},
    "M2": {"responds_to", "same_value", "corrects", "fix_chain"},
    "M3": {"same_path", "corrects", "fix_chain"},
    "M4": {"fix_chain", "same_path", "responds_to"},
    "M5": {"fix_chain", "same_value", "responds_to"},
    "M6": {"responds_to", "same_value", "corrects", "fix_chain"},
}


def _ids(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        return list(dict.fromkeys(
            item for item in value if isinstance(item, str) and item
        ))
    return []


def _record_id(record: Mapping[str, Any]) -> str | None:
    value = record.get("id")
    return value if isinstance(value, str) and value else None


def _text(record: Mapping[str, Any]) -> str:
    value = record.get("text", record.get("content", ""))
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return "\n".join(str(item) for item in value if item).strip()
    return ""


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


def keys(text: str) -> set[str]:
    """Return stable searchable keys without common prose words."""
    if not isinstance(text, str):
        return set()
    latin = set(re.findall(r"[a-z0-9_.-]{2,}", text.casefold()))
    chinese = set(re.findall(r"[\u4e00-\u9fff]{2}", text))
    return {value for value in latin | chinese if value not in _STOPWORDS}


def _literal_values(text: str) -> set[str]:
    if not isinstance(text, str):
        return set()
    quoted = set(re.findall(r"""[`'"]([^`'"]+)[`'"]""", text))
    values = set(re.findall(r"\b\d+(?:\.\d+)?\b", text))
    return {value.casefold() for value in quoted | values if value.strip()}


def _noise(record: Mapping[str, Any]) -> str | None:
    text = _text(record)
    if not text:
        return "empty"
    if _LISTING.match(text):
        return "directory_listing"
    if _RECEIPT.match(text):
        return "success_receipt"
    return None


def _call_id(record: Mapping[str, Any]) -> str | None:
    for key in ("call_id", "tool_call_id", "parent_call_id"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _command(record: Mapping[str, Any]) -> str:
    for key in ("command", "tool_name", "name"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    action = record.get("action")
    if isinstance(action, Mapping):
        for key in ("command", "tool_name", "name"):
            value = action.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def _failure_signature(parts: Iterable[Mapping[str, Any]], text: str) -> str | None:
    """Return a compact failure signature from structured or textual output."""
    for part in parts:
        if part.get("is_error") is True or part.get("success") is False:
            message = _text(part).splitlines()
            first = next((line.strip() for line in message if line.strip()), "tool error")
            return first[:240]
        exit_code = part.get("exit_code", part.get("returncode"))
        if isinstance(exit_code, int) and exit_code != 0:
            message = _text(part).splitlines()
            first = next((line.strip() for line in message if line.strip()), "")
            return ("exit %s: %s" % (exit_code, first)).strip()[:240]
    for line in text.splitlines():
        if _FAILURE.search(line):
            stripped = line.strip()
            match = re.search(r"\b([A-Za-z_][A-Za-z0-9]*(?:Error|Exception))\b", stripped)
            prefix = match.group(1) + ": " if match else ""
            return (prefix + stripped)[:240]
    return None


def _node_order(node: Mapping[str, Any]) -> int:
    value = node.get("order", 0)
    return value if isinstance(value, int) else 0


def _node_keys(node: Mapping[str, Any]) -> set[str]:
    value = node.get("keys")
    return set(value) if isinstance(value, list) else keys(str(node.get("text", "")))


def _node_literals(node: Mapping[str, Any]) -> set[str]:
    return _literal_values(str(node.get("text", "")))


def _node_source_ids(node: Mapping[str, Any]) -> set[str]:
    return set(_ids(node.get("record_ids"))) | set(_ids(node.get("source_ids")))


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[。！？.!?])\s+|\n+", text)
            if part.strip()]


def _node_lookup(nodes: Iterable[Mapping[str, Any]]) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for node in nodes:
        copied = dict(node)
        node_id = copied.get("id")
        if isinstance(node_id, str):
            lookup[node_id] = copied
        for source_id in _node_source_ids(copied):
            lookup[source_id] = copied
    return lookup


def _action_node(call_id: str, parts: list[dict]) -> dict:
    parts.sort(key=lambda row: (row.get("order", 0), row.get("id", "")))
    evidence_parts = [row for row in parts if _noise(row) is None]
    text = "\n".join(dict.fromkeys(
        _text(row) for row in evidence_parts if _text(row)
    ))
    paths = sorted(set().union(*(_paths(row) for row in parts)))
    failure = _failure_signature(parts, text)
    has_patch = any(
        row.get("kind") == "patch" or isinstance(row.get("changes"), Mapping)
        for row in parts
    )
    has_result = any(row.get("kind") == "result" for row in parts)
    outcome = "failed" if failure else ("edited" if has_patch else (
        "ok" if has_result else "info"
    ))
    tool = next((_command(row) for row in parts if _command(row)), "")
    command = next((
        (row.get("action") or {}).get("command")
        for row in parts if isinstance(row.get("action"), Mapping)
        and isinstance((row.get("action") or {}).get("command"), str)
    ), "")
    if not command:
        command = next((_command(row) for row in parts
                        if row.get("kind") == "call"), "")
    return {
        "id": "action:" + call_id,
        "kind": "action",
        "call_id": call_id,
        "record_ids": [_record_id(row) for row in parts if _record_id(row)],
        "order": min((row.get("order", 0) for row in parts), default=0),
        "task_index": next(
            (row.get("task_index") for row in parts
             if isinstance(row.get("task_index"), int)), None
        ),
        "paths": paths,
        "tool": tool,
        "command": command,
        "text": text,
        "chars": len(text),
        "keys": sorted(keys(text)),
        "outcome": outcome,
        "error_signature": failure,
        "noise": next((_noise(row) for row in parts if _noise(row)), None),
        "parts": parts,
    }


def _standalone_node(record: dict) -> dict:
    text = _text(record)
    return {
        **record,
        "id": _record_id(record),
        "kind": record.get("kind", "message"),
        "record_ids": [_record_id(record)] if _record_id(record) else [],
        "order": record.get("order", 0),
        "task_index": record.get("task_index"),
        "paths": sorted(_paths(record)),
        "text": text,
        "chars": len(text),
        "keys": sorted(keys(text)),
        "noise": None,
    }


def _explicit_edges(nodes: list[dict], lookup: dict[str, dict]) -> set[tuple[str, str, str]]:
    edges: set[tuple[str, str, str]] = set()
    for node in nodes:
        source = node.get("id")
        if not isinstance(source, str):
            continue
        for record in [node, *node.get("parts", [])]:
            for field, relation in _RELATIONS.items():
                for reference in _ids(record.get(field)):
                    target = lookup.get(reference, {}).get("id")
                    if isinstance(target, str) and target != source:
                        edges.add((source, target, relation))
    for index, left in enumerate(nodes):
        for right in nodes[index + 1:]:
            if (left.get("kind") == "action" and right.get("kind") == "action"
                    and set(left.get("paths", [])) & set(right.get("paths", []))):
                edges.add((left["id"], right["id"], "same_path"))
    return edges


def build_action_graph(records: Iterable[Mapping[str, Any]]) -> dict:
    """Build a compressed action graph from public dialogue records."""
    rows = [deepcopy(dict(record)) for record in records]
    groups: dict[str, list[dict]] = defaultdict(list)
    standalone: list[dict] = []
    noise: list[dict] = []
    seen_text: set[str] = set()
    for record in rows:
        normalized = re.sub(r"\s+", " ", _text(record)).casefold()
        reason = _noise(record)
        if reason is None and normalized and normalized in seen_text:
            reason = "duplicate"
        if normalized:
            seen_text.add(normalized)
        if reason:
            noise.append({"id": _record_id(record), "reason": reason})
            continue
        call_id = _call_id(record)
        (groups[call_id] if call_id else standalone).append(record)

    nodes = [_action_node(call_id, parts) for call_id, parts in groups.items()]
    nodes.extend(_standalone_node(record) for record in standalone)
    nodes = [node for node in nodes if isinstance(node.get("id"), str)]
    nodes.sort(key=lambda node: (_node_order(node), node["id"]))
    lookup = _node_lookup(nodes)
    edges = _explicit_edges(nodes, lookup)

    users = [
        node for node in nodes
        if node.get("kind") == "message" and node.get("role") == "user"
    ]
    for node in nodes:
        if node.get("kind") != "action":
            continue
        prior = [user for user in users if _node_order(user) < _node_order(node)]
        if prior:
            edges.add((node["id"], max(prior, key=_node_order)["id"], "responds_to"))

    messages = [
        node for node in nodes
        if node.get("kind") == "message" and node.get("role") == "user"
    ]
    for earlier_index, earlier in enumerate(messages):
        for later in messages[earlier_index + 1:]:
            if _CORRECTION_WORDS.search(str(later.get("text", ""))):
                shared = ((_node_keys(earlier) & _node_keys(later))
                          | (_node_literals(earlier) & _node_literals(later)))
                if shared:
                    edges.add((later["id"], earlier["id"], "corrects"))

    frequency: defaultdict[str, int] = defaultdict(int)
    for node in nodes:
        for value in _node_keys(node) | _node_literals(node):
            frequency[value] += 1
    for index, left in enumerate(nodes):
        for right in nodes[index + 1:]:
            shared = (_node_literals(left) & _node_literals(right)) | {
                value for value in _node_keys(left) & _node_keys(right)
                if frequency[value] <= max(2, len(nodes) // 10)
            }
            if shared:
                edges.add((left["id"], right["id"], "same_value"))

    actions = [node for node in nodes if node.get("kind") == "action"]
    for failed in actions:
        if failed.get("outcome") != "failed":
            continue
        failure_paths = set(failed.get("paths", []))
        edits = [
            node for node in actions
            if node.get("outcome") == "edited"
            and _node_order(node) > _node_order(failed)
            and (failure_paths & set(node.get("paths", []))
                 or _node_keys(node) & _node_keys(failed))
        ]
        for edit in edits:
            edges.add((failed["id"], edit["id"], "fix_chain"))
            for passed in actions:
                if (passed.get("outcome") == "ok"
                        and _node_order(passed) > _node_order(edit)
                        and passed.get("command") == failed.get("command")
                        and (failure_paths & set(passed.get("paths", []))
                             or not failure_paths)):
                    edges.add((edit["id"], passed["id"], "fix_chain"))

    return {
        "nodes": nodes,
        "edges": [
            {"from": left, "to": right, "relation": relation}
            for left, right, relation in sorted(edges)
        ],
        "noise": noise,
        "node_by_id": lookup,
    }


def anchor_seeds(event: Mapping[str, Any], graph: Mapping[str, Any]) -> dict:
    """Map an external event's declared records to graph seed nodes."""
    lookup = graph.get("node_by_id") or _node_lookup(graph.get("nodes", []))
    seeds: list[dict] = []
    unresolved: list[str] = []
    for field in ("source_ids", "used_by", "context_ids"):
        for reference in _ids(event.get(field)):
            node = lookup.get(reference)
            if node is None:
                unresolved.append(reference)
            elif node["id"] not in {item["id"] for item in seeds}:
                seeds.append(node)
    focus = event.get("focus", "")
    focus = focus if isinstance(focus, str) else ""
    focus_keys = keys(focus)
    source_refs = set(_ids(event.get("source_ids")))
    literal_values: set[str] = set(_literal_values(focus))
    source_orders: list[int] = []
    for node in seeds:
        if source_refs & _node_source_ids(node):
            if isinstance(node.get("order"), int):
                source_orders.append(node["order"])
            candidates = [node]
            candidates.extend(
                part for part in node.get("parts", [])
                if isinstance(part, Mapping)
            )
            for candidate in candidates:
                if candidate.get("kind") not in {"message", "result", "observation"}:
                    continue
                for sentence in _sentences(_text(candidate)):
                    sentence_keys = keys(sentence)
                    if not focus_keys or len(sentence_keys & focus_keys) >= 1:
                        literal_values.update(_literal_values(sentence))
    cutoff = event.get("cutoff")
    return {
        "anchor_id": event.get("id"),
        "seeds": seeds,
        "seed_ids": [node["id"] for node in seeds],
        "keywords": sorted(focus_keys),
        "literal_values": sorted(literal_values),
        "source_orders": sorted(source_orders),
        "cutoff": cutoff if isinstance(cutoff, int) else None,
        "unresolved_ids": list(dict.fromkeys(unresolved)),
    }


def _edge_neighbours(graph: Mapping[str, Any], admitted: set[str]) -> list[dict]:
    neighbours = []
    for edge in graph.get("edges", []):
        if edge.get("from") in admitted:
            neighbours.append(edge)
        elif edge.get("to") in admitted:
            neighbours.append({
                **edge,
                "from": edge.get("to"),
                "to": edge.get("from"),
            })
    return neighbours


def expand_anchor(
    anchor_or_group: Mapping[str, Any],
    graph: Mapping[str, Any],
    budget_chars: int,
) -> dict:
    """Expand seeds only with useful, non-noise graph evidence."""
    if not isinstance(budget_chars, int) or budget_chars <= 0:
        raise ValueError("budget_chars must be positive")
    seed = (
        anchor_or_group
        if "seeds" in anchor_or_group
        else anchor_seeds(anchor_or_group, graph)
    )
    lookup = graph.get("node_by_id") or _node_lookup(graph.get("nodes", []))
    nodes = list(seed.get("seeds", []))
    admitted = {node["id"] for node in nodes}
    anchor_keys = set(seed.get("keywords", []))
    known = set(anchor_keys)
    literals = set(seed.get("literal_values", []))
    memory_kind = anchor_or_group.get("memory_kind")
    allowed_edges = _ALLOWED_EDGES.get(memory_kind, set().union(*_ALLOWED_EDGES.values()))
    min_overlap = max(1, min(2, len(anchor_keys))) if anchor_keys else 1
    source_orders = [value for value in seed.get("source_orders", [])
                     if isinstance(value, int)]
    cutoff = seed.get("cutoff")
    seed_chars = len(json.dumps(nodes, ensure_ascii=False, separators=(",", ":")))
    seen_results = {
        (node.get("outcome"), node.get("error_signature"), node.get("command"))
        for node in nodes if node.get("kind") == "action"
    }
    rejected: list[dict] = []
    audit: list[dict] = [
        {"node_id": node["id"], "reason": "seed"}
        for node in nodes
    ]
    if seed_chars > budget_chars:
        return {
            "anchor_id": seed.get("anchor_id"),
            "nodes": nodes,
            "edges": [
                edge for edge in graph.get("edges", [])
                if edge.get("from") in admitted and edge.get("to") in admitted
            ],
            "admitted": audit,
            "rejected": [{"reason": "seed_over_budget",
                          "node_ids": [node["id"] for node in nodes]}],
            "budget": {"chars": seed_chars, "max_chars": budget_chars},
            "unresolved_ids": list(seed.get("unresolved_ids", [])),
            "status": "unrealizable:seed_over_budget",
        }
    visited: set[str] = set(admitted)
    while True:
        candidates = []
        for edge in _edge_neighbours(graph, admitted):
            node = lookup.get(edge.get("to"))
            if node is None or node["id"] in visited:
                continue
            candidate_keys = _node_keys(node)
            candidate_literals = _node_literals(node)
            overlap = len(candidate_keys & anchor_keys)
            literal_match = bool(candidate_literals & literals)
            priority = (
                0 if edge["relation"] in {"fix_chain", "corrects"} else 1,
                0 if literal_match else 1,
                -overlap,
                _node_order(node),
                node["id"],
            )
            candidates.append((priority, edge, node))
        if not candidates:
            break
        candidates.sort(key=lambda value: value[0])
        _, edge, candidate = candidates[0]
        visited.add(candidate["id"])
        candidate_keys = _node_keys(candidate)
        candidate_literals = _node_literals(candidate)
        overlap = len(candidate_keys & anchor_keys)
        literal_match = bool(candidate_literals & literals)
        matched = (candidate_keys & known) | (candidate_literals & literals)
        if edge["relation"] not in allowed_edges:
            rejected.append({"node_id": candidate["id"], "reason": "edge_not_allowed"})
            continue
        if cutoff is not None and isinstance(candidate.get("order"), int):
            if candidate["order"] > cutoff:
                rejected.append({"node_id": candidate["id"], "reason": "after_cutoff"})
                continue
        if (source_orders and isinstance(candidate.get("order"), int)
                and candidate["order"] < min(source_orders)
                and edge["relation"] != "fix_chain"):
            rejected.append({"node_id": candidate["id"], "reason": "order_violation"})
            continue
        if candidate.get("kind") != "action" and candidate.get("noise"):
            rejected.append({"node_id": candidate["id"], "reason": "noise"})
            continue
        if (edge["relation"] not in {"fix_chain", "corrects"}
                and not literal_match and overlap < min_overlap):
            rejected.append({"node_id": candidate["id"], "reason": "irrelevant"})
            continue
        result_key = (candidate.get("outcome"), candidate.get("error_signature"),
                      candidate.get("command"))
        if result_key in seen_results and candidate.get("kind") == "action":
            rejected.append({"node_id": candidate["id"], "reason": "not_novel"})
            continue
        if not (candidate_keys - known or candidate_literals - literals
                or edge["relation"] in {"fix_chain", "corrects"}
                or candidate.get("outcome") in {"failed", "ok"}):
            rejected.append({"node_id": candidate["id"], "reason": "not_novel"})
            continue
        proposed = nodes + [candidate]
        chars = len(json.dumps(proposed, ensure_ascii=False, separators=(",", ":")))
        if chars > budget_chars:
            rejected.append({"node_id": candidate["id"], "reason": "budget"})
            continue
        new_keys = (candidate_keys - known) | (candidate_literals - literals)
        nodes.append(candidate)
        admitted.add(candidate["id"])
        known.update(candidate_keys)
        literals.update(candidate_literals)
        if candidate.get("kind") == "action":
            seen_results.add(result_key)
        audit.append({
            "node_id": candidate["id"],
            "edge": edge["relation"],
            "from": edge["from"],
            "matched_keys": sorted(matched),
            "new_keys": sorted(new_keys),
            "reason": "useful",
        })
    return {
        "anchor_id": seed.get("anchor_id"),
        "nodes": nodes,
        "edges": [
            edge for edge in graph.get("edges", [])
            if edge.get("from") in admitted and edge.get("to") in admitted
        ],
        "admitted": audit,
        "rejected": rejected,
        "budget": {
            "chars": len(json.dumps(nodes, ensure_ascii=False, separators=(",", ":"))),
            "max_chars": budget_chars,
        },
        "unresolved_ids": list(seed.get("unresolved_ids", [])),
        "status": "ready" if not seed.get("unresolved_ids") else "needs_review",
    }


def combine_anchor_group(
    anchors: Iterable[Mapping[str, Any]],
    records: Iterable[Mapping[str, Any]],
    *,
    required_anchor_ids: Iterable[str] | None = None,
    combination_reason: str | None = None,
    max_chars: int = 48000,
) -> dict:
    """Combine independent anchor expansions without inferring a QA type."""
    rows = [dict(anchor) for anchor in anchors]
    ids = [row["id"] for row in rows if isinstance(row.get("id"), str)]
    required = list(dict.fromkeys(
        _ids(required_anchor_ids) if required_anchor_ids is not None else []
    ))
    invalid = [value for value in required if value not in ids]
    graph = build_action_graph(records)
    expanded = [expand_anchor(row, graph, max_chars) for row in rows]
    nodes_by_id = {
        node["id"]: node
        for item in expanded
        for node in item.get("nodes", [])
    }
    nodes = list(nodes_by_id.values())
    size = len(json.dumps(nodes, ensure_ascii=False, separators=(",", ":")))
    status = (
        "ready"
        if required and not invalid and size <= max_chars
        and all(not item.get("unresolved_ids") for item in expanded
                if item.get("anchor_id") in required)
        and all(item.get("status") == "ready" for item in expanded
                if item.get("anchor_id") in required)
        else "needs_review"
    )
    return {
        "anchor_ids": ids,
        "required_anchor_ids": required,
        "supporting_anchor_ids": [value for value in ids if value not in required],
        "memory_kinds": sorted({
            row.get("memory_kind")
            for row in rows
            if row.get("id") in required and isinstance(row.get("memory_kind"), str)
        }),
        "combination_reason": combination_reason,
        "external_lineage": {
            str(row["id"]): {
                "source_ids": _ids(row.get("source_ids")),
                "used_by": _ids(row.get("used_by")),
                "context_ids": _ids(row.get("context_ids")),
            }
            for row in rows if isinstance(row.get("id"), str)
        },
        "nodes": nodes,
        "subgraphs": expanded,
        "budget": {"chars": size, "max_chars": max_chars},
        "invalid_required_anchor_ids": invalid,
        "status": status,
    }


def _fact_count(required_facts: int | Iterable[Any]) -> int:
    if isinstance(required_facts, int):
        return max(0, required_facts)
    values = {
        item.get("id", item.get("statement"))
        if isinstance(item, Mapping) else item
        for item in required_facts or []
    }
    return len({value for value in values if value is not None})


def anchor_difficulty(
    required_facts: int | Iterable[Any],
    *,
    info_nodes: int | Iterable[Any] | None = None,
    revision: bool = False,
    cross_stage: bool = False,
    complete: bool = True,
) -> str:
    """Return evidence-complexity difficulty, independent of memory kind."""
    count = _fact_count(required_facts)
    if not complete or count <= 0:
        return "unknown"
    if revision or count >= 3:
        return "hard"
    if count == 2:
        return "medium"
    return "easy"


def difficulty_basis(
    required_facts: int | Iterable[Any],
    *,
    info_nodes: int | Iterable[Any] | None = None,
    revision: bool = False,
    cross_stage: bool = False,
    complete: bool = True,
    context_nodes: int | Iterable[Any] | None = None,
) -> dict:
    """Return the inputs used for the deterministic difficulty label."""
    count = _fact_count(required_facts)
    node_count = info_nodes if isinstance(info_nodes, int) else len(info_nodes or [])
    context_count = (
        context_nodes if isinstance(context_nodes, int)
        else len(context_nodes or [])
    )
    return {
        "difficulty": anchor_difficulty(
            count, revision=revision, cross_stage=cross_stage, complete=complete
        ),
        "required_facts": count,
        "info_nodes": node_count,
        "revision": bool(revision),
        "cross_stage": bool(cross_stage),
        "context_nodes": context_count,
        "complete": bool(complete),
    }
