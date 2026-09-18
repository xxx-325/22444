"""Event-sourced partial file states and time-bounded graph queries."""

import copy
import hashlib
import posixpath

from .normalize import normalize_path, is_truncated, source_view
from .patches import apply_diff
from .symbols import inspect_code


def build_graph(records):
    versions, events, snapshots, diagnostics = [], [], [], []
    state, calls = {}, {}

    def install(path, content, event, status="known", previous=None):
        version = {"id": "v%d" % (len(versions) + 1), "path": path,
                   "observed_at": event["order"], "source": event["id"],
                   "status": status, "previous": previous,
                   "content": content, "sha256": hashlib.sha256(content.encode()).hexdigest()
                   if content is not None else None,
                   "code": inspect_code(path, content) if content is not None else None}
        versions.append(version)
        state[path] = version["id"]
        return version

    for record in records:
        event = {"id": record["id"], "order": record["order"], "kind": record["kind"],
                 "source_line": record["source_line"], "affected_paths": []}
        kind = record["kind"]
        if kind == "call":
            call_id = record.get("call_id")
            if call_id:
                if call_id in calls:
                    calls[call_id] = None
                    diagnostics.append({"source": record["id"], "reason": "duplicate_call_id"})
                else:
                    calls[call_id] = record["id"]
        elif kind == "result":
            event["call_source"] = calls.get(record.get("call_id"))
            if not event["call_source"]:
                diagnostics.append({"source": record["id"], "reason": "unmatched_result"})
            if is_truncated(record.get("text", "")):
                diagnostics.append({"source": record["id"], "reason": "truncated_output"})
        elif kind == "observation":
            path = normalize_path(record["path"], record.get("workspace"))
            event["affected_paths"] = [path]
            previous = state.get(path)
            complete = record.get("complete") is True
            install(path, record["content"] if complete else None, event,
                    "known" if complete else "unknown", previous)
            if not complete:
                diagnostics.append({"source": record["id"], "reason": "partial_observation"})
        elif kind == "patch":
            event["success"] = record.get("success") is True
            changes = record.get("changes", {})
            if not isinstance(changes, dict):
                raise ValueError("Patch changes must be an object")
            for raw_path, change in changes.items():
                path = normalize_path(raw_path, record.get("workspace"))
                event["affected_paths"].append(path)
                if not event["success"]:
                    continue
                previous = state.get(path)
                old = next((v for v in reversed(versions) if v["id"] == previous), None)
                destination = normalize_path(change.get("move_path") or raw_path, record.get("workspace"))
                if destination != path:
                    event["affected_paths"].append(destination)
                try:
                    operation = change.get("type")
                    if operation == "add":
                        if old and old["status"] != "deleted":
                            raise ValueError("Add conflicts with observed file")
                        content = change["content"]
                    elif operation == "delete":
                        install(path, None, event, "deleted", previous)
                        continue
                    elif operation == "update":
                        if old is None or old["status"] != "known":
                            raise ValueError("Missing complete base")
                        content = apply_diff(old["content"], change["unified_diff"])
                    else:
                        raise ValueError("Unsupported patch operation")
                    if destination != path and destination in state:
                        target = next(v for v in versions if v["id"] == state[destination])
                        if target["status"] != "deleted":
                            raise ValueError("Move destination already observed")
                    install(destination, content, event, previous=previous)
                    if destination != path:
                        install(path, None, event, "deleted", previous)
                except (ValueError, KeyError, TypeError) as error:
                    # A successful but unreplayable edit invalidates the old known state.
                    install(path, None, event, "unknown", previous)
                    if destination != path:
                        install(destination, None, event, "unknown", state.get(destination))
                    diagnostics.append({"source": record["id"], "path": path,
                                        "reason": str(error)})
        events.append(event)
        snapshots.append({"at": record["order"], "event": event["id"], "files": dict(state)})
    return {"versions": versions, "events": events, "snapshots": snapshots,
            "diagnostics": diagnostics}


def graph_at(graph, cutoff):
    snapshot = next((s for s in reversed(graph["snapshots"]) if s["at"] <= cutoff), {"files": {}})
    by_id = {v["id"]: v for v in graph["versions"]}
    files = {p: by_id[v] for p, v in snapshot["files"].items()}
    nodes, edges, unresolved = [], [], []
    for path, version in files.items():
        if version["status"] != "known":
            continue
        nodes.append({"id": path, "kind": "file", "version": version["id"]})
        code = version["code"]
        for symbol in code["symbols"]:
            target = path + "::" + symbol["name"]
            nodes.append(dict(symbol, id=target, version=version["id"]))
            edges.append({"from": path, "to": target, "kind": "contains", "source": version["source"]})
    ids = {n["id"] for n in nodes}
    for path, version in files.items():
        if version["status"] != "known":
            continue
        code = version["code"]
        for ref in code["references"]:
            caller = path + "::" + ref["from"] if ref["from"] != "<module>" else path
            target = path + "::" + str(ref["simple_name"])
            imported = code["imports"].get(ref["simple_name"])
            if imported and imported["level"]:
                parent = posixpath.dirname(path)
                for _ in range(imported["level"] - 1):
                    parent = posixpath.dirname(parent)
                module = posixpath.join(parent, imported["module"].replace(".", "/")) + ".py"
                target = module + "::" + imported["name"]
            # Local/global binding shadowing is not resolved by this lightweight parser.
            # References are candidates, never proven runtime calls.
            edge = {"from": caller, "to": target, "kind": "call_reference",
                    "source": version["source"], "expression": ref["expression"],
                    "resolution": "syntactic_candidate"}
            if target in ids and ref["simple_name"]:
                edges.append(edge)
            else:
                unresolved.append({"from": caller, "expression": ref["expression"],
                                   "source": version["source"]})
    return {"at": cutoff, "nodes": nodes, "edges": edges, "unresolved": unresolved}


def query_scope(graph, records, seed, cutoff, hops=2, max_chars=60000):
    current = graph_at(graph, cutoff)
    if seed not in {n["id"] for n in current["nodes"]}:
        raise ValueError("Seed is not a known node at the cutoff")
    historical_edges = []
    for at in sorted({v["observed_at"] for v in graph["versions"] if v["observed_at"] <= cutoff}):
        historical_edges.extend(dict(edge, observed_snapshot=at) for edge in graph_at(graph, at)["edges"])
    selected = {seed}
    for _ in range(hops):
        expanded = set(selected)
        for edge in historical_edges:
            if edge["from"] in selected or edge["to"] in selected:
                expanded.update((edge["from"], edge["to"]))
        selected = expanded
    paths = {node.split("::")[0] for node in selected}
    # Follow version ancestry through moves without equating unrelated basenames.
    by_id = {v["id"]: v for v in graph["versions"]}
    relevant = {v["id"] for v in graph["versions"]
                if v["path"] in paths and v["observed_at"] <= cutoff}
    pending = list(relevant)
    while pending:
        previous = by_id[pending.pop()].get("previous")
        if previous and previous not in relevant:
            relevant.add(previous)
            pending.append(previous)
    paths.update(by_id[vid]["path"] for vid in relevant)
    events = [e for e in graph["events"] if e["order"] <= cutoff
              and paths.intersection(e["affected_paths"])]
    sources = {e["id"] for e in events}
    if not sources:
        raise ValueError("No evidence for query")
    first = min(e["order"] for e in events)
    preceding_user = [r["order"] for r in records if r["order"] < first
                      and r["kind"] == "message" and r.get("role") == "user"]
    if preceding_user:
        first = preceding_user[-1]
    # Keep the contiguous local event window, including execution results and explanations.
    # This is retrieval context, not a claim that every event proves a relationship.
    dialogue = [source_view(r) for r in records if first <= r["order"] <= cutoff]
    import json
    scope = {"seed": seed, "cutoff": cutoff,
            "nodes": [n for n in current["nodes"] if n["id"] in selected],
            "edges": [e for e in current["edges"] if e["from"] in selected and e["to"] in selected],
            "historical_edges": [e for e in historical_edges
                                 if e["from"] in selected and e["to"] in selected],
            "versions": [copy.deepcopy(v) for v in graph["versions"]
                         if v["path"] in paths and v["observed_at"] <= cutoff],
            "events": events, "dialogue": dialogue}
    size = len(json.dumps(scope, ensure_ascii=False))
    return dict(scope, context_chars=size, max_context_chars=max_chars, over_budget=size > max_chars)


def query_scope_adaptive(graph, records, seed, cutoff, initial_hops=1,
                         max_hops=4, max_chars=60000, min_gain=1):
    """Expand until the next layer adds too little evidence or hits a budget.

    The gain is deliberately structural: new nodes, edges, versions, events, and
    dialogue records. It does not pretend to measure semantic usefulness before
    the model extracts facts. Every attempted layer is recorded for audit.
    """
    if initial_hops < 0 or max_hops < initial_hops:
        raise ValueError("invalid adaptive hop range")
    layers = []
    previous = None
    previous_candidate = None
    selected_hops = None
    chosen = None
    for depth in range(initial_hops, max_hops + 1):
        try:
            candidate = query_scope(graph, records, seed, cutoff, depth, max_chars)
        except ValueError:
            if chosen is None:
                raise
            break
        current_keys = {
            "nodes": {(n["id"], n.get("version")) for n in candidate["nodes"]},
            "edges": {(e["from"], e["to"], e["kind"], e.get("observed_snapshot"))
                      for e in candidate.get("historical_edges", candidate["edges"])},
            "versions": {(v["path"], v["id"]) for v in candidate["versions"]},
            "events": {e["id"] for e in candidate["events"]},
            "dialogue": {r["id"] for r in candidate["dialogue"]},
        }
        gain = (0 if previous is None else sum(len(current_keys[name] - previous[name])
                                                for name in current_keys))
        budget_stop = candidate["over_budget"]
        layers.append({"hops": depth, "gain": gain, "counts":
                       {name: len(value) for name, value in current_keys.items()},
                       "stop": budget_stop or (previous is not None and gain < min_gain)})
        if budget_stop or (previous is not None and gain < min_gain):
            chosen = previous_candidate if previous_candidate is not None else candidate
            selected_hops = depth - 1 if previous_candidate is not None else depth
            break
        chosen = candidate
        previous_candidate = candidate
        previous = current_keys
    if chosen is None:
        raise ValueError("No adaptive scope")
    if selected_hops is None:
        selected_hops = layers[-1]["hops"]
    return dict(chosen, adaptive={"initial_hops": initial_hops, "max_hops": max_hops,
                                  "min_gain": min_gain, "layers": layers,
                                  "selected_hops": selected_hops})
