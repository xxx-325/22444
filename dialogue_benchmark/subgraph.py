"""Small adaptive evidence subgraphs for dialogue-to-QA construction."""

import copy
import json
import re

from .graph import graph_at


_FEEDBACK = re.compile(r"反馈|不对|没[有能]|差点意思|不符合|问题是|要求|希望|应该", re.I)
_FAILURE = re.compile(r"失败|报错|错误|异常|crash|error|failed|failure|bug", re.I)
_VALIDATION = re.compile(r"测试|实验|验证|结果|通过|命中|改善|提升|回归|benchmark", re.I)
_DESIGN = re.compile(r"方案|设计|改成|改为|采用|重构|实现|增加|引入", re.I)


def _text(record):
    return record.get("text", "") if isinstance(record.get("text", ""), str) else ""


def _labels(text):
    labels = set()
    if _FEEDBACK.search(text):
        labels.add("feedback")
    if _FAILURE.search(text):
        labels.add("failure")
    if _VALIDATION.search(text):
        labels.add("validation")
    if _DESIGN.search(text):
        labels.add("design")
    return sorted(labels)


def build_event_index(records, graph, neighbor_window=6):
    """Attach nearby dialogue evidence to code events without semantic guessing."""
    by_order = {record["order"]: record for record in records}
    indexed = []
    for event in graph.get("events", []):
        if event.get("kind") not in {"patch", "observation", "result"}:
            continue
        order = event["order"]
        nearby = [record for record in records
                  if abs(record["order"] - order) <= neighbor_window]
        texts = "\n".join(_text(record) for record in nearby)
        labels = set(_labels(texts))
        if event.get("kind") == "patch":
            labels.add("change")
        if event.get("success") is False:
            labels.add("failure")
        source_ids = {event["id"]}
        source_ids.update(record["id"] for record in nearby)
        indexed.append({
            "id": event["id"],
            "order": order,
            "kind": event.get("kind"),
            "paths": list(event.get("affected_paths", [])),
            "labels": sorted(labels),
            "source_ids": sorted(source_ids),
        })
    return indexed


def _score(events, versions, source_ids):
    labels = {label for event in events for label in event["labels"]}
    score = 0
    if "change" in labels:
        score += 2
    if len(versions) >= 2:
        score += 3
    if labels & {"feedback", "failure", "validation"}:
        score += 2
    if len({path for event in events for path in event["paths"]}) > 1:
        score += 2
    if len(events) >= 2:
        score += 2
    if len(source_ids) >= 2:
        score += 1
    return score


def _graph_context(graph, cutoff):
    snapshots = {at: graph_at(graph, at) for at in sorted(
        {cutoff} | {v["observed_at"] for v in graph.get("versions", [])
                    if isinstance(v.get("observed_at"), int) and v["observed_at"] <= cutoff})}
    historical = [dict(edge, observed_snapshot=at)
                  for at, snapshot in snapshots.items()
                  if any(v.get("observed_at") == at for v in graph.get("versions", []))
                  for edge in snapshot.get("edges", [])]
    by_source, by_path = {}, {}
    for index, edge in enumerate(historical):
        by_source.setdefault(edge.get("source"), []).append(index)
        for path in {edge.get("from", "").split("::", 1)[0],
                     edge.get("to", "").split("::", 1)[0]}:
            by_path.setdefault(path, []).append(index)
    return {"current": snapshots[cutoff], "historical": historical,
            "by_source": by_source, "by_path": by_path}


def _scope(graph, records, events, cutoff, seed, max_chars, context):
    selected_event_ids = {event["id"] for event in events}
    source_ids = {source for event in events for source in event["source_ids"]}
    orders = {event["order"] for event in events}
    versions_by_id = {version.get("id"): version for version in graph.get("versions", [])
                      if isinstance(version, dict) and isinstance(version.get("id"), str)}
    version_ids = {version.get("id") for version in graph.get("versions", [])
                   if version.get("observed_at") in orders}
    # A later version without its ancestors cannot support a historical
    # comparison. Include the complete previous chain before projecting it.
    pending = list(version_ids)
    while pending:
        version_id = pending.pop()
        previous = versions_by_id.get(version_id, {}).get("previous")
        if previous and previous not in version_ids:
            version_ids.add(previous)
            pending.append(previous)
    versions = [copy.deepcopy(versions_by_id[version_id]) for version_id in version_ids
                if version_id in versions_by_id]
    seed_path = seed.split("::")[0] if seed else ""
    event_paths = {path for event in events for path in event["paths"]}
    # A patch may touch many files. Keep the seed path by default; additional
    # files need an explicit graph edge and are added by a later expansion.
    paths = {path for path in event_paths if path == seed_path or not seed_path}
    if not paths and event_paths:
        paths = {next(iter(event_paths))}
    versions.extend(copy.deepcopy(version) for version in graph.get("versions", [])
                    if version.get("path") in paths and version.get("id") not in
                    {item["id"] for item in versions}
                    and version.get("observed_at", cutoff + 1) <= cutoff
                    and version.get("previous") in {item["id"] for item in versions})
    versions.sort(key=lambda item: item.get("observed_at", 0))
    source_ids.update(version.get("source") for version in versions
                      if isinstance(version.get("source"), str))
    selected_event_ids.update(version.get("source") for version in versions
                              if isinstance(version.get("source"), str))
    selected_records = [record for record in records
                        if record["id"] in source_ids and record["order"] <= cutoff]
    selected_records.sort(key=lambda record: record["order"])
    event_map = {event["id"]: event for event in graph.get("events", [])}
    graph_events = [copy.deepcopy(event_map[event_id]) for event_id in selected_event_ids
                    if event_id in event_map]
    graph_events.sort(key=lambda event: event["order"])
    current = context["current"]
    selected_paths = {path for event in events for path in event.get("paths", [])}
    selected_paths.update(version.get("path") for version in versions
                          if isinstance(version.get("path"), str))

    # The authoring graph is also a locator: retain one-hop code references
    # touching the selected files even when the referenced file had no
    # separate event in this candidate.  Without this, the default adaptive
    # path silently drops an already recovered cross-file relation.
    reachable_paths = set(selected_paths)
    for edge in current.get("edges", []):
        left, right = edge.get("from", ""), edge.get("to", "")
        left_path, right_path = left.split("::", 1)[0], right.split("::", 1)[0]
        if left_path in selected_paths or right_path in selected_paths:
            reachable_paths.update((left_path, right_path))
    selected_paths = {path for path in reachable_paths if path}

    def edge_in_scope(edge):
        left, right = edge.get("from", ""), edge.get("to", "")
        left_path, right_path = left.split("::", 1)[0], right.split("::", 1)[0]
        return (edge.get("source") in selected_event_ids
                or (left_path in selected_paths and right_path in selected_paths))

    edges = [copy.deepcopy(edge) for edge in current.get("edges", [])
             if edge_in_scope(edge)]
    related = {index for source in selected_event_ids
               for index in context["by_source"].get(source, [])}
    related.update(index for path in selected_paths for index in context["by_path"].get(path, []))
    historical_edges = [context["historical"][index] for index in sorted(related)
                        if edge_in_scope(context["historical"][index])]
    payload = {
        "seed": seed,
        "cutoff": cutoff,
        "nodes": [copy.deepcopy(node) for node in current.get("nodes", [])
                  if node.get("id", "").split("::", 1)[0] in selected_paths],
        "edges": edges,
        "historical_edges": historical_edges,
        "versions": versions,
        "events": graph_events,
        "dialogue": [dict(record) for record in selected_records],
        "subgraph_events": events,
    }
    payload["context_chars"] = len(json.dumps(payload, ensure_ascii=False))
    payload["max_context_chars"] = max_chars
    payload["over_budget"] = payload["context_chars"] > max_chars
    payload["score"] = _score(events, versions, source_ids)
    return payload


def _select_candidates(candidates, root_order, limit):
    """Prefer one useful scope across the timeline before filling by score."""
    if len(candidates) <= limit:
        return candidates
    by_root = {}
    for candidate in candidates:
        by_root.setdefault(candidate.get("candidate_seed_event"), []).append(candidate)
    roots = [root for root in root_order if root in by_root]
    if len(roots) > limit:
        if limit == 1:
            roots = [roots[len(roots) // 2]]
        else:
            roots = [roots[round(index * (len(roots) - 1) / (limit - 1))]
                     for index in range(limit)]
    selected = [by_root[root][0] for root in roots]
    selected_ids = {id(candidate) for candidate in selected}
    selected.extend(candidate for candidate in candidates
                    if id(candidate) not in selected_ids)
    return selected[:limit]


def _select_root_seeds(index, limit):
    """Score the full event range, then expand a bounded timeline-diverse set."""
    if len(index) <= limit:
        return list(index)
    chronological = sorted(index, key=lambda event: (event["order"], event["id"]))
    timeline_count = max(2, limit // 2)
    timeline = [chronological[round(position * (len(chronological) - 1)
                                     / (timeline_count - 1))]
                for position in range(timeline_count)]
    ranked = sorted(
        index,
        key=lambda event: (
            -(3 * len(set(event["labels"]) & {"feedback", "failure", "validation"})
              + 2 * ("change" in event["labels"])
              + min(len(event["paths"]), 3)),
            event["order"], event["id"],
        ),
    )
    selected = []
    seen = set()
    for event in timeline + ranked:
        if event["id"] in seen:
            continue
        selected.append(event)
        seen.add(event["id"])
        if len(selected) >= limit:
            break
    return selected


def adaptive_subgraphs(graph, records, cutoff, seed=None, max_chars=80000,
                       beam_width=3, max_depth=5, min_score=8,
                       max_candidates=None):
    """Grow small event-linked candidates and stop once they are structurally useful."""
    index = build_event_index(records, graph)
    if seed:
        seed_paths = {seed.split("::")[0]}
        index = [event for event in index
                 if seed in event["paths"] or seed_paths.intersection(event["paths"])]
    if not index:
        return [], {"seeds": [], "layers": []}
    index_by_id = {event["id"]: event for event in index}
    ranked = sorted(index, key=lambda event: (event["order"], len(event["labels"])), reverse=True)
    # Without a seed, score every event but expand only a bounded mix of useful
    # and timeline-diverse roots. Expanding every event against every other event
    # is quadratic and does not improve the final bounded candidate set.
    if seed is None:
        root_limit = max(12, (max_candidates or beam_width * 2) * 3)
        seeds = _select_root_seeds(index, root_limit)
    else:
        seeds = ranked[:min(beam_width, len(ranked))]
    states = [{"event_ids": (event["id"],), "root_id": event["id"],
               "history": ["seed:" + event["id"]]}
              for event in seeds]
    accepted, layers = [], []
    context = _graph_context(graph, cutoff)
    for depth in range(max_depth + 1):
        next_states = []
        for state in states:
            events = [index_by_id[event_id] for event_id in state["event_ids"]]
            scope = _scope(graph, records, events, cutoff, seed, max_chars, context)
            scope["candidate_seed_event"] = state["root_id"]
            if (scope["score"] >= min_score and len(scope["versions"]) >= 2
                    and len(scope["dialogue"]) >= 2):
                accepted.append(scope)
            # A structurally useful oversized scope is handed to the lossless
            # chunker. Expanding it further would add cost without helping find
            # a smaller evidence group.
            if depth >= max_depth or scope["over_budget"]:
                continue
            paths = {path for event in events for path in event["paths"]}
            labels = {label for event in events for label in event["labels"]}
            candidates = []
            for event in index:
                if event["id"] in state["event_ids"]:
                    continue
                shared_path = paths.intersection(event["paths"])
                useful_label = (labels.intersection(event["labels"])
                                & {"feedback", "failure", "validation"})
                distance = min(abs(event["order"] - item["order"]) for item in events)
                if shared_path or useful_label or distance <= 30:
                    candidates.append((0 if shared_path else 1, 0 if useful_label else 1,
                                       distance, event["id"]))
            for _, _, _, event_id in sorted(candidates)[:4]:
                next_states.append({
                    "event_ids": tuple(sorted(state["event_ids"] + (event_id,))),
                    "root_id": state["root_id"],
                    "history": state["history"] + ["add:" + event_id],
                })
        unique = {}
        for state in next_states:
            unique[(state["root_id"], state["event_ids"])] = state
        scored_by_root = {}
        for state in unique.values():
            events = [index_by_id[event_id] for event_id in state["event_ids"]]
            score = _score(events, [], set().union(*(set(e["source_ids"]) for e in events)))
            scored_by_root.setdefault(state["root_id"], []).append((score, state))
        states = []
        for root_id in [event["id"] for event in seeds]:
            ranked_states = sorted(
                scored_by_root.get(root_id, []),
                key=lambda item: (-item[0], item[1]["event_ids"]),
            )
            states.extend(state for _, state in ranked_states[:beam_width])
        layers.append({"depth": depth, "states": len(states), "accepted": len(accepted)})
        if not states:
            break
    # Deduplicate equivalent event sets, retaining the smallest useful evidence.
    unique = {}
    for scope in accepted:
        key = tuple(event["id"] for event in scope["subgraph_events"])
        if key not in unique or (scope["score"], -scope["context_chars"]) > \
                (unique[key]["score"], -unique[key]["context_chars"]):
            unique[key] = scope
    candidates = sorted(unique.values(),
                        key=lambda item: (-item["score"], item["context_chars"],
                                          len(item["subgraph_events"])))
    limit = max_candidates if max_candidates is not None else max(beam_width * 2, 1)
    if limit <= 0:
        raise ValueError("max_candidates must be positive")
    root_order = [event["id"] for event in sorted(seeds, key=lambda item: item["order"])]
    returned = _select_candidates(candidates, root_order, limit)
    return returned, {
        "enumerated_seed_count": len(index),
        "seeds": [event["id"] for event in seeds],
        "layers": layers,
        "accepted_before_dedup": len(accepted),
        "returned": len(returned),
    }
