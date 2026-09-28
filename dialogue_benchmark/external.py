"""Build QA scopes from externally supplied dialogue events.

The external-only path deliberately does not construct or search the evidence
graph.  The dialogue producer records small, public-source event bundles and
the QA runner turns each bundle into one bounded evidence scope.
"""

import json
import re
from copy import deepcopy
from pathlib import Path

from .protocol import MEMORY_TYPES


EXTERNAL_EVENT_KINDS = frozenset({
    "user_correction",
    "external_observation",
    "environment_observation",
    "perturbation_revealed",
    "compatibility_contract",
    "verification_result",
    "failure_avoidance",
})

def external_usage_review(document, scope, reference_map):
    """Separate the event-use judgment from immutable answer-point judgments."""
    cleaned = deepcopy(document)
    reviews = cleaned.get("reviews", []) if isinstance(cleaned, dict) else []
    decision = {"status": "uncertain", "reason": "missing_usage_review", "sources": []}
    if len(reviews) != 1 or not isinstance(reviews[0], dict):
        return cleaned, decision
    review = reviews[0]
    value = review.pop("usage", "")
    reason = review.pop("usage_reason", "")
    if value:
        decision["reason"] = "invalid_usage_evidence"
    if isinstance(value, str) and value in {"not_applied", "uncertain"}:
        decision.update(status=value, reason=reason if isinstance(reason, str) and reason.strip()
                        else "usage_not_established")
    elif isinstance(value, str) and value.startswith(("applied@", "confirmed@")):
        status = value.partition("@")[0]
        refs = [ref.strip() for ref in re.split(r"[,，]", value.partition("@")[2])]
        sources = [reference_map[ref] for ref in refs if ref in reference_map]
        records = {row["id"]: row for row in scope.get("dialogue", [])}
        eligible = set(scope.get("external_usage_ids", []))
        if status == "confirmed":
            eligible = {source for source in scope.get("external_source_ids", [])
                        if records.get(source, {}).get("role") == "user"}
        else:
            eligible = {source for source in eligible
                        if records.get(source, {}).get("kind") == "result"}
        if (refs and len(sources) == len(refs) and isinstance(reason, str) and reason.strip()
                and set(sources) & eligible):
            decision.update(status=status, reason=reason, sources=sources)
        else:
            decision["reason"] = "invalid_usage_evidence"
    return cleaned, decision


def external_review_projection(group, candidate):
    """Review the supplied event boundary without requiring code-graph anchors."""
    from .llm import simple_evidence_request_size

    scope = deepcopy(group["scope"])
    records = {row["id"]: row for row in scope["dialogue"]}
    required = set(scope["external_source_ids"]) | set(scope["external_usage_ids"])
    cited = {source for key in ("answer_points", "forbidden_points")
             for point in candidate.get(key, []) for source in point.get("sources", [])}
    audit = {"complete": False, "reason": "unresolved_external_source",
             "required_source_ids": sorted(required | cited), "request_chars": None}
    if (required | cited) - records.keys():
        return None, audit
    if any(row.get("order", 0) > scope["cutoff"] for row in records.values()):
        audit["reason"] = "external_source_after_cutoff"
        return None, audit
    sources = sorted(records)
    scope.update(review_guard_sources=sources, review_guard_complete=True,
                 review_guard_reason="external_event_complete")
    size = simple_evidence_request_size(scope, sources, group["facts"], candidate)
    audit.update(request_chars=size, guard_source_count=len(sources))
    if size > scope["model_request_chars"]:
        audit["reason"] = "candidate_guard_over_budget"
        return None, audit
    audit.update(complete=True, reason="external_event_complete")
    return dict(group, scope=scope, review_guard_complete=True), audit


def _read_events(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        value = {"version": 1, "events": [json.loads(line) for line in text.splitlines()
                                             if line.strip()]}
    else:
        value = json.loads(text)
    if isinstance(value, list):
        value = {"version": 1, "events": value}
    if not isinstance(value, dict) or value.get("version") != 1:
        raise ValueError("external events require version=1")
    events = value.get("events")
    if not isinstance(events, list):
        raise ValueError("external events require events[]")
    return value, events


def _identity_map(records):
    result = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        identity = record.get("id")
        if isinstance(identity, str):
            result[identity] = identity
        original = record.get("original_id")
        if isinstance(original, str):
            result[original] = identity
    return result


def _resolve_ids(values, identities, label):
    if not isinstance(values, list) or not values:
        raise ValueError("external event %s requires non-empty %s" % (label, label))
    resolved = []
    for value in values:
        if not isinstance(value, str) or value not in identities:
            raise ValueError("unknown external event source: %s" % value)
        identity = identities[value]
        if identity not in resolved:
            resolved.append(identity)
    return resolved


def _event_scope(event, records, records_by_id, cutoff, index, max_chars):
    source_ids = list(event["source_ids"])
    used_by = list(event["used_by"])
    # The producer already declares the evidence boundary. Do not add
    # chronological neighbours: a nearby turn can discuss another decision.
    context_ids = list(event.get("context_ids", []))
    selected_ids = set(source_ids) | set(used_by) | set(context_ids)
    selected_orders = [records_by_id[item].get("order", 0) for item in selected_ids]
    dialogue = [record for record in records if record.get("id") in selected_ids]
    group_id = "external-%s" % event["id"]
    target_type = event["memory_kind"]
    return {
        "cutoff": cutoff,
        "track": "memory",
        "scope_index": index,
        "chunk_index": index,
        "chunk_window": [min(selected_orders), max(selected_orders)],
        "dialogue": dialogue,
        "events": [],
        "versions": [],
        "edges": [],
        "historical_edges": [],
        "full_range_covered": False,
        "model_request_chars": max_chars,
        "external_event_id": event["id"],
        "external_event_ids": event.get("event_ids", [event["id"]]),
        "external_kind": event["kind"],
        "memory_kind": target_type,
        "memory_kinds": event.get("memory_kinds", [target_type]),
        "external_source_ids": source_ids,
        "external_usage_ids": used_by,
        # Later public use/result is context for composing a useful question;
        # it remains separate from the source IDs that ground extracted facts.
        "generation_extra_sources": list(dict.fromkeys(used_by + context_ids)),
        "evidence_group": {
            "id": group_id,
            "qa_mode": "memory",
            "target_types": [target_type],
            "allowed_types": [target_type],
            "eligible_types": [target_type],
            "type_selection": "preselected",
        },
    }


def _has_public_source(source_ids, records_by_id):
    """Require at least one visible User/Code message, not only a tool row."""
    return any(records_by_id[item].get("kind") == "message"
               and isinstance(records_by_id[item].get("text"), str)
               and records_by_id[item].get("text", "").strip()
               for item in source_ids)


def _event_groups(events, records_by_id):
    """Connect explicit task membership and revisions, never topic similarity."""
    by_id = {event["id"]: event for event in events}
    links = {identity: set() for identity in by_id}
    tasks = {}
    for event in events:
        identity = event["id"]
        task = event.get("task_id")
        if task:
            if task in tasks:
                links[identity].add(tasks[task])
                links[tasks[task]].add(identity)
            tasks[task] = identity
        for previous in event.get("supersedes", []):
            if previous in by_id:
                links[identity].add(previous)
                links[previous].add(identity)
    remaining = set(by_id)
    for event in events:
        if event["id"] not in remaining:
            continue
        members, pending = [], [event["id"]]
        while pending:
            identity = pending.pop()
            if identity not in remaining:
                continue
            remaining.remove(identity)
            members.append(by_id[identity])
            pending.extend(sorted(links[identity]))
        members.sort(key=lambda row: (max(records_by_id[source].get("order", 0)
                                          for source in row["source_ids"]), row["id"]))
        # The latest disclosure supplies the primary static type. All member
        # types and original records remain available for review.
        merged = dict(members[-1])
        merged["event_ids"] = [row["id"] for row in members]
        merged["memory_kinds"] = sorted({row["memory_kind"] for row in members})
        for field in ("source_ids", "used_by", "context_ids"):
            merged[field] = list(dict.fromkeys(source for row in members for source in row[field]))
        yield merged


def load_external_scopes(path, records, cutoff, max_chars=32000,
                         max_groups=None):
    """Load validated public-source events and build bounded QA scopes.

    The sidecar contains provenance and event kind, not a second private answer.
    Fact text is extracted from the referenced public dialogue records.
    """
    document, raw_events = _read_events(path)
    identities = _identity_map(records)
    records_by_id = {record["id"]: record for record in records}
    scopes, accepted, rejected = [], [], []
    seen = set()
    for raw in raw_events:
        if not isinstance(raw, dict):
            rejected.append({"reason": "malformed_external_event"})
            continue
        event_id = raw.get("id")
        kind = raw.get("kind")
        if not isinstance(event_id, str) or not event_id:
            rejected.append({"reason": "external_event_id_missing"})
            continue
        if event_id in seen:
            rejected.append({"id": event_id, "reason": "duplicate_external_event"})
            continue
        seen.add(event_id)
        if kind not in EXTERNAL_EVENT_KINDS:
            rejected.append({"id": event_id, "reason": "unknown_external_event_kind"})
            continue
        if raw.get("memory_kind") not in MEMORY_TYPES:
            rejected.append({"id": event_id, "reason": "invalid_memory_kind"})
            continue
        if raw.get("status", "active") != "active":
            rejected.append({"id": event_id, "reason": "external_event_inactive"})
            continue
        try:
            source_ids = _resolve_ids(raw.get("source_ids"), identities, "source_ids")
            used_by = (_resolve_ids(raw["used_by"], identities, "used_by")
                       if raw.get("used_by") else [])
            context_ids = raw.get("context_ids", [])
            if context_ids:
                context_ids = _resolve_ids(context_ids, identities, "context_ids")
            else:
                context_ids = []
        except ValueError as error:
            rejected.append({"id": event_id, "reason": str(error)})
            continue
        raw = dict(raw, id=event_id, kind=kind, source_ids=source_ids,
                   used_by=used_by, context_ids=context_ids)
        if ("task_id" in raw and (not isinstance(raw["task_id"], str) or not raw["task_id"].strip())
                or not isinstance(raw.get("supersedes", []), list)
                or any(not isinstance(item, str) or not item for item in raw.get("supersedes", []))):
            rejected.append({"id": event_id, "reason": "invalid_external_relation"})
            continue
        source_orders = [records_by_id[item].get("order", 0) for item in source_ids]
        usage_orders = [records_by_id[item].get("order", 0) for item in used_by]
        context_orders = [records_by_id[item].get("order", 0) for item in context_ids]
        orders = source_orders + usage_orders + context_orders
        if any(order > cutoff for order in orders):
            rejected.append({"id": event_id, "reason": "event_after_cutoff"})
            continue
        if not _has_public_source(source_ids, records_by_id):
            rejected.append({"id": event_id, "reason": "no_public_message_source"})
            continue
        if usage_orders and min(usage_orders) <= min(source_orders):
            rejected.append({"id": event_id, "reason": "usage_not_after_source"})
            continue
        accepted.append(raw)
    for event in _event_groups(accepted, records_by_id):
        if max_groups is not None and len(scopes) >= max_groups:
            rejected.append({"id": event["id"], "reason": "external_group_budget"})
            continue
        scopes.append(_event_scope(event, records, records_by_id, cutoff, len(scopes), max_chars))
    return {"version": document.get("version"), "events": accepted,
            "scopes": scopes, "rejected": rejected}


def filter_external_facts(facts, scopes):
    """Keep facts grounded in an event's declared public source records."""
    source_to_scope = {}
    for scope in scopes:
        event_id = scope.get("external_event_id")
        for source in scope.get("external_source_ids", []):
            source_to_scope.setdefault(source, []).append((event_id, scope))
    kept = []
    for fact in facts:
        sources = set(fact.get("sources", [])) if isinstance(fact, dict) else set()
        matches = []
        for source in sources:
            matches.extend(source_to_scope.get(source, []))
        if not matches:
            continue
        event_ids = sorted({item[0] for item in matches})
        fact["external"] = True
        fact["external_event_ids"] = event_ids
        fact["external_kind"] = matches[0][1].get("external_kind")
        fact["memory_kinds"] = sorted({scope["memory_kind"] for _, scope in matches
                                       if scope.get("memory_kind") in MEMORY_TYPES})
        kept.append(fact)
    return kept
