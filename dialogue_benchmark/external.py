"""Build QA scopes from externally supplied dialogue events.

The external-only path deliberately does not construct or search the evidence
graph.  The dialogue producer records small, public-source event bundles and
the QA runner turns each bundle into one bounded evidence scope.
"""

import json
import re
from copy import deepcopy
from pathlib import Path

from .normalize import source_kind_for


EXTERNAL_EVENT_KINDS = frozenset({
    "user_correction",
    "external_observation",
    "environment_observation",
    "perturbation_revealed",
    "compatibility_contract",
    "verification_result",
    "failure_avoidance",
})

EVENT_TYPE = {
    "user_correction": "correction_update",
    "external_observation": "external_state_application",
    "environment_observation": "external_state_application",
    "perturbation_revealed": "failure_avoidance",
    "compatibility_contract": "compatibility_preservation",
    "verification_result": "verification_reuse",
    "failure_avoidance": "failure_avoidance",
}


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
    if isinstance(value, str) and value in {"not_applied", "uncertain"}:
        decision.update(status=value, reason=reason if isinstance(reason, str) and reason.strip()
                        else "usage_not_established")
    elif isinstance(value, str) and value.startswith("applied@"):
        refs = [ref.strip() for ref in re.split(r"[,，]", value.partition("@")[2])]
        sources = [reference_map[ref] for ref in refs if ref in reference_map]
        if (refs and len(sources) == len(refs) and isinstance(reason, str) and reason.strip()
                and set(sources) & set(scope.get("external_usage_ids", []))):
            decision.update(status="applied", reason=reason, sources=sources)
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


def _track(event, records_by_id):
    value = event.get("qa_mode", event.get("track"))
    if value in {"general", "code"}:
        return value
    source_ids = list(event.get("source_ids", [])) + list(event.get("used_by", []))
    if any(source_kind_for(records_by_id[item]) in {"code", "test"}
           for item in source_ids if item in records_by_id):
        return "code"
    return "general"


def _event_scope(event, records, records_by_id, cutoff, track, index, max_chars):
    source_ids = list(event["source_ids"])
    used_by = list(event["used_by"])
    # The producer already declares the evidence boundary. Do not add
    # chronological neighbours: a nearby turn can discuss another decision.
    context_ids = list(event.get("context_ids", []))
    selected_ids = set(source_ids) | set(used_by) | set(context_ids)
    selected_orders = [records_by_id[item].get("order", 0) for item in selected_ids]
    dialogue = [record for record in records if record.get("id") in selected_ids]
    group_id = "external-%s" % event["id"]
    target_type = EVENT_TYPE[event["kind"]]
    return {
        "cutoff": cutoff,
        "track": track,
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
        "external_kind": event["kind"],
        "memory_kind": event.get("memory_kind"),
        "external_source_ids": source_ids,
        "external_usage_ids": used_by,
        # Later public use/result is context for composing a useful question;
        # it remains separate from the source IDs that ground extracted facts.
        "generation_extra_sources": used_by,
        "evidence_group": {
            "id": group_id,
            "qa_mode": track,
            "target_types": [target_type],
            "allowed_types": [target_type],
            "eligible_types": [target_type],
            "type_selection": "preselected",
            "stage_count": 0,
            "graph_hops": 0,
            "reasoning_hops": 1,
        },
    }


def _has_public_source(source_ids, records_by_id):
    """Require at least one visible User/Code message, not only a tool row."""
    return any(records_by_id[item].get("kind") == "message"
               and isinstance(records_by_id[item].get("text"), str)
               and records_by_id[item].get("text", "").strip()
               for item in source_ids)


def load_external_scopes(path, records, cutoff, enabled_tracks, max_chars=32000,
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
        if raw.get("memory_kind") is not None and raw["memory_kind"] not in {
                "M1", "M2", "M3", "M4", "M5", "M6"}:
            rejected.append({"id": event_id, "reason": "unknown_memory_kind"})
            continue
        if raw.get("status", "active") != "active":
            rejected.append({"id": event_id, "reason": "external_event_inactive"})
            continue
        try:
            source_ids = _resolve_ids(raw.get("source_ids"), identities, "source_ids")
            used_by = _resolve_ids(raw.get("used_by"), identities, "used_by")
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
        source_orders = [records_by_id[item].get("order", 0) for item in source_ids]
        usage_orders = [records_by_id[item].get("order", 0) for item in used_by]
        context_orders = [records_by_id[item].get("order", 0) for item in context_ids]
        orders = source_orders + usage_orders + context_orders
        if any(order > cutoff for order in orders):
            rejected.append({"id": event_id, "reason": "event_after_cutoff"})
            continue
        track = _track(raw, records_by_id)
        if track not in enabled_tracks:
            rejected.append({"id": event_id, "reason": "track_disabled", "track": track})
            continue
        if not _has_public_source(source_ids, records_by_id):
            rejected.append({"id": event_id, "reason": "no_public_message_source"})
            continue
        if min(usage_orders) <= min(source_orders):
            rejected.append({"id": event_id, "reason": "usage_not_after_source"})
            continue
        scope = _event_scope(raw, records, records_by_id, cutoff, track,
                             len(scopes), max_chars)
        scopes.append(scope)
        accepted.append(raw)
    if max_groups is not None:
        selected = []
        counts = {track: 0 for track in enabled_tracks}
        for scope, event in zip(scopes, accepted):
            track = scope["track"]
            if counts[track] >= max_groups.get(track, 0):
                rejected.append({"id": event["id"], "reason": "external_group_budget",
                                 "track": track})
                continue
            counts[track] += 1
            selected.append((scope, event))
        scopes = [item[0] for item in selected]
        accepted = [item[1] for item in selected]
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
        kept.append(fact)
    return kept
