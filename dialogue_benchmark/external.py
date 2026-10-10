"""Build QA scopes from externally supplied dialogue events.

The external path keeps the public event sidecar as its source of truth and
uses the deterministic action graph only to bound each event to useful public
evidence before model extraction.
"""

import json
import re
from copy import deepcopy
from pathlib import Path

from .protocol import MEMORY_TYPES
from .anchor_graph import build_action_graph, expand_anchor


EXTERNAL_EVENT_KINDS = frozenset({
    "user_correction",
    "external_observation",
    "environment_observation",
    "perturbation_revealed",
    "compatibility_contract",
    "verification_result",
    "failure_avoidance",
})

_HISTORY_CHANGE = re.compile(
    r"纠正|撤销|改为|改成|例外|除外|仅限|仅在|只适用|确认|不再|取消|更正|但是|不过|"
    r"correct|revoke|replace|instead|except|only|no longer|however|optional", re.I)


def relevant_public_history(records, anchors, text, cutoff, *, include_ambiguous=True):
    """Select same-object amendments and their immediate reference context."""
    from .fact_index import _entities, _semantic_shared_entities

    from .fact_index import _labels
    view = {"entities": _entities(text), "statement_labels": _labels(text)}
    anchor_ids = {row["id"] for row in anchors}
    selected = set(anchor_ids)
    visible = [row for row in records if row.get("order", 0) <= cutoff]
    previous_related = False
    for position, row in enumerate(visible):
        body = str(row.get("text", ""))
        related = bool(_semantic_shared_entities(view, {"entities": _entities(body), "statement_labels": _labels(body)}))
        amendment = bool(_HISTORY_CHANGE.search(body))
        referential = bool(re.search(r"\b(it|that|this|they)\b|the rule|这个|该项|上述|它|此规则", body, re.I))
        ambiguous_revision = (row.get("role") == "user" and amendment and bool(re.search(
            r"仅在|只适用|默认|之前的|only when|only if|default|previous rule", body, re.I))
            and bool(_labels(body)))
        if (row.get("id") in anchor_ids
                or amendment and related
                or previous_related and referential
                or include_ambiguous and ambiguous_revision):
            selected.add(row["id"])
            if referential and position:
                selected.add(visible[position - 1]["id"])
        previous_related = related or row.get("id") in selected
    return [row for row in visible if row.get("id") in selected]


def _event_focus(event):
    """Return only an explicitly supplied target label.

    Event IDs are control metadata. Their slugs must never be promoted to
    factual targets because the public dialogue may describe a different
    object (or several objects).
    """
    for key in ("focus", "target", "object", "label"):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _canonical_source(source):
    """Ignore a local fragment suffix when matching an anchor source."""
    return source.partition("#fragment-")[0] if isinstance(source, str) else source


def derive_external_anchor_metadata(scope, question):
    """Derive control-side anchor metadata from cited public evidence.

    External QA does not receive a single ``type`` label.  The only
    deterministic source of necessity is the answer's public citations and the
    event-to-source map recorded when the scope was built.  A supporting anchor
    may remain in ``anchor_ids`` without entering ``required_anchor_ids``.
    """
    if not isinstance(scope, dict) or not isinstance(question, dict):
        return None, "invalid_external_anchor_input"
    anchor_ids = list(dict.fromkeys(
        value for value in scope.get("anchor_ids", [])
        if isinstance(value, str) and value.strip()))
    source_map = scope.get("anchor_source_map", {})
    if not isinstance(source_map, dict):
        source_map = {}
    required = []
    cited_sources = []
    for point in question.get("answer_points", []):
        if not isinstance(point, dict):
            continue
        for source in point.get("sources", []):
            if not isinstance(source, str):
                continue
            if source not in cited_sources:
                cited_sources.append(source)
            base = _canonical_source(source)
            values = source_map.get(source, source_map.get(base, []))
            if isinstance(values, str):
                values = [values]
            for anchor in values if isinstance(values, (list, tuple, set)) else []:
                if anchor in anchor_ids and anchor not in required:
                    required.append(anchor)
    if not anchor_ids:
        return None, "external_anchor_ids_missing"
    if not required:
        return None, "required_anchor_ids_unresolved"
    kinds_by_anchor = scope.get("anchor_memory_kinds", {})
    if not isinstance(kinds_by_anchor, dict):
        kinds_by_anchor = {}
    memory_kinds = sorted({
        str(kind).upper()
        for anchor in required
        for kind in (kinds_by_anchor.get(anchor, []) if isinstance(
            kinds_by_anchor.get(anchor, []), (list, tuple, set))
            else [kinds_by_anchor.get(anchor)])
        if isinstance(kind, str) and kind.upper() in MEMORY_TYPES
    })
    metadata = {
        "anchor_ids": anchor_ids,
        "required_anchor_ids": required,
        "memory_kinds": memory_kinds,
    }
    if len(required) > 1:
        reason = question.get("combination_reason")
        for review_key in ("target_review", "relevance_review",
                           "evidence_review", "review"):
            review = question.get(review_key)
            if not isinstance(reason, str) and isinstance(review, dict):
                reason = review.get("reason")
        if isinstance(reason, str) and reason.strip():
            metadata["combination_reason"] = reason.strip()
    return metadata, None

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
            # A later public User message can confirm a scoped correction.
            # Evidence review establishes its relation to the question.
            eligible = {source for source, record in records.items()
                        if record.get("kind") == "message" and record.get("role") == "user"}
        else:
            eligible = {source for source in eligible
                        if records.get(source, {}).get("kind") == "result"}
        if (refs and len(sources) == len(refs) and isinstance(reason, str) and reason.strip()
                and set(sources) & eligible):
            decision.update(status=status, reason=reason, sources=sources)
        else:
            decision["reason"] = "invalid_usage_evidence"
    return cleaned, decision


def event_sources(scope):
    """Return the sources that may ground facts and answers of one event.

    These are the event source, its declared uses and context, and later
    public User messages (corrections of the rule) that are not the declared
    source of another event.  Earlier, undeclared messages belong to other
    events.
    """
    sources = set(scope.get("external_source_ids") or [])
    allowed = (sources | set(scope.get("external_usage_ids") or [])
               | set(scope.get("external_declared_context_ids") or []))
    foreign = set(scope.get("external_foreign_source_ids") or [])
    rows = [row for row in scope.get("dialogue", []) if isinstance(row, dict)]
    source_orders = [row.get("order") for row in rows
                     if row.get("id") in sources and isinstance(row.get("order"), int)]
    if source_orders:
        allowed.update(row.get("id") for row in rows
                       if row.get("kind") == "message" and row.get("role") == "user"
                       and isinstance(row.get("order"), int)
                       and row["order"] > min(source_orders)
                       and row.get("id") not in foreign)
    return allowed


def _anchor_record_ids(event, records, graph, max_chars, cutoff):
    """Return public record IDs admitted by one event's anchor expansion."""
    members = event.get("anchor_events")
    if not isinstance(members, list) or not members:
        members = [event]
    selected: set[str] = set()
    audits = []
    for member in members:
        expanded = expand_anchor(dict(member, cutoff=cutoff), graph, max_chars)
        audits.append(expanded)
        for node in expanded.get("nodes", []):
            selected.update(
                value for value in node.get("record_ids", [])
                if isinstance(value, str)
            )
    # Always retain the declared boundary even when expansion is under review.
    for field in ("source_ids", "used_by", "context_ids"):
        selected.update(value for value in event.get(field, [])
                       if isinstance(value, str))
    return selected, audits


def external_anchor_reason(question, scope):
    """Reject an external answer that does not rest on its own event."""
    if not isinstance(scope, dict) or not scope.get("external_event_id"):
        return None
    cited = {source.partition("#fragment-")[0]
             for point in (question.get("answer_points") or [])
             if isinstance(point, dict)
             for source in (point.get("sources") or []) if isinstance(source, str)}
    if not cited:
        return "no_public_source_cited"
    metadata_required = question.get("required_anchor_ids")
    if metadata_required is not None:
        required = [value for value in metadata_required
                    if isinstance(value, str) and value.strip()]
        if not required:
            return "required_anchor_ids_missing"
        source_map = scope.get("anchor_source_map", {})
        if not isinstance(source_map, dict):
            return "external_anchor_not_cited"
        reverse = {}
        for source, anchors in source_map.items():
            values = [anchors] if isinstance(anchors, str) else anchors
            if not isinstance(values, (list, tuple, set)):
                continue
            for anchor in values:
                reverse.setdefault(anchor, set()).add(
                    _canonical_source(source))
        allowed = {
            _canonical_source(source) for source in event_sources(scope)
        }
        for anchor in required:
            if not (cited & reverse.get(anchor, set()) & allowed):
                return ("anchor_not_cited:" + anchor
                        if len(required) > 1 else "external_anchor_not_cited")
        return None
    return None if cited & {
        _canonical_source(source) for source in event_sources(scope)
    } else "external_anchor_not_cited"


def external_review_projection(group, candidate):
    """Review the supplied event boundary without requiring code-graph anchors."""
    from .llm import simple_evidence_request_size

    scope = deepcopy(group["scope"])
    anchors = scope["dialogue"]
    text = "\n".join([candidate.get("question", "")]
                     + [point.get("text", "") for point in candidate.get("answer_points", [])])
    scope["dialogue"] = relevant_public_history(
        scope.pop("public_history", anchors), anchors, text, scope["cutoff"])
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


def _event_scope(event, records, records_by_id, cutoff, index, max_chars,
                 evidence_graph=None):
    source_ids = list(event["source_ids"])
    used_by = list(event["used_by"])
    # Keep the declared seed boundary separate from public correction context.
    context_ids = list(event.get("context_ids", []))
    selected_ids = set(source_ids) | set(used_by) | set(context_ids)
    anchor_audits = []
    if evidence_graph is not None:
        selected_ids, anchor_audits = _anchor_record_ids(
            event, records, evidence_graph, max_chars, cutoff)
    selected_orders = [records_by_id[item].get("order", 0) for item in selected_ids]
    # Graph expansion adds context nodes, but those nodes must not become
    # independent history anchors.  Public amendments are resolved against
    # the event's declared sources/uses; otherwise an admitted correction can
    # broaden the semantic scope and pull in unrelated later corrections.
    declared_anchor_ids = set(source_ids) | set(used_by) | set(context_ids)
    anchors = [record for record in records
               if record.get("id") in declared_anchor_ids]
    anchor_text = "\n".join(str(row.get("text", "")) for row in anchors)
    # Public amendments are conversation messages.  Undeclared tool rows
    # (for example ``str_replace`` edits) would otherwise match the change
    # vocabulary and pull most of the session into every event.
    source_order = min(selected_orders) if selected_orders else cutoff
    history = [
        record for record in records
        if record.get("id") in selected_ids
        or (record.get("kind") == "message"
            and isinstance(record.get("order"), int)
            and record["order"] > source_order)
    ]
    candidate_dialogue = relevant_public_history(
        history, anchors, anchor_text, cutoff)
    candidate_grounded = relevant_public_history(
        history, anchors, anchor_text, cutoff, include_ambiguous=False)

    def keep_context(row, previous, candidate_ids=()):
        # ``relevant_public_history`` has already applied the semantic and
        # revision filters for this scope.  Do not apply a second whitespace
        # token test here: Chinese messages, in particular, are often one
        # whitespace-delimited sentence even when they clearly correct the
        # selected rule.  The candidate set is still bounded by ``history``
        # and the cutoff, while the grounded projection below deliberately
        # uses its stricter, non-ambiguous candidate set.
        if (row.get("id") in selected_ids
                or row.get("role") == "assistant"
                or row.get("id") in candidate_ids):
            return True
        if row.get("role") != "user":
            return False
        text = str(row.get("text", ""))
        if re.search(r"\b(it|that|this|they)\b|the rule|这个|该项|上述|它|此规则",
                     text, re.I) and previous and previous.get("role") == "assistant":
            return True
        return bool(re.search(r"纠正|更正|改为|改成|例外|only|except|instead|no longer",
                              text, re.I)
                    and any(token in text for token in anchor_text.split()[:4]))

    dialogue = []
    previous = None
    for row in candidate_dialogue:
        if keep_context(row, previous, {item["id"] for item in candidate_dialogue}):
            dialogue.append(row)
        previous = row
    grounded_context = []
    previous = None
    for row in candidate_grounded:
        if keep_context(row, previous, {item["id"] for item in candidate_grounded}):
            grounded_context.append(row)
        previous = row
    group_id = "external-%s" % event["id"]
    target_type = event["memory_kind"]
    # A merged disclosure is one public anchor even if it has several
    # controller-side event ids.  An explicit combination keeps one anchor
    # per contributing event so a later candidate can distinguish required
    # from supporting evidence.
    if event.get("combination"):
        anchor_ids = list(dict.fromkeys(
            event.get("anchor_ids") or event.get("event_ids", [event["id"]])))
    else:
        anchor_ids = [event["id"]]
    anchor_memory_kinds = event.get("anchor_memory_kinds")
    if not isinstance(anchor_memory_kinds, dict):
        anchor_memory_kinds = {
            anchor_ids[0]: list(event.get("memory_kinds", [target_type]))
        }
    anchor_memory_kinds = {
        str(anchor): (list(dict.fromkeys(value)) if isinstance(value, (list, tuple, set))
                      else [value])
        for anchor, value in anchor_memory_kinds.items()
        if isinstance(anchor, str)
    }
    source_map = event.get("anchor_source_map")
    if not isinstance(source_map, dict):
        source_map = {}
    source_map = {
        str(source): list(dict.fromkeys(values if isinstance(values, (list, tuple, set))
                                        else [values]))
        for source, values in source_map.items()
        if isinstance(source, str)
    }
    if not source_map:
        for source in selected_ids:
            source_map[source] = list(anchor_ids)
    # A later public correction is part of the event's evidence boundary, but
    # an earlier related message is only context.  Keep the source map aligned
    # with event_sources so context cannot be promoted to a required anchor.
    allowed_ids = set(source_ids) | set(used_by) | set(context_ids)
    source_orders = [
        records_by_id[item].get("order", 0)
        for item in source_ids
        if item in records_by_id and isinstance(records_by_id[item].get("order"), int)
    ]
    foreign = set(event.get("foreign_source_ids", []))
    if source_orders:
        allowed_ids.update(
            row["id"] for row in dialogue
            if row.get("kind") == "message"
            and row.get("role") == "user"
            and isinstance(row.get("order"), int)
            and row["order"] > min(source_orders)
            and row.get("id") not in foreign
        )
    for row in dialogue:
        if row["id"] in allowed_ids:
            source_map.setdefault(row["id"], list(anchor_ids))
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
        "anchor_ids": anchor_ids,
        "anchor_memory_kinds": anchor_memory_kinds,
        "anchor_source_map": source_map,
        "anchor_audits": anchor_audits,
        "external_kind": event["kind"],
        "memory_kind": target_type,
        "memory_kinds": event.get("memory_kinds", [target_type]),
        "external_source_ids": source_ids,
        "external_usage_ids": used_by,
        "external_declared_context_ids": context_ids,
        "external_context_source_ids": [row["id"] for row in grounded_context if row["id"] not in selected_ids],
        "external_focus": event.get("focus"),
        # These are control-side labels used to relate QA to a business
        # behavior. They are never copied into the public question projection.
        "external_behavior": event.get("behavior") or event.get("focus"),
        "external_impact": (event.get("impact") or event.get("behavior_impact")
                             or event.get("constraint")),
        # Later public use/result is context for composing a useful question;
        # it remains separate from the source IDs that ground extracted facts.
        "generation_extra_sources": list(dict.fromkeys(used_by + context_ids + [
            row["id"] for row in dialogue if row["id"] not in source_ids])),
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


def _event_groups(events, records_by_id, *, merge_task_events=True):
    """Connect revisions and repeated disclosures without hiding new facts.

    An event is a new group only when its public anchor carries a new fact.
    Events that point at the same anchors, or repeat the same public wording,
    are one group even when their controller-side ids differ.  Explicit
    supersession and (when requested) task links still connect revisions.
    """
    by_id = {event["id"]: event for event in events}
    links = {identity: set() for identity in by_id}
    tasks = {}
    anchor_groups = {}
    wording_groups = {}

    def public_wording(event):
        texts = []
        for source in event.get("source_ids", []):
            text = records_by_id[source].get("text", "")
            text = re.sub(r"\s+", " ", str(text)).strip().casefold()
            if text:
                texts.append(text)
        return tuple(texts)

    for event in events:
        identity = event["id"]
        task = event.get("task_id") if merge_task_events else None
        if task:
            if task in tasks:
                links[identity].add(tasks[task])
                links[tasks[task]].add(identity)
            tasks[task] = identity
        # A repeated disclosure with a different event id is still the same
        # public fact.  Exact anchor equality also coalesces multiple labels
        # attached to one disclosure while preserving distinct anchors.
        focus = (event.get("behavior") or event.get("focus") or "").strip().casefold()
        anchor_key = (focus, tuple(event.get("source_ids", [])),
                      tuple(event.get("used_by", [])),
                      tuple(event.get("context_ids", [])))
        if anchor_key in anchor_groups:
            other = anchor_groups[anchor_key]
            links[identity].add(other)
            links[other].add(identity)
        else:
            anchor_groups[anchor_key] = identity
        wording_key = (focus, public_wording(event), tuple(event.get("used_by", [])),
                       tuple(event.get("context_ids", [])))
        if wording_key[1] and wording_key in wording_groups:
            other = wording_groups[wording_key]
            links[identity].add(other)
            links[other].add(identity)
        elif wording_key[1]:
            wording_groups[wording_key] = identity
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
        canonical_id = merged["id"]
        merged["anchor_memory_kinds"] = {
            canonical_id: sorted({row["memory_kind"] for row in members})
        }
        for field in ("source_ids", "used_by", "context_ids"):
            merged[field] = list(dict.fromkeys(source for row in members for source in row[field]))
        merged["anchor_source_map"] = {
            source: [canonical_id]
            for source in (merged["source_ids"] + merged["used_by"]
                           + merged["context_ids"])
        }
        yield merged


def _combination_groups(events):
    """Offer one supplemental judgment per explicit business object."""
    related = {}
    for event in events:
        focus = event.get("behavior") or event.get("focus")
        if isinstance(focus, str) and focus.strip():
            related.setdefault(focus.casefold(), []).append(event)
    for members in related.values():
        if len(members) < 2:
            continue
        combined = dict(members[-1])
        combined["id"] = "combined-" + "-".join(row["id"] for row in members)
        combined["anchor_events"] = [dict(row) for row in members]
        # Each merged member is one independent anchor.  ``event_ids`` keeps
        # the full disclosure lineage, while ``anchor_ids`` names the
        # canonical member anchors used by the QA contract.
        combined["anchor_ids"] = [row["id"] for row in members]
        combined["event_ids"] = list(dict.fromkeys(identity for row in members for identity in row["event_ids"]))
        combined["memory_kinds"] = sorted({kind for row in members for kind in row["memory_kinds"]})
        combined["anchor_memory_kinds"] = {
            anchor: sorted({kind for kind in row.get("anchor_memory_kinds", {}).get(anchor, [])
                            if kind in MEMORY_TYPES})
            for row in members
            for anchor in row.get("anchor_memory_kinds", {})
        }
        for field in ("source_ids", "used_by", "context_ids"):
            combined[field] = list(dict.fromkeys(source for row in members for source in row[field]))
        combined["anchor_source_map"] = {}
        for row in members:
            row_map = row.get("anchor_source_map", {})
            for source in row["source_ids"] + row["used_by"] + row["context_ids"]:
                anchors = row_map.get(source, [row["id"]])
                combined["anchor_source_map"].setdefault(source, [])
                combined["anchor_source_map"][source].extend(anchors)
        combined["anchor_source_map"] = {
            source: list(dict.fromkeys(anchors))
            for source, anchors in combined["anchor_source_map"].items()
        }
        combined["combination"] = True
        yield combined


def load_external_scopes(path, records, cutoff, max_chars=32000,
                         max_groups=None, *, merge_task_events=True):
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
                   used_by=used_by, context_ids=context_ids,
                   focus=_event_focus(raw))
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
    groups = list(_event_groups(accepted, records_by_id, merge_task_events=merge_task_events))
    combinations = list(_combination_groups(groups)) if not merge_task_events else []
    evidence_graph = build_action_graph(records)
    for event in groups + combinations:
        if max_groups is not None and len(scopes) >= max_groups:
            rejected.append({"id": event["id"], "reason": "external_group_budget"})
            continue
        scopes.append(_event_scope(
            event, records, records_by_id, cutoff, len(scopes), max_chars,
            evidence_graph=evidence_graph))
    # A later message may correct this event's rule, unless it is the
    # declared source of another event; then it states that event's rule.
    for scope in scopes:
        own = set(scope["external_source_ids"])
        scope["external_foreign_source_ids"] = sorted(
            {source for event in accepted for source in event["source_ids"]} - own)
    return {"version": document.get("version"), "events": accepted,
            "scopes": scopes, "rejected": rejected}


def filter_external_facts(facts, scopes):
    """Keep facts grounded in an event's own public sources (``event_sources``).

    A fact sourced only from another event's message or earlier shared
    context is not attributed to this event, so it cannot seed this event's
    question.
    """
    source_to_scope = {}
    for scope in scopes:
        event_id = scope.get("external_event_id")
        for source in sorted(event_sources(scope)):
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
