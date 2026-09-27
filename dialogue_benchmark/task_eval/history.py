"""Freeze public history and inspect its use independently of task correctness."""

from pathlib import Path
import re

from ..llm import parse_text_response
from .artifacts import read, save


def historical_question(message):
    """An explicit solver request, never a classifier call on a completion report."""
    match = re.fullmatch(r"\s*HISTORY_QUESTION:\s*(\S[\s\S]*)", message)
    return match.group(1).strip() if match else None


def _refs(value):
    return [] if not value or value == "none" else [s.strip() for s in value.split(",") if s.strip()]


def answer_quote_supported(quote, answer):
    """Check that a short review quote is grounded in the injected answer.

    Models sometimes join two adjacent answer points with punctuation or omit
    a small subject prefix. Accept literal fragments with whitespace changes;
    a quote that only appears in the historical source still fails.
    """
    if not isinstance(quote, str) or not quote.strip() or not isinstance(answer, str):
        return False
    if quote in answer:
        return True
    lines = [re.sub(r"^[-*]\s*", "", line).strip()
             for line in answer.splitlines() if line.strip()]
    parts = [part.strip(" \t\r\n;；。.!！?") for part in re.split(r"[;；。.!！?]", quote)
             if part.strip(" \t\r\n;；。.!！?")]
    if not parts or not lines:
        return False
    for part in parts:
        compact_part = re.sub(r"\s+", "", part)
        if not compact_part:
            continue
        if any(compact_part in re.sub(r"\s+", "", line) for line in lines):
            continue
        return False
    return True


def _normalise_target(row, aliases, events, known):
    """Validate one immutable historical target before task-specific labels."""
    target_id = row.get("id")
    refs = [aliases.get(ref, ref) for ref in _refs(row.get("sources"))]
    supersedes = _refs(row.get("supersedes"))
    if (not isinstance(target_id, str) or not target_id.strip() or target_id in known
            or not refs or set(refs) - events.keys()
            or any(not isinstance(row.get(key), str) or not row[key].strip()
                   for key in ("statement", "scope", "behavior"))
            or set(supersedes) - known.keys()):
        raise ValueError("Invalid historical target or public source reference")
    order = max(events[ref]["order"] for ref in refs)
    if any(order <= known[prior]["order"] for prior in supersedes):
        raise ValueError("Historical replacement must cite a later public event")
    return {"id": target_id, "statement": row["statement"], "scope": row["scope"],
            "behavior": row["behavior"], "sources": refs, "supersedes": supersedes,
            "order": order}


def freeze_targets(rows, history):
    """Freeze historical truth without deciding repository availability or QA coverage."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("No historical targets")
    events = {event["id"]: event for event in history.get("events", [])}
    aliases = history.get("source_aliases", {})
    known, targets = {}, []
    for row in rows:
        target = _normalise_target(row, aliases, events, known)
        known[target["id"]] = target
        targets.append(target)
    return {"targets": targets, "source_aliases": aliases,
            "cutoff_event_id": history.get("cutoff_event_id"),
            "events": history.get("events", [])}


def validate_contract_targets(spec_rows, frozen_targets):
    """Ensure private history-contract cannot rewrite the already frozen targets."""
    expected = {row["id"]: row for row in frozen_targets.get("targets", [])}
    seen = set()
    for row in spec_rows:
        target_id = row.get("id")
        if target_id not in expected or target_id in seen:
            raise ValueError("History contract changed frozen targets")
        target = expected[target_id]
        for key in ("statement", "scope", "behavior", "sources", "supersedes"):
            actual = row.get(key)
            if key in {"sources", "supersedes"}:
                actual = _refs(actual)
                if key == "sources":
                    aliases = frozen_targets.get("source_aliases", {})
                    actual = [aliases.get(ref, ref) for ref in actual]
            if actual != target[key]:
                raise ValueError("History contract changed frozen target " + target_id)
        seen.add(target_id)
    if seen != set(expected):
        raise ValueError("History contract omitted frozen target")


def write_contract_from_targets(spec, frozen_targets, review=None):
    """Materialize the private contract from immutable H and finite review states."""
    review_rows = {row.get("id"): row for row in (review or {}).get("history_rows", [])}
    lines = []
    for target in frozen_targets.get("targets", []):
        row = review_rows.get(target["id"], {})
        applicable = row.get("applicable")
        active = "yes" if applicable == "yes" else "no" if applicable == "no" else "yes"
        public = row.get("public")
        repository = ("recoverable" if public == "full" else
                      "external" if public in {"partial", "none"} else "uncertain")
        lines.extend([
            "REVIEW %s" % target["id"],
            "statement: %s" % target["statement"],
            "scope: %s" % target["scope"],
            "sources: %s" % ",".join(target["sources"]),
            "supersedes: %s" % (",".join(target["supersedes"]) or "none"),
            "behavior: %s" % target["behavior"],
            "active: %s" % active,
            "repository: %s" % repository,
            "END_REVIEW",
        ])
    path = Path(spec) / "history-contract.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def prepare_history(records, generation_input, *, qa_source_ids=None):
    """Bind QA source events and retain the public cutoff for checking later updates."""
    request = read(generation_input)
    if qa_source_ids is None:
        qa_source_ids = set(request.get("ref_to_source", {}).values())
    else:
        qa_source_ids = set(qa_source_ids)
    refs = set(qa_source_ids)
    payload_scope = request.get("payload", {}).get("scope", {})
    refs.update(r["id"] for r in payload_scope.get("dialogue", []))
    evidence, selected, cited = [], [], []
    for record in records:
        if record["id"] in qa_source_ids or any(
                ref.startswith(record["id"] + "#fragment-") for ref in qa_source_ids):
            cited.append(record["original_id"])
        if record["id"] in refs or any(ref.startswith(record["id"] + "#fragment-") for ref in refs):
            selected.append(record["original_id"])
        evidence.append({"id": record["original_id"], "order": record["order"],
                         "kind": record["kind"], "role": record.get("role"),
                         "text": record.get("text", "")})
    if not selected:
        raise ValueError("No public source records for historical task")
    aliases = {"source%d" % (i + 1): event["id"] for i, event in enumerate(evidence)}
    # Keep the selected source closure small.  Previously every user message
    # in the session was added here, which let task selection drift to an
    # unrelated later topic even when the QA pointed at one external fact.
    initial_ids = set(selected)
    for i, event in enumerate(evidence):
        if event["id"] not in initial_ids:
            continue
        if event.get("role") == "assistant":
            # A selected assistant report may be followed by the user's
            # confirmation in the next visible turn.  Skip tool records, but
            # stop at another assistant message so we do not import a topic
            # from a distant conversation branch.
            for following in evidence[i + 1:]:
                if following.get("role") == "assistant":
                    break
                if following.get("role") == "user":
                    initial_ids.add(following["id"])
                    break
            continue
        if event.get("role") != "user":
            continue
        # Include only the preceding visible reply, which may explain the
        # selected user correction.  The selected usage/reply is already in
        # `selected` when the generation request cited it.
        prior = next((e for e in reversed(evidence[:i]) if e.get("role") == "assistant"
                      and e["kind"] == "message"), None)
        if prior:
            initial_ids.add(prior["id"])
    return {"cutoff_event_id": records[-1]["original_id"], "selected_event_ids": selected,
            "qa_source_ids": cited,
            "source_aliases": aliases,
            "initial_events": [dict(e, source="source%d" % (i + 1))
                               for i, e in enumerate(evidence) if e["id"] in initial_ids],
            "events": evidence}


def freeze_contract(spec, history, oracle, *, require_external=True, targets=None):
    """Validate source closure and scoped updates; semantics remain a validator check."""
    spec = Path(spec)
    path = spec / "history-contract.txt"
    rows = parse_text_response(path.read_text())["reviews"]
    sources = {r["id"]: r for r in history["events"]}
    contracts = []
    known = {}
    for row in rows:
        cid = row.get("id")
        aliases = history.get("source_aliases", {})
        refs = [aliases.get(ref, ref) for ref in _refs(row.get("sources"))]
        replaced = _refs(row.get("supersedes"))
        if (not cid or cid in known or not refs or set(refs) - sources.keys()
                or any(not isinstance(row.get(k), str) or not row[k].strip()
                       for k in ("statement", "scope", "behavior"))
                or set(replaced) - known.keys()
                or row.get("active") not in {"yes", "no"}
                or row.get("repository") not in {"recoverable", "external", "uncertain"}):
            raise ValueError("Invalid historical contract or public source reference")
        order = max(sources[r]["order"] for r in refs)
        if any(order <= known[prior]["order"] for prior in replaced):
            raise ValueError("Historical replacement must cite a later public event")
        contract = {k: row[k] for k in ("id", "statement", "scope", "behavior", "repository")}
        contract.update(sources=refs, supersedes=replaced, order=order, active=row["active"] == "yes")
        known[cid] = contract
        contracts.append(contract)
    if not contracts:
        raise ValueError("No historical contract")
    if targets:
        validate_contract_targets(rows, targets)
    active = [c for c in contracts if c["active"]]
    if require_external and (not any(c["repository"] == "external" for c in active) or any(
            c["repository"] == "uncertain" for c in active)):
        raise ValueError("Require an active external rule and resolve uncertain rules")
    result = {"cutoff_event_id": history["cutoff_event_id"], "contracts": contracts,
              "public_task": (spec / "task.md").read_text() if (spec / "task.md").exists() else "",
              "events": history["events"], "oracle_answer": oracle,
              "reference_information": "oracle plus scoped historical contract",
              "oracle_sufficiency": "not_established_by_reference"}
    save(spec / "history.json", result)
    return result


def historical_context(history):
    """Do not give the reference solver acceptance actions or a future implementation."""
    statements = {row["id"]: row["statement"] for row in history["contracts"]}
    return "\n".join("%s（适用范围：%s；替代记录：%s）" % (
        row["statement"], row["scope"], "；".join(statements[cid] for cid in row["supersedes"]) or "无")
        for row in history["contracts"] if row.get("active", True))


def read_history_review(acceptance, history):
    """History is a projection of the same mandatory acceptance results."""
    states = {"applied", "violated", "not_applicable", "insufficient"}
    rows = []
    for contract in history["contracts"]:
        matches = [r for r in acceptance["rows"] if contract["id"] in r["basis"]]
        status = ("not_applicable" if not contract.get("active", True) else
                  "violated" if any(r["status"] == "failed" for r in matches) else
                  "applied" if matches and all(r["status"] == "passed" for r in matches) else "insufficient")
        evidence = "; ".join(r["id"] + ": " + r["evidence"] for r in matches) or "No applicable check established"
        rows.append({"id": contract["id"], "statement": contract["statement"],
                     "status": status, "evidence": evidence})
    return {"rows": rows, "counts": {s: sum(r["status"] == s for r in rows) for s in sorted(states)}}


def oracle_coverage(path, history):
    """Require an exact answer excerpt for each external rule used in acceptance."""
    if not Path(path).is_file():
        return False
    rows = parse_text_response(Path(path).read_text()).get("reviews", [])
    missing_information = 0
    for rule in history["contracts"]:
        if not rule["active"] or rule["repository"] != "external":
            continue
        matched = [r for r in rows if r.get("id") == rule["id"]]
        if len(matched) != 1:
            return False
        row = matched[0]
        quote = row.get("quote", "")
        if row.get("coverage") == "provided":
            if not quote.strip() or quote not in history.get("public_task", ""):
                return False
            continue
        if row.get("coverage") != "complete" or not quote.strip() \
                or not answer_quote_supported(quote, history["oracle_answer"]):
            return False
        missing_information += 1
    return missing_information > 0


def answer_clarification(message, history, exchanges, config, output):
    from .runtime import ask_model
    from .prompts import CLARIFY
    refs = {source for row in history["contracts"] for source in row["sources"]}
    payload = {"message": message, "exchange": exchanges,
               "supplied_history": {"contracts": [{k: row[k] for k in (
                   "id", "statement", "scope", "sources", "supersedes")} for row in history["contracts"]],
                   "events": [e for e in history["events"] if e["id"] in refs]}}
    prompt = CLARIFY + "\n本次可引用的公开事件来源：" + ",".join(sorted(refs)) + "。\n"
    response = ask_model(prompt, payload, config, output)
    rows = response.get("reviews", [])
    row = rows[0] if len(rows) == 1 else {}
    # A frozen rule is an exact alias for its public sources, not a new source.
    # Keep the model's raw response on disk while resolving citations here.
    aliases = {rule["id"]: rule["sources"] for rule in history["contracts"]}
    sources = list(dict.fromkeys(source for ref in _refs(row.get("sources"))
                                for source in ([ref] if ref in refs else aliases.get(ref, [ref]))))
    status = row.get("status")
    delivered = {source for exchange in exchanges if exchange.get("delivered")
                 for source in exchange.get("sources", [])}
    if (status not in {"answer", "no_question", "unavailable"}
            or row.get("kind") not in {"historical_reask", "same_session_repeat", "update_confirmation", "none"}
            or set(sources) - refs or (status == "answer" and
                (not sources or not row.get("reply") or row["reply"] == "none"))
            or (status != "answer" and row.get("reply") != "none")
            or (row.get("kind") == "same_session_repeat" and not (delivered & set(sources)))
            or (status == "no_question" and row.get("kind") != "none")
            or (status == "answer" and row.get("kind") == "none")):
        raise ValueError("Invalid clarification response or source")
    return dict(row, sources=sources)
