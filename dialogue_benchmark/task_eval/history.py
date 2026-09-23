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


def prepare_history(records, generation_input):
    """Bind QA source events and retain the public cutoff for checking later updates."""
    request = read(generation_input)
    refs = set(request.get("ref_to_source", {}).values())
    payload_scope = request.get("payload", {}).get("scope", {})
    refs.update(r["id"] for r in payload_scope.get("dialogue", []))
    evidence, selected = [], []
    for record in records:
        if record["id"] in refs or any(ref.startswith(record["id"] + "#fragment-") for ref in refs):
            selected.append(record["original_id"])
        evidence.append({"id": record["original_id"], "order": record["order"],
                         "kind": record["kind"], "role": record.get("role"),
                         "text": record.get("text", "")})
    if not selected:
        raise ValueError("No public source records for historical task")
    return {"cutoff_event_id": records[-1]["original_id"], "selected_event_ids": selected,
            "events": evidence}


def freeze_contract(spec, history, oracle):
    """Validate source closure and scoped updates; semantics remain a validator check."""
    spec = Path(spec)
    path = spec / "history-contract.txt"
    rows = parse_text_response(path.read_text())["reviews"]
    sources = {r["id"]: r for r in history["events"]}
    contracts = []
    known = {}
    for row in rows:
        cid = row.get("id")
        refs = _refs(row.get("sources"))
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
    active = [c for c in contracts if c["active"]]
    if not active or any(c["repository"] != "external" for c in active):
        raise ValueError("Main historical tasks require active external rules; uncertain rules need review")
    result = {"cutoff_event_id": history["cutoff_event_id"], "contracts": contracts,
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
    """Require an exact answer excerpt for each active rule, reviewed before trials."""
    if not Path(path).is_file():
        return False
    rows = parse_text_response(Path(path).read_text()).get("reviews", [])
    for rule in history["contracts"]:
        if not rule["active"]:
            continue
        matched = [r for r in rows if r.get("id") == rule["id"]]
        if len(matched) != 1:
            return False
        row = matched[0]
        quote = row.get("quote", "")
        if row.get("coverage") != "complete" or not quote.strip() or quote not in history["oracle_answer"]:
            return False
    return True


def answer_clarification(message, history, exchanges, config, output):
    from .runtime import ask_model
    from .prompts import CLARIFY
    refs = {source for row in history["contracts"] for source in row["sources"]}
    payload = {"message": message, "exchange": exchanges,
               "supplied_history": {"contracts": [{k: row[k] for k in (
                   "id", "statement", "scope", "sources", "supersedes")} for row in history["contracts"]],
                   "events": [e for e in history["events"] if e["id"] in refs]}}
    response = ask_model(CLARIFY, payload, config, output)
    rows = response.get("reviews", [])
    row = rows[0] if len(rows) == 1 else {}
    sources = _refs(row.get("sources"))
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
