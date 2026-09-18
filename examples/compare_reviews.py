"""Run a redacted review-mode comparison on a frozen private bundle.

The bundle contains private evidence and is intentionally kept outside public
artifacts.  This runner never prints candidate text, provider responses, URLs,
or credentials.  Without ``--allow-network`` it only reports the expected
request plan.
"""

import argparse
import copy
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dialogue_benchmark.llm import ChatClient, review_candidates, stage_error


MODES = ("single", "split")
ALL_MODES = MODES + ("simple",)
MODE_REQUEST_UPPER_BOUNDS = {"single": 1, "split": 2, "simple": 4}
# This comparison reviews existing frozen candidates; generation is counted by
# the separate smoke runner, and no post-review annotation call is enabled.
POST_ANNOTATION_CALLS = 0
_DANGEROUS_CANDIDATE_FIELDS = {
    "review", "question_review", "answer_review", "status", "review_error",
    "missing_review_fields", "review_conflicts", "repair_attempted",
    "atomicity_review", "completeness_review", "evidence_review",
}


def _source_label(value):
    """Keep provenance to a run name; never echo a private absolute path."""
    if not isinstance(value, str) or not value:
        return "unspecified"
    return Path(value).name or "unspecified"


def load_bundle(path):
    """Load and validate the fixed bundle without exposing its text."""
    with Path(path).open(encoding="utf-8") as stream:
        bundle = json.load(stream)
    if not isinstance(bundle, dict) or bundle.get("schema_version") != 1:
        raise ValueError("unsupported comparison bundle")
    cases = bundle.get("cases")
    groups = bundle.get("groups")
    if not isinstance(cases, list) or not 8 <= len(cases) <= 12:
        raise ValueError("comparison bundle must contain 8-12 cases")
    if not isinstance(groups, dict):
        raise ValueError("comparison bundle groups must be an object")
    seen = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str):
            raise ValueError("malformed comparison case")
        case_id = case["id"]
        if case_id in seen:
            raise ValueError("duplicate comparison case")
        seen.add(case_id)
        group_id = case.get("group_id")
        candidate = case.get("candidate")
        group = groups.get(group_id)
        if (not isinstance(group_id, str) or not isinstance(candidate, dict)
                or not isinstance(group, dict)):
            raise ValueError("comparison case has no complete group")
        if candidate.get("id") != case_id:
            raise ValueError("candidate identity mismatch")
        if _DANGEROUS_CANDIDATE_FIELDS & set(candidate):
            raise ValueError("comparison candidate contains prior review state")
        if candidate.get("evidence_group_id") != group_id:
            raise ValueError("candidate group mismatch")
        if case.get("expected_decision") not in {"approve", "reject"}:
            raise ValueError("comparison case needs an expected decision")
        if not isinstance(group.get("facts"), list) or not isinstance(group.get("scope"), dict):
            raise ValueError("comparison group needs facts and scope")
    return bundle


def select_cases(bundle, raw_case_ids=None):
    """Select a validated case subset without changing the frozen source file."""
    if raw_case_ids is None:
        return bundle
    requested = [item.strip() for item in raw_case_ids.split(",")]
    requested = [item for item in requested if item]
    if not requested:
        raise ValueError("--case-ids must contain at least one case ID")
    if len(requested) != len(set(requested)):
        raise ValueError("--case-ids contains duplicate case IDs")
    available = {case["id"] for case in bundle["cases"]}
    unknown = sorted(set(requested) - available)
    if unknown:
        raise ValueError("--case-ids contains unknown case ID(s)")
    selected = [case for case in bundle["cases"] if case["id"] in requested]
    if not selected:
        raise ValueError("--case-ids selected no cases")
    selected_bundle = dict(bundle)
    selected_bundle["cases"] = selected
    return selected_bundle


def _decision(result):
    questions = result.get("questions", []) if isinstance(result, dict) else []
    if any(isinstance(item, dict) and item.get("status") == "approved"
           for item in questions):
        return "approve"
    if any(isinstance(item, dict) and item.get("status") == "needs_review"
           for item in questions):
        return "needs_review"
    if isinstance(result, dict) and result.get("rejected"):
        return "reject"
    return "unknown"


def _usage_summary(receipts):
    request_chars = [item.get("request_chars") for item in receipts
                     if isinstance(item.get("request_chars"), (int, float))]
    elapsed = [item.get("elapsed_seconds") for item in receipts
               if isinstance(item.get("elapsed_seconds"), (int, float))]

    def summary(values):
        if not values:
            return {"count": 0}
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": round(statistics.mean(values), 3),
            "median": round(statistics.median(values), 3),
        }

    statuses = {}
    for item in receipts:
        status = item.get("status", "unknown")
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "calls": len(receipts),
        "statuses": statuses,
        "request_chars": summary(request_chars),
        "elapsed_seconds": summary(elapsed),
    }


def _checkpoint_name(checkpoint_dir, mode, index, case_id):
    safe_id = "".join(char if char.isalnum() or char in "-_" else "_"
                       for char in case_id)
    return checkpoint_dir / ("%s-%04d-%s.json" % (mode, index + 1, safe_id))


def _write_checkpoint(checkpoint_dir, mode, index, case_id, payload):
    path = _checkpoint_name(checkpoint_dir, mode, index, case_id)
    with path.open("x", encoding="utf-8") as stream:
        os.chmod(path, 0o600)
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _run_case(bundle, mode, index, endpoint, model, key_env, timeout):
    """Run one case with an independent client; no repair is ever enabled."""
    case = bundle["cases"][index]
    group = bundle["groups"][case["group_id"]]
    client = None
    started = time.monotonic()
    try:
        client = ChatClient(endpoint, model, key_env=key_env, timeout=timeout)
        result = review_candidates(
            copy.deepcopy(group["scope"]), copy.deepcopy(group["facts"]),
            [copy.deepcopy(case["candidate"])], client,
            qa_mode=group.get("qa_mode", case["candidate"].get("qa_mode", "code")),
            allow_repair=False, review_mode=mode)
        error = None
    except Exception as exc:  # Keep provider diagnostics type-only.
        result = {"questions": [], "rejected": [], "stage_errors": []}
        error = stage_error("comparison", exc)
    receipts = list(getattr(client, "usage", []) or [])
    # ChatClient appends only the post-guard visible message content here;
    # reasoning fields are never retained by the transport or this checkpoint.
    responses = list(getattr(client, "responses", []) or [])
    wall_seconds = round(time.monotonic() - started, 3)
    row = {
        "id": case["id"],
        "expected_decision": case["expected_decision"],
        "expected_basis": case.get("expected_basis", "unspecified"),
        "observed_decision": _decision(result),
        "calls": len(receipts),
        "request_chars": [item.get("request_chars") for item in receipts
                          if isinstance(item.get("request_chars"), (int, float))],
        "receipt_statuses": [item.get("status", "unknown") for item in receipts],
        "wall_seconds": wall_seconds,
        "stage_error_count": len(result.get("stage_errors", [])),
        "error": error,
    }
    checkpoint = {
        "schema_version": 1,
        "mode": mode,
        "case_index": index,
        "id": case["id"],
        "expected_decision": case["expected_decision"],
        "expected_basis": case.get("expected_basis", "unspecified"),
        "result": result,
        "usage": receipts,
        "responses": responses,
        "wall_seconds": wall_seconds,
        "error": error,
    }
    return index, row, receipts, checkpoint


def _run_mode(bundle, mode, endpoint, model, key_env, timeout, checkpoint_dir):
    """Review cases concurrently while keeping each case's stages serial."""
    rows = [None] * len(bundle["cases"])
    all_usage = []
    workers = min(10, len(bundle["cases"]))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_case, bundle, mode, index, endpoint, model, key_env, timeout): index
            for index in range(len(bundle["cases"]))
        }
        for future in as_completed(futures):
            index, row, receipts, checkpoint = future.result()
            _write_checkpoint(checkpoint_dir, mode, index, row["id"], checkpoint)
            rows[index] = row
            for receipt in receipts:
                tagged = dict(receipt)
                tagged.update(mode=mode, case_index=index, case_id=row["id"])
                all_usage.append(tagged)
    summary = _usage_summary(all_usage)
    expected = sum(_case_request_upper_bound(bundle, case, mode)
                   for case in bundle["cases"])
    expected_reject_approve = sum(
        row["expected_decision"] == "reject" and row["observed_decision"] == "approve"
        for row in rows)
    expected_approve_reject = sum(
        row["expected_decision"] == "approve" and row["observed_decision"] == "reject"
        for row in rows)
    summary.update({
        "mode": mode,
        "expected_requests_upper_bound_without_repair": expected,
        "request_count_within_upper_bound": summary["calls"] <= expected,
        "request_count_exceeded_upper_bound": summary["calls"] > expected,
        # These are directional mismatches against static expected labels,
        # not false-negative/false-deletion rates against human gold.
        "static_expected_reject_but_observed_approve": expected_reject_approve,
        "static_expected_approve_but_observed_reject": expected_approve_reject,
    })
    return {"summary": summary, "cases": rows, "all_usage": all_usage}


def _plan(bundle, mode):
    count = len(bundle["cases"])
    modes = MODES if mode == "both" else (mode,)
    return {
        "status": "dry_run_no_model_request",
        "generation_requests": 0,
        "case_count": count,
        "modes": {item: {"allow_repair": False,
                         "expected_requests": sum(
                             _case_request_upper_bound(bundle, case, item)
                             for case in bundle["cases"]),
                         "post_annotation_calls": POST_ANNOTATION_CALLS}
                  for item in modes},
        "post_annotation_enabled": bool(POST_ANNOTATION_CALLS),
        "selection_basis": bundle.get("selection_basis", "unspecified"),
    }


def _case_request_upper_bound(bundle, case, mode):
    """Count the code-only distinctiveness pass without charging general QA."""
    if mode != "simple":
        return MODE_REQUEST_UPPER_BOUNDS[mode]
    group = bundle.get("groups", {}).get(case.get("group_id"), {})
    track = group.get("qa_mode", case.get("candidate", {}).get("qa_mode"))
    return MODE_REQUEST_UPPER_BOUNDS[mode] + (1 if track == "code" else 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--mode", choices=ALL_MODES + ("both",), default="both")
    parser.add_argument("--output", type=Path,
                        help="New private JSON output; required with --allow-network")
    parser.add_argument("--case-ids",
                        help="Comma-separated case IDs to run after full bundle validation")
    parser.add_argument("--endpoint")
    parser.add_argument("--model")
    parser.add_argument("--key-env", default="BENCHMARK_API_KEY")
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--allow-network", action="store_true",
                        help="Explicitly permit provider requests")
    args = parser.parse_args(argv)
    # Validate the complete frozen bundle before applying an optional smoke
    # subset, so unknown IDs or malformed source cases cannot be hidden by the
    # filter.
    bundle = select_cases(load_bundle(args.bundle), args.case_ids)
    if not args.allow_network:
        print(json.dumps(_plan(bundle, args.mode), ensure_ascii=False, sort_keys=True))
        return 0
    if not args.endpoint or not args.model:
        parser.error("--allow-network requires --endpoint and --model")
    if not args.output:
        parser.error("--allow-network requires --output")
    if args.output.exists():
        parser.error("output already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output.parent / (args.output.stem + ".checkpoints")
    if checkpoint_dir.exists():
        parser.error("checkpoint directory already exists")
    checkpoint_dir.mkdir(mode=0o700)
    os.chmod(checkpoint_dir, 0o700)
    modes = MODES if args.mode == "both" else (args.mode,)
    results = {
        mode: _run_mode(bundle, mode, args.endpoint, args.model, args.key_env,
                        args.timeout, checkpoint_dir)
        for mode in modes
    }
    payload = {
        "schema_version": 1,
        "source_run": _source_label(bundle.get("source_run")),
        "case_count": len(bundle["cases"]),
        "allow_repair": False,
        "modes": results,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        os.chmod(args.output, 0o600)
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "status": "completed",
        "output": args.output.name,
        "case_count": len(bundle["cases"]),
        "modes": {mode: results[mode]["summary"] for mode in modes},
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
