"""Command line orchestration for private dialogue benchmark runs."""

import argparse
from copy import deepcopy
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .chunking import split_scope
from .fact_index import (build_evidence_groups, build_evidence_index,
                         merge_scopes,
                         candidate_review_projection, coverage_report,
                         expand_evidence_group_once,
                         static_candidate_labels, static_evidence_check)
from .general import build_general_scope, identify_stages
from .graph import build_graph, graph_at, query_scope_adaptive
from .llm import (ChatClient, extract_facts, generate_from_facts,
                  repair_simple_validation_rejection, review_candidates,
                  stage_error)
from .normalize import load_dialogue
from .protocol import MISSING_KINDS
from .quality import CODE_QA_TYPES, GENERAL_QA_TYPES
from .security import credential_detected
from .storage import save_projection
from .subgraph import adaptive_subgraphs
from .selection import (build_audit, select_approved, deduplicate,
                        deduplicate_reviewed, replenish, globally_blocked,
                        review_duplicate_clusters, apply_duplicate_decisions)


DEFAULT_GENERAL_TYPES = tuple(sorted(GENERAL_QA_TYPES))
DEFAULT_CODE_TYPES = tuple(sorted(CODE_QA_TYPES))
_USER_PATH = re.compile(r"(?<![A-Za-z0-9:/])(?:/(?:Users|home)/[^/\s`'\"<>，。]+|[A-Za-z]:[\\/]Users[\\/][^\\/\s`'\"<>，。]+)(?=[\\/\s`'\"<>，。]|$)")
_INTERNAL_PUBLIC_ID = re.compile(
    r"(?<![A-Za-z0-9_])(?:e|f)\d+(?![A-Za-z0-9_.(])|"
    r"(?<![A-Za-z0-9_])(?:code|general)_s\d+_c\d+_f\d+(?![A-Za-z0-9_])|"
    r"(?<![A-Za-z0-9_])(?:stage|chunk|scope)[-_]?\d+(?![A-Za-z0-9_])",
    re.I,
)
CHUNK_EVIDENCE_FIELDS = (
    "cutoff", "dialogue", "events", "versions", "edges",
    "historical_edges", "stages",
)


def save(directory, name, data):
    path = directory / name
    if name in {"scope.json", "scopes.json", "general-scope.json", "evidence-groups.json"}:
        if path.exists():
            raise FileExistsError(path)
        save_projection(path, data)
        return
    with path.open("x", encoding="utf-8") as output:
        os.chmod(path, 0o600)
        json.dump(data, output, ensure_ascii=False, indent=2)
        output.write("\n")


def _chunk_identity(track, chunk):
    evidence = {key: chunk.get(key) for key in CHUNK_EVIDENCE_FIELDS
                if key in chunk}
    serialized = json.dumps(
        evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return track + ":" + hashlib.sha256(serialized.encode()).hexdigest()


def _csv_types(value, allowed, option):
    if value is None:
        return None
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("%s must contain at least one type" % option)
    unknown = [item for item in values if item not in allowed]
    if unknown:
        raise ValueError("Invalid %s: %s" % (option, ", ".join(unknown)))
    if len(set(values)) != len(values):
        raise ValueError("Duplicate type in %s" % option)
    return values


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Build local dialogue evidence and optional QA candidates")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True,
                        help="New private run directory; must not exist")
    parser.add_argument("--cutoff", type=int,
                        help="Normalized event order, not raw line number")
    parser.add_argument("--seed", help="Observed relative file path or path::symbol")
    parser.add_argument("--source-event", action="append", default=[],
                        help="Public original event ID to seed either track; repeatable")
    parser.add_argument("--source-object", action="append", default=[],
                        help="Exact public file or file::symbol mention to seed either track")
    parser.add_argument("--initial-hops", type=int, default=1)
    parser.add_argument("--max-hops", type=int, default=4)
    parser.add_argument("--max-context-chars", type=int, default=120000,
                        help="Maximum candidate subgraph size before bounded evidence chunking")
    parser.add_argument("--model-request-chars", type=int, default=32000,
                        help="Maximum serialized request size for a model stage")
    parser.add_argument("--qa-mode", choices=("general", "code", "both"), default="code",
                        help="QA tracks to generate (default: code)")
    parser.add_argument("--general-types",
                        help="Comma-separated memory-purpose types for the general track")
    parser.add_argument("--code-types", help="Comma-separated code QA categories")
    parser.add_argument("--general-count", type=int,
                        help="Approved, safe, unique general QA target and publication cap")
    parser.add_argument("--code-count", type=int,
                        help="Approved, safe, unique code QA target and publication cap")
    parser.add_argument("--general-group-budget", type=int,
                        help="Maximum general evidence groups to explore")
    parser.add_argument("--code-group-budget", type=int,
                        help="Maximum code evidence groups to explore")
    parser.add_argument("--questions-per-group", type=int, default=1,
                        help="Legacy generation cap; simple mode emits at most one question per eligible type")
    parser.add_argument("--max-questions", type=int,
                        help="Deprecated alias for --code-count (global, not per chunk)")
    parser.add_argument("--chunk-chars", type=int, default=24000,
                        help="Per-scope model context budget")
    parser.add_argument("--chunk-overlap", type=int, default=2,
                        help="Overlapping dialogue records between chunks")
    parser.add_argument("--parallel-workers", type=int, default=6,
                        help="Shared concurrent model tasks across both tracks")
    parser.add_argument("--expansion-budget", type=int, default=3,
                        help="Maximum directed evidence expansions per fixed target type")
    parser.add_argument("--review-mode", choices=("simple", "split", "single"),
                        default="simple",
                        help="Simple completeness/evidence review (default), or legacy comparison modes")
    parser.add_argument("--adaptive-subgraphs", action="store_true",
                        help="Use event-linked adaptive candidate subgraphs")
    parser.add_argument("--allow-network", action="store_true",
                        help="Consent to transmit selected private evidence")
    parser.add_argument("--endpoint", help="Full HTTPS chat/completions endpoint")
    parser.add_argument("--model")
    parser.add_argument("--key-env", default="BENCHMARK_API_KEY")
    parser.add_argument("--reuse-facts", type=Path,
                        help="Reuse saved facts/errors from an identical normalized input and chunk layout")
    return parser


def _parse_options(args, parser):
    try:
        general_types = _csv_types(args.general_types, GENERAL_QA_TYPES, "--general-types")
        code_types = _csv_types(args.code_types, CODE_QA_TYPES, "--code-types")
    except ValueError as error:
        parser.error(str(error))
    for name in ("general_count", "code_count", "general_group_budget",
                 "code_group_budget", "questions_per_group", "max_questions"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error("--%s must be positive" % name.replace("_", "-"))
    if args.max_questions is not None and args.code_count is not None:
        parser.error("--max-questions conflicts with --code-count")
    if args.qa_mode == "code" and (args.general_types is not None
                                    or args.general_count is not None
                                    or args.general_group_budget is not None):
        parser.error("general options require --qa-mode general or both")
    if args.qa_mode == "general" and (args.code_types is not None
                                       or args.code_count is not None
                                       or args.code_group_budget is not None
                                       or args.max_questions is not None
                                       or args.seed is not None
                                       or args.adaptive_subgraphs):
        parser.error("code options require --qa-mode code or both")
    if (args.endpoint or args.model) and not args.allow_network:
        parser.error("Network access requires --allow-network")
    if args.allow_network and not (args.endpoint and args.model):
        parser.error("Network mode requires --endpoint and --model")
    if (args.initial_hops < 0 or args.max_hops < args.initial_hops
            or args.max_context_chars <= 0 or args.model_request_chars <= 0
            or args.chunk_chars <= 0 or args.chunk_overlap < 0
            or args.parallel_workers <= 0 or args.expansion_budget < 0):
        parser.error("Invalid hop, chunk, worker, or positive budget")

    enabled_general = args.qa_mode in {"general", "both"}
    enabled_code = args.qa_mode in {"code", "both"}
    general_count = (args.general_count if args.general_count is not None else 10)
    code_count = (args.code_count if args.code_count is not None else
                  args.max_questions if args.max_questions is not None else 10)
    return {
        "enabled_general": enabled_general,
        "enabled_code": enabled_code,
        "general_types": tuple(general_types or DEFAULT_GENERAL_TYPES) if enabled_general else (),
        "code_types": tuple(code_types or DEFAULT_CODE_TYPES) if enabled_code else (),
        "general_count": general_count if enabled_general else 0,
        "code_count": code_count if enabled_code else 0,
        "general_group_budget": (
            args.general_group_budget if args.general_group_budget is not None
            else max(10, general_count * 4)) if enabled_general else 0,
        "code_group_budget": (
            args.code_group_budget if args.code_group_budget is not None
            else max(10, code_count * 4)) if enabled_code else 0,
        "questions_per_group": min(3, args.questions_per_group),
    }


def _prefix_facts(result, prefix, track):
    for fact in result.get("facts", []):
        fact["id"] = prefix + fact["id"]
        fact.setdefault("qa_mode", track)


def _prefix_questions(result, prefix):
    for question in result.get("questions", []):
        question["model_id"] = question["id"]
        question["id"] = prefix + question["id"]
        question["candidate_id"] = question["id"]
        if isinstance(question.get("review"), dict):
            question["review"]["id"] = question["id"]
    for rejection in result.get("rejected", []):
        question = rejection.get("question") if isinstance(rejection, dict) else None
        if isinstance(question, dict) and isinstance(question.get("id"), str):
            question["id"] = prefix + question["id"]
    return result


def _checkpoint(checkpoint_dir, track, phase, index):
    if checkpoint_dir is None:
        return None

    def write(name, data):
        stage = Path(name).stem
        save(checkpoint_dir, "%s-%s-%04d-%s.json" %
             (track, phase, index, stage), data)
    return write


def _run_fact_tasks(tasks, endpoint, model, key_env, workers, checkpoint_dir=None, reuse_dir=None):
    """Extract every chunk's facts through one bounded shared executor."""
    if not tasks:
        return {"facts": [], "questions": [], "rejected": [], "usage": [],
                "stage_errors": [], "stage_status": [],
                "scopes": {"general": [], "code": []}}

    def run(item):
        index, track, scope = item
        client = None
        try:
            saved = (Path(reuse_dir) / "stages" / ("%s-chunk-%04d-facts.json" % (track, index))) if reuse_dir else None
            error_path = saved.with_name(saved.name.replace("facts.json", "facts-error.json")) if saved else None
            write = _checkpoint(checkpoint_dir, track, "chunk", index)
            if saved is not None and saved.exists():
                facts = json.loads(saved.read_text())
                result = {"facts": facts, "stage_status": {"facts": "completed", "reused": True}}
                if write:
                    write("facts.json", facts)
            elif error_path is not None and error_path.exists():
                error = json.loads(error_path.read_text())
                result = {"facts": [], "stage_errors": [error],
                          "stage_status": {"facts": "failed", "reused": True}}
                if write:
                    write("facts-error.json", error)
            elif reuse_dir:
                raise ValueError("Missing saved fact-stage result")
            else:
                client = ChatClient(endpoint, model, key_env)
                result = extract_facts(
                    scope, client, qa_mode=track,
                    checkpoint=_checkpoint(checkpoint_dir, track, "chunk", index))
            prefix = "%s_s%d_c%d_" % (track, scope.get("scope_index", 0),
                                        scope.get("chunk_index", index))
            _prefix_facts(result, prefix, track)
            return index, track, scope, result, client.usage if client else []
        except Exception as error:
            diagnostic = stage_error("facts", error)
            checkpoint = _checkpoint(checkpoint_dir, track, "chunk", index)
            if checkpoint:
                checkpoint("facts-error.json", diagnostic)
            return index, track, scope, {
                "facts": [], "questions": [], "rejected": [],
                "stage_errors": [diagnostic],
                "stage_status": {"facts": "failed"},
            }, client.usage if client is not None else []

    results = []
    with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
        for start in range(0, len(tasks), workers):
            batch = [future.result() for future in as_completed(
                [pool.submit(run, task) for task in tasks[start:start + workers]])]
            results.extend(batch)
            if globally_blocked([error for row in batch for error in row[3].get("stage_errors", [])]):
                for index, track, scope in tasks[start + workers:]:
                    results.append((index, track, scope, {
                        "facts": [], "stage_errors": [{"stage": "facts", "error_code": "global_blocker"}],
                        "stage_status": {"facts": "not_submitted"}}, []))
                break
    merged = {"facts": [], "questions": [], "rejected": [], "usage": [],
              "stage_errors": [], "stage_status": [],
              "scopes": {"general": [], "code": []}}
    facts_by_key = {}
    for index, track, scope, result, usage in sorted(results, key=lambda item: item[0]):
        merged["usage"].extend(usage)
        merged["stage_errors"].extend(dict(error, track=track, task_index=index)
                                      for error in result.get("stage_errors", []))
        merged["stage_status"].append({"track": track, "phase": "facts",
                                        "task_index": index,
                                        **result.get("stage_status", {})})
        scope_for_merge = scope
        if result.get("stage_status", {}).get("facts") != "completed":
            # A failed fact task means the selected range is incomplete for
            # explicit full-range requests; preserve the scope for
            # diagnostics but make that incompleteness explicit.
            scope_for_merge = dict(scope)
            scope_for_merge["facts_failed"] = True
        merged["scopes"][track].append(scope_for_merge)
        for fact in result.get("facts", []):
            fact_mode = fact.get("qa_mode", track)
            key = (fact_mode, fact.get("statement", "").strip(),
                   tuple(sorted(fact.get("sources", []))))
            if key not in facts_by_key:
                facts_by_key[key] = fact["id"]
                merged["facts"].append(fact)
        merged["rejected"].extend(
            dict(item, track=track, stage="facts_validation") if isinstance(item, dict) else
            {"reason": "malformed_rejection", "track": track}
            for item in result.get("rejected", []))
    return merged


def _static_generation_rejection(group, target_type, check):
    return {
        "all_candidates": [], "questions": [], "raw_generated": 0,
        "facts": group.get("facts", []),
        "rejected": [{
            "reason": "type_evidence_static_insufficient",
            "static_reason": check.get("reason"),
            "target_type": target_type,
            "stage": "static_pre_generation",
            "static_type_evidence": check,
        }],
        "stage_errors": [], "stage_status": {"qa": "skipped"},
    }


def generate_simple_target(group, evidence_index, target_type, client,
                           qa_mode, candidate_prefix=None, checkpoint=None,
                           expansion_budget=3):
    """Generate one fixed target with a bounded directed-expansion loop."""
    if qa_mode not in {"general", "code"}:
        raise ValueError("qa_mode must be general or code")
    if not isinstance(expansion_budget, int) or expansion_budget < 0:
        raise ValueError("expansion_budget must be a non-negative integer")

    def phase_checkpoint(phase):
        if checkpoint is None:
            return None
        return lambda name, data: checkpoint(phase, name, data)

    def bind_target(active_group):
        scope = dict(active_group["scope"])
        scope["evidence_group"] = dict(
            scope.get("evidence_group", {}), target_type=target_type)
        return dict(active_group, scope=scope)

    generation_attempts = 0
    generation_requests = 0
    repair_context = None
    expansion_rounds = 0
    expansion_audits = []
    expansion_stop_reason = None
    static_checks = []

    def generate(active_group, phase):
        nonlocal generation_attempts, generation_requests, repair_context
        generation_attempts += 1
        current = generate_from_facts(
            active_group["scope"], active_group["facts"], client,
            max_questions=1, qa_mode=qa_mode, allowed_types=(target_type,),
            target_type=target_type, generation_mode="simple",
            checkpoint=phase_checkpoint(phase),
            candidate_prefix=candidate_prefix)
        generation_requests += int(current.get("generation_request_count", 0) or 0)
        context = current.pop("_repair_context", None)
        if context is not None:
            repair_context = context
        return current

    def expand(active_group, missing_kind, missing_object=None,
               static_direction=False):
        scope = active_group["scope"]
        expanded, audit = expand_evidence_group_once(
            active_group, evidence_index, missing_kind, missing_object,
            target_chars=min(16000, scope.get("model_request_chars", 16000)),
            max_chars=scope.get("model_request_chars", 32000),
            static_direction=static_direction)
        return (bind_target(expanded) if expanded is not None else None), audit

    def merge_generated(total, current):
        if total is None:
            total = dict(current)
            for key in ("all_candidates", "rejected", "stage_errors"):
                total[key] = list(current.get(key, []))
            return total
        total["all_candidates"].extend(current.get("all_candidates", []))
        total["rejected"].extend(current.get("rejected", []))
        total["stage_errors"].extend(current.get("stage_errors", []))
        total["questions"] = list(current.get("questions", []))
        total["raw_generated"] = (int(total.get("raw_generated", 0) or 0)
                                  + int(current.get("raw_generated", 0) or 0))
        total["stage_status"] = dict(total.get("stage_status", {}),
                                     **current.get("stage_status", {}))
        if "focus" in current:
            total["focus"] = deepcopy(current["focus"])
        else:
            total.pop("focus", None)
        for key in ("missing_kind", "missing_object"):
            if key in current:
                total[key] = current[key]
            else:
                total.pop(key, None)
        return total

    active_group = bind_target(group)
    static_precheck = static_evidence_check(active_group, evidence_index, target_type)
    if static_precheck is not None:
        static_checks.append(static_precheck)
    expanded_static_precheck = None
    generated = None
    missing_by_reason = {
        "correction_missing_old_or_new": "earlier_state",
    }

    def record_expansion(missing_kind, missing_object, static_direction=False):
        nonlocal active_group, expansion_rounds, expanded_static_precheck
        if expansion_rounds >= expansion_budget:
            return False, "expansion_budget_exhausted"
        expanded, audit = expand(
            active_group, missing_kind, missing_object,
            static_direction=static_direction)
        audit = dict(audit, attempt=len(expansion_audits) + 1,
                     expansion_round=expansion_rounds + 1,
                     static_direction=bool(static_direction))
        expansion_audits.append(audit)
        if expanded is None:
            return False, audit.get("reason", "expansion_failed")
        active_group = expanded
        expansion_rounds += 1
        expanded_static_precheck = static_evidence_check(
            active_group, evidence_index, target_type)
        static_checks.append(expanded_static_precheck)
        if checkpoint is not None:
            checkpoint("expanded-%d" % expansion_rounds, "evidence.json", {
                "target_type": target_type,
                "missing_kind": missing_kind,
                "missing_object": missing_object,
                "scope": active_group["scope"],
                "facts": active_group["facts"],
                "audit": audit,
                "static_precheck": expanded_static_precheck,
            })
        return True, None

    def satisfy_static(check):
        nonlocal expansion_stop_reason
        while check is not None and check.get("status") == "insufficient":
            missing_kind = missing_by_reason.get(check.get("reason"))
            if missing_kind is None:
                expansion_stop_reason = "static_type_evidence_insufficient"
                return False, check
            ok, reason = record_expansion(
                missing_kind, static_direction=True)
            if not ok:
                expansion_stop_reason = reason
                return False, check
            check = expanded_static_precheck
        return True, check

    static_ok, effective_static = satisfy_static(static_precheck)
    if not static_ok:
        generated = _static_generation_rejection(
            active_group, target_type, effective_static)
    else:
        while True:
            phase = ("initial" if expansion_rounds == 0
                     else "expanded-%d" % expansion_rounds)
            current = generate(active_group, phase)
            generated = merge_generated(generated, current)
            if current.get("questions"):
                expansion_stop_reason = "candidate_generated"
                break
            if current.get("stage_status", {}).get("qa") != "completed":
                expansion_stop_reason = "generation_failed"
                break
            missing_kind = current.get("missing_kind")
            missing_object = current.get("missing_object")
            if not (missing_kind in MISSING_KINDS
                    and isinstance(missing_object, str) and missing_object):
                expansion_stop_reason = "no_qa_without_missing_evidence"
                break
            ok, reason = record_expansion(missing_kind, missing_object)
            if not ok:
                expansion_stop_reason = reason
                break
            if qa_mode == "code":
                static_ok, effective_static = satisfy_static(
                    expanded_static_precheck)
                if not static_ok:
                    break
        if generated is None:
            generated = {"questions": [], "all_candidates": [],
                         "rejected": [], "stage_errors": [],
                         "stage_status": {"qa": "skipped"},
                         "raw_generated": 0}
    generated["facts"] = active_group["facts"]
    return {
        "generated": generated, "active_group": active_group,
        "static_precheck": static_precheck,
        "expanded_static_precheck": expanded_static_precheck,
        "static_checks": static_checks,
        "expansion_audits": expansion_audits,
        "expansion_rounds": expansion_rounds,
        "expansion_stop_reason": expansion_stop_reason,
        "generation_attempt_count": generation_attempts,
        "generation_request_count": generation_requests,
        "repair_context": repair_context,
    }


def _run_qa_tasks(tasks, endpoint, model, key_env, workers, checkpoint_dir=None,
                  deduplicate_results=True, review_mode="single",
                  evidence_indexes=None, expansion_budget=3):
    """Generate bounded candidates per group, reviewing each independently."""
    if not tasks:
        return {"questions": [], "rejected": [], "usage": [],
                "stage_errors": [], "stage_status": [],
                "question_stats": _empty_question_stats()}

    def run(item):
        index, track, group = item
        client = None
        try:
            client = ChatClient(endpoint, model, key_env)
            if review_mode == "simple":
                evidence_index = (evidence_indexes or {}).get(track)
                if evidence_index is None:
                    raise ValueError("simple review requires a reusable evidence index")
                combined = {
                    "all_candidates": [], "revisions": [], "questions": [],
                    "raw_generated": 0, "generated_unique": 0,
                    "rejected": [], "stage_errors": [], "stage_status": {},
                    "expansion_audits": [], "type_attempts": [],
                    "candidate_review_guards": [],
                }
                seen_questions = set()
                target_types = tuple(
                    group["eligible_types"] if "eligible_types" in group
                    else group.get("allowed_types", ()))
                for type_index, target_type in enumerate(target_types, 1):
                    prefix = "%s_g%d_t%d_" % (track, index, type_index)
                    attempt_scope = dict(group["scope"])
                    evidence_group = dict(attempt_scope.get("evidence_group", {}),
                                          target_type=target_type)
                    attempt_scope["evidence_group"] = evidence_group
                    attempt_group = dict(group, scope=attempt_scope)
                    base_checkpoint = _checkpoint(
                        checkpoint_dir, track, "group-" + target_type, index)

                    def scoped_checkpoint(label):
                        if base_checkpoint is None:
                            return None
                        return lambda name, data: base_checkpoint(label + "-" + name, data)

                    def generation_checkpoint(phase, name, data):
                        if base_checkpoint is not None:
                            base_checkpoint(phase + "-" + name, data)

                    generation = generate_simple_target(
                        attempt_group, evidence_index, target_type, client, track,
                        candidate_prefix=prefix,
                        checkpoint=(generation_checkpoint
                                    if base_checkpoint is not None else None),
                        expansion_budget=expansion_budget)
                    generated = generation["generated"]
                    active_group = generation["active_group"]
                    static_precheck = generation["static_precheck"]
                    expanded_static_precheck = generation[
                        "expanded_static_precheck"]
                    expansion_audits = generation["expansion_audits"]
                    combined["expansion_audits"].extend(expansion_audits)
                    missing_kind = generated.get("missing_kind")
                    missing_object = generated.get("missing_object")

                    type_questions = list(generated.get("questions", []))
                    for question in generated.get("all_candidates", []):
                        if isinstance(question, dict):
                            question.update(static_candidate_labels(
                                active_group, question, evidence_index, target_type))
                            question.setdefault("evidence_group_id", group.get("id"))
                    for question in type_questions:
                        question.update(static_candidate_labels(
                            active_group, question, evidence_index, target_type))
                        question["evidence_group_id"] = group.get(
                            "id", "%s-group-%d" % (track, index))
                    static_post_rejected = []
                    if type_questions:
                        checked_questions = []
                        checks_by_id = {}
                        for question in type_questions:
                            check = static_evidence_check(
                                active_group, evidence_index, target_type,
                                candidate=question)
                            question["static_type_evidence"] = check
                            checks_by_id[question.get("id")] = check
                            if check.get("status") == "insufficient":
                                static_post_rejected.append({
                                    "question": dict(question, status="rejected"),
                                    "reason": "type_evidence_static_insufficient",
                                    "static_reason": check.get("reason"),
                                    "failed_checks": ["type_evidence_sufficient"],
                                    "static_type_evidence": check,
                                })
                            else:
                                checked_questions.append(question)
                        type_questions = checked_questions
                        for question in generated.get("all_candidates", []):
                            check = checks_by_id.get(question.get("id"))
                            if check is not None:
                                question["static_type_evidence"] = check
                    reviewed_questions = type_questions
                    validation_rejected = list(generated.get("rejected", []))
                    type_rejected = [dict(item, stage="qa_validation")
                                     for item in validation_rejected]
                    type_rejected.extend(
                        dict(item, stage="static_post_generation")
                        for item in static_post_rejected)
                    type_revisions = []
                    repair_state = {"remaining": 1}
                    generation_context = generation.get("repair_context")
                    def resolve_review_scope(review_candidate):
                        projected_group, guard_audit = candidate_review_projection(
                            active_group, evidence_index, review_candidate)
                        guard_audit.update(
                            candidate_id=review_candidate.get("id"),
                            target_type=target_type)
                        return ((projected_group or active_group)["scope"],
                                guard_audit)
                    already_reviewed = False
                    if not type_questions:
                        repaired = repair_simple_validation_rejection(
                            active_group["scope"], active_group["facts"],
                            validation_rejected, client, qa_mode=track,
                            checkpoint=scoped_checkpoint("validation-repair"),
                            generation_context=generation_context,
                            repair_state=repair_state,
                            review_scope_resolver=resolve_review_scope)
                        if repaired:
                            revision, revised = repaired
                            type_revisions.append(revision)
                            if revised is not None:
                                type_rejected = [dict(
                                    item, stage="static_post_generation")
                                    for item in static_post_rejected]
                                reviewed_questions = list(
                                    revised.get("questions", []))
                                type_questions = list(reviewed_questions)
                                type_rejected.extend(
                                    dict(item, stage="semantic_review")
                                    for item in revised.get("rejected", []))
                                combined["stage_errors"].extend(
                                    revised.get("stage_errors", []))
                                combined["stage_status"].update(
                                    revised.get("stage_status", {}))
                                combined["candidate_review_guards"].extend(
                                    revised.get("candidate_review_guards", []))
                                already_reviewed = True
                            elif revision.get("error"):
                                combined["stage_errors"].append(
                                    revision["error"])
                    if (generated.get("stage_status", {}).get("qa") == "completed"
                            and type_questions and not already_reviewed):
                        reviewed = review_candidates(
                            active_group["scope"], active_group["facts"],
                            type_questions, client, qa_mode=track,
                            checkpoint=scoped_checkpoint("review"),
                            review_mode="simple",
                            generation_context=generation_context,
                            repair_state=repair_state,
                            review_scope_resolver=resolve_review_scope)
                        reviewed_questions = reviewed.get("questions", type_questions)
                        type_rejected.extend(
                            dict(item, stage="semantic_review")
                            for item in reviewed.get("rejected", []))
                        combined["stage_errors"].extend(
                            reviewed.get("stage_errors", []))
                        combined["stage_status"].update(
                            reviewed.get("stage_status", {}))
                        combined["candidate_review_guards"].extend(
                            reviewed.get("candidate_review_guards", []))
                        type_revisions.extend(reviewed.get("revisions", []))
                    for question in reviewed_questions:
                        if isinstance(question, dict):
                            question.update(static_candidate_labels(
                                active_group, question, evidence_index, target_type))
                    for revision in type_revisions:
                        after = revision.get("after") if isinstance(revision, dict) else None
                        if isinstance(after, dict):
                            after.update(static_candidate_labels(
                                active_group, after, evidence_index, target_type))
                    if reviewed_questions:
                        final_questions = []
                        for question in reviewed_questions:
                            check = static_evidence_check(
                                active_group, evidence_index, target_type,
                                candidate=question)
                            question["static_type_evidence"] = check
                            if check.get("status") == "insufficient":
                                type_rejected.append({
                                    "question": dict(question, status="rejected"),
                                    "reason": "type_evidence_static_insufficient",
                                    "static_reason": check.get("reason"),
                                    "failed_checks": ["type_evidence_sufficient"],
                                    "static_type_evidence": check,
                                    "stage": "static_post_review",
                                })
                            else:
                                final_questions.append(question)
                        reviewed_questions = final_questions
                    combined["all_candidates"].extend(
                        generated.get("all_candidates", type_questions))
                    combined["questions"].extend(reviewed_questions)
                    combined["rejected"].extend(type_rejected)
                    combined["revisions"].extend(type_revisions)
                    combined["stage_errors"].extend(generated.get("stage_errors", []))
                    combined["stage_status"].update(generated.get("stage_status", {}))
                    raw_count = int(generated.get("raw_generated", len(type_questions)) or 0)
                    combined["raw_generated"] += raw_count
                    for question in type_questions:
                        text = question.get("question") if isinstance(question, dict) else None
                        if isinstance(text, str):
                            seen_questions.add((track, text.strip().casefold()))
                    combined["type_attempts"].append({
                        "target_type": target_type,
                        "generation_attempt_count": generation[
                            "generation_attempt_count"],
                        "request_count": generation["generation_request_count"],
                        "missing_kind": missing_kind,
                        "missing_object": missing_object,
                        "expansion_rounds": generation["expansion_rounds"],
                        "expansion_stop_reason": generation[
                            "expansion_stop_reason"],
                        "generated": raw_count,
                        "static_precheck": static_precheck,
                        "expanded_static_precheck": expanded_static_precheck,
                    })
                combined["generated_unique"] = len(seen_questions)
                combined["stage_status"].setdefault("qa", "completed")
                combined["stage_status"].setdefault(
                    "review", "completed" if combined["questions"] else "skipped")
                checkpoint = _checkpoint(checkpoint_dir, track, "group", index)
                if checkpoint:
                    checkpoint("task-result.json", combined)
                return index, track, combined, client.usage

            checkpoint = _checkpoint(checkpoint_dir, track, "group", index)
            generated = generate_from_facts(
                group["scope"], group["facts"], client,
                max_questions=min(3, max(1, group.get("max_questions", 1))),
                qa_mode=track, allowed_types=group["allowed_types"],
                checkpoint=checkpoint, candidate_prefix="%s_g%d_" % (track, index),
                generation_mode="legacy")
            generated_questions = list(generated.get("questions", []))
            generated_keys = set()
            for question in generated_questions:
                if not isinstance(question, dict):
                    continue
                question.setdefault("qa_mode", track)
                text = question.get("question", "")
                if isinstance(text, str):
                    generated_keys.add((question.get("qa_mode", track),
                                        text.strip().casefold()))
            prefix = "%s_g%d_" % (track, index)
            if any(not q.get("candidate_id") for q in generated.get("questions", [])):
                _prefix_questions(generated, prefix)
            for question in generated.get("questions", []):
                question["evidence_group_id"] = group.get("id", "%s-group-%d" % (track, index))
            result = {
                "all_candidates": generated.get("all_candidates", list(generated.get("questions", []))),
                "revisions": [],
                "questions": generated.get("questions", []),
                "raw_generated": generated.get("raw_generated", len(generated_questions)),
                "generated_unique": len(generated_keys),
                "rejected": [dict(r, stage="qa_validation") for r in generated.get("rejected", [])],
                "stage_errors": list(generated.get("stage_errors", [])),
                "stage_status": dict(generated.get("stage_status", {})),
            }
            for question in result["all_candidates"]:
                if isinstance(question, dict):
                    question.setdefault("evidence_group_id", group.get("id", "%s-group-%d" % (track, index)))
            if (result["stage_status"].get("qa") == "completed"
                    and result["questions"]):
                try:
                    reviewed = review_candidates(
                        group["scope"], group["facts"], result["questions"], client,
                        qa_mode=track, checkpoint=checkpoint, review_mode=review_mode)
                    result["questions"] = reviewed.get("questions", result["questions"])
                    result["rejected"].extend(dict(r, stage="review_id" if r.get("reason") in {
                        "unknown_review_id", "duplicate_review_id"} else "semantic_review")
                        for r in reviewed.get("rejected", []))
                    result["stage_errors"].extend(reviewed.get("stage_errors", []))
                    result["stage_status"].update(reviewed.get("stage_status", {}))
                    result["revisions"].extend(reviewed.get("revisions", []))
                except Exception as error:
                    diagnostic = stage_error("review", error)
                    result["stage_errors"].append(diagnostic)
                    if checkpoint:
                        checkpoint("review-error.json", diagnostic)
                    result["stage_status"]["review"] = "failed"
                    result["questions"] = [
                        dict(question, status="needs_review")
                        for question in result["questions"]
                    ]
            else:
                result["stage_status"].setdefault("review", "skipped")
            if checkpoint:
                checkpoint("task-result.json", result)
            return index, track, result, client.usage
        except Exception as error:
            usage = client.usage if client is not None else []
            return index, track, {
                "questions": [], "rejected": [],
                "raw_generated": 0, "generated_unique": 0,
                "stage_errors": [stage_error("qa_task", error)],
                "stage_status": {"qa": "failed", "review": "skipped"},
            }, usage

    results = []
    with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
        for future in as_completed([pool.submit(run, task) for task in tasks]):
            results.append(future.result())
    merged = {"questions": [], "rejected": [], "usage": [],
              "stage_errors": [], "stage_status": [], "all_candidates": [],
              "reviewed_questions": [], "revisions": [],
              "expansion_audits": [], "type_attempts": []}
    raw_generated = 0
    generated_unique = 0
    raw_by_track = {}
    unique_by_track = {}
    stage_limit_rejected = 0
    stage_limit_by_track = {}

    for index, track, result, usage in sorted(results, key=lambda item: item[0]):
        merged["all_candidates"].extend(result.get("all_candidates", []))
        merged["reviewed_questions"].extend(result.get("questions", []))
        merged["revisions"].extend(result.get("revisions", []))
        merged["expansion_audits"].extend(result.get("expansion_audits", []))
        merged["type_attempts"].extend(
            dict(item, track=track, task_index=index)
            for item in result.get("type_attempts", []))
        merged["usage"].extend(usage)
        raw_count = int(result.get("raw_generated", 0) or 0)
        unique_count = int(result.get("generated_unique", 0) or 0)
        raw_generated += raw_count
        generated_unique += unique_count
        raw_by_track[track] = raw_by_track.get(track, 0) + raw_count
        unique_by_track[track] = unique_by_track.get(track, 0) + unique_count
        merged["stage_errors"].extend(dict(error, track=track, task_index=index)
                                      for error in result.get("stage_errors", []))
        merged["stage_status"].append({"track": track, "phase": "qa_review",
                                        "task_index": index,
                                        "generated": raw_count,
                                        "approved": sum(q.get("status") == "approved" for q in result.get("questions", [])),
                                        "needs_review": sum(q.get("status") == "needs_review" for q in result.get("questions", [])),
                                        "rejected": sum(isinstance(r.get("question"), dict) for r in result.get("rejected", [])),
                                        "type_attempts": len(result.get("type_attempts", [])),
                                        "expansion_attempts": len(result.get("expansion_audits", [])),
                                        "target_type_attempts": result.get("type_attempts", []),
                                        "expansions": result.get("expansion_audits", []),
                                        **result.get("stage_status", {})})
        for question in result.get("questions", []):
            if not isinstance(question, dict):
                merged["rejected"].append({
                    "question": question, "reason": "question_not_object",
                    "track": track,
                })
                continue
            question.setdefault("qa_mode", track)
            text = question.get("question", "")
            if not isinstance(text, str):
                merged["rejected"].append({
                    "question": question, "reason": "missing_question_text",
                    "track": track,
                })
                continue
            merged["questions"].append(question)
        task_rejections = []
        for item in result.get("rejected", []):
            rejection = (dict(item, track=track) if isinstance(item, dict)
                         else {"reason": "malformed_rejection", "track": track})
            task_rejections.append(rejection)
            if rejection.get("reason") in {
                    "question_limit_exceeded", "global_question_limit_exceeded"}:
                stage_limit_rejected += 1
                stage_limit_by_track[track] = stage_limit_by_track.get(track, 0) + 1
        merged["rejected"].extend(task_rejections)
    if deduplicate_results:
        merged["questions"], duplicates = deduplicate(merged["questions"])
        merged["rejected"].extend(duplicates)
    merged["question_stats"] = {
        # ``raw_generated`` counts model-emitted candidates before local
        # cross-group de-duplication. ``deduplicated`` is the globally unique
        # set that remains after review; ``pre_review_unique`` is retained for
        # diagnosing candidates removed by review.
        "raw_generated": raw_generated,
        "deduplicated": len(merged["questions"]),
        "pre_review_unique": generated_unique,
        "post_review": len(merged["questions"]),
        "limit_rejected": stage_limit_rejected,
        "by_track": {
            track: {
                "raw_generated": raw_by_track.get(track, 0),
                "deduplicated": len([
                    question for question in merged["questions"]
                    if question.get("qa_mode", track) == track
                ]),
                "pre_review_unique": unique_by_track.get(track, 0),
                "post_review": len([
                    question for question in merged["questions"]
                    if question.get("qa_mode", track) == track
                ]),
                "limit_rejected": stage_limit_by_track.get(track, 0),
            }
            for track in ("general", "code")
            if track in raw_by_track or track in unique_by_track
        },
    }
    return merged


def _limit_questions(result, limits):
    kept, selection, counts = select_approved(result.get("questions", []), limits)
    result.update(questions=kept, counts=counts)
    result.setdefault("selection", []).extend(selection)
    stats = result.setdefault("question_stats", _empty_question_stats())
    stats.update(post_limit=len(kept), limit_rejected=0,
                 over_quota=sum(s["selection_status"] == "over_quota" for s in selection))
    for mode, count in counts.items():
        stats.setdefault("by_track", {}).setdefault(mode, {}).update(post_limit=count, limit_rejected=0)
    return result


def _publication_view(questions, limits, workspaces=(), duplicate_decisions=()):
    # Project separately so approved-first deduplication also works after redaction.
    safe, rejected = [], []
    totals = {"total": 0, "path_redacted": 0, "credential_detected": 0,
              "by_track": {}, "track_details": {}}
    for q in questions:
        projected, stats = _filter_private_questions([q], rejected, workspaces)
        safe.extend(projected)
        for key in ("total", "path_redacted", "credential_detected"):
            totals[key] += stats.get(key, 0)
        mode = q.get("qa_mode", "code")
        totals["by_track"][mode] = totals["by_track"].get(mode, 0) + stats["total"]
        details = totals["track_details"].setdefault(mode, {})
        for key, value in stats.get("track_details", {}).get(mode, {}).items():
            details[key] = details.get(key, 0) + value
    safe, reviewed_duplicates = apply_duplicate_decisions(safe, duplicate_decisions)
    unique, duplicates = deduplicate_reviewed(safe)
    kept, selection, counts = select_approved(unique, limits)
    selection = reviewed_duplicates + duplicates + selection + [
        {"candidate_id": r["question"].get("id"),
         "selection_status": "safety_blocked",
         "reason": r["reason"]} for r in rejected if isinstance(r.get("question"), dict)]
    return {"questions": kept, "counts": counts, "selection": selection,
            "publication_rejected": rejected, "path_stats": totals,
            "deduplicated_count": len(unique),
            "eligible_counts": {m: sum(q.get("status") == "approved" and q.get("qa_mode", "code") == m
                                       for q in unique) for m in limits}}


def _question_mode(item, default="code"):
    if isinstance(item, dict):
        mode = item.get("qa_mode")
        if mode in {"general", "code"}:
            return mode
        if item.get("track") in {"general", "code"}:
            return item["track"]
        question = item.get("question")
        if isinstance(question, dict) and question.get("qa_mode") in {"general", "code"}:
            return question["qa_mode"]
    return default


def _status(result, enabled=True, network=True):
    """Return the status of the usable candidate set.

    Stage failures are fatal only when no candidate survived.  When there is a
    usable set alongside an unrelated failed chunk/review, callers must not
    present it as fully approved; ``needs_review`` makes that distinction
    explicit.  Rejected candidates alone do not downgrade approved output.
    """
    if not enabled:
        return "disabled"
    if not network:
        return "static_only"
    questions = list(result.get("questions") or [])
    stage_errors = list(result.get("stage_errors") or [])
    if questions:
        if stage_errors or any(not isinstance(question, dict)
                               or question.get("status") != "approved"
                               for question in questions):
            return "needs_review"
        return "approved"
    if stage_errors:
        return "failed"
    if result.get("rejected"):
        return "rejected"
    return "completed_no_questions"


def _public_question(question):
    mode = question.get("qa_mode", "code")
    item = {
        "id": question.get("id"), "qa_mode": mode,
        "type": question.get("type", question.get("category")),
        "question": question.get("question", ""),
        "difficulty": question.get("difficulty"),
        "type_origin": question.get("type_origin"),
        "difficulty_origin": question.get("difficulty_origin"),
        "difficulty_distance": question.get("difficulty_distance"),
        "status": question.get("status", "needs_review"),
        "stage_count": question.get("stage_count"),
        "reasoning_hops": question.get("reasoning_hops"),
        "graph_hops": question.get("graph_hops"),
        "answer_points": [{"text": point.get("text", "")}
                          for point in question.get("answer_points", [])],
        "forbidden_points": [{"text": point.get("text", "")}
                             for point in question.get("forbidden_points", [])],
    }
    if mode == "code":
        item.update(category=question.get("category", item["type"]),
                    track=question.get("track"))
    return item


def _redact_public_text(value, workspaces=()):
    if not isinstance(value, str):
        return value, 0
    count = 0
    parts = re.split(r"(https?://[^\s<>]+)", value)
    for index in range(0, len(parts), 2):
        for root in sorted({root.rstrip('/\\') for root in workspaces if root}, key=len, reverse=True):
            if root:
                parts[index], n = re.subn(r"(?<![A-Za-z0-9:/\\])" + re.escape(root) + r"(?=[/\\\s`'\"<>，。]|$)",
                                         "<workspace>", parts[index])
                count += n
        parts[index], n = _USER_PATH.subn("~", parts[index])
        count += n
        parts[index], n = _INTERNAL_PUBLIC_ID.subn("该条记录", parts[index])
        count += n
    return ''.join(parts), count


def _project_public_question(question, workspaces=()):
    projected = dict(question)
    redactions = 0
    for field in ("id", "question", "difficulty_reason", "memory_requirement", "use_case", "external_knowledge"):
        if field in projected:
            projected[field], count = _redact_public_text(projected.get(field), workspaces)
            redactions += count
    for field in ("answer_points", "forbidden_points"):
        projected[field] = []
        for point in question.get(field, []):
            item = dict(point)
            item["text"], count = _redact_public_text(item.get("text"), workspaces)
            redactions += count
            projected[field].append(item)
    return projected, redactions


def _question_has_credential(question):
    values = [question.get(key, "") for key in ("id", "question", "use_case", "difficulty_reason", "memory_requirement", "external_knowledge")]
    for field in ("answer_points", "forbidden_points"):
        values.extend(point.get("text", "") for point in question.get(field, []) if isinstance(point, dict))
    return any(credential_detected(value) for value in values if isinstance(value, str))


def _safe_public_questions(questions, rejected, workspaces=()):
    """Project candidates for publication, retaining raw audit data separately."""
    safe, _ = _filter_private_questions(questions, rejected, workspaces)
    return [_public_question(question) for question in safe]


def _filter_private_questions(questions, rejected, workspaces=()):
    """Remove unsafe candidates before applying global quotas.

    The original candidate is retained in the rejection record for the local
    audit, while the returned list is the only set allowed to consume a
    per-track quota.  This lets a safe candidate later in deterministic merge
    order fill a slot occupied by an unsafe one.
    """
    safe, counts = [], {"total": 0, "by_track": {"general": 0, "code": 0}}
    counts.update(path_redacted=0, credential_detected=0, redaction_deduplicated=0)
    counts["track_details"] = {}
    seen = set()
    for question in questions:
        if not isinstance(question, dict):
            rejected.append({"question": question, "reason": "question_not_object"})
            continue
        mode = _question_mode(question)
        track_counts = counts["track_details"].setdefault(mode, {
            "path_redacted": 0, "credential_detected": 0,
            "redaction_deduplicated": 0,
        })
        if _question_has_credential(question):
            mode = _question_mode(question)
            rejected.append({"question": question, "reason": "credential_detected", "track": mode})
            counts["total"] += 1
            counts["credential_detected"] += 1
            track_counts["credential_detected"] += 1
            counts["by_track"][mode] = counts["by_track"].get(mode, 0) + 1
            continue
        projected, count = _project_public_question(question, workspaces)
        projected["path_redacted"] = count
        counts['path_redacted'] += bool(count)
        track_counts["path_redacted"] += bool(count)
        key = (_question_mode(question), projected.get('question', '').strip().casefold())
        if key in seen:
            counts['redaction_deduplicated'] += 1
            track_counts["redaction_deduplicated"] += 1
            rejected.append({'question': question, 'reason': 'duplicate_after_redaction', 'track': _question_mode(question)})
            continue
        seen.add(key)
        safe.append(projected)
    return safe, counts


def _empty_question_stats():
    return {
        "raw_generated": 0,
        "deduplicated": 0,
        "pre_review_unique": 0,
        "post_review": 0,
        "post_limit": 0,
        "limit_rejected": 0,
        "path_rejected": 0,
        "path_redacted": 0,
        "credential_detected": 0,
        "redaction_deduplicated": 0,
        "published": 0,
        "by_track": {},
    }


def _add_path_stats(result, stats):
    """Merge path-filter counts into the shared question statistics."""
    question_stats = result.setdefault("question_stats", _empty_question_stats())
    question_stats["path_rejected"] = (
        question_stats.get("path_rejected", 0) + stats.get("total", 0))
    question_stats.setdefault("by_track", {})
    for key in ('path_redacted', 'credential_detected', 'redaction_deduplicated'):
        question_stats[key] = question_stats.get(key, 0) + stats.get(key, 0)
    for mode in ("general", "code"):
        track_stats = question_stats["by_track"].setdefault(mode, {})
        for key, value in stats.get("track_details", {}).get(mode, {}).items():
            track_stats[key] = track_stats.get(key, 0) + value
        track_stats["path_rejected"] = (
            track_stats.get("path_rejected", 0)
            + stats.get("by_track", {}).get(mode, 0))


def _prepare_code_scopes(graph, records, cutoff, args, candidate_limit=None):
    if args.seed and not args.adaptive_subgraphs:
        scopes = [query_scope_adaptive(graph, records, args.seed, cutoff,
                                       args.initial_hops, args.max_hops,
                                       args.max_context_chars)]
        meta = None
    else:
        scopes, meta = adaptive_subgraphs(graph, records, cutoff, args.seed,
                                          args.max_context_chars,
                                          beam_width=3, max_depth=args.max_hops,
                                          max_candidates=candidate_limit)
    # Discussion stages are computed once over the selected visible dialogue
    # range, then projected into every code scope.  This keeps stage IDs stable
    # across chunks/candidates; character chunks must not masquerade as stages.
    visible = [record for record in records
               if record.get("kind") == "message"
               and record.get("role") in {"user", "assistant"}
               and record.get("order", 0) <= cutoff]
    stages = identify_stages(visible)
    stage_by_record = {
        record_id: stage["id"]
        for stage in stages for record_id in stage.get("record_ids", [])
    }

    def stage_for_order(order):
        if not isinstance(order, int):
            return None
        containing = [stage for stage in stages
                      if stage.get("start_order", order) <= order <= stage.get("end_order", order)]
        if containing:
            return containing[-1]["id"]
        preceding = [stage for stage in stages if stage.get("start_order", order) <= order]
        return preceding[-1]["id"] if preceding else (stages[0]["id"] if stages else None)

    for index, scope in enumerate(scopes):
        for record in scope.get("dialogue", []):
            stage_id = stage_by_record.get(record.get("id"))
            if stage_id:
                record["stage_id"] = stage_id
        for collection, order_key in (("events", "order"), ("versions", "observed_at")):
            for record in scope.get(collection, []):
                stage_id = stage_for_order(record.get(order_key))
                if stage_id:
                    record["stage_id"] = stage_id
        selected_stage_ids = {
            record.get("stage_id") for record in scope.get("dialogue", [])
            if record.get("stage_id")
        }
        scope["stages"] = [dict(stage, record_ids=[
            record.get("id") for record in scope.get("dialogue", [])
            if record.get("id") in stage.get("record_ids", [])
        ]) for stage in stages if stage["id"] in selected_stage_ids]
        scope["stage_count"] = len(scope["stages"])
        scope["scope_index"], scope["track"] = index, "code"
    return scopes, meta


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    options = _parse_options(args, parser)
    created = False
    try:
        records = load_dialogue(args.input)
        if not records:
            raise ValueError("No supported records found")
        cutoff = args.cutoff if args.cutoff is not None else records[-1]["order"]
        if not 1 <= cutoff <= records[-1]["order"]:
            raise ValueError("Cutoff outside normalized event range")
        records = [record for record in records if record["order"] <= cutoff]
        from .normalize import resolve_source_events, resolve_source_objects
        seed_sources = resolve_source_events(records, args.source_event)
        seed_sources.update(resolve_source_objects(records, args.source_object))
        public_input = all(r.get("input_schema") == "model-visible-dialogue-v1" for r in records)
        if args.seed and public_input:
            seed_sources.update(resolve_source_objects(records, [args.seed]))
        if args.reuse_facts and json.loads((args.reuse_facts / "normalized.json").read_text()) != records:
            raise ValueError("Saved facts belong to a different normalized input")
        args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
        created = True
        os.chmod(args.output, 0o700)
        graph = build_graph(records)
        save(args.output, "normalized.json", records)
        save(args.output, "graph.json", graph)
        save(args.output, "current-graph.json", graph_at(graph, cutoff))

        general_scope = None
        code_scopes, adaptive_meta = [], None
        if options["enabled_general"]:
            general_scope = build_general_scope(records, cutoff, args.max_context_chars, graph)
            general_scope["track"] = "general"
            save(args.output, "general-scope.json", general_scope)
        if options["enabled_code"]:
            if seed_sources or public_input:
                scope = build_general_scope(records, cutoff, args.max_context_chars, graph)
                scope.update(versions=graph["versions"], events=graph["events"], track="code")
                code_scopes = [scope]
            else:
                code_scopes, adaptive_meta = _prepare_code_scopes(
                    graph, records, cutoff, args,
                    max(options["code_count"], options["code_group_budget"]))
            if adaptive_meta is not None:
                save(args.output, "adaptive-subgraphs.json", {
                    "meta": adaptive_meta,
                    "candidates": [{
                        "seed": item.get("seed"), "score": item.get("score"),
                        "context_chars": item.get("context_chars"),
                        "events": len(item.get("events", [])),
                        "versions": len(item.get("versions", [])),
                        "dialogue": len(item.get("dialogue", [])),
                    } for item in code_scopes],
                })
            if code_scopes:
                save(args.output, "scope.json", code_scopes[0])
                if len(code_scopes) > 1:
                    save(args.output, "scopes.json", code_scopes)

        result = {"facts": [], "questions": [], "rejected": [],
                  "stage_errors": [], "stage_status": [], "usage": [],
                  "question_stats": _empty_question_stats(),
                  "all_questions": []}
        track_results = {
            "general": {"facts": [], "questions": [], "rejected": [],
                        "stage_errors": [], "stage_status": [], "usage": []},
            "code": {"facts": [], "questions": [], "rejected": [],
                     "stage_errors": [], "stage_status": [], "usage": []},
        }
        chunk_summaries, fact_tasks = [], []
        track_task_counts = {"general": 0, "code": 0}
        groups = []
        unique_chunk_count = 0
        if args.allow_network:
            scopes_by_track = []
            if general_scope is not None:
                scopes_by_track.append(("general", [general_scope]))
            if options["enabled_code"]:
                scopes_by_track.append(("code", code_scopes))
            task_index = 0
            seen_chunks = {}
            structural_scopes = {"general": [], "code": []}
            for track, scopes in scopes_by_track:
                # Extract each source once across overlapping graph candidates.
                # QA grouping still uses the source-linked fact index afterwards.
                extraction_scopes = ([merge_scopes(scopes, track, args.model_request_chars)]
                                     if len(scopes) > 1 else scopes)
                for scope_index, candidate in enumerate(extraction_scopes):
                    # Code relations locate QA evidence after facts exist; they
                    # need not be repeated in every source-extraction request.
                    structural_scopes[track].append({
                        "cutoff": cutoff, "full_range_covered": True,
                        "edges": candidate.get("edges", []),
                        "historical_edges": candidate.get("historical_edges", []),
                    })
                    chunks = split_scope(candidate, args.chunk_chars, args.chunk_overlap,
                                         include_code_edges=False)
                    for chunk in chunks:
                        chunk["scope_index"] = scope_index
                        chunk["track"] = track
                        chunk["model_request_chars"] = args.model_request_chars
                        summary = {
                            "task_index": task_index,
                            "track": track,
                            "scope_index": scope_index,
                            "chunk_index": chunk.get("chunk_index"),
                            "chunk_window": chunk.get("chunk_window"),
                            "context_chars": chunk.get("context_chars"),
                            "dialogue_records": len(chunk.get("dialogue", [])),
                            "versions": len(chunk.get("versions", [])),
                        }
                        identity = _chunk_identity(track, chunk)
                        if identity in seen_chunks:
                            summary["duplicate_of"] = seen_chunks[identity]
                            summary["task_index"] = None
                            chunk_summaries.append(summary)
                            continue
                        seen_chunks[identity] = task_index
                        chunk_summaries.append(summary)
                        fact_tasks.append((task_index, track, chunk))
                        track_task_counts[track] += 1
                        task_index += 1
            unique_chunk_count = sum(1 for item in chunk_summaries
                                     if item.get("task_index") is not None)
            save(args.output, "chunks.json", chunk_summaries)
            if args.reuse_facts and json.loads((args.reuse_facts / "chunks.json").read_text()) != chunk_summaries:
                raise ValueError("Saved facts use a different chunk layout")
            # Report missing evidence per enabled track.  A healthy other
            # track must not hide a scope failure in this one.
            for track in ("general", "code"):
                if options["enabled_" + track] and not track_task_counts[track]:
                    result["stage_errors"].append({
                        "track": track, "stage": "scope",
                        "error_type": "no_evidence",
                    })
                    result["stage_status"].append({
                        "track": track, "phase": "scope",
                        "scope": "failed", "error_type": "no_evidence",
                    })
            if not fact_tasks:
                # All enabled tracks were already annotated above.  There is
                # no fact/QA work to schedule, but still emit the normal stage
                # artifacts below.
                pass
            else:
                checkpoint_dir = args.output / "stages"
                checkpoint_dir.mkdir(mode=0o700)
                fact_options = {"reuse_dir": args.reuse_facts} if args.reuse_facts else {}
                facts_result = _run_fact_tasks(
                    fact_tasks, args.endpoint, args.model, args.key_env,
                    args.parallel_workers, checkpoint_dir, **fact_options)
                save(args.output, "fact-extraction.json", {
                    "reused_from": str(args.reuse_facts) if args.reuse_facts else None,
                    "usage": facts_result["usage"], "stage_errors": facts_result["stage_errors"],
                    "stage_status": facts_result["stage_status"],
                })
                for track, relation_scopes in structural_scopes.items():
                    facts_result["scopes"][track].extend(relation_scopes)
                result = {
                    "facts": facts_result["facts"], "questions": [],
                    "rejected": list(facts_result["rejected"]),
                    "usage": list(facts_result["usage"]),
                    "stage_errors": list(facts_result["stage_errors"]),
                    "stage_status": list(facts_result["stage_status"]),
                    "question_stats": _empty_question_stats(),
                    "all_questions": [],
                }
                # Preserve per-track scope errors collected before fact
                # extraction.  They are independent of successful chunks.
                result["stage_errors"].extend(
                    {"track": track, "stage": "scope",
                     "error_type": "no_evidence"}
                    for track in ("general", "code")
                    if options["enabled_" + track] and not track_task_counts[track]
                )
                result["stage_status"].extend(
                    {"track": track, "phase": "scope",
                     "scope": "failed", "error_type": "no_evidence"}
                    for track in ("general", "code")
                    if options["enabled_" + track] and not track_task_counts[track]
                )

                save(args.output, "facts.json", result["facts"])
                groups, group_tasks = [], []
                evidence_indexes = {}
                group_index = 0
                for track in ("general", "code"):
                    if not options["enabled_" + track]:
                        continue
                    track_facts = [fact for fact in result["facts"]
                                   if fact.get("qa_mode") == track]
                    allowed_types = options[track + "_types"]
                    count = options[track + "_group_budget"]
                    try:
                        print("Evidence index: %s, %d facts" % (track, len(track_facts)), flush=True)
                        evidence_index = build_evidence_index(
                            track_facts, facts_result["scopes"][track], track,
                            args.model_request_chars)
                        evidence_indexes[track] = evidence_index
                        print("Evidence groups: %s index ready" % track, flush=True)
                        track_groups = build_evidence_groups(
                            track_facts, facts_result["scopes"][track], track,
                            allowed_types, count,
                            target_chars=min(16000, args.model_request_chars),
                            max_chars=args.model_request_chars,
                            evidence_index=evidence_index,
                            seed_sources=seed_sources,
                            static_selection=args.review_mode == "simple")
                        result["stage_status"].append({
                            "track": track, "phase": "grouping",
                            "grouping": "completed", "groups": len(track_groups),
                            "eligible_type_skips": list(
                                evidence_index.get("eligible_skips", [])),
                        })
                    except (ValueError, KeyError, TypeError) as error:
                        track_groups = []
                        result["stage_errors"].append({
                            "track": track, "stage": "grouping",
                            "error_type": type(error).__name__,
                        })
                        result["stage_status"].append({
                            "track": track, "phase": "grouping",
                            "grouping": "failed", "groups": 0,
                        })
                    groups.extend(track_groups)
                    for group in track_groups:
                        group["max_questions"] = min(
                            options["questions_per_group"], group.get("max_questions", 1))
                        group_tasks.append((group_index, track, group))
                        group_index += 1
                save(args.output, "evidence-groups.json", groups)

                limits = {t: options[t + "_count"] for t in ("general", "code") if options["enabled_" + t]}
                budgets = {t: options[t + "_group_budget"] for t in limits}
                workspaces = [r["workspace"] for r in records if r.get("workspace")]
                duplicate_state = {"reviewed_pairs": set(), "decisions": []}

                def adjudicate_duplicates(merged, batch_result, batch_number):
                    candidates = [question for question in merged.get("all_questions", [])
                                  if isinstance(question, dict)
                                  and question.get("status") in {"approved", "needs_review"}]
                    if len(candidates) < 2:
                        return {}
                    try:
                        client = ChatClient(args.endpoint, args.model, args.key_env)
                        reviewed = review_duplicate_clusters(
                            candidates, client, duplicate_state["reviewed_pairs"])
                    except Exception as error:
                        return {"errors": [{
                            "stage": "duplicate_review", "error_code": "dedup_error",
                            "error_type": type(error).__name__, "batch": batch_number,
                        }]}
                    duplicate_state["reviewed_pairs"] = set(reviewed["reviewed_pairs"])
                    duplicate_state["decisions"].extend(reviewed["decisions"])
                    errors = [dict(error, stage="duplicate_review",
                                   error_code="dedup_error", batch=batch_number)
                              for error in reviewed["errors"]]
                    return {"decisions": reviewed["decisions"],
                            "usage": reviewed["usage"], "errors": errors}

                def batch_checkpoint(receipt, partial):
                    number = receipt["batch"]
                    partial["candidate_records"] = build_audit(
                        partial["all_candidates"], partial["all_questions"],
                        partial["rejected"] + partial.get("publication_rejected", []),
                        partial["selection"], partial["questions"], partial["revisions"])
                    save(args.output, "batch-%03d.json" % number, receipt)
                    save(args.output, "batch-%03d-candidates.json" % number, partial["all_candidates"])
                    save(args.output, "batch-%03d-audit.json" % number, partial)
                    save(args.output, "batch-%03d-public.json" % number,
                         [_public_question(q) for q in partial["questions"]])
                    print("Batch %d: approved unique %s; missing %s" %
                          (number, receipt["counts"], receipt["missing"]), flush=True)
                qa_result = replenish(
                    group_tasks, limits, budgets, args.parallel_workers,
                    lambda batch: _run_qa_tasks(batch, args.endpoint, args.model, args.key_env,
                                               args.parallel_workers, checkpoint_dir,
                                               deduplicate_results=False,
                                               review_mode=args.review_mode,
                                               evidence_indexes=evidence_indexes,
                                               expansion_budget=args.expansion_budget),
                    lambda questions, caps: _publication_view(
                        questions, caps, workspaces, duplicate_state["decisions"]),
                    checkpoint=batch_checkpoint, initial_errors=result["stage_errors"],
                    initial_request_count=sum(u.get("request_count", 1) for u in result["usage"]),
                    after_batch=adjudicate_duplicates)
                for key in ("rejected", "usage", "stage_errors", "stage_status"):
                    result[key].extend(qa_result[key])
                for key in ("questions", "all_questions", "all_candidates", "selection", "revisions",
                            "duplicate_decisions", "dedup_errors", "progress",
                            "expansion_audits", "type_attempts"):
                    result[key] = qa_result.get(key, [])
                result["expansion_audits"] = [
                    audit for row in qa_result.get("stage_status", [])
                    for audit in row.get("expansions", [])
                ]
                result["type_attempts"] = [
                    attempt for row in qa_result.get("stage_status", [])
                    for attempt in row.get("target_type_attempts", [])
                ]
                result["rejected"].extend(qa_result.get("publication_rejected", []))
                result["question_stats"].update(
                    raw_generated=len(result["all_candidates"]), post_review=len(result["all_questions"]),
                    post_limit=len(result["questions"]),
                    deduplicated=qa_result["deduplicated_count"],
                    over_quota=sum(x["selection_status"] == "over_quota" for x in result["selection"]))
                for track in limits:
                    result["question_stats"]["by_track"][track] = {
                        "raw_generated": sum(q.get("qa_mode") == track for q in result["all_candidates"] if isinstance(q, dict)),
                        "post_review": sum(q.get("qa_mode") == track for q in result["all_questions"]),
                        "post_limit": qa_result["counts"][track]}
                _add_path_stats(result, qa_result["path_stats"])
                for track in ("general", "code"):
                    track_results[track]["questions"] = [
                        q for q in result["questions"] if q.get("qa_mode", track) == track]
                    track_results[track]["facts"] = [
                        fact for fact in result["facts"] if fact.get("qa_mode", track) == track]
                    track_results[track]["rejected"] = [
                        item for item in result.get("rejected", [])
                        if _question_mode(item, track) == track]
                    track_results[track]["stage_errors"] = [
                        error for error in result.get("stage_errors", [])
                        if error.get("track") == track]
                    track_results[track]["stage_status"] = [
                        status for status in result.get("stage_status", [])
                        if status.get("track") == track]
                    track_results[track]["counts"] = {
                        "questions": len(track_results[track]["questions"]),
                        "facts": len(track_results[track]["facts"]),
                    }
            if not (args.output / "facts.json").exists():
                save(args.output, "facts.json", result.get("facts", []))
            save(args.output, "candidates.json",
                 result.get("all_candidates", result.get("all_questions", result.get("questions", []))))
            save(args.output, "stage-status.json", result.get("stage_status", []))
            save(args.output, "stage-errors.json", result.get("stage_errors", []))
        else:
            result["status"] = "static_only"
            # The dual-track files are written once below, after the common
            # public projection logic.  Keeping that write in one place also
            # makes static and network runs share the same file contract.

        # Project public questions only after all stages have completed.  A
        # raw candidate remains in private audit artifacts; public fields use
        # the projected paths and must not contain detected credentials.
        public_questions = [_public_question(q) for q in result.get("questions", [])]
        safe_question_ids = {item.get("id") for item in public_questions}
        # These values are needed by both per-track files and the combined
        # manifest; compute them before writing either view.
        normalized_question_stats = dict(_empty_question_stats())
        normalized_question_stats.update(result.get("question_stats", {}))
        result["question_stats"] = normalized_question_stats
        result["question_stats"]["published"] = len(public_questions)
        result["question_stats"].setdefault("by_track", {})
        public_counts = {track: len([
            item for item in public_questions if item.get("qa_mode") == track
        ]) for track in ("general", "code")}
        for track in ("general", "code"):
            track_stats = dict(_empty_question_stats())
            track_stats.update(
                result["question_stats"]["by_track"].get(track, {}))
            track_stats.pop("by_track", None)
            track_stats["published"] = public_counts[track]
            result["question_stats"]["by_track"][track] = track_stats
        for track in ("general", "code"):
            track_results[track]["questions"] = [
                question for question in result.get("questions", [])
                if question.get("qa_mode", track) == track
                and question.get("id") in safe_question_ids
            ]
            track_results[track]["facts"] = [
                fact for fact in result.get("facts", [])
                if fact.get("qa_mode", track) == track
            ]
            track_results[track]["rejected"] = [
                item for item in result.get("rejected", [])
                if _question_mode(item, track) == track
            ]
            track_results[track]["stage_errors"] = [
                error for error in result.get("stage_errors", [])
                if error.get("track") == track
            ]
            track_results[track]["stage_status"] = [
                status for status in result.get("stage_status", [])
                if status.get("track") == track
            ]
            track_results[track]["counts"] = {
                "questions": len(track_results[track]["questions"]),
                "facts": len(track_results[track]["facts"]),
            }
            track_results[track]["enabled"] = options["enabled_" + track]
            track_stats = dict(_empty_question_stats())
            track_stats.update(
                result.get("question_stats", {}).get("by_track", {}).get(track, {}))
            track_stats.pop("by_track", None)
            track_results[track]["question_stats"] = track_stats
            save(args.output, "facts-%s.json" % track,
                 track_results[track]["facts"])
            track_status = _status(
                track_results[track],
                enabled=options["enabled_" + track],
                network=args.allow_network,
            )
            track_results[track]["status"] = track_status
            save(args.output, "%s-qa.json" % track, {
                "status": track_status,
                "enabled": options["enabled_" + track],
                "qa_mode": track,
                "questions": [_public_question(q)
                              for q in track_results[track]["questions"]],
                "counts": track_results[track]["counts"],
                "question_stats": track_results[track]["question_stats"],
                "stage_errors": track_results[track]["stage_errors"],
            })

        if args.allow_network:
            # Use the publishable set when determining the run status.  A
            # rejected private-path candidate must not make an otherwise valid
            # public set appear approved, and must not abort the run.
            status_view = dict(result, questions=[
                question for question in result.get("questions", [])
                if question.get("id") in safe_question_ids])
            result["status"] = _status(status_view)
        public = {"status": result.get("status", "static_only"),
                  "qa_mode": args.qa_mode, "questions": [],
                  "counts": {"general": 0, "code": 0},
                  "tracks": {
                      track: {
                          "enabled": options["enabled_" + track],
                          "status": track_results[track].get(
                              "status", "disabled"),
                          "count": public_counts[track],
                      }
                      for track in ("general", "code")
                  }}
        for item in public_questions:
            public["questions"].append(item)
            public["counts"].setdefault(item["qa_mode"], 0)
            public["counts"][item["qa_mode"]] += 1
        audit = {"status": public["status"], "review_mode": args.review_mode,
                 "rejected": result.get("rejected", []),
                 "candidates": result.get("all_candidates", []),
                 "questions": result.get("all_questions",
                                           result.get("questions", [])),
                 "stage_errors": result.get("stage_errors", []),
                 "stage_status": result.get("stage_status", []),
                 "duplicate_decisions": result.get("duplicate_decisions", []),
                 "dedup_errors": result.get("dedup_errors", []),
                 "expansion_audits": result.get("expansion_audits", []),
                 "type_attempts": result.get("type_attempts", []),
                 "selection": result.get("selection", []), "revisions": result.get("revisions", []),
                 "candidate_records": build_audit(result.get("all_candidates", []),
                     result.get("all_questions", []), result.get("rejected", []),
                     result.get("selection", []), result.get("questions", []),
                     result.get("revisions", [])),
                 "progress": result.get("progress", {})}
        save(args.output, "qa-public.json", public)
        save(args.output, "qa-audit.json", audit)
        save(args.output, "qa.json", public)
        status_counts = {}
        type_counts = {}
        for item in public_questions:
            status = item.get("status", "needs_review")
            status_counts[status] = status_counts.get(status, 0) + 1
            question_type = item.get("type")
            if question_type:
                type_counts[question_type] = type_counts.get(question_type, 0) + 1
        rejected_by_reason = {}
        for item in result.get("rejected", []):
            reason = item.get("reason", "unknown") if isinstance(item, dict) else "malformed"
            rejected_by_reason[reason] = rejected_by_reason.get(reason, 0) + 1
        stage_error_types = {}
        for item in result.get("stage_errors", []):
            error_type = item.get("error_type", "unknown") if isinstance(item, dict) else "malformed"
            stage_error_types[error_type] = stage_error_types.get(error_type, 0) + 1
        code_stage_ids = {
            stage.get("id")
            for scope in code_scopes
            for stage in scope.get("stages", [])
            if isinstance(stage, dict) and isinstance(stage.get("id"), str)
        }
        stage_counts = {
            "general": len(general_scope.get("stages", []))
            if general_scope is not None else 0,
            "code": len(code_stage_ids),
            "code_by_scope": [len(scope.get("stages", []))
                              for scope in code_scopes],
        }
        manifest = {
            "quality_contract_version": 3,
            "count_semantics": "approved_safe_unique_target",
            "progress": result.get("progress", {}),
            "coverage": coverage_report(
                [r for r in records if r.get("order", 0) <= cutoff],
                [v for v in graph["versions"] if v.get("observed_at", 0) <= cutoff],
                facts_result["scopes"] if args.allow_network and fact_tasks else {},
                result.get("facts", []), groups, result.get("stage_status", []),
                result.get("questions", []),
                attempted_group_ids=result.get("progress", {}).get("attempted_group_ids", [])),
            "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "seed_event_ids": args.source_event,
            "seed_objects": args.source_object,
            "seed_source_ids": sorted(seed_sources),
            "cutoff": cutoff, "events": len(records),
            "file_versions": len(graph["versions"]),
            "diagnostics": len(graph["diagnostics"]), "mode": public["status"],
            "qa_mode": args.qa_mode,
            "review_mode": args.review_mode,
            "general_types": list(options["general_types"]),
            "code_types": list(options["code_types"]),
            "general_count": options["general_count"],
            "code_count": options["code_count"],
            "general_group_budget": options["general_group_budget"],
            "code_group_budget": options["code_group_budget"],
            "questions_per_group": options["questions_per_group"],
            "expansion_budget": args.expansion_budget,
            "configured_limits": {
                "general": options["general_count"],
                "code": options["code_count"],
            },
            "model": args.model, "usage": result.get("usage", []),
            "source_kinds": {
                "records": {kind: sum(1 for record in records
                                       if record.get("source_kind") == kind)
                            for kind in ("conversation", "document", "tool", "code", "test")},
                "facts": {kind: sum(1 for fact in result.get("facts", [])
                                     if kind in fact.get("source_kinds", [fact.get("source_kind")]))
                          for kind in ("conversation", "document", "tool", "code", "test")},
            },
            "network": bool(args.allow_network),
            "scopes": {
                "general": 1 if general_scope is not None else 0,
                "code": len(code_scopes),
            },
            "stages": stage_counts,
            "chunks": {
                "total": len(chunk_summaries),
                "unique": unique_chunk_count,
                "duplicates": len(chunk_summaries) - unique_chunk_count,
            },
            "evidence_groups": len(groups),
            "tracks": {
                track: {
                    "enabled": options["enabled_" + track],
                    "status": track_results[track].get("status", "disabled"),
                    "counts": dict(track_results[track].get("counts", {})),
                    "question_stats": dict(
                        track_results[track].get("question_stats", {})),
                }
                for track in ("general", "code")
            },
            "facts": {
                "total": len(result.get("facts", [])),
                "general": len([fact for fact in result.get("facts", [])
                                if fact.get("qa_mode") == "general"]),
                "code": len([fact for fact in result.get("facts", [])
                             if fact.get("qa_mode") == "code"]),
            },
            "questions": {
                # ``generated`` remains a compatibility alias for the
                # post-limit candidate count used by the original manifest.
                # The explicit fields below
                # make each filtering step auditable.
                "generated": result["question_stats"].get(
                    "post_limit", len(result.get("questions", []))),
                "raw_generated": result["question_stats"].get("raw_generated", 0),
                "deduplicated": result["question_stats"].get("deduplicated", 0),
                "post_review": result["question_stats"].get("post_review", 0),
                "post_limit": result["question_stats"].get(
                    "post_limit", len(result.get("questions", []))),
                "published": len(public_questions),
                "limit_rejected": result["question_stats"].get(
                    "limit_rejected", 0),
                "over_quota": result["question_stats"].get("over_quota", 0),
                "path_rejected": result["question_stats"].get(
                    "path_rejected", 0),
                "path_redacted": result["question_stats"].get("path_redacted", 0),
                "credential_detected": result["question_stats"].get("credential_detected", 0),
                "redaction_deduplicated": result["question_stats"].get("redaction_deduplicated", 0),
                "status": status_counts,
                "type": type_counts,
                "by_track": result["question_stats"].get("by_track", {}),
            },
            "rejections": {
                "total": len(result.get("rejected", [])),
                "by_reason": rejected_by_reason,
            },
            "failures": {
                "total": len(result.get("stage_errors", [])),
                "by_type": stage_error_types,
            },
            "limitations": ["Partial evidence, not a complete repository",
                            "Python AST only; arbitrary shell output remains raw evidence",
                            "Syntactic call references require semantic verification",
                            "Human review required; difficulty is provisional"],
        }
        save(args.output, "manifest.json", manifest)
        print("Completed: %s; %d QA candidates" %
              (args.output, len(result.get("questions", []))))
        return 0
    except (ValueError, KeyError, TypeError, OSError) as error:
        if created:
            save(args.output, "failure.json", {
                "status": "failed", "error_type": type(error).__name__,
                "message": "Run stopped; inspect local inputs and stage artifacts.",
            })
        print("Run failed (%s). No QA should be treated as validated." % type(error).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
