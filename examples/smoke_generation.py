"""Opt-in frozen-group generation/review smoke runner.

This runner is intentionally inert unless ``--allow-network`` is supplied.
It reads only the selected frozen evidence groups, writes private checkpoints
as each stage returns, and emits a small CLI-style public projection.  It does
not retry provider calls; each target type is run in its own one-question unit.
Split and simple review may invoke one bounded repair/re-review path when
``allow_repair`` is enabled. Simple generation uses focus selection followed by
QA generation. Review checks atomicity, completeness and evidence; code adds an
answer-basis check. Directed expansion is bounded and requires new evidence.
The dry-run plan reports the request bound, including evidence supplementation
and repair. No type, difficulty, or other annotation call is made.
"""

import argparse
import copy
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


EXAMPLES_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dialogue_benchmark.cli import (_safe_public_questions,
                                    generate_simple_target)
from dialogue_benchmark.fact_index import (build_evidence_index,
                                            candidate_review_projection,
                                            static_candidate_labels,
                                            static_code_evidence_check)
from dialogue_benchmark.llm import (ChatClient, generate_from_facts,
                                    repair_simple_validation_rejection,
                                    review_candidates, stage_error)

from compare_reviews import load_bundle


TRACKS = ("general", "code")
REVIEW_MODES = ("split", "simple")
SIMPLE_BASE_CALLS = {"general": 5, "code": 6}
EVIDENCE_SUPPLEMENT_CALLS_PER_GROUP = 1
SIMPLE_REPAIR_CALLS = {"general": 4, "code": 5}
SPLIT_BASE_CALLS_PER_GROUP = 3
SPLIT_REPAIR_CALLS_PER_GROUP = 3
POST_ANNOTATION_CALLS_PER_GROUP = 0
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_name(value):
    text = _SAFE_NAME.sub("_", str(value)).strip("._")
    return text or "item"


def _write_json(path, data):
    path = Path(path)
    with path.open("x", encoding="utf-8") as stream:
        os.chmod(path, 0o600)
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def load_generation_bundle(path):
    """Load a legacy comparison bundle or a group-only generation bundle."""
    with Path(path).open(encoding="utf-8") as stream:
        bundle = json.load(stream)
    if bundle.get("cases"):
        return load_bundle(path)
    groups = bundle.get("groups")
    if (not isinstance(bundle, dict) or bundle.get("schema_version") != 1
            or bundle.get("cases") != [] or not isinstance(groups, dict)
            or not groups):
        raise ValueError("unsupported generation bundle")
    for group in groups.values():
        if (not isinstance(group, dict)
                or group.get("qa_mode") not in TRACKS
                or not isinstance(group.get("facts"), list)
                or not isinstance(group.get("scope"), dict)):
            raise ValueError("generation bundle has an incomplete group")
    pool = bundle.get("evidence_pool")
    if pool is not None:
        if not isinstance(pool, dict):
            raise ValueError("evidence_pool must be an object")
        for track, item in pool.items():
            scopes = item.get("scopes") if isinstance(item, dict) else None
            if (track not in TRACKS or not isinstance(item, dict)
                    or not isinstance(item.get("facts"), list)
                    or not isinstance(scopes, (dict, list))):
                raise ValueError("evidence_pool has an incomplete track")
    return bundle


def _pool_scopes(value):
    if isinstance(value, dict):
        return [value]
    return [scope for scope in value if isinstance(scope, dict)]


def _source_closure(scopes):
    records = [record for scope in scopes
               for key in ("dialogue", "events", "versions")
               for record in scope.get(key, []) if isinstance(record, dict)]
    exact = {record.get("id") for record in records
             if isinstance(record.get("id"), str)}
    parents = {record.get("parent_id") for record in records
               if isinstance(record.get("parent_id"), str)}
    return exact, parents


def build_shared_evidence_indexes(bundle):
    """Build one read-only evidence index per track and audit source closure."""
    pool = bundle.get("evidence_pool")
    indexes, metadata = {}, {}
    for track in TRACKS:
        if isinstance(pool, dict) and isinstance(pool.get(track), dict):
            facts = list(pool[track].get("facts", []))
            scopes = _pool_scopes(pool[track].get("scopes", []))
            scope_label = "saved_extraction_pool"
        else:
            track_groups = [group for group in bundle["groups"].values()
                            if group.get("qa_mode") == track]
            facts = [fact for group in track_groups for fact in group.get("facts", [])]
            scopes = [group["scope"] for group in track_groups]
            scope_label = "bundle_groups_partial"
        exact, parents = _source_closure(scopes)
        unresolved = {}
        resolved = []
        for fact in facts:
            missing = sorted({source for source in fact.get("sources", [])
                              if isinstance(source, str)
                              and source not in exact and source not in parents})
            if missing:
                unresolved[fact.get("id")] = missing
            else:
                resolved.append(fact)
        indexes[track] = build_evidence_index(
            copy.deepcopy(resolved), copy.deepcopy(scopes), track)
        metadata[track] = {
            "scope": scope_label,
            "input_fact_count": len(facts),
            "indexed_fact_count": len(resolved),
            "unresolved_fact_count": len(unresolved),
            "_unresolved_fact_ids": set(unresolved),
            "unresolved_source_ids": sorted({source for values in unresolved.values()
                                             for source in values}),
        }
    return indexes, metadata


def _public_index_metadata(metadata):
    return {track: {key: (sorted(value) if isinstance(value, set) else value)
                    for key, value in item.items() if not key.startswith("_")}
            for track, item in metadata.items()}


def _checkpoint_writer(directory):
    directory = Path(directory)
    counter = 0

    def save(name, data):
        nonlocal counter
        counter += 1
        path = directory / ("%04d-%s.json" % (counter, _safe_name(name)))
        _write_json(path, data)

    return save


def _csv_values(value, option):
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("%s must contain at least one ID" % option)
    if len(values) != len(set(values)):
        raise ValueError("%s contains duplicate IDs" % option)
    return values


def select_groups(bundle, raw_group_ids=None):
    """Pick the default two groups or an explicit ordered group cohort."""
    groups = bundle["groups"]
    by_track = {}
    for case in bundle["cases"]:
        group_id = case["group_id"]
        group = groups[group_id]
        track = group.get("qa_mode", case["candidate"].get("qa_mode"))
        if track in TRACKS and track not in by_track:
            by_track[track] = group_id

    if raw_group_ids is not None:
        requested = _csv_values(raw_group_ids, "--group-ids")
        unknown = sorted(set(requested) - set(groups))
        if unknown:
            raise ValueError("--group-ids contains unknown group ID(s)")
        selected = []
        for group_id in requested:
            group = groups[group_id]
            track = group.get("qa_mode")
            if track not in TRACKS:
                raise ValueError("selected group has no supported QA track")
            selected.append((track, group_id, group))
        return selected

    missing = [track for track in TRACKS if track not in by_track]
    if missing:
        raise ValueError("bundle has no selected %s group" % ",".join(missing))
    return [(track, by_track[track], groups[by_track[track]]) for track in TRACKS]


def _target_types(group, require_static=False):
    """Return deterministic one-call target types for one evidence group.

    New evidence groups expose ``eligible_types``/``allowed_types``.  The
    fallback keeps the dry-run helper usable with small synthetic groups used
    by offline tests; a real network run will fail closed if no valid simple
    target type is supplied.
    """
    values = group.get("eligible_types")
    if require_static and not values:
        return ()
    values = values or group.get("allowed_types")
    if not values and not require_static:
        values = group.get("scope", {}).get("evidence_group", {}).get("target_types", [])
    if isinstance(values, str):
        values = [values]
    normalized = sorted({value for value in (values or [])
                         if isinstance(value, str) and value.strip()})
    return tuple(normalized) or (() if require_static else (None,))


def _scope_for_target(group, target_type):
    """Copy a group scope and bind its advertised types to one target."""
    scope = copy.deepcopy(group["scope"])
    if target_type is not None:
        evidence_group = dict(scope.get("evidence_group", {}))
        evidence_group["target_types"] = [target_type]
        scope["evidence_group"] = evidence_group
    return scope


def _stage_receipt(directory, stage, client, usage_before, responses_before,
                   result=None, error=None):
    usage = list(getattr(client, "usage", []) or [])
    responses = list(getattr(client, "responses", []) or [])
    _write_json(directory / ("stage-%s.json" % _safe_name(stage)), {
        "stage": stage,
        "usage": usage[usage_before:],
        # ChatClient stores only post-outbound-guard visible content here.
        "responses": responses[responses_before:],
        "status": (result.get("stage_status", {}).get(stage)
                    if isinstance(result, dict) else None),
        "error": error,
    })


def _public_questions(questions):
    """Project approved questions only; retain every raw item in private audit."""
    approved = [question for question in questions
                if isinstance(question, dict)
                and question.get("status") == "approved"]
    return _safe_public_questions(approved, [], workspaces=())


def _apply_static_labels(items, group, evidence_index, target_type):
    """Attach deterministic type/difficulty labels without a provider call."""
    for item in items or []:
        if isinstance(item, dict):
            item.update(static_candidate_labels(
                group, item, evidence_index, target_type))


def _run_target_type(track, group_id, group, target_type, endpoint, model,
                     key_env, timeout, review_mode, target_dir, evidence_index,
                     evidence_index_scope, expansion_budget):
    """Run one independently grounded generation/review target."""
    client = None
    save = _checkpoint_writer(target_dir)
    started = time.monotonic()
    qa_result = None
    review_result = None
    error = None
    static_precheck = None
    expanded_static_precheck = None
    expansion_audits = []
    static_postchecks = []
    candidate_review_guards = []
    active_group = group
    generation_context = None
    try:
        client = ChatClient(endpoint, model, key_env=key_env, timeout=timeout)
        usage_before = len(client.usage)
        responses_before = len(client.responses)
        if review_mode == "simple":
            def generation_checkpoint(phase, name, data):
                save(phase + "-" + name, data)

            generation = generate_simple_target(
                group, evidence_index, target_type, client, track,
                candidate_prefix=(
                    _safe_name(group_id) + "-" + _safe_name(target_type or "unbound") + "-"),
                checkpoint=generation_checkpoint,
                expansion_budget=expansion_budget)
            qa_result = generation["generated"]
            active_group = generation["active_group"]
            static_precheck = generation["static_precheck"]
            expanded_static_precheck = generation["expanded_static_precheck"]
            expansion_audits = generation["expansion_audits"]
            generation_context = generation.get("repair_context")
        else:
            qa_result = generate_from_facts(
                _scope_for_target(group, target_type), copy.deepcopy(group["facts"]),
                client, max_questions=1, qa_mode=track,
                allowed_types=((target_type,) if target_type is not None
                               else group.get("allowed_types")),
                checkpoint=save, candidate_prefix=(
                    _safe_name(group_id) + "-" + _safe_name(target_type or "unbound") + "-"),
                generation_mode="legacy", target_type=target_type)
        _apply_static_labels(
            qa_result.get("all_candidates", []), active_group, evidence_index, target_type)
        _apply_static_labels(
            qa_result.get("questions", []), active_group, evidence_index, target_type)
        if track == "code" and review_mode == "simple":
            checked = []
            checks_by_id = {}
            for question in qa_result.get("questions", []):
                check = static_code_evidence_check(
                    active_group, evidence_index, target_type, candidate=question)
                question["static_code_evidence"] = check
                checks_by_id[question.get("id")] = check
                static_postchecks.append(check)
                if check.get("status") == "insufficient":
                    qa_result.setdefault("rejected", []).append({
                        "question": dict(question, status="rejected"),
                        "reason": "code_evidence_static_insufficient",
                        "static_reason": check.get("reason"),
                        "failed_checks": ["code_evidence_sufficient"],
                        "static_code_evidence": check,
                    })
                else:
                    checked.append(question)
            qa_result["questions"] = checked
            for question in qa_result.get("all_candidates", []):
                check = checks_by_id.get(question.get("id"))
                if check is not None:
                    question["static_code_evidence"] = check
        _stage_receipt(target_dir, "qa", client, usage_before,
                       responses_before, result=qa_result)

        usage_before = len(client.usage)
        responses_before = len(client.responses)
        repair_state = {"remaining": 1}
        def resolve_review_scope(review_candidate):
            projected_group, guard_audit = candidate_review_projection(
                active_group, evidence_index, review_candidate)
            guard_audit.update(
                candidate_id=review_candidate.get("id"),
                target_type=target_type)
            return (_scope_for_target(
                projected_group or active_group, target_type), guard_audit)
        if review_mode == "simple" and not qa_result.get("questions"):
            repaired = repair_simple_validation_rejection(
                _scope_for_target(active_group, target_type),
                copy.deepcopy(active_group["facts"]),
                qa_result.get("rejected", []), client, qa_mode=track,
                checkpoint=save, generation_context=generation_context,
                repair_state=repair_state,
                review_scope_resolver=resolve_review_scope)
            if repaired:
                revision, revised = repaired
                qa_result.setdefault("revisions", []).append(revision)
                if revised is not None:
                    qa_result["rejected"] = []
                    review_result = revised
                elif revision.get("error"):
                    qa_result.setdefault("stage_errors", []).append(
                        revision["error"])
        if review_result is None:
            review_result = review_candidates(
                _scope_for_target(active_group, target_type),
                copy.deepcopy(active_group["facts"]),
                copy.deepcopy(qa_result.get("questions", [])), client,
                qa_mode=track, checkpoint=save, allow_repair=True,
                review_mode=review_mode,
                generation_context=generation_context,
                repair_state=repair_state,
                review_scope_resolver=(resolve_review_scope
                                       if review_mode == "simple" else None))
        candidate_review_guards.extend(
            review_result.get("candidate_review_guards", []))
        if track == "code" and review_mode == "simple":
            final_questions = []
            for question in review_result.get("questions", []):
                check = static_code_evidence_check(
                    active_group, evidence_index, target_type, candidate=question)
                question["static_code_evidence"] = check
                static_postchecks.append(check)
                if check.get("status") == "insufficient":
                    review_result.setdefault("rejected", []).append({
                        "question": dict(question, status="rejected"),
                        "reason": "code_evidence_static_insufficient",
                        "static_reason": check.get("reason"),
                        "failed_checks": ["code_evidence_sufficient"],
                        "static_code_evidence": check,
                        "stage": "static_post_review",
                    })
                else:
                    final_questions.append(question)
            review_result["questions"] = final_questions
        _stage_receipt(target_dir, "review", client, usage_before,
                       responses_before, result=review_result)
        _apply_static_labels(
            review_result.get("questions", []), active_group, evidence_index, target_type)
        for revision in (qa_result.get("revisions", [])
                         + review_result.get("revisions", [])):
            if isinstance(revision, dict) and isinstance(revision.get("after"), dict):
                _apply_static_labels(
                    [revision["after"]], active_group, evidence_index, target_type)
    except Exception as exc:
        error = stage_error("smoke", exc)

    usage = list(getattr(client, "usage", []) or [])
    responses = list(getattr(client, "responses", []) or [])
    if qa_result is None:
        qa_result = {"questions": [], "rejected": [], "facts": [],
                     "stage_errors": [], "stage_status": {}}
    if review_result is None:
        review_result = {"questions": [], "rejected": [], "facts": [],
                         "stage_errors": [], "stage_status": {}}
    public_questions = _public_questions(review_result.get("questions", []))
    audit = {
        "schema_version": 1,
        "track": track,
        "group_id": group_id,
        "allow_repair": True,
        "review_mode": review_mode,
        "qa_result": qa_result,
        "review_result": review_result,
        "responses": responses,
        "static_precheck": static_precheck,
        "expanded_static_precheck": expanded_static_precheck,
        "expansion_audits": expansion_audits if review_mode == "simple" else [],
        "expansion_rounds": (generation["expansion_rounds"]
                             if review_mode == "simple" else 0),
        "expansion_stop_reason": (generation["expansion_stop_reason"]
                                  if review_mode == "simple" else None),
        "evidence_index_scope": evidence_index_scope,
        "static_postchecks": static_postchecks,
        "candidate_review_guards": candidate_review_guards,
        "error": error,
        "wall_seconds": round(time.monotonic() - started, 3),
    }
    usage_payload = {
        "schema_version": 1,
        "track": track,
        "group_id": group_id,
        "review_mode": review_mode,
        "calls": len(usage),
        "usage": usage,
    }
    audit["target_type"] = target_type
    usage_payload["target_type"] = target_type
    _write_json(target_dir / "audit.json", audit)
    _write_json(target_dir / "usage.json", usage_payload)
    return {
        "track": track,
        "group_id": group_id,
        "target_type": target_type,
        "review_mode": review_mode,
        "question_count": len(public_questions),
        "public_questions": public_questions,
        "calls": len(usage),
        "stage_status": {
            "qa": qa_result.get("stage_status", {}).get("qa"),
            "review": review_result.get("stage_status", {}).get("review"),
        },
        "error": error,
    }, usage


def _run_group(track, group_id, group, endpoint, model, key_env, timeout,
               output_dir, review_mode, evidence_index, evidence_index_metadata,
               expansion_budget):
    """Run one group as one private unit per eligible target type."""
    group_dir = Path(output_dir) / _safe_name(group_id)
    group_dir.mkdir(mode=0o700)
    os.chmod(group_dir, 0o700)
    unresolved = evidence_index_metadata.get("_unresolved_fact_ids", set())
    group_unresolved = sorted({fact.get("id") for fact in group.get("facts", [])
                               if fact.get("id") in unresolved})
    if group_unresolved:
        summary = {
            "track": track, "group_id": group_id, "review_mode": review_mode,
            "target_types": [], "units": [], "question_count": 0,
            "public_questions": [], "calls": 0, "error": None,
            "skip_reason": "unresolved_source",
            "unresolved_fact_ids": group_unresolved,
        }
        _write_json(group_dir / "summary.json", summary)
        return summary, []
    target_types = _target_types(group, require_static=review_mode == "simple")
    if not target_types:
        summary = {
            "track": track,
            "group_id": group_id,
            "review_mode": review_mode,
            "target_types": [],
            "units": [],
            "question_count": 0,
            "public_questions": [],
            "calls": 0,
            "error": None,
            "skip_reason": "no_static_eligible_target_type",
        }
        _write_json(group_dir / "summary.json", summary)
        return summary, []
    units, usage = [], []
    for target_type in target_types:
        target_dir = group_dir / _safe_name(target_type or "unbound")
        target_dir.mkdir(mode=0o700)
        os.chmod(target_dir, 0o700)
        summary, receipts = _run_target_type(
            track, group_id, group, target_type, endpoint, model, key_env,
            timeout, review_mode, target_dir, evidence_index,
            evidence_index_metadata.get("scope"), expansion_budget)
        units.append(summary)
        usage.extend(receipts)
    public_questions = [question for unit in units
                        for question in unit.get("public_questions", [])]
    errors = [unit.get("error") for unit in units if unit.get("error")]
    group_summary = {
        "track": track,
        "group_id": group_id,
        "review_mode": review_mode,
        "target_types": [unit.get("target_type") for unit in units],
        "units": units,
        "question_count": len(public_questions),
        "public_questions": public_questions,
        "calls": len(usage),
        "error": errors or None,
    }
    _write_json(group_dir / "summary.json", group_summary)
    return group_summary, usage


def _plan(selected, review_mode="split", expansion_budget=3):
    if review_mode not in REVIEW_MODES:
        raise ValueError("unsupported smoke review mode")
    units = [{"track": track, "group_id": group_id,
              "target_type": target_type}
             for track, group_id, group in selected
             for target_type in _target_types(group, require_static=review_mode == "simple")]
    supplement_calls = (EVIDENCE_SUPPLEMENT_CALLS_PER_GROUP
                        if review_mode == "simple" else 0)
    expansion_calls = expansion_budget * 2 if review_mode == "simple" else 0

    def bounds(track):
        if review_mode == "simple":
            return SIMPLE_BASE_CALLS[track], SIMPLE_REPAIR_CALLS[track]
        return SPLIT_BASE_CALLS_PER_GROUP, SPLIT_REPAIR_CALLS_PER_GROUP

    group_plans = [{
        "track": track,
        "group_id": group_id,
        "target_types": list(_target_types(group, require_static=review_mode == "simple")),
        "units": len(_target_types(group, require_static=review_mode == "simple")),
        "expected_calls_upper_bound": len(_target_types(
            group, require_static=review_mode == "simple")) * (
                sum(bounds(track)) + supplement_calls + expansion_calls),
    } for track, group_id, group in selected]
    base_by_track = {track: bounds(track)[0] for track in TRACKS}
    repair_by_track = {track: bounds(track)[1] for track in TRACKS}
    return {
        "status": "dry_run_no_model_request",
        "groups": [{"track": track, "group_id": group_id}
                   for track, group_id, _ in selected],
        "group_plans": group_plans,
        "target_type_units": units,
        "max_questions_per_group": 1,
        "max_questions_per_target_type": 1,
        "review_mode": review_mode,
        "allow_repair": True,
        # Kept for compatibility with the original two-group plan; the
        # per-target fields above are authoritative when a group has >1 type.
        "base_calls_per_group": max(base_by_track.values()),
        "base_calls_per_target_type": max(base_by_track.values()),
        "base_calls_per_target_type_by_track": base_by_track,
        "repair_calls_per_group_upper_bound": max(repair_by_track.values()),
        "repair_calls_per_target_type_upper_bound": max(repair_by_track.values()),
        "repair_calls_per_target_type_upper_bound_by_track": repair_by_track,
        "evidence_supplement_calls_per_group_upper_bound": (
            supplement_calls),
        "evidence_supplement_calls_per_target_type_upper_bound": (
            supplement_calls),
        "directed_expansion_calls_per_target_type_upper_bound": expansion_calls,
        "generation_calls_per_attempt_upper_bound": (
            2 if review_mode == "simple" else 1),
        "post_annotation_calls_per_group": POST_ANNOTATION_CALLS_PER_GROUP,
        "post_annotation_calls_per_target_type": POST_ANNOTATION_CALLS_PER_GROUP,
        "expected_calls_upper_bound": sum(
            bounds(unit["track"])[0] + bounds(unit["track"])[1]
            + supplement_calls + expansion_calls for unit in units),
    }


def _execute_groups(selected, endpoint, model, key_env, timeout, output_dir,
                    review_mode, evidence_indexes, evidence_index_metadata,
                    expansion_budget=3, parallel_workers=1):
    """Run each selected group exactly once with bounded shared concurrency."""
    completed = []

    def run_group(index, item):
        track, group_id, group = item
        summary, receipts = _run_group(
            track, group_id, group, endpoint, model, key_env, timeout,
            output_dir, review_mode, evidence_indexes[track],
            evidence_index_metadata[track], expansion_budget)
        return index, summary, receipts

    if parallel_workers == 1 or len(selected) < 2:
        completed = [run_group(index, item)
                     for index, item in enumerate(selected)]
    else:
        with ThreadPoolExecutor(max_workers=min(
                parallel_workers, len(selected))) as pool:
            futures = [pool.submit(run_group, index, item)
                       for index, item in enumerate(selected)]
            completed = [future.result()
                         for future in as_completed(futures)]
    summaries = [item[1] for item in sorted(completed)]
    usage = [receipt for _, _, receipts in sorted(completed)
             for receipt in receipts]
    return summaries, usage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--group-ids",
                        help="Optional comma-separated general and code group IDs")
    parser.add_argument("--review-mode", choices=REVIEW_MODES, default="split",
                        help="Split structure/evidence review or simple atomicity/completeness/evidence review")
    parser.add_argument("--output", type=Path,
                        help="New private output directory; required with --allow-network")
    parser.add_argument("--endpoint")
    parser.add_argument("--model")
    parser.add_argument("--key-env", default="BENCHMARK_API_KEY")
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--parallel-workers", type=int, default=1,
                        help="Run independent groups concurrently (1-10)")
    parser.add_argument("--expansion-budget", type=int, default=3,
                        help="Maximum directed evidence expansions per fixed target type")
    parser.add_argument("--allow-network", action="store_true",
                        help="Explicitly permit provider requests")
    args = parser.parse_args(argv)
    if not 1 <= args.parallel_workers <= 10:
        parser.error("--parallel-workers must be between 1 and 10")
    if args.expansion_budget < 0:
        parser.error("--expansion-budget must be non-negative")
    bundle = load_generation_bundle(args.bundle)
    selected = select_groups(bundle, args.group_ids)
    evidence_indexes, evidence_index_metadata = build_shared_evidence_indexes(bundle)
    if not args.allow_network:
        plan = _plan(selected, args.review_mode, args.expansion_budget)
        plan["evidence_indexes"] = _public_index_metadata(evidence_index_metadata)
        print(json.dumps(plan, ensure_ascii=False, sort_keys=True))
        return 0
    if not args.endpoint or not args.model:
        parser.error("--allow-network requires --endpoint and --model")
    if not args.output:
        parser.error("--allow-network requires --output")
    if args.output.exists():
        parser.error("output already exists")
    args.output.mkdir(parents=True, mode=0o700)
    os.chmod(args.output, 0o700)
    summaries, usage = _execute_groups(
        selected, args.endpoint, args.model, args.key_env, args.timeout,
        args.output, args.review_mode, evidence_indexes,
        evidence_index_metadata, args.expansion_budget, args.parallel_workers)
    public = {
        "status": "smoke",
        "review_mode": args.review_mode,
        "allow_repair": True,
        "groups": [{"track": item["track"], "group_id": item["group_id"],
                    "questions": item["public_questions"]}
                   for item in summaries],
    }
    audit = {
        "schema_version": 1,
        "source_run": Path(args.bundle).name,
        "selection_basis": bundle.get("selection_basis"),
        "evidence_indexes": _public_index_metadata(evidence_index_metadata),
        "groups": summaries,
    }
    usage_payload = {
        "schema_version": 1,
        "calls": len(usage),
        "usage": usage,
    }
    _write_json(args.output / "qa-public.json", public)
    _write_json(args.output / "audit.json", audit)
    _write_json(args.output / "usage.json", usage_payload)
    print(json.dumps({
        "status": "completed",
        "output": args.output.name,
        "groups": [{"track": item["track"], "group_id": item["group_id"],
                    "question_count": item["question_count"],
                    "calls": item["calls"]}
                   for item in summaries],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
