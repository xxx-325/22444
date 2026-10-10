"""Sequential composition of the existing project, dialogue and QA commands."""

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from .task_eval.artifacts import read, save
from .task_eval.metrics import scored_task
from .llm import read_usage_ledger
from .task_eval.metrics import cache_usage
from .task_eval.runtime import shared_task_slot

EVALUATION_DEFAULTS = dict(qa_count=8, task_count=1, task_budget=2,
                           parallel_workers=6, task_workers=2, max_task_workers=3,
                           revisions=3,
                           model_request_chars=96000, qa_only=False,
                           general_count=50, code_count=50, design_probe=True,
                           clarification_diagnostic=True,
                           request_timeout=600, agent_seconds=1200)
EXTERNAL_MEMORY_KINDS = ("M1", "M2", "M3", "M4", "M5", "M6")
# Keep one bounded retry for a model-authored stage.  The checkpoint-aware
# path below never discards an accepted prefix.
# A failed stage gets one bounded retry.  Valid dialogue/scenario checkpoints
# remain in place for that retry; later resumes require an explicit run.
MAX_STAGE_RETRIES = 1


def sum_usage(rows):
    rows = list(rows)
    result = dict(requests=0, prompt_tokens=0, completion_tokens=0,
                  transient_failures=0, complete=True)
    for row in rows:
        count = row.get("attempts", row.get("requests", row.get("request_count", 1)))
        result["requests"] += count
        result["transient_failures"] += row.get("transient_failures", 0)
        for key in ("prompt_tokens", "completion_tokens"):
            result[key] += row.get(key, 0)
            if count and not isinstance(row.get(key), int):
                result["complete"] = False
        result["complete"] &= not bool(row.get("usage_missing") or row.get("pending"))
        result["complete"] &= row.get("complete", row.get("usage_complete", True))
    result["total_tokens"] = result["prompt_tokens"] + result["completion_tokens"]
    result.update(cache_usage(rows))
    return result


def episode_usage(root):
    """Read each primary ledger once, before compaction removes stage directories."""
    root = Path(root)
    saved = _read_if(root / "usage.json")
    receipts_by_path = {row["path"]: row for row in saved.get("receipts", [])}
    receipts = []
    qa = root / "qa/manifest.json"
    ledger = root / "qa/usage-ledger.jsonl"
    if ledger.is_file():
        # The ledger records every request of the QA output, including those
        # of interrupted runs that a resumed manifest no longer lists.
        receipt = dict(path="qa/usage-ledger.jsonl", **sum_usage(read_usage_ledger(ledger)))
        if not qa.is_file():
            receipt["complete"] = False
        receipts.append(receipt)
        receipts_by_path.pop("qa/manifest.json", None)
    elif qa.is_file():
        receipts.append(dict(path="qa/manifest.json", **sum_usage(read(qa).get("usage", []))))
    elif (root / "qa").exists():
        receipts.append(dict(sum_usage([]), path="qa/manifest.json", complete=False))
    for directory, dirs, files in os.walk(root / "tasks"):
        dirs[:] = [name for name in dirs if name not in {
            "workspace", "runtime", ".git", "reference-input", "author-reference"}]
        folder = Path(directory)
        for name in sorted(set(files) & {"usage.json", "budget.json"}):
            path = folder / name
            value = read(path)
            rows = value if isinstance(value, list) else [value]
            receipts.append(dict(path=str(path.relative_to(root)), **sum_usage(rows)))
    receipts_by_path.update({row["path"]: row for row in receipts})
    receipts = list(receipts_by_path.values())
    return dict(**sum_usage(receipts), receipts=receipts)


def _read_if(path):
    return read(path) if path.is_file() else {}


def _recover_receipt(receipt):
    """Recover an atomically written receipt left as a temporary file."""
    receipt = Path(receipt)
    if receipt.is_file():
        try:
            return read(receipt)
        except (OSError, ValueError, TypeError):
            pass
    temporary = receipt.with_suffix(receipt.suffix + ".tmp")
    if not temporary.is_file():
        return {}
    try:
        recovered = read(temporary)
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(recovered, dict):
        return {}
    temporary.replace(receipt)
    return recovered


def _dialogue_package_is_current(package, source):
    """Accept a saved dialogue package only when its source has stopped.

    A provider failure may leave a valid prefix package next to a checkpoint
    that is already being resumed.  Reusing that prefix while a new turn is
    in flight would silently send stale history to QA.  Compare the package's
    public cutoff with the source log; this is deliberately a content check,
    not another integrity field.
    """
    package, source = Path(package), Path(source)
    manifest = _read_if(package / "manifest.json")
    cutoff = (manifest.get("dialogue") or {}).get("cutoff_event_id")
    if not cutoff:
        return True
    checkpoint = source / "private/checkpoint.json"
    if checkpoint.is_file():
        try:
            saved = read(checkpoint)
        except (OSError, ValueError, TypeError):
            return False
        if saved.get("in_flight"):
            return False
    session = source / "session.jsonl"
    if not session.is_file():
        return True
    last_id = None
    try:
        for raw in session.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            row = json.loads(raw)
            if row.get("kind") == "assistant" and row.get("phase") not in {
                    None, "final", "commentary"}:
                continue
            if row.get("delivery") == "closing_not_forwarded_to_code":
                continue
            if isinstance(row.get("id"), str):
                last_id = row["id"]
    except (OSError, ValueError, TypeError):
        return False
    return last_id is None or last_id == cutoff


def _scenario_resume_checkpoint(folder):
    """Return whether a scenario has a structurally usable prefix checkpoint.

    The simulator performs the full identity and checksum validation.  The
    collection layer only decides whether preserving this output is safe to
    request; malformed checkpoints fall back to the normal archived retry.
    """
    checkpoint = Path(folder) / "frozen/resume.json"
    if not checkpoint.is_file():
        return False
    try:
        value = read(checkpoint)
    except (OSError, ValueError, TypeError):
        return False
    shape_valid = (
        isinstance(value, dict)
        and value.get("schema") == "scenario-resume-v1"
        and isinstance(value.get("identity"), dict)
        and isinstance(value.get("scenario"), dict)
        and value["scenario"].get("schema") == "continuous-commit-scenario-v1"
        and type(value.get("next_index")) is int
        and value["next_index"] >= 0
    )
    if not shape_valid:
        return False
    # Early checkpoints predate the checksum fields.  The simulator's
    # legacy-recovery path independently reconstructs their accepted prefix;
    # do not discard that prefix here.  Partially present checksum metadata is
    # unsafe and must fall back to an archived retry.
    checksum_fields = (
        value.get("scenario_sha256"),
        value.get("external_plan_sha256"),
        (value.get("pending") or {}).get("raw_draft_sha256")
        if isinstance(value.get("pending"), dict) else None,
    )
    return all(item is None for item in checksum_fields) or (
        isinstance(value.get("scenario_sha256"), str)
        and isinstance(value.get("external_plan_sha256"), str)
    )


def _restore_archived_scenario_checkpoint(collection):
    """Put the last retry checkpoint back at its stable stage path.

    Older collection retries archived the scenario directory before the
    checkpoint-aware retry existed.  The simulator can verify that legacy
    checkpoint independently, so restore only that exact archived directory;
    any other retry remains untouched.
    """
    collection = Path(collection)
    state_path = collection / "collection.json"
    if not state_path.is_file():
        return
    state = read(state_path)
    stages = state.get("stages", [])
    for index in range(len(stages) - 1, -1, -1):
        row = stages[index]
        if row.get("name") != "scenario" or row.get("status") in {"completed", "failed"}:
            continue
        target = collection / row.get("target", "")
        if target.exists():
            continue
        # Retry-exhausted rows keep the archived attempt separately; older
        # rows only exposed the stable path.  Try the explicit archive first,
        # then walk earlier archived attempts if the last attempt was empty.
        names = [row.get("retry_exhausted_path"), row.get("path")]
        if row.get("status") == "retry_exhausted":
            names.extend(
                prior.get("path")
                for prior in reversed(stages[:index])
                if prior.get("name") == "scenario"
            )
        archived = next(
            (collection / name for name in names
             if isinstance(name, str) and _scenario_resume_checkpoint(collection / name)),
            None,
        )
        if archived is None:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(archived), str(target))
        row["status"] = "failed"
        row["retry"] = max(0, MAX_STAGE_RETRIES - 1)
        row["path"] = str(target.relative_to(collection))
        row["restored_from_retry_exhausted"] = True
        row["restored_from"] = str(archived.relative_to(collection))
        save(state_path, state)
        return


def _dialogue_resume_checkpoint(folder):
    checkpoint = Path(folder) / "private/checkpoint.json"
    if not checkpoint.is_file():
        return False
    try:
        value = read(checkpoint)
    except (OSError, ValueError, TypeError):
        return False
    return (
        isinstance(value, dict)
        and isinstance(value.get("tasks"), list)
        and isinstance(value.get("public"), list)
        and isinstance(value.get("state"), dict)
        and value.get("schema", "").startswith("openhands-progressive-")
    )


def _dialogue_round_count(folder):
    try:
        value = read(Path(folder) / "private/checkpoint.json")
    except (OSError, ValueError, TypeError):
        return -1
    pending = 0
    rounds = 0
    for row in value.get("public", []):
        if row.get("kind") == "user" and row.get("text", "").strip():
            pending += 1
        elif (
            row.get("kind") == "assistant"
            and row.get("phase", "final") == "final"
            and row.get("text", "").strip()
            and pending
        ):
            pending -= 1
            rounds += 1
    return rounds


def _restore_archived_dialogue_checkpoint(collection):
    """Restore the richest interrupted dialogue checkpoint before retrying."""
    collection = Path(collection)
    state_path = collection / "collection.json"
    if not state_path.is_file():
        return
    state = read(state_path)
    stages = state.get("stages", [])
    for index in range(len(stages) - 1, -1, -1):
        row = stages[index]
        if row.get("name") != "dialogue" or row.get("status") == "completed":
            continue
        target = collection / row.get("target", "")
        candidates = [row.get("retry_exhausted_path"), row.get("path")]
        candidates.extend(
            prior.get("path")
            for prior in reversed(stages[:index])
            if prior.get("name") == "dialogue"
        )
        folders = [
            collection / name for name in candidates
            if isinstance(name, str) and _dialogue_resume_checkpoint(collection / name)
        ]
        if target.exists() and _dialogue_resume_checkpoint(target):
            folders.append(target)
        if not folders:
            continue
        archived = max(folders, key=_dialogue_round_count)
        if target.exists() and archived != target:
            backup = collection / "attempts" / row.get("target", "dialogue") / "recovery-current"
            suffix = 1
            while backup.exists():
                backup = backup.with_name("recovery-current-%d" % suffix)
                suffix += 1
            backup.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            shutil.move(str(target), str(backup))
        if archived != target:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(archived), str(target))
        row["status"] = "failed"
        row["retry"] = max(0, MAX_STAGE_RETRIES - 1)
        row["path"] = str(target.relative_to(collection))
        row["restored_from"] = str(archived.relative_to(collection))
        save(state_path, state)
        return


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _accepted_receipt(result, expected):
    status = result.get("status")
    if status in {"blocked", "failed", "rejected"}:
        return False
    if status in expected:
        return True
    if status == "completed_with_warnings":
        return "completed" in expected or result.get("schema") in expected
    return status in {None, "completed"} and result.get("schema") in expected


def _command(command, cwd, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        log.chmod(0o600)
        return subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT).returncode


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", value):
        raise ValueError("Use short lowercase project and scenario identifiers")
    return value


def _aggregate_status(statuses, empty="no_scenarios"):
    """Collapse child outcomes without treating an empty set as success."""
    statuses = list(statuses)
    if not statuses:
        return empty
    # ``qa_only`` is a successful terminal route: it deliberately skips
    # repository tasks and paired execution while still producing QA.
    if all(status in {"completed", "completed_with_warnings", "qa_only", "dialogue_only"}
           for status in statuses):
        return "completed_with_warnings" if "completed_with_warnings" in statuses else "completed"
    if any(status in {"failed", "stopped", "interrupted", "project_rejected",
                      "scenario_rejected", "requirements_rejected",
                      "dialogue_incomplete", "evaluation_failed", "partial_failure"}
           for status in statuses):
        return "partial_failure"
    if any(status == "below_target" for status in statuses):
        return "below_target"
    return "partial_failure"


def _terminal_collection_status(state):
    """Return a terminal status without promoting local failures to blockers."""
    aggregate = _aggregate_status(
        project.get("status") for project in state.get("projects", []))
    if aggregate == "completed" and not state.get("warnings"):
        return "completed"
    if aggregate in {"completed", "completed_with_warnings", "partial_failure", "below_target", "no_scenarios"}:
        return "completed_with_warnings"
    return "blocked"


def plan_external_information(projects):
    """Create the control-side opportunity budget before repository work.

    The plan describes opportunities to try, not a required number of
    accepted facts. Business agreements are designed before repository work;
    implementation and runtime observations must come from the actual project.
    Neither becomes QA evidence until disclosed in the public conversation.
    """
    rows = []
    for project in projects:
        for scenario in project.get("scenarios", []):
            kinds = tuple(scenario.get("memory_kinds", EXTERNAL_MEMORY_KINDS))
            budget = scenario.get("external_attempt_budget", 0)
            if type(budget) is not int or budget < 0:
                raise ValueError("external_attempt_budget must be a nonnegative integer")
            if not kinds or any(kind not in EXTERNAL_MEMORY_KINDS for kind in kinds):
                raise ValueError("memory_kinds must select M1..M6")
            increments = project.get("increments", 0)
            if type(increments) is not int or increments < 0:
                raise ValueError("increments must be a nonnegative integer")
            stage_count = max(1, increments)
            raw_distribution = scenario.get("external_attempt_distribution")
            if raw_distribution is None:
                distribution = {kind: budget for kind in kinds}
                stages = stage_count
                stage_base, stage_remainder = divmod(budget, stages)
                opportunities = [
                    {
                        "stage": index + 1,
                        "behavior": str(scenario.get("brief", project.get("brief", ""))).strip(),
                        "memory_kinds": list(kinds),
                        "attempts": stage_base + (index < stage_remainder),
                    }
                    for index in range(stages)
                    if stage_base + (index < stage_remainder)
                ]
            else:
                if not isinstance(raw_distribution, list) or not raw_distribution:
                    raise ValueError("external_attempt_distribution must be a nonempty list")
                opportunities = []
                distribution = {kind: 0 for kind in kinds}
                total = 0
                for row in raw_distribution:
                    if not isinstance(row, dict):
                        raise ValueError("external attempt distribution rows must be objects")
                    allowed = row.get("memory_kinds", list(kinds))
                    attempts = row.get("attempts")
                    stage = row.get("stage")
                    behavior = row.get("behavior")
                    if (type(stage) is not int or stage < 1
                            or type(attempts) is not int or attempts < 0
                            or not isinstance(behavior, str) or not behavior.strip()
                            or not isinstance(allowed, list) or not allowed
                            or any(kind not in kinds for kind in allowed)):
                        raise ValueError("invalid external attempt distribution row")
                    opportunities.append({
                        "stage": stage,
                        "behavior": behavior.strip(),
                        "memory_kinds": list(dict.fromkeys(allowed)),
                        "attempts": attempts,
                    })
                    total += attempts
                    for kind in dict.fromkeys(allowed):
                        distribution[kind] += attempts
                if total != budget:
                    raise ValueError(
                        "external attempt distribution must sum to external_attempt_budget")
            # Opportunity stages describe business development, not a required
            # chain of prebuilt implementation commits.
            rows.append({
                "project": project["id"],
                "scenario": scenario["id"],
                "attempt_budget": budget,
                "memory_kinds": list(kinds),
                "distribution": distribution,
                "opportunities": opportunities,
                "status": "planned",
            })
    return {"schema": "external-information-plan-v2", "scenarios": rows}


def _external_attempt_stats(scenario_plan, report=None):
    """Keep observed counters even when scenario preparation is incomplete."""
    budget = scenario_plan.get("attempt_budget", 0)
    counts = {
        "planned": budget,
        "started": 0,
        "candidate_groups": 0,
        "candidate_facts": 0,
        "accepted_facts": 0,
        "unused_or_rejected": budget,
    }
    report = report or {}
    attempts = report.get("attempts", {})
    counts.update({key: attempts[key] for key in counts if key in attempts})
    outcomes = attempts.get("opportunity_outcomes", [])
    counts["unused_attempts"] = max(0, counts["planned"] - counts["started"])
    counts["rejected_facts"] = sum(len(row.get("rejected_fact_ids", [])) for row in outcomes)
    if "accepted_facts" not in attempts:
        counts["accepted_facts"] = sum(
            value.get("facts", 0) for value in report.get("memory_counts", {}).values()
            if isinstance(value, dict))
    counts["opportunity_outcomes"] = outcomes
    return counts


def _merge_dynamic_attempt_stats(attempts, dialogue_root):
    """Fold live follow-up opportunity receipts into the collection counters."""
    root = Path(dialogue_root) / "private" / "followups"
    receipts = sorted(root.glob("task-*/scenario/attempts.json"))
    if not receipts:
        return attempts
    merged = dict(attempts)
    for path in receipts:
        try:
            row = read(path)
        except (OSError, ValueError, TypeError):
            continue
        for key in ("started", "candidate_groups", "candidate_facts", "accepted_facts"):
            merged[key] = merged.get(key, 0) + int(row.get(key.replace("started", "started_attempts"),
                                                     row.get(key, 0)) or 0)
        merged["rejected_facts"] = merged.get("rejected_facts", 0) + len(
            row.get("rejected_fact_ids", []))
        merged["unused_or_rejected"] = merged.get("unused_or_rejected", 0) + int(
            row.get("unused_or_rejected", 0) or 0)
    merged["unused_attempts"] = max(0, merged.get("planned", 0) - merged.get("started", 0))
    return merged


def _completed_pair(task):
    """Count only a fully evaluated, scored pair with terminal arm results."""
    if task.get("status") != "evaluated" or not scored_task(task):
        return False
    comparison = task.get("comparison", {})
    if set(comparison) != {"without_memory", "with_memory"}:
        return False
    return all(
        trial.get("result") in {"passed", "failed"}
        and trial.get("execution_status") != "interrupted"
        for trial in comparison.values()
    )


def _transient_stage_failure(text):
    """Identify a bounded provider/gateway failure safe to retry once."""
    return bool(re.search(
        r"relay request failed|badgatewayerror|conversationrunerror|provider[_ ]503|gateway error",
        text,
        re.IGNORECASE,
    ))


def _authentication_failure(text):
    """Match explicit provider authentication failures, not model prose."""
    patterns = (
        r"\b(?:http(?:error)?|http_status|status(?:_code)?|response\s+status)\s*[:=]?\s*4(?:01|03)\b",
        r"\b4(?:01|03)\b[^\n]{0,60}\b(?:unauthorized|forbidden)\b",
        r"\b(?:invalid|missing|expired|revoked)\s+(?:api[_ -]?key|access[_ -]?token|bearer(?:\s+token)?|credentials?)\b",
        r"\b(?:api[_ -]?key|access[_ -]?token|bearer(?:\s+token)?|credentials?)\s*(?:is\s+)?(?:invalid|missing|expired|revoked)\b",
        r"\b(?:missing_api_key|invalid_api_key|authentication_error)\b",
        r"\bmissing\s+model\s+credential\b",
    )
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def validate_plan(plan):
    if not plan.get("projects") or not plan.get("runtime_config"):
        raise ValueError("A runtime_config and fixed projects list are required")
    sources = plan.get("qa_sources", ["external"])
    if (not isinstance(sources, list) or not sources
            or any(source not in ("graph", "external") for source in sources)
            or len(set(sources)) != len(sources)):
        raise ValueError("qa_sources must select distinct graph/external routes")
    quality = plan.get("dialogue_quality")
    if (quality is not None and not isinstance(quality, dict)
            and quality not in {"scale", "long_dialogue"}):
        raise ValueError("dialogue_quality must be 'scale', 'long_dialogue', or a quality mapping")
    if isinstance(quality, dict) and not quality:
        raise ValueError("dialogue_quality mapping must not be empty")
    if isinstance(quality, dict):
        minimums = {"min_increments", "min_visible_messages", "min_public_tool_events",
                    "min_external_events", "min_distinct_external_foci",
                    "min_user_code_rounds"}
        flags = {"require_external_event_closure", "require_declared_external_scope",
                 "forbid_memory_docs"}
        for key, value in quality.items():
            if key in minimums:
                if type(value) is not int or value < 0:
                    raise ValueError(key + " must be a nonnegative integer")
            elif key in flags:
                if type(value) is not bool:
                    raise ValueError(key + " must be boolean")
            else:
                raise ValueError("Unknown dialogue_quality option: " + key)
    for key in ("max_total_requests", "max_total_tokens"):
        if type(plan.get(key)) is not int or plan[key] <= 0:
            raise ValueError(key + " must be a positive stage-admission budget")
    for key, value in plan.get("evaluation", {}).items():
        if key not in EVALUATION_DEFAULTS and key != "group_budget":
            raise ValueError("Unknown evaluation option: " + key)
        if key == "qa_only":
            if type(value) is not bool:
                raise ValueError("qa_only must be a boolean")
            continue
        if key in {"request_timeout", "agent_seconds"}:
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or value <= 0):
                raise ValueError(key + " must be positive")
            continue
        if type(value) is not int or value <= 0:
            raise ValueError(key + " must be positive")
    seen = set()
    for project in plan["projects"]:
        name = _identifier(project.get("id"))
        if name in seen:
            raise ValueError("Duplicate project id")
        seen.add(name)
        if bool(project.get("brief")) == bool(project.get("prepared_config")):
            raise ValueError("Choose a business brief or an existing prepared_config")
        if not project.get("scenarios"):
            raise ValueError("Each project requires a fixed scenario list")
        scenarios = [_identifier(s.get("id")) for s in project["scenarios"]]
        if len(set(scenarios)) != len(scenarios):
            raise ValueError("Duplicate scenario id")
    plan_external_information(plan["projects"])


def run_collection(plan_path, output, simulator, env_file, python=sys.executable,
                   resume=False, dialogue_only=False):
    plan_path, output, simulator = (Path(p).resolve() for p in (plan_path, output, simulator))
    # Keep a virtualenv launcher symlink intact; resolving it loses the
    # environment's site-packages and can silently switch to bare Python.
    python = Path(python).absolute()
    plan = read(plan_path)
    validate_plan(plan)
    runtime_path = (plan_path.parent / plan["runtime_config"]).resolve()
    runtime = read(runtime_path)
    for role in ("user", "code", "judge", "decomposer"):
        if isinstance(runtime.get(role), dict):
            runtime[role]["max_output_tokens"] = None
    identity = dict(plan_sha256=_sha256(plan_path), runtime_sha256=_sha256(runtime_path),
                    simulator=str(simulator), python=str(python),
                    dialogue_only=bool(dialogue_only or plan.get("dialogue_only", False)),
                    prepared_configs={p["id"]: _sha256((plan_path.parent / p["prepared_config"]).resolve())
                                      for p in plan["projects"] if p.get("prepared_config")})
    if resume:
        state = read(output / "collection.json")
        if state.get("identity") != identity:
            raise ValueError("Collection resume requires the same plan, runtime and prepared inputs")
        _restore_archived_scenario_checkpoint(output)
        _restore_archived_dialogue_checkpoint(output)
        state = read(output / "collection.json")
    else:
        output.mkdir(parents=True, exist_ok=False, mode=0o700)
        state = dict(status="running", projects=[], stages=[], identity=identity,
                 qa_only=plan.get("evaluation", {}).get("qa_only", False),
                 dialogue_only=bool(dialogue_only or plan.get("dialogue_only", False)),
                 qa_sources=plan.get("qa_sources", ["external"]),
                 plan_sha256=identity["plan_sha256"],
                 limits={k: plan[k] for k in ("max_total_requests", "max_total_tokens")},
                 budget_boundary="Finish each started stage, then check cumulative usage before the next stage")
        save(output / "plan.json", plan)
        information_plan = plan_external_information(plan["projects"])
        save(output / "external-information-plan.json", information_plan)
        state["external_information_plan"] = "external-information-plan.json"
    state["status"] = "running"
    dialogue_only = bool(dialogue_only or state.get("dialogue_only", False))
    state["dialogue_only"] = dialogue_only
    information_plan_path = output / state.get(
        "external_information_plan", "external-information-plan.json")
    if "external_information_plan" not in state or not information_plan_path.is_file():
        information_plan = plan_external_information(plan["projects"])
        save(output / "external-information-plan.json", information_plan)
        state["external_information_plan"] = "external-information-plan.json"
    state.pop("stop_reason", None)
    state.pop("error_type", None)
    state.setdefault("warnings", [])

    def warning(code, **details):
        item = dict(code=code, **details)
        if item not in state["warnings"]:
            state["warnings"].append(item)
        state["usage_status"] = "pending" if code == "usage_incomplete" else state.get(
            "usage_status", "ready")

    def persist():
        state["usage"] = sum_usage([s.get("usage", dict(sum_usage([]), complete=False))
                                    for s in state["stages"]])
        state["usage_status"] = "ready" if state["usage"]["complete"] else "pending"
        save(output / "collection.json", state)

    def stage(name, folder, command, cwd, receipt, expected, ledger, budget_key=None):
        def next_archive(target, requested):
            """Choose a free diagnostic directory after an interrupted retry."""
            archive_root = output / "attempts" / target
            archive_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            number = int(requested)
            candidate = archive_root / ("attempt-%d" % number)
            while candidate.exists():
                number += 1
                candidate = archive_root / ("attempt-%d" % number)
            candidate.mkdir(parents=True, exist_ok=False, mode=0o700)
            return candidate

        def usage():
            budget = _read_if(ledger)
            if name == "dialogue" and not budget:
                budget = {"budget": _read_if(folder / "private/budget.json")}
            if budget_key:
                budget = budget.get(budget_key)
            return sum_usage([budget]) if budget else dict(sum_usage([]), complete=False)

        target = str(folder.relative_to(output))
        # The config file is enriched as upstream stages complete; command and
        # workspace identity remain stable across that expected mutation.
        stage_identity = dict(command=command, cwd=str(cwd))
        previous = next((s for s in reversed(state["stages"])
                         if s.get("target", s["path"]) == target), None)
        if previous:
            if previous.get("identity") != stage_identity:
                raise ValueError("Collection stage inputs changed: " + target)
            # A dialogue can export a usable warning package after a
            # transient provider failure.  Reuse that package on resume
            # instead of sending the dialogue again.
            if (name == "dialogue"
                    and previous.get("status") in {"failed", "interrupted"}
                    and not previous.get("terminal")):
                recovered = _recover_receipt(receipt)
                if (_accepted_receipt(recovered, expected)
                        and (name != "dialogue"
                             or _dialogue_package_is_current(receipt.parent, folder))):
                    previous["status"] = "completed"
                    previous["outcome"] = recovered.get(
                        "status", recovered.get("schema", "completed"))
                    previous["output_ready"] = True
                    previous["result"] = "completed_with_warnings"
                    previous["usage"] = usage()
                    persist()
                    return recovered
            if previous["status"] == "completed" or (
                    previous["status"] == "failed" and previous.get("terminal")):
                if previous.get("receipt_sha256") != _sha256(receipt):
                    raise ValueError("Collection stage receipt changed: " + target)
                if previous["status"] == "completed":
                    result = read(receipt)
                    if not _accepted_receipt(result, expected):
                        raise ValueError("Collection stage receipt is no longer accepted: " + target)
                    return result
                return None
            previous["usage"] = usage()
            previous["status"] = "interrupted"
        persist()
        if not state["usage"]["complete"]:
            warning("usage_incomplete")
        if (state["usage"]["requests"] >= plan["max_total_requests"]
                or state["usage"]["total_tokens"] >= plan["max_total_tokens"]):
            raise RuntimeError("collection_budget_exhausted")
        completed_qa = all((folder / "qa" / filename).is_file()
                           for filename in ("manifest.json", "qa-public.json"))
        if previous and name == "evaluation" and completed_qa:
            # The episode owns its append-only QA/task recovery and cumulative ledger.
            row = previous
            row["status"] = "running"
            row["resume_count"] = row.get("resume_count", 0) + 1
            command = [*command, "--resume"]
        elif (previous and name == "scenario"
              and previous.get("resume_count", 0) < MAX_STAGE_RETRIES
              and _scenario_resume_checkpoint(folder)):
            # The simulator validates checkpoint identity and checksums. Keep
            # the accepted prefix in place and charge all calls to its one
            # cumulative budget journal rather than creating a new attempt.
            row = previous
            row["status"] = "running"
            row["resume_count"] = row.get("resume_count", 0) + 1
            retry_log = folder.with_suffix(".log")
            if retry_log.is_file():
                diagnostics = output / "attempts" / target
                diagnostics.mkdir(parents=True, exist_ok=True, mode=0o700)
                suffix = row["resume_count"]
                diagnostic = diagnostics / ("resume-%d.log" % suffix)
                while diagnostic.exists():
                    suffix += 1
                    diagnostic = diagnostics / ("resume-%d.log" % suffix)
                retry_log.rename(diagnostic)
                row["retry_diagnostic"] = str(diagnostic.relative_to(output))
            command = [*command, "--resume-existing"]
        elif (previous and name == "dialogue"
              and _dialogue_resume_checkpoint(folder)
              and (resume or previous.get("resume_count", 0) < MAX_STAGE_RETRIES)):
            row = previous
            row["status"] = "running"
            row["resume_count"] = row.get("resume_count", 0) + 1
            command = [*command, "--resume"]
        elif (
            previous
            and name == "dialogue"
            and previous.get("resume_count", 0) < MAX_STAGE_RETRIES
            and (
                _restore_archived_dialogue_checkpoint(output) is None
                and _dialogue_resume_checkpoint(folder)
            )
        ):
            # A failed child may have left its valid checkpoint in an
            # archived attempt. Restore it before the automatic retry so
            # the dialogue continues instead of silently starting over.
            row = previous
            row["status"] = "running"
            row["resume_count"] = row.get("resume_count", 0) + 1
            command = [*command, "--resume"]
        else:
            if previous:
                if previous.get("retry", 0) >= MAX_STAGE_RETRIES:
                    archive = next_archive(target, previous.get("retry", 0) + 1)
                    for artifact in (folder, folder.with_suffix(".log"),
                                     folder.with_name("dialogue-package") if name == "dialogue" else None):
                        if artifact is not None and artifact.exists():
                            artifact.rename(archive / artifact.name)
                    previous["status"] = "retry_exhausted"
                    previous["receipt_sha256"] = _sha256(receipt)
                    previous["retry_exhausted_path"] = str(archive.relative_to(output))
                    persist()
                    return None
                archive = next_archive(target, int(previous.get("retry", 0)) + 1)
                for artifact in (folder, folder.with_suffix(".log"),
                                 folder.with_name("dialogue-package") if name == "dialogue" else None):
                    if artifact is not None and artifact.exists():
                        artifact.rename(archive / artifact.name)
                previous["path"] = str((archive / folder.name).relative_to(output))
                previous["archived"] = True
                persist()
                row = dict(name=name, path=target, target=target, status="running", command=command,
                           identity=stage_identity,
                           retry=(int(previous.get("retry", 0)) + 1) if previous else 0)
            else:
                row = dict(name=name, path=target, target=target, status="running", command=command,
                           identity=stage_identity, retry=0)
            state["stages"].append(row)
        persist()
        print("Collection:", row["path"], name, flush=True)
        try:
            if name == "dialogue":
                with shared_task_slot():
                    row["returncode"] = _command(command, cwd, folder.with_suffix(".log"))
            else:
                row["returncode"] = _command(command, cwd, folder.with_suffix(".log"))
            result = _recover_receipt(receipt)
            row["outcome"] = result.get("status", result.get("schema", "missing_receipt"))
            text = ""
            if row["returncode"]:
                log = folder.with_suffix(".log")
                text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
                if _authentication_failure(text):
                    raise RuntimeError("collection_authentication_failed")
                if re.search(r"unrecognized arguments|the following arguments are required", text, re.I):
                    raise RuntimeError("collection_input_failed")
                hard_codes = (
                    "source_closure_failed", "credential_detected", "private_content_leaked",
                    "snapshot_corrupt", "identity_mismatch", "host_unavailable",
                    "collection_budget_exhausted", "total_budget_exhausted",
                )
                diagnostics = "\n".join([text, *(
                    str(result.get(key, "")) for key in ("stop_reason", "reason", "error_type"))])
                blocker = next((code for code in hard_codes
                                if re.search(r"\b" + code + r"\b", diagnostics)), None)
                if blocker:
                    raise ValueError("Collection stage hard failure: " + blocker + " (" + target + ")")
            row["status"] = "completed" if _accepted_receipt(result, expected) else "failed"
            if row["status"] == "completed" and row["returncode"]:
                row["result"] = "completed_with_warnings"
                warning("stage_nonzero_exit", stage=target, returncode=row["returncode"],
                        outcome=row["outcome"])
            if row["status"] == "completed" and row["outcome"] == "completed_with_warnings":
                warning("stage_output_warnings", stage=target,
                        warnings=result.get("quality", {}).get("warnings", result.get("warnings", [])))
            row["terminal"] = row["returncode"] == 0 and row["status"] == "failed"
            if (row["status"] == "failed"
                    and row["outcome"] in {"missing_receipt", "timeout",
                                           "connection_error", "protocol_error",
                                           "request_budget"}
                    and row.get("retry", 0) < MAX_STAGE_RETRIES):
                row["status"] = "failed"
                row["terminal"] = False
                persist()
                retry_command = command[:-1] if command and command[-1] in {
                    "--resume", "--resume-existing"
                } else command
                return stage(name, folder, retry_command, cwd, receipt, expected, ledger, budget_key)
            if row["returncode"] and row["status"] != "completed":
                retryable = (
                    _transient_stage_failure(text)
                    or row["outcome"] in {"missing_receipt", "timeout", "connection_error",
                                           "protocol_error", "request_budget"}
                )
                if retryable and row.get("retry", 0) < MAX_STAGE_RETRIES:
                    # Preserve the failed attempt, then rerun this same
                    # stage from its saved inputs.  A local provider failure
                    # must not require a manual resume.
                    row["status"] = "failed"
                    persist()
                    retry_command = command[:-1] if command and command[-1] in {
                        "--resume", "--resume-existing"
                    } else command
                    return stage(name, folder, retry_command, cwd, receipt, expected, ledger, budget_key)
                warning("stage_needs_review", stage=target, outcome=row["outcome"],
                        returncode=row["returncode"])
            elif row["status"] != "completed":
                warning("stage_needs_review", stage=target, outcome=row["outcome"],
                        returncode=row["returncode"])
            return result if row["status"] == "completed" else None
        except BaseException as error:
            row.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                       error_type=type(error).__name__)
            if (not isinstance(error, Exception) or isinstance(error, (ValueError, FileNotFoundError))
                    or str(error) in {"collection_authentication_failed", "collection_input_failed"}):
                raise
            return None
        finally:
            try:
                row["usage"] = usage()
            except (OSError, ValueError, TypeError) as error:
                row["usage"] = dict(sum_usage([]), complete=False, error_type=type(error).__name__)
            row["receipt_sha256"] = _sha256(receipt)
            row["output_ready"] = (row["status"] == "completed"
                                   and _accepted_receipt(_recover_receipt(receipt), expected))
            row["usage_ready"] = bool(row["usage"].get("complete"))
            persist()

    def upstream(module, config, target):
        return [str(python), "-m", module, "--config", str(config),
                "--env-file", str(env_file), "--output", str(target)]

    try:
        for project in plan["projects"]:
            root = output / project["id"]
            root.mkdir(exist_ok=resume)
            project_information_plan = {
                "schema": "external-information-plan-v2",
                "project": project["id"],
                "scenarios": [row for row in plan_external_information([project])["scenarios"]],
            }
            save(root / "external-information-plan.json", project_information_plan)
            entry = next((p for p in state["projects"] if p["id"] == project["id"]), None)
            if entry is None:
                entry = dict(id=project["id"], status="running", scenarios=[])
                state["projects"].append(entry)
            entry["status"] = "running"
            if project.get("prepared_config"):
                config = read((plan_path.parent / project["prepared_config"]).resolve())
                for role in ("user", "code", "judge", "decomposer"):
                    config.setdefault(role, {}).update(runtime.get(role, {}))
                entry["preparation"] = "reused; original cost not charged to this collection"
            else:
                config = copy.deepcopy(runtime)
                config["project"] = dict(brief=project["brief"], increments=project.get("increments", 0))
                config["project"]["behavior_goals"] = list(dict.fromkeys(
                    row["behavior"] for scene in project_information_plan["scenarios"]
                    for row in scene["opportunities"] if row["behavior"]))
                config["project"]["business_context"] = "\n".join(
                    scenario.get("brief", "") for scenario in project["scenarios"])
                cfg = root / "project-input.json"
                save(cfg, config)
                target = root / "project"
                result = stage("project", target, upstream("simulator.openhands.prepare_project", cfg, target),
                               simulator, target / "project.json", {"completed"}, target / "private/budget.json")
                if result is None:
                    entry["status"] = "project_rejected"
                    warning("project_rejected", project=project["id"])
                    continue
                config = read(target / "config.json")
            if plan.get("dialogue_quality"):
                config["dialogue_quality"] = plan["dialogue_quality"]
            repo = Path(config["repository"])
            for role in ("user", "code", "judge", "decomposer"):
                if isinstance(config.get(role), dict):
                    config[role]["max_output_tokens"] = None
            entry["lineage"] = subprocess.check_output(
                ["git", "rev-list", "--max-parents=0", config["base"]], cwd=repo, text=True).splitlines()
            entry["base"] = config["base"]
            for scenario in project["scenarios"]:
                scenario_root = root / scenario["id"]
                scenario_root.mkdir(exist_ok=resume)
                scenario_plan = next(
                    row for row in project_information_plan["scenarios"]
                    if row["scenario"] == scenario["id"]
                )
                current = copy.deepcopy(config)
                current.pop("scenario_file", None)
                current.pop("prepared_issues", None)
                # Keep opportunity budgets and distributions control-side.  A
                # scenario author receives the natural business brief, never
                # the host's counters, IDs, or acceptance quotas.
                private_keys = {
                    "id", "external_fact_target", "external_attempt_budget",
                    "external_attempt_distribution", "preplan_external_facts",
                }
                current["scenario_design"] = {
                    k: v for k, v in scenario.items() if k not in private_keys
                }
                current["external_attempt_plan"] = scenario_plan
                # Retain the artifact name used by existing reports, but it is
                # a control-side plan and is not included in scenario_design.
                current["external_information_plan"] = scenario_plan
                cfg = scenario_root / "config.json"
                if not (resume and cfg.is_file()):
                    save(cfg, current)
                record = next((s for s in entry["scenarios"] if s["id"] == scenario["id"]), None)
                if record is None:
                    record = dict(id=scenario["id"], status="running")
                    entry["scenarios"].append(record)
                record["status"] = "running"
                record["external_information_plan"] = scenario_plan
                # Seed counters before invoking the scenario model.  A
                # rejected stage must still report its planned opportunities
                # and zero observed facts instead of dropping coverage data.
                record["external_attempts"] = _external_attempt_stats(scenario_plan)
                record["external_information_coverage"] = {
                    "attempt_budget": scenario_plan.get("attempt_budget", 0),
                    "actual": 0,
                    "distribution": {kind: 0 for kind in EXTERNAL_MEMORY_KINDS},
                }
                persist()
                target = scenario_root / "scenario"
                designed = stage("scenario", target, upstream("simulator.openhands.prepare_scenario", cfg, target),
                    simulator, target / "frozen/report.json", {"completed", "candidate_pass"}, target / "budget.json")
                frozen_report = _recover_receipt(target / "frozen/report.json")
                record["external_attempts"] = _external_attempt_stats(scenario_plan, frozen_report)
                if frozen_report.get("warnings"):
                    warning("scenario_coverage_warnings", project=project["id"],
                            scenario=scenario["id"], warnings=frozen_report["warnings"])
                if designed is None:
                    record["status"] = "scenario_rejected"
                    warning("scenario_rejected", project=project["id"], scenario=scenario["id"])
                    continue
                record["requested_memory_kinds"] = designed.get("design", {}).get("memory_kinds", [])
                record["designed_memory_counts"] = designed.get("memory_counts", {})
                current["scenario_file"] = str(target / "frozen/scenario.json")
                save(cfg, current)
                target = scenario_root / "requirements"
                if stage("requirements", target, upstream("simulator.openhands.prepare_progressive", cfg, target),
                         simulator, target / "report.json", {"candidate_pass"}, target / "budget.json") is None:
                    record["status"] = "requirements_rejected"
                    warning("requirements_rejected", project=project["id"], scenario=scenario["id"])
                    continue
                current["prepared_issues"] = str(target / "report.json")
                save(cfg, current)
                target = scenario_root / "dialogue"
                package = target.with_name("dialogue-package")
                if stage("dialogue", target, upstream("simulator", cfg, target), simulator,
                         package / "manifest.json", {"memory-episode-v1"},
                         package / "private/review.json", "budget") is None:
                    record["status"] = "dialogue_incomplete"
                    warning("dialogue_incomplete", project=project["id"], scenario=scenario["id"])
                    continue
                from .episode_input import load_episode_manifest
                exported = load_episode_manifest(package / "manifest.json")
                manifest_record = exported.get("manifest", {})
                manifest_quality = manifest_record.get("quality", {})
                quality_status = manifest_record.get("status")
                record["quality_status"] = quality_status or "completed"
                if quality_status == "completed_with_warnings":
                    warning("dialogue_quality_warnings", project=project["id"],
                            scenario=scenario["id"],
                            warnings=manifest_quality.get("warnings", []))
                events = read(exported["external_events"])["events"] if exported.get("external_events") else []
                record["public_memory_counts"] = {kind: sum(e.get("memory_kind") == kind for e in events)
                                                   for kind in ("M1", "M2", "M3", "M4", "M5", "M6")}
                actual = sum(record["public_memory_counts"].values())
                record["external_attempts"] = _merge_dynamic_attempt_stats(
                    record.get("external_attempts", {}), target)
                record["external_attempts"]["public_facts"] = actual
                record["external_information_coverage"] = {
                    "attempt_budget": scenario_plan.get("attempt_budget", 0),
                    "actual": actual,
                    "distribution": record["public_memory_counts"],
                }
                if dialogue_only:
                    record["status"] = "dialogue_only"
                    record["dialogue_manifest"] = str((package / "manifest.json").relative_to(output))
                    entry["status"] = _aggregate_status(
                        scenario.get("status") for scenario in entry["scenarios"])
                    continue
                record["evaluations"] = {}
                for qa_source in state["qa_sources"]:
                    evaluated = record["evaluations"][qa_source] = {}
                    options = dict(EVALUATION_DEFAULTS)
                    options.update(plan.get("evaluation", {}))
                    qa_only = options.pop("qa_only") or (qa_source == "graph" and len(state["qa_sources"]) > 1)
                    evaluated["target"] = dict(
                        published_qa=(options["general_count"] + options["code_count"]
                                      if qa_source == "graph" else options["qa_count"]),
                        paired_tasks=0 if qa_only else options["task_count"])
                    evaluated["qa_only"] = qa_only
                    if qa_source == "external" and not events:
                        evaluated.update(status="no_external_history", published_qa=0, paired_tasks=0, tasks=[])
                        continue
                    target = scenario_root / "evaluation" / qa_source
                    command = [str(python), str(Path(__file__).resolve().parents[1] / "run_episode.py"),
                        "--episode-manifest", str(package / "manifest.json"), "--qa-source", qa_source,
                        "--simulator-path", str(simulator), "--env-file", str(env_file), "--output", str(target)]
                    if qa_only:
                        command.append("--qa-only")
                    excluded = ({"qa_count", "group_budget"} if qa_source == "graph"
                                else {"general_count", "code_count"})
                    excluded.update({"max_task_workers", "design_probe",
                                     "clarification_diagnostic"})
                    for key, value in options.items():
                        if key not in excluded:
                            command.extend(["--" + key.replace("_", "-"), str(value)])
                    if options.get("design_probe"):
                        command.append("--design-probe")
                    if options.get("clarification_diagnostic"):
                        command.append("--clarification-diagnostic")
                    completed = stage("evaluation", target, command, Path(__file__).resolve().parents[1],
                                      target / "pipeline.json", {"completed"}, target / "usage.json")
                    evaluated.update(status=completed.get("stop_reason", completed.get("status", "completed")) if completed else "evaluation_failed",
                                     qa_only=qa_only, path=str(target.relative_to(output)))
                    if completed is None:
                        warning("evaluation_failed", project=project["id"],
                                scenario=scenario["id"], route=qa_source)
                    public = _read_if(target / "qa/qa-public.json")
                    evaluated["qa_status"] = public.get("status")
                    evaluated["published_qa"] = len(public.get("questions", []))
                    evaluated["counts"] = public.get("counts", {})
                    tasks = _read_if(target / "tasks/manifest.json").get("tasks", [])
                    evaluated["tasks"] = tasks
                    evaluated["paired_tasks"] = sum(_completed_pair(task) for task in tasks)
                for evaluated in record["evaluations"].values():
                    evaluated["shortfall"] = {
                        key: max(0, value - evaluated[key])
                        for key, value in evaluated["target"].items()}
                    evaluated["eligible"] = (
                        evaluated["status"] in {"completed", "completed_with_warnings", "qa_only"}
                        and evaluated["published_qa"] > 0
                        and (evaluated["qa_only"] or evaluated["paired_tasks"] > 0))
                    if not evaluated["eligible"] and evaluated["status"] in {"completed", "completed_with_warnings", "qa_only"}:
                        evaluated["status"] = "below_target"
                for qa_source, evaluated in record["evaluations"].items():
                    if any(evaluated["shortfall"].values()):
                        warning("coverage_shortfall", project=project["id"],
                                scenario=scenario["id"], route=qa_source,
                                shortfall=evaluated["shortfall"])
                outcomes = [row["status"] for row in record["evaluations"].values()]
                record["status"] = ((outcomes[0] if len(outcomes) == 1 else "completed") if all(
                    row["eligible"]
                    for row in record["evaluations"].values())
                    else "below_target" if "below_target" in outcomes
                    else outcomes[0] if len(outcomes) == 1
                    else "partial_failure" if "evaluation_failed" in outcomes
                    else "partial_failure")
            if entry["scenarios"]:
                entry["status"] = _aggregate_status(
                    scenario.get("status") for scenario in entry["scenarios"])
            elif entry["status"] == "running":
                entry["status"] = "no_scenarios"
        persist()
        if not state["usage"]["complete"]:
            warning("usage_incomplete")
        if dialogue_only:
            state["stop_reason"] = "dialogue_only"
        state["status"] = _terminal_collection_status(state)
    except BaseException as error:
        state.update(status="blocked",
                     stop_reason=str(error), error_type=type(error).__name__)
        for project in state["projects"]:
            if project["status"] == "running":
                project["status"] = "blocked"
            for scenario in project["scenarios"]:
                if scenario["status"] == "running":
                    scenario["status"] = "blocked"
        raise
    finally:
        persist()
        write_collection_report(output, state)
    return state


def write_collection_report(output, state):
    """Reuse the paired-trial report and keep construction statistics separate."""
    from .task_eval.report import write_report
    tasks = []
    qa_only = state.get("qa_only", False)
    lines = ["# Collection construction", "", "Status: " + state["status"], "",
             "| Project | Scenario | Route | Outcome | Published QA | QA target | QA shortfall | Completed pairs | Pair target | Pair shortfall |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for project in state["projects"]:
        for scenario in project["scenarios"]:
            routes = scenario.get("evaluations", {})
            if not routes:
                lines.append("| %s | %s | — | %s | 0 | — | — | 0 | — | — |" % (
                    project["id"], scenario["id"], scenario["status"]))
            for qa_source, evaluated in routes.items():
                target = evaluated.get("target", {})
                shortfall = evaluated.get("shortfall", {})
                lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    project["id"], scenario["id"], qa_source, evaluated.get("status", "not_started"),
                    evaluated.get("published_qa", 0),
                    target.get("published_qa", "—"), shortfall.get("published_qa", "—"),
                    "Not scheduled" if evaluated.get("qa_only", qa_only) else evaluated.get("paired_tasks", 0),
                    target.get("paired_tasks", "—"), shortfall.get("paired_tasks", "—")))
                tasks.extend(dict(t, task=evaluated["path"] + "/tasks/" + t["task"])
                             for t in evaluated.get("tasks", []))
    lines += ["", "QA counts are reported per route. They are not added into a combined unique count."]
    lines += ["Targets and shortfalls describe output volume, not admission. A completed route has nonempty published QA and, unless QA-only, at least one complete terminal pair. These are run-completeness checks, not manual quality acceptance."]
    lines += ["", "## Planned external information opportunities", "",
              "The opportunity budget is created before repository generation. Attempts and accepted facts are reported separately; a shortfall is retained as coverage data.",
              "", "| Project | Scenario | Planned attempts | Started | Candidate groups | Candidate facts | Accepted facts | Rejected facts | Public facts | External QA | Requirement candidates | Unused attempts |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for project in state["projects"]:
        for scenario in project["scenarios"]:
            coverage = scenario.get("external_information_coverage", {})
            attempts = scenario.get("external_attempts", {})
            plan_row = scenario.get("external_information_plan", {})
            external = scenario.get("evaluations", {}).get("external", {})
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                project["id"], scenario["id"],
                attempts.get("planned", plan_row.get("attempt_budget", "—")),
                attempts.get("started", "—"), attempts.get("candidate_groups", "—"),
                attempts.get("candidate_facts", "—"), attempts.get("accepted_facts", "—"),
                attempts.get("rejected_facts", "—"), coverage.get("actual", "—"),
                external.get("published_qa", "—"), len(external.get("tasks", [])),
                attempts.get("unused_attempts", "—")))
    if state.get("warnings"):
        lines += ["", "## Warnings", ""]
        lines.extend("- " + json.dumps(item, ensure_ascii=False, sort_keys=True)
                     for item in state["warnings"])
    totals = {}
    for project in state["projects"]:
        for scenario in project["scenarios"]:
            for source, evaluated in scenario.get("evaluations", {}).items():
                total = totals.setdefault(source, {"qa": 0, "tasks": 0, "pairs": 0})
                total["qa"] += evaluated.get("published_qa", 0)
                total["tasks"] += len(evaluated.get("tasks", []))
                total["pairs"] += evaluated.get("paired_tasks", 0)
    lines += ["", "| Route | Published QA | Task records | Completed pairs |",
              "|---|---:|---:|---:|"]
    for source, total in totals.items():
        lines.append("| %s | %s | %s | %s |" % (
            source, total["qa"], total["tasks"], total["pairs"]))
    lines += ["", "## Construction and evaluation usage", "",
              "Each primary ledger is counted once, including failed stages. Counts exclude reused project preparation.",
              "", "| Stage | Status | Requests | Tokens | Complete usage |", "|---|---|---:|---:|---|"]
    for stage in state["stages"]:
        usage = stage.get("usage", {})
        lines.append("| [%s](%s.log) | %s | %s | %s | %s |" % (
            stage["path"], stage["path"], stage["status"], usage.get("requests", "unknown"),
            usage.get("total_tokens", "unknown"), usage.get("complete", False)))
    usage = state["usage"]
    lines += ["", "Total: %s requests; %s recorded tokens; complete=%s." % (
        usage["requests"], usage["total_tokens"], usage["complete"]), ""]
    if qa_only:
        lines += ["This collection produces dialogue and recall QA. Repository tasks and paired trials are not scheduled.", ""]
    else:
        lines += ["[Paired execution results and totals](report.md)", ""]
    (output / "collection.md").write_text("\n".join(lines), encoding="utf-8")
    if not qa_only:
        write_report(output, {"tasks": tasks})
