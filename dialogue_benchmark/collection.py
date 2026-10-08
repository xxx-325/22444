"""Sequential composition of the existing project, dialogue and QA commands."""

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from .task_eval.artifacts import read, save
from .task_eval.metrics import cache_usage

EVALUATION_DEFAULTS = dict(qa_count=8, task_count=1, task_budget=2,
                           parallel_workers=6, task_workers=2, revisions=3,
                           model_request_chars=96000, qa_only=False,
                           general_count=50, code_count=50)
MAX_STAGE_RETRIES = 2


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
    if qa.is_file():
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


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


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
    if all(status in {"completed", "qa_only", "dialogue_only"} for status in statuses):
        return "completed"
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
    if aggregate in {"completed", "partial_failure", "below_target", "no_scenarios"}:
        return "completed_with_warnings"
    return "blocked"


def _completed_pair(task):
    """Count only a fully evaluated pair with terminal arm results."""
    if task.get("status") != "evaluated":
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
        for label, repo in (("qa", Path(__file__).resolve().parents[1]), ("simulator", simulator)):
            result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True)
            state[label + "_revision"] = result.stdout.strip() if result.returncode == 0 else None
    state["status"] = "running"
    dialogue_only = bool(dialogue_only or state.get("dialogue_only", False))
    state["dialogue_only"] = dialogue_only
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
            if previous["status"] == "completed" or (
                    previous["status"] == "failed" and previous.get("terminal")):
                if previous.get("receipt_sha256") != _sha256(receipt):
                    raise ValueError("Collection stage receipt changed: " + target)
                if previous["status"] == "completed":
                    result = read(receipt)
                    if result.get("status", result.get("schema")) not in expected:
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
        else:
            if previous:
                if previous.get("retry", 0) >= MAX_STAGE_RETRIES:
                    archive = output / "attempts" / target / ("attempt-%d" % (previous.get("retry", 0) + 1))
                    archive.mkdir(parents=True, exist_ok=False, mode=0o700)
                    for artifact in (folder, folder.with_suffix(".log"),
                                     folder.with_name("dialogue-package") if name == "dialogue" else None):
                        if artifact is not None and artifact.exists():
                            artifact.rename(archive / artifact.name)
                    previous["status"] = "retry_exhausted"
                    previous["receipt_sha256"] = _sha256(receipt)
                    previous["retry_exhausted_path"] = str(archive.relative_to(output))
                    persist()
                    return None
                archive = output / "attempts" / target / "attempt-1"
                archive.mkdir(parents=True, exist_ok=False, mode=0o700)
                for artifact in (folder, folder.with_suffix(".log"),
                                 folder.with_name("dialogue-package") if name == "dialogue" else None):
                    if artifact is not None and artifact.exists():
                        artifact.rename(archive / artifact.name)
                previous["path"] = str((archive / folder.name).relative_to(output))
                previous["archived"] = True
                persist()
            row = dict(name=name, path=target, target=target, status="running", command=command,
                       identity=stage_identity, retry=1 if previous else 0)
            state["stages"].append(row)
        persist()
        print("Collection:", row["path"], name, flush=True)
        try:
            row["returncode"] = _command(command, cwd, folder.with_suffix(".log"))
            result = _recover_receipt(receipt)
            row["outcome"] = result.get("status", result.get("schema", "missing_receipt"))
            row["status"] = ("completed" if row["returncode"] == 0
                             and row["outcome"] in expected else "failed")
            row["terminal"] = row["returncode"] == 0 and row["status"] == "failed"
            if (row["status"] == "failed"
                    and row["outcome"] in {"missing_receipt", "timeout",
                                           "connection_error", "protocol_error",
                                           "request_budget"}
                    and row.get("retry", 0) < MAX_STAGE_RETRIES):
                row["status"] = "failed"
                persist()
                return stage(name, folder, command, cwd, receipt, expected, ledger, budget_key)
            if row["returncode"]:
                log = folder.with_suffix(".log")
                text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
                if re.search(r"authentication|invalid api key|unauthorized|missing.*credential", text, re.I):
                    raise RuntimeError("collection_authentication_failed")
                if re.search(r"unrecognized arguments|the following arguments are required", text, re.I):
                    raise RuntimeError("collection_input_failed")
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
                    return stage(name, folder, command, cwd, receipt, expected, ledger, budget_key)
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
            row["output_ready"] = receipt.is_file()
            row["usage_ready"] = bool(row["usage"].get("complete"))
            persist()

    def upstream(module, config, target):
        return [str(python), "-m", module, "--config", str(config),
                "--env-file", str(env_file), "--output", str(target)]

    try:
        for project in plan["projects"]:
            root = output / project["id"]
            root.mkdir(exist_ok=resume)
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
                config["project"] = dict(brief=project["brief"], increments=project.get("increments", 2))
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
                current = copy.deepcopy(config)
                current.pop("scenario_file", None)
                current.pop("prepared_issues", None)
                current["scenario_design"] = {k: v for k, v in scenario.items() if k != "id"}
                cfg = scenario_root / "config.json"
                if not (resume and cfg.is_file()):
                    save(cfg, current)
                record = next((s for s in entry["scenarios"] if s["id"] == scenario["id"]), None)
                if record is None:
                    record = dict(id=scenario["id"], status="running")
                    entry["scenarios"].append(record)
                record["status"] = "running"
                target = scenario_root / "scenario"
                designed = stage("scenario", target, upstream("simulator.openhands.prepare_scenario", cfg, target),
                    simulator, target / "frozen/report.json", {"completed", "candidate_pass"}, target / "budget.json")
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
                events = read(exported["external_events"])["events"] if exported.get("external_events") else []
                record["public_memory_counts"] = {kind: sum(e.get("memory_kind") == kind for e in events)
                                                   for kind in ("M1", "M2", "M3", "M4", "M5", "M6")}
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
                    for key, value in options.items():
                        if key not in excluded:
                            command.extend(["--" + key.replace("_", "-"), str(value)])
                    completed = stage("evaluation", target, command, Path(__file__).resolve().parents[1],
                                      target / "pipeline.json", {"completed"}, target / "usage.json")
                    evaluated.update(status=completed.get("stop_reason", "completed") if completed else "evaluation_failed",
                                     qa_only=qa_only, path=str(target.relative_to(output)))
                    if completed is None:
                        warning("evaluation_failed", project=project["id"],
                                scenario=scenario["id"], route=qa_source)
                    public = _read_if(target / "qa/qa-public.json")
                    if public.get("status") == "failed":
                        evaluated["status"] = "evaluation_failed"
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
                        evaluated["status"] in {"completed", "qa_only"}
                        and evaluated["published_qa"] > 0
                        and (evaluated["qa_only"] or evaluated["paired_tasks"] > 0))
                    if not evaluated["eligible"] and evaluated["status"] in {"completed", "qa_only"}:
                        evaluated["status"] = "below_target"
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
                lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    project["id"], scenario["id"], qa_source, evaluated["status"],
                    evaluated.get("published_qa", 0),
                    evaluated["target"]["published_qa"], evaluated["shortfall"]["published_qa"],
                    "Not scheduled" if evaluated.get("qa_only", qa_only) else evaluated.get("paired_tasks", 0),
                    evaluated["target"]["paired_tasks"], evaluated["shortfall"]["paired_tasks"]))
                tasks.extend(dict(t, task=evaluated["path"] + "/tasks/" + t["task"])
                             for t in evaluated.get("tasks", []))
    lines += ["", "QA counts are reported per route. They are not added into a combined unique count."]
    lines += ["Targets and shortfalls describe output volume, not admission. A completed route has nonempty published QA and, unless QA-only, at least one complete terminal pair. These are run-completeness checks, not manual quality acceptance."]
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
