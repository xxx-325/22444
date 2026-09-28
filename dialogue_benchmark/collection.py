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

EVALUATION_DEFAULTS = dict(qa_count=8, task_count=1, task_budget=2,
                           parallel_workers=2, task_workers=1, revisions=3,
                           model_request_chars=96000, qa_only=False)


def sum_usage(rows):
    result = dict(requests=0, prompt_tokens=0, completion_tokens=0, complete=True)
    for row in rows:
        count = row.get("attempts", row.get("requests", row.get("request_count", 1)))
        result["requests"] += count
        for key in ("prompt_tokens", "completion_tokens"):
            result[key] += row.get(key, 0)
            if count and not isinstance(row.get(key), int):
                result["complete"] = False
        result["complete"] &= not bool(row.get("usage_missing") or row.get("pending"))
        result["complete"] &= row.get("complete", row.get("usage_complete", True))
    result["total_tokens"] = result["prompt_tokens"] + result["completion_tokens"]
    return result


def episode_usage(root):
    """Read each primary ledger once, before compaction removes stage directories."""
    root = Path(root)
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
    return dict(**sum_usage(receipts), receipts=receipts)


def _read_if(path):
    return read(path) if path.is_file() else {}


def _command(command, cwd, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        log.chmod(0o600)
        return subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT).returncode


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", value):
        raise ValueError("Use short lowercase project and scenario identifiers")
    return value


def validate_plan(plan):
    if not plan.get("projects") or not plan.get("runtime_config"):
        raise ValueError("A runtime_config and fixed projects list are required")
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


def run_collection(plan_path, output, simulator, env_file, python=sys.executable):
    plan_path, output, simulator = (Path(p).resolve() for p in (plan_path, output, simulator))
    plan = read(plan_path)
    validate_plan(plan)
    runtime = read((plan_path.parent / plan["runtime_config"]).resolve())
    for role in ("user", "code", "judge", "decomposer"):
        if isinstance(runtime.get(role), dict):
            runtime[role]["max_output_tokens"] = None
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    state = dict(status="running", projects=[], stages=[],
                 qa_only=plan.get("evaluation", {}).get("qa_only", False),
                 plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                 limits={k: plan[k] for k in ("max_total_requests", "max_total_tokens")},
                 budget_boundary="Finish each started stage, then check cumulative usage before the next stage")
    save(output / "plan.json", plan)
    for label, repo in (("qa", Path(__file__).resolve().parents[1]), ("simulator", simulator)):
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True)
        state[label + "_revision"] = result.stdout.strip() if result.returncode == 0 else None

    def persist():
        state["usage"] = sum_usage([s["usage"] for s in state["stages"] if "usage" in s])
        save(output / "collection.json", state)

    def stage(name, folder, command, cwd, receipt, expected, ledger, budget_key=None):
        persist()
        if not state["usage"]["complete"]:
            raise RuntimeError("usage_incomplete")
        if (state["usage"]["requests"] >= plan["max_total_requests"]
                or state["usage"]["total_tokens"] >= plan["max_total_tokens"]):
            raise RuntimeError("collection_budget_exhausted")
        row = dict(name=name, path=str(folder.relative_to(output)), status="running", command=command)
        state["stages"].append(row)
        persist()
        print("Collection:", row["path"], name, flush=True)
        try:
            row["returncode"] = _command(command, cwd, folder.with_suffix(".log"))
            result = _read_if(receipt)
            row["outcome"] = result.get("status", result.get("schema", "missing_receipt"))
            row["status"] = ("completed" if row["returncode"] == 0
                             and row["outcome"] in expected else "failed")
            return result if row["status"] == "completed" else None
        except BaseException as error:
            row.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                       error_type=type(error).__name__)
            raise
        finally:
            try:
                budget = _read_if(ledger)
                if name == "dialogue" and not budget:
                    budget = {"budget": _read_if(folder / "private/budget.json")}
                if budget_key:
                    budget = budget.get(budget_key)
                row["usage"] = sum_usage([budget]) if budget else dict(sum_usage([]), complete=False)
            except (OSError, ValueError, TypeError) as error:
                row["usage"] = dict(sum_usage([]), complete=False, error_type=type(error).__name__)
            persist()

    def upstream(module, config, target):
        return [str(python), "-m", module, "--config", str(config),
                "--env-file", str(env_file), "--output", str(target)]

    try:
        for project in plan["projects"]:
            root = output / project["id"]
            root.mkdir()
            entry = dict(id=project["id"], status="running", scenarios=[])
            state["projects"].append(entry)
            if project.get("prepared_config"):
                config = read((plan_path.parent / project["prepared_config"]).resolve())
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
                    continue
                config = read(target / "config.json")
            repo = Path(config["repository"])
            for role in ("user", "code", "judge", "decomposer"):
                if isinstance(config.get(role), dict):
                    config[role]["max_output_tokens"] = None
            entry["lineage"] = subprocess.check_output(
                ["git", "rev-list", "--max-parents=0", config["base"]], cwd=repo, text=True).splitlines()
            entry["base"] = config["base"]
            for scenario in project["scenarios"]:
                scenario_root = root / scenario["id"]
                scenario_root.mkdir()
                current = copy.deepcopy(config)
                current.pop("scenario_file", None)
                current.pop("prepared_issues", None)
                current["scenario_design"] = {k: v for k, v in scenario.items() if k != "id"}
                cfg = scenario_root / "config.json"
                save(cfg, current)
                record = dict(id=scenario["id"], status="running")
                entry["scenarios"].append(record)
                target = scenario_root / "scenario"
                designed = stage("scenario", target, upstream("simulator.openhands.prepare_scenario", cfg, target),
                    simulator, target / "frozen/report.json", {"completed", "candidate_pass"}, target / "budget.json")
                if designed is None:
                    record["status"] = "scenario_rejected"
                    continue
                record["requested_memory_kinds"] = designed.get("design", {}).get("memory_kinds", [])
                record["designed_memory_counts"] = designed.get("memory_counts", {})
                current["scenario_file"] = str(target / "frozen/scenario.json")
                save(cfg, current)
                target = scenario_root / "requirements"
                if stage("requirements", target, upstream("simulator.openhands.prepare_progressive", cfg, target),
                         simulator, target / "report.json", {"candidate_pass"}, target / "budget.json") is None:
                    record["status"] = "requirements_rejected"
                    continue
                current["prepared_issues"] = str(target / "report.json")
                save(cfg, current)
                target = scenario_root / "dialogue"
                package = target.with_name("dialogue-package")
                if stage("dialogue", target, upstream("simulator", cfg, target), simulator,
                         package / "manifest.json", {"memory-episode-v1"},
                         package / "private/review.json", "budget") is None:
                    record["status"] = "dialogue_incomplete"
                    continue
                from .episode_input import load_episode_manifest
                exported = load_episode_manifest(package / "manifest.json")
                events = read(exported["external_events"])["events"] if exported.get("external_events") else []
                record["public_memory_counts"] = {kind: sum(e.get("memory_kind") == kind for e in events)
                                                   for kind in ("M1", "M2", "M3", "M4", "M5", "M6")}
                if not events:
                    record.update(status="no_external_history", published_qa=0, paired_tasks=0, tasks=[])
                    continue
                target = scenario_root / "evaluation"
                command = [str(python), str(Path(__file__).resolve().parents[1] / "run_episode.py"),
                    "--episode-manifest", str(package / "manifest.json"), "--qa-source", "external",
                    "--simulator-path", str(simulator), "--env-file", str(env_file), "--output", str(target)]
                defaults = dict(EVALUATION_DEFAULTS)
                defaults.update(plan.get("evaluation", {}))
                if defaults.pop("qa_only"):
                    command.append("--qa-only")
                for key, value in defaults.items():
                    command.extend(["--" + key.replace("_", "-"), str(value)])
                completed = stage("evaluation", target, command, Path(__file__).resolve().parents[1],
                                  target / "pipeline.json", {"completed"}, target / "usage.json")
                record["status"] = completed.get("stop_reason", "completed") if completed else "evaluation_failed"
                record["published_qa"] = len(_read_if(target / "qa/qa-public.json").get("questions", []))
                tasks = _read_if(target / "tasks/manifest.json").get("tasks", [])
                record["tasks"] = tasks
                record["paired_tasks"] = sum(set(t.get("comparison", {})) >= {"with_memory", "without_memory"}
                                              for t in tasks)
            entry["status"] = "completed"
        state["status"] = "completed"
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "stopped",
                     stop_reason=str(error), error_type=type(error).__name__)
        for project in state["projects"]:
            if project["status"] == "running":
                project["status"] = state["status"]
            for scenario in project["scenarios"]:
                if scenario["status"] == "running":
                    scenario["status"] = state["status"]
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
             "| Project | Scenario | Outcome | Published QA | Completed pairs |",
             "|---|---|---|---:|---:|"]
    for project in state["projects"]:
        for scenario in project["scenarios"]:
            path = project["id"] + "/" + scenario["id"]
            lines.append("| %s | %s | %s | %s | %s |" % (
                project["id"], scenario["id"], scenario["status"],
                scenario.get("published_qa", 0),
                "Not scheduled" if qa_only else scenario.get("paired_tasks", 0)))
            tasks.extend(dict(t, task=path + "/evaluation/tasks/" + t["task"])
                         for t in scenario.get("tasks", []))
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
