"""Generate, validate, freeze, and evaluate repository tasks from historical QA."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shutil

from . import prompts
from .artifacts import copy_tree, fingerprint, labels, qa_inputs, read, save, write_diff
from .checks import run_checks, acceptance_items, assess_acceptance, check_history_mutations
from .metrics import compare_trials
from .runtime import configure, review_task, run_agent
from .report import write_report
from .versions import baseline_version, export_change, pin_baseline
from .history import prepare_history, freeze_contract, historical_context, read_history_review, oracle_coverage

def solver_input(task, answer=None):
    message = prompts.SOLVER + "\n\n" + task
    if answer is not None:
        message += "\n\n历史经验（描述过去，适用性请结合当前需求判断）：\n" + answer
    return message


def answer_text(question):
    points = question.get("answer_points", [])
    return "\n".join("- " + (p if isinstance(p, str) else p.get("text", p.get("claim", "")))
                     for p in points)


def prepare(root, baseline):
    copy_tree(baseline, Path(root) / "workspace/candidate")


def freeze(spec, output, baseline):
    output = Path(output)
    # Freeze the same support files that preflight tests used, including fixtures.
    copy_tree(spec, output)
    for name in ("task.md", "acceptance.md", "acceptance.json"):
        if not (output / name).is_file() or not (output / name).read_text().strip():
            raise ValueError("Missing task artifact: " + name)
    return {"baseline_sha256": fingerprint(baseline), "spec_sha256": fingerprint(output)}


def unchanged(receipt, spec, baseline):
    return (receipt["baseline_sha256"] == fingerprint(baseline)
            and receipt["spec_sha256"] == fingerprint(spec))


def admission(validation, baseline_checks, reference_checks):
    """Tests cannot be overridden by a model's PASS; absent tests use explicit review."""
    if not (validation.get("BASELINE") == "unmet" and validation.get("REFERENCE") == "pass"
            and validation.get("VERDICT") == "accept"
            and validation.get("COVERAGE") == "complete"):
        return False
    if validation.get("MUTATIONS") == "missed" or baseline_checks["status"] == "error":
        return False
    if reference_checks["status"] in {"failed", "error"}:
        return False
    if validation.get("TESTS") == "executable":
        return (baseline_checks["status"] == "failed"
                and reference_checks["status"] == "passed"
                and not reference_checks.get("skipped", 0)
                and validation.get("MUTATIONS") == "caught")
    return (validation.get("TESTS") in {"partial", "unavailable"}
            and validation.get("MUTATIONS") in {"caught", "unavailable"})


def validated_spec(spec, validator_checks, output):
    """Persist the validator's checks in the exact spec used by both trials."""
    coverage = Path(validator_checks) / "coverage.md"
    if not coverage.is_file() or not coverage.read_text().strip():
        return None
    copy_tree(spec, output)
    for name in ("coverage.md", "test_interactions.py"):
        source = Path(validator_checks) / name
        if source.is_file():
            shutil.copy2(source, Path(output) / name)
    return Path(output)


def agent_finished(outcome):
    return str(outcome.get("status")) in {"finished", "ConversationExecutionStatus.FINISHED"}


def construct(item, root, baseline, config, revisions, agent_options, *, design_probe=False):
    direction = prompts.task_direction(item["qa"]["type"])
    feedback = ""
    reference = root / "author-reference"
    reference.mkdir(parents=True)
    save(reference / "qa.json", item["qa"])
    shutil.copy2(item["generation_input"], reference / "qa-input.json")
    save(reference / "provenance.json", {k: v for k, v in item.items() if k not in {"qa", "public_records"}})
    public_history = (prepare_history(item["public_records"], item["generation_input"])
                      if item.get("public_records") else None)
    if public_history:
        save(reference / "history.json", public_history)
    history_prompt = prompts.HISTORY_AUTHOR if public_history else ""
    attempts = []
    for attempt in range(revisions + 1):
        run = root / ("construction-%02d" % attempt)
        author = run / "author"
        prepare(author, baseline)
        print(root.name, "author", attempt, flush=True)
        authored = run_agent(author, config, "judge", prompts.AUTHOR + direction + history_prompt + feedback,
                             reference=reference, **agent_options)
        spec = author / "workspace/checks"
        record = {"attempt": attempt, "author_status": authored["status"]}
        attempts.append(record)
        if (spec / "NO_TASK.md").exists():
            record["reason"] = (spec / "NO_TASK.md").read_text()
            break
        required = ("task.md", "acceptance.md")
        if any(not (spec / name).exists() for name in required):
            record["reason"] = "missing_author_artifacts"
            feedback = ("\n上一轮状态：%s。没有写齐 task.md、acceptance.md。"
                        "请完成文件。搜索无结果不代表环境故障，不要反复执行同一空搜索；"
                        "无法提出适当需求时写 NO_TASK.md。" % authored["status"])
            save(root / "construction.json", attempts)
            continue
        task_review = review_task((spec / "task.md").read_text(), answer_text(item["qa"]),
                                  config, run / "task-review")
        record["task_review"] = task_review
        if task_review["status"] != "clean":
            record.update(accepted=False, reason="task_" + task_review["status"])
            prior = reference / ("previous-%02d" % attempt)
            copy_tree(spec, prior)
            feedback = ("\n阅读 /reference/previous-%02d，修正公开需求中的解题提示：%s。"
                        "保留所有可观察行为、触发条件和兼容要求；只删除内部定位与历史改法。"
                        "重新写齐文件。" % (attempt, task_review["issue"]))
            save(root / "construction.json", attempts)
            continue
        history = None
        if public_history:
            try:
                history = freeze_contract(spec, public_history, answer_text(item["qa"]))
            except (ValueError, KeyError, OSError) as error:
                record.update(accepted=False, reason="invalid_history_contract", detail=str(error))
                feedback = "\n上一轮历史契约错误，请根据公开来源重写：" + str(error)
                save(root / "construction.json", attempts)
                continue
        try:
            items = acceptance_items(spec, history)
            save(spec / "acceptance.json", items)
        except ValueError as error:
            record.update(accepted=False, reason="invalid_acceptance", detail=str(error))
            feedback = "\n验收表需要修正：" + str(error)
            save(root / "construction.json", attempts)
            continue
        baseline_checks = run_checks(baseline, spec, run / "baseline-checks", config["execution_image"],
                                         candidate_pythonpath=config.get("code", {}).get("candidate_pythonpath"))
        implementation = run / "reference-solver"
        prepare(implementation, baseline)
        print(root.name, "reference implementation", attempt, flush=True)
        reference_answer = (answer_text(item["qa"]) + "\n" + historical_context(history)) if history else None
        solved = run_agent(implementation, config, "code",
                           solver_input((spec / "task.md").read_text(), reference_answer), **agent_options)
        candidate = implementation / "workspace/candidate"
        record["reference_version"] = export_change(baseline, candidate, implementation)
        reference_checks = run_checks(candidate, spec, run / "reference-checks", config["execution_image"],
                                         candidate_pythonpath=config.get("code", {}).get("candidate_pythonpath"))
        record.update(baseline_checks=baseline_checks, reference_checks=reference_checks,
                      reference_status=solved["status"])
        validation_reference = run / "validator-reference"
        copy_tree(spec, validation_reference / "spec")
        copy_tree(candidate, validation_reference / "implementation")
        save(validation_reference / "checks.json", record)
        validator = run / "validator"
        prepare(validator, baseline)
        print(root.name, "preflight validation", attempt, flush=True)
        validated = run_agent(validator, config, "judge", prompts.VALIDATOR + (
                              prompts.HISTORY_VALIDATOR if history else ""),
                              reference=validation_reference, **agent_options)
        verdict_file = validator / "workspace/checks/validation.txt"
        feedback = verdict_file.read_text() if verdict_file.exists() else "验收者未完成验证，请核查需求和测试。"
        record["validation"] = labels(feedback)
        record["validation_evidence"] = feedback
        record["validator_status"] = validated["status"]
        final_spec = validated_spec(spec, validator / "workspace/checks", run / "validated-spec")
        if final_spec is not None:
            # Never freeze model-written extra tests without executing those exact files.
            baseline_checks = run_checks(baseline, final_spec, run / "final-baseline-checks",
                                         config["execution_image"],
                                         candidate_pythonpath=config.get("code", {}).get("candidate_pythonpath"))
            reference_checks = run_checks(candidate, final_spec, run / "final-reference-checks",
                                          config["execution_image"],
                                         candidate_pythonpath=config.get("code", {}).get("candidate_pythonpath"))
            record.update(final_baseline_checks=baseline_checks, final_reference_checks=reference_checks)
        else:
            record["reason"] = "missing_coverage_checks"
        oracle_complete = not history or oracle_coverage(validator / "workspace/checks/oracle-review.txt", history)
        if history and final_spec is not None and oracle_complete:
            record["history_mutations"] = check_history_mutations(
                candidate, final_spec, validator / "workspace/checks",
                run / "history-mutations", config["execution_image"],
                                         candidate_pythonpath=config.get("code", {}).get("candidate_pythonpath"))
        reference_acceptance = (assess_acceptance(items, reference_checks,
            validator / "workspace/checks/acceptance-review.txt",
            {"/reference/implementation": candidate,
             "/workspace/checks": validator / "workspace/checks",
             "/workspace/experiments": validator / "workspace/experiments"})
            if final_spec else {"status": "uncertain"})
        record["reference_acceptance"] = reference_acceptance
        record["oracle_complete"] = oracle_complete
        if not oracle_complete:
            record["reason"] = "oracle_answer_incomplete"
        elif history and record.get("history_mutations", {}).get("status") != "caught":
            record["reason"] = "historical_mutation_not_verified"
        record["validation_accepted"] = (agent_finished(solved) and agent_finished(validated)
                                         and final_spec is not None
                                         and reference_acceptance["status"] == "passed"
                                         and oracle_complete
                                         and (not history or (
                                             record["validation"].get("HISTORY") == "supported"
                                             and record["history_mutations"]["status"] == "caught"))
                                         and admission(record["validation"], baseline_checks, reference_checks))
        record["accepted"] = False
        save(root / "construction.json", attempts)
        if record["validation_accepted"]:
            if history and design_probe:
                probe_root = run / "design-probe"
                prepare(probe_root, baseline)
                probe = run_agent(probe_root, config, "code",
                                  solver_input((final_spec / "task.md").read_text()) + prompts.HISTORY_REQUEST,
                                  history=history, **agent_options)
                record["design_probe"] = {"status": probe["status"], "used_for_admission": False,
                    "checks": run_checks(probe_root / "workspace/candidate", final_spec,
                                         run / "probe-checks", config["execution_image"],
                                         candidate_pythonpath=config.get("code", {}).get("candidate_pythonpath"))}
            record["accepted"] = True
            save(root / "construction.json", attempts)
            receipt = freeze(final_spec, root / "frozen", baseline)
            receipt.update(qa_id=item["qa"]["id"], accepted_attempt=attempt,
                           validation=record["validation"],
                           baseline_checks=baseline_checks, reference_checks=reference_checks)
            if history:
                receipt.update(comparison="without_memory_vs_oracle_history",
                               reference_information=history["reference_information"],
                               oracle_sufficiency="validated_against_active_rules")
            save(root / "frozen.json", receipt)
            return receipt
        # New author conversation receives the previous artifacts and concrete verifier feedback.
        prior = reference / ("previous-%02d" % attempt)
        copy_tree(final_spec or spec, prior)
        feedback = ("\n上一轮未通过。阅读 /reference/previous-%02d。依据以下具体问题修正，"
                    "重新写出完整文件，勿降低任务原有正确性标准：\n%s\n"
                    "补充检查实际重跑：基线 %s；参考实现 %s。%s" % (
                        attempt, feedback, baseline_checks, reference_checks, record.get("reason", "")))
    save(root / "construction.json", attempts)
    return None


def evaluate(item, root, baseline, receipt, config, agent_options, index):
    spec = root / "frozen"
    task = (spec / "task.md").read_text()
    items = read(spec / "acceptance.json")
    history = read(spec / "history.json") if (spec / "history.json").exists() else None
    result = {}
    order = ("without_memory", "with_memory") if index % 2 == 0 else ("with_memory", "without_memory")
    for slot, condition in enumerate(order, 1):
        if not unchanged(receipt, spec, baseline):
            raise ValueError("Frozen inputs changed before evaluation")
        trial = root / ("trial-%d" % slot)
        prepare(trial, baseline)
        oracle = history["oracle_answer"] if history else answer_text(item["qa"])
        message = solver_input(task, oracle if condition == "with_memory" else None)
        if history:
            message += prompts.HISTORY_REQUEST
        print(root.name, "evaluation", condition, flush=True)
        solved = run_agent(trial, config, "code", message, **agent_options,
                           **({"history": history} if history else {}))
        candidate = trial / "workspace/candidate"
        changed = write_diff(baseline, candidate, trial / "changes.patch")
        checks = run_checks(candidate, spec, trial / "checks", config["execution_image"],
                                         candidate_pythonpath=config.get("code", {}).get("candidate_pythonpath"))
        reference = trial / "judge-reference"
        copy_tree(spec, reference / "spec")
        if history:
            # The judge sees criteria and observable actions, never condition labels or injected answers.
            judge_history = read(reference / "spec/history.json")
            for key in ("oracle_answer", "reference_information", "oracle_sufficiency"):
                judge_history.pop(key, None)
            save(reference / "spec/history.json", judge_history)
        save(reference / "checks.json", checks)
        judge = trial / "judge"
        prepare(judge, candidate)
        if any(not row["tests"] for row in items):
            judged = run_agent(judge, config, "judge", prompts.JUDGE,
                               reference=reference, **agent_options)
        else:
            judged = {"status": "not_needed"}
        verdict_path = judge / "workspace/checks/verdict.txt"
        verdict = verdict_path.read_text() if verdict_path.exists() else ""
        acceptance = assess_acceptance(items, checks, judge / "workspace/checks/acceptance-review.txt",
            {"/workspace/candidate": candidate, "/workspace/checks": judge / "workspace/checks",
             "/workspace/experiments": judge / "workspace/experiments"})
        status = acceptance["status"]
        result[condition] = {"result": status, "solver_status": solved["status"],
                             "judge_status": judged["status"],
                             "metrics": solved["metrics"], "checks": checks,
                             "acceptance": acceptance, "changed_files": changed,
                             "judge_evidence": verdict, "trial": trial.name}
        counterexample = judge / "workspace/checks/counterexample.md"
        if counterexample.is_file():
            result[condition]["counterexample_pending_shared_review"] = counterexample.read_text()
        if history:
            application = read_history_review(acceptance, history)
            exchanges = solved.get("clarifications", [])
            result[condition].update(history_application=application,
                history_question_count=len(exchanges),
                information_condition="oracle_history" if condition == "with_memory" else "without_memory",
                clarifications=exchanges, responder_cost=solved.get("responder_cost"),
                clarification_status=solved.get("clarification_status", "unavailable"),
                interaction_counts={kind: sum(e.get("kind") == kind for e in exchanges)
                    for kind in ("historical_reask", "same_session_repeat", "update_confirmation")})
        if not unchanged(receipt, spec, baseline):
            raise ValueError("Frozen inputs changed during evaluation")
        save(root / "comparison.json", result)
        save(root / "paired-differences.json", compare_trials(result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("simulator-path", "source-run", "qa-run", "env-file", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--control-config", type=Path,
                        help="Independent non-secret runtime configuration")
    parser.add_argument("--design-probe", action="store_true",
                        help="Run a separate no-memory design probe; never an admission gate")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--baseline", type=Path,
                        help="Already pinned independent dialogue-end repository")
    parser.add_argument("--task-budget", type=int,
                        help="Maximum distinct QA-derived requirements to attempt (default: twice count)")
    parser.add_argument("--revisions", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--agent-requests", type=int, default=80)
    parser.add_argument("--agent-tokens", type=int, default=1500000)
    args = parser.parse_args(argv)
    task_budget = args.task_budget if args.task_budget is not None else args.count * 2
    if min(args.count, args.workers, args.agent_requests, args.agent_tokens) < 1 or args.revisions < 0:
        parser.error("Counts and budgets must be positive; revisions must be nonnegative")
    if task_budget < 1:
        parser.error("Task budget must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists() or (output / "baseline").exists():
        parser.error("Use a new output directory; previous experiments are retained")
    items = qa_inputs(args.qa_run)
    if not items:
        parser.error("No approved QA with saved generation inputs")
    config = configure(args.simulator_path, args.source_run / "private/checkpoint.json", args.env_file,
                       **({"control_config": args.control_config} if args.control_config else {}))
    baseline = args.baseline.resolve() if args.baseline else output / "baseline"
    if args.baseline:
        version = baseline_version(baseline)
    else:
        copy_tree(args.source_run / "workspace/candidate", baseline)
        version = pin_baseline(baseline)
    save(output / "baseline.json", dict(version,
         source=str((args.source_run / "workspace/candidate").resolve())))
    # Prefer distinct evidence targets, using only pre-evaluation metadata.
    selected, seen = [], set()
    for item in items:
        target = (item.get("original_candidate", {}).get("evidence_group_id")
                  or item["qa"].get("evidence_group_id") or item["qa"]["id"])
        if target not in seen:
            selected.append(item)
            seen.add(target)
        if len(selected) == task_budget:
            break
    for item in items:
        if len(selected) < task_budget and item not in selected:
            selected.append(item)
    manifest = {"source_run": str(args.source_run.resolve()), "qa_run": str(args.qa_run.resolve()),
                "baseline_sha256": fingerprint(baseline), "config": config,
                "comparison": "Historical answer injection; no memory retriever",
                "baseline_version": version, "baseline": str(baseline),
                "target": args.count, "task_budget": task_budget,
                "selected_qa_ids": [i["qa"]["id"] for i in selected], "tasks": []}
    save(output / "manifest.json", manifest)
    agent_options = {"max_requests": args.agent_requests, "max_tokens": args.agent_tokens}

    def run(index, item):
        root = output / ("task-%02d" % (index + 1))
        try:
            receipt = construct(item, root, baseline, config, args.revisions, agent_options,
                                **({"design_probe": True} if args.design_probe else {}))
            if receipt is None:
                return {"task": root.name, "status": "not_admitted", "qa_id": item["qa"]["id"],
                        "type": item["qa"]["type"]}
            comparison = evaluate(item, root, baseline, receipt, config, agent_options, index)
            return {"task": root.name, "status": "evaluated", "qa_id": item["qa"]["id"],
                    "type": item["qa"]["type"],
                    "comparison": comparison, "paired_differences": compare_trials(comparison)}
        except Exception as error:
            failure = {"task": root.name, "status": "error", "error_type": type(error).__name__,
                       "detail": str(error)}
            save(root / "failure.json", failure)
            return failure

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        next_index = 0
        while next_index < len(selected):
            completed = sum(task["status"] == "evaluated" for task in manifest["tasks"])
            if completed >= args.count:
                break
            size = min(args.workers, args.count - completed, len(selected) - next_index)
            futures = [pool.submit(run, index, selected[index])
                       for index in range(next_index, next_index + size)]
            next_index += size
            for future in as_completed(futures):
                manifest["tasks"].append(future.result())
                manifest["tasks"].sort(key=lambda item: item["task"])
                save(output / "manifest.json", manifest)
                write_report(output, manifest)
    completed = sum(task["status"] == "evaluated" for task in manifest["tasks"])
    manifest.update(completed=completed, shortfall=max(0, args.count - completed),
                    stop_reason="target_met" if completed >= args.count else
                    "task_budget_exhausted" if len(selected) >= task_budget else "qa_pool_exhausted")
    save(output / "manifest.json", manifest)
    write_report(output, manifest)
    print("Task pilot complete:", output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
