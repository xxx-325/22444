"""Generate, validate, freeze, and evaluate repository tasks from historical QA."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
import shutil

from . import prompts
from .artifacts import copy_tree, fingerprint, labels, qa_fingerprint, qa_inputs, read, save, write_diff
from .checks import run_checks, acceptance_items, assess_acceptance, check_history_mutations
from .metrics import compare_trials
from .runtime import configure, review_task, review_checks, repair_tests, write_tests, write_history_mutation, run_agent
from .report import write_report
from .versions import baseline_version, export_change, pin_baseline, source_version
from .history import (prepare_history, freeze_contract, historical_context, read_history_review,
                      write_contract_from_targets, review_sources, qualified_oracle_complete)
from .selection import SelectionBudget, select_task, write_draft, write_private_draft

def solver_input(task, answer=None):
    message = prompts.SOLVER + "\n\n" + task
    if answer is not None:
        message += "\n\n历史经验（描述过去，适用性请结合当前需求判断）：\n" + answer
    return message


def answer_text(question):
    points = question.get("answer_points", [])
    return "\n".join("- " + (p if isinstance(p, str) else p.get("text", p.get("claim", "")))
                     for p in points)


def reference_solver_answer(item, history=None):
    """Return the private information available to the reference solver.

    External-only QA has no public history contract, but its approved answer
    is still the fact that makes the task constructible.  Graph tasks keep the
    historical contract context used by the existing construction flow.
    """
    if item.get("qa_source") == "external":
        return answer_text(item["qa"])
    if history:
        return answer_text(item["qa"]) + "\n" + historical_context(history)
    return None


def prepare(root, baseline):
    copy_tree(baseline, Path(root) / "workspace/candidate")


def prepare_test_reference(spec, reference, output):
    """Expose fixed criteria and previous tests, keeping raw history for review."""
    output.mkdir(parents=True)
    for previous in reference.glob("previous-*"):
        if previous.is_dir():
            copy_tree(previous, output / previous.name)
    # qualify() already retained these exact files in qualified-draft. They
    # are regenerated from public_history for the independent validator.
    for directory in (spec, *output.glob("previous-*")):
        for name in ("history.json", "history-review.md"):
            (directory / name).unlink(missing_ok=True)
    return output


def explore_repository(root, baseline, item, config, options, public_history=None):
    """Use a read-only OpenHands worker to map a QA to a natural code area.

    This is deliberately separate from task authoring.  The worker may inspect
    the whole snapshot, but it cannot change the candidate or write the public
    task.  Its short report is passed to the host-controlled selector and is
    retained for audit.
    """
    from .prompts import REPOSITORY_EXPLORER

    root = Path(root)
    explorer = root / "repository-explorer"
    prepare(explorer, baseline)
    # Judge sandboxes require a reference mount even for this read-only
    # exploration stage. Keep it empty and private: the explorer should learn
    # from the candidate snapshot and QA, not from task acceptance artifacts.
    explorer_reference = explorer / "reference"
    explorer_reference.mkdir(parents=True, exist_ok=True)
    qa = item.get("qa", {})
    lines = [REPOSITORY_EXPLORER, "\nQA 定位问题:",
             "QUESTION: " + str(qa.get("question", "")),
             "TYPE: " + str(qa.get("type", "")),
             "不要读取或猜测答案；只用问题中的对象定位当前代码。"]
    remaining = dict(max_requests=min(40, options.get("max_requests", 80)),
                     max_tokens=min(300000, options.get("max_tokens", 1500000)),
                     max_seconds=600)
    outcome = run_agent(explorer, config, "judge", "\n".join(lines),
                        reference=explorer_reference, **remaining)
    report = explorer / "workspace/checks/repository-exploration.md"
    text = report.read_text(encoding="utf-8") if report.is_file() else ""
    save(root / "repository-exploration.json", {
        "status": outcome.get("status"),
        "metrics": outcome.get("metrics", {}),
        "report": text,
        "final": outcome.get("final", ""),
    })
    return text, outcome


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


def admission(validation, baseline_checks, reference_checks, baseline_acceptance=None,
              reference_acceptance=None):
    """Tests cannot be overridden by a model's PASS; absent tests use explicit review."""
    if not (validation.get("BASELINE") == "unmet" and validation.get("REFERENCE") == "pass"
            and validation.get("VERDICT") == "accept"
            and validation.get("COVERAGE") == "complete"):
        return False
    if validation.get("MUTATIONS") == "missed" or baseline_checks["status"] == "error":
        return False
    if reference_checks["status"] in {"failed", "error"}:
        return False
    if baseline_acceptance is not None and reference_acceptance is not None:
        return (validation.get("TESTS") in {"executable", "partial", "unavailable"}
                and baseline_acceptance["status"] == "failed"
                and reference_acceptance["status"] == "passed"
                and not reference_checks.get("skipped", 0)
                and validation.get("MUTATIONS") == "caught")
    if validation.get("TESTS") == "executable":
        return (baseline_checks["status"] == "failed"
                and reference_checks["status"] == "passed"
                and not reference_checks.get("skipped", 0)
                and validation.get("MUTATIONS") == "caught")
    return (validation.get("TESTS") in {"partial", "unavailable"}
            and validation.get("MUTATIONS") in {"caught", "unavailable"})


def validated_spec(spec, validator_checks, output, *, allow_new_tests=True):
    """Persist the validator's checks in the exact spec used by both trials."""
    coverage = Path(validator_checks) / "coverage.md"
    if not coverage.is_file() or not coverage.read_text().strip():
        return None
    copy_tree(spec, output)
    for name in (("coverage.md", "test_interactions.py") if allow_new_tests else ("coverage.md",)):
        source = Path(validator_checks) / name
        if source.is_file():
            shutil.copy2(source, Path(output) / name)
    return Path(output)


def agent_finished(outcome):
    return str(outcome.get("status")) in {"finished", "ConversationExecutionStatus.FINISHED"}


def inspect_acceptance(candidate, spec, items, checks, root, config, agent_options, *, budget=None):
    """Inspect this candidate against frozen criteria, retaining its own evidence."""
    reference = root / "judge-reference"
    copy_tree(spec, reference / "spec")
    if (reference / "spec/history.json").exists():
        # Expose criteria and observations, never condition labels or injected answers.
        for name in ("history-review.md", "memory-use.md"):
            (reference / "spec" / name).unlink(missing_ok=True)
        judge_history = read(reference / "spec/history.json")
        for key in ("oracle_answer", "reference_information", "oracle_sufficiency"):
            judge_history.pop(key, None)
        save(reference / "spec/history.json", judge_history)
    save(reference / "checks.json", checks)
    judge = root / "judge"
    prepare(judge, candidate)
    roots = {"/workspace/candidate": judge / "workspace/candidate",
             "/workspace/checks": judge / "workspace/checks",
             "/workspace/experiments": judge / "workspace/experiments"}
    judged = {"status": "not_needed"}
    if any(not row["tests"] for row in items):
        try:
            options = budget.remaining() if budget is not None else agent_options
            judged = run_agent(judge, config, "judge", prompts.JUDGE, reference=reference, **options)
        except Exception as error:
            result_path = judge / "result.json"
            judged = read(result_path) if result_path.exists() else {}
            judged.update(status="error", error_type=type(error).__name__, detail=str(error))
            save(result_path, judged)
        if budget is not None and "metrics" in judged:
            metrics = judged["metrics"]
            budget.record([dict(request_count=metrics.get("attempted_requests", 0),
                **({k: metrics[k] for k in ("prompt_tokens", "completion_tokens") if k in metrics}
                   if metrics.get("usage_complete") else {}))])
        if not agent_finished(judged) or (budget is not None and not budget.usage_complete):
            return judged, None, roots
    return judged, judge / "workspace/checks/acceptance-review.txt", roots


def load_preparation(attempt, item, baseline, public_history):
    """Reuse model-authored criteria only for the same QA, evidence and baseline."""
    attempt = Path(attempt).resolve()
    task = attempt.parent
    authored = read(attempt / "author/result.json")
    spec = attempt / "author/workspace/checks"
    if (not agent_finished(authored)
            or read(task / "author-reference/qa.json") != item["qa"]
            or read(task / "author-reference/qa-input.json") != read(item["generation_input"])
            or read(task.parent / "manifest.json")["baseline_sha256"] != fingerprint(baseline)):
        raise ValueError("Prepared task does not match completed author, QA, evidence or baseline")
    protected = ["task.md", "memory-use.md"]
    if item.get("qa_source") == "external":
        protected.append("oracle-answer.json")
    history = read(spec / "history.json") if (spec / "history.json").exists() else None
    if public_history:
        if not history or any(history[key] != public_history[key] for key in ("events", "cutoff_event_id")):
            raise ValueError("Prepared historical sources differ")
        protected.append("history-contract.txt")
    elif history:
        raise ValueError("Prepared history is not available to this QA")
    if any((spec / name).read_bytes() != (attempt / "qualified-draft" / name).read_bytes()
           for name in protected):
        raise ValueError("Prepared task changed after qualification")
    def requirements(directory):
        return [{key: row[key] for key in ("id", "requirement", "basis")}
                for row in acceptance_items(directory, history)]
    if requirements(spec) != requirements(attempt / "qualified-draft"):
        raise ValueError("Prepared acceptance requirements changed")
    return read(task / "selection/result.json"), authored, spec


def construct(item, root, baseline, config, revisions, agent_options, *, design_probe=False,
              selection_only=False, reuse_preparation=None, preparation_feedback=None):
    feedback = ""
    reference = root / "author-reference"
    reference.mkdir(parents=True)
    save(reference / "qa.json", item["qa"])
    shutil.copy2(item["generation_input"], reference / "qa-input.json")
    save(reference / "provenance.json", {k: v for k, v in item.items() if k not in {"qa", "public_records"}})
    # Use the exact source closure saved with the QA request for graph-mode
    # tasks.  External-only tasks keep the dialogue only as QA provenance: the
    # saved answer is the sole private information supplied to the memory arm.
    qa_source_ids = set()
    candidate = item.get("reviewed_candidate", item.get("original_candidate", {}))
    for point in candidate.get("answer_points", []) + candidate.get("forbidden_points", []):
        if isinstance(point, dict):
            qa_source_ids.update(source for source in point.get("sources", [])
                                 if isinstance(source, str) and source)
    # External-only QA intentionally keeps its source dialogue as provenance,
    # but does not turn that dialogue into a task-time history contract.  The
    # task experiment compares the saved external answer with no answer; if we
    # materialize a public history here, the construction path can leak the
    # same rule into task authoring and both solver arms receive a different
    # contract than the one being tested.
    public_history = (prepare_history(
        item["public_records"], item["generation_input"],
        qa_source_ids=qa_source_ids or None)
                      if item.get("public_records")
                      and item.get("qa_source", "graph") != "external" else None)
    if public_history:
        save(reference / "history.json", public_history)
        save(reference / "history-focus.json", public_history["initial_events"])
    attempts = []
    fixed_draft = None
    fixed_qualification = None
    previous_tests = None
    budget = SelectionBudget(root, agent_options)
    reused = load_preparation(reuse_preparation, item, baseline, public_history) if reuse_preparation else None
    if preparation_feedback:
        if not reused:
            raise ValueError("Preparation feedback requires an existing qualified task")
        feedback = "\n以下是已保存测试的审阅问题。保持需求与历史规则不变，修正测试后结束：\n" + Path(preparation_feedback).read_text()
        (reference / "test-repair-feedback.md").write_text(feedback, encoding="utf-8")
    exploration_text = ""
    # Malformed/offline selection fixtures may not contain a question.  There
    # is nothing meaningful for a repository explorer to anchor on, and
    # selection-only mode explicitly promises not to start OpenHands.
    if not reused and not selection_only and item.get("qa", {}).get("question"):
        try:
            exploration_text, exploration = explore_repository(
                root, baseline, item, config, budget.remaining(), public_history)
            metrics = exploration.get("metrics", {})
            # Charge attempted calls even if exploration failed. A zero-call
            # startup failure consumes nothing; missing provider usage stays unknown.
            usage = {"request_count": metrics.get("attempted_requests", 0)}
            if metrics.get("usage_complete"):
                usage.update({k: metrics[k] for k in ("prompt_tokens", "completion_tokens")
                              if k in metrics})
            budget.record([usage])
        except Exception as error:
            save(root / "repository-exploration.json", {
                "status": "error", "error_type": type(error).__name__,
                "detail": str(error), "report": "",
            })
    selection = (reused[0] if reused else select_task(item["qa"], public_history, baseline, config,
                 root / "selection", budget, exploration=exploration_text,
                 workflow=item.get("development_workflow")))
    if reused:
        save(root / "selection/result.json", selection)
    selection["qa_source"] = item.get("qa_source", "graph")
    if selection["status"] != "candidate":
        save(root / "construction.json", [{"attempt": 0, "accepted": False, "status": selection["status"],
                                           "reason": selection["reason"]}])
        return None
    if selection.get("history_targets"):
        save(reference / "history-targets.json", selection["history_targets"])
    for attempt in range(revisions + 1):
        run = root / ("construction-%02d" % attempt)
        author = run / "author"
        print(root.name, "author", attempt, flush=True)
        spec = author / "workspace/checks"
        gate_state = {}
        draft_selection = dict(selection)
        draft_selection["historical_question"] = item["qa"].get("question", "")
        if item.get("qa_source") == "external":
            draft_selection["historical_answer"] = answer_text(item["qa"])
        if public_history:
            draft_selection.update(public_history=public_history,
                                   historical_answer=answer_text(item["qa"]),
                                   history_targets=selection.get("history_targets", {}))

        def qualify():
            if (spec / "NO_TASK.md").exists():
                gate_state["review"] = {"status": "ineligible", "issue": (spec / "NO_TASK.md").read_text()}
                return gate_state["review"]
            try:
                frozen_targets = (selection.get("history_targets", {})
                                  if selection.get("history_targets", {}).get("targets") else None)
                draft_history = (freeze_contract(spec, public_history, answer_text(item["qa"],),
                                                 require_external=False, targets=frozen_targets)
                                 if public_history else None)
                draft_items = acceptance_items(spec, draft_history)
                if item.get("qa_source") == "external":
                    if read(spec / "oracle-answer.json") != {"answer": answer_text(item["qa"])}:
                        raise ValueError("Private acceptance answer differs from the injected QA answer")
                protected = {name: (spec / name).read_text() for name in
                             ("task.md", "memory-use.md") + (("history-contract.txt",) if public_history else ())
                             + (("oracle-answer.json",) if item.get("qa_source") == "external" else ())}
                reviewed_task, reviewed_answer = protected["task.md"], answer_text(item["qa"])
                if fixed_qualification is not None:
                    requirements = [{k: r[k] for k in ("id", "requirement", "basis")} for r in draft_items]
                    if (protected != fixed_qualification["protected"]
                            or requirements != fixed_qualification["requirements"]):
                        raise ValueError("qualified_draft_changed")
                    if (draft_history != fixed_qualification["history"]
                            or reviewed_answer != fixed_qualification["answer"]
                            or (draft_history and not qualified_oracle_complete(
                                draft_history, fixed_qualification["review"]))):
                        raise ValueError("qualified_history_not_verified")
                    gate_state.update(deepcopy(fixed_qualification))
                    decision = dict(gate_state["review"], usage=[],
                                    reused_from=str(fixed_draft.parent.relative_to(root) / "task-review"))
                    gate_state["review"] = decision
                    save(run / "task-review/result.json", decision)
                    copy_tree(spec, run / "qualified-draft")
                    return decision
                refs = {ref for c in (draft_history or {}).get("contracts", []) for ref in c["sources"]}
                evidence = {"memory_use": protected["memory-use.md"], "acceptance": draft_items,
                            "qa_source": item.get("qa_source", "graph"),
                            "development_workflow": item.get("development_workflow", ""),
                            "repository_exploration": selection.get("repository_exploration", ""),
                            "repository_queries": [q for q in selection.get("evidence", {}).get("queries", [])
                                                   if q["query"]["target"] == "repo"],
                            "contracts": (draft_history or {}).get("contracts", []),
                            "history_targets": (frozen_targets or {}).get("targets", []),
                            "sources": [e for e in (public_history or {}).get("events", [])
                                        if e["id"] in refs or e.get("role") == "user"]}
                decision = review_task(reviewed_task, reviewed_answer,
                                       config, run / "task-review", evidence=evidence, budget=budget)
                if decision.get("status") == "clean" and frozen_targets:
                    # H is immutable; only the finite review may classify its
                    # applicability and repository availability.
                    write_contract_from_targets(spec, frozen_targets, decision)
                    inactive = {row["id"] for row in decision["history_rows"]
                                if row["applicable"] == "no"}
                    if any(inactive.intersection(row["basis"]) for row in draft_items):
                        write_private_draft(
                            draft_selection, config, run / "draft/private-applicability",
                            spec, budget, history_review=decision,
                            feedback="Only the supplied historical targets apply to this task. "
                                     "Rebuild acceptance from the unchanged public task and these targets.")
                    draft_history = freeze_contract(spec, public_history, answer_text(item["qa"]),
                                                    targets=frozen_targets)
                    draft_items = acceptance_items(spec, draft_history)
                    protected = {name: (spec / name).read_text() for name in
                                 ("task.md", "memory-use.md", "history-contract.txt")}
                if draft_history and decision.get("status") == "clean" and (
                        draft_history["public_task"] != reviewed_task
                        or draft_history["oracle_answer"] != reviewed_answer
                        or not qualified_oracle_complete(draft_history, decision)):
                    decision = dict(decision, status="uncertain", issue="historical_answer_not_verified")
                    save(run / "task-review/result.json", decision)
                gate_state.update(review=decision, protected=protected, history=draft_history, answer=reviewed_answer,
                                  requirements=[{k: r[k] for k in ("id", "requirement", "basis")} for r in draft_items])
                copy_tree(spec, run / "qualified-draft")
                return decision
            except (ValueError, KeyError, OSError) as error:
                decision = {"status": "uncertain", "issue": str(error)}
                gate_state["review"] = decision
                save(run / "task-review/result.json", decision)
                return decision

        try:
            if fixed_draft:
                copy_tree(fixed_draft, spec)
            elif reused and attempt == 0:
                copy_tree(reused[2], spec)
            else:
                write_draft(draft_selection, config, run / "draft", spec, budget, feedback=feedback)
            task_review = qualify()
            if task_review["status"] != "clean":
                attempts.append(dict(attempt=attempt, accepted=False, reason="task_" + task_review["status"],
                                     status="pending" if task_review["status"] == "uncertain" else "stop",
                                     task_review=task_review))
                break
            if fixed_draft is None:
                fixed_draft = run / "qualified-draft"
                fixed_qualification = deepcopy(gate_state)
            if selection_only:
                save(root / "construction.json", [dict(attempt=attempt, status="qualified",
                     accepted=False, task_review=task_review)])
                return {"selection_only": True, "status": "qualified"}
            if previous_tests:
                # Qualification stays fixed; repairs consume the complete suite
                # that produced the latest feedback, including validator checks.
                shutil.copytree(previous_tests, spec, dirs_exist_ok=True)
            remaining = budget.remaining()
            private_memory_answer = (answer_text(item["qa"])
                                     if item.get("qa_source") == "external" else "")
            if reused and attempt == 0 and not preparation_feedback:
                authored = reused[1]
            else:
                if preparation_feedback or previous_tests:
                    if private_memory_answer:
                        authored = repair_tests(
                            spec, baseline, config, author, budget, feedback,
                            private_memory_answer=private_memory_answer)
                    else:
                        authored = repair_tests(spec, baseline, config, author, budget, feedback)
                elif private_memory_answer:
                    authored = write_tests(
                        spec, baseline, config, author, budget, feedback,
                        private_memory_answer=private_memory_answer)
                else:
                    authored = write_tests(spec, baseline, config, author, budget, feedback)
                if authored is None:
                    prepare(author, baseline)
                    test_reference = prepare_test_reference(spec, reference, run / "test-reference")
                    authored = run_agent(author, config, "judge", prompts.AUTHOR_TESTS + feedback,
                                         system=prompts.PREPARATION_SYSTEM, reference=test_reference, **remaining)
                    metrics = authored.get("metrics", {})
                    budget.record([dict(request_count=metrics.get("attempted_requests", 0),
                                        **({k: metrics[k] for k in ("prompt_tokens", "completion_tokens") if k in metrics}
                                           if metrics.get("usage_complete") else {}))])
            budget.record([])
        except Exception as error:
            attempts.append(dict(attempt=attempt, accepted=False, status="pending", reason=str(error)))
            break
        record = {"attempt": attempt, "author_status": authored["status"],
                  "author_metrics": authored.get("metrics", {}),
                  "construction_budget": read(root / "selection-budget.json")}
        if reused and attempt == 0:
            record["preparation_reused_from"] = str(Path(reuse_preparation).resolve())
            if not preparation_feedback:
                record["author_reused_from"] = str(Path(reuse_preparation).resolve())
            else:
                record["test_repair_feedback"] = "author-reference/test-repair-feedback.md"
        if authored.get("error_code") or authored.get("error_type"):
            record["author_error"] = {k: authored[k] for k in ("error_code", "error_type", "detail") if k in authored}
        attempts.append(record)
        if (spec / "NO_TASK.md").exists():
            record["reason"] = (spec / "NO_TASK.md").read_text()
            break
        if authored.get("error_code") in {"token_budget_exhausted", "request_budget_exhausted", "runtime_budget_exhausted"}:
            record["reason"] = authored["error_code"]
            break
        required = ("task.md", "acceptance.md")
        if any(not (spec / name).exists() for name in required):
            record["reason"] = authored.get("error_code", "missing_author_artifacts")
            if record["reason"] in {"token_budget_exhausted", "request_budget_exhausted", "runtime_budget_exhausted"}:
                save(root / "construction.json", attempts)
                break
            feedback = ("\n上一轮状态：%s。没有写齐 task.md、acceptance.md。"
                        "请完成文件。搜索无结果不代表环境故障，不要反复执行同一空搜索；"
                        "无法提出适当需求时写 NO_TASK.md。" % authored["status"])
            save(root / "construction.json", attempts)
            continue
        task_review = gate_state.get("review", {"status": "uncertain", "issue": "qualification_not_completed"})
        record["task_review"] = task_review
        if task_review["status"] != "clean":
            record.update(accepted=False, reason="task_" + task_review["status"])
            prior = reference / ("previous-%02d" % attempt)
            copy_tree(spec, prior)
            save(root / "construction.json", attempts)
            break
        changed = [name for name, text in gate_state["protected"].items()
                   if not (spec / name).exists() or (spec / name).read_text() != text]
        if changed or not agent_finished(authored):
            record.update(accepted=False, status="pending", changed_qualified_files=changed,
                          reason="qualified_draft_changed" if changed else "author_incomplete")
            break
        previous_tests = spec
        history = None
        if public_history:
            try:
                frozen_targets = (selection.get("history_targets", {})
                                  if selection.get("history_targets", {}).get("targets") else None)
                history = freeze_contract(spec, public_history, answer_text(item["qa"]),
                                          targets=frozen_targets)
            except (ValueError, KeyError, OSError) as error:
                record.update(accepted=False, reason="invalid_history_contract", detail=str(error))
                feedback = "\n上一轮历史契约错误，请根据公开来源重写：" + str(error)
                save(root / "construction.json", attempts)
                continue
        oracle_complete = not history or (
            history == read(run / "qualified-draft/history.json")
            and qualified_oracle_complete(history, task_review))
        record["oracle_complete"] = oracle_complete
        if not oracle_complete:
            record.update(accepted=False, reason="qualified_history_not_verified")
            save(root / "construction.json", attempts)
            break
        try:
            items = acceptance_items(spec, history)
            if [{k: r[k] for k in ("id", "requirement", "basis")} for r in items] != gate_state["requirements"]:
                raise ValueError("Qualified acceptance requirements changed during test construction")
            save(spec / "acceptance.json", items)
        except ValueError as error:
            record.update(accepted=False, reason="invalid_acceptance", detail=str(error))
            feedback = "\n验收表需要修正：" + str(error)
            save(root / "construction.json", attempts)
            continue
        baseline_checks = run_checks(baseline, spec, run / "baseline-checks", config["execution_image"],
                                         candidate_pythonpath=config.get("code", {}).get("candidate_pythonpath"))
        if baseline_checks["status"] == "error":
            record.update(accepted=False, reason="baseline_check_error", baseline_checks=baseline_checks)
            previous_tests = reference / ("previous-%02d" % attempt)
            copy_tree(spec, previous_tests)
            feedback = ("\n上一轮检查未正常执行。已有测试保存在 /reference/previous-%02d。"
                        "修正测试收集、依赖或路径问题，保留要求和历史规则。"
                        "尚不存在的新接口须在测试函数内导入。执行结果：%s" % (attempt, baseline_checks))
            save(root / "construction.json", attempts)
            continue
        implementation = run / "reference-solver"
        prepare(implementation, baseline)
        print(root.name, "reference implementation", attempt, flush=True)
        reference_answer = reference_solver_answer(item, history)
        solved = run_agent(implementation, config, "code",
                           solver_input((spec / "task.md").read_text(), reference_answer), **agent_options)
        candidate = implementation / "workspace/candidate"
        record["reference_version"] = export_change(baseline, candidate, implementation)
        reference_checks = run_checks(candidate, spec, run / "reference-checks", config["execution_image"],
                                         candidate_pythonpath=config.get("code", {}).get("candidate_pythonpath"))
        record.update(baseline_checks=baseline_checks, reference_checks=reference_checks,
                      reference_status=solved["status"])
        if reference_checks["status"] in {"failed", "error"}:
            record.update(accepted=False, reason="reference_check_" + reference_checks["status"])
            previous_tests = reference / ("previous-%02d" % attempt)
            copy_tree(spec, previous_tests)
            feedback = ("\n参考实现未通过实际检查。先核查测试调用与题面是否一致；"
                        "测试有错则修正，实现有错则保留能发现问题的测试。"
                        "不要降低已冻结的要求。下一轮会重新生成参考实现。\n"
                        "执行结果：%s\n本次参考实现改动（用于定位，不是正确性标准）：\n%s" % (
                            dict(reference_checks, cases=[case for case in reference_checks.get("cases", [])
                                 if case.get("status") not in {"passed", "skipped"}]),
                            (implementation / "changes.patch").read_text()))
            save(root / "construction.json", attempts)
            continue
        preflight_budget = SelectionBudget(run / "preflight", agent_options)
        source_review = None
        if history:
            source_review = review_sources(history, config, run / "history-review", preflight_budget)
            record["history_source_review"] = source_review
            if source_review["support"] != "supported":
                record.update(accepted=False, reason="history_source_not_verified")
                save(root / "construction.json", attempts)
                break
            coverage_review = review_checks(spec, baseline, candidate,
                record["reference_version"]["changed_files"],
                {"baseline": baseline_checks, "reference": reference_checks},
                config, run / "checks-review", preflight_budget)
            record["checks_review"] = coverage_review
            if coverage_review["status"] != "complete":
                record.update(accepted=False, reason="checks_" + coverage_review["status"])
                save(root / "construction.json", attempts)
                if coverage_review["status"] == "uncertain":
                    break
                previous_tests = reference / ("previous-%02d" % attempt)
                copy_tree(spec, previous_tests)
                feedback = "\n上一轮测试审核发现具体问题，请保留目标并修正：\n" + str(coverage_review)
                continue
        validation_reference = run / "validator-reference"
        copy_tree(spec, validation_reference / "spec")
        for name in ("history.json", "history-review.md"):
            (validation_reference / "spec" / name).unlink(missing_ok=True)
        copy_tree(candidate, validation_reference / "implementation")
        save(validation_reference / "checks.json", record)
        validator = run / "validator"
        print(root.name, "preflight validation", attempt, flush=True)
        if history and not any(item["check"].startswith("inspect:") for item in items):
            validated = write_history_mutation(spec, candidate, record["reference_version"]["changed_files"],
                                                config, validator, preflight_budget)
        else:
            prepare(validator, baseline)
            validated = run_agent(validator, config, "judge", prompts.HISTORY_MUTATION if history else prompts.VALIDATOR,
                                  system=prompts.PREPARATION_SYSTEM,
                                  reference=validation_reference, **preflight_budget.remaining())
        verdict_file = validator / "workspace/checks/validation.txt"
        feedback = verdict_file.read_text() if verdict_file.exists() else "验收者未完成验证，请核查需求和测试。"
        record["validation"] = labels(feedback)
        if source_review:
            record["validation"] = {
                "BASELINE": "uncertain", "REFERENCE": "uncertain",
                "TESTS": "executable" if reference_checks.get("cases") else "unavailable",
                "MUTATIONS": "unavailable", "COVERAGE": coverage_review["status"],
                "VERDICT": "accept" if coverage_review["status"] == "complete" else "revise",
                "HISTORY": source_review["support"],
            }
            feedback = (run / "checks-review/coverage.md").read_text()
        record["validation_evidence"] = feedback
        record["validator_status"] = validated["status"]
        record["validator_method"] = validated.get("method", "openhands")
        record["validator_metrics"] = validated.get("metrics", {})
        metrics = record["validator_metrics"]
        preflight_budget.record([] if validated.get("method") == "model_file_generation" else
            [dict(request_count=metrics.get("attempted_requests", 0),
            **({k: metrics[k] for k in ("prompt_tokens", "completion_tokens") if k in metrics}
               if metrics.get("usage_complete") else {}))])
        record["preflight_budget"] = read(run / "preflight/selection-budget.json")
        if not agent_finished(validated):
            record.update(accepted=False, reason="validator_incomplete",
                          validator_error={k: validated[k] for k in
                                           ("error_code", "error_type", "detail") if k in validated})
            save(root / "construction.json", attempts)
            break
        if history:
            shutil.copy2(run / "checks-review/coverage.md", validator / "workspace/checks/coverage.md")
        final_spec = validated_spec(spec, validator / "workspace/checks", run / "validated-spec",
                                    allow_new_tests=not history)
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

        memory_check = bool(history) or item.get("qa_source") == "external"
        if memory_check and final_spec is not None and oracle_complete:
            def inspect_mutant(mutant, criteria, checks, output):
                _, review_path, roots = inspect_acceptance(
                    mutant, criteria, items, checks, output, config, agent_options, budget=preflight_budget)
                return review_path, roots

            record["history_mutations"] = check_history_mutations(
                candidate, final_spec, validator / "workspace/checks",
                run / "history-mutations", config["execution_image"],
                candidate_pythonpath=config.get("code", {}).get("candidate_pythonpath"), inspector=inspect_mutant)
            record["validation"]["MUTATIONS"] = record["history_mutations"]["status"]
            record["preflight_budget"] = read(run / "preflight/selection-budget.json")
            if record["history_mutations"].get("detail"):
                feedback += "\n历史变体验证错误：" + record["history_mutations"]["detail"]
                record["validation_evidence"] = feedback
        reference_acceptance = (assess_acceptance(items, reference_checks,
            validator / "workspace/checks/acceptance-review.txt",
            {"/reference/implementation": candidate,
             "/workspace/checks": validator / "workspace/checks",
             "/workspace/experiments": validator / "workspace/experiments"})
            if final_spec else {"status": "uncertain"})
        record["reference_acceptance"] = reference_acceptance
        baseline_acceptance = None
        if memory_check and final_spec is not None:
            baseline_acceptance = assess_acceptance(items, baseline_checks)
            if (any(not item["tests"] for item in items)
                    and baseline_acceptance["status"] != "failed"
                    and baseline_checks["status"] in {"passed", "unavailable"}
                    and reference_acceptance["status"] == "passed"
                    and record["history_mutations"]["status"] == "caught"):
                try:
                    judged, review_path, roots = inspect_acceptance(
                        baseline, final_spec, items, baseline_checks, run / "baseline-inspection",
                        config, agent_options, budget=preflight_budget)
                    record["baseline_inspection"] = {
                        key: judged[key] for key in ("status", "error_type", "detail", "metrics")
                        if key in judged}
                except Exception as error:
                    review_path, roots = None, None
                    record["baseline_inspection"] = {
                        "status": "error", "error_type": type(error).__name__, "detail": str(error)}
                baseline_acceptance = assess_acceptance(items, baseline_checks, review_path, roots)
                record["preflight_budget"] = read(run / "preflight/selection-budget.json")
            record["baseline_acceptance"] = baseline_acceptance
            record["validation"].update(
                BASELINE={"failed": "unmet", "passed": "met"}.get(baseline_acceptance["status"], "uncertain"),
                REFERENCE={"passed": "pass", "failed": "fail"}.get(reference_acceptance["status"], "uncertain"))
        if not agent_finished(validated):
            record["reason"] = "validator_incomplete"
        elif final_spec is None:
            record["reason"] = "missing_coverage_checks"
        elif memory_check and record.get("history_mutations", {}).get("status") != "caught":
            record["reason"] = "historical_mutation_not_verified"
        record["validation_accepted"] = (agent_finished(solved) and agent_finished(validated)
                                         and final_spec is not None
                                         and reference_acceptance["status"] == "passed"
                                         and oracle_complete
                                         and (not memory_check or record["history_mutations"]["status"] == "caught")
                                         and (not history or (
                                             record["validation"].get("HISTORY") == "supported"
                                             and record["history_mutations"]["status"] == "caught"))
                                         and admission(record["validation"], baseline_checks, reference_checks,
                                                       baseline_acceptance, reference_acceptance))
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
                           qa_sha256=qa_fingerprint(item["qa"]),
                           validation=record["validation"],
                           baseline_checks=baseline_checks, reference_checks=reference_checks)
            if history:
                receipt.update(comparison="without_memory_vs_oracle_history",
                               reference_information=history["reference_information"],
                               oracle_sufficiency="validated_against_external_rules")
            save(root / "frozen.json", receipt)
            return receipt
        # New author conversation receives the previous artifacts and concrete verifier feedback.
        prior = reference / ("previous-%02d" % attempt)
        copy_tree(final_spec or spec, prior)
        previous_tests = prior
        feedback = ("\n上一轮未通过。阅读 /reference/previous-%02d。依据以下具体问题修正，"
                    "重新写出完整文件，勿降低任务原有正确性标准：\n%s\n"
                    "补充检查实际重跑：基线 %s；参考实现 %s。%s" % (
                        attempt, feedback, baseline_checks, reference_checks, record.get("reason", "")))
    save(root / "construction.json", attempts)
    return None


def evaluate(item, root, baseline, receipt, config, agent_options, index):
    if "qa_sha256" in receipt and receipt["qa_sha256"] != qa_fingerprint(item["qa"]):
        raise ValueError("Frozen QA changed before evaluation")
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
        judged, review_path, roots = inspect_acceptance(
            candidate, spec, items, checks, trial, config, agent_options)
        judge = trial / "judge"
        verdict_path = judge / "workspace/checks/verdict.txt"
        verdict = verdict_path.read_text() if verdict_path.exists() else ""
        acceptance = assess_acceptance(items, checks, review_path, roots)
        status = acceptance["status"]
        result[condition] = {"result": status, "solver_status": solved["status"],
                             "judge_status": judged["status"],
                             "metrics": solved["metrics"], "checks": checks,
                             "history_available": history is not None,
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
    parser.add_argument("--selection-only", action="store_true",
                        help="Stop after controlled selection, draft and qualification; no OpenHands execution")
    parser.add_argument("--reuse-preparation", type=Path,
                        help="Reuse one completed construction-NN author; requalify and rerun preflight in a new output")
    parser.add_argument("--preparation-feedback", type=Path,
                        help="Repair reused tests from saved review feedback before rerunning preflight")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--baseline", type=Path,
                        help="Already pinned independent dialogue-end repository")
    parser.add_argument("--task-budget", type=int,
                        help="Maximum distinct QA-derived requirements to attempt (default: twice count)")
    parser.add_argument("--revisions", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--agent-requests", type=int, default=80)
    parser.add_argument("--agent-tokens", type=int, default=1500000)
    parser.add_argument("--model-request-chars", type=int, default=60000,
                        help="Maximum serialized task-construction request size, including evidence and prompt")
    args = parser.parse_args(argv)
    task_budget = args.task_budget if args.task_budget is not None else args.count * 2
    if min(args.count, args.workers, args.agent_requests, args.agent_tokens,
           args.model_request_chars) < 1 or args.revisions < 0:
        parser.error("Counts and budgets must be positive; revisions must be nonnegative")
    if task_budget < 1:
        parser.error("Task budget must be positive")
    if args.preparation_feedback and not args.reuse_preparation:
        parser.error("--preparation-feedback requires --reuse-preparation")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists() or (output / "baseline").exists():
        parser.error("Use a new output directory; previous experiments are retained")
    items = qa_inputs(args.qa_run)
    if not items:
        parser.error("No approved QA with saved generation inputs")
    if args.reuse_preparation:
        source_qa = read(args.reuse_preparation.parent / "author-reference/qa.json")
        items = [item for item in items if item["qa"] == source_qa]
        if len(items) != 1 or args.count != 1:
            parser.error("Reusing preparation requires its exact QA and --count 1")
    config = configure(args.simulator_path, args.source_run / "private/checkpoint.json", args.env_file,
                       **({"control_config": args.control_config} if args.control_config else {}))
    config["model_request_chars"] = args.model_request_chars
    baseline = args.baseline.resolve() if args.baseline else output / "baseline"
    if args.baseline:
        version = baseline_version(baseline)
    else:
        copy_tree(args.source_run / "workspace/candidate", baseline)
        version = pin_baseline(baseline)
    save(output / "baseline.json", dict(version,
         source=str(baseline if args.baseline else (args.source_run / "workspace/candidate").resolve())))
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
                "evaluator_version": source_version(Path(__file__).resolve().parents[2], "dialogue_benchmark"),
                "simulator_version": source_version(args.simulator_path, "simulator"),
                "execution": {"workers": args.workers, "revisions": args.revisions,
                              "agent_requests": args.agent_requests, "agent_tokens": args.agent_tokens,
                              "agent_seconds": 1200, "design_probe": args.design_probe,
                              "reuse_preparation": str(args.reuse_preparation.resolve()) if args.reuse_preparation else None,
                              "preparation_feedback": str(args.preparation_feedback.resolve()) if args.preparation_feedback else None},
                "comparison": "Historical answer injection; no memory retriever",
                "baseline_version": version, "baseline": str(baseline),
                "target": args.count, "task_budget": task_budget,
                "selection_only": args.selection_only,
                "selected_qa_ids": [i["qa"]["id"] for i in selected], "tasks": []}
    save(output / "manifest.json", manifest)
    agent_options = {"max_requests": args.agent_requests, "max_tokens": args.agent_tokens}

    def run(index, item):
        root = output / ("task-%02d" % (index + 1))
        try:
            receipt = construct(item, root, baseline, config, args.revisions, agent_options,
                                **({"design_probe": True} if args.design_probe else {}),
                                selection_only=args.selection_only,
                                **({"reuse_preparation": args.reuse_preparation} if args.reuse_preparation else {}),
                                **({"preparation_feedback": args.preparation_feedback} if args.preparation_feedback else {}))
            if receipt is None:
                records = read(root / "construction.json") if (root / "construction.json").exists() else [{}]
                return {"task": root.name, "status": records[-1].get("status", "not_admitted"), "qa_id": item["qa"]["id"],
                        "type": item["qa"]["type"]}
            if args.selection_only:
                return {"task": root.name, "status": "qualified", "qa_id": item["qa"]["id"],
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
            completed = sum(task["status"] == ("qualified" if args.selection_only else "evaluated") for task in manifest["tasks"])
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
    completed = sum(task["status"] == ("qualified" if args.selection_only else "evaluated") for task in manifest["tasks"])
    manifest.update(completed=completed, shortfall=max(0, args.count - completed),
                    stop_reason="target_met" if completed >= args.count else
                    "task_budget_exhausted" if len(selected) >= task_budget else "qa_pool_exhausted")
    save(output / "manifest.json", manifest)
    write_report(output, manifest)
    print("Task pilot complete:", output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
