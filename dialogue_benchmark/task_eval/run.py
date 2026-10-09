"""Generate, validate, freeze, and evaluate repository tasks from historical QA."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import time

from . import prompts
from .artifacts import (copy_tree, fingerprint, labels, qa_fingerprint, qa_inputs,
                        group_qa_inputs, generation_request, read, save, write_diff)
from .checks import (run_checks, acceptance_items, acceptance_has_inspect,
                     _acceptance_row_cells,
                     assess_acceptance, check_history_mutations, _inspect_command_cases)
from .metrics import compare_trials
from .runtime import (bounded_model_config, configure, preflight_openhands_runtime,
                      review_task, review_checks, repair_tests, write_tests,
                      write_history_mutation, run_agent, shared_task_slot)
from .report import write_report
from .versions import baseline_version, export_change, pin_baseline, source_version
from .history import (prepare_history, freeze_contract, historical_context, read_history_review,
                      write_contract_from_targets, review_sources, qualified_oracle_complete)
from .selection import SelectionBudget, select_task, write_draft, write_private_draft

# The task and historical obligations are frozen after qualification.  The
# test author is explicitly allowed to replace only the Check column with an
# executable test or command, so it must not participate in this signature.
_FROZEN_ACCEPTANCE_FIELDS = ("id", "requirement", "basis")


def acceptance_signature(rows):
    return [{key: row[key] for key in _FROZEN_ACCEPTANCE_FIELDS} for row in rows]


def solver_input(task, answer=None):
    message = prompts.SOLVER + "\n\n" + task
    if answer is not None:
        message += "\n\n历史经验（描述过去，适用性请结合当前需求判断）：\n" + answer
    return message


def implementation_pollution(changed_files):
    """Return candidate paths that alter frozen checks or their inputs.

    The scored repository is the candidate only.  Changes under test/fixture
    directories (or to task/acceptance artifacts) are therefore not normal
    implementation work and must not silently count as a passing delivery.
    """
    protected_roots = {"test", "tests", "fixtures", "fixture", "testdata", "test-data"}
    protected_names = {
        "task.md", "acceptance.md", "acceptance.json", "history.json",
        "history-contract.txt", "memory-use.md", "history-review.md",
        "tests_unavailable.md", "oracle-answer.json", "applicable-answer.txt",
    }
    polluted = []
    for raw in changed_files or ():
        path = PurePosixPath(str(raw).replace("\\", "/"))
        if any(part.lower() in protected_roots for part in path.parts[:-1]):
            polluted.append(str(raw))
        elif path.name.lower() in protected_names:
            polluted.append(str(raw))
    return sorted(set(polluted))


def answer_text(question):
    points = question.get("answer_points", [])
    return "\n".join("- " + (p if isinstance(p, str) else p.get("text", p.get("claim", "")))
                     for p in points)


def available_answer(item):
    text = answer_text(item["qa"])
    globals_text = "\n".join("- " + row["text"] for row in item.get("global_agreements", []))
    return text + ("\n" + globals_text if globals_text else "")


def injected_answer(item, spec):
    path = Path(spec) / "oracle-answer.json"
    return read(path)["answer"] if path.is_file() else answer_text(item["qa"])


def save_task_progress(root, stage, **details):
    """Write a small live marker while a task is between final receipts."""
    payload = {"stage": stage, "updated_at": round(time.time(), 3)}
    payload.update(details)
    save(Path(root) / "progress.json", payload)


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
    # Each remaining inspect row must have one deterministic frozen command.
    # The same command is run for reference, both formal arms, and history
    # mutants; independent Judge calls must not reinterpret its call shape.
    inspect_ids = []
    acceptance = Path(output) / "acceptance.md"
    if acceptance.is_file():
        for line in acceptance.read_text(encoding="utf-8").splitlines():
            cells = _acceptance_row_cells(line)
            if (len(cells) == 4 and cells[0].lower().startswith("a")
                    and acceptance_has_inspect(cells[3].strip())):
                inspect_ids.append(cells[0].lower())
    source_commands = Path(validator_checks) / "commands"
    target_commands = Path(output) / "commands"
    command_files = (sorted(source_commands.glob("inspect-a*.sh"))
                     if source_commands.is_dir() else ())
    for path in command_files:
        if path.is_file():
            target_commands.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target_commands / path.name)
    missing = [identity for identity in inspect_ids
               if not (target_commands / ("inspect-" + identity + ".sh")).is_file()]
    if missing:
        (Path(output) / "missing-inspect-commands.txt").write_text(
            "Missing deterministic inspect commands: " + ", ".join(missing) + "\n",
            encoding="utf-8")
        return None
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
    inspect_rows = [row for row in items if not row["tests"]]
    deterministic = _inspect_command_cases(inspect_rows, checks)
    if inspect_rows and len(deterministic) != len(inspect_rows):
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
            or read(task / "author-reference/qa-input.json") != generation_request(item)
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
        return acceptance_signature(acceptance_items(directory, history))
    if requirements(spec) != requirements(attempt / "qualified-draft"):
        raise ValueError("Prepared acceptance requirements changed")
    return read(task / "selection/result.json"), authored, spec


def construct(item, root, baseline, config, revisions, agent_options, *, design_probe=False,
              selection_only=False, reuse_preparation=None, preparation_feedback=None):
    feedback = ""
    reference = root / "author-reference"
    reference.mkdir(parents=True)
    save(reference / "qa.json", item["qa"])
    save(reference / "qa-input.json", generation_request(item))
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
    reference_feedback = ""
    previous_reference = None
    reuse_reference = False
    repeated_format_errors = set()
    budget = SelectionBudget(root, agent_options)
    reused = load_preparation(reuse_preparation, item, baseline, public_history) if reuse_preparation else None
    if reused and (Path(reuse_preparation) / "reference-solver/result.json").is_file():
        prior_solver = Path(reuse_preparation) / "reference-solver"
        if agent_finished(read(prior_solver / "result.json")):
            previous_reference = prior_solver / "workspace/candidate"
    if reused and (Path(reuse_preparation) / "reference-checks/result.json").is_file():
        prior_checks = read(Path(reuse_preparation) / "reference-checks/result.json")
        if prior_checks["status"] in {"failed", "error"}:
            reference_feedback = "\n已保存的参考实现失败，请核对并修正，不改变固定需求：\n" + str({
                "status": prior_checks["status"],
                "cases": [case for case in prior_checks.get("cases", [])
                          if case.get("status") not in {"passed", "skipped"}]})
    if preparation_feedback:
        if not reused:
            raise ValueError("Preparation feedback requires an existing qualified task")
        feedback = "\n以下是已保存测试的审阅问题。保持需求与历史规则不变，修正测试后结束：\n" + Path(preparation_feedback).read_text()
        (reference / "test-repair-feedback.md").write_text(feedback, encoding="utf-8")
    exploration_text = ""
    save_task_progress(root, "selection")
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
                 workflow=item.get("development_workflow"),
                 global_agreements=item.get("global_agreements", []),
                 qa_pool=[member["qa"] for member in item.get("qa_members", [item])]))
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
        save_task_progress(root, "author", attempt=attempt)
        spec = author / "workspace/checks"
        gate_state = {}
        draft_selection = dict(selection)
        draft_selection["historical_question"] = item["qa"].get("question", "")
        if item.get("qa_members"):
            draft_selection["historical_questions"] = [
                {"question": member["qa"]["question"], "answer": answer_text(member["qa"])}
                for member in item["qa_members"]]
        if item.get("qa_source") == "external":
            draft_selection["historical_answer"] = available_answer(item)
            draft_selection["global_agreements"] = [
                {"text": row["text"], "context": row.get("context", "")}
                for row in item.get("global_agreements", [])]
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
                    answer = injected_answer(item, spec)
                    available = {line.removeprefix("- ").strip() for line in available_answer(item).splitlines()}
                    if not answer.strip() or any(line.removeprefix("- ").strip() not in available
                                                 for line in answer.splitlines() if line.strip()):
                        raise ValueError("Private acceptance answer is not grounded in supplied public facts")
                    used = {line.removeprefix("- ").strip() for line in answer.splitlines()}
                    source_rows = [dict(qa_id=member["qa"]["id"], text=(point if isinstance(point, str)
                                   else point.get("text", point.get("claim", ""))),
                                   sources=point.get("sources", []) if isinstance(point, dict) else [])
                                   for member in item.get("qa_members", [item])
                                   for point in member.get("reviewed_candidate", member["qa"]).get("answer_points", [])
                                   if (point if isinstance(point, str) else point.get("text", point.get("claim", ""))) in used]
                    source_rows.extend(dict(text=row["text"], sources=row["sources"], scope="global")
                                       for row in item.get("global_agreements", []) if row["text"] in used)
                    save(run / "used-history-facts.json", source_rows)
                protected = {name: (spec / name).read_text() for name in
                             ("task.md", "memory-use.md") + (("history-contract.txt",) if public_history else ())
                             + (("oracle-answer.json",) if item.get("qa_source") == "external" else ())}
                reviewed_task, reviewed_answer = protected["task.md"], injected_answer(item, spec)
                if fixed_qualification is not None:
                    requirements = acceptance_signature(draft_items)
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
                            "global_agreements": draft_selection.get("global_agreements", []),
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
                                  requirements=acceptance_signature(draft_items))
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
                save(root / "construction.json", attempts)
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
            private_memory_answer = (injected_answer(item, spec)
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
            code = getattr(error, "code", None)
            record = {"attempt": attempt, "accepted": False, "status": "pending",
                      "reason": code or str(error)}
            attempts.append(record)
            save(root / "construction.json", attempts)
            # A provider protocol/transport failure does not say that the QA
            # or task is invalid.  Spend the explicitly allowed next round on
            # a clean model request, while keeping the failed response visible.
            retryable = {"protocol_error", "response_envelope", "timeout",
                         "connection_error", "http_error"}
            if code in retryable and attempt < revisions:
                feedback = ("\n上一轮模型响应未能按约定格式完成（%s）。"
                            "保持同一题面和历史答案，重新输出所需文件；"
                            "不要新增要求或改写公开需求。" % code)
                continue
            break
        save_task_progress(root, "acceptance", attempt=attempt)
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
            if acceptance_signature(items) != gate_state["requirements"]:
                raise ValueError("Qualified acceptance requirements changed during test construction")
            save(spec / "acceptance.json", items)
        except ValueError as error:
            record.update(accepted=False, reason="invalid_acceptance", detail=str(error))
            signature = str(error).strip()
            if signature in repeated_format_errors:
                record["terminal_reason"] = "repeated_invalid_acceptance"
                save(root / "construction.json", attempts)
                break
            repeated_format_errors.add(signature)
            feedback = "\n验收表需要修正：" + str(error)
            save(root / "construction.json", attempts)
            continue
        baseline_checks = run_checks(baseline, spec, run / "baseline-checks", config["execution_image"],
                                     candidate_pythonpath=config.get("code", {}).get("candidate_pythonpath"))
        save_task_progress(root, "baseline_checks", attempt=attempt)
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
        prepare(implementation, previous_reference or baseline)
        if reuse_reference and previous_reference is not None:
            solved = {"status": "finished",
                      "metrics": {"attempted_requests": 0,
                                  "usage_complete": True},
                      "reused_reference": True}
            reuse_reference = False
        else:
            print(root.name, "reference implementation", attempt, flush=True)
            save_task_progress(root, "reference_solver", attempt=attempt)
            reference_answer = (injected_answer(item, spec) if item.get("qa_source") == "external"
                                else reference_solver_answer(item, history))
            solved = run_agent(implementation, config, "code",
                               solver_input((spec / "task.md").read_text(), reference_answer)
                               + reference_feedback, **agent_options)
        candidate = implementation / "workspace/candidate"
        # A solver can exhaust its request/token/runtime budget after saving a
        # useful partial implementation.  Keep that workspace as the starting
        # point for the next repair attempt; completion is still required for
        # admission below, so this does not weaken the acceptance gate.
        if candidate.is_dir():
            previous_reference = candidate
        record["reference_version"] = export_change(baseline, candidate, implementation)
        reference_checks = run_checks(candidate, spec, run / "reference-checks", config["execution_image"],
                                         candidate_pythonpath=config.get("code", {}).get("candidate_pythonpath"))
        record.update(baseline_checks=baseline_checks, reference_checks=reference_checks,
                      reference_status=solved["status"])
        if reference_checks["status"] in {"failed", "error"}:
            reference_feedback = "\n参考实现上一轮实际失败，请核对并修正，不改变固定需求：\n" + str({
                "status": reference_checks["status"],
                "cases": [case for case in reference_checks.get("cases", [])
                          if case.get("status") not in {"passed", "skipped"}]})
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
        memory_check = bool(history) or item.get("qa_source") == "external"
        source_review = None
        if history:
            source_review = review_sources(history, config, run / "history-review", preflight_budget)
            record["history_source_review"] = source_review
            if source_review["support"] != "supported":
                record.update(accepted=False, reason="history_source_not_verified")
                save(root / "construction.json", attempts)
                break
        coverage_review = None
        if memory_check:
            save_task_progress(root, "checks_review", attempt=attempt)
            coverage_review = review_checks(spec, baseline, candidate,
                record["reference_version"]["changed_files"],
                {"baseline": baseline_checks, "reference": reference_checks},
                config, run / "checks-review", preflight_budget)
            record["checks_review"] = coverage_review
            if coverage_review["status"] != "complete":
                record.update(accepted=False, reason="checks_" + coverage_review["status"])
                save(root / "construction.json", attempts)
                if coverage_review["status"] == "uncertain":
                    # A failed coverage request is an evaluation-stage
                    # failure, not evidence that the task or reference
                    # implementation is wrong. Keep the receipt and stop this
                    # task; do not loop back into AUTHOR.
                    break
                previous_tests = reference / ("previous-%02d" % attempt)
                copy_tree(spec, previous_tests)
                feedback = "\n上一轮测试审核发现具体问题，请保留目标并修正：\n" + str(coverage_review)
                reuse_reference = previous_reference is not None
                continue
        validation_reference = run / "validator-reference"
        copy_tree(spec, validation_reference / "spec")
        for name in ("history.json", "history-review.md"):
            (validation_reference / "spec" / name).unlink(missing_ok=True)
        copy_tree(candidate, validation_reference / "implementation")
        save(validation_reference / "checks.json", record)
        validator = run / "validator"
        print(root.name, "preflight validation", attempt, flush=True)
        save_task_progress(root, "preflight_validation", attempt=attempt)
        if memory_check and not any(acceptance_has_inspect(item["check"]) for item in items):
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
        if coverage_review:
            record["validation"] = {
                "BASELINE": "uncertain", "REFERENCE": "uncertain",
                "TESTS": "executable" if reference_checks.get("cases") else "unavailable",
                "MUTATIONS": "unavailable", "COVERAGE": coverage_review["status"],
                "VERDICT": "accept" if coverage_review["status"] == "complete" else "revise",
                "HISTORY": source_review["support"] if source_review else "not_applicable",
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
        if coverage_review:
            shutil.copy2(run / "checks-review/coverage.md", validator / "workspace/checks/coverage.md")
        final_spec = validated_spec(spec, validator / "workspace/checks", run / "validated-spec",
                                    allow_new_tests=not memory_check)
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
            missing = run / "validated-spec/missing-inspect-commands.txt"
            record["reason"] = ("missing_inspect_commands"
                                if missing.is_file() else "missing_coverage_checks")
            if missing.is_file():
                feedback += "\n" + missing.read_text(encoding="utf-8")

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
        if (final_spec is not None and reference_acceptance["status"] == "uncertain"
                and any(not item["tests"] for item in items)):
            judged, review_path, roots = inspect_acceptance(
                candidate, final_spec, items, reference_checks, run / "reference-inspection",
                config, agent_options, budget=preflight_budget)
            reference_acceptance = assess_acceptance(items, reference_checks, review_path, roots)
            record["reference_acceptance"] = reference_acceptance
            record["reference_inspection"] = {key: judged[key] for key in
                ("status", "error_type", "detail", "metrics") if key in judged}
            record["preflight_budget"] = read(run / "preflight/selection-budget.json")
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
        baseline_already_satisfies = (
            item.get("qa_source") == "external"
            and baseline_acceptance is not None
            and baseline_acceptance.get("status") == "passed"
            and reference_acceptance.get("status") == "passed"
            and not record.get("reference_version", {}).get("changed_files"))
        if not agent_finished(validated):
            record["reason"] = "validator_incomplete"
        elif final_spec is None:
            record["reason"] = ("missing_inspect_commands"
                                if (run / "validated-spec/missing-inspect-commands.txt").is_file()
                                else "missing_coverage_checks")
        elif memory_check and record.get("history_mutations", {}).get("status") != "caught":
            record["reason"] = "historical_mutation_not_verified"
        if baseline_already_satisfies:
            record["reason"] = "baseline_already_satisfies_task"
        admitted = admission(record["validation"], baseline_checks, reference_checks,
                             baseline_acceptance, reference_acceptance)
        record["validation_accepted"] = (not baseline_already_satisfies
                                         and agent_finished(solved) and agent_finished(validated)
                                         and final_spec is not None
                                         and reference_acceptance["status"] == "passed"
                                         and oracle_complete
                                         and (not memory_check or record["history_mutations"]["status"] == "caught")
                                         and (not history or (
                                             record["validation"].get("HISTORY") == "supported"
                                             and record["history_mutations"]["status"] == "caught"))
                                         and admitted)
        if not admitted and not record.get("reason"):
            record["reason"] = "admission_rejected"
        record["accepted"] = False
        save(root / "construction.json", attempts)
        if baseline_already_satisfies:
            break
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
            receipt.update(qa_ids=item.get("qa_ids", [item["qa"]["id"]]),
                           qa_statuses={member["qa"]["id"]: member["qa"].get("status")
                                        for member in item.get("qa_members", [item])},
                           provisional=item.get("provisional", False),
                           source_ids=item.get("source_ids", []))
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


def evaluate(item, root, baseline, receipt, config, agent_options, index, *, resume=False):
    if "qa_sha256" in receipt and receipt["qa_sha256"] != qa_fingerprint(item["qa"]):
        raise ValueError("Frozen QA changed before evaluation")
    spec = root / "frozen"
    task = (spec / "task.md").read_text()
    items = read(spec / "acceptance.json")
    history = read(spec / "history.json") if (spec / "history.json").exists() else None
    result = read(root / "comparison.json") if resume and (root / "comparison.json").is_file() else {}
    order = ("without_memory", "with_memory") if index % 2 == 0 else ("with_memory", "without_memory")
    for slot, condition in enumerate(order, 1):
        if not unchanged(receipt, spec, baseline):
            raise ValueError("Frozen inputs changed before evaluation")
        if condition in result:
            if result[condition].get("result") not in {"passed", "failed", "uncertain"}:
                raise ValueError("Saved evaluation arm has no terminal result")
            continue
        trial = root / ("trial-%d" % slot)
        saved_solver = resume and trial.exists()
        solved = read(trial / "result.json") if saved_solver and (trial / "result.json").is_file() else {}
        if saved_solver and not agent_finished(solved):
            # A started arm spent its original budget. Do not reset it or
            # overwrite its candidate merely because its score was not saved.
            result[condition] = {"result": "uncertain", "execution_status": "interrupted",
                "solver_status": solved.get("status", "interrupted"),
                "judge_status": "unavailable", "metrics": solved.get("metrics", {}),
                "history_available": history is not None, "trial": trial.name,
                "detail": "Started trial retained without a complete acceptance score"}
            save(root / "comparison.json", result)
            save(root / "paired-differences.json", compare_trials(result))
            continue
        if not saved_solver:
            prepare(trial, baseline)
            oracle = history["oracle_answer"] if history else injected_answer(item, spec)
            message = solver_input(task, oracle if condition == "with_memory" else None)
            if history:
                message += prompts.HISTORY_REQUEST
            print(root.name, "evaluation", condition, flush=True)
            def record_round(number, outcome):
                round_root = trial / ("round-%02d" % number)
                candidate = trial / "workspace/candidate"
                # The candidate snapshot and the solver receipt are the
                # authoritative record for this round.  Persist them before
                # running any host-side checks or Judge so a failed scorer
                # cannot erase the solver's completed round.
                copy_tree(candidate, round_root / "candidate")
                save(round_root / "result.json", {
                    "result": "uncertain", "solver_status": outcome.get("status"),
                    "metrics": outcome.get("metrics", {}), "judge_status": "pending"})
                try:
                    checks = run_checks(
                        candidate, spec, round_root / "checks", config["execution_image"],
                        candidate_pythonpath=config.get("code", {}).get("candidate_pythonpath"))
                    save(round_root / "checks/result.json", checks)
                except Exception as error:
                    checks = {"status": "error", "cases": [],
                              "error_type": type(error).__name__, "detail": str(error)}
                    save(round_root / "checks/result.json", checks)
                judged = {"status": "not_run"}
                review_path, roots = None, {
                    "/workspace/candidate": round_root / "candidate",
                    "/workspace/checks": round_root / "checks",
                    "/workspace/experiments": round_root / "experiments",
                }
                if checks.get("status") != "error":
                    try:
                        judged, review_path, roots = inspect_acceptance(
                            candidate, spec, items, checks, round_root, config, agent_options)
                        if not isinstance(judged, dict) or not isinstance(judged.get("status"), str):
                            raise ValueError("invalid_judge_result")
                    except Exception as error:
                        # A Judge failure is an evaluation failure.  Keep the
                        # solver receipt and candidate status independent.
                        judged = {"status": "error", "error_type": type(error).__name__,
                                  "detail": str(error)}
                save(round_root / "judge/result.json", judged)
                acceptance = assess_acceptance(items, checks, review_path, roots)
                changed = write_diff(baseline, candidate, round_root / "changes.patch")
                pollution = implementation_pollution(changed)
                status = "uncertain" if pollution and acceptance["status"] == "passed" else acceptance["status"]
                save(round_root / "result.json", dict(result=status, solver_status=outcome["status"],
                     metrics=outcome["metrics"], acceptance=acceptance, checks=checks,
                     judge_status=judged["status"], changed_files=changed,
                     implementation_pollution=pollution))
                return {"continue": status != "passed"}
            solved = run_agent(trial, config, "code", message, **agent_options,
                               max_rounds=2, on_round=record_round,
                               **({"history": history} if history else {}))
        candidate = trial / "workspace/candidate"
        completed_rounds = sorted(trial.glob("round-*/result.json"))
        final_round_root = completed_rounds[-1].parent if completed_rounds else None
        changed = (read(trial / "version.json")["changed_files"]
                   if saved_solver and (trial / "version.json").is_file()
                   else write_diff(baseline, candidate, trial / "changes.patch"))
        if final_round_root is not None:
            checks = read(final_round_root / "checks/result.json")
        elif saved_solver and (trial / "checks/result.json").is_file():
            checks = read(trial / "checks/result.json")
        else:
            check_root = trial / "checks"
            while check_root.exists():
                check_root = check_root.with_name(check_root.name + "-resume")
            checks = run_checks(candidate, spec, check_root, config["execution_image"],
                                candidate_pythonpath=config.get("code", {}).get("candidate_pythonpath"))
        inspection = (final_round_root if final_round_root is not None else
                      trial / "resume-inspection" if (trial / "resume-inspection").exists() else trial)
        judge = inspection / "judge"
        if (saved_solver or final_round_root is not None) and judge.exists():
            judged = read(judge / "result.json") if (judge / "result.json").is_file() else {"status": "interrupted"}
            review_path = judge / "workspace/checks/acceptance-review.txt"
            roots = {"/workspace/" + name: judge / "workspace" / name
                     for name in ("candidate", "checks", "experiments")}
        else:
            if saved_solver and (trial / "judge-reference").exists():
                inspection = trial / "resume-inspection"
            try:
                judged, review_path, roots = inspect_acceptance(
                    candidate, spec, items, checks, inspection, config, agent_options)
                if not isinstance(judged, dict) or not isinstance(judged.get("status"), str):
                    raise ValueError("invalid_judge_result")
            except Exception as error:
                judged = {"status": "error", "error_type": type(error).__name__,
                          "detail": str(error)}
                review_path, roots = None, {
                    "/workspace/candidate": inspection / "candidate",
                    "/workspace/checks": inspection / "checks",
                    "/workspace/experiments": inspection / "experiments",
                }
                save(inspection / "judge/result.json", judged)
            judge = inspection / "judge"
        verdict_path = judge / "workspace/checks/verdict.txt"
        verdict = verdict_path.read_text() if verdict_path.exists() else ""
        acceptance = assess_acceptance(items, checks, review_path, roots)
        status = acceptance["status"]
        polluted = implementation_pollution(changed)
        if polluted and status == "passed":
            # Keep the functional result visible, but do not count a delivery
            # that changed its frozen tests/fixtures as an ordinary pass.
            status = "uncertain"
        information_condition = (
            "memory_not_required" if condition == "without_memory" and status == "passed"
            else "oracle_history" if condition == "with_memory" and history else "without_memory"
        )
        result[condition] = {"result": status, "solver_status": solved["status"],
                             "judge_status": judged["status"],
                             "metrics": solved["metrics"], "checks": checks,
                             "history_available": history is not None,
                             "information_condition": information_condition,
                             "acceptance": acceptance, "changed_files": changed,
                             "implementation_pollution": polluted,
                             "judge_evidence": verdict, "trial": trial.name}
        rounds = [read(path) for path in sorted(trial.glob("round-*/result.json"))]
        if rounds:
            result[condition].update(rounds=rounds, first_round=rounds[0], final_round=rounds[-1])
        counterexample = judge / "workspace/checks/counterexample.md"
        if counterexample.is_file():
            result[condition]["counterexample_pending_shared_review"] = counterexample.read_text()
        if history:
            application = read_history_review(acceptance, history)
            exchanges = solved.get("clarifications", [])
            result[condition].update(history_application=application,
                history_question_count=len(exchanges),
                information_condition=information_condition,
                clarifications=exchanges, responder_cost=solved.get("responder_cost"),
                clarification_status=solved.get("clarification_status", "unavailable"),
                interaction_counts={kind: sum(e.get("kind") == kind for e in exchanges)
                    for kind in ("historical_reask", "same_session_repeat", "update_confirmation")})
        if not unchanged(receipt, spec, baseline):
            raise ValueError("Frozen inputs changed during evaluation")
        save(root / "comparison.json", result)
        save(root / "paired-differences.json", compare_trials(result))
    return result


def selected_input_hash(item):
    payload = dict(item)
    if "generation_input" in payload:
        payload["generation_input"] = generation_request(item)
    if payload.get("qa_members"):
        payload["qa_members"] = [dict(member, generation_input=generation_request(member))
                                 for member in payload["qa_members"]]
    return qa_fingerprint(payload)


def validate_resume(manifest, expected, selected, output, baseline):
    """Check experiment identity before changing any retained artifact."""
    for key in ("source_run", "qa_run", "baseline", "baseline_sha256", "baseline_version",
                "config", "execution", "target", "task_budget", "selection_only",
                "selected_qa_ids", "qa_group_size", "evaluator_version", "simulator_version"):
        # Legacy manifests predate the two source-version receipts.  They may
        # still be inspected, but a current resume must not silently run the
        # old artifact under a new evaluator.
        if key in expected and key in manifest and manifest.get(key) != expected[key]:
            raise ValueError("Resume inputs changed: " + key)
        if key in expected and key in {"evaluator_version", "simulator_version"}:
            if key not in manifest:
                raise ValueError("Resume manifest missing: " + key)
    if "allow_provisional" in manifest and manifest.get("allow_provisional") != expected.get(
            "allow_provisional", False):
        raise ValueError("Resume provisional QA policy changed")
    hashes = expected["selected_inputs_sha256"]
    if "selected_inputs_sha256" in manifest and manifest["selected_inputs_sha256"] != hashes:
        raise ValueError("Resume QA or generation inputs changed")
    if "config_sha256" in manifest and manifest["config_sha256"] != expected["config_sha256"]:
        raise ValueError("Resume configuration hash changed")
    items = {item["qa"]["id"]: item for item in selected}
    identities = {}
    for row in manifest["tasks"]:
        name, identity = row["task"], row.get("qa_id")
        if identity is None and row.get("status") == "error":
            saved_qa = output / name / "author-reference/qa.json"
            if saved_qa.is_file():
                identity = read(saved_qa).get("id")
        if identity not in items:
            raise ValueError("Resume task selection changed: " + name)
        identities[name] = identity
        item = items[identity]
        root = output / name
        for saved, current in ((root / "author-reference/qa.json", item["qa"]),
                               (root / "author-reference/qa-input.json",
                                generation_request(item) if "generation_input" in item else None)):
            if saved.is_file() and read(saved) != current:
                raise ValueError("Resume saved QA or generation input changed: " + name)
        if (root / "frozen.json").is_file():
            receipt = read(root / "frozen.json")
            if (receipt.get("qa_id") != item["qa"]["id"]
                    or ("qa_sha256" in receipt and receipt["qa_sha256"] != qa_fingerprint(item["qa"]))
                    or not unchanged(receipt, root / "frozen", baseline)):
                raise ValueError("Resume frozen inputs changed: " + name)
    return identities


def recover_orphan_tasks(manifest, selected, output):
    """Register task directories written before their manifest row was saved."""
    output = Path(output)
    known = {row.get("task") for row in manifest.get("tasks", [])}
    known_qa = {row.get("qa_id") for row in manifest.get("tasks", [])}
    selected_by_id = {item["qa"]["id"]: item for item in selected}
    for root in sorted(output.glob("task-*")):
        if root.name in known:
            continue
        saved = root / "author-reference/qa.json"
        if not saved.is_file():
            continue
        try:
            qa_id = read(saved).get("id")
        except (OSError, ValueError, TypeError):
            continue
        if qa_id not in selected_by_id or qa_id in known_qa:
            continue
        item = selected_by_id[qa_id]
        manifest.setdefault("tasks", []).append({
            "task": root.name,
            "status": "pending",
            "qa_id": qa_id,
            "type": item["qa"].get("type"),
            "recovered_orphan": True,
        })
        known.add(root.name)
        known_qa.add(qa_id)


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
    parser.add_argument("--resume", action="store_true",
                        help="Continue an existing task output, preserving completed task records")
    parser.add_argument("--allow-provisional", action="store_true",
                        help="Use QA items marked needs_review as provisional task seeds")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--qa-group-size", type=int,
                        help="Maximum related QA in each optional pool (default: 4 external, 1 graph)")
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
    qa_source = getattr(args, "qa_source", "graph")
    qa_group_size = getattr(args, "qa_group_size", None)
    if qa_group_size is not None and qa_group_size < 1:
        parser.error("QA group size must be positive")
    if args.preparation_feedback and not args.reuse_preparation:
        parser.error("--preparation-feedback requires --reuse-preparation")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    existing_manifest = (read(output / "manifest.json")
                         if args.resume and (output / "manifest.json").is_file() else None)
    if args.resume and existing_manifest is None:
        parser.error("--resume requires an existing task manifest")
    if not args.resume and ((output / "manifest.json").exists() or (output / "baseline").exists()):
        parser.error("Use a new output directory; previous experiments are retained")
    input_diagnostics = []
    items = qa_inputs(args.qa_run, include_provisional=args.allow_provisional,
                      diagnostics=input_diagnostics)
    qa_source = (items[0].get("qa_source") if items else
                 read(args.qa_run / "manifest.json").get("qa_source", "graph")
                 if (args.qa_run / "manifest.json").is_file() else "graph")
    group_size = qa_group_size if qa_group_size is not None else (
        4 if qa_source == "external" else 1)
    grouping_diagnostics = []
    usable_count = len(items)
    items = group_qa_inputs(items, group_size, diagnostics=grouping_diagnostics)
    if not items and existing_manifest and existing_manifest.get("tasks"):
        raise ValueError("Resume task selection changed: no usable QA groups")
    save(output / "qa-grouping.json", {"group_size": group_size,
         "groups": [{"qa_id": item["qa"]["id"], "qa_ids": item.get("qa_ids", [item["qa"]["id"]]),
                     "source_ids": item.get("source_ids", []),
                     # Lineage is a control-side receipt.  It lets the report
                     # explain why questions were grouped without entering
                     # the natural-language task input.
                     "external_lineage": item.get("external_lineage", {}),
                     "provisional": item.get("provisional", False)}
                    for item in items],
         "rejected_inputs": input_diagnostics, "ungrouped": grouping_diagnostics})
    if not items:
        stop_reason = "no_related_external_qa" if usable_count else "no_eligible_qa"
        manifest = {"source_run": str(args.source_run.resolve()), "qa_run": str(args.qa_run.resolve()),
                    "target": args.count, "task_budget": task_budget, "selection_only": args.selection_only,
                    "tasks": [], "completed": 0, "accepted": 0, "shortfall": args.count,
                    "status": "incomplete",
                    "stop_reason": stop_reason,
                    "notes": ["%s. Completed tasks: 0; target: %d; shortfall: %d."
                              % (stop_reason, args.count, args.count)],
                    "qa_group_size": group_size, "selected_qa_ids": [],
                    "qa_input_diagnostics": input_diagnostics,
                    "qa_grouping_diagnostics": grouping_diagnostics}
        save(output / "manifest.json", manifest)
        write_report(output, manifest)
        return 0
    if args.reuse_preparation:
        source_qa = read(args.reuse_preparation.parent / "author-reference/qa.json")
        items = [item for item in items if item["qa"] == source_qa]
        if len(items) != 1 or args.count != 1:
            parser.error("Reusing preparation requires its exact QA and --count 1")
    config = configure(args.simulator_path, args.source_run / "private/checkpoint.json", args.env_file,
                       **({"control_config": args.control_config} if args.control_config else {}))
    # Host-side selection and review calls use the judge transport directly.
    # Keep a stalled provider response from holding the whole task batch past
    # the stage budget; code-agent workers retain their separate execution
    # deadline and cumulative request/token budgets.
    config = bounded_model_config(config, 600)
    preflight_openhands_runtime(args.simulator_path)
    config["model_request_chars"] = args.model_request_chars
    baseline = (Path(existing_manifest["baseline"]).resolve() if args.resume
                else args.baseline.resolve() if args.baseline else output / "baseline")
    if args.resume:
        version = baseline_version(baseline)
    elif args.baseline:
        version = baseline_version(baseline)
    else:
        copy_tree(args.source_run / "workspace/candidate", baseline)
        version = pin_baseline(baseline)
    if not args.resume:
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
    config_sha256 = qa_fingerprint(config)
    selected_inputs_sha256 = [selected_input_hash(item) for item in selected]
    expected_manifest = {"source_run": str(args.source_run.resolve()), "qa_run": str(args.qa_run.resolve()),
                "baseline": str(baseline),
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
                "selected_qa_ids": [i["qa"]["id"] for i in selected],
                "config_sha256": config_sha256, "selected_inputs_sha256": selected_inputs_sha256,
                "allow_provisional": args.allow_provisional}
    expected_manifest.update(qa_group_size=group_size,
         selected_qa_groups=[item.get("qa_ids", [item["qa"]["id"]]) for item in selected],
         qa_input_diagnostics=input_diagnostics, qa_grouping_diagnostics=grouping_diagnostics)
    if args.resume:
        manifest = existing_manifest
        recover_orphan_tasks(manifest, selected, output)
        identities = validate_resume(manifest, expected_manifest, selected, output, baseline)
        for row in manifest["tasks"]:
            if row.get("qa_id") is None:
                row["qa_id"] = identities[row["task"]]
        manifest.setdefault("resume_history", []).append({"source": "manifest", "config_sha256": config_sha256})
    else:
        manifest = dict(expected_manifest, tasks=[])
        save(output / "manifest.json", manifest)
    agent_options = {"max_requests": args.agent_requests, "max_tokens": args.agent_tokens}

    # A construction directory is an append-only attempt.  Keep its record
    # and give an unfinished construction a fresh task directory on resume;
    # frozen tasks can still resume their missing scored arm in place.
    root_by_index = {index: "task-%02d" % (index + 1) for index in range(len(selected))}
    if args.resume:
        used = [int(path.name.split("-")[-1]) for path in output.glob("task-*")
                if path.name.split("-")[-1].isdigit()]
        used.extend(int(row["task"].split("-")[-1]) for row in manifest["tasks"]
                    if row.get("task", "").startswith("task-") and row["task"].split("-")[-1].isdigit())
        next_number = max([len(selected), *used]) + 1
        for index, item in enumerate(selected):
            rows = [row for row in manifest["tasks"] if row.get("qa_id") == item["qa"]["id"]]
            prior = rows[-1] if rows else None
            if prior is None and (output / root_by_index[index]).exists():
                replacement = "task-%02d" % next_number
                next_number += 1
                root_by_index[index] = replacement
                manifest["tasks"].append({"task": replacement, "status": "pending",
                                           "qa_id": item["qa"]["id"], "type": item["qa"]["type"]})
                continue
            if prior and prior.get("status") in {"evaluated", "qualified", "not_admitted"}:
                root_by_index[index] = prior["task"]
            elif prior and (output / prior["task"] / "frozen.json").is_file():
                root_by_index[index] = prior["task"]
            elif prior:
                replacement = "task-%02d" % next_number
                next_number += 1
                prior["status"] = "interrupted"
                prior["resume_replaced_by"] = replacement
                root_by_index[index] = replacement
                manifest["tasks"].append({"task": replacement, "status": "pending",
                                           "qa_id": item["qa"]["id"], "type": item["qa"]["type"]})
        save(output / "manifest.json", manifest)

    def run_task(index, item):
        root = output / root_by_index[index]
        prior = next((row for row in manifest["tasks"] if row.get("task") == root.name), None)
        if args.resume and prior and prior.get("status") in {"evaluated", "qualified", "not_admitted"}:
            complete = (prior.get("status") in {"qualified", "not_admitted"} or
                        set(prior.get("comparison", {})) == {"without_memory", "with_memory"})
            if complete:
                return prior
            prior = dict(prior, status="interrupted", interruption_reason="incomplete_terminal_record")
        if args.resume and prior and (root / "frozen.json").is_file() and not args.selection_only:
            try:
                receipt = read(root / "frozen.json")
                comparison = evaluate(item, root, baseline, receipt, config, agent_options, index, resume=True)
                interrupted = any(trial.get("execution_status") == "interrupted"
                                  for trial in comparison.values())
                return {"task": root.name, "status": "interrupted" if interrupted else "evaluated",
                        "qa_id": item["qa"]["id"], "type": item["qa"]["type"],
                        "comparison": comparison, "paired_differences": compare_trials(comparison)}
            except Exception as error:
                return {**prior, "status": "interrupted", "error_type": type(error).__name__, "detail": str(error)}
        if args.resume and prior and prior.get("status") in {"running", "error", "interrupted", "not_admitted"}:
            record = dict(prior, status="interrupted", interruption_reason="construction_not_frozen")
            save(root / "interrupted.json", record)
            return record
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
            failure = {"task": root.name, "qa_id": item["qa"]["id"], "type": item["qa"]["type"],
                       "status": "error", "error_type": type(error).__name__,
                       "detail": str(error)}
            save(root / "failure.json", failure)
            return failure

    def run(index, item):
        root = output / root_by_index[index]
        row = {"status": "interrupted"}
        try:
            with shared_task_slot():
                row = run_task(index, item)
            row.update(qa_ids=item.get("qa_ids", [item["qa"]["id"]]),
                       qa_statuses={member["qa"]["id"]: member["qa"].get("status")
                                    for member in item.get("qa_members", [item])},
                       provisional=item.get("provisional", False))
            return row
        except BaseException as error:
            row.update(error_type=type(error).__name__, detail=str(error))
            raise
        finally:
            save_task_progress(root, "terminal", status=row["status"],
                               finished_at=round(time.time(), 3),
                               **{key: row[key] for key in
                                  ("reason", "interruption_reason", "error_type", "detail")
                                  if key in row})

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        next_index = 0
        futures = set()
        while next_index < len(selected) or futures:
            completed = sum(task["status"] in (("qualified", "evaluated") if args.selection_only else ("evaluated",))
                            for task in manifest["tasks"])
            while (next_index < len(selected) and len(futures) < args.workers
                   and completed + len(futures) < args.count):
                futures.add(pool.submit(run, next_index, selected[next_index]))
                next_index += 1
            if not futures:
                break
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                row = future.result()
                manifest["tasks"] = [existing for existing in manifest["tasks"]
                                     if existing.get("task") != row.get("task")]
                manifest["tasks"].append(row)
                manifest["tasks"].sort(key=lambda item: item["task"])
                save(output / "manifest.json", manifest)
                write_report(output, manifest)
    completed = sum(task["status"] == ("qualified" if args.selection_only else "evaluated") for task in manifest["tasks"])
    manifest.update(completed=completed, shortfall=max(0, args.count - completed),
                    stop_reason="target_met" if completed >= args.count else
                    "task_budget_exhausted" if len(selected) >= task_budget else "qa_pool_exhausted")
    accepted = sum(task.get("status") == ("qualified" if args.selection_only else "evaluated")
                   and not task.get("provisional") for task in manifest["tasks"])
    manifest.update(accepted=accepted, provisional_completed=completed - accepted,
                    status="complete" if accepted >= args.count else "needs_review" if completed else "incomplete")
    save(output / "manifest.json", manifest)
    write_report(output, manifest)
    print("Task pilot complete:", output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
