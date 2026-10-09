"""A compact view of saved task outcomes and observable agent costs."""

from pathlib import Path
from html import escape
import json

from .metrics import compare_trials


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_report(output, manifest):
    output = Path(output)
    tasks = manifest.get("tasks", [])
    lines = ["# Repository task comparison", "",
             "The memory condition receives applicable saved public answers and global agreements as historical information.",
             "", "| Task | Condition | Result | History questions | Development tools | File views | Reads/searches | Solver tokens | Cache hit / rate |",
             "|---|---|---|---|---|---|---|---|---|"]
    if manifest.get("selection_only"):
        lines = ["# Controlled task selection", "", "No OpenHands agents or paired trials were started.", "",
                 "| Task | QA | Outcome | Queries | Requests | Tokens | Cache hit / rate |",
                 "|---|---|---|---|---|---|---|"]
        for task in tasks:
            root = output / task["task"]
            def saved(name):
                path = root / name
                return json.loads(path.read_text()) if path.exists() else {}
            budget, selection = saved("selection-budget.json"), saved("selection/result.json")
            cache = (str(budget.get("cache_hit_tokens")) + " / " +
                     ("%.1f%%" % (100 * budget["cache_hit_rate"])
                      if budget.get("cache_usage_complete") and budget.get("cache_hit_rate") is not None
                      else "unavailable"))
            lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
                task["task"], task.get("qa_id", ""), task["status"], selection.get("query_count", "—"),
                budget.get("requests", "—"), budget.get("total_tokens", "—"), cache))
            lines += ["", "- %s: %s" % (task["task"], selection.get("reason", "Not saved")),
                      "- [Saved decisions and queries](%s/selection/result.json)" % task["task"], ""]
        (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        write_html(output, manifest)
        return
    for task in tasks:
        for condition, trial in task.get("comparison", {}).items():
            m = trial.get("metrics", {})
            cache = (str(m.get("cache_hit_tokens")) + " / " +
                     ("%.1f%%" % (100 * m["cache_hit_rate"]) if m.get("cache_usage_complete") else "unavailable"))
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                task["task"], trial.get("information_condition", condition), trial["result"],
                ("unavailable" if trial.get("history_available") is False else
                 trial.get("history_question_count", "not saved")), m.get("tool_calls", "not saved"),
                m.get("file_view_calls", "not saved"), m.get("shell_read_or_search_calls", "not saved"),
                m.get("total_tokens", "not saved"), cache))
    lines += ["", "## First and final rounds", "",
              "Both conditions use at most two solver rounds with one shared budget. Costs are cumulative and include failures.",
              "", "| Condition | Checkpoint | Trials | Passed / failed / uncertain | Solver requests | Solver tokens |",
              "|---|---|---|---|---|---|"]
    for condition in ("without_memory", "with_memory"):
        for checkpoint in ("first_round", "final_round"):
            rows = [task["comparison"][condition][checkpoint] for task in tasks
                    if checkpoint in task.get("comparison", {}).get(condition, {})]
            def round_total(key):
                values = [row.get("metrics", {}).get(key) for row in rows]
                return sum(values) if rows and all(isinstance(value, (int, float)) for value in values) else "not saved"
            lines.append("| %s | %s | %s | %s | %s | %s |" % (
                condition, checkpoint, len(rows),
                "/".join(str(sum(row["result"] == value for row in rows))
                         for value in ("passed", "failed", "uncertain")),
                round_total("attempted_requests"), round_total("total_tokens")))
    lines += ["", "## Agent execution", "",
              "Results above describe acceptance of the saved code. An interrupted agent may still leave "
              "code that can be checked; its execution status is recorded separately.", "",
              "| Task | Condition | Solver status | Judge status | Solver receipt |",
              "|---|---|---|---|---|"]
    for task in tasks:
        for condition, trial in task.get("comparison", {}).items():
            receipt = output / task["task"] / trial.get("trial", "") / "result.json"
            lines.append("| %s | %s | %s | %s | %s |" % (
                _cell(task["task"]), _cell(condition), _cell(trial.get("solver_status", "not saved")),
                _cell(trial.get("judge_status", "not saved")),
                "[Execution result](%s)" % receipt.relative_to(output) if receipt.is_file() else "not saved"))
    lines += ["", "## Aggregate execution costs", "",
              "| Condition | Trials | Passed / failed / uncertain | Pass rate | History questions | Clarifications | Development tools | File views | Reads/searches | Solver prompt | Solver completion | Solver total | Responder tokens | Injection tokens | Cache hit / rate |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    rates = {}
    for condition in ("without_memory", "with_memory"):
        trials = [t["comparison"][condition] for t in tasks if condition in t.get("comparison", {})]
        def total(key):
            values = [t.get("metrics", {}).get(key) for t in trials]
            return sum(values) if trials and all(isinstance(v, (int, float)) for v in values) else "not saved"
        passed = sum(t["result"] == "passed" for t in trials)
        rates[condition] = passed / len(trials) if trials else None
        questions = [t.get("history_question_count") for t in trials]
        clarification_counts = [t.get("clarification_count", len(t.get("clarifications", []))) for t in trials]
        responder = [t.get("responder_cost") for t in trials]
        cache_complete = trials and all(t.get("metrics", {}).get("cache_usage_complete") for t in trials)
        cache_total = total("cache_hit_tokens") if cache_complete else "unavailable"
        cache_rate = (sum(t["metrics"]["cache_hit_tokens"] for t in trials) /
                      sum(t["metrics"]["cache_observed_prompt_tokens"] for t in trials)
                      if cache_complete and sum(t["metrics"]["cache_observed_prompt_tokens"] for t in trials) else None)
        injection = total("injection_tokens")
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            condition, len(trials),
            "/".join(str(sum(t["result"] == v for t in trials)) for v in ("passed", "failed", "uncertain")),
            "%.1f%%" % (100 * rates[condition]) if trials else "—",
            (sum(questions) if trials and all(isinstance(v, int) for v in questions) else
             "unavailable" if trials and all(t.get("history_available") is False for t in trials)
             else "not saved"),
            sum(clarification_counts) if trials and all(isinstance(v, int) for v in clarification_counts) else "not saved",
            total("tool_calls"), total("file_view_calls"), total("shell_read_or_search_calls"),
            total("prompt_tokens"), total("completion_tokens"),
            str(total("solver_tokens")) + ("" if all(t.get("metrics", {}).get("usage_complete") for t in trials) else " (incomplete)"),
            sum(r["tokens"] for r in responder) if trials and all(r and r.get("usage_complete") for r in responder) else "not saved",
            injection,
            str(cache_total) + (" / %.1f%%" % (100 * cache_rate) if cache_rate is not None else " / unavailable")))
    if all(v is not None for v in rates.values()) and all(
            compare_trials(task["comparison"])["completion_difference"] is not None
            for task in tasks if task.get("comparison")):
        lines += ["", "Pass-rate difference (with − without): %.1f percentage points." %
                  (100 * (rates["with_memory"] - rates["without_memory"]))]
    lines += ["", "## Paired differences", "",
              "Differences are with memory minus without memory. Raw costs are retained even when one arm "
              "fails; comparable costs are shown only when both arms pass. Token differences cover the "
              "solver; responder tokens are listed separately above.",
              "", "| Task | Pair class | Both passed | Efficiency comparable | History questions | Raw tools | Raw views | Raw reads | Raw tokens | Comparable tools | Comparable views | Comparable reads | Comparable tokens |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for task in tasks:
        pair = compare_trials(task.get("comparison", {}))
        raw, delta = pair["raw_cost_differences"], pair["comparable_cost_differences"]
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            task["task"], pair.get("pair_class", "not saved"), pair["both_passed"],
            pair.get("efficiency_comparable", False), pair["history_question_difference"],
            *(raw[k] for k in ("tool_calls", "file_view_calls", "shell_read_or_search_calls", "total_tokens")),
            *(delta[k] for k in ("tool_calls", "file_view_calls", "shell_read_or_search_calls", "total_tokens"))))
    successful = [compare_trials(t.get("comparison", {})) for t in tasks
                  if compare_trials(t.get("comparison", {}))["both_passed"]]
    lines += ["", "Both-passed pairs: %d." % len(successful)]
    for metric in ("tool_calls", "file_view_calls", "shell_read_or_search_calls", "total_tokens"):
        values = [p["cost_differences"][metric] for p in successful]
        delta = sum(values) if values and all(v is not None for v in values) else "not available"
        lines.append("- %s: total paired difference %s" % (metric, delta))
    lines += ["", "## No-memory, no-followup diagnostic", "",
              "This diagnostic is separate from the main pair and does not change pass rates.",
              "", "| Task | Result | Solver status | Judge status | Tokens |",
              "|---|---|---|---|---|"]
    for task in tasks:
        diagnostic = output / task["task"] / "diagnostic-no-memory-no-followup/result.json"
        if not diagnostic.is_file():
            continue
        row = json.loads(diagnostic.read_text())
        lines.append("| %s | %s | %s | %s | %s |" % (
            task["task"], row.get("result", "not saved"), row.get("solver_status", "not saved"),
            row.get("judge_status", "not saved"),
            row.get("metrics", {}).get("total_tokens", "not saved")))
    lines += ["", "## Frozen acceptance and historical rules", ""]
    for task in tasks:
        for condition, trial in task.get("comparison", {}).items():
            lines += ["", "### %s / %s" % (task["task"], condition), "",
                      "| Requirement | Basis | Result | Evidence |", "|---|---|---|---|"]
            for row in trial.get("acceptance", {}).get("rows", []):
                lines.append("| %s | %s | %s | %s |" % tuple(_cell(v) for v in (
                    row["requirement"], ", ".join(row["basis"]), row["status"], row["evidence"])))
            for row in trial.get("history_application", {}).get("rows", []):
                lines.append("- %s: %s — %s" % (row["id"], row["status"], _cell(row["evidence"])))
            if not trial.get("acceptance"):
                lines.append("Per-item acceptance was not saved in this older run.")
            if trial.get("counterexample_pending_shared_review"):
                lines += ["", "Counterexample pending shared verification:",
                          _cell(trial["counterexample_pending_shared_review"])]
    lines += ["", "Tool counts exclude think and finish; failed development calls count. "
              "Reads/searches are call details, not an additional score. Old traces remain unchanged.",
              "", "Every trial, including failures and uncertain results, is listed. No route score is computed."]
    lines += manifest.get("notes", [])
    if tasks:
        lines += ["", "## QA to requirement", "",
                  "| Task | Source QA | Type | Outcome |", "|---|---|---|---|"]
        for task in tasks:
            lines.append("| %s | %s | %s | %s |" % tuple(_cell(v) for v in (
                task["task"], task.get("qa_id", "not saved"), task.get("type", "not saved"), task["status"])))
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_html(output, manifest)


def write_html(output, manifest):
    """Render saved model artifacts and trial receipts without adding judgments."""
    from ..cli import _redact_public_text
    from ..security import credential_detected

    def safe(value):
        text = str(value)
        if credential_detected(text):
            return "[Content hidden: possible credential]"
        return escape(_redact_public_text(text)[0])

    def text_file(path):
        return path.read_text(encoding="utf-8") if path.exists() else "Not saved"

    def link(path, label):
        if not path.exists() and path.with_suffix(path.suffix + ".gz").exists():
            path = path.with_suffix(path.suffix + ".gz")
        if not path.exists():
            return ""
        return '<a href="%s">%s</a>' % (escape(str(path.relative_to(output)), quote=True), escape(label))

    def trace(root):
        path = root / "trace.jsonl.gz"
        if not path.exists():
            return ""
        return '<p>%s · %s</p>' % (
            link(path, "Original model and tool trace (compressed)"),
            link(root / "trajectory.json", "Tool trajectory"))

    cards = []
    for item in manifest.get("tasks", []):
        root = output / item["task"]
        qa_path = root / "author-reference/qa.json"
        qa = json.loads(qa_path.read_text()) if qa_path.exists() else {}
        attempts = sorted(root.glob("construction-*/author/workspace/checks/task.md"))
        task_path = root / "frozen/task.md"
        if not task_path.exists() and attempts:
            task_path = attempts[-1]
        if not task_path.exists() and (root / "rejected-task.md").exists():
            task_path = root / "rejected-task.md"
        trials = []
        for condition, trial in item.get("comparison", {}).items():
            trial_root = root / trial["trial"]
            metrics = trial.get("metrics", {})
            trials.append('<h3>%s · %s</h3><p>%s</p><pre>%s</pre>' % (
                safe(condition), safe(trial.get("result")), " · ".join(filter(None, [
                    link(trial_root / "workspace/candidate", "Full code"),
                    link(trial_root / "changes.patch", "Patch"),
                    link(trial_root / "version.json", "Replay verification"),
                    link(trial_root / "trajectory.json", "Tool trajectory"),
                    link(trial_root / "checks/result.json", "Tests")])),
                safe(json.dumps({"solver_status": trial.get("solver_status"),
                                 "judge_status": trial.get("judge_status"),
                                 "metrics": metrics, "acceptance": trial.get("acceptance"),
                                 "history_application": trial.get("history_application"),
                                 "interactions": trial.get("interaction_counts"),
                                 "clarifications": trial.get("clarifications"),
                                 "responder_cost": trial.get("responder_cost"),
                                 "tests": trial.get("checks"), "judge": trial.get("judge_evidence")},
                                ensure_ascii=False, indent=2))))
            trials.append(trace(trial_root))
        frozen_path = root / "frozen.json"
        reference = ""
        accepted = json.loads(frozen_path.read_text()).get("accepted_attempt") if frozen_path.exists() else None
        if accepted is not None:
            ref = root / ("construction-%02d/reference-solver" % accepted)
            reference = '<h3>Reference implementation</h3><p>%s</p>%s' % (
                " · ".join(filter(None, [link(ref / "workspace/candidate", "Full reference code"),
                    link(ref / "changes.patch", "Patch"), link(ref / "version.json", "Replay verification")])),
                trace(ref))
        summary_path = root / "construction-summary.json"
        if not summary_path.exists():
            summary_path = root / "construction.json"
        cards.append('<article><h2>%s · %s · %s</h2><p>%s</p><details><summary>Source QA and actual generation input</summary>'
                     '<pre>%s</pre>%s</details><h3>Model-authored requirement</h3><pre>%s</pre>'
                     '<details><summary>Construction attempts and reasons</summary><pre>%s</pre></details>%s</article>' % (
            safe(item["task"]), safe(item.get("type", qa.get("type", "Type not saved"))), safe(item["status"]), " · ".join(filter(None, [
                link(root / "frozen", "Frozen criteria and tests"),
                link(summary_path, "Requirements and attempt reasons"),
                link(root / "comparison.json", "Comparison"),
                link(root, "All task files")])),
            safe(json.dumps(qa, ensure_ascii=False, indent=2)),
            link(root / "author-reference/qa-input.json", "Original QA generation input"),
            safe(text_file(task_path)), safe(text_file(summary_path)), reference + "".join(trials)))
    context = ("Controlled simulation: customer rules are simulated; dialogue and agent actions were actually executed."
               if manifest.get("evaluation_type") == "simulation_only" else "")
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Repository task results</title>
<style>body{font:16px/1.6 system-ui;background:#f4f6fa;color:#182336;margin:32px auto;max-width:1150px;padding:0 24px}
article{background:white;border:1px solid #dce3ed;border-radius:12px;padding:24px;margin:22px 0}a{color:#2157a5}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 ui-monospace,monospace;background:#f4f6fa;padding:16px}
summary{cursor:pointer;color:#2157a5}h1,h2{line-height:1.3}</style><h1>Repository task results</h1>
<p>Each requirement starts from the pinned dialogue-end code. Both trials use the same frozen criteria.
The memory trial receives only the historical answer as extra context.</p><p>%s</p>
<p><a href="report.md">Aggregate results, paired differences and per-item acceptance</a></p><p>%s</p>%s</html>''' % (
        safe(context),
        safe("Target: %s · Completed pairs: %s · Stop: %s" % (
            manifest.get("target", "—"), manifest.get("completed", sum(t["status"] == "evaluated" for t in manifest.get("tasks", []))),
            manifest.get("stop_reason", "running"))), "".join(cards))
    (output / "report.html").write_text(page, encoding="utf-8")
