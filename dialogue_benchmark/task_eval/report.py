"""A compact view of saved task outcomes and observable agent costs."""

from pathlib import Path
from html import escape
import json
import gzip

from .metrics import compare_checkpoints


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _coverage(checkpoints):
    counts = checkpoints.get("counts", {})
    if not counts:
        return "unavailable", "—"
    rate = checkpoints.get("action_coverage")
    value = "%s/%s (%s)" % (counts["observed"], checkpoints["total"],
                             "%.0f%%" % (rate * 100) if rate is not None else "unresolved")
    return value, "/".join(str(counts[key]) for key in ("alternative", "skipped", "uncertain"))


def _evidence(row):
    sources = ", ".join(source["action_id"] for source in row.get("sources", []))
    return _cell(row.get("evidence", "Not recorded") + ("; actions: " + sources if sources else ""))


def write_report(output, manifest):
    output = Path(output)
    tasks = manifest.get("tasks", [])
    lines = ["# Repository task pilot", "",
             "Model-authored requirements and tests, with frozen automated judging. "
             "The memory condition receives the historical answer directly."]
    for note in manifest.get("notes", []):
        lines += ["", note]
    lines += ["", "| Task | Condition | Judge result | Tests passed/total | Exploration observed/total | Alternative/skipped/uncertain | Tool calls | Provider tokens |",
              "|---|---|---|---|---|---|---|---|"]
    for task in tasks:
        for condition, trial in task.get("comparison", {}).items():
            metrics, checks = trial.get("metrics", {}), trial.get("checks", {})
            tokens = str(metrics.get("total_tokens", "unavailable"))
            if not metrics.get("usage_complete"):
                tokens += " (incomplete)"
            coverage, counts = _coverage(trial.get("checkpoints", {}))
            lines.append("| %s | %s | %s | %s/%s | %s | %s | %s | %s |" % (
                task["task"], condition, trial.get("result"),
                checks.get("passed", "?"), checks.get("tests", "?"),
                coverage, counts, metrics.get("tool_calls", "?"), tokens))
        if not task.get("comparison"):
            lines.append("| %s | — | %s | — | — | — | — | — |" % (task["task"], task["status"]))
    for task in tasks:
        comparison = task.get("comparison", {})
        if not comparison:
            continue
        delta = compare_checkpoints(comparison)
        lines += ["", "## %s: checkpoint comparison" % task["task"], ""]
        if delta["eligible"]:
            lines += ["Both tasks passed. Exploration coverage reduction: %.1f percentage points." % (
                100 * delta["coverage_reduction"]), "",
                "Observed without memory, skipped with memory: %s. "
                "Replaced by an alternative exploration: %s. "
                "Additional reference actions observed with memory: %s." % (
                    delta["skipped_in_memory"], delta["alternative_in_memory"], delta["additional_in_memory"])]
        else:
            lines.append("Efficiency comparison unavailable: %s." % delta["reason"])
        left = {row["index"]: row for row in comparison.get("without_memory", {}).get("checkpoints", {}).get("rows", [])}
        right = {row["index"]: row for row in comparison.get("with_memory", {}).get("checkpoints", {}).get("rows", [])}
        lines += ["", "| # | Exploration checkpoint | Without memory | With memory |",
                  "|---|---|---|---|"]
        for index in sorted(left.keys() | right.keys()):
            a, b = left.get(index, {}), right.get(index, {})
            lines.append("| %s | %s | %s | %s |" % (
                index, _cell(a.get("checkpoint", b.get("checkpoint", "Not recorded"))),
                a.get("status", "Not recorded"), b.get("status", "Not recorded")))
        lines += ["", "<details><summary>Solver trajectory evidence</summary>", "",
                  "| # | Without memory evidence | With memory evidence |", "|---|---|---|"]
        for index in sorted(left.keys() | right.keys()):
            lines.append("| %s | %s | %s |" % (index, _evidence(left.get(index, {})), _evidence(right.get(index, {}))))
        lines += ["", "</details>"]
    lines += ["", "In new task runs, checkpoints are exploration actions extracted from an accepted, independent "
              "no-memory reference trajectory and frozen before the paired trials. "
              "Implementation, final tests, and environment setup do not enter exploration coverage. "
              "Equivalent tools obtaining the same information count as observed. "
              "An uncertain match does not count as a skipped action. "
              "Efficiency comparisons require both tasks to pass and both checkpoint reviews to be resolved. "
              "Alternative routes and total tool/token costs remain visible alongside coverage.", "",
              "Full file-read counts are unavailable: explicit views and recognized shell "
              "read/search outputs are counted separately. Their token counts are estimates. "
              "Absent provider reasoning is recorded as unavailable, not zero. "
              "Elapsed time is not compared.", "",
              "Each task directory retains the generated requirement, frozen criteria, "
              "test receipts, trial changes, and judge evidence. Failed construction requirements "
              "and reasons remain in the construction summary."]
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
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        return '<details><summary>Original model and tool trace</summary>%s<pre>%s</pre></details>' % (
            link(path, "Download compressed trace"),
            safe(json.dumps(rows, ensure_ascii=False, indent=2)))

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
                safe(json.dumps({"metrics": metrics, "checkpoints": trial.get("checkpoints"),
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
        cards.append('<article><h2>%s · %s</h2><p>%s</p><details><summary>Source QA and actual generation input</summary>'
                     '<pre>%s</pre>%s</details><h3>Model-authored requirement</h3><pre>%s</pre>'
                     '<details><summary>Construction attempts and reasons</summary><pre>%s</pre></details>%s</article>' % (
            safe(item["task"]), safe(item["status"]), " · ".join(filter(None, [
                link(root / "frozen", "Frozen criteria and tests"),
                link(summary_path, "Requirements and attempt reasons"),
                link(root / "comparison.json", "Comparison"),
                link(root, "All task files")])),
            safe(json.dumps(qa, ensure_ascii=False, indent=2)),
            link(root / "author-reference/qa-input.json", "Original QA generation input"),
            safe(text_file(task_path)), safe(text_file(summary_path)), reference + "".join(trials)))
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Repository task results</title>
<style>body{font:16px/1.6 system-ui;background:#f4f6fa;color:#182336;margin:32px auto;max-width:1150px;padding:0 24px}
article{background:white;border:1px solid #dce3ed;border-radius:12px;padding:24px;margin:22px 0}a{color:#2157a5}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 ui-monospace,monospace;background:#f4f6fa;padding:16px}
summary{cursor:pointer;color:#2157a5}h1,h2{line-height:1.3}</style><h1>Repository task results</h1>
<p>Each requirement starts from the pinned dialogue-end code. Both trials use the same frozen criteria.
The memory trial receives only the historical answer as extra context.</p><p>%s</p>%s</html>''' % (
        safe("Target: %s · Completed pairs: %s · Stop: %s" % (
            manifest.get("target", "—"), manifest.get("completed", sum(t["status"] == "evaluated" for t in manifest.get("tasks", []))),
            manifest.get("stop_reason", "running"))), "".join(cards))
    (output / "report.html").write_text(page, encoding="utf-8")
