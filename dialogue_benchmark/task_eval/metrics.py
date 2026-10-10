"""Observable costs; model reasoning text is measured, never inferred."""

from collections import Counter
from pathlib import Path
import json
import re


def cache_usage(rows):
    """Summarize provider-reported prompt cache usage without guessing missing fields."""
    rows = list(rows or [])
    if not rows:
        return {"cache_hit_tokens": None, "cache_miss_tokens": None,
                "cache_observed_prompt_tokens": None, "cache_usage_complete": False,
                "cache_hit_rate": None}
    hit = miss = observed = 0
    complete = True
    for row in rows:
        usage = row.get("usage", row) if isinstance(row, dict) else {}
        # Stage ledgers may already contain a cache summary rather than raw
        # provider usage. Reuse it without counting it as a second request.
        if usage.get("cache_usage_complete") is True:
            summary_hit = usage.get("cache_hit_tokens")
            summary_miss = usage.get("cache_miss_tokens")
            if isinstance(summary_hit, int) and isinstance(summary_miss, int):
                hit += summary_hit
                miss += summary_miss
                observed += summary_hit + summary_miss
                continue
        elif usage.get("cache_usage_complete") is False:
            complete = False
            continue
        deep_hit = usage.get("prompt_cache_hit_tokens")
        deep_miss = usage.get("prompt_cache_miss_tokens")
        details = usage.get("prompt_tokens_details") or {}
        open_hit = details.get("cached_tokens")
        prompt = usage.get("prompt_tokens")
        if deep_hit is not None or deep_miss is not None:
            if deep_hit is None or deep_miss is None:
                complete = False
                continue
            row_hit, row_miss = deep_hit, deep_miss
        elif open_hit is not None and prompt is not None:
            row_hit, row_miss = open_hit, prompt - open_hit
        else:
            complete = False
            continue
        hit += row_hit
        miss += row_miss
        observed += row_hit + row_miss
    if not complete:
        return {"cache_hit_tokens": hit if observed else None,
                "cache_miss_tokens": miss if observed else None,
                "cache_observed_prompt_tokens": observed if observed else None,
                "cache_usage_complete": False, "cache_hit_rate": None}
    return {"cache_hit_tokens": hit, "cache_miss_tokens": miss,
            "cache_observed_prompt_tokens": observed,
            "cache_usage_complete": True,
            "cache_hit_rate": hit / observed if observed else None}


def text_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(x.get("text", "") for x in content if isinstance(x, dict))
    return ""


def measure(events, provider_path):
    actions = [e for e in events if e.get("kind") == "ActionEvent"]
    by_call = {e.get("tool_call_id"): e for e in actions}
    counts = Counter(e.get("tool_name") for e in actions)
    file_texts = []
    views = [e for e in actions if e.get("tool_name") == "file_editor"
             and (e.get("action") or {}).get("command") == "view"]
    shell_reads = [e for e in actions if e.get("tool_name") == "terminal" and re.search(
        r"(?:^|&&|\||;)\s*(?:cat|sed|head|tail|nl|rg|grep)\b",
        (e.get("action") or {}).get("command", ""))]
    shell_texts = []
    for e in events:
        call = by_call.get(e.get("tool_call_id"), {})
        if (e.get("kind") == "ObservationEvent" and call in views):
            file_texts.append(text_content((e.get("observation") or {}).get("content")))
        if e.get("kind") == "ObservationEvent" and call in shell_reads:
            shell_texts.append(text_content((e.get("observation") or {}).get("content")))
    responses = []
    if Path(provider_path).exists():
        responses = [json.loads(line) for line in Path(provider_path).read_text().splitlines()
                     if line.strip()]
        responses = [r for r in responses if r.get("kind") == "response"]
    reasoning = [choice.get("message", {}).get("reasoning_content")
                 for r in responses for choice in r.get("output", {}).get("choices", [])]
    reasoning = [r for r in reasoning if isinstance(r, str)]
    thoughts = [text_content(e.get("thought")) for e in actions if text_content(e.get("thought"))]
    development = [e for e in actions if e.get("tool_name") not in {"think", "finish"}]
    result = {"tool_calls": len(development), "raw_action_count": len(actions),
              "tool_calls_by_type": dict(Counter(e.get("tool_name") for e in development)),
              "raw_actions_by_type": dict(counts),
              "file_view_calls": len(views),
              "file_view_chars": sum(map(len, file_texts)),
              "file_read_scope": "Explicit file views plus separately labeled shell read/search observations; not all script reads",
              "file_view_tokens_estimate": None, "file_tokenizer": None,
              "shell_read_or_search_calls": len(shell_reads),
              "shell_read_or_search_output_chars": sum(map(len, shell_texts)),
              "shell_read_or_search_output_tokens_estimate": None,
              "file_read_count_complete": False,
              "provider_requests": len(responses),
              "prompt_tokens": sum(r.get("usage", {}).get("prompt_tokens", 0) for r in responses),
              "completion_tokens": sum(r.get("usage", {}).get("completion_tokens", 0) for r in responses),
              "reasoning_chars": sum(map(len, reasoning)) if reasoning else None,
              "reasoning_tokens_estimate": None,
              "visible_thought_chars": sum(map(len, thoughts)) if thoughts else None,
              "think_tool_calls": counts.get("think", 0),
              "usage_complete": bool(responses) and all(
                  "prompt_tokens" in r.get("usage", {}) and "completion_tokens" in r.get("usage", {})
                  for r in responses)}
    result.update(cache_usage(responses))
    result["total_tokens"] = result["prompt_tokens"] + result["completion_tokens"]
    # Explicit names make the accounting boundary clear to reports and
    # callers.  ``total_tokens`` remains the solver ledger for compatibility.
    result["solver_tokens"] = result["total_tokens"]
    result["responder_tokens"] = 0
    result["injection_tokens"] = None
    try:
        import tiktoken
        tokenizer = tiktoken.get_encoding("cl100k_base")
        result["file_tokenizer"] = "cl100k_base estimate, not DeepSeek billing tokens"
        result["file_view_tokens_estimate"] = sum(len(tokenizer.encode(t, disallowed_special=())) for t in file_texts)
        result["shell_read_or_search_output_tokens_estimate"] = sum(
            len(tokenizer.encode(t, disallowed_special=())) for t in shell_texts)
        if reasoning:
            result["reasoning_tokens_estimate"] = sum(len(tokenizer.encode(t, disallowed_special=())) for t in reasoning)
    except (ImportError, OSError, ValueError):
        pass
    return result



def evaluation_failure(trial):
    """Return why an uncertain trial reflects an evaluator failure, else ``None``.

    A Judge or check-runner error says nothing about the solver.  Such a
    trial must not count as a solver outcome in pair classes or pass rates.
    """
    if not isinstance(trial, dict) or trial.get("result") != "uncertain":
        return None
    if trial.get("judge_status") == "error":
        return "judge_error"
    if (trial.get("checks") or {}).get("status") == "error":
        return "checks_error"
    return None


def scored_task(task):
    """Return whether a task row counts toward approved pass-rate totals."""
    pair = compare_trials(task.get("comparison", {}))
    return (not task.get("provisional") and pair["pair_class"] != "evaluation_failed")


def compare_trials(comparison):
    """Compare outcomes and retain raw costs separately from fair comparisons.

    Raw deltas describe what each arm actually spent, even when one arm did
    not pass.  Comparable deltas are intentionally limited to pairs where
    both arms passed, so an early failure is never reported as an efficiency
    win.
    """
    left, right = (comparison.get(k, {}) for k in ("without_memory", "with_memory"))
    complete = all(trial.get("result") in {"passed", "failed", "uncertain"} for trial in (left, right))
    both = left.get("result") == right.get("result") == "passed"
    failures = {condition: reason for condition, reason in (
        ("without_memory", evaluation_failure(left)), ("with_memory", evaluation_failure(right)))
        if reason}
    if not complete:
        pair_class = "incomplete"
    elif failures:
        pair_class = "evaluation_failed"
        complete = False
    elif both:
        pair_class = "both_passed"
    elif right.get("result") == "passed":
        pair_class = "memory_capability_gain"
    elif left.get("result") == "passed":
        pair_class = "memory_harm_candidate"
    else:
        pair_class = "both_failed"
    solver_usage_complete = all(
        trial.get("metrics", {}).get("usage_complete") is True
        for trial in (left, right))
    responder_trials = [trial.get("responder_cost") for trial in (left, right)]
    responder_usage_complete = (
        not any(responder_trials)
        or all(cost and cost.get("usage_complete") is True for cost in responder_trials))
    efficiency_ready = both and solver_usage_complete and responder_usage_complete
    result = {"both_passed": both, "pair_class": pair_class,
              "efficiency_comparable": efficiency_ready,
              "completion_difference":
              int(right.get("result") == "passed") - int(left.get("result") == "passed") if complete else None,
              "raw_cost_differences": {}, "comparable_cost_differences": {}}
    if failures:
        result["evaluation_failures"] = failures
    for key in ("tool_calls", "file_view_calls", "shell_read_or_search_calls",
                "prompt_tokens", "completion_tokens", "cache_hit_tokens",
                "cache_miss_tokens", "total_tokens", "solver_tokens",
                "responder_tokens", "injection_tokens"):
        a, b = left.get("metrics", {}).get(key), right.get("metrics", {}).get(key)
        raw_valid = isinstance(a, (int, float)) and isinstance(b, (int, float))
        if key in {"prompt_tokens", "completion_tokens", "total_tokens", "solver_tokens"}:
            raw_valid = raw_valid and solver_usage_complete
        elif key in {"cache_hit_tokens", "cache_miss_tokens"}:
            raw_valid = raw_valid and all(
                t.get("metrics", {}).get("cache_usage_complete") is True
                for t in (left, right))
        elif key == "responder_tokens":
            raw_valid = raw_valid and responder_usage_complete
        raw = b - a if raw_valid else None
        result["raw_cost_differences"][key] = raw
        result["comparable_cost_differences"][key] = raw if efficiency_ready else None
    # Keep the old key as the fair, both-passed-only view for existing reports
    # and callers.  New consumers should use the explicit names above.
    result["cost_differences"] = dict(result["comparable_cost_differences"])
    a, b = left.get("history_question_count"), right.get("history_question_count")
    result["history_question_difference"] = b - a if isinstance(a, int) and isinstance(b, int) else None
    return result
