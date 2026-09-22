"""Observable costs; model reasoning text is measured, never inferred."""

from collections import Counter
from pathlib import Path
import json
import re


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
    result = {"tool_calls": len(actions), "tool_calls_by_type": dict(counts),
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
    result["total_tokens"] = result["prompt_tokens"] + result["completion_tokens"]
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


def checkpoint_summary(rows, expected):
    valid = (len(rows) == expected and {r["index"] for r in rows} == set(range(1, expected + 1))
             and all(r["status"] in {"observed", "alternative", "skipped", "uncertain"} for r in rows))
    counts = {status: sum(r["status"] == status for r in rows)
              for status in ("observed", "alternative", "skipped", "uncertain")}
    resolved = valid and expected > 0 and counts["uncertain"] == 0
    return {"rows": rows, "complete": valid, "resolved": resolved, "total": expected,
            "counts": counts, "action_coverage": counts["observed"] / expected if resolved else None}


def compare_checkpoints(comparison):
    """Compare exploration only when both tasks and both trace reviews are complete."""
    without = comparison.get("without_memory", {})
    with_memory = comparison.get("with_memory", {})
    both_passed = without.get("result") == with_memory.get("result") == "passed"
    left, right = without.get("checkpoints", {}), with_memory.get("checkpoints", {})
    eligible = (both_passed and left.get("resolved", False) and right.get("resolved", False)
                and left.get("total") == right.get("total"))
    result = {"eligible": bool(eligible), "both_passed": both_passed,
              "reason": ("both_passed_with_resolved_checkpoints" if eligible else
                         "task_not_passed" if not both_passed else "checkpoint_review_incomplete")}
    if not eligible:
        return result
    left_rows = {row["index"]: row["status"] for row in left["rows"]}
    right_rows = {row["index"]: row["status"] for row in right["rows"]}
    result.update(without_memory_coverage=left["action_coverage"],
                  with_memory_coverage=right["action_coverage"],
                  coverage_reduction=left["action_coverage"] - right["action_coverage"],
                  skipped_in_memory=[i for i in left_rows if left_rows[i] == "observed"
                                     and right_rows[i] == "skipped"],
                  alternative_in_memory=[i for i in left_rows if left_rows[i] == "observed"
                                         and right_rows[i] == "alternative"],
                  additional_in_memory=[i for i in left_rows if left_rows[i] != "observed"
                                        and right_rows[i] == "observed"])
    return result
