"""Derive a fixed exploration reference and match observable solver actions."""

from pathlib import Path
import json

from ..llm import stage_error
from ..security import redact_credential_assignments
from . import prompts
from .artifacts import save
from .metrics import checkpoint_summary, text_content
from .runtime import ask_model


def project_trajectory(events, output_limit=4000):
    """Keep tool actions and results, omitting duplicate editor bodies and reasoning."""
    observations = {}
    for event in events:
        if event.get("kind") == "ObservationEvent":
            observations.setdefault(event.get("tool_call_id"), []).append(event)
    steps, sources = [], {}
    for event in events:
        if event.get("kind") != "ActionEvent" or event.get("tool_name") in {"think", "finish"}:
            continue
        number = len(steps) + 1
        action = event.get("action") or {}
        step = {"step": number, "tool": event.get("tool_name"),
                "action": {key: (redact_credential_assignments(action[key])
                                 if isinstance(action[key], str) else action[key])
                           for key in ("command", "path", "view_range") if key in action},
                "results": []}
        results = observations.get(event.get("tool_call_id"), [])
        for result in results:
            observation = result.get("observation") or {}
            content = redact_credential_assignments(text_content(observation.get("content")))
            if len(content) > output_limit:
                tail = max(1, output_limit // 4)
                content = content[:output_limit - tail] + "\n[Middle output omitted]\n" + content[-tail:]
            step["results"].append({"text": content,
                                    "is_error": observation.get("is_error"),
                                    "exit_code": (observation.get("exit_code")
                                                  if observation.get("exit_code") is not None else
                                                  (observation.get("metadata") or {}).get("exit_code"))})
        steps.append(step)
        if results and event.get("id"):
            sources[str(number)] = {"action_id": event["id"],
                                    "observation_ids": [item["id"] for item in results if item.get("id")]}
    return steps, sources


def trajectory_payload(context, trajectory, prompt, max_chars=54000):
    """Keep every action and source while fitting literal output excerpts."""
    for limit in (4000, 2000, 1000, 500, 250, 125):
        steps, sources = project_trajectory(trajectory, output_limit=limit)
        payload = dict(context, steps=steps)
        if len(prompt) + len(json.dumps(payload, ensure_ascii=False)) <= max_chars:
            return payload, sources
    raise ValueError("trajectory_actions_exceed_request_budget")


def _references(value, sources):
    if isinstance(value, str):
        value = [] if value.strip().lower() == "none" else value.split(",")
    if not isinstance(value, list):
        raise ValueError("missing_step_references")
    references = list(dict.fromkeys(str(item).strip() for item in value))
    if any(item not in sources for item in references):
        raise ValueError("unknown_or_unobserved_step")
    return [sources[item] for item in references]


def extract_checkpoints(task, trajectory, config, output, *, source="accepted_reference_trajectory"):
    """Called once for the accepted reference run, before either scored trial."""
    try:
        prompt = prompts.CHECKPOINT_EXTRACT
        if source == "independent_design_probe":
            prompt = prompt.replace("一次已通过验收的无记忆参考运行", "一次独立无记忆设计探针（不保证通过验收）")
        payload, sources = trajectory_payload({"task": task}, trajectory, prompt)
        response = ask_model(prompt, payload, config, output)
        if not isinstance(response.get("facts"), list):
            raise ValueError("missing_checkpoint_list")
        checkpoints, seen = [], set()
        for fact in response["facts"]:
            text = fact.get("statement", "").strip()
            evidence = _references(fact.get("sources"), sources)
            if not text or not evidence:
                raise ValueError("unsupported_checkpoint")
            if text in seen:
                continue
            seen.add(text)
            checkpoints.append({"index": len(checkpoints) + 1, "text": text,
                                "reference_evidence": evidence})
        result = {"status": "completed", "source": source,
                  "checkpoints": checkpoints}
    except Exception as error:
        result = {"status": "failed", "error": stage_error("checkpoint_extraction", error)}
    save(Path(output) / "result.json", result)
    return result


def write_checkpoints(spec, extracted, reference_run):
    """Render the same fixed list for inspection and for later matching."""
    document = dict(extracted, reference_run=reference_run)
    save(Path(spec) / "checkpoints.json", document)
    lines = ["# Exploration checkpoints", "",
             "Source: %s. Status: %s. These are observed exploration actions, not required steps." % (
                 extracted.get("source", "unavailable"), extracted["status"]), ""]
    for row in document["checkpoints"]:
        lines.append("%d. %s" % (row["index"], row["text"]))
    if not document["checkpoints"]:
        lines.append("No supported exploration checkpoints were identified.")
    (Path(spec) / "checkpoints.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def match_checkpoints(checkpoints, trajectory, config, output, *, trajectory_complete):
    """Only this solver's trace is eligible evidence; all counts remain static."""
    rows = [{"index": cp["index"], "checkpoint": cp["text"], "status": "uncertain",
             "evidence": "No completed checkpoint review", "sources": []} for cp in checkpoints]
    error = None
    if rows and trajectory_complete:
        try:
            payload, sources = trajectory_payload(
                {"checkpoints": [{"index": cp["index"], "text": cp["text"]} for cp in checkpoints]},
                trajectory, prompts.CHECKPOINT_MATCH)
            response = ask_model(prompts.CHECKPOINT_MATCH, payload, config, output)
            reviews = response.get("reviews", [])
            expected = {str(row["index"]) for row in rows}
            if any(review.get("id") not in expected for review in reviews):
                raise ValueError("unknown_checkpoint")
            for row in rows:
                matches = [review for review in reviews if review.get("id") == str(row["index"])]
                try:
                    if len(matches) != 1:
                        raise ValueError("missing_or_duplicate_checkpoint")
                    review = matches[0]
                    status = review.get("status")
                    if status not in {"observed", "alternative", "skipped", "uncertain"}:
                        raise ValueError("invalid_checkpoint_status")
                    evidence = _references(review.get("sources"), sources)
                    if status in {"observed", "alternative"} and not evidence:
                        raise ValueError("missing_action_evidence")
                    explanation = review.get("evidence", "").strip()
                    if not explanation:
                        raise ValueError("missing_checkpoint_evidence")
                    row.update(status=status, evidence=explanation, sources=evidence)
                except (ValueError, TypeError, AttributeError) as problem:
                    row["evidence"] = str(problem)
        except Exception as problem:
            error = stage_error("checkpoint_matching", problem)
    elif not trajectory_complete:
        error = {"code": "incomplete_solver_trajectory"}
    result = checkpoint_summary(rows, len(checkpoints))
    if error:
        result["error"] = error
    save(Path(output) / "result.json", result)
    return result
