"""Read public OpenHands session exports without reading the live repository."""

import difflib
import json


def _tool_text(value):
    """Render native text with real newlines so record fragments keep code lines."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(str(k) + ":\n" + _tool_text(v) for k, v in value.items())
    if isinstance(value, list):
        return "\n".join(_tool_text(v) for v in value)
    return json.dumps(value, ensure_ascii=False)


def normalize_openhands(rows):
    from .normalize import is_truncated, source_kind_for, text_content

    records, calls = [], {}
    visible = bool(rows and rows[0][1].get("schema") == "model-visible-dialogue-v1")
    seen = set()

    def emit(item, row, line, suffix=""):
        records.append(dict(
            item, id="e%d" % (len(records) + 1), order=len(records) + 1,
            source_line=line, original_id=str(row.get("id", line)) + suffix,
            timestamp=row.get("timestamp"), workspace="/workspace/candidate",
            source_kind=source_kind_for(item)))
        if visible:
            records[-1]["input_schema"] = "model-visible-dialogue-v1"
            records[-1]["sequence"] = row["sequence"]

    for sequence, (line, row) in enumerate(rows, 1):
        if visible:
            if (row.get("schema") != "model-visible-dialogue-v1"
                    or row.get("sequence") != sequence
                    or not isinstance(row.get("id"), str) or not row["id"]
                    or row["id"] in seen):
                raise ValueError("Invalid public event identity or sequence")
            seen.add(row["id"])
        elif row.get("schema"):
            raise ValueError("Mixed or unsupported dialogue schema")
        kind = row.get("kind")
        if kind in {"user", "assistant"}:
            if visible and not isinstance(row.get("text"), str):
                raise ValueError("Public message requires actual text")
            emit({"kind": "message", "role": kind,
                  "text": text_content(row.get("text", ""))}, row, line)
        elif kind == "tool_call":
            if visible and (not row.get("call_id") or row["call_id"] in calls):
                raise ValueError("Missing or repeated public call ID")
            calls[row.get("call_id")] = row
            emit({"kind": "call", "call_id": row.get("call_id"),
                  "name": row.get("tool_name"),
                  "text": _tool_text(row.get("action", {}))}, row, line)
        elif kind == "tool_result":
            if visible:
                call = calls.get(row.get("call_id"))
                if (not call or call.get("tool_name") != row.get("tool_name")
                        or not isinstance(row.get("text"), str)):
                    raise ValueError("Unmatched public tool result or missing text")
                item = {"kind": "result", "call_id": row["call_id"],
                        "name": row["tool_name"], "text": row["text"]}
                if isinstance(row.get("is_error"), bool):
                    item["is_error"] = row["is_error"]
                emit(item, row, line)
                continue
            observation = row.get("observation") or {}
            emit({"kind": "result", "call_id": row.get("call_id"),
                  "name": row.get("tool_name"),
                  "text": _tool_text(observation)}, row, line)
            result_record = records[-1]
            call = calls.get(row.get("call_id"), {})
            action = call.get("action") or {}
            path = observation.get("path")
            command = observation.get("command")
            if (row.get("tool_name") != "file_editor"
                    or call.get("tool_name") != "file_editor"
                    or command not in {"str_replace", "create", "insert", "undo_edit"}
                    or command != action.get("command")
                    or not path or path != action.get("path")
                    or observation.get("is_error") is not False):
                continue
            before, after = observation.get("old_content"), observation.get("new_content")
            # A command alone does not establish a successful edit or a full file.
            if not isinstance(after, str) or is_truncated(after):
                continue
            # Complete versions are represented below, linked to this source line.
            # Keep the actual result and metadata without repeating those same bodies.
            represented = {"new_content"}
            if isinstance(before, str) and not is_truncated(before):
                represented.add("old_content")
            result_record["text"] = _tool_text({k: v for k, v in observation.items()
                                                 if k not in represented})
            if isinstance(before, str) and not is_truncated(before):
                emit({"kind": "observation", "path": path, "content": before, "complete": True},
                     row, line, ":before")
            elif observation.get("prev_exist") is not False:
                emit({"kind": "observation", "path": path, "content": after, "complete": True},
                     row, line, ":after")
                continue
            diff = "".join(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
                           for line in difflib.unified_diff(
                               (before or "").splitlines(True), after.splitlines(True)))
            change = ({"type": "add", "content": after} if observation.get("prev_exist") is False
                      else {"type": "update", "unified_diff": diff})
            emit({"kind": "patch", "call_id": row.get("call_id"), "success": True,
                  "changes": {path: change}}, row, line, ":after")
        else:
            raise ValueError("Unsupported OpenHands record at line %d" % line)
    return records
