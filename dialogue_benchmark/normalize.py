"""Normalize explicit records without executing or interpreting tool commands."""

import json
import re
from pathlib import Path

SOURCE_KINDS = {"conversation", "document", "tool", "code", "test"}


def source_kind_for(record, default=None):
    """Return an explicit source label, falling back to the record kind."""
    explicit = record.get("source_kind") if isinstance(record, dict) else None
    if explicit in SOURCE_KINDS:
        return explicit
    kind = record.get("kind") if isinstance(record, dict) else None
    return default or ({"message": "conversation", "call": "tool", "result": "tool",
                        "observation": "code", "patch": "code"}.get(kind, "conversation"))


def text_content(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(text_content(part) for part in value)
    if isinstance(value, dict):
        return text_content(value.get("text", ""))
    return ""


def normalize_path(value, workspace=None):
    path = value.replace("\\", "/")
    if workspace:
        root = workspace.replace("\\", "/").rstrip("/")
        if path.startswith(root + "/"):
            path = path[len(root) + 1:]
    # Absolute paths without a declared root stay distinct, never basename-merge.
    return path


def load_dialogue(path):
    raw = Path(path).read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows = [(i, json.loads(line)) for i, line in enumerate(raw.splitlines(), 1)
                if line.strip()]
        return normalize_codex(rows)
    document = json.loads(raw)
    if not isinstance(document, dict) or document.get("version") != 1:
        raise ValueError("Unified JSON requires version=1 and records[]")
    records = document.get("records")
    if not isinstance(records, list):
        raise ValueError("records must be a list")
    allowed = {"message", "call", "result", "observation", "patch"}
    result = []
    for index, item in enumerate(records, 1):
        if not isinstance(item, dict) or item.get("kind") not in allowed:
            raise ValueError("Invalid record at position %d" % index)
        record = dict(item, id="e%d" % index, order=index, source_line=index)
        record["source_kind"] = source_kind_for(record)
        record["workspace"] = item.get("workspace", document.get("workspace"))
        result.append(record)
    return result


def normalize_codex(rows):
    result = []
    workspace = None
    for line, row in rows:
        p = row.get("payload", {})
        kind = p.get("type")
        if row.get("type") in {"session_meta", "turn_context"}:
            workspace = p.get("cwd", workspace)
        item = None
        if row.get("type") == "response_item":
            if kind == "message" and p.get("role") in {"user", "assistant"}:
                item = {"kind": "message", "role": p["role"],
                        "text": text_content(p.get("content", []))}
            elif kind in {"function_call", "custom_tool_call"}:
                item = {"kind": "call", "call_id": p.get("call_id"),
                        "name": p.get("name"), "text": p.get("arguments", p.get("input", ""))}
            elif kind in {"function_call_output", "custom_tool_call_output"}:
                item = {"kind": "result", "call_id": p.get("call_id"),
                        "text": text_content(p.get("output", ""))}
        elif row.get("type") == "event_msg" and kind == "patch_apply_end":
            item = {"kind": "patch", "call_id": p.get("call_id"),
                    "success": p.get("success") is True, "changes": p.get("changes", {})}
        if item is not None:
            record = dict(item, id="e%d" % (len(result) + 1), order=len(result) + 1,
                          source_line=line, timestamp=row.get("timestamp"), workspace=workspace)
            metadata = p.get("metadata", {}) if isinstance(p, dict) else {}
            explicit = item.get("source_kind") or p.get("source_kind") or (
                metadata.get("source_kind") if isinstance(metadata, dict) else None)
            record["source_kind"] = source_kind_for(dict(record, source_kind=explicit))
            result.append(record)
    return result


def source_view(record):
    """Only these fields may be selected for model context."""
    return {key: record[key] for key in ("id", "order", "source_line", "timestamp",
                                         "kind", "source_kind", "role", "call_id", "name",
                                         "text", "path", "content", "success", "changes")
            if key in record}


def is_truncated(text):
    return bool(re.search(r"truncated output|output truncated|\[\.\.\.truncated", text, re.I))
