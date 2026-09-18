"""Split a graph scope into bounded, independently auditable evidence chunks."""

import copy
import json


def _size(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def _line_parts(text, target_chars):
    """Split text only at complete line boundaries and retain line locations."""
    lines = text.splitlines(keepends=True)
    if not lines:
        return [(text, 1, 1)]
    parts = []
    start = 0
    current = []
    current_chars = 0
    for index, line in enumerate(lines):
        if current and current_chars + len(line) > target_chars:
            parts.append(("".join(current), start + 1, index))
            start = index
            current = []
            current_chars = 0
        current.append(line)
        current_chars += len(line)
        # A single very long line remains intact. The resulting chunk is
        # explicitly marked over budget instead of silently truncating it.
        if len(current) == 1 and current_chars > target_chars:
            parts.append(("".join(current), start + 1, index + 1))
            start = index + 1
            current = []
            current_chars = 0
    if current:
        parts.append(("".join(current), start + 1, len(lines)))
    return parts


def _fragment_record(record, target_chars):
    """Return lossless, source-linked fragments for one oversized record."""
    if _size(record) <= target_chars:
        return [copy.deepcopy(record)]

    fragments = []
    fragment_fields = []
    if isinstance(record.get("text"), str):
        fragment_fields.append(("text", None, record["text"]))
    elif isinstance(record.get("content"), str):
        fragment_fields.append(("content", None, record["content"]))
    elif isinstance(record.get("changes"), dict):
        for path, change in record["changes"].items():
            if not isinstance(change, dict):
                fragment_fields.append(("raw_change", path, json.dumps(
                    change, ensure_ascii=False, separators=(",", ":"))))
                continue
            text_field = next((field for field in ("unified_diff", "content")
                               if isinstance(change.get(field), str)), None)
            if text_field is not None:
                fragment_fields.append((text_field, path, change[text_field]))
            else:
                fragment_fields.append(("change_metadata", path, ""))

    if not fragment_fields:
        return [copy.deepcopy(record)]

    payload_target = max(512, target_chars // 2)
    for field, path, value in fragment_fields:
        for text, start_line, end_line in _line_parts(value, payload_target):
            item = {key: copy.deepcopy(value) for key, value in record.items()
                    if key not in {"text", "content", "changes"}}
            if path is None:
                item[field] = text
            else:
                original_change = record["changes"][path]
                compact_change = ({key: copy.deepcopy(value)
                                   for key, value in original_change.items()
                                   if key not in {"unified_diff", "content"}}
                                  if isinstance(original_change, dict)
                                  else {"raw": copy.deepcopy(original_change)})
                if field not in {"change_metadata", "raw_change"}:
                    compact_change[field] = text
                item["changes"] = {path: compact_change}
            item["parent_id"] = record.get("id")
            item["fragment"] = {
                "field": field,
                "path": path,
                "line_range": [start_line, end_line],
                "complete_record": False,
            }
            fragments.append(item)

    total = len(fragments)
    for index, item in enumerate(fragments, 1):
        item["id"] = "%s#fragment-%d" % (record.get("id"), index)
        item["fragment"].update(index=index, count=total)
    return fragments or [copy.deepcopy(record)]


def _version_metadata(version):
    """Avoid sending code twice when its source record already carries it."""
    keys = ("id", "path", "observed_at", "source", "status", "previous", "sha256")
    result = {key: copy.deepcopy(version.get(key)) for key in keys if key in version}
    result["content_in_source_record"] = True
    return result


def split_scope(scope, max_chars=24000, overlap_records=2):
    """Return contiguous scope chunks without inventing or truncating records.

    Each chunk keeps only versions/events whose observation order intersects its
    dialogue window. A small record overlap preserves local cause/effect context.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    fragment_target = max(512, max_chars - 2048)
    dialogue = [fragment for record in scope.get("dialogue", [])
                for fragment in _fragment_record(record, fragment_target)]
    if not dialogue:
        return []
    chunks = []
    start = 0

    def make_chunk(window, index):
        lo, hi = window[0]["order"], window[-1]["order"]
        window_ids = {record.get("id") for record in window}
        parent_ids = {record.get("parent_id", record.get("id")) for record in window}
        result = dict(scope, dialogue=window, nodes=[], subgraph_events=[])
        # The full object index is retained on the parent general scope for
        # static cross-turn lookup.  It is deliberately omitted from model
        # chunks: facts extraction already has lossless source records and
        # their IDs, while repeating an index for a document-sized message can
        # push an otherwise valid chunk over the request budget.
        result.pop("object_index", None)
        if scope.get("stages"):
            result["stages"] = []
            for stage in scope["stages"]:
                original_ids = set(stage.get("record_ids", []))
                selected = [record.get("id") for record in window
                            if record.get("id") in original_ids
                            or record.get("parent_id") in original_ids]
                if selected:
                    result["stages"].append(dict(stage, record_ids=selected))
        result["events"] = [e for e in scope.get("events", [])
                             if lo <= e.get("order", -1) <= hi]
        result["versions"] = [_version_metadata(v) for v in scope.get("versions", [])
                               if lo <= v.get("observed_at", -1) <= hi]
        source_ids = {e["id"] for e in result["events"]}
        result["edges"] = [e for e in scope.get("edges", [])
                            if e.get("source") in source_ids]
        result["historical_edges"] = [e for e in scope.get("historical_edges", [])
                                       if e.get("source") in source_ids]
        result["chunk_index"] = index
        result["chunk_window"] = [lo, hi]
        result["chunk_record_ids"] = sorted(window_ids)
        result["chunk_parent_ids"] = sorted(parent_ids)
        result["context_chars"] = _size(result)
        result["max_context_chars"] = max_chars
        result["over_budget"] = result["context_chars"] > max_chars
        return result

    while start < len(dialogue):
        end = start
        candidate = None
        while end < len(dialogue):
            probe = make_chunk(dialogue[start:end + 1], len(chunks))
            if probe["context_chars"] > max_chars and end > start:
                break
            candidate = probe
            end += 1
            if candidate["context_chars"] > max_chars:
                break
        if candidate is None:
            candidate = make_chunk([dialogue[start]], len(chunks))
            end = start + 1
        chunks.append(candidate)
        # Once the window reaches the end, overlapping it would only create
        # progressively smaller duplicate tail chunks.
        if end >= len(dialogue):
            break
        next_start = max(end - overlap_records, start + 1)
        start = next_start
    return chunks
