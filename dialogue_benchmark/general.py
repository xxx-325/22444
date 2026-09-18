"""Build lightweight conversation-stage scopes for general memory QA.

The stage detector is deliberately conservative and local.  It does not claim
to understand the task; it only gives the model stable stage boundaries and
source IDs that can be used when combining facts across a long dialogue.
"""

import json
import re


_TOPIC_MARKERS = re.compile(
    r"(?:现在|然后|另外|接下来|换个|继续|最后|总结|部署|安装|测试|评估|设计|实现|修复|导出|分析|对比|"
    r"之前|后来|改成|改为|修改|保留|删除|移除|反馈|失败|报错|要求|确认|"
    r"可以吗|怎么|为什么|如何|需要|帮我|我想|我认为|同意|好的)", re.I)
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_.:/-]*|[\u4e00-\u9fff]{2,}")
_FILE = re.compile(r"(?<![A-Za-z0-9_])(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.(?:py|js|mjs|ts|tsx|jsx|json|yaml|yml)(?![A-Za-z0-9_])", re.I)
_SYMBOL = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}(?=\s*\(|\s*::)|\b(?:Attribute|stdin|stdout|SAFE_NAMES|EXPRESSION|checker)\b")
_RELATION_CUE = re.compile(r"之前|后来|现在|然后|改成|修改|保留|删除|移除|必须|不要|不能|通过|失败|报错|要求|反馈|确认|计划", re.I)
_REFERENCE_CUE = re.compile(r"之前|此前|前面|上述|原来|这个(?:决定|约束|方案)|该(?:决定|约束|方案)", re.I)
_WEAK_OBJECTS = {
    "之后", "后来", "现在", "然后", "这个", "那个", "一个", "问题",
    "方案", "要求", "内容", "部分", "结果", "实现", "修改", "确认",
    "支持", "使用", "需要", "应该", "不能", "可以", "用户", "助手",
}


def _tokens(text):
    return {item.lower() for item in _TOKEN.findall(text or "")}


def build_object_index(dialogue, graph=None):
    """Build a conservative reverse index from code objects to visible turns.

    This is a locator, not a semantic relation graph: an object is linked to a
    message only when its normalized spelling or an explicit alias occurs in
    the original text.  Code graph records provide aliases for files/symbols;
    they do not prove a user intention or causal relation.
    """
    objects = {}

    def add(kind, name, aliases, source_id=None, stage_id=None, source_bucket=None):
        normalized = name.casefold().replace("\\", "/")
        item = objects.setdefault(normalized, {
            "object_id": normalized, "object_kind": kind,
            "normalized_name": normalized, "aliases": sorted(set(aliases)),
            "message_ids": [], "event_ids": [], "version_ids": [], "stage_ids": [],
        })
        item["aliases"] = sorted(set(item["aliases"]) | set(aliases))
        if source_id:
            bucket = source_bucket or "event_ids"
            if source_id not in item[bucket]:
                item[bucket].append(source_id)
        if stage_id and stage_id not in item["stage_ids"]:
            item["stage_ids"].append(stage_id)
        return item

    graph = graph or {}
    for version in graph.get("versions", []):
        path = version.get("path")
        if not isinstance(path, str):
            continue
        aliases = {path, path.rsplit("/", 1)[-1]}
        add("file", path, aliases, version.get("source"), source_bucket="event_ids")
        version_id = version.get("id")
        item = objects[path.casefold().replace("\\", "/")]
        if version_id and version_id not in item["version_ids"]:
            item["version_ids"].append(version_id)
        for symbol in (version.get("code") or {}).get("symbols", []):
            if isinstance(symbol, dict) and isinstance(symbol.get("name"), str):
                add("symbol", symbol["name"], {symbol["name"]},
                    version.get("source"), source_bucket="event_ids")

    for record in dialogue:
        text = record.get("text", "") if isinstance(record.get("text"), str) else ""
        stage_id = record.get("stage_id")
        names = set(_FILE.findall(text)) | set(_SYMBOL.findall(text))
        for name in names:
            kind = "file" if "." in name and name.rsplit(".", 1)[-1].lower() in {
                "py", "js", "mjs", "ts", "tsx", "jsx", "json", "yaml", "yml"} else "symbol"
            aliases = {name}
            if kind == "file":
                aliases.add(name.rsplit("/", 1)[-1])
            item = add(kind, name, aliases, record.get("id"), stage_id,
                       source_bucket="message_ids")
            if record.get("id") not in item["message_ids"]:
                item["message_ids"].append(record.get("id"))
        # Add explicit Chinese/English decision objects only when a cue and a
        # concrete token coexist; generic prose is intentionally excluded.
        if _RELATION_CUE.search(text):
            for token in sorted(_tokens(text)):
                if len(token) >= 3 and token not in {"user", "assistant", "需要", "可以"}:
                    add("conversation_object", token, {token}, record.get("id"), stage_id,
                        source_bucket="message_ids")
    return sorted(objects.values(), key=lambda item: item["object_id"])


def identify_stages(records, gap_threshold=180):
    """Return explicit, reproducible stage metadata for visible dialogue.

    A new stage starts at a user turn when the turn introduces a new vocabulary
    cluster, contains a clear task marker, or is separated by a large normalized
    order gap.  This is a candidate boundary, not a semantic ground truth.
    """
    visible = [record for record in records
               if record.get("kind") == "message" and record.get("role") in {"user", "assistant"}]
    stages = []
    current = None
    previous_user_tokens = set()
    for record in visible:
        role = record.get("role")
        text = record.get("text", "") if isinstance(record.get("text", ""), str) else ""
        tokens = _tokens(text)
        starts = current is None
        if role == "user" and current is not None:
            overlap = len(tokens & previous_user_tokens)
            novel = len(tokens - previous_user_tokens)
            starts = (record.get("order", 0) - current["last_order"] >= gap_threshold
                      or bool(_TOPIC_MARKERS.search(text)) and novel >= 2
                      or (novel >= 5 and overlap == 0))
        if starts:
            stage_id = "stage-%d" % (len(stages) + 1)
            current = {"id": stage_id, "record_ids": [], "start_order": record.get("order"),
                       "end_order": record.get("order"), "label": text[:80]}
            stages.append(current)
        current["record_ids"].append(record["id"])
        current["end_order"] = record.get("order")
        current["last_order"] = record.get("order")
        if role == "user":
            previous_user_tokens = tokens
    for stage in stages:
        stage.pop("last_order", None)
    return stages


def _message_objects(record):
    """Return concrete repeated objects suitable for an explicit relation."""
    text = record.get("text", "") if isinstance(record.get("text"), str) else ""
    objects = set(_FILE.findall(text)) | set(_SYMBOL.findall(text))
    objects.update(token.casefold() for token in _tokens(text)
                   if token.casefold() not in _WEAK_OBJECTS)
    return objects


def _discussion_edges(dialogue, stages):
    """Link only explicit cross-stage revisions and feedback references.

    Shared vocabulary is used as an object anchor, never as a relation by
    itself.  The later message must also contain a change/reference cue.
    """
    stage_by_id = {stage["id"]: stage for stage in stages}
    edges = []
    for left_index, left in enumerate(dialogue):
        left_stage = left.get("stage_id")
        left_objects = _message_objects(left)
        if not left_stage or not left_objects:
            continue
        for right in dialogue[left_index + 1:]:
            right_stage = right.get("stage_id")
            if not right_stage or right_stage == left_stage:
                continue
            if not (left_objects & _message_objects(right)):
                continue
            right_text = right.get("text", "")
            if not (_RELATION_CUE.search(right_text) or _REFERENCE_CUE.search(right_text)):
                continue
            relation = "feedback" if re.search(r"反馈|问题是|不对|不符合", right_text, re.I) else "revision"
            edges.append({
                "from": left["id"], "to": right["id"],
                "kind": "discussion_relation", "relation": relation,
                "source": right["id"], "source_kind": "conversation",
                "stage_from": left_stage, "stage_to": right_stage,
            })
    return edges


def build_general_scope(records, cutoff, max_chars=24000, graph=None):
    """Create a dialogue-only scope; no repository facts are added."""
    dialogue = [dict(record) for record in records
                if record.get("kind") == "message"
                and record.get("role") in {"user", "assistant"}
                and record.get("order", 0) <= cutoff]
    stages = identify_stages(dialogue)
    stage_by_record = {
        record_id: stage["id"]
        for stage in stages for record_id in stage["record_ids"]
    }
    for record in dialogue:
        record["stage_id"] = stage_by_record.get(record["id"])
    object_index = build_object_index(dialogue, graph)
    edges = _discussion_edges(dialogue, stages)
    scope = {
        "seed": None,
        "cutoff": cutoff,
        "nodes": [],
        "edges": edges,
        "historical_edges": [],
        "versions": [],
        "events": [],
        "dialogue": dialogue,
        "stages": stages,
        "stage_count": len(stages),
        "object_index": object_index,
    }
    scope["context_chars"] = len(json.dumps(scope, ensure_ascii=False, separators=(",", ":")))
    scope["max_context_chars"] = max_chars
    scope["over_budget"] = scope["context_chars"] > max_chars
    return scope
