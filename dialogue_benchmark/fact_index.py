"""Build small cross-chunk evidence groups from source-grounded facts."""

import json
import re
from collections import deque

from .normalize import source_kind_for
from .protocol import MISSING_KINDS


_IDENTIFIER = re.compile(
    r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+|"
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+|"
    r"[A-Za-z_][A-Za-z0-9_]{2,}")
_CHINESE = re.compile(r"[\u4e00-\u9fff]{2,}")
# Quoted or code-marked Chinese terms are usually deliberate object names
# (for example, `缓存策略`) rather than incidental sentence n-grams. Keep a
# separate namespace so one such term can link facts without weakening the
# default collision guard for ordinary Chinese prose.
_CHINESE_EXPLICIT = re.compile(
    r"`([^`\n]{3,48})`|[\"“「《]([^\"”」》\n]{3,48})[\"”」》]")
_QUALIFIED_OBJECT = re.compile(
    r"(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+::[A-Za-z_][A-Za-z0-9_.-]*")
_CODE_PATH_OBJECT = re.compile(
    r"(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+(?:\.py|\.js|\.mjs|\.ts|\.tsx|"
    r"\.jsx|\.json|\.yaml|\.yml)")
_STOPWORDS = {
    "assistant", "dialogue", "error", "file", "function", "model", "result",
    "source", "test", "tests", "true", "false", "none", "python", "用户", "助手",
}
_CHINESE_STOP = {
    "用户", "助手", "要求", "认为", "可以", "需要", "目前", "这个", "那个", "然后",
    "已经", "进行", "问题", "内容", "对应", "相关", "时候", "里面", "一下", "一个",
    "版本", "快照", "事件", "文件", "代码", "函数", "模块", "旧版", "新版", "当前",
    "历史", "工具", "结果", "输出", "记录", "数据", "实现", "逻辑", "路径", "修改",
}
_GENERIC_ENTITIES = {
    "version", "snapshot", "event", "record", "source", "file", "module", "function",
    "code", "current", "latest", "old", "new", "history", "change", "result", "output",
    "error", "failure", "test", "tests", "true", "false", "none", "unknown",
    "script", "command", "time", "wall", "seconds", "completed", "success", "successful",
    "run", "running", "status", "message", "text", "data", "value", "values",
    "str", "int", "bool", "float", "list", "dict", "tuple", "set", "bytes",
    "mapping", "optional", "literal", "typealias", "dataclass", "self",
}
_RELATION_LABELS = {
    "change", "failure", "validation", "feedback", "constraint",
    "decision", "temporal", "conditional",
}
_WEAK_ENTITIES = {
    "logger", "parser", "client", "handler", "manager", "helper", "util", "utils",
    "service", "config", "common", "base", "data", "request", "response",
}
_EXTERNAL_KNOWLEDGE_CUE = re.compile(
    r"常识|标准|通常|一般来说|规范|协议|语义|复杂度|YAML|JSON|HTTP|REST|"
    r"Unicode|UTF-?8|正则|时间格式|时区|操作系统", re.I)
_CODE_FILE_SUFFIXES = (".py", ".js", ".mjs", ".ts", ".tsx", ".jsx", ".json", ".yaml", ".yml")
_OBSERVED_FAILURE = re.compile(
    r"历史失败|实际(?:运行)?失败|运行[^。\n]{0,120}(?:抛出|报错|异常|错误)|"
    r"(?:曾|已|出现|发生|遇到|触发|抛出|收到)[^。\n]{0,100}"
    r"(?:失败|报错|错误|异常|crash|error|failed|failure)|"
    r"(?:traceback|错误日志|失败回溯)", re.I)
_FAILURE_NEGATION = re.compile(
    r"(?:不要|不能|避免|防止|如何|怎么|如果|否则|无需|未(?:曾)?|没有)"
    r"[^。\n]{0,30}(?:失败|报错|错误|异常|crash|error|bug)", re.I)
_FAILURE_ABSENCE = re.compile(
    r"(?:无(?:错误|异常|报错)|没有(?:错误|异常|报错|失败)|"
    r"未(?:发生|出现|记录|提供)[^。\n]{0,30}(?:失败|错误|异常|报错)|"
    r"正常(?:结束|完成|通过)|success(?:fully)?\s+(?:completed|finished))", re.I)
_COUNTEREVIDENCE_CUE = re.compile(
    r"更正|纠正|并非|不是|不对|误报|实际(?:为|是|结果)|撤回|"
    r"correct(?:ion|ed)?|incorrect|wrong|revert(?:ed)?", re.I)
_HISTORICAL_EVENT_QUESTION = re.compile(
    r"当时|该次|此次|这次|在[^。？\n]{0,160}失败中|针对[^。？\n]{0,160}失败|"
    r"(?:应用|打上|完成)(?:该)?(?:补丁|修复|改动)后|补丁后|修复后|重新执行(?:的)?结果",
    re.I)
_OLD_STATE_CUE = re.compile(
    r"(?:之前|此前|原来|原先|曾经|当时|旧版|旧版本|旧实现|before|previous(?:ly)?|"
    r"formerly|used\s+to)", re.I)
_NEW_STATE_CUE = re.compile(
    r"(?:之后|后来|现在|当前|新版|新版本|新实现|改为|改成|替换为|"
    r"after(?:wards)?|now|current(?:ly)?|changed?\s+to|replaced?\s+with)", re.I)
_INLINE_TRANSITION = re.compile(
    r"(?:从|由)[^。；;\n]{1,160}(?:改为|改成|变为|替换为)|"
    r"(?:之前|此前|原来|旧版|before|previous(?:ly)?)[^。；;\n]{1,200}"
    r"(?:之后|后来|现在|当前|新版|after(?:wards)?|now)|"
    r"(?:之后|后来|现在|当前|新版|after(?:wards)?|now)[^。；;\n]{1,200}"
    r"(?:之前|此前|原来|旧版|before|previous(?:ly)?)", re.I)
_BEHAVIOR_OUTCOME = re.compile(
    r"(?:返回|抛出|触发|调用|写入|读取|发送|传递|拒绝|接受|跳过|保留|清除|"
    r"导致|生成|选择|使用|return(?:s|ed)?|raise[sd]?|throw[sn]?|call[sed]?|"
    r"write[sn]?|read[sd]?|send[sd]?|pass(?:es|ed)?|reject[sed]?|accept[sed]?|"
    r"skip(?:s|ped)?|retain[sed]?|clear[sed]?|cause[sd]?|produce[sd]?|use[sd]?)",
    re.I)
_LABELS = {
    "failure": re.compile(r"失败|报错|错误|异常|crash|error|failed|failure|bug", re.I),
    "validation": re.compile(r"测试|实验|验证|通过|passed|pytest|结果|回归", re.I),
    "change": re.compile(r"修改|改为|改成|删除|新增|移除|替换|升级|迁移|补丁|版本", re.I),
    "feedback": re.compile(r"反馈|不对|不需要|不能|应该|希望|要求|同意|确认", re.I),
    "constraint": re.compile(r"必须|只允许|不要|不能|限制|兼容|默认|约束", re.I),
    "decision": re.compile(r"决定|采用|选择|方案|最终|保留|放弃", re.I),
    "temporal": re.compile(r"之前|之后|后来|先|再|当前|当时|历史|旧版|新版|曾经", re.I),
    "conditional": re.compile(r"如果|当.+时|只有|否则|unless|when|if\b", re.I),
}


def _size(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def _dedupe(items, key="id"):
    result = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        value = item.get(key) if key is not None else None
        marker = (value if value is not None else
                  json.dumps(item, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")))
        if marker in seen:
            continue
        seen.add(marker)
        result.append(item)
    return result


def merge_scopes(scopes, qa_mode, model_request_chars=32000):
    """Merge chunk projections into one source index without restoring big graphs."""
    scopes = list(scopes)
    dialogue = _dedupe(
        [record for scope in scopes for record in scope.get("dialogue", [])])
    dialogue.sort(key=lambda item: (item.get("order", 0), item.get("id", "")))
    events = _dedupe([record for scope in scopes for record in scope.get("events", [])])
    events.sort(key=lambda item: (item.get("order", 0), item.get("id", "")))
    versions = _dedupe([record for scope in scopes for record in scope.get("versions", [])])
    versions.sort(key=lambda item: (item.get("observed_at", 0), item.get("id", "")))

    stages = {}
    for scope in scopes:
        for stage in scope.get("stages", []):
            if not isinstance(stage, dict) or not isinstance(stage.get("id"), str):
                continue
            target = stages.setdefault(stage["id"], dict(stage, record_ids=[]))
            target["record_ids"] = list(dict.fromkeys(
                target["record_ids"] + list(stage.get("record_ids", []))))
            target["start_order"] = min(target.get("start_order", stage.get("start_order", 0)),
                                        stage.get("start_order", target.get("start_order", 0)))
            target["end_order"] = max(target.get("end_order", stage.get("end_order", 0)),
                                      stage.get("end_order", target.get("end_order", 0)))

    source_graph_hops = {}
    for scope in scopes:
        hops = scope.get("adaptive", {}).get("selected_hops")
        if not isinstance(hops, int):
            continue
        for collection, order_key in (("dialogue", "order"), ("events", "order"),
                                      ("versions", "observed_at")):
            for item in scope.get(collection, []):
                if isinstance(item, dict) and isinstance(item.get("id"), str):
                    source_graph_hops[item["id"]] = hops

    coverage_flags = []
    for scope in scopes:
        if not isinstance(scope, dict):
            coverage_flags.append(False)
        elif (scope.get("full_range_covered") is False
              or scope.get("facts_failed") is True
              or scope.get("stage_failed") is True):
            coverage_flags.append(False)
        else:
            coverage_flags.append(not scope.get("over_budget", False))

    result = {
        "seed": next((scope.get("seed") for scope in scopes if scope.get("seed")), None),
        "cutoff": max((scope.get("cutoff", 0) for scope in scopes), default=0),
        "qa_mode": qa_mode,
        "nodes": [],
        "edges": _dedupe([edge for scope in scopes for edge in scope.get("edges", [])],
                           key=None),
        "historical_edges": _dedupe(
            [edge for scope in scopes for edge in scope.get("historical_edges", [])], key=None),
        "versions": versions,
        "events": events,
        "dialogue": dialogue,
        "stages": sorted(stages.values(), key=lambda item: item.get("start_order", 0)),
        "source_graph_hops": source_graph_hops,
        # An explicitly incomplete scope must stay incomplete after merging;
        # otherwise adversarial questions could mistake a partial range for a
        # complete history merely because each retained chunk fits its budget.
        "full_range_covered": bool(scopes) and all(coverage_flags),
        "model_request_chars": model_request_chars,
        "max_context_chars": model_request_chars,
    }
    result["stage_count"] = len(result["stages"])
    result["context_chars"] = _size(result)
    result["over_budget"] = result["context_chars"] > model_request_chars
    return result


def _record_text(record):
    values = []
    for key in ("text", "content"):
        if isinstance(record.get(key), str):
            values.append(record[key])
    if isinstance(record.get("changes"), dict):
        for path, change in record["changes"].items():
            values.append(path)
            if isinstance(change, dict):
                values.extend(value for key, value in change.items()
                              if key in {"unified_diff", "content"}
                              and isinstance(value, str))
    return "\n".join(values)


def _entities(text):
    result = set()
    for token in _IDENTIFIER.findall(text or ""):
        normalized = token.casefold()
        # Version/source IDs and generic prose are bookkeeping, not a shared
        # business object.  In particular, v930/e16924 must not connect every
        # fact extracted from the same version.
        if (normalized in _STOPWORDS or normalized in _GENERIC_ENTITIES
                or re.fullmatch(r"[ev]\d+", normalized)
                or re.fullmatch(r"(?:stage|chunk|scope)[-_]?\d+", normalized)
                or len(normalized) < 3
                or normalized.endswith(_CODE_FILE_SUFFIXES)):
            continue
        result.add(normalized)
    # Short Chinese n-grams have a very high accidental collision rate (for
    # example, "错误信息" appears in unrelated API descriptions).  Retain
    # only four-character terms and require two matching terms when they are
    # later used as a relation signal.
    for segment in _CHINESE.findall(text or ""):
        width = 4
        for start in range(max(0, len(segment) - width + 1)):
            token = segment[start:start + width]
            if (token not in _CHINESE_STOP
                    and not any(generic in token for generic in _CHINESE_STOP)):
                result.add("zh:" + token)
    # Quoted terms are deliberate references supplied by the speaker/model.
    # Keep them source-grounded, but distinguish them from accidental
    # four-character overlaps so a single explicit term can be trusted.
    for match in _CHINESE_EXPLICIT.finditer(text or ""):
        term = (match.group(1) or match.group(2) or "").strip()
        if any("\u4e00" <= char <= "\u9fff" for char in term):
            normalized = re.sub(r"\s+", "", term).casefold()
            if normalized and normalized not in _CHINESE_STOP:
                result.add("zhx:" + normalized)
    return result


def _semantic_shared_entities(left, right):
    """Return concrete shared objects, filtering weak Chinese collisions."""
    shared = left["entities"] & right["entities"]
    strong = {item for item in shared
              if not item.startswith(("zh:", "zhx:")) and item not in _WEAK_ENTITIES}
    weak = {item for item in shared if item in _WEAK_ENTITIES}
    chinese = {item for item in shared if item.startswith(("zh:", "zhx:"))}
    if len(weak) >= 2:
        strong |= weak
    if len(chinese) >= 2:
        return strong | chinese
    if len(chinese) == 1 and _single_chinese_link_allowed(chinese, (left, right)):
        return strong | chinese
    return strong


def _single_chinese_link_allowed(shared, infos):
    """Allow one Chinese object term only in an explicit change context.

    Four-character n-grams are useful for Chinese but collide frequently.
    Requiring a high-signal statement label on every participating fact keeps
    the relaxed path limited to facts about a recorded change, failure,
    validation, feedback, decision, or temporal transition.
    """
    if not shared or len(shared) != 1 or not infos:
        return False
    token = next(iter(shared))
    if token.startswith("zhx:"):
        return True
    labels = [set(info.get("statement_labels", set())) & _RELATION_LABELS
              for info in infos]
    return bool(labels and all(labels)
                and set.union(*labels) & _RELATION_LABELS)


def _shared_entities_for_infos(infos):
    """Return concrete entities shared by every fact in ``infos``."""
    if not infos:
        return set()
    shared = set.intersection(*(set(info["entities"]) for info in infos))
    strong = {item for item in shared
              if not item.startswith(("zh:", "zhx:")) and item not in _WEAK_ENTITIES}
    weak = {item for item in shared if item in _WEAK_ENTITIES}
    chinese = {item for item in shared if item.startswith(("zh:", "zhx:"))}
    if len(weak) >= 2:
        strong |= weak
    if len(chinese) >= 2:
        return strong | chinese
    if len(chinese) == 1 and _single_chinese_link_allowed(chinese, infos):
        return strong | chinese
    return strong


def _labels(text):
    return {name for name, pattern in _LABELS.items() if pattern.search(text or "")}


def _normalized_fact_statement(statement):
    """Return an intentionally strict fingerprint for duplicate extraction.

    Only outer whitespace is normalized. Case, internal whitespace, numbers,
    negation, conditions, and temporal wording remain significant so code
    literals or old/new states never collapse merely because they came from
    one source record.
    """
    return str(statement or "").strip()


def _merge_duplicate_facts(facts):
    """Collapse exact repeated extraction from the same source set.

    The first fact remains canonical and keeps its original statement and ID.
    Alias IDs are retained as metadata instead of being rewritten globally.
    """
    merged = []
    aliases = {}
    merged_ids = {}
    by_fingerprint = {}
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        sources = tuple(sorted(source for source in fact.get("sources", [])
                               if isinstance(source, str)))
        statement = _normalized_fact_statement(fact.get("statement", ""))
        marker = (sources, statement)
        canonical = by_fingerprint.get(marker) if sources and statement else None
        if canonical is None:
            canonical = dict(fact)
            merged.append(canonical)
            if isinstance(canonical.get("id"), str):
                merged_ids.setdefault(canonical["id"], [])
            if sources and statement:
                by_fingerprint[marker] = canonical
        else:
            duplicate_id = fact.get("id")
            canonical_id = canonical.get("id")
            if isinstance(duplicate_id, str) and isinstance(canonical_id, str):
                merged_ids[canonical_id] = list(dict.fromkeys(
                    merged_ids.get(canonical_id, []) + [duplicate_id]))
        fact_id = fact.get("id")
        canonical_id = canonical.get("id")
        if isinstance(fact_id, str) and isinstance(canonical_id, str):
            aliases[fact_id] = canonical_id
    return merged, aliases, merged_ids


def _source_index(scope):
    result = {}

    def add_source(source_id, info):
        if not isinstance(source_id, str):
            return
        result[source_id] = info
        # Oversized records are split losslessly and retain the original ID in
        # ``parent_id``. Treat that parent as an alias for grouping metadata.
        parent_id = info.get("parent_id")
        if isinstance(parent_id, str):
            result.setdefault(parent_id, info)

    for record in scope.get("dialogue", []):
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            continue
        paths = set(record.get("changes", {})) if isinstance(record.get("changes"), dict) else set()
        add_source(record["id"], {
            "order": record.get("order"), "stage_id": record.get("stage_id"),
            "paths": paths, "text": _record_text(record),
            "kinds": {record.get("kind")} if record.get("kind") else set(),
            "source_kinds": {source_kind_for(record)},
            "parent_id": record.get("parent_id"),
            "call_ids": {record["call_id"]} if isinstance(record.get("call_id"), str) else set(),
            "call_sources": set(),
        })
    for event in scope.get("events", []):
        if isinstance(event, dict) and isinstance(event.get("id"), str):
            target = result.setdefault(event["id"], {})
            target["order"] = event.get("order")
            target["paths"] = set(target.get("paths", set())) | set(
                event.get("affected_paths", []))
            if event.get("stage_id"):
                target["stage_id"] = event["stage_id"]
            if event.get("kind"):
                target.setdefault("kinds", set()).add(event["kind"])
            target.setdefault("source_kinds", set()).add(source_kind_for(event))
            if isinstance(event.get("call_source"), str):
                target.setdefault("call_sources", set()).add(event["call_source"])

    calls = {}
    for source_id, source in result.items():
        for call_id in source.get("call_ids", set()):
            calls.setdefault(call_id, set()).add(source_id)
    for sources in calls.values():
        for source_id in sources:
            result[source_id].setdefault("call_sources", set()).update(
                sources - {source_id})
    # Event normalization also records the originating call directly. Make
    # that edge symmetric so either a call or its result can seed review
    # guards without guessing from nearby text.
    for source_id, source in list(result.items()):
        for peer_id in tuple(source.get("call_sources", set())):
            if peer_id in result:
                result[peer_id].setdefault("call_sources", set()).add(source_id)
    cutoff = scope.get("cutoff")
    versions = [version for version in scope.get("versions", [])
                if isinstance(version, dict) and isinstance(version.get("id"), str)
                and (not isinstance(cutoff, int)
                     or not isinstance(version.get("observed_at"), int)
                     or version.get("observed_at") <= cutoff)]
    versions_by_id = {version["id"]: version for version in versions}

    def lineage(version_id, visiting=None):
        if not isinstance(version_id, str):
            return set()
        visiting = set(visiting or ())
        if version_id in visiting:
            return set()
        visiting.add(version_id)
        version = versions_by_id.get(version_id)
        if not version:
            return {version_id}
        result_ids = {version_id}
        previous = version.get("previous")
        if previous:
            result_ids.update(lineage(previous, visiting))
        return result_ids

    for version in versions:
        if isinstance(version, dict) and isinstance(version.get("id"), str):
            source_info = result.get(version.get("source"), {})
            add_source(version["id"], {
                "id": version["id"],
                "order": version.get("observed_at"),
                "stage_id": version.get("stage_id") or source_info.get("stage_id"),
                "paths": {version["path"]} if isinstance(version.get("path"), str) else set(),
                "text": version.get("path", ""),
                "kinds": {"version"},
                "source_kinds": {source_kind_for(version, default="code")},
                # Keep ancestry on the provenance record.  A fact need not
                # repeat "旧版/新版" in its wording for the index to know
                # that two cited versions are a real transition.
                "previous": version.get("previous"),
                "historical_transition": bool(version.get("previous")),
                "lineage": lineage(version["id"]),
                "version_ids": {version["id"]},
            })
            if version.get("previous") and isinstance(version.get("source"), str):
                source = result.setdefault(version["source"], {})
                source["historical_transition"] = True
                source.setdefault("kinds", set()).add("version_transition")
            if isinstance(version.get("source"), str):
                source = result.setdefault(version["source"], {})
                source.setdefault("lineage", set()).update(lineage(version["id"]))
                source.setdefault("version_ids", set()).add(version["id"])
    # A version's ``previous`` chain is explicit evidence, unlike a mere
    # shared filename. Include both ancestors and descendants (plus their
    # producing records) so a guard seeded from an old version sees later
    # updates before the cutoff. Sibling branches remain unrelated.
    for version in versions:
        chain_ids = lineage(version["id"])
        chain_sources = {
            versions_by_id[item].get("source") for item in chain_ids
            if item in versions_by_id and isinstance(versions_by_id[item].get("source"), str)
        }
        connected = chain_ids | chain_sources
        for source_id in connected:
            result.setdefault(source_id, {}).setdefault("version_sources", set()).update(
                connected - {source_id})
    return result


def _graph_path_links(scope):
    """Return one-hop file links explicitly present in the authoring graph."""
    links = set()
    for edge in scope.get("edges", []) + scope.get("historical_edges", []):
        if not isinstance(edge, dict):
            continue
        left = edge.get("from")
        right = edge.get("to")
        if not isinstance(left, str) or not isinstance(right, str):
            continue
        left_path, right_path = left.split("::", 1)[0], right.split("::", 1)[0]
        if left_path != right_path:
            links.add(frozenset((left_path, right_path)))
    return links


def _fact_info(fact, source_index, graph_links=None):
    sources = [source_index.get(source, {}) for source in fact.get("sources", [])]
    statement = fact.get("statement", "")
    source_text = "\n".join(source.get("text", "")[:1000] for source in sources)
    source_kinds = set().union(*(source.get("kinds", set()) for source in sources)) if sources else set()
    labels = _labels(statement + "\n" + source_text)
    statement_labels = _labels(statement)
    # Source type and version ancestry are structural evidence.  They may add
    # a transition/validation label, but never create a relation by themselves.
    if "patch" in source_kinds or "version_transition" in source_kinds:
        labels.add("change")
    if "version" in source_kinds:
        labels.add("temporal")
    graph_links = graph_links or set()
    orders = sorted({source.get("order") for source in sources
                     if isinstance(source.get("order"), int)})
    lineage = set().union(*(source.get("lineage", set()) for source in sources)) \
        if sources else set()
    version_ids = set().union(*(source.get("version_ids", set()) for source in sources)) \
        if sources else set()
    statement_paths = {token for token in _IDENTIFIER.findall(statement)
                       if token.casefold().endswith(_CODE_FILE_SUFFIXES)}
    info = {
        "fact": fact,
        "entities": _entities(statement),
        "labels": labels,
        "statement_labels": statement_labels,
        "source_kinds": source_kinds,
        # A fact extracted from a result/tool record can state an observed
        # failure without using the exact "运行失败" wording.  Keep requests
        # such as "不要报错" out of this signal so they do not become
        # failure-diagnosis roots by themselves.
        "observed_failure": bool(
            (_OBSERVED_FAILURE.search(statement + "\n" + source_text)
             or ("failure" in _labels(statement)
                 and "result" in source_kinds
                 and not _FAILURE_NEGATION.search(statement)
                 and not _FAILURE_ABSENCE.search(statement)))
        ),
        "paths": ((set().union(*(source.get("paths", set()) for source in sources))
                   if sources else set()) | statement_paths),
        "stages": {source.get("stage_id") for source in sources if source.get("stage_id")},
        "orders": orders,
        "lineage": lineage,
        "version_ids": version_ids,
        "version_sources": set().union(*(source.get("version_sources", set())
                                         for source in sources)) if sources else set(),
        "historical_transition": any(source.get("historical_transition")
                                      for source in sources),
        "call_sources": set().union(*(source.get("call_sources", set())
                                      for source in sources)) if sources else set(),
        "external_cue": bool(_EXTERNAL_KNOWLEDGE_CUE.search(statement + "\n" + source_text)),
    }
    # Expand each cited path only through an explicit graph edge.  This gives
    # cross-file call/reference evidence a chance to form a group while still
    # avoiding semantic guesses from merely similar wording.
    info["graph_neighbors"] = set()
    for path in info["paths"]:
        for link in graph_links:
            if path in link:
                info["graph_neighbors"].update(link - {path})
    return info


def _explicit_graph_link(left, right):
    """Return whether two facts cite paths joined by an explicit graph edge."""
    return bool(left.get("graph_neighbors", set()) & right["paths"]
                or right.get("graph_neighbors", set()) & left["paths"])


def _version_ancestry_link(left, right):
    """Return a concrete version transition shared by two facts.

    A source marked as a transition is not enough: two unrelated branches can
    both contain patches to the same filename, and two symbols in one file can
    be unrelated. Require distinct cited version points, a common ancestor,
    and a shared concrete semantic object. An explicit graph edge is handled
    separately by ``_explicit_graph_link``.
    """
    left_versions = set(left.get("version_ids", set()))
    right_versions = set(right.get("version_ids", set()))
    if not left_versions or not right_versions or left_versions == right_versions:
        return False
    # A shared root is not enough: sibling versions (v2 <- v1 and v3 <- v1)
    # are alternative branches, not a chronological transition. One cited
    # version must be an ancestor of the other.
    left_lineage = set(left.get("lineage", set()))
    right_lineage = set(right.get("lineage", set()))
    if not ((left_versions & right_lineage) or (right_versions & left_lineage)):
        return False
    return bool(_semantic_shared_entities(left, right))


def _call_result_link(left, right):
    """Return a normalized call/result edge in either direction."""
    left_sources = set(left["fact"].get("sources", []))
    right_sources = set(right["fact"].get("sources", []))
    return bool(left.get("call_sources", set()) & right_sources
                or right.get("call_sources", set()) & left_sources)


def _explicit_update_link(left, right):
    """Link a later update/validation only through one shared concrete object."""
    if not _semantic_shared_entities(left, right):
        return False
    left_orders, right_orders = left.get("orders", []), right.get("orders", [])
    if not left_orders or not right_orders or max(left_orders) == max(right_orders):
        return False
    later = right if max(right_orders) > max(left_orders) else left
    earlier = left if later is right else right
    later_signal = set(later.get("statement_labels", set())) & {
        "change", "validation", "feedback", "decision", "temporal",
    }
    earlier_signal = set(earlier.get("labels", set())) & _RELATION_LABELS
    return bool(later_signal and earlier_signal)


def _guard_link(left, right, allow_graph=True):
    """Use only explicit structural or same-object update relationships."""
    if (_version_ancestry_link(left, right)
            or _call_result_link(left, right)
            or (allow_graph and _explicit_graph_link(left, right))):
        return True
    if not _explicit_update_link(left, right):
        return False
    left_paths = set(left.get("paths", set()))
    right_paths = set(right.get("paths", set()))
    left_qualified = set(_QUALIFIED_OBJECT.findall(
        left.get("fact", {}).get("statement", "")))
    right_qualified = set(_QUALIFIED_OBJECT.findall(
        right.get("fact", {}).get("statement", "")))
    # When both facts identify code paths, a generic shared command or test
    # result must not move the guard to another file.  Pathless feedback can
    # still bind through the existing concrete-entity rules.
    if left_paths and right_paths:
        return bool(_matching_code_paths(left_paths, right_paths)
                    or left_qualified & right_qualified)
    if left_qualified & right_qualified:
        return True
    shared = _semantic_shared_entities(left, right)
    if len(shared) >= 2:
        return True
    # One common property without a file/symbol binding is too weak across a
    # long session. Retain it only when the text explicitly corrects an older
    # statement; ordinary validation requirements are not counterevidence.
    return bool(shared and (_COUNTEREVIDENCE_CUE.search(
        left.get("guard_statement", "")) or _COUNTEREVIDENCE_CUE.search(
            right.get("guard_statement", ""))))


def _review_guard_closure(base_infos, all_infos):
    """Add evidence linked to the original objects without bridge expansion.

    Structural version and call sources remain closed by ``_guard_sources``.
    Facts added here may provide a later same-object revision or rebuttal, but
    they never become new anchors that can carry the guard to another object.
    """
    selected = list(base_infos)
    selected_ids = {info["fact"]["id"] for info in selected}
    base_anchors = tuple(base_infos)
    dependency_objects = set().union(*(
        set(anchor.get("graph_neighbors", set())) for anchor in base_anchors
    )) if base_anchors else set()
    dependency_anchors = tuple(
        candidate for candidate in all_infos
        if candidate["fact"]["id"] not in selected_ids
        and dependency_objects & set(candidate.get("paths", set()))
        and any(_explicit_graph_link(anchor, candidate)
                for anchor in base_anchors)
    )
    for candidate in dependency_anchors:
        selected.append(candidate)
        selected_ids.add(candidate["fact"]["id"])
    for candidate in all_infos:
        if candidate["fact"]["id"] in selected_ids:
            continue
        if (any(_guard_link(anchor, candidate) for anchor in base_anchors)
                or (dependency_objects & set(candidate.get("paths", set()))
                    and any(_guard_link(anchor, candidate, allow_graph=False)
                            for anchor in dependency_anchors))):
            selected.append(candidate)
            selected_ids.add(candidate["fact"]["id"])
    selected.sort(key=lambda info: (
        min(info.get("orders", [10 ** 18])) if info.get("orders") else 10 ** 18,
        info["fact"]["id"],
    ))
    return selected


def _fact_sources(infos):
    return {source for info in infos
            for source in info["fact"].get("sources", [])
            if isinstance(source, str)}


def _guard_sources(infos):
    sources = _fact_sources(infos)
    sources.update(source for info in infos
                   for source in info.get("call_sources", set())
                   if isinstance(source, str))
    sources.update(source for info in infos
                   for source in info.get("version_sources", set())
                   if isinstance(source, str))
    return sources


def _candidate_text(candidate):
    values = [candidate.get("question", "")]
    for key in ("answer_points", "forbidden_points"):
        values.extend(point.get("text", "") for point in candidate.get(key, [])
                      if isinstance(point, dict))
    return "\n".join(value for value in values if isinstance(value, str))


def _candidate_concrete_entities(text):
    """Keep explicit symbol/property names, not every word in QA prose."""
    entities = _entities(text)
    concrete = {item for item in entities if item.startswith("zhx:")}
    explicit, named = set(), set()
    for match in _IDENTIFIER.finditer(text or ""):
        token = match.group(0)
        normalized = token.casefold()
        if normalized not in entities:
            continue
        before = text[match.start() - 1] if match.start() else ""
        after = text[match.end()] if match.end() < len(text) else ""
        # ``3_600`` must not yield ``_600`` and a hyphenated package name
        # must not yield arbitrary capitalized fragments such as ``SWE``.
        if before.isdigit() or before == "-" or after == "-":
            continue
        if any(mark in token for mark in ("_", ".", "/", "-")):
            explicit.add(normalized)
        elif any(char.isupper() for char in token[1:]) or token.isupper():
            named.add(normalized)
    for match in re.finditer(r"`([^`\n]{1,120})`", text or ""):
        explicit.update(_entities(match.group(1)))
    if explicit:
        named = {item for item in named
                 if not item.endswith(("error", "exception"))
                 and not (len(item) <= 4 and item.isalpha())}
    return concrete | explicit | named


def _matching_code_paths(left_paths, right_paths):
    """Match a project-relative path with the same basename/full suffix."""
    matches = set()
    for left in left_paths:
        for right in right_paths:
            if (left == right or left.endswith("/" + right)
                    or right.endswith("/" + left)):
                matches.update((left, right))
    return matches


def _statement_guard_view(info, graph_links, allowed_objects=None):
    """Remove source-wide path metadata from one candidate guard anchor."""
    statement = info.get("fact", {}).get("statement", "")
    entities = _entities(statement)
    paths = set(_CODE_PATH_OBJECT.findall(statement))
    source_paths = set(info.get("paths", set()))
    # A single-file source can safely disambiguate the same property name in
    # another module. Multi-file source metadata remains excluded because it
    # cannot bind a fact to one of those files.
    source_basenames = {path.rsplit("/", 1)[-1] for path in source_paths}
    if not paths and len(source_basenames) == 1:
        paths = source_paths
    qualified = set(_QUALIFIED_OBJECT.findall(statement))
    if allowed_objects is not None:
        entities &= allowed_objects
        matched_paths = paths & allowed_objects
        matched_qualified = qualified & allowed_objects
        # A statement containing one path binds its matched symbol to that
        # path. Multiple paths are never cross-paired merely because the raw
        # source was one patch record.
        paths = (paths if entities and len(paths) == 1 else matched_paths)
        qualified = matched_qualified
    view = dict(info)
    view["guard_statement"] = statement
    view["entities"] = entities
    view["paths"] = paths
    view["fact"] = dict(info.get("fact", {}), statement="\n".join(sorted(qualified)))
    view["graph_neighbors"] = set()
    for path in paths:
        for link in graph_links:
            if path in link:
                view["graph_neighbors"].update(link - {path})
    return view


def _candidate_sources(candidate):
    return {
        source for key in ("answer_points", "forbidden_points")
        for point in candidate.get(key, []) if isinstance(point, dict)
        for source in point.get("sources", []) if isinstance(source, str)
    }


def _candidate_guard_closure(base_infos, all_infos, historical_event=False):
    """Close a candidate guard without turning later history into current truth."""
    if not historical_event:
        return _review_guard_closure(base_infos, all_infos)
    selected = list(base_infos)
    selected_ids = {info["fact"]["id"] for info in selected}
    for candidate in all_infos:
        if candidate["fact"]["id"] in selected_ids:
            continue
        structural = any(
            _version_ancestry_link(anchor, candidate)
            or _call_result_link(anchor, candidate)
            or _explicit_graph_link(anchor, candidate)
            for anchor in base_infos)
        labels = set(candidate.get("statement_labels", set()))
        correction = bool(
            labels & {"feedback", "decision"}
            or _COUNTEREVIDENCE_CUE.search(
                candidate.get("guard_statement", "")))
        if structural or (correction and any(
                _guard_link(anchor, candidate, allow_graph=False)
                for anchor in base_infos)):
            selected.append(candidate)
            selected_ids.add(candidate["fact"]["id"])
    selected.sort(key=lambda info: (
        min(info.get("orders", [10 ** 18])) if info.get("orders") else 10 ** 18,
        info["fact"]["id"],
    ))
    return selected


def candidate_review_projection(group, evidence_index, candidate,
                                target_chars=16000, max_chars=None):
    """Build a bounded counterevidence scope from one candidate's real anchors.

    Candidate prose and cited facts define fixed objects. Raw source-wide path
    metadata never becomes an anchor, while explicit call/result, version,
    update, dependency, and feedback links remain available before the frozen
    cutoff.
    """
    scope = group.get("scope", {})
    max_chars = max_chars or scope.get(
        "model_request_chars", scope.get("max_context_chars", 32000))
    audit = {
        "complete": False,
        "reason": "candidate_guard_unavailable",
        "base_fact_ids": [],
        "anchor_objects": [],
        "guard_source_count": 0,
        "request_chars": None,
    }
    if not isinstance(candidate, dict) or not isinstance(evidence_index, dict):
        return None, audit
    if candidate.get("type") == "adversarial":
        audit["reason"] = "full_range_required"
        return None, audit
    cited_sources = _candidate_sources(candidate)
    unresolved = sorted(source for source in cited_sources
                        if not _source_material_ids(
                            evidence_index.get("universe", {}), source))
    if unresolved:
        audit.update(reason="unresolved_candidate_source",
                     unresolved_source_ids=unresolved)
        return None, audit

    text = _candidate_text(candidate)
    candidate_view = {
        # General dialogue questions often name a decision object with an
        # ordinary noun (for example ``receipt``), while code questions need
        # the stricter path/symbol filter to avoid linking a whole project by
        # prose words alone.
        "entities": (_entities(text) if group.get("qa_mode") == "general"
                     else _candidate_concrete_entities(text)),
        "paths": set(_CODE_PATH_OBJECT.findall(text)),
        "statement_labels": _labels(text),
        "fact": {"statement": text},
    }
    candidate_qualified = set(_QUALIFIED_OBJECT.findall(text))
    aliases = evidence_index.get("fact_aliases", {})
    group_fact_ids = {
        aliases.get(fact.get("id"), fact.get("id"))
        for fact in group.get("facts", []) if isinstance(fact, dict)
    }
    cutoff = scope.get("cutoff")

    def source_before_cutoff(source):
        order = _source_order(evidence_index.get("source_index", {}), source)
        return not isinstance(cutoff, int) or not isinstance(order, int) or order <= cutoff

    after_cutoff = sorted(source for source in cited_sources
                          if not source_before_cutoff(source))
    if after_cutoff:
        audit.update(reason="candidate_source_after_cutoff",
                     unresolved_source_ids=after_cutoff)
        return None, audit

    def at_cutoff(info):
        sources = [source for source in info.get("fact", {}).get("sources", [])
                   if isinstance(source, str) and source_before_cutoff(source)]
        if not sources:
            return None
        filtered = dict(info)
        filtered["fact"] = dict(info["fact"], sources=sources)
        filtered["orders"] = [order for order in info.get("orders", [])
                              if not isinstance(cutoff, int) or order <= cutoff]
        filtered["call_sources"] = {
            source for source in info.get("call_sources", set())
            if source_before_cutoff(source)
        }
        filtered["version_sources"] = {
            source for source in info.get("version_sources", set())
            if source_before_cutoff(source)
        }
        return filtered

    available_infos = [filtered for info in evidence_index.get("infos", [])
                       for filtered in [at_cutoff(info)] if filtered is not None]
    base, anchor_objects = [], set()
    for info in available_infos:
        fact_id = aliases.get(info.get("fact", {}).get("id"),
                              info.get("fact", {}).get("id"))
        if fact_id not in group_fact_ids:
            continue
        statement_view = _statement_guard_view(
            info, evidence_index.get("graph_links", set()))
        shared = _semantic_shared_entities(candidate_view, statement_view)
        shared |= _matching_code_paths(
            candidate_view["paths"], statement_view["paths"])
        shared |= candidate_qualified & set(_QUALIFIED_OBJECT.findall(
            info.get("fact", {}).get("statement", "")))
        fact_sources = set(info.get("fact", {}).get("sources", []))
        if not shared and not (cited_sources & fact_sources):
            continue
        # A citation can seed structural call/version closure, but it cannot
        # turn unrelated objects co-recorded in the same source into anchors.
        anchor_objects.update(shared)
        base.append((info, shared))
    if not base:
        audit["reason"] = "no_candidate_fact_anchor"
        return None, audit
    if not anchor_objects:
        audit["reason"] = "no_concrete_candidate_object"
        return None, audit

    base_views = [_statement_guard_view(
        info, evidence_index.get("graph_links", set()), anchor_objects)
        for info, _ in base]
    all_views = [_statement_guard_view(
        info, evidence_index.get("graph_links", set()))
        for info in available_infos]
    historical_event = bool(
        candidate.get("type") == "failure_diagnosis"
        and _HISTORICAL_EVENT_QUESTION.search(candidate.get("question", "")))
    guarded_infos = _candidate_guard_closure(
        base_views, all_views, historical_event=historical_event)
    guard_sources = _guard_sources(guarded_infos) | cited_sources
    guard_sources.update(source for source in scope.get(
        "generation_extra_sources", []) if isinstance(source, str))
    if isinstance(cutoff, int):
        guard_sources = {
            source for source in guard_sources
            if not isinstance(_source_order(
                evidence_index.get("source_index", {}), source), int)
            or _source_order(evidence_index.get("source_index", {}), source) <= cutoff
        }
    base_infos = [
        info for info in available_infos
        if aliases.get(info["fact"].get("id"), info["fact"].get("id"))
        in group_fact_ids
        and set(info["fact"].get("sources", [])) & guard_sources
    ]
    audit.update(
        base_fact_ids=sorted({info["fact"]["id"] for info, _ in base}),
        projected_fact_ids=sorted({info["fact"]["id"] for info in base_infos}),
        anchor_objects=sorted(anchor_objects),
        guard_source_count=len(guard_sources),
        required_source_ids=sorted(guard_sources),
    )
    unresolved_guard = sorted(source for source in guard_sources
                              if not _source_material_ids(
                                  evidence_index["universe"], source))
    if unresolved_guard:
        audit.update(reason="unresolved_guard_source",
                     unresolved_source_ids=unresolved_guard)
        return None, audit

    from .llm import evidence_projection, simple_evidence_request_size
    projected = evidence_projection(
        evidence_index["universe"], guard_sources,
        padding_records=0, max_chars=max_chars)
    projected["cutoff"] = scope.get("cutoff", projected.get("cutoff"))
    projected["review_guard_sources"] = sorted(guard_sources)
    projected["review_guard_complete"] = True
    projected["review_guard_reason"] = "candidate_scoped_complete"
    projected["review_guard_omitted_count"] = 0
    projected["generation_extra_sources"] = sorted({
        source for source in scope.get("generation_extra_sources", [])
        if isinstance(source, str)
    })
    projected["evidence_group"] = dict(scope.get("evidence_group", {}))
    request_chars = simple_evidence_request_size(
        projected, guard_sources, group.get("facts", []), candidate)
    audit["request_chars"] = request_chars
    if request_chars > max_chars:
        audit["reason"] = "candidate_guard_over_budget"
        return None, audit
    projected["model_request_chars"] = max_chars
    projected["max_context_chars"] = max_chars
    projected["review_request_chars"] = request_chars
    projected["over_budget"] = False
    projected_group = dict(
        group, scope=projected,
        review_guard_sources=projected["review_guard_sources"],
        review_guard_complete=True, request_chars=max_chars)
    audit.update(complete=True, reason="candidate_scoped_complete",
                 request_chars=request_chars)
    return projected_group, audit


def _relation(left, right, qa_mode):
    shared_paths = left["paths"] & right["paths"]
    shared_entities = _semantic_shared_entities(left, right)
    shared_sources = set(left["fact"].get("sources", [])) & set(right["fact"].get("sources", []))
    graph_linked = _explicit_graph_link(left, right)
    version_link = _version_ancestry_link(left, right)
    score = min(len(shared_entities), 3) * 3 + min(len(shared_paths), 2) * 2 \
        + len(shared_sources) * 6 + (5 if graph_linked else 0)
    if qa_mode == "general" and left["stages"] != right["stages"] and shared_entities:
        score += 3
    labels = left["labels"] | right["labels"]
    if "failure" in labels and labels & {"change", "validation", "decision"}:
        score += 3
    if labels & {"change", "temporal"} and (shared_paths or shared_entities):
        score += 2
    complementary = any(item["observed_failure"] for item in (left, right)) \
        and bool(labels & {"change", "validation"})
    # A filename alone is too weak: unrelated concerns often live in one
    # module.  Keep path-only links for an observed failure/change/validation
    # chain, where the shared path is the concrete object being repaired.
    path_link = bool(shared_paths and (complementary or version_link or (
        qa_mode == "general" and bool(labels & {"change", "temporal", "feedback"}))))
    # A common source is useful only when the facts also mention a concrete
    # object or describe complementary stages of one event.
    source_link = bool(shared_sources and (shared_entities or complementary))
    linked = bool(shared_entities or source_link or path_link or graph_linked)
    return score, linked


def _candidate_types(infos, qa_mode, allowed_types):
    stages = set().union(*(info["stages"] for info in infos))
    labels = set().union(*(info["labels"] for info in infos))
    entity_sets = [info["entities"] for info in infos]
    path_sets = [info["paths"] for info in infos]
    shared_entities = (_shared_entities_for_infos(infos)
                       if len(entity_sets) > 1 else set())
    shared_paths = set.intersection(*path_sets) if len(path_sets) > 1 else set()
    graph_linked = any(
        info.get("graph_neighbors", set()) & other["paths"]
        for index, info in enumerate(infos)
        for other in infos[index + 1:]
    )
    shared = shared_entities or shared_paths or graph_linked
    order_values = [order for info in infos for order in info["orders"]]
    source_ids = {source for info in infos for source in info["fact"].get("sources", [])}
    distinct_time = len(set(order_values)) >= 2
    version_link = any(
        _version_ancestry_link(left, right)
        for index, left in enumerate(infos)
        for right in infos[index + 1:]
    )
    explicit_temporal = bool(set().union(*(
        info.get("statement_labels", set()) for info in infos)) &
        {"temporal", "feedback"})
    if qa_mode == "general":
        possible = set()
        if len(stages) <= 1:
            possible.add("single-hop")
        if len(stages) >= 2 and shared:
            possible.add("multi-hop")
        if ((len(set(order_values)) >= 2 and shared) or "temporal" in labels) and len(infos) >= 2:
            possible.add("temporal")
        if any(info.get("external_cue") for info in infos):
            possible.add("open-domain")
    else:
        possible = {"fact_recall"} if len(infos) == 1 else set()
        # History tracking must cross an actual recorded point in time.  Two
        # independent facts extracted from one version are behavior/context
        # material, not a historical transition, even when they share a path.
        # ``labels`` includes source-derived markers (for example, every
        # version source receives ``temporal``). Use statement-level cues or a
        # concrete ancestry link so an unrelated same-file pair cannot pass
        # merely because it came from versioned records.
        history_signal = bool(explicit_temporal
                              or version_link
                              or (shared_entities and "change" in labels)
                              or (graph_linked and "change" in labels))
        # A shared filename is a useful index hint but not a semantic link:
        # separate functions in one file can evolve independently. Require a
        # concrete entity or an explicit graph/ancestry relation before a
        # history question is eligible.
        history_relation = bool(shared_entities or graph_linked or version_link)
        # A single fact may still be a valid historical question when it is
        # explicitly tied to a recorded patch/version transition.  The
        # current snapshot cannot answer “what did this old version contain?”
        # without that source, so do not force an unrelated second fact merely
        # to manufacture a multi-hop group.
        historical_single = (
            len(infos) == 1
            and (bool(_INLINE_TRANSITION.search(
                infos[0]["fact"].get("statement", "")))
                 or (bool(infos[0].get("historical_transition"))
                     and bool(infos[0].get("statement_labels", set())
                              & {"change", "temporal"})))
        )
        if ((len(infos) >= 2 and len(source_ids) >= 2 and distinct_time
             and history_relation and history_signal)
                or historical_single):
            possible.add("history_tracking")
        inline_behavior = (len(infos) == 1
                           and "conditional" in infos[0].get("statement_labels", set())
                           and _BEHAVIOR_OUTCOME.search(
                               infos[0]["fact"].get("statement", "")))
        if ((shared_entities or graph_linked) and labels & {"change", "conditional"}) \
                or inline_behavior:
            possible.add("behavior_inference")
        # A failure diagnosis must connect an observed failure to a separate
        # change, validation, or decision fact.  A lone statement describing a
        # validator/error is current behavior, not a historical diagnosis.
        has_failure_chain = (
            len(infos) >= 2
            and any(info.get("observed_failure") for info in infos)
            and labels & {"change", "validation", "decision"}
        )
        inline_failure_chain = (len(infos) == 1
                                and infos[0].get("observed_failure")
                                and bool(infos[0].get("statement_labels", set())
                                         & {"change", "validation", "decision", "feedback"}))
        if has_failure_chain or inline_failure_chain:
            possible.add("failure_diagnosis")
    return possible & set(allowed_types)


def _project_group(universe, infos, target_chars, max_chars, full_range=False,
                   required_sources=(), include_padding=True):
    # Import locally to keep the low-level projection reusable without a module cycle.
    from .llm import evidence_projection

    facts = [info["fact"] for info in infos]
    sources = ({source for fact in facts for source in fact.get("sources", [])}
               | {source for source in required_sources if isinstance(source, str)})
    if full_range:
        # Adversarial review needs the complete selected dialogue range, but it
        # still does not need adaptive-search bookkeeping or a duplicate graph.
        # Use the same source projection path as other stages with an explicit
        # full-range flag so the request surface stays predictable.
        projected = evidence_projection(universe, sources, padding_records=0,
                                        max_chars=max_chars, full_range=True)
        payload_chars = _size({"scope": projected, "facts": facts})
        if not universe.get("full_range_covered") or payload_chars + 4500 > max_chars:
            return None, None, None
        request_chars = target_chars if payload_chars + 4500 <= target_chars else max_chars
        projected["model_request_chars"] = request_chars
        projected["max_context_chars"] = request_chars
        projected["over_budget"] = False
        projected["full_range_required"] = True
        return projected, request_chars, 0
    for padding in ((1, 0) if include_padding else (0,)):
        projected = evidence_projection(universe, sources, padding_records=padding,
                                        max_chars=max_chars)
        payload_chars = _size({"scope": projected, "facts": facts})
        reserve = 4500
        if payload_chars + reserve <= target_chars:
            request_chars = target_chars
        elif payload_chars + reserve <= max_chars:
            request_chars = max_chars
        else:
            continue
        projected["model_request_chars"] = request_chars
        projected["max_context_chars"] = request_chars
        projected["over_budget"] = False
        return projected, request_chars, padding
    return None, None, None


def build_evidence_index(facts, scopes, qa_mode, model_request_chars=32000):
    """Build one reusable, in-memory index for grouping and one-shot expansion.

    The returned object is intentionally not embedded in evidence groups: it
    contains the merged source universe and normalized sets used by the static
    linker.  Groups receive only a small JSON-safe expansion pointer.
    """
    if qa_mode not in {"general", "code"}:
        raise ValueError("qa_mode must be general or code")
    facts, fact_aliases, merged_fact_ids = _merge_duplicate_facts(facts)
    universe = merge_scopes(scopes, qa_mode, model_request_chars)
    source_index = _source_index(universe)
    graph_links = _graph_path_links(universe)
    infos = [_fact_info(fact, source_index, graph_links) for fact in facts]
    # A tool result may contain one file while individual extracted facts omit
    # its path. If every fact from that source names the same basename, bind
    # the pathless siblings to that file so a property name cannot connect to
    # an unrelated module. Multi-file sources remain unbound.
    paths_by_source = {}
    for info in infos:
        for source in info["fact"].get("sources", []):
            paths_by_source.setdefault(source, set()).update(info.get("paths", set()))
    for info in infos:
        if info.get("paths"):
            continue
        inferred = set().union(*(
            paths_by_source.get(source, set())
            for source in info["fact"].get("sources", [])))
        code_paths = {path for path in inferred
                      if path.casefold().endswith(
                          (".py", ".js", ".mjs", ".ts", ".tsx", ".jsx"))}
        if len({path.rsplit("/", 1)[-1] for path in code_paths}) != 1:
            continue
        info["paths"] = code_paths
        for path in code_paths:
            for link in graph_links:
                if path in link:
                    info["graph_neighbors"].update(link - {path})
    index = {
        "qa_mode": qa_mode,
        "universe": universe,
        "source_index": source_index,
        "graph_links": graph_links,
        "infos": infos,
        "info_by_id": {info["fact"]["id"]: info for info in infos},
        "fact_aliases": fact_aliases,
        "merged_fact_ids": merged_fact_ids,
        "skipped_types": [],
        "eligible_skips": [],
    }
    for alias, canonical in fact_aliases.items():
        if canonical in index["info_by_id"]:
            index["info_by_id"].setdefault(alias, index["info_by_id"][canonical])
    index["direct_graph"] = _build_direct_evidence_graph(index)
    index["expansion_candidates"] = _build_expansion_candidates(index)
    return index


def _build_direct_evidence_graph(index):
    """Build direct 0/1 evidence edges; never flatten transitive closures."""
    graph = {}

    def link(left, right, weight, relation):
        if not isinstance(left, str) or not isinstance(right, str) or left == right:
            return
        graph.setdefault(left, []).append((right, weight, relation))
        graph.setdefault(right, []).append((left, weight, relation))

    universe = index["universe"]
    source_index = index["source_index"]
    infos = index["infos"]
    for edge in universe.get("edges", []) + universe.get("historical_edges", []):
        if not isinstance(edge, dict):
            continue
        left, right = edge.get("from"), edge.get("to")
        if isinstance(left, str) and isinstance(right, str):
            link(left, right, 1, edge.get("relation", edge.get("kind", "graph")))
    for info in infos:
        for source in info["fact"].get("sources", []):
            if isinstance(source, str):
                graph.setdefault(source, [])
    for record in universe.get("dialogue", []):
        if isinstance(record, dict) and isinstance(record.get("parent_id"), str):
            link(record.get("id"), record["parent_id"], 0, "fragment")
    versions = {version.get("id"): version for version in universe.get("versions", [])
                if isinstance(version, dict) and isinstance(version.get("id"), str)}
    for version_id, version in versions.items():
        link(version_id, version.get("source"), 0, "version_source")
        if version.get("previous") in versions:
            link(version_id, version["previous"], 1, "version_previous")
    seen_call_pairs = set()
    for source_id, source in source_index.items():
        for peer in source.get("call_sources", set()):
            marker = tuple(sorted((source_id, peer)))
            if marker not in seen_call_pairs:
                seen_call_pairs.add(marker)
                link(source_id, peer, 1, "call_result")
    for left_index, left in enumerate(infos):
        left_sources = set(left["fact"].get("sources", []))
        for right in infos[left_index + 1:]:
            right_sources = set(right["fact"].get("sources", []))
            graph_link = _explicit_graph_link(left, right)
            if graph_link:
                for left_source in left_sources:
                    for right_source in right_sources:
                        link(left_source, right_source, 1, "graph")
            explicitly_co_recorded = bool(left_sources & right_sources)
            if not (graph_link or explicitly_co_recorded):
                continue
            # A shared record proves the two extracted facts were co-recorded,
            # but it does not make every other citation on the left equivalent
            # to every other citation on the right.  Only an explicit graph
            # edge may add a direct cross-source reasoning hop here.
            if not graph_link:
                continue
            if "feedback" in (left.get("statement_labels", set())
                               | right.get("statement_labels", set())):
                for left_source in left_sources:
                    for right_source in right_sources:
                        if left_source != right_source:
                            link(left_source, right_source, 1, "feedback")
            if "validation" in (left.get("statement_labels", set())
                                 | right.get("statement_labels", set())):
                for left_source in left_sources:
                    for right_source in right_sources:
                        if left_source != right_source:
                            link(left_source, right_source, 1, "test")
    for node in graph:
        graph[node] = sorted(set(graph[node]), key=lambda item: (item[1], item[0], item[2]))
    return graph


def _shortest_relation_path(graph, start, target):
    if start == target:
        return {"nodes": [start], "relations": [], "distance": 0}
    queue = deque([start])
    distances = {start: 0}
    previous = {}
    while queue:
        node = queue.popleft()
        for neighbor, weight, relation in graph.get(node, []):
            distance = distances[node] + weight
            if neighbor in distances and distances[neighbor] <= distance:
                continue
            distances[neighbor] = distance
            previous[neighbor] = (node, relation, weight)
            if weight == 0:
                queue.appendleft(neighbor)
            else:
                queue.append(neighbor)
    if target not in distances:
        return None
    nodes, relations = [target], []
    cursor = target
    while cursor != start:
        parent, relation, weight = previous[cursor]
        relations.append({"from": parent, "to": cursor,
                          "relation": relation, "distance": weight})
        nodes.append(parent)
        cursor = parent
    nodes.reverse()
    relations.reverse()
    return {"nodes": nodes, "relations": relations,
            "distance": distances[target]}


def _shortest_from_sources(graph, sources, target):
    paths = [_shortest_relation_path(graph, source, target)
             for source in sources if isinstance(source, str)]
    paths = [path for path in paths if path is not None]
    return min(paths, key=lambda path: (path["distance"], path["nodes"])) if paths else None


def _relation_metadata(infos, evidence_index, seed_node=None, extra_nodes=()):
    ordered = sorted(infos, key=lambda info: (
        min(info.get("orders", [10 ** 18])) if info.get("orders") else 10 ** 18,
        info["fact"]["id"]))
    if not ordered and not seed_node:
        return {"seed_node": None, "seed_fact_id": None,
                "necessary_nodes": [], "paths": [],
                "max_distance": None, "path_complete": False,
                "difficulty": "unknown"}
    seed_fact_id = ordered[0]["fact"]["id"] if ordered else None
    source_index = evidence_index["source_index"]
    seed_sources = [source for source in ordered[0]["fact"].get("sources", [])
                    if isinstance(source, str)] if ordered else []
    seed_sources.sort(key=lambda source: (
        _source_order(source_index, source)
        if _source_order(source_index, source) is not None else 10 ** 18,
        source))
    seed = seed_node or (seed_sources[0] if seed_sources else None)
    necessary = []
    for info in ordered:
        for source in info["fact"].get("sources", []):
            if isinstance(source, str) and source not in necessary:
                necessary.append(source)
    necessary.extend(node for node in extra_nodes
                     if isinstance(node, str) and node not in necessary)
    if seed is None:
        return {"seed_node": None, "seed_fact_id": seed_fact_id,
                "necessary_nodes": necessary, "paths": [],
                "max_distance": None, "path_complete": False,
                "difficulty": "unknown"}
    if seed not in evidence_index.get("direct_graph", {}):
        return {"seed_node": seed, "seed_fact_id": seed_fact_id,
                "necessary_nodes": necessary, "paths": [],
                "max_distance": None, "path_complete": False,
                "difficulty": "unknown"}
    paths, distances, complete = [], [], True
    for node in necessary:
        path = _shortest_relation_path(evidence_index["direct_graph"], seed, node)
        if path is None:
            complete = False
            paths.append({"target": node, "distance": None,
                          "nodes": [], "relations": []})
        else:
            paths.append(dict(path, target=node))
            distances.append(path["distance"])
    maximum = max(distances) if complete and distances else (0 if complete else None)
    difficulty = ("unknown" if maximum is None else
                  "easy" if maximum <= 1 else
                  "medium" if maximum == 2 else "hard")
    return {"seed_node": seed, "seed_fact_id": seed_fact_id,
            "necessary_nodes": necessary, "paths": paths,
            "max_distance": maximum, "path_complete": complete,
            "difficulty": difficulty}


def _order(info, default=None):
    values = info.get("orders", []) if isinstance(info, dict) else []
    return max(values) if values else default


def _source_order(source_index, source_id):
    value = source_index.get(source_id, {}).get("order")
    return value if isinstance(value, int) else None


def _expansion_entry(missing_kind, relation, fact_ids=(), source_ids=(), order=None,
                     distance=None):
    return {
        "missing_kind": missing_kind,
        "relation": relation,
        "fact_ids": sorted({item for item in fact_ids if isinstance(item, str)}),
        "source_ids": sorted({item for item in source_ids if isinstance(item, str)}),
        "order": order,
        "distance": distance,
    }


def _info_objects(info):
    statement = info.get("fact", {}).get("statement", "")
    return (set(info.get("paths", set())) | set(info.get("entities", set()))
            | set(_QUALIFIED_OBJECT.findall(statement)))


def _entry_objects(entry, evidence_index, include_base=False):
    objects = set()
    fact_ids = list(entry.get("fact_ids", []))
    if include_base and isinstance(entry.get("base_fact_id"), str):
        fact_ids.append(entry["base_fact_id"])
    for fact_id in fact_ids:
        info = evidence_index["info_by_id"].get(fact_id)
        if info is not None:
            objects.update(_info_objects(info))
    # Source-only relations (for example a call result without a separately
    # extracted fact) still need exact object matching. Fact-backed entries use
    # their fact objects so an unrelated co-recorded object cannot qualify.
    if not entry.get("fact_ids"):
        source_ids = list(entry.get("source_ids", []))
        if include_base and isinstance(entry.get("base_source_id"), str):
            source_ids.append(entry["base_source_id"])
        for source_id in source_ids:
            source = evidence_index["source_index"].get(source_id, {})
            objects.update(source.get("paths", set()))
            objects.update(_entities(source.get("text", "")))
            objects.update(_QUALIFIED_OBJECT.findall(source.get("text", "")))
    return objects


def _source_material_ids(scope, source_id):
    """Resolve one citation to the exact retained material records it exposes."""
    if not isinstance(source_id, str):
        return set()
    return {
        record["id"]
        for key in ("dialogue", "events", "versions")
        for record in scope.get(key, [])
        if isinstance(record, dict) and isinstance(record.get("id"), str)
        and (record["id"] == source_id or record.get("parent_id") == source_id)
    }


def _qualified_group_binding(infos, value):
    """Return sources that bind an exact path to its symbol without cross-pairing."""
    if not isinstance(value, str) or "::" not in value:
        return set()
    path, symbol = value.rsplit("::", 1)
    if not path or not symbol:
        return set()
    basename = path.rsplit("/", 1)[-1]
    path_infos, symbol_infos = [], []
    for info in infos:
        statement = info["fact"].get("statement", "")
        sources = {source for source in info["fact"].get("sources", [])
                   if isinstance(source, str)}
        if path in statement or path in info.get("paths", set()):
            path_infos.append((info, sources))
        mentioned_basenames = {
            item.rsplit("/", 1)[-1]
            for item in _CODE_PATH_OBJECT.findall(statement)
        }
        if ((basename in statement and symbol in statement
             and mentioned_basenames == {basename})
                or value in _info_objects(info)):
            symbol_infos.append((info, sources))
    bound = set()
    for _, path_sources in path_infos:
        for _, symbol_sources in symbol_infos:
            bound.update(path_sources & symbol_sources)
    return bound


def _entry_matches_object(entry, evidence_index, value, group_infos):
    include_base = entry.get("relation") == "code_dependency"
    if value in _entry_objects(entry, evidence_index, include_base=include_base):
        return True
    if isinstance(value, str) and "::" in value:
        path, symbol = value.rsplit("::", 1)
        for fact_id in entry.get("fact_ids", []):
            info = evidence_index["info_by_id"].get(fact_id)
            if info is None:
                continue
            paths = set(info.get("paths", set()))
            statement = info["fact"].get("statement", "")
            if paths == {path} and symbol in statement:
                return True
    bound_sources = _qualified_group_binding(group_infos, value)
    if not bound_sources:
        return False
    entry_sources = {source for source in entry.get("source_ids", [])
                     if isinstance(source, str)}
    for fact_id in ([entry.get("base_fact_id")] + list(entry.get("fact_ids", []))):
        info = evidence_index["info_by_id"].get(fact_id)
        if info is not None:
            entry_sources.update(source for source in info["fact"].get("sources", [])
                                 if isinstance(source, str))
    return bool(bound_sources & entry_sources)


def _general_expansion_entry_allowed(entry, evidence_index):
    """Keep general expansion on conversation/document evidence only."""
    allowed = {"conversation", "document"}
    kinds = set()
    for fact_id in entry.get("fact_ids", []):
        info = evidence_index["info_by_id"].get(fact_id)
        if info is None:
            continue
        declared = set(info["fact"].get("source_kinds", []))
        if not declared and isinstance(info["fact"].get("source_kind"), str):
            declared.add(info["fact"]["source_kind"])
        if declared:
            kinds.update(declared)
        else:
            for source_id in info["fact"].get("sources", []):
                kinds.update(evidence_index["source_index"].get(
                    source_id, {}).get("source_kinds", set()))
    for source_id in entry.get("source_ids", []):
        kinds.update(evidence_index["source_index"].get(
            source_id, {}).get("source_kinds", set()))
    return bool(kinds) and kinds <= allowed


def _entry_sources_resolved(entry, evidence_index):
    """Require candidate citations to resolve to real retained records."""
    universe = evidence_index["universe"]
    records = [record for key in ("dialogue", "events", "versions")
               for record in universe.get(key, []) if isinstance(record, dict)]
    exact_ids = {
        record.get("id") for record in records
        if isinstance(record.get("id"), str)
        and any(key in record for key in (
            "text", "content", "changes", "success", "partial", "status", "complete"))
    }
    parent_ids = {record.get("parent_id") for record in records
                  if isinstance(record.get("parent_id"), str)}
    sources = {source for source in entry.get("source_ids", [])
               if isinstance(source, str)}
    for fact_id in entry.get("fact_ids", []):
        info = evidence_index["info_by_id"].get(fact_id)
        if info is None:
            return False
        sources.update(source for source in info["fact"].get("sources", [])
                       if isinstance(source, str))
    return bool(sources) and all(source in exact_ids or source in parent_ids
                                 for source in sources)


def _build_expansion_candidates(index):
    """Index only explicit relations that can answer a declared missing kind."""
    infos = index["infos"]
    source_index = index["source_index"]
    candidates = {info["fact"]["id"]: [] for info in infos}

    def add(base_id, entry):
        if not entry["fact_ids"] and not entry["source_ids"]:
            return
        marker = (entry["missing_kind"], entry["relation"],
                  tuple(entry["fact_ids"]), tuple(entry["source_ids"]))
        existing = {
            (item["missing_kind"], item["relation"],
             tuple(item["fact_ids"]), tuple(item["source_ids"]))
            for item in candidates[base_id]
        }
        if marker not in existing:
            candidates[base_id].append(entry)

    for base in infos:
        base_id = base["fact"]["id"]
        base_order = _order(base)
        base_sources = {source for source in base["fact"].get("sources", [])
                        if isinstance(source, str)}
        # Source-level links matter when a related version or call result did
        # not produce a standalone fact.  Direction is enforced here rather
        # than treating every record on one path as related.
        for source_id in base_sources:
            source = source_index.get(source_id, {})
            for peer_id in source.get("version_sources", set()):
                peer_order = _source_order(source_index, peer_id)
                if (peer_id not in base_sources and peer_order is not None
                        and base_order is not None and peer_order != base_order):
                    path = _shortest_from_sources(
                        index["direct_graph"], base_sources, peer_id)
                    add(base_id, _expansion_entry(
                        ("earlier_state" if peer_order < base_order
                         else "later_state"),
                        "version_previous", source_ids=[peer_id],
                        order=peer_order,
                        distance=path["distance"] if path else None))
            for peer_id in source.get("call_sources", set()):
                peer = source_index.get(peer_id, {})
                peer_order = _source_order(source_index, peer_id)
                if (peer_id not in base_sources and "result" in peer.get("kinds", set())
                        and (base_order is None or peer_order is None
                             or peer_order >= base_order)):
                    path = _shortest_from_sources(
                        index["direct_graph"], base_sources, peer_id)
                    add(base_id, _expansion_entry(
                        "outcome", "call_result", source_ids=[peer_id],
                        order=peer_order,
                        distance=path["distance"] if path else None))

        for candidate in infos:
            candidate_id = candidate["fact"]["id"]
            if candidate_id == base_id:
                continue
            candidate_order = _order(candidate)
            candidate_sources = candidate["fact"].get("sources", [])
            if (_version_ancestry_link(base, candidate)
                    and base_order is not None and candidate_order is not None
                    and candidate_order != base_order):
                candidate_paths = [_shortest_from_sources(
                    index["direct_graph"], base_sources, source)
                    for source in candidate_sources]
                candidate_paths = [path for path in candidate_paths if path]
                path = min(candidate_paths, key=lambda item: item["distance"]) \
                    if candidate_paths else None
                add(base_id, _expansion_entry(
                    ("earlier_state" if candidate_order < base_order
                     else "later_state"),
                    "version_previous", [candidate_id], candidate_sources,
                    candidate_order, path["distance"] if path else None))
            shared_sources = bool(
                set(base["fact"].get("sources", [])) & set(candidate_sources))
            graph_linked = _explicit_graph_link(base, candidate)
            if ((shared_sources or graph_linked)
                    and candidate.get("statement_labels", set())
                    & {"feedback", "decision", "constraint", "failure"}
                    and (base_order is None or candidate_order is None
                         or candidate_order < base_order)):
                candidate_paths = [_shortest_from_sources(
                    index["direct_graph"], base_sources, source)
                    for source in candidate_sources]
                candidate_paths = [path for path in candidate_paths if path]
                path = min(candidate_paths, key=lambda item: item["distance"]) \
                    if candidate_paths else None
                add(base_id, _expansion_entry(
                    "reason", "feedback", [candidate_id], candidate_sources,
                    candidate_order, path["distance"] if path else None))
            if graph_linked:
                candidate_paths = [_shortest_from_sources(
                    index["direct_graph"], base_sources, source)
                    for source in candidate_sources]
                candidate_paths = [path for path in candidate_paths if path]
                path = min(candidate_paths, key=lambda item: item["distance"]) \
                    if candidate_paths else None
                add(base_id, _expansion_entry(
                    "dependency", "code_dependency", [candidate_id],
                    candidate_sources, candidate_order,
                    path["distance"] if path else None))
            if (_call_result_link(base, candidate)):
                candidate_is_result = any(
                    "result" in source_index.get(source, {}).get("kinds", set())
                    for source in candidate_sources)
                if candidate_is_result and (base_order is None or candidate_order is None
                                            or candidate_order >= base_order):
                    candidate_paths = [_shortest_from_sources(
                        index["direct_graph"], base_sources, source)
                        for source in candidate_sources]
                    candidate_paths = [path for path in candidate_paths if path]
                    path = min(candidate_paths, key=lambda item: item["distance"]) \
                        if candidate_paths else None
                    add(base_id, _expansion_entry(
                        "outcome", "call_result", [candidate_id], candidate_sources,
                        candidate_order, path["distance"] if path else None))
            if ((shared_sources or graph_linked)
                    and "validation" in candidate.get("statement_labels", set())
                    and (base_order is None or candidate_order is None
                         or candidate_order > base_order)):
                candidate_paths = [_shortest_from_sources(
                    index["direct_graph"], base_sources, source)
                    for source in candidate_sources]
                candidate_paths = [path for path in candidate_paths if path]
                path = min(candidate_paths, key=lambda item: item["distance"]) \
                    if candidate_paths else None
                add(base_id, _expansion_entry(
                    "outcome", "test", [candidate_id], candidate_sources,
                    candidate_order, path["distance"] if path else None))

    relation_priority = {"version_previous": 0, "feedback": 0,
                         "call_result": 0, "code_dependency": 0, "test": 1}
    for base_id, entries in candidates.items():
        entries.sort(key=lambda item: (
            relation_priority.get(item["relation"], 9),
            item["distance"] if isinstance(item.get("distance"), int) else 10 ** 18,
            tuple(item["fact_ids"]), tuple(item["source_ids"]),
        ))
    return candidates


def _source_frontier_candidates(source_ids, evidence_index):
    """Return explicit one-hop candidates rooted at retained source-only evidence."""
    source_index = evidence_index["source_index"]
    graph = evidence_index["direct_graph"]
    items = []
    for source_id in sorted(set(source_ids)):
        source = source_index.get(source_id, {})
        source_order = _source_order(source_index, source_id)
        for peer_id, distance, relation in graph.get(source_id, []):
            if peer_id in source_ids or distance > 1:
                continue
            peer = source_index.get(peer_id, {})
            peer_order = _source_order(source_index, peer_id)
            missing_kind = None
            if relation == "version_previous" and source_order != peer_order:
                missing_kind = ("earlier_state"
                                if peer_order is not None and source_order is not None
                                and peer_order < source_order else "later_state")
            elif relation in {"call_result", "test"}:
                missing_kind = "outcome"
            elif relation == "feedback":
                missing_kind = "reason"
            elif relation == "graph":
                missing_kind = "dependency"
            if missing_kind is not None:
                entry = _expansion_entry(
                    missing_kind, relation, source_ids=[peer_id],
                    order=peer_order, distance=distance)
                entry["base_source_id"] = source_id
                items.append(entry)
    return items


def _group_expansion_pointer(infos, evidence_index, source_frontier=(),
                             selected_fact_ids=None):
    selected_ids = (set(selected_fact_ids) if selected_fact_ids is not None
                    else {info["fact"]["id"] for info in infos})
    items = []
    seen = set()
    for info in infos:
        for entry in evidence_index["expansion_candidates"].get(
                info["fact"]["id"], []):
            if set(entry["fact_ids"]) <= selected_ids and not entry["source_ids"]:
                continue
            marker = ((info["fact"]["id"]
                       if entry["relation"] == "code_dependency" else None),
                      entry["missing_kind"], entry["relation"],
                      tuple(entry["fact_ids"]), tuple(entry["source_ids"]))
            if marker in seen:
                continue
            seen.add(marker)
            public = {key: entry[key] for key in
                      ("missing_kind", "relation", "fact_ids", "source_ids", "distance")}
            public["base_fact_id"] = info["fact"]["id"]
            if info["fact"]["id"] not in selected_ids:
                public["fact_ids"] = sorted(set(public["fact_ids"])
                                            | {info["fact"]["id"]})
            items.append(public)
    for entry in _source_frontier_candidates(source_frontier, evidence_index):
        marker = (entry.get("base_source_id"), entry["missing_kind"],
                  entry["relation"], tuple(entry["fact_ids"]),
                  tuple(entry["source_ids"]))
        if marker in seen:
            continue
        seen.add(marker)
        items.append(entry)
    items.sort(key=lambda item: (
        item["missing_kind"],
        item["distance"] if isinstance(item.get("distance"), int) else 10 ** 18,
        item["relation"], item.get("base_fact_id", ""), tuple(item["fact_ids"]),
        tuple(item["source_ids"])))
    return {"candidates": items}


def expand_evidence_group_once(group, evidence_index, missing_kind,
                               missing_object=None, target_chars=16000,
                               max_chars=32000, static_direction=False):
    """Expand one group once through a declared, explicit evidence relation."""
    audit = {
        "group_id": group.get("id"),
        "missing_kind": missing_kind,
        "missing_object": missing_object,
        "status": "not_applicable",
        "before_fact_ids": [fact.get("id") for fact in group.get("facts", [])],
        "before_source_ids": sorted({source for fact in group.get("facts", [])
                                     for source in fact.get("sources", [])}),
    }
    if missing_kind not in MISSING_KINDS:
        audit["reason"] = "unsupported_missing_kind"
        return None, audit
    if static_direction and missing_kind not in {"earlier_state", "later_state"}:
        audit["reason"] = "static_direction_not_allowed"
        return None, audit
    if (not isinstance(missing_object, str) or not missing_object.strip()) \
            and not static_direction:
        audit["reason"] = "missing_object_required"
        return None, audit
    missing_object = missing_object.strip() if isinstance(missing_object, str) else None
    group_infos = _group_infos(group, evidence_index)
    current_sources = ({source for fact in group.get("facts", [])
                        for source in fact.get("sources", [])
                        if isinstance(source, str)}
                       | {source for source in group.get("scope", {}).get(
                           "generation_extra_sources", [])
                          if isinstance(source, str)})
    audit["before_source_ids"] = sorted(current_sources)
    audit["before_source_material_ids"] = sorted(set().union(*(
        _source_material_ids(group.get("scope", {}), source)
        for source in current_sources))) if current_sources else []
    extra_sources = set(group.get("scope", {}).get(
        "generation_extra_sources", []))
    frontier_infos = list(group_infos)
    frontier_ids = {info["fact"]["id"] for info in frontier_infos}
    for info in evidence_index["infos"]:
        fact_id = info["fact"]["id"]
        if (fact_id not in frontier_ids and extra_sources
                & set(info["fact"].get("sources", []))):
            frontier_infos.append(info)
            frontier_ids.add(fact_id)
    pointer = _group_expansion_pointer(
        frontier_infos, evidence_index, source_frontier=current_sources,
        selected_fact_ids={info["fact"]["id"] for info in group_infos})
    audit["pointer_source"] = "derived_current_frontier"
    choices = [item for item in pointer.get("candidates", [])
               if item.get("missing_kind") == missing_kind
               and (static_direction
                    or _entry_matches_object(
                        item, evidence_index, missing_object, group_infos))]
    choices = [item for item in choices
               if _entry_sources_resolved(item, evidence_index)]
    if group.get("qa_mode") == "general":
        choices = [item for item in choices
                   if _general_expansion_entry_allowed(item, evidence_index)]
    if not choices:
        audit["reason"] = "no_matching_explicit_relation"
        return None, audit
    aliases = evidence_index.get("fact_aliases", {})
    existing_fact_ids = {
        aliases.get(fact.get("id"), fact.get("id"))
        for fact in group.get("facts", []) if isinstance(fact, dict)
    }
    audit["before_fact_ids"] = sorted(existing_fact_ids)
    existing_generation_sources = ({source for fact in group.get("facts", [])
                                    for source in fact.get("sources", [])}
                                   | set(group.get("scope", {}).get(
                                       "generation_extra_sources", [])))
    existing_source_materials = set().union(*(
        _source_material_ids(group.get("scope", {}), source)
        for source in existing_generation_sources)) \
        if existing_generation_sources else set()
    choices = [item for item in choices
               if (set(item.get("fact_ids", [])) - existing_fact_ids
                   or set().union(*(
                       _source_material_ids(evidence_index["universe"], source)
                       for source in item.get("source_ids", [])))
                   - existing_source_materials)]
    if not choices:
        audit["reason"] = "no_new_evidence"
        return None, audit
    choices = choices[:12]
    info_by_id = evidence_index["info_by_id"]
    existing_extra_sources = {
        source for source in group.get("scope", {}).get(
            "generation_extra_sources", []) if isinstance(source, str)
    }
    over_budget = []
    no_progress = 0
    for chosen in choices:
        facts = list(group.get("facts", []))
        fact_ids = {aliases.get(fact.get("id"), fact.get("id"))
                    for fact in facts if isinstance(fact, dict)}
        for fact_id in chosen.get("fact_ids", []):
            info = info_by_id.get(fact_id)
            if info is not None and fact_id not in fact_ids:
                facts.append(info["fact"])
                fact_ids.add(fact_id)
        infos = [info_by_id[fact_id] for fact_id in fact_ids
                 if fact_id in info_by_id]
        extra_sources = existing_extra_sources | {
            source for source in chosen.get("source_ids", [])
            if isinstance(source, str)
        }
        guarded_infos = _review_guard_closure(infos, evidence_index["infos"])
        guard_sources = _guard_sources(guarded_infos) | extra_sources
        projected, request_chars, padding = _project_group(
            evidence_index["universe"], infos, target_chars, max_chars,
            required_sources=guard_sources)
        if projected is None:
            over_budget.append(chosen["relation"])
            continue

        projected["review_guard_sources"] = sorted(guard_sources)
        projected["review_guard_complete"] = True
        projected["review_guard_reason"] = "complete"
        projected["review_guard_omitted_count"] = 0
        projected["generation_extra_sources"] = sorted(extra_sources)
        original_relation = group.get("relation_path", {})
        extra_nodes = list(chosen.get("fact_ids", [])) + list(extra_sources)
        relation_path = _relation_metadata(
            infos, evidence_index, seed_node=original_relation.get("seed_node"),
            extra_nodes=extra_nodes)
        base_meta = dict(group.get("scope", {}).get("evidence_group", {}))
        base_meta.update(
            fact_count=len(facts), evidence_fact_count=len(facts),
            expansion="directed_bounded", padding_records=padding,
            relation_max_distance=relation_path["max_distance"],
            relation_path_complete=relation_path["path_complete"])
        projected["evidence_group"] = base_meta
        expanded = dict(group)
        expanded.update(
            facts=facts, scope=projected,
            review_guard_sources=projected["review_guard_sources"],
            review_guard_complete=True, request_chars=request_chars,
            max_questions=1,
            relation_path=relation_path,
            static_difficulty=relation_path["difficulty"],
            expansion_pointer={"candidates": []},
        )
        added_fact_ids = sorted(fact_ids - existing_fact_ids)
        resulting_sources = ({source for fact in facts
                              for source in fact.get("sources", [])
                              if isinstance(source, str)} | extra_sources)
        added_source_ids = sorted(resulting_sources - existing_generation_sources)
        resulting_source_materials = set().union(*(
            _source_material_ids(projected, source)
            for source in resulting_sources)) if resulting_sources else set()
        added_source_material_ids = sorted(
            resulting_source_materials - existing_source_materials)
        if not added_fact_ids and not added_source_material_ids:
            no_progress += 1
            continue
        audit.update(
            status="expanded", reason="explicit_relation",
            relation=chosen["relation"], tried_over_budget=len(over_budget),
            added_fact_ids=added_fact_ids,
            added_source_ids=added_source_ids,
            added_source_material_ids=added_source_material_ids,
            after_fact_ids=[fact.get("id") for fact in facts],
            after_source_ids=sorted(_fact_sources(infos) | extra_sources),
            after_source_material_ids=sorted(resulting_source_materials),
        )
        expanded["expansion_audit"] = audit
        return expanded, audit
    if no_progress == len(choices):
        audit.update(status="not_applicable", reason="no_new_evidence",
                     tried_no_progress=no_progress)
    elif no_progress:
        audit.update(status="not_applicable", reason="no_fitting_new_evidence",
                     tried_over_budget=len(over_budget),
                     tried_no_progress=no_progress)
    else:
        audit.update(status="over_budget", reason="all_matching_expansions_over_budget",
                     tried_over_budget=len(over_budget),
                     tried_no_progress=no_progress)
    return None, audit


def _code_evidence_result(status, reason, infos, source_ids=()):
    return {
        "status": status,
        "reason": reason,
        "fact_ids": sorted({info["fact"].get("id") for info in infos
                            if isinstance(info["fact"].get("id"), str)}),
        "source_ids": sorted({source for source in source_ids
                              if isinstance(source, str)}),
    }


def _answer_evidence(candidate):
    texts, sources = [], []
    for point in candidate.get("answer_points", []) if isinstance(candidate, dict) else []:
        if not isinstance(point, dict):
            continue
        if isinstance(point.get("text"), str):
            texts.append(point["text"])
        for source in point.get("sources", []):
            if isinstance(source, str) and source not in sources:
                sources.append(source)
    return "\n".join(texts), sources


def _group_infos(group, evidence_index, cited_sources=None):
    aliases = evidence_index.get("fact_aliases", {})
    selected, seen = [], set()
    for fact in group.get("facts", []):
        if not isinstance(fact, dict):
            continue
        fact_id = fact.get("id")
        canonical = aliases.get(fact_id, fact_id)
        info = evidence_index.get("info_by_id", {}).get(canonical)
        if info is None or info["fact"].get("id") in seen:
            continue
        if cited_sources is not None and not (
                set(info["fact"].get("sources", [])) & set(cited_sources)):
            continue
        selected.append(info)
        seen.add(info["fact"].get("id"))
    return selected


def _source_relation_kinds(evidence_index, sources):
    kinds = set()
    ordered = sorted({source for source in sources if isinstance(source, str)})
    for position, left in enumerate(ordered):
        for right in ordered[position + 1:]:
            path = _shortest_relation_path(evidence_index["direct_graph"], left, right)
            if path:
                kinds.update(edge["relation"] for edge in path["relations"])
    return kinds


def _strong_business_link(left, right):
    """Recognize only an explicit code-graph or version relation."""
    return bool(_explicit_graph_link(left, right)
                or _version_ancestry_link(left, right))


def _evaluate_code_evidence(infos, evidence_index, target_type, text, sources,
                            complete=True, post_generation=False):
    statements = [str(info["fact"].get("statement", "")) for info in infos]
    statement_text = "\n".join(statements)
    evidence_text = text if post_generation else statement_text
    source_ids = list(sources)
    labels = set().union(*(info.get("statement_labels", set()) for info in infos)) \
        if infos else set()
    relation_kinds = _source_relation_kinds(evidence_index, source_ids)

    if target_type == "history_tracking":
        # A multi-source fact may summarize both sides of a transition.  At
        # review time it proves that transition only when the answer cites
        # every source used by that fact; an overlap with the new side must
        # not import the uncited old side through the extracted statement.
        cited = set(source_ids)
        proof_infos = [
            info for info in infos
            if not post_generation
            or set(info["fact"].get("sources", [])).issubset(cited)
        ]
        proof_statements = [
            str(info["fact"].get("statement", "")) for info in proof_infos
        ]
        proof_text = "\n".join(proof_statements)
        source_transition = "version_previous" in relation_kinds
        fact_inline = any(
            _INLINE_TRANSITION.search(statement) for statement in proof_statements)
        fact_old_new = (bool(_OLD_STATE_CUE.search(proof_text))
                        and bool(_NEW_STATE_CUE.search(proof_text)))
        paired_facts = any(
            _strong_business_link(left, right)
            for position, left in enumerate(proof_infos)
            for right in proof_infos[position + 1:]
        )
        if (fact_inline or (fact_old_new and paired_facts)
                or (source_transition and len(set(source_ids)) >= 2)):
            return _code_evidence_result(
                "supported", "explicit_old_new_transition", infos, source_ids)
        missing_state_kinds = {
            entry.get("missing_kind")
            for info in infos
            for entry in evidence_index.get("expansion_candidates", {}).get(
                info["fact"].get("id"), [])
            if entry.get("missing_kind") in {"earlier_state", "later_state"}
        }
        if complete and (missing_state_kinds or any(
                info.get("historical_transition") for info in infos)):
            if len(missing_state_kinds) == 1:
                missing = next(iter(missing_state_kinds))
                reason = "history_missing_" + missing
            else:
                reason = "history_missing_state"
            return _code_evidence_result(
                "insufficient", reason, infos, source_ids)
        return _code_evidence_result(
            "unknown", "history_relation_not_statically_proven", infos, source_ids)

    if target_type == "failure_diagnosis":
        failures = [info for info in infos if info.get("observed_failure")]
        complements = [info for info in infos
                       if info.get("statement_labels", set())
                       & {"change", "validation", "decision", "feedback"}]
        for failure in failures:
            for complement in complements:
                if failure is complement or _strong_business_link(failure, complement):
                    return _code_evidence_result(
                        "supported", "failure_linked_to_diagnostic_evidence",
                        infos, source_ids)
        return _code_evidence_result(
            "unknown", "failure_relation_not_statically_proven", infos, source_ids)

    if target_type == "behavior_inference":
        conditions = [info for info in infos
                      if "conditional" in info.get("statement_labels", set())]
        outcomes = [info for info in infos
                    if _BEHAVIOR_OUTCOME.search(info["fact"].get("statement", ""))]
        answer_has_shape = (not post_generation or (
            "conditional" in _labels(evidence_text)
            and bool(_BEHAVIOR_OUTCOME.search(evidence_text))))
        inline = any(
            "conditional" in info.get("statement_labels", set())
            and _BEHAVIOR_OUTCOME.search(info["fact"].get("statement", ""))
            for info in infos)
        linked = any(
            _strong_business_link(condition, outcome)
            for condition in conditions for outcome in outcomes
            if condition is not outcome)
        if answer_has_shape and (inline or linked):
            return _code_evidence_result(
                "supported", "explicit_condition_to_behavior", infos, source_ids)
        weak_only = bool({"call_result"} & relation_kinds) or any(
            left.get("paths", set()) & right.get("paths", set())
            for position, left in enumerate(infos)
            for right in infos[position + 1:])
        if weak_only:
            return _code_evidence_result(
                "unknown", "only_nonsemantic_or_shared_file_link", infos, source_ids)
        return _code_evidence_result(
            "unknown", "behavior_relation_not_statically_proven", infos, source_ids)

    if target_type == "fact_recall":
        durable = bool(labels & {"constraint", "failure", "decision", "feedback"})
        historical = bool(_OLD_STATE_CUE.search(statement_text)
                          or "temporal" in labels)
        if len(infos) == 1 and durable and historical:
            return _code_evidence_result(
                "supported", "historical_constraint_failure_or_decision",
                infos, source_ids)
        return _code_evidence_result(
            "unknown", "fact_recall_not_statically_classified", infos, source_ids)

    return _code_evidence_result(
        "unknown", "unsupported_static_code_type", infos, source_ids)


def static_code_evidence_check(group, evidence_index, target_type, candidate=None):
    """Check conservative code-evidence necessities before or after generation.

    ``candidate=None`` checks the selected facts before generation. Otherwise
    only facts cited by ``answer_points`` participate. ``unknown`` means the
    static graph lacks semantic authority; callers should continue with their
    semantic answer-basis review rather than treating it as acceptance or
    rejection.
    """
    if group.get("qa_mode") != "code":
        return _code_evidence_result(
            "unknown", "not_code_mode", [], ())
    text, cited_sources = _answer_evidence(candidate or {})
    post_generation = candidate is not None
    complete = group.get("review_guard_complete", True) is True
    all_infos = _group_infos(group, evidence_index)
    if post_generation:
        scope_sources = {
            item.get("id")
            for field in ("dialogue", "events", "versions")
            for item in group.get("scope", {}).get(field, [])
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
        if not cited_sources:
            return _code_evidence_result(
                "insufficient" if complete else "unknown",
                "answer_has_no_source_citations", [], ())
        unknown_sources = set(cited_sources) - scope_sources
        if unknown_sources:
            return _code_evidence_result(
                "insufficient", "answer_source_out_of_scope", [], cited_sources)
        infos = _group_infos(group, evidence_index, cited_sources)
        if not infos:
            return _code_evidence_result(
                "unknown", "cited_source_has_no_extracted_fact", [], cited_sources)
        selected_ids = {info["fact"].get("id") for info in infos}
        if target_type == "failure_diagnosis" and complete:
            selected_failures = [info for info in infos if info.get("observed_failure")]
            omitted_diagnostics = [
                info for info in all_infos
                if info["fact"].get("id") not in selected_ids
                and info.get("statement_labels", set())
                & {"change", "validation", "decision", "feedback"}
                and any(_strong_business_link(failure, info)
                        for failure in selected_failures)
            ]
            if selected_failures and omitted_diagnostics:
                return _code_evidence_result(
                    "insufficient", "answer_cites_failure_but_omits_linked_diagnosis",
                    infos, cited_sources)
        if target_type == "behavior_inference" and complete:
            explicit_pairs = [
                (condition, outcome)
                for condition in all_infos
                if "conditional" in condition.get("statement_labels", set())
                for outcome in all_infos
                if _BEHAVIOR_OUTCOME.search(
                    outcome["fact"].get("statement", ""))
                and condition is not outcome
                and _strong_business_link(condition, outcome)
            ]
            if explicit_pairs and not any(
                    condition["fact"].get("id") in selected_ids
                    and outcome["fact"].get("id") in selected_ids
                    for condition, outcome in explicit_pairs):
                return _code_evidence_result(
                    "insufficient", "answer_omits_condition_or_linked_outcome",
                    infos, cited_sources)
    else:
        infos = all_infos
        cited_sources = [source for info in infos
                         for source in info["fact"].get("sources", [])]
    return _evaluate_code_evidence(
        infos, evidence_index, target_type, text, cited_sources,
        complete=complete, post_generation=post_generation)


def _static_eligible_types(infos, qa_mode, proposed, evidence_index):
    """Apply conservative necessary conditions without claiming semantics."""
    proposed = set(proposed)
    unsupported = proposed & {"open-domain"}
    if "adversarial" in proposed and not evidence_index["universe"].get(
            "full_range_covered"):
        unsupported.add("adversarial")
    evidence_index["skipped_types"] = sorted(
        set(evidence_index.get("skipped_types", [])) | unsupported)
    proposed -= unsupported
    relation = _relation_metadata(infos, evidence_index)
    stages = set().union(*(info.get("stages", set()) for info in infos))
    orders = {order for info in infos for order in info.get("orders", [])}
    labels = set().union(*(info.get("labels", set()) for info in infos))
    explicit_change = bool(set().union(*(
        info.get("statement_labels", set()) for info in infos))
        & {"change", "temporal", "feedback"})
    connected = relation["path_complete"] and relation["max_distance"] is not None
    crossed = connected and relation["max_distance"] >= 1
    relation_kinds = {
        edge["relation"] for path in relation["paths"]
        for edge in path.get("relations", [])
    }
    version_change = "version_previous" in relation_kinds
    eligible = set()
    if qa_mode == "general":
        if "adversarial" in proposed and evidence_index["universe"].get(
                "full_range_covered"):
            eligible.add("adversarial")
        if "single-hop" in proposed and len(stages) <= 1:
            eligible.add("single-hop")
        if "multi-hop" in proposed and len(stages) >= 2 and crossed:
            eligible.add("multi-hop")
        if ("temporal" in proposed and len(orders) >= 2 and crossed
                and (explicit_change or version_change)):
            eligible.add("temporal")
    else:
        source_ids = [source for info in infos
                      for source in info["fact"].get("sources", [])]
        for question_type in proposed:
            check = _evaluate_code_evidence(
                infos, evidence_index, question_type, "", source_ids,
                complete=True, post_generation=False)
            if check["status"] != "insufficient":
                eligible.add(question_type)
    return eligible, relation


def static_candidate_labels(group, candidate, evidence_index, target_type):
    """Label one generated candidate from its actually cited answer evidence."""
    answer_sources = []
    for point in candidate.get("answer_points", []):
        for source in point.get("sources", []):
            if isinstance(source, str) and source not in answer_sources:
                answer_sources.append(source)
    # Difficulty describes the evidence actually needed by this question,
    # not the arbitrary root chosen while exploring the group.  A question
    # citing only a late snapshot is therefore easy even if its group was
    # reached through an older seed; a true old/new comparison still spans
    # the lineage and remains hard.
    source_index = evidence_index.get("source_index", {})
    answer_sources.sort(key=lambda source: (
        source_index.get(source, {}).get("order", 10 ** 18), source))
    candidate_seed = answer_sources[0] if answer_sources else None
    relation = _relation_metadata(
        [], evidence_index,
        seed_node=candidate_seed,
        extra_nodes=answer_sources)
    labels = {
        "type": target_type,
        "type_origin": "static_target",
        "difficulty": relation["difficulty"],
        "difficulty_origin": "static_graph_distance",
        "difficulty_distance": relation["max_distance"],
        "relation_path_complete": relation["path_complete"],
        "static_evidence_path": relation,
    }
    if group.get("qa_mode") == "code":
        requirement = static_code_evidence_check(
            group, evidence_index, target_type, candidate)
        labels.update(
            category=target_type,
            track="unknown",
            track_origin="static_graph_insufficient",
            static_evidence_status=requirement["status"],
            static_evidence_reason=requirement["reason"],
        )
    return labels


def build_evidence_groups(facts, scopes, qa_mode, allowed_types, max_groups,
                          target_chars=16000, max_chars=32000,
                          evidence_index=None, static_selection=False):
    """Create deterministic minimal fact groups, including cross-chunk links.

    The index proposes evidence groups; it never answers a question. A group is
    expanded by at most one additional fact layer before it is skipped.
    """
    if qa_mode not in {"general", "code"}:
        raise ValueError("qa_mode must be general or code")
    if max_groups <= 0:
        return []
    evidence_index = evidence_index or build_evidence_index(
        facts, scopes, qa_mode, max_chars)
    if evidence_index.get("qa_mode") != qa_mode:
        raise ValueError("evidence_index qa_mode mismatch")
    universe = evidence_index["universe"]
    infos = evidence_index["infos"]
    info_by_id = evidence_index["info_by_id"]

    if static_selection:
        unsupported = set(allowed_types) & {"open-domain"}
        if ("adversarial" in set(allowed_types)
                and not evidence_index["universe"].get("full_range_covered")):
            unsupported.add("adversarial")
        reasons = {
            "open-domain": "external_knowledge_not_separated",
            "adversarial": "full_range_not_proven",
        }
        for question_type in sorted(unsupported):
            diagnostic = {"type": question_type, "reason": reasons[question_type]}
            if diagnostic not in evidence_index["eligible_skips"]:
                evidence_index["eligible_skips"].append(diagnostic)

    def eligible(infos, proposed):
        if not static_selection:
            return set(proposed)
        selected, _ = _static_eligible_types(
            infos, qa_mode, proposed, evidence_index)
        return selected

    candidates = []
    for info in infos:
        proposed = _candidate_types([info], qa_mode, allowed_types)
        types = eligible([info], proposed)
        if types:
            utility = len(info["labels"] & {
                "failure", "validation", "change", "feedback", "constraint", "decision"})
            candidates.append(((info["fact"]["id"],), types, 10 + utility, "minimal"))

    pair_scores = []
    for left_index, left in enumerate(infos):
        ranked = []
        for right in infos[left_index + 1:]:
            score, linked = _relation(left, right, qa_mode)
            if linked and score > 0:
                ranked.append((score, right))
        ranked = sorted(ranked, key=lambda item: (-item[0], item[1]["fact"]["id"]))
        for rank, (score, right) in enumerate(ranked):
            # Keep the normal top-3 bound for lexical links, but never discard
            # a pair backed by version ancestry or an explicit graph edge.
            structural = _version_ancestry_link(left, right) \
                or _explicit_graph_link(left, right)
            if rank >= 3 and not structural:
                continue
            pair = [left, right]
            proposed = _candidate_types(pair, qa_mode, allowed_types)
            types = eligible(pair, proposed)
            if types:
                ids = tuple(sorted((left["fact"]["id"], right["fact"]["id"])))
                pair_scores.append((ids, types, score + 20, "expanded_once"))
    candidates.extend(pair_scores)

    if qa_mode == "general" and "adversarial" in allowed_types \
            and universe.get("full_range_covered"):
        ranked_infos = sorted(
            infos,
            key=lambda item: (-len(item["statement_labels"] & {
                "feedback", "constraint", "decision", "temporal"}),
                item["fact"]["id"]),
        )
        for info in ranked_infos[:3]:
            candidates.append(((info["fact"]["id"],), {"adversarial"},
                               9 + len(info["labels"]), "full_range"))

    # A three-step failure/change/validation chain is often the smallest useful
    # code-memory unit. Add only one best third fact to a connected pair.
    if qa_mode == "code" and {"failure_diagnosis", "history_tracking"} & set(allowed_types):
        for ids, _, score, _ in pair_scores:
            base = [info_by_id[fid] for fid in ids]
            base_entities = set().union(*(
                {entity for entity in item["entities"]
                 if not entity.startswith(("zh:", "zhx:"))
                 and entity not in _WEAK_ENTITIES}
                | item["paths"] | item.get("graph_neighbors", set())
                for item in base))
            base_labels = set().union(*(item["labels"] for item in base))
            ranked = []
            for extra in infos:
                if extra["fact"]["id"] in ids:
                    continue
                shared = base_entities & ({entity for entity in extra["entities"]
                                           if not entity.startswith(("zh:", "zhx:"))
                                           and entity not in _WEAK_ENTITIES}
                                          | extra["paths"]
                                          | extra.get("graph_neighbors", set()))
                complement = len((base_labels | extra["statement_labels"])
                                 & {"failure", "change", "validation"})
                extra_linked = any(_relation(item, extra, qa_mode)[1] for item in base)
                if shared and extra_linked and complement >= 2 and (
                        any(item.get("observed_failure") for item in base + [extra])):
                    ranked.append((complement * 3 + min(len(shared), 3), extra))
            if ranked:
                _, extra = sorted(ranked, key=lambda item: (-item[0], item[1]["fact"]["id"]))[0]
                triple = base + [extra]
                proposed = _candidate_types(triple, qa_mode, allowed_types)
                types = eligible(triple, proposed)
                if types:
                    triple_ids = tuple(sorted(ids + (extra["fact"]["id"],)))
                    candidates.append((triple_ids, types, score + 5, "expanded_once"))

    unique = {}
    for ids, types, score, expansion in candidates:
        key = (ids, tuple(sorted(types)))
        if key not in unique or score > unique[key][2]:
            unique[key] = (ids, types, score, expansion)

    by_type = {question_type: [] for question_type in allowed_types}
    for candidate in unique.values():
        for question_type in candidate[1]:
            by_type.setdefault(question_type, []).append(candidate)
    for question_type in by_type:
        by_type[question_type].sort(key=lambda item: (-item[2], len(item[0]), item[0]))

    selected = []
    seen = set()
    used_fact_counts = {}
    used_object_counts = {}
    used_labels = {}
    groups, projections = [], {}
    def materialize(ids, types, score, expansion):
        group_infos = [info_by_id[fid] for fid in ids]
        # Recover related clauses about the same object already present in
        # cited original records. This adds no new raw context and avoids
        # asking questions from one arbitrarily isolated field of a function.
        if expansion != "full_range":
            source_ids = {source for info in group_infos
                          for source in info["fact"].get("sources", [])}
            extras = [info for info in infos if info["fact"]["id"] not in ids
                      and source_ids.intersection(info["fact"].get("sources", []))
                      and any(_semantic_shared_entities(info, base)
                              for base in group_infos)]
            extras.sort(key=lambda info: (
                used_fact_counts.get(info["fact"]["id"], 0),
                -len(info["statement_labels"] & _RELATION_LABELS),
                info["fact"]["id"]))
            group_infos.extend(extras[:max(0, 6 - len(group_infos))])
        base_infos = list(group_infos)
        guarded_infos = (base_infos if expansion == "full_range" else
                         _review_guard_closure(base_infos, infos))
        complete_guard_sources = _guard_sources(guarded_infos)
        projected, request_chars, padding = _project_group(
            universe, base_infos, target_chars, max_chars,
            full_range=expansion == "full_range",
            required_sources=complete_guard_sources)
        guard_complete = projected is not None
        group_infos = base_infos
        if not guard_complete:
            # Keep an otherwise valid evidence group when its explicit update
            # closure exceeds the request budget. Downstream review must leave
            # it as needs_review rather than silently approving partial proof.
            projected, request_chars, padding = _project_group(
                universe, group_infos, target_chars, max_chars,
                full_range=expansion == "full_range",
                required_sources=_fact_sources(group_infos))
            while projected is None and len(group_infos) > len(ids):
                group_infos.pop()
                projected, request_chars, padding = _project_group(
                    universe, group_infos, target_chars, max_chars,
                    full_range=expansion == "full_range",
                    required_sources=_fact_sources(group_infos))
        if projected is None:
            return None
        included_guard_sources = (_fact_sources(group_infos) if not guard_complete
                                  else complete_guard_sources)
        projected["review_guard_sources"] = sorted(included_guard_sources)
        projected["review_guard_complete"] = guard_complete
        projected["review_guard_reason"] = (
            "complete" if guard_complete else "over_budget")
        projected["review_guard_omitted_count"] = (
            0 if guard_complete else
            len(complete_guard_sources - included_guard_sources))
        ids = tuple(info["fact"]["id"] for info in group_infos)
        stages = set().union(*(info["stages"] for info in group_infos))
        sources = {source for info in group_infos for source in info["fact"].get("sources", [])}
        graph_hops = [universe["source_graph_hops"].get(source) for source in sources]
        graph_hops = [value for value in graph_hops if isinstance(value, int)]
        group_id = "%s-group-%d" % (qa_mode, len(groups) + 1)
        projected["evidence_group"] = {
            "id": group_id,
            "target_types": sorted(types),
            "fact_count": len(ids),
            "stage_count": len(stages),
            "reasoning_hops": None,
            "evidence_fact_count": len(ids),
            "graph_hops": max(graph_hops) if graph_hops else None,
            "expansion": expansion,
            "padding_records": padding,
        }
        eligible_types = eligible(group_infos, types)
        relation_path = _relation_metadata(group_infos, evidence_index)
        if not eligible_types:
            return None
        types = eligible_types
        projected["evidence_group"].update(
            target_types=sorted(types),
            relation_max_distance=relation_path["max_distance"],
            relation_path_complete=relation_path["path_complete"],
        )
        group = {
            "id": group_id,
            "qa_mode": qa_mode,
            "allowed_types": tuple(sorted(types)),
            "facts": [info["fact"] for info in group_infos],
            "scope": projected,
            "review_guard_sources": projected["review_guard_sources"],
            "review_guard_complete": projected["review_guard_complete"],
            "score": score,
            "request_chars": request_chars,
            "eligible_types": tuple(sorted(types)),
            "relation_path": relation_path,
            "static_difficulty": relation_path["difficulty"],
            "expansion_pointer": _group_expansion_pointer(group_infos, evidence_index),
            # Two independent grounded statements permit, but do not require,
            # two distinct answer targets in the same small request.
            "max_questions": 2 if len({
                (info["fact"]["statement"].strip(), tuple(sorted(info["fact"].get("sources", []))))
                for info in group_infos}) >= 2 else 1,
        }
        if qa_mode == "code":
            group["static_evidence_requirements"] = {
                question_type: static_code_evidence_check(
                    group, evidence_index, question_type)
                for question_type in group["allowed_types"]
            }
        return group
    while len(selected) < max_groups:
        added = False
        for question_type in sorted(by_type):
            while by_type[question_type]:
                ranked = sorted(
                    by_type[question_type],
                    key=lambda item: (
                        -(item[2]
                          - 12 * sum(used_fact_counts.get(fid, 0) for fid in item[0])
                          - 4 * sum(used_object_counts.get(entity, 0)
                                     for fid in item[0]
                                     for entity in info_by_id[fid]["entities"]
                                     if not entity.startswith(("zh:", "zhx:")))
                          - 2 * sum(used_labels.get(label, 0)
                                     for fid in item[0]
                                     for label in info_by_id[fid]["labels"])),
                        len(item[0]), item[0], item[3],
                    ),
                )
                candidate = ranked[0]
                by_type[question_type].remove(candidate)
                ids, types, score, expansion = candidate
                # A source set is one group even when it supports several
                # question types. The model may generate distinct targets
                # in a single call instead of receiving the same graph again.
                marker = (ids, expansion == "full_range")
                if marker in seen:
                    continue
                seen.add(marker)
                group = materialize(ids, types, score, expansion)
                if group is None:
                    continue
                projection_key = json.dumps({
                    "facts": sorted(f["id"] for f in group["facts"]),
                    "sources": {key: group["scope"].get(key, [])
                                for key in ("dialogue", "events", "versions", "edges", "historical_edges")},
                    "full_range": expansion == "full_range",
                }, sort_keys=True, ensure_ascii=False)
                if projection_key in projections:
                    previous = projections[projection_key]
                    previous["allowed_types"] = tuple(sorted(set(previous["allowed_types"]) | set(types)))
                    previous["eligible_types"] = previous["allowed_types"]
                    previous["scope"]["evidence_group"]["target_types"] = list(previous["allowed_types"])
                    continue
                projections[projection_key] = group
                groups.append(group)
                selected.append((ids, types, score, expansion))
                for fid in ids:
                    used_fact_counts[fid] = used_fact_counts.get(fid, 0) + 1
                    for entity in info_by_id[fid]["entities"]:
                        if not entity.startswith(("zh:", "zhx:")):
                            used_object_counts[entity] = used_object_counts.get(entity, 0) + 1
                    for label in info_by_id[fid]["labels"]:
                        used_labels[label] = used_labels.get(label, 0) + 1
                added = True
                break
            if len(selected) >= max_groups:
                break
        if not added:
            break

    return groups


def coverage_report(records, versions, scopes, facts, groups, stage_status, questions,
                    attempted_group_ids=None):
    if attempted_group_ids is not None:
        attempted = set(attempted_group_ids)
        pool = coverage_report(records, versions, scopes, facts, groups, [], [])
        actual = coverage_report(records, versions, scopes, facts,
                                 [g for g in groups if g["id"] in attempted],
                                 stage_status, questions)
        for mode in actual:
            actual[mode]["candidate_pool"] = pool[mode]
            actual[mode]["groups"]["attempted"] = actual[mode]["groups"]["total"]
        return actual
    """Count source, fact and group coverage without claiming semantic completeness."""
    def sources(scope):
        return {item.get("parent_id", item.get("id"))
                for field in ("dialogue", "events", "versions")
                for item in scope.get(field, []) if item.get("id")}

    result = {}
    for track in ("general", "code"):
        track_scopes = scopes.get(track, [])
        track_facts = [f for f in facts if f.get("qa_mode", "code") == track]
        track_groups = [g for g in groups if g.get("qa_mode") == track]
        eligible_records = records if track == "code" else [
            r for r in records if r.get("kind") == "message" or r.get("source_kind") == "document"]
        universe = {r["id"] for r in eligible_records}
        if track == "code":
            universe |= {v["id"] for v in versions}
        scope_sources = set().union(*(sources(s) for s in track_scopes))
        fact_sources = {s for f in track_facts for s in f.get("sources", [])}
        grouped_sources = set().union(*(sources(g["scope"]) for g in track_groups))
        all_ids = {f["id"] for f in track_facts}
        used_ids = {f["id"] for g in track_groups for f in g["facts"]} & all_ids
        # Object coverage is deliberately lexical and source-grounded. It is
        # a diagnostic for selection diversity, not a claim that every token
        # is a semantic entity.
        symbol_names = {
            symbol.get("name").casefold()
            for version in versions
            if isinstance(version, dict)
            for symbol in (version.get("code") or {}).get("symbols", [])
            if isinstance(symbol, dict) and isinstance(symbol.get("name"), str)
        }
        object_by_fact = {}
        for fact in track_facts:
            info = _fact_info(fact, {}, set())
            if track == "code":
                statement = str(fact.get("statement", ""))
                identifiers = set(_IDENTIFIER.findall(statement))
                identifiers = {
                    item.casefold() for item in identifiers
                    if (item.casefold() in symbol_names or "." in item
                        or item.casefold().endswith(_CODE_FILE_SUFFIXES)
                        or re.search(r"\b" + re.escape(item) + r"\s*\(", statement))
                }
                info["entities"] = identifiers
            object_by_fact[fact["id"]] = {
                "paths": set(info.get("paths", set())),
                "entities": {item for item in info.get("entities", set())
                              if not item.startswith(("zh:", "zhx:"))
                              and item not in _WEAK_ENTITIES},
            }
        all_objects = set().union(*(
            item["paths"] | item["entities"] for item in object_by_fact.values()))
        used_objects = set().union(*(
            object_by_fact.get(fact["id"], {}).get("paths", set())
            | object_by_fact.get(fact["id"], {}).get("entities", set())
            for group in track_groups for fact in group.get("facts", [])))
        relation_labels = set().union(*(
            _labels(fact.get("statement", "")) for fact in track_facts))
        grouped_relation_labels = set().union(*(
            _labels(fact.get("statement", ""))
            for group in track_groups for fact in group.get("facts", [])))
        version_ids = {version.get("id") for version in versions
                       if isinstance(version, dict) and isinstance(version.get("id"), str)}
        version_source_map = {
            source: version.get("id")
            for version in versions if isinstance(version, dict)
            for source in (version.get("id"), version.get("source"))
            if isinstance(source, str) and isinstance(version.get("id"), str)
        }
        cited_versions = {version_source_map[source] for source in fact_sources
                          if source in version_source_map}
        grouped_versions = {version_source_map[source]
                            for group in track_groups
                            for fact in group.get("facts", [])
                            for source in fact.get("sources", [])
                            if source in version_source_map}
        rows = [s for s in stage_status if s.get("track") == track and s.get("phase") == "qa_review"]
        published_groups = {q.get("evidence_group_id") for q in questions
                            if q.get("qa_mode") == track} - {None}
        result[track] = {
            "sources": {"eligible": len(universe), "in_scopes": len(universe & scope_sources),
                        "cited_by_facts": len(universe & fact_sources),
                        "in_groups": len(universe & grouped_sources),
                        "uncovered_ids": sorted(universe - scope_sources)},
            "chunks": {"total": len(track_scopes),
                       "facts_completed": sum(not s.get("facts_failed", False) for s in track_scopes)},
            "facts": {"total": len(all_ids), "in_groups": len(used_ids),
                      "group_coverage": len(used_ids) / len(all_ids) if all_ids else None,
                      "unused_ids": sorted(all_ids - used_ids)},
            "objects": {
                "total": len(all_objects),
                "in_groups": len(all_objects & used_objects),
                "coverage": (len(all_objects & used_objects) / len(all_objects)
                              if all_objects else None),
                "uncovered": sorted(all_objects - used_objects),
            },
            "relations": {
                "total": len(relation_labels),
                "in_groups": len(relation_labels & grouped_relation_labels),
                "coverage": (len(relation_labels & grouped_relation_labels) /
                              len(relation_labels) if relation_labels else None),
                "labels": sorted(relation_labels),
            },
            "versions": {
                "total": len(version_ids) if track == "code" else 0,
                "cited_by_facts": len(cited_versions) if track == "code" else 0,
                "in_groups": len(grouped_versions) if track == "code" else 0,
                "uncovered": sorted(version_ids - grouped_versions) if track == "code" else [],
            },
            "groups": {"total": len(track_groups),
                       "generated": sum(s.get("generated", 0) > 0 for s in rows),
                       "approved": sum(s.get("approved", 0) > 0 for s in rows),
                       "needs_review": sum(s.get("needs_review", 0) > 0 for s in rows),
                       "published": len(published_groups)},
        }
    return result
