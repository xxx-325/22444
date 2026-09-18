"""Validate QA contracts and source closure before model review."""

import re

from .normalize import SOURCE_KINDS, source_kind_for
from .protocol import (
    CODE_DISTINCTIVENESS_BASES,
    SIMPLE_ATOMICITY_STATES,
    SIMPLE_EVIDENCE_STATES,
    simple_point_ids,
)

CODE_QA_TYPES = {
    "fact_recall",
    "history_tracking",
    "behavior_inference",
    "failure_diagnosis",
}
GENERAL_QA_TYPES = {
    "single-hop",
    "multi-hop",
    "temporal",
    "open-domain",
    "adversarial",
}
QA_MODES = {"general", "code"}
DIFFICULTIES = {"easy", "medium", "hard"}
TRACKS = {"history_core", "inference_control"}
_COMPOUND_POINT = re.compile(
    r"(?:并将|并把|然后将|另外(?:新增|删除|修改)|(?:并且|同时|以及)(?:新增|删除|修改|安装|启用|禁用|写入|保留))", re.I)

_INTERNAL_PUBLIC_ID = re.compile(
    # e1/f1 are valid project symbols and filenames.  Treat them as internal
    # references only when they are used as standalone prose tokens, not when
    # followed by a call or filename suffix.
    r"(?<![A-Za-z0-9_])(?:e|f)\d+(?![A-Za-z0-9_.(])|"
    r"(?<![A-Za-z0-9_])(?:code|general)_s\d+_c\d+_f\d+(?![A-Za-z0-9_])|"
    r"(?<![A-Za-z0-9_])(?:stage|chunk|scope)[-_]?\d+(?![A-Za-z0-9_])",
    re.I,
)
_LOW_VALUE_CODE_SURFACE = re.compile(
    r"(?:格式化为多行|单行格式|多行格式|第\s*\d+\s*行|行号|"
    r"哪些导出名称|__all__\s*列表|导入语句.*(?:哪个|之后)|"
    r"函数签名.*格式|签名.*格式)", re.I)
_VACUOUS_QUESTION = re.compile(r"^(?:用户?是否要求过|有没有要求过|是否说过|用户发送了什么(?:简短)?指令|用户说了什么)\s*[？?]?$", re.I)
_UNSUPPORTED_TEMPORAL_REFERENCE = re.compile(r"上一次|最近一次|最后一次|上次")
_QUOTED_LITERAL = re.compile(
    r"`[^`]*`|\"[^\"]*\"|'[^']*'|“[^”]*”|‘[^’]*’")
_FLOW_QUESTION = re.compile(
    r"如何.*(?:传入|传给|传到|传递|流向|消费|插入|展开)|"
    r"(?:一路|经).*(?:传入|传给|传到|传递)|最终.*(?:消费|传入|成为)", re.I)
_SIGNATURE_ONLY_POINT = re.compile(
    r"(?:新增|增加|声明|定义|改为).{0,80}(?:参数|形参)|"
    r"(?:参数|形参).{0,40}(?:新增|增加|声明|定义)|"
    r"签名.{0,40}(?:新增|增加|包含|改为)", re.I)
_FLOW_ACTION_POINT = re.compile(
    r"调用|传入|传给|继续传|作为.*(?:实参|timeout)|"
    r"设置|插入|展开|捕获|抛出|返回|导致|使|=timeout_seconds", re.I)
_SIGNATURE_REQUEST = re.compile(r"签名|形参|参数定义|默认值|声明", re.I)


def _has_unsupported_temporal_reference(text):
    return bool(isinstance(text, str) and _UNSUPPORTED_TEMPORAL_REFERENCE.search(
        _QUOTED_LITERAL.sub("", text)))

# Compatibility names used by the original code-only pipeline.
CATEGORIES = CODE_QA_TYPES
CODE_TYPES = CODE_QA_TYPES
GENERAL_TYPES = GENERAL_QA_TYPES


def validate_sources(sources, allowed):
    """Return whether a non-empty source list is unique and in ``allowed``."""
    return (isinstance(sources, list) and bool(sources)
            and all(isinstance(source, str) and source in allowed for source in sources)
            and len(sources) == len(set(sources)))


def _dialogue_source_ids(scope, visible_messages_only=False):
    ids = set()
    for record in scope.get("dialogue", []):
        if not isinstance(record, dict):
            continue
        if visible_messages_only and not (
                record.get("kind") == "message"
                and record.get("role") in {"user", "assistant"}):
            continue
        if isinstance(record.get("id"), str):
            ids.add(record["id"])
        # Chunking may split one oversized record into lossless fragments. The
        # parent ID remains a valid provenance alias for the selected chunk.
        if isinstance(record.get("parent_id"), str):
            ids.add(record["parent_id"])
    return ids


def scope_source_ids(scope, qa_mode=None):
    """IDs a mode may cite from the supplied evidence scope.

    General QA is intentionally limited to visible user/assistant messages. Code
    QA may additionally cite tool events and reconstructed file versions. Omitting
    the mode retains the original permissive code-pipeline behavior.
    """
    if qa_mode is not None and qa_mode not in QA_MODES | {"both"}:
        raise ValueError("qa_mode must be general, code, or both")
    if qa_mode == "general":
        return _dialogue_source_ids(scope, visible_messages_only=True)
    ids = _dialogue_source_ids(scope)
    for collection in ("events", "versions"):
        for record in scope.get(collection, []):
            if not isinstance(record, dict):
                continue
            if isinstance(record.get("id"), str):
                ids.add(record["id"])
            if isinstance(record.get("parent_id"), str):
                ids.add(record["parent_id"])
    return ids


def _direct_source_links(scope):
    """Build explicit one-edge source links without inferring semantic relations."""
    known = scope_source_ids(scope)
    links = {source: set() for source in known}

    def connect(left, right):
        if left in known and right in known and left != right:
            links[left].add(right)
            links[right].add(left)

    versions_by_id = {
        version.get("id"): version for version in scope.get("versions", [])
        if isinstance(version, dict) and isinstance(version.get("id"), str)
    }
    for version in scope.get("versions", []):
        if not isinstance(version, dict):
            continue
        connect(version.get("id"), version.get("source"))
        connect(version.get("id"), version.get("previous"))
        # A later patch's source is directly associated with the source of its
        # predecessor. This permits an answer to cite both sides of a real
        # version transition when the selected fact cites either side.
        previous = versions_by_id.get(version.get("previous"))
        if previous:
            connect(version.get("source"), previous.get("source"))
    for record in scope.get("dialogue", []):
        if isinstance(record, dict):
            connect(record.get("id"), record.get("parent_id"))
    for event in scope.get("events", []):
        if isinstance(event, dict):
            connect(event.get("id"), event.get("parent_id"))
            connect(event.get("id"), event.get("call_source"))

    calls = {}
    for record in scope.get("dialogue", []):
        if not isinstance(record, dict) or not isinstance(record.get("call_id"), str):
            continue
        calls.setdefault(record["call_id"], []).append(record.get("id"))
    for sources in calls.values():
        for left in sources:
            for right in sources:
                connect(left, right)
    return links


def directly_related_sources(scope, sources, qa_mode=None):
    """Return cited sources plus explicitly adjacent provenance records."""
    allowed = scope_source_ids(scope, qa_mode)
    links = _direct_source_links(scope)
    related = set(sources) & allowed
    for source in tuple(related):
        related.update(links.get(source, set()) & allowed)
    return related


def validate_facts(document, scope, return_rejected=False, qa_mode=None):
    if not isinstance(document, dict) or not isinstance(document.get("facts"), list):
        raise ValueError("LLM facts response must contain facts[]")
    allowed = scope_source_ids(scope, qa_mode)
    source_kind_by_id = {}
    for collection in ("dialogue", "events", "versions"):
        for record in scope.get(collection, []):
            if isinstance(record, dict) and isinstance(record.get("id"), str):
                default = "code" if collection == "versions" else None
                kind = source_kind_for(record, default=default)
                # Explicit source labels from normalized input are authoritative;
                # graph projections must not silently replace them.
                if record.get("source_kind") in SOURCE_KINDS or record["id"] not in source_kind_by_id:
                    source_kind_by_id[record["id"]] = kind
                if isinstance(record.get("parent_id"), str):
                    if record.get("source_kind") in SOURCE_KINDS or record["parent_id"] not in source_kind_by_id:
                        source_kind_by_id[record["parent_id"]] = kind
    seen = set()
    accepted, rejected = [], []
    for fact in document["facts"]:
        if not isinstance(fact, dict):
            rejected.append({"reason": "fact_not_object"})
            continue
        fid = fact.get("id")
        if not isinstance(fid, str) or not fid or fid in seen:
            rejected.append({"fact": fact, "reason": "invalid_or_duplicate_id"})
            continue
        seen.add(fid)
        if not isinstance(fact.get("statement"), str) or not fact["statement"].strip():
            rejected.append({"fact": fact, "reason": "missing_statement"})
            continue
        if not validate_sources(fact.get("sources"), allowed):
            rejected.append({"fact": fact, "reason": "unknown_or_missing_evidence"})
            continue
        kinds = sorted({source_kind_by_id[source] for source in fact["sources"]
                        if source in source_kind_by_id})
        if not kinds:
            rejected.append({"fact": fact, "reason": "untyped_evidence"})
            continue
        declared = fact.get("source_kind")
        if declared is not None and declared not in SOURCE_KINDS:
            rejected.append({"fact": fact, "reason": "invalid_source_kind"})
            continue
        fact["source_kinds"] = kinds
        if declared in SOURCE_KINDS:
            fact["source_kind"] = declared
        elif len(kinds) == 1:
            fact["source_kind"] = kinds[0]
        accepted.append(fact)
    if return_rejected:
        return accepted, rejected
    return accepted


def _candidate_mode_and_type(question, expected_mode):
    explicit_mode = question.get("qa_mode")
    question_type = question.get("type")
    category = question.get("category")
    if explicit_mode is not None:
        mode = explicit_mode
    elif question_type in GENERAL_QA_TYPES:
        mode = "general"
    elif question_type in CODE_QA_TYPES or category in CODE_QA_TYPES:
        mode = "code"
    elif expected_mode in QA_MODES:
        mode = expected_mode
    else:
        mode = None
    if question_type is None and mode == "code":
        question_type = category
    return mode, question_type


def _validate_point_list(points, source_closure, required, label, incompatible_texts=None):
    if not isinstance(points, list) or (required and not points):
        return "missing_%s_points" % label
    seen = set()
    incompatible_texts = incompatible_texts or set()
    for point in points:
        if not isinstance(point, dict) or not isinstance(point.get("text"), str):
            return "invalid_%s_point" % label
        normalized = point["text"].strip().casefold()
        if not normalized:
            return "invalid_%s_point" % label
        if _COMPOUND_POINT.search(normalized):
            return "compound_%s_point" % label
        if normalized in seen:
            return "duplicate_%s_point" % label
        if normalized in incompatible_texts:
            return "conflicting_answer_and_forbidden_point"
        if not validate_sources(point.get("sources"), source_closure):
            return "invalid_%s_evidence" % label
        seen.add(normalized)
    return None


def _source_stage_ids(scope, sources):
    return {record.get("stage_id") for record in scope.get("dialogue", [])
            if isinstance(record, dict) and record.get("id") in sources
            and record.get("stage_id")}


def _source_orders(scope, sources):
    orders = set()
    for collection in ("dialogue", "events", "versions"):
        for record in scope.get(collection, []):
            if not isinstance(record, dict) or record.get("id") not in sources:
                continue
            value = record.get("order", record.get("observed_at"))
            if isinstance(value, int):
                orders.add(value)
    return orders


def _code_historical_signal(scope, sources, facts):
    """Return whether selected evidence contains an explicit historical fact."""
    text = "\n".join(
        str(fact.get("statement", "")) for fact in facts if isinstance(fact, dict))
    text += "\n" + "\n".join(
        str(record.get("text", record.get("content", "")))
        for collection in ("dialogue", "events", "versions")
        for record in scope.get(collection, [])
        if isinstance(record, dict) and record.get("id") in sources)
    if re.search(r"之前|后来|旧版|新版|历史|曾经|修改|改成|修复|补丁|失败|报错|测试|验证|反馈|previous|patch|fixed|failed|error|test|version", text, re.I):
        return True
    return any(
        isinstance(version, dict) and version.get("id") in sources
        and (version.get("previous") or version.get("observed_at") is not None)
        for version in scope.get("versions", []))


def _deterministic_type_guard(question, mode, question_type, scope, fact_sources,
                              selected_facts=None):
    """Reject type labels contradicted by observable source structure."""
    if mode == "general":
        stages = _source_stage_ids(scope, fact_sources)
        if question_type == "multi-hop" and len(stages) < 2:
            return "multi_hop_same_stage"
        if question_type == "temporal":
            # Temporal reasoning may happen inside one discussion stage. It
            # still needs two source positions or an explicit before/after
            # cue; stage count is a boundary for multi-hop only.
            orders = _source_orders(scope, fact_sources)
            source_text = "\n".join(
                str(record.get("text", "")) for record in scope.get("dialogue", [])
                if record.get("id") in fact_sources)
            if len(orders) < 2 and not re.search(
                    r"之前|后来|之后|先|再|当前|当时|旧版|新版|previous|later|before|after",
                    source_text, re.I):
                return "temporal_without_change"
        if question_type == "adversarial" and not (
                scope.get("full_range_required") or scope.get("full_range_covered")):
            return "adversarial_without_full_range"
        return None
    if mode == "code":
        selected_facts = selected_facts or []
        track = question.get("track", "history_core")
        if track == "history_core" and not _code_historical_signal(
                scope, fact_sources, selected_facts):
            return "history_without_historical_signal"
        if question_type == "failure_diagnosis":
            text = "\n".join(str(f.get("statement", "")) for f in selected_facts)
            if not re.search(r"失败|报错|错误|异常|failure|failed|error|exception|bug", text, re.I):
                return "failure_without_observed_failure"
    return None


def validate_candidates(document, facts, scope, qa_mode=None, allowed_types=None):
    """Validate general/code QA candidates while preserving legacy code input.

    ``qa_mode`` optionally constrains the expected branch. ``both`` accepts both
    branches. ``allowed_types`` applies an additional caller-selected type filter.
    Legacy candidates containing only ``category`` and ``track`` are normalized to
    ``qa_mode=code`` and expose the same value through the new ``type`` field.
    """
    if qa_mode not in {None, "general", "code", "both"}:
        raise ValueError("qa_mode must be general, code, both, or None")
    if allowed_types is not None:
        if isinstance(allowed_types, str):
            raise ValueError("allowed_types must be a collection, not text")
        allowed_types = set(allowed_types)
    if not isinstance(document, dict) or not isinstance(document.get("questions"), list):
        raise ValueError("LLM response must contain questions[]")

    facts_by_id = {fact.get("id"): fact for fact in facts
                   if isinstance(fact, dict) and isinstance(fact.get("id"), str)}
    accepted, rejected, seen = [], [], set()
    repairable_semantic = {
        "compound_answer_point", "compound_forbidden_point",
        "multi_hop_same_stage", "temporal_without_change",
        "history_without_historical_signal", "failure_without_observed_failure",
    }
    for question in document["questions"]:
        reason = None
        warnings = []
        if not isinstance(question, dict):
            rejected.append({"reason": "question_not_object"})
            continue
        qid = question.get("id")
        mode, question_type = _candidate_mode_and_type(question, qa_mode)
        if not isinstance(qid, str) or not qid or qid in seen:
            reason = "invalid_or_duplicate_id"
        elif mode not in QA_MODES:
            reason = "unknown_qa_mode"
        elif qa_mode in QA_MODES and mode != qa_mode:
            reason = "qa_mode_mismatch"
        elif mode == "general" and question_type not in GENERAL_QA_TYPES:
            reason = "unknown_type"
        elif mode == "code" and question_type not in CODE_QA_TYPES:
            reason = "unknown_category" if question.get("category") is not None else "unknown_type"
        elif allowed_types is not None and question_type not in allowed_types:
            reason = "type_not_requested"
        elif mode == "general" and (question.get("category") is not None
                                     or question.get("track") is not None):
            reason = "code_fields_on_general_question"
        elif mode == "code" and question.get("category") not in {None, question_type}:
            reason = "type_category_mismatch"
        elif mode == "code" and question.get("track") not in TRACKS:
            reason = "unknown_track"
        elif question.get("difficulty") not in DIFFICULTIES:
            reason = "unknown_difficulty"
        elif not all(isinstance(question.get(key), str) and question[key].strip()
                     for key in ("question", "difficulty_reason", "memory_requirement",
                                 "answer_target")):
            reason = "missing_question_or_rationale"
        elif _VACUOUS_QUESTION.search(question.get("question", "").strip()):
            reason = "vacuous_question"
        elif not isinstance(question.get("use_case"), str) or not question["use_case"].strip():
            reason = "missing_practical_use"
        elif (mode == "general" and question_type == "open-domain"
              and (not isinstance(question.get("external_knowledge"), str)
                   or not question["external_knowledge"].strip())):
            reason = "missing_external_knowledge"

        fact_ids = question.get("fact_ids")
        if not reason and not validate_sources(fact_ids, set(facts_by_id)):
            reason = "unknown_fact"

        selected_facts = [facts_by_id[fid] for fid in fact_ids or [] if fid in facts_by_id]
        mode_sources = scope_source_ids(scope, mode) if mode in QA_MODES else set()
        fact_sources = {source for fact in selected_facts for source in fact.get("sources", [])
                        if isinstance(source, str)}
        if not reason and (not fact_sources or not fact_sources <= mode_sources):
            reason = "fact_source_not_allowed_for_mode"
        if not reason:
            type_reason = _deterministic_type_guard(
                question, mode, question_type, scope, fact_sources, selected_facts)
            if type_reason in repairable_semantic:
                warnings.append(type_reason)
            else:
                reason = type_reason

        source_closure = directly_related_sources(scope, fact_sources, mode) if not reason else set()
        if not reason:
            point_reason = _validate_point_list(question.get("answer_points"), source_closure,
                                                True, "answer")
            if point_reason in repairable_semantic:
                warnings.append(point_reason)
            else:
                reason = point_reason
        answer_texts = {
            point.get("text", "").strip().casefold()
            for point in question.get("answer_points", [])
            if isinstance(point, dict) and isinstance(point.get("text"), str)
        }
        if not reason:
            point_reason = _validate_point_list(question.get("forbidden_points"), source_closure,
                                                False, "forbidden", answer_texts)
            if point_reason in repairable_semantic:
                warnings.append(point_reason)
            else:
                reason = point_reason
        if not reason:
            public_points = list(question.get("answer_points", [])) + list(
                question.get("forbidden_points", []))
            if (_INTERNAL_PUBLIC_ID.search(question.get("question", ""))
                    or any(isinstance(point, dict) and _INTERNAL_PUBLIC_ID.search(
                        point.get("text", "")) for point in public_points)):
                reason = "internal_id_in_public_text"
        if isinstance(qid, str):
            seen.add(qid)
        if reason:
            rejected.append({"question": question, "reason": reason})
            continue

        normalized = dict(question, qa_mode=mode, type=question_type,
                          use_case=question.get("use_case", question.get("memory_requirement")),
                          cutoff=scope.get("cutoff"), status="awaiting_semantic_review")
        if warnings:
            normalized["pre_review_warnings"] = sorted(set(warnings))
        if mode == "code":
            normalized["category"] = question_type
        accepted.append(normalized)
    return accepted, rejected


def validate_simple_candidates(document, facts, scope, qa_mode="code"):
    """Validate the small generation contract without inventing quality labels.

    Type, difficulty, and track are annotations performed only after evidence
    review.  This gate therefore checks only stable IDs, public text, and source
    closure; it deliberately does not reject compound wording or infer labels.
    """
    if qa_mode not in QA_MODES:
        raise ValueError("qa_mode must be general or code")
    if not isinstance(document, dict) or not isinstance(document.get("questions"), list):
        raise ValueError("LLM response must contain questions[]")
    facts_by_id = {fact.get("id"): fact for fact in facts
                   if isinstance(fact, dict) and isinstance(fact.get("id"), str)}
    allowed_sources = scope_source_ids(scope, qa_mode)

    def contains_source_id(text):
        if not isinstance(text, str):
            return False
        return any(re.search(
            r"(?<![A-Za-z0-9_])" + re.escape(source)
            + r"(?![A-Za-z0-9_])", text)
                   for source in allowed_sources if source)
    accepted, rejected, seen = [], [], set()
    for question in document["questions"]:
        reason = None
        if not isinstance(question, dict):
            rejected.append({"reason": "question_not_object"})
            continue
        qid = question.get("id")
        if not isinstance(qid, str) or not qid or qid in seen:
            reason = "invalid_or_duplicate_id"
        elif not isinstance(question.get("question"), str) or not question["question"].strip():
            reason = "missing_question"
        elif _has_unsupported_temporal_reference(question["question"]):
            reason = "unsupported_temporal_reference"
        seen.add(qid) if isinstance(qid, str) else None

        answer_points = question.get("answer_points")
        forbidden_points = question.get("forbidden_points", [])
        if not reason and (not isinstance(answer_points, list) or not answer_points):
            reason = "missing_answer_points"
        if not reason and not isinstance(forbidden_points, list):
            reason = "invalid_forbidden_points"
        texts = set()
        cited_sources = set()
        if not reason:
            for label, points in (("answer", answer_points),
                                  ("forbidden", forbidden_points)):
                for point in points:
                    if (not isinstance(point, dict)
                            or not isinstance(point.get("text"), str)
                            or not point["text"].strip()):
                        reason = "invalid_%s_point" % label
                        break
                    normalized = point["text"].strip().casefold()
                    if normalized in texts:
                        reason = "duplicate_or_conflicting_point"
                        break
                    if not validate_sources(point.get("sources"), allowed_sources):
                        reason = "invalid_%s_evidence" % label
                        break
                    texts.add(normalized)
                    cited_sources.update(point["sources"])
                if reason:
                    break
        if not reason:
            public_points = list(answer_points) + list(forbidden_points)
            if any(_has_unsupported_temporal_reference(point["text"])
                   for point in public_points):
                reason = "unsupported_temporal_reference"
            elif question.get("local_reference_leak") is True:
                reason = "local_reference_in_public_text"
            elif (_INTERNAL_PUBLIC_ID.search(question["question"])
                    or any(_INTERNAL_PUBLIC_ID.search(point["text"])
                           for point in public_points)):
                reason = "internal_id_in_public_text"
            elif (contains_source_id(question["question"])
                    or any(contains_source_id(point["text"])
                           for point in public_points)):
                reason = "internal_id_in_public_text"
        if reason:
            rejected.append({"question": question, "reason": reason})
            continue

        fact_ids = [fid for fid, fact in facts_by_id.items()
                    if cited_sources & set(fact.get("sources", []))]
        normalized = dict(
            ((key, question[key]) for key in
             ("id", "candidate_id", "model_id", "question", "answer_points",
              "forbidden_points", "origin_qa_mode", "evidence_group_id")
             if key in question),
            qa_mode=qa_mode,
            fact_ids=fact_ids,
            answer_target=question["question"].strip(),
            cutoff=scope.get("cutoff"),
            status="awaiting_semantic_review",
            annotation_status="pending",
        )
        accepted.append(normalized)
    return accepted, rejected


def _match_review_decisions(candidates, review, local_single_id=None):
    if not isinstance(review, dict) or not isinstance(review.get("reviews"), list):
        raise ValueError("Review must contain reviews[]")
    decisions = {}
    malformed = []
    aliases = {}
    for question in candidates:
        for alias in {question.get("id"), question.get("candidate_id"), question.get("model_id")} - {None, ""}:
            aliases.setdefault(alias, set()).add(question["id"])
    blocked = set()
    for item in review["reviews"]:
        if not isinstance(item, dict):
            malformed.append({"reason": "malformed_review_item", "review": item})
            continue
        review_id = item.get("id")
        matches = aliases.get(review_id, set()) if isinstance(review_id, str) else set()
        # A single-candidate, single-review call is unambiguous even when the
        # model emits its generic ``q1`` label instead of the supplied short
        # ID. Keep arbitrary unknown IDs rejected; only the conventional qN
        # placeholder is safely inferable in this narrow shape.
        inferred_single = False
        if local_single_id is not None:
            if (len(candidates) == len(review["reviews"]) == 1
                    and review_id in (None, "", local_single_id)):
                matches = {candidates[0]["id"]}
                inferred_single = review_id not in aliases
            else:
                matches = set()
        elif len(candidates) == len(review["reviews"]) == 1 and (
                review_id in (None, "") or
                (isinstance(review_id, str) and re.fullmatch(r"q\d+", review_id))):
            matches = {candidates[0]["id"]}
            inferred_single = review_id not in aliases
        if len(matches) != 1:
            malformed.append({"reason": "unknown_review_id", "review": item})
            blocked.update(matches)
            continue
        candidate_id = next(iter(matches))
        if candidate_id in decisions:
            malformed.append({"reason": "duplicate_review_id", "review": item})
            blocked.add(candidate_id)
            continue
        decisions[candidate_id] = dict(item, id=candidate_id, returned_id=review_id,
                                       id_inferred=inferred_single)
    return decisions, malformed, blocked


def _csv_tokens(value):
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if not isinstance(value, str):
        return None
    if value.strip().casefold() in {"", "none", "n/a", "-"}:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _assignments(value):
    """Parse the deliberately small ``key=value;...`` review protocol."""
    if isinstance(value, dict):
        return {str(key).strip(): str(item).strip() for key, item in value.items()
                if str(key).strip() and str(item).strip()}
    if not isinstance(value, str) or not value.strip():
        return None
    parsed = {}
    for item in value.split(";"):
        key, marker, assigned = item.strip().partition("=")
        if not marker or not key.strip() or not assigned.strip() or key.strip() in parsed:
            return None
        parsed[key.strip()] = assigned.strip()
    return parsed


def _structured_question_diagnostics(question, decision, allowed_sources=None,
                                     allowed_stage_ids=None):
    malformed, conflicts = [], []
    if decision.get("review_contract") != "structured_v2":
        return ["review_contract"], []
    for key in ("type_basis", "useful_task", "useful_decision", "answer_effect",
                "recommended_type"):
        if not isinstance(decision.get(key), str) or not decision[key].strip():
            malformed.append(key)
    sources = _csv_tokens(decision.get("necessary_source_ids"))
    if sources is None or not sources:
        malformed.append("necessary_source_ids")
    elif allowed_sources is not None and not set(sources) <= set(allowed_sources):
        malformed.append("necessary_source_ids_out_of_scope")
    mode, _ = _candidate_mode_and_type(question, None)
    recommended_type = decision.get("recommended_type")
    if isinstance(recommended_type, str):
        allowed_types = GENERAL_QA_TYPES if mode == "general" else CODE_QA_TYPES
        if recommended_type not in allowed_types:
            malformed.append("recommended_type")
        derived_type_correct = recommended_type == question.get("type", question.get("category"))
        if mode == "general" and isinstance(decision.get("type_correct"), bool) and (
                decision["type_correct"] != derived_type_correct):
            conflicts.append("type_correct")

    # Validate the reviewer's recommended shape. A coherent negative judgment
    # on an incorrectly labelled multi-hop question must remain repairable,
    # rather than becoming a malformed review merely because the original
    # candidate claimed the wrong type.
    if mode == "general" and recommended_type == "multi-hop":
        stages = _csv_tokens(decision.get("necessary_stage_ids"))
        if stages is None or len(set(stages)) < 2:
            malformed.append("necessary_stage_ids")
        elif allowed_stage_ids is not None and not set(stages) <= set(allowed_stage_ids):
            malformed.append("necessary_stage_ids_out_of_scope")
    if mode == "code":
        recommended_track = decision.get("recommended_track")
        if recommended_track not in TRACKS:
            malformed.append("recommended_track")
        elif isinstance(decision.get("type_correct"), bool):
            derived_type_correct = (
                decision.get("recommended_type") == question.get("type", question.get("category"))
                and recommended_track == question.get("track")
            )
            if decision["type_correct"] != derived_type_correct:
                conflicts.append("type_correct")
        history_fact = decision.get("history_only_fact")
        history_sources = _csv_tokens(decision.get("history_source_ids"))
        if history_sources is None:
            malformed.append("history_source_ids")
        elif allowed_sources is not None and not set(history_sources) <= set(allowed_sources):
            malformed.append("history_source_ids_out_of_scope")
        history_claimed = (isinstance(history_fact, str)
                           and history_fact.strip().casefold() not in {"", "none", "n/a", "-"})
        derived_history_required = history_claimed and bool(history_sources)
        if recommended_track == "history_core":
            expected = (False, True, True)
        else:
            expected = (True, False, False)
        actual = (decision.get("current_snapshot_alone_sufficient"),
                  decision.get("history_evidence_required"), derived_history_required)
        if all(isinstance(item, bool) for item in actual[:2]) and actual != expected:
            conflicts.append("history_requirement")
        derived_history_correct = recommended_track == question.get("track")
        if isinstance(decision.get("history_requirement_correct"), bool) and (
                decision["history_requirement_correct"] != derived_history_correct):
            conflicts.append("history_requirement_correct")
    return sorted(set(malformed)), sorted(set(conflicts))


def _structured_answer_structure_diagnostics(question, decision):
    malformed, conflicts = [], []
    if decision.get("review_contract") != "structured_v2":
        return ["review_contract"], [], {}
    if not isinstance(decision.get("reason"), str) or not decision["reason"].strip():
        malformed.append("reason")
    requirements = _assignments(decision.get("answer_requirements"))
    coverage = _assignments(decision.get("requirement_coverage"))
    claims = _assignments(decision.get("point_claims"))
    if not requirements or any(not re.fullmatch(r"R\d+", key) for key in requirements):
        malformed.append("answer_requirements")
    if not coverage or (requirements and set(coverage) != set(requirements)):
        malformed.append("requirement_coverage")

    point_ids = (["A%d" % (index + 1) for index in range(len(question.get("answer_points", [])))]
                 + ["F%d" % (index + 1) for index in range(len(question.get("forbidden_points", [])))])
    point_text = {}
    for prefix, points in (("A", question.get("answer_points", [])),
                           ("F", question.get("forbidden_points", []))):
        for index, point in enumerate(points):
            point_id = "%s%d" % (prefix, index + 1)
            point_text[point_id] = re.sub(r"\s+", "", str(point.get("text", ""))).casefold()
    answer_ids = {point for point in point_ids if point.startswith("A")}
    coverage_targets = {}
    if coverage:
        for requirement_id, value in coverage.items():
            targets = _csv_tokens(value)
            coverage_targets[requirement_id] = targets
            if (targets is None or not targets
                    or ("MISSING" in targets and targets != ["MISSING"])
                    or any(target != "MISSING" and target not in answer_ids
                           for target in targets)):
                malformed.append("requirement_coverage_target")
    derived_complete = bool(coverage_targets) and all(
        targets and all(target in answer_ids for target in targets)
        for targets in coverage_targets.values())
    if isinstance(decision.get("answer_complete"), bool) and (
            decision["answer_complete"] != derived_complete):
        conflicts.append("answer_complete")

    claim_groups = {point: [] for point in point_ids}
    if not claims:
        malformed.append("point_claims")
    else:
        for key, text in claims.items():
            match = re.fullmatch(r"([AF]\d+)\.(\d+)", key)
            if not match or match.group(1) not in claim_groups or not text:
                malformed.append("point_claims_key")
                continue
            point_id = match.group(1)
            claim_span = re.sub(r"\s+", "", text).casefold()
            if not claim_span or claim_span not in point_text.get(point_id, ""):
                malformed.append("point_claim_not_verbatim_span")
                continue
            claim_groups[point_id].append(int(match.group(2)))
        if any(sorted(numbers) != list(range(1, len(numbers) + 1))
               for numbers in claim_groups.values()):
            malformed.append("point_claims_sequence")
    derived_atomic = bool(claim_groups) and all(len(numbers) == 1
                                                for numbers in claim_groups.values())
    if isinstance(decision.get("atomic_points_correct"), bool) and (
            decision["atomic_points_correct"] != derived_atomic):
        conflicts.append("atomic_points_correct")

    derived = {
        "answer_complete": derived_complete,
        "atomic_points_correct": derived_atomic,
    }
    for key, value in derived.items():
        if isinstance(decision.get(key), bool) and decision[key] != value:
            conflicts.append(key)
    return sorted(set(malformed)), sorted(set(conflicts)), derived


def _structured_answer_evidence_diagnostics(question, decision, allowed_sources=None):
    malformed, conflicts = [], []
    evidence = _assignments(decision.get("point_evidence"))
    point_ids = (["A%d" % (index + 1) for index in range(len(question.get("answer_points", [])))]
                 + ["F%d" % (index + 1) for index in range(len(question.get("forbidden_points", [])))])
    declared_sources = {}
    for prefix, points in (("A", question.get("answer_points", [])),
                           ("F", question.get("forbidden_points", []))):
        for index, point in enumerate(points):
            declared_sources["%s%d" % (prefix, index + 1)] = set(point.get("sources", []))
    evidence_status = {}
    if not evidence or set(evidence) != set(point_ids):
        malformed.append("point_evidence")
    else:
        for point_id, value in evidence.items():
            status, marker, source_text = value.partition("@")
            sources = _csv_tokens(source_text) if marker else None
            if status not in {"supported", "unsupported", "contradicted", "stale"}:
                malformed.append("point_evidence_status")
                continue
            if sources is None or not sources:
                malformed.append("point_evidence_sources")
                continue
            if allowed_sources is not None and not set(sources) <= set(allowed_sources):
                malformed.append("point_evidence_out_of_scope")
                continue
            if status == "supported" and not set(sources) & declared_sources.get(point_id, set()):
                malformed.append("point_evidence_missing_citation")
                continue
            evidence_status[point_id] = status
    derived_supported = bool(evidence_status) and all(
        status == ("supported" if point_id.startswith("A") else "contradicted")
        for point_id, status in evidence_status.items())
    derived_consistent = bool(evidence_status) and all(
        status != "stale" for status in evidence_status.values())
    derived = {
        "evidence_supported": derived_supported,
        "version_consistent": derived_consistent,
    }
    for key, value in derived.items():
        if isinstance(decision.get(key), bool) and decision[key] != value:
            conflicts.append(key)
    return sorted(set(malformed)), sorted(set(conflicts)), derived


def _structured_answer_diagnostics(question, decision, allowed_sources=None):
    structure_missing, structure_conflicts, structure_derived = (
        _structured_answer_structure_diagnostics(question, decision))
    evidence_missing, evidence_conflicts, evidence_derived = (
        _structured_answer_evidence_diagnostics(question, decision, allowed_sources))
    return (sorted(set(structure_missing + evidence_missing)),
            sorted(set(structure_conflicts + evidence_conflicts)),
            dict(structure_derived, **evidence_derived))


def apply_answer_structure_review(candidates, review):
    """Validate evidence-blind completeness and point atomicity."""
    decisions, malformed, blocked = _match_review_decisions(candidates, review)
    rejected = list(malformed)
    kept = []
    for question in candidates:
        decision = decisions.get(question["id"], {})
        if not decision or question["id"] in blocked:
            kept.append(dict(question, status="needs_review",
                             review_error="structure_review_unmatched"))
            continue
        missing, conflicts, derived = _structured_answer_structure_diagnostics(
            question, decision)
        for key, value in derived.items():
            decision.setdefault(key, value)
        if missing or conflicts:
            kept.append(dict(
                question, status="needs_review", structure_review=decision,
                review_error="incomplete_or_conflicting_structure_review",
                missing_review_fields=missing, review_conflicts=conflicts))
        elif all(decision.get(key) is True
                 for key in ("answer_complete", "atomic_points_correct")):
            kept.append(dict(question, status="awaiting_evidence_review",
                             structure_review=decision))
        else:
            rejected.append({
                "question": dict(question, status="rejected"),
                "reason": "semantic_answer_structure_failed",
                "failed_checks": [key for key in
                                  ("answer_complete", "atomic_points_correct")
                                  if decision.get(key) is False],
                "review": decision,
            })
    return kept, rejected


def apply_simple_completeness_review(candidates, review):
    """Apply the evidence-blind simple completeness contract."""
    decisions, malformed, blocked = _match_review_decisions(
        candidates, review, local_single_id="q1")
    rejected = list(malformed)
    kept = []
    for question in candidates:
        decision = decisions.get(question["id"], {})
        if not decision or question["id"] in blocked:
            kept.append(dict(question, status="needs_review",
                             review_error="completeness_review_unmatched"))
            continue
        unexpected = set(decision) - {
            "id", "returned_id", "id_inferred", "review_contract",
            "completeness", "missing",
        }
        completeness = decision.get("completeness")
        missing = decision.get("missing")
        if (decision.get("review_contract") != "simple_v1" or unexpected
                or completeness not in {"complete", "missing", "uncertain"}
                or (completeness == "missing" and (
                    not isinstance(missing, str) or not missing.strip()))):
            kept.append(dict(
                question, status="needs_review", completeness_review=decision,
                review_error="invalid_completeness_review"))
        elif completeness == "complete":
            kept.append(dict(question, status="awaiting_evidence_review",
                             completeness_review=decision))
        elif completeness == "uncertain":
            kept.append(dict(question, status="needs_review",
                             completeness_review=decision,
                             review_error="uncertain_completeness"))
        else:
            rejected.append({
                "question": dict(question, status="rejected"),
                "reason": "semantic_answer_incomplete",
                "failed_checks": ["answer_complete"],
                "review": dict(decision, answer_complete=False),
            })
    return kept, rejected


def apply_simple_atomicity_review(candidates, review):
    """Apply one evidence-blind per-point atomicity decision."""
    decisions, malformed, blocked = _match_review_decisions(
        candidates, review, local_single_id="q1")
    rejected = list(malformed)
    kept = []
    for question in candidates:
        decision = decisions.get(question["id"], {})
        if not decision or question["id"] in blocked:
            kept.append(dict(question, status="needs_review",
                             review_error="atomicity_review_unmatched"))
            continue
        unexpected = set(decision) - {
            "id", "returned_id", "id_inferred", "review_contract",
            "point_atomicity",
        }
        assignments = _assignments(decision.get("point_atomicity"))
        point_ids = simple_point_ids(question)
        invalid = (
            decision.get("review_contract") != "simple_atomicity_v1"
            or bool(unexpected)
            or assignments is None
            or set(assignments) != set(point_ids)
            or any(value not in SIMPLE_ATOMICITY_STATES
                   for value in (assignments or {}).values())
        )
        if invalid:
            kept.append(dict(
                question, status="needs_review", atomicity_review=decision,
                review_error="invalid_atomicity_review"))
        elif any(value == "compound" for value in assignments.values()):
            rejected.append({
                "question": dict(question, status="rejected",
                                 atomicity_review=decision),
                "reason": "semantic_answer_atomicity_failed",
                "failed_checks": ["atomic_points_correct"],
                "review": dict(decision, atomic_points_correct=False),
            })
        elif any(value == "uncertain" for value in assignments.values()):
            kept.append(dict(
                question, status="needs_review", atomicity_review=decision,
                review_error="uncertain_atomicity"))
        else:
            kept.append(dict(
                question, status="awaiting_completeness_review",
                atomicity_review=decision))
    return kept, rejected


def apply_simple_relevance_review(candidates, review):
    """Drop independently judged side points before structural review."""
    decisions, malformed, blocked = _match_review_decisions(
        candidates, review, local_single_id="q1")
    rejected = list(malformed)
    kept = []
    for question in candidates:
        decision = decisions.get(question["id"], {})
        if not decision or question["id"] in blocked:
            kept.append(dict(question, status="needs_review",
                             review_error="relevance_review_unmatched"))
            continue
        unexpected = set(decision) - {
            "id", "returned_id", "id_inferred", "review_contract",
            "point_relevance",
        }
        relevance_value = decision.get("point_relevance")
        if isinstance(relevance_value, str):
            relevance_value = re.sub(r",\s*(?=[AF]\d+\s*=)", ";",
                                     relevance_value)
        assignments = _assignments(relevance_value)
        point_ids = simple_point_ids(question)
        invalid = (
            decision.get("review_contract") != "simple_relevance_v1"
            or bool(unexpected)
            or assignments is None
            or set(assignments) != set(point_ids)
            or any(value not in {"direct", "extra", "uncertain"}
                   for value in (assignments or {}).values())
        )
        if invalid:
            kept.append(dict(question, status="needs_review",
                             relevance_review=decision,
                             review_error="invalid_relevance_review"))
            continue
        if any(value == "uncertain" for value in assignments.values()):
            kept.append(dict(question, status="needs_review",
                             relevance_review=decision,
                             review_error="uncertain_relevance"))
            continue
        static_extra_ids = set()
        question_text = question.get("question", "")
        if (question.get("qa_mode") == "code"
                and question.get("type", question.get("category")) == "behavior_inference"
                and _FLOW_QUESTION.search(question_text)
                and not _SIGNATURE_REQUEST.search(question_text)):
            for index, point in enumerate(question.get("answer_points", []), 1):
                text = point.get("text", "") if isinstance(point, dict) else ""
                if (_SIGNATURE_ONLY_POINT.search(text)
                        and not _FLOW_ACTION_POINT.search(text)):
                    static_extra_ids.add("A%d" % index)
        filtered = dict(question, relevance_review=decision)
        if static_extra_ids:
            filtered["static_relevance_filtered"] = sorted(static_extra_ids)
        filtered["answer_points"] = [
            point for index, point in enumerate(question.get("answer_points", []), 1)
            if (assignments.get("A%d" % index) == "direct"
                and "A%d" % index not in static_extra_ids)]
        filtered["forbidden_points"] = [
            point for index, point in enumerate(question.get("forbidden_points", []), 1)
            if assignments.get("F%d" % index) == "direct"]
        if not filtered["answer_points"]:
            rejected.append({
                "question": dict(question, status="rejected",
                                 relevance_review=decision),
                "reason": "semantic_answer_irrelevant",
                "failed_checks": ["answer_relevant"],
                "review": dict(decision, answer_relevant=False),
            })
            continue
        filtered["status"] = "awaiting_atomicity_review"
        kept.append(filtered)
    return kept, rejected


def apply_code_distinctiveness_review(candidates, review):
    """Apply the code-only answer-basis selection contract."""
    decisions, malformed, blocked = _match_review_decisions(
        candidates, review, local_single_id="q1")
    rejected = list(malformed)
    kept = []
    for question in candidates:
        decision = decisions.get(question["id"], {})
        if not decision or question["id"] in blocked:
            kept.append(dict(
                question, status="needs_review",
                review_error="code_distinctiveness_review_unmatched"))
            continue
        unexpected = set(decision) - {
            "id", "returned_id", "id_inferred", "review_contract",
            "answer_basis", "target_alignment",
        }
        basis = decision.get("answer_basis")
        requires_alignment = bool(question.get("_generation_focus"))
        alignment = decision.get("target_alignment")
        if (decision.get("review_contract") != "code_distinctiveness_v1"
                or unexpected or basis not in CODE_DISTINCTIVENESS_BASES):
            kept.append(dict(
                question, status="needs_review",
                distinctiveness_review=decision,
                review_error="invalid_code_distinctiveness_review"))
        elif requires_alignment and alignment not in {
                "aligned", "mixed", "drifted", "uncertain"}:
            kept.append(dict(
                question, status="needs_review",
                distinctiveness_review=decision,
                review_error="missing_code_target_alignment"))
        elif requires_alignment and alignment in {"mixed", "drifted"}:
            rejected.append({
                "question": dict(
                    question, status="rejected",
                    distinctiveness_review=decision),
                "reason": "code_answer_basis_target_mismatch",
                "failed_checks": ["code_answer_target_aligned"],
                "review": dict(decision, code_answer_target_aligned=False),
            })
        elif requires_alignment and alignment == "uncertain":
            kept.append(dict(
                question, status="needs_review",
                distinctiveness_review=decision,
                review_error="uncertain_code_target_alignment"))
        elif basis == "D":
            rejected.append({
                "question": dict(
                    question, status="rejected",
                    distinctiveness_review=decision),
                "reason": "code_answer_not_distinctive",
                "failed_checks": ["code_answer_distinctive"],
                "review": dict(decision, code_answer_distinctive=False),
            })
        elif basis not in {
                "history_tracking": {"A"},
                "behavior_inference": {"C"},
                "failure_diagnosis": {"B", "C"},
                "fact_recall": {"B"},
        }.get(question.get("type"), set()):
            rejected.append({
                "question": dict(
                    question, status="rejected",
                    distinctiveness_review=decision),
                "reason": "code_answer_basis_target_mismatch",
                "failed_checks": ["code_answer_target_aligned"],
                "review": dict(decision, code_answer_target_aligned=False),
            })
        else:
            kept.append(dict(
                question, status="awaiting_atomicity_review",
                distinctiveness_review=decision))
    return kept, rejected


def _simple_evidence_assignments(decision, expected_ids, allowed_sources=None,
                                 require_complete=True):
    """Parse and validate the shared simple point-evidence schema."""
    malformed_fields = []
    unexpected = set(decision) - {
        "id", "returned_id", "id_inferred", "review_contract", "point_evidence",
    }
    if decision.get("review_contract") != "simple_v1":
        malformed_fields.append("review_contract")
    if unexpected:
        malformed_fields.append("unexpected_fields")
    raw_evidence = decision.get("point_evidence")
    evidence = ({} if not require_complete and isinstance(raw_evidence, str)
                and not raw_evidence.strip()
                else _assignments(raw_evidence))
    if (evidence is None
            or (require_complete and set(evidence) != set(expected_ids))
            or (not require_complete and not set(evidence) <= set(expected_ids))):
        malformed_fields.append("point_evidence")
        return evidence, {}, sorted(set(malformed_fields))
    statuses = {}
    for point_id, value in evidence.items():
        status, marker, source_text = value.partition("@")
        sources = _csv_tokens(source_text) if marker else None
        if status not in SIMPLE_EVIDENCE_STATES:
            malformed_fields.append("point_evidence_status")
        elif status != "insufficient" and (sources is None or not sources):
            malformed_fields.append("point_evidence_sources")
        elif (sources and allowed_sources is not None
                and not set(sources) <= set(allowed_sources)):
            malformed_fields.append("point_evidence_out_of_scope")
        else:
            statuses[point_id] = status
    return evidence, statuses, sorted(set(malformed_fields))


def _simple_evidence_fragment(candidates, review, expected_ids,
                              allowed_sources=None):
    """Return one valid partial simple review, or ``None`` when malformed."""
    decisions, malformed, blocked = _match_review_decisions(
        candidates, review, local_single_id="q1")
    if len(candidates) != 1 or malformed:
        return None
    question = candidates[0]
    decision = decisions.get(question["id"])
    if not decision or question["id"] in blocked:
        return None
    evidence, _, malformed_fields = _simple_evidence_assignments(
        decision, expected_ids, allowed_sources, require_complete=False)
    if malformed_fields:
        return None
    return decision, evidence


def simple_evidence_review_omissions(candidates, review, allowed_sources=None):
    """Return ordered omitted point IDs for a valid partial review.

    ``None`` means the response is malformed rather than merely incomplete.
    """
    if len(candidates) != 1:
        return None
    question = candidates[0]
    point_ids = simple_point_ids(question)
    fragment = _simple_evidence_fragment(
        candidates, review, point_ids, allowed_sources)
    if fragment is None:
        return None
    _, evidence = fragment
    return [point_id for point_id in point_ids if point_id not in evidence]


def merge_simple_evidence_reviews(candidates, initial, supplement,
                                  allowed_sources=None):
    """Merge valid supplemental decisions without replacing initial ones."""
    missing = simple_evidence_review_omissions(
        candidates, initial, allowed_sources)
    if not missing:
        return None
    initial_fragment = _simple_evidence_fragment(
        candidates, initial, simple_point_ids(candidates[0]), allowed_sources)
    supplement_fragment = _simple_evidence_fragment(
        candidates, supplement, simple_point_ids(candidates[0]), allowed_sources)
    if initial_fragment is None or supplement_fragment is None:
        return None
    initial_decision, initial_evidence = initial_fragment
    _, supplement_evidence = supplement_fragment
    merged = dict(initial_evidence)
    merged.update({point_id: supplement_evidence[point_id]
                   for point_id in missing if point_id in supplement_evidence})
    point_ids = simple_point_ids(candidates[0])
    returned_id = initial_decision.get("returned_id")
    return {"reviews": [{
        "id": returned_id if isinstance(returned_id, str) else "q1",
        "review_contract": "simple_v1",
        "point_evidence": ";".join(
            "%s=%s" % (point_id, merged[point_id])
            for point_id in point_ids if point_id in merged),
    }]}


def apply_simple_evidence_review(candidates, review, allowed_sources=None):
    """Apply per-point truth/version decisions without other quality judgments."""
    decisions, malformed, blocked = _match_review_decisions(
        candidates, review, local_single_id="q1")
    rejected = list(malformed)
    kept = []
    for question in candidates:
        decision = decisions.get(question["id"], {})
        if not decision or question["id"] in blocked:
            kept.append(dict(question, status="needs_review",
                             review_error="evidence_review_unmatched"))
            continue
        _, statuses, malformed_fields = _simple_evidence_assignments(
            decision, simple_point_ids(question), allowed_sources)
        if malformed_fields:
            kept.append(dict(
                question, status="needs_review", evidence_review=decision,
                review_error="invalid_evidence_review",
                missing_review_fields=sorted(set(malformed_fields))))
            continue
        if any(status == "insufficient" for status in statuses.values()):
            kept.append(dict(question, status="needs_review", evidence_review=decision,
                             review_error="insufficient_evidence"))
            continue
        evidence_supported = all(
            status == ("supported" if point_id.startswith("A") else "contradicted")
            for point_id, status in statuses.items())
        version_consistent = all(status != "stale" for status in statuses.values())
        semantic = dict(decision, evidence_supported=evidence_supported,
                        version_consistent=version_consistent)
        if evidence_supported and version_consistent:
            kept.append(dict(question, status="approved", evidence_review=decision,
                             review=semantic))
        else:
            failed = []
            if not evidence_supported:
                failed.append("evidence_supported")
            if not version_consistent:
                failed.append("version_consistent")
            rejected.append({
                "question": dict(question, status="rejected"),
                "reason": "semantic_evidence_review_failed",
                "failed_checks": failed,
                "review": semantic,
            })
    return kept, rejected


def apply_question_review(candidates, review, allowed_sources=None, allowed_stage_ids=None):
    """Apply the first split-review lane without approving a candidate."""
    decisions, malformed, blocked = _match_review_decisions(candidates, review)
    rejected = list(malformed)
    kept = []
    checks = ("unambiguous", "difficulty_justified", "not_answer_leaking",
              "natural_wording", "practical_useful", "type_correct")
    for question in candidates:
        decision = decisions.get(question["id"], {})
        if not decision or question["id"] in blocked:
            kept.append(dict(question, status="needs_review", review_error="review_unmatched"))
            continue
        approval_checks = list(checks)
        required = list(approval_checks)
        mode, _ = _candidate_mode_and_type(question, None)
        if mode == "code":
            approval_checks.append("history_requirement_correct")
            required += ["history_requirement_correct", "current_snapshot_alone_sufficient",
                         "history_evidence_required"]
        if question.get("type") == "open-domain":
            approval_checks += ["external_knowledge_separated", "external_knowledge_necessary"]
            required += ["external_knowledge_separated", "external_knowledge_necessary"]
        if question.get("type") == "adversarial":
            approval_checks.append("full_range_checked")
            required += ["full_range_checked"]
        missing = [key for key in required if not isinstance(decision.get(key), bool)]
        structure_missing, conflicts = _structured_question_diagnostics(
            question, decision, allowed_sources, allowed_stage_ids)
        if missing or structure_missing or conflicts:
            kept.append(dict(question, status="needs_review", question_review=decision,
                             review_error="incomplete_or_conflicting_question_review",
                             missing_review_fields=missing + structure_missing,
                             review_conflicts=conflicts))
        elif all(decision.get(key) is True for key in approval_checks):
            kept.append(dict(question, status="awaiting_answer_review",
                             question_review=decision))
        else:
            rejected.append({"question": dict(question, status="rejected"),
                             "reason": "semantic_question_review_failed",
                             "failed_checks": [key for key in approval_checks
                                               if decision.get(key) is False],
                             "review": decision})
    return kept, rejected


def apply_review(candidates, review, require_structured=False, allowed_sources=None,
                 allowed_stage_ids=None):
    decisions, malformed, blocked = _match_review_decisions(candidates, review)
    rejected = list(malformed)
    kept = []
    for question in candidates:
        decision = decisions.get(question["id"], {})
        if not decision or question["id"] in blocked:
            reason = "review_id_ambiguous" if question["id"] in blocked else "review_unmatched"
            kept.append(dict(question, status="needs_review", review_error=reason))
            continue
        mode, _ = _candidate_mode_and_type(question, None)
        common_checks = ("evidence_supported", "version_consistent", "unambiguous",
                         "difficulty_justified", "not_answer_leaking", "natural_wording",
                         "practical_useful", "answer_complete", "atomic_points_correct",
                         "type_correct")
        causal = (question.get("type", question.get("category")) in
                  {"behavior_inference", "failure_diagnosis"}
                  or bool(re.search(r"为什么|为何|原因|导致|why|cause", question.get("question", ""), re.I)))
        if causal:
            common_checks += ("causal_support",)
        required = list(common_checks)
        if mode == "code":
            required += ["history_requirement_correct", "current_snapshot_alone_sufficient",
                         "history_evidence_required"]
        if question.get("type") == "open-domain":
            required += ["external_knowledge_separated", "external_knowledge_necessary"]
        if question.get("type") == "adversarial":
            required += ["full_range_checked"]
        structure_missing, structure_conflicts = [], []
        if require_structured:
            question_missing, question_conflicts = _structured_question_diagnostics(
                question, decision, allowed_sources, allowed_stage_ids)
            answer_missing, answer_conflicts, derived_answer = _structured_answer_diagnostics(
                question, decision, allowed_sources)
            for key, value in derived_answer.items():
                decision.setdefault(key, value)
            if causal:
                bridge = decision.get("causal_bridge")
                if isinstance(bridge, str) and bridge.strip():
                    derived_causal = bridge.strip().casefold() not in {"none", "n/a", "-"}
                    if (isinstance(decision.get("causal_support"), bool)
                            and decision["causal_support"] != derived_causal):
                        answer_conflicts.append("causal_support")
                    decision.setdefault("causal_support", derived_causal)
                elif not isinstance(decision.get("causal_support"), bool):
                    answer_missing.append("causal_bridge")
            structure_missing = question_missing + answer_missing
            structure_conflicts = question_conflicts + answer_conflicts
        missing = [key for key in required if not isinstance(decision.get(key), bool)]
        type_correct = decision.get("type_correct") is True
        mode_correct = decision.get("qa_mode_correct", True) is True
        review_valid = (all(decision.get(key) is True for key in common_checks)
                        and type_correct and mode_correct)
        if mode == "code":
            history_fields_present = all(isinstance(decision.get(key), bool) for key in
                                         ("current_snapshot_alone_sufficient",
                                          "history_evidence_required"))
            track_consistent = (
                (question.get("track") == "history_core"
                 and decision.get("current_snapshot_alone_sufficient") is False
                 and decision.get("history_evidence_required") is True)
                or
                (question.get("track") == "inference_control"
                 and decision.get("current_snapshot_alone_sufficient") is True
                 and decision.get("history_evidence_required") is False)
            )
            review_valid = (review_valid
                            and decision.get("history_requirement_correct") is True
                            and history_fields_present and track_consistent)
        elif question.get("type") == "open-domain":
            review_valid = (review_valid and decision.get("external_knowledge_separated") is True
                            and decision.get("external_knowledge_necessary") is True)
        elif question.get("type") == "adversarial":
            review_valid = review_valid and decision.get("full_range_checked") is True
        rationale = decision.get("reason", "")
        if isinstance(rationale, str):
            contradictory = (
                decision.get("history_requirement_correct") is True
                and re.search(r"应为\s*false|should be false|不符合.*(?:history|历史)|不成立",
                              rationale, re.I)
            ) or (
                decision.get("history_requirement_correct") is False
                and re.search(r"符合|正确|consistent|satisfied", rationale, re.I)
                and not re.search(r"不符合|不正确|not\s+(?:consistent|satisfied)", rationale, re.I)
            )
            if contradictory:
                decision["review_consistency"] = "reason_boolean_conflict"
                review_valid = False
        if (isinstance(decision.get("type_correct"), bool)
                and isinstance(decision.get("category_correct"), bool)
                and decision.get("category_correct") != decision.get("type_correct")):
            decision["review_consistency"] = "type_boolean_conflict"
        if structure_conflicts:
            decision["review_consistency"] = "structured_boolean_conflict"
        if missing or structure_missing or decision.get("review_consistency"):
            kept.append(dict(question, status="needs_review", review=decision,
                             review_error="incomplete_or_conflicting_review",
                             missing_review_fields=missing + structure_missing,
                             review_conflicts=structure_conflicts))
            continue
        if (review_valid and isinstance(decision.get("reason"), str)
                and decision["reason"].strip()):
            kept.append(dict(question, status="approved", review=decision))
        else:
            rejected.append({"question": dict(question, status="rejected"),
                             "reason": "semantic_review_failed",
                             "failed_checks": [key for key in common_checks if decision.get(key) is False],
                             "review": decision})
    return kept, rejected
