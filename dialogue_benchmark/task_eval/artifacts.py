"""Private task artifacts and immutable source snapshots."""

import hashlib
import json
import re
import shutil
from pathlib import Path

from ..storage import load
from ..protocol import QA_TYPES, MEMORY_TYPES
from ..security import credential_detected


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.chmod(0o600)
    temp.replace(path)


def read(path):
    return load(path)


def qa_fingerprint(question):
    return hashlib.sha256(json.dumps(question, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def install_candidate_fixture(directory):
    """Provide the same repository location in authoring and scored executions."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "conftest.py").write_text(
        'from pathlib import Path\nimport pytest\n\n'
        '@pytest.fixture(scope="session")\ndef candidate_root():\n'
        '    return Path("/workspace/candidate")\n', encoding="utf-8")


def copy_tree(source, target, *, include_caches=False):
    source, target = Path(source), Path(target)
    ignored_artifact_dirs = {".tox", ".venv", "_build"}
    if any(p.is_symlink() and not any(part in ignored_artifact_dirs
                                      for part in p.relative_to(source).parts)
           for p in source.rglob("*")):
        raise ValueError("Snapshot contains a symlink")
    ignored = [".git", "__pycache__", ".pytest_cache", ".venv", ".tox", "_build", "*.pyc"]
    if not include_caches:
        ignored += [".mypy_cache", ".ruff_cache"]
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(*ignored))


def fingerprint(root):
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and not any(x in {"__pycache__", ".pytest_cache", ".git"}
                                      for x in path.parts):
            digest.update(str(path.relative_to(root)).encode() + b"\0")
            digest.update(b"x" if path.stat().st_mode & 0o111 else b"-")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def write_diff(base, candidate, output):
    from .versions import export_change
    output = Path(output)
    receipt = export_change(base, candidate, output.parent)
    if output.name != receipt["patch"]:
        (output.parent / receipt["patch"]).rename(output)
        receipt["patch"] = output.name
        save(output.parent / "version.json", receipt)
    return receipt["changed_files"]


def _point_text(point):
    return point if isinstance(point, str) else point.get("text", point.get("claim", ""))


def public_global_agreements(records):
    """Index only public wording that explicitly gives a project-wide scope."""
    from ..external import relevant_public_history

    scope = re.compile(r"(?<!\w)(?:所有任务|所有代码)|全项目|整个项目|项目统一|全局(?:编码|代码|交付)?(?:约定|规范|规则)|"
                       r"\b(?:all tasks|all code)\b(?!\s+(?:for|in|under|during|within|of)\b)|"
                       r"\b(?:project[- ]wide|across the project|globally)\b", re.I)
    result = []
    cutoff = max((record.get("order", 0) for record in records or []), default=0)
    for record in records or []:
        if record.get("role") != "user":
            continue
        text = record.get("text", "")
        for line in text.splitlines():
            if scope.search(line) and line.strip() and not credential_detected(line):
                related = relevant_public_history(records, [record], line, cutoff)
                result.append({"text": line.strip(),
                               "sources": [record.get("original_id", record.get("id"))],
                               "context": "\n".join(row.get("text", "") for row in related)})
    return result


def _input_rejection(question, candidate, request, records, *, require_sources):
    """Apply hard content/source guards independently of review status."""
    for document in (question, candidate):
        if credential_detected(json.dumps(document, ensure_ascii=False)):
            return "credential_detected"
    if not require_sources:
        return None
    points = candidate.get("answer_points", [])
    if not points or any(not isinstance(point, dict) or not _point_text(point).strip()
                         or not point.get("sources") for point in points):
        return "missing_answer_sources"
    references = request.get("ref_to_source", {})
    payload = request.get("payload", {})
    known = set(references.values())
    known.update(row.get("id") for row in payload.get("scope", {}).get("dialogue", []))
    cited = {source for key in ("answer_points", "forbidden_points")
             for point in candidate.get(key, []) if isinstance(point, dict)
             for source in point.get("sources", [])}
    if any(not isinstance(source, str) or source not in known for source in cited):
        return "unresolved_answer_source"
    public_ids = {row.get("id") for row in records}
    if records and any(source.partition("#fragment-")[0] not in public_ids for source in cited):
        return "source_missing_from_dialogue"
    represented = {references[row["reference"]] for row in payload.get("materials", [])
                   if row.get("reference") in references}
    represented.update(row.get("id") for row in payload.get("scope", {}).get("dialogue", []))
    if not records and cited - represented:
        return "source_missing_from_generation_input"
    public_points = [_point_text(point) for point in question.get("answer_points", [])]
    if public_points != [_point_text(point) for point in points]:
        return "answer_differs_from_source_candidate"
    return None


def qa_inputs(qa_run, *, include_provisional=False, diagnostics=None):
    """Join QA questions to their saved generation request.

    Approved questions are always eligible.  A caller may explicitly opt into
    the provisional pool so a local review warning does not discard otherwise
    usable questions; the original status is retained on the task record and
    never turns into a public approval.
    """
    qa_run = Path(qa_run)
    public = read(qa_run / "qa-public.json")["questions"]
    if include_provisional:
        candidates = read(qa_run / "qa-candidates.json") if (
            qa_run / "qa-candidates.json").is_file() else {}
        provisional = [
            question for question in candidates.get("questions", [])
            if isinstance(question, dict)
            and question.get("status") in {"approved", "needs_review", "provisional"}
        ]
        # Keep the publishable projection authoritative when it already has
        # the question.  Only use the extra provisional pool to fill the gap.
        known = {question.get("id") for question in public}
        public = list(public) + [
            question for question in provisional if question.get("id") not in known
        ]
    run_manifest = read(qa_run / "manifest.json") if (qa_run / "manifest.json").exists() else {}
    types = MEMORY_TYPES if run_manifest.get("qa_source") == "external" else QA_TYPES
    # Type inference is intentionally non-blocking.  A candidate whose
    # answer evidence is valid but cannot be classified yet is retained as
    # ``unknown``/``unresolved`` for review instead of dropping the whole
    # route.  Concrete values still must belong to the selected QA track.
    if any(question.get("type") not in types
           and question.get("type") != "unknown"
           and not (question.get("type") is None
                    and question.get("type_status") == "unresolved")
           for question in public):
        raise ValueError("QA types do not match the source mode; regenerate QA")
    normalized_path = qa_run / "normalized.json"
    normalized = read(normalized_path) if normalized_path.exists() else []
    public_records = None
    if normalized and all(r.get("input_schema") == "model-visible-dialogue-v1" for r in normalized):
        public_records = normalized
    elif normalized and all(
            isinstance(r, dict) and r.get("kind") == "message"
            and r.get("role") in {"user", "assistant"}
            and isinstance(r.get("text"), str) and isinstance(r.get("id"), str)
            for r in normalized):
        # A plain user/assistant dialogue is still a public history.  Older
        # list inputs predate the envelope marker and therefore do not carry
        # original_id/input_schema; add only the local identity needed by
        # task_eval, never tool or controller records.
        public_records = [dict(r, original_id=r.get("original_id", r["id"]),
                               input_schema="model-visible-dialogue-v1")
                          for r in normalized]
    requests = {}
    for path in sorted((qa_run / "stages").glob("*raw-candidates*")):
        input_path = path.with_name(path.name.replace("raw-candidates", "qa-input"))
        if not input_path.exists():
            continue
        for question in read(path).get("questions", []):
            requests[question["id"]] = (input_path, question)
    audit_path = qa_run / "qa-audit.json"
    reviewed = {question["id"]: question
                for question in (read(audit_path).get("questions", []) if audit_path.exists() else [])
                if question.get("status") == "approved"}
    result = []
    diagnostics = diagnostics if diagnostics is not None else []
    external_path = qa_run / "external-events.json"
    external_events = read(external_path).get("events", []) if external_path.is_file() else []
    global_agreements = public_global_agreements(public_records)
    for question in public:
        eligible = question.get("status") == "approved" or (
            include_provisional and question.get("status") in {"needs_review", "provisional"}
        )
        if eligible and question["id"] in requests:
            path, original = requests[question["id"]]
            request = read(path)
            candidate = reviewed.get(question["id"], original)
            rejection = _input_rejection(question, candidate, request, normalized,
                require_sources=run_manifest.get("qa_source") == "external")
            if rejection:
                diagnostics.append({"qa_id": question["id"], "status": question.get("status"),
                                    "reason": rejection})
                continue
            item = {"qa": question, "generation_input": str(path.resolve()),
                    "original_candidate": original,
                    "reviewed_candidate": candidate,
                    "qa_source": run_manifest.get("qa_source", "graph"),
                    "provisional": question.get("status") != "approved"}
            payload = request.get("payload", {}) if item["qa_source"] == "external" else {}
            workflow = payload.get("workflow", {})
            focus = payload.get("focus", {})
            workflow_sources = {
                str(source) for source in (workflow.get("sources", []) if isinstance(workflow, dict) else [])
                if str(source).strip()
            }
            focus_sources = {
                str(source) for source in (focus.get("sources", []) if isinstance(focus, dict) else [])
                if str(source).strip()
            }
            # Workflow and focus are generated in separate calls.  A workflow
            # from a disjoint evidence group is background for another QA,
            # not a valid task seed for this one.
            if (isinstance(workflow, dict) and isinstance(workflow.get("text"), str)
                    and (not workflow_sources or not focus_sources
                         or workflow_sources & focus_sources)):
                item["development_workflow"] = workflow["text"]
            if public_records is not None:
                item["public_records"] = public_records
            if item["qa_source"] == "external":
                item["global_agreements"] = global_agreements
                cited = sorted({source for key in ("answer_points", "forbidden_points")
                                for point in candidate.get(key, []) if isinstance(point, dict)
                                for source in point.get("sources", [])})
                associations = set()
                lineage_events = []
                for event in external_events:
                    if set(event.get("source_ids", [])) & set(cited):
                        lineage_events.append(event)
                        for key in ("task_id", "focus", "object", "target",
                                    "behavior", "impact", "behavior_impact", "constraint"):
                            if isinstance(event.get(key), str) and event[key].strip():
                                associations.add(key + ":" + event[key].strip())
                for document in (original, candidate):
                    group = document.get("evidence_group_id")
                    if group:
                        associations.add("evidence:" + group)
                if item.get("development_workflow"):
                    associations.add("workflow:" + item["development_workflow"].strip())
                item.update(
                    source_ids=cited,
                    associations=sorted(associations),
                    external_lineage={
                        "source_ids": cited,
                        "event_ids": sorted({event.get("id") for event in lineage_events
                                              if isinstance(event.get("id"), str)}),
                        "business_behavior": sorted({value.strip() for event in lineage_events
                                                      for value in (event.get("behavior"),
                                                                    event.get("focus"),
                                                                    event.get("task_id"))
                                                      if isinstance(value, str) and value.strip()}),
                        "impact": sorted({value.strip() for event in lineage_events
                                           for value in (event.get("impact"),
                                                         event.get("behavior_impact"),
                                                         event.get("constraint"))
                                           if isinstance(value, str) and value.strip()}),
                    })
            result.append(item)
        elif eligible:
            diagnostics.append({"qa_id": question.get("id"), "status": question.get("status"),
                                "reason": "generation_input_missing"})
    return result


def has_eligible_qa(qa_run, *, include_provisional=False):
    return bool(qa_inputs(qa_run, include_provisional=include_provisional))


def generation_request(item):
    """Keep every original request in a grouped task's private receipt."""
    members = item.get("qa_members")
    if not members:
        return read(item["generation_input"])
    return {"questions": [{"qa_id": member["qa"]["id"],
                           "request": read(member["generation_input"])} for member in members]}


def group_qa_inputs(items, group_size=4, *, diagnostics=None):
    """Build bounded related QA pools; a singleton remains a valid task seed."""
    if group_size < 1:
        raise ValueError("QA group size must be positive")
    diagnostics = diagnostics if diagnostics is not None else []
    result = [item for item in items if item.get("qa_source") != "external"]
    pending = [item for item in items if item.get("qa_source") == "external"]
    while pending:
        members = [pending.pop(0)]
        while len(members) < group_size:
            match = next((index for index, item in enumerate(pending)
                if item["qa"].get("question") not in {member["qa"].get("question") for member in members}
                and any(set(item.get("associations", [])) & set(member.get("associations", []))
                        for member in members)), None)
            if match is None:
                break
            members.append(pending.pop(match))
        if len(members) == 1:
            result.append(members[0])
            continue
        ids = [member["qa"]["id"] for member in members]
        question = dict(members[0]["qa"], id="group-" + qa_fingerprint(ids)[:16],
            question="\n".join(member["qa"]["question"] for member in members),
            answer_points=[point for member in members for point in member["qa"].get("answer_points", [])],
            forbidden_points=[point for member in members for point in member["qa"].get("forbidden_points", [])],
            status="needs_review" if any(member.get("provisional") for member in members) else "approved")
        grouped = dict(members[0], qa=question, qa_members=members, qa_ids=ids,
                       provisional=any(member.get("provisional") for member in members),
                       source_ids=sorted({source for member in members for source in member.get("source_ids", [])}),
                       associations=sorted({value for member in members for value in member.get("associations", [])}))
        lineage = {
            "source_ids": sorted({source for member in members
                                   for source in member.get("external_lineage", {}).get("source_ids", [])}),
            "event_ids": sorted({event for member in members
                                  for event in member.get("external_lineage", {}).get("event_ids", [])}),
            "business_behavior": sorted({value for member in members
                                          for value in member.get("external_lineage", {}).get("business_behavior", [])}),
            "impact": sorted({value for member in members
                               for value in member.get("external_lineage", {}).get("impact", [])}),
        }
        grouped["external_lineage"] = lineage
        workflows = list(dict.fromkeys(member["development_workflow"] for member in members
                                       if member.get("development_workflow")))
        if workflows:
            grouped["development_workflow"] = "\n".join(workflows)
        result.append(grouped)
    return result


def labels(text):
    return {key: value.lower() for key, value in re.findall(
        r"^([A-Z_]+):\s*([a-z_]+)\s*$", text, re.M)}
