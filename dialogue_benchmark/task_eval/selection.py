"""Host-controlled task selection using exact, read-only evidence requests."""

import json
from pathlib import Path
import re
import subprocess

from ..llm import ModelStageError
from .artifacts import read, save
from .runtime import ask_model, bounded_model_config
from .history import freeze_targets, write_contract_from_targets
from .metrics import cache_usage


MAX_SELECTION_REQUEST_SECONDS = 600


def _source_tokens(value):
    """Parse the small source list used by selector decisions."""
    if value is None:
        return []
    return [item.strip() for item in re.split(r"[,，、]", str(value))
            if item.strip() and item.strip().casefold() != "none"]


class SelectionBudget:
    """Share request/token limits; each agent retains its execution timeout."""

    def __init__(self, root, options):
        self.root = Path(root)
        self.max_requests = options.get("max_requests", 80)
        self.max_tokens = options.get("max_tokens", 1500000)
        self.agent_seconds = options.get("max_seconds", 1200)
        self.requests = self.prompt_tokens = self.completion_tokens = 0
        self.usage_complete = True
        self._cache_rows = []

    def remaining(self):
        if not self.usage_complete:
            raise ModelStageError("usage_missing")
        if self.requests >= self.max_requests or self.tokens >= self.max_tokens:
            raise ModelStageError("selection_budget_exhausted")
        return dict(max_requests=self.max_requests - self.requests,
                    max_tokens=self.max_tokens - self.tokens,
                    max_seconds=self.agent_seconds)

    @property
    def tokens(self):
        return self.prompt_tokens + self.completion_tokens

    def record(self, usage):
        for row in usage:
            # A stage that made no model request (for example a mocked or
            # skipped read-only explorer) has no provider usage to validate.
            # Do not turn that harmless zero into a global ``usage_missing``
            # failure; an actual request without usage still fails closed.
            if row.get("request_count", 1) == 0 and "prompt_tokens" not in row and "completion_tokens" not in row:
                continue
            self.requests += row.get("request_count", 1)
            self.usage_complete &= "prompt_tokens" in row and "completion_tokens" in row
            self.prompt_tokens += row.get("prompt_tokens", 0)
            self.completion_tokens += row.get("completion_tokens", 0)
            if row.get("request_count", 1) != 0 or "prompt_tokens" in row or "completion_tokens" in row:
                self._cache_rows.append(row)
        cache = cache_usage(self._cache_rows)
        save(self.root / "selection-budget.json", dict(requests=self.requests,
             prompt_tokens=self.prompt_tokens, completion_tokens=self.completion_tokens,
             total_tokens=self.tokens, usage_complete=self.usage_complete, **cache))

    def call(self, prompt, payload, config, output):
        self.remaining()
        error = None
        try:
            return ask_model(prompt, payload,
                             bounded_model_config(config, min(self.agent_seconds,
                                                               MAX_SELECTION_REQUEST_SECONDS)),
                             output)
        except Exception as exc:
            error = exc
            raise
        finally:
            usage = Path(output) / "usage.json"
            if usage.exists():
                self.record(read(usage) or [{}])
            else:
                self.record([{}])
            # Preserve the provider/protocol error.  A missing usage record is
            # a second diagnostic, not a reason to hide the original failure.
            if error is None and not self.usage_complete:
                raise ModelStageError("usage_missing")


def _path(root, value):
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("Expected a repository-relative path")
    if any(p.startswith(".") and p != "." for p in Path(value).parts):
        raise ValueError("Hidden paths and parent traversal are not query targets")
    path = (root / value).resolve()
    if path != root and root not in path.parents:
        raise ValueError("Path outside baseline")
    if not path.exists():
        raise ValueError("Query path does not exist")
    return path


def _query_from_text(value):
    """Decode the five short fields used by the selection prompt."""
    if not isinstance(value, str) or value.strip().casefold() == "none":
        return None
    fields = [part.strip() for part in value.split("|")]
    if len(fields) != 5:
        raise ValueError("Invalid selection query")
    op, target, path, text, offset = fields
    if path == "-":
        path = None
    if text == "-":
        text = None
    # The text protocol uses ``-`` for an omitted path/text field.  Models
    # occasionally use the same marker for the first page; treat that as the
    # documented zero offset instead of rejecting an otherwise safe query.
    if offset == "-":
        offset = 0
    try:
        offset = int(offset)
    except (TypeError, ValueError):
        raise ValueError("Invalid selection query offset")
    query = {"op": op, "target": target, "offset": offset}
    if op == "read":
        if target == "repo":
            query["path"] = path
        elif target == "history":
            query["source"] = text
    elif op == "lookup":
        query["path" if target == "repo" else "text"] = path if target == "repo" else text
        if target == "repo":
            query["text"] = text
        elif target == "history":
            query["text"] = text
    else:
        raise ValueError("Invalid selection query operation")
    return query


def query_evidence(query, baseline, history):
    """Execute one structured lookup/read, never a model-authored command."""
    if not isinstance(query, dict) or query.get("op") not in {"lookup", "read"}:
        raise ValueError("Expected lookup or read")
    if set(query) - {"op", "target", "path", "source", "text", "offset"}:
        raise ValueError("Unknown query parameters")
    offset = query.get("offset", 0)
    if type(offset) is not int or offset < 0:
        raise ValueError("Invalid page offset")
    target, op = query.get("target"), query["op"]
    allowed = {"op", "target", "offset"} | ({"source"} if target == "history" and op == "read"
              else {"text"} if target == "history" else {"path", "text"} if op == "lookup" else {"path"})
    if set(query) - allowed:
        raise ValueError("Parameters do not match query operation")
    if target == "history":
        events = {e["id"]: e for e in (history or {}).get("events", [])}
        aliases = (history or {}).get("source_aliases", {})
        if op == "read":
            source = query.get("source")
            identity = aliases.get(source, source)
            if identity not in events:
                raise ValueError("Unknown exact history source")
            text = events[identity]["text"]
            if offset > len(text):
                raise ValueError("Offset beyond source")
            return dict(source=identity, text=text[offset:offset + 6000], offset=offset,
                        next_offset=offset + 6000 if len(text) > offset + 6000 else None)
        keyword = query.get("text")
        if not isinstance(keyword, str) or not keyword.strip():
            raise ValueError("History lookup requires literal text")
        matches = []
        inverse = {v: k for k, v in aliases.items()}
        for event in events.values():
            position = event["text"].find(keyword)
            if position >= 0:
                matches.append(dict(source=inverse.get(event["id"], event["id"]),
                                    order=event["order"], role=event.get("role"),
                                    excerpt=event["text"][max(0, position - 100):position + 400]))
    elif target == "repo":
        root = Path(baseline).resolve()
        path = _path(root, query.get("path"))
        if op == "read":
            if not path.is_file() or path.stat().st_size > 2000000:
                # A directory read is a compact, deterministic file index.
                # It lets selection discover the relevant entry point without
                # forcing the model to guess a filename; file contents still
                # require an explicit subsequent read.
                if path.is_dir():
                    names = sorted(str(item.relative_to(root)) for item in path.rglob("*")
                                   if item.is_file() and not any(part.startswith(".") for part in item.parts))
                    return {"path": str(path.relative_to(root)) or ".", "files": names[offset:offset + 80],
                            "offset": offset, "next_offset": offset + 80 if len(names) > offset + 80 else None}
                raise ValueError("Read requires a text file smaller than 2 MB")
            lines = path.read_text(encoding="utf-8").splitlines()
            if offset > len(lines):
                raise ValueError("Offset beyond file")
            page, size = [], 0
            for number in range(offset, min(len(lines), offset + 80)):
                if size + len(lines[number]) > 8000:
                    break
                page.append({"line": number + 1, "text": lines[number]})
                size += len(lines[number])
            if not page and offset < len(lines):
                raise ValueError("Single line exceeds read projection")
            end = offset + len(page)
            return dict(path=str(path.relative_to(root)), lines=page, offset=offset,
                        next_offset=end if end < len(lines) else None)
        keyword = query.get("text")
        if not isinstance(keyword, str) or not keyword.strip():
            raise ValueError("Repository lookup requires text or a filename fragment")
        # rg's fixed-string mode supplies content matches; --files supplies names.
        files = subprocess.run(["rg", "--files", "--", str(path)], cwd=root,
                               capture_output=True, text=True, timeout=15)
        if path.is_file():
            names = [str(path)]
        elif files.returncode in {0, 1}:
            names = files.stdout.splitlines()
        else:
            raise ValueError("Repository file lookup failed")
        matches = [{"path": str(Path(p).relative_to(root)), "kind": "filename"}
                   for p in sorted(names) if keyword in str(Path(p).relative_to(root))]
        found = subprocess.run(["rg", "-n", "-F", "--no-heading", "--with-filename", "--color", "never",
                                "--max-columns", "500", "--max-columns-preview", "--", keyword, str(path)],
                               cwd=root, capture_output=True, text=True, timeout=15)
        if found.returncode not in {0, 1}:
            raise ValueError("Repository text lookup failed")
        matches += [{"match": line.replace(str(root) + "/", "", 1)} for line in found.stdout.splitlines()]
    else:
        raise ValueError("Expected repo or history target")
    return dict(matches=matches[offset:offset + 20], offset=offset, total_matches=len(matches),
                next_offset=offset + 20 if len(matches) > offset + 20 else None,
                scope="Literal lookup only; zero matches do not establish absence of a rule")


def _focused_history(history):
    """Keep the selected public event and one nearby visible reply in the first call.

    The complete public history remains host-side and can be read by exact
    source query.  Sending every user turn up front both wastes context and
    makes a real selection impossible for long dialogues.
    """
    events = list((history or {}).get("initial_events", []))
    selected_ids = set((history or {}).get("selected_event_ids", []))
    if not selected_ids:
        return events
    selected = [index for index, event in enumerate(events)
                if event.get("id") in selected_ids]
    keep = set(selected)
    for index in selected:
        for candidate in range(index - 1, -1, -1):
            if events[candidate].get("role") == "assistant":
                keep.add(candidate)
                break
        for candidate in range(index + 1, len(events)):
            if events[candidate].get("role") == "assistant":
                keep.add(candidate)
                break
    return [events[index] for index in sorted(keep)]


def repository_overview(root, limit=80):
    """Return a small static project map for task authoring."""
    root = Path(root).resolve()
    if not root.is_dir():
        return {"purpose": "", "top_level": [], "entry_points": [],
                "test_areas": [], "relevant_paths": []}

    entries = [item for item in sorted(root.iterdir(), key=lambda item: item.name)
               if not item.name.startswith(".")]
    top_level = [item.name + ("/" if item.is_dir() else "") for item in entries[:limit]]
    purpose = ""
    for name in ("README.md", "README.rst", "README.txt"):
        path = root / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("```"):
                continue
            lines.append(line)
            if len(" ".join(lines)) >= 700:
                break
        purpose = " ".join(lines)[:900]
        if purpose:
            break

    entry_points = [name for name in (
        "pyproject.toml", "setup.cfg", "setup.py", "package.json",
        "Cargo.toml", "go.mod", "Makefile", "tox.ini") if (root / name).is_file()]
    test_areas = [name for name in top_level
                  if name.rstrip("/").casefold() in {"test", "tests", "spec", "specs"}]
    relevant_paths = [name for name in top_level
                      if name.rstrip("/").casefold() in {
                          "src", "lib", "app", "docs", "examples", "scripts", "cli"}]
    return {"purpose": purpose, "top_level": top_level,
            "entry_points": entry_points, "test_areas": test_areas,
            "relevant_paths": relevant_paths}


def _model_repository_evidence(rows):
    """Project host query receipts to observations, not control metadata.

    The task writer needs to know what the repository contained, but it does
    not need the host's query IDs, request shapes, or cache bookkeeping.
    Keeping those fields out of the model payload also prevents a public task
    from accidentally repeating the construction protocol.
    """
    projected = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        value = row.get("result", row)
        if isinstance(value, dict):
            projected.append(value)
    return projected


def select_task(qa, history, baseline, config, output, budget, *, exploration=None, workflow=None, feedback=None):
    from .prompts import SELECT_TASK
    output = Path(output)
    focus = _focused_history(history)
    cited = set((history or {}).get("qa_source_ids", []))
    sources = [{"source": e.get("source", e["id"]), "role": e.get("role"),
                "answer_source": e["id"] in cited}
               for e in focus]
    # Keep the stable part of this state before the per-turn query history.
    # DeepSeek's prefix cache matches from token zero; putting the growing
    # query log (and turn-specific choices) first would invalidate the useful
    # prefix on every evidence request.
    state = {
        # The QA route and internal type labels are control-side metadata.  The
        # selector only needs the question and the answer material to decide
        # whether a real follow-up task exists.
        "qa": {"question": qa.get("question", ""),
               "answer_points": [point if isinstance(point, str)
                                 else point.get("text", point.get("claim", ""))
                                 for point in qa.get("answer_points", [])]},
        "history_sources": sources,
        "repository_overview": repository_overview(baseline),
        "repository_exploration": exploration or "",
        "repository_entries": sorted(
            p.name + ("/" if p.is_dir() else "")
            for p in Path(baseline).iterdir()
            if not p.name.startswith(".")
        ),
        "development_workflow": workflow or "",
        "rejected_draft": feedback or "",
        "queries": [],
        "turn_context": {},
    }
    seen = set()
    known = {"qa"}
    if state["repository_exploration"]:
        # This is a host-produced repository observation, not a dialogue
        # source.  Give it one stable citation name so the selector can use
        # the explorer's report without inventing a query ID.
        known.add("repository_exploration")
    known.add("repository_overview")
    status, reason, decision = "pending", "not_started", None
    try:
        while True:
            step = output / ("step-%03d" % len(state["queries"]))
            allowed = "need_evidence, stop, pending" if not state["queries"] else "need_evidence, candidate, stop, pending"
            state["turn_context"] = {
                "allowed_decisions": allowed.split(", "),
                "repository_read_hint": "若仓库还没读过，请先查入口或说明文档。",
                "duplicate_query_rule": (
                    "上一轮查询已经执行并保存在 queries 中；不要重复相同的查询，"
                    "零匹配也不要重复原文字查找。"
                    if state["queries"] else ""
                ),
                "available_sources": sorted(known),
            }
            response = budget.call(
                SELECT_TASK
                + "\n本轮决定、查询去重规则和可引用来源见 DATA.turn_context；"
                  "只按其中约束作答，不要输出 JSON。\n",
                state,
                config,
                step,
            )
            rows = response.get("reviews", [])
            if len(rows) != 1:
                raise ValueError("Expected one selection decision")
            decision = rows[0]
            action, reason = decision.get("decision"), decision.get("reason")
            if action not in {"stop", "candidate", "need_evidence", "pending"} or not isinstance(reason, str) or not reason.strip():
                raise ValueError("Invalid selection decision")
            refs = _source_tokens(decision.get("sources", "none"))
            request_value = decision.get("request", "none")
            if action in {"candidate", "pending"} and request_value != "none":
                # A useful candidate with one unresolved read is not a format
                # failure. Execute the read first and let the next decision
                # confirm the conclusion against the new evidence.  DeepSeek
                # sometimes calls this pending while still naming its next
                # read; the host can safely normalize that to need_evidence.
                decision["decision"] = "need_evidence"
                action = "need_evidence"
                reason = str(reason) + "; unresolved evidence requested"
            if action != "need_evidence" and request_value != "none":
                raise ValueError("Conclusion still requests unresolved evidence")
            # Index entries can identify the next source to read. Only sources
            # actually read may support a conclusion.
            allowed_refs = known | ({s["source"] for s in sources} if action == "need_evidence" else set())
            if set(refs) - allowed_refs:
                raise ValueError("Selection cites evidence not provided")
            if action in {"stop", "candidate"}:
                if not refs:
                    raise ValueError("Selection conclusion needs evidence")
                repo_refs = {q["id"] for q in state["queries"] if q["query"]["target"] == "repo"}
                if state["repository_exploration"]:
                    repo_refs.add("repository_exploration")
                repo_refs.add("repository_overview")
                if action == "candidate" and not set(refs) & repo_refs:
                    raise ValueError("Candidate needs repository observations")
                status = action
                break
            if action == "pending":
                break
            query = _query_from_text(request_value)
            if not isinstance(query, dict):
                raise ValueError("Evidence request is required")
            query.setdefault("offset", 0)
            if query.get("target") == "repo":
                root = Path(baseline).resolve()
                query["path"] = str(_path(root, query.get("path")).relative_to(root))
            if query.get("target") == "history" and "source" in query:
                query["source"] = (history or {}).get("source_aliases", {}).get(query["source"], query["source"])
            key = json.dumps(query, sort_keys=True)
            if key in seen:
                raise ValueError("duplicate_query_no_new_evidence")
            result = query_evidence(query, baseline, history)
            identity = "query%d" % (len(state["queries"]) + 1)
            receipt = dict(id=identity, query=query, result=result)
            state["queries"].append(receipt)
            save(step / "query.json", receipt)
            known.add(identity)
            if query["target"] == "history":
                if "source" in result:
                    known.add(result["source"])
                known.update(row["source"] for row in result.get("matches", []))
                aliases = (history or {}).get("source_aliases", {})
                known.update(alias for alias, identity in aliases.items() if identity in known)
            seen.add(key)
    except Exception as error:
        status, reason = "pending", str(error)
    public = {}
    if isinstance(decision, dict):
        for key in ("public_goal", "agreement_object", "agreement_scope"):
            value = decision.get(key)
            if isinstance(value, str) and value.strip():
                public[key] = value.strip()
    repo_evidence = [q for q in state["queries"] if q["query"].get("target") == "repo"]
    result = dict(status=status, reason=reason, decision=decision, public=public,
                  repository_overview=state["repository_overview"],
                  repository_exploration=state["repository_exploration"],
                  public_repository_evidence=repo_evidence, evidence=state,
                  query_count=len(state["queries"]), requests=budget.requests, total_tokens=budget.tokens)
    if status == "candidate" and public and history:
        answer_fragments = [point.get("text", point.get("claim", ""))
                            for point in qa.get("answer_points", [])
                            if isinstance(point, dict)]
        answer_fragments.extend(point for point in qa.get("answer_points", [])
                                if isinstance(point, str))
        # The selector is shown the answer to decide whether a task is worth
        # constructing, but its public goal must remain a natural business
        # request.  Reject both verbatim answer points and distinctive
        # clauses, since a leaked direction lets the no-memory arm solve the
        # task without recovering the historical fact.
        leaked = []
        for fragment in answer_fragments:
            words = re.findall(r"[\w\u4e00-\u9fff]+", fragment)
            if not fragment or len(fragment) < 8:
                continue
            for key, value in public.items():
                if fragment in value:
                    leaked.append((key, fragment))
                    continue
                significant = [word for word in words if len(word) > 1]
                if len(significant) >= 3:
                    hits = sum(word in value for word in significant)
                    if hits >= max(2, len(significant) - 2):
                        leaked.append((key, fragment))
        if leaked:
            result["status"] = "pending"
            result["reason"] = "public_goal_contains_answer"
        else:
            target_result = extract_history_targets(qa, history, public, config,
                                                    output / "history-targets", budget)
            result["history_targets"] = target_result
            if target_result.get("status") != "candidate":
                result["status"] = target_result.get("status", "pending")
                result["reason"] = target_result.get("reason", "history_targets_unavailable")
    save(output / "result.json", result)
    return result


def _parse_files(response, allowed, required):
    files = {}
    for row in response.get("files", []):
        name = row.get("name")
        if name not in allowed or name in files:
            raise ValueError("Unexpected draft file")
        content = row.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Missing draft content")
        files[name] = content
    if set(files) != set(required):
        raise ValueError("Incomplete draft")
    return files


def extract_history_targets(qa, history, public, config, output, budget):
    """Freeze H from original sources before any task-specific answer review."""
    from .prompts import HISTORY_TARGETS
    output = Path(output)
    focused = _focused_history(history)
    # Keep the QA's saved public boundary: an uncited correction can still
    # constrain a cited rule. Do not add other dialogue topics to this closure.
    source_ids = (set((history or {}).get("qa_source_ids", []))
                  | set((history or {}).get("selected_event_ids", [])))
    if source_ids:
        selected = [index for index, event in enumerate(focused)
                    if event.get("id") in source_ids]
        keep = set(selected)
        for index in selected:
            for candidate in range(index - 1, -1, -1):
                if focused[candidate].get("role") == "assistant":
                    keep.add(candidate)
                    break
            for candidate in range(index + 1, len(focused)):
                if focused[candidate].get("role") == "assistant":
                    keep.add(candidate)
                    break
        focused = [focused[index] for index in sorted(keep)]
    initial = [{"source": event.get("source", event["id"]), "role": event.get("role"),
                "text": event.get("text", "")} for event in focused]
    # Type labels (including M1--M6) are classification metadata, not facts
    # needed to freeze a historical rule.  Keeping them out of the prompt
    # prevents the author from treating the construction label as a clue.
    payload = {"question": qa.get("question", ""),
               "agreement": public, "history": initial}
    try:
        response = budget.call(HISTORY_TARGETS, payload, config, output)
        rows = response.get("reviews", [])
        if not rows:
            result = {"status": "pending", "reason": "no_historical_target", "targets": []}
        else:
            frozen = freeze_targets(rows, history)
            result = {"status": "candidate", "reason": "historical_targets_frozen", **frozen}
    except Exception as error:
        result = {"status": "pending", "reason": str(error), "targets": []}
    save(output / "result.json", result)
    return result


def write_public_task(selection, config, output, spec, budget, feedback=""):
    """Generate only the public task from a whitelisted selection envelope."""
    from .prompts import PUBLIC_TASK_SIMPLE
    public = selection.get("public") or {}
    if not isinstance(public.get("public_goal"), str) or not public["public_goal"].strip():
        raise ValueError("public_task_input_missing")
    # The selector sees the answer and may repeat it in its agreement fields.
    # Keep the original workflow with the goal and repository evidence. The explorer's
    # proposed interfaces are suggestions, not additional task requirements.
    payload = {"public_goal": public["public_goal"],
               "historical_question": selection.get("historical_question", ""),
               "repository_overview": selection.get("repository_overview", {}),
               "repository_evidence": _model_repository_evidence(
                   selection.get("public_repository_evidence", [])),
               "feedback": feedback}
    workflow = selection.get("evidence", {}).get("development_workflow")
    if workflow:
        payload["development_workflow"] = workflow
    response = budget.call(PUBLIC_TASK_SIMPLE, payload, config, output)
    task = response.get("task")
    if not isinstance(task, str) or not task.strip():
        # A previously configured provider may still return the old one-file
        # envelope; accept only that exact public file, never private files.
        files = _parse_files(response, {"task.md"}, {"task.md"})
        task = files["task.md"]
    Path(spec).mkdir(parents=True, exist_ok=True)
    (Path(spec) / "task.md").write_text(task, encoding="utf-8")
    return {"task.md": task}


def write_private_draft(selection, config, output, spec, budget, feedback="", history_review=None):
    """Generate private history use and acceptance after task.md is fixed."""
    from .prompts import PRIVATE_DRAFT_SIMPLE
    targets = selection.get("history_targets", {}).get("targets", [])
    if history_review is not None:
        active = {row["id"] for row in history_review["history_rows"]
                  if row["applicable"] == "yes"}
        targets = [target for target in targets if target["id"] in active]
    history = selection.get("public_history") or {}
    payload = {"task": (Path(spec) / "task.md").read_text(encoding="utf-8"),
               "history_targets": targets,
               "history_sources": [{"id": event["id"], "text": event.get("text", ""),
                                    "role": event.get("role")} for event in history.get("events", [])
                                   if event["id"] in {source for target in targets for source in target["sources"]}],
               "historical_answer": selection.get("historical_answer", ""),
               "repository_evidence": _model_repository_evidence(
                   selection.get("public_repository_evidence", [])),
               "feedback": feedback}
    response = budget.call(PRIVATE_DRAFT_SIMPLE, payload, config, output)
    if not isinstance(response.get("use"), str) or not response["use"].strip():
        raise ValueError("private_use_missing")
    rows = response.get("acceptance", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("private_acceptance_missing")
    if not any(row.get("basis") == "task" for row in rows):
        raise ValueError("private_acceptance_missing_task")
    (Path(spec) / "memory-use.md").write_text(response["use"] + "\n", encoding="utf-8")
    lines = ["| ID | Requirement | Basis | Check |", "| --- | --- | --- | --- |"]
    for row in rows:
        lines.append("| %s | %s | %s | %s |" %
                     (row["id"], row["requirement"], row["basis"], row["check"]))
    (Path(spec) / "acceptance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if selection.get("history_targets", {}).get("targets"):
        write_contract_from_targets(spec, selection["history_targets"], history_review)
    files = {"memory-use.md": response["use"], "acceptance.md": "\n".join(lines) + "\n"}
    if targets:
        files["history-contract.txt"] = (Path(spec) / "history-contract.txt").read_text()
    return files


def write_draft(selection, config, output, spec, budget, feedback=""):
    """Run the public and private generators in order.

    The old four-file response is retained only for selections without frozen
    history (the ordinary no-history path); historical candidates always use
    the split calls above.
    """
    if selection.get("history_targets", {}).get("targets"):
        write_public_task(selection, config, Path(output) / "public", spec, budget, feedback)
        return write_private_draft(selection, config, Path(output) / "private", spec, budget, feedback)
    if selection.get("qa_source") == "external":
        from .prompts import EXTERNAL_ACCEPTANCE, TASK_ONLY_DRAFT
        # Do not pass the selector's agreement object/scope to the public
        # task writer.  For external-only QA those fields may contain the
        # observed URL, error, or log result; the public writer should see the
        # natural goal and current repository shape, while the answer remains
        # private to the with-memory trial.
        public = selection.get("public", {})
        payload = {"selection": {"public_goal": public.get("public_goal", "")},
                   "repository_overview": selection.get("repository_overview", {}),
                   "repository_exploration": selection.get("repository_exploration", ""),
                   "evidence": _model_repository_evidence(
                       selection.get("public_repository_evidence", [])),
                   "feedback": feedback}
        response = budget.call(TASK_ONLY_DRAFT, payload, config, Path(output) / "public")
        files = _parse_files(response, {"task.md"}, {"task.md"})
        Path(spec).mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (Path(spec) / name).write_text(content, encoding="utf-8")
        answer = selection["historical_answer"]
        if not answer.strip():
            raise ValueError("External task requires the exact injected answer")
        response = budget.call(EXTERNAL_ACCEPTANCE, dict(
            public_task=files["task.md"], historical_answer=answer,
            historical_questions=selection.get("historical_questions", []),
            repository_overview=payload["repository_overview"],
            repository_evidence=payload["evidence"]), config, Path(output) / "private")
        required = ({"NO_TASK.md"} if any(row.get("name") == "NO_TASK.md" for row in
                    response.get("files", [])) else {"memory-use.md", "acceptance.md"})
        private = _parse_files(response, required, required)
        for name, content in private.items():
            (Path(spec) / name).write_text(content, encoding="utf-8")
        save(Path(spec) / "oracle-answer.json", {"answer": answer})
        files.update(private)
        return files
    from .prompts import DRAFT_TASK, HISTORY_CONTRACT
    response = budget.call(DRAFT_TASK + HISTORY_CONTRACT, dict(selection=selection.get("public", {}),
                           evidence=selection.get("public_repository_evidence", []), feedback=feedback), config, output)
    files = _parse_files(response, {"task.md", "memory-use.md", "history-contract.txt", "acceptance.md"},
                          {"task.md", "memory-use.md", "history-contract.txt", "acceptance.md"})
    Path(spec).mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (Path(spec) / name).write_text(content, encoding="utf-8")
