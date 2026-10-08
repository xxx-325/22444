"""Small localhost reviewer for provisional QA and task candidates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .security import credential_detected

PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QA review</title>
<style>
body{font:15px system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#202124;background:#f7f8fa}
header{display:flex;justify-content:space-between;align-items:center;gap:12px}
button{border:0;border-radius:6px;padding:7px 12px;cursor:pointer}
.tabs button{background:#e4e7ec;margin-right:6px}.tabs button.active{background:#2257d6;color:#fff}
.card{background:#fff;border:1px solid #d9dde5;border-radius:10px;padding:16px;margin:14px 0;box-shadow:0 1px 2px #0000000b}
.meta{color:#667085;font-size:13px}.warning{color:#9a3412;background:#fff7ed;padding:8px;border-radius:6px}
.answer{white-space:pre-wrap;margin:8px 0}.actions{display:flex;gap:8px;margin-top:12px}
.approve{background:#15803d;color:#fff}.reject{background:#b91c1c;color:#fff}.defer{background:#e5e7eb}
#empty{color:#667085;margin:30px 0}
</style>
</head>
<body>
<header><h1>QA review queue</h1><span id="updated"></span></header>
<p>这是异步审核页。生成流程不会等待这里的点击。</p>
<nav class="tabs">
<button data-filter="pending" class="active">待审核</button>
<button data-filter="approved">已通过</button>
<button data-filter="rejected">已拒绝</button>
<button data-filter="warnings">运行警告</button>
</nav>
<main id="list"></main>
<script>
let data={items:[],decisions:[],stage_errors:[]}, filter="pending";
const esc=s=>String(s??"");
function latest(id){return [...data.decisions].reverse().find(x=>x.candidate_id===id)}
function matches(item){
  const d=latest(item.candidate_id||item.id);
  if(filter==="warnings") return false;
  const status=d?.action || (item.status==="approved"?"approve":item.status) || "needs_review";
  return filter==="pending" ? !["approve","reject"].includes(status)
       : filter==="approved" ? status==="approve"
       : status==="reject";
}
function card(item){
  const id=item.candidate_id||item.id||"unknown", d=latest(id);
  const box=document.createElement("article"); box.className="card";
  const title=document.createElement("div"); title.className="meta";
  title.textContent=`${id} · ${item.status||"needs_review"}${item.evidence_group_id?" · "+item.evidence_group_id:""}`;
  box.append(title);
  const q=document.createElement("h3"); q.textContent=item.question||"(无题干)"; box.append(q);
  for(const p of (item.answer_points||[])){const a=document.createElement("div");a.className="answer";a.textContent="答案："+(p.text||"");box.append(a)}
  if(item.reason||item.failed_checks?.length){const w=document.createElement("div");w.className="warning";w.textContent="原因："+(item.reason||item.failed_checks.join(", "));box.append(w)}
  const actions=document.createElement("div"); actions.className="actions";
  for(const [label,action,cls] of [["通过","approve","approve"],["拒绝","reject","reject"],["暂缓","defer","defer"]]){
    const b=document.createElement("button");b.textContent=label;b.className=cls;
    b.onclick=async()=>{await fetch("/api/decision",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({candidate_id:id,action})});await load()}; actions.append(b)
  }
  if(d){const m=document.createElement("div");m.className="meta";m.textContent=`最近决定：${d.action} · ${d.at}`;box.append(m)}
  box.append(actions); return box
}
function render(){
  const list=document.querySelector("#list");list.replaceChildren();
  if(filter==="warnings"){
    for(const e of data.stage_errors||[]){const w=document.createElement("div");w.className="card warning";w.textContent=JSON.stringify(e);list.append(w)}
    if(!data.stage_errors?.length) list.textContent="没有运行警告"; return;
  }
  const rows=data.items.filter(matches);
  if(!rows.length){list.id="empty";list.textContent="当前没有记录";return}
  for(const item of rows)list.append(card(item));
}
async function load(){const r=await fetch("/api/queue");data=await r.json();document.querySelector("#updated").textContent="更新："+new Date().toLocaleTimeString();render()}
document.querySelectorAll("[data-filter]").forEach(b=>b.onclick=()=>{filter=b.dataset.filter;document.querySelectorAll("[data-filter]").forEach(x=>x.classList.toggle("active",x===b));render()});
load();setInterval(load,5000);
</script>
</body></html>"""


def _read(path, fallback):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return fallback


def _decisions(path):
    rows = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    except FileNotFoundError:
        pass
    return rows


def queue_payload(root: Path) -> dict:
    queue = _read(root / "qa-review-queue.json", {})
    public = _read(root / "qa-public.json", {})
    items = list(queue.get("items", []))
    known = {item.get("candidate_id", item.get("id")) for item in items}
    items.extend(
        item for item in public.get("questions", [])
        if item.get("id") not in known
    )
    return {
        "items": items,
        "stage_errors": queue.get("stage_errors", []),
        "decisions": _decisions(root / "review-decisions.jsonl"),
    }


def _published_question(item):
    """Keep the human-reviewed projection as small as the public projection."""
    answer_points = [
        {"text": point.get("text", "") if isinstance(point, dict) else str(point)}
        for point in item.get("answer_points", [])
    ]
    forbidden_points = [
        {"text": point.get("text", "") if isinstance(point, dict) else str(point)}
        for point in item.get("forbidden_points", [])
    ]
    return {
        key: item.get(key)
        for key in ("id", "qa_mode", "type", "question", "difficulty",
                    "type_origin", "type_status", "difficulty_origin",
                    "difficulty_distance", "stage_count", "reasoning_hops",
                    "graph_hops", "track", "category")
        if key in item
    } | {
        "status": "approved",
        "answer_points": answer_points,
        "forbidden_points": forbidden_points,
    }


def materialize_public(root: Path) -> dict:
    """Apply append-only decisions to the public QA projection."""
    public_path = root / "qa-public.json"
    public = _read(public_path, {})
    if not public_path.is_file():
        return public
    queue = _read(root / "qa-review-queue.json", {})
    items = list(queue.get("items", []))
    items.extend(public.get("questions", []))
    decisions = {row.get("candidate_id"): row for row in _decisions(
        root / "review-decisions.jsonl")}
    selected = {}
    for item in items:
        candidate_id = item.get("candidate_id", item.get("id"))
        if not candidate_id:
            continue
        decision = decisions.get(candidate_id, {})
        action = decision.get("action")
        if action == "reject":
            selected.pop(candidate_id, None)
            continue
        if item.get("status") == "approved" or action == "approve":
            projected = _published_question(item)
            visible = " ".join([
                projected.get("question", ""),
                *(point["text"] for point in projected["answer_points"]),
                *(point["text"] for point in projected["forbidden_points"]),
            ])
            if not credential_detected(visible):
                selected[candidate_id] = projected
    public["questions"] = list(selected.values())
    public["counts"] = {}
    for item in public["questions"]:
        mode = item.get("qa_mode", "code")
        public["counts"][mode] = public["counts"].get(mode, 0) + 1
    pending = any(
        item.get("candidate_id", item.get("id")) not in decisions
        and item.get("status") != "approved"
        for item in queue.get("items", [])
    )
    public["status"] = "needs_review" if pending or queue.get("stage_errors") else (
        "approved" if public["questions"] else "completed_no_questions")
    public["human_review"] = {
        "decisions": len(decisions),
        "published": len(public["questions"]),
    }
    from .task_eval.artifacts import save
    save(public_path, public)
    save(root / "qa.json", public)
    return public


def append_decision(root: Path, body: dict) -> dict:
    candidate_id = body.get("candidate_id")
    action = body.get("action")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValueError("candidate_id is required")
    if action not in {"approve", "reject", "defer"}:
        raise ValueError("action must be approve, reject, or defer")
    decision = {
        "candidate_id": candidate_id,
        "action": action,
        "at": datetime.now(timezone.utc).isoformat(),
    }
    path = root / "review-decisions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(decision, ensure_ascii=False) + "\n")
    path.chmod(0o600)
    materialize_public(root)
    return decision


def make_handler(root: Path):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, content, content_type):
            raw = content.encode("utf-8") if isinstance(content, str) else content
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                return self._send(200, PAGE, "text/html; charset=utf-8")
            if path == "/api/queue":
                return self._send(200, json.dumps(queue_payload(root), ensure_ascii=False),
                                  "application/json; charset=utf-8")
            self._send(404, "not found", "text/plain; charset=utf-8")

        def do_POST(self):
            if urlparse(self.path).path != "/api/decision":
                return self._send(404, "not found", "text/plain; charset=utf-8")
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size > 65536:
                    raise ValueError("request too large")
                body = json.loads(self.rfile.read(size).decode("utf-8"))
                decision = append_decision(root, body)
            except (ValueError, json.JSONDecodeError) as error:
                return self._send(400, json.dumps({"error": str(error)}),
                                  "application/json; charset=utf-8")
            self._send(200, json.dumps(decision, ensure_ascii=False),
                       "application/json; charset=utf-8")

        def log_message(self, *_):
            return

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="QA output directory containing qa-review-queue.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.run.resolve()))
    print(f"Review page: http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
