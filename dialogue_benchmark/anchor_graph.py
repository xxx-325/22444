"""Pure deterministic action graphs for anchored QA."""
from __future__ import annotations
import re
from collections import defaultdict
from copy import deepcopy
from typing import Any, Iterable, Mapping

REL = {"fix_of":"fix_chain","fixes":"fix_chain","fixed":"fix_chain","previous":"fix_chain","previous_id":"fix_chain","supersedes":"corrects","corrects":"corrects","corrected_from":"corrects","replaces":"corrects"}
RECEIPT = re.compile(r"^(?:ok|done|completed|success(?:fully)?|tests? passed|passed|[✓✔]\s*\d*\s*passed)[.!\s]*$", re.I)
LISTING = re.compile(r"^(?:ls(?:\s+-[\w-]+)?|find\s+[^\n]*|tree(?:\s+[^\n]*)?)\s*(?:\n|$)", re.I)

def _ids(v: Any) -> list[str]:
    if isinstance(v,str): return [v] if v else []
    return list(dict.fromkeys(x for x in v if isinstance(x,str) and x)) if isinstance(v,(list,tuple,set)) else []
def _text(r: Mapping[str,Any]) -> str:
    v=r.get("text",r.get("content","")); return v.strip() if isinstance(v,str) else ""
def _paths(r: Mapping[str,Any]) -> set[str]:
    out={r[k] for k in ("path","file","affected_path") if isinstance(r.get(k),str) and r[k]}
    for k in ("paths","affected_paths"):
        if isinstance(r.get(k),(list,tuple,set)): out.update(x for x in r[k] if isinstance(x,str) and x)
    if isinstance(r.get("changes"),Mapping): out.update(x for x in r["changes"] if isinstance(x,str))
    return out
def _keys(t:str)->set[str]: return set(re.findall(r"[a-z0-9_.-]{2,}",t.casefold()))|set(re.findall(r"[\u4e00-\u9fff]{2}",t))
def _rid(r): return r.get("id") if isinstance(r.get("id"),str) else None
def _noise(r):
    t=_text(r)
    if not t:return "empty"
    if LISTING.match(t):return "directory_listing"
    if RECEIPT.match(t):return "success_receipt"
    return None

def build_action_graph(records:Iterable[Mapping[str,Any]])->dict:
    rows=[deepcopy(dict(r)) for r in records]; groups=defaultdict(list); standalone=[]; noise=[]; seen=set()
    for r in rows:
        norm=re.sub(r"\s+"," ",_text(r)).casefold(); reason=_noise(r) or ("duplicate" if norm and norm in seen else None)
        if norm:seen.add(norm)
        if reason: noise.append({"id":_rid(r),"reason":reason}); continue
        cid=next((r.get(k) for k in ("call_id","tool_call_id","parent_call_id") if isinstance(r.get(k),str) and r[k]),None)
        (groups[cid] if cid else standalone).append(r)
    nodes=[]
    for cid,parts in groups.items():
        parts.sort(key=lambda r:(r.get("order",0),r.get("id",""))); ids=[_rid(r) for r in parts if _rid(r)]
        nodes.append({"id":"action:"+cid,"kind":"action","call_id":cid,"record_ids":ids,"order":min((r.get("order",0) for r in parts),default=0),"paths":sorted(set().union(*(_paths(r) for r in parts))),"text":"\n".join(dict.fromkeys(_text(r) for r in parts if _text(r))),"parts":parts})
    for r in standalone:nodes.append({**r,"id":_rid(r),"kind":r.get("kind","message"),"record_ids":[_rid(r)] if _rid(r) else [],"order":r.get("order",0),"paths":sorted(_paths(r)),"text":_text(r)})
    nodes=[n for n in nodes if isinstance(n.get("id"),str)]; nodes.sort(key=lambda n:(n["order"],n["id"]))
    lookup={x:n for n in nodes for x in [n["id"],*n["record_ids"]]}; edges=set()
    for i,a in enumerate(nodes):
        for b in nodes[i+1:]:
            if set(a["paths"])&set(b["paths"]):edges.add((a["id"],b["id"],"same_path"))
        for f,rel in REL.items():
            for ref in _ids(a.get(f)):
                if ref in lookup and lookup[ref]["id"]!=a["id"]:edges.add((a["id"],lookup[ref]["id"],rel))
    users=[n for n in nodes if n.get("kind")=="message" and n.get("role")=="user"]
    for n in nodes:
        if n.get("kind")=="action":
            prior=[u for u in users if u["order"]<n["order"]]
            if prior:edges.add((n["id"],max(prior,key=lambda x:x["order"])["id"],"responds_to"))
    return {"nodes":nodes,"edges":[{"from":a,"to":b,"relation":r} for a,b,r in sorted(edges)],"noise":noise,"node_by_id":lookup}

def anchor_seeds(event:Mapping[str,Any],graph:Mapping[str,Any])->dict:
    lookup=graph.get("node_by_id") or {n["id"]:n for n in graph.get("nodes",[])}; seeds=[]; unresolved=[]
    for f in ("source_ids","used_by","context_ids"):
        for ref in _ids(event.get(f)):
            n=lookup.get(ref)
            if n and n["id"] not in {x["id"] for x in seeds}:seeds.append(n)
            elif not n:unresolved.append(ref)
    focus=event.get("focus","") if isinstance(event.get("focus"),str) else ""
    return {"anchor_id":event.get("id"),"seeds":seeds,"seed_ids":[n["id"] for n in seeds],"keywords":sorted(_keys(focus)|set().union(*(_keys(n.get("text","")) for n in seeds))),"unresolved_ids":list(dict.fromkeys(unresolved))}

def expand_anchor(anchor_or_group:Mapping[str,Any],graph:Mapping[str,Any],budget_chars:int)->dict:
    if not isinstance(budget_chars,int) or budget_chars<=0:raise ValueError("budget_chars must be positive")
    seed=anchor_or_group if "seeds" in anchor_or_group else anchor_seeds(anchor_or_group,graph); nodes=list(seed.get("seeds",[])); known=set(seed.get("keywords",[])); admitted={n["id"] for n in nodes}; rejected=[]; lookup=graph.get("node_by_id") or {n["id"]:n for n in graph.get("nodes",[])}
    for e in graph.get("edges",[]):
        if e["from"] not in admitted:continue
        cand=lookup.get(e["to"]); overlap=_keys(cand.get("text",""))&known if cand else set()
        if not cand or cand["id"] in admitted:continue
        if e["relation"] not in {"same_path","fix_chain","corrects","same_value"} or not overlap:rejected.append({"node_id":cand["id"],"reason":"irrelevant"});continue
        if len(str(nodes+[cand]))>budget_chars:rejected.append({"node_id":cand["id"],"reason":"budget"});continue
        nodes.append(cand); admitted.add(cand["id"]); known|=_keys(cand.get("text",""))
    return {"nodes":nodes,"edges":[e for e in graph.get("edges",[]) if e["from"] in admitted and e["to"] in admitted],"admitted":[{"node_id":n["id"]} for n in nodes],"rejected":rejected,"budget":{"chars":len(str(nodes)),"max_chars":budget_chars}}

def combine_anchor_group(anchors:Iterable[Mapping[str,Any]],records:Iterable[Mapping[str,Any]],*,required_anchor_ids:Iterable[str]|None=None,combination_reason:str|None=None,max_chars:int=48000)->dict:
    graph=build_action_graph(records); rows=[dict(a) for a in anchors]; ids=[a["id"] for a in rows if isinstance(a.get("id"),str)]; required=list(dict.fromkeys(required_anchor_ids or [])); invalid=[x for x in required if x not in ids]; expanded=[expand_anchor(a,graph,max_chars) for a in rows]; all_nodes={n["id"]:n for x in expanded for n in x["nodes"]}
    return {"anchor_ids":ids,"required_anchor_ids":required,"supporting_anchor_ids":[x for x in ids if x not in required],"memory_kinds":sorted({a.get("memory_kind") for a in rows if a.get("id") in required and a.get("memory_kind")}),"combination_reason":combination_reason,"nodes":list(all_nodes.values()),"subgraphs":expanded,"invalid_required_anchor_ids":invalid,"status":"ready" if required and not invalid else "needs_review"}

def anchor_difficulty(required_facts:int|Iterable[Any],*,info_nodes:int|Iterable[Any]|None=None,revision:bool=False,cross_stage:bool=False,complete:bool=True)->str:
    n=required_facts if isinstance(required_facts,int) else len({(x.get("id",x.get("statement")) if isinstance(x,Mapping) else x) for x in required_facts if x is not None}); return "unknown" if not complete or n<=0 else ("hard" if revision or cross_stage or n>=3 else "medium" if n==2 else "easy")
def difficulty_basis(required_facts:int|Iterable[Any],*,info_nodes:int|Iterable[Any]|None=None,revision:bool=False,cross_stage:bool=False,complete:bool=True)->dict:
    n=required_facts if isinstance(required_facts,int) else len({(x.get("id",x.get("statement")) if isinstance(x,Mapping) else x) for x in required_facts if x is not None}); m=info_nodes if isinstance(info_nodes,int) else len(info_nodes or []); return {"difficulty":anchor_difficulty(n,revision=revision,cross_stage=cross_stage,complete=complete),"required_facts":n,"info_nodes":m,"revision":bool(revision),"cross_stage":bool(cross_stage),"complete":bool(complete)}
