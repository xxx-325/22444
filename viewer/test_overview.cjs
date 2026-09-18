const assert = require("node:assert/strict");
const test = require("node:test");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const {execFileSync} = require("node:child_process");
const G = require("./graph.js");
const O = require("./overview.js");
const context = {window:{}};
vm.runInNewContext(fs.readFileSync(path.join(__dirname,"data.js"),"utf8"),context);
const currentData = JSON.parse(JSON.stringify(context.window.BENCHMARK_DATA));
// Keep the regression case fixed when the user opens a different run.
const data = JSON.parse(execFileSync(path.join(__dirname,"../.venv/bin/python"),
  ["-c", "import json; from pathlib import Path; from viewer.build_data import build; print(json.dumps(build(Path('runs/validation-quality-final-20260909-v5'))))"],
  {cwd:path.join(__dirname,".."),encoding:"utf8",maxBuffer:16*1024*1024}));
const m=G.model(data), view=O.build(m);

test("current run retains audit candidates and valid evidence references",()=>{
  const model=G.model(currentData), overview=O.build(model);
  assert(currentData.candidate_records.length>=currentData.questions.length);
  for(const group of currentData.groups)for(let step=0;step<6;step++){
    const frame=G.frame(model,group,G.seedFor(model,group),step);
    assert(O.project(overview,frame,step,group).active);
  }
  for(const q of currentData.questions)for(const field of ["answer_points","forbidden_points"])
    for(const point of q[field]||[])for(const id of point.sources||[])assert(model.nodes.has(id),id);
});

test("overview collapses the saved Lambda case to nine meaningful cards",()=>{
  assert.equal(view.cards.length,9);
  assert.equal(view.cards.filter(c=>c.kind==="phase").length,5);
  assert.equal(view.cards.filter(c=>c.kind==="file").length,2);
  assert.equal(view.cards.filter(c=>c.kind==="output").length,2);
  assert(!view.cards.some(c=>/^e\d+$/.test(c.title)));
});
test("all original records and versions remain reachable through details",()=>{
  for(const n of data.nodes.filter(n=>n.kind!=="fact"))assert(view.sourceCards.get(n.id)?.size,`Unmapped ${n.id}`);
  for(const card of view.cards)for(const id of [...card.key,...card.members])assert(m.nodes.has(id),id);
});
test("summary edges close and sequence is not represented as causality",()=>{
  const ids=new Set(view.cards.map(c=>c.id));
  for(const edge of view.edges){assert(ids.has(edge.source));assert(ids.has(edge.target));}
  assert.equal(view.edges.filter(e=>e.kind==="sequence").length,4);
  assert(view.edges.filter(e=>e.kind==="sequence").every(e=>e.label.includes("非因果")));
});
test("every saved group can play all steps without losing source IDs",()=>{
  const initial=JSON.stringify(data);
  for(const group of data.groups){
    for(let step=0;step<6;step++){
      const frame=G.frame(m,group,G.seedFor(m,group),step), projected=O.project(view,frame,step,group);
      assert(projected.active.size<=9);
      for(const id of projected.active)assert(view.cards.some(c=>c.id===id));
      if(step===4)for(const source of group.projection_ids)assert(frame.active.has(source),source);
    }
  }
  assert.equal(JSON.stringify(data),initial);
});
test("QA answer links and group links still point to the original evidence",()=>{
  for(const q of data.questions){
    assert(data.groups.some(g=>g.id===q.evidence_group_id));
    for(const p of [...q.answer_points,...q.forbidden_points])for(const id of p.sources)assert(m.nodes.has(id),id);
    for(const id of q.fact_ids)assert(m.nodes.has(id),id);
  }
});
test("a different run does not inherit this case's handwritten interpretation",()=>{
  const other=G.model({...data,meta:{...data.meta,run:"different-run"}});
  const result=O.build(other);
  assert.equal(result.annotated,false);
  assert(result.cards.filter(c=>c.kind==="phase").every(c=>c.title.startsWith("记录区间")));
});

const codeGroup=data.groups.find(g=>g.id==="code-group-20");
test("initial version survives as background when the saved input does not use it",()=>{
  const expanded=G.frame(m,codeGroup,"e10",3);
  assert(expanded.explored.has("v1"));
  for(const step of [4,5]){
    const frame=G.frame(m,codeGroup,"e10",step);
    assert(frame.shown.has("v1"));
    assert(frame.background.has("v1"));
    assert(!frame.adopted.has("v1"));
    assert(!frame.active.has("v1"));
    const projected=O.project(view,frame,step,codeGroup);
    assert(projected.shown.has("phase-0"));
    assert(!projected.active.has("phase-0"));
    assert.equal(projected.states.get("phase-0").role,"background");
  }
});
test("a version-only visit is never captioned as evidence of the user's request",()=>{
  const card=view.cards.find(c=>c.id==="phase-0");
  for(const step of [3,4,5]){
    const frame=G.frame(m,codeGroup,"e10",step),projected=O.project(view,frame,step,codeGroup);
    const state=projected.states.get(card.id),caption=O.caption(view,card,state,step);
    assert.deepEqual(state.refs,["v1"]);
    assert.equal(caption.title,"初始代码版本");
    assert(caption.lines.some(l=>l.includes("非需求引用")));
    assert(!state.refs.includes("e1"));
    if(step>=4)assert(caption.badge.includes("未纳入"));
  }
});
test("fixed seed and adopted/background membership stay separate across all groups",()=>{
  for(const group of data.groups){
    const seed=G.seedFor(m,group);let explored=new Set();
    for(let step=1;step<6;step++){
      const frame=G.frame(m,group,seed,step);
      assert.equal(frame.seed,seed);
      assert(frame.shown.has(seed));
      for(const id of explored)assert(frame.shown.has(id));
      explored=frame.explored;
      for(const id of frame.background)assert(!frame.adopted.has(id));
      if(step>=4)assert.deepEqual(frame.adopted,G.groupIds(m,group));
      if(step===5)assert.equal(frame.added.size,0);
    }
  }
});
test("one file can contain both adopted evidence and unadopted exploration context",()=>{
  const frame=G.frame(m,codeGroup,"e10",4),projected=O.project(view,frame,4,codeGroup);
  const state=projected.states.get("file-0");
  assert.equal(state.role,"adopted");
  assert(state.adopted.includes("v5"));
  assert(state.background.includes("v1"));
  assert(O.caption(view,view.cards.find(c=>c.id==="file-0"),state,4).badge.includes("背景"));
});
test("choosing a requirement seed prefers a group that cites it over padded context",()=>{
  const groups=G.groupsForSeed(m,"e1");
  assert(groups.length);
  assert(groups.every(g=>g.source_ids.includes("e1")));
  assert(!groups.some(g=>g.id==="general-group-1"));
  assert.equal(G.groupsForSeed(m,"not-a-source").length,0);
});
test("rewinding or selecting a different seed does not retain another path's background",()=>{
  assert(G.frame(m,codeGroup,"e10",4).background.has("v1"));
  assert(!G.frame(m,codeGroup,"e10",1).shown.has("v1"));
  const general=G.groupsForSeed(m,"e1")[0];
  const frame=G.frame(m,general,"e1",1);
  assert.deepEqual([...frame.shown],["e1"]);
  assert.equal(frame.seed,"e1");
});
test("final QA edges only represent this group's published questions",()=>{
  const frame=G.frame(m,codeGroup,"e10",5),projected=O.project(view,frame,5,codeGroup);
  for(const edge of view.edges.filter(e=>e.kind==="evidence")){
    const active=O.edgeRole(edge,projected,frame,5,codeGroup)==="active";
    assert.equal(active,edge.question_ids.some(id=>codeGroup.question_ids.includes(id)));
    if(edge.target==="general")assert(!active);
    assert.equal(O.edgeRole(edge,projected,frame,3,codeGroup),"idle");
  }
});

test("middle-detail view exposes 24 meaningful nodes without raw call nodes",()=>{
  const detail=O.detailed(view);
  assert.equal(detail.cards.length,24);
  assert.equal(detail.cards.filter(c=>c.material_kind==="version").length,8);
  assert.equal(detail.cards.filter(c=>c.material_kind==="symbol").length,2);
  assert.equal(detail.cards.filter(c=>c.material_kind==="result").length,5);
  assert(!detail.cards.some(c=>c.material_kind==="call"));
  const ids=new Set(detail.cards.map(c=>c.id));
  for(const edge of detail.edges){assert(ids.has(edge.source),edge.source);assert(ids.has(edge.target),edge.target);}
  for(const card of detail.cards)for(const id of card.members)assert(m.nodes.has(id));
  for(const card of detail.cards){
    assert(card.x-(card.w||212)/2>=0);
    assert(card.x+(card.w||212)/2<=detail.width);
    assert(card.y-(card.h||118)/2>=0);
    assert(card.y+(card.h||118)/2<=detail.height);
  }
  for(let i=0;i<detail.cards.length;i++)for(let j=i+1;j<detail.cards.length;j++){
    const a=detail.cards[i],b=detail.cards[j];
    const overlap=Math.abs(a.x-b.x)<((a.w||212)+(b.w||212))/2&&Math.abs(a.y-b.y)<((a.h||118)+(b.h||118))/2;
    assert(!overlap,`${a.id} overlaps ${b.id}`);
  }
  assert.equal(view.cards.length,9);
  assert.equal(view.cards.find(c=>c.id==="phase-0").y,350);
});
test("changing density preserves the exact input, seed and background distinctions",()=>{
  const detail=O.detailed(view);
  for(const step of [1,2,3,4,5]){
    const frame=G.frame(m,codeGroup,"e10",step),original=[...frame.active];
    for(const display of [view,detail,view]){
      const projected=O.project(display,frame,step,codeGroup);
      assert.deepEqual([...frame.active],original);
      assert.equal(frame.seed,"e10");
      if(step>=4){
        assert.equal(projected.states.get("phase-0").role,"background");
        if(display===detail)assert.equal(projected.states.get("detail:v1").role,"background");
      }
    }
  }
});
test("version edges never highlight an unseen endpoint as adopted",()=>{
  const detail=O.detailed(view);
  for(const step of [1,2,3,4,5]){
    const frame=G.frame(m,codeGroup,"e10",step),projected=O.project(detail,frame,step,codeGroup);
    for(const edge of detail.edges.filter(e=>e.kind==="version_chain")){
      const role=O.edgeRole(edge,projected,frame,step,codeGroup);
      if(role==="active")assert(edge.sources.every(id=>frame.active.has(id)));
      if(role==="background")assert(edge.sources.every(id=>frame.shown.has(id)));
    }
  }
});
