/* Presentation-only aggregation. The saved evidence graph remains untouched. */
(function (root) {
  "use strict";
  const caseRun = "validation-quality-final-20260909-v5";
  // These source-grounded captions explain this saved case, not inferred stages
  // from the QA engine. Intervals group supporting records; they prove no cause.
  const casePhases = [
    {end: 7, title: "提升题目难度", lines: ["单个长 Lambda 表达式", "创建挑战与解题脚本"], key: ["e1", "e2", "e6"]},
    {end: 11, title: "支持多行粘贴", lines: ["只读一行 → 完整 stdin", "修改挑战输入方式"], key: ["e8", "e10"]},
    {end: 30, title: "修正表达式组合", lines: ["enumerate 输入不符", "修正后本地执行通过"], key: ["e16", "e24", "e28", "e30"]},
    {end: 43, title: "调整 join 接入方式", lines: ["属性访问被 checker 拒绝", "改为显式安全函数绑定"], key: ["e32", "e33", "e35", "e38"]},
    {end: 55, title: "修正局部名检查", lines: ["形参 J 被误当外部名字", "允许局部参数后验证通过"], key: ["e44", "e46", "e48", "e53"]}
  ];

  function build(model) {
    const data = model.data;
    const maximum = Math.max(1, ...data.nodes.map(n => n.order || 0));
    const phases = data.meta.run === caseRun ? casePhases : Array.from({length:5}, (_,i)=>({end: Math.ceil(maximum*(i+1)/5), title:`记录区间 ${i+1}`, lines:["按记录顺序折叠", "未补充语义或因果标签"], key:[]}));
    const cards = phases.map((p, i) => ({id:`phase-${i}`, kind:"phase", title:p.title, lines:p.lines, color:i===0?"#2563d9":"#b46614", x:125+i*238, y:350,
      members:data.nodes.filter(n=>n.kind!=="fact"&&n.lane!=="object"&&n.order>(i?phases[i-1].end:0)&&n.order<=p.end).map(n=>n.id), key:p.key.filter(id=>model.nodes.has(id)), end:p.end}));
    for (const card of cards) if(!card.key.length)card.key=card.members.slice(0,2);
    const changedFiles=data.nodes.filter(n=>n.kind==="file"&&data.nodes.filter(v=>v.kind==="version"&&v.path===n.path).length>1);
    const files=changedFiles.slice(0,2).map((file,i)=>({id:`file-${i}`,kind:"file", title:file.label,
      lines:[data.nodes.filter(n=>n.kind==="symbol"&&n.path===file.path).map(n=>n.label).join(" · ") || "解题表达式与运行环境", `${data.nodes.filter(n=>n.kind==="version"&&n.path===file.path).length} 个保存版本 · 点击查看代码`],
      color:"#7252ad",x:363+i*476,y:110,path:file.path,key:[file.id],members:data.nodes.filter(n=>n.path===file.path).map(n=>n.id)}));
    // Put unchanged support files under the first creation block, not new dots.
    for(const n of data.nodes.filter(n=>n.lane==="object"&&!files.some(f=>f.path===n.path)))cards[0].members.push(n.id);
    const sourceCards=new Map(data.nodes.map(n=>[n.id,new Set()]));
    for(const card of [...cards,...files])for(const id of card.members)sourceCards.get(id)?.add(card.id);
    for(const n of data.nodes.filter(n=>n.kind==="fact"))for(const source of n.sources||[])for(const cid of sourceCards.get(source)||[])sourceCards.get(n.id).add(cid);
    const outputs=["general","code"].map((mode,i)=>({id:mode,kind:"output",mode,title:mode==="general"?"普通 QA":"代码 QA",lines:[`${data.questions.filter(q=>q.qa_mode===mode).length} 道已保存题目`,"查看问题、答案点与审核"],color:mode==="general"?"#119481":"#ce486a",x:363+i*476,y:605,members:[],key:[]}));
    const edges=[];
    cards.slice(1).forEach((card,i)=>edges.push({source:cards[i].id,target:card.id,kind:"sequence",label:"记录先后 · 非因果边"}));
    for(const file of files)for(const card of cards){
      const sources=card.members.filter(id=>model.nodes.get(id)?.kind==="version"&&model.nodes.get(id)?.path===file.path);
      if(sources.length)edges.push({source:file.id,target:card.id,kind:"file",sources,label:"这一段包含该文件的修改版本"});
    }
    for(const output of outputs){
      const sources=new Set(data.questions.filter(q=>q.qa_mode===output.mode).flatMap(q=>[...q.answer_points,...q.forbidden_points].flatMap(p=>p.sources||[])));
      for(const card of cards)if(card.members.some(id=>sources.has(id))){
        const questionIds=data.questions.filter(q=>q.qa_mode===output.mode&&[...q.answer_points,...q.forbidden_points].some(p=>(p.sources||[]).some(id=>card.members.includes(id)))).map(q=>q.id);
        edges.push({source:card.id,target:output.id,kind:"evidence",question_ids:questionIds,label:"已保存 QA 引用了这一段的来源"});
      }
    }
    return {cards:[...files,...cards,...outputs],edges,sourceCards,nodes:model.nodes,symbol_links:model.data.edges.filter(e=>e.kind==="call_reference"),annotated:data.meta.run===caseRun,width:1200,height:730,
      lanes:[[25,"项目对象 · 版本和函数收在文件里"],[253,"关键变化 · 按记录先后排列"],[500,"最终输出 · 点击查看题目"]]};
  }

  function detailed(base) {
    const phases=base.cards.filter(c=>c.kind==="phase");
    const cards=base.cards.map(card=>({...card,x:card.kind==="phase"?150+phases.indexOf(card)*300:card.id==="file-0"?350:card.id==="file-1"?1150:card.id==="general"?470:1030,
      y:card.kind==="phase"?610:card.kind==="file"?90:890}));
    const sourceCards=new Map([...base.sourceCards].map(([id,ids])=>[id,new Set(ids)]));
    const edges=base.edges.filter(e=>e.kind!=="file");
    const add=(card)=>{cards.push(card);for(const id of card.members)sourceCards.get(id)?.add(card.id);};
    const fileCards=base.cards.filter(c=>c.kind==="file");
    for(const [index,file] of fileCards.entries()){
      const versions=file.members.map(id=>base.nodes.get(id)).filter(n=>n?.kind==="version").sort((a,b)=>a.order-b.order);
      for(const phase of phases){
        const local=versions.filter(v=>phase.members.includes(v.id));
        for(const [i,version] of local.entries()){
          const width=local.length>1?132:180;
          add({id:`detail:${version.id}`,kind:"detail",material_kind:"version",rawId:version.id,title:`${version.id} · ${version.label}`,
            lines:[version.previous?`${version.previous} → ${version.id}`:"初始代码版本",`补丁来源 ${version.source}`],color:"#7252ad",x:150+phases.indexOf(phase)*300+(i-(local.length-1)/2)*160,
            y:310+index*130,w:width,h:84,members:[version.id],key:[version.id]});
          edges.push({source:`detail:${version.id}`,target:phase.id,kind:"file",sources:[version.id],label:"该版本由这一段的补丁产生"});
        }
      }
      for(const version of versions){
        const previous=versions.find(v=>v.id===version.previous);
        edges.push({source:previous?`detail:${previous.id}`:file.id,target:`detail:${version.id}`,kind:previous?"version_chain":"file",sources:previous?[previous.id,version.id]:[version.id],all_sources:Boolean(previous),
          label:previous?"同一文件的前后版本（不等于已部署）":"所属文件的初始版本"});
      }
    }
    const symbols=fileCards.flatMap(file=>file.members.map(id=>base.nodes.get(id)).filter(n=>n?.kind==="symbol").map(n=>({node:n,file})));
    symbols.forEach(({node,file},i)=>{
      add({id:`detail:${node.id}`,kind:"detail",material_kind:"symbol",rawId:node.id,title:node.label,lines:[node.path.split("/").at(-1),"对话代码中可见的函数"],
        color:"#45586c",x:200+i*330,y:200,w:230,h:84,members:[node.id],key:[node.id]});
      edges.push({source:file.id,target:`detail:${node.id}`,kind:"file",sources:[node.id],label:"文件中存在的函数符号"});
    });
    for(const link of base.symbol_links)if(symbols.some(s=>s.node.id===link.source)&&symbols.some(s=>s.node.id===link.target)){
      edges.push({source:`detail:${link.source}`,target:`detail:${link.target}`,kind:"syntax",sources:[link.provenance].filter(Boolean),label:`语法调用候选，来源 ${link.provenance}；不保证运行调用`});
    }
    // These are concrete saved result IDs, not generic "tests passed" claims.
    const results=base.annotated?[
      {ids:["e14"],title:"表达式验证失败",lines:["实际错误输出", "对应组合表达式调试"],phase:2,x:670},
      {ids:["e28"],title:"本地执行通过",lines:["solve.py 的运行结果", "不代表 checker 已通过"],phase:2,x:830},
      {ids:["e32"],title:"属性访问被拒绝",lines:["正式 checker 返回拒绝", "需要调整 join 接入方式"],phase:3,x:1050},
      {ids:["e44"],title:"局部名被拒绝",lines:["J 被当作外部名字", "查看实际错误输出"],phase:4,x:1260},
      {ids:["e53","e55"],title:"最终验证结果",lines:["checker / 编译", "两份结果分别保留"],phase:4,x:1430}
    ]:[];
    for(const result of results){
      const ids=result.ids.filter(id=>base.nodes.has(id));if(!ids.length)continue;
      const card={id:`detail:${ids[0]}`,kind:"detail",material_kind:"result",rawId:ids[0],title:result.title,lines:result.lines,
        color:result.ids.includes("e28")||result.ids.includes("e53")?"#119481":"#ce486a",x:result.x,y:745,w:result.phase===4?130:150,h:88,members:ids,key:ids};
      add(card);edges.push({source:phases[result.phase].id,target:card.id,kind:"file",sources:ids,label:"这一段的实际运行结果（不是额外推断的因果边）"});
    }
    return {...base,cards,edges,sourceCards,width:1500,height:1000,
      lanes:[[24,"文件与关键函数"],[263,`${fileCards[0]?.title||"文件一"} · 版本变化`],[393,`${fileCards[1]?.title||"文件二"} · 版本变化`],[530,base.annotated?"对话变化主线":"记录顺序区间（非语义阶段）"],[689,base.annotated?"实际失败与验证":"更多原始记录见来源详情"],[824,"两类 QA 输出"]]};
  }

  function project(view, frame, step, group) {
    const mapped=ids=>new Set([...ids].flatMap(id=>[...(view.sourceCards.get(id)||[])]));
    const active=step===0?new Set(view.cards.map(c=>c.id)):mapped(frame.active);
    if(step===5&&group.question_ids.length)active.add(group.qa_mode);
    const shown=step===0?active:new Set([...mapped(frame.shown),...active]);
    const states=new Map();
    for(const card of view.cards){
      const belongs=id=>view.sourceCards.get(id)?.has(card.id);
      const refs=[...frame.shown].filter(belongs);
      const adopted=[...frame.adopted].filter(belongs), background=[...frame.background].filter(belongs);
      const role=step===0?"overview":adopted.length?"adopted":background.length&&step>=4?"background":active.has(card.id)?"exploring":"idle";
      states.set(card.id,{refs,adopted,background,role,seed:step>0&&Boolean(belongs(frame.seed))});
    }
    return {active,shown,states,added:step===0?new Set():mapped(frame.added)};
  }

  function caption(view, card, state, step) {
    const original={title:card.title,lines:card.lines,badge:card.kind==="phase"?`${card.members.length} 条记录 / 版本 · 展开`:card.kind==="file"?"展开版本与函数 →":card.kind==="detail"?"点击查看原始来源":"查看已保存结果 →"};
    if(step===0||state.role==="idle"||card.kind==="output")return original;
    const refs=state.refs.map(id=>view.nodes.get(id)).filter(Boolean);
    const material=refs.filter(n=>n.kind!=="fact");
    const types={version:"版本",patch:"补丁",message:"消息",call:"调用",result:"结果",file:"文件",symbol:"函数",fact:"事实"};
    const counts=new Map();
    for(const n of refs)counts.set(types[n.kind]||n.kind,(counts.get(types[n.kind]||n.kind)||0)+1);
    const kindText=[...counts].slice(0,2).map(([kind,count])=>`${kind} ${count}`).join(" · ")+(counts.size>2?" 等":"");
    const badge=(state.seed?"含起点 · ":"")+(step>=4?(state.adopted.length?`采用 ${state.adopted.length} 项${state.background.length?` / 背景 ${state.background.length}`:""}`:"探索背景 · 未纳入"):"正在探索");
    if(card.kind==="detail")return {...original,badge};
    if(card.kind==="phase"&&material.length&&material.every(n=>n.kind==="version")){
      return {title:material.every(n=>!n.previous)?"初始代码版本":"历史代码版本",
        lines:[`${material[0].label} · ${material.length===1?material[0].id:material.length+" 个版本"}`,"仅关联代码，非需求引用"],badge};
    }
    return {...original,lines:[kindText,"仅关联证据，非整段采用"],badge};
  }

  function edgeRole(edge, projected, frame, step, group) {
    if(step===0)return "overview";
    if(edge.kind==="evidence")return step===5&&edge.question_ids.some(id=>group.question_ids.includes(id))?"active":"idle";
    if(["file","version_chain","syntax"].includes(edge.kind)){
      if(edge.all_sources){
        if(edge.sources.every(id=>frame.active.has(id)))return "active";
        if(edge.sources.every(id=>frame.shown.has(id))&&edge.sources.some(id=>frame.background.has(id)))return "background";
        return "idle";
      }
      if(edge.sources.some(id=>frame.active.has(id)))return "active";
      if(edge.sources.some(id=>frame.background.has(id)))return "background";
      return "idle";
    }
    // Chronology is a guide, not an explored evidence edge.
    return "guide";
  }
  const api={build,detailed,project,caption,edgeRole};
  if(typeof module!=="undefined"&&module.exports)module.exports=api;else root.EvidenceOverview=api;
})(typeof window!=="undefined"?window:globalThis);
