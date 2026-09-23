(function () {
  "use strict";
  const data = window.BENCHMARK_DATA;
  if (!data) {
    document.getElementById("run-description").textContent = "未加载本地数据。请先生成同目录的 data.js，再重新打开此页。";
    return;
  }
  const G = window.EvidenceGraph, model = G.model(data);
  const summaryView = window.EvidenceOverview.build(model);
  const detailView = window.EvidenceOverview.detailed(summaryView);
  let overview = detailView;
  const $ = id => document.getElementById(id);
  $("run-title").textContent = data.meta.title || "真实对话";
  function el(tag, text, cls) {
    const item = document.createElement(tag);
    if (text !== undefined && text !== null) item.textContent = String(text);
    if (cls) item.className = cls;
    return item;
  }
  function button(text, action, cls) {
    const b = el("button", text, cls); b.type = "button"; b.addEventListener("click", action); return b;
  }
  function tag(text) { return el("span", text, "tag"); }
  function sourceLinks(ids) {
    const list = el("div", null, "source-links");
    for (const id of [...new Set(ids || [])]) {
      const b = button(id, () => selectNode(id, true), "source-link");
      b.disabled = !model.nodes.has(id); b.title = model.nodes.get(id)?.label || "未找到来源"; list.append(b);
    }
    return list;
  }
  const modeName = mode => mode === "general" ? "普通 QA" : "代码 QA";
  const types = {constraint_followthrough:'约束遵循',correction_update:'纠正应用',external_state_application:'外部状态应用',failure_avoidance:'失败规避',verification_reuse:'验证复用',compatibility_preservation:'兼容保留'};
  const difficulties = {easy: "简单", medium: "中等", hard: "困难"};
  const steps = ["关键主线", "挑选种子", "扩展一圈", "继续延伸", "实际证据组", "查看 QA"];
  const defaultGroup = data.groups.find(g => g.qa_mode === "code" && g.question_ids.length && g.source_ids.length > 2) || data.groups[0];
  if (!defaultGroup) { $("run-description").textContent = "该次运行没有保存证据组，无法回放抽取过程。"; return; }
  const state = {step: 0, group: defaultGroup, seed: G.seedFor(model, defaultGroup), selected: null, mode: "general", groupOnly: false, timer: null, qaView: "final", auditStatus: "all"};
  let transform = {x: 0, y: 0, scale: 1};
  const nodeElements = new Map(), edgeElements = [];
  let cardsById=new Map();

  $("run-description").textContent = `${data.meta.records} 条规范化记录 · 其中 ${data.meta.visible_messages} 条可见消息（不是 ${data.meta.records} 轮对话） · ${data.meta.versions} 个文件版本`;
  for (const [value, label] of [[data.meta.facts, "抽取事实"], [data.meta.groups, "证据组"], [data.questions.filter(q => q.qa_mode === "general").length, "普通题"], [data.questions.filter(q => q.qa_mode === "code").length, "代码题"]]) {
    const box = el("div", null, "metric"); box.append(el("strong", value), el("span", label)); $("metrics").append(box);
  }
  steps.forEach((name, i) => {
    const b = button(null, () => { stop(); go(i); }, "step");
    b.append(el("span", String(i + 1).padStart(2, "0"), "number"), el("span", name)); $("steps").append(b);
  });
  for (const mode of ["general", "code"]) {
    const section = el("optgroup"); section.label = modeName(mode);
    data.groups.filter(g => g.qa_mode === mode).forEach(g => {
      const first = model.nodes.get(g.fact_ids[0]);
      const option = el("option", `${g.id.split("-").at(-1).padStart(2, "0")} · ${first?.text.slice(0, 52) || "证据组"}… · ${g.question_ids.length} 题`);
      option.value = g.id; section.append(option);
    });
    $("group-select").append(section);
  }
  $("group-select").value = state.group.id;
  $("group-select").addEventListener("change", e => chooseGroup(e.target.value));

  const NS = "http://www.w3.org/2000/svg";
  function svg(tagName, attrs = {}, text) {
    const item = document.createElementNS(NS, tagName);
    for (const [key, val] of Object.entries(attrs)) item.setAttribute(key, val);
    if (text !== undefined) item.textContent = text;
    return item;
  }
  function drawGraph() {
  nodeElements.clear();edgeElements.length=0;$("scene").replaceChildren();
  $("graph").setAttribute("viewBox", `0 0 ${overview.width} ${overview.height}`);
  $("graph").parentElement.classList.toggle("detail-map",overview===detailView);
  const laneGroup = svg("g"), edgeGroup = svg("g"), nodeGroup = svg("g");
  $("scene").append(laneGroup, edgeGroup, nodeGroup);
  for (const [y, name] of overview.lanes) {
    laneGroup.append(svg("text", {x: 25,y,class:"overview-lane"}, name));
  }
  cardsById=new Map(overview.cards.map(card=>[card.id,card]));
  for (const edge of overview.edges) {
    const a = cardsById.get(edge.source), b = cardsById.get(edge.target);
    if (!a || !b) continue;
    const ah=(a.h||118)/2,bh=(b.h||118)/2,aw=(a.w||212)/2,bw=(b.w||212)/2;
    const path=a.y===b.y?`M${a.x+aw},${a.y} L${b.x-bw},${b.y}`:`M${a.x},${a.y+ah} C${a.x},${(a.y+b.y)/2} ${b.x},${(a.y+b.y)/2} ${b.x},${b.y-bh}`;
    const line = svg("path", {d:path,class:"overview-edge "+edge.kind,"marker-end":"url(#arrow)"});
    line.append(svg("title", {}, `${a.title} → ${b.title} · ${edge.label}`));
    edgeGroup.append(line); edgeElements.push({edge, line});
  }
  for (const card of overview.cards) {
    const width=card.w||212,height=card.h||118,left=-width/2+12;
    const group=svg("g",{class:`graph-node summary-node ${card.kind} ${width<180?"compact":""}`,transform:`translate(${card.x},${card.y})`,tabindex:"0",role:"button","aria-label":`${card.title}，展开证据`});
    group.append(svg("rect",{x:-width/2,y:-height/2,width,height,rx:10,class:"card-surface"}),svg("rect",{x:-width/2,y:-height/2,width:5,height,rx:2,fill:card.color}));
    group.append(svg("text",{x:left,y:-height/2+23,class:"card-title"},card.title));
    card.lines.forEach((line,i)=>group.append(svg("text",{x:left,y:-height/2+44+i*(height<100?16:23),class:"card-description"},line)));
    group.append(svg("text",{x:left,y:height/2-8,class:"card-count"},""));
    group.append(svg("title",{},`${card.title}：${card.lines.join("；")}`));
    group.addEventListener("click",e=>{e.stopPropagation();selectSummary(card);});
    group.addEventListener("keydown",e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();selectSummary(card);}});
    nodeGroup.append(group);nodeElements.set(card.id,group);
  }
  }

  function paint() {
    const frame = G.frame(model, state.group, state.seed, state.step);
    const view=window.EvidenceOverview.project(overview,frame,state.step,state.group);
    for (const [id, item] of nodeElements) {
      const cardState=view.states.get(id);
      const caption=window.EvidenceOverview.caption(overview,cardsById.get(id),cardState,state.step);
      item.querySelector(".card-title").textContent=caption.title;
      [...item.querySelectorAll(".card-description")].forEach((line,i)=>{line.textContent=caption.lines[i]||"";});
      item.querySelector(".card-count").textContent=caption.badge;
      item.setAttribute("aria-label",`${caption.title}，${caption.badge}。点击仅查看详情，不改变起点`);
      item.querySelector("title").textContent=`${caption.title}：${caption.lines.join("；")}。${caption.badge}`;
      item.classList.toggle("dim", !view.shown.has(id));
      item.classList.toggle("active", state.step > 0 && cardState.role==="exploring");
      item.classList.toggle("adopted", cardState.role==="adopted");
      item.classList.toggle("background", cardState.role==="background");
      item.classList.toggle("seed", cardState.seed);
      item.classList.toggle("new", state.step > 0 && view.added.has(id));
      item.classList.toggle("inspected", overview.sourceCards.get(state.selected)?.has(id) || false);
    }
    for (const {edge,line} of edgeElements) {
      const role=window.EvidenceOverview.edgeRole(edge,view,frame,state.step,state.group);
      line.classList.toggle("active", role==="active");
      line.classList.toggle("background", role==="background");
      line.classList.toggle("dim", role==="idle");
    }
    const seed=model.nodes.get(state.seed);
    $("seed-label").textContent=`${state.step===0?"播放起点":"固定起点"}：${seed?.label||"来源"} ${state.seed}`;
    $("inspection-label").textContent=state.selected&&state.selected!==state.seed?`正在查看 ${state.selected} · 起点未改变`:"点击卡片仅查详情；点击“设为起点”才会重选";
    frame.visible=view;
    return frame;
  }

  function renderStep() {
    const frame = paint(), group = state.group;
    [...$("steps").children].forEach((b, i) => {if(i === state.step)b.setAttribute("aria-current","step");else b.removeAttribute("aria-current");});
    $("step-count").textContent = `${state.step + 1} / 6`;
    $("back").disabled = state.step === 0; $("next").disabled = state.step === 5;
    $("graph-title").textContent = state.step === 0 ? `${overview===detailView?"关键关系":"主线概览"} · ${overview.cards.length} 个节点` : `${steps[state.step]} · ${frame.visible.shown.size} 个相关节点（含探索背景）`;
    $("group-badge").textContent = modeName(group.qa_mode);
    $("step-label").textContent = state.step === 0 || state.step >= 4 ? "已保存运行产物" : "扩展演示 · 非真实模型决策";
    $("step-title").textContent = ["先看发生了什么变化", "一个具体对象作起点", "沿直接关系补信息", "连接更远的相关证据", "实际送题的证据投影", "结果分成两条轨道"][state.step];
    $("step-description").textContent = [
      "默认展开文件、函数、代码版本和关键失败 / 验证，保留五个变化作定位。重复工具日志仍放在详情；也可切换到九张卡片的主线概览。",
      `以 ${state.seed} 为种子。后续只沿已记录关系展开；时间接近、主题相似本身不证明因果。`,
      "沿底层来源关系展开，高亮涉及的变化卡片。蓝边表示本步涉及的新证据；高亮卡片不代表它的所有记录都进入模型。",
      "再向外走一圈，可能连接另一时刻的记录。没有新连接就停止扩展，不凭空补足因果链。",
      "绿色为保存证据组包含的内容；灰色虚线保留探索过但未纳入的背景。不会删除探索记录，也不会把背景塞回实际输入。没有日志证明它们被模型判为无用。",
      "读取该证据组最终保留的题目。某组没有公开 QA，不等于没有可问的信息；可能生成失败、被审核拒绝、去重或配额截断。"
    ][state.step];
    $("selection-stats").replaceChildren();
    for (const [number,label] of [[frame.visible.shown.size,"卡片（含背景）"],[state.step > 0 ? frame.added.size : 0,"新增底层证据"],[group.question_ids.length,"本组公开 QA"]]) {
      const item = el("div");item.append(el("strong", number),el("span",label));$("selection-stats").append(item);
    }
    const details = $("step-details"); details.replaceChildren();
    if (state.step === 0) {
      details.append(el("h3","按关键关系看，不逐条画日志"), el("p","关键关系图展开版本、函数和真实失败 / 验证，让代码变化能被看见。点击节点查看完整原文；facts 在实际证据组阶段查看。"));
      for (const [title, content] of [["中间粒度","展开关键代码与运行证据；不把重复调用和每条 fact 铺到画布上。"],["变化卡片","经原文核对的展示分组，不冒充模型自动识别的语义阶段。"],["关系边界","版本链表示前后关系；结果节点保留真实来源，不把先后等同因果。"],["原始证据未删","完整记录仍可按来源链接查看；切换视图不改变起点或证据组。"]]) {
        const row = el("div",null,"relation");row.append(el("strong",title),el("span",content));details.append(row);
      }
    } else if (state.step < 4) {
      details.append(el("h3", state.step === 1 ? "起点原文" : "本步补进来的内容"));
      const added = [...frame.added].slice(0, 12);
      if (!added.length) details.append(el("p","这一圈没有新增连接。可继续查看实际保存的证据组。"));
      for (const id of added) {
        const n = model.nodes.get(id); if (!n) continue;
        const card=el("div",null,"fact-card");card.append(tag(n.label),el("p",n.text.slice(0,150)+(n.text.length>150?"…":"")),sourceLinks([id]));
        const links = frame.links.filter(e=>e.source===id || e.target===id).slice(0,2);
        for(const link of links)card.append(el("div",`${link.source} → ${link.target} · ${G.relationNames[link.kind] || link.kind}`,"relation"));
        details.append(card);
      }
      if(frame.added.size>12)details.append(el("p",`另有 ${frame.added.size-12} 个新增节点，请在图上点击查看。`));
      if(frame.context.length)details.append(el("p",`${frame.context.length} 个节点是连接背景，并不表示实际送给 QA 模型。`,"muted"));
    } else {
      if(state.step===4){
        details.append(el("h3",`${group.fact_ids.length} 条事实 · ${group.projection_ids.length} 条来源投影`));
        if(frame.disconnected.length)details.append(el("p",`${frame.disconnected.length} 个实际投影节点未在上述两圈出现，在此按保存结果补入。`,"muted"));
        details.append(el("p",`运行标注：${group.metadata.stage_count ?? "未记录"} 个语义阶段；图跳数 ${group.metadata.graph_hops ?? "未记录"}。阶段数、圈数和难度互不等同。`,"muted"));
        for(const id of group.fact_ids){const n=model.nodes.get(id);if(!n)continue;const card=el("div",null,"fact-card");card.append(tag(n.source_kind || "来源事实"),el("p",n.text),sourceLinks(n.sources));details.append(card);}
      } else {
        const qs=data.questions.filter(q=>group.question_ids.includes(q.id));
        details.append(el("h3",`${modeName(group.qa_mode)} · ${qs.length} 道公开题`));
        if(!qs.length)details.append(el("p","该组没有最终公开题。页面不会补造 QA 或“证据足够”的判断。","empty"));
        for(const q of qs){const card=el("div",null,"fact-card");card.append(tag(types[q.type] || q.type),el("p",q.question),button("查看答案与审核 ↓",()=>$("qa-section").scrollIntoView({behavior:"auto"}),"text-button"));details.append(card);}
        details.append(el("p","自动审核通过不等于人工定稿。答案点可能仍含并列主张，类型也可能需要调整。","muted"));
      }
      if(frame.background.size){
        const box=el("div",null,"background-explanation");
        box.append(el("h3",`保留探索背景 · ${frame.background.size} 项未纳入`),el("p","这些来源仅沿关系补入展示，没有出现在本组保存的 QA 输入里。未记录排除原因，不能说是模型判定“不重要”。"),sourceLinks([...frame.background]));
        details.append(box);
      }
    }
  }

  function chooseGroup(id, source) {
    const group=data.groups.find(g=>g.id===id);if(!group)return;
    if(source&&!G.groupIds(model,group).has(source))return;
    stop();state.group=group;state.seed=source || G.seedFor(model,group);state.mode=group.qa_mode;state.groupOnly=true;state.step=1;
    $("group-select").value=id;renderStep();renderQA();selectNode(state.seed);
  }
  function selectSummary(card) {
    if(card.kind==="output"){state.mode=card.mode;state.groupOnly=false;renderQA();$("qa-section").scrollIntoView({behavior:"auto"});return;}
    const frame=G.frame(model,state.group,state.seed,state.step);
    const cardState=window.EvidenceOverview.project(overview,frame,state.step,state.group).states.get(card.id);
    const caption=window.EvidenceOverview.caption(overview,card,cardState,state.step);
    const actual=state.step>0?cardState.refs:[];
    const primary=card.key.find(id=>actual.includes(id))||actual.find(id=>model.nodes.get(id)?.kind!=="fact")||actual[0]||card.key[0]||card.members[0];
    if(!primary)return;
    selectNode(primary,true);
    const block=el("div",null,"summary-evidence");
    block.append(el("h3",caption.title),el("p",caption.lines.join("；")),el("p",actual.length?"本步实际关联（不是整段）：":"概览的关键证据：","muted"),sourceLinks(actual.length?actual:card.key));
    if(state.step>=4&&cardState.background.length)block.append(el("p","灰色背景未纳入保存证据组。此处保留它的探索来路，不补造排除理由。","muted"));
    const rest=el("details");rest.append(el("summary",`所属“${card.title}”的全部 ${card.members.length} 条来源（不代表采用）`),sourceLinks(card.members));
    block.append(rest);$("source-detail").lastElementChild.prepend(block);
  }
  function selectNode(id, scroll=false) {
    const n=model.nodes.get(id);if(!n)return;
    stop();
    state.selected=id;paint();
    $("source-meta").textContent=`${id} · ${n.source_kind || n.kind}`;
    const detail=$("source-detail");detail.replaceChildren();
    const info=el("div");info.append(el("h3",n.lane==="object" ? n.label : `${n.label} · ${id}`));
    const dl=el("dl");
    for(const [label,value] of [["项目路径",n.path],["来源顺序",n.order || null],["原始时间",n.timestamp],["原始来源行",n.original_source_line],["上一文件版本",n.previous],["原始来源",n.source],["版本状态",n.status]])if(value){dl.append(el("dt",label),el("dd",value));}
    info.append(dl);
    const related=G.groupsForSeed(model,id);
    info.append(el("p",`当前只是查看。固定起点仍是 ${state.seed}。`,"muted"));
    if(related.length){info.append(button("设为起点，重新演示",()=>{chooseGroup((related.find(g=>g.id===state.group.id)||related[0]).id,id);$("steps").scrollIntoView({behavior:"auto"});},"primary"));info.append(el("p",`可在 ${related.length} 个证据组中演示；优先选直接引用它的组。`,"muted"));}
    else info.append(el("p","此节点未进入已保存的证据投影，不补造出题路径。","muted"));
    const content=el("div");
    const relatedSources=[...(n.sources||[]),n.source,n.previous].filter(Boolean);
    if(relatedSources.length)content.append(sourceLinks(relatedSources));
    const pre=el("pre",n.text,(n.kind==="message"||n.kind==="fact"||n.lane==="object")?"prose":"");
    content.append(pre);
    if(n.kind==="file")content.append(sourceLinks(data.nodes.filter(v=>v.kind==="version"&&v.path===n.path).map(v=>v.id)));
    if(n.kind==="symbol"){
      const versions=(model.adjacent.get(id)||[]).map(link=>model.nodes.get(link.id)).filter(item=>item?.kind==="version");
      content.append(el("p","此函数所在的代码版本（点击查看完整代码）：","muted"),sourceLinks(versions.map(v=>v.id)));
    }
    detail.append(info,content);
    if(scroll)$("source-detail").scrollIntoView({behavior:"auto",block:"center"});
  }

  const stopNames = {target_reached:"已达到目标", pool_exhausted:"证据组候选池已耗尽", budget_exhausted:"探索预算已用完", global_blocker:"鉴权等全局错误"};
  const selectionNames = {published:"已发布", over_quota:"审核通过 · 超出配额", duplicate:"重复 · 未发布", not_approved:"未通过完整审核", rejected:"拒绝", safety_blocked:"敏感信息拦截", not_selected:"未选择"};
  function renderAudit() {
    const rows=(data.candidate_records||[]).filter(row => {
      const q=row.current||row.original||{};
      return (q.origin_qa_mode||q.qa_mode||"code")===state.mode && (!state.groupOnly||q.evidence_group_id===state.group.id) &&
        (state.auditStatus==="all" || row.review_status===state.auditStatus || row.selection_status===state.auditStatus);
    });
    const panel=$("qa-list");panel.replaceChildren();
    $("qa-total").textContent=`共 ${(data.candidate_records||[]).length} 个已保存候选`;
    $("qa-context").textContent=`${modeName(state.mode)} · 当前筛选 ${rows.length} 个；审核状态与发布结果独立，待复核不计目标。`;
    if(!rows.length)panel.append(el("p","当前筛选没有候选。旧运行未保存的内容不会补造。","empty"));
    for(const row of rows){
      const q=row.current||row.original||{}, card=el("article",null,"qa-card"), head=el("header");
      head.append(tag(row.candidate_id), tag(types[q.type||q.category]||q.type||q.category||"类型未保存"),
                  tag(difficulties[q.difficulty]||q.difficulty||"难度未保存"),
                  tag(({approved:"自动审核通过",needs_review:"待复核",rejected:"审核或校验拒绝",not_reviewed:"审核未保存"})[row.review_status]||row.review_status),
                  tag(selectionNames[row.selection_status]||row.selection_status));
      card.append(head,el("h3",q.question||"题干未保存"));
      if(row.record_note)card.append(el("p",row.record_note,"review-warning"));
      for(const [field,label] of [["answer_points","要回答的原子点"],["forbidden_points","有依据的禁止点"]]){
        card.append(el("h4",label));
        if(!Array.isArray(q[field])||!q[field].length)card.append(el("p",field==="answer_points"?"答案点未保存":"未保存具体禁止点","muted"));
        for(const point of Array.isArray(q[field])?q[field]:[]){
          const item=el("div",null,"point");item.append(el("p",typeof point==="string"?point:point.text),sourceLinks(point.sources));card.append(item);
        }
      }
      if(q.use_case)card.append(el("p",`预期用途：${q.use_case}`));
      if(q.difficulty_reason)card.append(el("p",`难度依据：${q.difficulty_reason}`));
      card.append(sourceLinks(q.fact_ids));
      if(q.evidence_group_id&&data.groups.some(g=>g.id===q.evidence_group_id))
        card.append(button("定位证据子图 ↑",()=>{chooseGroup(q.evidence_group_id);go(4);$("steps").scrollIntoView({behavior:"auto"});},"text-button"));
      const detail=el("details");detail.append(el("summary","展开原始审核、拒绝原因与修正前后"));
      for(const [label,value] of [["原始审核",q.review],["拒绝记录",row.rejections],["发布选择原因",row.selection_reason],["重复目标",row.duplicate_of],["校验前候选",row.original],["修正前后版本",row.revisions]]){
        detail.append(el("h4",label),el("pre",value==null||Array.isArray(value)&&!value.length?"未保存 / 未发生":JSON.stringify(value,null,2),"audit-json"));
      }
      card.append(detail);panel.append(card);
    }
  }
  function renderQA() {
    for(const view of ["final","audit"])$("view-"+view).setAttribute("aria-pressed",String(state.qaView===view));
    $("audit-status").disabled=state.qaView!=="audit";
    $("target-progress").textContent=["general","code"].map(mode=>{
      const count=data.questions.filter(q=>q.qa_mode===mode).length,target=data.targets?.[mode];
      return `${modeName(mode)}：${count} / ${target??"目标未保存"} · ${stopNames[data.progress?.stop_reasons?.[mode]]||"旧运行未保存补题停止原因"}`;
    }).join("　｜　");
    for(const mode of ["general","code"]){const b=$("tab-"+mode);b.textContent=`${modeName(mode)} · ${data.questions.filter(q=>q.qa_mode===mode).length}`;b.setAttribute("aria-selected",String(mode===state.mode));b.tabIndex=mode===state.mode?0:-1;}
    $("qa-list").setAttribute("aria-labelledby","tab-"+state.mode);
    if(state.qaView==="audit"){renderAudit();return;}
    const list=data.questions.filter(q=>q.qa_mode===state.mode&&(!state.groupOnly||q.evidence_group_id===state.group.id));
    $("qa-total").textContent=`共 ${data.questions.length} 道`;
    $("qa-context").textContent=state.groupOnly?`仅看当前证据组：${state.group.id} · ${list.length} 道${modeName(state.mode)}`:`全部已保存${modeName(state.mode)} · ${list.length} 道`;
    const panel=$("qa-list");panel.replaceChildren();
    if(!list.length)panel.append(el("p","当前筛选没有公开题。可以切换轨道，或点击“查看全部题目”。","empty"));
    for(const q of list){
      const card=el("article",null,"qa-card");card.id="qa-"+q.id;
      const head=el("header");head.append(tag(`${types[q.type]||q.type} · ${q.type}`),tag(`${difficulties[q.difficulty]||q.difficulty}（运行标注）`));
      const status=tag(q.status==="approved"?"原记录：自动通过":"原记录：待复核");status.classList.add("status");if(q.status!=="approved")status.classList.add("pending");head.append(status);
      card.append(head,el("h3",q.question));
      for(const [field,title,css] of [["answer_points","要回答的点",""],["forbidden_points","不能答错的点","forbidden"]]){
        card.append(el("div",title,"point-heading "+css));
        if(!q[field]?.length)card.append(el("p","该题没有保存具体禁止点。","muted"));
        (q[field]||[]).forEach((point,i)=>{const item=el("div",null,"point "+css);item.append(el("p",`${i+1}. ${point.text}`),sourceLinks(point.sources));card.append(item);});
      }
      const actions=el("div",null,"qa-actions");
      actions.append(button("定位证据子图 ↑",()=>{chooseGroup(q.evidence_group_id);go(4);$("steps").scrollIntoView({behavior:"auto"});},"text-button"));
      card.append(actions);
      const review=el("details");review.append(el("summary","查看用途、难度依据与原始审核"));
      review.append(el("p","这是原始模型审核结果。请同时读状态和理由；本页不重新判题或消除二者的冲突。","review-warning"));
      for(const [label,value] of [["预期用途",q.use_case],["难度理由",q.difficulty_reason],["轨道边界",q.track],["适用截止",q.cutoff===undefined?null:`第 ${q.cutoff} 条规范化记录`],["审核理由",q.review?.reason]])if(value)review.append(el("p",`${label}：${value}`));
      for(const [key,label] of [["history_evidence_required","历史证据必需"],["current_snapshot_alone_sufficient","仅当前快照即可回答"],["history_requirement_correct","历史定位正确"]])if(q.review&&key in q.review)review.append(el("p",`${label}（原始布尔标注）：${q.review[key]}`));
      review.append(el("p",`语义阶段 ${q.stage_count??"未记录"} · 图跳数 ${q.graph_hops??"未记录"} · 推理跳数 ${q.reasoning_hops??"未记录"}`),sourceLinks(q.fact_ids));
      card.append(review);panel.append(card);
    }
  }

  function go(step){state.step=Math.max(0,Math.min(5,step));if(state.step===5){state.mode=state.group.qa_mode;state.groupOnly=true;renderQA();}renderStep();}
  function stop(){if(state.timer)clearInterval(state.timer);state.timer=null;$("play").textContent="▶ 播放讲解";}
  $("play").addEventListener("click",()=>{if(state.timer){stop();return;}if(state.step===5)go(0);$("play").textContent="Ⅱ 暂停";go(state.step+1);state.timer=setInterval(()=>{if(state.step>=5){stop();return;}go(state.step+1);if(state.step===5)stop();},2400);});
  $("back").addEventListener("click",()=>{stop();go(state.step-1);});
  $("next").addEventListener("click",()=>{stop();go(state.step+1);});
  for(const view of ["final","audit"])$("view-"+view).addEventListener("click",()=>{state.qaView=view;state.groupOnly=false;renderQA();});
  $("audit-status").addEventListener("change",e=>{state.auditStatus=e.target.value;renderQA();});
  $("all-qa").addEventListener("click",()=>{state.groupOnly=false;renderQA();});
  for(const mode of ["general","code"]){const b=$("tab-"+mode);b.addEventListener("click",()=>{state.mode=mode;state.groupOnly=false;renderQA();});b.addEventListener("keydown",e=>{if(e.key==="ArrowLeft"||e.key==="ArrowRight"){e.preventDefault();const other=mode==="general"?"code":"general";state.mode=other;state.groupOnly=false;renderQA();$("tab-"+other).focus();}});}
  $("method-link").addEventListener("click",()=>{$("method").open=true;$("method").scrollIntoView({behavior:"auto"});});
  for(const [id,view] of [["density-detail",detailView],["density-summary",summaryView]])$(id).addEventListener("click",()=>{
    stop();overview=view;transform={x:0,y:0,scale:1};drawGraph();applyTransform();renderStep();
    $("density-detail").setAttribute("aria-pressed",String(overview===detailView));$("density-summary").setAttribute("aria-pressed",String(overview===summaryView));
  });
  for(const layer of data.adaptive.layers || [])$("layer-log").append(tag(`层 ${layer.depth} · ${layer.states} 个状态 · 累计接受 ${layer.accepted}`));
  if(!(data.adaptive.layers || []).length)$("layer-log").textContent="本次运行未保存扩展层统计。";
  $("run-file").textContent=`数据目录：runs/${data.meta.run}/。隐私投影：${data.meta.redactions.path_replacements || 0} 处路径替换，${data.meta.redactions.credential_fields_hidden || 0} 个疑似凭据字段隐藏。原始内容仍保留在原运行目录。`;
  function applyTransform(){$("scene").setAttribute("transform",`translate(${transform.x},${transform.y}) scale(${transform.scale})`);}
  function zoom(factor){const previous=transform.scale;transform.scale=Math.max(.55,Math.min(4,previous*factor));const ratio=transform.scale/previous,cx=overview.width/2,cy=overview.height/2;transform.x=cx-(cx-transform.x)*ratio;transform.y=cy-(cy-transform.y)*ratio;applyTransform();}
  $("zoom-in").addEventListener("click",()=>zoom(1.25));$("zoom-out").addEventListener("click",()=>zoom(.8));$("zoom-fit").addEventListener("click",()=>{transform={x:0,y:0,scale:1};applyTransform();});
  $("graph").addEventListener("wheel",e=>{e.preventDefault();zoom(e.deltaY<0?1.08:1/1.08);},{passive:false});
  let drag=null;
  $("graph").addEventListener("pointerdown",e=>{if(e.target.closest(".graph-node"))return;drag={x:e.clientX,y:e.clientY,tx:transform.x,ty:transform.y};$("graph").setPointerCapture(e.pointerId);});
  $("graph").addEventListener("pointermove",e=>{if(!drag)return;const box=$("graph").getBoundingClientRect(),ratio=Math.max(overview.width/box.width,overview.height/box.height);transform.x=drag.tx+(e.clientX-drag.x)*ratio;transform.y=drag.ty+(e.clientY-drag.y)*ratio;applyTransform();});
  for(const event of ["pointerup","pointercancel"])$("graph").addEventListener(event,()=>{drag=null;});
  document.addEventListener("visibilitychange",()=>{if(document.hidden)stop();});
  drawGraph();renderStep();renderQA();selectNode(state.seed);
})();
