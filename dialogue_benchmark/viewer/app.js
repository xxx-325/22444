(() => {
  'use strict';
  const data = JSON.parse(document.getElementById('run-data').textContent);
  const $ = id => document.getElementById(id);
  const records = new Map(data.records.map(r => [r.id, r]));
  const events = new Map(data.graph.events.map(e => [e.id, e]));
  const versions = new Map(data.graph.versions.map(v => [v.id, v]));
  const facts = new Map(data.facts.map(f => [f.id, f]));
  const stages = [
    ['原始证据', '规范化 · 静态', '消息、调用与补丁'],
    ['事件快照', '逐事件 · 静态', '每个时点的已知文件状态'],
    ['图结构', 'AST 与版本 · 静态', '实际提取的关系'],
    ['Facts', '模型生成 · 待核验', '从局部原文抽取事实'],
    ['候选 QA', '模型生成 · 待核验', '问题、答题要点与来源']
  ];
  const kindNames = {message:'消息',call:'工具调用',result:'工具结果',patch:'补丁',observation:'代码观察'};
  const categories = {fact_recall:'事实回忆',history_tracking:'历史追踪',behavior_inference:'行为推演',failure_diagnosis:'故障诊断'};
  const difficulties = {easy:'简单',medium:'中等',hard:'困难'};
  const state = {stage:0, index:0, selected:'', graphMode:'events', playing:null, filter:'all', search:''};

  function el(tag, attrs={}, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) {
      if (key === 'class') node.className = value;
      else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, value);
    }
    for (const child of children.flat()) {
      if (child !== null && child !== undefined) node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
  }
  const tag = (text, cls='') => el('span',{class:'tag '+cls},text);
  const pathName = path => path.split(/[\\/]/).pop();
  const current = () => data.records[state.index];
  const snapshot = () => data.graph.snapshots.find(s => s.at === current().order);
  const currentGraph = () => data.states[data.snapshot_states[String(current().order)]] || {nodes:[],edges:[],unresolved:[]};
  function preview(r) {
    if (r.kind === 'patch') return Object.entries(r.changes||{}).map(([p,c]) => `${c.type}  ${p}`).join('\n');
    return r.text || r.content || '';
  }
  function sourceButtons(ids=[]) {
    return el('div',{class:'sources'}, ids.map(id => el('button',{class:'source', onclick:() => showEvidence(id)},id)));
  }
  function codeBlock(text, diff=false) {
    if (!diff) return el('pre',{class:'code'}, text || '无代码内容');
    return el('pre',{class:'code'}, (text||'').split('\n').map(line => el('span',{
      class:'diff-line '+(line.startsWith('+')?'add':line.startsWith('-')?'del':line.startsWith('@@')?'hunk':'')
    }, line || ' ')));
  }
  function evidenceBody(id) {
    const r = records.get(id);
    if (!r) return el('div',{class:'empty'},'来源未在本次产物中找到：'+id);
    const ev = events.get(id);
    const body = el('div',{},el('div',{class:'detail-meta'},tag(id,'green'),tag(kindNames[r.kind]||r.kind),
      r.role?tag(r.role):null,r.timestamp?tag(r.timestamp):null),
      el('p',{class:'muted'},`规范化序号 ${r.order} · 产物 source_line ${r.source_line}`));
    if(r.original_source_line) body.append(el('p',{class:'muted'},'原始 session 行号 '+r.original_source_line));
    if(ev?.call_source) body.append(el('div',{class:'detail-block'},el('h3',{},'对应调用'),sourceButtons([ev.call_source])));
    if(r.kind === 'patch') {
      body.append(el('div',{class:'detail-meta'},tag(r.success===true?'补丁应用成功':'未确认应用成功',r.success?'green':'warn')));
      for (const [p,c] of Object.entries(r.changes||{})) body.append(el('div',{class:'detail-block'},el('h3',{class:'path'},p),tag(c.type),codeBlock(c.unified_diff||c.content||'',Boolean(c.unified_diff))));
    } else body.append(codeBlock(r.text||r.content||''));
    return body;
  }
  function showEvidence(id) {
    $('dialog-title').textContent = '原始证据 · '+id;
    $('dialog-body').replaceChildren(evidenceBody(id));
    if (!$('evidence-dialog').open) $('evidence-dialog').showModal();
  }
  function metric(value, label) {return el('div',{class:'metric'},el('strong',{},value),el('span',{},label));}
  function stop() { if(state.playing) clearInterval(state.playing); state.playing=null; $('play').textContent='播放'; }
  function setIndex(index) {state.index=Math.max(0,Math.min(data.records.length-1,index));state.selected='';render();}
  function setStage(stage) {stop();state.stage=stage;state.selected='';render();}

  function render() {
    $('workspace').classList.toggle('graph-layout',state.stage===2);
    $('steps').replaceChildren(...stages.map(([label,sub],i) => el('button',{
      class:'step','aria-current':i===state.stage?'step':'false',onclick:()=>setStage(i)
    },el('span',{class:'step-num'},String(i+1).padStart(2,'0')),label,el('small',{},sub))));
    $('stage-kicker').textContent=stages[state.stage][1];
    $('stage-title').textContent=stages[state.stage][2];
    $('previous-step').disabled=state.stage===0;$('next-step').disabled=state.stage===4;
    $('timeline').hidden=state.stage>=3;
    $('event-range').max=String(data.records.length);$('event-range').value=String(state.index+1);
    $('event-position').textContent=`${current().id} / ${data.records.length}`;
    $('event-summary').textContent=`${kindNames[current().kind]}${current().role?' · '+current().role:''}`;
    $('prev-event').disabled=state.index===0;$('next-event').disabled=state.index===data.records.length-1;
    $('patch-stops').replaceChildren(...data.records.filter(r=>r.kind==='patch').map(r=>el('button',{
      class:r.id===current().id?'current':'',onclick:()=>{stop();setIndex(data.records.indexOf(r));},title:`${r.id} · 补丁`
    },r.id, r.success?' · 修改':' · 失败')));
    const nowSnapshot=snapshot();
    const before=data.graph.snapshots.filter(s=>s.at<current().order).at(-1);
    const changed=Object.entries(nowSnapshot?.files||{}).filter(([p,v])=>before?.files[p]!==v);
    $('event-delta').replaceChildren(el('span',{class:'delta-label'},'本步变化'),...changed.map(([p,v])=>el('span',{class:'delta-item'},`${pathName(p)}：${before?.files[p]||'未观察'} → ${v}`)));
    if(!changed.length)$('event-delta').append(el('span',{class:'muted'},current().kind==='patch'?'文件状态没有推进':'新增观察记录 · 代码状态不变'));
    $('primary').replaceChildren();$('inspector').replaceChildren();
    const g=currentGraph(),s=snapshot();
    if(state.stage<3) $('metrics').replaceChildren(metric(state.index+1,'已观察事件'),metric(Object.keys(s?.files||{}).length,'文件状态'),metric(g.nodes.length,'代码图节点'),metric(g.edges.length,'代码图关系'));
    else if(state.stage===3) $('metrics').replaceChildren(metric(data.facts.length,'模型事实'),metric(data.scope?.dialogue?.length||0,'回取记录'),metric(data.scope?.nodes?.length||0,'局部图节点'));
    else $('metrics').replaceChildren(metric(data.candidates.length,'结构检查后候选'),metric(data.result?.questions?.length||0,'完成审核后候选'),metric(data.result?.rejected?.length||0,'已记录淘汰'));
    $('footer-state').textContent=state.stage<3?`截止 ${current().id} · 只展示截至此刻的状态`:'整段运行产物 · 不属于早期事件快照';
    [renderEvidence,renderSnapshots,renderGraph,renderFacts,renderQA][state.stage]();
  }

  function renderEvidence() {
    const filter=el('select',{'aria-label':'证据类型'},el('option',{value:'all'},'全部类型'),Object.entries(kindNames).map(([v,t])=>el('option',{value:v},t)));
    filter.value=state.filter;filter.addEventListener('change',()=>{state.filter=filter.value;renderEvidenceList();});
    const search=el('input',{type:'search',placeholder:'搜索当前已观察证据','aria-label':'搜索证据'});search.value=state.search;
    search.addEventListener('input',()=>{state.search=search.value;renderEvidenceList();});
    $('primary').append(el('div',{class:'filters'},filter,search),el('div',{id:'evidence-list',class:'list'}));
    renderEvidenceList();
  }
  function renderEvidenceList() {
    const list=$('evidence-list');if(!list)return;
    const rows=data.records.slice(0,state.index+1).filter(r=>(state.filter==='all'||r.kind===state.filter)&&(`${r.id} ${preview(r)}`).toLowerCase().includes(state.search.toLowerCase()));
    const selected=state.selected||current().id;
    list.replaceChildren(...rows.map(r=>el('button',{class:'row '+(selected===r.id?'selected':''),onclick:()=>{state.selected=r.id;renderEvidenceList();}},
      el('div',{class:'row-head'},el('span',{class:'row-id'},r.id),tag(kindNames[r.kind]),r.role?tag(r.role):null),el('div',{class:'row-text row-preview'},preview(r)))));
    if(!rows.length)list.append(el('div',{class:'empty'},'无匹配证据'));
    $('inspector').replaceChildren(el('h3',{},'证据原文'),evidenceBody(selected));
  }

  function renderSnapshots() {
    const snap=snapshot();
    $('primary').append(el('div',{class:'section-top'},el('h3',{},'当前文件状态'),tag(current().id,'green')));
    const entries=Object.entries(snap?.files||{});
    if(!entries.length){$('primary').append(el('div',{class:'empty'},'尚未观察到可回放的文件'));$('inspector').append(evidenceBody(current().id));return;}
    for(const [p,id]of entries){const v=versions.get(id);$('primary').append(el('div',{class:'file-entry'},el('button',{onclick:()=>{state.selected=id;render();}},el('span',{class:'path'},p),tag(id,'blue'),tag(v.status)),el('div',{class:'muted'},`来源 ${v.source} · ${v.previous?'前版 '+v.previous:'首次观察'}`)));}
    const selected=versions.get(state.selected)||versions.get(entries[0][1]);
    renderVersion(selected);
  }
  function renderVersion(v) {
    $('inspector').replaceChildren(el('h3',{class:'path'},v.path),el('div',{class:'detail-meta'},tag(v.id,'blue'),tag(v.status),tag('观察于 '+v.source)),sourceButtons([v.source]));
    if(v.previous){const prev=versions.get(v.previous);const record=records.get(v.source);const change=Object.entries(record?.changes||{}).find(([p,c])=>p.replaceAll('\\','/')===v.path||c.move_path?.replaceAll('\\','/')===v.path)?.[1];
      const d=el('details',{},el('summary',{},`本次补丁 · ${v.previous} → ${v.id}`),change?.unified_diff?codeBlock(change.unified_diff,true):el('p',{class:'muted'},'本次事件未提供匹配的 unified diff'));
      $('inspector').append(d,el('details',{},el('summary',{},'上一版本代码 · '+prev.id),codeBlock(prev.content)));
    }
    $('inspector').append(el('div',{class:'detail-block'},el('h3',{},'回放后的完整代码'),codeBlock(v.content)));
  }

  const NS='http://www.w3.org/2000/svg';
  function svgEl(name,attrs={},text){const n=document.createElementNS(NS,name);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,String(v));if(text!==undefined)n.textContent=text;return n;}
  function renderGraph() {
    $('primary').append(el('div',{class:'graph-controls'},...['events','code','versions'].map((mode,i)=>el('button',{'aria-pressed':state.graphMode===mode,onclick:()=>{state.graphMode=mode;state.selected='';render();}},['事件驱动图','代码关系','版本演变'][i]))));
    const g=currentGraph();let nodes=[],edges=[];
    if(state.graphMode==='code'||state.graphMode==='events'){
      const offset=state.graphMode==='events'?250:0;
      const files=g.nodes.filter(n=>n.kind==='file');let y=35;
      for(const file of files){const functions=g.nodes.filter(n=>n.id.startsWith(file.id+'::'));nodes.push({...file,x:25+offset,y,label:pathName(file.id),sub:file.version,changed:versions.get(file.version)?.source===current().id});functions.forEach((n,i)=>nodes.push({...n,x:310+offset,y:y+i*105,label:n.name,sub:n.version,changed:versions.get(n.version)?.source===current().id}));y+=Math.max(1,functions.length)*105+30;}
      edges=[...g.edges];
      if(state.graphMode==='events'){
        const patches=data.graph.events.filter(e=>e.kind==='patch'&&e.success&&e.order<=current().order);
        patches.forEach((e,i)=>{nodes.push({id:e.id,kind:'event',x:15,y:35+i*90,label:e.id+' · 成功补丁',sub:e.affected_paths.length+' 个文件',changed:e.id===current().id});
          for(const p of e.affected_paths)if(files.some(f=>f.id===p))edges.push({from:e.id,to:p,kind:'modifies',source:e.id});
        });
      }
    }else{
      const all=data.graph.versions.filter(v=>v.observed_at<=current().order);const lanes=[...new Set(all.map(v=>v.path))];
      for(const[pIndex,path]of lanes.entries()){const lane=all.filter(v=>v.path===path);lane.forEach((v,i)=>{nodes.push({id:v.id,version:v.id,kind:'version',x:25+i*235,y:35+pIndex*140,label:v.id+' · '+v.source,sub:pathName(v.path)});if(v.previous&&all.some(p=>p.id===v.previous))edges.push({from:v.previous,to:v.id,kind:'version',source:v.source});});}
    }
    const width=Math.max(570,...nodes.map(n=>n.x+260)),height=Math.max(220,...nodes.map(n=>n.y+105));
    const wrap=el('div',{class:'graph-wrap'});const svg=svgEl('svg',{class:'diagram',viewBox:`0 0 ${width} ${height}`,role:'img','aria-label':state.graphMode==='code'?'截至当前事件的代码关系图':'截至当前事件的文件版本演变图'});
    if(width>650){svg.style.width=width+'px';svg.style.maxWidth='none';}
    const defs=svgEl('defs');const marker=svgEl('marker',{id:'arrow',markerWidth:8,markerHeight:8,refX:7,refY:4,orient:'auto'});marker.append(svgEl('path',{d:'M0,0 L8,4 L0,8',fill:'#829b8b'}));defs.append(marker);svg.append(defs);
    const map=new Map(nodes.map(n=>[n.id,n]));
    edges.forEach((edge,i)=>{const a=map.get(edge.from),b=map.get(edge.to);if(!a||!b)return;let d;
      if(a.x<b.x)d=`M${a.x+205},${a.y+30} C${a.x+250},${a.y+30} ${b.x-45},${b.y+30} ${b.x},${b.y+30}`;
      else if(a.x===b.x)d=`M${a.x+205},${a.y+40} C${a.x+245},${a.y+40} ${b.x+245},${b.y+20} ${b.x+205},${b.y+20}`;
      else d=`M${a.x},${a.y+40} C${a.x-25},${a.y+50} ${b.x+240},${b.y+50} ${b.x+205},${b.y+40}`;
      const line=svgEl('path',{d,class:'edge '+(edge.kind==='call_reference'?'call':edge.kind==='version'?'version-edge':edge.kind==='modifies'?'modifies':'')+(edge.source===current().id?' latest':''),'marker-end':'url(#arrow)'});line.append(svgEl('title',{},`${edge.kind} · ${edge.source}`));svg.append(line);
    });
    for(const n of nodes){const group=svgEl('g',{class:`node ${n.kind} ${state.selected===n.id?'active':''} ${n.changed?'changed':''}`,role:'button',tabindex:0,'aria-label':n.id});group.append(svgEl('rect',{x:n.x,y:n.y,width:205,height:65,rx:5}));
      const title=n.label.length>27?n.label.slice(0,25)+'…':n.label;group.append(svgEl('text',{x:n.x+12,y:n.y+25},title),svgEl('text',{x:n.x+12,y:n.y+47,style:'fill:#65706a;font-size:11px'},n.sub),svgEl('title',{},n.id));
      const select=()=>{state.selected=n.id;render();};group.addEventListener('click',select);group.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select();}});svg.append(group);}
    wrap.append(svg);$('primary').append(wrap,el('div',{class:'legend'},el('span',{},el('i'),'包含关系'),el('span',{},el('i',{class:'call'}),'语法调用引用'),el('span',{},el('i',{class:'version'}),'版本前后继'),el('span',{},el('i',{class:'modify'}),'补丁改动文件')));
    if(!nodes.length)$('primary').append(el('div',{class:'empty'},'此刻尚无图节点'));
    const rows=edges.map(e=>el('div',{class:'detail-block'},el('div',{class:'path'},`${e.from} → ${e.to}`),tag(e.kind),sourceButtons([e.source])));
    $('primary').append(el('details',{},el('summary',{},`关系明细 · ${edges.length}`),rows));
    if(state.graphMode==='versions'){
      const v=versions.get(state.selected)||versions.get(nodes[0]?.version);if(v)renderVersion(v);
    }else if(state.graphMode==='events'&&events.get(state.selected)?.kind==='patch'){
      $('inspector').append(el('h3',{},'改动事件'),evidenceBody(state.selected));
    }else{
      const n=g.nodes.find(n=>n.id===state.selected)||g.nodes[0];if(n){const v=versions.get(n.version);$('inspector').append(el('h3',{class:'path'},n.id),el('div',{class:'detail-meta'},tag(n.kind),tag(n.version,'blue')),sourceButtons([v.source]));
        const code=n.line?v.content.split('\n').slice(n.line-1,n.end_line).join('\n'):v.content;
        $('inspector').append(codeBlock(code));
      }
      $('inspector').append(el('details',{},el('summary',{},`未解析引用 · ${g.unresolved.length}`),g.unresolved.map(r=>el('div',{class:'detail-block'},el('div',{class:'path'},r.from),el('code',{},r.expression),sourceButtons([r.source])))));
    }
    $('inspector').append(el('div',{class:'notice'},'图中只展示静态产物已有关系。命令管道传递和字符串内 Lambda 数据流未自动建边。'));
  }

  function renderFacts() {
    $('primary').append(el('div',{class:'section-top'},el('h3',{},'Facts · 模型原始输出'),tag('整段运行后','blue')));
    const list=el('div',{class:'list'});$('primary').append(list);
    const selected=facts.get(state.selected)||data.facts[0];
    list.append(...data.facts.map(f=>el('button',{class:'row '+(selected?.id===f.id?'selected':''),onclick:()=>{state.selected=f.id;render();}},el('div',{class:'row-head'},el('span',{class:'row-id'},f.id),tag(`${f.sources?.length||0} 个来源`)),el('div',{class:'row-text row-preview'},f.statement))));
    if(!selected){list.append(el('div',{class:'empty'},'本次运行尚无 Facts'));return;}
    $('inspector').append(el('h3',{},selected.id+' · 模型陈述'),el('div',{class:'detail-block detail-text'},selected.statement),sourceButtons(selected.sources));
    for(const id of selected.sources||[]) $('inspector').append(el('details',{},el('summary',{},'证据 '+id),evidenceBody(id)));
    $('inspector').append(el('div',{class:'notice'},'Facts 未经人工确认；原文来源存在不等于结论得到充分支持。'));
  }

  function renderQA() {
    const candidates=data.candidates.length?data.candidates:(data.result?.questions||[]);
    const selected=candidates.find(q=>q.id===state.selected)||candidates[0];
    $('primary').append(el('div',{class:'section-top'},el('h3',{},'候选题 · 原始生成结果'),tag('难度为预估','warn')));
    if(data.failure&&!data.review)$('primary').append(el('div',{class:'notice'},`审核未完成 · ${data.failure.error_type||'运行中断'}。当前候选不代表通过审核。`));
    for(const q of candidates)$('primary').append(el('button',{class:'row '+(selected?.id===q.id?'selected':''),onclick:()=>{state.selected=q.id;render();}},el('div',{class:'row-head'},el('span',{class:'row-id'},q.id),tag(categories[q.category]||q.category,'blue'),tag(difficulties[q.difficulty]||q.difficulty)),el('div',{class:'row-text'},q.question)));
    if(!selected){$('primary').append(el('div',{class:'empty'},'本次运行没有候选题'));return;}
    $('inspector').append(el('h3',{},selected.id+' · 答题要点'),...selected.answer_points.map((point,i)=>el('div',{class:'answer'},el('div',{class:'answer-num'},'原子点 '+(i+1)),el('div',{class:'detail-text'},point.text),sourceButtons(point.sources))));
    $('inspector').append(el('div',{class:'detail-block'},el('h3',{},'难度依据 · 模型判断'),el('p',{class:'detail-text'},selected.difficulty_reason)),el('div',{class:'detail-block'},el('h3',{},'记忆依赖 · 模型判断'),el('p',{class:'detail-text'},selected.memory_requirement)));
    for(const id of selected.fact_ids||[]) {const f=facts.get(id);$('inspector').append(el('details',{},el('summary',{},'关联 Fact · '+id),el('p',{class:'detail-text'},f?.statement||'未找到'),sourceButtons(f?.sources)));}
    const review=data.review?.reviews?.find(r=>r.id===selected.id);
    $('inspector').append(el('details',{},el('summary',{},'审核状态'),review?codeBlock(JSON.stringify(review,null,2)):el('p',{class:'muted'},'没有完成的模型审核记录')));
  }

  $('run-name').textContent=data.run;
  $('run-status').textContent=data.failure&&!data.review?'审核阶段中断 · 候选未完成核验':data.review?'自动审核已有记录 · 仍需人工确认':'本地阶段产物 · 待审核';
  $('previous-step').addEventListener('click',()=>setStage(Math.max(0,state.stage-1)));
  $('next-step').addEventListener('click',()=>setStage(Math.min(4,state.stage+1)));
  $('prev-event').addEventListener('click',()=>{stop();setIndex(state.index-1);});
  $('next-event').addEventListener('click',()=>{stop();setIndex(state.index+1);});
  $('event-range').addEventListener('input',e=>{stop();setIndex(Number(e.target.value)-1);});
  $('play').addEventListener('click',()=>{if(state.playing){stop();return;}if(state.index===data.records.length-1)setIndex(0);$('play').textContent='暂停';state.playing=setInterval(()=>{if(state.index===data.records.length-1){stop();return;}setIndex(state.index+1);},1100);});
  $('close-dialog').addEventListener('click',()=>$('evidence-dialog').close());
  $('evidence-dialog').addEventListener('click',e=>{if(e.target===$('evidence-dialog')){const r=e.target.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)e.target.close();}});
  window.addEventListener('pagehide',stop);
  render();
})();
