/* Pure display helpers. These do not reproduce the production search engine. */
(function (root) {
  "use strict";
  const colors = {message: "#2563d9", tool: "#7a8798", patch: "#df821a", version: "#9147d9", object: "#45586c", general: "#119481", code: "#db5473"};
  const relationNames = {tool_result: "调用 → 返回结果", version_source: "补丁 → 产生版本", previous_version: "旧版本 → 新版本", file_version: "版本 → 所属文件", symbol_version: "版本 → 包含符号", contains: "文件 → 包含函数", call_reference: "语法调用候选（非运行保证）", fact_source: "原始来源 → 抽取事实"};

  function model(data) {
    const nodes = new Map(data.nodes.map(n => [n.id, n]));
    const adjacent = new Map(data.nodes.map(n => [n.id, []]));
    for (const edge of data.edges) {
      if (!nodes.has(edge.source) || !nodes.has(edge.target)) continue;
      adjacent.get(edge.source).push({id: edge.target, edge});
      adjacent.get(edge.target).push({id: edge.source, edge});
    }
    return {data, nodes, adjacent};
  }

  function groupIds(m, group) {
    return new Set([...group.projection_ids, ...group.fact_ids].filter(id => m.nodes.has(id)));
  }

  function seedFor(m, group) {
    return group.source_ids.filter(id => m.nodes.has(id)).sort((a, b) => m.nodes.get(a).order - m.nodes.get(b).order)[0] || group.fact_ids[0];
  }

  function groupsForSeed(m, source) {
    // Prefer a group that actually cites this source over one carrying it only
    // as neighboring/full-range context. Never silently bind an unrelated seed.
    const direct = m.data.groups.filter(g => g.source_ids.includes(source) || g.fact_ids.includes(source));
    return direct.length ? direct : m.data.groups.filter(g => g.projection_ids.includes(source));
  }

  function frame(m, group, seed, step) {
    const target = groupIds(m, group);
    const pool = new Set(target);
    // One layer of recorded version/object context helps explain connectivity.
    // It is display context, not an assertion about the model's input.
    for (const id of target) for (const link of m.adjacent.get(id) || []) {
      if (["version", "file", "symbol"].includes(m.nodes.get(link.id).kind)) pool.add(link.id);
    }
    const rings = [new Set([seed])];
    for (let depth = 0; depth < 2; depth++) {
      const next = new Set(rings[depth]);
      for (const id of rings[depth]) for (const link of m.adjacent.get(id) || []) {
        if (pool.has(link.id)) next.add(link.id);
      }
      rings.push(next);
    }
    const active = step === 0 ? new Set(m.nodes.keys()) : step < 4 ? rings[step - 1] : target;
    const previous = step === 0 ? active : step === 1 ? new Set() : step < 4 ? rings[step - 2] : step === 4 ? rings[2] : target;
    const added = new Set([...active].filter(id => !previous.has(id)));
    const explored = step === 0 ? new Set() : rings[Math.min(step - 1, 2)];
    const background = new Set([...explored].filter(id => !target.has(id)));
    const adopted = step >= 4 ? target : new Set();
    const shown = new Set([...active, ...explored]);
    const links = m.data.edges.filter(e => active.has(e.source) && active.has(e.target));
    return {seed, active, added, explored, background, adopted, shown, links, target, rings,
      disconnected: [...target].filter(id => !rings[2].has(id)), context: [...background]};
  }

  function layout(m) {
    const positions = new Map();
    const maximum = Math.max(1, ...m.data.nodes.map(n => n.order || 0));
    const xFor = n => 175 + (Math.max(1, n.order) - 1) / Math.max(1, maximum - 1) * 1200;
    const paths = [...new Set(m.data.nodes.filter(n => n.kind === "version").map(n => n.path))];
    const items = m.data.nodes.filter(n => n.lane === "object");
    const factRows = {general: Array.from({length: 3}, () => []), code: Array.from({length: 8}, () => [])};
    for (const node of [...m.data.nodes].sort((a, b) => a.order - b.order || a.id.localeCompare(b.id))) {
      let x = xFor(node), y = 80;
      if (node.lane === "tool") y = 180 + (node.order % 3) * 26;
      if (node.lane === "patch") y = 302;
      if (node.lane === "version") y = 380 + paths.indexOf(node.path) * 37;
      if (node.lane === "object") { x = 220 + items.indexOf(node) * 1120 / Math.max(1, items.length - 1); y = 585; }
      if (factRows[node.lane]) {
        const rows = factRows[node.lane];
        let row = rows.findIndex(values => values.every(px => Math.abs(px - x) > 33));
        if (row < 0) {
          row = rows.map(v => v.length).indexOf(Math.min(...rows.map(v => v.length)));
          const offsets = [0, 18, -18, 36, -36, 54, -54, 72, -72];
          x = offsets.map(d => Math.max(160, Math.min(1390, x + d))).find(px => rows[row].every(v => Math.abs(v - px) > 15)) || x;
        }
        rows[row].push(x);
        y = (node.lane === "general" ? 673 : 797) + row * 23;
      }
      positions.set(node.id, {x, y});
    }
    return positions;
  }

  const api = {colors, relationNames, model, groupIds, seedFor, groupsForSeed, frame, layout};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.EvidenceGraph = api;
})(typeof window !== "undefined" ? window : globalThis);
