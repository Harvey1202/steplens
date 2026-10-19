// StepLens views: performance matrix, compare strip, slice table, task graph, timeline.
import { state, $, stepsFor, predOf, isPos, inPhase, scoreOf, outcome, metricsOf, matchesFilters,
  catNameOf, activityName, fmtPct, fmtTime, api, setFilters, setModelA, thresholdFor, RULE_LABEL } from './app.js';

// ---------------------------------------------------------------- color
const RAMP = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b'];
const seq = d3.scaleLinear().domain(RAMP.map((_, i) => i / (RAMP.length - 1))).range(RAMP).interpolate(d3.interpolateRgb).clamp(true);
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const catColor = (key) => cssVar(`--cat-${key}`) || '#898781';
const inkOn = (fill) => (d3.hcl(fill).l < 58 ? '#fcfcfb' : '#0b0b0b');
const OUTCOME = { TP: { color: () => cssVar('--good'), label: 'caught', icon: '✓' }, FN: { color: () => cssVar('--critical'), label: 'missed', icon: '✗' }, FP: { color: () => cssVar('--serious'), label: 'false alarm', icon: '!' } };

// ---------------------------------------------------------------- tooltip
let tip;
export function initTooltip() { tip = $('tooltip'); }
function showTip(evt, title, rows = [], sub = '') {
  tip.replaceChildren();
  if (title) { const t = document.createElement('div'); t.className = 't-title'; t.textContent = title; tip.appendChild(t); }
  for (const r of rows) {
    const row = document.createElement('div'); row.className = 't-row';
    const l = document.createElement('span');
    if (r.color) { const k = document.createElement('span'); k.className = 'dot'; k.style.background = r.color; l.appendChild(k); }
    l.appendChild(document.createTextNode(r.label));
    const v = document.createElement('b'); v.textContent = r.value;
    row.append(l, v); tip.appendChild(row);
  }
  if (sub) { const s = document.createElement('div'); s.className = 't-sub'; s.textContent = sub; tip.appendChild(s); }
  tip.hidden = false;
  moveTip(evt);
}
function moveTip(evt) {
  const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = evt.clientX + pad, y = evt.clientY + pad;
  if (x + w > window.innerWidth - 8) x = evt.clientX - w - pad;
  if (y + h > window.innerHeight - 8) y = evt.clientY - h - pad;
  tip.style.left = `${x}px`; tip.style.top = `${y}px`;
}
function hideTip() { tip.hidden = true; }

function el(tag, attrs = {}, text) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) { if (k === 'class') e.className = v; else if (k === 'style') e.style.cssText = v; else e.setAttribute(k, v); }
  if (text !== undefined) e.textContent = text;
  return e;
}

// ---------------------------------------------------------------- summary
export function renderSummary() {
  const box = $('summary');
  box.replaceChildren();
  const describe = (id, tag) => {
    if (!id) return;
    const items = stepsFor(id), m = metricsOf(items), model = state.modelById.get(id);
    const parts = [[`${tag}`, model.label], ['steps', `${m.n} (${m.nError} with errors)`], ['threshold', `${thresholdFor(id).toFixed(2)} (${RULE_LABEL[state.opRule]}${state.norm ? ', normalized' : ''})`],
      ['acc', fmtPct(m.accuracy)], ['precision', fmtPct(m.precision)], ['recall', fmtPct(m.recall)], ['F1', fmtPct(m.f1)],
      ['AUC', Number.isFinite(m.auc) ? m.auc.toFixed(3) : '–'], ['false alarm', fmtPct(m.falseAlarm)]];
    for (const [k, v] of parts) { const s = el('span'); s.append(document.createTextNode(`${k} `), el('b', {}, v)); box.appendChild(s); }
  };
  describe(state.modelA, 'A');
  if (state.modelB) { box.appendChild(el('span', { style: 'flex-basis:100%;height:0' })); describe(state.modelB, 'B'); }
}

// ---------------------------------------------------------------- matrix
export function renderMatrix() {
  const box = $('matrix');
  box.replaceChildren();
  if (!state.models.length) { box.appendChild(el('div', { class: 'empty' }, 'No trained models found in data/predictions.')); return; }
  const keys = state.meta.category_keys, names = state.meta.categories;
  const perModel = state.models.map((m) => ({ m, items: stepsFor(m.model_id) }));

  const table = el('table', { class: 'matrix' });
  const thead = el('thead'), hr = el('tr');
  hr.appendChild(el('th', {}, 'Error category'));
  hr.appendChild(el('th', { class: 'num' }, 'steps'));
  for (const { m, items } of perModel) {
    const th = el('th', { class: `model${m.model_id === state.modelA ? ' a' : ''}${m.model_id === state.modelB ? ' b' : ''}` });
    const b = el('button', { type: 'button', class: 'ghost', title: 'use as model A' }, m.label);
    b.style.color = 'inherit';
    b.addEventListener('click', () => setModelA(m.model_id));
    th.appendChild(b);
    th.appendChild(el('div', { style: 'font-weight:400;color:var(--muted)' }, `${state.phase} · ${items.length} steps · thr ${thresholdFor(m.model_id).toFixed(2)}`));
    hr.appendChild(th);
  }
  thead.appendChild(hr); table.appendChild(thead);
  const tbody = el('tbody');

  const rows = keys.map((k, i) => ({ key: k, name: names[i], color: catColor(k), pick: (d) => d.s[`err_${k}`], missRate: true }));
  rows.push({ key: 'normal', name: 'Normal steps (false-alarm rate)', color: cssVar('--normal-fill'), pick: (d) => !d.s.has_errors, missRate: false });

  for (const r of rows) {
    const tr = el('tr');
    const td = el('td');
    const dot = el('span', { class: 'dot' }); dot.style.background = r.color;
    td.append(dot, document.createTextNode(r.name));
    tr.appendChild(td);
    const total = state.steps.filter((s) => (r.key === 'normal' ? !s.has_errors : s[`err_${r.key}`])).length;
    tr.appendChild(el('td', { class: 'num' }, String(total)));
    for (const { m, items } of perModel) {
      const sub = items.filter(r.pick);
      const n = sub.length;
      const bad = r.missRate ? sub.filter((d) => !d.pos).length : sub.filter((d) => d.pos).length;
      const rate = n ? bad / n : NaN;
      const td2 = el('td', { class: `cell${state.filters.category === r.key && m.model_id === state.modelA ? ' selected' : ''}` });
      const btn = el('button', { type: 'button' });
      if (n) {
        const fill = seq(rate); btn.style.background = fill; btn.style.color = inkOn(fill);
        btn.append(el('span', { class: 'v' }, fmtPct(rate)), el('span', { class: 'n' }, `${bad} / ${n}`));
        btn.addEventListener('click', () => {
          const same = state.filters.category === r.key && state.modelA === m.model_id;
          state.modelA = m.model_id;
          setFilters({ category: same || r.key === 'normal' ? '' : r.key });
        });
        btn.addEventListener('pointermove', (e) => showTip(e, `${r.name} · ${m.label}`, [
          { label: r.missRate ? 'missed' : 'false alarms', value: `${bad} of ${n}` },
          { label: r.missRate ? 'miss rate' : 'false-alarm rate', value: fmtPct(rate) }], r.key === 'missing' ? '' : 'click to focus this category and model'));
        btn.addEventListener('pointerleave', hideTip);
      } else {
        btn.style.background = 'transparent'; btn.style.color = 'var(--muted)';
        btn.append(el('span', { class: 'v' }, '–'));
        btn.disabled = true;
        if (r.key === 'missing') btn.title = 'Skipped steps have no video and no features, so the supervised models never score them.';
      }
      td2.appendChild(btn);
      tr.appendChild(td2);
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  box.appendChild(table);
  const legend = el('div', { class: 'legend' });
  legend.append(document.createTextNode('0% missed'), el('span', { class: 'ramp' }), document.createTextNode(`100% missed · thresholds: ${RULE_LABEL[state.opRule]}` + (state.norm ? ' on min-max normalized step scores' : ' on mean per-second score')));
  box.appendChild(legend);
}

// ---------------------------------------------------------------- compare strip
export function renderCompare() {
  const box = $('compare');
  box.replaceChildren();
  if (!state.modelB || state.modelB === state.modelA) return;
  const A = stepsFor(state.modelA), B = stepsFor(state.modelB);
  const bByKey = new Map(B.map((d) => [d.s.step_key, d]));
  const keys = [...state.meta.category_keys.filter((k) => k !== 'missing'), 'normal'];
  const rows = keys.map((k) => {
    const sub = A.filter((d) => (k === 'normal' ? !d.s.has_errors : d.s[`err_${k}`]) && bByKey.has(d.s.step_key));
    const c = { key: k, name: k === 'normal' ? 'Normal (flagged)' : catNameOf(k), n: sub.length, both: 0, aOnly: 0, bOnly: 0, neither: 0 };
    for (const d of sub) { const pb = bByKey.get(d.s.step_key).pos; if (d.pos && pb) c.both++; else if (d.pos) c.aOnly++; else if (pb) c.bOnly++; else c.neither++; }
    return c;
  }).filter((r) => r.n);
  const la = state.modelById.get(state.modelA).label, lb = state.modelById.get(state.modelB).label;
  const colors = { both: cssVar('--cat-preparation'), aOnly: cssVar('--cat-order'), bOnly: cssVar('--cat-measurement'), neither: cssVar('--normal-fill') };
  const legend = el('div', { class: 'legend' });
  legend.appendChild(el('span', {}, 'Which model catches each error:'));
  for (const [k, label] of [['both', 'both'], ['aOnly', `A only (${la})`], ['bOnly', `B only (${lb})`], ['neither', 'neither']]) {
    const s = el('span'); const key = el('span', { class: 'key' }); key.style.background = colors[k]; s.append(key, document.createTextNode(label)); legend.appendChild(s);
  }
  box.appendChild(legend);
  const W = 640, rowH = 26, labelW = 170, barH = 16;
  const svg = d3.create('svg').attr('viewBox', `0 0 ${W} ${rows.length * rowH + 4}`).attr('role', 'img').attr('aria-label', 'agreement between models per error category');
  const x = d3.scaleLinear().domain([0, 1]).range([labelW, W - 60]);
  rows.forEach((r, i) => {
    const g = svg.append('g').attr('transform', `translate(0,${i * rowH + 2})`);
    g.append('text').attr('x', labelW - 8).attr('y', barH / 2 + 4).attr('text-anchor', 'end').attr('fill', 'var(--ink-2)').attr('font-size', 12).text(r.name);
    let acc = 0;
    for (const part of ['both', 'aOnly', 'bOnly', 'neither']) {
      const frac = r[part] / r.n; if (!frac) continue;
      const x0 = x(acc), x1 = x(acc + frac);
      g.append('rect').attr('x', x0 + 1).attr('y', 0).attr('width', Math.max(0, x1 - x0 - 2)).attr('height', barH).attr('rx', 2).attr('fill', colors[part])
        .on('pointermove', (e) => showTip(e, r.name, [{ label: { both: 'caught by both', aOnly: 'A only', bOnly: 'B only', neither: 'missed by both' }[part], value: `${r[part]} of ${r.n} (${fmtPct(frac)})` }]))
        .on('pointerleave', hideTip);
      acc += frac;
    }
    g.append('text').attr('x', W - 54).attr('y', barH / 2 + 4).attr('fill', 'var(--ink-2)').attr('font-size', 11.5).text(`n=${r.n}`);
  });
  box.appendChild(svg.node());
}

// ---------------------------------------------------------------- slices
export async function renderSlices() {
  const box = $('slices');
  if (!state.modelA) { box.replaceChildren(el('div', { class: 'empty' }, 'No model.')); return; }
  box.style.opacity = '0.5';
  const holdout = state.phase === 'val' ? 'test' : 'val';
  const q = new URLSearchParams({ model_id: state.modelA, phase: state.phase, threshold: thresholdFor(state.modelA), norm: state.norm, min_size: 10, min_errors: state.minSize, sort: state.sliceSort, categories: state.sliceCats, limit: 150, confirm: holdout });
  const f = state.filters;
  if (f.activity_id) q.append('activity_id', f.activity_id);
  if (f.person_id) q.append('person_id', f.person_id);
  if (f.environment_id) q.append('environment_id', f.environment_id);
  if (f.category) q.append('category', f.category);
  let data;
  try { data = await api.json(`/api/slices?${q}`); } catch (e) { box.replaceChildren(el('div', { class: 'empty' }, `slices failed: ${e.message}`)); box.style.opacity = ''; return; }
  box.replaceChildren();
  const o = data.overall || {};
  const ci = (lo, hi) => (Number.isFinite(lo) && Number.isFinite(hi) ? ` [${Math.round(lo * 100)}–${Math.round(hi * 100)}]` : '');
  const confirmNote = data.confirm ? ` Hold-out column = the same slice on the ${data.confirm} steps (overall miss rate there ${fmtPct(data.confirm_overall_miss)}); a slice that stays above it on both phases is a confirmed failure mode.` : '';
  box.appendChild(el('div', { class: 'legend' }, `Overall on this selection: miss rate ${fmtPct(1 - (o.recall ?? NaN))} (${(o.n_error ?? 0) - (o.tp ?? 0)} of ${o.n_error ?? 0} errors), false-alarm rate ${fmtPct(o.false_alarm)}. ${data.n_slices ?? 0} slices shown; brackets are 95% Wilson intervals.${confirmNote}`));
  const table = el('table'), thead = el('thead'), tr = el('tr');
  for (const [t, cls, title] of [['Slice', ''], ['errors', 'num'], ['missed', 'num'], ['miss rate', 'num'], ['Δ vs overall', 'num'], ['z', 'num'], ['hold-out', 'num', 'same slice on the hold-out phase; ✓ = also above the overall miss rate there'], ['AUC', 'num', 'error-vs-normal AUC inside the slice: low means the model cannot separate them there, high with a high miss rate means its scores are just shifted'], ['false alarm', 'num'], ['Δ', 'num']]) { const th = el('th', { class: cls }, t); if (title) th.title = title; tr.appendChild(th); }
  thead.appendChild(tr); table.appendChild(thead);
  const tbody = el('tbody');
  for (const r of data.slices || []) {
    const row = el('tr', { class: 'clickable' });
    row.appendChild(el('td', { style: 'white-space:normal' }, r.slice));
    row.appendChild(el('td', { class: 'num' }, String(r.n_error)));
    row.appendChild(el('td', { class: 'num' }, String(r.n_missed ?? Math.round((r.miss_rate ?? 0) * r.n_error))));
    row.appendChild(el('td', { class: 'num' }, fmtPct(r.miss_rate) + ci(r.miss_lo, r.miss_hi)));
    row.appendChild(el('td', { class: 'num' }, Number.isFinite(r.miss_lift) ? `${r.miss_lift >= 0 ? '+' : '−'}${Math.abs(Math.round(r.miss_lift * 100))} pts` : '–'));
    row.appendChild(el('td', { class: 'num' }, Number.isFinite(r.z_miss) ? r.z_miss.toFixed(1) : '–'));
    const confirmed = Number.isFinite(r.confirm_miss_lift) && r.confirm_miss_lift > 0;
    const hold = el('td', { class: 'num' }, r.confirm_n_error ? `${confirmed ? '✓ ' : ''}${fmtPct(r.confirm_miss_rate)} (n=${r.confirm_n_error})` : '–');
    if (r.confirm_n_error) hold.title = `${fmtPct(r.confirm_miss_rate)}${ci(r.confirm_miss_lo, r.confirm_miss_hi)} missed on the ${data.confirm} steps; ` + (confirmed ? 'above the overall miss rate there too (confirmed)' : 'not above the overall miss rate there');
    row.appendChild(hold);
    row.appendChild(el('td', { class: 'num' }, Number.isFinite(r.auc) ? r.auc.toFixed(2) : '–'));
    row.appendChild(el('td', { class: 'num' }, fmtPct(r.false_alarm)));
    row.appendChild(el('td', { class: 'num' }, Number.isFinite(r.false_alarm_lift) ? `${r.false_alarm_lift >= 0 ? '+' : '−'}${Math.abs(Math.round(r.false_alarm_lift * 100))} pts` : '–'));
    row.addEventListener('click', () => {
      const patch = {};
      if (r.attr_activity_name) { const a = state.meta.activities.find((x) => x.activity_name === r.attr_activity_name); if (a) patch.activity_id = String(a.activity_id); }
      if (r.attr_person_id != null) patch.person_id = String(r.attr_person_id);
      if (r.attr_environment_id != null) patch.environment_id = String(r.attr_environment_id);
      if (r.attr_duration_bucket) patch.duration_bucket = r.attr_duration_bucket;
      if (r.attr_position_bucket) patch.position_bucket = r.attr_position_bucket;
      if (r.category) { const i = state.meta.categories.indexOf(r.category); if (i >= 0) patch.category = state.meta.category_keys[i]; }
      setFilters(patch);
    });
    tbody.appendChild(row);
  }
  if (!(data.slices || []).length) tbody.appendChild(el('tr')).appendChild(el('td', { colspan: 10, class: 'empty' }, 'No slice meets the minimum size.'));
  table.appendChild(tbody);
  box.appendChild(table);
  box.style.opacity = '';
}

// ---------------------------------------------------------------- task graph
function nodeStats(items) {
  const byStep = new Map();
  for (const d of items) {
    const st = byStep.get(d.s.step_id) || { n: 0, missed: 0, normal: 0, fa: 0, cats: {} };
    if (d.s.has_errors) { st.n++; if (!d.pos) st.missed++; for (const t of d.s.error_tags || []) st.cats[t] = (st.cats[t] || 0) + 1; } else { st.normal++; if (d.pos) st.fa++; }
    byStep.set(d.s.step_id, st);
  }
  return byStep;
}

export function renderGraph() {
  const box = $('graph');
  box.replaceChildren();
  $('graphBack').hidden = !state.filters.activity_id;
  if (!state.modelA) return;
  if (!state.filters.activity_id) {
    $('graphTitle').textContent = 'Recipes by miss rate';
    $('graphHint').textContent = 'Share of error steps the model misses in each recipe. Click a recipe to open its task graph.';
    const items = stepsFor(state.modelA);
    const by = d3.rollups(items.filter((d) => d.s.has_errors), (v) => ({ n: v.length, missed: v.filter((d) => !d.pos).length }), (d) => d.activity_id ?? d.s.activity_id)
      .map(([id, v]) => ({ id, name: activityName(id), ...v, rate: v.missed / v.n })).sort((a, b) => b.rate - a.rate);
    const list = el('div', { class: 'bar-list' });
    for (const r of by) {
      const lbl = el('div', { class: 'lbl', title: r.name }, r.name);
      const track = el('div', { class: 'track' });
      const bar = el('div', { class: 'bar' }); bar.style.width = `${Math.max(1, r.rate * 100)}%`;
      track.appendChild(bar);
      const val = el('div', { class: 'val' }, `${fmtPct(r.rate)} (${r.missed}/${r.n})`);
      const open = () => setFilters({ activity_id: String(r.id), step_id: '' });
      lbl.addEventListener('click', open); bar.addEventListener('click', open);
      bar.addEventListener('pointermove', (e) => showTip(e, r.name, [{ label: 'missed', value: `${r.missed} of ${r.n}` }, { label: 'miss rate', value: fmtPct(r.rate) }], 'click to open the task graph'));
      bar.addEventListener('pointerleave', hideTip);
      list.append(lbl, track, val);
    }
    if (!by.length) list.appendChild(el('div', { class: 'empty' }, 'No error steps in this selection.'));
    box.appendChild(list);
    return;
  }
  const g = state.graphs[state.filters.activity_id];
  if (!g) { box.appendChild(el('div', { class: 'empty' }, 'No task graph for this recipe.')); return; }
  $('graphTitle').textContent = `Task graph · ${g.activity_name}`;
  $('graphHint').textContent = 'Steps colored by the share of their error steps the model misses (across the recordings in the current selection). Arrows are the recipe\'s ordering constraints. Click a step to filter the other views to it.';
  const items = stepsFor(state.modelA, { step_id: true });
  const stats = nodeStats(items);

  // layered layout: depth = longest path from START
  const nodes = g.nodes.map((n) => ({ ...n, in: [], out: [] }));
  const byId = new Map(nodes.map((n) => [n.id, n]));
  for (const e of g.edges) { const a = byId.get(e.source), b = byId.get(e.target); if (a && b) { a.out.push(b); b.in.push(a); } }
  const depth = new Map(); const indeg = new Map(nodes.map((n) => [n.id, n.in.length]));
  const queue = nodes.filter((n) => !n.in.length); for (const n of queue) depth.set(n.id, 0);
  while (queue.length) { const n = queue.shift(); for (const m of n.out) { depth.set(m.id, Math.max(depth.get(m.id) ?? 0, depth.get(n.id) + 1)); indeg.set(m.id, indeg.get(m.id) - 1); if (!indeg.get(m.id)) queue.push(m); } }
  for (const n of nodes) if (!depth.has(n.id)) depth.set(n.id, 0);
  const cols = d3.groups(nodes, (n) => depth.get(n.id)).sort((a, b) => a[0] - b[0]);
  const nodeW = 150, nodeH = 36, colGap = 190, rowGap = 48, ml = 16, mt = 12;
  const pos = new Map();
  cols.forEach(([d, list], ci) => {
    if (ci) list.sort((a, b) => d3.mean(a.in, (p) => pos.get(p.id)?.y ?? 0) - d3.mean(b.in, (p) => pos.get(p.id)?.y ?? 0));
    list.forEach((n, i) => pos.set(n.id, { x: ml + d * colGap, y: mt + i * rowGap }));
  });
  const maxRows = d3.max(cols, (c) => c[1].length);
  const W = ml + cols.length * colGap, H = mt + maxRows * rowGap + 10;
  const svg = d3.create('svg').attr('width', W).attr('height', H).attr('viewBox', `0 0 ${W} ${H}`).style('width', `${W}px`).style('max-width', 'none');
  const wrap = el('div', { style: 'overflow:auto' });
  const link = d3.linkHorizontal().x((d) => d.x).y((d) => d.y);
  svg.append('g').selectAll('path').data(g.edges.filter((e) => pos.has(e.source) && pos.has(e.target))).join('path').attr('class', 'edge')
    .attr('d', (e) => link({ source: { x: pos.get(e.source).x + nodeW, y: pos.get(e.source).y + nodeH / 2 }, target: { x: pos.get(e.target).x, y: pos.get(e.target).y + nodeH / 2 } }));
  const gn = svg.append('g').selectAll('g').data(nodes).join('g').attr('class', (n) => `node${String(n.step_id) === String(state.filters.step_id) && n.step_id != null ? ' selected' : ''}`)
    .attr('transform', (n) => `translate(${pos.get(n.id).x},${pos.get(n.id).y})`).style('cursor', (n) => (n.step_id != null ? 'pointer' : 'default'));
  const fillOf = (n) => { const st = stats.get(n.step_id); return st && st.n ? seq(st.missed / st.n) : null; };
  gn.append('rect').attr('width', (n) => (n.kind === 'step' ? nodeW : 60)).attr('height', nodeH).attr('rx', 4)
    .attr('fill', (n) => fillOf(n) || 'var(--surface-1)').attr('stroke', (n) => (fillOf(n) ? 'none' : 'var(--axis)')).attr('stroke-width', 1);
  gn.append('text').attr('x', 8).attr('y', 15).attr('fill', (n) => (fillOf(n) ? inkOn(fillOf(n)) : 'var(--ink)')).text((n) => {
    const l = n.kind === 'step' ? n.label.split(' -')[0].replace(/^[a-z]/, (c) => c.toUpperCase()) : n.label; return l.length > 20 ? `${l.slice(0, 19)}…` : l;
  });
  gn.filter((n) => n.kind === 'step').append('text').attr('x', 8).attr('y', 29).attr('font-size', 10.5).attr('fill', (n) => (fillOf(n) ? inkOn(fillOf(n)) : 'var(--muted)'))
    .text((n) => { const st = stats.get(n.step_id); return st && st.n ? `${st.missed}/${st.n} missed${st.fa ? ` · ${st.fa} FA` : ''}` : (st && st.normal ? `${st.fa}/${st.normal} false alarms` : 'no scored steps'); });
  gn.on('pointermove', (e, n) => {
    if (n.kind !== 'step') return;
    const st = stats.get(n.step_id) || { n: 0, missed: 0, normal: 0, fa: 0, cats: {} };
    const rows = [{ label: 'error steps', value: String(st.n) }, { label: 'missed', value: st.n ? `${st.missed} (${fmtPct(st.missed / st.n)})` : '–' }, { label: 'normal steps', value: String(st.normal) }, { label: 'false alarms', value: st.normal ? `${st.fa} (${fmtPct(st.fa / st.normal)})` : '–' }];
    for (const [c, k] of Object.entries(st.cats)) rows.push({ label: c, value: String(k), color: catColor(state.meta.category_keys[state.meta.categories.indexOf(c)]) });
    showTip(e, n.label, rows, 'click to filter the other views to this step');
  }).on('pointerleave', hideTip).on('click', (e, n) => { if (n.step_id == null) return; setFilters({ step_id: String(state.filters.step_id) === String(n.step_id) ? '' : String(n.step_id) }); });
  wrap.appendChild(svg.node());
  box.appendChild(wrap);
  const legend = el('div', { class: 'legend' });
  legend.append(document.createTextNode('0% missed'), el('span', { class: 'ramp' }), document.createTextNode('100% missed · white = no error steps scored for this step in the selection'));
  box.appendChild(legend);
  if (state.graphTable) {
    const table = el('table'), tr = el('tr');
    for (const [t, c] of [['Step', ''], ['error steps', 'num'], ['missed', 'num'], ['miss rate', 'num'], ['normal steps', 'num'], ['false alarms', 'num']]) tr.appendChild(el('th', { class: c }, t));
    table.appendChild(el('thead')).appendChild(tr);
    const tb = el('tbody');
    for (const n of nodes.filter((x) => x.kind === 'step')) {
      const st = stats.get(n.step_id) || { n: 0, missed: 0, normal: 0, fa: 0 };
      const r = el('tr'); r.appendChild(el('td', { style: 'white-space:normal' }, n.label));
      for (const v of [st.n, st.missed, st.n ? fmtPct(st.missed / st.n) : '–', st.normal, st.fa]) r.appendChild(el('td', { class: 'num' }, String(v)));
      tb.appendChild(r);
    }
    table.appendChild(tb); box.appendChild(table);
  }
}

// ---------------------------------------------------------------- timeline
let timelineToken = 0;
export async function renderTimeline() {
  const box = $('timeline'), sel = $('recording');
  if (!state.modelA) { box.replaceChildren(el('div', { class: 'empty' }, 'No model.')); return; }
  // candidate recordings under the current filters (ignoring step/category for membership but counting them for ranking)
  const items = stepsFor(state.modelA, { category: true, step_id: true });
  const groups = d3.groups(items, (d) => d.s.recording_id).map(([rid, v]) => {
    const match = v.filter((d) => matchesFilters(d.s));
    const errs = match.filter((d) => d.s.has_errors);
    const first = v[0].s;
    return { rid, name: first.activity_name, person: first.person_id, env: first.environment_id, nErr: errs.length, missed: errs.filter((d) => !d.pos).length, fa: match.filter((d) => !d.s.has_errors && d.pos).length, keep: match.length > 0 };
  }).filter((r) => r.keep).sort((a, b) => b.missed - a.missed || b.nErr - a.nErr || a.rid.localeCompare(b.rid));
  sel.replaceChildren();
  for (const r of groups) {
    const o = document.createElement('option'); o.value = r.rid;
    o.textContent = `${r.rid} · ${r.name} · person ${r.person} · kitchen ${r.env} · missed ${r.missed}/${r.nErr}${r.fa ? ` · ${r.fa} false alarm${r.fa > 1 ? 's' : ''}` : ''}${state.meta.videos.includes(r.rid) ? ' · video' : ''}`;
    sel.appendChild(o);
  }
  if (!groups.length) { box.replaceChildren(el('div', { class: 'empty' }, 'No recordings match the current filters.')); $('stepDetail').replaceChildren(); $('videoBox').replaceChildren(); return; }
  if (!groups.some((r) => r.rid === state.recording)) state.recording = groups[0].rid;
  sel.value = state.recording;
  const rid = state.recording, token = ++timelineToken;
  const [segs, secA, secB] = await Promise.all([api.fetchSegments(rid), api.fetchSeconds(state.modelA, rid), state.modelB ? api.fetchSeconds(state.modelB, rid) : null]);
  if (token !== timelineToken) return; // superseded
  drawTimeline(rid, segs, secA, secB);
  renderVideo(rid);
  renderDetail();
}

function secondsPoints(segs, sec) {
  // Map each occurrence's rows to timestamps. Rows for a step cover its occurrences in annotation order.
  const L = sec.segment_len || 1, pts = [], used = new Map();
  for (const sg of segs.filter((s) => !s.is_missing).sort((a, b) => a.ann_index - b.ann_index)) {
    const arr = sec.seconds[sg.step_key]; if (!arr) continue;
    const a = Math.floor(sg.start_time / L), b = Math.max(Math.ceil(sg.end_time / L), a + 1);
    const off = used.get(sg.step_key) || 0;
    const rows = arr.slice(off, off + (b - a)); used.set(sg.step_key, off + (b - a));
    pts.push({ step_key: sg.step_key, points: rows.map((v, i) => ({ t: (a + i) * L + L / 2, v })) });
  }
  return pts;
}

function drawTimeline(rid, segs, secA, secB) {
  const box = $('timeline');
  box.replaceChildren();
  const steps = state.steps.filter((s) => s.recording_id === rid);
  const performed = segs.filter((s) => !s.is_missing);
  const W = Math.max(600, box.clientWidth || 1100), ml = 44, mr = 16;
  const gtY = 14, gtH = 22, outY = gtY + gtH + 4, outH = 8, scY = outY + outH + 14, scH = 130, axisY = scY + scH + 6, H = axisY + 26;
  const dur = Math.max(d3.max(performed, (s) => s.end_time) || 0, d3.max(Object.values(secA.seconds), (a) => a.length) * (secA.segment_len || 1) || 0);
  const x = d3.scaleLinear().domain([0, dur]).range([ml, W - mr]);
  const y = d3.scaleLinear().domain([0, 1]).range([scY + scH, scY]);
  const svg = d3.create('svg').attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('height', H);

  // ground truth track
  const tagKey = (s) => { const t = (s.error_tags || [])[0]; return t ? state.meta.category_keys[state.meta.categories.indexOf(t)] : null; };
  const stepOf = (key) => state.stepByKey.get(key);
  const gt = svg.append('g');
  svg.append('text').attr('class', 'axis').attr('x', ml - 6).attr('y', gtY + gtH / 2 + 4).attr('text-anchor', 'end').text('steps');
  svg.append('text').attr('class', 'axis').attr('x', ml - 6).attr('y', outY + outH).attr('text-anchor', 'end').text('A');
  svg.append('text').attr('class', 'axis').attr('x', ml - 6).attr('y', scY + 4).attr('text-anchor', 'end').text('score');
  for (const sg of performed) {
    const s = stepOf(sg.step_key), p = predOf(state.modelA, sg.step_key);
    const k = tagKey(s), fill = k ? catColor(k) : cssVar('--normal-fill');
    const x0 = x(sg.start_time) + 1, w = Math.max(2, x(sg.end_time) - x(sg.start_time) - 2);
    const inP = p && inPhase(p);
    const r = gt.append('rect').attr('x', x0).attr('y', gtY).attr('width', w).attr('height', gtH).attr('rx', 3).attr('fill', fill).attr('opacity', inP ? 1 : 0.45).style('cursor', 'pointer')
      .attr('stroke', state.selectedStep === sg.step_key ? 'var(--ink)' : 'none').attr('stroke-width', 2);
    const short = s.step_description.split(' -')[0];
    if (w > short.length * 6.8 + 10) gt.append('text').attr('x', x0 + 6).attr('y', gtY + gtH / 2 + 4).attr('font-size', 11).attr('fill', k ? inkOn(fill) : 'var(--ink)').attr('pointer-events', 'none').text(short);
    r.on('pointermove', (e) => showTip(e, s.step_description, [
      { label: 'time', value: `${fmtTime(sg.start_time)} – ${fmtTime(sg.end_time)}` },
      ...(s.error_tags || []).map((t) => ({ label: t, value: '', color: catColor(state.meta.category_keys[state.meta.categories.indexOf(t)]) })),
      ...(p ? [{ label: `score A${state.norm ? ' (norm)' : ''}`, value: scoreOf(p).toFixed(3) }, { label: 'phase for model A', value: p.phase }] : []),
    ], s.has_errors ? '' : 'normal step')).on('pointerleave', hideTip).on('click', () => selectStep(sg.step_key, sg.start_time));
    if (inP) {
      const o = outcome(s, isPos(p, state.modelA));
      if (o !== 'TN') gt.append('rect').attr('x', x0).attr('y', outY).attr('width', w).attr('height', outH).attr('rx', 2).attr('fill', OUTCOME[o].color()).style('cursor', 'pointer')
        .on('pointermove', (e) => showTip(e, `${OUTCOME[o].icon} ${OUTCOME[o].label}`, [{ label: 'step', value: short }, { label: 'score', value: scoreOf(p).toFixed(3) }])).on('pointerleave', hideTip).on('click', () => selectStep(sg.step_key, sg.start_time));
    }
  }
  // score area
  for (const v of [0, 0.5, 1]) {
    svg.append('line').attr('class', 'gridline').attr('x1', ml).attr('x2', W - mr).attr('y1', y(v)).attr('y2', y(v));
    svg.append('text').attr('class', 'axis').attr('x', ml - 6).attr('y', y(v) + 4).attr('text-anchor', 'end').text(v.toFixed(1));
  }
  const line = d3.line().x((d) => x(d.t)).y((d) => y(d.v)).curve(d3.curveMonotoneX);
  const series = [{ id: state.modelA, sec: secA, color: 'var(--model-a)', tag: 'A' }];
  if (secB) series.push({ id: state.modelB, sec: secB, color: 'var(--model-b)', tag: 'B' });
  for (const sr of series) {
    const thr = thresholdFor(sr.id);
    svg.append('line').attr('x1', ml).attr('x2', W - mr).attr('y1', y(thr)).attr('y2', y(thr)).attr('stroke', sr.color).attr('stroke-width', 1).attr('opacity', 0.7);
    svg.append('text').attr('class', 'axis').attr('x', W - mr).attr('y', y(thr) - 4).attr('text-anchor', 'end').text(`${sr.tag} threshold ${thr.toFixed(2)}`);
  }
  const lookup = [];
  for (const sr of series) {
    const pts = secondsPoints(segs, sr.sec);
    sr.pts = pts;
    for (const seg of pts) svg.append('path').attr('d', line(seg.points)).attr('fill', 'none').attr('stroke', sr.color).attr('stroke-width', 2).attr('stroke-linejoin', 'round').attr('stroke-linecap', 'round').attr('opacity', sr.tag === 'A' ? 0.95 : 0.9);
    // step-level mean marks
    for (const sg of performed) {
      const p = predOf(sr.id, sg.step_key); if (!p) continue;
      svg.append('line').attr('x1', x(sg.start_time) + 1).attr('x2', x(sg.end_time) - 1).attr('y1', y(scoreOf(p))).attr('y2', y(scoreOf(p))).attr('stroke', sr.color).attr('stroke-width', inPhase(p) ? 3 : 1).attr('opacity', inPhase(p) ? 0.9 : 0.5);
    }
  }
  // time axis
  const ticks = x.ticks(Math.min(12, Math.floor(W / 90)));
  for (const t of ticks) {
    svg.append('line').attr('class', 'hair').attr('x1', x(t)).attr('x2', x(t)).attr('y1', axisY).attr('y2', axisY + 4);
    svg.append('text').attr('class', 'axis').attr('x', x(t)).attr('y', axisY + 16).attr('text-anchor', 'middle').text(fmtTime(t));
  }
  svg.append('line').attr('class', 'hair').attr('x1', ml).attr('x2', W - mr).attr('y1', axisY).attr('y2', axisY);
  // crosshair
  const cross = svg.append('line').attr('class', 'hair').attr('y1', gtY).attr('y2', axisY).style('display', 'none');
  svg.append('rect').attr('x', ml).attr('y', scY).attr('width', W - ml - mr).attr('height', scH).attr('fill', 'transparent')
    .on('pointermove', (e) => {
      const [mx] = d3.pointer(e); const t = x.invert(mx);
      cross.attr('x1', x(t)).attr('x2', x(t)).style('display', null);
      const sg = performed.find((s) => t >= s.start_time && t <= s.end_time);
      const rows = [];
      for (const sr of series) {
        let v = null;
        for (const seg of sr.pts) { const L = sr.sec.segment_len || 1; const hit = seg.points.find((pt) => Math.abs(pt.t - t) <= L / 2); if (hit) { v = hit.v; break; } }
        rows.push({ label: `${sr.tag} per-second`, value: v == null ? '–' : v.toFixed(3) });
      }
      if (sg) { const s = stepOf(sg.step_key); rows.unshift({ label: 'ground truth', value: s.has_errors ? (s.error_tags || []).join(', ') : 'normal' }); }
      showTip(e, `${fmtTime(t)}${sg ? ` · ${stepOf(sg.step_key).step_description.split(' -')[0]}` : ''}`, rows);
    }).on('pointerleave', () => { cross.style('display', 'none'); hideTip(); });
  box.appendChild(svg.node());
  // legend
  const legend = el('div', { class: 'legend' });
  for (const [i, k] of state.meta.category_keys.entries()) { const s = el('span'); const key = el('span', { class: 'key' }); key.style.background = catColor(k); s.append(key, document.createTextNode(state.meta.categories[i])); legend.appendChild(s); }
  { const s = el('span'); const key = el('span', { class: 'key' }); key.style.background = cssVar('--normal-fill'); s.append(key, document.createTextNode('normal')); legend.appendChild(s); }
  for (const o of ['TP', 'FN', 'FP']) { const s = el('span'); const key = el('span', { class: 'key' }); key.style.background = OUTCOME[o].color(); s.append(key, document.createTextNode(`${OUTCOME[o].icon} ${OUTCOME[o].label} (model A, ${state.phase} steps)`)); legend.appendChild(s); }
  for (const sr of series) { const s = el('span'); const ln = el('span', { class: 'line' }); ln.style.background = sr.color; s.append(ln, document.createTextNode(`${sr.tag}: ${state.modelById.get(sr.id).label} (line = per-second score, bar = step mean${state.norm ? ', normalized' : ''})`)); legend.appendChild(s); }
  box.appendChild(legend);
  const skipped = steps.filter((s) => s.is_missing);
  if (skipped.length) {
    const d = el('div', { class: 'legend' }); d.appendChild(el('span', {}, `Skipped steps (Missing Step errors, never scored): `));
    for (const s of skipped) { const sp = el('span'); const key = el('span', { class: 'dot' }); key.style.background = catColor('missing'); sp.append(key, document.createTextNode(s.step_description.split(' -')[0])); d.appendChild(sp); }
    box.appendChild(d);
  }
}

function selectStep(key, start) {
  state.selectedStep = key;
  renderDetail();
  const v = $('videoBox').querySelector('video');
  if (v && Number.isFinite(start)) { v.currentTime = Math.max(0, start); v.play().catch(() => {}); }
  renderTimeline();
}

function renderDetail() {
  const box = $('stepDetail');
  box.replaceChildren();
  const s = state.selectedStep ? state.stepByKey.get(state.selectedStep) : null;
  if (!s) { box.appendChild(el('div', { class: 'empty' }, 'Click a step in the timeline to see its errors and scores.')); return; }
  box.appendChild(el('h3', {}, s.step_description));
  box.appendChild(el('div', {}, `${s.activity_name} · recording ${s.recording_id} · person ${s.person_id} · kitchen ${s.environment_id} · ${fmtTime(s.start_time)}–${fmtTime(s.end_time)} (${Math.round(s.duration)} s, position ${s.position_idx + 1} of ${s.n_performed})`));
  if (s.has_errors) {
    const ul = el('ul');
    for (const e of s.error_list || s.error_tags || []) {
      const li = el('li'); const tag = String(e).split(': ')[0]; const k = state.meta.category_keys[state.meta.categories.indexOf(tag)];
      const dot = el('span', { class: 'dot' }); dot.style.background = catColor(k); li.append(dot, document.createTextNode(e)); ul.appendChild(li);
    }
    box.appendChild(ul);
  } else box.appendChild(el('div', {}, 'Ground truth: normal step.'));
  for (const [tag, id] of [['A', state.modelA], ['B', state.modelB]]) {
    if (!id) continue;
    const p = predOf(id, s.step_key); if (!p) { box.appendChild(el('div', {}, `${tag}: not scored`)); continue; }
    const o = outcome(s, isPos(p, id)); const oi = OUTCOME[o];
    box.appendChild(el('div', {}, `${tag} (${state.modelById.get(id).label}): score ${scoreOf(p).toFixed(3)} · ${p.phase} step · ${oi ? `${oi.icon} ${oi.label}` : 'correctly normal'}`));
  }
}

function renderVideo(rid) {
  const box = $('videoBox');
  const cur = box.querySelector('video');
  if (cur && cur.dataset.rid === rid) return;
  box.replaceChildren();
  if (state.meta.videos.includes(rid)) {
    const v = document.createElement('video'); v.controls = true; v.preload = 'metadata'; v.src = `/videos/${rid}_360p.mp4`; v.dataset.rid = rid;
    box.appendChild(v);
  } else box.appendChild(el('div', { class: 'nov' }, `No local 360p video for ${rid}. Run scripts/download_videos.sh ${rid} to fetch it (GoPro, ~1 MB per 10 s).`));
}
