// StepLens front end: shared state, data loading, controls and view linking.
import * as V from './views.js';

export const state = {
  meta: null,
  steps: [],
  stepByKey: new Map(),
  graphs: {},
  models: [],
  modelById: new Map(),
  preds: new Map(),              // model_id -> Map(step_key -> {score, score_norm, phase, label})
  modelA: null,
  modelB: '',
  phase: 'test',
  norm: false,
  threshold: 0.5,
  opRule: 'balanced',            // balanced | f1 | paper | custom
  thresholds: new Map(),         // model_id -> threshold under the current rule
  filters: { activity_id: '', person_id: '', environment_id: '', category: '', duration_bucket: '', position_bucket: '', step_id: '' },
  recording: '',
  selectedStep: null,
  minSize: 10,
  sliceSort: 'z_miss',
  sliceCats: true,
  graphTable: false,
  multiLabel: 'all',             // all | single: how the matrix counts steps with several error categories
  secondsCache: new Map(),       // `${model}|${rid}` -> {segment_len, seconds}
  segmentsCache: new Map(),      // rid -> [segments]
};

export const $ = (id) => document.getElementById(id);
const j = (url) => fetch(url).then((r) => { if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.json(); });

// ---------------------------------------------------------------- selectors
export function predOf(model, key) { const m = state.preds.get(model); return m ? m.get(key) : undefined; }
export function scoreOf(p) { return state.norm ? p.score_norm : p.score; }
export function thresholdFor(modelId) { return state.opRule === 'custom' ? state.threshold : (state.thresholds.get(modelId) ?? state.threshold); }
export function isPos(p, modelId) { return scoreOf(p) > thresholdFor(modelId); }
export const RULE_LABEL = { balanced: 'balanced on val', f1: 'best F1 on val', paper: 'paper protocol', custom: 'custom' };
export function inPhase(p) { return state.phase === 'all' || p.phase === state.phase; }
export function outcome(s, pos) { return s.has_errors ? (pos ? 'TP' : 'FN') : (pos ? 'FP' : 'TN'); }

export function matchesFilters(s, ignore = {}) {
  const f = state.filters;
  if (f.activity_id && !ignore.activity_id && String(s.activity_id) !== String(f.activity_id)) return false;
  if (f.person_id && !ignore.person_id && String(s.person_id) !== String(f.person_id)) return false;
  if (f.environment_id && !ignore.environment_id && String(s.environment_id) !== String(f.environment_id)) return false;
  if (f.duration_bucket && !ignore.duration_bucket && s.duration_bucket !== f.duration_bucket) return false;
  if (f.position_bucket && !ignore.position_bucket && s.position_bucket !== f.position_bucket) return false;
  if (f.step_id && !ignore.step_id && String(s.step_id) !== String(f.step_id)) return false;
  if (f.category && !ignore.category && s.has_errors && !s['err_' + f.category]) return false;
  return true;
}

/** Steps scored by `model` in the current phase that pass the filters. */
export function stepsFor(model, ignore = {}) {
  const out = [];
  const preds = state.preds.get(model);
  if (!preds) return out;
  for (const s of state.steps) {
    const p = preds.get(s.step_key);
    if (!p || !inPhase(p)) continue;
    if (!matchesFilters(s, ignore)) continue;
    out.push({ s, p, pos: isPos(p, model) });
  }
  return out;
}

export function auc(items) {
  const pos = items.filter((d) => d.y).length, neg = items.length - pos;
  if (!pos || !neg) return NaN;
  const sorted = items.slice().sort((a, b) => a.s - b.s);
  let i = 0, sum = 0;
  while (i < sorted.length) {
    let k = i;
    while (k + 1 < sorted.length && sorted[k + 1].s === sorted[i].s) k++;
    const rank = (i + k) / 2 + 1;
    for (let q = i; q <= k; q++) if (sorted[q].y) sum += rank;
    i = k + 1;
  }
  return (sum - (pos * (pos + 1)) / 2) / (pos * neg);
}

export function metricsOf(items) {
  let tp = 0, fp = 0, fn = 0, tn = 0;
  for (const d of items) {
    if (d.s.has_errors) { if (d.pos) tp++; else fn++; } else if (d.pos) fp++; else tn++;
  }
  const n = items.length;
  const precision = tp + fp ? tp / (tp + fp) : 0, recall = tp + fn ? tp / (tp + fn) : 0;
  return {
    n, nError: tp + fn, tp, fp, fn, tn,
    accuracy: n ? (tp + tn) / n : NaN, precision, recall,
    f1: precision + recall ? (2 * precision * recall) / (precision + recall) : 0,
    falseAlarm: fp + tn ? fp / (fp + tn) : NaN,
    auc: auc(items.map((d) => ({ y: d.s.has_errors, s: scoreOf(d.p) }))),
  };
}

/** Threshold that maximizes F1 ('f1') or TPR − FPR ('balanced') for a model on one phase (all steps, no filters). */
export function bestThreshold(modelId, phase = 'val', criterion = 'f1') {
  const preds = state.preds.get(modelId);
  if (!preds) return null;
  const items = [];
  for (const s of state.steps) { const p = preds.get(s.step_key); if (p && (phase === 'all' || p.phase === phase)) items.push({ y: s.has_errors, s: scoreOf(p) }); }
  if (!items.length) return null;
  const nPos = items.filter((d) => d.y).length, nNeg = items.length - nPos;
  let best = { value: -Infinity, t: 0.5 };
  for (let i = 1; i < 100; i++) {
    const t = i / 100;
    let tp = 0, fp = 0;
    for (const d of items) { if (d.s > t) { if (d.y) tp++; else fp++; } }
    const fn = nPos - tp;
    const value = criterion === 'f1' ? (tp ? (2 * tp) / (2 * tp + fp + fn) : 0) : (nPos ? tp / nPos : 0) - (nNeg ? fp / nNeg : 0);
    if (value > best.value + 1e-9) best = { value, t };
  }
  return best;
}

/** Recompute every model's threshold under the current operating-point rule. */
export function recomputeThresholds() {
  if (state.opRule === 'paper') state.norm = true;
  for (const m of state.models) {
    let t = state.threshold;
    if (state.opRule === 'paper') t = ['step', 'step_random'].includes(m.split) ? 0.6 : 0.4;
    else if (state.opRule === 'f1' || state.opRule === 'balanced') { const b = bestThreshold(m.model_id, 'val', state.opRule); if (b) t = b.t; }
    state.thresholds.set(m.model_id, t);
  }
  if (state.opRule !== 'custom' && state.modelA) state.threshold = thresholdFor(state.modelA);
}

export const catKeyOf = (name) => { const i = state.meta.categories.indexOf(name); return i >= 0 ? state.meta.category_keys[i] : null; };
export const catNameOf = (key) => { const i = state.meta.category_keys.indexOf(key); return i >= 0 ? state.meta.categories[i] : key; };
export const activityName = (id) => (state.meta.activities.find((a) => String(a.activity_id) === String(id)) || {}).activity_name || `recipe ${id}`;
export const fmtPct = (x) => (Number.isFinite(x) ? `${Math.round(x * 100)}%` : '–');
export const fmtTime = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`;

// ---------------------------------------------------------------- data
async function fetchSeconds(model, rid) {
  const key = `${model}|${rid}`;
  if (!state.secondsCache.has(key)) state.secondsCache.set(key, await j(`/api/seconds/${model}?recording_id=${encodeURIComponent(rid)}`));
  return state.secondsCache.get(key);
}
async function fetchSegments(rid) {
  if (!state.segmentsCache.has(rid)) state.segmentsCache.set(rid, await j(`/api/segments?recording_id=${encodeURIComponent(rid)}`));
  return state.segmentsCache.get(rid);
}
export const api = { fetchSeconds, fetchSegments, json: j };

async function load() {
  const [meta, steps, graphs, models] = await Promise.all([j('/api/meta'), j('/api/steps'), j('/api/task_graphs'), j('/api/models')]);
  state.meta = meta;
  state.steps = steps;
  for (const s of steps) state.stepByKey.set(s.step_key, s);
  state.graphs = graphs;
  const order = { step: 0, step_random: 1, recordings: 2, person: 3, environment: 4 };
  state.models = models.sort((a, b) => (order[a.split] - order[b.split]) || a.backbone.localeCompare(b.backbone) || a.variant.localeCompare(b.variant));
  for (const m of state.models) state.modelById.set(m.model_id, m);
  await Promise.all(state.models.map(async (m) => {
    const rows = await j(`/api/predictions/${m.model_id}`);
    state.preds.set(m.model_id, new Map(rows.map((r) => [r.step_key, r])));
  }));
  // default model A: the trained (not derived) model with the best validation AUC
  let best = null, bestAuc = -1;
  for (const m of state.models) {
    if (m.model_id.includes('_cal-')) continue;
    const items = [];
    for (const s of state.steps) { const p = state.preds.get(m.model_id).get(s.step_key); if (p && p.phase === 'val') items.push({ y: s.has_errors, s: p.score }); }
    const a = auc(items);
    if (Number.isFinite(a) && a > bestAuc) { bestAuc = a; best = m.model_id; }
  }
  state.modelA = best || (state.models[0] || {}).model_id || null;
}

// ---------------------------------------------------------------- controls
function fill(select, options, keepFirst = true) {
  const first = keepFirst ? select.querySelector('option') : null;
  select.replaceChildren();
  if (first) select.appendChild(first);
  for (const [value, label] of options) {
    const o = document.createElement('option');
    o.value = value; o.textContent = label;
    select.appendChild(o);
  }
}

function initControls() {
  const m = state.meta;
  fill($('modelA'), state.models.map((x) => [x.model_id, x.label]), false);
  fill($('modelB'), state.models.map((x) => [x.model_id, x.label]));
  fill($('fActivity'), m.activities.map((a) => [a.activity_id, a.activity_name]));
  fill($('fPerson'), m.persons.map((p) => [p, `person ${p}`]));
  fill($('fEnvironment'), m.environments.map((e) => [e, `kitchen ${e}`]));
  fill($('fCategory'), m.category_keys.map((k, i) => [k, m.categories[i]]));
  fill($('fDuration'), ['<15s', '15-30s', '30-60s', '60-120s', '>120s'].map((d) => [d, d]));
  fill($('fPosition'), [['Q1', 'Q1 (first quarter)'], ['Q2', 'Q2'], ['Q3', 'Q3'], ['Q4', 'Q4 (last quarter)']]);

  $('modelA').addEventListener('change', (e) => { state.modelA = e.target.value; if (state.opRule !== 'custom') state.threshold = thresholdFor(state.modelA); update(); });
  $('modelB').addEventListener('change', (e) => { state.modelB = e.target.value; update(); });
  $('phase').addEventListener('change', (e) => { state.phase = e.target.value; update(); });
  $('norm').addEventListener('change', (e) => { state.norm = e.target.checked; if (state.opRule === 'paper' && !state.norm) state.opRule = 'custom'; recomputeThresholds(); update(); });
  $('opRule').addEventListener('change', (e) => { state.opRule = e.target.value; recomputeThresholds(); update(); });
  $('threshold').addEventListener('input', (e) => { state.threshold = +e.target.value; state.opRule = 'custom'; $('thresholdOut').value = state.threshold.toFixed(2); update(); });
  const bind = (id, name) => $(id).addEventListener('change', (e) => { state.filters[name] = e.target.value; update(); });
  bind('fActivity', 'activity_id'); bind('fPerson', 'person_id'); bind('fEnvironment', 'environment_id');
  bind('fCategory', 'category'); bind('fDuration', 'duration_bucket'); bind('fPosition', 'position_bucket');
  $('reset').addEventListener('click', () => { for (const k in state.filters) state.filters[k] = ''; state.selectedStep = null; update(); });
  $('theme').addEventListener('change', (e) => { applyTheme(e.target.value); update(); });
  $('multiLabel').addEventListener('change', (e) => { state.multiLabel = e.target.value; V.renderMatrix(); });
  $('minSize').addEventListener('change', (e) => { state.minSize = +e.target.value || 10; scheduleSlices(); });
  $('sliceSort').addEventListener('change', (e) => { state.sliceSort = e.target.value; scheduleSlices(); });
  $('sliceCats').addEventListener('change', (e) => { state.sliceCats = e.target.checked; scheduleSlices(); });
  $('graphBack').addEventListener('click', () => { state.filters.activity_id = ''; state.filters.step_id = ''; update(); });
  $('graphTable').addEventListener('change', (e) => { state.graphTable = e.target.checked; V.renderGraph(); });
  $('recording').addEventListener('change', (e) => { state.recording = e.target.value; state.selectedStep = null; V.renderTimeline(); });
}

function syncControls() {
  $('modelA').value = state.modelA || '';
  $('modelB').value = state.modelB || '';
  $('phase').value = state.phase;
  $('norm').checked = state.norm;
  $('opRule').value = state.opRule;
  const tA = state.modelA ? thresholdFor(state.modelA) : state.threshold;
  $('threshold').value = tA; $('thresholdOut').value = tA.toFixed(2);
  $('fActivity').value = state.filters.activity_id; $('fPerson').value = state.filters.person_id;
  $('fEnvironment').value = state.filters.environment_id; $('fCategory').value = state.filters.category;
  $('fDuration').value = state.filters.duration_bucket; $('fPosition').value = state.filters.position_bucket;
}

function renderChips() {
  const box = $('chips');
  box.replaceChildren();
  const f = state.filters;
  const labels = {
    activity_id: (v) => `recipe: ${activityName(v)}`, person_id: (v) => `person ${v}`, environment_id: (v) => `kitchen ${v}`,
    category: (v) => `category: ${catNameOf(v)}`, duration_bucket: (v) => `duration ${v}`, position_bucket: (v) => `position ${v}`,
    step_id: (v) => { const s = state.steps.find((x) => String(x.step_id) === String(v)); return `step: ${s ? s.step_description.split(' -')[0] : v}`; },
  };
  for (const k of Object.keys(f)) {
    if (!f[k]) continue;
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.appendChild(document.createTextNode(labels[k](f[k])));
    const x = document.createElement('button');
    x.type = 'button'; x.textContent = '×'; x.setAttribute('aria-label', `remove filter ${k}`);
    x.addEventListener('click', () => { state.filters[k] = ''; update(); });
    chip.appendChild(x);
    box.appendChild(chip);
  }
}

let sliceTimer = null;
export function scheduleSlices() { clearTimeout(sliceTimer); sliceTimer = setTimeout(() => V.renderSlices(), 250); }

function applyTheme(v) {
  if (v) document.documentElement.dataset.theme = v; else delete document.documentElement.dataset.theme;
  $('theme').value = v || '';
  try { localStorage.setItem('steplens-theme', v || ''); } catch (e) { /* storage unavailable */ }
}

// URL parameters make a view shareable: ?modelA=..&modelB=..&phase=..&norm=1&threshold=0.3&activity_id=..&recording=..
const URL_KEYS = ['modelA', 'modelB', 'phase', 'norm', 'threshold', 'recording'];
function readUrlState() {
  const q = new URLSearchParams(location.search);
  if (q.get('theme') !== null) applyTheme(q.get('theme'));
  if (q.get('modelA') && state.modelById.has(q.get('modelA'))) state.modelA = q.get('modelA');
  if (q.get('modelB') && state.modelById.has(q.get('modelB'))) state.modelB = q.get('modelB');
  if (q.get('phase')) state.phase = q.get('phase');
  if (q.get('norm') !== null) state.norm = q.get('norm') === '1';
  if (q.get('rule') && q.get('rule') in RULE_LABEL) state.opRule = q.get('rule');
  if (q.get('threshold') !== null && Number.isFinite(+q.get('threshold'))) { state.threshold = +q.get('threshold'); if (!q.get('rule')) state.opRule = 'custom'; }
  if (q.get('recording')) state.recording = q.get('recording');
  for (const k of Object.keys(state.filters)) if (q.get(k)) state.filters[k] = q.get(k);
  return q.get('threshold') !== null;
}
function writeUrlState() {
  const q = new URLSearchParams();
  q.set('modelA', state.modelA || ''); if (state.modelB) q.set('modelB', state.modelB);
  q.set('phase', state.phase); if (state.norm) q.set('norm', '1'); q.set('rule', state.opRule);
  if (state.opRule === 'custom') q.set('threshold', state.threshold.toFixed(2));
  for (const [k, v] of Object.entries(state.filters)) if (v) q.set(k, v);
  if (state.recording) q.set('recording', state.recording);
  const theme = document.documentElement.dataset.theme; if (theme) q.set('theme', theme);
  history.replaceState(null, '', `?${q}`);
}

export function setFilters(patch) { Object.assign(state.filters, patch); update(); }
export function setModelA(id) { state.modelA = id; update(); }

export function update() {
  syncControls();
  writeUrlState();
  renderChips();
  V.renderSummary();
  V.renderMatrix();
  V.renderCompare();
  V.renderGraph();
  V.renderTimeline();
  scheduleSlices();
}

(async function main() {
  try {
    try { applyTheme(new URLSearchParams(location.search).get('theme') ?? localStorage.getItem('steplens-theme') ?? ''); } catch (e) { /* ignore */ }
    await load();
    V.initTooltip();
    initControls();
    readUrlState();
    recomputeThresholds();
    $('status').textContent = `${state.meta.n_recordings} recordings · ${state.meta.n_steps} steps · ${state.models.length} model${state.models.length === 1 ? '' : 's'}`;
    if (!state.models.length) $('status').textContent += ' — no predictions yet; train a model with python -m steplens.train';
    update();
  } catch (err) {
    $('status').textContent = `failed to load: ${err.message}`;
    console.error(err);
  }
})();
