// ChestForge UI — editor, 3D viewer, ND200 report, optimizer Pareto. Simulated — not NOCSAE certified.
import * as THREE from 'three';
import { OrbitControls } from '/vendor/OrbitControls.js';

const $ = (s, r = document) => r.querySelector(s);
const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') e.className = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null) e.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k !== null && k !== undefined) e.append(k.nodeType ? k : document.createTextNode(k));
  return e;
};
const SVGNS = 'http://www.w3.org/2000/svg';
const svg = (tag, attrs = {}, parent) => {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) e.setAttribute(k, v);
  if (parent) parent.append(e);
  return e;
};
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const tick = (v) => (Math.abs(v) >= 100 ? Math.round(v).toString() : Number(v.toPrecision(3)).toString());
const fmt = (v, d = 1) => (v === null || v === undefined || !isFinite(v) ? '–' : Number(v).toFixed(d));

async function api(path, body) {
  const r = await fetch(path, body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {});
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { const j = await r.json(); msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail); } catch { /* keep status */ }
    throw new Error(msg);
  }
  return r;
}

const state = { meta: null, design: null, materials: {}, report: null, run: null, runTimer: null };

// ------------------------------------------------------------------ editor
function runRequest() {
  return {
    design: state.design,
    projectile: $('#projectile').value,
    torso_length_in: Number($('#torso').value),
    mc_samples: Number($('#mc').value),
    seed: 0,
  };
}

function materialSelect(value, onchange) {
  const s = el('select', { 'aria-label': 'material', onchange: (e) => onchange(e.target.value) });
  for (const kind of ['plate', 'foam']) {
    const g = el('optgroup', { label: kind === 'plate' ? 'Plates' : 'Foams' });
    for (const m of state.meta.materials.filter((m) => m.kind === kind)) {
      g.append(el('option', { value: m.id, selected: m.id === value ? '' : null }, `${m.id}${m.estimated ? ' (est.)' : ''}`));
    }
    s.append(g);
  }
  return s;
}

function num(value, attrs, onchange) {
  return el('input', { type: 'number', value, step: attrs.step ?? 0.25, min: attrs.min, max: attrs.max,
    'aria-label': attrs.label, title: attrs.label, onchange: (e) => onchange(Number(e.target.value)) });
}

let dragFrom = null;
function renderEditor() {
  $('#dname').value = state.design.name || '';
  const host = $('#zones');
  host.replaceChildren();
  state.design.zones.forEach((z, zi) => {
    const zone = el('div', { class: 'zone' });
    zone.append(el('div', { class: 'zone-head' },
      el('label', {}, 'Zone', el('input', { class: 'zname', value: z.name, maxlength: 40, onchange: (e) => { z.name = e.target.value; changed(); } })),
      el('label', {}, '½ width in', num(z.outline.a, { min: 0.5, max: 20, label: 'zone half width (in)' }, (v) => { z.outline.a = v; changed(); })),
      el('label', {}, '½ height in', num(z.outline.b, { min: 0.5, max: 20, label: 'zone half height (in)' }, (v) => { z.outline.b = v; changed(); })),
      el('label', {}, 'cx in', num(z.outline.cx ?? 0, { min: -20, max: 20, label: 'zone centre x (in, + = wearer left)' }, (v) => { z.outline.cx = v; changed(); })),
      el('label', {}, 'cy in', num(z.outline.cy ?? 0, { min: -20, max: 20, label: 'zone centre y (in, + = up)' }, (v) => { z.outline.cy = v; changed(); })),
    ));
    z.stack.forEach((L, li) => {
      const kind = state.materials[L.material]?.kind || 'foam';
      const move = (to) => {
        if (to < 0 || to >= z.stack.length) return;
        const [m] = z.stack.splice(li, 1);
        z.stack.splice(to, 0, m);
        changed(true);
        document.querySelectorAll('.zone')[zi]?.querySelectorAll('.grip')[to]?.focus();
      };
      const grip = el('span', { class: 'grip', title: 'drag (or focus + ↑/↓) to reorder', tabindex: '0', role: 'button',
        'aria-label': `move layer ${li + 1}; use arrow keys`, draggable: 'true',
        onkeydown: (e) => { if (e.key === 'ArrowUp') { e.preventDefault(); move(li - 1); } if (e.key === 'ArrowDown') { e.preventDefault(); move(li + 1); } } }, '⋮⋮');
      const row = el('div', { class: `layer ${kind}` },
        grip,
        materialSelect(L.material, (v) => {
          L.material = v;
          if (state.materials[v].kind === 'plate' && !L.extent) L.extent = { cx: 0, cy: 0, a: Math.max(1, z.outline.a - 0.5), b: Math.max(1, z.outline.b - 0.5) };
          if (state.materials[v].kind === 'foam') delete L.extent;
          changed(true);
        }),
        el('label', { title: 'thickness (mm)' }, num(L.thickness_mm, { min: 0.5, max: 60, step: 0.5, label: 'thickness (mm)' }, (v) => { L.thickness_mm = v; changed(); })),
        el('button', { class: 'del', title: 'remove layer', 'aria-label': 'remove layer', onclick: () => { if (z.stack.length > 1) { z.stack.splice(li, 1); changed(true); } } }, '×'),
      );
      if (kind === 'plate') {
        const ext = L.extent || (L.extent = { cx: 0, cy: 0, a: z.outline.a, b: z.outline.b });
        row.append(el('div', { class: 'extent' },
          '½w', num(ext.a, { min: 0.5, max: 20, label: 'plate half width (in)' }, (v) => { ext.a = v; changed(); }),
          '½h', num(ext.b, { min: 0.5, max: 20, label: 'plate half height (in)' }, (v) => { ext.b = v; changed(); })));
      }
      grip.addEventListener('dragstart', (e) => {
        dragFrom = { zi, li };
        e.dataTransfer.setData('text/plain', `${zi}:${li}`);   // required by Firefox
        e.dataTransfer.setDragImage(row, 10, 10);
        row.classList.add('dragging');
      });
      grip.addEventListener('dragend', () => row.classList.remove('dragging'));
      row.addEventListener('dragover', (e) => { if (dragFrom?.zi === zi) { e.preventDefault(); row.classList.add('over'); } });
      row.addEventListener('dragleave', () => row.classList.remove('over'));
      row.addEventListener('drop', (e) => {
        e.preventDefault();
        if (dragFrom?.zi !== zi || dragFrom.li === li) return;
        row.classList.remove('over');
        const [moved] = z.stack.splice(dragFrom.li, 1);
        z.stack.splice(li, 0, moved);
        dragFrom = null;
        changed(true);
      });
      zone.append(row);
    });
    zone.append(el('button', { class: 'addlayer', onclick: () => { z.stack.push({ material: 'eva_100', thickness_mm: 10 }); changed(true); } }, '+ layer'));
    host.append(zone);
  });
}

let geomTimer = null;
function changed(rerender = false) {
  if (rerender) renderEditor();
  $('#report').classList.add('stale');
  $('#geomstatus').innerHTML = '<span class="spin"></span>updating…';
  clearTimeout(geomTimer);
  geomTimer = setTimeout(refreshGeometry, 600);
}

async function refreshMaxShift() {
  try {
    const r = await (await api(`/api/max_shift?torso_length_in=${Number($('#torso').value)}`)).json();
    $('#maxshift').textContent = `${r.max_shift_in} in`;
  } catch (e) { $('#maxshift').textContent = '–'; }
}

// ------------------------------------------------------------------ 3D viewer
const view = {};
function initViewer() {
  const host = $('#three');
  view.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  view.renderer.setPixelRatio(window.devicePixelRatio);
  view.renderer.setClearColor(css('--surface'));
  host.append(view.renderer.domElement);
  view.scene = new THREE.Scene();
  view.camera = new THREE.PerspectiveCamera(35, 1, 5, 5000);
  view.camera.position.set(520, 40, 1250);
  view.controls = new OrbitControls(view.camera, view.renderer.domElement);
  view.controls.target.set(0, -170, 0);
  view.controls.enableDamping = true;
  view.scene.add(new THREE.HemisphereLight(0xffffff, 0x303030, 1.6));
  const d = new THREE.DirectionalLight(0xffffff, 1.8);
  d.position.set(300, 400, 800);
  view.scene.add(d);
  view.group = new THREE.Group();
  view.scene.add(view.group);
  view.raycaster = new THREE.Raycaster();
  view.pointer = new THREE.Vector2();
  view.renderer.domElement.addEventListener('pointermove', onPointer3D);
  view.renderer.domElement.addEventListener('pointerleave', () => { $('#hover3d').hidden = true; });
  new ResizeObserver(resize3D).observe(host);
  resize3D();
  (function loop() { requestAnimationFrame(loop); view.controls.update(); view.renderer.render(view.scene, view.camera); })();
}

function resize3D() {
  const host = $('#three');
  const w = host.clientWidth || 1, h = host.clientHeight || 1;
  view.renderer.setSize(w, h);
  view.camera.aspect = w / h;
  view.camera.updateProjectionMatrix();
}

function onPointer3D(ev) {
  const rect = view.renderer.domElement.getBoundingClientRect();
  view.pointer.set(((ev.clientX - rect.left) / rect.width) * 2 - 1, -((ev.clientY - rect.top) / rect.height) * 2 + 1);
  view.raycaster.setFromCamera(view.pointer, view.camera);
  const hit = view.raycaster.intersectObjects(view.group.children.filter((o) => o.userData.layer), false)[0];
  const tip = $('#hover3d');
  if (!hit) { tip.hidden = true; return; }
  const L = hit.object.userData.layer;
  tip.textContent = `${L.name} · ${L.material} · ${fmt(L.mass_g, 0)} g`;
  tip.style.left = `${ev.clientX - rect.left + 12}px`;
  tip.style.top = `${ev.clientY - rect.top + 12}px`;
  tip.hidden = false;
}

function radialOffset(p, R, h) {
  const dx = p[0], dz = p[2] + R, n = Math.hypot(dx, dz) || 1;
  return new THREE.Vector3(p[0] + (dx / n) * h, p[1], p[2] + (dz / n) * h);
}

function buildScene(g) {
  for (const o of [...view.group.children]) { view.group.remove(o); o.geometry?.dispose(); o.material?.dispose(); }
  const R = g.landmarks.chest_radius_mm;
  const L = g.torso_length_mm;
  // torso context: cylinder patch (single-curvature chest model)
  const body = new THREE.Mesh(
    new THREE.CylinderGeometry(R - 0.5, R - 0.5, L + 180, 72, 1, true, -1.9, 3.8),
    new THREE.MeshStandardMaterial({ color: 0x77766f, transparent: true, opacity: 0.28, side: THREE.DoubleSide, depthWrite: false }));
  body.position.set(0, -L / 2 + 30, -R);
  view.group.add(body);
  let top = 0;
  for (const layer of g.layers) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(layer.positions, 3));
    geo.setIndex(layer.indices);
    geo.computeVertexNormals();
    const cardiacZone = g.layers[0].zone;
    const color = layer.kind === 'plate' ? css('--plate') : (layer.zone === cardiacZone ? css('--foam') : css('--torso'));
    const other = layer.zone !== cardiacZone && layer.kind !== 'plate';
    const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color, roughness: 0.75, metalness: 0.0,
      transparent: other, opacity: other ? 0.55 : 1, side: THREE.DoubleSide }));
    mesh.userData.layer = layer;
    view.group.add(mesh);
    geo.computeBoundingBox();
    top = Math.max(top, geo.boundingBox.max.z);
  }
  const lift = top + 3;
  const sil = g.landmarks.silhouette_outline.map((p) => radialOffset(p, R, lift));
  const silLine = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(sil), new THREE.LineBasicMaterial({ color: css('--bad'), depthTest: false }));
  const shift = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(g.landmarks.shift_circle.map((p) => radialOffset(p, R, lift))),
    new THREE.LineDashedMaterial({ color: css('--good'), dashSize: 6, gapSize: 4, depthTest: false }));
  shift.computeLineDistances();
  for (const line of [silLine, shift]) { line.renderOrder = 10; view.group.add(line); }
  const c = g.landmarks.silhouette_center;
  view.controls.target.set(c[0] * 0.3, c[1] - 40, 0);
}

let geomSeq = 0;
async function refreshGeometry() {
  const seq = ++geomSeq;
  const status = $('#geomstatus');
  status.innerHTML = '<span class="spin"></span>building geometry…';
  try {
    const g = await (await api('/api/geometry', { design: state.design, torso_length_in: Number($('#torso').value) })).json();
    if (seq !== geomSeq) return;
    buildScene(g);
    const mass = g.layers.reduce((s, l) => s + l.mass_g, 0);
    status.textContent = `CAD mass ${fmt(mass, 0)} g`;
  } catch (e) {
    if (seq === geomSeq) status.textContent = `geometry: ${e.message}`;
  }
}

// ------------------------------------------------------------------ report
function verdictClass(v) { return v === 'PASS' ? 'pass' : v === 'FAIL' ? 'fail' : 'beyond'; }

function renderReport(rep) {
  const host = $('#report');
  host.classList.remove('stale');
  host.replaceChildren();
  const mc = rep.p95_verdict !== null;
  host.append(el('div', { class: 'verdicts' },
    el('div', { class: 'card' }, el('div', { class: 'k' }, 'ND200 (nominal)'),
      el('div', { class: `v ${verdictClass(rep.verdict)}` }, rep.verdict.replace(' (not ND200)', '')),
      el('div', { class: 's' }, `min margin ${fmt(rep.min_margin_pct)}%`)),
    el('div', { class: 'card' }, el('div', { class: 'k' }, 'P95 Monte Carlo'),
      el('div', { class: `v ${mc ? verdictClass(rep.p95_verdict) : ''}` }, mc ? rep.p95_verdict : 'off'),
      el('div', { class: 's' }, mc ? `min P95 margin ${fmt(rep.min_p95_margin_pct)}%` : 'set samples > 0')),
    el('div', { class: 'card' }, el('div', { class: 'k' }, 'Protector'),
      el('div', { class: 'v' }, `${fmt(rep.metrics.mass_g, 0)} g`),
      el('div', { class: 's' }, `${fmt(rep.metrics.cardiac_thickness_mm)} mm cardiac stack`)),
  ));
  const notes = [];
  if (rep.beyond_standard) notes.push('Beyond standard: this projectile is not an ND200 projectile — no PASS/FAIL applies.');
  if (!rep.covers_silhouette) notes.push('Cardiac zone does not cover the silhouette + max shift radius.');
  notes.push(...rep.flags);
  for (const i of rep.impacts) if (i.bottomed.length) notes.push(`Bottomed out (S${i.sample} ${i.location}): ${i.bottomed.join(', ')}`);
  if (notes.length) host.append(el('ul', { class: 'flags' }, notes.map((n) => el('li', {}, n))));

  const t = el('table', { class: 'imp' });
  t.append(el('tr', {}, ['S', 'Location', 'Shift (in)', 'Peak', ...(mc ? ['P95'] : []), 'Limit', 'Margin', ''].map((h) => el('th', {}, h))));
  for (const i of rep.impacts) {
    const m = mc ? i.p95_margin_pct : i.margin_pct;
    const bar = el('span', { class: 'mbar', style: `width:${Math.min(Math.abs(m), 100) * 0.4}px;background:${m >= 0 ? css('--good') : css('--bad')}` });
    t.append(el('tr', { title: i.local },
      el('td', {}, `${i.sample}`), el('td', {}, i.location.replace('_', ' ')),
      el('td', {}, `${fmt(i.shift_in[0], 2)}, ${fmt(i.shift_in[1], 2)}`),
      el('td', {}, fmt(i.peak_lbf)), ...(mc ? [el('td', {}, fmt(i.p95_lbf))] : []),
      el('td', {}, fmt(i.limit_lbf, 0)), el('td', {}, `${fmt(m)}%`, bar),
      el('td', { class: i.passed ? 'pass' : 'fail' }, i.passed ? 'PASS' : 'FAIL')));
  }
  host.append(t);
  for (const cond of ['30mph', '50mph']) host.append(traceChart(rep.impacts.filter((i) => i.condition === cond)));
  host.append(el('p', { class: 'hint' }, `${rep.label}. Units lbf, CFC120-filtered. Hover a chart for values; hover a table row for the stack under the impact.`));
}

const SERIES = ['--s1', '--s2', '--s3'];
function traceChart(imps) {
  const W = 440, H = 190, m = { l: 40, r: 10, t: 22, b: 26 };
  const wrap = el('div', { class: 'chart' });
  wrap.append(el('div', { class: 'title' }, `Sample ${imps[0].sample} · ${fmt(imps[0].velocity_mph, 0)} mph — load-cell force`));
  const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `force-time traces at ${imps[0].velocity_mph} mph` });
  const tmax = 15;
  const ymax = Math.max(imps[0].limit_lbf * 1.15, ...imps.map((i) => i.peak_lbf * 1.08));
  const x = (v) => m.l + (v / tmax) * (W - m.l - m.r);
  const y = (v) => H - m.b - (v / ymax) * (H - m.t - m.b);
  const ink2 = css('--ink2'), grid = css('--line');
  for (let k = 0; k <= 4; k++) {
    const v = (ymax / 4) * k;
    svg('line', { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: grid, 'stroke-width': 0.6 }, s);
    svg('text', { x: m.l - 4, y: y(v) + 3, 'text-anchor': 'end', 'font-size': 9, fill: ink2 }, s).textContent = v.toFixed(0);
  }
  for (let k = 5; k <= tmax; k += 5) svg('text', { x: x(k), y: H - 8, 'text-anchor': k === tmax ? 'end' : 'middle', 'font-size': 9, fill: ink2 }, s).textContent = `${k} ms`;
  const lim = imps[0].limit_lbf;
  svg('line', { x1: m.l, x2: W - m.r, y1: y(lim), y2: y(lim), stroke: css('--bad'), 'stroke-dasharray': '5 4', 'stroke-width': 1.2 }, s);
  svg('text', { x: W - m.r, y: y(lim) - 4, 'text-anchor': 'end', 'font-size': 9, fill: css('--bad') }, s).textContent = `limit ${lim} lbf`;
  imps.forEach((i, k) => {
    const pts = i.t_ms.map((tt, j) => (tt >= 0 && tt <= tmax ? `${x(tt).toFixed(1)},${y(Math.max(i.force_lbf[j], 0)).toFixed(1)}` : null)).filter(Boolean);
    svg('polyline', { points: pts.join(' '), fill: 'none', stroke: css(SERIES[k]), 'stroke-width': 2, 'stroke-linejoin': 'round',
      'stroke-dasharray': k === 2 ? '6 4' : null }, s);
  });
  // legend (identity never colour-alone: name + peak)
  imps.forEach((i, k) => {
    const lx = m.l + 4 + k * 130;
    svg('line', { x1: lx, x2: lx + 12, y1: 7.5, y2: 7.5, stroke: css(SERIES[k]), 'stroke-width': 2, 'stroke-dasharray': k === 2 ? '4 2' : null }, s);
    svg('text', { x: lx + 14, y: 10, 'font-size': 9, fill: ink2 }, s).textContent = `${i.location.replace('_', ' ')} ${fmt(i.peak_lbf, 0)}`;
  });
  const cross = svg('line', { y1: m.t, y2: H - m.b, stroke: ink2, 'stroke-width': 0.8, visibility: 'hidden' }, s);
  const hit = svg('rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: 'transparent' }, s);
  const tip = $('#charttip');
  hit.addEventListener('pointermove', (ev) => {
    const r = s.getBoundingClientRect();
    const tt = ((ev.clientX - r.left) / r.width * W - m.l) / (W - m.l - m.r) * tmax;
    if (tt < 0 || tt > tmax) return;
    cross.setAttribute('x1', x(tt)); cross.setAttribute('x2', x(tt)); cross.setAttribute('visibility', 'visible');
    const rows = imps.map((i) => {
      const j = Math.max(0, Math.min(i.t_ms.length - 1, Math.round((tt - i.t_ms[0]) / (i.t_ms[1] - i.t_ms[0] || 0.05))));
      return `${i.location.replace('_', ' ')}: ${fmt(i.force_lbf[j] ?? 0)} lbf`;
    });
    tip.innerHTML = `<b>${tt.toFixed(2)} ms</b><br>${rows.join('<br>')}`;
    tip.style.left = `${ev.clientX + 14}px`; tip.style.top = `${ev.clientY + 14}px`; tip.hidden = false;
  });
  hit.addEventListener('pointerleave', () => { cross.setAttribute('visibility', 'hidden'); tip.hidden = true; });
  wrap.append(s);
  return wrap;
}

async function runND200() {
  const btn = $('#run');
  btn.disabled = true;
  btn.innerHTML = '<span class="spin"></span>Simulating 6 impacts…';
  try {
    state.report = await (await api('/api/nd200', runRequest())).json();
    renderReport(state.report);
    selectTab('report');
  } catch (e) {
    $('#report').replaceChildren(el('p', { class: 'fail' }, `ND200 run failed: ${e.message}`));
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run ND200';
  }
}

// ------------------------------------------------------------------ exports
async function doExport(kind) {
  const msg = $('#exportmsg');
  msg.innerHTML = '<span class="spin"></span>exporting…';
  try {
    const body = kind === 'cad' ? { design: state.design, torso_length_in: Number($('#torso').value) } : runRequest();
    const r = await api(`/api/export/${kind}`, body);
    const name = /filename="([^"]+)"/.exec(r.headers.get('Content-Disposition') || '')?.[1] || `chestforge.${kind}`;
    download(await r.blob(), name);
    msg.textContent = `saved ${name}`;
  } catch (e) { msg.textContent = `export failed: ${e.message}`; }
}

function download(blob, name) {
  const a = el('a', { href: URL.createObjectURL(blob), download: name });
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

// ------------------------------------------------------------------ optimizer
const OBJ_LABEL = { p95_50mph_lbf: 'P95 load @ 50 mph (lbf)', mass_g: 'Protector mass (g)', thickness_mm: 'Cardiac thickness (mm)', bending_Nm: 'Bending stiffness (N·m)' };

async function loadRuns(select) {
  const runs = await (await api('/api/runs')).json();
  const sel = $('#runsel');
  const cur = select || sel.value;
  sel.replaceChildren(...runs.map((r) => el('option', { value: r.name, selected: r.name === cur ? '' : null }, `${r.name} (${r.evaluations})${r.running ? ' ●' : ''}`)));
  if (!runs.length) sel.append(el('option', { value: '' }, 'no runs yet'));
  if (sel.value) await loadRun(sel.value);
}

async function loadRun(name) {
  clearTimeout(state.runTimer);
  if (!name) return;
  try {
    state.run = await (await api(`/api/runs/${encodeURIComponent(name)}`)).json();
  } catch (e) { $('#runstatus').textContent = e.message; return; }
  const r = state.run;
  const feas = r.points.filter((p) => p.feasible).length;
  $('#runstatus').textContent = `${r.running ? 'running · ' : ''}${feas}/${r.points.length} feasible`;
  for (const id of ['#xobj', '#yobj']) {
    const s = $(id);
    if (!s.options.length) {
      s.append(...r.objectives.map((o) => el('option', { value: o }, OBJ_LABEL[o] || o)));
      s.value = id === '#xobj' ? 'mass_g' : 'p95_50mph_lbf';
    }
  }
  drawPareto();
  if (r.running) {
    const log = await (await api(`/api/optimize/${encodeURIComponent(name)}/log`)).json();
    $('#runlog').hidden = false; $('#runlog').textContent = log.log;
    state.runTimer = setTimeout(() => loadRun(name), 4000);
  }
}

function drawPareto() {
  const host = $('#pareto');
  host.replaceChildren();
  const r = state.run;
  if (!r || !r.points.length) { host.append(el('p', { class: 'empty' }, 'No evaluations yet.')); return; }
  const xo = $('#xobj').value, yo = $('#yobj').value;
  const W = 440, H = 300, m = { l: 48, r: 10, t: 24, b: 34 };
  const pts = r.points.filter((p) => isFinite(p.objectives[xo]) && isFinite(p.objectives[yo]));
  const xs = pts.map((p) => p.objectives[xo]), ys = pts.map((p) => p.objectives[yo]);
  const pad = (a, b) => [a - (b - a) * 0.05, b + (b - a) * 0.05];
  const [x0, x1] = pad(Math.min(...xs), Math.max(...xs)), [y0, y1] = pad(Math.min(...ys), Math.max(...ys));
  const X = (v) => m.l + (v - x0) / (x1 - x0 || 1) * (W - m.l - m.r);
  const Y = (v) => H - m.b - (v - y0) / (y1 - y0 || 1) * (H - m.t - m.b);
  const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `${OBJ_LABEL[xo]} vs ${OBJ_LABEL[yo]}` });
  const ink2 = css('--ink2'), grid = css('--line');
  for (let k = 0; k <= 4; k++) {
    const yv = y0 + (y1 - y0) * k / 4, xv = x0 + (x1 - x0) * k / 4;
    svg('line', { x1: m.l, x2: W - m.r, y1: Y(yv), y2: Y(yv), stroke: grid, 'stroke-width': 0.6 }, s);
    svg('text', { x: m.l - 4, y: Y(yv) + 3, 'text-anchor': 'end', 'font-size': 9, fill: ink2 }, s).textContent = tick(yv);
    svg('text', { x: X(xv), y: H - m.b + 13, 'text-anchor': k === 4 ? 'end' : k === 0 ? 'start' : 'middle', 'font-size': 9, fill: ink2 }, s).textContent = tick(xv);
  }
  svg('text', { x: (W + m.l) / 2, y: H - 4, 'text-anchor': 'middle', 'font-size': 10, fill: ink2 }, s).textContent = OBJ_LABEL[xo];
  svg('text', { x: 10, y: (H - m.b + m.t) / 2, transform: `rotate(-90 10 ${(H - m.b + m.t) / 2})`, 'text-anchor': 'middle', 'font-size': 10, fill: ink2 }, s).textContent = OBJ_LABEL[yo];
  // legend
  const leg = [['infeasible', 'none', ink2], ['feasible', css('--s1'), css('--s1')], ['Pareto set', css('--s2'), css('--s2')]];
  leg.forEach(([name, fill, stroke], k) => {
    svg('circle', { cx: m.l + 6 + k * 90, cy: 10, r: 4, fill, stroke, 'stroke-width': 1 }, s);
    svg('text', { x: m.l + 14 + k * 90, y: 13, 'font-size': 9, fill: ink2 }, s).textContent = name;
  });
  const order = [...pts].sort((a, b) => (a.pareto - b.pareto) || (a.feasible - b.feasible));
  const tip = $('#charttip');
  for (const p of order) {
    const fill = p.pareto ? css('--s2') : p.feasible ? css('--s1') : 'none';
    const c = svg('circle', { cx: X(p.objectives[xo]), cy: Y(p.objectives[yo]), r: p.pareto ? 5 : 3.5, fill,
      stroke: p.pareto ? css('--surface') : p.feasible ? 'none' : ink2, 'stroke-width': p.pareto ? 2 : 0.8,
      'fill-opacity': p.feasible && !p.pareto ? 0.6 : 1, style: 'cursor:pointer' }, s);
    const hitc = svg('circle', { cx: c.getAttribute('cx'), cy: c.getAttribute('cy'), r: 8, fill: 'transparent', style: 'cursor:pointer' }, s);
    hitc.addEventListener('pointermove', (ev) => {
      tip.innerHTML = `<b>#${p.id} · ${p.method}</b><br>${p.design.name}<br>` +
        Object.entries(p.objectives).map(([k, v]) => `${OBJ_LABEL[k] || k}: ${fmt(v)}`).join('<br>') +
        `<br>min P95 margin: ${fmt(p.min_p95_margin_pct)}% ${p.feasible ? '✓' : '✗'}`;
      tip.style.left = `${ev.clientX + 14}px`; tip.style.top = `${ev.clientY + 14}px`; tip.hidden = false;
    });
    hitc.addEventListener('pointerleave', () => { tip.hidden = true; });
    hitc.addEventListener('click', () => loadDesignFromRun(p));
  }
  host.append(s);
}

function loadDesignFromRun(p) {
  state.design = structuredClone(p.design);
  if (state.run) { $('#torso').value = state.run.torso_length_in; $('#projectile').value = state.run.projectile; refreshMaxShift(); }
  renderEditor();
  changed();
  $('#report').replaceChildren(el('p', { class: 'empty' }, `Loaded design #${p.id} from run “${state.run.name}”. Press Run ND200 for the full report.`));
}

async function startRun() {
  const body = { name: $('#newname').value, method: $('#newmethod').value, trials: Number($('#newtrials').value), rounds: Number($('#newrounds').value) };
  try {
    await api('/api/optimize', body);
    setTimeout(() => loadRuns(body.name), 1500);
  } catch (e) { $('#runstatus').textContent = e.message; }
}

// ------------------------------------------------------------------ validation (measured traces)
async function loadTraces(select = []) {
  const list = await (await api('/api/validation/traces')).json();
  const t = el('table', { class: 'vlist' }, el('tr', {}, ['', 'Trace', 'Peak lbf'].map((h) => el('th', {}, h))));
  for (const x of list) {
    if (x.error) { t.append(el('tr', {}, el('td', {}), el('td', { colspan: 2, class: 'fail' }, `${x.file}: ${x.error}`))); continue; }
    t.append(el('tr', {},
      el('td', {}, el('input', { type: 'checkbox', value: x.file, checked: select.includes(x.file) || !select.length ? '' : null, 'aria-label': `use ${x.file}` })),
      el('td', { title: x.file }, x.file.split('/').pop(), ' ', x.synthetic ? el('span', { class: 'tag', title: 'generated, not measured' }, 'synthetic') : null,
        el('div', { class: 'hint' }, `${x.rig.replace('_', ' ')} · ${x.stack} · ${fmt(x.velocity_ms, 2)} m/s`),
        ...(x.warnings || []).map((w) => el('div', { class: 'hint' }, w))),
      el('td', {}, fmt(x.peak_lbf))));
  }
  $('#vallist').replaceChildren(list.length ? t : el('p', { class: 'empty' }, 'No traces yet — upload one.'));
}

const chosenTraces = () => [...document.querySelectorAll('#vallist input:checked')].map((c) => c.value);

function overlayChart(c) {
  const W = 220, H = 130, m = { l: 30, r: 6, t: 8, b: 18 };
  const series = [['measured', c.measured, css('--ink'), null], ['model', c.model, css('--s1'), null]];
  if (c.fitted) series.push(['fitted', c.fitted, css('--s2'), '5 3']);
  const tmax = Math.max(...c.measured.t_ms), ymax = Math.max(...series.flatMap(([, d]) => d.lbf)) * 1.08;
  const X = (v) => m.l + (v + 1) / (tmax + 1) * (W - m.l - m.r), Y = (v) => H - m.b - v / ymax * (H - m.t - m.b);
  const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `overlay ${c.trace}` });
  for (let k = 0; k <= 2; k++) {
    const v = ymax / 2 * k;
    svg('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: css('--line'), 'stroke-width': 0.5 }, s);
    svg('text', { x: m.l - 3, y: Y(v) + 3, 'text-anchor': 'end', 'font-size': 8, fill: css('--ink2') }, s).textContent = v.toFixed(0);
  }
  svg('text', { x: W - m.r, y: H - 4, 'text-anchor': 'end', 'font-size': 8, fill: css('--ink2') }, s).textContent = `${tmax.toFixed(0)} ms`;
  for (const [, d, color, dash] of series) {
    svg('polyline', { points: d.t_ms.map((tt, j) => `${X(tt).toFixed(1)},${Y(Math.max(d.lbf[j], 0)).toFixed(1)}`).join(' '),
      fill: 'none', stroke: color, 'stroke-width': color === css('--ink') ? 2.2 : 1.5, 'stroke-dasharray': dash }, s);
  }
  return el('div', { class: 'chart' }, el('div', { class: 'title' }, c.trace), s);
}

function summaryCards(before, after) {
  const card = (k, b, a, unit, d = 1) => el('div', { class: 'card' }, el('div', { class: 'k' }, k),
    el('div', { class: 'v' }, a === undefined ? `${fmt(b, d)}${unit}` : `${fmt(b, d)} → ${fmt(a, d)}${unit}`));
  return el('div', { class: 'verdicts' },
    card('mean |peak error|', before.mean_abs_peak_err_pct, after?.mean_abs_peak_err_pct, '%'),
    card('mean NRMSE', before.mean_abs_nrmse_pct, after?.mean_abs_nrmse_pct, '%'),
    card('mean R²', before.mean_r2, after?.mean_r2, '', 3));
}

function legendLine(fitted) {
  return el('p', { class: 'hint' }, el('span', { style: `color:${css('--ink')}` }, '━ measured  '),
    el('span', { style: `color:${css('--s1')}` }, '━ current model  '), fitted ? el('span', { style: `color:${css('--s2')}` }, '╍ fitted model') : null);
}

async function valCompare() {
  const out = $('#valout');
  out.innerHTML = '<span class="spin"></span>simulating each test…';
  try {
    const r = await (await api('/api/validation/compare', { files: chosenTraces() })).json();
    const t = el('table', { class: 'imp' }, el('tr', {}, ['Trace', 'Peak meas', 'Peak sim', 'Err', 'NRMSE', 'R²'].map((h) => el('th', {}, h))));
    for (const x of r.rows) t.append(el('tr', {}, el('td', {}, x.trace), el('td', {}, fmt(x.peak_exp_lbf)), el('td', {}, fmt(x.peak_sim_lbf)),
      el('td', {}, `${fmt(x.peak_err_pct)}%`), el('td', {}, `${fmt(x.nrmse_pct)}%`), el('td', {}, fmt(x.r2, 3))));
    out.replaceChildren(summaryCards(r.summary), t, legendLine(false), el('div', { class: 'charts-sm' }, r.curves.map(overlayChart)));
  } catch (e) { out.textContent = `compare failed: ${e.message}`; }
}

async function valFit(kind) {
  const out = $('#valout');
  out.innerHTML = '<span class="spin"></span>fitting (tens of simulations)…';
  const body = { kind, files: chosenTraces() };
  if (kind === 'material') { body.material = $('#valmat').value; if ($('#valnewid').value) body.new_id = $('#valnewid').value; }
  try {
    const r = await (await api('/api/validation/fit', body)).json();
    const p = r.params;
    const plist = kind === 'nts'
      ? Object.entries(p).map(([k, v]) => el('li', {}, `${k}: ${Number(v).toPrecision(4)}`))
      : [el('li', {}, `new material id: ${p.new_id} (from ${p.base})`), ...Object.entries(p.multipliers).map(([k, v]) => el('li', {}, `${k} × ${fmt(v, 3)}`)),
         ...(Object.keys(p.correlation).length ? [el('li', { class: 'beyond' }, `not separately identifiable: ${Object.keys(p.correlation).join(', ')}`)] : [])];
    const apply = el('button', { class: 'primary', disabled: r.synthetic ? '' : null, title: r.synthetic ? 'synthetic traces cannot be applied' : 'write to config' },
      r.synthetic ? 'Apply (blocked: synthetic data)' : 'Apply to configuration');
    apply.addEventListener('click', async () => {
      if (!confirm(kind === 'nts' ? 'Write config/nts_fitted.yaml and recalibrate the NTS?' : `Add material ${p.new_id} to config/materials_fitted.yaml?`)) return;
      try { const a = await (await api(`/api/validation/apply/${r.fit_id}`, {})).json(); apply.replaceWith(el('p', { class: 'pass' }, `Applied: ${JSON.stringify(a)}`)); state.meta = null; }
      catch (e) { apply.replaceWith(el('p', { class: 'fail' }, e.message)); }
    });
    out.replaceChildren(summaryCards(r.before, r.after), el('ul', { class: 'flags', style: `color:${css('--ink2')}` }, plist),
      el('div', { class: 'actions' }, apply), legendLine(true), el('div', { class: 'charts-sm' }, r.curves.map(overlayChart)));
  } catch (e) { out.textContent = `fit failed: ${e.message}`; }
}

async function valUpload(ev) {
  const f = ev.target.files[0];
  if (!f) return;
  const fd = new FormData();
  fd.append('file', f);
  try {
    const r = await (await fetch('/api/validation/upload', { method: 'POST', body: fd })).json();
    if (r.detail) throw new Error(typeof r.detail === 'string' ? r.detail : JSON.stringify(r.detail));
    await loadTraces([r.file]);
    $('#valout').textContent = `uploaded ${r.file}${r.warnings.length ? ' — ' + r.warnings.join('; ') : ''}`;
  } catch (e) { $('#valout').textContent = `upload failed: ${e.message}`; }
  ev.target.value = '';
}

// ------------------------------------------------------------------ tabs + boot
function selectTab(name) {
  for (const b of document.querySelectorAll('.tabs button')) b.setAttribute('aria-selected', String(b.dataset.tab === name));
  $('#tab-report').hidden = name !== 'report';
  $('#tab-opt').hidden = name !== 'opt';
  $('#tab-val').hidden = name !== 'val';
  if (name === 'opt') loadRuns();
  if (name === 'val') loadTraces();
}

async function boot() {
  state.meta = await (await api('/api/meta')).json();
  for (const m of state.meta.materials) state.materials[m.id] = m;
  $('#projectile').replaceChildren(...state.meta.projectiles.map((p) => el('option', { value: p.id }, p.nd200 ? p.id : `${p.id} (beyond standard)`)));
  state.design = structuredClone(state.meta.default_design);
  renderEditor();
  initViewer();
  refreshMaxShift();
  refreshGeometry();
  $('#dname').addEventListener('change', (e) => { state.design.name = e.target.value; });
  $('#torso').addEventListener('change', () => { refreshMaxShift(); changed(); });
  $('#projectile').addEventListener('change', () => $('#report').classList.add('stale'));
  $('#run').addEventListener('click', runND200);
  for (const b of document.querySelectorAll('[data-export]')) b.addEventListener('click', () => doExport(b.dataset.export));
  $('#savejson').addEventListener('click', () => download(new Blob([JSON.stringify(state.design, null, 1)], { type: 'application/json' }), 'design.json'));
  for (const b of document.querySelectorAll('.tabs button')) b.addEventListener('click', () => selectTab(b.dataset.tab));
  $('#runsel').addEventListener('change', (e) => loadRun(e.target.value));
  $('#xobj').addEventListener('change', drawPareto);
  $('#yobj').addEventListener('change', drawPareto);
  $('#startrun').addEventListener('click', startRun);
  $('#valmat').replaceChildren(...state.meta.materials.filter((m) => m.kind === 'foam').map((m) => el('option', { value: m.id }, m.id)));
  $('#valcompare').addEventListener('click', valCompare);
  $('#valfitnts').addEventListener('click', () => valFit('nts'));
  $('#valfitmat').addEventListener('click', () => valFit('material'));
  $('#valupload').addEventListener('change', valUpload);
}

boot().catch((e) => { document.body.prepend(el('p', { class: 'fail' }, `ChestForge failed to start: ${e.message}`)); });
