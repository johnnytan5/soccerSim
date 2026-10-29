"use strict";
/* World-model view for the Control tab: what each brain believes, drawn on the field map and listed in a panel.
   Data: status.worldmodel.robots[] from the match bridge (each brain's /brain/world_model snapshot, moved into the world
   frame and scored against the simulator's ground truth). Works for the 2D and the MuJoCo simulator alike.
   Hooks into app.js through window.SoccerUI. */
(() => {
const ui = window.SoccerUI;
if (!ui) return;
const {S, $, h, svgEl, fmt} = ui;

const K95 = 2.4477;   // 95 % ellipse of a 2D Gaussian: 2.4477 sigma
const STOP_DECEL = 0.4;
const KEY = "soccerLauncher.wm";
const LAYERS = [
  ["fused", "Fused ball", "Every robot's estimate combined; ellipse = 95 % uncertainty, arrow = velocity (1 s), + = where it stops"],
  ["local", "Own camera ball", "What this robot's own filter believes from its own detections"],
  ["sources", "Contributions", "Each robot's ball estimate and its weight in the fusion; red = rejected as inconsistent"],
  ["teammates", "Predicted teammates", "Teammates extrapolated from their last packet; circle = uncertainty"],
  ["oppLocal", "Own opponent tracks", "Opponents this robot tracks from its camera (teammates are filtered out)"],
  ["oppFused", "Fused opponents", "Opponent tracks combined from every robot's reports, including ones this robot cannot see"],
];
let saved = {};
try { saved = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { /* private mode */ }
const wm = {view: saved.view !== undefined ? saved.view : "all", layers: {fused: true, local: false, sources: true, teammates: true, oppLocal: false, oppFused: true, ...(saved.layers || {})},
  layers_el: null, panel: null, key: ""};
const persist = () => { try { localStorage.setItem(KEY, JSON.stringify({view: wm.view, layers: wm.layers})); } catch (e) { /* ignore */ } };

/* ------------------------------------------------------------------ drawing */
const deg = r => r * 180 / Math.PI;
function ellipse(x, y, smaj, smin, ang, cls, k = K95) {
  return svgEl("ellipse", {class: cls, cx: 0, cy: 0, rx: Math.max(.04, k * smaj), ry: Math.max(.04, k * smin), transform: `translate(${x} ${y}) rotate(${deg(ang)})`});
}
function arrow(x, y, vx, vy, cls, seconds = 1) {
  const sp = Math.hypot(vx, vy);
  if (sp < .15) return null;
  const ex = x + vx * seconds, ey = y + vy * seconds, a = Math.atan2(vy, vx), n = .28;
  return svgEl("path", {class: cls, d: `M${x} ${y}L${ex} ${ey}M${ex} ${ey}L${ex - Math.cos(a - .5) * n} ${ey - Math.sin(a - .5) * n}M${ex} ${ey}L${ex - Math.cos(a + .5) * n} ${ey - Math.sin(a + .5) * n}`});
}
function text(x, y, str, cls = "wm-label") {
  return svgEl("text", {class: cls, x: 0, y: 0, "text-anchor": "middle", transform: `translate(${x} ${y}) scale(1,-1)`}, str);
}
function diamond(x, y, r, cls) {
  return svgEl("path", {class: cls, d: `M${x} ${y + r}L${x + r} ${y}L${x} ${y - r}L${x - r} ${y}Z`});
}
function cross(x, y, r, cls) {
  return svgEl("path", {class: cls, d: `M${x - r} ${y - r}L${x + r} ${y + r}M${x - r} ${y + r}L${x + r} ${y - r}`});
}

function drawRobot(r, under, over, detail) {
  const L = wm.layers, team = r.team;
  const fb = r.fused_ball, lb = r.local_ball;
  if (detail && L.oppFused) {
    for (const o of r.opponents_fused || []) {
      under.append(ellipse(o.x, o.y, o.smaj, o.smin, o.ang, "wm-opp-ellipse"));
      over.append(diamond(o.x, o.y, .26, `wm-opp-fused ${o.n > 1 ? "shared" : ""}`));
      const v = arrow(o.x, o.y, o.vx, o.vy, "wm-opp-arrow"); if (v) over.append(v);
      if (o.n > 1) over.append(text(o.x, o.y - .55, `${o.n} robots`, "wm-label small"));
    }
  }
  if (detail && L.oppLocal) {
    for (const o of r.opponents_local || []) {
      over.append(svgEl("circle", {class: "wm-opp-local", cx: o.x, cy: o.y, r: .13}), text(o.x, o.y + .3, "#" + o.id, "wm-label small"));
    }
  }
  if (detail && L.teammates) {
    for (const t of r.teammates || []) {
      under.append(svgEl("circle", {class: `wm-tm ${team}`, cx: t.x, cy: t.y, r: Math.max(.2, K95 * t.sigma)}));
      over.append(text(t.x, t.y - .75, `P${t.player} ${fmt(t.age, 1)}s`, "wm-label small"));
    }
  }
  if (detail && L.local && lb && lb.valid) {
    under.append(ellipse(lb.x, lb.y, lb.smaj, lb.smin, lb.ang, "wm-local-ellipse"));
    over.append(svgEl("circle", {class: "wm-local", cx: lb.x, cy: lb.y, r: .14}));
    const v = arrow(lb.x, lb.y, lb.vx, lb.vy, "wm-local-arrow"); if (v) over.append(v);
  }
  if (fb && fb.valid) {
    if (detail && L.sources) {
      for (const s of fb.sources || []) {
        const w = s.rejected ? 0 : s.w;
        over.append(svgEl("path", {class: `wm-src ${s.rejected ? "rejected" : ""}`, d: `M${s.x} ${s.y}L${fb.x} ${fb.y}`, "stroke-width": .03 + .16 * w}));
        over.append(svgEl("circle", {class: `wm-src-dot ${s.rejected ? "rejected" : ""}`, cx: s.x, cy: s.y, r: .09}));
        if (s.rejected) over.append(cross(s.x, s.y, .2, "wm-reject"));
        over.append(text(s.x, s.y + .32, `${s.player === 0 ? "self" : "P" + s.player} ${s.rejected ? "✕" : fmt(s.w, 2)}`, "wm-label small"));
      }
    }
    if (L.fused) {
      under.append(ellipse(fb.x, fb.y, fb.smaj, fb.smin, fb.ang, `wm-fused-ellipse ${detail ? "" : team}`));
      over.append(svgEl("circle", {class: `wm-fused ${detail ? "" : team}`, cx: fb.x, cy: fb.y, r: detail ? .2 : .16}));
      const sp = Math.hypot(fb.vx, fb.vy);
      const v = arrow(fb.x, fb.y, fb.vx, fb.vy, "wm-fused-arrow"); if (v) over.append(v);
      if (detail && sp > .3) {
        const d = sp * sp / (2 * STOP_DECEL);
        over.append(cross(fb.x + fb.vx / sp * d, fb.y + fb.vy / sp * d, .14, "wm-stop"));
      }
      if (!detail) over.append(text(fb.x, fb.y - .42, r.label, "wm-label small"));
    }
  }
}

function redraw() {
  const layers = ui.layers;
  if (!layers) return;
  const robots = (S.status.worldmodel && S.status.worldmodel.robots) || [];
  layers.under.replaceChildren();
  layers.over.replaceChildren();
  if (wm.view === "off" || !S.ready) return;
  if (wm.view === "all") robots.forEach(r => drawRobot(r, layers.under, layers.over, false));
  else {
    const r = robots.find(x => x.index === wm.view);
    if (r) drawRobot(r, layers.under, layers.over, true);
  }
}

/* -------------------------------------------------------------------- panel */
function setView(v) { wm.view = v; persist(); renderControls(); redraw(); renderPanel(); }
function renderControls() {
  if (!wm.layers_el) return;
  const robots = (S.status.worldmodel && S.status.worldmodel.robots) || [];
  const views = [["off", "Off"], ["all", "All"], ...robots.map(r => [r.index, r.label])];
  wm.layers_el.views.replaceChildren(h("span", {class: "hint", text: "View of:"}), ...views.map(([v, t]) =>
    h("button", {class: `pick ${typeof v === "number" ? (robots.find(r => r.index === v) || {}).team || "" : ""} ${wm.view === v ? "on" : ""}`, type: "button",
      "aria-pressed": String(wm.view === v), onclick: () => setView(v)}, h("b", {text: t}))));
  wm.layers_el.toggles.replaceChildren(...LAYERS.map(([k, label, tip]) => h("label", {class: "check", title: tip},
    h("input", {type: "checkbox", checked: wm.layers[k] ? true : null, onchange: e => { wm.layers[k] = e.target.checked; persist(); redraw(); }}), label)));
}
function row(k, ...v) { return [h("div", {class: "k", text: k}), h("div", {class: "v"}, ...v)]; }
function tg(text, kind = "") { return h("span", {class: `tag ${kind}`, text}); }
function mean(list) { const a = list.filter(x => x !== null && x !== undefined); return a.length ? a.reduce((s, x) => s + x, 0) / a.length : null; }

function renderPanel() {
  if (!wm.panel) return;
  const w = S.status.worldmodel, robots = (w && w.robots) || [];
  if (!S.ready) { wm.panel.body.replaceChildren(h("p", {class: "hint", text: "Waiting for the sim…"})); return; }
  if (!robots.length) {
    wm.panel.body.replaceChildren(h("p", {class: "hint", text: "No brain is publishing a world model. This rcap checkout may not have one (use the fused-world-model branch), or world_model.enabled is off."}));
    return;
  }
  const rows = [];
  for (const team of ["home", "away"]) {
    const t = robots.filter(r => r.team === team);
    if (!t.length) continue;
    const bytes = mean(t.map(r => r.packet && r.packet.bytes));
    rows.push(...row(team === "home" ? "Home team" : "Away team",
      tg(`ball error: own camera ${fmt(mean(t.map(r => r.truth.local_ball_avg)), 2)} m → fused ${fmt(mean(t.map(r => r.truth.fused_ball_avg)), 2)} m`, "info"),
      tg(`${fmt(bytes, 0)} B / packet`, bytes > 400 ? "warn" : ""),
      tg(`${t.reduce((s, r) => s + (r.truth.teammates_as_opponents || 0), 0)} teammates tracked as opponents`)));
  }
  const r = robots.find(x => x.index === wm.view);
  if (r) {
    const t = r.truth, fb = r.fused_ball, lb = r.local_ball, p = r.packet || {}, c = r.counters || {};
    rows.push(...row(`Robot ${r.label}`, tg(r.method === "ci" ? "fusing" : r.method, r.enabled ? "good" : "bad"), r.stale ? tg("snapshot stale", "bad") : null,
      tg(`pose ${fmt(r.pose.x)}, ${fmt(r.pose.y)}`)));
    rows.push(...row("Fused ball", fb.valid ? [tg(`${fmt(fb.x)}, ${fmt(fb.y)}`), tg(`v ${fmt(Math.hypot(fb.vx, fb.vy))} m/s`), tg(`±${fmt(fb.smaj)} m`),
      tg(`error ${fmt(t.fused_ball_err)} m (avg ${fmt(t.fused_ball_avg)})`, "info")] : tg("unknown")));
    rows.push(...row("Own camera ball", lb.valid ? [tg(`${fmt(lb.x)}, ${fmt(lb.y)}`), tg(`error ${fmt(t.local_ball_err)} m (avg ${fmt(t.local_ball_avg)})`),
      tg(`unseen ${fmt(lb.unseen, 1)} s`)] : tg("not seen")));
    if (fb.valid) {
      rows.push(...row("Weights", (fb.sources || []).map(s => tg(`${s.player === 0 ? "self" : "P" + s.player} ${s.rejected ? "rejected" : fmt(s.w, 2)} · ${fmt(s.age, 1)}s old · ±${fmt(s.sigma)}`, s.rejected ? "bad" : (s.player === 0 ? "good" : "")))));
    }
    rows.push(...row("Teammates", (r.teammates || []).length ? r.teammates.map(m => tg(`P${m.player} ±${fmt(m.sigma)} m · packet ${fmt(m.rx_age, 1)} s ago`)) : tg("none heard")));
    rows.push(...row("Opponents", tg(`${(r.opponents_local || []).length} seen · ${(r.opponents_fused || []).length} fused`),
      tg(`error ${fmt(t.opp_fused_err)} m`), t.teammates_as_opponents ? tg(`${t.teammates_as_opponents} teammate(s) tracked as opponents`, "warn") : null,
      tg(`gate: ${c.robots_as_teammate ?? 0} teammate · ${c.robots_as_opponent ?? 0} opponent`)));
    const blocks = Object.entries(p.blocks || {}).map(([k, v]) => `${k} ${v || "–"}`).join(" · ");
    rows.push(...row("Last packet", tg(`${p.bytes ?? "–"} B of ${p.max ?? 512}`), tg(`fixed ${p.fixed ?? "–"}`), blocks ? tg(blocks) : null, tg(`${p.sent ?? 0} sent`)));
    rows.push(...row("Ball filter", tg(`${c.ball_accepted ?? 0} accepted`), tg(`${c.ball_rejected ?? 0} rejected`), tg(`${c.ball_reinit ?? 0} restarts`), tg(`${c.opponents_suppressed ?? 0} tracks not re-sent`)));
  }
  wm.panel.body.replaceChildren(h("div", {class: "grid"}, rows));
}

function buildPanel() {
  const side = document.querySelector(".col-side");
  if (!side || wm.panel) return;
  const views = h("div", {class: "picks"}), toggles = h("div", {class: "wm-toggles"});
  const legend = h("div", {class: "wm-legend"},
    h("span", {}, h("i", {class: "k fused"}), "fused ball"), h("span", {}, h("i", {class: "k local"}), "own camera"),
    h("span", {}, h("i", {class: "k src"}), "contribution"), h("span", {}, h("i", {class: "k tm"}), "predicted teammate"),
    h("span", {}, h("i", {class: "k opp"}), "opponent track"));
  const body = h("div", {id: "wm-body"});
  const card = h("div", {class: "card", id: "wm-card"}, h("h2", {text: "World model"}), views, toggles, legend, body);
  side.insertBefore(card, side.children[1] || null);
  wm.panel = {card, body};
  wm.layers_el = {views, toggles};
  renderControls();
}

ui.hooks.mapBuilt = () => { buildPanel(); };
ui.hooks.status = () => {
  buildPanel();
  const robots = (S.status.worldmodel && S.status.worldmodel.robots) || [];
  const key = robots.map(r => r.index + r.label).join();
  if (key !== wm.key) { wm.key = key; renderControls(); }
  if (typeof wm.view === "number" && !robots.some(r => r.index === wm.view) && robots.length) wm.view = "all";
  redraw();
  renderPanel();
};
window.addEventListener("load", () => { buildPanel(); renderPanel(); });
})();
