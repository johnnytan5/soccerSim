"use strict";
/* Soccer Sim Control Centre: launcher + controller front-end (no build step, no dependencies).
   All untrusted text (branch names, log lines, commit subjects) is inserted with textContent. */
(() => {
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "text") el.textContent = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid !== null && kid !== undefined && kid !== false) el.append(kid);
  return el;
}
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const fmt = (v, d = 2) => (typeof v === "number" && isFinite(v) ? v.toFixed(d) : "–");

function toast(msg, kind = "") {
  const t = h("div", {class: `toast ${kind}`, role: "status", text: msg});
  $("#toasts").append(t);
  setTimeout(() => t.remove(), kind === "bad" ? 7000 : 3500);
}
async function api(path, body) {
  try {
    const r = await fetch(path, body === undefined ? {} : {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
    let data = null;
    try { data = await r.json(); } catch (e) { /* non-JSON error page */ }
    return {ok: r.ok, status: r.status, data: data || {}};
  } catch (e) {
    return {ok: false, status: 0, data: {error: "launcher not reachable"}};
  }
}

/* ------------------------------------------------------------------ state */
const DEFAULTS = {role: "goal_keeper", player_id: 1, tree: "game.xml", match_mode: "quick", viewer: true, ball_detector: "yolo",
  spawn_x: null, spawn_y: null, spawn_yaw: null, ball_x: null, ball_y: null, referee_schedule: "", rcap_root: "",
  sim_vy_max: null, camera_gl: "auto", mujoco_gl: "", ros_domain_id: null, extra_bridge_args: "", extra_launch_args: "",
  players: [], half_minutes: 10, continuous: false, detailed_bodies: false, graphics: "gpu"};
const S = {
  cfg: {...DEFAULTS}, parseErrors: {}, srv: null, static: null, check: null, checkSeq: 0, checkTimer: null,
  ready: false, status: {}, lastShot: "", cur: "", autoSwitched: false, lastState: "idle", sel: 0, lastRoster: null,
};
const ACTIVE = new Set(["building", "starting", "running", "tool", "stopping"]);
const controls = {};

/* ------------------------------------------------------------ form schema */
const isMatch = c => (c.players || []).length > 0;
const robotCount = c => (c.players || []).length;
const FIELDS = [
  {group: "What to simulate", items: [
    {k: "_mode", label: "Mode", type: "mode", help: c => isMatch(c)
      ? "A match: several robots, each running its own brain, with a referee, score and clock."
      : "One robot alone on the pitch; shots come from the Control tab."},
  ]},
  {group: "Players", show: isMatch, items: [{k: "players", type: "roster", wide: true}]},
  {group: "Robot", show: c => !isMatch(c), items: [
    {k: "role", label: "Role", type: "select", opts: ["goal_keeper", "striker", "defender"], help: () => "Player 1 is always the keeper; use player 2-5 for a striker or defender."},
    {k: "player_id", label: "Player id", type: "select", opts: [1, 2, 3, 4, 5], num: true},
  ]},
  {group: "Match", items: [
    {k: "tree", label: "Behaviour tree", type: "select", opts: ["game.xml", "chase.xml", "demo.xml"]},
    {k: "match_mode", label: "Start", type: "seg", opts: [["quick", "Quick"], ["full", "Full procedure"]], help: c => isMatch(c)
      ? "Quick: robots wait on the sideline and the referee runs INITIAL→READY→SET→PLAYING by itself. Full: robots stand up and calibrate on their own, then you press READY in the Control tab."
      : "Quick: robot already standing, scripted referee INITIAL→READY→SET→PLAYING. Full: you stand it up and run the referee from the Control tab."},
    {k: "_length", label: "Length", type: "length", show: isMatch, help: c => c.continuous
      ? "No halves and no clock limit: play on until you press Finish."
      : `Two halves of ${c.half_minutes} min with a side swap at half-time (real rules: 10).`},
    {k: "viewer", label: "MuJoCo viewer window", type: "switch", help: () => "Needs a display (WSLg works)."},
    {k: "graphics", label: "Viewer graphics", type: "seg", opts: [["gpu", "GPU"], ["cpu", "CPU"]], show: c => c.viewer, help: c => c.graphics === "cpu"
      ? "Software rendering on the CPU (Mesa llvmpipe). Slow with many robots; only for when the GPU driver misbehaves. macOS ignores this."
      : "The viewer window is drawn by your graphics card (the default). macOS always uses the system's graphics."},
    {k: "detailed_bodies", label: "Detailed robot bodies", type: "switch", show: c => isMatch(c) && c.viewer, help: c => c.detailed_bodies
      ? `Real T2 meshes, about 1 million triangles each${robotCount(c) > 2 ? `: ${robotCount(c)} robots will make the viewer lag` : ""}. Light parts are tinted by team.`
      : "Off: simple bodies, home blue and away orange, no shadows. Keeps the viewer smooth with many robots."},
  ]},
  {group: "Perception", items: [
    {k: "ball_detector", label: "Ball detector", type: "seg", opts: [["yolo", "YOLO"], ["ground_truth", "Ground truth"]], help: c =>
      c.ball_detector === "yolo" && robotCount(c) > 1
        ? `▲ YOLO shares about ${robotCount(c) <= 5 ? 25 : 12.5} camera frames a second between all robots (${((robotCount(c) <= 5 ? 25 : 12.5) / robotCount(c)).toFixed(1)} each). Ground truth gives every robot full-rate perception.`
        : "Ground truth gives the brain a perfect ball: debug behaviour, not vision."},
  ]},
  {group: "Field & referee (blank = default)", items: [
    {k: ["spawn_x", "spawn_y"], label: "Robot spawn x / y (m)", type: "pair", ph: ["-6.0", "-7.5"], show: c => !isMatch(c)},
    {k: "spawn_yaw", label: "Robot spawn yaw (rad)", type: "num", ph: "1.571", show: c => !isMatch(c)},
    {k: ["ball_x", "ball_y"], label: "Ball x / y (m)", type: "pair", ph: ["-6.0", "-4.0"], phMatch: ["0.0", "0.0"]},
    {k: "referee_schedule", label: "Referee schedule", type: "text", ph: "0:3,1:20,2:3,3:0", show: c => !isMatch(c), help: () => "state:seconds pairs, the last is held. 0 INITIAL · 1 READY · 2 SET · 3 PLAYING · 4 FINISHED."},
  ]},
  {group: "Advanced", items: [
    {k: "rcap_root", label: "rcap clone", type: "text", ph: "auto-detect", help: () => "Blank: $ROBOTEDGE_ROOT, the sibling Robotedge-5v5-RCAP2026, or ~/Robotedge-5v5-RCAP2026."},
    {k: "sim_vy_max", label: "SIM_VY_MAX (m/s)", type: "num", ph: "0.8"},
    {k: "camera_gl", label: "Camera GL", type: "select", opts: ["auto", "software", "gpu"]},
    {k: "mujoco_gl", label: "MUJOCO_GL", type: "select", opts: ["", "egl", "glfw", "osmesa"], labels: {"": "(default)"}},
    {k: "ros_domain_id", label: "ROS_DOMAIN_ID", type: "num", int: true, ph: "default", help: c => isMatch(c)
      ? "Each robot gets its own ROS domain, counting up from this one (default 70)."
      : "Give each simultaneous sim its own domain: topics are global."},
    {k: "extra_bridge_args", label: "Extra bridge args", type: "text", ph: "-p team_id:=131"},
    {k: "extra_launch_args", label: "Extra launch args", type: "text", ph: "key:=value"},
  ]},
];
const uiRefs = [];   // {group el, items: [{el, f, help el}]}: visibility and help text follow the config

/* ---- roster editor: five fixed slots per team, so an invalid lineup cannot be built ---- */
const ENTRY = {1: [-9.0, "right"], 2: [-6.0, "left"], 3: [-6.0, "right"], 4: [-3.5, "left"], 5: [-3.5, "right"]};
const DEFAULT_ROLE = {1: "goal_keeper", 2: "striker", 3: "striker", 4: "defender", 5: "defender"};
const ROLE_LABEL = {goal_keeper: "Goalkeeper", striker: "Striker", defender: "Defender"};
const TEAM_LABEL = {home: "Home", away: "Away"};
const defaultRoster = () => [{team: "home", player_id: 1, role: "goal_keeper"}, {team: "home", player_id: 2, role: "striker"}, {team: "away", player_id: 2, role: "striker"}];
function setPlayers(list) {
  S.cfg.players = list;
  controls.players.set(list);
  S.cur = ""; syncDerived(); renderSidebar(); saveDraft(); scheduleCheck();
}
function setMode(match) {
  if (match === isMatch(S.cfg)) return;
  if (match) {
    S.cfg.players = S.lastRoster && S.lastRoster.length ? S.lastRoster : defaultRoster();
    if (S.cfg.ball_detector === "yolo") S.cfg.ball_detector = "ground_truth";  // many cameras + YOLO cannot keep up
    controls.ball_detector.set(S.cfg.ball_detector);
  } else {
    S.lastRoster = S.cfg.players;
    S.cfg.players = [];
  }
  controls.players.set(S.cfg.players);
  S.cur = ""; syncDerived(); renderSidebar(); saveDraft(); scheduleCheck();
}
function buildRoster() {
  const root = h("div", {class: "roster", id: "roster-card"});
  const render = () => {
    const players = S.cfg.players || [];
    const cols = ["home", "away"].map(team => {
      const mine = Object.fromEntries(players.filter(p => p.team === team).map(p => [p.player_id, p]));
      const others = players.filter(p => p.team !== team);
      const count = Object.keys(mine).length;
      const slots = [1, 2, 3, 4, 5].map(id => {
        const p = mine[id];
        if (!p) {
          return h("button", {class: "slot empty", type: "button", "aria-label": `Add ${TEAM_LABEL[team]} player ${id}`,
            onclick: () => setPlayers([...players, {team, player_id: id, role: DEFAULT_ROLE[id]}])},
            h("span", {class: "num", text: id}), h("span", {text: id === 1 ? "add goalkeeper" : "add player"}));
        }
        const role = id === 1
          ? h("span", {class: "role-fixed", text: "Goalkeeper"})
          : h("select", {"aria-label": `${TEAM_LABEL[team]} player ${id} role`, onchange: e => setPlayers(players.map(q => q === p ? {...q, role: e.target.value} : q))},
              ["striker", "defender"].map(r => h("option", {value: r, text: ROLE_LABEL[r], selected: p.role === r ? true : null})));
        return h("div", {class: "slot filled"}, h("span", {class: "num", text: id}), role,
          h("button", {class: "x", type: "button", title: "Remove this player", "aria-label": `Remove ${TEAM_LABEL[team]} player ${id}`, onclick: () => setPlayers(players.filter(q => q !== p)), text: "×"}));
      });
      return h("div", {class: `team ${team}`},
        h("div", {class: "team-head"}, h("strong", {text: `${TEAM_LABEL[team]} · ${count}`}),
          h("span", {class: "spacer"}),
          h("button", {class: "btn small", type: "button", disabled: count === 5 ? "true" : null, text: "Fill 5",
            onclick: () => setPlayers([...others, ...[1, 2, 3, 4, 5].map(id => mine[id] || {team, player_id: id, role: DEFAULT_ROLE[id]})])}),
          h("button", {class: "btn small", type: "button", disabled: count === 0 ? "true" : null, text: "Clear", onclick: () => setPlayers(others)})),
        slots);
    });
    const home = players.filter(p => p.team === "home").length, away = players.length - home;
    const err = !home ? "Add at least one home player." : "";
    root.replaceChildren(
      h("div", {class: "teams"}, cols),
      h("div", {class: "roster-foot"},
        h("span", {class: "tag info", text: `${players.length} robot${players.length === 1 ? "" : "s"}`}),
        h("span", {class: "hint", text: `${home} home · ${away} away. Player 1 is always the goalkeeper; add anyone else as a striker or defender.`}),
        err ? h("span", {class: "tag bad", text: err}) : null),
      miniPitch(players));
  };
  controls.players = {set: render};
  return root;
}
/* start spots seen from above: home on the left half, away mirrored on the right */
function miniPitch(players) {
  const svg = svgEl("svg", {viewBox: "-12.2 -8.3 24.4 16.6", class: "mini", role: "img", "aria-label": "Where each robot starts"});
  const g = svgEl("g", {transform: "scale(1,-1)"});
  g.append(svgEl("rect", {class: "line", x: -11, y: -7.06, width: 22, height: 14.12}), svgEl("path", {class: "line", d: "M0 -7.06V7.06"}), svgEl("circle", {class: "line", cx: 0, cy: 0, r: 1.99}));
  for (const p of players) {
    const e = ENTRY[p.player_id];
    if (!e) continue;
    const sgn = p.team === "home" ? 1 : -1;
    const x = e[0] * sgn, y = (e[1] === "right" ? -7.5 : 7.5) * sgn;
    g.append(svgEl("circle", {class: `dot ${p.team}`, cx: x, cy: y, r: .62}));
    g.append(svgEl("text", {x, y: -y, "text-anchor": "middle", "dominant-baseline": "central", transform: "scale(1,-1)"}, String(p.player_id)));
  }
  svg.append(g);
  return svg;
}
function focusRoster() {
  const el = $("#roster-card");
  if (!el) return;
  el.scrollIntoView({block: "center", behavior: "smooth"});
  el.classList.remove("flash"); void el.offsetWidth; el.classList.add("flash");
}

function numInput(key, ph, isInt) {
  const el = h("input", {id: "f-" + key, type: "text", inputmode: isInt ? "numeric" : "decimal", placeholder: ph || "", "aria-label": key, autocomplete: "off"});
  el.addEventListener("input", () => {
    const raw = el.value.trim();
    let v = null, bad = false;
    if (raw !== "") {
      v = Number(raw);
      bad = !isFinite(v) || (isInt && !Number.isInteger(v));
    }
    el.classList.toggle("invalid", bad);
    if (bad) S.parseErrors[key] = `${key}: "${raw}" is not a ${isInt ? "whole " : ""}number`; else delete S.parseErrors[key];
    setCfg(key, bad ? S.cfg[key] : v);
  });
  controls[key] = {set: v => { el.value = v === null || v === undefined ? "" : String(v); el.classList.remove("invalid"); delete S.parseErrors[key]; }};
  return el;
}

function buildForm() {
  const form = $("#form");
  form.replaceChildren();
  uiRefs.length = 0;
  for (const g of FIELDS) {
    const fs = h("fieldset", {}, h("legend", {text: g.group}));
    const ref = {g, el: fs, items: []};
    for (const f of g.items) {
      const id = "f-" + (Array.isArray(f.k) ? f.k[0] : f.k);
      let input;
      if (f.type === "select") {
        const sel = h("select", {id}, f.opts.map(o => h("option", {value: String(o), text: (f.labels && f.labels[o]) ?? String(o)})));
        sel.addEventListener("change", () => setCfg(f.k, f.num ? Number(sel.value) : sel.value));
        controls[f.k] = {set: v => { sel.value = String(v); }};
        input = sel;
      } else if (f.type === "seg") {
        input = h("div", {class: "seg", role: "radiogroup", "aria-label": f.label}, f.opts.map(([v, t]) =>
          h("label", {}, h("input", {type: "radio", name: id, value: v, onchange: () => setCfg(f.k, v)}), h("span", {text: t}))));
        controls[f.k] = {set: v => { $$("input", input).forEach(r => { r.checked = r.value === v; }); }};
      } else if (f.type === "switch") {
        const cb = h("input", {type: "checkbox", id, onchange: () => setCfg(f.k, cb.checked)});
        const word = h("span", {class: "switch-label"});
        const sync = () => { word.textContent = cb.checked ? "on" : "off"; };
        cb.addEventListener("change", sync);
        input = h("label", {class: "switch"}, cb, h("span", {class: "track"}), word);
        controls[f.k] = {set: v => { cb.checked = !!v; sync(); }};
      } else if (f.type === "mode") {
        input = h("div", {class: "seg", role: "radiogroup", "aria-label": "Mode"}, [["single", "Single robot"], ["match", "Match"]].map(([v, t]) =>
          h("label", {}, h("input", {type: "radio", name: "f-mode", value: v, onchange: () => setMode(v === "match")}), h("span", {text: t}))));
        controls._mode = {set: () => $$("input", input).forEach(r => { r.checked = (r.value === "match") === isMatch(S.cfg); })};
      } else if (f.type === "roster") {
        input = buildRoster();
      } else if (f.type === "length") {
        const seg = h("div", {class: "seg", role: "radiogroup", "aria-label": "Match length"}, [["halves", "Two halves"], ["continuous", "Continuous"]].map(([v, t]) =>
          h("label", {}, h("input", {type: "radio", name: "f-length", value: v, onchange: () => setCfg("continuous", v === "continuous")}), h("span", {text: t}))));
        const mins = numInput("half_minutes", "10", false);
        mins.classList.add("mins");
        const minsBox = h("span", {class: "mins-box"}, mins, h("span", {class: "hint", text: "min per half"}));
        input = h("div", {class: "length"}, seg, minsBox);
        controls.continuous = {set: v => { $$("input", seg).forEach(r => { r.checked = (r.value === "continuous") === !!v; }); minsBox.hidden = !!v; }};
      } else if (f.type === "pair") {
        input = h("div", {class: "pair"}, f.k.map((k, i) => numInput(k, f.ph[i], false)));
        f.phEls = $$("input", input);
      } else if (f.type === "num") {
        input = numInput(f.k, f.ph, !!f.int);
      } else {
        const el = h("input", {type: "text", id, placeholder: f.ph || "", autocomplete: "off", spellcheck: "false"});
        el.addEventListener("input", () => setCfg(f.k, el.value));
        controls[f.k] = {set: v => { el.value = v ?? ""; }};
        input = el;
      }
      const help = h("div", {class: "help"});
      const wrap = h("div", {class: "field" + (f.wide ? " wide" : "")}, f.label ? h("label", {for: id, text: f.label}) : null, input, help);
      fs.append(wrap);
      ref.items.push({f, el: wrap, help});
    }
    uiRefs.push(ref);
    form.append(fs);
  }
}
/* show only what applies to the current mode, and keep help texts and placeholders in step with the config */
function syncDerived() {
  for (const ref of uiRefs) {
    let any = false;
    for (const it of ref.items) {
      const on = !it.f.show || it.f.show(S.cfg);
      it.el.hidden = !on;
      any = any || on;
      const text = typeof it.f.help === "function" ? it.f.help(S.cfg) : (it.f.help || "");
      it.help.textContent = text;
      it.help.hidden = !text;
      if (it.f.phEls) it.f.phEls.forEach((el, i) => { el.placeholder = (isMatch(S.cfg) && it.f.phMatch ? it.f.phMatch : it.f.ph)[i]; });
    }
    ref.el.hidden = ref.g.show ? !ref.g.show(S.cfg) : !any;
  }
  controls._mode && controls._mode.set();
}

function setCfg(key, value, quiet) {
  S.cfg[key] = value;
  if (key === "role") {
    if (value === "goal_keeper") S.cfg.player_id = 1; else if (S.cfg.player_id === 1) S.cfg.player_id = 2;
    controls.player_id.set(S.cfg.player_id);
  } else if (key === "player_id") {
    if (value === 1) S.cfg.role = "goal_keeper"; else if (S.cfg.role === "goal_keeper") S.cfg.role = "striker";
    controls.role.set(S.cfg.role);
  }
  if (!quiet) { S.cur = ""; syncDerived(); renderSidebar(); saveDraft(); scheduleCheck(); }
}
function applyCfg(cfg, cur = "") {
  S.cfg = {...DEFAULTS, ...cfg, players: (cfg.players || []).map(p => ({...p}))};
  for (const k of Object.keys(S.cfg)) if (controls[k]) controls[k].set(S.cfg[k]);
  S.cur = cur;
  syncDerived(); renderSidebar(); saveDraft(); scheduleCheck(0);
}
function saveDraft() { try { localStorage.setItem("soccerLauncher.draft", JSON.stringify(S.cfg)); } catch (e) { /* private mode */ } }
function loadDraft() { try { return JSON.parse(localStorage.getItem("soccerLauncher.draft") || "null"); } catch (e) { return null; } }

/* --------------------------------------------------------- pre-flight etc */
function scheduleCheck(delay = 250) {
  clearTimeout(S.checkTimer);
  S.checkTimer = setTimeout(runCheck, delay);
}
async function runCheck() {
  const seq = ++S.checkSeq;
  const r = await api("/api/launcher/check", {config: S.cfg, force: $("#force").checked});
  if (seq !== S.checkSeq || !r.ok) return;
  S.check = r.data;
  renderChecks();
}
function renderChecks() {
  const list = $("#checks");
  list.replaceChildren();
  const icons = {ok: "✔", warn: "▲", fail: "✖"};
  const parse = Object.values(S.parseErrors).map(text => ({level: "fail", text}));
  const checks = [...parse, ...((S.check && S.check.checks) || [])];
  const seen = new Set();
  for (const c of checks) {
    if (seen.has(c.text)) continue;
    seen.add(c.text);
    const li = h("li", {class: c.level}, h("span", {class: "ic", text: icons[c.level] || "•"}), h("span", {}, c.text,
      c.fix ? [" ", h("button", {class: "btn small", text: "Fix", onclick: () => runTool(c.fix)})] : null));
    list.append(li);
  }
  const errs = (S.check && S.check.errors) || [];
  for (const e of errs) if (!seen.has(e)) { seen.add(e); list.append(h("li", {class: "fail"}, h("span", {class: "ic", text: "✖"}), h("span", {text: e}))); }
  $("#preview").textContent = (S.check && S.check.preview) || "(fix the errors above to see the command)";
  const stale = S.check && S.check.stale;
  $("#btn-launch").textContent = stale ? "▶ Build & Launch" : "▶ Launch";
}

/* -------------------------------------------------------------- sidebar */
function renderSidebar() {
  const st = S.static;
  if (!st) return;
  const idle = !S.srv || S.srv.state === "idle";
  const pl = $("#presets");
  const lone = st.presets.filter(p => !(p.config && p.config.players && p.config.players.length));
  const matchPresets = st.presets.filter(p => p.config && p.config.players && p.config.players.length);
  const card = p => {
    const i = st.presets.indexOf(p), n = p.config && p.config.players ? p.config.players.length : 0;
    const cur = S.cur === "preset:" + i || (p.custom && S.cur === "" && isMatch(S.cfg));
    return h("button", {
      class: "item" + (cur ? " cur" : ""), type: "button", "aria-disabled": p.available ? null : "true",
      title: p.available ? p.description : `${p.name}: ${p.blocked_reason}`,
      onclick: () => {
        if (!p.available) { toast(`${p.name}: ${p.blocked_reason}`, "warn"); return; }
        if (p.custom && isMatch(S.cfg)) { S.cur = "preset:" + i; renderSidebar(); focusRoster(); return; }
        applyCfg({...p.config, rcap_root: S.cfg.rcap_root}, "preset:" + i);
        if (p.custom) focusRoster();
      }},
      h("span", {class: "t"}, h("span", {text: p.name}),
        p.available ? (n ? h("span", {class: "count", text: p.custom ? "edit" : `${n} robots`}) : null) : h("span", {class: "soon", text: "soon"})),
      h("span", {class: "d", text: p.description}));
  };
  pl.replaceChildren(h("div", {class: "group-label", text: "One robot"}), ...lone.map(card),
    h("div", {class: "group-label", text: "Full match"}), ...matchPresets.map(card));
  const names = Object.keys(st.profiles);
  $("#profiles").replaceChildren(...(names.length ? names.map(n => h("div", {class: "item" + (S.cur === "profile:" + n ? " cur" : ""), role: "button", tabindex: "0",
      onclick: () => { applyCfg(st.profiles[n], "profile:" + n); $("#profile-name").value = n; },
      onkeydown: e => { if (e.key === "Enter") e.currentTarget.click(); }},
    h("span", {class: "t"}, h("span", {text: n}), h("button", {class: "x", type: "button", title: "Delete profile", "aria-label": "Delete profile " + n,
      onclick: async e => { e.stopPropagation(); if (confirm(`Delete profile "${n}"?`)) { const r = await api("/api/launcher/profile/delete", {name: n}); if (r.ok) { st.profiles = r.data.profiles; renderSidebar(); } } }, text: "×"})),
    h("span", {class: "d", text: (st.profiles[n].players || []).length ? `match · ${st.profiles[n].players.length} robots` : `${st.profiles[n].role} #${st.profiles[n].player_id} · ${st.profiles[n].match_mode}`})))
    : [h("div", {class: "hint", text: "None yet. Name the current form and press Save."})]));
  $("#tools").replaceChildren(...st.tools.map(t => h("button", {class: "item", type: "button", title: t.description, disabled: idle ? null : "true", onclick: () => runTool(t.key)},
    h("span", {class: "t", text: t.label}), h("span", {class: "d", text: t.description}))));
}
async function runTool(key) {
  const r = await api("/api/launcher/tool", {key, rcap_root: S.cfg.rcap_root});
  if (!r.ok) toast(r.data.error || "could not start the tool", "bad"); else openDrawer(true);
}
async function saveProfile() {
  const name = $("#profile-name").value.trim();
  if (!name) { $("#profile-name").focus(); toast("Type a profile name first", "warn"); return; }
  const r = await api("/api/launcher/profile", {name, config: S.cfg});
  if (r.ok) { S.static.profiles = r.data.profiles; S.cur = "profile:" + name; renderSidebar(); toast(`Saved profile "${name}"`, "good"); }
  else toast(r.data.error || "could not save", "bad");
}

/* --------------------------------------------------- top bar + rcap card */
function chip(text, kind = "", live = false, title = "") {
  return h("span", {class: `chip ${kind}${live ? " live" : ""}`, title}, h("span", {class: "dot"}), text);
}
function renderTop() {
  const s = S.srv;
  if (!s) return;
  const chips = [];
  const r = s.rcap;
  if (r) {
    if (!r.valid) chips.push(chip("rcap not found", "bad", false, r.root));
    else {
      chips.push(chip(`⎇ ${r.branch || "detached"} @ ${r.sha}`, "", false, `${r.root}\n${r.subject}`));
      if (r.dirty) chips.push(chip("local changes", "warn", false, "Uncommitted changes under src/ are what gets built"));
    }
  }
  if (s.build) chips.push(s.build.stale ? chip("build stale", "warn", false, s.build.reason) : chip("build up to date", "good"));
  const sim = {
    idle: ["Idle", ""], building: ["Building rcap…", "warn", true], starting: ["Starting sim…", "warn", true],
    running: S.ready ? ["Sim running", "good", true] : ["Sim starting…", "warn", true],
    tool: [`Running ${s.label}…`, "warn", true], stopping: ["Stopping…", "bad", true],
  }[s.state] || [s.state, ""];
  chips.push(chip(sim[0], sim[1], !!sim[2]));
  $("#chips").replaceChildren(...chips);
  const active = ACTIVE.has(s.state);
  $("#btn-launch").disabled = s.state !== "idle";
  $("#btn-stop").disabled = !active || s.state === "stopping";
  $("#control-dot").classList.toggle("on", S.ready);
  renderRcapCard();
  renderOverlay();
  renderSidebar();
}
function renderRcapCard() {
  const r = S.srv && S.srv.rcap, b = S.srv && S.srv.build;
  const box = $("#rcap");
  if (!r) { box.textContent = "detecting…"; return; }
  if (!r.valid) { box.replaceChildren(h("div", {class: "path", text: r.root}), h("p", {class: "hint", text: "No src/brain here. Clone Robotedge-5v5-RCAP2026 next to this repo or set the path under Advanced."})); return; }
  box.replaceChildren(
    h("div", {class: "branch"}, chip(`⎇ ${r.branch || "detached HEAD"}`), chip("@ " + r.sha), r.dirty ? chip("+ local changes", "warn") : null),
    h("div", {class: "subj", text: r.subject}),
    h("div", {class: "path", text: r.root}),
    h("div", {class: "branch"}, b ? (b.stale ? chip("build stale", "warn") : chip("build up to date", "good")) : null,
      b && b.stale ? h("span", {class: "hint", text: b.reason}) : null),
    h("div", {class: "btn-row"}, h("button", {class: "btn small", type: "button", text: "Build now", disabled: S.srv.state === "idle" ? null : "true", onclick: () => runTool("build-sim")}),
      h("button", {class: "btn small", type: "button", text: "Re-check", onclick: refreshRcap})));
}
async function refreshRcap() {
  await api("/api/launcher/refresh", {rcap_root: S.cfg.rcap_root});
  scheduleCheck(0);
}
function renderOverlay() {
  const s = S.srv, up = S.ready;
  $("#no-sim").hidden = up;
  if (!up && s) {
    $("#no-sim-text").textContent = ["building", "starting", "running"].includes(s.state)
      ? "The sim is coming up; this page switches on as soon as the controller answers."
      : "Launch it from the Launch tab to drive, place the ball and robot, and fire shots.";
  }
}
function onState(snap) {
  const prev = S.lastState;
  S.srv = snap;
  S.lastState = snap.state;
  if (snap.state !== "running" && snap.state !== "stopping") { S.ready = false; S.status = {}; }
  if (snap.state === "idle") { S.autoSwitched = false; if (prev !== "idle" && currentTab() === "control") showTab("launch"); if (prev !== "idle") scheduleCheck(0); }
  renderTop();
}

/* ---------------------------------------------------------------- launch */
async function launch() {
  const bad = Object.values(S.parseErrors);
  if (bad.length) { toast(bad[0], "bad"); $("#checks").scrollIntoView({block: "center", behavior: "smooth"}); return; }
  const r = await api("/api/launcher/launch", {config: S.cfg, force: $("#force").checked});
  if (r.ok) { S.autoSwitched = false; openDrawer(true); toast(r.data.stale ? "Building rcap, then launching…" : "Launching…", "good"); }
  else if (r.status === 422) {
    S.check = r.data; renderChecks();
    const first = (r.data.errors || [])[0] || ((r.data.checks || []).find(c => c.level === "fail") || {}).text;
    toast(first || "Pre-flight failed", "bad");
    $("#checks").scrollIntoView({block: "center", behavior: "smooth"});
  } else toast(r.data.error || "launch failed", "bad");
}
async function stop() {
  const r = await api("/api/launcher/stop", {});
  if (!r.ok) toast(r.data.error || "stop failed", "bad");
}

/* -------------------------------------------------------------- log drawer */
const logEl = () => $("#log");
let logQueue = [], logRaf = 0, logCount = 0;
const ANSI = /\x1b\[[0-9;?]*[A-Za-z]/g;
function addLog(raw) {
  logQueue.push(String(raw).replace(ANSI, ""));
  if (!logRaf) logRaf = requestAnimationFrame(flushLog);
}
function flushLog() {
  logRaf = 0;
  const log = logEl(), frag = document.createDocumentFragment();
  for (const text of logQueue) {
    let cls = "l";
    if (/^\$ |^== /.test(text)) cls += " cmd"; else if (/error|fail|traceback|✖|fatal/i.test(text)) cls += " err"; else if (/warn/i.test(text)) cls += " warn";
    frag.append(h("span", {class: cls, text}));
    logCount++;
  }
  $("#last-line").textContent = logQueue[logQueue.length - 1] || "";
  logQueue = [];
  log.append(frag);
  while (logCount > 3000) { log.firstChild.remove(); logCount--; }
  if ($("#log-follow").checked) log.scrollTop = log.scrollHeight;
}
function openDrawer(open) {
  const d = $("#drawer");
  const v = open === undefined ? d.dataset.open !== "true" : open;
  d.dataset.open = String(v);
  $("#drawer-toggle").setAttribute("aria-expanded", String(v));
  $("#drawer-toggle").textContent = (v ? "▼" : "▲") + " Logs";
  document.body.style.paddingBottom = v ? "calc(30vh + 56px)" : "56px";
  if (v) logEl().scrollTop = logEl().scrollHeight;
}
function connectEvents() {
  const es = new EventSource("/api/launcher/events");
  es.addEventListener("log", e => addLog(JSON.parse(e.data)));
  es.addEventListener("state", e => onState(JSON.parse(e.data)));
  es.onopen = () => { $("#banner").hidden = true; };
  es.onerror = () => { $("#banner").hidden = false; $("#banner").textContent = "Lost connection to the launcher. Retrying… (if you stopped `pixi run launch`, start it again)"; };
}

/* ---------------------------------------------------------------- tabs */
const currentTab = () => ($("#tab-control").getAttribute("aria-selected") === "true" ? "control" : "launch");
function showTab(name) {
  for (const t of ["launch", "control"]) {
    $("#tab-" + t).setAttribute("aria-selected", String(t === name));
    $("#view-" + t).hidden = t !== name;
  }
  if (location.hash !== "#" + name) history.replaceState(null, "", "#" + name);
  if (name === "control") { drawSticks(); }
}

/* ============================================================ CONTROL TAB */
const joy = {lx: 0, ly: 0, rx: 0, ry: 0};
const held = {};
const BUTTONS = ["a", "b", "x", "y", "lb", "rb", "lt", "rt", "back", "start", "hat_u", "hat_d", "hat_l", "hat_r"];
const LATCHED = new Set(["lt", "rt"]);  // click to hold, click again to release (a mouse can only press one button)
const PAD_BUTTONS = {0: "a", 1: "b", 2: "x", 3: "y", 4: "lb", 5: "rb", 6: "lt", 7: "rt", 8: "back", 9: "start", 12: "hat_u", 13: "hat_d", 14: "hat_l", 15: "hat_r"};
const MODE_NAMES = {0: "damping", 1: "prepare", 2: "walking", 3: "custom", 4: "soccer"};
const STATES = ["INITIAL", "READY", "SET", "PLAYING", "FINISHED", "SCHEDULE"];

// In a match the stick, reset and placement requests go to the selected robot; a lone sim has no robot field.
const withRobot = o => (S.status.robots ? {...o, robot: S.sel} : o);
async function selectRobot(i) {
  S.sel = i;
  if (S.status.robots) { api("/api/select", {robot: i}); updateMarkers(); renderStatus(); renderPicks(); }
}
async function act(path, body, okMsg) {
  const r = await api(path, body);
  if (!r.ok) toast(r.status === 503 ? "The sim isn't running" : (r.data.error || `request failed (${r.status})`), "bad");
  else if (okMsg) toast(okMsg, "good");
  return r;
}

const stickDraws = [];
function stick(id, kx, ky) {
  const c = document.getElementById(id), g = c.getContext("2d"), R = c.width / 2;
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim() || "#4da3ff";
  function draw() {
    g.clearRect(0, 0, c.width, c.height);
    g.strokeStyle = css("--border-strong"); g.lineWidth = 1;
    g.beginPath(); g.moveTo(R, 8); g.lineTo(R, c.height - 8); g.moveTo(8, R); g.lineTo(c.width - 8, R); g.stroke();
    g.beginPath(); g.arc(R + joy[kx] * R * 0.8, R + joy[ky] * R * 0.8, 18, 0, 2 * Math.PI);
    g.fillStyle = css("--accent"); g.fill();
  }
  function move(e) {
    const b = c.getBoundingClientRect(), sc = c.width / b.width;
    let x = ((e.clientX - b.left) * sc - R) / (R * 0.8), y = ((e.clientY - b.top) * sc - R) / (R * 0.8);
    const n = Math.hypot(x, y);
    if (n > 1) { x /= n; y /= n; }
    joy[kx] = x; joy[ky] = y;  // screen-down is +y, matching RemoteControllerState (push front = -1)
    draw();
  }
  c.addEventListener("pointerdown", e => { c.setPointerCapture(e.pointerId); move(e); });
  c.addEventListener("pointermove", e => { if (c.hasPointerCapture(e.pointerId)) move(e); });
  // fires after pointerup *and* pointercancel, so a cancelled touch never leaves the stick pushed
  c.addEventListener("lostpointercapture", () => { joy[kx] = 0; joy[ky] = 0; draw(); });
  stickDraws.push(draw);
  draw();
}
const drawSticks = () => stickDraws.forEach(f => f());

function gamepadState() {
  const pads = navigator.getGamepads ? [...navigator.getGamepads()] : [];
  const pad = pads.find(p => p && p.connected);
  if (!pad) return null;
  const s = {lx: pad.axes[0] || 0, ly: pad.axes[1] || 0, rx: pad.axes[2] || 0, ry: pad.axes[3] || 0};
  for (const [i, name] of Object.entries(PAD_BUTTONS)) s[name] = !!(pad.buttons[i] && pad.buttons[i].pressed);
  return s;
}
// A worker's timer is not throttled in a background tab (page timers drop to 1 Hz, past the bridge's 0.5 s joystick timeout).
function ticker(fn, ms) {
  try {
    const url = URL.createObjectURL(new Blob([`setInterval(() => postMessage(0), ${ms})`], {type: "text/javascript"}));
    const w = new Worker(url);
    w.onmessage = fn;
  } catch (e) { setInterval(fn, ms); }
}
let joyBusy = false;
function sendJoystick() {
  if (!S.ready || joyBusy || !S.srv || S.srv.state !== "running") return;
  const state = withRobot({...joy});
  for (const name of BUTTONS) state[name] = !!held[name];
  const pad = gamepadState();
  if (pad) {
    for (const k of ["lx", "ly", "rx", "ry"]) if (Math.abs(pad[k]) > Math.abs(state[k])) state[k] = pad[k];
    for (const name of BUTTONS) state[name] = state[name] || pad[name];
    $("#pad").textContent = "physical gamepad connected";
  }
  joyBusy = true;
  fetch("/api/joystick", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(state)})
    .catch(() => {}).finally(() => { joyBusy = false; });
}

function buildControls() {
  stick("left", "lx", "ly"); stick("right", "rx", "ry");
  const box = $("#buttons");
  for (const name of BUTTONS) {
    const btn = h("button", {class: "btn", type: "button", text: name.toUpperCase()});
    if (LATCHED.has(name)) btn.onclick = () => { held[name] = !held[name]; btn.classList.toggle("on", held[name]); };
    else {
      btn.onpointerdown = () => { held[name] = true; btn.classList.add("on"); };
      btn.onpointerup = btn.onpointerleave = btn.onpointercancel = () => { held[name] = false; btn.classList.remove("on"); };
    }
    box.append(btn);
  }
  $$("[data-combo]").forEach(btn => btn.onclick = () => {
    const k = btn.dataset.combo;
    held.lt = true; held[k] = true;
    setTimeout(() => { held.lt = false; held[k] = false; }, 300);
  });
  for (const s of STATES) $("#ref-states").append(h("button", {class: "btn", type: "button", "data-state": s, text: s, onclick: () => act("/api/referee", {state: s})}));
  $("#apply").onclick = () => act("/api/referee", {set_play: $("#setplay").value, kicking: $("input[name=kicking]:checked").value, stopped: $("#stopped").checked}, "Set play applied");
  $("#fire-panel").onclick = () => act("/api/shoot", {});
  $("#reset-ball").onclick = () => placeBall(0, 0);
  $("#reset-robot").onclick = () => act("/api/reset_robot", withRobot({}));
  $("#match-actions").append(
    h("button", {class: "btn", type: "button", text: "Kickoff", title: "Ball to the centre spot, then READY → SET → PLAYING", onclick: () => act("/api/referee", {action: "kickoff"})}),
    h("button", {class: "btn", type: "button", id: "next-half", text: "Next half", title: "End this half now", onclick: () => act("/api/referee", {action: "next_half"})}),
    h("button", {class: "btn", type: "button", text: "Finish", title: "Full time", onclick: () => act("/api/referee", {action: "finish"})}));
  // Ball spots = the preset shot origins: within the keeper's ~8 m detection range, >= 2.5 m away so it still retreats.
  for (const [label, x, y] of BALL_SPOTS) $("#ball-spots").append(h("button", {class: "btn", type: "button", text: label, onclick: () => placeBall(x, y)}));
  for (const speed of [4, 6, 8]) {
    const row = h("div", {class: "btn-row"});
    for (const [label, x, y, ty] of PRESET_SHOTS) {
      const direction_deg = Math.atan2(ty - y, -11.0 - x) * 180 / Math.PI;
      row.append(h("button", {class: "btn", type: "button", text: label, onclick: () => act("/api/shoot", {x, y, speed, direction_deg, elevation_deg: 0})}));
    }
    $("#shot-presets").append(h("details", {open: speed === 6 ? true : null}, h("summary", {text: `${speed} m/s`}), row));
  }
  $("#shot-speed").oninput = e => { $("#out-speed").textContent = Number(e.target.value).toFixed(1); };
  $("#shot-elev").oninput = e => { $("#out-elev").textContent = e.target.value; };
}
// Preset shots at our goal (goal line x = -11, posts y = +-1.3): [label, origin x, origin y, target y on goal line]
const PRESET_SHOTS = [
  ["7 m centre → left corner", -4.0, 0.0, 1.0], ["7 m centre → right corner", -4.0, 0.0, -1.0], ["7 m centre → middle", -4.0, 0.0, 0.0],
  ["7 m left → far corner", -4.0, 2.0, -1.0], ["7 m right → far corner", -4.0, -2.0, 1.0],
  ["4 m box → left corner", -7.0, 0.0, 1.0], ["4 m box → right corner", -7.0, 0.0, -1.0],
  ["wide left → near post", -6.5, 3.5, 1.0], ["wide right → near post", -6.5, -3.5, -1.0],
];
const BALL_SPOTS = [["7 m centre", -4.0, 0.0], ["7 m left", -4.0, 2.0], ["7 m right", -4.0, -2.0], ["4 m box", -7.0, 0.0], ["wide left", -6.5, 3.5], ["wide right", -6.5, -3.5]];
const placeBall = (x, y) => act("/api/shoot", {x, y, speed: 0, direction_deg: 0, elevation_deg: 0});

/* ---- field map (field frame: x toward the opponent, our goal line at x = -11, y to the robot's left) ---- */
const NS = "http://www.w3.org/2000/svg";
const L = 11.0015, W = 7.063;   // half length / half width of FD_ROBOLEAGUE
const svgEl = (tag, attrs = {}, ...kids) => { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); e.append(...kids); return e; };
let fieldG, ballEl, robotEl, ghostEl, aimEl, botLayer, botEls = [], botKey = "";
function buildPitch() {
  const svg = $("#pitch");
  svg.replaceChildren();
  fieldG = svgEl("g", {transform: "scale(1,-1)"});
  const line = (tag, a) => svgEl(tag, {class: "line", ...a});
  fieldG.append(
    line("rect", {x: -L, y: -W, width: 2 * L, height: 2 * W}), line("path", {d: `M0 ${-W}V${W}`}),
    line("circle", {cx: 0, cy: 0, r: 1.99}), svgEl("circle", {class: "fill", cx: 0, cy: 0, r: .12}));
  for (const s of [-1, 1]) {
    fieldG.append(
      line("rect", {x: s < 0 ? -L : L - 5.221, y: -4.0605, width: 5.221, height: 8.121}),
      line("rect", {x: s < 0 ? -L : L - 2.307, y: -2.5415, width: 2.307, height: 5.083}),
      svgEl("circle", {class: "fill", cx: s * (L - 3.635), cy: 0, r: .12}),
      svgEl("rect", {class: "goal", x: s < 0 ? -L - .6 : L, y: -1.3, width: .6, height: 2.6}));
  }
  ghostEl = svgEl("g", {visibility: "hidden"});
  aimEl = svgEl("g", {visibility: "hidden"});
  ballEl = svgEl("circle", {class: "ball", r: .22, visibility: "hidden"});
  robotEl = svgEl("g", {visibility: "hidden"}, svgEl("circle", {class: "robot-body", r: .38}), svgEl("path", {class: "robot-body", d: "M.2 -.2L.75 0L.2 .2Z"}));
  botLayer = svgEl("g");
  fieldG.append(ghostEl, aimEl, ballEl, robotEl, botLayer);
  svg.append(fieldG,
    svgEl("text", {x: -L + .4, y: W - .35}, "our goal"), svgEl("text", {x: L - .4, y: W - .35, "text-anchor": "end"}, "opponent"));
}
function fieldPoint(e) {
  const svg = $("#pitch"), pt = svg.createSVGPoint();
  pt.x = e.clientX; pt.y = e.clientY;
  const p = pt.matrixTransform(fieldG.getScreenCTM().inverse());
  return {x: clamp(p.x, -L, L), y: clamp(p.y, -W, W)};
}
function updateMarkers() {
  const st = S.status;
  if (st.ball) { ballEl.setAttribute("cx", st.ball.x); ballEl.setAttribute("cy", st.ball.y); ballEl.setAttribute("visibility", "visible"); }
  if (st.robots && st.robots.length) {
    robotEl.setAttribute("visibility", "hidden");
    const key = st.robots.map(r => r.label).join();
    if (key !== botKey) {  // the lineup changed: rebuild the markers
      botKey = key;
      botLayer.replaceChildren();
      botEls = st.robots.map(r => {
        const body = svgEl("g", {}, svgEl("circle", {class: "robot-body", r: .38}), svgEl("path", {class: "robot-body", d: "M.2 -.2L.75 0L.2 .2Z"}));
        const label = svgEl("text", {class: "bot-label", "text-anchor": "middle", "dominant-baseline": "central", transform: "scale(1,-1)"}, String(r.player_id));
        const g = svgEl("g", {class: `bot ${r.team}`, "data-robot": r.index}, svgEl("circle", {class: "ring", r: .62}), body, label);
        botLayer.append(g);
        return {g, body};
      });
    }
    st.robots.forEach((r, i) => {
      const b = botEls[i];
      b.g.setAttribute("transform", `translate(${r.x} ${r.y})`);
      b.body.setAttribute("transform", `rotate(${r.yaw * 180 / Math.PI})`);
      b.g.classList.toggle("sel", r.index === S.sel);
      b.g.classList.toggle("down", !!r.fallen);
    });
  } else {
    botKey = ""; botLayer.replaceChildren(); botEls = [];
    if (st.robot) { robotEl.setAttribute("transform", `translate(${st.robot.x} ${st.robot.y}) rotate(${st.robot.yaw * 180 / Math.PI})`); robotEl.setAttribute("visibility", "visible"); }
  }
}
let picksKey = "";
function renderPicks() {
  const box = $("#robot-picks"), robots = S.status.robots || [];
  box.hidden = !robots.length;
  if (!robots.length) { picksKey = ""; return; }
  if (S.sel >= robots.length) S.sel = 0;
  const key = robots.map(r => r.label + r.role).join();
  if (key !== picksKey) {
    picksKey = key;
    box.replaceChildren(h("span", {class: "hint", text: "Drive / place:"}), ...robots.map(r => h("button", {
      class: `pick ${r.team}`, type: "button", title: `${r.team} player ${r.player_id}, ${r.role.replace("_", " ")}`,
      onclick: () => selectRobot(r.index)}, h("b", {text: r.label}), h("span", {text: r.role === "goal_keeper" ? "keeper" : r.role}))));
  }
  $$("button.pick", box).forEach((b, i) => { b.classList.toggle("on", i === S.sel); b.setAttribute("aria-pressed", String(i === S.sel)); });
}
const mapTool = () => $("input[name=maptool]:checked").value;
const HINTS = {
  ball: "Click anywhere to put the ball there, at rest.",
  robot: "Click to place the robot, drag to set the way it faces, release to apply.",
  shot: "Press where the ball should start (near the ball snaps to it), drag toward the target, release to fire. Speed and elevation come from the sliders.",
};
function setupMap() {
  const svg = $("#pitch");
  let drag = null;
  const hint = () => { $("#map-hint").textContent = HINTS[mapTool()]; $("#shot-controls").hidden = mapTool() !== "shot"; };
  $$("input[name=maptool]").forEach(r => r.addEventListener("change", hint));
  hint();
  function clear() { drag = null; ghostEl.setAttribute("visibility", "hidden"); aimEl.setAttribute("visibility", "hidden"); $("#map-readout").textContent = " "; }
  function draw(p) {
    const dx = p.x - drag.x, dy = p.y - drag.y, dist = Math.hypot(dx, dy);
    if (dist > 0.15) drag.ang = Math.atan2(dy, dx);
    const deg = drag.ang * 180 / Math.PI;
    ghostEl.replaceChildren(svgEl("circle", {class: "ghost", cx: drag.x, cy: drag.y, r: drag.tool === "robot" ? .38 : .22}));
    ghostEl.setAttribute("visibility", "visible");
    if (dist > 0.15) {
      const hx = drag.x + Math.cos(drag.ang) * dist, hy = drag.y + Math.sin(drag.ang) * dist, a = .5, b = .3;
      aimEl.replaceChildren(svgEl("path", {class: "aim", d: `M${drag.x} ${drag.y}L${hx} ${hy}`}),
        svgEl("path", {class: "aimhead", d: `M${hx} ${hy}L${hx - Math.cos(drag.ang - .45) * a} ${hy - Math.sin(drag.ang - .45) * a}L${hx - Math.cos(drag.ang + .45) * a} ${hy - Math.sin(drag.ang + .45) * a}Z`}));
      aimEl.setAttribute("visibility", "visible");
    }
    $("#map-readout").textContent = drag.tool === "robot"
      ? `robot at (${fmt(drag.x)}, ${fmt(drag.y)}) m, yaw ${fmt(deg, 0)}°`
      : `shot from (${fmt(drag.x)}, ${fmt(drag.y)}) m, direction ${fmt(deg, 0)}°, ${$("#shot-speed").value} m/s, elevation ${$("#shot-elev").value}°`;
  }
  svg.addEventListener("pointerdown", e => {
    if (!S.ready || e.button > 0) return;
    const bot = e.target.closest && e.target.closest("[data-robot]");
    if (bot) { selectRobot(Number(bot.getAttribute("data-robot"))); return; }  // clicking a robot selects it
    const p = fieldPoint(e), tool = mapTool();
    if (tool === "ball") { placeBall(p.x, p.y); $("#map-readout").textContent = `ball placed at (${fmt(p.x)}, ${fmt(p.y)}) m`; return; }
    svg.setPointerCapture(e.pointerId);
    let o = p;
    const b = S.status.ball;
    if (tool === "shot" && b && Math.hypot(b.x - p.x, b.y - p.y) < 0.8) o = {x: b.x, y: b.y};
    drag = {tool, x: o.x, y: o.y, ang: tool === "robot" ? (S.status.robot ? S.status.robot.yaw : 0) : Math.PI};
    draw(p);
  });
  svg.addEventListener("pointermove", e => { if (drag) draw(fieldPoint(e)); });
  svg.addEventListener("pointerup", async e => {
    if (!drag) return;
    const d = drag, p = fieldPoint(e), dist = Math.hypot(p.x - d.x, p.y - d.y);
    clear();
    if (d.tool === "robot") await act("/api/reset_robot", withRobot({x: d.x, y: d.y, yaw: d.ang}));
    else if (dist < 0.3) toast("Drag away from the start point to aim the shot", "warn");
    else await act("/api/shoot", {x: d.x, y: d.y, speed: Number($("#shot-speed").value), direction_deg: d.ang * 180 / Math.PI, elevation_deg: Number($("#shot-elev").value)});
  });
  svg.addEventListener("pointercancel", clear);
  window.addEventListener("keydown", e => { if (e.key === "Escape" && drag) clear(); });
}

/* ---- status polling + rendering ---- */
function tag(text, kind = "") { return h("span", {class: `tag ${kind}`, text}); }
function fmtClock(secs) { secs = Math.max(0, Math.floor(secs || 0)); return `${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, "0")}`; }
function renderStatus() {
  const s = S.status, box = $("#status");
  if (!S.ready) { box.replaceChildren(h("p", {class: "hint", text: "Waiting for the sim…"})); return; }
  const match = !!s.match;
  const sel = match && s.robots ? s.robots[Math.min(S.sel, s.robots.length - 1)] : null;
  const stKind = {PLAYING: "good", READY: "warn", SET: "warn", INITIAL: "info", FINISHED: "bad"}[s.state] || "";
  const kick = match ? `${s.kicking_team} kicks` : (s.kicking === "us" ? "we kick" : "opp. kicks");
  const rows = [];
  if (match) {
    rows.push(["Score", [h("span", {class: "score"}, h("span", {class: "home", text: `Home ${(s.score || {}).home ?? 0}`}), " – ", h("span", {class: "away", text: `${(s.score || {}).away ?? 0} Away`}))]],
      ["Clock", [tag(s.continuous ? `${fmtClock(s.play_secs)} played` : fmtClock(s.secs_remaining) + " left"), s.continuous ? null : tag(`half ${s.first_half ? 1 : 2}`),
        s.awaiting_start ? tag("press READY to start", "warn") : null]]);
  }
  rows.push(
    ["Referee", [tag(s.state || "–", stKind), s.set_play && s.set_play !== "NONE" ? tag(s.set_play.replace("_", " "), "warn") : null,
      tag(kick), s.stopped ? tag("stopped", "bad") : null, s.set_play_secs_left ? tag(`${s.set_play_secs_left}s`) : null]],
    [sel ? `Robot ${sel.label}` : "Robot", [tag(MODE_NAMES[s.robot_mode] || String(s.robot_mode), s.robot_mode === 4 ? "good" : "warn"), s.visual_kick ? tag("kicking", "info") : null,
      s.manual_drive ? tag("manual drive", "info") : null, sel && sel.fallen ? tag("fallen", "bad") : null]],
    ["Brain decision", [tag(s.decision || "–", "info")]]);
  if (!match) rows.push(["Shots", [tag(`goal ${(s.shots || {}).goal ?? 0}`, "good"), tag(`no goal ${(s.shots || {}).no_goal ?? 0}`), s.last_shot ? h("span", {class: "hint", text: `last: ${s.last_shot}`}) : null]]);
  rows.push(
    ["Robot pose", [h("span", {text: s.robot ? `x ${fmt(s.robot.x)}  y ${fmt(s.robot.y)}  yaw ${fmt(s.robot.yaw * 180 / Math.PI, 0)}°` : "–"})]],
    ["Ball", [h("span", {text: s.ball ? `x ${fmt(s.ball.x)}  y ${fmt(s.ball.y)}` : "–"})]],
    ["Sim rate", [tag(`${fmt(s.hz, 0)} Hz`, s.hz && s.hz < 45 ? "warn" : "")]]);
  if (match && s.event) rows.push(["Last event", [h("span", {text: s.event})]]);
  box.replaceChildren(h("div", {class: "grid"}, rows.flatMap(([k, v]) => [h("div", {class: "k", text: k}), h("div", {class: "v"}, v)])));
  $$("#ref-states [data-state]").forEach(b => b.classList.toggle("cur", b.dataset.state === s.state));
  $("#match-actions").hidden = !match;
  $("#next-half").disabled = !!s.continuous;
  $("#kick-opp").textContent = match ? "away kicks" : "opponent kicks";
  $("#kick-us").textContent = match ? "home kicks" : "we kick";
  $("#reset-robot").textContent = match ? "Robot to its start spot" : "Robot to goal centre";
  $("#field-title").textContent = match ? "Field · home attacks →" : "Field";
  renderPicks();
}
let polling = false;
async function pollStatus() {
  if (polling || !S.srv || !["running", "stopping"].includes(S.srv.state)) return;
  polling = true;
  try {
    const r = await fetch("/api/status");
    const s = r.ok ? await r.json() : {};
    const was = S.ready;
    S.ready = !!(s && s.state);
    S.status = S.ready ? s : {};
    if (S.ready && s.last_shot && s.last_shot !== S.lastShot) { if (S.lastShot !== "" || s.shots) toast(`Shot: ${s.last_shot}`, /^goal/i.test(s.last_shot) ? "bad" : "good"); }
    S.lastShot = s.last_shot || S.lastShot;
    if (S.ready) { updateMarkers(); renderStatus(); }
    if (S.ready !== was) {
      renderTop();
      if (S.ready && !S.autoSwitched) { S.autoSwitched = true; showTab("control"); openDrawer(false); toast("Sim is up: controls are live", "good"); }
    }
  } catch (e) { /* launcher restarting */ } finally { polling = false; }
}

/* ------------------------------------------------------------------ boot */
async function boot() {
  buildForm();
  buildControls();
  buildPitch();
  setupMap();
  const r = await api("/api/launcher/state");
  if (!r.ok) { $("#banner").hidden = false; $("#banner").textContent = "Cannot reach the launcher backend."; return; }
  S.static = r.data;
  S.srv = r.data;
  const draft = loadDraft();
  applyCfg(draft || r.data.last);
  if (r.data.load_warning) toast(r.data.load_warning, "warn");
  if (draft) toast("Restored your unsaved form edits", "");
  $("#btn-launch").onclick = launch;
  $("#btn-stop").onclick = stop;
  $("#btn-refresh").onclick = async () => { await refreshRcap(); toast("rcap status refreshed"); };
  $("#btn-save-profile").onclick = saveProfile;
  $("#profile-name").addEventListener("keydown", e => { if (e.key === "Enter") saveProfile(); });
  $("#force").onchange = () => scheduleCheck(0);
  $("#btn-copy").onclick = async () => { try { await navigator.clipboard.writeText($("#preview").textContent); toast("Command copied", "good"); } catch (e) { toast("Copy failed", "warn"); } };
  $("#btn-go-launch").onclick = () => showTab("launch");
  $("#tab-launch").onclick = () => showTab("launch");
  $("#tab-control").onclick = () => showTab("control");
  $("#drawer-toggle").onclick = () => openDrawer();
  $("#log-clear").onclick = () => { logEl().replaceChildren(); logCount = 0; $("#last-line").textContent = ""; };
  $("#log-copy").onclick = async () => { try { await navigator.clipboard.writeText(logEl().innerText); toast("Log copied", "good"); } catch (e) { toast("Copy failed", "warn"); } };
  onState(r.data);
  connectEvents();
  showTab(location.hash === "#control" ? "control" : "launch");
  ticker(sendJoystick, 50);
  setInterval(pollStatus, 250);
  if (["running", "starting", "building", "tool"].includes(r.data.state)) openDrawer(true);
}
boot();
})();
