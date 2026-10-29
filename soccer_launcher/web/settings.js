"use strict";
/* Settings form: comms / tracking knobs, simulation realism and the RCAP checkout's world-model parameters.
   The entries come from the launcher (soccer_launcher/settings.py + world_model.schema.json of the rcap clone), so a
   new parameter in the C++ parameter table shows up here without touching this file. A blank field means "do not
   override": the brain keeps config.yaml's value (the competition one). Hooks into app.js through window.SoccerUI. */
(() => {
const ui = window.SoccerUI;
if (!ui) return;
const {S, $, $$, h} = ui;

const state = {root: null, inputs: {}, open: new Set(["Team comms & ball", "World model: fusion"])};
const registry = () => (S.static && S.static.settings) || [];
const values = () => S.cfg.settings || (S.cfg.settings = {});
const numberOf = s => s.type === "int" || s.type === "float";

function setValue(key, value) {
  const v = values();
  if (value === null || value === undefined || value === "") delete v[key]; else v[key] = value;
  ui.edited();
}

function placeholder(s) {
  if (s.default === null || s.default === undefined) return "default";
  return typeof s.default === "boolean" ? (s.default ? "on" : "off") : String(s.default);
}

function rangeText(s) {
  if (s.min === null || s.min === undefined) return "";
  return `${s.min} to ${s.max}${s.unit ? " " + s.unit : ""}`;
}

function inputFor(s) {
  const id = "set-" + s.key.replace(/[^a-z0-9]+/gi, "-");
  if (s.type === "bool") {
    const sel = h("select", {id}, [["", `(default: ${placeholder(s)})`], ["true", "on"], ["false", "off"]].map(([v, t]) => h("option", {value: v, text: t})));
    sel.addEventListener("change", () => setValue(s.key, sel.value === "" ? null : sel.value === "true"));
    return {el: sel, set: v => { sel.value = v === undefined ? "" : String(v); }};
  }
  if (s.type === "str") {
    const sel = h("select", {id}, [["", `(default: ${placeholder(s)})`], ...s.choices.map(c => [c, c])].map(([v, t]) => h("option", {value: v, text: t})));
    sel.addEventListener("change", () => setValue(s.key, sel.value));
    return {el: sel, set: v => { sel.value = v === undefined ? "" : String(v); }};
  }
  const el = h("input", {id, type: "text", inputmode: "decimal", placeholder: placeholder(s), autocomplete: "off", title: rangeText(s)});
  el.addEventListener("input", () => {
    const raw = el.value.trim();
    if (raw === "") { el.classList.remove("invalid"); delete S.parseErrors[s.key]; return setValue(s.key, null); }
    const n = Number(raw);
    const bad = !isFinite(n) || (s.type === "int" && !Number.isInteger(n));
    el.classList.toggle("invalid", bad);
    if (bad) { S.parseErrors[s.key] = `${s.label}: "${raw}" is not a ${s.type === "int" ? "whole " : ""}number`; return; }
    delete S.parseErrors[s.key];
    setValue(s.key, n);
  });
  return {el, set: v => { el.value = v === undefined || v === null ? "" : String(v); el.classList.remove("invalid"); delete S.parseErrors[s.key]; }};
}

function build() {
  if (!state.root) return;
  const list = registry();
  state.inputs = {};
  const groups = new Map();
  for (const s of list) { if (!groups.has(s.group)) groups.set(s.group, []); groups.get(s.group).push(s); }
  const body = [];
  for (const [name, items] of groups) {
    const rows = items.map(s => {
      const inp = inputFor(s);
      state.inputs[s.key] = {...inp, s};
      const help = [s.help, rangeText(s) ? `Range ${rangeText(s)}.` : "", s.inert ? "Has no effect: the brain never reads it." : ""].filter(Boolean).join(" ");
      if (s.inert) inp.el.disabled = true;
      const row = h("div", {class: "field", "data-key": s.key}, h("label", {for: inp.el.id, text: s.label + (s.unit ? ` (${s.unit})` : "")}), inp.el, h("div", {class: "help", text: help}));
      state.inputs[s.key].row = row;
      return row;
    });
    const count = h("span", {class: "count"});
    const det = h("details", {class: "settings-group", "data-group": name}, h("summary", {}, h("span", {text: name}), count), ...rows);
    det.open = state.open.has(name);
    det.addEventListener("toggle", () => { if (det.open) state.open.add(name); else state.open.delete(name); });
    state.inputs["#" + name] = {det, count, keys: items.map(s => s.key)};
    body.push(det);
  }
  const rcap = S.srv && S.srv.rcap;
  const hasWm = list.some(s => s.source === "rcap");
  const note = h("p", {class: "hint"}, hasWm ? "" : `This rcap checkout${rcap && rcap.branch ? ` (${rcap.branch})` : ""} has no world_model schema, so no world-model settings are offered. Check out the fused-world-model branch.`);
  const tools = h("div", {class: "btn-row"},
    h("button", {class: "btn small", type: "button", text: "Competition values", title: "Replace the settings with the Competition preset's", onclick: () => {
      S.cfg.settings = {...((S.static.default_config || {}).settings || {})}; sync(); ui.edited(); }}),
    h("button", {class: "btn small", type: "button", text: "Clear all", title: "Do not override anything: config.yaml decides", onclick: () => {
      S.cfg.settings = {}; sync(); ui.edited(); }}));
  state.root.replaceChildren(h("legend", {text: "Settings (blank = config.yaml / competition value)"}), tools, note, ...body);
  sync();
}

function sync() {
  if (!state.root) return;
  const cfg = S.cfg, match = ui.isMatch(cfg), v = values();
  state.root.hidden = !match;
  for (const [key, inp] of Object.entries(state.inputs)) {
    if (key[0] === "#") continue;
    if (document.activeElement !== inp.el) inp.set(v[key]);  // never rewrite what is being typed
    inp.row.hidden = !!(inp.s.only_sim && inp.s.only_sim !== cfg.simulator);
    inp.row.classList.toggle("set", v[key] !== undefined);
  }
  for (const [key, g] of Object.entries(state.inputs)) {
    if (key[0] !== "#") continue;
    const visible = g.keys.some(k => !state.inputs[k].row.hidden);
    g.det.hidden = !visible;
    const set = g.keys.filter(k => v[k] !== undefined).length;
    g.count.textContent = set ? `${set} set` : "";
  }
}

ui.hooks.formBuilt = form => {
  state.root = h("fieldset", {class: "settings"});
  form.insertBefore(state.root, form.lastElementChild);  // above Advanced
};
ui.hooks.settingsChanged = build;
ui.hooks.cfg = sync;
ui.hooks.sync = sync;
})();
