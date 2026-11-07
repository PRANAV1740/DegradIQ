"use strict";
/* DEGRADIQ — APEX CONTROL. Vanilla client for the DegradIQ FastAPI backend (same origin).
   Integrity: on API failure the real error is shown with a retry; predictions are
   never fabricated or replaced with sample values. */

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"];
const COLORS = ["#4fd6dd", "#e8c07d", "#5aa8ff", "#4fd38a", "#ff6b6b", "#b79bff", "#f3d14e", "#7fd0ff"];
const state = {
  calendar: [], lineup: [], event: null, mode: "pre-race", predictions: [], raceLaps: 57,
  intel: { circuits: [], tyreEvents: [] }, circuit: null, tyre: null,
  tyreHidden: {}, focusDriver: null, manual: {}, manualOn: false,
};
const COMPOUND_HEX = { SOFT: "#ff5a6a", MEDIUM: "#f3d14e", HARD: "#e9eef7", INTERMEDIATE: "#4fd38a", WET: "#5aa8ff" };
const slugify = (n) => (n || "").toLowerCase().replace(/ grand prix/g, "").trim().replace(/\s+/g, "-").replace(/ã/g, "a").replace(/ç/g, "c");

async function api(path, opts) {
  const r = await fetch(path, opts);
  const ct = r.headers.get("content-type") || "";
  const body = ct.includes("json") ? await r.json() : await r.text();
  if (!r.ok) throw { status: r.status, body };
  return body;
}
function setPill(id, cls, text) { const el = $("#" + id); el.className = "pill state " + cls; el.innerHTML = '<i class="dot"></i>' + text; }

/* ---- signature intro (short, skippable, session-only) ---- */
function setupIntro() {
  const intro = $("#intro");
  let seen = false;
  try { seen = sessionStorage.getItem("apex_intro") === "1"; } catch (e) {}
  const dismiss = () => {
    intro.classList.add("done");
    document.body.dataset.intro = "done";
    try { sessionStorage.setItem("apex_intro", "1"); } catch (e) {}
  };
  $("#intro-skip").addEventListener("click", dismiss);
  if (seen || window.matchMedia("(prefers-reduced-motion: reduce)").matches) { dismiss(); return; }
  setTimeout(dismiss, 2600);
}

/* ---- clock (UTC) ---- */
function tick() {
  const d = new Date();
  $("#st-clock").textContent = d.toISOString().slice(11, 19) + " UTC";
}
setInterval(() => { if (!document.hidden) tick(); }, 1000); tick();

/* ---- nav rail active state ---- */
function setupNav() {
  const items = $$(".rail-item");
  const secs = items.map(a => $(a.getAttribute("href")));
  const obs = new IntersectionObserver(ents => {
    ents.forEach(e => {
      if (e.isIntersecting) {
        const id = "#" + e.target.id;
        items.forEach(a => a.classList.toggle("active", a.getAttribute("href") === id));
      }
    });
  }, { rootMargin: "-45% 0px -50% 0px" });
  secs.forEach(s => s && obs.observe(s));
}

async function boot() {
  setupIntro();
  try { await api("/health"); setPill("st-api", "ok", "API ONLINE"); }
  catch { setPill("st-api", "bad", "API OFFLINE"); }
  try {
    const s = await api("/strategy/first-stop/status");
    if (s.ready === false || s.status === "model_not_trained") { setPill("st-model", "bad", "MODEL NOT TRAINED"); renderModel(null); }
    else { setPill("st-model", "ok", "MODEL READY"); renderModel(s.provenance); }
  } catch { setPill("st-model", "bad", "MODEL ERROR"); renderModel(null); }
  setPill("st-live", "warn", "LIVE TIMING UNAVAILABLE");
  renderSys();
  await loadIntel();
  await Promise.all([loadCalendar(), loadLineup()]);
  renderSources(); bindUI(); setupNav(); syncOverview();
}

/* ---- model status (Model Room + overview card + strategy fields) ---- */
function renderModel(p) {
  const room = $("#integrity"), ov = $("#ov-model");
  if (!p) { room.innerHTML = ov.innerHTML = '<div class="empty">Model status unavailable.</div>'; return; }
  $("#foot-build").textContent = (p.architecture || "model") + " · " + (p.label_version_status || "");
  const sha = (p.model_sha256 || "").slice(0, 12);
  const sb = p.label_version_status === "current" ? "ok" : (p.label_version_status === "stale" ? "bad" : "warn");
  const full = [
    ["Architecture", `<span class="badge ok">${p.architecture || "—"}</span>`],
    ["Label version", p.label_definition_version || "—"],
    ["Label status", `<span class="badge ${sb}">${p.label_version_status || "—"}</span>`],
    ["Model SHA-256", sha ? sha + "…" : "—"],
    ["Training seasons", (p.training_seasons || []).join(", ") || "—"],
    ["Readiness", `<span class="badge ok">READY</span>`],
  ];
  room.innerHTML = full.map(([k, v]) => `<div class="line"><span class="k">${k}</span><span class="val">${v}</span></div>`).join("");
  ov.innerHTML = [["Arch", p.architecture], ["Labels", p.label_definition_version], ["Seasons", (p.training_seasons || []).join("–")], ["SHA-256", sha + "…"]]
    .map(([k, v]) => `<div class="line"><span class="k">${k}</span><span class="val">${v || "—"}</span></div>`).join("");
}
function renderSys() {
  const api = $("#st-api").className.includes("ok"), model = $("#st-model").className.includes("ok");
  $("#ov-sys").innerHTML = [
    [api ? "ok" : "bad", (api ? "API ONLINE" : "API OFFLINE")],
    [model ? "ok" : "bad", (model ? "MODEL READY · calibrated_hgb" : "MODEL UNAVAILABLE")],
    ["warn", "LIVE TIMING UNAVAILABLE"],
  ].map(([c, t]) => `<div class="row ${c}"><i class="dot"></i>${t}</div>`).join("");
}

/* ---- calendar ---- */
async function loadCalendar() {
  try {
    const cal = await api("/calendar");
    state.calendar = Array.isArray(cal) ? cal : [];
    $("#in-event").innerHTML = state.calendar.map((e, i) => `<option value="${i}">R${e.RoundNumber ?? i + 1} · ${e.EventName}</option>`).join("");
    if (state.calendar.length) selectEvent(0);
  } catch (e) { $("#in-event").innerHTML = `<option>calendar unavailable (${e.status || "err"})</option>`; }
}
function selectEvent(i) {
  const e = state.calendar[i]; if (!e) return;
  state.event = e;
  $("#st-event").textContent = "GP " + (e.EventName || "—");
  syncOverview();
  loadCircuit(e);
  loadTyre(e);
}

/* ---- shared intelligence (circuit + tyre availability) ---- */
async function loadIntel() {
  try {
    const idx = await api("/api/intel/index");
    state.intel.circuits = (idx.circuits && idx.circuits.available) || [];
    state.intel.eventMap = (idx.circuits && idx.circuits.event_to_circuit) || {};
    state.intel.events = (idx.coverage && idx.coverage.events) || [];
    state.intel.tyreEvents = (idx.tyre && idx.tyre.events) || [];
  } catch (e) { state.intel = { circuits: [], eventMap: {}, events: [], tyreEvents: [] }; }
}
// Explicit, configuration-aware mapping: an available circuit lists the exact EventNames
// it serves, so the 2026 Barcelona GP and the new Madrid "Spanish GP" never share geometry.
function findCircuit(name) {
  const n = (name || "").toLowerCase();
  return state.intel.circuits.find(c => (c.events || []).includes(name))
    || state.intel.circuits.find(c => c.circuit_id === state.intel.eventMap[name])
    || state.intel.circuits.find(c => (c.name || "").toLowerCase() === n);
}
function eventCoverage(name) { return (state.intel.events || []).find(e => e.event === name) || null; }
function findTyre(name) { const s = slugify(name), n = (name || "").toLowerCase(); return state.intel.tyreEvents.find(t => t.event_id === s || (t.event || "").toLowerCase() === n); }

/* ---- lineup / grid ---- */
async function loadLineup() {
  try { state.lineup = (await api("/strategy/first-stop/lineup")).drivers || []; renderGrid(); }
  catch (e) { $("#grid-body").innerHTML = `<tr><td colspan="9" class="empty">Lineup unavailable (${e.status || "err"})</td></tr>`; }
}
function renderGrid() {
  const comp = COMPOUNDS.map(c => `<option ${c === "MEDIUM" ? "selected" : ""}>${c}</option>`).join("");
  $("#grid-body").innerHTML = state.lineup.map((d, i) => `<tr data-drv="${d.driver}" data-team="${(d.team || "").toLowerCase()}">
    <td class="c-sel"><input type="checkbox" class="sel" ${i < 10 ? "checked" : ""}></td>
    <td><span class="drv">${d.driver}</span></td><td>${d.team || "—"}</td>
    <td class="c-grid"><input type="number" class="grid" min="1" max="22" value="${i + 1}"></td>
    <td><select class="scomp">${comp}</select></td>
    <td class="c-age"><input type="number" class="sage" min="1" max="60" value="1"></td>
    <td class="live-col" hidden><input type="number" class="clap" min="1" max="100" value="1"></td>
    <td class="live-col" hidden><select class="ccomp">${comp}</select></td>
    <td class="live-col" hidden><input type="number" class="cpos" min="1" max="22" value="${i + 1}"></td></tr>`).join("");
  $$("#grid-body .scomp").forEach(paint);
  $$("#grid-body .scomp").forEach(s => s.addEventListener("change", () => { paint(s); syncOverview(); }));
  $$("#grid-body .sel").forEach(c => c.addEventListener("change", () => { c.closest("tr").classList.toggle("sel", c.checked); syncOverview(); }));
  $$("#grid-body .grid").forEach(g => g.addEventListener("change", syncOverview));
  $$("#grid-body tr").forEach(tr => tr.classList.toggle("sel", $(".sel", tr).checked));
  syncOverview();
}
function paint(sel) { const td = sel.closest("td"); td.className = "comp " + sel.value.toLowerCase(); if (!td.querySelector("i")) td.insertAdjacentHTML("afterbegin", "<i></i>"); }

function selectedDrivers() {
  return $$("#grid-body tr").filter(tr => $(".sel", tr).checked).map(tr => {
    const d = { driver: tr.dataset.drv, grid_position: +$(".grid", tr).value, starting_compound: $(".scomp", tr).value, starting_tyre_age: +$(".sage", tr).value };
    if (state.mode === "live") { d.current_lap = +$(".clap", tr).value; d.current_compound = $(".ccomp", tr).value; d.current_position = +$(".cpos", tr).value; }
    return d;
  });
}

/* ---- overview sync (driver strip + scenario summary) ---- */
function syncOverview() {
  const e = state.event || {};
  $("#ov-round").textContent = "ROUND " + (e.RoundNumber ?? "—");
  $("#ov-event").textContent = e.EventName || "—";
  $("#ov-loc").textContent = (e.Location || e.Country || "—") + "  ·  " + ((e.EventDate || "").slice(0, 10) || "—");
  $("#ov-laps").textContent = $("#in-laps") ? $("#in-laps").value : "57";
  $("#ov-mode").textContent = state.mode === "live" ? "LIVE" : "PRE";
  const rows = $$("#grid-body tr");
  const sel = rows.filter(tr => $(".sel", tr).checked);
  if ($("#m-count")) $("#m-count").textContent = sel.length;
  $("#ov-field").textContent = sel.length;
  const strip = $("#strip");
  if (!rows.length) return;
  const chips = rows.map(tr => ({ code: tr.dataset.drv, team: $("td:nth-child(3)", tr).textContent, grid: +$(".grid", tr).value, comp: $(".scomp", tr).value, on: $(".sel", tr).checked }))
    .sort((a, b) => a.grid - b.grid);
  strip.innerHTML = chips.map(c => `<div class="chip" style="${c.on ? "" : "opacity:.45"}">
    <div class="top"><span class="pos">P${c.grid}</span><span class="cc ${c.comp.toLowerCase()}"></span></div>
    <span class="code">${c.code}</span><span class="team">${c.team}</span></div>`).join("");
}

/* ---- UI ---- */
function bindUI() {
  $("#in-event").addEventListener("change", e => selectEvent(+e.target.value));
  $("#in-laps").addEventListener("change", e => { e.target.value = Math.max(5, Math.min(100, +e.target.value || 57)); syncOverview(); });
  $("#in-mode").addEventListener("change", e => {
    state.mode = e.target.value; const live = state.mode === "live";
    $$(".live-col").forEach(c => c.hidden = !live); $("#live-hint").hidden = !live; syncOverview();
  });
  $("#in-search").addEventListener("input", e => {
    const q = e.target.value.trim().toLowerCase();
    $$("#grid-body tr").forEach(tr => tr.hidden = q && !(tr.dataset.drv.toLowerCase().includes(q) || tr.dataset.team.includes(q)));
  });
  $("#h-sel").addEventListener("change", e => { $$("#grid-body .sel").forEach(c => { c.checked = e.target.checked; c.closest("tr").classList.toggle("sel", c.checked); }); syncOverview(); });
  $("#b-all").addEventListener("click", () => { $$("#grid-body .sel").forEach(c => { c.checked = true; c.closest("tr").classList.add("sel"); }); syncOverview(); });
  $("#b-none").addEventListener("click", () => { $$("#grid-body .sel").forEach(c => { c.checked = false; c.closest("tr").classList.remove("sel"); }); syncOverview(); });
  $("#b-run").addEventListener("click", runForecast);
  $("#board-sort").addEventListener("change", renderBoard);
  // Circuit controls
  const toggle = (btn, fn) => btn.addEventListener("click", () => { btn.classList.toggle("active"); btn.setAttribute("aria-pressed", btn.classList.contains("active")); fn(); });
  toggle($("#circ-sectors"), () => state.circuit && renderCircuit());
  toggle($("#circ-corners"), () => state.circuit && renderCircuit());
  $("#circ-reset").addEventListener("click", resetCircuit);
  // Stint Canvas manual scenario
  $("#canvas-manual").addEventListener("change", e => {
    state.manualOn = e.target.checked;
    $("#manual-hint").hidden = !state.manualOn;
    $$(".manual-col").forEach(c => c.hidden = !state.manualOn);
    renderResults(); renderBoard();
  });
}

/* ---- forecast ---- */
async function runForecast() {
  const drivers = selectedDrivers(), rs = $("#run-state"), btn = $("#b-run");
  removeRetry();
  if (!drivers.length) { rs.className = "run-state err"; rs.textContent = "Select at least one driver."; return; }
  if (!state.event) { rs.className = "run-state err"; rs.textContent = "Select a Grand Prix."; return; }
  const race_laps = Math.max(5, Math.min(100, +$("#in-laps").value || 57));
  const payload = { event_name: state.event.EventName, race_laps, mode: state.mode, drivers };
  btn.disabled = true; rs.className = "run-state"; rs.innerHTML = '<span class="spinner"></span>Forecasting…';
  try {
    const res = await api("/strategy/first-stop/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    state.predictions = res.predictions || []; state.raceLaps = res.race_laps || race_laps;
    rs.className = "run-state ok"; rs.textContent = `${state.predictions.length} forecasts · ${res.model_type || "model"}`;
    renderResults(); renderBoard(); renderChart();
    document.getElementById("sec-board").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    // Surface the real error; never fabricate predictions. Keep the user's scenario inputs.
    let msg = `Forecast failed (${e.status || "network"})`;
    if (e.body && e.body.detail) msg += ": " + (typeof e.body.detail === "string" ? e.body.detail : "validation error");
    rs.className = "run-state err"; rs.textContent = msg;
    state.predictions = [];
    $("#res-wrap").hidden = true; $("#res-empty").hidden = false; $("#res-empty").textContent = msg;
    $("#board").innerHTML = '<div class="empty">No forecast.</div>';
    addRetry();
  } finally { btn.disabled = false; }
}
function addRetry() {
  if ($("#b-retry")) return;
  const b = document.createElement("button"); b.id = "b-retry"; b.className = "btn retry"; b.textContent = "Retry forecast";
  b.addEventListener("click", runForecast); $("#run-state").after(b);
}
function removeRetry() { const b = $("#b-retry"); if (b) b.remove(); }

function renderResults() {
  const ps = state.predictions.filter(p => p.status === "forecast");
  if (!ps.length) { $("#res-wrap").hidden = true; $("#res-empty").hidden = false; $("#res-empty").textContent = "No forecasts returned."; return; }
  $("#res-empty").hidden = true; $("#res-wrap").hidden = false;
  const man = state.manualOn;
  $("#res-body").innerHTML = ps.map(p => {
    const w = p.likely_window_if_stops || {}, prob = p.probability_stop_before_finish ?? 0;
    const foc = state.focusDriver === p.driver ? " class=\"focus\"" : "";
    const mv = state.manual[p.driver] != null ? state.manual[p.driver] : "";
    return `<tr${foc} data-drv="${p.driver}"><td class="mono">${p.predicted_order_by_expected_lap ?? "—"}</td><td><span class="drv">${p.driver}</span></td>
      <td>${p.team || "—"}</td><td class="prob-cell">${(prob * 100).toFixed(1)}%<div class="bar"><i style="width:${Math.round(prob * 100)}%"></i></div></td>
      <td class="mono">${p.most_likely_stop_lap ?? "—"}</td><td class="mono">${p.median_stop_lap_if_stops ?? "—"}</td>
      <td class="mono">${p.expected_stop_lap_if_stops ?? "—"}</td><td class="mono">${w.from_lap != null ? w.from_lap + "–" + w.to_lap : "—"}</td>
      <td class="manual-col" ${man ? "" : "hidden"}><input type="number" class="man-in" min="1" max="${state.raceLaps}" value="${mv}" placeholder="—" data-drv="${p.driver}"></td></tr>`;
  }).join("");
  $$("#res-body .drv").forEach(el => el.addEventListener("click", () => setFocusDriver(el.closest("tr").dataset.drv)));
  $$("#res-body .man-in").forEach(inp => inp.addEventListener("change", () => {
    const v = parseInt(inp.value, 10);
    if (Number.isFinite(v) && v >= 1 && v <= state.raceLaps) state.manual[inp.dataset.drv] = v; else delete state.manual[inp.dataset.drv];
    renderBoard();
  }));
}

function renderBoard() {
  const host = $("#board");
  let ps = state.predictions.filter(p => p.status === "forecast");
  if (!ps.length) { host.innerHTML = '<div class="empty">Run a forecast to populate the canvas.</div>'; return; }
  const sort = $("#board-sort").value;
  ps = ps.slice().sort((a, b) =>
    sort === "prob" ? (b.probability_stop_before_finish || 0) - (a.probability_stop_before_finish || 0)
      : sort === "driver" ? a.driver.localeCompare(b.driver)
        : (a.expected_stop_lap_if_stops || 1e9) - (b.expected_stop_lap_if_stops || 1e9));
  const L = state.raceLaps, pos = v => Math.max(0, Math.min(100, (v / L) * 100));
  // lap-axis gridlines behind the lanes
  const gridMarks = []; for (let lx = 0; lx <= L; lx += Math.max(5, Math.round(L / 8))) gridMarks.push(lx);
  const grid = `<div class="canvas-grid">${gridMarks.map(lx => `<i style="left:${pos(lx)}%"></i>`).join("")}</div>`;
  host.innerHTML = grid + ps.map(p => {
    const w = p.likely_window_if_stops || {}, pct = Math.round((p.probability_stop_before_finish || 0) * 100);
    const focus = state.focusDriver === p.driver ? " focus" : "";
    const man = state.manualOn && state.manual[p.driver] != null ? `<div class="man-mark" style="left:${pos(state.manual[p.driver])}%" title="Manual what-if: L${state.manual[p.driver]}">✚</div>` : "";
    if (w.from_lap == null) return `<div class="lane${focus}" data-drv="${p.driver}"><span class="who"><b>${p.driver}</b><span>${pct}%</span></span><div class="track">${man}</div><span class="pw">no window · stops ${pct}%</span></div>`;
    const left = pos(w.from_lap), width = Math.max(1.2, pos(w.to_lap) - pos(w.from_lap));
    const marks = [];
    if (p.most_likely_stop_lap != null) marks.push(`<div class="mk dot" style="left:${pos(p.most_likely_stop_lap)}%" title="most likely L${p.most_likely_stop_lap}"></div>`);
    if (p.median_stop_lap_if_stops != null) marks.push(`<div class="mk med" style="left:${pos(p.median_stop_lap_if_stops)}%" title="median L${p.median_stop_lap_if_stops}"></div>`);
    if (p.expected_stop_lap_if_stops != null) marks.push(`<div class="mk exp" style="left:${pos(p.expected_stop_lap_if_stops)}%" title="expected L${p.expected_stop_lap_if_stops}"></div>`);
    return `<div class="lane${focus}" data-drv="${p.driver}">
      <span class="who"><b>${p.driver}</b><span>${pct}%</span></span>
      <div class="track" title="${p.driver}: 10–90% window L${w.from_lap}–${w.to_lap} (conditional on stopping)">
        <div class="win" style="left:${left}%;width:${width}%;opacity:${0.25 + 0.6 * (p.probability_stop_before_finish || 0)}"></div>
        ${marks.join("")}${man}</div>
      <span class="pw">L${w.from_lap}–${w.to_lap} · <b>${p.expected_stop_lap_if_stops ?? "—"}</b></span></div>`;
  }).join("") + `<div class="scale"><span>driver · P(stop)</span><div class="ticks">${gridMarks.map(lx => `<span>${lx}</span>`).join("")}</div><span>exp. lap</span></div>`;
  $$("#board .lane").forEach(l => l.addEventListener("click", () => setFocusDriver(l.dataset.drv)));
}
function setFocusDriver(drv) {
  state.focusDriver = state.focusDriver === drv ? null : drv;
  renderBoard(); renderResults(); renderChart(); syncOverview();
}

function renderChart() {
  const svg = $("#chart"), W = 760, H = 320, pad = { l: 40, r: 14, t: 14, b: 26 };
  const ps = state.predictions.filter(p => p.status === "forecast" && (p.cumulative_probabilities || []).length).slice(0, 6);
  svg.innerHTML = ""; $("#prob-legend").innerHTML = "";
  if (!ps.length) { svg.innerHTML = `<text x="${W / 2}" y="${H / 2}" text-anchor="middle">Run a forecast</text>`; return; }
  const L = state.raceLaps, x = lap => pad.l + (lap / L) * (W - pad.l - pad.r), y = p => pad.t + (1 - p) * (H - pad.t - pad.b);
  for (let g = 0; g <= 1.0001; g += 0.25) svg.insertAdjacentHTML("beforeend", `<line class="grid" x1="${pad.l}" y1="${y(g)}" x2="${W - pad.r}" y2="${y(g)}"/><text x="6" y="${y(g) + 3}">${Math.round(g * 100)}%</text>`);
  for (let lx = 0; lx <= L; lx += Math.max(5, Math.round(L / 6))) svg.insertAdjacentHTML("beforeend", `<text x="${x(lx)}" y="${H - 8}" text-anchor="middle">${lx}</text>`);
  ps.forEach((p, i) => {
    const col = COLORS[i % COLORS.length];
    const pts = p.cumulative_probabilities.map(c => `${x(c.lap)},${y(c.probability_stopped_by_lap)}`).join(" L ");
    svg.insertAdjacentHTML("beforeend", `<path class="series" d="M ${pts}" stroke="${col}"/>`);
    $("#prob-legend").insertAdjacentHTML("beforeend", `<span class="it"><i style="background:${col}"></i>${p.driver}</span>`);
  });
  const tip = $("#chart-tip");
  svg.onmousemove = ev => {
    const r = svg.getBoundingClientRect();
    const lap = Math.round(((ev.clientX - r.left) / r.width * W - pad.l) / (W - pad.l - pad.r) * L);
    if (lap < 0 || lap > L) { tip.hidden = true; return; }
    const parts = ps.map(p => { const c = p.cumulative_probabilities.reduce((a, b) => Math.abs(b.lap - lap) < Math.abs(a.lap - lap) ? b : a); return `${p.driver} ${Math.round(c.probability_stopped_by_lap * 100)}%`; });
    tip.hidden = false; tip.style.left = (ev.clientX - r.left) + "px"; tip.style.top = (ev.clientY - r.top) + "px"; tip.innerHTML = `L${lap} · ` + parts.join(" · ");
  };
  svg.onmouseleave = () => { tip.hidden = true; };
}

/* ============================ CIRCUIT INTELLIGENCE ============================ */
const circView = { x: 0, y: 0, w: 1000, h: 1000 };
let circCornerIdx = -1;

async function loadCircuit(e) {
  const tag = $("#circ-avail");
  const hit = findCircuit(e.EventName);
  $("#circ-name").textContent = e.EventName || "—";
  if (!hit) { renderCircuitUnavailable(e); return; }
  tag.className = "tag"; tag.textContent = "LOADING…";
  try {
    const data = await api("/api/intel/circuit/" + encodeURIComponent(hit.circuit_id));
    if (data.status !== "ok" || !data.outline) { renderCircuitUnavailable(e); return; }
    state.circuit = data; renderCircuit();
  } catch (err) { renderCircuitUnavailable(e, err); }
}
function renderCircuitUnavailable(e, err) {
  state.circuit = null; circCornerIdx = -1;
  const tag = $("#circ-avail"), svg = $("#circ-svg"), empty = $("#circ-empty");
  setCircuitAttribution("");
  const cov = eventCoverage(e.EventName) || {};
  // Distinct unavailable states: data-acquisition failure vs genuinely no geometry.
  const failed = !!err || cov.status === "FAILED";
  tag.className = "tag warn"; tag.textContent = failed ? "DATA ACQUISITION FAILED" : "GEOMETRY UNAVAILABLE";
  svg.innerHTML = ""; empty.hidden = false;
  const verified = state.intel.circuits.filter(c => c.geometry_type === "fastlap_trace").map(c => c.name);
  const approx = state.intel.circuits.filter(c => c.geometry_type === "osm_centreline").map(c => c.name);
  const reason = failed
    ? `We could not acquire or validate geometry for this circuit (${esc((err && (err.status || "network")) || cov.reason || "error")}). Nothing is shown rather than a guessed shape.`
    : `No verified racing-line telemetry and no usable OpenStreetMap trace for this circuit${cov.reason ? " — " + esc(cov.reason) : ""}. We never substitute a different circuit's shape.`;
  empty.innerHTML = `<div class="empty unavail"><b>${failed ? "Geometry could not be loaded" : "Geometry unavailable"} for ${esc(e.EventName || "this circuit")}.</b>
    <span>${reason}</span>
    <span class="mono avail-note">Verified (racing-line + corners): ${verified.join(", ") || "none"}.<br>Approximate (OpenStreetMap outline): ${approx.length} circuit${approx.length === 1 ? "" : "s"}.</span></div>`;
  $("#circ-loc").textContent = "—"; $("#circ-facts").innerHTML = '<div class="empty">—</div>';
  $("#circ-corner").innerHTML = '<div class="empty">Pick an event with available geometry.</div>';
  $("#circ-prov").innerHTML = ""; $("#circ-legend").hidden = true; $("#circ-ctx-tag").textContent = "UNAVAILABLE";
}
// Project each marshal post onto its nearest outline point (distance-based) and return the
// sorted unique boundary indices. These delimit marshal sectors — a traceable FastF1 source —
// not the 3 official timing sectors.
function marshalBoundaries(d) {
  const ol = d.outline || [], ms = d.marshal_sectors || [];
  if (!ms.length || !ol.length) return [];
  const idx = ms.map(m => {
    let best = 0, bd = Infinity;
    for (let i = 0; i < ol.length; i++) { const dx = ol[i][0] - m.xy[0], dy = ol[i][1] - m.xy[1], dd = dx * dx + dy * dy; if (dd < bd) { bd = dd; best = i; } }
    return best;
  });
  return [...new Set(idx)].sort((a, b) => a - b);
}
function setCircuitAttribution(text) {
  const stage = $("#map-stage"); let el = $("#circ-attrib");
  if (!text) { if (el) el.remove(); return; }
  if (!el) { el = document.createElement("div"); el.id = "circ-attrib"; el.className = "map-attrib"; stage.appendChild(el); }
  el.textContent = text;
}
function renderCircuit() {
  const d = state.circuit, svg = $("#circ-svg"), empty = $("#circ-empty");
  const osm = d.geometry_type === "osm_centreline";
  empty.hidden = true; $("#circ-ctx-tag").textContent = osm ? "OSM OUTLINE" : "CIRCUIT";
  $("#circ-avail").className = "tag " + (osm ? "approx" : "ok");
  $("#circ-avail").textContent = osm ? "APPROX · OPENSTREETMAP" : "VERIFIED GEOMETRY";
  $("#circ-legend").hidden = false;
  $("#circ-sectors").style.display = osm ? "none" : "";   // no verified sectors for OSM outlines
  $("#circ-corners").style.display = osm ? "none" : "";
  Object.assign(circView, { x: 0, y: 0, w: 1000, h: 1000 }); svg.setAttribute("viewBox", "0 0 1000 1000");
  const ol = d.outline, n = ol.length;
  const line = pts => pts.map((p, i) => (i ? "L" : "M") + p[0] + " " + p[1]).join(" ");
  // FastF1 laps naturally return to the start (close with Z); OSM outlines are never
  // force-closed — the validated near-zero gap is left as-is rather than drawing a segment.
  let h = `<path class="track-base" d="${line(ol)}${osm ? "" : " Z"}"/>`;
  if (osm) {
    // Single-colour outline only — never invent sector shading on an OSM centreline.
    h += `<path class="track-sec s1" d="${line(ol)}" style="stroke:var(--cyan)"/>`;
    if (d.start_finish) h += `<g class="startfin"><circle cx="${d.start_finish[0]}" cy="${d.start_finish[1]}" r="11"/><circle cx="${d.start_finish[0]}" cy="${d.start_finish[1]}" r="5"/></g>`;
    $("#circ-legend").innerHTML = `<span class="it"><i class="sw s1"></i>Circuit outline</span>
      ${d.start_finish ? '<span class="it"><i class="sw sf"></i>Start / finish (OSM)</span>' : ""}
      <span class="it osm-attr">© OpenStreetMap contributors</span>`;
  } else {
    const showSec = $("#circ-sectors").classList.contains("active");
    const showCor = $("#circ-corners").classList.contains("active");
    // Marshal sectors are the ONLY traceable sector-boundary source here. The build projects
    // each marshal post onto the nearest outline SEGMENT (same FastF1 coordinate frame) with
    // snap-distance validation; we shade the segments between accepted boundaries in alternating
    // tones. These are marshal sectors, NOT the 3 official timing sectors — see the legend.
    const bounds = (d.marshal_boundaries && d.marshal_boundaries.length) ? d.marshal_boundaries : marshalBoundaries(d);
    if (showSec && bounds.length >= 2) {
      for (let k = 0; k < bounds.length; k++) {
        const a = bounds[k], b = bounds[(k + 1) % bounds.length];
        const part = a < b ? ol.slice(a, b + 1) : ol.slice(a).concat(ol.slice(0, b + 1));
        h += `<path class="track-sec" d="${line(part)}" style="stroke:${k % 2 ? "#b79bff" : "var(--cyan)"}"/>`;
      }
    } else {
      h += `<path class="track-sec s1" d="${line(ol)}" style="stroke:var(--cyan)"/>`;
    }
    (d.marshal_sectors || []).forEach(m => { const xy = m.projected_xy || m.xy; h += `<circle class="marshal" cx="${xy[0]}" cy="${xy[1]}" r="4"/>`; });
    if (d.start_finish) h += `<g class="startfin"><circle cx="${d.start_finish[0]}" cy="${d.start_finish[1]}" r="11"/><circle cx="${d.start_finish[0]}" cy="${d.start_finish[1]}" r="5"/></g>`;
    if (showCor) (d.corners || []).forEach((c, i) => {
      h += `<g class="corner" data-idx="${i}" tabindex="0" role="button" aria-label="Turn ${c.number}">
        <circle class="hit" cx="${c.xy[0]}" cy="${c.xy[1]}" r="18"/>
        <circle class="dot" cx="${c.xy[0]}" cy="${c.xy[1]}" r="6"/>
        <text x="${c.xy[0]}" y="${c.xy[1]}" dy="-11" text-anchor="middle">${c.number}</text></g>`;
    });
    const marsh = (d.marshal_sectors || []).length;
    $("#circ-legend").innerHTML = (marsh
      ? `<span class="it"><i class="sw s1"></i><i class="sw s3"></i>Marshal sectors (${marsh})</span>`
      : "") +
      `<span class="it"><i class="sw sf"></i>Start / finish</span>
       <span class="it"><i class="sw mk"></i>Marshal post</span>
       <span class="it osm-attr">Official S1/S2/S3 timing sectors: not available</span>`;
  }
  svg.innerHTML = h;
  // Visible OpenStreetMap attribution on the map itself (ODbL requirement).
  setCircuitAttribution(osm ? (d.attribution || "© OpenStreetMap contributors") : "");
  $("#circ-loc").textContent = [d.location, d.country].filter(Boolean).join(", ") + (d.source_event ? "  ·  " + d.source_event : "");
  const val = d.validation || {}, mp = d.marshal_projection || {};
  const facts = osm
    ? [["Geometry", "centreline outline"], ["Corners", "not in OSM data"],
       ["Track length", d.track_length_m ? d.track_length_m + " m (OSM)" : "—"],
       ["Loop", d.loop_closed ? "closed" : "open / partial"],
       ["Source", val.from_relation ? "OSM relation" : "raceway ways"]]
    : [["Corners", (d.corners || []).length],
       ["Marshal sectors", mp.n_accepted != null ? `${mp.n_accepted} accepted${mp.n_rejected ? ` · ${mp.n_rejected} rejected` : ""}` : (d.marshal_sectors || []).length],
       ["Official S1/S2/S3", "unavailable"],
       ["Start / finish", d.start_finish ? "telemetry S/F" : "—"],
       ["Pit lane", d.available && d.available.pit_lane ? "yes" : "not derived"]];
  if (osm && (d.missing || []).length) facts.push(["Not available", d.missing.join(", ")]);
  $("#circ-facts").innerHTML = facts.map(([k, v]) => `<div class="line"><span class="k">${k}</span><span class="val">${v}</span></div>`).join("");
  const prov = [["Source", d.source], ["Provider", d.data_provider]];
  if (d.source_event) prov.push(["Event", d.source_event]);
  prov.push(["Retrieved", d.retrieved], ["Licence", d.licence]);
  if (d.attribution) prov.push(["Attribution", d.attribution]);
  if (d.osm && d.osm.relation_id) prov.push(["OSM relation", String(d.osm.relation_id)]);
  if (d.osm && (d.osm.way_ids || []).length) prov.push(["OSM way ids", d.osm.way_ids.slice(0, 6).join(", ") + (d.osm.way_ids.length > 6 ? "…" : "")]);
  $("#circ-prov").innerHTML = `<div class="prov-h">Provenance &amp; geometry type</div>
    <div class="prov-row"><span>Geometry</span><b>${esc(d.geometry_label || d.geometry_type)}</b></div>
    ${prov.map(([k, v]) => `<div class="prov-row"><span>${k}</span><b>${esc(v)}</b></div>`).join("")}
    ${d.identity_basis ? `<div class="prov-note"><b>Identity:</b> ${esc(d.identity_basis)}</div>` : ""}
    ${mp.n_accepted != null ? `<div class="prov-note"><b>Marshal projection:</b> ${mp.n_accepted} accepted, ${mp.n_rejected} rejected · ${esc(mp.coordinate_reference || "")} · nearest-segment, max snap ${mp.max_snap_units ?? "—"} units. Official S1/S2/S3 timing sectors remain unavailable.</div>` : ""}
    ${val.accepted_reason ? `<div class="prov-note"><b>Validation:</b> accepted — ${esc(val.accepted_reason)}${val.within_expected_window === false ? " (outside the heuristic length window; accepted on circuit-relation identity)" : ""}.</div>` : ""}
    ${d.licence_compliance ? `<div class="prov-note"><b>Licensing:</b> ${esc(d.licence_compliance)}</div>` : ""}
    ${d.config_note ? `<div class="prov-note"><b>Configuration:</b> ${esc(d.config_note)}</div>` : ""}
    <div class="prov-note">${esc(d.notes)}</div>`;
  circCornerIdx = -1;
  $("#circ-corner").innerHTML = osm
    ? '<div class="empty">OpenStreetMap provides the outline only — corner numbers and sectors are not available for this circuit.</div>'
    : '<div class="empty">Hover, click or arrow-key through the corners.</div>';
  if (!osm) bindCircuitInteractions(); else bindCircuitPanZoom();
}
function esc(s) { return String(s == null ? "" : s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
function showCorner(i, lock) {
  const d = state.circuit; if (!d || !d.corners[i]) return;
  const c = d.corners[i];
  $$("#circ-svg .corner").forEach(g => g.classList.toggle("on", +g.dataset.idx === i));
  $("#circ-corner").innerHTML = `<div class="corner-card"><span class="ct">TURN</span><b class="cn">${c.number}</b>
    <div class="cmeta mono">corner ${i + 1} of ${d.corners.length}<br>map x${c.xy[0]} y${c.xy[1]} (normalised)</div>
    ${lock ? '<span class="clk mono">selected</span>' : ""}</div>`;
}
function bindCircuitPanZoom() {
  const svg = $("#circ-svg");
  svg.onwheel = ev => { ev.preventDefault(); zoomCircuit(ev.deltaY < 0 ? 0.85 : 1.18, ev); };
  let drag = null;
  svg.onpointerdown = ev => { if (ev.target.closest(".corner")) return; drag = { x: ev.clientX, y: ev.clientY }; svg.setPointerCapture(ev.pointerId); };
  svg.onpointermove = ev => { if (!drag) return; const r = svg.getBoundingClientRect(); circView.x -= (ev.clientX - drag.x) / r.width * circView.w; circView.y -= (ev.clientY - drag.y) / r.height * circView.h; drag = { x: ev.clientX, y: ev.clientY }; applyCircView(); };
  svg.onpointerup = () => { drag = null; };
  svg.onkeydown = ev => {
    const d = state.circuit; if (!d) return;
    const corners = d.corners || [];
    if (corners.length && (ev.key === "ArrowRight" || ev.key === "ArrowDown")) { circCornerIdx = (circCornerIdx + 1) % corners.length; showCorner(circCornerIdx, true); ev.preventDefault(); }
    else if (corners.length && (ev.key === "ArrowLeft" || ev.key === "ArrowUp")) { circCornerIdx = (circCornerIdx - 1 + corners.length) % corners.length; showCorner(circCornerIdx, true); ev.preventDefault(); }
    else if (ev.key === "+" || ev.key === "=") zoomCircuit(0.85);
    else if (ev.key === "-") zoomCircuit(1.18);
    else if (ev.key === "0") resetCircuit();
  };
}
function bindCircuitInteractions() {
  $$("#circ-svg .corner").forEach(g => {
    const i = +g.dataset.idx;
    g.addEventListener("mouseenter", () => showCorner(i, false));
    g.addEventListener("click", () => { circCornerIdx = i; showCorner(i, true); });
    g.addEventListener("focus", () => { circCornerIdx = i; showCorner(i, true); });
  });
  bindCircuitPanZoom();
}
function applyCircView() { $("#circ-svg").setAttribute("viewBox", `${circView.x} ${circView.y} ${circView.w} ${circView.h}`); }
function zoomCircuit(f, ev) {
  const svg = $("#circ-svg"), r = svg.getBoundingClientRect();
  let cx = circView.x + circView.w / 2, cy = circView.y + circView.h / 2;
  if (ev) { cx = circView.x + (ev.clientX - r.left) / r.width * circView.w; cy = circView.y + (ev.clientY - r.top) / r.height * circView.h; }
  const nw = Math.max(120, Math.min(1000, circView.w * f)), nh = Math.max(120, Math.min(1000, circView.h * f));
  circView.x = cx - (cx - circView.x) * (nw / circView.w); circView.y = cy - (cy - circView.y) * (nh / circView.h);
  circView.w = nw; circView.h = nh;
  circView.x = Math.max(-200, Math.min(1200 - nw, circView.x)); circView.y = Math.max(-200, Math.min(1200 - nh, circView.y));
  applyCircView();
}
function resetCircuit() { Object.assign(circView, { x: 0, y: 0, w: 1000, h: 1000 }); applyCircView(); }

/* ============================== TYRE INTELLIGENCE ============================= */
async function loadTyre(e) {
  const hit = findTyre(e.EventName);
  $("#tyre-event").textContent = e.EventName || "—";
  if (!hit) { renderTyreUnavailable(e); return; }
  $("#tyre-avail").className = "tag"; $("#tyre-avail").textContent = "LOADING…";
  try {
    const data = await api("/api/intel/tyre/" + encodeURIComponent(hit.event_id));
    if (data.status !== "ok" || !data.compounds) { renderTyreUnavailable(e); return; }
    state.tyre = data; state.tyreHidden = {}; renderTyre();
  } catch (err) { renderTyreUnavailable(e, err); }
}
function renderTyreUnavailable(e) {
  state.tyre = null;
  $("#tyre-avail").className = "tag warn"; $("#tyre-avail").textContent = "NO CACHED DATA";
  $("#tyre-svg").innerHTML = ""; $("#tyre-compounds").innerHTML = "";
  const avail = state.intel.tyreEvents.length;
  $("#tyre-stats").innerHTML = `<div class="empty unavail"><b>No cached tyre data for ${e.EventName || "this event"}.</b>
    <span>Tyre intelligence is drawn only from cached race laps (${avail} events available in this deployment). No values are fabricated.</span></div>`;
  $("#tyre-compare").innerHTML = ""; $("#tyre-prov").innerHTML = "";
}
function renderTyre() {
  const d = state.tyre, comps = Object.keys(d.compounds);
  $("#tyre-avail").className = "tag ok"; $("#tyre-avail").textContent = d.clean_laps + " CLEAN LAPS";
  $("#tyre-compounds").innerHTML = comps.map(c =>
    `<button class="chip-btn comp-chip ${state.tyreHidden[c] ? "" : "active"}" data-c="${c}"><i style="background:${COMPOUND_HEX[c] || "#888"}"></i>${c}</button>`).join("");
  $$("#tyre-compounds .comp-chip").forEach(b => b.addEventListener("click", () => { const c = b.dataset.c; state.tyreHidden[c] = !state.tyreHidden[c]; renderTyre(); }));
  drawTyreChart();
  $("#tyre-stats").innerHTML = [
    ["Event", esc(d.event) + " " + d.season], ["Session", esc(d.session)],
    ["Clean laps", d.clean_laps + " / " + d.total_laps + " total"],
    ["Compounds", comps.join(", ")],
  ].map(([k, v]) => `<div class="line"><span class="k">${k}</span><span class="val">${v}</span></div>`).join("");
  // Compound comparison: median pace loss across each compound's observed age range.
  $("#tyre-compare").innerHTML = '<div class="prov-h">Degradation (median, clean laps)</div>' + comps.map(c => {
    const cc = d.compounds[c], obs = cc.observations, last = obs[obs.length - 1];
    const dd = last.pace_loss_s, perLap = obs.length > 1 ? (dd / (last.age - obs[0].age)) : 0;
    return `<div class="cmp-row ${state.tyreHidden[c] ? "off" : ""}"><span class="cmp-n"><i style="background:${COMPOUND_HEX[c] || "#888"}"></i>${c}</span>
      <span class="mono">${cc.n} laps · age ${cc.age_range[0]}–${cc.age_range[1]}</span>
      <span class="mono cmp-d">${dd >= 0 ? "+" : ""}${dd.toFixed(2)}s · ${(perLap * 1000).toFixed(0)} ms/lap</span></div>`;
  }).join("");
  $("#tyre-prov").innerHTML = `<div class="prov-h">Provenance &amp; limitations</div>
    <div class="prov-row"><span>Source</span><b>${esc(d.source)}</b></div>
    <div class="prov-row"><span>Metric</span><b>${esc(d.metric)}</b></div>
    <div class="prov-row"><span>Retrieved</span><b>${esc(d.retrieved)}</b></div>
    <div class="prov-note"><b>Cleaning:</b> ${esc(d.cleaning)}</div>
    <div class="prov-note"><b>Limitations:</b> ${esc(d.limitations)}</div>`;
}
function drawTyreChart() {
  const d = state.tyre, svg = $("#tyre-svg"), W = 760, H = 360, pad = { l: 52, r: 16, t: 16, b: 34 };
  const comps = Object.keys(d.compounds).filter(c => !state.tyreHidden[c]);
  svg.innerHTML = "";
  const all = comps.flatMap(c => d.compounds[c].observations);
  if (!all.length) { svg.innerHTML = `<text x="${W / 2}" y="${H / 2}" text-anchor="middle" class="axl">No compound selected</text>`; return; }
  const ages = all.map(o => o.age), times = all.flatMap(o => [o.p25_s, o.p75_s]);
  const aMin = Math.min(...ages), aMax = Math.max(...ages), tMin = Math.min(...times), tMax = Math.max(...times);
  const tPad = (tMax - tMin) * 0.08 || 0.5;
  const x = a => pad.l + (aMax === aMin ? 0.5 : (a - aMin) / (aMax - aMin)) * (W - pad.l - pad.r);
  const y = t => pad.t + (t - (tMin - tPad)) / ((tMax + tPad) - (tMin - tPad)) * (H - pad.t - pad.b);
  // gridlines + y labels (lap time seconds)
  for (let k = 0; k <= 4; k++) { const tv = (tMin - tPad) + k / 4 * ((tMax + tPad) - (tMin - tPad)); const yy = y((tMin - tPad) + (1 - k / 4) * ((tMax + tPad) - (tMin - tPad))); svg.insertAdjacentHTML("beforeend", `<line class="grid" x1="${pad.l}" y1="${yy}" x2="${W - pad.r}" y2="${yy}"/><text class="axl" x="8" y="${yy + 3}">${tv.toFixed(1)}s</text>`); }
  // x labels (tyre age)
  const step = Math.max(1, Math.round((aMax - aMin) / 7));
  for (let a = aMin; a <= aMax; a += step) svg.insertAdjacentHTML("beforeend", `<text class="axl" x="${x(a)}" y="${H - 12}" text-anchor="middle">${a}</text>`);
  svg.insertAdjacentHTML("beforeend", `<text class="axt" x="${(pad.l + W - pad.r) / 2}" y="${H - 1}" text-anchor="middle">tyre age (laps)</text>`);
  comps.forEach(c => {
    const col = COMPOUND_HEX[c] || "#8aa0bd", obs = d.compounds[c].observations;
    const band = obs.map(o => `${x(o.age)},${y(o.p25_s)}`).concat(obs.slice().reverse().map(o => `${x(o.age)},${y(o.p75_s)}`)).join(" ");
    svg.insertAdjacentHTML("beforeend", `<polygon class="tband" points="${band}" style="fill:${col}"/>`);
    const line = obs.map((o, i) => (i ? "L" : "M") + x(o.age) + " " + y(o.median_s)).join(" ");
    svg.insertAdjacentHTML("beforeend", `<path class="tline" d="${line}" style="stroke:${col}"/>`);
    obs.forEach(o => svg.insertAdjacentHTML("beforeend", `<circle class="tpt" cx="${x(o.age)}" cy="${y(o.median_s)}" r="2.6" style="fill:${col}" data-c="${c}" data-a="${o.age}" data-m="${o.median_s}" data-n="${o.n}"/>`));
  });
  const tip = $("#tyre-tip");
  svg.onmousemove = ev => {
    const el = ev.target.closest(".tpt"); if (!el) { tip.hidden = true; return; }
    const r = svg.getBoundingClientRect();
    tip.hidden = false; tip.style.left = (ev.clientX - r.left) + "px"; tip.style.top = (ev.clientY - r.top) + "px";
    tip.innerHTML = `${el.dataset.c} · age ${el.dataset.a}<br>median ${(+el.dataset.m).toFixed(3)}s · n=${el.dataset.n}`;
  };
  svg.onmouseleave = () => { tip.hidden = true; };
}

function renderSources() {
  const src = [["Backend API", true], ["Prediction engine", true], ["Calendar", true], ["Driver lineup", true],
    ["Live timing", false], ["Weather", false], ["GPS positioning", false], ["Safety-car / VSC", false], ["Team radio", false]];
  $("#sources").innerHTML = src.map(([n, ok]) => `<li><span>${n}</span><span class="${ok ? "s-ok" : "s-no"}">${ok ? "AVAILABLE" : "UNAVAILABLE"}</span></li>`).join("");
}

boot();
