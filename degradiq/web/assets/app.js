"use strict";
/* DEGRADIQ — APEX CONTROL. Vanilla client for the DegradIQ FastAPI backend (same origin).
   Integrity: on API failure the real error is shown with a retry; predictions are
   never fabricated or replaced with sample values. */

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"];
const COLORS = ["#4fd6dd", "#e8c07d", "#5aa8ff", "#4fd38a", "#ff6b6b", "#b79bff", "#f3d14e", "#7fd0ff"];
const state = { calendar: [], lineup: [], event: null, mode: "pre-race", predictions: [], raceLaps: 57 };

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
}

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
  $("#res-body").innerHTML = ps.map(p => {
    const w = p.likely_window_if_stops || {}, prob = p.probability_stop_before_finish ?? 0;
    return `<tr><td class="mono">${p.predicted_order_by_expected_lap ?? "—"}</td><td><span class="drv">${p.driver}</span></td>
      <td>${p.team || "—"}</td><td class="prob-cell">${(prob * 100).toFixed(1)}%<div class="bar"><i style="width:${Math.round(prob * 100)}%"></i></div></td>
      <td class="mono">${p.most_likely_stop_lap ?? "—"}</td><td class="mono">${p.median_stop_lap_if_stops ?? "—"}</td>
      <td class="mono">${p.expected_stop_lap_if_stops ?? "—"}</td><td class="mono">${w.from_lap != null ? w.from_lap + "–" + w.to_lap : "—"}</td></tr>`;
  }).join("");
}

function renderBoard() {
  const host = $("#board");
  let ps = state.predictions.filter(p => p.status === "forecast");
  if (!ps.length) { host.innerHTML = '<div class="empty">Run a forecast to populate the board.</div>'; return; }
  const sort = $("#board-sort").value;
  ps = ps.slice().sort((a, b) => sort === "prob" ? (b.probability_stop_before_finish || 0) - (a.probability_stop_before_finish || 0) : (a.expected_stop_lap_if_stops || 1e9) - (b.expected_stop_lap_if_stops || 1e9));
  const L = state.raceLaps;
  host.innerHTML = ps.map(p => {
    const w = p.likely_window_if_stops || {}, pct = Math.round((p.probability_stop_before_finish || 0) * 100);
    if (w.from_lap == null) return `<div class="lane"><span class="who"><b>${p.driver}</b></span><div class="track"></div><span class="pw">no window</span></div>`;
    const left = (w.from_lap / L) * 100, width = Math.max(1.5, ((w.to_lap - w.from_lap) / L) * 100);
    const med = p.median_stop_lap_if_stops != null ? (p.median_stop_lap_if_stops / L) * 100 : null;
    return `<div class="lane"><span class="who"><b>${p.driver}</b><span>${pct}%</span></span>
      <div class="track" title="${p.driver}: window L${w.from_lap}–${w.to_lap}, median L${p.median_stop_lap_if_stops ?? "—"}">
        <div class="win" style="left:${left}%;width:${width}%"></div>${med != null ? `<div class="med" style="left:${med}%"></div>` : ""}</div>
      <span class="pw">L${w.from_lap}–${w.to_lap} · <b>${p.expected_stop_lap_if_stops ?? "—"}</b></span></div>`;
  }).join("") + `<div class="scale"><span>driver</span><div class="ticks"><span>1</span><span>${Math.round(L / 2)}</span><span>${L}</span></div><span>exp. lap</span></div>`;
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

function renderSources() {
  const src = [["Backend API", true], ["Prediction engine", true], ["Calendar", true], ["Driver lineup", true],
    ["Live timing", false], ["Weather", false], ["GPS positioning", false], ["Safety-car / VSC", false], ["Team radio", false]];
  $("#sources").innerHTML = src.map(([n, ok]) => `<li><span>${n}</span><span class="${ok ? "s-ok" : "s-no"}">${ok ? "AVAILABLE" : "UNAVAILABLE"}</span></li>`).join("");
}

boot();
