"use strict";
/* DEGRADIQ Pit Wall — vanilla client for the DegradIQ FastAPI backend (same origin).
   No fabricated data: failures are surfaced, never replaced with sample values. */

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"];
const SERIES_COLORS = ["#35d0d6", "#ffb020", "#4aa8ff", "#36d17a", "#ff5d5d", "#c08bff", "#ffd23f", "#7fd0ff"];

const state = { calendar: [], lineup: [], event: null, laps: 57, mode: "pre-race", predictions: [], raceLaps: 57 };

async function api(path, opts) {
  const r = await fetch(path, opts);
  const ct = r.headers.get("content-type") || "";
  const body = ct.includes("json") ? await r.json() : await r.text();
  if (!r.ok) throw { status: r.status, body };
  return body;
}
function setPill(id, cls, text) {
  const el = $("#" + id); el.className = "pill state " + cls;
  el.innerHTML = '<i class="dot"></i>' + text;
}

/* ---- clock ---- */
function tick() {
  const d = new Date();
  $("#st-clock").textContent = d.toLocaleTimeString([], { hour12: false });
}
setInterval(tick, 1000); tick();

/* ---- boot ---- */
async function boot() {
  // Health / API status
  try { await api("/health"); setPill("st-api", "ok", "API ONLINE"); }
  catch { setPill("st-api", "bad", "API OFFLINE"); }

  // Model status
  try {
    const s = await api("/strategy/first-stop/status");
    if (s.ready === false || s.status === "model_not_trained") { setPill("st-model", "bad", "MODEL NOT TRAINED"); renderIntegrity(null); }
    else { setPill("st-model", "ok", "MODEL READY"); renderIntegrity(s.provenance); fillModelFields(s.provenance); }
  } catch { setPill("st-model", "bad", "MODEL ERROR"); renderIntegrity(null); }

  // Live timing: no real feed — always honest.
  setPill("st-live", "warn", "LIVE TIMING UNAVAILABLE");

  await Promise.all([loadCalendar(), loadLineup()]);
  renderSources();
  bindUI();
}

function fillModelFields(p) {
  if (!p) return;
  $("#m-arch").textContent = p.architecture || "—";
  $("#m-label").textContent = p.label_definition_version || "—";
  $("#m-lstat").textContent = p.label_version_status || "—";
  $("#foot-build").textContent = (p.architecture || "model") + " · " + (p.label_version_status || "");
}

function renderIntegrity(p) {
  const box = $("#integrity");
  if (!p) { box.innerHTML = '<div class="empty">Model status unavailable.</div>'; return; }
  const sha = (p.model_sha256 || "").slice(0, 12);
  const statusBadge = p.label_version_status === "current" ? "ok" : (p.label_version_status === "stale" ? "bad" : "warn");
  const rows = [
    ["Architecture", `<span class="badge ok">${p.architecture || "—"}</span>`],
    ["Label version", p.label_definition_version || "—"],
    ["Label status", `<span class="badge ${statusBadge}">${p.label_version_status || "—"}</span>`],
    ["Model SHA-256", sha ? sha + "…" : "—"],
    ["Training seasons", (p.training_seasons || []).join(", ") || "—"],
    ["Readiness", `<span class="badge ok">READY</span>`],
  ];
  box.innerHTML = rows.map(([k, v]) => `<div class="line"><span class="k">${k}</span><span class="val">${v}</span></div>`).join("");
}

/* ---- calendar ---- */
async function loadCalendar() {
  try {
    const cal = await api("/calendar");
    state.calendar = Array.isArray(cal) ? cal : [];
    const sel = $("#in-event");
    sel.innerHTML = state.calendar.map((e, i) =>
      `<option value="${i}">R${e.RoundNumber ?? i + 1} · ${e.EventName}</option>`).join("");
    if (state.calendar.length) selectEvent(0);
  } catch (e) { $("#in-event").innerHTML = `<option>calendar unavailable (${e.status || "err"})</option>`; }
}
function selectEvent(i) {
  const e = state.calendar[i]; if (!e) return;
  state.event = e;
  $("#m-round").textContent = e.RoundNumber ?? "—";
  $("#m-loc").textContent = e.Location || e.Country || "—";
  $("#m-date").textContent = (e.EventDate || "").slice(0, 10) || "—";
  $("#st-event").textContent = "GP " + (e.EventName || "—");
}

/* ---- lineup / grid ---- */
async function loadLineup() {
  try {
    const l = await api("/strategy/first-stop/lineup");
    state.lineup = l.drivers || [];
    renderGrid();
  } catch (e) { $("#grid-body").innerHTML = `<tr><td colspan="9" class="empty">Lineup unavailable (${e.status || "err"})</td></tr>`; }
}
function renderGrid() {
  const body = $("#grid-body");
  body.innerHTML = state.lineup.map((d, i) => {
    const comp = COMPOUNDS.map(c => `<option ${c === "MEDIUM" ? "selected" : ""}>${c}</option>`).join("");
    return `<tr data-drv="${d.driver}">
      <td class="c-sel"><input type="checkbox" class="sel" ${i < 10 ? "checked" : ""}></td>
      <td><span class="drv">${d.driver}</span></td>
      <td>${d.team || "—"}</td>
      <td class="c-grid"><input type="number" class="grid" min="1" max="22" value="${i + 1}"></td>
      <td><select class="scomp">${comp}</select></td>
      <td class="c-age"><input type="number" class="sage" min="1" max="60" value="1"></td>
      <td class="live-col" hidden><input type="number" class="clap" min="1" max="100" value="1"></td>
      <td class="live-col" hidden><select class="ccomp">${comp}</select></td>
      <td class="live-col" hidden><input type="number" class="cpos" min="1" max="22" value="${i + 1}"></td>
    </tr>`;
  }).join("");
  body.querySelectorAll(".scomp").forEach(paintCompound);
  body.querySelectorAll(".scomp").forEach(s => s.addEventListener("change", () => paintCompound(s)));
  body.querySelectorAll(".sel").forEach(c => c.addEventListener("change", updateCount));
  updateCount();
}
function paintCompound(sel) {
  const td = sel.closest("td"); td.className = "comp " + sel.value.toLowerCase();
  if (!td.querySelector("i")) td.insertAdjacentHTML("afterbegin", "<i></i>");
}
function updateCount() { $("#m-count").textContent = $$("#grid-body .sel:checked").length; }

function selectedDrivers() {
  return $$("#grid-body tr").filter(tr => $(".sel", tr).checked).map(tr => {
    const d = { driver: tr.dataset.drv, grid_position: +$(".grid", tr).value,
      starting_compound: $(".scomp", tr).value, starting_tyre_age: +$(".sage", tr).value };
    if (state.mode === "live") {
      d.current_lap = +$(".clap", tr).value;
      d.current_compound = $(".ccomp", tr).value;
      d.current_position = +$(".cpos", tr).value;
    }
    return d;
  });
}

/* ---- UI bindings ---- */
function bindUI() {
  $("#in-event").addEventListener("change", e => selectEvent(+e.target.value));
  $("#in-laps").addEventListener("change", e => { state.laps = Math.max(5, Math.min(100, +e.target.value || 57)); });
  $("#in-mode").addEventListener("change", e => {
    state.mode = e.target.value;
    const live = state.mode === "live";
    $$(".live-col").forEach(c => c.hidden = !live);
    $("#live-hint").hidden = !live;
  });
  $("#h-sel").addEventListener("change", e => { $$("#grid-body .sel").forEach(c => c.checked = e.target.checked); updateCount(); });
  $("#b-all").addEventListener("click", () => { $$("#grid-body .sel").forEach(c => c.checked = true); updateCount(); });
  $("#b-none").addEventListener("click", () => { $$("#grid-body .sel").forEach(c => c.checked = false); updateCount(); });
  $("#b-run").addEventListener("click", runForecast);
  $("#board-sort").addEventListener("change", renderBoard);
}

/* ---- forecast ---- */
async function runForecast() {
  const drivers = selectedDrivers();
  const rs = $("#run-state"), btn = $("#b-run");
  if (!drivers.length) { rs.className = "run-state err"; rs.textContent = "Select at least one driver."; return; }
  if (!state.event) { rs.className = "run-state err"; rs.textContent = "Select a Grand Prix."; return; }
  state.laps = Math.max(5, Math.min(100, +$("#in-laps").value || 57));
  const payload = { event_name: state.event.EventName, race_laps: state.laps, mode: state.mode, drivers };
  btn.disabled = true; rs.className = "run-state"; rs.innerHTML = '<span class="spinner"></span>Forecasting…';
  try {
    const res = await api("/strategy/first-stop/predict", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    state.predictions = res.predictions || [];
    state.raceLaps = res.race_laps || state.laps;
    rs.className = "run-state ok";
    rs.textContent = `${state.predictions.length} forecasts · ${res.model_type || "model"}`;
    renderResults(); renderBoard(); renderChart();
  } catch (e) {
    // Surface the real error; never fabricate predictions.
    let msg = `Forecast failed (${e.status || "network"})`;
    if (e.body && e.body.detail) msg += ": " + (typeof e.body.detail === "string" ? e.body.detail : "validation error");
    rs.className = "run-state err"; rs.textContent = msg;
    state.predictions = [];
    $("#res-wrap").hidden = true; $("#res-empty").hidden = false;
    $("#res-empty").textContent = msg;
    $("#board").innerHTML = '<div class="empty">No forecast.</div>';
  } finally { btn.disabled = false; }
}

function renderResults() {
  const ps = state.predictions.filter(p => p.status === "forecast");
  if (!ps.length) { $("#res-wrap").hidden = true; $("#res-empty").hidden = false; $("#res-empty").textContent = "No forecasts returned."; return; }
  $("#res-empty").hidden = true; $("#res-wrap").hidden = false;
  $("#res-body").innerHTML = ps.map(p => {
    const w = p.likely_window_if_stops || {};
    const prob = p.probability_stop_before_finish ?? 0;
    return `<tr>
      <td class="mono">${p.predicted_order_by_expected_lap ?? "—"}</td>
      <td><span class="drv">${p.driver}</span></td>
      <td>${p.team || "—"}</td>
      <td class="prob-cell">${(prob * 100).toFixed(1)}%<div class="bar"><i style="width:${Math.round(prob * 100)}%"></i></div></td>
      <td class="mono">${p.most_likely_stop_lap ?? "—"}</td>
      <td class="mono">${p.median_stop_lap_if_stops ?? "—"}</td>
      <td class="mono">${p.expected_stop_lap_if_stops ?? "—"}</td>
      <td class="mono">${w.from_lap != null ? w.from_lap + "–" + w.to_lap : "—"}</td>
    </tr>`;
  }).join("");
}

/* ---- pit window board ---- */
function renderBoard() {
  const host = $("#board");
  let ps = state.predictions.filter(p => p.status === "forecast");
  if (!ps.length) { host.innerHTML = '<div class="empty">No forecast yet.</div>'; return; }
  const sort = $("#board-sort").value;
  ps = ps.slice().sort((a, b) => sort === "prob"
    ? (b.probability_stop_before_finish || 0) - (a.probability_stop_before_finish || 0)
    : (a.expected_stop_lap_if_stops || 1e9) - (b.expected_stop_lap_if_stops || 1e9));
  const L = state.raceLaps;
  host.innerHTML = ps.map(p => {
    const w = p.likely_window_if_stops || {};
    if (w.from_lap == null) return `<div class="lane"><span class="who">${p.driver}</span><div class="track"></div><span class="pw">no window</span></div>`;
    const left = (w.from_lap / L) * 100, width = Math.max(1.5, ((w.to_lap - w.from_lap) / L) * 100);
    const med = p.median_stop_lap_if_stops != null ? (p.median_stop_lap_if_stops / L) * 100 : null;
    return `<div class="lane">
      <span class="who">${p.driver} <span class="mono" style="color:var(--faint)">${Math.round((p.probability_stop_before_finish||0)*100)}%</span></span>
      <div class="track"><div class="win" style="left:${left}%;width:${width}%"></div>${med != null ? `<div class="med" style="left:${med}%"></div>` : ""}</div>
      <span class="pw">L${w.from_lap}–${w.to_lap}</span></div>`;
  }).join("") + `<div class="scale"><span>driver</span><div class="ticks"><span>1</span><span>${Math.round(L/2)}</span><span>${L}</span></div><span>window</span></div>`;
}

/* ---- cumulative probability chart ---- */
function renderChart() {
  const svg = $("#chart"), W = 720, H = 300, pad = { l: 36, r: 12, t: 12, b: 22 };
  const ps = state.predictions.filter(p => p.status === "forecast" && (p.cumulative_probabilities || []).length).slice(0, 6);
  svg.innerHTML = ""; $("#prob-legend").innerHTML = "";
  if (!ps.length) { svg.innerHTML = `<text x="${W/2}" y="${H/2}" text-anchor="middle">No forecast yet</text>`; return; }
  const L = state.raceLaps;
  const x = lap => pad.l + (lap / L) * (W - pad.l - pad.r);
  const y = p => pad.t + (1 - p) * (H - pad.t - pad.b);
  // y gridlines
  for (let g = 0; g <= 1.0001; g += 0.25) {
    svg.insertAdjacentHTML("beforeend", `<line class="axis" x1="${pad.l}" y1="${y(g)}" x2="${W-pad.r}" y2="${y(g)}"/><text x="4" y="${y(g)+3}">${Math.round(g*100)}%</text>`);
  }
  for (let lx = 0; lx <= L; lx += Math.max(5, Math.round(L / 6))) {
    svg.insertAdjacentHTML("beforeend", `<text x="${x(lx)}" y="${H-6}" text-anchor="middle">${lx}</text>`);
  }
  ps.forEach((p, i) => {
    const col = SERIES_COLORS[i % SERIES_COLORS.length];
    const pts = p.cumulative_probabilities.map(c => `${x(c.lap)},${y(c.probability_stopped_by_lap)}`).join(" L ");
    svg.insertAdjacentHTML("beforeend", `<path class="series" d="M ${pts}" stroke="${col}" data-drv="${p.driver}"/>`);
    $("#prob-legend").insertAdjacentHTML("beforeend", `<span class="it"><i style="background:${col}"></i>${p.driver}</span>`);
  });
  // hover tip
  const tip = $("#chart-tip");
  svg.onmousemove = ev => {
    const r = svg.getBoundingClientRect();
    const lap = Math.round(((ev.clientX - r.left) / r.width * W - pad.l) / (W - pad.l - pad.r) * L);
    if (lap < 0 || lap > L) { tip.hidden = true; return; }
    const parts = ps.map((p, i) => {
      const c = p.cumulative_probabilities.reduce((a, b) => Math.abs(b.lap - lap) < Math.abs(a.lap - lap) ? b : a);
      return `${p.driver} ${Math.round(c.probability_stopped_by_lap * 100)}%`;
    });
    tip.hidden = false; tip.style.left = (ev.clientX - r.left) + "px"; tip.style.top = (ev.clientY - r.top) + "px";
    tip.innerHTML = `L${lap} · ` + parts.join(" · ");
  };
  svg.onmouseleave = () => { tip.hidden = true; };
}

/* ---- data sources ---- */
function renderSources() {
  const src = [
    ["First-stop prediction API", true], ["2026 calendar snapshot", true], ["22-driver lineup", true],
    ["Live F1 timing", false], ["GPS track positioning", false], ["Weather", false],
    ["Team radio", false], ["Safety-car / VSC feed", false],
  ];
  $("#sources").innerHTML = src.map(([n, ok]) =>
    `<li><span>${n}</span><span class="${ok ? "s-ok" : "s-no"}">${ok ? "AVAILABLE" : "UNAVAILABLE"}</span></li>`).join("");
}

boot();
