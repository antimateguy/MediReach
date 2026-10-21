/* MediReach web app. Layout from the design mock-up; every number comes from the engine.
   Plain JavaScript, no build step: a hash router, HTML templates and one delegated event handler.
   The public story loads from data/public.json (no server needed); the calculators and the clinic
   portal call the API at CONFIG.apiBase (empty = the same server that serves this page). */
"use strict";

const CONFIG = window.MEDIREACH_CONFIG || {};
const API = String(CONFIG.apiBase || "").replace(/\/+$/, "");
const TOKEN_KEY = "medireach_token";
const token = {
  get: () => { try { return sessionStorage.getItem(TOKEN_KEY); } catch (_) { return null; } },
  set: (value) => { try { sessionStorage.setItem(TOKEN_KEY, value); } catch (_) { /* private mode: lasts for this page only */ } },
  clear: () => { try { sessionStorage.removeItem(TOKEN_KEY); } catch (_) { /* nothing stored */ } },
};
const WAKING = "Waking the server: the free host sleeps when nobody uses it, so the first visit takes up to a minute.";

const ICON = (name) => `https://pub-821312cfd07a4061bf7b99c1f23ed29b.r2.dev/v1/dynamic/color/${name}-dynamic-color.png`;
const S = {
  status: null, summary: null, impact: null, pilot: null, data: null, path: "",
  assume: null, pilotIn: { reduction: 18, holdout: 50 }, menu: false, prefill: "", ui: {},
};

/* ------------------------------------------------------------------ helpers */
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const int = (n) => (n == null ? "–" : Math.round(n).toLocaleString("en-US"));
const pct = (x, d = 1) => (x == null ? "–" : (100 * x).toFixed(d) + "%");
const signed = (x, d = 1) => (x == null ? "–" : (x > 0 ? "+" : x < 0 ? "−" : "") + Math.abs(x).toFixed(d));
const pts = (x) => (x == null ? "–" : signed(100 * x) + " pts");
const spct = (x) => (x == null ? "–" : signed(100 * x) + "%");
const inr = (x) => (x == null ? "–" : (x < 0 ? "−" : "") + "INR " + int(Math.abs(x)));
const $ = (sel) => document.querySelector(sel);
const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };

async function api(path, body) {
  const opts = { headers: {} };
  const bearer = token.get();
  if (bearer) opts.headers.Authorization = "Bearer " + bearer;
  if (body !== undefined) {
    opts.method = "POST"; opts.headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(body);
  }
  let response;
  try { response = await fetch(API + path, opts); } catch (_) {
    const error = new Error("The server cannot be reached right now. If it was asleep, it is starting; try again in a minute.");
    error.offline = true;
    throw error;
  }
  let data = null;
  try { data = await response.json(); } catch (_) { /* not JSON */ }
  if (!response.ok) {
    if (response.status === 401 && bearer) token.clear();
    const error = new Error((data && data.error) || response.statusText || "The server answered with an error.");
    error.status = response.status;
    throw error;
  }
  return data;
}

/* Waits for the server, showing a "waking up" note in `box` if it takes more than a moment. */
async function withWaking(promise, box) {
  const timer = setTimeout(() => { if (box) box.innerHTML = `<p class="wrap loading muted">${WAKING}</p>`; }, 2500);
  try { return await promise; } finally { clearTimeout(timer); }
}

const signedIn = () => Boolean(S.status ? S.status.user : token.get());

function toast(message, bad = false) {
  const box = $("#toasts");
  const el = document.createElement("div");
  el.className = "toast" + (bad ? " bad" : "");
  el.textContent = message;
  box.appendChild(el);
  setTimeout(() => el.remove(), bad ? 7000 : 4500);
}

const go = (path) => { if (location.hash === "#/" + path) route(); else location.hash = "#/" + path; };
const tagFor = (verdict) => ({ "Supported": "tag-accent", "Partly supported": "tag-accent-2" }[verdict] || "tag-neutral");
const labels = () => {
  const s = S.status || {};
  return `<span class="tag tag-outline">${esc(s.replay_label || "Replay of 2016 data")}</span>` +
    (s.simulated ? `<span class="tag tag-accent-2" title="Results include a planted effect (Demo controls)">Simulated</span>` : "");
};

/* --------------------------------------------------------------------- router */
const VIEWS = {
  "admin/today": { label: "Today", load: () => api("/api/admin/today"), render: vToday },
  "admin/plan": { label: "Plan and approve", load: () => api("/api/admin/plan?offset=" + (S.ui.offset || 0)), render: vPlan },
  "admin/team": { label: "Team board", load: () => api("/api/admin/team"), render: vTeam },
  "admin/results": { label: "Results", load: () => api("/api/admin/results"), render: vResults },
  "admin/directory": { label: "Directory", load: () => api(`/api/admin/directory?day=${S.ui.dirDay || "all"}&q=${encodeURIComponent(S.ui.dirQ || "")}`), render: vDirectory },
  "admin/demo": { label: "Demo controls", load: () => api("/api/admin/demo"), render: vDemo },
  "admin/backtest": { label: "Backtest queue", load: () => api("/api/admin/backtest" + (S.ui.btDay ? "?day=" + S.ui.btDay : "")), render: vBacktest },
  "me/tasks": { label: "My tasks", load: () => api("/api/me/tasks"), render: vTasks },
  "me/day": { label: "My day", load: () => api("/api/me/day"), render: vDay },
};

async function route() {
  const path = location.hash.replace(/^#\/?/, "");
  if (path !== S.path) { S.ui = {}; S.menu = false; }
  S.path = path;
  if (!VIEWS[path] && path !== "login") {
    /* The story needs no server: draw it now, then fetch the live labels in the background. */
    renderLanding(path.startsWith("s/") ? path.slice(2) : null);
    api("/api/status").then((status) => { S.status = status; if (!VIEWS[S.path] && S.path !== "login") refreshHeader(); }).catch(() => {});
    return;
  }
  try { S.status = await withWaking(api("/api/status"), $("#app")); } catch (e) {
    $("#app").innerHTML = `<div class="wrap loading"><p class="error">${esc(e.message)}</p><p style="margin-top:var(--space-4)"><button class="btn btn-primary" data-act="retry">Try again</button> <button class="btn btn-ghost" data-act="home">Back to the story</button></p></div>`;
    return;
  }
  const user = S.status.user;
  if (VIEWS[path]) {
    if (!user) { S.prefill = ""; return go("login"); }
    if (path.startsWith("admin/") && user.role !== "ADMIN") return go("me/tasks");
    if (path.startsWith("me/") && user.role !== "RECEPTIONIST") return go("admin/today");
    renderShell(path);
    return loadView();
  }
  return renderLogin();
}

function refreshHeader() {
  const old = document.querySelector(".site-header");
  if (old) old.outerHTML = header();
}

async function loadView() {
  const view = VIEWS[S.path];
  if (!view) return;
  try {
    S.data = await view.load();
    drawView();
    (S.data.notes || []).forEach((note) => toast(note));
  } catch (e) {
    if (e.status === 401) return go("login");
    $("#view").innerHTML = `<p class="error">${esc(e.message)}</p>`;
  }
}

function drawView() {
  const main = $("#view");
  if (main && S.data) main.innerHTML = VIEWS[S.path].render(S.data);
}

/* ------------------------------------------------------------------- landing */
function header() {
  const user = signedIn();
  const nav = [["problem", "Problem"], ["method", "How it works"], ["data", "Data"], ["insights", "Insights"],
    ["models", "Models"], ["impact", "Impact"], ["limits", "Method and limits"]];
  return `<header class="site-header"><div class="inner">
    <button class="brand spacer" data-act="top">MediReach</button>
    <nav class="site-nav" aria-label="Sections">
      ${nav.map(([id, label]) => `<button class="link" data-act="scroll" data-id="${id}">${label}</button>`).join("")}
      ${labels()}
      <button class="btn btn-primary" data-act="portal">${user ? "Open portal" : "Log in"}</button>
    </nav></div></header>`;
}

function statCard(bg, icon, label, value, note) {
  return `<div class="stat-card ${bg} lift" data-reveal><div class="icon" aria-hidden="true"><img src="${ICON(icon)}" alt="" loading="lazy" onerror="this.parentElement.remove()"></div>
    <p class="tile-label">${label}</p><div><div class="tile-value">${value}</div><p class="tile-note">${note}</p></div></div>`;
}

function sectionHead(number, eyebrow, title, lede, icon) {
  return `<div class="row-between" style="gap:var(--space-8)"><div data-reveal style="flex:1 1 520px">
      <p class="eyebrow">${number} — ${eyebrow}</p><h2 class="display">${title}</h2>${lede ? `<p class="lede">${lede}</p>` : ""}</div>
    ${icon ? `<div class="art" aria-hidden="true"><img src="${ICON(icon)}" alt="" loading="lazy" onerror="this.parentElement.remove()"></div>` : ""}</div>`;
}

function bars(rows, scale, valueText, colour = () => "") {
  return `<div class="bars num">${rows.map((row) => `<span>${esc(row.label)}</span>
    <div class="track"><div class="fill ${colour(row)}" data-w="${Math.max(1, Math.min(100, (100 * row.value) / scale)).toFixed(1)}" style="width:0"></div></div>
    <span style="text-align:right">${valueText(row)}</span>`).join("")}</div>`;
}

async function loadPublic() {
  /* The exported file first (instant, works with the server asleep); the API only as a fallback. */
  try {
    const response = await fetch("data/public.json", { cache: "no-cache" });
    if (response.ok) return await response.json();
  } catch (_) { /* fall back to the API */ }
  const summary = await api("/api/public/summary");
  return { summary, impact: null, pilot: null };
}

function renderLanding(scrollTo) {
  if (!S.summary) {
    $("#app").innerHTML = `<p class="wrap loading">Loading MediReach…</p>`;
    return loadPublic().then((bundle) => {
      S.summary = bundle.summary; S.impact = bundle.impact; S.pilot = bundle.pilot;
      S.assume = { ...bundle.summary.impact.assumptions };
      S.pilotIn.reduction = Math.round(100 * bundle.summary.pilot.relative_reduction);
      renderLanding(scrollTo);
    }).catch((e) => { $("#app").innerHTML = `<p class="wrap loading error">${esc(e.message)}</p>`; });
  }
  const m = S.summary, d = m.data, h = m.headline, imp = m.impact;
  const selectedLabel = (m.models.find((row) => row.model === m.selected_model) || {}).label;
  const dropped = d.rows_in - d.rows_out;
  const marquee = [`${int(d.rows_out)} appointments analysed`, `${pct(d.no_show_rate)} no-show rate`,
    `${h.lift_top_decile.toFixed(2)}× lift in the top decile`, `${inr(imp.selected_cost_per_prevented)} per prevented no-show (est.)`,
    `${int(imp.selected_prevented)} no-shows prevented (est.)`, "One ranked list, every morning"];
  const validation = m.models.filter((row) => row.period === "validation");
  const test = m.models.filter((row) => row.period === "test");
  const maxLoss = Math.max(...m.features.left_out.map((row) => row.loss));

  $("#app").innerHTML = `${header()}
  <section class="hero-frame" aria-label="MediReach">
    <div class="hero">
      <video id="hero-video" muted loop autoplay playsinline preload="metadata" aria-hidden="true" hidden></video>
      <div class="shade"></div>
      <div class="content">
        <div class="stack" style="flex:1 1 340px;max-width:44ch;gap:var(--space-6)">
          <p class="tagline">A friendly call before the visit — for the people most likely to need one.</p>
          <div class="row"><button class="btn btn-primary btn-lg" data-act="scroll" data-id="problem">Why it matters</button>
          <button class="btn btn-secondary btn-lg" data-act="scroll" data-id="method">How it works</button></div>
        </div>
        <h1>MediReach</h1>
      </div>
    </div>
  </section>

  <section class="wrap row-between" style="padding-block:clamp(72px,8vw,128px) clamp(40px,5vw,72px);gap:var(--space-6)">
    <p data-reveal style="flex:1 1 520px;font-weight:700;font-size:clamp(28px,3.4vw,52px);line-height:1.1;letter-spacing:-0.025em;max-width:22ch;text-wrap:balance">Reminders that reach the right people — before the appointment, not after.</p>
    <div class="art" aria-hidden="true" style="flex:0 1 320px;height:clamp(200px,22vw,300px)"><img src="${ICON("phone-ringing")}" alt="" loading="lazy" onerror="this.parentElement.remove()"></div>
  </section>
  <div class="marquee" aria-hidden="true"><div class="track-x">${[...marquee, ...marquee].map((t, i) => `<span class="item"><span>${esc(t)}</span><span class="dot" style="${i % 3 === 1 ? "background:var(--color-accent-2)" : ""}"></span></span>`).join("")}</div></div>

  <section class="wrap section" id="problem">
    <div class="row-between" data-reveal style="align-items:flex-end;margin-bottom:clamp(40px,5vw,64px);gap:var(--space-6) clamp(40px,6vw,96px)">
      <div><p class="eyebrow">01 — The problem</p><h2 class="display">Empty chairs, missed care.</h2></div>
      <p class="body" style="flex:0 1 460px">When someone misses a visit, their care slips and another patient waits longer. Clinics can only make so many reminder calls a day.</p>
    </div>
    <div class="grid grid-260">
      ${statCard("bg-a1", "calender", "Appointments analysed", int(d.rows_out), `Outpatient visits, ${d.first_date} – ${d.last_date}, after cleaning.`)}
      ${statCard("bg-b1", "phone-missing", "No-show rate", pct(d.no_show_rate), "About one appointment in five goes unattended.")}
      ${statCard("bg-n1", "target", "No-shows in the top 10% by risk", pct(h.precision_at_10), `${h.lift_top_decile.toFixed(2)}× the test-week rate of ${pct(h.base_rate)}.`)}
      ${statCard("bg-a2", "rupee", "Net benefit, test week (est.)", inr(imp.selected_net), `Under stated assumptions. Random calling: ${inr(imp.random_net)}.`)}
    </div>
  </section>

  <section class="wrap section" id="method">
    <div data-reveal><p class="eyebrow">02 — How it works</p><h2 class="display">Five steps to a kinder call list.</h2></div>
    <div class="grid grid-200" style="margin-top:clamp(40px,5vw,64px)">
      ${[["bg-a1", "folder", "Clean and load", "The raw file goes into a database with keys and checks."],
         ["bg-b1", "chart", "Test hypotheses", "Simple statistics confirm which patterns are real."],
         ["bg-n1", "puzzle", "Build features once", "Only what is known before the visit; history from earlier dates only."],
         ["bg-a2", "trophy", "Compare three models", "Judged on later weeks. The simplest one that holds up wins."],
         ["bg-b2", "tick", "Make the list", "A fixed number of calls a day, each with a plain reason."]].map(([bg, icon, title, text], i) =>
        `<div class="step-card ${bg} lift" data-reveal><div class="row-between" style="align-items:flex-start"><span class="n">${i + 1}</span>
          <div style="width:72px;height:72px" aria-hidden="true" class="icon-inline"><img src="${ICON(icon)}" alt="" loading="lazy" style="width:100%;height:100%;object-fit:contain" onerror="this.remove()"></div></div>
          <h3>${title}</h3><p style="font-size:15px">${text}</p></div>`).join("")}
    </div>
  </section>

  <section class="wrap section" id="data">
    ${sectionHead("03", "Data", "One city, six weeks of visits.", "Every number in MediReach comes from one public dataset of outpatient appointments. The portals replay it day by day.", "folder")}
    <section class="sub-section split">
      <div class="side" data-reveal><p class="eyebrow">Source</p><h2 class="title">Kaggle <em>Medical Appointment No Shows</em></h2>
        <p class="body" style="margin-top:var(--space-4)">Appointments at public clinics in one Brazilian city. One row per visit: when it was booked, the visit date, age, sex, the clinic's neighbourhood, welfare enrolment, health flags, whether an SMS was sent — and whether the patient came. Licence CC BY-NC-SA 4.0.</p></div>
      <div class="main grid" style="grid-template-columns:repeat(2,minmax(0,1fr))">
        ${[["Appointments", int(d.rows_in)], ["Patients", int(d.patients)], ["Clinic sites", int(d.clinics)], ["Visit dates", `${d.first_date} – ${d.last_date}`]].map(([label, value], i) =>
          `<div class="card" data-reveal><p class="tile-label">${label}</p><div class="tile-value"${i === 3 ? ' style="font-size:clamp(22px,2vw,30px)"' : ""}>${value}</div></div>`).join("")}
      </div>
    </section>
    <section class="sub-section split">
      <div class="side" data-reveal><p class="eyebrow">Cleaning</p><h2 class="title">${int(dropped)} rows out, nothing guessed.</h2>
        <p class="body" style="margin-top:var(--space-4)">Rows that could not be true were dropped, not repaired. Every drop is counted by rule, so a silent loss cannot happen again.</p></div>
      <div class="main stack" style="gap:var(--space-6)">
        <div class="card num" data-reveal style="display:grid;grid-template-columns:minmax(0,1fr) auto;gap:var(--space-4) var(--space-6);font-size:18px;padding:var(--space-4) var(--space-6)">
          <span>Raw rows</span><span style="text-align:right">${int(d.rows_in)}</span>
          <span>Visit dated before it was booked</span><span style="text-align:right;color:var(--color-accent-2-700)">−${int(d.dropped.negative_lead_days)}</span>
          <span>Age below 0 or above 110</span><span style="text-align:right;color:var(--color-accent-2-700)">−${int(d.dropped.invalid_age)}</span>
          <span>Duplicate appointment id</span><span style="text-align:right;color:var(--color-accent-2-700)">−${int(d.dropped.duplicate_appointment_id)}</span>
          <strong>Kept for analysis</strong><strong style="text-align:right;color:var(--color-accent-700)">${int(d.rows_out)}</strong>
        </div>
        <div class="stack" data-reveal><p class="muted small">Also fixed</p>
          <p class="body">Misspelt columns renamed (<em>Hipertension</em>, <em>Handcap</em>).</p>
          <p class="body">"No-show = Yes" means the patient missed: stored as <code>no_show = 1</code>, so every rate here is a no-show rate.</p>
          <p class="body">Booking times compared as dates, so a same-day booking has lead time 0 instead of looking impossible.</p>
          <p class="body">Loaded into SQLite with primary keys, foreign keys and checks on the cleaning rules.</p></div>
      </div>
    </section>
    <section class="sub-section card bg-b1" style="border-radius:32px;padding:clamp(36px,5vw,80px)">
      <div class="split">
        <div class="side" data-reveal style="flex:1 1 360px">
          <p class="eyebrow" style="color:var(--color-accent-2-800)">The v1 bugs</p>
          <h2 class="display-sm">Version 1 let the answer leak in.</h2>
          <p class="body" style="margin-top:var(--space-6)">Its strongest feature, a patient's past no-show rate, was a running total that included the visit being predicted. The model was partly reading the answer.</p>
          <p class="body" style="margin-top:var(--space-4)">It also dropped ${int(d.v1_timestamp_filter_drops)} rows (${pct(d.v1_timestamp_filter_drops / d.rows_in, 0)}) by comparing a booking timestamp with a date at midnight. They were same-day bookings, the group that misses least (${pct(d.same_day_no_show_rate)}).</p>
        </div>
        <div class="main stack" style="flex:1 1 360px">
          <div class="grid" style="grid-template-columns:repeat(2,minmax(0,1fr))">
            <div class="card bg-page" data-reveal><p class="tile-label">v1 reported ROC-AUC</p><div class="tile-value strike">0.74</div><p class="tile-note">Leaky feature, random split, same-day rows missing</p></div>
            <div class="card bg-page" data-reveal><p class="tile-label">v2, test week</p><div class="tile-value" style="color:var(--color-accent-700)">${h.roc_auc.toFixed(3)}</div><p class="tile-note">Honest: history from earlier dates, later weeks held out</p></div>
          </div>
          <div class="code" data-reveal>
            <div class="old">history = cumsum(no_show) / visit_number   <span class="cmt"># includes today</span></div>
            <div class="new">history = earlier dates only, shifted by one date</div>
            <div class="old">scheduled_day &lt;= appointment_day   <span class="cmt"># 18:38 &gt; 00:00</span></div>
            <div class="new">scheduled_date &lt;= appointment_date</div>
          </div>
        </div>
      </div>
    </section>
  </section>

  <section class="wrap section" id="insights">
    ${sectionHead("04", "Insights", "Six hypotheses, tested.", "Each idea was checked on the cleaned data with a two-proportion or chi-square test and a 95% confidence interval, before any model was built.", "chart")}
    <div class="grid" style="margin-top:clamp(72px,9vw,140px);grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:var(--space-4)">
      ${m.hypotheses.map((hyp) => {
        const scale = 1.15 * Math.max(...hyp.bars.map((bar) => bar.rate));
        return `<article class="card card-lg stack lift" data-reveal style="padding:clamp(28px,3vw,40px);gap:var(--space-4)">
          <div class="row-between"><span style="font-weight:700;font-size:44px;line-height:1;letter-spacing:-0.04em;color:var(--color-accent)">${esc(hyp.id)}</span><span class="tag ${tagFor(hyp.verdict)}">${esc(hyp.verdict)}</span></div>
          <h3 style="font-size:24px;line-height:1.15;text-wrap:balance">${esc(hyp.statement)}</h3>
          ${bars(hyp.bars.map((bar) => ({ label: bar.label, value: bar.rate })), scale, (row) => pct(row.value), (row) => (row.value === Math.max(...hyp.bars.map((bar) => bar.rate)) ? "" : "grey"))}
          <p class="small num" style="color:var(--color-neutral-800)">${esc(hyp.summary)}</p></article>`;
      }).join("")}
    </div>
  </section>

  <section class="wrap section" id="models">
    ${sectionHead("05", "Models and features", "Three models, one kept.", "Each was trained on the first three weeks and judged on the weeks after. " + esc(m.selection_reason.split(". ")[0]) + ". The random forest was barely better on validation and harder to explain, so logistic regression stays.", "puzzle")}
    <section class="sub-section">
      <p class="eyebrow">Model comparison</p><p class="muted" style="margin-bottom:var(--space-3)">Validation ROC-AUC</p>
      <div class="stack">${validation.map((row) => `<div class="card ${row.selected ? "bg-a1" : ""}" data-reveal style="display:grid;grid-template-columns:minmax(140px,240px) minmax(0,1fr) auto;gap:var(--space-4) var(--space-6);align-items:center">
        <span style="font-size:19px;${row.selected ? "font-weight:600" : ""}">${esc(row.label)} ${row.selected ? '<span class="tag tag-accent">selected</span>' : ""}</span>
        <div class="track"><div class="fill ${row.selected ? "" : "grey"}" data-w="${(100 * row.roc_auc).toFixed(1)}" style="width:0"></div></div>
        <span class="num" style="font-weight:700;font-size:24px">${row.roc_auc.toFixed(3)}</span></div>`).join("")}</div>
      <div class="grid grid-200" style="margin:var(--space-8) 0">
        <div class="card" data-reveal><p class="tile-label">Test ROC-AUC</p><div class="tile-value">${h.roc_auc.toFixed(3)}</div></div>
        <div class="card" data-reveal><p class="tile-label">No-shows in the top 10%</p><div class="tile-value">${pct(h.precision_at_10)}</div></div>
        <div class="card" data-reveal><p class="tile-label">No-shows found in the top 30%</p><div class="tile-value">${pct(h.capture_at_30)}</div></div>
        <div class="card bg-accent" data-reveal><p class="tile-label">Lift, top decile</p><div class="tile-value">${h.lift_top_decile.toFixed(2)}×</div></div>
      </div>
      <div class="table-wrap"><table class="table num" style="min-width:720px"><thead><tr><th>Model</th><th>Period</th><th class="r">ROC-AUC</th><th class="r">No-shows in top 10%</th><th class="r">Found in top 10%</th><th class="r">Found in top 30%</th><th class="r">Lift, top decile</th></tr></thead>
        <tbody>${[...validation, ...test].sort((a, b) => a.label.localeCompare(b.label) || b.period.localeCompare(a.period)).map((row) => `<tr><td>${esc(row.label)} ${row.selected ? '<span class="tag tag-accent">selected</span>' : ""}</td><td>${row.period === "test" ? "Test" : "Validation"}</td>
          <td class="r">${row.roc_auc.toFixed(3)}</td><td class="r">${pct(row.precision_at_10)}</td><td class="r">${pct(row.capture_at_10)}</td><td class="r">${pct(row.capture_at_30)}</td><td class="r">${row.lift_top_decile.toFixed(2)}</td></tr>`).join("")}</tbody></table></div>
      <p class="small muted" style="margin-top:var(--space-3)">Train ${m.split.train.start} – ${m.split.train.end} 2016 (${int(m.split.train.n)} rows) · validation ${m.split.validation.start} – ${m.split.validation.end} (${int(m.split.validation.n)}) · test ${m.split.test.start} – ${m.split.test.end} (${int(m.split.test.n)}).</p>
    </section>
    <section class="sub-section split">
      <div class="side" data-reveal><p class="eyebrow">Feature study</p><h2 class="title">Lead time does most of the work.</h2>
        <p class="body" style="margin-top:var(--space-4)">Validation ROC-AUC lost when one group of features is left out and the model is refit. With lead time alone all three models score ${m.features.grid[0].roc_auc.toFixed(3)}; every group together takes logistic regression to ${m.features.added[m.features.added.length - 1].roc_auc.toFixed(3)}. The gain comes from information, not from a fancier model.</p></div>
      <div class="main stack" style="gap:var(--space-4)">
        ${bars(m.features.left_out.map((row) => ({ label: row.label, value: Math.max(0, row.loss) })), maxLoss, (row) => "−" + row.value.toFixed(3), (row) => (row.value >= 0.009 ? "" : "grey"))}
        <p class="small muted">Clinic location was tested as an extra feature and lowered validation ROC-AUC, so it is not used.</p>
      </div>
    </section>
  </section>

  <section class="wrap section" id="impact">
    ${sectionHead("06", "Impact", "Same budget, better aimed.", "The same number of calls and texts, sent to different people. Change the assumptions below and watch the comparison move.", "rupee")}
    <section class="sub-section"><p class="eyebrow">Strategy comparison</p>
      <p class="small" id="strategies-note" role="status" style="color:var(--color-accent-2-700);margin-bottom:var(--space-2)"></p>
      <div class="stack" id="strategies">${strategiesHtml()}</div>
      <p class="small muted" style="margin-top:var(--space-3)">Test week ${m.split.test.start} – ${m.split.test.end} 2016 · ${int(imp.n_calls)} calls and ${int(imp.n_sms)} SMS for every strategy (10% and 20% of each day). Random is the expected result of picking the same number at random; Oracle knows who will miss, an upper bound no model reaches.</p>
    </section>
    <section class="sub-section split">
      <div class="side" data-reveal><p class="eyebrow">Assumptions</p><h2 class="title">These are guesses, not measurements.</h2>
        <p class="body" style="margin-top:var(--space-4)">No one in 2016 was called, so how well a call works is assumed. At the default assumptions the programme breaks even if a connected call prevents ${pct(imp.break_even)} of no-shows. Only a randomised pilot can measure it.</p>
        <button class="btn btn-ghost" data-act="reset-assume" style="padding-left:0;margin-top:var(--space-4)">Reset to defaults</button></div>
      <div class="main card bg-a1 card-lg grid" style="grid-template-columns:repeat(auto-fit,minmax(min(100%,240px),1fr));gap:var(--space-6) var(--space-8);padding:clamp(24px,3vw,40px)" id="sliders">${slidersHtml()}</div>
    </section>
    <section class="sub-section card bg-b1" style="border-radius:32px;padding:clamp(36px,5vw,80px)">
      <div class="split">
        <div class="side stack" style="flex:1 1 360px;gap:var(--space-6)">
          <div data-reveal><p class="eyebrow" style="color:var(--color-accent-2-800)">Pilot size</p><h2 class="display-sm" style="max-width:12ch">How big a pilot?</h2>
            <p class="body" style="margin-top:var(--space-4)">Hold some patients on the call list back from calls at random, then compare. The smaller the effect you hope to see, the more patients you need.</p></div>
          <div class="field"><label for="pilot-red">Reduction in no-shows to detect — <strong id="pilot-red-text">${S.pilotIn.reduction}%</strong></label><input id="pilot-red" type="range" min="5" max="50" step="1" value="${S.pilotIn.reduction}" data-input="pilot" data-key="reduction"></div>
          <div class="field"><label for="pilot-hold">Hold-out share — <strong id="pilot-hold-text">${S.pilotIn.holdout}%</strong> not called</label><input id="pilot-hold" type="range" min="10" max="50" step="5" value="${S.pilotIn.holdout}" data-input="pilot" data-key="holdout"></div>
          <p class="small" style="color:var(--color-neutral-800)">Default reduction ${pct(m.pilot.relative_reduction, 0)} = calls that connect (${pct(imp.assumptions.call_reach_rate, 0)}) × no-shows a connected call prevents (${pct(imp.assumptions.call_effectiveness, 0)}).</p>
        </div>
        <div class="main" style="flex:1 1 360px"><p class="small" id="pilot-note" role="status" style="color:var(--color-accent-2-700);margin-bottom:var(--space-2)"></p><div id="pilot">${pilotHtml()}</div></div>
      </div>
    </section>
  </section>

  <section class="wrap" style="padding-top:clamp(72px,9vw,140px)">
    <div class="cta" data-reveal>
      <div class="stack" style="flex:1 1 360px;padding-bottom:clamp(32px,4vw,64px);gap:var(--space-6);align-items:flex-start">
        <h3 style="font-family:var(--font-display);font-weight:400;font-size:clamp(40px,4.4vw,68px);line-height:1">See the clinic side.<br>Plan, call, measure.</h3>
        <p style="color:var(--color-neutral-200);max-width:44ch">Log in as the clinic manager to approve tomorrow's list, or as a receptionist to work it. Everything inside replays 2016 data; no real patient is contacted.</p>
        <button class="btn btn-primary btn-lg" data-act="portal">${signedIn() ? "Open portal" : "Try the demo"}</button>
      </div>
      <div style="flex:0 1 380px;aspect-ratio:472/496;margin:0 auto"><img src="assets/phone.png" alt="A hand holding a phone with a video call"></div>
    </div>
  </section>

  <section class="wrap section" id="limits">
    ${sectionHead("07", "Method and limits", "What we did, and what it cannot tell you.", "A plain pipeline, judged on later weeks it never saw — and a list of what the numbers cannot say.", "lab")}
    <section class="sub-section"><p class="eyebrow">Method</p>
      <div class="stack" style="gap:var(--space-8);max-width:1100px">
        ${[["Clean and load", `The raw file goes into SQLite with keys and checks. ${int(dropped)} impossible rows are dropped.`],
           ["Test hypotheses", "SQL views and two-proportion tests with confidence intervals, before any model is built."],
           ["Build features once", "One feature builder for training and scoring; patient history from strictly earlier dates."],
           ["Compare three models", "A lead-time rule, logistic regression and a random forest, judged on later weeks. The simplest one not clearly beaten is kept."],
           ["Turn scores into a list", "A fixed daily capacity: the top 10% are called, the next 20% get an SMS, each with a plain reason."]].map(([title, text], i) =>
          `<div class="method-step" data-reveal><span class="n">${i + 1}</span><div class="t"><h3>${title}</h3><p>${text}</p></div></div>`).join("")}
      </div>
      <p class="muted" style="margin:clamp(56px,6vw,88px) 0 var(--space-3)">Time-ordered split · ${int(d.rows_out)} rows</p>
      <div class="split-bar" data-reveal>
        <div style="flex:${m.split.train.n};background:var(--color-neutral-300);border-radius:18px 6px 6px 18px"><strong>Train</strong><span>${m.split.train.start} – ${m.split.train.end} · ${int(m.split.train.n)}</span></div>
        <div style="flex:${m.split.validation.n};background:var(--color-accent-200);border-radius:6px"><strong>Validation</strong><span>${m.split.validation.start} – ${m.split.validation.end} · ${int(m.split.validation.n)}</span></div>
        <div style="flex:${m.split.test.n};background:var(--color-accent);color:var(--color-bg);border-radius:6px 18px 18px 6px"><strong>Test</strong><span>${m.split.test.start} – ${m.split.test.end} · ${int(m.split.test.n)}</span></div>
      </div>
    </section>
    <section class="sub-section split">
      <div class="side" data-reveal><p class="eyebrow">Checking the measurement</p><h2 class="title">The hold-out finds an effect only when there is one.</h2>
        <p class="body" style="margin-top:var(--space-4)">One replay week run through the whole workflow (plan, approve with a 50% hold-out, calls, day closing), twice.</p></div>
      <div class="main grid" style="grid-template-columns:repeat(auto-fit,minmax(min(100%,240px),1fr))">
        <div class="card bg-n1" data-reveal><p class="tile-label">Real 2016 outcomes (true effect: none)</p><div class="tile-value md">${spct(m.replay.historical.relative_reduction)}</div><p class="tile-note">95% CI ${spct(m.replay.historical.reduction_ci[0])} to ${spct(m.replay.historical.reduction_ci[1])}: includes zero, as it should.</p></div>
        <div class="card bg-a1" data-reveal><p class="tile-label">Simulation with a planted ${pct(m.replay.settings.planted_effect, 0)} effect</p><div class="tile-value md">${spct(m.replay.simulated.relative_reduction)}</div><p class="tile-note">95% CI ${spct(m.replay.simulated.reduction_ci[0])} to ${spct(m.replay.simulated.reduction_ci[1])}: the effect is found.</p></div>
      </div>
    </section>
    <section class="sub-section"><p class="eyebrow">Limits</p><h2 class="title" style="max-width:20ch;margin-bottom:clamp(32px,4vw,56px)">Read every number with these in mind.</h2>
      <div class="grid grid-300">
        ${[["bg-n1", "Observational data", "The model predicts who is likely to miss a visit, not who a call will help. Those can be different people."],
           ["bg-a1", "No uplift estimate", "Benefit figures rest on assumed call effectiveness. Only a randomised pilot can measure it."],
           ["bg-b1", "One city, 2016", "Six weeks of one city's clinics may not carry over to other clinics or years."],
           ["bg-n1", "SMS is confounded", "Reminders went mostly to visits booked days ahead, so a raw SMS comparison runs the wrong way (H3)."],
           ["bg-a1", "Thin history", "Six weeks of data: many patients have no earlier visit to learn from."],
           ["bg-b1", "A replay, not live", "The portals replay 2016 records on today's calendar. No real patient is called, texted or rebooked."]].map(([bg, title, text]) =>
          `<div class="card card-lg ${bg}" data-reveal style="padding:clamp(28px,3vw,40px)"><h3 style="font-size:24px;margin-bottom:var(--space-3)">${title}</h3><p class="body">${text}</p></div>`).join("")}
      </div>
    </section>
    <section class="sub-section"><p class="eyebrow">Labels you will see</p>
      <div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:var(--space-8) clamp(40px,5vw,80px)">
        <div class="stack" data-reveal style="align-items:flex-start"><span class="tag tag-outline">Replay of 2016 data</span><p class="body">On every screen. Clocks, bookings and outcomes come from the 2016 records.</p></div>
        <div class="stack" data-reveal style="align-items:flex-start"><span class="tag tag-accent-2">Simulated</span><p class="body">Shown whenever a planted call effect is switched on in Demo controls.</p></div>
        <div class="stack" data-reveal style="align-items:flex-start"><span class="tag tag-accent">Written by AI</span><p class="body">On a briefing or script worded by Gemini. AI text appears only when someone presses the button, and every number in it is checked.</p></div>
      </div>
    </section>
  </section>

  <footer class="wrap" style="padding-block:clamp(96px,10vw,160px) var(--space-8)">
    <div class="footer-word">MediReach</div>
    <div class="row muted small" style="margin-top:var(--space-6);gap:var(--space-2) var(--space-6)"><span>Model ${esc(selectedLabel)}, version ${esc(m.model_version)}</span><span>Test ROC-AUC ${h.roc_auc.toFixed(3)}</span><span>Data ${d.first_date} – ${d.last_date}</span><span>Dataset: Kaggle, CC BY-NC-SA 4.0</span></div>
  </footer>`;

  setupHeroVideo();
  armReveal();
  if (!S.impact) refreshImpact();
  if (!S.pilot) refreshPilot();
  if (scrollTo) setTimeout(() => scrollToId(scrollTo, true), 0);   /* a deep link jumps straight there */
}

function setupHeroVideo() {
  /* Only a video the owner added as web/assets/hero.mp4 is used; otherwise the gradient stays. */
  const video = $("#hero-video");
  if (!video || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  fetch("assets/hero.mp4", { method: "HEAD" }).then((response) => {
    if (!response.ok) return;
    video.src = "assets/hero.mp4"; video.hidden = false; video.play().catch(() => {});
  }).catch(() => {});
}

function armReveal() {
  const show = (el) => { el.classList.add("shown"); el.querySelectorAll("[data-w]").forEach((bar) => { bar.style.width = bar.dataset.w + "%"; }); };
  const targets = document.querySelectorAll("[data-reveal]");
  const looseBars = [...document.querySelectorAll("[data-w]")].filter((bar) => !bar.closest("[data-reveal]"));
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    targets.forEach(show); looseBars.forEach((bar) => { bar.style.width = bar.dataset.w + "%"; }); return;
  }
  /* A plain position check on scroll: works everywhere, and nothing can stay hidden once it is on screen. */
  document.body.classList.add("reveal-on");
  let waiting = [...targets, ...looseBars], queued = false;
  const check = () => {
    queued = false;
    const bottom = window.innerHeight * 0.92;
    waiting = waiting.filter((el) => {
      if (!el.isConnected) return false;
      const box = el.getBoundingClientRect();
      if (box.top > bottom) return true;   /* still below the fold; anything above (jumped past) is shown */
      if (el.dataset.w && !el.hasAttribute("data-reveal")) el.style.width = el.dataset.w + "%"; else show(el);
      return false;
    });
    if (!waiting.length) { window.removeEventListener("scroll", onScroll); clearInterval(armReveal.timer); }
  };
  const onScroll = () => { if (!queued) { queued = true; setTimeout(check, 60); } };
  window.removeEventListener("scroll", armReveal.listener || (() => {}));
  armReveal.listener = onScroll;
  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", onScroll, { passive: true });
  clearInterval(armReveal.timer);
  armReveal.timer = setInterval(check, 500);   /* safety net for jumps that fire no scroll event */
  check();
}

function scrollToId(id, instant = false) {
  const el = document.getElementById(id);
  if (el) window.scrollTo({ top: el.getBoundingClientRect().top + window.scrollY - 80, behavior: instant ? "auto" : "smooth" });
}

const SLIDERS = [
  ["call_reach_rate", "Calls that connect", 0.3, 0.9, 0.05, "%"],
  ["call_effectiveness", "No-shows a connected call prevents", 0.05, 0.5, 0.05, "%"],
  ["sms_reach_rate", "SMS delivered", 0.5, 1, 0.05, "%"],
  ["sms_effectiveness", "No-shows a delivered SMS prevents", 0, 0.3, 0.01, "%"],
  ["cost_per_call", "Cost per call", 10, 100, 5, "INR"],
  ["value_per_kept_visit", "Value of a kept visit", 100, 1500, 50, "INR"],
];
const sliderText = (key, unit) => (unit === "%" ? pct(S.assume[key], 0) : "INR " + int(S.assume[key]));

function slidersHtml() {
  return SLIDERS.map(([key, label, min, max, step, unit]) => `<div class="field"><label for="a-${key}">${label} — <strong id="t-${key}">${sliderText(key, unit)}</strong></label>
    <input id="a-${key}" type="range" min="${min}" max="${max}" step="${step}" value="${S.assume[key]}" data-input="assume" data-key="${key}" data-unit="${unit}"></div>`).join("");
}

const STRATEGY_TEXT = {
  "Random (no targeting)": "Same budget, patients picked at random",
  "Lead-time baseline": "Longest-booked first: one rule",
  "Logistic regression": "MediReach risk score (selected)",
  "Random forest": "Barely different, harder to explain",
  "Oracle (perfect knowledge)": "Knows who will miss: an upper bound",
};

function strategiesHtml() {
  if (!S.impact) return `<p class="muted">Loading the comparison…</p>`;
  const selected = (S.summary.models.find((row) => row.model === S.summary.selected_model) || {}).label;
  return S.impact.strategies.map((row) => `<div class="card ${row.strategy === selected ? "bg-a1" : ""} lift" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,180px),1fr));gap:var(--space-4) var(--space-6);align-items:end">
    <div><div style="font-size:21px;font-weight:700">${esc(row.strategy)}</div><div class="small muted">${esc(STRATEGY_TEXT[row.strategy] || "")}</div></div>
    <div><p class="small muted" style="margin-bottom:var(--space-2)">No-shows prevented (est.)</p><div class="tile-value md">${int(row.prevented)}</div></div>
    <div><p class="small muted" style="margin-bottom:var(--space-2)">Net benefit (est.)</p><div class="tile-value md">${inr(row.net)}</div></div>
    <div><p class="small muted" style="margin-bottom:var(--space-2)">Per prevented no-show</p><div class="tile-value md">${row.cost_per_prevented == null ? "–" : inr(row.cost_per_prevented)}</div></div></div>`).join("");
}

function pilotHtml() {
  const p = S.pilot;
  if (!p) return `<p class="muted">Sizing the pilot…</p>`;
  return `<div class="grid" style="grid-template-columns:repeat(2,minmax(0,1fr))">
    <div class="card bg-page" style="grid-column:1 / -1"><p class="tile-label">Patients needed</p><div class="tile-value" style="font-size:clamp(56px,7vw,104px);letter-spacing:-0.04em">${int(p.total)}</div></div>
    <div class="card bg-page"><p class="tile-label">Called</p><div class="tile-value md">${int(p.called)}</div></div>
    <div class="card bg-page"><p class="tile-label">Held out</p><div class="tile-value md">${int(p.held_out)}</div></div>
    <div class="card bg-ink" style="grid-column:1 / -1"><p class="tile-label">Clinic days, at about ${int(p.daily_volume)} call-list patients a day</p><div class="tile-value">${int(p.days)}</div></div>
    <p class="small" style="grid-column:1 / -1;color:var(--color-neutral-800)">80% power, two-sided 5% test, no-show rate on the call list ${pct(p.p_control)} (test week).</p></div>`;
}

const calculatorNote = (id, waking) => {
  const box = document.getElementById(id);
  if (box) box.textContent = waking ? "Recalculating needs the server, which is waking up (free host). The figures update as soon as it answers." : "";
};

const refreshImpact = debounce(async () => {
  calculatorNote("strategies-note", false);
  try {
    const query = new URLSearchParams(Object.entries(S.assume).map(([k, v]) => [k, String(v)]));
    S.impact = await api("/api/public/impact?" + query);
    const box = $("#strategies"); if (box) box.innerHTML = strategiesHtml();
  } catch (e) { if (e.offline) { calculatorNote("strategies-note", true); setTimeout(refreshImpact, 8000); } else toast(e.message, true); }
}, 180);

const refreshPilot = debounce(async () => {
  calculatorNote("pilot-note", false);
  try {
    S.pilot = await api(`/api/public/pilot?reduction=${S.pilotIn.reduction / 100}&holdout=${S.pilotIn.holdout / 100}`);
    const box = $("#pilot"); if (box) box.innerHTML = pilotHtml();
  } catch (e) { if (e.offline) { calculatorNote("pilot-note", true); setTimeout(refreshPilot, 8000); } else toast(e.message, true); }
}, 180);

/* --------------------------------------------------------------------- login */
function renderLogin() {
  const s = S.status, hint = s.demo_password_hint;
  $("#app").innerHTML = `<div style="min-height:100vh;display:flex;flex-direction:column">
    <header class="row" style="padding:var(--space-3) clamp(16px,3vw,48px);gap:var(--space-3) var(--space-6)">
      <button class="brand" data-act="home">MediReach</button>${labels()}
      <button class="btn btn-ghost" data-act="home" style="margin-left:auto">← Public site</button></header>
    <div class="row" style="flex:1;max-width:1240px;width:100%;margin:0 auto;padding:clamp(32px,6vw,96px) var(--gutter);gap:clamp(40px,6vw,96px);align-items:center">
      <div style="flex:1 1 380px"><p class="eyebrow">Staff sign-in</p>
        <h1 style="font-family:var(--font-display);font-weight:400;font-size:clamp(52px,6.4vw,100px);line-height:.96;max-width:14ch;text-wrap:balance">The desk opens at eight.</h1>
        <p class="lede">Sign in with a demo account. Everything inside is a replay of 2016 data — no real patient is contacted.</p></div>
      ${s.user ? `<div class="login-card"><p class="body">Signed in as <strong>${esc(s.user.name)}</strong>.</p>
        <button class="btn btn-primary btn-lg" data-act="portal">Open portal</button><button class="btn btn-secondary" data-act="logout">Log out</button></div>` :
      `<form class="login-card" id="login-form" autocomplete="on">
        <div class="field"><label for="login-user">Username</label><input class="input" id="login-user" name="username" autocomplete="username" placeholder="admin, reception1 or reception2" value="${esc(S.prefill)}" required></div>
        <div class="field"><label for="login-pass">Password</label><input class="input" id="login-pass" name="password" type="password" autocomplete="current-password" placeholder="Password" required></div>
        <p class="error" id="login-error" role="alert" hidden></p>
        <button class="btn btn-primary btn-lg" type="submit">Sign in</button>
        <p class="small muted" style="margin-top:var(--space-2)">Demo accounts${hint ? ` · password <code>${esc(hint)}</code>` : ""}</p>
        <div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:var(--space-2)">
          ${s.demo_accounts.map((a) => `<button type="button" class="account-btn" data-act="use-account" data-user="${esc(a.username)}"><strong>${esc(a.name)}</strong><span class="small muted">${esc(a.role_label)}</span><span class="small" style="color:var(--color-accent-700)">${esc(a.username)}</span></button>`).join("")}
        </div></form>`}
    </div></div>`;
  const form = $("#login-form");
  if (form) {
    form.addEventListener("submit", (event) => { event.preventDefault(); signIn($("#login-user").value, $("#login-pass").value); });
    (S.prefill ? $("#login-pass") : $("#login-user")).focus();
  }
}

async function signIn(username, password) {
  const error = $("#login-error");
  try {
    const result = await api("/api/login", { username, password });
    token.set(result.token);
    S.prefill = "";
    go(result.user.role === "ADMIN" ? "admin/today" : "me/tasks");
  } catch (e) {
    if (error) { error.textContent = e.message; error.hidden = false; }
  }
}

/* --------------------------------------------------------------- portal shell */
function renderShell(path) {
  const s = S.status, user = s.user;
  const items = Object.entries(VIEWS).filter(([key]) => key.startsWith(user.role === "ADMIN" ? "admin/" : "me/"));
  $("#app").innerHTML = `<div class="portal">
    <header class="portal-header">
      <div class="row" style="gap:var(--space-3) var(--space-4)">
        <div class="row spacer" style="align-items:baseline"><button class="brand" data-act="home">MediReach</button><span class="role-label">${esc(user.role_label)}</span></div>
        <span class="clock-pill num" title="${s.demo_time ? "Demo time, set on Demo controls" : "Real clock, " + esc(s.timezone)}"><span class="dot ${s.demo_time ? "demo" : ""}"></span>${esc(s.now_label)} · replaying ${esc(s.replaying_label)}</span>
        ${labels()}
        <div class="acct"><button class="acct-btn" data-act="menu" aria-haspopup="menu" aria-expanded="${S.menu}"><span class="avatar">${esc(user.initials)}</span>
          <span style="display:flex;flex-direction:column;line-height:1.15"><strong style="font-size:14px">${esc(user.name)}</strong><span class="small muted">${esc(user.username)}</span></span><span aria-hidden="true">▾</span></button>
          ${S.menu ? `<div class="menu" role="menu"><p class="role-label" style="margin:var(--space-2) var(--space-3)">Switch account</p>
            ${s.demo_accounts.map((a) => `<button role="menuitem" data-act="switch" data-user="${esc(a.username)}"><span class="avatar" style="width:30px;height:30px">${esc(a.initials)}</span><span class="spacer" style="display:flex;flex-direction:column;line-height:1.2"><strong style="font-size:14px">${esc(a.name)}</strong><span class="small muted">${esc(a.role_label)}</span></span>${a.username === user.username ? '<span class="small" style="color:var(--color-accent-700)">Current</span>' : ""}</button>`).join("")}
            <button role="menuitem" class="danger" data-act="logout">Log out</button></div>` : ""}
        </div>
      </div>
      <nav class="pill-nav" aria-label="Pages">${items.map(([key, view]) => `<a href="#/${key}" ${key === path ? 'aria-current="page"' : ""}>${view.label}</a>`).join("")}</nav>
    </header>
    <main class="view" id="view"><p class="loading muted">Loading…</p></main></div>`;
}

const eyebrow = (text) => `<p class="eyebrow">${text}</p>`;
const tile = (label, value, extra = "", cls = "") => `<div class="card ${cls}"><p class="tile-label">${label}</p><div class="tile-value">${value}</div>${extra}</div>`;
const notice = (html, cls = "bg-b1") => `<div class="card card-lg ${cls}" style="margin-top:var(--space-6)">${html}</div>`;
const progressRow = (label, done, total) => `<div style="display:grid;grid-template-columns:minmax(64px,120px) minmax(0,1fr) 9ch;gap:var(--space-4);align-items:center">
  <span style="font-size:15px">${esc(label)}</span><div class="track sm"><div class="fill" style="width:${total ? Math.round((100 * done) / total) : 0}%"></div></div><span class="small num" style="text-align:right">${int(done)} of ${int(total)}</span></div>`;
const STATUS_LABEL = { PROPOSED: "Proposed — not approved", APPROVED: "Approved", CLOSED: "Closed" };
const ARM_LABEL = { CALL: "Call", HOLDOUT: "Held out", SMS: "SMS", NONE: "No contact" };

/* -------------------------------------------------------------- admin: today */
function vToday(d) {
  const next = d.next_plan, working = d.working_plan, b = d.briefing;
  const planTag = next ? `<span class="tag ${next.status === "PROPOSED" ? "tag-accent-2" : "tag-accent"}">${STATUS_LABEL[next.status]}</span>`
    : `<span class="tag ${d.plan_due ? "tag-accent-2" : "tag-neutral"}">${d.plan_due ? "Due now" : "Not yet due"}</span>`;
  return `${eyebrow(`${esc(d.now_label)} · replaying ${esc(d.replaying_label)}`)}<h1 class="page-title">Today</h1>
  ${d.is_open ? "" : notice(`<p class="body">The clinic is closed today. Next clinic day: <strong>${esc(d.next_clinic_day_label)}</strong>.</p>`)}
  <div class="row" style="margin-top:var(--space-8);align-items:stretch">
    <div class="card card-lg bg-ink stack" style="flex:2 1 340px;gap:var(--space-4)">
      <p class="muted">Clinic clock (${esc(S.status.timezone)})</p>
      <div class="num" style="font-weight:700;font-size:clamp(72px,9vw,128px);line-height:.9;letter-spacing:-0.04em">${esc(d.time_label)}</div>
      <div class="row-between"><span class="small muted">${S.status.demo_time ? "Demo time. " : ""}Bookings appear as their 2016 booking time passes.</span>
        <a class="btn btn-secondary" href="#/admin/demo">Demo controls</a></div>
    </div>
    <div class="grid grid-160 tiles-sm" style="flex:3 1 460px">
      ${tile("Appointments today", int(d.appointments_today), '<p class="tile-note small muted">Booked so far</p>')}
      ${tile("Same-day bookings so far", int(d.same_day_bookings_so_far))}
      ${tile(`Known for ${esc(d.next_clinic_day_label)}`, int(d.appointments_next_day_known))}
      ${tile("Feed batches this week", int(d.feed.batches), d.feed.last_rows != null ? `<p class="tile-note small muted">Last: ${int(d.feed.last_rows)} bookings, ${d.feed.last_failed_checks ? d.feed.last_failed_checks + " check(s) failed" : "all checks passed"}</p>` : "")}
    </div>
  </div>
  <div class="row" style="margin-top:var(--space-3);align-items:stretch">
    <div class="card card-lg bg-a1 stack" style="flex:1 1 340px;align-items:flex-start;gap:var(--space-4)">
      <div class="row-between" style="width:100%"><p class="muted">Plan for ${esc(d.next_clinic_day_label)}</p>${planTag}</div>
      <div style="font-weight:700;font-size:clamp(24px,2.4vw,32px);line-height:1.15">${next ? `${int(next.calls)} calls · ${int(next.sms)} SMS` : "No plan yet"}</div>
      <p class="body" style="font-size:16px">${next ? (next.status === "PROPOSED" ? "Review the list, choose the hold-out and approve to release tasks to reception." : `About ${int(next.expected_no_shows)} no-shows expected. Tasks are with reception.`) : (d.plan_due ? `Bookings up to ${String(d.plan_hour).padStart(2, "0")}:00 are in. Make the plan now.` : `The plan can be made from ${String(d.plan_hour).padStart(2, "0")}:00 on the clinic day before.`)}</p>
      <a class="btn btn-primary" href="#/admin/plan" style="margin-top:auto">${next && next.status !== "PROPOSED" ? "View plan" : "Review and approve"}</a>
    </div>
    <div class="card card-lg stack" style="flex:1.4 1 460px;gap:var(--space-4)">
      <div class="row-between"><p class="muted">Task progress${working ? ` · calls about ${esc(working.target_label)}` : ""}</p><span class="num" style="font-weight:700;font-size:20px">${int(d.progress.done)} of ${int(d.progress.total)} done</span></div>
      ${d.progress.total ? d.progress.rows.map((row) => progressRow(row.label, row.done, row.total)).join("") : `<p class="italic-note">Tasks appear here once a plan is approved; they are worked on the clinic day before the visits.</p>`}
    </div>
  </div>
  <section class="card card-lg bg-page stack" style="margin-top:var(--space-3);box-shadow:var(--shadow-md);padding:clamp(24px,3vw,40px);gap:var(--space-4)">
    <div class="row"><h3 class="spacer" style="font-size:26px">Daily briefing</h3>
      <span class="tag ${b.source === "AI" ? "tag-accent" : "tag-neutral"}">${b.source === "AI" ? "Written by AI" : "Template"}</span>
      ${d.ai_available && b.source !== "AI" ? `<button class="btn btn-secondary" data-act="brief" ${S.ui.briefBusy ? "disabled" : ""}>${S.ui.briefBusy ? "Asking Gemini…" : "Write it with AI"}</button>` : ""}</div>
    <p style="font-size:19px;font-weight:600">${esc(b.headline)}</p>
    <ul class="body" style="margin:0;padding-left:1.2em;max-width:72ch">${b.bullets.map((line) => `<li>${esc(line)}</li>`).join("")}</ul>
    <p class="small muted">${b.source === "AI" ? `Worded by ${esc(b.model)} from the numbers below; every number was checked against them.` : "Built from the numbers below." + (b.note ? " " + esc(b.note) : "")}</p>
    <details class="facts"><summary>The numbers behind this briefing</summary><table class="table" style="margin-top:var(--space-2)"><tbody>${Object.entries(d.facts).map(([k, v]) => `<tr><td>${esc(k.replace(/_/g, " "))}</td><td>${esc(v)}</td></tr>`).join("")}</tbody></table></details>
  </section>`;
}

/* --------------------------------------------------------------- admin: plan */
function vPlan(d) {
  const head = `${eyebrow(`Calls today about ${esc(d.target_label)} visits · replays ${esc(d.replays_label)}`)}<h1 class="page-title">Plan and approve</h1>
    <p class="body" style="margin-top:var(--space-4);max-width:64ch">MediReach proposes who to contact about the next clinic day. Change a channel (with a reason), choose the hold-out, then approve to send tasks to reception.</p>`;
  if (!d.plan) {
    const f = S.ui.staff || d.defaults;
    S.ui.staff = f;
    return `${head}<div class="card card-lg bg-a1 stack" style="margin-top:var(--space-8);max-width:900px;gap:var(--space-4)">
      <h3 style="font-size:24px">No plan yet for ${esc(d.target_label)}</h3>
      <div class="grid grid-200">
        <div class="field"><label for="st-r">Receptionists on shift</label><input class="input" id="st-r" type="number" min="1" max="20" value="${f.receptionists}" data-input="staff" data-key="receptionists"></div>
        <div class="field"><label for="st-c">Calls per receptionist</label><input class="input" id="st-c" type="number" min="10" max="300" step="10" value="${f.calls_per_receptionist}" data-input="staff" data-key="calls_per_receptionist"></div>
        <div class="field"><label for="st-s">SMS budget</label><input class="input" id="st-s" type="number" min="0" max="5000" step="20" value="${f.sms_budget}" data-input="staff" data-key="sms_budget"></div>
      </div>
      <p class="small muted" id="cap-text">Call capacity: ${int(f.receptionists * f.calls_per_receptionist)} calls. The demo has two receptionist logins; tasks are shared between them.</p>
      ${d.can_propose ? "" : `<p class="small" style="color:var(--color-accent-2-700)">Not yet: ${esc(d.why_not)}.</p>`}
      <div><button class="btn btn-primary btn-lg" data-act="propose" ${d.can_propose && !S.ui.busy ? "" : "disabled"}>${S.ui.busy ? "Scoring the appointments…" : "Propose the plan"}</button></div></div>`;
  }
  const p = d.plan, n = d.numbers, proposed = p.status === "PROPOSED";
  const mode = S.ui.mode || "PILOT";
  const band = proposed ? `<div class="field" style="flex:1 1 260px;max-width:420px"><span class="label">Hold-out — drawn at random at approval, by patient</span>
      <div class="seg" role="group" aria-label="Hold-out">${d.holdout_modes.map((m) => `<button aria-pressed="${m.mode === mode}" data-act="mode" data-mode="${m.mode}">${m.mode === "PILOT" ? "Pilot" : "Monitoring"} · ${pct(m.share, 0)}</button>`).join("")}</div>
      <p class="small muted" style="margin-top:6px">Costs about ${d.holdout_cost[mode].toFixed(1)} prevented no-shows on that day (illustrative). It is what it costs to learn whether calls work.</p></div>`
    : `<div class="stack" style="flex:1 1 260px;gap:6px"><span class="tag tag-accent" style="width:fit-content">Approved by ${esc(p.approved_by)} · ${p.holdout_mode === "PILOT" ? "Pilot" : "Monitoring"} ${pct(p.holdout_share, 0)}</span>
      <span class="small muted">Randomised: ${Object.entries(d.arms).map(([arm, count]) => `${ARM_LABEL[arm] || arm} ${int(count)}`).join(" · ")}</span></div>`;
  const rows = d.rows.map((r) => {
    const pending = S.ui.pending && S.ui.pending.patient_id === r.patient_id ? S.ui.pending : null;
    const channel = proposed ? `<div class="row" style="gap:4px">${["Call", "SMS", "None"].map((a) => `<button class="btn btn-sm ${(pending ? pending.action : r.action) === a ? "is-on" : "btn-ghost"}" data-act="channel" data-pid="${r.patient_id}" data-action="${a}">${a === "None" ? "Skip" : a}</button>`).join("")}</div>
      ${pending ? `<div class="reason-pick"><select class="input" id="reason-${r.patient_id}" aria-label="Reason for the change">${d.override_reasons.map((reason) => `<option>${esc(reason)}</option>`).join("")}</select>
        <button class="btn btn-sm btn-primary" data-act="save-override" data-pid="${r.patient_id}">Save</button><button class="btn btn-sm btn-ghost" data-act="cancel-override">Cancel</button></div>` : ""}`
      : `<span>${esc(r.action === "None" ? "No contact" : r.action)}</span>`;
    return `<tr style="opacity:${r.action === "None" ? 0.5 : r.arm === "HOLDOUT" ? 0.75 : 1}">
      <td class="num">${r.rank}</td><td class="r num">${pct(r.risk)}</td>
      <td><div>${r.patient_id}</div><div class="small muted">${esc(r.clinic)} · age ${int(r.age)}${r.appointments > 1 ? ` · ${r.appointments} appts` : ""}</div></td>
      <td class="r num">${int(r.lead_days)} d</td>
      <td class="small" style="max-width:280px;color:var(--color-neutral-800)">${r.reasons.map(esc).join(" · ")}${r.override ? `<div class="small" style="color:var(--color-accent-2-700)">Changed: ${esc(r.override)}${r.recommended !== r.action ? ` (was ${esc(r.recommended)})` : ""}</div>` : ""}</td>
      <td>${channel}</td><td>${r.arm ? `<span class="tag ${r.arm === "HOLDOUT" ? "tag-accent-2" : "tag-neutral"}">${ARM_LABEL[r.arm]}</span>` : '<span class="small muted">at approval</span>'}</td>
      <td class="small">${esc(r.assignee || "—")}</td></tr>`;
  }).join("");
  const late = proposed && d.late_count ? `<section class="card card-lg" style="margin-top:var(--space-4)"><h3 style="font-size:22px">Late bookings <span class="muted small">${int(d.late_count)} since the plan was made</span></h3>
      <div class="table-wrap bg-page" style="margin-top:var(--space-3)"><table class="table"><thead><tr><th>Appointment</th><th>Patient</th><th>Clinic</th><th class="r">Age</th><th>Booked</th><th>Add as</th></tr></thead><tbody>
      ${d.late_bookings.slice(0, 15).map((l) => `<tr><td class="num">${l.appointment_id}</td><td>${l.patient_id}</td><td>${esc(l.clinic)}</td><td class="r">${int(l.age)}</td><td>${esc(l.booked_label)}</td>
        <td><button class="btn btn-sm btn-ghost" data-act="late" data-id="${l.appointment_id}" data-action="Call">Call</button><button class="btn btn-sm btn-ghost" data-act="late" data-id="${l.appointment_id}" data-action="SMS">SMS</button></td></tr>`).join("")}</tbody></table></div></section>` : "";
  const last = Math.min(d.offset + d.page_size, d.patients);
  return `${head}
  <div class="card card-lg bg-a1 row" style="margin:var(--space-8) 0 var(--space-4);align-items:flex-end;gap:var(--space-6)">
    ${band}
    <div class="row num spacer" style="font-size:16px;gap:var(--space-2) var(--space-4)"><span><strong>${int(n.calls)}</strong> calls</span><span><strong>${int(n.sms)}</strong> SMS</span><span><strong>${int(n.patients)}</strong> patients</span><span><strong>${int(n.expected_no_shows)}</strong> expected no-shows</span></div>
    ${!proposed ? `<a class="btn btn-secondary" href="#/admin/team">Team board</a>`
      : S.ui.confirmApprove ? `<div class="stack" style="gap:6px;max-width:340px"><span class="small">The randomisation and the tasks are fixed once approved.</span>
          <div class="row" style="gap:var(--space-2)"><button class="btn btn-primary" data-act="approve" ${S.ui.busy ? "disabled" : ""}>${S.ui.busy ? "Approving…" : `Confirm: approve with ${mode === "PILOT" ? "50%" : "10%"} hold-out`}</button><button class="btn btn-ghost" data-act="cancel-approve">Cancel</button></div></div>`
      : `<button class="btn btn-primary btn-lg" data-act="ask-approve">Approve and send to reception</button>`}
  </div>
  <p class="small muted" style="margin-bottom:var(--space-3)">Plan ${p.plan_id}: ${int(n.appointments)} appointments known at ${esc(p.as_of_label)} (data time), model ${esc(p.model_version)}. ${pct(n.call_list_capture)} of expected no-shows are on the call list. Risk is recalibrated to the last clinic days.</p>
  <div class="table-wrap"><table class="table" style="min-width:1000px"><thead><tr><th>Rank</th><th class="r">Risk</th><th>Patient</th><th class="r">Booked ahead</th><th>Why on the list</th><th>Channel</th><th>Arm</th><th>Assigned to</th></tr></thead><tbody>${rows}</tbody></table></div>
  <div class="row-between" style="margin-top:var(--space-3)"><span class="small muted">Patients ${int(d.offset + 1)}–${int(last)} of ${int(d.patients)}, highest risk first.</span>
    <div class="row"><button class="btn btn-secondary btn-sm" data-act="page" data-step="-1" ${d.offset === 0 ? "disabled" : ""}>Previous</button><button class="btn btn-secondary btn-sm" data-act="page" data-step="1" ${last >= d.patients ? "disabled" : ""}>Next</button></div></div>
  ${late}`;
}

/* --------------------------------------------------------------- admin: team */
function vTeam(d) {
  const outcomeTag = (t) => t.status === "PENDING" ? '<span class="tag tag-outline">Open</span>'
    : t.status === "SKIPPED" ? '<span class="tag tag-neutral">Skipped</span>'
    : t.status === "NOT_ATTEMPTED" ? '<span class="tag tag-neutral">Not attempted</span>'
    : `<span class="tag ${["CONFIRMED", "SMS_SENT"].includes(t.outcome) ? "tag-accent" : t.outcome === "NO_ANSWER" || t.outcome === "WRONG_NUMBER" ? "tag-neutral" : "tag-accent-2"}">${esc(t.outcome_label)}</span>`;
  const head = `${eyebrow(d.plan ? `Calls about ${esc(d.plan.target_label)} · plan ${d.plan.plan_id}` : "No plan is being worked today")}<h1 class="page-title">Team board</h1>`;
  if (!d.plan) {
    return `${head}${notice(`<div class="row"><p class="body spacer">No tasks today. ${d.next_plan && d.next_plan.status === "PROPOSED" ? "The next plan is proposed but not approved." : "Approve the next clinic day's plan to hand work to reception."}</p><a class="btn btn-primary" href="#/admin/plan">Go to plan</a></div>`)}`;
  }
  return `${head}<div class="grid grid-440" style="margin-top:var(--space-8)">${d.team.map((person) => {
    const other = d.team.find((p) => p.username !== person.username);
    return `<section class="card card-lg stack" style="gap:var(--space-4)">
      <div class="row"><span class="avatar lg">${esc(person.initials)}</span><div class="spacer"><div style="font-size:20px;font-weight:700">${esc(person.name)}</div><div class="small muted">${esc(person.username)} · ${int(person.open)} open</div></div>
        <span class="num" style="font-weight:700;font-size:22px">${int(person.done)} of ${int(person.total)}</span></div>
      <div class="track sm"><div class="fill" style="width:${person.total ? Math.round((100 * person.done) / person.total) : 0}%"></div></div>
      <div class="task-list">${person.tasks.map((t) => `<div class="item"><span class="small muted num">${t.rank}</span>
        <div style="min-width:0"><div>${t.patient_id}</div><div class="small muted">${t.kind === "CALL" ? "Call" + (t.attempt > 1 ? `, attempt ${t.attempt}` : "") : "SMS"}${t.notes && t.status === "SKIPPED" ? " · " + esc(t.notes) : ""}</div></div>
        <div class="row" style="gap:var(--space-2);justify-content:flex-end">${outcomeTag(t)}${t.status === "PENDING" && other ? `<button class="btn btn-ghost btn-sm" data-act="move" data-id="${t.task_id}" data-to="${esc(other.username)}">Move to ${esc(other.name)}</button>` : ""}</div></div>`).join("")}</div>
      ${person.total > d.page_size ? `<p class="small muted">Showing ${d.page_size} of ${int(person.total)}: open tasks first, by priority.</p>` : ""}</section>`;
  }).join("")}</div>`;
}

/* ------------------------------------------------------------ admin: results */
function vResults(d) {
  const banner = d.source === "simulated" || d.source === "mixed" || d.setting === "simulated"
    ? notice(`<div class="row"><span class="tag tag-accent-2">Simulated</span><span class="body">Outcomes ${d.setting === "simulated" ? "are" : "include days"} simulated with a planted ${pct(d.planted_effect, 0)} cut in no-shows for reached would-be no-shows. This tests the method; it is not evidence that calls work. Switch it in Demo controls.</span></div>`)
    : `<p class="italic-note" style="margin-top:var(--space-6)">Real 2016 outcomes: nobody in 2016 was called, so the true effect is zero and any gap between the arms is chance. A correct analysis should show an interval that includes zero.</p>`;
  if (!d.closed_days) {
    return `${eyebrow("Call arm against the randomised hold-out")}<h1 class="page-title">Results</h1>${banner}
      ${notice(`<p class="body">No planned day has closed yet. Approve a plan, let reception work it, then move to the next clinic day (Demo controls): attendance is revealed once the day is over.</p>`, "bg-n1")}`;
  }
  const ci = d.difference_ci, diff = d.difference;
  let chart = "";
  if (ci) {
    const lim = Math.max(10, Math.ceil((Math.max(Math.abs(ci[0]), Math.abs(ci[1])) * 100) / 5) * 5);
    const pos = (x) => `${(((100 * x + lim) / (2 * lim)) * 100).toFixed(2)}%`;
    const excludes = ci[1] < 0 || ci[0] > 0;
    chart = `<section class="card card-lg" style="margin-top:var(--space-3)"><p class="muted" style="margin-bottom:var(--space-6)">Difference in no-show rate (call arm minus hold-out), percentage points, with 95% confidence interval</p>
      <div class="ci-chart" role="img" aria-label="Difference ${pts(diff)}, 95% CI ${pts(ci[0])} to ${pts(ci[1])}"><div class="axis"></div><div class="zero" style="left:${pos(0)}"></div>
        <div class="band" style="left:${pos(ci[0])};width:calc(${pos(ci[1])} - ${pos(ci[0])});background:${excludes ? "var(--color-accent)" : "var(--color-neutral-400)"}"></div><div class="point" style="left:${pos(diff)}"></div></div>
      <div class="row-between small muted num" style="margin:var(--space-2) var(--space-2) 0"><span>−${lim}</span><span>0</span><span>+${lim}</span></div>
      <p class="body" style="margin-top:var(--space-6);max-width:70ch">${ci[1] < 0 ? "The whole interval sits below zero: patients on the call arm missed fewer visits than the hold-out." : ci[0] > 0 ? "The whole interval sits above zero: the call arm missed more often. Check the setup before reading anything into it." : "The interval includes zero: no difference between the arms is shown yet."}</p></section>`;
  }
  const mix = Object.entries(d.outcome_mix);
  return `${eyebrow(`${int(d.closed_days)} closed day${d.closed_days === 1 ? "" : "s"} · intention to treat`)}<h1 class="page-title">Results</h1>
    <p class="body" style="margin-top:var(--space-4);max-width:64ch">Patients on the call list, split at random: called, or held out. Everyone assigned to the call arm counts, answered or not.</p>${banner}
    <div class="row" style="margin-top:var(--space-8);align-items:stretch">
      <div class="card card-lg bg-a1" style="flex:1 1 240px"><p class="tile-label">Call arm · no-show rate</p><div class="tile-value xl">${pct(d.no_show_call)}</div><p class="tile-note">${int(d.patients_call)} patients</p></div>
      <div class="card card-lg" style="flex:1 1 240px"><p class="tile-label">Hold-out · no-show rate</p><div class="tile-value xl">${pct(d.no_show_holdout)}</div><p class="tile-note">${int(d.patients_holdout)} patients, not called</p></div>
      <div class="card card-lg bg-ink" style="flex:1.4 1 360px"><p class="tile-label">Relative reduction in no-shows</p><div class="tile-value xl">${spct(d.relative_reduction)}</div>
        <p class="tile-note" style="color:var(--color-neutral-200)">${d.reduction_ci ? `95% CI ${spct(d.reduction_ci[0])} to ${spct(d.reduction_ci[1])}` : "Not defined yet"}${d.p_value != null ? ` · difference ${pts(diff)}, p = ${d.p_value.toFixed(3)}` : ""}</p></div>
    </div>${chart}
    <div class="grid grid-300" style="margin-top:var(--space-3)">
      <section class="card card-lg stack"><p class="muted">Is the pilot big enough?</p>
        ${d.required_holdout ? progressRow("Hold-out", Math.min(d.patients_holdout, d.required_holdout), d.required_holdout) + progressRow("Call arm", Math.min(d.patients_call, d.required_call), d.required_call)
          + `<p class="small muted">Patients needed for 80% power at 5% significance, at the current hold-out share.</p>` : `<p class="small muted">Shown once a plan with a hold-out is approved.</p>`}</section>
      <section class="card card-lg stack"><p class="muted">Operations</p>
        <div class="grid" style="grid-template-columns:repeat(3,minmax(0,1fr))">
          <div><p class="small muted">Reach</p><div class="tile-value md">${pct(d.reach, 0)}</div><p class="small muted">${int(d.patients_reached)} of ${int(d.patients_attempted)} (assumed ${pct(d.assumed_reach, 0)})</p></div>
          <div><p class="small muted">Slots released</p><div class="tile-value md">${int(d.released_slots)}</div></div>
          <div><p class="small muted">Overrides</p><div class="tile-value md">${int(Object.values(d.overrides).reduce((a, b) => a + b, 0))}</div></div></div>
        ${mix.length ? `<div class="row" style="gap:6px">${mix.map(([label, count]) => `<span class="chip small">${esc(label)} ${int(count)}</span>`).join("")}</div>` : ""}</section>
    </div>`;
}

/* ---------------------------------------------------------- admin: directory */
function vDirectory(d) {
  const day = S.ui.dirDay || "all";
  return `${eyebrow("Read-only · as the clinic knows it now")}<h1 class="page-title">Directory</h1>
    <div class="row" style="margin:var(--space-8) 0 var(--space-4);gap:var(--space-3) var(--space-4)">
      <input class="input" id="dir-q" value="${esc(S.ui.dirQ || "")}" data-input="dirq" placeholder="Search patient, appointment or clinic" aria-label="Search" style="flex:1 1 300px;max-width:520px;border-radius:999px;padding-left:20px">
      <div class="seg" role="group" aria-label="Day">${d.tabs.map((t) => `<button aria-pressed="${t.key === day}" data-act="dir-day" data-day="${t.key}">${esc(t.label)}</button>`).join("")}</div>
      <span class="small muted">${int(d.count)} appointments${d.count > d.shown ? `, first ${int(d.shown)} shown` : ""}</span></div>
    <div class="table-wrap"><table class="table" style="min-width:860px"><thead><tr><th>Appointment</th><th>Patient</th><th>Visit</th><th>Clinic</th><th class="r">Age</th><th class="r">Booked ahead</th><th>SMS</th><th>Status</th></tr></thead><tbody>
      ${d.rows.map((r) => `<tr><td class="num">${r.appointment_id}</td><td>${r.patient_id}</td><td style="white-space:nowrap">${esc(r.day_label)}</td><td>${esc(r.clinic)}</td><td class="r">${int(r.age)}</td><td class="r num">${int(r.lead_days)} d</td><td>${r.sms ? "Yes" : "No"}</td>
        <td><span class="tag ${r.status === "No-show" ? "tag-accent-2" : r.status === "Attended" ? "tag-accent" : "tag-neutral"}">${esc(r.status)}</span></td></tr>`).join("")}</tbody></table>
      ${d.rows.length ? "" : '<p class="italic-note" style="padding:var(--space-6) 0;text-align:center">No appointments match.</p>'}</div>
    <p class="small muted" style="margin-top:var(--space-3)">Only bookings made by now (replayed time) are listed; dates are shown on today's calendar. Attendance becomes known after the day closes.</p>`;
}

/* --------------------------------------------------------------- admin: demo */
function vDemo(d) {
  const effect = S.ui.effect ?? Math.round(100 * d.planted_effect);
  const reach = S.ui.reach ?? Math.round(100 * d.assumed_reach);
  return `${eyebrow("For demonstrations only · shared by every user")}<h1 class="page-title">Demo controls</h1>
  <div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(min(100%,340px),1fr));gap:var(--space-4);margin-top:var(--space-8)">
    <section class="card card-lg stack" style="gap:var(--space-4)"><h3 style="font-size:22px">Clinic clock</h3>
      <div class="num" style="font-weight:700;font-size:clamp(56px,6vw,88px);line-height:.9;letter-spacing:-0.04em">${esc(d.time_label)}</div>
      <p class="small muted">${esc(d.date_label)} · replaying ${esc(d.replaying_label)} · ${d.demo_time ? "demo time" : "real clock"}</p>
      <div class="row" style="gap:var(--space-2)">${[[-60, "−1 h"], [15, "+15 min"], [60, "+1 h"]].map(([m, l]) => `<button class="btn btn-secondary" data-act="time" data-kind="shift" data-minutes="${m}">${l}</button>`).join("")}</div>
      <div class="row" style="gap:var(--space-2)"><span class="small muted">Jump to</span>${["08:00", "12:00", "17:00"].map((t) => `<button class="btn btn-ghost" data-act="time" data-kind="set" data-time="${t}">${t}</button>`).join("")}</div>
      <div class="row" style="gap:var(--space-2)"><button class="btn btn-primary" data-act="time" data-kind="next_day">Next clinic day, 08:00</button><button class="btn btn-ghost" data-act="time" data-kind="real" ${d.demo_time ? "" : "disabled"}>Use the real clock</button></div>
      <p class="small muted">Moving to the next clinic day closes the day that has passed and reveals its attendance.</p></section>
    <section class="card card-lg bg-a1 stack" style="gap:var(--space-4)"><h3 style="font-size:22px">Simulate reception</h3>
      <p class="body">${d.working_plan ? `${int(d.open_tasks)} open of ${int(d.total_tasks)} tasks (calls about ${esc(d.working_plan.target_label)}).` : "No approved plan is being worked today."}</p>
      <div class="field"><label for="reach">Share of calls answered — <strong id="reach-text">${reach}%</strong></label><input id="reach" type="range" min="0" max="100" step="5" value="${reach}" data-input="reach"></div>
      <div class="row" style="gap:var(--space-2);margin-top:auto"><button class="btn btn-primary" data-act="simulate" data-limit="5" ${d.open_tasks ? "" : "disabled"}>Complete next 5</button><button class="btn btn-secondary" data-act="simulate" ${d.open_tasks ? "" : "disabled"}>Complete all</button></div>
      <p class="small muted">Answered calls mostly confirm; unanswered calls get a retry, then an SMS. Logged as "Simulated".</p></section>
    <section class="card card-lg bg-b1 stack" style="gap:var(--space-4)"><div class="row"><h3 class="spacer" style="font-size:22px">Outcomes after a day</h3>${d.source === "simulated" ? '<span class="tag tag-accent-2">Simulated</span>' : ""}</div>
      <label class="row" style="gap:var(--space-3);font-size:17px;align-items:flex-start;cursor:pointer"><input type="checkbox" id="planted" ${(S.ui.planted ?? d.source === "simulated") ? "checked" : ""} data-input="planted" style="width:22px;height:22px;flex:none">Use a simulation with a planted effect instead of the real 2016 outcomes</label>
      <div class="field"><label for="effect">Planted effect: share of reached would-be no-shows who attend — <strong id="effect-text">${effect}%</strong></label><input id="effect" type="range" min="0" max="100" step="5" value="${effect}" data-input="effect"></div>
      <div><button class="btn btn-primary" data-act="outcomes">Save</button></div>
      <p class="small" style="color:var(--color-neutral-800)">Applies to days revealed from now on, and marks every screen as Simulated while on.</p></section>
  </div>
  <section class="card card-lg" style="margin-top:var(--space-4)"><div class="row"><label class="row spacer" style="gap:var(--space-2);cursor:pointer"><input type="checkbox" id="reset-ok" data-input="reset-ok" ${S.ui.resetOk ? "checked" : ""}>I understand this deletes every plan, task, result and the audit trail (logins are kept).</label>
    <button class="btn btn-secondary" data-act="reset" ${S.ui.resetOk ? "" : "disabled"} style="color:var(--color-accent-2-700)">Reset the whole demo</button></div></section>`;
}

/* ----------------------------------------------------------- admin: backtest */
function vBacktest(d) {
  return `${eyebrow("Historical queue · test week · possible only because the days are past")}<h1 class="page-title">Backtest queue</h1>
    <div class="seg" role="group" aria-label="Day" style="margin-top:var(--space-8)">${d.days.map((x) => `<button aria-pressed="${x.key === d.day}" data-act="bt-day" data-day="${x.key}">${esc(x.label)}</button>`).join("")}</div>
    <div class="grid grid-160" style="margin:var(--space-6) 0">${tile("Visits", int(d.visits))}${tile("Calls", int(d.calls))}${tile("SMS", int(d.sms))}
      ${tile("No-shows, call list vs day", `${pct(d.call_rate)}<small> vs ${pct(d.day_rate)}</small>`, "", "bg-a1")}</div>
    <div class="table-wrap"><table class="table" style="min-width:820px"><thead><tr><th>Rank</th><th class="r">Risk</th><th>Patient</th><th>Clinic</th><th class="r">Booked ahead</th><th>Why on the list</th><th>What happened</th></tr></thead><tbody>
      ${d.rows.map((r) => `<tr><td>${r.rank}</td><td class="r num">${pct(r.risk)}</td><td>${r.patient_id}</td><td>${esc(r.clinic)}</td><td class="r num">${int(r.lead_days)} d</td><td class="small" style="max-width:300px;color:var(--color-neutral-800)">${r.reasons.map(esc).join(" · ")}</td>
        <td><span class="tag ${r.outcome === "No-show" ? "tag-accent-2" : "tag-accent"}">${r.outcome}</span></td></tr>`).join("")}</tbody></table></div>
    <p class="small muted" style="margin-top:var(--space-3)">Top ${d.rows.length} of ${int(d.calls)} calls on ${esc(d.day_label)}; stored run ${esc(d.run.run_id)}, model ${esc(d.run.model_version)}. Outcomes are what happened in 2016 — nobody was contacted.</p>`;
}

/* --------------------------------------------------------- reception: tasks */
function vTasks(d) {
  const head = `${eyebrow(`${esc(d.date_label)}${d.plan ? ` · calling about ${esc(d.plan.target_label)} visits` : ""}`)}<h1 class="page-title">My tasks</h1>`;
  if (!d.plan) return `${head}${notice(`<h2 class="display-sm">Waiting for today's plan.</h2><p class="body" style="margin-top:var(--space-4)">Your list appears as soon as the clinic manager approves it. In this demo, sign in as <strong>admin</strong> to approve.</p>`)}`;
  if (!d.next) return `${head}${notice(`<div class="stack" style="align-items:flex-start;gap:var(--space-4)"><h2 class="display-sm">All done for today.</h2><p class="body">${int(d.done)} tasks logged. Thank you.</p><a class="btn btn-primary" href="#/me/day">See my day</a></div>`, "bg-a1")}`;
  const n = d.next, polished = S.ui.script && S.ui.script.task_id === n.task_id ? S.ui.script : null;
  const script = polished && !polished.original ? polished.text : n.script;
  return `${head}<div class="row" style="align-items:flex-start;gap:var(--space-6) clamp(24px,3vw,48px);margin-top:var(--space-8)">
    <article class="card bg-page stack" style="flex:1.6 1 520px;min-width:0;border-radius:32px;box-shadow:var(--shadow-md);padding:clamp(24px,3vw,44px);gap:var(--space-6)">
      <div class="row"><span class="tag ${n.kind === "CALL" ? "tag-accent" : "tag-neutral"}">${n.kind === "CALL" ? "Call" + (n.attempt > 1 ? `, attempt ${n.attempt}` : "") : "SMS"}</span>
        <span class="small muted spacer">Task ${int(n.position)} of ${int(d.total)}</span><span class="small num">Risk <strong>${pct(n.risk, 0)}</strong></span></div>
      <div><div style="font-weight:700;font-size:clamp(28px,3vw,40px);line-height:1.1">Patient ${n.patient_id}</div>
        <div class="muted" style="font-size:16px;margin-top:var(--space-2)">Age ${int(n.age)} · visit ${esc(n.visit_label)}${n.appointments > 1 ? ` (${n.appointments} appointments)` : ""} · ${esc(n.clinic)} · booked ${int(n.lead_days)} days ahead</div></div>
      <div><p class="small muted" style="margin-bottom:var(--space-3)">Why this patient</p><div class="row" style="gap:var(--space-2)">${n.reasons.map((r) => `<span class="chip">${esc(r)}</span>`).join("")}</div></div>
      <div class="card stack" style="border-radius:22px;gap:var(--space-4)">
        <div class="row" style="gap:var(--space-2)"><span class="small muted spacer">${n.kind === "CALL" ? "Script" : "Message"}</span>
          ${polished && polished.source === "AI" && !polished.original ? `<span class="tag tag-accent">Written by AI</span><button class="btn btn-ghost btn-sm" data-act="use-original">Use original</button>` : ""}
          ${n.kind === "CALL" && d.ai_available && !(polished && polished.source === "AI" && !polished.original) ? `<button class="btn btn-secondary btn-sm" data-act="polish" data-id="${n.task_id}" ${S.ui.polishing ? "disabled" : ""}>${S.ui.polishing ? "Asking Gemini…" : "Polish with AI"}</button>` : ""}</div>
        <p style="font-size:19px;line-height:1.55">${esc(script)}</p>
        ${polished && polished.note ? `<p class="small" style="color:var(--color-accent-2-700)">${esc(polished.note)}</p>` : ""}
        ${polished && polished.source === "AI" && !polished.original ? '<p class="small muted">Only the reasons, the clinic and the day were sent to the AI: no patient id, name or age.</p>' : ""}</div>
      <div class="field"><label for="notes">Notes (optional)</label><input class="input" id="notes" maxlength="500" placeholder="Anything the next person should know"></div>
      <div><p class="small muted" style="margin-bottom:var(--space-3)">Outcome</p><div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:var(--space-2)">
        ${n.outcomes.map((o, i) => `<button class="btn ${i === 0 ? "btn-primary" : "btn-secondary"}" style="padding:12px 16px" data-act="record" data-id="${n.task_id}" data-outcome="${o.value}">${esc(o.label)}</button>`).join("")}</div></div>
      <details><summary class="small" style="cursor:pointer;color:var(--color-accent-700)">Skip this task</summary><div class="row" style="margin-top:var(--space-2)">
        <select class="input" id="skip-reason" style="max-width:340px" aria-label="Reason">${d.skip_reasons.map((r) => `<option>${esc(r)}</option>`).join("")}</select><button class="btn btn-secondary" data-act="skip" data-id="${n.task_id}">Skip</button></div></details>
    </article>
    <aside class="card card-lg stack" style="flex:1 1 280px;min-width:240px;gap:var(--space-2)">
      <div class="row-between"><p class="muted">Up next</p><span class="small num">${int(d.open)} left</span></div>
      ${d.up_next.length ? `<div class="task-list">${d.up_next.map((u) => `<div class="item"><span class="small muted">${u.rank}</span><div style="min-width:0"><div>${u.patient_id}</div><div class="small muted">${esc(u.clinic)}</div></div><span class="tag ${u.kind === "CALL" ? "tag-accent" : "tag-neutral"}">${u.kind === "CALL" ? "Call" + (u.attempt > 1 ? " " + u.attempt : "") : "SMS"}</span></div>`).join("")}</div>` : '<p class="italic-note">This is your last task.</p>'}
      <p class="small muted" style="margin-top:var(--space-2)">No answer: the call comes back once, then becomes an SMS.</p>
    </aside></div>`;
}

function vDay(d) {
  const head = `${eyebrow(esc(d.date_label))}<h1 class="page-title">My day</h1>`;
  if (!d.plan) return `${head}${notice('<p class="body">No tasks today yet.</p>', "bg-n1")}`;
  return `${head}<div class="grid grid-160" style="margin:var(--space-8) 0 var(--space-4)">
      ${tile("Tasks done", `${int(d.done)}<small> of ${int(d.total)}</small>`, "", "bg-a1")}${tile("Calls made", int(d.calls))}${tile("Will attend", int(d.confirmed))}${tile("SMS sent", int(d.sms))}</div>
    <section class="card card-lg stack" style="max-width:900px;gap:var(--space-2)"><p class="muted" style="margin-bottom:var(--space-2)">What I did today</p>
      ${d.log.length ? d.log.map((l) => `<div style="display:grid;grid-template-columns:52px minmax(0,1fr) auto;gap:var(--space-3);align-items:center;padding:var(--space-2) 0;font-size:15px">
        <span class="muted num">${esc(l.time || "")}</span><span>${l.kind === "CALL" ? "Called" : "Texted"} patient ${l.patient_id} · rank ${l.rank}${l.notes ? ` <span class="small muted">· ${esc(l.notes)}</span>` : ""}</span>
        <span class="tag ${l.status === "SKIPPED" ? "tag-neutral" : ["Will attend", "SMS sent"].includes(l.label) ? "tag-accent" : "tag-accent-2"}">${esc(l.label)}</span></div>`).join("") : '<p class="italic-note">Nothing logged yet.</p>'}
      ${d.skipped ? `<p class="small muted">${int(d.skipped)} skipped.</p>` : ""}</section>`;
}

/* -------------------------------------------------------------------- actions */
async function act(promise, after) {
  try { const result = await promise; if (after) after(result); return result; }
  catch (e) { toast(e.message, true); if (e.status === 401) go("login"); return null; }
}

const ACTIONS = {
  top: () => window.scrollTo({ top: 0, behavior: "smooth" }),
  home: () => go(""),
  scroll: (el) => { if (S.path === "" || S.path.startsWith("s/")) scrollToId(el.dataset.id); else go("s/" + el.dataset.id); },
  portal: () => {
    const u = S.status && S.status.user;
    go(u ? (u.role === "ADMIN" ? "admin/today" : "me/tasks") : token.get() ? "admin/today" : "login");
  },
  retry: () => route(),
  "reset-assume": () => { S.assume = { ...S.summary.impact.assumptions }; $("#sliders").innerHTML = slidersHtml(); refreshImpact(); },
  "use-account": (el) => {
    const hint = S.status.demo_password_hint;
    $("#login-user").value = el.dataset.user;
    if (hint) signIn(el.dataset.user, hint); else $("#login-pass").focus();
  },
  menu: () => { S.menu = !S.menu; renderShell(S.path); drawView(); },
  switch: (el) => { token.clear(); S.status = null; S.prefill = el.dataset.user; go("login"); },
  logout: () => { token.clear(); S.status = null; S.prefill = ""; go(""); },
  brief: async () => {
    S.ui.briefBusy = true; drawView();
    const result = await act(api("/api/admin/briefing", {}));
    S.ui.briefBusy = false;
    if (result && result.briefing.source !== "AI") toast(result.briefing.note || "AI text was not used.", true);
    loadView();
  },
  propose: async () => {
    const f = S.ui.staff;
    S.ui.busy = true; drawView();
    await act(api("/api/admin/plan/propose", { calls: f.receptionists * f.calls_per_receptionist, sms: f.sms_budget }), (data) => { S.data = data; });
    S.ui.busy = false; drawView();
  },
  mode: (el) => { S.ui.mode = el.dataset.mode; drawView(); },
  channel: (el) => {
    const pid = Number(el.dataset.pid), row = S.data.rows.find((r) => r.patient_id === pid);
    S.ui.pending = row && row.action === el.dataset.action ? null : { patient_id: pid, action: el.dataset.action };
    drawView();
  },
  "cancel-override": () => { S.ui.pending = null; drawView(); },
  "save-override": async (el) => {
    const pid = Number(el.dataset.pid), reason = $("#reason-" + pid).value;
    await act(api("/api/admin/plan/override", { plan_id: S.data.plan.plan_id, patient_id: pid, action: S.ui.pending.action, reason, offset: S.ui.offset || 0 }),
      (data) => { S.data = data; S.ui.pending = null; toast(`Patient ${pid}: changed (${reason}).`); });
    drawView();
  },
  late: async (el) => {
    await act(api("/api/admin/plan/late", { plan_id: S.data.plan.plan_id, appointment_id: Number(el.dataset.id), action: el.dataset.action, offset: S.ui.offset || 0 }),
      (data) => { S.data = data; toast("Late booking added."); });
    drawView();
  },
  "ask-approve": () => { S.ui.confirmApprove = true; drawView(); },
  "cancel-approve": () => { S.ui.confirmApprove = false; drawView(); },
  approve: async () => {
    const mode = S.ui.mode || "PILOT";
    S.ui.busy = true; drawView();
    await act(api("/api/admin/plan/approve", { plan_id: S.data.plan.plan_id, mode }), (data) => { S.data = data; S.ui.offset = 0; toast("Approved. Tasks are with reception."); });
    S.ui.busy = false; S.ui.confirmApprove = false; drawView();
  },
  page: (el) => { S.ui.offset = Math.max(0, (S.ui.offset || 0) + Number(el.dataset.step) * S.data.page_size); S.ui.pending = null; loadView(); },
  move: async (el) => {
    await act(api("/api/admin/team/reassign", { task_ids: [Number(el.dataset.id)], to: el.dataset.to }), (data) => { S.data = data; toast("Task moved."); });
    drawView();
  },
  "dir-day": (el) => { S.ui.dirDay = el.dataset.day; loadView(); },
  "bt-day": (el) => { S.ui.btDay = el.dataset.day; loadView(); },
  time: async (el) => {
    const body = { action: el.dataset.kind };
    if (el.dataset.minutes) body.minutes = Number(el.dataset.minutes);
    if (el.dataset.time) body.time = el.dataset.time;
    await act(api("/api/admin/demo/time", body), (data) => { S.data = data; (data.notes || []).forEach((note) => toast(note)); });
    S.status = await api("/api/status"); renderShell(S.path); drawView();
  },
  simulate: async (el) => {
    const body = { reach: (S.ui.reach ?? Math.round(100 * S.data.assumed_reach)) / 100 };
    if (el.dataset.limit) body.limit = Number(el.dataset.limit);
    await act(api("/api/admin/demo/simulate", body), (data) => { S.data = data; toast(`${int(data.completed)} tasks completed (simulated).`); });
    drawView();
  },
  outcomes: async () => {
    const planted = $("#planted").checked, effect = Number($("#effect").value) / 100;
    await act(api("/api/admin/demo/outcomes", { source: planted ? "simulated" : "historical", effect }), (data) => { S.data = data; toast(planted ? "Simulation on: results will be marked Simulated." : "Real 2016 outcomes."); });
    S.ui.planted = undefined; S.status = await api("/api/status"); renderShell(S.path); drawView();
  },
  reset: async () => {
    await act(api("/api/admin/demo/reset", { confirm: true }), (data) => { S.data = data; S.ui.resetOk = false; toast("The demo was reset."); });
    S.status = await api("/api/status"); renderShell(S.path); drawView();
  },
  record: async (el) => {
    const notes = ($("#notes") || {}).value || "";
    await act(api(`/api/me/tasks/${el.dataset.id}/outcome`, { outcome: el.dataset.outcome, notes }), (data) => { S.data = data; S.ui.script = null; });
    drawView();
  },
  skip: async (el) => {
    await act(api(`/api/me/tasks/${el.dataset.id}/skip`, { reason: $("#skip-reason").value }), (data) => { S.data = data; S.ui.script = null; toast("Task skipped."); });
    drawView();
  },
  polish: async (el) => {
    S.ui.polishing = true; drawView();
    const result = await act(api(`/api/me/tasks/${el.dataset.id}/polish`, {}));
    S.ui.polishing = false;
    if (result) S.ui.script = { task_id: Number(el.dataset.id), text: result.script, source: result.source, note: result.note, original: false };
    drawView();
  },
  "use-original": () => { if (S.ui.script) S.ui.script.original = true; drawView(); },
};

document.addEventListener("click", (event) => {
  const el = event.target.closest("[data-act]");
  if (el && ACTIONS[el.dataset.act] && !el.disabled) { event.preventDefault(); ACTIONS[el.dataset.act](el, event); return; }
  if (S.menu && !event.target.closest(".acct")) { S.menu = false; renderShell(S.path); drawView(); }
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && S.menu) { S.menu = false; renderShell(S.path); drawView(); }
});

const searchDirectory = debounce(() => loadView(), 300);
document.addEventListener("input", (event) => {
  const el = event.target, kind = el.dataset && el.dataset.input;
  if (!kind) return;
  if (kind === "assume") {
    S.assume[el.dataset.key] = Number(el.value);
    $("#t-" + el.dataset.key).textContent = sliderText(el.dataset.key, el.dataset.unit);
    refreshImpact();
  } else if (kind === "pilot") {
    S.pilotIn[el.dataset.key] = Number(el.value);
    $(el.dataset.key === "reduction" ? "#pilot-red-text" : "#pilot-hold-text").textContent = el.value + "%";
    refreshPilot();
  } else if (kind === "staff") {
    S.ui.staff = { ...S.ui.staff, [el.dataset.key]: Math.max(0, Number(el.value) || 0) };
    $("#cap-text").textContent = `Call capacity: ${int(S.ui.staff.receptionists * S.ui.staff.calls_per_receptionist)} calls. The demo has two receptionist logins; tasks are shared between them.`;
  } else if (kind === "dirq") {
    S.ui.dirQ = el.value; searchDirectory();
  } else if (kind === "reach") {
    S.ui.reach = Number(el.value); $("#reach-text").textContent = el.value + "%";
  } else if (kind === "effect") {
    S.ui.effect = Number(el.value); $("#effect-text").textContent = el.value + "%";
  } else if (kind === "planted") {
    S.ui.planted = el.checked;
  } else if (kind === "reset-ok") {
    S.ui.resetOk = el.checked; drawView();
  }
});

/* Keep live screens current: a quiet reload every minute while the tab is visible and nothing is being typed. */
setInterval(() => {
  const live = ["admin/today", "admin/team", "me/tasks", "me/day"];
  if (document.visibilityState !== "visible" || !live.includes(S.path) || S.ui.pending || S.ui.polishing || S.ui.busy) return;
  if (document.activeElement && ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
  api("/api/status").then((status) => { S.status = status; renderShell(S.path); return loadView(); }).catch(() => {});
}, 60000);

window.addEventListener("hashchange", route);
fetch(API + "/api/health").catch(() => {});   /* wake a sleeping free host as early as possible */
route();
