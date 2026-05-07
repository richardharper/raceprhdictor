import { useState, useEffect, useMemo } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";

// ── Race definitions ──────────────────────────────────────────────────────────
const RACES = [
  { id: "marathon", label: "Marathon",      short: "42.2K",  detail: "26.2 mi · 42.2 km", m: 42195   },
  { id: "half",     label: "Half Marathon", short: "21.1K",  detail: "13.1 mi · 21.1 km", m: 21097.5 },
  { id: "10k",      label: "10K",           short: "10K",    detail: "6.2 mi · 10 km",     m: 10000   },
  { id: "5k",       label: "5K",            short: "5K",     detail: "3.1 mi · 5 km",      m: 5000    },
];

// ── Design tokens ─────────────────────────────────────────────────────────────
const T = {
  bg:      "#F2F1EC",
  surface: "#FFFFFF",
  border:  "#E3E1DA",
  text:    "#111111",
  muted:   "#7A7870",
  faint:   "#B0AEA7",
  accent:  "#00C48C",
  accentBg:"#E8FAF4",
  dark:    "#111111",
  radius:  "8px",
  radiusLg:"12px",
};

// ── Auth / API ────────────────────────────────────────────────────────────────
const STRAVA_API = "https://www.strava.com/api/v3";
const CLIENT_ID  = import.meta.env.VITE_STRAVA_CLIENT_ID;

function redirectToStrava() {
  const url = new URL("https://www.strava.com/oauth/authorize");
  url.searchParams.set("client_id",    CLIENT_ID);
  url.searchParams.set("redirect_uri", `${window.location.origin}/auth/callback`);
  url.searchParams.set("response_type","code");
  url.searchParams.set("scope",        "activity:read_all");
  window.location.href = url.toString();
}

async function apiPost(action, body) {
  const res = await fetch("/api/strava", {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ action, ...body }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `API error ${res.status}`);
  return data;
}

async function fetchActivities(token) {
  const after = Math.floor((Date.now() - 183 * 24 * 60 * 60 * 1000) / 1000); // ~6 months
  let all = [], page = 1;
  while (true) {
    const res = await fetch(
      `${STRAVA_API}/athlete/activities?after=${after}&per_page=100&page=${page}`,
      { headers: { Authorization: `Bearer ${token}` } }
    );
    if (!res.ok) { if (res.status === 401) throw new Error("TOKEN_EXPIRED"); throw new Error(`Strava ${res.status}`); }
    const batch = await res.json();
    if (!Array.isArray(batch) || !batch.length) break;
    all = [...all, ...batch];
    if (batch.length < 100) break;
    page++;
  }
  return all;
}

async function fetchAthlete(token) {
  const res = await fetch(`${STRAVA_API}/athlete`, { headers: { Authorization: `Bearer ${token}` } });
  if (!res.ok) throw new Error("Could not fetch athlete");
  return res.json();
}

// ── LocalStorage helpers ──────────────────────────────────────────────────────
const ls  = { get: (k) => { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : null; } catch { return null; } },
              set: (k,v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
              del: (k)   => { try { localStorage.removeItem(k); } catch {} } };

// ── Helpers ───────────────────────────────────────────────────────────────────
const RUN_TYPES = ["Run", "TrailRun", "VirtualRun"];

function isRun(a) {
  return RUN_TYPES.includes(a.type || a.sport_type) &&
    a.distance >= 1000 &&
    a.moving_time > 0 &&
    (a.moving_time / (a.distance / 1609.34)) < 900;
}

// ── Polynomial regression (least squares) ────────────────────────────────────
// Fits log(time) = a + b·log(dist) + c·log(dist)² to personal-best data.
// Returns { a, b, c } or null if not enough points.
function fitCurve(points) {
  // points = [{ x: log(distM), y: log(timeSec) }, ...]
  if (points.length < 3) return null;
  const n = points.length;
  // Build basis: [1, x, x²]
  const X = points.map(p => [1, p.x, p.x * p.x]);
  const Y = points.map(p => p.y);
  // Normal equations: (XᵀX)β = XᵀY — solved directly for 3×3 system
  function dot(a, b) { return a.reduce((s, v, i) => s + v * b[i], 0); }
  const cols = [0,1,2].map(j => X.map(r => r[j]));
  const XtX  = [0,1,2].map(i => [0,1,2].map(j => dot(cols[i], cols[j])));
  const XtY  = [0,1,2].map(i => dot(cols[i], Y));
  // Gaussian elimination
  const M = XtX.map((r, i) => [...r, XtY[i]]);
  for (let col = 0; col < 3; col++) {
    let maxR = col;
    for (let r = col+1; r < 3; r++) if (Math.abs(M[r][col]) > Math.abs(M[maxR][col])) maxR = r;
    [M[col], M[maxR]] = [M[maxR], M[col]];
    if (Math.abs(M[col][col]) < 1e-12) return null;
    for (let r = 0; r < 3; r++) {
      if (r === col) continue;
      const f = M[r][col] / M[col][col];
      for (let k = col; k <= 3; k++) M[r][k] -= f * M[col][k];
    }
  }
  const [a, b, c] = [0,1,2].map(i => M[i][3] / M[i][i]);
  return { a, b, c };
}

// ── Core prediction engine ────────────────────────────────────────────────────
// Three-layer approach (inspired by Alex Gasconn's model):
//
// Layer 1 — Real Anchors: actual best efforts within ±10% of each distance.
//           These are weighted 3× — hard facts beat extrapolations.
//
// Layer 2 — Personal-best Riegel: for each distance bracket, find the best
//           effort and apply Riegel. Faster PBs get higher weight. Uses the
//           runner's own exponent derived from their PB spread where possible,
//           falling back to the standard 1.06.
//
// Layer 3 — Personal fatigue curve: polynomial regression on distance-bucketed
//           PBs. Derives your personal fatigue exponent and provides a fast/slow
//           range as honest uncertainty bounds.
//
// Final: pool all predictions, trim top/bottom 25%, weighted-average the middle
// 50%. Also returns { lo, hi } range from that trimmed set.

const DIST_BUCKETS = [
  { id: "5k",   min: 4750,  max: 5250,  label: "5K"   },
  { id: "10k",  min: 9500,  max: 10500, label: "10K"  },
  { id: "half", min: 20000, max: 22200, label: "Half" },
  { id: "mar",  min: 40000, max: 44400, label: "Mar"  },
  { id: "3k",   min: 2800,  max: 3200,  label: "3K"   },
  { id: "park", min: 4500,  max: 5500,  label: "Pk"   }, // overlaps 5K intentionally
];

function extractPBs(runs) {
  // For each distance bucket, find the fastest effort (highest speed).
  const pbs = {};
  for (const bucket of DIST_BUCKETS) {
    const inBucket = runs.filter(r => r.distance >= bucket.min && r.distance <= bucket.max);
    if (!inBucket.length) continue;
    const best = inBucket.reduce((b, r) => (r.distance / r.moving_time > b.distance / b.moving_time) ? r : b);
    // Normalise to the exact bucket target distance via Riegel so different
    // distances within a bucket are comparable.
    const targetD = (bucket.min + bucket.max) / 2;
    const normTime = best.moving_time * Math.pow(targetD / best.distance, 1.06);
    pbs[bucket.id] = { distM: targetD, timeSec: normTime, rawRun: best };
  }
  // Also store any run longer than 15K that isn't in a standard bucket as a
  // "long effort" anchor — useful for marathon predictions.
  const longRuns = runs.filter(r => r.distance > 15000 &&
    !DIST_BUCKETS.some(b => r.distance >= b.min && r.distance <= b.max));
  if (longRuns.length) {
    const best = longRuns.reduce((b, r) => (r.distance / r.moving_time > b.distance / b.moving_time) ? r : b);
    pbs["long"] = { distM: best.distance, timeSec: best.moving_time, rawRun: best };
  }
  return pbs;
}

function riegelPredict(fromTimeSec, fromDistM, toDistM, exponent = 1.06) {
  return fromTimeSec * Math.pow(toDistM / fromDistM, exponent);
}

function deriveExponent(pbs) {
  // Use two well-separated PBs to derive a personal fatigue exponent:
  //   exponent = log(T2/T1) / log(D2/D1)
  const entries = Object.values(pbs).sort((a, b) => a.distM - b.distM);
  if (entries.length < 2) return 1.06;
  // Use the furthest-apart pair for stability.
  const lo = entries[0], hi = entries[entries.length - 1];
  const exp = Math.log(hi.timeSec / lo.timeSec) / Math.log(hi.distM / lo.distM);
  // Clamp to a sensible range — bad data can produce extreme values.
  return Math.min(Math.max(exp, 0.95), 1.20);
}

function predictRace(activities, targetM) {
  const runs = activities.filter(isRun);
  if (!runs.length) return null;

  const pbs = extractPBs(runs);
  const pbEntries = Object.values(pbs);
  const personalExp = deriveExponent(pbs);

  const allPredictions = [];

  // ── Layer 1: Real Anchors (±10% of target distance) ──────────────────────
  const anchorRuns = runs.filter(r => {
    const ratio = r.distance / targetM;
    return ratio >= 0.90 && ratio <= 1.10;
  });
  // Sort by speed, take top 3.
  const topAnchors = anchorRuns
    .sort((a, b) => (b.distance / b.moving_time) - (a.distance / a.moving_time))
    .slice(0, 3);
  for (const r of topAnchors) {
    const days    = (Date.now() - new Date(r.start_date)) / 86400000;
    const recency = Math.exp(-days / 120);       // slower decay for anchors — a marathon from 4 months ago is still gold
    const pred    = riegelPredict(r.moving_time, r.distance, targetM, personalExp);
    allPredictions.push({ time: pred, weight: 3.0 * recency }); // 3× base weight
  }

  // ── Layer 2: Personal-best Riegel from each distance bucket ──────────────
  for (const pb of pbEntries) {
    if (pb.distM === targetM) continue; // handled in Layer 1
    const days    = (Date.now() - new Date(pb.rawRun.start_date)) / 86400000;
    const recency = Math.exp(-days / 90);
    const speed   = pb.distM / pb.timeSec;
    // Prefer PBs at distances closer to target.
    const ratio   = targetM / pb.distM;
    const distSim = Math.exp(-Math.abs(Math.log(ratio)) * 0.4);
    const pred    = riegelPredict(pb.timeSec, pb.distM, targetM, personalExp);
    allPredictions.push({ time: pred, weight: 1.5 * recency * distSim * speed });
  }

  // ── Layer 3: Polynomial fatigue curve ────────────────────────────────────
  if (pbEntries.length >= 3) {
    const points = pbEntries.map(pb => ({
      x: Math.log(pb.distM),
      y: Math.log(pb.timeSec),
    }));
    const curve = fitCurve(points);
    if (curve) {
      const lx   = Math.log(targetM);
      const logy = curve.a + curve.b * lx + curve.c * lx * lx;
      const pred = Math.exp(logy);
      // Compute residual std-dev for uncertainty.
      const residuals = points.map(p => {
        const yhat = curve.a + curve.b * p.x + curve.c * p.x * p.x;
        return p.y - yhat;
      });
      const mse = residuals.reduce((s, r) => s + r*r, 0) / residuals.length;
      const se  = Math.sqrt(mse);
      // Weight by how tight the fit is — lower se = higher confidence.
      const confidence = Math.exp(-se * 5);
      allPredictions.push({ time: pred, weight: 1.2 * confidence });
      // Also push fast/slow bounds as lower-weight data points.
      allPredictions.push({ time: Math.exp(logy - se), weight: 0.4 * confidence });
      allPredictions.push({ time: Math.exp(logy + se), weight: 0.4 * confidence });
    }
  }

  if (!allPredictions.length) return null;

  // ── Ensemble: trim outliers, weighted-average the middle 50% ─────────────
  allPredictions.sort((a, b) => a.time - b.time);
  const trim = Math.floor(allPredictions.length * 0.25);
  const middle = allPredictions.slice(trim, allPredictions.length - trim || undefined);
  if (!middle.length) return null;

  let totalW = 0, weightedT = 0;
  for (const p of middle) { weightedT += p.time * p.weight; totalW += p.weight; }
  return totalW > 0 ? weightedT / totalW : null;
}

// Also export a version that returns { time, lo, hi } for the UI range display.
function predictRaceWithRange(activities, targetM) {
  const runs = activities.filter(isRun);
  if (!runs.length) return null;

  const pbs = extractPBs(runs);
  const pbEntries = Object.values(pbs);
  const personalExp = deriveExponent(pbs);
  const allPredictions = [];

  const anchorRuns = runs
    .filter(r => { const ratio = r.distance / targetM; return ratio >= 0.90 && ratio <= 1.10; })
    .sort((a, b) => (b.distance / b.moving_time) - (a.distance / a.moving_time))
    .slice(0, 3);
  for (const r of anchorRuns) {
    const days = (Date.now() - new Date(r.start_date)) / 86400000;
    const recency = Math.exp(-days / 120);
    allPredictions.push({ time: riegelPredict(r.moving_time, r.distance, targetM, personalExp), weight: 3.0 * recency });
  }
  for (const pb of pbEntries) {
    if (pb.distM === targetM) continue;
    const days = (Date.now() - new Date(pb.rawRun.start_date)) / 86400000;
    const recency = Math.exp(-days / 90);
    const distSim = Math.exp(-Math.abs(Math.log(targetM / pb.distM)) * 0.4);
    allPredictions.push({ time: riegelPredict(pb.timeSec, pb.distM, targetM, personalExp), weight: 1.5 * recency * distSim });
  }
  if (pbEntries.length >= 3) {
    const points = pbEntries.map(pb => ({ x: Math.log(pb.distM), y: Math.log(pb.timeSec) }));
    const curve = fitCurve(points);
    if (curve) {
      const lx = Math.log(targetM);
      const logy = curve.a + curve.b * lx + curve.c * lx * lx;
      const residuals = points.map(p => { const yhat = curve.a + curve.b * p.x + curve.c * p.x * p.x; return p.y - yhat; });
      const se = Math.sqrt(residuals.reduce((s, r) => s + r*r, 0) / residuals.length);
      const confidence = Math.exp(-se * 5);
      allPredictions.push({ time: Math.exp(logy), weight: 1.2 * confidence });
      allPredictions.push({ time: Math.exp(logy - se), weight: 0.4 * confidence });
      allPredictions.push({ time: Math.exp(logy + se), weight: 0.4 * confidence });
    }
  }
  if (!allPredictions.length) return null;

  allPredictions.sort((a, b) => a.time - b.time);
  const trim = Math.floor(allPredictions.length * 0.25);
  const middle = allPredictions.slice(trim, allPredictions.length - trim || undefined);
  if (!middle.length) return null;

  let totalW = 0, weightedT = 0;
  for (const p of middle) { weightedT += p.time * p.weight; totalW += p.weight; }
  const time = totalW > 0 ? weightedT / totalW : null;
  const lo = middle[0].time;
  const hi = middle[middle.length - 1].time;
  return time ? { time, lo, hi } : null;
}

function buildChartData(activities, targetM) {
  const pts = [];
  for (let i = 13; i >= 0; i--) {
    const d = new Date(); d.setDate(d.getDate() - i); d.setHours(23,59,59,0);
    const subset = activities.filter(a => new Date(a.start_date) <= d);
    const pred   = predictRace(subset, targetM);
    if (pred != null) pts.push({ d: d.toLocaleDateString("en-US",{month:"short",day:"numeric"}), s: Math.round(pred) });
  }
  return pts;
}

// ── Format helpers ────────────────────────────────────────────────────────────
const KM = 1.60934;

function fmtTime(s) {
  s = Math.round(Math.abs(s));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return h > 0
    ? `${h}:${String(m).padStart(2,"0")}:${String(sec).padStart(2,"0")}`
    : `${m}:${String(sec).padStart(2,"0")}`;
}

function fmtDiff(s) {
  const abs = Math.abs(Math.round(s));
  const m = Math.floor(abs/60), sec = abs%60;
  const str = m > 0 ? `${m}m ${sec}s` : `${sec}s`;
  return s > 0 ? `↑ ${str}` : `↓ ${str}`;
}

function fmtPace(secPerMi, unit) {
  const s = Math.round(unit === "km" ? secPerMi / KM : secPerMi);
  return `${Math.floor(s/60)}:${String(s%60).padStart(2,"0")}/${unit}`;
}

function fmtDist(mi, unit) {
  return `${(unit === "km" ? mi * KM : mi).toFixed(2)} ${unit}`;
}

function fmtDate(iso) {
  return new Date(iso).toLocaleDateString("en-GB",{weekday:"short",day:"numeric",month:"short"});
}

function actType(a) {
  const t = (a.sport_type || a.type || "").toLowerCase();
  if (["walk","hike"].includes(t)) return "walk";
  if (a.workout_type === 1 || (a.distance > 40000 && t === "run")) return "race";
  if (/interval|tempo|track|strides|800|1200|400/.test((a.name||"").toLowerCase())) return "intervals";
  if (t === "run") return "run";
  return "other";
}

// ── Icons ─────────────────────────────────────────────────────────────────────
const RunIcon      = () => <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#A09E97" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><circle cx="14" cy="4" r="1.5"/><path d="M8 9.5L11.5 7l4 2 2-2.5"/><path d="M10 20L12.5 14l3.5 2 2.5-3"/><path d="M11.5 7L10 13 7 15"/></svg>;
const WalkIcon     = () => <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#A09E97" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="3.5" r="1.5"/><path d="M9.5 7L8 13l4 2-1.5 4"/><path d="M9.5 7L12 12l4.5-2"/><path d="M12 12l2 4"/></svg>;
const RaceIcon     = () => <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#E6960C" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><circle cx="14" cy="4" r="1.5"/><path d="M8 9.5L11.5 7l4 2 2-2.5"/><path d="M10 20L12.5 14l3.5 2 2.5-3"/><path d="M11.5 7L10 13 7 15"/><polyline points="4 5 8 5"/><polyline points="4 9 7 9"/></svg>;
const IntIcon      = () => <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#818CF8" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="2 12 5 6 8 15 11 9 14 12 17 6 20 12"/></svg>;
const OtherIcon    = () => <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#A09E97" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="8"/><path d="M12 8v4l3 3"/></svg>;
const AIcon = ({ type }) => { if (type==="walk") return <WalkIcon/>; if (type==="race") return <RaceIcon/>; if (type==="intervals") return <IntIcon/>; if (type==="run") return <RunIcon/>; return <OtherIcon/>; };

// ── Logo ──────────────────────────────────────────────────────────────────────
function Logo({ dark }) {
  const color = dark ? "white" : T.text;
  return (
    <div style={{ display:"flex", alignItems:"center", gap:"9px" }}>
      <svg width="26" height="26" viewBox="0 0 26 26" fill="none">
        <circle cx="13" cy="13" r="11" stroke={T.accent} strokeWidth="2"/>
        <polyline points="13 7 13 13 17 15" stroke={T.accent} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        <circle cx="13" cy="13" r="1.5" fill={T.accent}/>
      </svg>
      <span style={{ fontWeight:"800", fontSize:"15px", color, letterSpacing:"-0.4px" }}>Race<span style={{ color: T.accent }}>Predictor</span></span>
    </div>
  );
}

// ── Chart tooltip ─────────────────────────────────────────────────────────────
function ChartTip({ active, payload, label, distM, unit }) {
  if (!active || !payload?.length) return null;
  const timeSec = payload[0].value;
  // Pace = time / distance-in-chosen-unit
  const distInUnit = unit === "km" ? distM / 1000 : distM / 1609.34;
  const paceSecPerUnit = distInUnit > 0 ? timeSec / distInUnit : 0;
  return (
    <div style={{ background:"white", border:`1px solid ${T.border}`, borderRadius:T.radius, padding:"9px 14px", boxShadow:"0 4px 14px rgba(0,0,0,0.08)", minWidth:"120px" }}>
      <div style={{ color:T.accent, fontWeight:"700", fontSize:"16px", fontVariantNumeric:"tabular-nums" }}>{fmtTime(timeSec)}</div>
      {paceSecPerUnit > 0 && (
        <div style={{ color:T.muted, fontSize:"12px", fontVariantNumeric:"tabular-nums", marginTop:"2px" }}>
          {fmtPace(paceSecPerUnit, unit)} avg
        </div>
      )}
      <div style={{ color:T.faint, fontSize:"11px", marginTop:"3px" }}>{label}</div>
    </div>
  );
}

// ── Race prediction card ──────────────────────────────────────────────────────
function RaceCard({ race, pred, prevPred, selected, onClick }) {
  // pred / prevPred are { time, lo, hi } | null
  const diff = pred && prevPred ? pred.time - prevPred.time : null;
  return (
    <button onClick={onClick} style={{
      flex:1, textAlign:"left", background: selected ? T.accentBg : T.surface,
      border: `1.5px solid ${selected ? T.accent : T.border}`,
      borderRadius: T.radiusLg, padding:"18px 20px", cursor:"pointer",
      transition:"all .15s", outline:"none", display:"flex", flexDirection:"column", gap:"6px",
      boxShadow: selected ? `0 0 0 3px ${T.accent}22` : "none",
    }}>
      <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between" }}>
        <span style={{ fontSize:"11px", fontWeight:"700", color: selected ? T.accent : T.faint, letterSpacing:"0.1em", textTransform:"uppercase" }}>{race.label}</span>
        {selected && <span style={{ width:"7px", height:"7px", borderRadius:"50%", background:T.accent, flexShrink:0 }}/>}
      </div>
      <div style={{ fontSize:"30px", fontWeight:"800", color: T.text, letterSpacing:"-1px", lineHeight:1, fontVariantNumeric:"tabular-nums" }}>
        {pred ? fmtTime(pred.time) : <span style={{ color:T.faint }}>–:––:––</span>}
      </div>
      {pred && pred.lo && pred.hi && pred.lo !== pred.hi && (
        <div style={{ fontSize:"10px", color:T.faint, fontVariantNumeric:"tabular-nums" }}>
          {fmtTime(pred.lo)} – {fmtTime(pred.hi)}
        </div>
      )}
      <div style={{ display:"flex", alignItems:"center", gap:"8px" }}>
        <span style={{ fontSize:"11px", color:T.faint }}>{race.detail}</span>
        {diff !== null && (
          <span style={{ fontSize:"11px", fontWeight:"600", color: diff > 0 ? "#EF4444" : T.accent }}>
            {fmtDiff(diff)}
          </span>
        )}
      </div>
    </button>
  );
}

// ── Connect screen ────────────────────────────────────────────────────────────
function ConnectScreen({ error }) {
  return (
    <div style={{ minHeight:"100vh", background:T.dark, display:"flex", flexDirection:"column" }}>
      <div style={{ flex:1, display:"flex", alignItems:"center", justifyContent:"center", padding:"40px 24px" }}>
        <div style={{ width:"100%", maxWidth:"440px" }}>
          <div style={{ marginBottom:"40px" }}><Logo dark/></div>

          <h1 style={{ fontSize:"42px", fontWeight:"900", color:"white", letterSpacing:"-1.5px", lineHeight:1.1, marginBottom:"16px" }}>
            Know your<br/>
            <span style={{ color:T.accent }}>real pace.</span>
          </h1>
          <p style={{ fontSize:"15px", color:"rgba(255,255,255,0.5)", lineHeight:"1.7", marginBottom:"40px" }}>
            Predicts your Marathon, Half, 10K, and 5K finish times from your Strava training data — updated every time you sync.
          </p>

          {error && (
            <div style={{ background:"rgba(239,68,68,0.1)", border:"1px solid rgba(239,68,68,0.3)", borderRadius:T.radius, padding:"12px 16px", marginBottom:"20px", fontSize:"13px", color:"#F87171" }}>
              {error}
            </div>
          )}

          <button onClick={redirectToStrava} style={{ display:"flex", alignItems:"center", justifyContent:"center", gap:"10px", width:"100%", padding:"14px 24px", background:T.accent, color:"white", border:"none", borderRadius:T.radiusLg, fontSize:"15px", fontWeight:"700", cursor:"pointer", letterSpacing:"-0.2px" }}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="white">
              <path d="M15.387 17.944l-2.089-4.116h-3.065L15.387 24l5.15-10.172h-3.066m-7.008-5.599l2.836 5.598h4.172L10.463 0l-7 13.828h4.169"/>
            </svg>
            Connect with Strava
          </button>
          <p style={{ fontSize:"12px", color:"rgba(255,255,255,0.25)", marginTop:"14px", textAlign:"center" }}>
            Redirects to Strava to authorize read access to your activities.
          </p>
        </div>
      </div>
      <style>{`@keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}`}</style>
    </div>
  );
}

// ── Main app ──────────────────────────────────────────────────────────────────
export default function App() {
  const [phase,       setPhase]       = useState("init");
  const [tokenData,   setTokenData]   = useState(null);
  const [activities,  setActivities]  = useState([]);
  const [athlete,     setAthlete]     = useState(null);
  const [unit,        setUnit]        = useState(() => ls.get("rp-unit") || "mi");
  const [selectedRace,setSelectedRace]= useState("marathon");
  const [syncing,     setSyncing]     = useState(false);
  const [syncMsg,     setSyncMsg]     = useState("");
  const [goalRace,    setGoalRace]    = useState(() => ls.get("rp-goal") || null);
  const [showGoal,    setShowGoal]    = useState(false);
  const [goalForm,    setGoalForm]    = useState({ name:"", date:"", target:"" });
  const [err,         setErr]         = useState("");

  // ── Boot ────────────────────────────────────────────────────────────────────
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const code   = params.get("code");
    const oerr   = params.get("error");
    if (code || oerr) window.history.replaceState({}, "", window.location.pathname);
    if (oerr) { setErr("Strava authorization was cancelled."); setPhase("connect"); return; }
    if (code) { handleCodeExchange(code); return; }
    restoreSession();
  }, []);

  async function restoreSession() {
    const stored = ls.get("rp-token");
    if (!stored?.access_token) { setPhase("connect"); return; }
    let token = stored;
    if (stored.expires_at * 1000 < Date.now() + 60000) {
      if (!stored.refresh_token) { ls.del("rp-token"); setPhase("connect"); return; }
      try { token = await apiPost("refresh", { refresh_token: stored.refresh_token }); ls.set("rp-token", token); }
      catch  { ls.del("rp-token"); setPhase("connect"); return; }
    }
    setTokenData(token);
    const acts = ls.get("rp-acts"); const ath = ls.get("rp-athlete");
    if (acts?.length) { setActivities(acts); if (ath) setAthlete(ath); setPhase("dashboard"); }
    else await doSync(token.access_token);
  }

  async function handleCodeExchange(code) {
    setPhase("loading");
    try {
      const data = await apiPost("exchange", { code });
      ls.set("rp-token", data); setTokenData(data);
      if (data.athlete) { setAthlete(data.athlete); ls.set("rp-athlete", data.athlete); }
      await doSync(data.access_token);
    } catch(e) { setErr(e.message); setPhase("connect"); }
  }

  async function doSync(accessToken) {
    setSyncing(true); setSyncMsg("Syncing…"); setErr("");
    try {
      const [acts, ath] = await Promise.all([fetchActivities(accessToken), fetchAthlete(accessToken)]);
      setActivities(acts); setAthlete(ath);
      ls.set("rp-acts", acts); ls.set("rp-athlete", ath);
      setSyncMsg("Up to date"); setPhase("dashboard");
    } catch(e) {
      if (e.message === "TOKEN_EXPIRED") {
        const st = ls.get("rp-token");
        if (st?.refresh_token) {
          try { const ref = await apiPost("refresh",{ refresh_token: st.refresh_token }); ls.set("rp-token",ref); setTokenData(ref); await doSync(ref.access_token); return; } catch {}
        }
        ls.del("rp-token"); setPhase("connect");
      } else { setErr(e.message); setPhase("dashboard"); }
    } finally { setSyncing(false); setTimeout(() => setSyncMsg(""), 5000); }
  }

  async function handleSync() {
    if (syncing || !tokenData) return;
    let token = tokenData;
    if (token.expires_at * 1000 < Date.now() + 60000) {
      if (!token.refresh_token) { disconnect(); return; }
      try { token = await apiPost("refresh",{refresh_token:token.refresh_token}); setTokenData(token); ls.set("rp-token",token); } catch { disconnect(); return; }
    }
    await doSync(token.access_token);
  }

  function disconnect() {
    ["rp-token","rp-acts","rp-athlete"].forEach(ls.del);
    setTokenData(null); setActivities([]); setAthlete(null); setPhase("connect");
  }

  function setUnit2(u) { setUnit(u); ls.set("rp-unit", u); }

  // ── Computed ────────────────────────────────────────────────────────────────
  // Full predictions with lo/hi range for display.
  const predictions = useMemo(() => {
    if (!activities.length) return {};
    return Object.fromEntries(RACES.map(r => [r.id, predictRaceWithRange(activities, r.m)]));
  }, [activities]);

  // Predictions from 7 days ago — used to show trend direction on each card.
  const prevPredictions = useMemo(() => {
    if (!activities.length) return {};
    const sevenDaysAgo = new Date(); sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7); sevenDaysAgo.setHours(23,59,59,0);
    const subset = activities.filter(a => new Date(a.start_date) <= sevenDaysAgo);
    if (!subset.length) return {};
    return Object.fromEntries(RACES.map(r => [r.id, predictRaceWithRange(subset, r.m)]));
  }, [activities]);

  const selectedM   = RACES.find(r => r.id === selectedRace).m;
  const chartData   = useMemo(() => activities.length ? buildChartData(activities, selectedM) : [], [activities, selectedM]);

  const sixMonthsAgo = Date.now() - 183 * 24 * 60 * 60 * 1000;
  const recentRuns  = useMemo(() =>
    activities.filter(a => ["Run","TrailRun","VirtualRun"].includes(a.sport_type||a.type) && new Date(a.start_date).getTime() >= sixMonthsAgo),
  [activities]);

  const { totalMi, avgPaceSec, runCount } = useMemo(() => {
    const mi  = recentRuns.reduce((s,a) => s + a.distance/1609.34, 0);
    const sec = recentRuns.reduce((s,a) => s + a.moving_time, 0);
    return { totalMi: mi, avgPaceSec: mi > 0 ? sec/mi : 0, runCount: recentRuns.length };
  }, [recentRuns]);

  const chartMin = chartData.length ? Math.min(...chartData.map(d=>d.s)) - 60 : 0;
  const chartMax = chartData.length ? Math.max(...chartData.map(d=>d.s)) + 60 : 3600;
  const chartTicks = useMemo(() => {
    if (!chartData.length) return [];
    const step = Math.ceil((chartMax - chartMin) / 3 / 15) * 15;
    return [0,1,2,3].map(i => Math.round(chartMin + i * step));
  }, [chartData, chartMin, chartMax]);

  const sortedActs = useMemo(() => [...activities].sort((a,b) => new Date(b.start_date)-new Date(a.start_date)), [activities]);

  // The predicted time for whichever race is set as the goal target.
  const goalRacePred = useMemo(() => {
    if (!goalRace?.race) return null;
    return predictions[goalRace.race]?.time ?? null;
  }, [goalRace, predictions]);

  // ── Render phases ────────────────────────────────────────────────────────────
  if (phase === "init" || phase === "loading") return (
    <div style={{ display:"flex", alignItems:"center", justifyContent:"center", minHeight:"100vh", background:T.dark }}>
      <style>{`@keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}`}</style>
      <div style={{ textAlign:"center" }}>
        <div style={{ width:"34px", height:"34px", border:`3px solid ${T.accent}`, borderTopColor:"transparent", borderRadius:"50%", animation:"spin .8s linear infinite", margin:"0 auto 14px" }}/>
        <p style={{ fontSize:"13px", color:"rgba(255,255,255,0.4)" }}>{phase === "loading" ? "Connecting to Strava…" : "Loading…"}</p>
      </div>
    </div>
  );

  if (phase === "connect") return <ConnectScreen error={err}/>;

  // ── Dashboard ────────────────────────────────────────────────────────────────
  return (
    <div style={{ fontFamily:"system-ui,-apple-system,BlinkMacSystemFont,sans-serif", background:T.bg, minHeight:"100vh", color:T.text }}>
      <style>{`
        @keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
        .row:hover{background:#F9F8F4}
        .sl{color:#FC4C02;text-decoration:none;font-size:12px;font-weight:500}
        .sl:hover{text-decoration:underline}
        .gh:hover{background:rgba(255,255,255,0.1)}
        .ghw:hover{background:#F4F3EE}
        input:focus{outline:none;border-color:${T.accent} !important;box-shadow:0 0 0 2px ${T.accent}30}
      `}</style>

      {/* ── Header (dark) ── */}
      <header style={{ background:T.dark, padding:"13px 24px", display:"flex", alignItems:"center", justifyContent:"space-between", position:"sticky", top:0, zIndex:10 }}>
        <Logo dark/>
        <div style={{ display:"flex", alignItems:"center", gap:"10px" }}>
          {/* Unit toggle */}
          <div style={{ display:"flex", background:"rgba(255,255,255,0.08)", borderRadius:"6px", padding:"2px" }}>
            {["mi","km"].map(u => (
              <button key={u} onClick={() => setUnit2(u)} style={{ padding:"4px 11px", fontSize:"12px", fontWeight:u===unit?"600":"400", background:u===unit?"rgba(255,255,255,0.15)":"transparent", border:"none", borderRadius:"4px", cursor:"pointer", color:u===unit?"white":"rgba(255,255,255,0.45)", transition:"all .15s" }}>{u}</button>
            ))}
          </div>
          {/* Sync */}
          <button onClick={handleSync} className="gh" style={{ display:"flex", alignItems:"center", gap:"6px", padding:"6px 13px", fontSize:"12px", fontWeight:"500", background:"transparent", border:"1px solid rgba(255,255,255,0.18)", borderRadius:"6px", cursor:syncing?"default":"pointer", color:"rgba(255,255,255,0.75)", transition:"all .15s" }}>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" style={{ animation:syncing?"spin .9s linear infinite":"none" }}>
              <polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/>
              <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
            </svg>
            {syncMsg || "Sync"}
          </button>
          {/* Athlete */}
          {athlete && (
            <div style={{ display:"flex", alignItems:"center", gap:"7px" }}>
              {athlete.profile_medium
                ? <img src={athlete.profile_medium} alt="" style={{ width:"28px", height:"28px", borderRadius:"50%", objectFit:"cover", border:"1.5px solid rgba(255,255,255,0.15)" }}/>
                : <div style={{ width:"28px", height:"28px", borderRadius:"50%", background:T.accent, display:"flex", alignItems:"center", justifyContent:"center", fontSize:"11px", color:"white", fontWeight:"700" }}>{(athlete.firstname?.[0]||"")+(athlete.lastname?.[0]||"")}</div>}
              <span style={{ fontSize:"12px", color:"rgba(255,255,255,0.65)", fontWeight:"500" }}>{athlete.firstname}</span>
            </div>
          )}
          {/* Disconnect */}
          <button onClick={disconnect} className="gh" style={{ display:"flex", alignItems:"center", padding:"6px 8px", background:"transparent", border:"1px solid rgba(255,255,255,0.1)", borderRadius:"6px", cursor:"pointer", color:"rgba(255,255,255,0.3)", fontSize:"11px", gap:"4px" }}>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/></svg>
            Disconnect
          </button>
        </div>
      </header>

      <div style={{ maxWidth:"1060px", margin:"0 auto", padding:"28px 24px 56px" }}>

        {/* Page title */}
        <div style={{ marginBottom:"22px" }}>
          <h1 style={{ fontSize:"20px", fontWeight:"800", letterSpacing:"-0.5px", margin:0 }}>Performance Predictions</h1>
          <p style={{ fontSize:"13px", color:T.muted, marginTop:"3px" }}>Based on your last 6 months of Strava activities.</p>
        </div>

        {err && <div style={{ background:"#FEF2F2", border:"1px solid #FCA5A5", borderRadius:T.radius, padding:"10px 14px", marginBottom:"16px", fontSize:"13px", color:"#B91C1C" }}>{err}</div>}

        {/* ── 4 Race prediction cards ── */}
        <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:"12px", marginBottom:"16px" }}>
          {RACES.map(race => (
            <RaceCard key={race.id} race={race}
              pred={predictions[race.id] ?? null}
              prevPred={prevPredictions[race.id] ?? null}
              selected={selectedRace === race.id}
              onClick={() => setSelectedRace(race.id)}
            />
          ))}
        </div>

        {/* ── Chart + stats row ── */}
        <div style={{ display:"grid", gridTemplateColumns:"1fr 180px", gap:"12px", marginBottom:"16px", alignItems:"stretch" }}>

          {/* Chart */}
          <div style={{ background:T.surface, border:`1px solid ${T.border}`, borderRadius:T.radiusLg, padding:"18px 20px" }}>
            <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:"4px" }}>
              <span style={{ fontSize:"13px", fontWeight:"600" }}>
                {RACES.find(r=>r.id===selectedRace).label} trend
              </span>
              <span style={{ fontSize:"11px", color:T.faint }}>Last 14 days</span>
            </div>

            {chartData.length > 1 ? (
              <ResponsiveContainer width="100%" height={190}>
                <LineChart data={chartData} margin={{ top:12, right:8, left:4, bottom:0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#EDEBE5" vertical={false}/>
                  <XAxis dataKey="d" axisLine={false} tickLine={false} tick={{ fontSize:10.5, fill:T.faint }}/>
                  <YAxis domain={[chartMin, chartMax]} ticks={chartTicks} tickFormatter={v => fmtTime(v)} axisLine={false} tickLine={false} width={56} tick={{ fontSize:10.5, fill:T.faint }}/>
                  <Tooltip content={<ChartTip distM={selectedM} unit={unit}/>}/>
                  <Line type="monotone" dataKey="s" stroke={T.accent} strokeWidth={2.5} dot={false} activeDot={{ r:5, fill:T.accent, stroke:"white", strokeWidth:2 }}/>
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ height:"190px", display:"flex", alignItems:"center", justifyContent:"center", color:T.faint, fontSize:"13px", gap:"8px" }}>
                {syncing ? <><div style={{ width:"18px", height:"18px", border:`2px solid ${T.accent}`, borderTopColor:"transparent", borderRadius:"50%", animation:"spin .8s linear infinite" }}/>Loading…</> : "Sync activities to see the trend"}
              </div>
            )}
          </div>

          {/* Stats column */}
          <div style={{ display:"flex", flexDirection:"column", gap:"10px" }}>
            {[
              { label:"6-month distance", value: totalMi > 0 ? fmtDist(totalMi, unit) : "–" },
              { label:"Avg run pace",    value: avgPaceSec > 0 ? fmtPace(avgPaceSec, unit) : "–" },
              { label:"Runs logged",     value: runCount > 0 ? `${runCount} runs` : "–" },
            ].map(s => (
              <div key={s.label} style={{ background:T.surface, border:`1px solid ${T.border}`, borderRadius:T.radius, padding:"14px 16px", flex:1 }}>
                <div style={{ fontSize:"11px", color:T.faint, fontWeight:"500", marginBottom:"5px", textTransform:"uppercase", letterSpacing:"0.06em" }}>{s.label}</div>
                <div style={{ fontSize:"17px", fontWeight:"800", letterSpacing:"-0.5px", fontVariantNumeric:"tabular-nums" }}>{s.value}</div>
              </div>
            ))}
          </div>
        </div>

        {/* ── Goal Race ── */}
        <div style={{ background:T.surface, border:`1px solid ${T.border}`, borderRadius:T.radiusLg, padding:"20px 22px", marginBottom:"12px" }}>
          <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:"16px" }}>
            <div>
              <span style={{ fontSize:"11px", color:T.faint, fontWeight:"600", letterSpacing:"0.08em", textTransform:"uppercase" }}>Goal Race</span>
              <h2 style={{ fontSize:"14px", fontWeight:"700", margin:"2px 0 0" }}>Target</h2>
            </div>
            {!goalRace && !showGoal && (
              <button onClick={() => setShowGoal(true)} className="ghw" style={{ display:"flex", alignItems:"center", gap:"5px", padding:"6px 14px", fontSize:"12px", fontWeight:"500", background:"white", border:`1px solid ${T.border}`, borderRadius:T.radius, cursor:"pointer", color:T.muted }}>
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
                Add goal race
              </button>
            )}
          </div>

          {!goalRace && !showGoal && (
            <div style={{ textAlign:"center", padding:"16px 0 20px", color:T.faint, fontSize:"13px" }}>
              No goal race set. Add one to see your predicted time vs target.
            </div>
          )}

          {showGoal && !goalRace && (
            <div style={{ maxWidth:"440px", display:"flex", flexDirection:"column", gap:"12px" }}>
              <div>
                <label style={{ fontSize:"12px", color:T.muted, fontWeight:"600", display:"block", marginBottom:"5px" }}>Race Name</label>
                <input type="text" value={goalForm.name} onChange={e => setGoalForm({...goalForm,name:e.target.value})} placeholder="e.g. London Marathon 2027" style={{ width:"100%", padding:"8px 12px", border:`1px solid ${T.border}`, borderRadius:T.radius, fontSize:"13px", boxSizing:"border-box", background:"white" }}/>
              </div>
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:"10px" }}>
                <div>
                  <label style={{ fontSize:"12px", color:T.muted, fontWeight:"600", display:"block", marginBottom:"5px" }}>Distance</label>
                  <select value={goalForm.race || "marathon"} onChange={e => setGoalForm({...goalForm,race:e.target.value})} style={{ width:"100%", padding:"8px 10px", border:`1px solid ${T.border}`, borderRadius:T.radius, fontSize:"13px", background:"white" }}>
                    {RACES.map(r => <option key={r.id} value={r.id}>{r.label}</option>)}
                  </select>
                </div>
                <div>
                  <label style={{ fontSize:"12px", color:T.muted, fontWeight:"600", display:"block", marginBottom:"5px" }}>Date</label>
                  <input type="date" value={goalForm.date} onChange={e => setGoalForm({...goalForm,date:e.target.value})} style={{ width:"100%", padding:"8px 10px", border:`1px solid ${T.border}`, borderRadius:T.radius, fontSize:"13px", boxSizing:"border-box", background:"white" }}/>
                </div>
                <div>
                  <label style={{ fontSize:"12px", color:T.muted, fontWeight:"600", display:"block", marginBottom:"5px" }}>Target time</label>
                  <input type="text" value={goalForm.target} onChange={e => setGoalForm({...goalForm,target:e.target.value})} placeholder="e.g. 1:45:00" style={{ width:"100%", padding:"8px 10px", border:`1px solid ${T.border}`, borderRadius:T.radius, fontSize:"13px", boxSizing:"border-box", fontFamily:"monospace", background:"white" }}/>
                </div>
              </div>
              <div style={{ display:"flex", gap:"8px" }}>
                <button onClick={() => { if (!goalForm.name||!goalForm.date) return; const g={...goalForm}; ls.set("rp-goal",g); setGoalRace(g); setShowGoal(false); }} style={{ padding:"8px 20px", background:T.accent, color:"white", border:"none", borderRadius:T.radius, fontSize:"13px", fontWeight:"600", cursor:"pointer" }}>Save</button>
                <button onClick={() => setShowGoal(false)} className="ghw" style={{ padding:"8px 14px", background:"white", color:T.muted, border:`1px solid ${T.border}`, borderRadius:T.radius, fontSize:"13px", cursor:"pointer" }}>Cancel</button>
              </div>
            </div>
          )}

          {goalRace && (() => {
            const raceObj  = RACES.find(r => r.id === goalRace.race) || RACES[0];
            const pred     = predictions[raceObj.id];
            const raceDate = goalRace.date ? new Date(goalRace.date + "T00:00:00") : null;
            const daysLeft = raceDate ? Math.ceil((raceDate - new Date()) / 86400000) : null;
            return (
              <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", flexWrap:"wrap", gap:"16px" }}>
                <div style={{ display:"flex", alignItems:"center", gap:"14px" }}>
                  <div style={{ width:"42px", height:"42px", borderRadius:T.radius, background:T.accentBg, border:`1.5px solid ${T.accent}40`, display:"flex", alignItems:"center", justifyContent:"center", flexShrink:0 }}>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke={T.accent} strokeWidth="1.8"><path d="M4 5h16v6l-4 3 4 3v4H4v-4l4-3-4-3V5z"/></svg>
                  </div>
                  <div>
                    <div style={{ fontWeight:"700", fontSize:"14px" }}>{goalRace.name}</div>
                    <div style={{ fontSize:"12px", color:T.faint, marginTop:"2px" }}>
                      {raceObj.label}
                      {raceDate && ` · ${raceDate.toLocaleDateString("en-GB",{day:"numeric",month:"short",year:"numeric"})}`}
                      {daysLeft != null && daysLeft > 0 && ` · ${daysLeft} days to go`}
                    </div>
                  </div>
                </div>
                <div style={{ display:"flex", alignItems:"center", gap:"24px" }}>
                  {goalRace.target && (
                    <div style={{ textAlign:"right" }}>
                      <div style={{ fontSize:"11px", color:T.faint, fontWeight:"500", textTransform:"uppercase", letterSpacing:"0.06em" }}>Target</div>
                      <div style={{ fontSize:"20px", fontWeight:"800", letterSpacing:"-0.5px", fontVariantNumeric:"tabular-nums" }}>{goalRace.target}</div>
                    </div>
                  )}
                  {goalRacePred && (
                    <div style={{ textAlign:"right" }}>
                      <div style={{ fontSize:"11px", color:T.accent, fontWeight:"600", textTransform:"uppercase", letterSpacing:"0.06em" }}>Predicted</div>
                      <div style={{ fontSize:"20px", fontWeight:"800", letterSpacing:"-0.5px", color:T.accent, fontVariantNumeric:"tabular-nums" }}>{fmtTime(goalRacePred)}</div>
                    </div>
                  )}
                  <button onClick={() => { ls.del("rp-goal"); setGoalRace(null); }} className="ghw" style={{ padding:"5px 12px", fontSize:"12px", color:T.faint, background:"white", border:`1px solid ${T.border}`, borderRadius:T.radius, cursor:"pointer" }}>Remove</button>
                </div>
              </div>
            );
          })()}
        </div>

        {/* ── Activities table ── */}
        <div style={{ background:T.surface, border:`1px solid ${T.border}`, borderRadius:T.radiusLg, padding:"20px 22px" }}>
          <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:"16px" }}>
            <h2 style={{ fontSize:"14px", fontWeight:"700", margin:0 }}>Recent Activities</h2>
            {sortedActs.length > 0 && <span style={{ fontSize:"12px", color:T.faint }}>{sortedActs.length} in the last 6 months</span>}
          </div>

          {sortedActs.length === 0 ? (
            <div style={{ textAlign:"center", padding:"28px 0", color:T.faint, fontSize:"13px" }}>
              {syncing ? <span style={{ display:"flex", alignItems:"center", justifyContent:"center", gap:"8px" }}><div style={{ width:"14px", height:"14px", border:`2px solid ${T.accent}`, borderTopColor:"transparent", borderRadius:"50%", animation:"spin .8s linear infinite" }}/>Loading activities…</span> : "No activities in the last 6 months."}
            </div>
          ) : (
            <div style={{ overflowX:"auto" }}>
              <table style={{ width:"100%", borderCollapse:"collapse", minWidth:"520px" }}>
                <thead>
                  <tr style={{ borderBottom:`1px solid ${T.bg}` }}>
                    {["Name","Distance","Pace","Heart Rate",""].map((h,i) => (
                      <th key={i} style={{ textAlign: i > 0 && i < 4 ? "right":"left", fontSize:"11px", fontWeight:"600", color:T.faint, padding:"0 0 10px", paddingLeft: i===0 ? "8px":0, textTransform:"uppercase", letterSpacing:"0.06em", width: i===4?"1%":undefined }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sortedActs.map((a, i) => {
                    const at      = actType(a);
                    const distMi  = a.distance / 1609.34;
                    const paceSec = distMi > 0 ? a.moving_time / distMi : 0;
                    return (
                      <tr key={a.id} className="row" style={{ borderBottom: i < sortedActs.length-1 ? `1px solid ${T.bg}` : "none", transition:"background .1s" }}>
                        <td style={{ padding:"9px 8px" }}>
                          <div style={{ display:"flex", alignItems:"center", gap:"9px" }}>
                            <div style={{ width:"30px", height:"30px", borderRadius:"7px", background:T.bg, border:`1px solid ${T.border}`, display:"flex", alignItems:"center", justifyContent:"center", flexShrink:0 }}>
                              <AIcon type={at}/>
                            </div>
                            <div>
                              <div style={{ fontSize:"13px", fontWeight:"600", lineHeight:1.3 }}>{a.name}</div>
                              <div style={{ fontSize:"11px", color:T.faint, marginTop:"1px" }}>{fmtDate(a.start_date_local || a.start_date)}</div>
                            </div>
                          </div>
                        </td>
                        <td style={{ textAlign:"right", fontSize:"13px", color:T.muted, padding:"9px 8px", whiteSpace:"nowrap", fontVariantNumeric:"tabular-nums" }}>{fmtDist(distMi, unit)}</td>
                        <td style={{ textAlign:"right", fontSize:"13px", color:T.muted, padding:"9px 8px", whiteSpace:"nowrap", fontVariantNumeric:"tabular-nums" }}>{paceSec > 0 && paceSec < 1800 ? fmtPace(paceSec, unit) : "–"}</td>
                        <td style={{ textAlign:"right", fontSize:"13px", color:T.muted, padding:"9px 8px", whiteSpace:"nowrap", fontVariantNumeric:"tabular-nums" }}>{a.average_heartrate ? `${Math.round(a.average_heartrate)} bpm` : "–"}</td>
                        <td style={{ textAlign:"right", padding:"9px 0 9px 12px", whiteSpace:"nowrap" }}>
                          <a href={`https://www.strava.com/activities/${a.id}`} target="_blank" rel="noreferrer" className="sl">Strava ↗</a>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
