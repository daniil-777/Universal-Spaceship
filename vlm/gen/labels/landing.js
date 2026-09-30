// vlm/gen/labels/landing.js — landing (L) labels (spec §4.2 L, §4.3 landing, §7.3 L): landingNow() reduces the sim to
// the combinator's inputs exactly as autoland.js:92-99 evaluates them, landingFacts() writes the facts, and replays from
// createLandingSim(conditions(seed)) give the CONTINUE and GO_AROUND branches (review probe_landing_ga.mjs).
import { createLandingSim } from '../../../src/landing/sim.js';
import { bodyToWorld } from '../../../src/landing/flight.js';
import { KT, FT } from '../../../src/landing/vehicle.js';
import { AIRPORT, RWY } from '../../../src/landing/airport.js';
import { landingSafety, eyeView } from '../safety.js';
import { fact } from '../schema.js';
import { project, cameraPosition } from './camera.js';

export const H_L = 1 / 120;
const DEG = Math.PI / 180, GSA = 3 * DEG, r2 = (x) => (Number.isFinite(x) ? +x.toFixed(2) : null);
export const stepOf = (sim) => Math.round(sim.flight.t / H_L);
export const papiWhites = (eye) => AIRPORT.papi.reduce((n, u) => n + (Math.atan2(eye[1] - u.y, Math.hypot(eye[0] - u.x, eye[2] - u.z)) > u.angle ? 1 : 0), 0);
export const lOutcome = (rep) => (rep.result === 'go-around' ? 'go_around' : rep.result);

export function landingNow(sim) {
  const f = sim.flight, A = f.air, st = sim.gnc.st, d = st.dev || {}, P = sim.gnc.params, ft = A.hRA / FT, wc = sim.wind.components(10);
  const vG = st.vG ?? A.V, vMin = P.vapp - 5 * KT, vMax = Math.max(P.vapp + 15 * KT, (st.vTgt ?? P.vapp) + 10 * KT), vsLim = -Math.max(1000, A.gs * Math.tan(GSA) / FT * 60 + 300);
  const gate = { lateral: st.lat === 'LOC', vertical: st.vert === 'GS', loc: Math.abs(d.loc ?? 9) < 1, gs: Math.abs(d.gs ?? 9) < 1, speed: vG >= vMin && vG <= vMax, vs: (st.vzS ?? f.v[1]) > vsLim * FT / 60, gear: f.gear > 0.99 };
  return { airborne: !f.wow, wow: f.wow, vert: st.vert, lat: st.lat, hRAft: ft, thrNm: Math.max(0, -f.p[0]) / 1852, locDots: d.locValid ? d.loc : 0, gsDots: d.gsValid ? d.gs : 0,
    crossing500: st.lastFt > 500 && ft <= 500, gate, speedErrKt: (vG < vMin ? vG - vMin : vG > vMax ? vG - vMax : 0) / KT, crossKt: wc.cross, tailKt: Math.max(0, -wc.head),
    turbSevere: sim.rep.cond.wind.turb === 'severe', mainGearZ: bodyToWorld(f, [-2.5, -6.6, 0])[2], halfWidthM: RWY.width / 2,
    stopNeedM: f.wow && A.gs > 5 ? A.gs * A.gs / (2 * 2.5) : 0, runwayLeftM: RWY.length - f.p[0], retard: !!st.retard };
}

export function landingFacts(sim, cam, { W = 896, H = 504, scene = {} } = {}) {
  const F = {}, put = (id, v, unit, obs) => { F[id] = fact(v, unit, obs); }, f = sim.flight, A = f.air, st = sim.gnc.st, d = st.dev || {}, w = sim.rep.cond.wind, wc = sim.wind.components(10);
  put('mode', st.mode, null, 'context'); put('lat_mode', st.lat, null, 'context'); put('vert_mode', st.vert, null, 'context');
  put('pos.x_m', r2(f.p[0]), 'm', 'context'); put('pos.z_m', r2(f.p[2]), 'm', 'context'); put('alt_ft', r2(f.p[1] / FT), 'ft', 'context'); put('ra_ft', r2(Math.max(0, A.hRA / FT)), 'ft', 'context');
  put('thr_nm', r2(Math.max(0, -f.p[0]) / 1852), 'NM', 'context'); put('ias_kt', r2(A.V / KT), 'kt', 'context'); put('gs_kt', r2(A.gs / KT), 'kt', 'context');
  put('vs_fpm', r2(f.v[1] / FT * 60), 'fpm', 'context'); put('vapp_kt', r2(st.vapp / KT), 'kt', 'context');
  put('att.pitch_deg', r2(A.theta / DEG), 'deg', 'context'); put('att.bank_deg', r2(A.phi / DEG), 'deg', 'context'); put('att.heading_deg', Math.round(((A.psi / DEG) + RWY.heading + 360) % 360), 'deg', 'context');
  put('att.alpha_deg', r2(A.alpha / DEG), 'deg', 'context'); put('att.beta_deg', r2(A.beta / DEG), 'deg', 'context'); put('att.fpa_deg', r2(A.gamma / DEG), 'deg', 'context');
  put('ils.loc_dots', d.locValid ? r2(d.loc) : null, 'dots', 'context'); put('ils.gs_dots', d.gsValid ? r2(d.gs) : null, 'dots', 'context');
  const inFrame = (p) => { const q = project(cam, p, W, H); return q.front && q.x >= 0 && q.x <= W && q.y >= 0 && q.y <= H; };
  put('papi_whites_cam', AIRPORT.papi.some((u) => inFrame([u.x, u.y, u.z])) ? papiWhites(cameraPosition(cam)) : null, 'count', 'visual');
  const ws = AIRPORT.windsock; put('windsock.from_deg', inFrame([ws.x, ws.h, ws.z]) ? w.dir : null, 'deg', 'visual');
  put('cfg.gear', f.gear > 0.99 ? 'down' : f.gear > 0.01 ? 'transit' : 'up', null, 'visual'); put('cfg.spoilers', r2(f.spoil), null, 'visual');
  put('thrust', r2(f.spool), null, 'context'); put('wow', f.wow, null, 'context');
  put('wind.metar', `${String(w.dir).padStart(3, '0')}${String(w.kt).padStart(2, '0')}${w.gust ? 'G' + w.gust : ''}KT`, null, 'context');
  put('wind.head_kt', r2(wc.head), 'kt', 'context'); put('wind.cross_kt', r2(wc.cross), 'kt', 'context'); put('wind.turb', w.turb, null, 'context');
  put('gates', { ...st.gates }, null, 'context');
  for (const k of ['time', 'vis', 'clouds', 'rain']) put(`scene.${k}`, scene[k] ?? null, null, 'visual');
  return F;
}

export function forceGoAround(sim) { const st = sim.gnc.st; st.vert = 'GA'; st.tGa = sim.flight.t; st.why = 'forced'; }
export function applyLandingKick(sim, inj) {
  const f = sim.flight;
  if (inj.kind === 'alt_plus_60') f.p[1] += 60;
  else if (inj.kind === 'speed_plus_25') { const v = Math.hypot(f.v[0], f.v[1], f.v[2]); for (let i = 0; i < 3; i++) f.v[i] += 25 * f.v[i] / v; }
  else if (inj.kind === 'lateral_25') { f.p[2] += 25; f.v[2] += 3; }
  else if (inj.kind === 'forced_ga') forceGoAround(sim);
  else throw new Error(`not a runtime L injection: ${inj.kind}`);
}
export function replayLanding(cond, { params = {}, injection = null, step }) {
  const sim = createLandingSim(cond, { params });
  for (let n = 0; n < step && !sim.rep.done; n++) { if (injection && injection.step === n) applyLandingKick(sim, injection); sim.step(); }
  return sim;
}
export function landingBranches({ cond, params = {}, injection = null, step, liveResult, airborne, retard }) {
  const rt = injection && injection.step !== null && injection.step !== undefined ? injection : null;
  let CONTINUE = lOutcome({ result: liveResult });
  if (rt && rt.step > step) { const s = replayLanding(cond, { params, step }); s.run(); CONTINUE = lOutcome(s.rep); }
  let GO_AROUND = null;
  if (airborne && !retard) { const s = replayLanding(cond, { params, injection: rt && rt.step < step ? rt : null, step }); forceGoAround(s); s.run(); GO_AROUND = lOutcome(s.rep); }
  return { CONTINUE, GO_AROUND };
}
export function landingLabel(sim, branches) {
  const input = { now: landingNow(sim), outcome: branches };
  return { safety: landingSafety(input), safety_eye: eyeView('L', input), input };
}
