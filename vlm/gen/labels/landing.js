// vlm/gen/labels/landing.js — landing (L) labels (spec §4.2 L, §4.3 landing, §7.3 L): landingNow() reduces the sim to
// the combinator's inputs exactly as autoland.js:92-99 evaluates them, landingFacts() writes the facts, and replays from
// createLandingSim(conditions(seed)) give the CONTINUE and GO_AROUND branches (review probe_landing_ga.mjs).
import { createLandingSim } from '../../../src/landing/sim.js';
import { bodyToWorld } from '../../../src/landing/flight.js';
import { KT, FT } from '../../../src/landing/vehicle.js';
import { AIRPORT, RWY } from '../../../src/landing/airport.js';
import { landingSafety, eyeView } from '../safety.js';
import { fact, Discard } from '../schema.js';
import { project, cameraPosition } from './camera.js';

export const H_L = 1 / 120;
const DEG = Math.PI / 180, GSA = 3 * DEG, r2 = (x) => (Number.isFinite(x) ? +x.toFixed(2) : null);
export const stepOf = (sim) => Math.round(sim.flight.t / H_L);
export const papiWhites = (eye) => AIRPORT.papi.reduce((n, u) => n + (Math.atan2(eye[1] - u.y, Math.hypot(eye[0] - u.x, eye[2] - u.z)) > u.angle ? 1 : 0), 0);
export const lOutcome = (rep) => (rep.result === 'go-around' ? 'go_around' : rep.result);
// world boxes [lo, hi] of what the visual facts read: the PAPI bar (4 units with their 2.2 m light sprites, airportmesh.js:101,
// lights.js:10) and the windsock's sock (4.5 m, swinging round the pole top, airportmesh.js:112)
const PZ = AIRPORT.papi.map((u) => u.z), PX = AIRPORT.papi[0].x, WS = AIRPORT.windsock;
export const PAPI_BOX = Object.freeze([[PX - 1.1, 0, Math.min(...PZ) - 1.1], [PX + 1.1, 2.1, Math.max(...PZ) + 1.1]]);
export const SOCK_BOX = Object.freeze([[WS.x - 4.5, WS.h - 0.9, WS.z - 4.5], [WS.x + 4.5, WS.h + 0.9, WS.z + 4.5]]);
// the in-frame part of a box's projection (px at W x H), or null when a corner is behind the camera or nothing is in frame
export function pixelBox(cam, [lo, hi], W, H) {
  const c = [];
  for (const x of [lo[0], hi[0]]) for (const y of [lo[1], hi[1]]) for (const z of [lo[2], hi[2]]) c.push(project(cam, [x, y, z], W, H));
  if (c.some((q) => !q.front)) return null;
  const x0 = Math.max(0, Math.min(...c.map((q) => q.x))), x1 = Math.min(W, Math.max(...c.map((q) => q.x))), y0 = Math.max(0, Math.min(...c.map((q) => q.y))), y1 = Math.min(H, Math.max(...c.map((q) => q.y)));
  return x1 > x0 && y1 > y0 ? [x0, y0, x1, y1] : null;
}
// a visual fact counts (non-null) only when its box is in frame with its longest side >= 2 px at the 896x504 capture
// resolution (controller ruling: these facts feed Narrator's 512^2 input; Pilot Eye's scope is safety_eye/EYE)
export const CAPTURE_W = 896, CAPTURE_H = 504, MIN_SIDE_PX = 2;
export const boxVisible = (cam, box, W, H) => { const b = pixelBox(cam, box, W, H); return !!b && Math.max((b[2] - b[0]) * CAPTURE_W / W, (b[3] - b[1]) * CAPTURE_H / H) >= MIN_SIDE_PX; };
const sameBits = (a, b) => a.every((v, i) => Object.is(v, b[i]));
// T10-v: fog transmittance gate. The page's fog is exponential-squared (lights.js:49, closed in by the cloud deck per
// scene.js:93): transmittance(rho, d) = exp(-(rho*d)^2). A camera-visual fact of an object is non-null only when its box
// is in frame (boxVisible) and this transmittance, at the distance from the camera eye to the object's nearest point,
// is >= T_VIS; T_VIS is the same 10% cutoff for the PAPI, the windsock and the aircraft itself (d there is the chase
// camera's distance to the aircraft, since the page has no separate box for it).
export const T_VIS = 0.1;
export const transmittance = (rho, d) => Math.exp(-((rho * d) ** 2));
const dist3 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const clamp1 = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
// the nearest point of an axis-aligned box [lo, hi] to a point p, for the fog-distance measurement
const distToBox = (p, [lo, hi]) => dist3(p, [0, 1, 2].map((i) => clamp1(p[i], lo[i], hi[i])));
const visibleThroughFog = (cam, box, W, H, eye, rho) => boxVisible(cam, box, W, H) && transmittance(rho, distToBox(eye, box)) >= T_VIS;

export function landingNow(sim) {
  const f = sim.flight, A = f.air, st = sim.gnc.st, d = st.dev || {}, P = sim.gnc.params, ft = A.hRA / FT, wc = sim.wind.components(10);
  const vG = st.vG ?? A.V, vMin = P.vapp - 5 * KT, vMax = Math.max(P.vapp + 15 * KT, (st.vTgt ?? P.vapp) + 10 * KT), vsLim = -Math.max(1000, A.gs * Math.tan(GSA) / FT * 60 + 300);
  const gate = { lateral: st.lat === 'LOC', vertical: st.vert === 'GS', loc: Math.abs(d.loc ?? 9) < 1, gs: Math.abs(d.gs ?? 9) < 1, speed: vG >= vMin && vG <= vMax, vs: (st.vzS ?? f.v[1]) > vsLim * FT / 60, gear: f.gear > 0.99 };
  return { airborne: !f.wow, wow: f.wow, vert: st.vert, lat: st.lat, hRAft: ft, thrNm: Math.max(0, -f.p[0]) / 1852, locDots: d.locValid ? d.loc : 0, gsDots: d.gsValid ? d.gs : 0,
    crossing500: st.lastFt > 500 && ft <= 500, gate, speedErrKt: (vG < vMin ? vG - vMin : vG > vMax ? vG - vMax : 0) / KT, crossKt: wc.cross, tailKt: Math.max(0, -wc.head),
    turbSevere: sim.rep.cond.wind.turb === 'severe', mainGearZ: bodyToWorld(f, [-2.5, -6.6, 0])[2], halfWidthM: RWY.width / 2,
    stopNeedM: f.wow && A.gs > 5 ? A.gs * A.gs / (2 * 2.5) : 0, runwayLeftM: RWY.length - f.p[0], retard: !!st.retard };
}

export function landingFacts(sim, cam, { W = 896, H = 504, scene = {}, fogDensity = 0 } = {}) {
  const F = {}, put = (id, v, unit, obs) => { F[id] = fact(v, unit, obs); }, f = sim.flight, A = f.air, st = sim.gnc.st, d = st.dev || {}, w = sim.rep.cond.wind, wc = sim.wind.components(10);
  put('mode', st.mode, null, 'context'); put('lat_mode', st.lat, null, 'context'); put('vert_mode', st.vert, null, 'context');
  put('pos.x_m', r2(f.p[0]), 'm', 'context'); put('pos.z_m', r2(f.p[2]), 'm', 'context'); put('alt_ft', r2(f.p[1] / FT), 'ft', 'context'); put('ra_ft', r2(Math.max(0, A.hRA / FT)), 'ft', 'context');
  put('thr_nm', r2(Math.max(0, -f.p[0]) / 1852), 'NM', 'context'); put('ias_kt', r2(A.V / KT), 'kt', 'context'); put('gs_kt', r2(A.gs / KT), 'kt', 'context');
  put('vs_fpm', r2(f.v[1] / FT * 60), 'fpm', 'context'); put('vapp_kt', r2(st.vapp / KT), 'kt', 'context');
  put('att.pitch_deg', r2(A.theta / DEG), 'deg', 'context'); put('att.bank_deg', r2(A.phi / DEG), 'deg', 'context'); put('att.heading_deg', Math.round(((A.psi / DEG) + RWY.heading + 360) % 360), 'deg', 'context');
  put('att.alpha_deg', r2(A.alpha / DEG), 'deg', 'context'); put('att.beta_deg', r2(A.beta / DEG), 'deg', 'context'); put('att.fpa_deg', r2(A.gamma / DEG), 'deg', 'context');
  put('ils.loc_dots', d.locValid ? r2(d.loc) : null, 'dots', 'context'); put('ils.gs_dots', d.gsValid ? r2(d.gs) : null, 'dots', 'context');
  const eye = cameraPosition(cam), papiVis = visibleThroughFog(cam, PAPI_BOX, W, H, eye, fogDensity), sockVis = visibleThroughFog(cam, SOCK_BOX, W, H, eye, fogDensity);
  put('papi_whites_cam', papiVis ? papiWhites(eye) : null, 'count', 'visual');
  put('windsock.from_deg', sockVis ? w.dir : null, 'deg', 'visual');
  // the aircraft's own visual facts: no separate box on the page, so the gate is transmittance alone at the chase
  // camera's distance to the aircraft (it is always framed by the chase view)
  const acVis = transmittance(fogDensity, dist3(eye, f.p)) >= T_VIS;
  put('cfg.gear', acVis ? (f.gear > 0.99 ? 'down' : f.gear > 0.01 ? 'transit' : 'up') : null, null, 'visual'); put('cfg.spoilers', acVis ? r2(f.spoil) : null, null, 'visual');
  put('thrust', r2(f.spool), null, 'context'); put('wow', f.wow, null, 'context');
  put('wind.metar', `${String(w.dir).padStart(3, '0')}${String(w.kt).padStart(2, '0')}${w.gust ? 'G' + w.gust : ''}KT`, null, 'context');
  put('wind.head_kt', r2(wc.head), 'kt', 'context'); put('wind.cross_kt', r2(wc.cross), 'kt', 'context'); put('wind.turb', w.turb, null, 'context');
  put('gates', { ...st.gates }, null, 'context');
  for (const k of ['time', 'vis', 'clouds', 'rain', 'in_cloud']) put(`scene.${k}`, scene[k] ?? null, null, 'visual');
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
// replay to `step`; with the live flight's p and v at that step (liveP, liveV) it must match them bit for bit
export function replayLanding(cond, { params = {}, injection = null, step, liveP = null, liveV = null }) {
  const sim = createLandingSim(cond, { params });
  for (let n = 0; n < step && !sim.rep.done; n++) { if (injection && injection.step === n) applyLandingKick(sim, injection); sim.step(); }
  if (liveP && !(sameBits(sim.flight.p, liveP) && sameBits(sim.flight.v, liveV))) throw new Error(`landing replay is not bit-exact at t_sample (step ${step})`);
  return sim;
}
// branches at the sample `step`; a branch that ends in timeout is a Discard (spec §4.3: a timeout run is dropped)
export function landingBranches({ cond, params = {}, injection = null, step, liveResult, airborne, retard, liveP = null, liveV = null }) {
  const rt = injection && injection.step !== null && injection.step !== undefined ? injection : null;
  if (rt && rt.step === step) throw new Error(`injection ${rt.kind} at the sample step ${step}: the timing rule forbids it`);
  let CONTINUE = lOutcome({ result: liveResult });
  if (rt && rt.step > step) { const s = replayLanding(cond, { params, step, liveP, liveV }); s.run(); CONTINUE = lOutcome(s.rep); }
  if (CONTINUE === 'timeout') throw new Discard(`L CONTINUE ends in timeout (step ${step})`);
  let GO_AROUND = null;
  if (airborne && !retard) {
    const s = replayLanding(cond, { params, injection: rt && rt.step < step ? rt : null, step, liveP, liveV }); forceGoAround(s); s.run(); GO_AROUND = lOutcome(s.rep);
    if (GO_AROUND === 'timeout') throw new Discard(`L GO_AROUND replay ends in timeout (step ${step})`);
  }
  return { CONTINUE, GO_AROUND };
}
export function landingLabel(sim, branches) {
  const input = { now: landingNow(sim), outcome: branches };
  return { safety: landingSafety(input), safety_eye: eyeView('L', input), input };
}
