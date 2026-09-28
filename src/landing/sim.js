// One landing, start to stop: the flight model, the weather, the ILS and the autoland stepped together at 120 Hz, with the
// numbers a flight-data monitor keeps — the touchdown (sink rate, g, where, how aligned), the approach (stabilized gates,
// beam tracking), the rollout — and a grade. drawConditions(seed) gives Monte Carlo campaigns their weather and start:
// headwind −10…+25 kt, crosswind ±20 kt, gusts in 30 %, turbulence none/light/moderate, anywhere 16–30 km out at
// 3000–5000 ft on any heading (or, with final: true, vectored onto an 8.4–10.3 NM final: 15.5–19 km, ≥ 2 NM before the glideslope intercept as ATC rules ask).
import { newFlight, stepFlight, trimFlight, bodyToWorld } from './flight.js';
import { createWind } from './wind.js';
import { createAutoland } from './autoland.js';
import { createIlsNoise } from './nav.js';
import { KT, FT } from './vehicle.js';
import { RWY } from './airport.js';
import { mulberry32 } from '../mathx.js';

const DEG = Math.PI / 180, H = 1 / 120;
export function drawConditions(seed, { final = false } = {}) {
  const r = mulberry32((seed * 2654435761) >>> 0);
  const head = -10 + 35 * r(), cross = (r() * 2 - 1) * 20, kt = Math.round(Math.hypot(head, cross)), dir = Math.round((RWY.heading + Math.atan2(cross, head) / DEG + 360) % 360) || 360;
  const gust = r() < 0.3 ? kt + 5 + Math.round(10 * r()) : 0, tr = r(), turb = tr < 0.2 ? 'none' : tr < 0.7 ? 'light' : 'moderate';
  const start = final ? { x: -15500 - 3500 * r(), z: (r() * 2 - 1) * 600, h: 560 + 100 * r(), hdg: (r() * 2 - 1) * 10 * DEG, V: (175 + 10 * r()) * KT }
    : { x: -16000 - 14000 * r(), z: (r() * 2 - 1) * 12000, h: 915 + 610 * r(), hdg: (r() * 2 - 1) * Math.PI, V: (200 + 30 * r()) * KT };
  return { seed, wind: { dir, kt, gust, turb }, start };
}
export const grade = (fps) => (fps < 1.5 ? 'butter' : fps < 3 ? 'smooth' : fps < 6 ? 'firm' : 'hard');

export function createLandingSim(cond, { params = {} } = {}) {
  const rng = mulberry32((cond.seed ^ 0x2545f491) >>> 0), f = newFlight(), s0 = cond.start;
  trimFlight(f, { x: s0.x, z: s0.z, h: s0.h, V: s0.V, gamma: 0, heading: s0.hdg, gear: s0.gear ?? 0 });
  const wind = createWind({ ...cond.wind, seed: cond.seed, rwyHdg: RWY.heading }), gnc = createAutoland({ params, noise: createIlsNoise(rng), atis: cond.wind });
  gnc.plan(f);
  const rep = { cond, done: false, result: '', td: null, tdG: 1, maxG: 1, maxZ: 0, stopX: null, tStop: null, gates: gnc.st.gates, maxLoc: 0, maxGs: 0, calls: gnc.st.calls, tailStrike: false };
  const _m = new Float64Array(3), _p = new Float64Array(3);
  function finish(result) { rep.done = true; rep.result = result; rep.t = f.t; rep.maxLoc = Math.max(-gnc.st.minLoc, gnc.st.maxLoc); rep.maxGs = gnc.st.maxGs; rep.tailStrike = f.tailStrike; }
  function step() {
    if (rep.done) return;
    const cmd = gnc.update(f, H);
    wind.mean(Math.max(1, f.p[1]), _m); let hx = f.v[0] - _m[0], hz = f.v[2] - _m[2]; const hl = Math.hypot(hx, hz) || 1; hx /= hl; hz /= hl;
    wind.step(H, f.air.V, hx, hz, Math.max(1, f.air.hRA));
    stepFlight(f, cmd, wind, H);
    rep.maxG = Math.max(rep.maxG, f.gearG);
    if (f.touchdown && !rep.td) rep.td = { ...f.touchdown, sinkFps: f.touchdown.sinkFps, mode: gnc.st.mode, grade: grade(f.touchdown.sinkFps) };
    if (rep.td && f.t - rep.td.t < 2) rep.tdG = Math.max(rep.tdG, f.gearG);
    if (f.wow) { const m = bodyToWorld(f, [-2.5, -6.6, 0], _p), off = gnc.st.lat === 'EXIT' || gnc.st.lat === 'CLEAR';   // on an exit taxiway it is meant to leave the runway
      if (!off) { rep.maxZ = Math.max(rep.maxZ, Math.abs(m[2])); if (Math.abs(m[2]) > RWY.width / 2) return finish('excursion'); } if (m[0] > RWY.length) return finish('overrun'); }
    if (rep.td && rep.td.x < 0) return finish('short');
    if (f.tailStrike) return finish('tailstrike');
    if (!f.wow && f.air.hRA < -0.5) return finish('crash');
    if (gnc.st.vert === 'GA' && f.t - gnc.st.tGa > 20) return finish('go-around');
    if (gnc.st.vert === 'STOP') { rep.stopX = f.p[0]; rep.stopZ = f.p[2]; rep.tStop = f.t; rep.exit = gnc.st.exit ? { name: gnc.st.exit.name, kt: gnc.st.exitKt } : null; return finish(rep.td.sinkFps > 10 ? 'hard' : 'landed'); }
  }
  function run(maxT = 900) { while (!rep.done && f.t < maxT) step(); if (!rep.done) finish('timeout'); return rep; }
  return { flight: f, wind, gnc, rep, step, run };
}
