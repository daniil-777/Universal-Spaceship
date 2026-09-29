// The real-spacecraft simulation (S1, headless): truth dynamics (exact CW ZOH + RK4 rigid body; 10 ms steps in any
// 100 ms cycle with a jet on or |we| > 0.2 deg/s, else 100 ms), the dispersed jets, nav, 1 Hz guidance + safety filter,
// 10 Hz control, and the judge: collisions (7 ship spheres vs the station capsules within 60 m), IDSS contact, KOS and
// corridor, keep-in, propellant, and the truth Rule-P / breakout checks after every TRANSFER burn, MCC and BREAKOUT.
import { OMEGA_L, DEG, DRAG_B, T_ORB, CYCLE, MASS0, PROP0, INERTIA, KOS_R, KEEPIN_R, NARROW_R, IDSS, JETS, SHIP_PORT, STATION_PORT, SHIP_SPHERES, STATION_CAPSULES, RULE_P_TRANSFER, portRel, inCone } from './consts.js';
import { phi, gamma, stepWith } from './cw.js';
import { rigidStep, makeInertia, omegaRel, attError } from './rigid.js';
import { createJets } from './jets.js';
import { createNav } from './nav.js';
import { createControl } from './control.js';
import { createGuidance, PH, CORRIDOR_PHASES } from './guidance.js';
import { filterVelocity } from './safety.js';
import { passiveMin, breakoutOk } from './passive.js';
import { mulberry32, qRotate, qInvRotate, qFromAxisAngle, qMul } from '../mathx.js';

const P1 = phi(CYCLE), G1 = gamma(CYCLE), P01 = phi(0.01), G01 = gamma(0.01);
const U = (rng, a) => (2 * rng() - 1) * a;
const QREF = [0, 0, 0, 1];

// the run's dispersions and start (spec sections 3 and 10); disp = false gives the nominal vehicle and a_d = 0
export function drawRun(seed, { start = 'far', disp = true } = {}) {
  const rng = mulberry32(seed * 2654435761 + 7), d = disp ? 1 : 0, k = (a) => U(rng, a) * d;
  const primaries = JETS.map((j, i) => (j.kind === 'P' ? i : -1)).filter((i) => i >= 0);
  const out = {
    seed, start, ad: k(DRAG_B),
    thrustScale: JETS.map(() => 1 + k(0.05)),
    dirTilt: JETS.map(() => { const a = k(0.5 * DEG), th = 2 * Math.PI * rng(); return [0, a * Math.cos(th), a * Math.sin(th)]; }),
    com: [k(0.05), k(0.05), k(0.05)],
    failed: disp && rng() < 0.05 ? [primaries[Math.floor(rng() * primaries.length)]] : [],
    inertia: (() => { const c = 1 + k(0.15); return { diag: INERTIA.map((v) => v * c * (1 + k(0.015))), prod: [k(0.01), k(0.01), k(0.01)].map((v) => v * INERTIA[1]) }; })(),
    mib: { P: disp ? 0.03 + 0.02 * rng() : 0.04, V: disp ? 0.06 + 0.04 * rng() : 0.08 },
    navScale: 1,
  };
  const s = (a) => U(rng, a);
  if (start === 'far') { out.x = [-2000 + s(30), s(30), 250 + s(30), s(0.02), s(0.02), s(0.02)]; out.phase = PH.TRANSFER; }
  else if (start === 'near') { out.x = [-250 + s(3), s(1), s(1), s(0.002), s(0.002), s(0.002)]; out.phase = PH.H1; }
  else { out.x = [STATION_PORT[0] - 20 - SHIP_PORT[0] + s(0.3), -SHIP_PORT[1] + s(0.3), s(0.3), s(0.002), s(0.002), s(0.002)]; out.phase = PH.H2; }
  out.att = [s(1 * DEG), s(1 * DEG), s(1 * DEG)];
  out.rate = [s(0.02 * DEG), s(0.02 * DEG), s(0.02 * DEG)];
  return out;
}

export function createRealSim({ seed = 1, start = 'far', nav = 'noisy', disp = true, filter = true, tMax = 20000, run = null } = {}) {
  const R = run || drawRun(seed, { start, disp }), rng = mulberry32(seed + 99);
  const x = Float64Array.from(R.x), q = new Float64Array(4), w = new Float64Array(3);
  const qa = qFromAxisAngle(1, 0, 0, R.att[0]), qb = qFromAxisAngle(0, 1, 0, R.att[1]), qc = qFromAxisAngle(0, 0, 1, R.att[2]);
  qMul(qMul(qa, qb), qc, q);
  const inertia = makeInertia(R.inertia.diag, R.inertia.prod);
  const truthJets = createJets({ thrustScale: R.thrustScale, dirTilt: R.dirTilt, com: R.com, failed: R.failed, mib: R.mib });
  const ctrl = createControl({ jets: createJets({ failed: R.failed }) }), navf = createNav({ mode: nav, rng, scale: R.navScale });
  const events = [], log = [], guid = createGuidance({ phase: R.phase, events, failed: R.failed });
  qInvRotate(q, OMEGA_L, w); for (let i = 0; i < 3; i++) w[i] += R.rate[i];
  const F = new Float64Array(3), Tq = new Float64Array(3), aL = new Float64Array(3), aSum = new Float64Array(3), pv = new Float64Array(3), pp = new Float64Array(3);
  const rep = { done: false, result: null, reason: null, t: 0, dv: 0, prop: 0, contact: null, latMax: 0, drops: 0, brakes: 0, bypass: 0, rMin: Infinity, run: R };
  let t = 0, m = MASS0 - (R.propUsed || 0), cmd = { vDes: null, mode: 'P' }, k = 0, onT = new Float64Array(JETS.length), seen = 0, tDepart = null, burnV0 = null, burnKey = null, pendingAbort = null;
  const navState = () => ({ x: navf.xh, q, w, mass: m, fuel: (m - (MASS0 - PROP0)) / PROP0, valid: navf.valid });
  const sim = { get t() { return t; }, x, q, w, get mass() { return m; }, get cmd() { return cmd; }, get onTimes() { return onT; }, guid, ctrl, nav: navf, events, log, rep, step, run: runFor, R,
    abort(why = 'manual') { pendingAbort = why; k = 0; } };

  function finish(result, reason) { if (rep.done) return; Object.assign(rep, { done: true, result, reason, t, prop: MASS0 - m }); }
  const segDist = (c, a, b) => {
    const ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], L = ab[0] ** 2 + ab[1] ** 2 + ab[2] ** 2;
    let s = ((c[0] - a[0]) * ab[0] + (c[1] - a[1]) * ab[1] + (c[2] - a[2]) * ab[2]) / L; s = Math.max(0, Math.min(1, s));
    return Math.hypot(c[0] - a[0] - s * ab[0], c[1] - a[1] - s * ab[1], c[2] - a[2] - s * ab[2]);
  };
  function collided() {
    for (const sp of SHIP_SPHERES) {
      qRotate(q, sp, pv); const c = [x[0] + pv[0], x[1] + pv[1], x[2] + pv[2]];
      for (const [a, b, rc] of STATION_CAPSULES) if (segDist(c, a, b) < sp[3] + rc) return true;
    }
    return false;
  }
  function contact(p) {
    const wr = omegaRel(q, w), rv = qRotate(q, [wr[1] * SHIP_PORT[2] - wr[2] * SHIP_PORT[1], wr[2] * SHIP_PORT[0] - wr[0] * SHIP_PORT[2], wr[0] * SHIP_PORT[1] - wr[1] * SHIP_PORT[0]]);
    const v = [x[3] + rv[0], x[4] + rv[1], x[5] + rv[2]], e = attError(q, QREF);
    const c = { close: v[0], lat: Math.hypot(v[1], v[2]), rate: Math.max(...wr.map(Math.abs)), mis: Math.hypot(p[1], p[2]), ang: Math.hypot(...e) };
    rep.contact = { ...c, rate: c.rate / DEG, ang: c.ang / DEG };
    const ok = c.close >= IDSS.closeMin && c.close <= IDSS.closeMax && c.lat <= IDSS.lat && c.rate <= IDSS.rate && c.mis <= IDSS.mis && c.ang <= IDSS.ang;
    finish(ok ? 'capture' : 'fail', ok ? null : 'non-IDSS contact');
  }
  function judgeEvents() {
    for (; seen < events.length; seen++) {
      const ev = events[seen];
      if (burnV0 && ev.screened) ev.executed = [x[3] - burnV0[0], x[4] - burnV0[1], x[5] - burnV0[2]];
      if (ev.kind === 'DEPART' || ev.kind === 'MCC') {
        const pm = passiveMin(x, 86400, DRAG_B);
        ev.truthRMin = pm.rMin;
        if (pm.rMin < RULE_P_TRANSFER) finish('fail', `Rule P after ${ev.kind}: ${pm.rMin.toFixed(1)} m`);
      } else if (ev.kind === 'BREAKOUT') {
        const bo = breakoutOk(x, { inside: ev.inside !== false }); ev.truth = bo; tDepart = t;
        if (!bo.ok) finish('fail', 'breakout not passively safe');
      }
    }
  }

  function step() {
    if (rep.done) return;
    const rho = Math.hypot(...portRel(x, q, pp));
    const xh = navf.update(x, rho, aSum);
    if (k % 10 === 0) {
      if (pendingAbort) { guid.breakout(t, pendingAbort, navState()); pendingAbort = null; }
      cmd = guid.update(t, navState());
      if (cmd.vDes && filter && !cmd.bypass) {
        const f = filterVelocity(cmd.vDes, xh, { inCorridor: cmd.inCorridor, axial: cmd.axial }, log, t);
        cmd.vDes = f.v; rep.drops += f.dropped.length; if (f.brake) rep.brakes++;
      }
      const key = cmd.burn ? guid.st.burn.t0 : null;
      if (key !== burnKey) { burnKey = key; if (key !== null) { burnV0 = [x[3], x[4], x[5]]; if (cmd.bypass) { log.push({ t, kind: 'bypass', burn: cmd.burnKind }); rep.bypass++; } } }
    }
    k++;
    const req = ctrl.update(xh, q, w, m, cmd);
    onT = truthJets.schedule(req);
    const fine = onT.some((v) => v > 0) || Math.max(...omegaRel(q, w).map(Math.abs)) > 0.2 * DEG;
    aSum.fill(0);
    if (fine) {
      for (let s = 0; s < 10; s++) {
        const md = truthJets.active(onT, s * 0.01, F, Tq);
        qRotate(q, F, aL); for (let i = 0; i < 3; i++) { aL[i] /= m; aSum[i] += aL[i] / 10; }
        stepWith(P01, G01, x, aL, R.ad, x);
        rigidStep(q, w, 0.01, { inertia, torque: Tq });
        m -= md * 0.01; rep.dv += Math.hypot(aL[0], aL[1], aL[2]) * 0.01;
      }
    } else {
      stepWith(P1, G1, x, null, R.ad, x);
      rigidStep(q, w, CYCLE, { inertia });
    }
    t += CYCLE; rep.t = t;
    judge();
  }
  function judge() {
    judgeEvents(); if (rep.done) return;
    const r = Math.hypot(x[0], x[1], x[2]), p = portRel(x, q, pp), ph = guid.st.phase;
    rep.rMin = Math.min(rep.rMin, r);
    if (ph === PH.FINAL) rep.latMax = Math.max(rep.latMax, Math.hypot(p[1], p[2]));
    if (m <= MASS0 - PROP0) return finish('fail', 'propellant exhausted');
    if (r > KEEPIN_R) return finish('fail', 'keep-in exit');
    if (r < NARROW_R && collided()) return finish('fail', 'collision');
    if (-p[0] <= 0 && Math.hypot(p[1], p[2]) < 2) return contact(p);
    const cone = inCone(p);
    if (r < KOS_R && !(CORRIDOR_PHASES.includes(ph) && cone) && ph !== PH.BREAKOUT && ph !== PH.DEPART) return finish('fail', `KOS violation in ${ph}`);
    if (ph === PH.DEPART && tDepart !== null && (r > KOS_R + 10 || t - tDepart > T_ORB)) return finish(r > KOS_R ? 'breakout' : 'fail', r > KOS_R ? null : 'still in the KOS one orbit after the breakout');
    if (t >= tMax) return finish('fail', 'timeout');
  }
  function runFor(sec = Infinity) { const end = t + sec; while (!rep.done && t < end - 1e-9) step(); return rep; }
  return sim;
}
