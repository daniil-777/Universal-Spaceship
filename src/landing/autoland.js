// The autoland — how the ship plans an approach and flies it to a stop, the way an airliner's autopilot and autothrottle
// do in CAT III:
//  NAV    a Dubins path (bank ≤ 25°) from wherever it is to the extended centreline 4 km before the glideslope intercept,
//         flown with L1 guidance (Park–Deyst–How), descending at ≤ 3.5° to 2000 ft at 200 kt;
//  LOC    the localizer captured and tracked (L1 on the beam), 180 kt;  G/S  the glideslope captured from below (or
//         above) and tracked — the path on the pitch (γ = −3° + a height correction), speed and energy on the thrust (a
//         TECS law) — gear down, V_app; stabilized-approach gates at 1000 ft and 500 ft (else GO AROUND);
//  FLARE  an exponential flare ḣ = −h/τ + c whose touchdown sink c is re-solved five times a second so the predicted
//         touchdown falls on the aim point (model-predictive touchdown control); ALIGN (de-crab: the sideslip takes most of
//         the wind-correction angle, a wing-low bank holds the track); RETARD;
//  ROLLOUT spoilers, derotation, the chute, autobrake, nose-wheel steering on the centreline; STOP.
import { dubins, poseAt } from './dubins.js';
import { VEH, G, KT, FT, liftCoeff } from './vehicle.js';
import { ilsDeviation, locMetres } from './nav.js';
import { AIRPORT } from './airport.js';
import { GNC } from './gnc_params.js';
import { bodyToWorld } from './flight.js';
import { clamp } from '../mathx.js';

const DEG = Math.PI / 180, GSA = 3 * DEG, RX = [16, -1, 0], GSD = AIRPORT.ils.gs.dot, GSX = AIRPORT.ils.gs.x, GSZ = AIRPORT.ils.gs.z;
const EXITS = AIRPORT.taxiways.filter((t) => t.kind === 'rapid').map((t) => ({ name: t.name, x0: t.pts[0][0], ang: Math.atan2(t.pts[1][1] - t.pts[0][1], t.pts[1][0] - t.pts[0][0]) })).sort((a, b) => a.x0 - b.x0);
const VEX = 40 * 0.514444, LWB = 16, ALAT = 1.5, REX = 400;   // turn-off speed, wheelbase, lateral-acceleration budget (m/s²), the exit's lead-in radius (m)
function exitPath(e) {                                    // the runway centreline, a 400-m arc, the exit's centreline — sampled every 2 m
  const T = REX * Math.tan(e.ang / 2), pts = [];
  for (let x = e.x0 - T - 300; x < e.x0 - T; x += 2) pts.push([x, 0]);
  for (let a = 0; a < e.ang; a += 2 / REX) pts.push([e.x0 - T + REX * Math.sin(a), REX * (1 - Math.cos(a))]);
  for (let d = T; d < T + 500; d += 2) pts.push([e.x0 + d * Math.cos(e.ang), d * Math.sin(e.ang)]);
  return pts;
}
const CALLS = [[1000, '1000'], [500, '500'], [100, '100'], [50, 'MINIMUMS'], [40, '40'], [30, '30'], [20, '20'], [10, '10']];

export function createAutoland({ params = {}, noise = null, atis = null } = {}) {   // atis: the reported wind { dir (from, °), kt, gust } for V_app
  const P = { ...GNC, ...params };
  const st = { mode: 'NAV', lat: 'NAV', vert: 'ALT', dub: null, xJ: 0, s: 0, px: null, pz: null, eI: null, vPrev: null, vdot: 0, gear: 0, phiHold: 0, tLat: -1,
    fl: null, tMpc: 0, tdT: null, noseT: null, brakeI: 0.25, gsPrev: null, decel: 0, gates: {}, calls: [], lastFt: Infinity, why: '', retard: false,
    vRef: 0, vapp: P.vapp, gCmd: 0, dev: {}, rx: new Float64Array(3), minLoc: 0, maxLoc: 0, maxGs: 0, tGa: null, vS: null, head: null, psiPrev: null, psiRate: 0 };
  const cmd = { nz: 1, phi: 0, beta: 0, thr: null, gear: 0, spoil: 0, brake: 0, chute: 0, steer: 0, q: NaN };
  const xGSI = GSX - Math.sqrt(Math.max(0, (P.hInt / Math.tan(GSA)) ** 2 - GSZ * GSZ));   // where the GS cone reaches hInt on the centreline

  function plan(s) {                                       // where to join the localizer, as ATC would vector the ship: 4 km before the glideslope intercept,
    const V = Math.max(s.air.V, 80), R = V * V / (G * Math.tan(P.bankPlan)) * P.radiusMargin, q = {}, x0 = xGSI - P.finalStraight;   // unless a join further in saves a
    const from = { x: s.p[0], z: s.p[2], hdg: Math.atan2(s.v[2], s.v[0]) }, need = 1.1 * Math.max(0, s.p[1] - P.hInt) / Math.tan(P.gMaxDesc);   // loop (≥ 1 km) — and only
    let best = null;                                        // on a path long enough to descend to the intercept altitude at ≤ 3.5° before the glideslope (else a longer final)
    for (let xj = x0 - 20000; xj <= xGSI - 1500 + 1e-6; xj += 250) {
      const d = dubins(from, { x: xj, z: 0, hdg: 0 }, R); if (d.length + (xGSI - xj) < need) continue;
      const c = d.length - xj + 0.1 * Math.abs(xj - x0) - (Math.abs(xj - x0) < 1 ? 1000 : 0);   // the distance to the threshold; near the standard join preferred
      if (!best || c < best.c) best = { d, xj, c };
    }
    if (!best) best = { d: dubins(from, { x: x0, z: 0, hdg: 0 }, R), xj: x0 };
    st.xJ = best.xj; st.dub = best.d; st.s = 0;
    const n = Math.ceil(st.dub.length / 10) + 1; st.px = new Float64Array(n); st.pz = new Float64Array(n);
    for (let i = 0; i < n; i++) { poseAt(st.dub, Math.min(i * 10, st.dub.length), q); st.px[i] = q.x; st.pz[i] = q.z; }
    st.lat = 'NAV'; st.vert = 'ALT'; st.mode = 'NAV'; return st.dub;
  }
  const _r = [0, 0];
  function pathXZ(sp, out) {                               // the whole path: the Dubins leg, then the extended centreline
    const L = st.dub.length; if (sp >= L) { out[0] = st.xJ + (sp - L); out[1] = 0; return out; }
    const f = sp / 10, i = Math.min(Math.floor(f), st.px.length - 2), u = f - i; out[0] = st.px[i] + u * (st.px[i + 1] - st.px[i]); out[1] = st.pz[i] + u * (st.pz[i + 1] - st.pz[i]); return out;
  }
  function progress(x, z) {                                // the closest point, searched forward from the last one
    const L = st.dub.length; if (st.s >= L) { st.s = Math.max(st.s, L + (x - st.xJ)); return; }
    let best = st.s, bd = Infinity; const i0 = Math.max(0, Math.floor(st.s / 10) - 5), i1 = Math.min(st.px.length - 1, i0 + 80);
    for (let i = i0; i <= i1; i++) { const d = (st.px[i] - x) ** 2 + (st.pz[i] - z) ** 2; if (d < bd) { bd = d; best = i * 10; } }
    if (i1 === st.px.length - 1 && best >= (i1 - 1) * 10) best = Math.max(best, L + (x - st.xJ));
    st.s = Math.max(st.s, best);
  }
  function l1(s, x, z, T, onBeam) {                        // L1 lateral acceleration → bank
    const vx = s.v[0], vz = s.v[2], V = Math.max(20, Math.hypot(vx, vz)), L1 = Math.max(120, T * P.l1Damp * V / Math.PI);
    let dx, dz; if (onBeam) { dx = L1; dz = -locMetres(st.dev.loc, st.rx[0]); } else { pathXZ(st.s + L1, _r); dx = _r[0] - x; dz = _r[1] - z; }
    const eta = Math.atan2(vx * dz - vz * dx, vx * dx + vz * dz);
    return Math.atan(2 * V * V / L1 * Math.sin(clamp(eta, -Math.PI / 2, Math.PI / 2)) / G);
  }

  function update(s, dt) {
    const A = s.air, x = s.p[0], z = s.p[2], hRA = A.hRA, ft = hRA / FT, dev = st.dev;
    bodyToWorld(s, RX, st.rx); ilsDeviation(st.rx, noise, s.t, dev);
    const vI = Math.hypot(s.v[0], s.v[1], s.v[2]);          // the speed rate from the inertial velocity (gusts do not shake it), as TECS takes it
    if (st.vPrev === null) st.vPrev = vI; st.vdot += ((vI - st.vPrev) / dt - st.vdot) * Math.min(1, dt / 0.5); st.vPrev = vI;
    st.vS = st.vS === null ? A.V : st.vS + (A.V - st.vS) * Math.min(1, dt / 2);          // smoothed airspeed and vertical speed for the gates (sustained, not a gust)
    st.vzS = st.vzS === undefined ? s.v[1] : st.vzS + (s.v[1] - st.vzS) * Math.min(1, dt / 3);
    st.vG = st.vG === undefined ? A.V : st.vG + (A.V - st.vG) * Math.min(1, dt / 5);   // the gates' sustained speed: momentary gust overshoots are accepted (FCTM)
    st.head = st.head === null ? A.V - A.gs : st.head + (A.V - A.gs - st.head) * Math.min(1, dt / 10);
    const head = atis ? atis.kt * KT * Math.cos((atis.dir - AIRPORT.runway.heading) * DEG) : st.head, gustAdd = (atis && atis.gust ? (atis.gust - atis.kt) * KT : 0) + (atis && atis.turb === 'moderate' ? 5 * KT : atis && atis.turb === 'severe' ? 8 * KT : 0);   // + the turbulence additive crews fly
    st.vapp = P.vapp + clamp(P.vAdd * Math.max(0, head) + gustAdd, 0, P.vAddMax);   // ½ the headwind + the gust increment (the usual wind additive)
    st.hNow = st.hNow === undefined ? A.V - A.gs : st.hNow + (A.V - A.gs - st.hNow) * Math.min(1, dt / 1.5);
    const vTgt = st.vTgt = clamp(st.vapp - Math.max(0, head) + st.hNow, st.vapp, st.vapp + 10 * KT);           // ground-speed mini: a gust above the reported wind raises the target, so a lull finds the energy there
    for (const [f, label] of CALLS) if (st.lastFt > f && ft <= f && !s.wow && st.vert !== 'GA') st.calls.push({ t: s.t, label });
    // modes
    if (st.lat === 'NAV' && dev.locValid && Math.abs(dev.loc) < 1.5 && st.s > st.dub.length - 2500) st.lat = 'LOC';
    if (st.vert === 'ALT' && st.lat === 'LOC' && dev.gsValid && Math.abs(dev.gs) < 0.15) st.vert = 'GS';   // captured at the beam, from below or (descending) from above
    if (st.vert === 'GS' || hRA < 1500 * FT) st.gear = 1;
    for (const g of [1000, 500]) if (st.lastFt > g && ft <= g && st.vert !== 'GA') {       // stabilized-approach gates
      const ok = st.lat === 'LOC' && st.vert === 'GS' && Math.abs(dev.loc) < 1 && Math.abs(dev.gs) < 1 && st.vG >= P.vapp - 5 * KT && st.vG <= Math.max(P.vapp + 15 * KT, st.vTgt + 10 * KT) && st.vzS > -Math.max(1000, A.gs * Math.tan(GSA) / FT * 60 + 300) * FT / 60 && s.gear > 0.99;   // V/S: 1000 fpm, or the ILS rate + 300 when the ground speed needs more (FSF: briefed)
      st.gates[g] = ok; if (!ok && g === 500) goAround(s, 'unstable at 500 ft');
    }
    if (st.vert === 'GS' && ft < 500 && ft > 30 && (Math.abs(dev.loc) > 1.5 || Math.abs(dev.gs) > 1.5)) goAround(s, 'excessive deviation');
    if (st.vert === 'GS' && ft < 1000) { st.minLoc = Math.min(st.minLoc, dev.loc); st.maxLoc = Math.max(st.maxLoc, dev.loc); st.maxGs = Math.max(st.maxGs, Math.abs(dev.gs)); }
    if (st.vert === 'GS' && hRA <= P.hFlare) { st.vert = 'FLARE'; const hd = s.v[1]; st.fl = { tau: P.hFlare / Math.max(0.5, -hd - P.sinkTD), c: -P.sinkTD, g0: A.gamma }; }
    if (s.wow && st.vert !== 'ROLLOUT' && st.vert !== 'STOP' && st.vert !== 'GA') { st.vert = 'ROLLOUT'; st.lat = 'ROLLOUT'; st.tdT = s.t; }
    st.lastFt = ft;
    cmd.gear = st.gear; cmd.q = NaN; cmd.spoil = 0; cmd.steer = 0; if (st.vert !== 'FLARE') cmd.beta = 0;
    if (st.vert === 'ROLLOUT' || st.vert === 'STOP') return rollout(s, dt);
    // lateral: L1 on the path or the beam, at 10 Hz (a held bank between)
    if (s.t - st.tLat >= 0.1 - 1e-9) { st.tLat = s.t; if (st.lat === 'NAV') progress(x, z);
      const lim = hRA < P.lowAlt ? P.bankLow : P.bankMax; st.phiHold = clamp(l1(s, x, z, st.lat === 'LOC' ? P.l1PeriodFinal : P.l1Period, st.lat === 'LOC'), -lim, lim); }
    cmd.phi = st.phiHold;
    if (st.vert === 'FLARE' && hRA <= P.hAlign) {        // align: the sideslip takes most of the wind-correction angle, a wing-low bank balances its side force
      const psiAir = A.psi + A.beta, bc = clamp(P.decrab * psiAir, -P.betaMax, P.betaMax); cmd.beta = bc;
      cmd.phi = clamp(st.phiHold + Math.atan(A.qbar * VEH.S * VEH.cyBeta * bc / (VEH.mass * G)), -P.bankLow, P.bankLow);
    }
    // vertical: the path on the pitch
    let gCmd, gff = 0, kg = P.kGamma;
    if (st.vert === 'ALT') { gCmd = clamp(P.kH * (P.hInt - s.p[1]) / Math.max(40, A.gs), -P.gMaxDesc, P.gMaxClimb); st.vRef = st.lat === 'LOC' ? P.vLoc : P.vNav; }
    else if (st.vert === 'GS') { const dh = dev.gs * GSD * Math.hypot(st.rx[0] - GSX, st.rx[2] - GSZ); gCmd = -GSA + clamp(-P.kH * dh / Math.max(40, A.gs), -2 * DEG, 2 * DEG); st.vRef = vTgt; }
    else if (st.vert === 'FLARE') {
      const f = st.fl, V = Math.max(40, A.gs);
      if (s.t - st.tMpc >= 0.2) {                          // model-predictive touchdown control: the sink c that lands the flare on the aim point
        st.tMpc = s.t; const xm = x - 2.5 * Math.cos(A.psi), tNeed = (P.aim - xm) / V;
        if (tNeed > 0.6) { const want = -hRA / (f.tau * (Math.exp(Math.min(tNeed / f.tau, 12)) - 1)); f.c += clamp(clamp(want, -P.sinkMax, -P.sinkMin) - f.c, -0.1, 0.1); }
      }
      const hdc = -Math.max(0, hRA) / f.tau + f.c; gCmd = Math.max(Math.asin(clamp(hdc / V, -0.2, 0.2)), f.g0 - 0.5 * DEG); gff = (-s.v[1] / f.tau) / V; kg = P.kGammaFlare; st.vRef = vTgt;   // never steeper than the approach
      if (hRA <= P.hRetard && !st.retard) { st.retard = true; st.calls.push({ t: s.t, label: 'RETARD' }); }
    } else { gCmd = st.tGa !== null && s.p[1] > 900 ? clamp(0.1 * (915 - s.p[1]) / 50, -0.05, 0.1) : 6 * DEG; st.vRef = P.vLoc; if (s.t - st.tGa > 5) st.gear = 0; }   // GO AROUND
    st.gCmd = gCmd; const V = Math.max(A.V, 30);
    cmd.nz = clamp((Math.cos(A.gamma) + V / G * (kg * (gCmd - A.gamma) + gff)) / Math.max(0.5, Math.cos(A.phi)), st.vert === 'FLARE' ? 0.8 : -0.5, 2.5);   // no push-over near the ground
    if (hRA < 30) cmd.nz = Math.min(cmd.nz, Math.max(0.8, liftCoeff(Math.max(0, P.pitchMax - A.gamma)) * A.qbar * VEH.S / (VEH.mass * G)));   // tail-strike protection: no more lift than at θ = pitchMax
    // thrust: total energy rate (TECS) — path and speed errors together
    const W = VEH.mass * G, Tmax = VEH.thrMax * (A.rho / 1.225) ** 0.7, vd = clamp(P.kV * (st.vRef - st.vS), -P.aMax, P.aMax), E = (gCmd - A.gamma) + (vd - st.vdot) / G;
    if (st.eI === null) st.eI = s.spool;
    st.eI = clamp(st.eI + P.kEI * E * W / Tmax * dt, VEH.idle, 1); let thr = clamp(st.eI + P.kE * E * W / Tmax, VEH.idle, 1);   // the integrator clamped to the throttle's range
    if (st.vert === 'GA') thr = 1; if (st.retard) thr = VEH.idle;
    cmd.thr = thr; st.mode = st.vert === 'GA' ? 'GO AROUND' : st.vert === 'FLARE' ? (st.retard ? 'RETARD' : 'FLARE') : st.vert === 'GS' ? (ft < 1500 ? 'LAND' : 'G/S') : st.lat;
    return cmd;
  }
  function goAround(s, why) { if (st.vert === 'GA') return; st.vert = 'GA'; st.why = why; st.tGa = s.t; st.calls.push({ t: s.t, label: 'GO AROUND' }); }

  function rollout(s, dt) {                                // spoilers, derotation, chute, autobrake, steering on the centreline
    const A = s.air, V = A.gs;
    cmd.thr = VEH.idle; cmd.spoil = 1; cmd.phi = 0; cmd.nz = 1;
    if (!s.noseOn && st.noseT === null) cmd.q = -P.derot; else { if (st.noseT === null) st.noseT = s.t; cmd.q = -0.01; }
    if (st.noseT !== null && V < P.chuteKt * KT && s.chute === 0) cmd.chute = 1;
    if (V < P.chuteOffKt * KT && s.chute === 1) cmd.chute = 2;
    if (st.gsPrev === null) st.gsPrev = V; st.decel += ((st.gsPrev - V) / dt - st.decel) * Math.min(1, dt / 0.4); st.gsPrev = V;
    if (st.noseT !== null && !st.exit && s.t - st.noseT > 2) for (const e of EXITS) {   // the first rapid exit it can slow down for (≤ 3 m/s²)
      const dist = e.x0 - s.p[0] - 60; if (dist > 0 && (V * V - VEX * VEX) / (2 * dist) <= 3) { st.exit = { ...e, pts: exitPath(e), i: 0 }; break; } }
    if (st.exit && st.lat === 'ROLLOUT' && s.p[0] >= st.exit.x0 - REX * Math.tan(st.exit.ang / 2) - 60) st.lat = 'EXIT';
    if (st.psiPrev === null) st.psiPrev = A.psi; st.psiRate += ((A.psi - st.psiPrev) / dt - st.psiRate) * Math.min(1, dt / 0.2); st.psiPrev = A.psi;
    let dWant = P.decel;
    if (st.exit && st.lat === 'ROLLOUT') { const vT = Math.sqrt(VEX * VEX + 2 * 1.4 * Math.max(0, st.exit.x0 - REX * Math.tan(st.exit.ang / 2) - s.p[0])); dWant = clamp(0.6 * (V - vT), 0, 3); }   // brake-to-vacate: a speed profile that reaches the exit at 40 kt
    if (st.lat === 'EXIT' || st.lat === 'CLEAR') {       // pure pursuit: the centreline to the exit point, then the exit; curvature within the lateral budget
      const e = st.exit, P2 = e.pts, Ld = clamp(2.2 * V, 25, 70); let bd = Infinity;
      for (let i = e.i; i < Math.min(P2.length, e.i + 60); i++) { const d = (P2[i][0] - s.p[0]) ** 2 + (P2[i][1] - s.p[2]) ** 2; if (d < bd) { bd = d; e.i = i; } }
      if (st.exitKt === undefined && e.i >= 150) st.exitKt = V / KT;   // the turn-off speed: where the lead-in curve begins
      const tgt = P2[Math.min(P2.length - 1, e.i + Math.round(Ld / 2))], pr = e.i * 2 - 300, th = Math.atan2(tgt[1] - s.p[2], tgt[0] - s.p[0]), eta = Math.atan2(Math.sin(th - A.psi), Math.cos(th - A.psi));
      const kap = clamp(2 * Math.sin(eta) / Ld, -ALAT / Math.max(V * V, 1), ALAT / Math.max(V * V, 1)); cmd.steer = Math.atan(LWB * kap) - 0.3 * st.psiRate * (V > 5 ? 1 : 0);
      const vT = s.p[2] > 110 ? 0 : pr < 200 ? VEX : 15 * KT; dWant = clamp(0.5 * (V - vT), 0, 2.5); if (s.p[2] > 110) { st.lat = 'CLEAR'; dWant = 2; }
    } else {
      const zm = bodyToWorld(s, [-2.5, -6.6, 0], st.rx)[2], psiRef = clamp(-P.kSz * zm - P.kSzd * s.v[2], -6 * DEG, 6 * DEG);   // head back toward the centreline
      cmd.steer = clamp(P.kSpsi * (psiRef - A.psi) - P.kSr * st.psiRate, -VEH.steerMax, VEH.steerMax);
    }
    if (st.noseT !== null && s.t - st.noseT >= P.brakeDelay) { st.brakeI = clamp(st.brakeI + P.kBrake * (dWant - st.decel) * dt, 0, 1); cmd.brake = V < 0.5 || (st.lat === 'CLEAR' && V < 3) ? 1 : st.brakeI; }
    if (V < 0.3) { st.vert = 'STOP'; if (st.lat !== 'CLEAR') st.lat = 'STOP'; }
    st.mode = st.vert; return cmd;
  }
  return { st, cmd, plan, update, params: P };
}
