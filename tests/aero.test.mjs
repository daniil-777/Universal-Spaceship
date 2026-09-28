import { test } from 'node:test';
import assert from 'node:assert/strict';
import { AERO, airDensity, liftCoeff, stepAir, newAirState, trimAttitude } from '../src/aero.js';
import { Dryden } from '../src/turbulence.js';
import { mulberry32, qIdentity, qAxes, qFromAxisAngle, qMul } from '../src/mathx.js';

const H = 1 / 60, DEG = Math.PI / 180;
const stubAir = (wind, sigma = 0, field = null) => ({   // uniform air (or a wind field) over a valley floor at y = −26
  windAt: (x, y, z, o) => { if (field) return field(x, y, z, o); o[0] = wind[0]; o[1] = wind[1]; o[2] = wind[2]; return o; },
  turbAt: (x, y, z, o) => { o.sigma = sigma; o.hA = y + 26; o.share = 0; o.Lc = 0; return o; } });
function airEnv({ y = -12, V = 14, bank = 0, wind = null, sigma = 0, auto = true } = {}) {
  const ship = { p: new Float64Array([0, y, 0]), v: new Float64Array([V, 0, 0]), q: qIdentity(), w: new Float64Array(3), f: new Float64Array(3), u: new Float64Array(3), r: new Float64Array(3) };
  if (bank) qMul(ship.q, qFromAxisAngle(1, 0, 0, bank, new Float64Array(4)), ship.q);
  qAxes(ship.q, ship.f, ship.u, ship.r);
  const weather = wind || sigma ? stubAir(wind || [0, 0, 0], sigma) : null;
  const env = { ship, air: newAirState(), cmd: new Float32Array(4), weather, hf: null, dryden: new Dryden(mulberry32(1)), rngAir: mulberry32(2), autoThrottle: auto, speedTarget: V };
  trimAttitude(env); return env;
}
function fly(env, sec, cmd = [0, 0, 0, 0], each = null) {
  env.cmd.set(cmd);
  for (let i = 0; i < Math.round(sec / H); i++) { stepAir(env, H); for (let k = 0; k < 3; k++) env.ship.p[k] += H * env.ship.v[k]; if (each) each(env); }
  return env;
}
const speed = (e) => Math.hypot(...e.ship.v);

test('air density: 1 on the valley floor, ≈ 0.55 at the thin-air ceiling', () => {
  assert.ok(Math.abs(airDensity(-26) - 1) < 1e-12); assert.ok(Math.abs(airDensity(12) - 0.552) < 0.01);
});
test('lift: linear up to the stall at 16°, then it falls; symmetric', () => {
  assert.ok(Math.abs(liftCoeff(5 * DEG) - 4 * 5 * DEG) < 1e-9);
  assert.ok(liftCoeff(24 * DEG) < liftCoeff(16 * DEG)); assert.ok(Math.abs(liftCoeff(-10 * DEG) + liftCoeff(10 * DEG)) < 1e-12);
});
test('trim: neutral stick + auto-throttle holds level flight at cruise for 10 s', () => {
  const e = fly(airEnv(), 10);
  assert.ok(Math.abs(e.ship.p[1] + 12) < 0.5, 'height ' + e.ship.p[1]); assert.ok(Math.abs(speed(e) - 14) < 0.8, 'speed ' + speed(e));
  assert.ok(e.air.throttle > 0.3 && e.air.throttle < 0.6, 'throttle ' + e.air.throttle); assert.ok(!e.air.stalled && !e.air.overstressed);
});
test('stall speed ≈ 9 u/s: at 7 u/s the protected wing cannot hold the ship, at 11 u/s it can', () => {
  const slow = fly(airEnv({ V: 7, auto: false }), 2), ok = fly(airEnv({ V: 11 }), 2);
  assert.ok(slow.ship.p[1] < -13, 'slow sinks ' + slow.ship.p[1]); assert.ok(Math.abs(ok.ship.p[1] + 12) < 1, '11 u/s holds ' + ok.ship.p[1]);
});
test('gust load grows with airspeed', () => {
  const peak = (V) => { let m = 0; fly(airEnv({ V, wind: [0, 2.5, 0] }), 0.5, [0, 0, 0, 0], (e) => { m = Math.max(m, e.air.n); }); return m; };
  const p10 = peak(10), p18 = peak(18); assert.ok(p18 > p10 + 0.3, `n at 18: ${p18}, at 10: ${p10}`);
});
test('overstress: a violent updraft at high speed breaks the airframe, a mild one at cruise does not', () => {
  assert.ok(fly(airEnv({ V: 22, wind: [0, 9, 0] }), 0.5).air.overstressed);
  assert.ok(!fly(airEnv({ V: 14, wind: [0, 1, 0] }), 1).air.overstressed);
});
test('banked turn: neutral stick holds height at 60° of bank, radius ≈ 12', () => {
  const e = fly(airEnv({ bank: 60 * DEG }), 2), h0 = e.ship.p[1], a0 = Math.atan2(e.ship.v[2], e.ship.v[0]);
  fly(e, 1); let da = Math.atan2(e.ship.v[2], e.ship.v[0]) - a0; da = Math.atan2(Math.sin(da), Math.cos(da));
  const radius = speed(e) / Math.abs(da); assert.ok(Math.abs(e.ship.p[1] - h0) < 1.5, 'height change ' + (e.ship.p[1] - h0)); assert.ok(radius > 9 && radius < 15, 'radius ' + radius);
});
test('power: full throttle accelerates, idle decelerates', () => {
  assert.ok(speed(fly(airEnv({ auto: false }), 5, [0, 0, 0, 1])) > 16); assert.ok(speed(fly(airEnv({ auto: false }), 5, [0, 0, 0, -1])) < 12);
});
test('near-zero airspeed stays finite', () => {
  const e = fly(airEnv({ V: 0.3, auto: false }), 3);
  for (const a of [e.ship.p, e.ship.v, e.ship.q, e.ship.w]) for (const v of a) assert.ok(Number.isFinite(v)); assert.ok(Number.isFinite(e.air.n));
});
const bankOf = (e) => Math.atan2(-e.ship.r[1], e.ship.u[1]);
test('bank protection: full roll stick stops near 67°', () => {
  let worst = 0; fly(airEnv(), 3, [0, 0, 1, 0], (e) => { worst = Math.max(worst, Math.abs(bankOf(e))); });
  assert.ok(worst < 70 * DEG, 'bank reached ' + (worst / DEG).toFixed(0) + '°');
});
test('bank protection: an upset to 120° with neutral stick rolls back without pulling into the ground', () => {
  const e = airEnv({ bank: 120 * DEG }), y0 = e.ship.p[1]; fly(e, 2);
  assert.ok(Math.abs(bankOf(e)) < 70 * DEG, 'bank ' + (bankOf(e) / DEG).toFixed(0) + '°'); assert.ok(y0 - e.ship.p[1] < 16, 'height lost ' + (y0 - e.ship.p[1]).toFixed(1));   // ballistic while rolling back, then the held (descending) path: 13.5; the old law pulled 2.5 g into the ground: 24.7
});

test('a sharp-edged gust builds up over the chord: the load peaks ≥ 3 substeps after it hits, the first substep feels < 40 % of it', () => {
  const e = fly(airEnv({ y: 5 }), 1), n0 = e.air.n, n = []; e.weather = stubAir([0, 2.5, 0]);
  fly(e, 0.5, [0, 0, 0, 0], (x) => n.push(x.air.n)); const top = Math.max(...n), i = n.indexOf(top);
  assert.ok(i >= 3, 'peak at substep ' + i); assert.ok(n[0] - n0 < 0.4 * (top - n0), `first substep ${(n[0] - n0).toFixed(3)} of ${(top - n0).toFixed(3)}`);
});
test('gusts in a headwind as fast as the ship are as smooth as in calm air: their axes follow the track through the air', () => {
  const worstJump = (head) => {
    const e = airEnv({ y: 5, sigma: 1.5, wind: [head ? -13.5 : 0, 0, 0] }); if (head) e.ship.v[0] = 0.5;   // 0.5 u/s over the ground, 14 through the air
    let prev = null, worst = 0; fly(e, 30, [0, 0, 0, 0], (x) => { const g = x.air.gust; if (prev) worst = Math.max(worst, Math.hypot(g[0] - prev[0], g[1] - prev[1], g[2] - prev[2])); prev = Array.from(g); });
    return worst;
  };
  const calm = worstJump(false), head = worstJump(true); assert.ok(head < 1.3 * calm, `headwind ${head.toFixed(2)} u/s vs calm ${calm.toFixed(2)} u/s`);
});
test('rough air (σ 1, 31 u above the ground): the load factor varies about as much as before the Dryden rework (RMS n − 1 0.128)', () => {
  const e = airEnv({ y: 5, sigma: 1 }); let s2 = 0, k = 0; fly(e, 60, [0, 0, 0, 0], (x) => { s2 += (x.air.n - 1) ** 2; k++; });   // first-order filters, no chord lag, same setup: 0.128
  const rms = Math.sqrt(s2 / k); assert.ok(rms > 0.128 * 0.85 && rms < 0.128 * 1.15, 'RMS n − 1: ' + rms.toFixed(3));
});
test('a pass beside an updraft core rolls the ship away from it (more lift under the inner wing), turbulence off', () => {
  const core = (x, y, z, o) => { o[0] = 0; o[1] = 2 * Math.exp(-((x - 30) ** 2 + (z - 8) ** 2) / 100); o[2] = 0; return o; };   // W 2, R 10, the axis 0.8 R to the right of the track
  const e = airEnv({ y: 5 }); e.weather = stubAir(null, 0, core); let lo = 0, hi = 0;
  fly(e, 4, [0, 0, 0, 0], (x) => { const b = bankOf(x); lo = Math.min(lo, b); hi = Math.max(hi, b); });
  assert.ok(lo < -1 * DEG && lo > -8 * DEG, 'rolled left by ' + (-lo / DEG).toFixed(1) + '°'); assert.ok(hi < -0.5 * lo, 'right by ' + (hi / DEG).toFixed(1) + '°');
});
test('turbulence rocks the wings a little: RMS roll rate at σ 0.5 with the stick at neutral', () => {
  const e = airEnv({ y: 5, sigma: 0.5 }); let s2 = 0, k = 0; fly(e, 60, [0, 0, 0, 0], (x) => { s2 += x.ship.w[0] ** 2; k++; });
  const rms = Math.sqrt(s2 / k); assert.ok(rms > 0.015 && rms < 0.03, 'RMS roll rate ' + rms.toFixed(4));
});
