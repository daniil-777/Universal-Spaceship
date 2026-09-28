import { test } from 'node:test';
import assert from 'node:assert/strict';
import { VEH, liftCoeff, alphaForLift, isaDensity } from '../src/landing/vehicle.js';
import { newFlight, stepFlight, trimFlight, bodyToWorld } from '../src/landing/flight.js';

const H = 1 / 120, KT = 0.514444, DEG = Math.PI / 180, calm = { at: (p, t, o) => { o[0] = 0; o[1] = 0; o[2] = 0; return o; } };
const cmd = (o = {}) => ({ nz: 1, phi: 0, beta: 0, thr: null, gear: 1, spoil: 0, brake: 0, chute: 0, steer: 0, ...o });
const fly = (s, sec, c, wind = calm, each = null) => { for (let i = 0; i < Math.round(sec / H); i++) { stepFlight(s, c, wind, H); if (each) each(s, i); } return s; };
const gamma = (s) => Math.asin(s.v[1] / Math.hypot(s.v[0], s.v[1], s.v[2]));

test('vehicle: Polhamus lift — 0.59 at 11°, 1.14 at the 20° limit; the inverse finds α', () => {
  assert.ok(Math.abs(liftCoeff(11 * DEG) - 0.59) < 0.01, 'CL(11°) ' + liftCoeff(11 * DEG)); assert.ok(Math.abs(liftCoeff(20 * DEG) - 1.14) < 0.02, 'CL(20°) ' + liftCoeff(20 * DEG));
  for (const a of [2, 7, 11, 16, 19]) assert.ok(Math.abs(alphaForLift(liftCoeff(a * DEG)) - a * DEG) < 1e-3, 'α ' + a);
  assert.ok(Math.abs(isaDensity(0) - 1.225) < 1e-9 && isaDensity(1000) < 1.13 && isaDensity(1000) > 1.1);
});

test('flight: on a 3° glide path at 155 kt with the gear down it trims at α ≈ 11° on 60–160 kN of thrust and holds it', () => {
  const s = newFlight(); trimFlight(s, { x: -3000, h: 160, V: 155 * KT, gamma: -3 * DEG, gear: 1 });
  assert.ok(Math.abs(s.air.alpha - 11 * DEG) < 1.5 * DEG, 'α ' + (s.air.alpha / DEG).toFixed(1)); assert.ok(s.trimThrust > 60e3 && s.trimThrust < 160e3, 'thrust ' + s.trimThrust.toFixed(0));
  const g0 = gamma(s); fly(s, 10, cmd({ thr: s.trimThr }));
  assert.ok(Math.abs(gamma(s) - g0) < 0.3 * DEG, 'path ' + (gamma(s) / DEG).toFixed(2)); assert.ok(Math.abs(s.air.V - 155 * KT) < 2, 'speed ' + (s.air.V / KT).toFixed(1));
});

test('flight: full back stick is held at the α-limit, never past it', () => {
  const s = newFlight(); trimFlight(s, { x: -6000, h: 600, V: 140 * KT, gamma: 0, gear: 1 }); let top = 0;
  fly(s, 8, cmd({ nz: 3, thr: 1 }), calm, (x) => { top = Math.max(top, x.air.alpha); });
  assert.ok(top < VEH.alphaMax + 0.5 * DEG, 'α reached ' + (top / DEG).toFixed(1)); assert.ok(top > VEH.alphaMax - 2 * DEG, 'α ' + (top / DEG).toFixed(1));
});

test('flight: ground effect — less drag and more lift a few metres up than at altitude, same α and speed', () => {
  const hi = newFlight(), lo = newFlight(); trimFlight(hi, { x: -3000, h: 300, V: 150 * KT, gamma: 0, gear: 1 }); trimFlight(lo, { x: 100, h: 1.5, V: 150 * KT, gamma: 0, gear: 1, alpha: hi.air.alpha });
  stepFlight(hi, cmd({ thr: 0.3 }), calm, H); stepFlight(lo, cmd({ thr: 0.3 }), calm, H);
  assert.ok(lo.air.CL > hi.air.CL * 1.03, `CL ${lo.air.CL.toFixed(3)} vs ${hi.air.CL.toFixed(3)}`); assert.ok(lo.air.CDi < hi.air.CDi * 0.95, `CDi ${lo.air.CDi.toFixed(3)} vs ${hi.air.CDi.toFixed(3)}`);
});

test('gear: parked, the three legs carry the weight (within 2 %) and the tail clears the ground', () => {
  const s = newFlight(); trimFlight(s, { x: 500, onGround: true }); fly(s, 6, cmd({ thr: 0, brake: 1 }));
  const load = s.legs.reduce((a, l) => a + l.force, 0); assert.ok(Math.abs(load / (VEH.mass * 9.81) - 1) < 0.02, 'gear load ' + (load / (VEH.mass * 9.81)).toFixed(3));
  assert.ok(Math.abs(s.v[0]) < 0.05 && s.legs.every((l) => l.onGround), 'at rest on all legs'); assert.ok(bodyToWorld(s, VEH.tail)[1] > 1, 'tail height ' + bodyToWorld(s, VEH.tail)[1]);
});

test('gear: a 3 ft/s touchdown is soft (1.2–1.45 g), the 10 ft/s design case 1.9–2.5 g, neither bounces', () => {
  const drop = (fps) => { const s = newFlight(); trimFlight(s, { x: 400, h: 0.3, V: 150 * KT, gamma: -Math.asin(fps * 0.3048 / (150 * KT)), gear: 1 }); let peak = 0, air = 0, first = null;
    fly(s, 3, cmd({ nz: 1, thr: 0.04, spoil: 1 }), calm, (x) => { peak = Math.max(peak, x.gearG); if (first && !x.legs.some((l) => l.onGround)) air = Math.max(air, bodyToWorld(x, VEH.legs[1].contact)[1]); if (!first && x.touchdown) first = x.touchdown; });
    return { peak, air, td: first }; };
  const soft = drop(3), hard = drop(10);
  assert.ok(soft.td && Math.abs(soft.td.sinkFps - 3) < 0.6, 'recorded sink ' + (soft.td && soft.td.sinkFps)); assert.ok(soft.peak > 1.2 && soft.peak < 1.45 && soft.air < 0.05, `soft: ${soft.peak.toFixed(2)} g, bounce ${soft.air.toFixed(2)} m`);   // oleos with rebound damping: 1.36 g
  assert.ok(hard.peak > 1.9 && hard.peak < 2.5 && hard.air < 0.05 && hard.td.sinkFps > 9, `hard: ${hard.peak.toFixed(2)} g at ${hard.td.sinkFps.toFixed(1)} ft/s, bounce ${hard.air.toFixed(2)}`);   // the 10 ft/s design case: 2.26 g
});

test('rollout: 45 % brake pressure decelerates at 2.5–3.5 m/s² (spoilers, drag, tyres), the chute adds, and the nose wheel steers', () => {
  const dec = (brake, chute) => { const s = newFlight(); trimFlight(s, { x: 600, onGround: true, V: 120 * KT }); fly(s, 0.5, cmd({ thr: 0.04, spoil: 1, brake, chute })); const v0 = s.v[0]; fly(s, 2, cmd({ thr: 0.04, spoil: 1, brake, chute })); return (v0 - s.v[0]) / 2; };
  const med = dec(0.45, 0); assert.ok(med > 2.5 && med < 3.5, 'brakes at 45 % ' + med.toFixed(2)); assert.ok(dec(0.45, 1) > med + 0.8, 'chute adds');
  const s = newFlight(); trimFlight(s, { x: 600, onGround: true, V: 30 * KT }); fly(s, 3, cmd({ thr: 0.1, steer: 5 * DEG }));
  assert.ok(s.v[2] > 0.5, 'turned right: v_z ' + s.v[2].toFixed(2));
});
