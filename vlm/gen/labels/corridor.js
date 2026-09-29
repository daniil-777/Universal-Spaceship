// vlm/gen/labels/corridor.js — corridor (S/A) labels on the site's own SpaceEnv (spec R5, §4.3): a state-transplant clone
// with fresh noise streams (probe_corridor.mjs), the 7 macro actions (CONTINUE = HANDS_OFF, rotational pulses of 8 steps
// in space and 15 in the air, SPEED_UP/SLOW_DOWN held), K-draw rollouts with common random numbers, the crash cause, the
// board's warning, the search pilot of §3.4. Browser-loadable: the probe runs it in the page on the rendered env.
import { ENV, OBS_DIM, ACT_DIM } from '../../../src/envconst.js';
import { mulberry32, wrapX } from '../../../src/mathx.js';
import { Dryden } from '../../../src/turbulence.js';
import { createLongField } from '../../../src/cityfield.js';
import { CELL } from '../../../src/weather.js';
import { CORRIDOR_ACTIONS, corridorSafety, eyeView } from '../safety.js';

export const H_STEPS = 45, K_DRAWS = 4, M_PER_U = 19, PULSE = Object.freeze({ space: 8, air: 15 }), SEARCH = Object.freeze({ every: 8, H: 30 });
export const HAZARD_KIND = Object.freeze(['rock', 'comet', 'satellite', 'airliner', 'birds']);
export const atanhClamp = (c) => { const x = Math.max(-0.995, Math.min(0.995, c)); return 0.5 * Math.log((1 + x) / (1 - x)); };
const TERRAIN_WORLDS = ['mountains', 'pillars', 'meshy'];

export function cloneEnv(src, noiseSeed = 1) {
  const e = Object.create(Object.getPrototypeOf(src));
  for (const [k, v] of Object.entries(src)) {
    if (typeof v === 'function' || k === 'hf' || k === 'weather' || k === 'dryden' || k === 'radarSky' || k === 'lastAction') continue;
    if (ArrayBuffer.isView(v)) e[k] = v.slice();
    else if (k === 'asteroids') e[k] = v.map((h) => { const c = { ...h }; for (const f of ['p', 'v', 'q', 'axis']) if (h[f]) c[f] = h[f].slice(); return c; });
    else e[k] = v && typeof v === 'object' ? structuredClone(v) : v;
  }
  e.rng = mulberry32((noiseSeed * 2654435761) >>> 0); e.rngAir = mulberry32((noiseSeed ^ 0x27d4eb2f) >>> 0); e.rngSky = mulberry32((noiseSeed ^ 0x165667b1) >>> 0);
  e.dryden = new Dryden(mulberry32((noiseSeed ^ 0x5bd1e995) >>> 0)); e.dryden.x.set(src.dryden.x); e.dryden.p = src.dryden.p;
  if (src.hf && src.hf.lap) { e.hf = createLongField(src.hf.city, ENV.mountains.y0); e.hf.lap.shift = src.hf.lap.shift; } else e.hf = src.hf;
  e.weather = null; if (src.weather) e.buildWeather(src.weather);
  e.radarSky = e.weather; e.lastAction = e.prevA;
  return e;
}

export function macroAction(action, { air, uThr, pulse = air ? PULSE.air : PULSE.space }) {
  const hold = [0, 0, 0, uThr], rot = { CLIMB: [2, 0, 0, uThr], DESCEND: [-2, 0, 0, uThr], TURN_LEFT: [0, 2, 0, uThr], TURN_RIGHT: [0, -2, 0, uThr] }[action];
  if (action === 'CONTINUE') return () => hold;
  if (rot) return (i) => (i < pulse ? rot : hold);
  if (action === 'SPEED_UP') return () => [0, 0, 0, 2];
  if (action === 'SLOW_DOWN') return () => [0, 0, 0, -2];
  throw new Error(`unknown corridor action ${action}`);
}

export function clearance(env) {
  const s = env.ship.p, R = ENV.ship.radius; let value = Infinity, source = null;
  for (const a of env.asteroids) { const d = Math.hypot(wrapX(a.p[0] - s[0], ENV.xHalf), a.p[1] - s[1], a.p[2] - s[2]) - a.r - R; if (d < value) { value = d; source = 'hazard'; } }
  if (env.hf) { const g = env.groundClearance(); if (g < value) { value = g; source = TERRAIN_WORLDS.includes(env.world) ? 'terrain' : 'building'; } }
  return { value, source };
}

export function crashCause(env) {
  if (env.crashCause === 'overstress') return 'overstress';
  const s = env.ship.p, R = ENV.ship.radius; let near = null, nd = Infinity;
  for (const a of env.asteroids) { const d = Math.hypot(wrapX(a.p[0] - s[0], ENV.xHalf), a.p[1] - s[1], a.p[2] - s[2]) - a.r - R; if (d < 1e-6) return HAZARD_KIND[a.kind || 0]; if (d < nd) { nd = d; near = a; } }
  const hf = env.hf;
  if (hf) {
    const e = 0.7 * R, [x, y, z] = s;
    if (y - R < hf.height(x, z) || y - e < Math.max(hf.height(x + e, z), hf.height(x - e, z), hf.height(x, z + e), hf.height(x, z - e))) return TERRAIN_WORLDS.includes(env.world) ? 'terrain' : 'building';
    if (hf.ceiling) return 'roof';
  }
  return near ? HAZARD_KIND[near.kind || 0] : 'terrain';
}

export function rollout(env, action, { H = H_STEPS, noiseSeed = 1000, pulse } = {}) {
  const c = cloneEnv(env, noiseSeed), fn = typeof action === 'function' ? null : macroAction(action, { air: !!env.atmosphere, uThr: atanhClamp(env.cmd[3]), pulse });
  let minClr = Infinity;
  for (let i = 0; i < H; i++) {
    const r = c.step(fn ? fn(i) : action(c)), cl = clearance(c).value; if (cl < minClr) minClr = cl;
    if (r.done) return { crashed: true, step: i + 1, minClr, cause: crashCause(c) };
  }
  return { crashed: false, step: H, minClr, cause: null };
}

export function actionSearch(env, { H = H_STEPS, K = K_DRAWS, pulse } = {}) {
  const out = {};
  for (const a of CORRIDOR_ACTIONS) {
    const runs = []; for (let k = 0; k < K; k++) runs.push(rollout(env, a, { H, noiseSeed: 1000 + k, pulse }));
    const m = runs.map((r) => r.minClr).sort((x, y) => x - y), crashed = runs.filter((r) => r.crashed);
    out[a] = { k: K, crashes: crashed.length, minClr: m[0], medClr: K % 2 ? m[(K - 1) / 2] : (m[K / 2 - 1] + m[K / 2]) / 2, causes: crashed.map((r) => r.cause) };
  }
  return out;
}

export function hazardGeometry(env, a) {
  const s = env.ship, rel = [wrapX(a.p[0] - s.p[0], ENV.xHalf), a.p[1] - s.p[1], a.p[2] - s.p[2]], rv = [a.v[0] - s.v[0], a.v[1] - s.v[1], a.v[2] - s.v[2]];
  const cd = Math.hypot(...rel), dist = cd - a.r - ENV.ship.radius, closing = -(rv[0] * rel[0] + rv[1] * rel[1] + rv[2] * rel[2]) / Math.max(1e-6, cd), vv = rv[0] ** 2 + rv[1] ** 2 + rv[2] ** 2;
  const tca = vv > 1e-9 ? Math.max(0, -(rel[0] * rv[0] + rel[1] * rv[1] + rel[2] * rv[2]) / vv) : 0;
  return { rel, dist, closing, ttc: closing > 0 ? Math.max(0, dist) / closing : null, tca, miss: Math.hypot(rel[0] + rv[0] * tca, rel[1] + rv[1] * tca, rel[2] + rv[2] * tca) - a.r - ENV.ship.radius };
}
export function nearestThreat(env) {
  let best = null;
  for (const j of env.nearest) { const a = env.asteroids[j], g = hazardGeometry(env, a); if (g.ttc !== null && (!best || g.ttc < best.ttc_s)) best = { index: j, kind: HAZARD_KIND[a.kind || 0], ttc_s: g.ttc, tca_s: g.tca, cpa_u: g.miss, cpa_m: g.miss * M_PER_U }; }
  return best;
}
export const cpaTrigger = (env) => env.asteroids.some((a) => { const g = hazardGeometry(env, a); return g.closing > 0 && g.tca <= 3 && g.miss < 3; });

export function boardWarning(env) {
  const s = env.ship, air = !!env.atmosphere, ra = env.hf ? s.p[1] - env.hf.height(s.p[0], s.p[2]) : Infinity;
  let prox = Infinity, idx = -1; for (let i = 0; i < env.rayHit.length; i++) if (env.rayHit[i] < prox) { prox = env.rayHit[i]; idx = i; }
  if (env.done) return air && env.air.overstressed ? 'OVERSTRESS' : 'CRASH';
  if (air && env.air.stalled) return 'STALL';
  if (air && ra < 3 && s.v[1] < -1.5) return 'PULL UP';
  if (env.guard > 0.25 || (prox < 4 && idx >= 0 && env.rayEdge[idx])) return 'EDGE';
  return prox < 4 ? 'PROXIMITY' : '';
}

export function nowState(env) {
  const s = env.ship, air = !!env.atmosphere, ai = env.air, ra = env.hf ? s.p[1] - env.hf.height(s.p[0], s.p[2]) : Infinity, t = nearestThreat(env);
  let stormCell = false;
  if (air && env.weather) for (const c of env.weather.cellsNear(s.p[0], 60)) if (c.type === CELL.TOWERING && Math.hypot(c.x - s.p[0], c.z - s.p[2]) < c.R + 10) stormCell = true;
  return { overstressed: air && !!ai.overstressed, stalled: air && !!ai.stalled, pullUp: air && ra < 3 && s.v[1] < -1.5, turbSevere: air && ai.sigmaFelt >= 2,
    edge: boardWarning(env) === 'EDGE', stormCell, closingFast: !!t && t.cpa_u < 3 && t.tca_s <= 3, clrSource: clearance(env).source };
}

export function corridorLabel(env, { cNear, pilot = null, K = K_DRAWS } = {}) {
  const pulse = env.atmosphere ? PULSE.air : PULSE.space, branches = actionSearch(env, { K, pulse }), t = nearestThreat(env);
  let p_pilot = null;
  if (pilot) { let n = 0; for (let k = 0; k < K; k++) if (rollout(env, pilot, { noiseSeed: 1000 + k }).crashed) n++; p_pilot = n / K; }
  const input = { branches, now: nowState(env), ttc_s: t ? t.ttc_s : null, cpa_m: t ? t.cpa_m : null, p_pilot, pulse };
  return { safety: corridorSafety(input, { cNear }), safety_eye: eyeView(env.atmosphere ? 'A' : 'S', input, { cNear }), input };
}

export function ppoPilot(agent) { const obsN = new Float32Array(OBS_DIM); return (env) => { const out = new Float32Array(ACT_DIM); agent.normalize(env.obs, 1, obsN); agent.actMean(obsN, out); return out; }; }

export function createSearchPilot({ every = SEARCH.every, H = SEARCH.H } = {}) {
  let plan = null, i = 0;
  function replan(env) {
    const top = env.hf ? ENV.mountains.ceiling : ENV.yHalf, lz = ENV.zHalf + ENV.ship.wallClamp; let best = null;
    for (const a of CORRIDOR_ACTIONS) {
      const fn = macroAction(a, { air: !!env.atmosphere, uThr: atanhClamp(env.cmd[3]) }), c = cloneEnv(env, 7); let crash = 0, m = Infinity;
      for (let k = 0; k < H; k++) { const r = c.step(fn(k)), p = c.ship.p; m = Math.min(m, clearance(c).value, lz - Math.abs(p[2]), top - p[1]); if (r.done) { crash = 1; break; } }
      if (!best || crash < best.crash || (crash === best.crash && m > best.m)) best = { a, crash, m, fn };
    }
    return best;
  }
  return { act(env) { if (!plan || i >= every) { plan = replan(env); i = 0; } return Float32Array.from(plan.fn(i++)); }, get action() { return plan ? plan.a : null; } };
}
