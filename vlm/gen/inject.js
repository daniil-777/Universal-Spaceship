// vlm/gen/inject.js — the fault-injection catalogue of spec §3.3 (verified in probe_hazards/probe_scen and the review's
// inject_timing_probe): one injection per episode, drawn uniformly over its interval; the failed-P6 seed list.
// Controller rulings (Task 4): L alt_plus_60 only at 550-700 ft; D attitude_kick only at rho >= 20 m (closer, the port
// swing leaves the cone: an immediate KOS fail); lateral_drift in two bands, 0.05-0.1 m/s at 20-40 m (recovered) and
// 0.5 m/s at 20-25 m in CORRIDOR (KOS fail ~10 s later); inbound at 1.2-1.6 m/s (above speedLimit) at 300-400 m.
// A trigger fires when its qty drops to the drawn value (and, with trigger.phase, only in that guidance phase).
export const INJECT_RATE = Object.freeze({ S: 0.25, A: 0.10, L: 0.35, D: 0.35 });
export const INJECTIONS = Object.freeze({
  S: [{ kind: 'collision_course', at: 'runtime', qty: 'hit_s', lo: 1, hi: 4 }],
  A: [{ kind: 'collision_course', at: 'runtime', qty: 'hit_s', lo: 1, hi: 4 }],
  L: [{ kind: 'alt_plus_60', at: 'runtime', qty: 'hRAft', lo: 550, hi: 700 }, { kind: 'speed_plus_25', at: 'runtime', qty: 'hRAft', lo: 800, hi: 1500 },
    { kind: 'lateral_25', at: 'runtime', qty: 'hRAft', lo: 30, hi: 200 }, { kind: 'hflare', at: 'load', params: { hFlare: 0.3 } },
    { kind: 'wind_severe', at: 'load', url: { wind: '35040G55', turb: 'severe' } }, { kind: 'tailwind', at: 'load', url: { wind: '08025' } },
    { kind: 'forced_ga', at: 'runtime', qty: 'hRAft', lo: 50, hi: 1500 }],
  D: [{ kind: 'closing_plus_0.2', at: 'runtime', qty: 'axial_m', lo: 1.5, hi: 2.5 }, { kind: 'radial_plus_0.08', at: 'runtime', qty: 'axial_m', lo: 0.5, hi: 1.5 },
    { kind: 'lateral_drift', band: 'recovered', at: 'runtime', qty: 'rho_m', lo: 20, hi: 40, dv: [0.05, 0.1] },
    { kind: 'lateral_drift', band: 'kos', at: 'runtime', qty: 'rho_m', lo: 20, hi: 25, dv: [0.5, 0.5], phase: 'CORRIDOR' },
    { kind: 'inbound', at: 'runtime', qty: 'rho_m', lo: 300, hi: 400, dv: [1.2, 1.6] },
    { kind: 'failed_p6', at: 'construction' }, { kind: 'abort', at: 'runtime', qty: 'rho_m', lo: 0.5, hi: 20 }, { kind: 'attitude_kick', at: 'runtime', qty: 'rho_m', lo: 20, hi: 40 }],
});
// seeds 1..20000 whose drawRun(seed) fails P6 (start-independent); tests/vlm_docking.test.mjs recomputes the list
export const P6_SEEDS = Object.freeze([614, 637, 660, 734, 912, 1910, 2590, 2642, 3216, 3875, 4128, 4407, 5492, 5501, 5887, 5964, 6243, 6340, 6448, 7440, 7810, 7860, 7965, 8019, 8027,
  8743, 8908, 8980, 9648, 9670, 9830, 10216, 11003, 11466, 12680, 12740, 12754, 13291, 13729, 13833, 13986, 14379, 14832, 15164, 16154, 16319, 17349, 17377, 17429, 17678,
  17681, 17762, 18011, 18035, 18164, 18532, 18761, 19393, 19615]);
export function drawInjection(family, rng) {
  if (rng() >= INJECT_RATE[family]) return null;
  const list = INJECTIONS[family], e = list[Math.floor(rng() * list.length)], params = { ...(e.params || {}) };
  if (e.dv) params.dv = e.dv[0] + rng() * (e.dv[1] - e.dv[0]);
  if (e.band) params.band = e.band;
  if (e.kind === 'attitude_kick') { const a = [rng() - 0.5, rng() - 0.5, rng() - 0.5], l = Math.hypot(...a); params.axis = a.map((v) => v / l); }
  if (e.kind === 'failed_p6') params.seed = P6_SEEDS[Math.floor(rng() * P6_SEEDS.length)];
  return { kind: e.kind, at: e.at, trigger: e.qty ? { qty: e.qty, value: e.lo + rng() * (e.hi - e.lo), phase: e.phase ?? null } : null, params, url: e.url || null };
}
