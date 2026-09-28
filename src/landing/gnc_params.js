// The autoland's tunable parameters (SI; angles in radians). Tuned by CMA-ES (test/landing_tune.mjs: λ 14, 80 seeded
// final approaches in drawn weather, cost = mean + ½ the worst decile) from standard-practice defaults — round 3,
// 2026-09-28. Validated on 2400 unseen landings with the descent-aware localizer join (test/landing_mc.mjs: seeds 20001,
// 40001, 60001 × 500 full profile; 30001, 50001, 70001 × 300 vectored finals): all landed, stabilized at 1000 ft 99.9 %;
// sink 2.3 ft/s median (spec §8 FOQA band 1.5–3), 5.2 max; touchdown 393–564 m (median 456); |y| ≤ 5.4 m; bank ≤ 3.0°;
// crab ≤ 3.5°; pitch ≤ 10.8°; ≤ 1.6 g; rollout |y| ≤ 8.0 m (the 8.0 in a crosswind gusting 29 kt, beyond the envelope).
// Round 2 (softer, 1.9 ft/s, more crab: 5.2°) is out/gnc_params_r2_backup.js on LaCie; round 4 drifted 8.6 m on a rollout.
const DEG = Math.PI / 180, KT = 0.514444;
export const GNC = Object.freeze({
  hInt: 610, finalStraight: 4000, bankPlan: 25 * DEG, radiusMargin: 1.15,        // approach geometry: GS intercept 2000 ft, 4 km of LOC before it
  bankMax: 25 * DEG, bankLow: 3.03 * DEG, lowAlt: 50.0, l1Period: 20, l1PeriodFinal: 32.5, l1Damp: 0.800,
  kH: 0.0550, gMaxDesc: 3.5 * DEG, gMaxClimb: 3 * DEG, kGamma: 1.187, kGammaFlare: 0.953,
  vNav: 200 * KT, vLoc: 180 * KT, vapp: 157 * KT, vAdd: 0.5, vAddMax: 15 * KT,   // V_app + half the headwind + gust and turbulence additives, ≤ 15 kt
  kV: 0.240, aMax: 0.9, kE: 0.683, kEI: 0.108,
  hFlare: 11.08, sinkTD: 0.715, sinkMin: 0.490, sinkMax: 0.891, aim: 541.5, hRetard: 1.25, hAlign: 1.085, decrab: 0.754, betaMax: 10 * DEG, pitchMax: 12.46 * DEG,
  derot: 3.42 * DEG, decel: 2.5, brakeDelay: 1.5, kBrake: 0.35, kSz: 0.0169, kSzd: 0.0705, kSpsi: 1.740, kSr: 1.159, chuteKt: 170, chuteOffKt: 60,
});
