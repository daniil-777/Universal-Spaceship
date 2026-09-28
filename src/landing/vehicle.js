// The landing scenario's vehicle: the Astro Pilot ship at full scale — the model ×9, a 36-m delta-wing spaceplane with
// a powered approach, tricycle landing gear, ground spoilers and a drag chute. SI units; body frame x forward, y up,
// z right, origin at the centre of mass. Numbers are of the Space Shuttle orbiter / Concorde class (mass, wing loading,
// approach α, gear geometry sized for a 15° tail-strike angle).
const DEG = Math.PI / 180;
export const SCALE = 9;                                   // model units (src/ship.js) → metres
export const G = 9.81, KT = 0.514444, FT = 0.3048;

export const VEH = Object.freeze({
  mass: 72000, S: 300, span: 30.6, chord: 13, AR: 30.6 * 30.6 / 300,
  Kp: 2.6, Kv: 3.2, alphaMax: 20 * DEG, alphaBreak: 30 * DEG,   // Polhamus potential + vortex lift; α-protection; vortex breakdown
  cd0: 0.021, suction: 0.3, cdGear: 0.018, cdSpoil: 0.05, liftDump: 0.35, chuteCdS: 70, cyBeta: 0.9,
  thrMax: 400e3, idle: 0.04, spoolUp: 1.8, spoolDown: 1.2,   // two engines; spool time constants (s)
  I: [1.3e6, 8.9e6, 8.6e6],                               // roll (x), yaw (y), pitch (z) inertia, kg·m²
  ctrl: [0.05, 0.04, 0.12],                               // control-moment coefficients (per q·S·b for roll/yaw, per q·S·c for pitch)
  legs: Object.freeze([                                   // attachment (body), wheel contact when extended (body), spring, damper (×rebound when extending: no bounce)
    { name: 'nose', attach: [13.5, -4.0, 0], contact: [13.5, -6.6, 0], stroke: 0.45, k: 4.1e5, c: 7e4, rebound: 4, brake: false, steer: true },
    { name: 'left', attach: [-2.5, -4.0, -4.6], contact: [-2.5, -6.6, -4.6], stroke: 0.45, k: 1.1e6, c: 1.2e5, rebound: 4, brake: true, steer: false },
    { name: 'right', attach: [-2.5, -4.0, 4.6], contact: [-2.5, -6.6, 4.6], stroke: 0.45, k: 1.1e6, c: 1.2e5, rebound: 4, brake: true, steer: false },
  ]),
  tail: [-17.1, -2.7, 0], eye: [15.5, 1.2, -0.6], wing: [-1.0, -1.0, 0],   // tail-strike point, pilot's eye, wing reference (ground effect)
  muRoll: 0.015, muBrake: 0.55, muSide: 0.6, cornering: 8, steerMax: 8 * DEG, tailStrike: 15 * DEG,
});

export function liftCoeff(a) {                            // Polhamus: K_p sin α cos²α + K_v cos α |sin α| sin α; past the breakdown the vortex lift fades
  const s = Math.sin(a), c = Math.cos(a), vx = a > VEH.alphaBreak ? Math.max(0, 1 - (a - VEH.alphaBreak) / (10 * DEG)) : 1;
  return VEH.Kp * s * c * c + VEH.Kv * c * Math.abs(s) * s * vx;
}
const AT = new Float64Array(401), CT = new Float64Array(401);   // the inverse on a table (−5° … +25°), refined by Newton
for (let i = 0; i <= 400; i++) { AT[i] = (-5 + i * 30 / 400) * DEG; CT[i] = liftCoeff(AT[i]); }
export function alphaForLift(cl) {
  if (cl <= CT[0]) return AT[0]; if (cl >= CT[400]) return AT[400];
  let lo = 0, hi = 400; while (hi - lo > 1) { const m = (lo + hi) >> 1; if (CT[m] < cl) lo = m; else hi = m; }
  let a = AT[lo] + (cl - CT[lo]) / (CT[hi] - CT[lo]) * (AT[hi] - AT[lo]);
  for (let k = 0; k < 2; k++) { const e = 1e-5, d = (liftCoeff(a + e) - liftCoeff(a - e)) / (2 * e); a -= (liftCoeff(a) - cl) / d; }
  return a;
}
export const inducedDrag = (cl, a) => (1 - VEH.suction) * cl * Math.tan(Math.abs(a)) + VEH.suction * cl * cl / (Math.PI * VEH.AR);   // a sharp delta loses most leading-edge suction
export function groundEffect(hw) {                        // wing height above the ground → [lift factor, induced-drag factor] (McCormick; a delta's lift gain)
  const r = Math.max(0.02, hw / VEH.span), m = 16 * r;
  return [1 + 0.25 * Math.exp(-5 * r), (m * m) / (1 + m * m)];
}
export const isaDensity = (h) => 1.225 * (1 - 2.25577e-5 * h) ** 4.25588;   // ISA troposphere
