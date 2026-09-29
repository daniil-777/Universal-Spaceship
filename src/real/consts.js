// Real-spacecraft scenario: every constant, the state layout and the geometry, in SI units.
// Frame: LVLH (orbit.js) with X along-track, Y radial up, Z = X x Y (the orbit anti-normal), origin at the station CoM.
// Translational state: Float64Array(6) [X, Y, Z, Xd, Yd, Zd] in m and m/s.
// Attitude: q = Float64Array(4) [x, y, z, w] body -> LVLH; w = body rate relative to inertial (rad/s, body axes).
// Body axes: x nose, y up, z right; origin at the ship CoM. Jet d = plume direction, so the force is -T * d.
import { circularOrbit } from '../orbit.js';
import { qRotate } from '../mathx.js';

const ORBIT = circularOrbit({ h: 420 });
export const N = ORBIT.n;
export const T_ORB = ORBIT.period;
export const DAY = 86400;
export const DEG = Math.PI / 180;
export const G0 = 9.80665;
// differential-drag bound (m/s^2) and the LVLH frame rate in LVLH coordinates
export const DRAG_B = 2e-7;
export const OMEGA_L = Object.freeze([0, 0, -N]);

export const MASS0 = 95000;
export const PROP0 = 600;
export const INERTIA = Object.freeze([1.715e6, 11.74e6, 11.35e6]);
export const QUANTUM = 0.01;
export const CYCLE = 0.1;
export const KIND = Object.freeze({
  P: Object.freeze({ thrust: 3870, isp: 280, mib: 0.04 }),
  V: Object.freeze({ thrust: 107, isp: 265, mib: 0.08 }),
});

const jet = (name, kind, pos, d) => Object.freeze({ name, kind, pos: Object.freeze(pos), d: Object.freeze(d) });
// Spec section 5 layout (body m); P1-P12 are ship.js RCS[0..11] x 9, P13/P14 the aft nozzles, P15/P16 the nose.
export const JETS = Object.freeze([
  jet('P1', 'P', [13.05, 0.78, 0], [0, 1, 0]), jet('P2', 'P', [13.05, -1.73, 0], [0, -1, 0]),
  jet('P3', 'P', [-13.5, 1.13, 0], [0, 1, 0]), jet('P4', 'P', [-13.5, -1.70, 0], [0, -1, 0]),
  jet('P5', 'P', [13.05, -0.65, 1.77], [0, 0, 1]), jet('P6', 'P', [13.05, -0.65, -1.77], [0, 0, -1]),
  jet('P7', 'P', [-16.2, 0, 6.48], [0, 0, 1]), jet('P8', 'P', [-16.2, 0, -6.48], [0, 0, -1]),
  jet('P9', 'P', [-13.5, 1.26, 14.4], [0, 1, 0]), jet('P10', 'P', [-13.5, 0.72, 14.4], [0, -1, 0]),
  jet('P11', 'P', [-13.5, 1.26, -14.4], [0, 1, 0]), jet('P12', 'P', [-13.5, 0.72, -14.4], [0, -1, 0]),
  jet('P13', 'P', [-16.9, 0, 2.0], [-1, 0, 0]), jet('P14', 'P', [-16.9, 0, -2.0], [-1, 0, 0]),
  jet('P15', 'P', [17.3, -0.9, 0.4], [1, 0, 0]), jet('P16', 'P', [17.3, -0.9, -0.4], [1, 0, 0]),
  jet('V1', 'V', [11.7, 1.13, 0], [0, 1, 0]), jet('V2', 'V', [11.7, -1.78, 0], [0, -1, 0]),
  jet('V3', 'V', [11.7, -0.52, 2.07], [0, 0, 1]), jet('V4', 'V', [11.7, -0.52, -2.07], [0, 0, -1]),
  jet('V5', 'V', [-13.5, 0.72, 14.4], [0, -1, 0]), jet('V6', 'V', [-13.5, 0.72, -14.4], [0, -1, 0]),
]);
export const JET_INDEX = Object.freeze(Object.fromEntries(JETS.map((j, i) => [j.name, i])));
const ids = (...names) => Object.freeze(names.map((n) => JET_INDEX[n]));
// translation sets by body axis and sign: [+X, -X, +Y, -Y, +Z, -Z]
export const TRANS_SETS = Object.freeze([
  ids('P13', 'P14'), ids('P15', 'P16'), ids('P2', 'P4'), ids('P1', 'P3'), ids('P6', 'P8'), ids('P5', 'P7'),
]);
// the Y and Z sets: a lone primary there is torque-balanced only by its pair partner, which cancels its force
export const LATERAL_SETS = Object.freeze(TRANS_SETS.slice(2));
// vernier fine-trim jets by body axis and sign: [+Y, -Y, +Z, -Z] (a single primary on Y or Z needs the opposite
// member of its torque pair to balance it, which also cancels its force)
export const VERNIER_TRANS = Object.freeze(ids('V2', 'V1', 'V4', 'V3'));
export const ATT_JETS = Object.freeze({
  P: ids('P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12'),
  V: ids('V1', 'V2', 'V3', 'V4', 'V5', 'V6'),
});
// ship.js glow slots 0-13 -> jets (slots 0/1 were the main nozzles, 2-13 the RCS puffs)
export const GLOW_SLOTS = Object.freeze(ids('P13', 'P14', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12'));

// Docking ports. SHIP_PORT = the nose tip (ship.js HULL_KEYS x 2.10, centreY -0.110) x 9, body m.
// STATION_PORT = the Kvant-1 aft port, satellites.js (y -2.06 - 0.25) x 6 with the station long axis on LVLH +X.
export const SHIP_PORT = Object.freeze([18.9, -0.99, 0]);
export const STATION_PORT = Object.freeze([-13.86, 0, 0]);
export const PORT_SUM = SHIP_PORT[0] - STATION_PORT[0];
export const rFromRho = (rho) => rho + PORT_SUM;
export const rhoFromR = (r) => r - PORT_SUM;
const _pv = new Float64Array(3);
// station port -> ship port in LVLH (m); x = translational state, q = attitude
export function portRel(x, q, out = new Float64Array(3)) {
  qRotate(q, SHIP_PORT, _pv);
  out[0] = x[0] + _pv[0] - STATION_PORT[0];
  out[1] = x[1] + _pv[1] - STATION_PORT[1];
  out[2] = x[2] + _pv[2] - STATION_PORT[2];
  return out;
}

// Ship collision spheres [x, y, z, radius], body m: 98 % of the x9 hull vertices lie within 0.5 m of them.
export const SHIP_SPHERES = Object.freeze([
  [15.0, -0.7, 0, 2.4], [9.6, 0.6, 0, 4.2], [-0.8, 0.6, 0, 7.5], [-11.0, 1.3, -9.6, 7.5],
  [-11.0, 1.3, 9.6, 7.5], [-10.8, -0.2, 0, 3.7], [-16.9, 0.2, 0, 5.8],
].map((s) => Object.freeze(s)));
// Station capsules [[ax, ay, az], [bx, by, bz], radius], LVLH m (satellites.js r 13.2 = scale 6, held in LVLH).
const cap = (a, b, r) => Object.freeze([Object.freeze(a), Object.freeze(b), r]);
export const STATION_CAPSULES = Object.freeze([
  cap([-11.24, 0, 0], [12.12, 0, 0], 1.86),
  cap([-5.1, 2.2, 0], [-5.1, 13.45, 0], 1.65), cap([-5.1, -2.2, 0], [-5.1, -13.45, 0], 1.65),
  cap([-10.5, 0, 2.2], [-10.5, 0, 8.1], 1.3), cap([-10.5, 0, -2.2], [-10.5, 0, -8.1], 1.3),
  cap([4.5, 0, 0], [4.5, 8.6, 0], 1.7), cap([4.5, 0, 0], [4.5, -10.4, 0], 1.7),
  cap([4.5, 0, 0], [4.5, 0, 8.62], 1.7), cap([4.5, 0, 0], [4.5, 0, -8.62], 1.7),
  cap([-0.2, 7.8, 0], [9.2, 7.8, 0], 1.26), cap([-0.2, -7.8, 0], [9.2, -7.8, 0], 1.26),
  cap([-1.38, 0, 7.8], [10.38, 0, 7.8], 1.26), cap([-1.38, 0, -7.8], [10.38, 0, -7.8], 1.26),
  cap([12.06, 1.4, 0], [12.06, 4.74, 0], 0.78), cap([12.06, -1.4, 0], [12.06, -4.74, 0], 0.78),
  cap([-10.2, -1.68, 0], [-12.3, -9.6, 0], 0.6), cap([-10.2, 1.7, 0], [-10.2, 3.4, 0], 1.4),
  cap([1.5, 0, 0.9], [1.5, 0, 4.5], 0.1), cap([1.5, 0, -0.9], [1.5, 0, -4.5], 0.1),
  cap([8.2, 0.78, 0], [8.2, 2.82, 0], 0.06), cap([8.2, -0.78, 0], [8.2, -2.82, 0], 0.06),
]);

// Mission geometry (m, m/s, s)
export const KOS_R = 200;
export const KEEPIN_R = 3000;
export const NARROW_R = 60;
export const H1_POINT = Object.freeze([-250, 0, 0]);
export const H2_RHO = 20;
export const RULE_P_TRANSFER = 240;
export const CORRIDOR_HALF = 10 * DEG;
// the approach cone around the docking axis (-X from the station port), with a 1 m floor at the port itself
export const CONE_FLOOR = 1.0;
export const inCone = (p) => -p[0] > -0.5 && Math.hypot(p[1], p[2]) <= Math.max(Math.tan(CORRIDOR_HALF) * -p[0], CONE_FLOOR);
export const FINAL_SPEED = 0.07;
export const IDSS = Object.freeze({ closeMin: 0.05, closeMax: 0.10, lat: 0.04, rate: 0.2 * DEG, mis: 0.10, ang: 4 * DEG });
export const GO = Object.freeze({ att: 2 * DEG, rate: 0.1 * DEG, propFrac: 0.4 });
export const A_AV = 0.04;
export const BREAKOUT_V = Object.freeze([0.02, 0.10, 0]);
