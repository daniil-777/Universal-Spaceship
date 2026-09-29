// vlm/gen/labels/corridor_facts.js — the S/A facts of spec §4.2 from the env and the capture camera, each {v, unit, obs}.
// Non-finite values (no closing, no ground in space) are stored as null: validateRecord rejects Infinity (Review Focus 1).
import { ENV } from '../../../src/envconst.js';
import { qInvRotate } from '../../../src/mathx.js';
import { AERO } from '../../../src/aero.js';
import { fact } from '../schema.js';
import { projectSphere } from './camera.js';
import { HAZARD_KIND, M_PER_U, boardWarning, clearance, hazardGeometry } from './corridor.js';

const DEG = 180 / Math.PI, fin = (x) => (Number.isFinite(x) ? x : null), r3 = (x) => (Number.isFinite(x) ? +x.toFixed(3) : null);
export const clockOf = (b) => { const h = Math.round(Math.atan2(b[2], b[0]) * DEG / 30); return ((h % 12) + 12) % 12 || 12; };
const sizeBin = (px) => (px < 8 ? 'tiny' : px < 24 ? 'small' : px < 64 ? 'medium' : 'large');
const sideOf = (x, W) => (x < W / 3 ? 'left' : x > 2 * W / 3 ? 'right' : 'centre');
const turbClass = (s) => (s < 0.8 ? 'LIGHT' : s < 2 ? 'MOD' : 'SEVERE');

export function corridorFacts(env, cam, { W = 896, H = 504, sky = null, route = null, space = null } = {}) {
  const F = {}, put = (id, v, unit, obs) => { F[id] = fact(v, unit, obs); }, s = env.ship, air = !!env.atmosphere, body = new Float64Array(3);
  const speed = Math.hypot(...s.v);
  put('ship.speed_u_s', r3(speed), 'u/s', 'context'); put('ship.speed_m_s', r3(speed * M_PER_U), 'm/s', 'context');
  put('ship.pitch_deg', r3(Math.asin(s.f[1]) * DEG), 'deg', 'context'); put('ship.bank_deg', r3(Math.atan2(-s.r[1], s.u[1]) * DEG), 'deg', 'context');
  put('ship.heading_deg', r3(Math.atan2(-s.f[2], s.f[0]) * DEG), 'deg', 'context');
  put('ship.cmd', [...env.cmd].map(r3), null, 'context'); put('ship.applied', [...env.prevA].map(r3), null, 'context');
  const inFrameKinds = new Set(); let nIn = 0;
  for (const a of env.asteroids) { const g = hazardGeometry(env, a), p = projectSphere(cam, [s.p[0] + g.rel[0], s.p[1] + g.rel[1], s.p[2] + g.rel[2]], a.r, W, H); if (p.inFrame) { nIn++; inFrameKinds.add(HAZARD_KIND[a.kind || 0]); } }
  put('hazards.count_in_frame', nIn, 'count', 'visual'); put('kinds_in_frame', [...inFrameKinds].sort(), null, 'visual');
  env.nearest.slice(0, 6).forEach((j, i) => {
    const a = env.asteroids[j], g = hazardGeometry(env, a), c = [s.p[0] + g.rel[0], s.p[1] + g.rel[1], s.p[2] + g.rel[2]], p = projectSphere(cam, c, a.r, W, H), vis = p.inFrame ? 'visual' : 'context';
    qInvRotate(s.q, g.rel, body);
    put(`hazard.${i}.kind`, HAZARD_KIND[a.kind || 0], null, vis); put(`hazard.${i}.in_frame`, p.inFrame, null, 'visual');
    put(`hazard.${i}.box_px`, p.inFrame ? p.box.map((v) => +v.toFixed(1)) : null, 'px', 'visual'); put(`hazard.${i}.side`, p.inFrame ? sideOf(p.cx, W) : null, null, 'visual');
    put(`hazard.${i}.clock`, clockOf(body), 'clock', vis); put(`hazard.${i}.size_bin`, p.inFrame ? sizeBin(2 * p.rPx) : null, null, 'visual');
    put(`hazard.${i}.r_u`, r3(a.r), 'u', 'context'); put(`hazard.${i}.dist_u`, r3(g.dist), 'u', 'context'); put(`hazard.${i}.closing_u_s`, r3(g.closing), 'u/s', 'context');
    put(`hazard.${i}.ttc_s`, g.ttc === null ? null : r3(g.ttc), 's', 'context'); put(`hazard.${i}.cpa_u`, r3(g.miss), 'u', 'context'); put(`hazard.${i}.tca_s`, r3(g.tca), 's', 'context');
  });
  let nh = Infinity, ni = -1; for (let i = 0; i < env.rayHit.length; i++) if (env.rayHit[i] < nh) { nh = env.rayHit[i]; ni = i; }
  put('beams.nearest_u', r3(nh), 'u', 'context'); put('beams.nearest_is_edge', ni >= 0 && !!env.rayEdge[ni], null, 'context');
  const top = env.hf ? ENV.mountains.ceiling : ENV.yHalf;
  put('clearance.ground_u', env.hf ? r3(fin(env.groundClearance())) : null, 'u', 'context'); put('clearance.min_u', r3(fin(clearance(env).value)), 'u', 'context');
  put('edges.left_u', r3(ENV.zHalf + s.p[2]), 'u', 'context'); put('edges.right_u', r3(ENV.zHalf - s.p[2]), 'u', 'context');
  put('edges.ceiling_u', r3(top - s.p[1]), 'u', 'context'); put('edges.floor_u', env.hf ? null : r3(s.p[1] + ENV.yHalf), 'u', 'context');
  if (air) {
    const ai = env.air;
    put('air.V_u_s', r3(ai.V), 'u/s', 'context'); put('air.alpha_deg', r3(ai.alpha * DEG), 'deg', 'context'); put('air.beta_deg', r3(ai.beta * DEG), 'deg', 'context');
    put('air.stall_margin_deg', r3((AERO.alphaStall - ai.alpha) * DEG), 'deg', 'context'); put('air.n_g', r3(ai.n), 'g', 'context');
    put('air.stalled', !!ai.stalled, null, 'context'); put('air.overstressed', !!ai.overstressed, null, 'context'); put('air.rho', r3(ai.rho), null, 'context');
    put('air.sigma_felt', r3(ai.sigmaFelt), 'g', 'context'); put('air.turbulence', turbClass(ai.sigmaFelt), null, 'context');
    put('air.wind_u_s', [...ai.wind].map(r3), 'u/s', 'context'); put('air.climb_u_s', r3(ai.climb), 'u/s', 'context');
    put('air.speed_band', ai.V < 0.8 * AERO.cruise ? 'slow' : ai.V > AERO.vne ? 'overspeed' : 'normal', null, 'context');
    put('air.agl_m', env.hf ? r3((s.p[1] - env.hf.height(s.p[0], s.p[2])) * M_PER_U) : null, 'm', 'context');
    put('air.in_cloud', env.weather ? r3(env.weather.cloudAt(s.p[0], s.p[1], s.p[2])) : 0, null, 'visual');
    put('weather.cells', env.weather ? env.weather.cellsNear(s.p[0], 60).map((c) => ({ type: c.type, dist_u: r3(Math.hypot(c.x - s.p[0], c.z - s.p[2])), bearing_deg: r3(Math.atan2(c.z - s.p[2], c.x - s.p[0]) * DEG) })) : [], null, 'context');
  }
  put('weather.preset', sky, null, 'visual'); put('weather.knobs', [env.weatherSeverity, env.weatherWind, env.weatherCover, env.weatherTurb].map(r3), null, 'context');
  put('world', env.world, null, 'visual'); put('route', route, null, 'visual'); put('in_tunnel', !!(env.tunnelZone && env.tunnelZone(s.p[0])), null, 'visual');
  put('warning', boardWarning(env), null, 'context');
  if (space) { put('sun.lit', r3(space.sunLit), null, 'visual'); put('orbit.body', space.body, null, 'visual'); put('orbit.lat_deg', r3(space.lat), 'deg', 'context'); put('orbit.lon_deg', r3(space.lon), 'deg', 'context');
    put('earth_in_frame', !!space.earthInFrame, null, 'visual'); put('moon_in_frame', !!space.moonInFrame, null, 'visual'); }
  return F;
}
