// vlm/gen/labels/corridor_facts.js — the S/A facts of spec §4.2 from the env and the capture camera, each {v, unit, obs}.
// Non-finite values (no closing, no ground in space) are stored as null: validateRecord rejects Infinity (Review Focus 1).
import { ENV } from '../../../src/envconst.js';
import { qInvRotate } from '../../../src/mathx.js';
import { AERO } from '../../../src/aero.js';
import { fact, HAZARD_KINDS } from '../schema.js';
import { projectSphere, cameraPosition } from './camera.js';
import { M_PER_U, boardWarning, clearance, hazardGeometry } from './corridor.js';

const DEG = 180 / Math.PI, fin = (x) => (Number.isFinite(x) ? x : null), r3 = (x) => (Number.isFinite(x) ? +x.toFixed(3) : null);
// T10-v: the atmospheric corridor's own scene fog (src/app.js:267,307 — a THREE.Fog, linear/smoothstep, not the
// landing's exponential-squared FogExp2: near/far by terrain, always on while `atmosphere` is set). The same 10%
// transmittance cutoff (landing.js T_VIS) gates hazard visual facts here too, using three.js's own linear-fog law
// (fog_fragment.glsl.js: fogFactor = smoothstep(near, far, d); transmittance is the rest of the signal).
const T_VIS = 0.1, smooth3 = (t) => t * t * (3 - 2 * t);
const fogTransmittance = (near, far, d) => (d <= near ? 1 : d >= far ? 0 : 1 - smooth3((d - near) / (far - near)));
const dist3 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
// T10-v live check (t10v-smoke, alps storm, task-10v-report.md): the linear scene fog above never fires at real hazard
// distances (all < 62 u across 22 captured records, well inside its 70 u near plane), yet a frame at air.in_cloud ~1 is
// a near-total whiteout: a hazard ~4-10 u from the camera is still a faint blur, one >= ~24 u is not visible at all. The
// storm's own volumetric cloud, not the corridor's atmospheric haze, is what hides a hazard at gameplay range, so it
// gates too, measured (not invented): IN_CLOUD_VIS_U sits between the two clusters.
const IN_CLOUD_A = 0.5, IN_CLOUD_VIS_U = 12;
export const clockOf = (b) => { const h = Math.round(Math.atan2(b[2], b[0]) * DEG / 30); return ((h % 12) + 12) % 12 || 12; };
// screen-relative clock (spec §4.2: "pixel box and clock sector projected with the capture camera"): the angle of the
// hazard's projected pixel centre about the image centre, 12 o'clock = up in the image; clockOf above stays the ship-
// body (pilot-relative) bearing, exposed separately as the `bearing_clock` fact.
const screenClockOf = (cx, cy, W, H) => { const h = Math.round(Math.atan2(cx - W / 2, H / 2 - cy) * DEG / 30); return ((h % 12) + 12) % 12 || 12; };
const sizeBin = (px) => (px < 8 ? 'tiny' : px < 24 ? 'small' : px < 64 ? 'medium' : 'large');
const sideOf = (x, W) => (x < W / 3 ? 'left' : x > 2 * W / 3 ? 'right' : 'centre');
const turbClass = (s) => (s < 0.8 ? 'LIGHT' : s < 2 ? 'MOD' : 'SEVERE');

export function corridorFacts(env, cam, { W = 896, H = 504, sky = null, route = null, space = null, fog = null } = {}) {
  const F = {}, put = (id, v, unit, obs) => { F[id] = fact(v, unit, obs); }, s = env.ship, air = !!env.atmosphere, body = new Float64Array(3);
  const speed = Math.hypot(...s.v);
  put('ship.speed_u_s', r3(speed), 'u/s', 'context'); put('ship.speed_m_s', r3(speed * M_PER_U), 'm/s', 'context');
  put('ship.pitch_deg', r3(Math.asin(s.f[1]) * DEG), 'deg', 'context'); put('ship.bank_deg', r3(Math.atan2(-s.r[1], s.u[1]) * DEG), 'deg', 'context');
  put('ship.heading_deg', r3(Math.atan2(-s.f[2], s.f[0]) * DEG), 'deg', 'context');
  put('ship.cmd', [...env.cmd].map(r3), null, 'context'); put('ship.applied', [...env.prevA].map(r3), null, 'context');
  // visible = in the projected frame and, when a scene fog is given (A: the corridor's own THREE.Fog), transmittance
  // at the camera's distance to the hazard's surface is still >= T_VIS; and, in a storm's volumetric cloud, still
  // within the measured close range. S has no fog and no weather (fog stays null, cloudNow stays 0): unchanged.
  const eye = cameraPosition(cam), cloudNow = env.weather ? env.weather.cloudAt(s.p[0], s.p[1], s.p[2]) : 0;
  const visible = (p, c, r) => {
    if (!p.inFrame) return false;
    const d = Math.max(0, dist3(eye, c) - r);
    if (fog && fogTransmittance(fog.near, fog.far, d) < T_VIS) return false;
    return !(cloudNow >= IN_CLOUD_A && d > IN_CLOUD_VIS_U);
  };
  const inFrameKinds = new Set(); let nIn = 0;
  for (const a of env.asteroids) { const g = hazardGeometry(env, a), c = [s.p[0] + g.rel[0], s.p[1] + g.rel[1], s.p[2] + g.rel[2]], p = projectSphere(cam, c, a.r, W, H); if (visible(p, c, a.r)) { nIn++; inFrameKinds.add(HAZARD_KINDS[a.kind || 0]); } }
  put('hazards.count_in_frame', nIn, 'count', 'visual'); put('kinds_in_frame', [...inFrameKinds].sort(), null, 'visual');
  env.nearest.slice(0, 6).forEach((j, i) => {
    const a = env.asteroids[j], g = hazardGeometry(env, a), c = [s.p[0] + g.rel[0], s.p[1] + g.rel[1], s.p[2] + g.rel[2]], p = projectSphere(cam, c, a.r, W, H), vv = visible(p, c, a.r), vis = vv ? 'visual' : 'context';
    qInvRotate(s.q, g.rel, body);
    put(`hazard.${i}.kind`, HAZARD_KINDS[a.kind || 0], null, vis); put(`hazard.${i}.in_frame`, vv, null, 'visual');
    put(`hazard.${i}.box_px`, vv ? p.box.map((v) => +v.toFixed(1)) : null, 'px', 'visual'); put(`hazard.${i}.side`, vv ? sideOf(p.cx, W) : null, null, 'visual');
    put(`hazard.${i}.clock`, vv ? screenClockOf(p.cx, p.cy, W, H) : null, 'clock', 'visual'); put(`hazard.${i}.bearing_clock`, clockOf(body), 'clock', 'context');
    put(`hazard.${i}.size_bin`, vv ? sizeBin(2 * p.rPx) : null, null, 'visual');
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
    put('air.in_cloud', env.weather ? r3(cloudNow) : 0, null, 'visual');
    put('weather.cells', env.weather ? env.weather.cellsNear(s.p[0], 60).map((c) => ({ type: c.type, dist_u: r3(Math.hypot(c.x - s.p[0], c.z - s.p[2])), bearing_deg: r3(Math.atan2(c.z - s.p[2], c.x - s.p[0]) * DEG) })) : [], null, 'context');
  }
  put('weather.preset', sky, null, 'visual'); put('weather.knobs', [env.weatherSeverity, env.weatherWind, env.weatherCover, env.weatherTurb].map(r3), null, 'context');
  put('world', env.world, null, 'visual'); put('route', route, null, 'visual'); put('in_tunnel', !!(env.tunnelZone && env.tunnelZone(s.p[0])), null, 'visual');
  put('warning', boardWarning(env), null, 'context');
  // a value the page does not provide stays null (unknown), never false or undefined
  if (space) { put('sun.lit', r3(space.sunLit), null, 'visual'); put('orbit.body', space.body ?? null, null, 'visual'); put('orbit.lat_deg', r3(space.lat), 'deg', 'context'); put('orbit.lon_deg', r3(space.lon), 'deg', 'context');
    put('earth_in_frame', space.earthInFrame ?? null, null, 'visual'); put('moon_in_frame', space.moonInFrame ?? null, null, 'visual'); }
  return F;
}
