// vlm/gen/labels/rawstate.js — V1-2 (APV v1 plan): the raw per-frame world state a record keeps so new label types (2D boxes
// and grounding, landing-site hazard maps, pose) can be derived at build time without re-capturing. Pure functions of the sim
// state (the probes call them in the page at every capture; Node tests call them directly). Every number is finite or null
// (validateRecord rejects non-finite numbers). Frames: S/A the env's corridor frame (u; hazard centres at the rendered,
// x-wrapped position, as corridorFacts projects them), L the runway frame (m), D LVLH (m, station at the origin).
import { wrapX, qRotate } from '../../../src/mathx.js';
import { ENV } from '../../../src/envconst.js';
import { RWY, AIRPORT } from '../../../src/landing/airport.js';
import { STATION_PORT, SHIP_PORT, portRel } from '../../../src/real/consts.js';
import { attError, omegaRel } from '../../../src/real/rigid.js';
import { HAZARD_KINDS } from '../schema.js';
import { projectSphere, cameraPosition } from './camera.js';

export const RAW_VERSION = 1;
const rn = (d) => (x) => (Number.isFinite(x) ? +x.toFixed(d) : null), r4 = rn(4), r6 = rn(6), arr = (a, f = r4) => Array.from(a, f);
// S/A visibility as corridorFacts gates it (T10-v): in the projected frame, the scene fog's transmittance at the camera's
// distance to the hazard surface >= T_VIS (A's linear THREE.Fog; S has none), and inside a storm cloud (cloudAt >= 0.5) only
// within IN_CLOUD_VIS_U. Occlusion by terrain or another hazard is not modelled (null: unknown).
const T_VIS = 0.1, IN_CLOUD_A = 0.5, IN_CLOUD_VIS_U = 12, smooth3 = (t) => t * t * (3 - 2 * t);
const fogT = (near, far, d) => (d <= near ? 1 : d >= far ? 0 : 1 - smooth3((d - near) / (far - near)));
// S/A: the ship pose, and every hazard that is in this frame (projected with the frame's own camera, any size) or among the
// env's nearest (the ones the facts describe), with its world centre, velocity, orientation, radius, kind, pixel box (null out
// of frame) and the fog/cloud visibility flag
export function corridorState(env, cam, { W = 896, H = 504, fog = null } = {}) {
  const s = env.ship, near = new Set(env.nearest || []), hazards = [], eye = cameraPosition(cam), cloudNow = env.weather ? env.weather.cloudAt(s.p[0], s.p[1], s.p[2]) : 0;
  env.asteroids.forEach((a, i) => {
    const c = [s.p[0] + wrapX(a.p[0] - s.p[0], ENV.xHalf), a.p[1], a.p[2]], p = projectSphere(cam, c, a.r, W, H);
    if (!p.inFrame && !near.has(i)) return;
    const d = Math.max(0, Math.hypot(eye[0] - c[0], eye[1] - c[1], eye[2] - c[2]) - a.r), fogged = !!fog && fogT(fog.near, fog.far, d) < T_VIS, clouded = cloudNow >= IN_CLOUD_A && d > IN_CLOUD_VIS_U;
    hazards.push({ i, kind: HAZARD_KINDS[a.kind || 0], r: r4(a.r), c: c.map(r4), v: arr(a.v), q: a.q ? arr(a.q, r6) : null, in_frame: !!p.inFrame, box_px: p.inFrame ? p.box.map(rn(1)) : null,
      visible: !!p.inFrame && !fogged && !clouded, fogged, in_cloud: clouded, occluded: null, cam_dist: r4(d) });
  });
  return { ship: { p: arr(s.p), v: arr(s.v), q: arr(s.q, r6) }, cloud_at_ship: r4(cloudNow), fog: fog ? { near: r4(fog.near), far: r4(fog.far) } : null, hazards };
}
// L: the aircraft pose in the runway frame and the fog density the frame was rendered with (the geometry is sceneOf('L'))
export function landingState(sim, fogDensity = 0) {
  const f = sim.flight;
  return { aircraft: { p: arr(f.p), v: arr(f.v), q: arr(f.q, r6), w: f.w ? arr(f.w, r6) : null, wow: !!f.wow, gear: r4(f.gear ?? null) }, fog_density: rn(6)(fogDensity) };
}
// D: the ship pose in LVLH (x = [position, velocity]) and its port's world position; the station port is sceneOf('D')
export function dockingState(s) {
  const x = s.x, pr = qRotate(s.q, SHIP_PORT, [0, 0, 0]);
  return { ship: { x: arr(x.slice(0, 3)), v: arr(x.slice(3, 6)), q: arr(s.q, r6), w: s.w ? arr(s.w, r6) : null }, ship_port_w: [0, 1, 2].map((i) => r4(x[i] + pr[i])),
    port_rel: arr(portRel(x, s.q, [0, 0, 0])), att_err_rad: arr(attError(s.q, [0, 0, 0, 1], [0, 0, 0]), r6), omega_rel: s.w ? arr(omegaRel(s.q, s.w, [0, 0, 0]), r6) : null };
}
// the static scene geometry per family (null for S/A/Z: the corridor hazards are per frame, Z keeps grid.latlon and the camera)
export function sceneOf(family) {
  // L: the runway corners (threshold 26 at x 0 to the far end at x = length, +-width/2 across, elevation y), PAPI units, windsock, ILS
  if (family === 'L') { const hw = RWY.width / 2, y = RWY.elevation; return { frame: 'runway_m', runway: { ...RWY, corners: [[0, y, -hw], [0, y, hw], [RWY.length, y, hw], [RWY.length, y, -hw]] }, papi: AIRPORT.papi.map((u) => ({ ...u })), windsock: { ...AIRPORT.windsock }, ils: JSON.parse(JSON.stringify(AIRPORT.ils)) }; }
  if (family === 'D') return { frame: 'lvlh_m', station_port: [...STATION_PORT], ship_port: [...SHIP_PORT] };
  return null;
}
// the record's snapshot block: per-frame state keyed by frame name (plus the D chase frame), the scene; null when no frame has state
export function rawOf(family, named, chase = null) {
  const frames = named.filter(([, st]) => st).concat(chase && chase[1] ? [chase] : []);
  return frames.length ? { v: RAW_VERSION, frames: Object.fromEntries(frames), scene: sceneOf(family) } : null;
}
