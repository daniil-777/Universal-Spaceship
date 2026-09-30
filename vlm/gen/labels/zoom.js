// vlm/gen/labels/zoom.js — Z facts from zoom.info and the captured camera matrices (spec §4.2 Z, §4.3 zoom): the 16x9
// ground grid by ray-sphere intersection (R = 6371 km, centre (0, -R, 0) in the view's local frame; approximate by
// h*tan(tilt)), lighting, r_min, tags. poseCamera() rebuilds the camera from cameraPose() for the Node sampler and tests.
import { R_KM, createLocalFrame, cameraPose, sunLocal, tileSizeKm, sourceForLevel } from '../../../src/earthtiles.js';
import { fact, rangeBin } from '../schema.js';
import { lookAtCamera, pixelRay, cameraPosition } from './camera.js';

export const GRID_NX = 16, GRID_NY = 9, ALWAYS_DAY = Object.freeze({ elev: 66.7, az: 54.5 });
const DEG = Math.PI / 180;
export function raySphere(o, d) { const c = [o[0], o[1] + R_KM, o[2]], b = c[0] * d[0] + c[1] * d[1] + c[2] * d[2], q = c[0] ** 2 + c[1] ** 2 + c[2] ** 2 - R_KM * R_KM, disc = b * b - q; return disc < 0 ? null : -b - Math.sqrt(disc); }
export function poseCamera(view, { W = 896, H = 504 } = {}) {
  const pose = cameraPose(createLocalFrame(view.lat, view.lon), { groundKm: 0, ...view }), clear = Math.max(0.02, pose.camAltKm);
  const cam = lookAtCamera({ eye: pose.pos, target: pose.look, up: pose.up, fovDeg: 45, aspect: W / H, near: Math.min(50, Math.max(0.02, 0.3 * clear)), far: 1e5 });
  cam.eye = pose.pos.slice(); return cam;
}
export function matrixGrid(cam, origin, { nx = GRID_NX, ny = GRID_NY, W = 896, H = 504, margin = 0 } = {}) {
  const frame = createLocalFrame(origin.lat, origin.lon), out = [];
  for (let j = -margin; j < ny + margin; j++) for (let i = -margin; i < nx + margin; i++) {
    const px = (i + 0.5) * W / nx, py = (j + 0.5) * H / ny, r = pixelRay(cam, px, py, W, H), t = raySphere(r.origin, r.dir);
    if (t === null || t < 0) { out.push(null); continue; }
    const ll = frame.toLatLon(r.origin.map((o, k) => o + t * r.dir[k])); out.push({ i, j, px, py, lat: ll.lat, lon: ((ll.lon + 540) % 360) - 180 });
  }
  return out;
}
export function sunAngles(utcMs, lat, lon) { const s = sunLocal(utcMs, createLocalFrame(lat, lon)); return { elev: Math.asin(s[1]) / DEG, az: ((Math.atan2(s[0], -s[2]) / DEG) + 360) % 360 }; }
export const lightingClass = (e) => (e < -6 ? 'night' : e < 0 ? 'twilight' : e < 10 ? 'golden' : 'day');
export function lightingOf({ mode, utcMs, lat, lon }) {
  if (mode === 'always_day') return { mode, sun_elev_deg: ALWAYS_DAY.elev, sun_az_deg: ALWAYS_DAY.az, class: 'day' };
  const s = sunAngles(utcMs, lat, lon); return { mode: 'utc', sun_elev_deg: +s.elev.toFixed(2), sun_az_deg: +s.az.toFixed(2), class: lightingClass(s.elev) };
}
export const rMinKm = (latDeg, Lmax) => 0.011533 * Math.cos(latDeg * DEG) * 504 * 2 ** (14 - Lmax) * 1.15;
export function zoomFacts(info, cam, { origin, lighting, rings }) {
  const F = {}, put = (id, v, unit) => { F[id] = fact(v, unit, 'visual'); }, r2 = (x) => +(+x).toFixed(4), grid = matrixGrid(cam, origin);
  put('view.lat_deg', r2(info.lat), 'deg'); put('view.lon_deg', r2(info.lon), 'deg'); put('view.range_km', r2(info.rangeKm), 'km'); put('view.tilt_deg', r2(info.tilt / DEG), 'deg');
  put('view.heading_deg', r2(info.heading / DEG), 'deg'); put('view.cam_alt_km', r2(info.camAltKm), 'km'); put('view.clear_km', r2(info.clearKm), 'km'); put('view.L0', info.L0, null);
  put('view.rings', rings, null); put('view.gsd_m', r2(Math.max(tileSizeKm(info.L0, info.lat) * 1000 / 256, sourceForLevel(info.L0).maxLevel > 8 ? 10 : 0)), 'm');
  put('view.footprint_km', r2(info.rangeKm * 2 * Math.tan(22.5 * DEG)), 'km'); put('view.range_bin', rangeBin(info.rangeKm), null);
  put('sun.elev_deg', lighting.sun_elev_deg, 'deg'); put('sun.az_deg', lighting.sun_az_deg, 'deg'); put('sun.class', lighting.class, null);
  put('grid.latlon', grid.map((g) => (g ? [r2(g.lat), r2(g.lon)] : null)), 'deg'); put('grid.approx_km', r2(Math.max(0, info.clearKm) * Math.tan(info.tilt)), 'km');
  put('camera.eye_km', cameraPosition(cam).map(r2), 'km');
  return F;
}
export function zoomTags({ geo, view, lighting }) {
  const t = new Set(), land = 1 - geo.sea_frac;
  if (geo.sea_frac > 0.6) t.add('WATER_DOMINANT');
  if (geo.sea_frac >= 0.05 && geo.sea_frac <= 0.95) t.add('COASTLINE');
  if (land > 0 && geo.relief !== null) {
    const mtn = view.rangeKm < 100 ? geo.relief > 1500 : (geo.cellReliefFrac[1500] ?? 0) >= 0.3, hill = !mtn && (view.rangeKm < 100 ? geo.relief >= 300 : (geo.cellReliefFrac[300] ?? 0) >= 0.3);
    if (mtn) t.add('MOUNTAINS'); else if (hill) t.add('HILLS'); else t.add('FLAT');
  }
  if (geo.mean !== null && geo.mean > 2000) t.add('HIGH_TERRAIN');
  if (geo.urban) t.add('URBAN'); if (geo.desert) t.add('DESERT'); if (geo.ice || Math.abs(view.lat) > 70) t.add('ICE');
  if (lighting.class === 'night') t.add('NIGHT');
  return [...t];
}
