// vlm/gen/build/ground.js — v1 grounding facts, derived at build time from what a record already holds (ruling V1-2, the
// research report's #1/#3/#9 and the controller's rulings on the v1 snapshot). Coordinates are integers 0-100 on the 896x504
// frame, x first (GeoChat/Molmo style; the square() stretch to 512² keeps fractions): a box is [x0,y0,x1,y1], a point [x,y].
//   ground.hazards   S/A (visual): the visible hazards of the narrator frame, nearest first: {kind, box, pt, alone}; from
//                    rec.snapshot (every hazard whose sphere projects into the frame, the fog/cloud gate applied) or, for a
//                    v0 record, from the six nearest hazards' facts. Boxes are clipped to the frame; a clipped area under
//                    2 px² is left out. Occlusion is not modelled: `alone` marks a hazard no nearer hazard's box covers by
//                    more than half, and only those are counted by pointing.
//   ground.complete  S/A (visual): true when ground.hazards lists every visible hazard (the snapshot, or a v0 record whose
//                    list matches hazards.count_in_frame)
//   ground.cv_sim    S/A (context): per listed hazard, the constant-velocity closest approach if nothing changes (sim units:
//                    cpa_cv_sim_u the miss distance to its surface, tca_cv_sim_s, ttc_cv_sim_s); for Pilot Eye and later
//                    Context lines, not for Narrator text (the §5.7 Context does not carry it)
//   ground.runway    L (visual): the runway outline in view from the chase camera: {box, pt (the threshold centre)}, when
//                    all four corners are in front of the camera, it overlaps the frame, and the scene fog leaves the
//                    threshold at transmittance >= T_VIS (exp² fog, as Task 10v), else null
// Z gets ground.features in zoomgeo.js (Natural Earth features at their pixel positions).
import path from 'node:path';

const W = 896, H = 504, T_VIS = 0.1;
export const nx = (x) => Math.round(Math.max(0, Math.min(100, (100 * x) / W)));
export const ny = (y) => Math.round(Math.max(0, Math.min(100, (100 * y) / H)));
export function clipBox(b) {
  if (!Array.isArray(b) || b.length !== 4 || b.some((v) => !Number.isFinite(v))) return null;
  const c = [Math.max(0, b[0]), Math.max(0, b[1]), Math.min(W, b[2]), Math.min(H, b[3])];
  return c[2] > c[0] && c[3] > c[1] ? c : null;
}
const area = (b) => Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
const overlap = (a, b) => area([Math.max(a[0], b[0]), Math.max(a[1], b[1]), Math.min(a[2], b[2]), Math.min(a[3], b[3])]);
export const normBox = (b) => [nx(b[0]), ny(b[1]), nx(b[2]), ny(b[3])];
export const centre = (b) => [nx((b[0] + b[2]) / 2), ny((b[1] + b[3]) / 2)];
export const iou = (a, b) => { const i = overlap(a, b), u = area(a) + area(b) - i; return u > 0 ? i / u : 0; };
const fact = (v, obs) => ({ v, unit: null, obs });
const sub = (a, b) => a.map((x, i) => x - b[i]), dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
// constant-velocity closest approach of a sphere (centre c, velocity v, radius r) to the ship (p, vs)
export function cvApproach(p, vs, c, v, r) {
  const d = sub(c, p), w = sub(v, vs), ww = dot(w, w), tca = ww > 1e-9 ? Math.max(0, -dot(d, w) / ww) : 0;
  const miss = Math.hypot(...d.map((x, i) => x + w[i] * tca)) - r, b = dot(d, w), cc = dot(d, d) - r * r, disc = b * b - ww * cc;
  const ttc = cc <= 0 ? 0 : ww > 1e-9 && disc >= 0 && b < 0 ? (-b - Math.sqrt(disc)) / ww : null;
  const r3 = (x) => (x === null ? null : +x.toFixed(3));
  return { cpa_cv_sim_u: r3(miss), tca_cv_sim_s: r3(tca), ttc_cv_sim_s: r3(ttc) };
}
function listed(rec) {
  const F = rec.facts || {}, snap = rec.snapshot && rec.snapshot.frames && rec.snapshot.frames[path.basename(rec.narrator_frame || '')];
  if (snap && Array.isArray(snap.hazards)) {
    const hs = snap.hazards.filter((h) => h.visible && h.in_frame).map((h) => ({ h, box: clipBox(h.box_px) })).filter((x) => x.box && area(x.box) >= 2)
      .sort((a, b) => a.h.cam_dist - b.h.cam_dist);
    const vec = (x) => Array.isArray(x) && x.length === 3 && x.every(Number.isFinite), cvOk = (h) => snap.ship && vec(snap.ship.p) && vec(snap.ship.v) && vec(h.c) && vec(h.v) && Number.isFinite(h.r);
    return { source: 'snapshot', complete: true, items: hs.map(({ h, box }) => ({ kind: h.kind, box, cv: cvOk(h) ? cvApproach(snap.ship.p, snap.ship.v, h.c, h.v, h.r) : null })) };
  }
  const items = [];
  for (let i = 0; i < 6; i++) {
    const g = (k) => (F[`hazard.${i}.${k}`] ? F[`hazard.${i}.${k}`].v : null), box = g('in_frame') === true ? clipBox(g('box_px')) : null;
    if (box && area(box) >= 2) items.push({ kind: g('kind'), box, cv: null });
  }
  const n = F['hazards.count_in_frame'] ? F['hazards.count_in_frame'].v : null;
  return { source: 'facts', complete: n === items.length, items };
}
export function groundHazards(rec) {
  const L = listed(rec);
  const items = L.items.filter((x) => typeof x.kind === 'string').map((x, j, all) => ({ kind: x.kind, box: normBox(x.box), pt: centre(x.box), alone: !all.slice(0, j).some((n) => overlap(n.box, x.box) > 0.5 * area(x.box)) }));
  const cv = L.items.filter((x) => typeof x.kind === 'string' && x.cv).map((x) => ({ kind: x.kind, ...x.cv }));
  return { hazards: items, complete: L.complete, cv: cv.length ? cv : null, source: L.source };
}
// the three.js camera (column-major matrixWorldInverse and projectionMatrix) applied to a world point: pixels, or null behind
export function project(cam, p) {
  const m = cam.matrixWorldInverse, q = cam.projectionMatrix, v = [0, 1, 2].map((r) => m[r] * p[0] + m[4 + r] * p[1] + m[8 + r] * p[2] + m[12 + r]);
  const c = [0, 1, 2, 3].map((r) => q[r] * v[0] + q[4 + r] * v[1] + q[8 + r] * v[2] + q[12 + r]);
  if (!(c[3] > 1e-6)) return null;
  return [((c[0] / c[3] + 1) / 2) * W, ((1 - c[1] / c[3]) / 2) * H];
}
const camPos = (cam) => { const m = cam.matrixWorldInverse, t = [m[12], m[13], m[14]]; return [0, 1, 2].map((i) => -(m[i * 4] * t[0] + m[i * 4 + 1] * t[1] + m[i * 4 + 2] * t[2])); };
export function groundRunway(rec) {
  const name = path.basename(rec.narrator_frame || ''), snap = rec.snapshot, cam = rec.cameras && rec.cameras[name];
  const rw = snap && snap.scene && snap.scene.runway, st = snap && snap.frames && snap.frames[name];
  if (!rw || !cam || !st || (rec.facts['scene.in_cloud'] && rec.facts['scene.in_cloud'].v === true)) return null;
  const px = rw.corners.map((c) => project(cam, c));
  if (px.some((x) => !x)) return null;
  const box = clipBox([Math.min(...px.map((x) => x[0])), Math.min(...px.map((x) => x[1])), Math.max(...px.map((x) => x[0])), Math.max(...px.map((x) => x[1]))]);
  const thr = [(rw.corners[0][0] + rw.corners[1][0]) / 2, 0, (rw.corners[0][2] + rw.corners[1][2]) / 2], d = Math.hypot(...sub(thr, camPos(cam)));
  if (!box || Math.exp(-(((st.fog_density || 0) * d) ** 2)) < T_VIS) return null;
  const tp = project(cam, thr);
  if (!tp || tp[0] < 0 || tp[0] > W || tp[1] < 0 || tp[1] > H) return null;
  return { box: normBox(box), pt: [nx(tp[0]), ny(tp[1])] };
}
// adds the v1 grounding facts to a record (build.mjs, after the image facts); a record whose snapshot does not have the
// expected shape gets no grounding facts (and so no grounding texts) rather than stopping the build
export function addGroundFacts(rec, { onError = null } = {}) {
  try {
    if (rec.family === 'S' || rec.family === 'A') {
      const g = groundHazards(rec);
      rec.facts['ground.hazards'] = fact(g.hazards, 'visual'); rec.facts['ground.complete'] = fact(g.complete, 'visual');
      if (g.cv) rec.facts['ground.cv_sim'] = fact(g.cv, 'context');
    }
    if (rec.family === 'L') rec.facts['ground.runway'] = fact(groundRunway(rec), 'visual');
  } catch (e) {
    for (const k of ['ground.hazards', 'ground.complete', 'ground.cv_sim', 'ground.runway']) delete rec.facts[k];
    if (onError) onError(rec, e);
  }
  return rec;
}
