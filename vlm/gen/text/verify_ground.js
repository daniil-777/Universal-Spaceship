// vlm/gen/text/verify_ground.js — the v1 grounding claims (the research report's slot types) and what supports them:
//   box    [x0,y0,x1,y1]  integers 0-100 on the 896x504 frame, x first: holds for a listed object of the claim's subject
//                         (its kind, or the runway) when IoU >= 0.5 or every edge is within 2 units
//   point  (x,y)          holds inside such an object's box widened by 2 units, or within 2 units of its point; for a Z
//                         feature (a named subject) within 4 units of the feature's pixel position
//   latlon 46° N, 8° E    holds within 1° (latitude and longitude) of the view centre (Z)
// The objects are the build's facts: ground.hazards (S/A), ground.runway (L), ground.features (Z; vlm/gen/build/ground.js
// and zoomgeo.js). Unlike o'clock and regions, these are not in v0 text, so v0 answers never meet them.
const norm = (s) => String(s).normalize('NFC').toLowerCase().replace(/[^\p{L}\p{N} ]+/gu, '').replace(/\s+/g, ' ').trim();

export const GROUND_TYPES = ['box', 'point', 'latlon'];
export const GROUND_RES = [
  ['box', /\[\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\]/g, (m) => [+m[1], +m[2], +m[3], +m[4]]],
  ['point', /\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)/g, (m) => [+m[1], +m[2]]],
  ['latlon', /(?<![\d.])(\d{1,2}(?:\.\d+)?)\s*°\s*([NS])\b\s*,?\s*(?:and\s+)?(\d{1,3}(?:\.\d+)?)\s*°\s*([EW])\b/g, (m) => [(m[2] === 'S' ? -1 : 1) * +m[1], (m[4] === 'W' ? -1 : 1) * +m[3]]]];
const area = (b) => Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
export const iou = (a, b) => { const i = area([Math.max(a[0], b[0]), Math.max(a[1], b[1]), Math.min(a[2], b[2]), Math.min(a[3], b[3])]), u = area(a) + area(b) - i; return u > 0 ? i / u : 0; };
export const boxHolds = (a, b) => iou(a, b) >= 0.5 || a.every((x, i) => Math.abs(x - b[i]) <= 2);
export const pointHolds = (p, box, pt, tol = 2) => (!!box && p[0] >= box[0] - tol && p[0] <= box[2] + tol && p[1] >= box[1] - tol && p[1] <= box[3] + tol)
  || (!!pt && Math.abs(p[0] - pt[0]) <= tol && Math.abs(p[1] - pt[1]) <= tol);
const val = (F, id) => (F[id] && F[id].v !== null && F[id].v !== undefined ? F[id].v : null);
const wrap = (d) => ((d + 540) % 360) - 180;
// the objects a claim may be about: [fact id, {box?, pt}] of the claim's subject
function objectsOf(F, subj) {
  const out = [], hz = val(F, 'ground.hazards'), rw = val(F, 'ground.runway'), ft = val(F, 'ground.features');
  const k = subj && subj.kind === 'hazard' ? subj.value : null, named = subj && subj.kind === 'entity' ? norm(subj.value) : null;
  if (Array.isArray(hz) && (!subj || subj.kind === 'hazard')) for (const h of hz) if (!k || h.kind === k) out.push(['ground.hazards', h, 2]);
  if (rw && (!subj || (subj.kind === 'object' && subj.value === 'runway'))) out.push(['ground.runway', rw, 2]);
  if (Array.isArray(ft) && (!subj || named)) for (const f of ft) if (!named || norm(f.name) === named) out.push(['ground.features', f, 4]);
  return out;
}
export function groundSupport(cl, F) {
  if (cl.type === 'latlon') {
    const lat = val(F, 'view.lat_deg'), lon = val(F, 'view.lon_deg');
    return typeof lat === 'number' && typeof lon === 'number' && Math.abs(lat - cl.value[0]) <= 1 && Math.abs(wrap(lon - cl.value[1])) <= 1 ? ['view.lat_deg', 'view.lon_deg'] : [];
  }
  const hits = objectsOf(F, cl.subject).filter(([, o, tol]) => (cl.type === 'box' ? !!o.box && boxHolds(cl.value, o.box) : pointHolds(cl.value, o.box, o.pt, tol)));
  return [...new Set(hits.map(([id]) => id))];
}
// template slots: the value a slot states is checked against its fact (slotError) and keyed like the parsed claim
export const boxText = (b) => `[${b.join(',')}]`, pointText = (p) => `(${p.join(',')})`;
export const latlonText = ([lat, lon]) => `${Math.abs(lat)}° ${lat < 0 ? 'S' : 'N'}, ${Math.abs(lon)}° ${lon < 0 ? 'W' : 'E'}`;
export const groundSlot = (type, id, value, text, derive = null) => ({ text, fact_id: id, value, unit: null, lo: null, hi: null, type, ...(derive ? { derive } : {}) });
export function groundSlotError(s, F) {
  if (s.type === 'latlon') return groundSupport({ type: 'latlon', value: s.value }, F).length ? null : `slot latlon ${JSON.stringify(s.value)} is not the view centre`;
  return null;
}
