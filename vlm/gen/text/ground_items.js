// vlm/gen/text/ground_items.js — the v1 grounding text items (task 'grounding'; ruling V1-2 and the controller's label
// priorities #1, #3, #8, #9): every claim is visual, from the build's ground.* facts (vlm/gen/build/ground.js, zoomgeo.js)
// and the Z view facts, and every item is verified like any template item (items.js keep() -> verifyTemplateItem, with the
// box/point/latlon slot types of verify_ground.js).
//   S/A  ground_box    the nearest visible hazard's box          ground_point  its point
//        ground_count  counting by pointing: every visible hazard as a point, only when the list is complete and no hazard's
//                      box is more than half covered by a nearer one (occlusion is not modelled)
//        ground_detect detection: every visible hazard as its kind and box, under the same condition (2-6 hazards)
//        ground_refer  a point in the question, the hazard kind there in the answer (a hazard no other box covers)
//   L    runway_box / runway_point   the runway outline in view and its near threshold (chase camera, fog-gated)
//   Z    feature_point  a Natural Earth feature's pixel position   geo  the view centre as latitude/longitude (whole
//        degrees, the verifier allows 1°) and the country at the centre
import { pickForm, render, polish } from './paraphrase.js';
import { factsOf, catSlot, entSlot, countSlot } from './verify.js';
import { groundSlot, boxText, pointText, latlonText } from './verify_ground.js';
import { KIND_A, KIND_N } from './vqa.js';

// the answer templates groundItems() produces (tests/vlm_text.test.mjs counts them as produced)
export const GROUND_ANSWERS = Object.freeze(['ground_box_a', 'ground_point_a', 'ground_count_a', 'ground_detect_a', 'ground_refer_a', 'runway_box_a', 'runway_point_a', 'feature_point_a', 'geo_a', 'geo_sea_a']);
const v = (F, id) => (F[id] && F[id].v !== null && F[id].v !== undefined ? F[id].v : null);
const list = (xs) => (xs.length < 2 ? xs.join('') : `${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}`);
const inside = (p, b) => p[0] >= b[0] && p[0] <= b[2] && p[1] >= b[1] && p[1] <= b[3];
// one question/answer pair: {q: {slot: text}, a: {slot: slot object or text}, slots: the checked slot objects}
function item(bank, rng, split, id, q, a, slots, qSlots = []) {
  const qf = pickForm(bank, `${id}_q`, rng, split), af = pickForm(bank, a.aid || `${id}_a`, rng, split);
  const text = Object.fromEntries(Object.entries(a.text).map(([k, s]) => [k, typeof s === 'string' ? s : s.text]));
  return { task: 'grounding', prompt: polish(render(qf.form, q)), answer: polish(render(af.form, text)), slots, checks: qSlots, template_id: `${id}_q+${a.aid || `${id}_a`}`,
    paraphrase_id: `${qf.paraphrase_id}+${af.paraphrase_id}`, family_q: id, generator: 'template', false_premise: null };
}
export function groundItems(rec, { bank, rng, split = 'train' }) {
  const F = factsOf(rec), out = [], has = (id) => bank[`${id}_q`] && (bank[`${id}_a`] || id === 'geo');
  if (rec.family === 'S' || rec.family === 'A') {
    const hz = v(F, 'ground.hazards') || [], near = hz[0], complete = v(F, 'ground.complete') === true;
    if (near && KIND_A[near.kind] && has('ground_box')) {
      const box = groundSlot('box', 'ground.hazards', near.box, boxText(near.box), ['idx', 0, 'box']), kind = catSlot('ground.hazards', near.kind, KIND_A[near.kind], ['idx', 0, 'kind']);
      out.push(item(bank, rng, split, 'ground_box', { kind: KIND_N[near.kind] }, { text: { a_kind: kind, box } }, [kind, box]));
    }
    if (near && KIND_N[near.kind] && has('ground_point')) {
      const pt = groundSlot('point', 'ground.hazards', near.pt, pointText(near.pt), ['idx', 0, 'pt']);
      out.push(item(bank, rng, split, 'ground_point', { kind: KIND_N[near.kind] }, { text: { pt } }, [pt]));
    }
    if (complete && hz.length >= 1 && hz.length <= 8 && hz.every((h) => h.alone) && has('ground_count')) {
      const n = countSlot('ground.hazards', hz.length, 'hazard', ['len']), pts = hz.map((h, i) => groundSlot('point', 'ground.hazards', h.pt, pointText(h.pt), ['idx', i, 'pt']));
      out.push(item(bank, rng, split, 'ground_count', {}, { text: { count: n, points: list(pts.map((p) => p.text)) } }, [n, ...pts]));
    }
    // detection: every visible hazard as its kind and box (a complete list of 2-6, none more than half covered)
    if (complete && hz.length >= 2 && hz.length <= 6 && hz.every((h) => h.alone && KIND_A[h.kind]) && has('ground_detect')) {
      const kinds = hz.map((h, i) => catSlot('ground.hazards', h.kind, KIND_A[h.kind], ['idx', i, 'kind'])), boxes = hz.map((h, i) => groundSlot('box', 'ground.hazards', h.box, boxText(h.box), ['idx', i, 'box']));
      out.push(item(bank, rng, split, 'ground_detect', {}, { text: { detections: list(hz.map((_, i) => `${kinds[i].text} at ${boxes[i].text}`)) } }, [...kinds, ...boxes]));
    }
    // a hazard whose point no other listed box covers, so the kind at that point is unambiguous
    const clear = hz.map((h, i) => [h, i]).filter(([h, i]) => KIND_A[h.kind] && hz.every((o, j) => j === i || !inside(h.pt, [o.box[0] - 2, o.box[1] - 2, o.box[2] + 2, o.box[3] + 2])));
    if (clear.length && has('ground_refer')) {
      const [h, i] = clear[Math.floor(rng() * clear.length)], kind = catSlot('ground.hazards', h.kind, KIND_A[h.kind], ['idx', i, 'kind']);
      out.push(item(bank, rng, split, 'ground_refer', { pt: pointText(h.pt) }, { text: { a_kind: kind } }, [kind], [groundSlot('point', 'ground.hazards', h.pt, pointText(h.pt), ['idx', i, 'pt'])]));
    }
  }
  if (rec.family === 'L') {
    const rw = v(F, 'ground.runway');
    if (rw && has('runway_box')) { const box = groundSlot('box', 'ground.runway', rw.box, boxText(rw.box), ['field', 'box']); out.push(item(bank, rng, split, 'runway_box', {}, { text: { box } }, [box])); }
    if (rw && has('runway_point')) { const pt = groundSlot('point', 'ground.runway', rw.pt, pointText(rw.pt), ['field', 'pt']); out.push(item(bank, rng, split, 'runway_point', {}, { text: { pt } }, [pt])); }
  }
  if (rec.family === 'Z') {
    const ft = (v(F, 'ground.features') || []).map((f, i) => [f, i]).filter(([f]) => f.name && Array.isArray(f.pt) && (v(F, 'ground.features') || []).filter((g) => g.name === f.name).length === 1);
    if (ft.length && has('feature_point')) {
      const [f, i] = ft[Math.floor(rng() * ft.length)], name = entSlot('ground.features', f.name, ['idx', i, 'name']), pt = groundSlot('point', 'ground.features', f.pt, pointText(f.pt), ['idx', i, 'pt']);
      out.push(item(bank, rng, split, 'feature_point', { feature: f.name }, { text: { feature: name, pt } }, [name, pt]));
    }
    const lat = v(F, 'view.lat_deg'), lon = v(F, 'view.lon_deg'), c = v(F, 'place.country');
    if (typeof lat === 'number' && typeof lon === 'number' && has('geo')) {
      const ll = [Math.round(lat), ((Math.round(lon) + 540) % 360) - 180], s = groundSlot('latlon', 'view.lat_deg', ll, latlonText(ll));
      out.push(typeof c === 'string' && bank.geo_a ? item(bank, rng, split, 'geo', {}, { text: { latlon: s, country: entSlot('place.country', c) } }, [s, entSlot('place.country', c)])
        : item(bank, rng, split, 'geo', {}, { aid: 'geo_sea_a', text: { latlon: s } }, [s]));
    }
  }
  return out;
}
