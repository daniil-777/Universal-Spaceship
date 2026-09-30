// vlm/gen/schema.js — the enums and the raw-record contract of the Astro Pilot Vision dataset (spec §4.1-§4.3).
// Browser-loadable (no imports). validateRecord(rec) returns { ok, errors } and never throws.
export const FAMILIES = Object.freeze(['S', 'A', 'L', 'D', 'Z']);
export const FLIGHT_FAMILIES = Object.freeze(['S', 'A', 'L', 'D']);
export const VERDICTS = Object.freeze(['SAFE', 'CAUTION', 'UNSAFE']);
export const SEVERITY = Object.freeze(['SAFE', 'CAUTION', 'DANGER_ESCAPABLE', 'DANGER', 'CRITICAL']);
export const REASONS = Object.freeze([
  'HAZARD_AHEAD', 'HAZARD_CLOSING_FAST', 'TERRAIN_CLOSE', 'BUILDING_CLOSE', 'CORRIDOR_EDGE',
  'STALL', 'OVERSTRESS', 'SEVERE_TURBULENCE', 'STORM_CELL', 'PULL_UP',
  'UNSTABLE_APPROACH', 'LOCALIZER_DEVIATION', 'GLIDESLOPE_DEVIATION', 'SPEED_OUT_OF_BAND', 'HIGH_SINK_RATE', 'STRONG_CROSSWIND', 'TAILWIND', 'RUNWAY_EDGE', 'CANNOT_STOP',
  'KOS_VIOLATION', 'CLOSING_TOO_FAST', 'LATERAL_MISALIGNMENT', 'ATTITUDE_ERROR', 'JET_FAILURE', 'LOW_FUEL', 'NO_BREAKOUT_AVAILABLE']);
export const ACTIONS = Object.freeze(['CONTINUE', 'CLIMB', 'DESCEND', 'TURN_LEFT', 'TURN_RIGHT', 'SPEED_UP', 'SLOW_DOWN', 'GO_AROUND', 'HOLD_POSITION', 'BREAKOUT', 'NONE_SAFE']);
const CORRIDOR = ['CONTINUE', 'CLIMB', 'DESCEND', 'TURN_LEFT', 'TURN_RIGHT', 'SPEED_UP', 'SLOW_DOWN'];
export const FAMILY_ACTIONS = Object.freeze({ S: CORRIDOR, A: CORRIDOR, L: ['CONTINUE', 'GO_AROUND', 'TURN_LEFT', 'TURN_RIGHT', 'CLIMB', 'DESCEND', 'SPEED_UP', 'SLOW_DOWN'], D: ['CONTINUE', 'BREAKOUT', 'HOLD_POSITION', 'SLOW_DOWN'], Z: [] });
export const OUTCOMES = Object.freeze({ S: ['clear', 'crash_possible', 'crash_certain'], A: ['clear', 'crash_possible', 'crash_certain'],
  L: ['landed', 'go_around', 'hard', 'excursion', 'overrun', 'short', 'tailstrike', 'crash'], D: ['capture', 'breakout', 'fail'] });
export const HAZARD_KINDS = Object.freeze(['rock', 'comet', 'satellite', 'airliner', 'birds']);
export const CAUSES = Object.freeze([...HAZARD_KINDS, 'terrain', 'building', 'roof', 'overstress', 'ground', 'station', 'runway']);
export const ZOOM_TAGS = Object.freeze(['WATER_DOMINANT', 'COASTLINE', 'MOUNTAINS', 'HILLS', 'FLAT', 'HIGH_TERRAIN', 'URBAN', 'DESERT', 'ICE', 'NIGHT']);
export const RANGE_EDGES_KM = Object.freeze([20, 100, 400, 1500]);
export const OBS = Object.freeze(['visual', 'context']);
export const POLICY_IDS = Object.freeze(['belt_ppo', 'atmo_ppo', 'search_v1', 'autoland', 'gnc']);
export const CAPTURE_MODES = Object.freeze(['clock', 'freeze']);
export const LAYERS = Object.freeze({
  gibs: { source: 'NASA GIBS Blue Marble', year: 2004, licence: 'public domain' },
  s2cloudless_3857: { source: 'EOX Sentinel-2 cloudless', year: 2016, licence: 'CC BY 4.0' },
  's2cloudless-2025_3857': { source: 'EOX Sentinel-2 cloudless', year: 2025, licence: 'CC BY-NC-SA 4.0' },
  terrarium: { source: 'AWS Terrain Tiles', year: null, licence: 'open data (see ATTRIBUTION.txt)' },
});
export const PROFILE_LAYERS = Object.freeze({ open: ['gibs', 's2cloudless_3857', 'terrarium'], nc: ['gibs', 's2cloudless-2025_3857', 'terrarium'] });
export const NOMINAL_FRAME_DT = Object.freeze({ S: { steps: 3, s: 0.2 }, A: { steps: 3, s: 0.2 }, L: { steps: 30, s: 0.25 }, D: { steps: 20, s: 2 } });
export const FRAME_DT_STEPS_RANGE = Object.freeze({ S: [2, 4], A: [2, 4], L: [22, 38], D: [19, 21] });
export const STEP_S = Object.freeze({ S: 1 / 15, A: 1 / 15, L: 1 / 120, D: 0.1 });

// ---- the text contract (controller ruling, T7 fix round 1): one definition for the text layer (T7), the build (T11: zoomgeo.js,
// imagefacts.js, export.js) and the slot evaluation (T15). Image facts are computed from narrator_frame at build.
export const PALETTE_NAMES = Object.freeze(['black', 'grey', 'white', 'red', 'orange', 'yellow', 'green', 'teal', 'blue', 'purple', 'pink', 'brown']);
// mean luminance of narrator_frame in [0, 1]: dark < 0.25 <= medium < 0.6 <= bright
export const BRIGHTNESS_BINS = Object.freeze(['dark', 'medium', 'bright']);
export const BRIGHTNESS_EDGES = Object.freeze([0.25, 0.6]);
export const EDGE_BINS = Object.freeze(['smooth', 'textured', 'highly textured']);
export const COAST_SIDES = Object.freeze(['left', 'right', 'top', 'bottom']);
// zoom tag -> the phrases text states it with; the first is the caption phrase, and slot evaluation credits any of them
export const TAG_WORDS = Object.freeze({ WATER_DOMINANT: ['mostly open sea', 'open sea'], COASTLINE: ['a coastline', 'coastline'], MOUNTAINS: ['mountains', 'mountainous'],
  HILLS: ['hills', 'hilly'], FLAT: ['flat land', 'mostly flat'], HIGH_TERRAIN: ['high ground'], URBAN: ['city areas', 'city'], DESERT: ['desert'], ICE: ['ice'], NIGHT: ['at night', 'night'] });
// the build-time facts the text layer reads, all obs visual and required for their families. Shapes: place = {name, km,
// bearing (deg from north), compass (one of the 8 compass words)} | null; features = [{name, kind, region (3x3 phrase)}];
// frac = number in [0, 1]; palette / brightness / edge / coast = one of PALETTE_NAMES / BRIGHTNESS_BINS / EDGE_BINS / COAST_SIDES
// (coast may be null); place.admin1 is the admin-1 name at the view centre (T11 writes it; text asks "which region is this");
// clouds = the landing scene's METAR group: an optional light-rain prefix "-RA ", then NSC / SKC or FEW|SCT|BKN|OVC with a
// three-digit base in hundreds of feet ("SCT030", "-RA BKN012"), or null
const ALL = FAMILIES;
export const TEXT_FACTS = Object.freeze({ 'place.nearest': { families: ['Z'], shape: 'place' }, 'place.country': { families: ['Z'], shape: 'name' }, 'place.admin1': { families: ['Z'], shape: 'name' },
  'place.in_view': { families: ['Z'], shape: 'features' }, 'geo.sea_frac': { families: ['Z'], shape: 'frac' }, 'geo.coast_side': { families: ['Z'], shape: 'coast' },
  'image.palette_0': { families: ALL, shape: 'palette' }, 'image.palette_1': { families: ALL, shape: 'palette' }, 'image.palette_2': { families: ALL, shape: 'palette' },
  'image.brightness_bin': { families: ALL, shape: 'brightness' }, 'image.edge_bin': { families: ALL, shape: 'edge' },
  'scene.clouds': { families: ['L'], shape: 'clouds' }, 'scene.in_cloud': { families: ['L'], shape: 'bool' } });
// safety.* ids text items cite; a Context line with a monitor supplies them (§5.7, controller ruling)
export const SAFETY_TEXT_IDS = Object.freeze(['safety.verdict', 'safety.reasons', 'safety.best_action', 'safety.action_outcome', 'safety.cause', 'safety.p_ref', 'safety.ttc_s', 'safety.clearance']);
const COMPASS8 = ['north', 'north-east', 'east', 'south-east', 'south', 'south-west', 'west', 'north-west'];
const REGION9 = ['upper left', 'upper centre', 'upper right', 'middle left', 'centre', 'middle right', 'lower left', 'lower centre', 'lower right'];
const SHAPE = { place: (v) => v === null || (v && typeof v.name === 'string' && v.km >= 0 && Number.isFinite(v.bearing) && COMPASS8.includes(v.compass)),
  name: (v) => v === null || (typeof v === 'string' && v.length > 0), features: (v) => Array.isArray(v) && v.every((x) => x && typeof x.name === 'string' && typeof x.kind === 'string' && REGION9.includes(x.region)),
  frac: (v) => typeof v === 'number' && v >= 0 && v <= 1, coast: (v) => v === null || COAST_SIDES.includes(v), palette: (v) => PALETTE_NAMES.includes(v),
  brightness: (v) => BRIGHTNESS_BINS.includes(v), edge: (v) => EDGE_BINS.includes(v),
  clouds: (v) => v === null || (typeof v === 'string' && /^(?:-RA )?(?:NSC|SKC|(?:FEW|SCT|BKN|OVC)\d{3})$/.test(v)),
  bool: (v) => typeof v === 'boolean' };
export function validateTextFacts(rec) {
  const e = [], facts = (rec && rec.facts) || {};
  for (const [id, spec] of Object.entries(TEXT_FACTS)) {
    if (!spec.families.includes(rec.family)) continue;
    const f = facts[id];
    if (!f) { e.push(`${id} missing`); continue; }
    if (f.obs !== 'visual') e.push(`${id} obs ${f.obs} (visual expected)`);
    if (!SHAPE[spec.shape](f.v)) e.push(`${id} = ${JSON.stringify(f.v)} is not a ${spec.shape}`);
  }
  if (rec.family === 'Z' && !(rec.zoom && Array.isArray(rec.zoom.tags) && rec.zoom.tags.every((t) => ZOOM_TAGS.includes(t)))) e.push('zoom.tags missing or not ZOOM_TAGS');
  return { ok: e.length === 0, errors: e };
}

export const fact = (v, unit, obs) => ({ v, unit, obs });
export const recordKey = (family, run, episode, step) => `${family}_${run}_${String(episode).padStart(5, '0')}_${String(step).padStart(6, '0')}`;
export const rangeBin = (km) => RANGE_EDGES_KM.filter((e) => km >= e).length;
// a sample the capture drops instead of labelling (e.g. a branch that ends in timeout): the label modules throw it and the
// capture catches it by `e.discard === true`; any other error is a bug and propagates
export class Discard extends Error { constructor(why) { super(`discard: ${why}`); this.name = 'Discard'; this.discard = true; } }

function numbers(x, at, errors) {
  if (typeof x === 'number') { if (!Number.isFinite(x)) errors.push(`non-finite number at ${at}`); return; }
  if (Array.isArray(x)) x.forEach((v, i) => numbers(v, `${at}[${i}]`, errors));
  else if (x && typeof x === 'object') for (const [k, v] of Object.entries(x)) numbers(v, `${at}.${k}`, errors);
}
function safetyErrors(s, fam, at, e) {
  if (!s || typeof s !== 'object') { e.push(`${at} missing`); return; }
  if (!VERDICTS.includes(s.verdict)) e.push(`${at}.verdict ${s.verdict}`);
  if (!(s.severity >= 0 && s.severity <= 4)) e.push(`${at}.severity ${s.severity}`);
  for (const r of s.reasons || []) if (!REASONS.includes(r)) e.push(`${at} reason ${r}`);
  for (const a of s.safe_actions || []) if (!FAMILY_ACTIONS[fam].includes(a)) e.push(`${at}.safe_actions has ${a}`);
  if (!ACTIONS.includes(s.best_action)) e.push(`${at}.best_action ${s.best_action}`);
  for (const [a, o] of Object.entries(s.action_outcome || {})) if (!OUTCOMES[fam].includes(o)) e.push(`${at}.action_outcome.${a} ${o} is not in OUTCOMES.${fam}`);
}

export function validateRecord(rec) {
  const e = [];
  if (!rec || typeof rec !== 'object') return { ok: false, errors: ['not an object'] };
  const fam = rec.family, r = rec.render || {}, pv = rec.provenance || {}, prof = r.licence_profile;
  if (!FAMILIES.includes(fam)) return { ok: false, errors: [`family ${fam}`] };
  numbers(rec, 'record', e);
  const facts = rec.facts || {};
  if (!Object.keys(facts).length) e.push('no facts');
  for (const [id, f] of Object.entries(facts)) {
    if (!f || typeof f !== 'object' || !('v' in f)) e.push(`fact ${id} is not {v, unit, obs}`);
    else if (!OBS.includes(f.obs)) e.push(`fact ${id} has no obs`);
  }
  const want = fam === 'Z' ? 1 : 3;
  if (!Array.isArray(rec.frames) || rec.frames.length !== want) e.push(`frames.length must be ${want}`);
  if (fam === 'D' && !(typeof rec.narrator_frame === 'string' && rec.narrator_frame.endsWith('.chase.png'))) e.push('D record without chase.png');
  if (r.renderScale !== 1 || r.dpr !== 1) e.push('renderScale and dpr must be 1');
  if (!CAPTURE_MODES.includes(r.capture_mode)) e.push(`capture_mode ${r.capture_mode}`);
  if (!PROFILE_LAYERS[prof]) e.push(`licence_profile ${prof}`);
  for (const [i, im] of (r.imagery || []).entries()) {
    if (!im || !im.licence || !im.layer_id) { e.push(`imagery[${i}] missing licence or layer_id`); continue; }
    if (/arcgis|esri/i.test(`${im.source} ${im.layer_id}`)) e.push(`imagery[${i}] is an Esri source`);
    if (prof === 'open' && /s2cloudless-20(1[89]|2[0-5])/.test(im.layer_id)) e.push(`imagery[${i}] EOX 2018-2025 in the open profile`);
    else if (PROFILE_LAYERS[prof] && !PROFILE_LAYERS[prof].includes(im.layer_id)) e.push(`imagery[${i}] layer ${im.layer_id} not allowed in ${prof}`);
  }
  if (fam === 'L') {
    if (!r.view || r.view.eye !== 'chase') e.push('L eye view must be chase');
    if (r.path !== false) e.push('L path must be false');
    if (!(pv.seed >= 1)) e.push('L seed must be >= 1');
  }
  if (fam === 'A' && !(pv.atmosphere >= 0.995)) e.push('A atmosphere < 0.995');
  if (fam === 'Z') {
    if (rec.safety !== null || rec.safety_eye !== null) e.push('Z safety and safety_eye must be null');
    if (!rec.zoom || !(rec.zoom.range_bin >= 0 && rec.zoom.range_bin <= 4) || !rec.zoom.lighting) e.push('Z zoom block missing');
    return { ok: e.length === 0, errors: e };
  }
  safetyErrors(rec.safety, fam, 'safety', e);
  safetyErrors(rec.safety_eye, fam, 'safety_eye', e);
  const dt = rec.frame_dt_steps, [lo, hi] = FRAME_DT_STEPS_RANGE[fam];
  if (!Array.isArray(dt) || dt.length !== 2 || dt.some((n) => !Number.isInteger(n) || n < lo || n > hi)) e.push(`frame_dt_steps outside [${lo}, ${hi}]`);
  const inj = pv.injection;
  if (inj && inj.step !== null && inj.step !== undefined && Array.isArray(dt)) {
    const lead = dt[0] + dt[1] + 1;
    if (pv.step - inj.step < lead) e.push(`runtime injection ${pv.step - inj.step} steps before f2 (< ${lead})`);
  }
  return { ok: e.length === 0, errors: e };
}
