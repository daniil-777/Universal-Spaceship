// vlm/gen/text/paraphrase.js — the static paraphrase bank (spec §5.3): slot protection, 20 % held-out paraphrase ids for
// the test split, rendering that refuses undefined, NaN and empty slots. checkBank also lints what the rules forbid
// outside slots: digits, proper nouns, claim words the verifier would read (numbers, counts, o'clock, regions, compass,
// verdict/action/reason words in anything but a question) and "about" in front of a number slot (fmtSlot says it).
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseClaims, makeGazetteer } from './verify.js';

export const KINDS = Object.freeze(['caption_short', 'caption_detail', 'caption_part', 'vqa_q', 'vqa_a', 'safety', 'safety_part', 'negative_q', 'negative_a']);
const FAMS = ['S', 'A', 'L', 'D', 'Z'];
// slots whose text starts with its own "about" (fmtSlot) or bin wording ("between ... and ...", "under ...", "over ...")
export const NUMBER_SLOTS = new Set(['dist', 'dist_bin', 'clr_bin', 'ceiling', 'rate', 'ttc', 'cpa', 'speed', 'pitch', 'cell_dist', 'stall_margin', 'g', 'gs_dev', 'loc_dev', 'head', 'cross', 'sock',
  'station_dist', 'closing', 'limit', 'att', 'fuel', 'place_km', 'range', 'range_bin', 'sea', 'gsd', 'dist_stated', 'dist_true', 'speed_stated', 'speed_true', 'station_stated', 'range_stated', 'range_true']);
const QUESTION = new Set(['vqa_q', 'negative_q']);
// place slots never open a sentence: a place named like a common word (Split, Nice, Orange) is not read as a place there
export const ENTITY_SLOTS = new Set(['place', 'country', 'region_name', 'feature', 'route', 'wrong', 'wrong_route']);
// slots that hold whole sentences (the safety chain's perception and prediction): the word after them opens a sentence
const SENTENCE_SLOTS = new Set(['perception', 'prediction']);
const slotBefore = (f, i) => { const w = f.split(/\s+/).filter(Boolean)[i - 1] || ''; const m = /^\{([a-z_]+)\}$/.exec(w); return m ? m[1] : null; };
// capitalised words a form may use mid-sentence
const PROPER_OK = new Set(['I', 'Earth', 'Moon', 'Sun', 'PAPI', 'ILS']);
const EMPTY_GAZ = makeGazetteer([]);
export const slotsOf = (form) => [...new Set([...form.matchAll(/\{([a-z_]+)\}/g)].map((m) => m[1]))].sort();
const fnv = (s) => { let h = 0x811c9dc5; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; } return h; };
export const heldOut = (templateId, i) => fnv(`${templateId}#${i}`) % 5 === 0;
function lintForm(id, t, f) {
  const e = [], bare = f.replace(/\{[a-z_]+\}/g, 'Xslot');
  if (/\d/.test(f.replace(/\{[a-z_]+\}/g, ''))) e.push(`${id}: "${f}" has a digit outside a slot`);
  for (const m of f.matchAll(/(?:^|[.!?:;]\s+)\{([a-z_]+)\}/g)) if (ENTITY_SLOTS.has(m[1])) e.push(`${id}: "${f}" opens a sentence with the place slot {${m[1]}}`);
  for (const m of f.matchAll(/\b(about|roughly|around|approximately|nearly|almost|some|over|under|between)\s+\{([a-z_]+)\}/gi)) if (NUMBER_SLOTS.has(m[2])) e.push(`${id}: "${f}" puts "${m[1]}" before the number slot {${m[2]}}`);
  const words = bare.split(/\s+/).filter(Boolean);
  words.forEach((w, i) => {
    const c = w.replace(/^[^\p{L}]+|[^\p{L}'-]+$/gu, '').replace(/'s$/, '');
    if (i > 0 && /^\p{Lu}/u.test(c) && c !== 'Xslot' && !PROPER_OK.has(c) && !/^[\p{Lu}-]+$/u.test(c) && !/[.!?:;]["')]*$/.test(words[i - 1]) && !SENTENCE_SLOTS.has(slotBefore(f, i))) e.push(`${id}: "${f}" has the proper noun "${c}" outside a slot`);
  });
  const claims = parseClaims(bare, EMPTY_GAZ).filter((c) => c.type !== 'unknown_entity' && !(QUESTION.has(t.kind) && ['verdict', 'action', 'reason'].includes(c.type)));
  for (const c of claims) e.push(`${id}: "${f}" states a ${c.type} (${JSON.stringify(c.value ?? [c.lo, c.hi])}) outside a slot`);
  return e;
}
export function checkBank(bank) {
  const e = [];
  for (const [id, t] of Object.entries(bank)) {
    if (!KINDS.includes(t.kind)) e.push(`${id}: kind ${t.kind}`);
    if (!Array.isArray(t.families) || !t.families.length || t.families.some((x) => !FAMS.includes(x))) e.push(`${id}: families ${t.families}`);
    if (!Array.isArray(t.forms) || t.forms.length < 10 || t.forms.length > 20) { e.push(`${id}: ${t.forms && t.forms.length} forms (need 10-20)`); continue; }
    if (new Set(t.forms).size !== t.forms.length) e.push(`${id}: duplicate forms`);
    for (const f of t.forms) if (JSON.stringify(slotsOf(f)) !== JSON.stringify([...t.slots].sort())) e.push(`${id}: "${f}" changes the slot set`);
    for (const f of t.forms) e.push(...lintForm(id, t, f));
  }
  return e;
}
export function render(form, slots) {
  return form.replace(/\{([a-z_]+)\}/g, (_, k) => { const v = slots[k]; if (v === undefined || v === null || v === '' || (typeof v === 'number' && !Number.isFinite(v)) || /undefined|NaN/.test(String(v))) throw new Error(`slot ${k} is empty in "${form}"`); return String(v); });
}
// sentence-initial capitals and single spaces (slot values are lower case wherever they may open a sentence)
export const polish = (s) => s.replace(/\s+/g, ' ').replace(/\s+([,.;:!?])/g, '$1').trim().replace(/(^|[.!?]\s+)(\p{Ll})/gu, (m, a, b) => a + b.toUpperCase());
// the test split (and OOD) draws only held-out paraphrase ids; train and val never do
export function pickForm(bank, templateId, rng, split) {
  const t = bank[templateId], test = split === 'test' || split === 'ood', ids = t.forms.map((_, i) => i).filter((i) => (test ? heldOut(templateId, i) : !heldOut(templateId, i))), pool = ids.length ? ids : t.forms.map((_, i) => i);
  const i = pool[Math.floor(rng() * pool.length)]; return { form: t.forms[i], paraphrase_id: i };
}
export function loadBank(dir) { const b = {}; for (const f of fs.readdirSync(dir).filter((n) => n.endsWith('.json')).sort()) Object.assign(b, JSON.parse(fs.readFileSync(path.join(dir, f), 'utf8'))); return b; }
let DEFAULT = null;
export const defaultBank = () => DEFAULT || (DEFAULT = loadBank(fileURLToPath(new URL('./bank/', import.meta.url))));
