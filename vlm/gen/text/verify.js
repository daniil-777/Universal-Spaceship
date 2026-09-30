// vlm/gen/text/verify.js — the §5.4 verifier. Failures are rejected, never repaired. The work is split over four modules:
//   verify_numbers.js  the one rounding table (nice steps 1, 2, 2.5, 5), units, slot formatting, number patterns, binding
//   verify_words.js    the word tables (W), claim phrases, subjects and the common-word lexicon
//   verify_claims.js   the claim parser (parseClaims) and the gazetteer
//   verify_rules.js    facts, slot constructors, derivations, category checks and support()
// This module judges parsed claims against a record and its row Context, and keeps the public API in one place.
//
// Scope (controller ruling, fix round 3): for template text the verifier is authoritative and exact (every slot holds and the
// parser recovers exactly the slot set). For free text (Narrator, teacher) it is a conservative screen, not a proof. Known
// limitations, with the catch rates of the round-3 probe (t7_probe3/adv3.mjs; template / free text / Narrator path):
//   N3  verdict synonyms ("fine", "risky", "all clear") are not verdict claims (0 / 0 / 0 %).
//   N4  comparatives with an earlier moment or another object ("closer than before", "bigger than the rock") are not
//       claims (0 / 0 / 0 %).
//   N5  superlatives, "only" and "both" ("the only hazard in view", "the largest hazard") are not claims (20 / 20 / 20 %).
//   N6  binding is by clause and nearest subject: a size stated of the wrong hazard across "while", a closing relation of a
//       body (the Moon, the Earth) and a place's distance given to another place still pass (67 / 47 / 73 %).
//   N8  the teacher path sees visual facts only, so vert_mode and wow are absent and "approaching the runway" cannot be
//       checked against the rollout (the L 'family' pseudo-support accepts it).
//   R2  double negation is resolved by parity within a clause (128/128 caught); a negation that reaches across clauses or
//       sentences is not.
import { normContext } from './context.js';
import { SAFETY_TEXT_IDS } from '../schema.js';
import { UNITS, unitOf, inIv } from './verify_numbers.js';
import { CONT_CUE } from './verify_words.js';
import { parseClaims, norm } from './verify_claims.js';
import { factsOf, derive, support } from './verify_rules.js';

export { niceStep, roundNice, UNITS, unitOf, numText, unitText, fmtSlot, rangeText } from './verify_numbers.js';
export { W, NOUNS, ROUTE_NAMES, TERMS, BASE_NAMES, SENTENCE_WORDS } from './verify_words.js';
export { makeGazetteer, parseClaims } from './verify_claims.js';
export { factsOf, derive, countPhrase, catSlot, entSlot, countSlot, clockSlot, regionSlot, compassSlot, rangeSlot, outcomeSlot, prefOutcome,
  outcomeAgrees } from './verify_rules.js';

const MONITOR_TYPES = ['verdict', 'action', 'reason'];
// claims a template answer checks by meaning, not by slot
const SEMANTIC = ['category', 'outcome', 'adjective', 'kind', 'cause', 'disagree', 'presence'];
// claims only a Context line can ground in a flight row
const CONTEXT_TYPES = [...MONITOR_TYPES, 'outcome', 'cause'];
// claims a question states (verdict/action/reason/category/outcome/kind/cause words in a question are not claims)
const PROMPT_TYPES = ['number', 'range', 'count', 'clock', 'bearing_clock', 'region', 'compass', 'entity', 'unknown_entity', 'adjective'];
const sentencesOf = (text) => [...String(text).matchAll(/(?:[^.!?;]|\.(?=\d))+[.!?;]?/g)].map((m) => m[0].trim());
// a claim is the monitor's when its sentence names the monitor, or opens with a continuation cue ("It advises...", "Best
// action:") after one that does
function attribution(text) {
  const out = [];
  for (const s of sentencesOf(text)) out.push(/monitor/i.test(s) || (out.some(Boolean) && CONT_CUE.test(s)));
  return out;
}
const pick = (all, factIds) => (factIds ? Object.fromEntries(Object.entries(all).filter(([id]) => factIds.includes(id))) : all);
// one parsed claim against the record (and the row's Context): true when some fact, the safety block or the monitor holds it
export const checkClaim = (cl, rec, { context = null, attributed = false, factIds = null } = {}) =>
  support(cl, pick(factsOf(rec), factIds), rec, normContext(context), attributed).length > 0;
// obsRule (§5.7): a claim rests on a visual fact, on a fact the Context supplies (telemetry ids; a monitor supplies the
// safety.* ids text cites, except the S/A ground-truth outcome), on the monitor itself, or on the family ('family')
function obsAllowed(id, F, ctx, rec) {
  if (id === 'family') return true;
  if (id === 'monitor') return !!(ctx && ctx.monitor);
  if (id === 'safety') return false;
  if (F[id] && F[id].obs === 'visual') return true;
  if (!ctx) return false;
  if (ctx.ids.has(id)) return true;
  return !!ctx.monitor && SAFETY_TEXT_IDS.includes(id) && !(id === 'safety.action_outcome' && (rec.family === 'S' || rec.family === 'A'));
}
const shown = (cl) => (cl.type === 'range' ? [cl.lo, cl.hi, cl.unit] : cl.type === 'number' ? [cl.value, cl.unit] : cl.value);
const label = (cl) => `${cl.negated ? 'negated ' : ''}${cl.type}${cl.dim ? `/${cl.dim}` : ''} ${JSON.stringify(shown(cl))}`;
function judge(claims, rec, F, ctx, { obsRule, attr }) {
  const errors = [];
  for (const cl of claims) {
    const attributed = !!attr[cl.sent];
    cl.monitor = attributed;
    if (cl.type === 'unknown_entity') { errors.push(`unknown entity ${cl.value}`); continue; }
    if (cl.type === 'adjective') { errors.push(`ungrounded descriptor "${cl.value}"`); continue; }
    if (cl.type === 'disagree') { errors.push(`ungrounded disagreement with the monitor ("${cl.value}")`); continue; }
    const sup = support(cl, F, rec, ctx, attributed);
    // a negated claim holds when nothing supports its positive reading
    if (cl.negated) { if (sup.length) errors.push(`${label(cl)} is false (the positive holds)`); continue; }
    if (!(obsRule ? sup.filter((id) => obsAllowed(id, F, ctx, rec)) : sup).length) errors.push(`${label(cl)} not supported`);
  }
  return errors;
}
// obsRule (Narrator rows, §5.7): a claim must rest on a visual fact, on a fact the row's Context supplies, or on its monitor
export function verifyFreeText(text, rec, { gaz, context = null, factIds = null, obsRule = false } = {}) {
  const ctx = normContext(context), F = pick(factsOf(rec), factIds);
  const claims = parseClaims(text, gaz), errors = judge(claims, rec, F, ctx, { obsRule, attr: attribution(text) });
  if (rec.family !== 'Z' && !ctx) {
    if (claims.some((cl) => CONTEXT_TYPES.includes(cl.type))) errors.push('a flight row without Context states a context-class fact');
    else if (/\bmonitor\b/i.test(text)) errors.push('a flight row without Context mentions the monitor');
  }
  return { verified: errors.length === 0, errors, claims };
}

const deq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
function slotError(s, F) {
  if (MONITOR_TYPES.includes(s.type) || s.type === 'monitor_range' || (s.type === 'outcome' && s.src === 'monitor')) return null;
  const f = F[s.fact_id];
  if (!f || f.v === null || f.v === undefined) return `slot ${s.fact_id} has no fact`;
  const x = derive(s.derive, f.v, F);
  if (s.type === 'number') {
    return typeof x === 'number' && inIv(x, s.lo, s.hi) ? null : `slot ${s.fact_id}: ${x} outside [${s.lo}, ${s.hi}]`;
  }
  if (s.type === 'range') {
    if (typeof x === 'string') return x === s.value ? null : `slot ${s.fact_id}: ${x} is not ${s.value}`;
    const fu = UNITS[unitOf(s.fact_id, f)], y = typeof x === 'number' && fu ? (Math.abs(x) * fu[1]) / UNITS[s.unit][1] : NaN;
    return inIv(y, s.lo, s.hi) ? null : `slot ${s.fact_id}: ${x} outside ${s.text}`;
  }
  if (s.type === 'entity') {
    return typeof x === 'string' && norm(x) === norm(s.value) ? null : `slot ${s.fact_id}: ${JSON.stringify(x)} is not ${s.value}`;
  }
  return deq(x, s.value) ? null : `slot ${s.fact_id}: ${JSON.stringify(x)} is not ${JSON.stringify(s.value)}`;
}
const keyOfClaim = (c) => `${c.type}:${c.type === 'range' ? `${c.lo}-${c.hi === Infinity ? 'inf' : c.hi}` : c.value}`;
function slotKeys(s, gaz) {
  if (s.type === 'number') return [`number:${s.shown ?? +s.text.replace(/[^\d.]/g, '')}`];
  if (s.type === 'range' || s.type === 'monitor_range') return [`range:${s.shownLo}-${s.shownHi === Infinity ? 'inf' : s.shownHi}`];
  if (['count', 'clock', 'bearing_clock', 'region', 'compass'].includes(s.type)) return [`${s.type}:${s.value}`];
  if (s.type === 'entity') return [`entity:${gaz.canonical(String(s.value)) ?? s.value}`];
  const place = s.type === 'category' && typeof s.value === 'string' && gaz.has(s.value) && /^\p{Lu}/u.test(String(s.text));
  if (place) return [`entity:${gaz.canonical(s.value)}`];
  return [];
}
// template text: (a) every slot (and hidden check) holds for the facts; (b) the parser recovers exactly the slot set (category,
// outcome, kind, cause and descriptor words are checked by meaning, not by slot); plus the free-text check of the answer and
// of the prompt (a negative's false premise, matched by `premise`, is exempt)
export function verifyTemplateItem(item, rec, { gaz, context = null, premise = null } = {}) {
  const F = factsOf(rec), errors = [];
  for (const s of [...(item.slots || []), ...(item.checks || [])]) {
    const e = slotError(s, F);
    if (e) errors.push(e);
  }
  const bySlot = (c) => !MONITOR_TYPES.includes(c.type) && !SEMANTIC.includes(c.type);
  const got = parseClaims(item.answer, gaz).filter(bySlot).map(keyOfClaim).sort();
  const want = (item.slots || []).flatMap((s) => slotKeys(s, gaz)).sort();
  const parserOk = deq(got, want);
  if (!parserOk) errors.push(`parser recovered ${got} but the slots are ${want}`);
  for (const e of verifyFreeText(item.answer, rec, { gaz, context }).errors) if (!errors.includes(e)) errors.push(e);
  if (item.prompt) {
    const claims = parseClaims(item.prompt, gaz).filter((c) => PROMPT_TYPES.includes(c.type) && !(premise && premise(c)));
    const ctx = normContext(context);
    for (const e of judge(claims, rec, F, ctx, { obsRule: false, attr: attribution(item.prompt) })) errors.push(`prompt: ${e}`);
  }
  return { verified: errors.length === 0, parserOk, errors };
}
