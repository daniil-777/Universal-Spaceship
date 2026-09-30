// vlm/web/templates.js — the 0 ms template sentence from Pilot Eye's heads (spec approach B): enum words only, no free text.
// Flight families: "<VERDICT>[: <reason> and <reason>]. Advice: <action>." (REASON_TEXT / ACTION_TEXT, the Context words);
// Z: "A view from <range bin>[ showing <tag>, <tag>]." with each tag's caption phrase (TAG_WORDS[tag][0], ruling T7-b).
import { REASON_TEXT, ACTION_TEXT } from '../gen/text/context.js';
import { TAG_WORDS, RANGE_EDGES_KM } from '../gen/schema.js';
const E = RANGE_EDGES_KM;
export const RANGE_TEXT = Object.freeze([`under ${E[0]} km`, ...E.slice(1).map((b, i) => `${E[i]} to ${b} km`), `over ${E[E.length - 1]} km`]);
export function templateSentence(h, family) {
  if (family === 'Z') {
    const tags = (h.tags || []).map((t) => (TAG_WORDS[t] || [])[0]).filter(Boolean);
    return `A view from ${RANGE_TEXT[h.range_bin] || 'orbit'}${tags.length ? ` showing ${tags.join(', ')}` : ''}.`;
  }
  const r = (h.reasons || []).map((x) => REASON_TEXT[x]).filter(Boolean);
  return `${h.verdict}${r.length ? `: ${r.join(' and ')}` : ''}. Advice: ${ACTION_TEXT[h.action] || 'continue'}.`;
}
