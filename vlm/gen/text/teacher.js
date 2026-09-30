// vlm/gen/text/teacher.js — the optional, gated teacher (spec §5.6): an injected send() transport, the cost printed first,
// no call without --yes-spend and ANTHROPIC_API_KEY, output through the same verifier, the key never logged or stored.
// The teacher sees the visual facts only, and its caption is verified as a Narrator row without Context (obsRule): a
// teacher caption may state nothing a single frame cannot show. This module makes no network call itself.
import crypto from 'node:crypto';
import { verifyFreeText } from './verify.js';

export const estimateCost = ({ n, tokensIn, tokensOut, priceIn, priceOut }) => (n * (tokensIn * priceIn + tokensOut * priceOut)) / 1e6;
export const TEACHER_PROMPT = 'Describe this image for a pilot, using ONLY the facts in the JSON below. Do not add any place, number or object that is not in the facts.';
export const visualFacts = (facts) => Object.fromEntries(Object.entries(facts || {}).filter(([, f]) => f && f.obs === 'visual'));
const redact = (s, key) => String(s).split(key).join('[redacted]');
export async function runTeacher(records, { send, model, temperature = 0.7, priceIn, priceOut, yesSpend = false, env = process.env, log = console.log, gaz = null, tokensIn = 1500, tokensOut = 200 }) {
  log(`teacher: estimated cost $${estimateCost({ n: records.length, tokensIn, tokensOut, priceIn, priceOut }).toFixed(2)} for ${records.length} records (model ${model}, ${tokensIn}+${tokensOut} tokens each)`);
  if (!yesSpend) { log('teacher: not run (pass --yes-spend to spend)'); return []; }
  const key = env.ANTHROPIC_API_KEY;
  if (!key) { log('teacher: not run (ANTHROPIC_API_KEY is not set)'); return []; }
  const out = [], sha = crypto.createHash('sha256').update(TEACHER_PROMPT).digest('hex');
  for (const r of records) {
    const facts = visualFacts(r.facts);
    let res;
    try { res = await send({ model, temperature, system: TEACHER_PROMPT, facts, image: r.narrator_frame, apiKey: key }); } catch (e) { log(`teacher: ${r.key}: request failed (${redact(e && e.message, key)})`); continue; }
    const text = res && typeof res.text === 'string' ? res.text.trim() : '';
    const chk = gaz && text ? verifyFreeText(text, { ...r, facts }, { gaz, obsRule: true }) : { verified: false };
    if (chk.verified) out.push({ key: r.key, task: 'caption_teacher', prompt: 'Describe the image in detail.', answer: text, fact_ids: Object.keys(facts), template_id: null, paraphrase_id: null,
      generator: { model, prompt_sha256: sha, temperature }, verified: true, false_premise: null, needsContext: false, context_facts: [] });
  }
  log(`teacher: kept ${out.length} of ${records.length}`);
  return out;
}
