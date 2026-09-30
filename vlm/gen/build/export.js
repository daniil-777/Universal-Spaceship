// vlm/gen/build/export.js — the §9 exports: SmolVLM chat rows with the §5.7 Context line and context_src, LLaVA and COCO views,
// the Pilot Eye JSONL and its uint8 cache. The one Context rule is context.js rowContext()/contextSupplies() (ruling T7-c):
// an item is kept only when the row's Context supplies every context-class fact it cites, and it carries the Context when it
// needs it, else with p = 0.5. A row's Context is the record's telemetry (exact) and, for flight, its row monitor
// (rec.row_monitor: gt+noise or oof, set by build.mjs) — the same Context its texts were generated and verified against.
import fs from 'node:fs';
import path from 'node:path';
import sharp from 'sharp';
import { renderContext, telemetryOf, rowContext } from '../text/context.js';
import { verifyFreeText } from '../text/verify.js';
import { SAFETY_TEXT_IDS } from '../schema.js';
import { boxResize } from '../boxresize.js';

export const LACIE = '/Volumes/LaCie/astro-pilot/vlm';
// ruling T10-g: inside 0.5 m axial the untextured port face fills the D eye view; such records stay for Narrator only
export const EYE_MIN_AXIAL_M = 0.5;
// ruling T11-a: in train, the injected member of a pixel-identical minimal pair is left out (identical pixels with a different
// label are label noise for a vision-only model); it stays in val/test for the paired metric and in every Narrator row
export function eyeExcluded(rec, split = null) {
  const a = rec.family === 'D' && rec.facts.axial_m ? rec.facts.axial_m.v : null;
  if (typeof a === 'number' && a < EYE_MIN_AXIAL_M) return `D axial ${a} m < ${EYE_MIN_AXIAL_M} m (ruling T10-g)`;
  return split === 'train' && rec.pixel_identical_pair && rec.provenance && rec.provenance.injection ? 'injected member of a pixel-identical pair, train (ruling T11-a)' : null;
}
// a dataset name is one plain path segment (build.mjs deletes datasets/<name> before writing it)
export const datasetNameOk = (n) => typeof n === 'string' && /^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(n) && n !== '.' && n !== '..';
// the context-class ids of a record: its obs:context facts and the safety.* ids text cites
export function contextClassIds(rec) {
  const s = new Set(Object.entries(rec.facts).filter(([, f]) => f && f.obs === 'context').map(([id]) => id));
  for (const k of SAFETY_TEXT_IDS) s.add(k);
  return s;
}
export const rowContextOf = (rec) => ({ telemetry: telemetryOf(rec.facts, rec.family), monitor: rec.family === 'Z' ? null : rec.row_monitor ?? null });
// template items carry context_facts/needsContext (items.js); an item without them (e.g. a hand-made one) gets them from its fact ids
const withContextFacts = (it, ids) => (Array.isArray(it.context_facts) ? it : { ...it, context_facts: it.fact_ids.filter((f) => ids.has(f)), needsContext: it.fact_ids.some((f) => ids.has(f)) });
// drops (optional): counts {unsupplied, screen} of the items left out. gaz (optional): a flight row that goes out without its
// Context is screened once more as free text with no Context (it may state visual facts only), and left out if it fails.
export function narratorRows(rec, { rng, split, mode, gaz = null, drops = null }) {
  const ctx = rowContextOf(rec), ids = contextClassIds(rec), flight = rec.family !== 'Z', line = renderContext(ctx), rows = [];
  for (const raw of rec.texts) {
    const it = withContextFacts(raw, ids), { keep, withCtx } = rowContext(it, ctx, rng);
    if (!keep) { if (drops) drops.unsupplied = (drops.unsupplied || 0) + 1; continue; }
    if (!withCtx && flight && gaz && !verifyFreeText(it.answer, rec, { gaz, context: null }).verified) { if (drops) drops.screen = (drops.screen || 0) + 1; continue; }
    const prompt = withCtx ? `${line}\n${it.prompt}` : it.prompt;
    rows.push({ images: [rec.narrator_frame], messages: [{ role: 'user', content: [{ type: 'image' }, { type: 'text', text: prompt }] }, { role: 'assistant', content: [{ type: 'text', text: it.answer }] }],
      task: it.task, key: rec.key, family: rec.family, context: withCtx, context_src: withCtx && flight ? mode : null, fact_ids: it.fact_ids, template_id: it.template_id ?? null, split });
  }
  return rows;
}
export const llavaOf = (rows) => rows.map((x, i) => ({ id: `${x.key}-${i}`, image: x.images[0], conversations: [{ from: 'human', value: `<image>\n${x.messages[0].content[1].text}` }, { from: 'gpt', value: x.messages[1].content[0].text }] }));
export function cocoOf(recs) {
  let id = 0;
  return { images: recs.map((r, i) => ({ id: i, file_name: r.narrator_frame, key: r.key })),
    annotations: recs.flatMap((r, i) => r.texts.filter((t) => t.task.startsWith('caption')).map((t) => ({ id: id++, image_id: i, caption: t.answer }))) };
}
export const EYE_SIZE = (process.env.APV_EYE_SIZE || '160x96').split('x').map(Number);
async function eyeFrame(file, [EW, EH]) { const { data, info } = await sharp(file).ensureAlpha().raw().toBuffer({ resolveWithObject: true }); return boxResize(data, info.width, info.height, EW, EH, info.channels); }
// cache/pilot_eye_<split>.u8: 3 rows of EW x EH x 3 per record in order (a Z record repeats its one frame, §10.1); returns
// {key: first row}
export async function writeCache(outDir, split, recs, { root = LACIE, size = EYE_SIZE } = {}) {
  fs.mkdirSync(path.join(outDir, 'cache'), { recursive: true });
  const fd = fs.openSync(path.join(outDir, 'cache', `pilot_eye_${split}.u8`), 'w'), index = {}; let row = 0;
  try {
    for (const r of recs) {
      index[r.key] = row;
      const px = []; for (const f of r.frames) px.push(await eyeFrame(path.join(root, f), size));
      while (px.length < 3) px.push(px[px.length - 1]);
      for (const x of px) fs.writeSync(fd, x);
      row += 3;
    }
  } finally { fs.closeSync(fd); }
  return index;
}
export function writeJsonl(file, rows) { fs.mkdirSync(path.dirname(file), { recursive: true }); fs.writeFileSync(file, rows.map((r) => JSON.stringify(r)).join('\n') + (rows.length ? '\n' : '')); }
