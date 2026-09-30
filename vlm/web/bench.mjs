// vlm/web/bench.mjs — Node benches (spec §10.3): `narrator` (G2 / G6: load the exported folder with transformers.js at the
// chosen decoder and vision dtypes; per parity.json sample check prompt length, 64 image tokens and pixel_values within
// 1e-2 of PyTorch on the same 512² PNG, and the image features on the device at mean cos >= 0.99 vs PyTorch fp32; greedy
// generation; cold load, TTFT, ms/token and vision time after warm-up; on WebGPU the decoder must stay within
// --max-ms-per-token). Later tasks add `pilot-eye` (Task 13) and `runtime` (Task 16).
// Run from the repo: node vlm/web/bench.mjs <cmd> …
import fs from 'node:fs';
import path from 'node:path';
const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
const med = (a) => { const s = [...a].sort((x, y) => x - y); return s[Math.floor((s.length - 1) / 2)]; };
const IMG = 49190, EOU = 49279, PIXEL_TOL = 1e-2, HIDDEN = 576;
export function maxAbsDiff(a, b) {
  if (a.length !== b.length) throw new Error(`length ${a.length} != ${b.length}`);
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}
export const perToken = ({ ttft, total, n }) => (total - ttft) / (n - 1);
export function meanRowCos(a, b, d) {
  let s = 0;
  for (let r = 0; r < a.length / d; r++) {
    let ab = 0, aa = 0, bb = 0;
    for (let k = r * d; k < (r + 1) * d; k++) { ab += a[k] * b[k]; aa += a[k] * a[k]; bb += b[k] * b[k]; }
    s += ab / Math.sqrt(aa * bb);
  }
  return s / (a.length / d);
}
export const gateFailures = ({ ms_per_token, device, maxMsPerToken, vision_cos, minCos }) => [
  ...(device === 'webgpu' && ms_per_token > maxMsPerToken ? [`${ms_per_token.toFixed(1)} ms/token > ${maxMsPerToken} ms on WebGPU`] : []),
  ...(vision_cos < minCos ? [`vision features mean cos ${vision_cos.toFixed(4)} < ${minCos} on ${device}`] : []),
];
const timed = async (fn) => { const t = performance.now(), r = await fn(); return [performance.now() - t, r]; };
export async function benchNarrator({ modelDir, device = 'webgpu', tokens = 40, decoder = 'q4', vision = 'q4' }) {
  const tf = await import('@huggingface/transformers');
  tf.env.localModelPath = path.dirname(modelDir) + '/'; tf.env.allowRemoteModels = false; tf.env.allowLocalModels = true;
  const id = path.basename(modelDir), parity = JSON.parse(fs.readFileSync(path.join(modelDir, 'parity.json'), 'utf8'));
  const [cold, [processor, model]] = await timed(async () => [await tf.AutoProcessor.from_pretrained(id),
    await tf.AutoModelForVision2Seq.from_pretrained(id, { device, dtype: { embed_tokens: 'fp16', vision_encoder: vision, decoder_model_merged: decoder } })]);
  const ttft = [], total = [], vis = [], cos = [], texts = [];
  let pixel_max_abs = 0;
  for (const [i, s] of parity.samples.entries()) {
    const text = processor.apply_chat_template([{ role: 'user', content: [{ type: 'image' }, { type: 'text', text: s.prompt }] }], { add_generation_prompt: true });
    const inputs = await processor(text, [await tf.RawImage.read(path.join(modelDir, s.png))], { do_image_splitting: false }), ids = Array.from(inputs.input_ids.data, Number);
    if (ids.length !== parity.d.prompt_len[i]) throw new Error(`sample ${i}: prompt length ${ids.length} != PyTorch ${parity.d.prompt_len[i]}`);
    if (ids.filter((t) => t === IMG).length !== 64) throw new Error(`sample ${i}: ${ids.filter((t) => t === IMG).length} image tokens, expected 64`);
    const ref = fs.readFileSync(path.join(modelDir, s.pixels)), dpix = maxAbsDiff(inputs.pixel_values.data, new Float32Array(ref.buffer, ref.byteOffset, ref.byteLength / 4));
    if (dpix > PIXEL_TOL) throw new Error(`sample ${i}: max |Δpixel_values| ${dpix} > ${PIXEL_TOL} vs PyTorch`);
    pixel_max_abs = Math.max(pixel_max_abs, dpix);
    const gen = (n) => model.generate({ ...inputs, max_new_tokens: n, min_new_tokens: n, do_sample: false, eos_token_id: EOU });
    await gen(tokens);
    const [tv, feats] = await timed(() => model.encode_image({ pixel_values: inputs.pixel_values, pixel_attention_mask: inputs.pixel_attention_mask })), fr = fs.readFileSync(path.join(modelDir, s.feats));
    vis.push(tv); cos.push(meanRowCos(Float32Array.from(feats.data), new Float32Array(fr.buffer, fr.byteOffset, fr.byteLength / 4), HIDDEN));
    ttft.push((await timed(() => gen(1)))[0]);
    const [dt, out] = await timed(() => gen(tokens)); total.push(dt);
    texts.push(processor.batch_decode(out.slice(null, [ids.length, null]), { skip_special_tokens: true })[0]);
    if (i < 3) console.log(`sample ${i}: ttft ${ttft.at(-1).toFixed(0)} ms, ${tokens} tokens ${dt.toFixed(0)} ms :: ${JSON.stringify(texts.at(-1))}`);
  }
  const r = { decoder, vision, device, tokens, cold_ms: Math.round(cold), ttft_ms: med(ttft), total_ms: med(total), vision_ms: med(vis), pixel_max_abs, vision_cos: cos.reduce((x, y) => x + y, 0) / cos.length, vision_cos_min: Math.min(...cos) };
  return { ...r, ms_per_token: perToken({ ttft: r.ttft_ms, total: r.total_ms, n: tokens }), texts };
}
if (process.argv[1] && process.argv[1].endsWith('bench.mjs')) {
  const cmd = process.argv[2];
  if (cmd === 'narrator') {
    const r = await benchNarrator({ modelDir: arg('model-dir'), device: arg('device', 'webgpu'), tokens: +arg('tokens', 40), decoder: arg('decoder', 'q4'), vision: arg('vision', 'q4') });
    const { texts, ...summary } = r, bad = gateFailures({ ...r, maxMsPerToken: +arg('max-ms-per-token', 12), minCos: +arg('min-cos', 0.99) });
    console.log(JSON.stringify(summary));
    if (arg('texts-out')) fs.writeFileSync(arg('texts-out'), JSON.stringify(texts, null, 1));
    if (bad.length) { console.log('G2 FAIL: ' + bad.join('; ')); process.exit(1); }
    console.log('G2 transformers.js OK');
  } else { console.log('usage: bench.mjs narrator --model-dir <folder> [--device webgpu|cpu] [--tokens 40] [--decoder q4|q4f16|fp16|fp32] [--vision q4|q8|int8|fp16] [--max-ms-per-token 12] [--min-cos 0.99] [--texts-out <json>]'); process.exit(2); }
  process.exit(0);
}
