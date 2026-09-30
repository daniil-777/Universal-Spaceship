// vlm/web/bench.mjs — Node benches (spec §10.3): `narrator` (G2 / G6: load the exported folder with transformers.js at the
// chosen decoder and vision dtypes — default q4f16 + q8, the accepted deploy pair; per parity.json sample check prompt
// length, 64 image tokens and pixel_values within 1e-2 of PyTorch on the same 512² PNG, and the image features on the
// device at mean cos >= 0.99 vs PyTorch fp32; greedy generation; cold load, TTFT, ms/token and vision time after warm-up;
// on WebGPU the decoder must stay within --max-ms-per-token). `--vision all` instead gates every vision_encoder dtype in
// the folder on the device (check_parity picks its candidate on the CPU only). Later tasks add `pilot-eye` (Task 13) and
// `runtime` (Task 16). Run from the repo: node vlm/web/bench.mjs <cmd> …
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
const med = (a) => { const s = [...a].sort((x, y) => x - y); return s[Math.floor((s.length - 1) / 2)]; };
const readF32 = (p) => { const b = fs.readFileSync(p); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };
const IMG = 49190, EOU = 49279, PIXEL_TOL = 1e-2, HIDDEN = 576;
const VISION_SUFFIX = { fp32: '', fp16: '_fp16', q8: '_quantized', int8: '_int8', uint8: '_uint8', q4: '_q4', q4f16: '_q4f16', bnb4: '_bnb4' };
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
// written as !(x <= max) and !(x >= min) so that a NaN measurement fails
export const latencyFailures = ({ ms_per_token, device, maxMsPerToken }) => (device === 'webgpu' && !(ms_per_token <= maxMsPerToken) ? [`${ms_per_token?.toFixed(1)} ms/token > ${maxMsPerToken} ms on WebGPU`] : []);
export const visionFailures = ({ vision_cos, device, minCos }) => (!(vision_cos >= minCos) ? [`vision features mean cos ${vision_cos?.toFixed(4)} < ${minCos} on ${device}`] : []);
export const gateFailures = (r) => [...latencyFailures(r), ...visionFailures(r)];
export const visionVariants = (names) => Object.keys(VISION_SUFFIX).filter((d) => names.includes(`vision_encoder${VISION_SUFFIX[d]}.onnx`));
const timed = async (fn) => { const t = performance.now(), r = await fn(); return [performance.now() - t, r]; };
export async function loadNarrator({ modelDir, device = 'webgpu', decoder = 'q4f16', vision = 'q8' }) {
  const tf = await import('@huggingface/transformers');
  tf.env.localModelPath = path.dirname(modelDir) + '/'; tf.env.allowRemoteModels = false; tf.env.allowLocalModels = true;
  const id = path.basename(modelDir), parity = JSON.parse(fs.readFileSync(path.join(modelDir, 'parity.json'), 'utf8'));
  const [cold, [processor, model]] = await timed(async () => [await tf.AutoProcessor.from_pretrained(id),
    await tf.AutoModelForVision2Seq.from_pretrained(id, { device, dtype: { embed_tokens: 'fp16', vision_encoder: vision, decoder_model_merged: decoder } })]);
  return { tf, processor, model, parity, cold };
}
export async function prepare(tf, processor, modelDir, s) {
  const text = processor.apply_chat_template([{ role: 'user', content: [{ type: 'image' }, { type: 'text', text: s.prompt }] }], { add_generation_prompt: true });
  const inputs = await processor(text, [await tf.RawImage.read(path.join(modelDir, s.png))], { do_image_splitting: false });
  return { inputs, ids: Array.from(inputs.input_ids.data, Number) };
}
const encode = (model, inputs) => model.encode_image({ pixel_values: inputs.pixel_values, pixel_attention_mask: inputs.pixel_attention_mask });
export async function benchNarrator({ modelDir, device = 'webgpu', tokens = 40, decoder = 'q4f16', vision = 'q8' }) {
  const { tf, processor, model, parity, cold } = await loadNarrator({ modelDir, device, decoder, vision });
  const ttft = [], total = [], vis = [], cos = [], texts = [];
  let pixel_max_abs = 0;
  for (const [i, s] of parity.samples.entries()) {
    const { inputs, ids } = await prepare(tf, processor, modelDir, s), nImg = ids.filter((t) => t === IMG).length;
    if (ids.length !== parity.d.prompt_len[i]) throw new Error(`sample ${i}: prompt length ${ids.length} != PyTorch ${parity.d.prompt_len[i]}`);
    if (nImg !== 64) throw new Error(`sample ${i}: ${nImg} image tokens, expected 64`);
    const dpix = maxAbsDiff(inputs.pixel_values.data, readF32(path.join(modelDir, s.pixels)));
    if (dpix > PIXEL_TOL) throw new Error(`sample ${i}: max |Δpixel_values| ${dpix} > ${PIXEL_TOL} vs PyTorch`);
    pixel_max_abs = Math.max(pixel_max_abs, dpix);
    const gen = (n) => model.generate({ ...inputs, max_new_tokens: n, min_new_tokens: n, do_sample: false, eos_token_id: EOU });
    await gen(tokens);
    const [tv, feats] = await timed(() => encode(model, inputs));
    vis.push(tv); cos.push(meanRowCos(Float32Array.from(feats.data), readF32(path.join(modelDir, s.feats)), HIDDEN));
    ttft.push((await timed(() => gen(1)))[0]);
    const [dt, out] = await timed(() => gen(tokens)); total.push(dt);
    texts.push(processor.batch_decode(out.slice(null, [ids.length, null]), { skip_special_tokens: true })[0]);
    if (i < 3) console.log(`sample ${i}: ttft ${ttft.at(-1).toFixed(0)} ms, ${tokens} tokens ${dt.toFixed(0)} ms :: ${JSON.stringify(texts.at(-1))}`);
  }
  await model.dispose();
  const r = { decoder, vision, device, tokens, cold_ms: Math.round(cold), ttft_ms: med(ttft), total_ms: med(total), vision_ms: med(vis), pixel_max_abs, vision_cos: cos.reduce((x, y) => x + y, 0) / cos.length, vision_cos_min: Math.min(...cos) };
  return { ...r, ms_per_token: perToken({ ttft: r.ttft_ms, total: r.total_ms, n: tokens }), texts };
}
export async function benchVision({ modelDir, device = 'webgpu', minCos = 0.99, decoder = 'q4f16' }) {
  const rows = [];
  for (const vision of visionVariants(fs.readdirSync(path.join(modelDir, 'onnx')))) {
    const { tf, processor, model, parity } = await loadNarrator({ modelDir, device, decoder, vision }), cos = [], ms = [];
    for (const s of parity.samples) {
      const { inputs } = await prepare(tf, processor, modelDir, s);
      await encode(model, inputs);
      const [t, feats] = await timed(() => encode(model, inputs));
      ms.push(t); cos.push(meanRowCos(Float32Array.from(feats.data), readF32(path.join(modelDir, s.feats)), HIDDEN));
    }
    await model.dispose();
    const vision_cos = cos.reduce((x, y) => x + y, 0) / cos.length, mb = +(fs.statSync(path.join(modelDir, 'onnx', `vision_encoder${VISION_SUFFIX[vision]}.onnx`)).size / 1e6).toFixed(1);
    rows.push({ vision, mb, device, vision_cos, vision_cos_min: Math.min(...cos), vision_ms: med(ms), pass: visionFailures({ vision_cos, device, minCos }).length === 0 });
  }
  return rows;
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const cmd = process.argv[2], modelDir = arg('model-dir'), device = arg('device', 'webgpu'), minCos = +arg('min-cos', 0.99);
  if (cmd === 'runtime') { console.log(JSON.stringify(await benchRuntime({ eyeDir: arg('eye'), narratorDir: arg('narrator'), frames: +arg('frames', 60), device: arg('device', 'webgpu'), decoder: arg('decoder', 'q4f16'), vision: arg('vision', 'q8') }))); process.exit(0); }
  if (cmd === 'narrator' && arg('vision') === 'all') {
    const rows = await benchVision({ modelDir, device, minCos, decoder: arg('decoder', 'q4f16') }), cand = JSON.parse(fs.readFileSync(path.join(modelDir, 'parity.json'), 'utf8')).b.candidate;
    for (const r of rows) console.log(JSON.stringify(r));
    const ok = rows.filter((r) => r.pass).sort((a, b) => a.mb - b.mb), confirmed = rows.some((r) => r.vision === cand && r.pass);
    console.log(`smallest vision dtype at mean cos >= ${minCos} on ${device}: ${ok[0]?.vision ?? 'none'}; check_parity (CPU) candidate ${cand}: ${confirmed ? 'confirmed' : 'NOT confirmed'} on ${device}`);
    if (!confirmed) { console.log('G2 FAIL: the vision candidate does not hold on ' + device); process.exit(1); }
    console.log('G2 vision OK');
  } else if (cmd === 'narrator') {
    const r = await benchNarrator({ modelDir, device, tokens: +arg('tokens', 40), decoder: arg('decoder', 'q4f16'), vision: arg('vision', 'q8') });
    const { texts, ...summary } = r, bad = gateFailures({ ...r, maxMsPerToken: +arg('max-ms-per-token', 12), minCos });
    console.log(JSON.stringify(summary));
    if (arg('texts-out')) fs.writeFileSync(arg('texts-out'), JSON.stringify(texts, null, 1));
    if (bad.length) { console.log('G2 FAIL: ' + bad.join('; ')); process.exit(1); }
    console.log('G2 transformers.js OK');
  } else { console.log('usage: bench.mjs narrator --model-dir <folder> [--device webgpu|cpu] [--tokens 40] [--decoder q4f16|q4|fp16|fp32] [--vision q8|q4|int8|fp16|all] [--max-ms-per-token 12] [--min-cos 0.99] [--texts-out <json>]'); process.exit(2); }
  process.exit(0);
}

// ---- runtime (Task 16): the browser runtime modules run in Node — Pilot Eye (pilot-eye.js, ORT-web WASM, 1 thread) per-frame
// latency on the S grid after 8 warm-up frames (the first two only fill the ring), and the Narrator (narrator.js) cold load and
// the median of 3 greedy 40-token describes of parity sample 0 squared to 512². Either part runs alone (--eye / --narrator).
// An untrained Pilot Eye export has no labels.json: its parity.json size stands in. Declared after the CLI (hoisted).
// node vlm/web/bench.mjs runtime --eye <pilot eye export> --narrator <narrator web folder> [--frames 60] [--device webgpu|cpu] [--decoder q4f16|q4] [--vision q8]
export async function benchRuntime({ eyeDir = null, narratorDir = null, frames = 60, device = 'webgpu', decoder = 'q4f16', vision = 'q8' }) {
  const pct = (a, p) => { const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * (s.length - 1)))]; }, r = {};
  if (eyeDir) {
    const ort = await import('onnxruntime-web'), { createPilotEye } = await import('./pilot-eye.js'), lp = path.join(eyeDir, 'labels.json');
    const labels = fs.existsSync(lp) ? JSON.parse(fs.readFileSync(lp, 'utf8')) : { input: (JSON.parse(fs.readFileSync(path.join(eyeDir, 'parity.json'), 'utf8')).size || '160x96').split('x').map(Number) };
    const eye = await createPilotEye({ ort, encoderUrl: path.join(eyeDir, 'encoder.onnx'), headsUrl: path.join(eyeDir, 'heads.onnx'), labels }), [EW, EH] = labels.input || [160, 96], ms = [];
    const rgb = new Uint8Array(EW * EH * 3).map((_, i) => (i * 2654435761) >>> 24), statuses = new Set();
    for (let i = 0; i < frames + 8; i++) { const o = await eye.push({ rgb, sim_t_s: 0.2 * (i + 1), family: 'S', episode_id: 1 }); if (i >= 8) { ms.push(o.ms); statuses.add(o.status); } }
    r.pilot_eye_frame_ms = { median: med(ms), p90: pct(ms, 0.9), min: Math.min(...ms), n: ms.length, size: `${EW}x${EH}`, all_ok: statuses.size === 1 && statuses.has('ok') };
  }
  if (narratorDir) {
    const tf = await import('@huggingface/transformers'), { createNarrator } = await import('./narrator.js');
    const nar = await createNarrator({ tf, modelId: path.basename(narratorDir), localModelPath: path.dirname(path.resolve(narratorDir)) + '/', device, fallback: 'cpu', decoder, vision });
    const img = await tf.RawImage.read(JSON.parse(fs.readFileSync(path.join(narratorDir, 'parity.json'), 'utf8')).samples[0].image), d = [];
    await nar.describe(img, { maxNewTokens: 8 });
    for (let i = 0; i < 3; i++) d.push((await nar.describe(img, { maxNewTokens: 40, minNewTokens: 40 })).ms);
    const text = (await nar.describe(img, { maxNewTokens: 40 })).text; await nar.dispose();
    r.narrator = { device: nar.device, decoder, vision, cold_load_ms: Math.round(nar.loadMs), median_ms_40: Math.round(med(d)), gate_40_tokens_le_1000_ms: med(d) <= 1000, sample_text: text };
  }
  return r;
}
