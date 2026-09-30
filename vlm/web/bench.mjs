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
// Task 13 — `pilot-eye`: ORT-web (wasm, 1 thread) latency of an export_onnx.py folder: encoder median/min/p90 after 8 warm-up
// runs, heads median (R13 gate: 160x96 encoder median <= 10 ms, heads <= 1 ms; 224x128 is the alternate only if <= 16 ms)
export async function benchPilotEye({ dir, runs = 80 }) {
  const ort = await import('onnxruntime-web'); ort.env.wasm.numThreads = 1;
  const q = (a, p) => { const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * (s.length - 1)))]; };
  const load = (f) => ort.InferenceSession.create(new Uint8Array(fs.readFileSync(path.join(dir, f))), { executionProviders: ['wasm'] });
  const enc = await load('encoder.onnx'), hd = await load('heads.onnx');
  const [W, H] = (JSON.parse(fs.readFileSync(path.join(dir, 'parity.json'), 'utf8')).size || '160x96').split('x').map(Number), x = new ort.Tensor('float32', new Float32Array(3 * H * W).fill(0.1), [1, 3, H, W]);
  for (let i = 0; i < 8; i++) await enc.run({ pixels: x });
  const te = [], th = []; let map = null;
  for (let i = 0; i < runs; i++) { const t0 = performance.now(); map = (await enc.run({ pixels: x })).map; te.push(performance.now() - t0); }
  const dt = new ort.Tensor('float32', new Float32Array([1, 1]), [1, 2]);
  for (let i = 0; i < runs; i++) { const t0 = performance.now(); await hd.run({ m0: map, m1: map, m2: map, dt }); th.push(performance.now() - t0); }
  return { size: `${W}x${H}`, encoder: { median_ms: med(te), min_ms: Math.min(...te), p90_ms: q(te, 0.9), runs }, heads: { median_ms: med(th) } };
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const cmd = process.argv[2], modelDir = arg('model-dir'), device = arg('device', 'webgpu'), minCos = +arg('min-cos', 0.99);
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
  } else if (cmd === 'pilot-eye') { console.log(JSON.stringify(await benchPilotEye({ dir: arg('dir'), runs: +arg('runs', 80) })));
  } else { console.log('usage: bench.mjs narrator --model-dir <folder> [--device webgpu|cpu] [--tokens 40] [--decoder q4f16|q4|fp16|fp32] [--vision q8|q4|int8|fp16|all] [--max-ms-per-token 12] [--min-cos 0.99] [--texts-out <json>]'); process.exit(2); }
  process.exit(0);
}
