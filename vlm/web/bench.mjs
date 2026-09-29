// vlm/web/bench.mjs — Node benches (spec §10.3): `narrator` (G2 / G6: load the exported folder with transformers.js,
// check prompt length and 64 image tokens against parity.json, greedy generation, 40-token latency after warm-up).
// Later tasks add `pilot-eye` (Task 13) and `runtime` (Task 16). Run from the repo: node vlm/web/bench.mjs <cmd> …
import fs from 'node:fs';
import path from 'node:path';
const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
const med = (a) => { const s = [...a].sort((x, y) => x - y); return s[Math.floor((s.length - 1) / 2)]; }, q = (a, p) => { const s = [...a].sort((x, y) => x - y); return s[Math.floor((s.length - 1) * p)]; };
export async function benchNarrator({ modelDir, device = 'webgpu', tokens = 40 }) {
  const tf = await import('@huggingface/transformers');
  tf.env.localModelPath = path.dirname(modelDir) + '/'; tf.env.allowRemoteModels = false; tf.env.allowLocalModels = true;
  const id = path.basename(modelDir), parity = JSON.parse(fs.readFileSync(path.join(modelDir, 'parity.json'), 'utf8')), t0 = performance.now();
  const processor = await tf.AutoProcessor.from_pretrained(id);
  const model = await tf.AutoModelForVision2Seq.from_pretrained(id, { device, dtype: { embed_tokens: 'fp16', vision_encoder: 'q4', decoder_model_merged: 'q4' } });
  const cold = performance.now() - t0, imgId = 49190, lat = [];
  for (const [i, s] of parity.samples.entries()) {
    const text = processor.apply_chat_template([{ role: 'user', content: [{ type: 'image' }, { type: 'text', text: s.prompt }] }], { add_generation_prompt: true });
    const inputs = await processor(text, [await tf.RawImage.read(s.image)], { do_image_splitting: false }), ids = Array.from(inputs.input_ids.data, Number);
    if (ids.length !== parity.d.prompt_len[i]) throw new Error(`sample ${i}: prompt length ${ids.length} != PyTorch ${parity.d.prompt_len[i]}`);
    if (ids.filter((t) => t === imgId).length !== 64) throw new Error(`sample ${i}: ${ids.filter((t) => t === imgId).length} image tokens, expected 64`);
    for (let r = 0; r < 2; r++) {
      const t1 = performance.now(), out = await model.generate({ ...inputs, max_new_tokens: tokens, min_new_tokens: tokens, do_sample: false, eos_token_id: 49279 }), dt = performance.now() - t1;
      if (r === 1) { lat.push(dt); console.log(`sample ${i}: ${dt.toFixed(0)} ms for ${tokens} tokens :: ${JSON.stringify(processor.batch_decode(out.slice(null, [ids.length, null]), { skip_special_tokens: true })[0])}`); }
    }
  }
  return { cold_ms: cold, median_ms_40: med(lat), p90_ms_40: q(lat, 0.9) };
}
if (process.argv[1] && process.argv[1].endsWith('bench.mjs')) {
  const cmd = process.argv[2];
  if (cmd === 'narrator') { const r = await benchNarrator({ modelDir: arg('model-dir'), device: arg('device', 'webgpu'), tokens: +arg('tokens', 40) }); console.log(JSON.stringify(r)); console.log('G2 transformers.js OK'); }
  else { console.log('usage: bench.mjs narrator --model-dir <folder> [--device webgpu|cpu] [--tokens 40]'); process.exit(2); }
  process.exit(0);
}
