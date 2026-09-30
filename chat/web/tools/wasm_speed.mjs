// chat/web/tools/wasm_speed.mjs — onnxruntime-web 1.30 WASM (Node, same V8/wasm as Chrome) decode speed of a CAPCOM q4 graph.
// A fresh --prompt-token prefill, then --steps single-token decode steps with the KV/conv cache. Needs a GatherBlockQuantized-free
// graph (tools/wasm_rewrite.py; its --accuracy-level makes the MatMulNBits variants compared here).
//   node chat/web/tools/wasm_speed.mjs <model.onnx> [--threads 1] [--steps 16] [--prompt 300]
import fs from 'node:fs';
const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
const NM = process.env.NODE_MODULES || '/Volumes/LaCie/astro-pilot/vlm/node/node_modules';
const ort = await import(`${NM}/onnxruntime-web/dist/ort.node.min.mjs`).catch(() => import(`${NM}/onnxruntime-web/dist/ort.wasm.min.mjs`));
const threads = +arg('threads', 1), steps = +arg('steps', 16), plen = +arg('prompt', 300);
const bytes = fs.readFileSync(process.argv[2]);
ort.env.wasm.numThreads = threads;
const t0 = performance.now(), s = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
console.log(`session ${(performance.now() - t0).toFixed(0)} ms, ${threads} thread(s)`);
const feed0 = {}, meta = Object.fromEntries(s.inputNames.map((n, i) => [n, s.inputMetadata[i]]));
for (const [n, md] of Object.entries(meta)) if (n.startsWith('past_')) { const shape = md.shape.map((d) => (d === 'batch_size' ? 1 : typeof d === 'string' ? 0 : d)); feed0[n] = new ort.Tensor('float32', new Float32Array(shape.reduce((a, b) => a * b, 1)), shape); }
const i64 = (a, dims) => new ort.Tensor('int64', BigInt64Array.from(a.map(BigInt)), dims);
let past = feed0, len = plen;
const run = async (ids) => {
  const r = await s.run({ input_ids: i64(ids, [1, ids.length]), attention_mask: i64(new Array(len).fill(1), [1, len]), num_logits_to_keep: new ort.Tensor('int64', BigInt64Array.from([1n]), []), ...past });
  past = {}; for (const [k, v] of Object.entries(r)) if (k.startsWith('present')) past[k.replace('present_conv', 'past_conv').replace('present.', 'past_key_values.')] = v;
};
let t = performance.now(); await run([1, ...new Array(plen - 1).fill(1098)]); const prefill = performance.now() - t;
t = performance.now(); for (let i = 0; i < steps; i++) { len++; await run([1098]); } const per = (performance.now() - t) / steps;
console.log(JSON.stringify({ model: process.argv[2].split('/').slice(-3).join('/'), threads, prefill_ms: Math.round(prefill), prompt: plen, decode_ms_per_token: +per.toFixed(1), tok_s: +(1000 / per).toFixed(1) }));
