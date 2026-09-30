import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createRequire } from 'node:module';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createFrameRing, pickAction, gridIndex, decodeHeads, prepareFrame, createPilotEye, ACTIONS } from '../vlm/web/pilot-eye.js';
import { templateSentence } from '../vlm/web/templates.js';
import { monitorFromHeads, contextLine, createNarrator, NARRATOR_EOS } from '../vlm/web/narrator.js';
import { vlmFile, checkPort } from '../vlm/web/serve.mjs';
import { renderContext, parseContext } from '../vlm/gen/text/context.js';
import { REASONS } from '../vlm/gen/schema.js';

test('frame ring: resets on family/episode change, non-increasing time or an off-grid step; ready at 3 frames', () => {
  const r = createFrameRing({ frameDt: 0.2, stepS: 1 / 15 }), f = (t, fam = 'S', ep = 1) => r.push({ sim_t_s: t, family: fam, episode_id: ep, map: t });
  assert.equal(f(0.2), null); assert.equal(f(0.4), null); const ok = f(0.6); assert.deepEqual(ok.maps, [0.2, 0.4, 0.6]); assert.ok(Math.abs(ok.dt[0] - 1) < 1e-9);
  assert.equal(f(0.6), null, 'time did not increase'); assert.equal(f(0.8), null); assert.ok(f(1.0));
  assert.equal(f(2.0), null, 'off-grid jump resets'); assert.equal(f(2.2, 'A'), null, 'family change resets'); assert.equal(f(2.4, 'A', 2), null, 'episode change resets');
  const j = createFrameRing({ frameDt: 0.2, stepS: 1 / 15 }); j.push({ sim_t_s: 1, family: 'S', episode_id: 1, map: 0 }); j.push({ sim_t_s: 1.2667, family: 'S', episode_id: 1, map: 1 });
  assert.ok(j.push({ sim_t_s: 1.4, family: 'S', episode_id: 1, map: 2 }), 'R7 one-step jitter (frame_dt_s rounded to 4 decimals) stays on the grid');
});
test('the sim-time grid posts once per crossing and never while paused', () => {
  const posts = []; let last = -1; for (const t of [0, 0.05, 0.19, 0.2, 0.2, 0.2, 0.41, 1.3]) { const g = gridIndex(t, 0.2); if (g > last) { posts.push(t); last = g; } }
  assert.deepEqual(posts, [0, 0.2, 0.41, 1.3]);
});
test('action choice: argmax over the family subset, ties to CONTINUE, NONE_SAFE when every sigmoid < 0.5', () => {
  const p = ACTIONS.map(() => 0.1); assert.equal(pickAction(p, 'S'), 'NONE_SAFE');
  p[ACTIONS.indexOf('CLIMB')] = 0.9; assert.equal(pickAction(p, 'S'), 'CLIMB'); p[0] = 0.9; assert.equal(pickAction(p, 'S'), 'CONTINUE');
  const d = ACTIONS.map(() => 0.1); d[ACTIONS.indexOf('CLIMB')] = 0.99; d[ACTIONS.indexOf('BREAKOUT')] = 0.8; assert.equal(pickAction(d, 'D'), 'BREAKOUT');
});
const OUT = { verdict: [0.1, 0.2, 3], severity: [0, 0, 0, 5, 0], reasons: Array(26).fill(-5).map((v, i) => (i === 0 ? 5 : v)), actions: ACTIONS.map((a) => (a === 'CLIMB' ? 5 : -5)), p_ref: [1.1], reg: [Math.log1p(2), Math.log1p(3), 0, 0, 0, 0], tags: Array(10).fill(-5), range: [0, 0, 5, 0, 0] };
test('heads decode into a template sentence and a Context line that parses back', () => {
  const h = decodeHeads(OUT, 'S'); assert.equal(h.verdict, 'UNSAFE'); assert.deepEqual(h.reasons, ['HAZARD_AHEAD']); assert.equal(h.action, 'CLIMB');
  const s = templateSentence(h, 'S'); assert.ok(!/undefined|NaN/.test(s)); assert.match(s, /^UNSAFE/);
  const m = monitorFromHeads(h, 'S'), line = renderContext({ telemetry: [], monitor: m }); assert.deepEqual(parseContext(line).monitor, m);
  assert.deepEqual([m.ttc_bin, m.clr_bin, m.p_ref], ['1-3 s', '<5 u', 0.75]);
});
test('decodeHeads follows the reason order of labels.json; L/D monitors carry no p_ref or bins; Z reads range and tags', () => {
  const order = [...REASONS].reverse(), h = decodeHeads(OUT, 'L', order); assert.deepEqual(h.reasons, [order[0]]); assert.equal(h.action, 'CLIMB');
  assert.deepEqual(monitorFromHeads(h, 'D'), { verdict: 'UNSAFE', severity: 3, reasons: [order[0]], action: 'CLIMB', p_ref: null, ttc_bin: 'none', clr_bin: 'none' });
  const z = decodeHeads({ ...OUT, tags: OUT.tags.map((v, i) => (i === 2 || i === 9 ? 4 : v)) }, 'Z'); assert.equal(z.range_bin, 2); assert.deepEqual(z.tags, ['MOUNTAINS', 'NIGHT']);
  const zs = templateSentence(z, 'Z'); assert.equal(zs, 'A view from 100 to 400 km showing mountains, at night.');
  assert.equal(templateSentence({ ...z, tags: [], range_bin: 4 }, 'Z'), 'A view from over 1500 km.');
  assert.equal(templateSentence({ ...h, verdict: 'SAFE', reasons: [], action: 'NONE_SAFE' }, 'L'), 'SAFE. Advice: no safe action.');
  assert.equal(contextLine({ heads: z, family: 'Z', facts: {} }), 'Context: telemetry: none.', 'Z has no monitor tuple');
});
test('prepareFrame box-resizes RGBA to the labels.input size as RGB', () => {
  const rgba = new Uint8Array(320 * 192 * 4).fill(77); assert.equal(prepareFrame(rgba, 320, 192).length, 160 * 96 * 3);
  const big = prepareFrame(rgba, 320, 192, [224, 128]); assert.equal(big.length, 224 * 128 * 3); assert.ok(big.every((v) => v === 77));
});

// a tiny untrained Pilot Eye written as ONNX bytes (no Python): encoder pixels [1,3,H,W] -> AveragePool 32 -> 1x1 Conv -> map
// [1,64,H/32,W/32]; heads (m0, m1, m2, dt) -> GlobalAveragePool(m2 - m1) ++ dt -> one Gemm per head, in the HEADS order
const HEADS = { verdict: 3, severity: 5, reasons: 26, actions: 11, p_ref: 1, reg: 6, tags: 10, range: 5 };
const vint = (n) => { const b = []; let x = BigInt(n); do { let c = Number(x & 0x7fn); x >>= 7n; if (x) c |= 0x80; b.push(c); } while (x); return b; };
const fld = (no, bytes) => [...vint((no << 3) | 2), ...vint(bytes.length), ...bytes], num = (no, n) => [...vint(no << 3), ...vint(n)], str = (no, s) => fld(no, [...Buffer.from(s)]);
const vinfo = (name, dims, no = 11) => fld(no, [...str(1, name), ...fld(2, fld(1, [...num(1, 1), ...fld(2, dims.flatMap((d) => fld(1, num(1, d))))]))]);
const out = (name, dims) => vinfo(name, dims, 12);
const tensor = (name, dims, vals) => fld(5, [...dims.flatMap((d) => num(1, d)), ...num(2, 1), ...str(8, name), ...fld(9, [...new Uint8Array(Float32Array.from(vals).buffer)])]);
const ints = (name, a) => fld(5, [...str(1, name), ...num(20, 7), ...a.flatMap((v) => num(8, v))]), int = (name, v) => fld(5, [...str(1, name), ...num(20, 2), ...num(3, v)]);
const node = (op, ins, outs, attrs = []) => fld(1, [...ins.flatMap((s) => str(1, s)), ...outs.flatMap((s) => str(2, s)), ...str(4, op), ...attrs.flat()]);
const model = (graph) => Uint8Array.from([...num(1, 8), ...fld(8, [...str(1, ''), ...num(2, 17)]), ...fld(7, [...str(2, 'g'), ...graph])]);
const wts = (n, k) => Array.from({ length: n }, (_, i) => Math.sin(i * 1.7 + k) * 0.5);
function tinyPilotEye(W = 160, H = 96) {
  const h = H / 32, w = W / 32;
  const enc = model([...node('AveragePool', ['pixels'], ['p'], [ints('kernel_shape', [32, 32]), ints('strides', [32, 32])]), ...node('Conv', ['p', 'cw'], ['map']),
    ...tensor('cw', [64, 3, 1, 1], wts(192, 0)), ...vinfo('pixels', [1, 3, H, W]), ...out('map', [1, 64, h, w])]);
  const heads = model([...node('Sub', ['m2', 'm1'], ['d']), ...node('GlobalAveragePool', ['d'], ['g']), ...node('Flatten', ['g'], ['f'], [int('axis', 1)]), ...node('Concat', ['f', 'dt'], ['x'], [int('axis', 1)]),
    ...Object.entries(HEADS).flatMap(([k, n], i) => [...node('Gemm', ['x', `w_${k}`], [k]), ...tensor(`w_${k}`, [66, n], wts(66 * n, i + 1))]),
    ...['m0', 'm1', 'm2'].flatMap((m) => vinfo(m, [1, 64, h, w])), ...vinfo('dt', [1, 2]), ...Object.entries(HEADS).flatMap(([k, n]) => out(k, [1, n]))]);
  return { enc, heads };
}
async function loadOrt() {
  try { const main = createRequire(fileURLToPath(new URL('../vlm/web/bench.mjs', import.meta.url))).resolve('onnxruntime-web'); return await import(pathToFileURL(path.join(path.dirname(main), 'ort.node.min.mjs')).href); } catch { return null; }
}
test('createPilotEye runs a tiny ONNX pair through ORT-web: warming up until the ring holds 3 frames, Z at once', async (t) => {
  const ort = await loadOrt(); if (!ort) return t.skip('onnxruntime-web is not installed under vlm/node_modules');
  const { enc, heads } = tinyPilotEye(), labels = { input: [160, 96], nominal_frame_dt: { S: 0.2, A: 0.2, L: 0.25, D: 2 }, reasons: [...REASONS] };
  const eye = await createPilotEye({ ort, encoderUrl: enc, headsUrl: heads, labels }), rgb = (v) => new Uint8Array(160 * 96 * 3).map((_, i) => (i * v) % 256);
  const push = (i, family = 'S', episode_id = 1) => eye.push({ rgb: rgb(i + 1), sim_t_s: 0.2 * (i + 1), family, episode_id });
  assert.equal((await push(0)).status, 'warming up'); assert.equal((await push(1)).status, 'warming up');
  const r = await push(2); assert.equal(r.status, 'ok'); assert.ok(['SAFE', 'CAUTION', 'UNSAFE'].includes(r.verdict)); assert.ok(r.severity >= 0 && r.severity <= 4);
  assert.ok(ACTIONS.includes(r.action) && r.reg.length === 6 && r.range_bin >= 0 && r.range_bin <= 4 && Number.isFinite(r.ms)); assert.ok(!/undefined|NaN/.test(r.sentence), r.sentence);
  assert.equal((await push(3, 'S', 2)).status, 'warming up', 'a new episode resets the ring');
  const z = await eye.push({ rgb: rgb(9), sim_t_s: 0, family: 'Z', episode_id: 7 }); assert.equal(z.status, 'ok'); assert.match(z.sentence, /^A view from /);
});

test('bench.mjs runtime: the Pilot Eye part times every frame after warm-up on an untrained-style export (parity.json size, no labels.json)', async (t) => {
  if (!(await loadOrt())) return t.skip('onnxruntime-web is not installed under vlm/node_modules');
  const { benchRuntime } = await import('../vlm/web/bench.mjs'), dir = fs.mkdtempSync(path.join(os.tmpdir(), 'apv-runtime-')), { enc, heads } = tinyPilotEye();
  try {
    fs.writeFileSync(path.join(dir, 'encoder.onnx'), enc); fs.writeFileSync(path.join(dir, 'heads.onnx'), heads); fs.writeFileSync(path.join(dir, 'parity.json'), JSON.stringify({ size: '160x96' }));
    const r = await benchRuntime({ eyeDir: dir, frames: 5 }); assert.equal(r.pilot_eye_frame_ms.n, 5); assert.ok(r.pilot_eye_frame_ms.all_ok); assert.equal(r.pilot_eye_frame_ms.size, '160x96');
    assert.ok(r.pilot_eye_frame_ms.median > 0 && r.pilot_eye_frame_ms.p90 >= r.pilot_eye_frame_ms.median); assert.equal(r.narrator, undefined);
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});

// transformers.js is injected: a stand-in at the tf boundary records what the Narrator loader asks of it
function fakeTf({ failOn = null } = {}) {
  const log = { loads: [], seen: null, gen: null, streamer: null };
  const processor = Object.assign(async (text, images, opts) => { log.seen = { text, images, opts }; return { input_ids: { dims: [1, 7] }, pixel_values: 'pv' }; },
    { apply_chat_template: (msgs) => msgs[0].content.map((c) => c.text || '<image>').join('|'), batch_decode: (ids) => [` decoded ${ids} `], tokenizer: {} });
  const mdl = { generate: async (o) => { log.gen = o; return { slice: (a, b) => `${a}:${b.join(',')}` }; } };
  const tf = { env: {}, log, AutoProcessor: { from_pretrained: async () => processor },
    AutoModelForVision2Seq: { from_pretrained: async (id, o) => { log.loads.push({ id, ...o }); if (o.device === failOn) throw new Error('no adapter'); return mdl; } },
    TextStreamer: class { constructor(tok, o) { log.streamer = o; } } };
  return tf;
}
test('createNarrator: deploy dtypes (q4f16 decoder, q8 vision), WASM fallback, 512² square, Context before the task, eos 49279', async () => {
  const tf = fakeTf({ failOn: 'webgpu' }), nar = await createNarrator({ tf, modelId: 'narrator_v0/web', localModelPath: '/__vlm/models/' });
  assert.deepEqual(tf.log.loads.map((l) => l.device), ['webgpu', 'wasm']); assert.deepEqual(tf.log.loads[0].dtype, { embed_tokens: 'fp16', vision_encoder: 'q8', decoder_model_merged: 'q4f16' });
  assert.deepEqual([tf.env.localModelPath, tf.env.allowRemoteModels, tf.env.allowLocalModels], ['/__vlm/models/', false, true]); assert.ok(nar.loadMs >= 0);
  let resized = null; const img = { width: 1600, height: 900, resize: async (w, h, o) => { resized = [w, h, o.resample]; return { width: w, height: h }; } }, toks = [];
  const d = await nar.describe(img, { context: 'Context: telemetry: none.', task: 'Is it safe?', maxNewTokens: 12, onToken: (x) => toks.push(x) });
  assert.deepEqual(resized, [512, 512, 1], 'LANCZOS (resample 1) to 512x512 before the processor');
  assert.equal(tf.log.seen.text, '<image>|Context: telemetry: none.\nIs it safe?'); assert.deepEqual(tf.log.seen.images[0], { width: 512, height: 512 }); assert.equal(tf.log.seen.opts.do_image_splitting, false);
  assert.deepEqual([tf.log.gen.max_new_tokens, tf.log.gen.do_sample, tf.log.gen.eos_token_id, NARRATOR_EOS], [12, false, 49279, 49279]); assert.ok(tf.log.streamer.skip_prompt);
  assert.equal(d.text, 'decoded null:7,'); assert.ok(d.ms >= 0);
  const q4 = fakeTf(); await createNarrator({ tf: q4, modelId: 'm', decoder: 'q4', vision: 'q8' }); assert.equal(q4.log.loads[0].dtype.decoder_model_merged, 'q4');
  const lfm = fakeTf(); await createNarrator({ tf: lfm, modelId: 'm', model: 'lfm' }); assert.equal(lfm.log.loads[0].dtype, 'q4f16', 'Plan B keeps its own processor and one dtype');
});

test('serve.mjs: /__vlm/ maps only models, datasets and raw under the LaCie root; reserved ports are refused', () => {
  const R = '/L/vlm';
  assert.equal(vlmFile('/__vlm/models/pe/encoder.onnx', R), '/L/vlm/models/pe/encoder.onnx'); assert.equal(vlmFile('/__vlm/raw/run/S/f0.png', R), '/L/vlm/raw/run/S/f0.png');
  for (const bad of ['/__vlm/logs/x.json', '/__vlm/models/../logs/x', '/__vlm/../../etc/passwd', '/__vlm/models', '/__vlm/modelsX/a', '/vlm/web/demo.html', '/__vlm/models/._encoder.onnx']) assert.equal(vlmFile(bad, R), null, bad);
  for (const p of [8788, 8790, 8791]) assert.throws(() => checkPort(p), /reserved/); assert.equal(checkPort(0), 0); assert.equal(checkPort(8795), 8795);
});
test('serve.mjs serves the demo page and 404s outside the /__vlm/ allow-list (spawned on a free port)', { timeout: 20000 }, async () => {
  const srv = spawn(process.execPath, [fileURLToPath(new URL('../vlm/web/serve.mjs', import.meta.url)), '--port', '0'], { stdio: ['ignore', 'pipe', 'pipe'] });
  try {
    const base = await new Promise((ok, no) => { let s = ''; srv.stdout.on('data', (d) => { s += d; const m = /demo: (http:\/\/127\.0\.0\.1:\d+)\//.exec(s); if (m) ok(m[1]); }); srv.on('exit', (c) => no(new Error(`serve.mjs exited ${c}`))); });
    const page = await fetch(`${base}/vlm/web/demo.html`); assert.equal(page.status, 200); assert.match(await page.text(), /Astro Pilot Vision/);
    assert.equal((await fetch(`${base}/__vlm/logs/nothing.json`)).status, 404); assert.equal((await fetch(`${base}/__vlm/models/no-such-model/labels.json`)).status, 404);
  } finally { srv.kill(); }
});
