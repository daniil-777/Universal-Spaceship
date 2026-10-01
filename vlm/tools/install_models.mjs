// vlm/tools/install_models.mjs — the seamless install of the Colab zip (ruling V1-9):
//   node vlm/tools/install_models.mjs <apv-models-v1.zip> [--models-dir /Volumes/LaCie/astro-pilot/vlm/models] [--no-deep]
// 1. unpacks the zip into a staging folder and checks every file against MANIFEST.json (SHA-256 and size);
// 2. Pilot Eye: re-runs the export's parity sample in Node with onnxruntime-web, the runtime the page uses (the input is
//    rebuilt from its formula; every head within the sample's tolerance, 1e-3);
// 3. Narrator: the narrator-v0 file manifest (q4f16 decoder, q8 vision encoder, fp16 embed_tokens, tokenizer and configs),
//    the configs the browser relies on (preprocessor do_resize false and no splitting at 512, eos 49279, a chat template)
//    and, unless --no-deep, an onnxruntime-node session per ONNX file with the transformers.js input/output names;
// 4. moves pilot-eye-v1/ and narrator-v1/ into the models folder (an existing folder is renamed <name>.prev-<time>, never
//    deleted) and writes models/current.json {eye, narrator, ...}, which the site's Narrator card and vlm/web/demo.html read
//    at /__vlm/models/current.json (node vlm/web/serve.mjs), falling back to v0 without it.
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import AdmZip from 'adm-zip';

export const MODELS = '/Volumes/LaCie/astro-pilot/vlm/models';
export const NARRATOR_FILES = ['onnx/decoder_model_merged_q4f16.onnx', 'onnx/embed_tokens_fp16.onnx', 'onnx/vision_encoder_quantized.onnx', 'config.json', 'tokenizer.json', 'tokenizer_config.json',
  'chat_template.json', 'preprocessor_config.json', 'processor_config.json', 'generation_config.json'];
export const EYE_FILES = ['encoder.onnx', 'heads.onnx', 'labels.json', 'parity_sample.json'];
const sha = (p) => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const readJson = (p) => JSON.parse(fs.readFileSync(p, 'utf8'));
const stamp = () => new Date().toISOString().replace(/[:.]/g, '-');

// unpack into dir and check it against the manifest; returns the manifest
export function unpack(zipFile, dir) {
  const zip = new AdmZip(zipFile), entry = zip.getEntry('MANIFEST.json');
  if (!entry) throw new Error(`${zipFile}: no MANIFEST.json (not an apv-models zip)`);
  const man = JSON.parse(entry.getData().toString('utf8'));
  if (man.format !== 'apv-models/1') throw new Error(`MANIFEST.json format ${man.format}: expected apv-models/1`);
  for (const e of zip.getEntries()) {
    const name = e.entryName;
    if (e.isDirectory) continue;
    if (name.includes('..') || path.isAbsolute(name) || path.basename(name).startsWith('._') || name.startsWith('__MACOSX/')) continue;
    const out = path.join(dir, name); fs.mkdirSync(path.dirname(out), { recursive: true }); fs.writeFileSync(out, e.getData());
  }
  const bad = [];
  for (const [f, m] of Object.entries(man.files || {})) {
    const p = path.join(dir, f);
    if (!fs.existsSync(p)) bad.push(`${f} missing`); else if (fs.statSync(p).size !== m.bytes || sha(p) !== m.sha256) bad.push(`${f} differs from MANIFEST.json`);
  }
  if (bad.length) throw new Error(`the zip does not match its manifest (a truncated download?): ${bad.slice(0, 5).join('; ')}`);
  return man;
}
// the parity input install and export share: byte k of frame f is (imul(k + f*n, 2654435761) >>> 24) / 255
export const nodeInput = (n, frame) => Float32Array.from({ length: n }, (_, k) => (Math.imul(k + frame * n, 2654435761) >>> 24) / 255);
export async function checkEye(dir, { ort = null } = {}) {
  for (const f of EYE_FILES) if (!fs.existsSync(path.join(dir, f))) throw new Error(`Pilot Eye: ${f} is missing`);
  ort = ort || (await import('onnxruntime-web')); if (ort.env && ort.env.wasm) ort.env.wasm.numThreads = 1;
  const s = readJson(path.join(dir, 'parity_sample.json')), [W, H] = s.size.split('x').map(Number), n = 3 * H * W;
  const load = (f) => ort.InferenceSession.create(new Uint8Array(fs.readFileSync(path.join(dir, f))), { executionProviders: ['wasm'] });
  const enc = await load('encoder.onnx'), hd = await load('heads.onnx'), maps = [];
  for (let f = 0; f < 3; f++) maps.push((await enc.run({ pixels: new ort.Tensor('float32', nodeInput(n, f), [1, 3, H, W]) })).map);
  const out = await hd.run({ m0: maps[0], m1: maps[1], m2: maps[2], dt: new ort.Tensor('float32', Float32Array.from(s.dt), [1, 2]) });
  let max = 0;
  for (const [k, want] of Object.entries(s.outputs)) {
    const got = out[k] && out[k].data; if (!got || got.length !== want.length) throw new Error(`Pilot Eye: head ${k} has ${got ? got.length : 0} values, the sample ${want.length}`);
    for (let i = 0; i < want.length; i++) max = Math.max(max, Math.abs(got[i] - want[i]));
  }
  if (!(max <= s.tol)) throw new Error(`Pilot Eye parity in Node: max |d| ${max} > ${s.tol}`);
  return { max_abs_diff: max, heads: Object.keys(s.outputs).length, size: s.size };
}
export async function checkNarrator(dir, { deep = true, ort = null } = {}) {
  const miss = NARRATOR_FILES.filter((f) => !fs.existsSync(path.join(dir, f)) || fs.statSync(path.join(dir, f)).size === 0);
  if (miss.length) throw new Error(`Narrator: missing or empty ${miss.join(', ')}`);
  const pre = readJson(path.join(dir, 'preprocessor_config.json')), gen = readJson(path.join(dir, 'generation_config.json')), cfg = readJson(path.join(dir, 'config.json'));
  const errs = [];
  if (pre.do_resize !== false || pre.do_image_splitting !== false || (pre.max_image_size || {}).longest_edge !== 512) errs.push('preprocessor_config.json must have do_resize false, do_image_splitting false, max_image_size 512 (callers square to 512²)');
  if (gen.eos_token_id !== 49279 && !(Array.isArray(gen.eos_token_id) && gen.eos_token_id.includes(49279))) errs.push('generation_config.json eos_token_id is not 49279 (<end_of_utterance>)');
  if (cfg.model_type !== 'idefics3') errs.push(`config.json model_type ${cfg.model_type}, expected idefics3 (SmolVLM)`);
  if (!readJson(path.join(dir, 'chat_template.json')).chat_template) errs.push('chat_template.json has no chat_template');
  readJson(path.join(dir, 'tokenizer.json'));
  if (errs.length) throw new Error(`Narrator: ${errs.join('; ')}`);
  const res = { files: NARRATOR_FILES.length, mb: +(NARRATOR_FILES.filter((f) => f.endsWith('.onnx')).reduce((a, f) => a + fs.statSync(path.join(dir, f)).size, 0) / 1e6).toFixed(1) };
  if (!deep) return res;
  ort = ort || (await import('onnxruntime-node'));
  const io = async (f) => { const s = await ort.InferenceSession.create(path.join(dir, f)); const r = { inputs: [...s.inputNames], outputs: [...s.outputNames] }; if (s.release) await s.release(); return r; };
  const dec = await io('onnx/decoder_model_merged_q4f16.onnx'), emb = await io('onnx/embed_tokens_fp16.onnx'), vis = await io('onnx/vision_encoder_quantized.onnx');
  const need = [[dec.inputs, ['inputs_embeds', 'attention_mask', 'past_key_values.0.key']], [dec.outputs, ['logits', 'present.0.key']], [emb.inputs, ['input_ids']], [vis.inputs, ['pixel_values']]];
  for (const [have, names] of need) for (const n of names) if (!have.includes(n)) throw new Error(`Narrator: an ONNX file lacks ${n} (the transformers.js contract)`);
  return { ...res, decoder_inputs: dec.inputs.length, decoder_outputs: dec.outputs.length };
}
// move a staged folder into place, keeping any older one as <name>.prev-<time>
function place(from, to) {
  let prev = null;
  if (fs.existsSync(to)) { prev = `${to}.prev-${stamp()}`; fs.renameSync(to, prev); }
  fs.renameSync(from, to); return prev;
}
export async function install(zipFile, { models = MODELS, deep = true, log = console.log } = {}) {
  if (!fs.existsSync(zipFile)) throw new Error(`${zipFile} not found`);
  fs.mkdirSync(models, { recursive: true });
  const stage = path.join(models, `.install-${stamp()}`); fs.mkdirSync(stage);
  let man, eye, nar;
  try {
    man = unpack(zipFile, stage); log(`unpacked ${Object.keys(man.files).length} files; every SHA-256 matches MANIFEST.json`);
    eye = await checkEye(path.join(stage, man.eye)); log(`Pilot Eye: parity in Node (onnxruntime-web) max |d| ${eye.max_abs_diff.toExponential(2)} over ${eye.heads} heads at ${eye.size}`);
    nar = await checkNarrator(path.join(stage, man.narrator), { deep }); log(`Narrator: ${nar.files} files, ${nar.mb} MB of ONNX, configs ok${deep ? ', sessions load with the transformers.js names' : ''}`);
  } catch (e) { fs.rmSync(stage, { recursive: true, force: true }); throw e; }
  const cur = path.join(models, 'current.json'), before = fs.existsSync(cur) ? readJson(cur) : null;
  const prevEye = place(path.join(stage, man.eye), path.join(models, man.eye)), prevNar = place(path.join(stage, man.narrator), path.join(models, man.narrator));
  if (fs.existsSync(path.join(stage, 'metrics'))) place(path.join(stage, 'metrics'), path.join(models, man.narrator, 'metrics'));
  fs.renameSync(path.join(stage, 'MANIFEST.json'), path.join(models, man.narrator, 'MANIFEST.json'));
  const current = { eye: man.eye, narrator: man.narrator, installed: new Date().toISOString(), zip: path.basename(zipFile), zip_sha256: sha(zipFile), dataset: man.dataset ?? null,
    checks: { eye, narrator: nar }, previous: before ? { eye: before.eye, narrator: before.narrator } : null, replaced: { eye: prevEye, narrator: prevNar } };
  fs.writeFileSync(cur, JSON.stringify(current, null, 1)); fs.rmSync(stage, { recursive: true, force: true });
  log(`installed: models/${man.eye} and models/${man.narrator}; models/current.json now points at them. Start the site with: node vlm/web/serve.mjs --port 8795`);
  return current;
}
if (process.argv[1] && pathToFileURL(fs.realpathSync(process.argv[1])).href === import.meta.url) {
  const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; }, zip = process.argv[2];
  if (!zip || zip.startsWith('--')) { console.error('usage: node vlm/tools/install_models.mjs <apv-models-v1.zip> [--models-dir <folder>] [--no-deep]'); process.exit(2); }
  try { await install(path.resolve(zip), { models: arg('models-dir', MODELS), deep: !process.argv.includes('--no-deep') }); }
  catch (e) { console.error(`install failed, nothing was switched: ${e.message}`); process.exit(1); }
}
