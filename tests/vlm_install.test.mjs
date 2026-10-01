// The seamless install (V1-9): vlm/tools/install_models.mjs checks the Colab zip against its MANIFEST, re-runs the Pilot Eye
// parity sample in Node (onnxruntime-web), checks the Narrator files and configs, moves the folders into place (an older
// one is kept as .prev-<time>) and writes models/current.json. The end-to-end case zips the v0 models with the notebook's
// own Python zipper (a fake v1 zip) and installs it into a temporary models folder.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { install, nodeInput, checkNarrator, unpack } from '../vlm/tools/install_models.mjs';

const require = createRequire(new URL('../vlm/package.json', import.meta.url)), AdmZip = require('adm-zip');
const V0EYE = '/Volumes/LaCie/astro-pilot/vlm/models/pilot-eye-v0/export', V0NAR = '/Volumes/LaCie/astro-pilot/vlm/models/narrator-v0';
const PY = '/Volumes/LaCie/astro-pilot/vlm/venv/bin/python', ROOT = fileURLToPath(new URL('..', import.meta.url));
const tmpRoot = fs.existsSync('/Volumes/LaCie/astro-pilot/vlm/tmp') ? '/Volumes/LaCie/astro-pilot/vlm/tmp' : os.tmpdir();
const tmp = (p) => fs.mkdtempSync(path.join(tmpRoot, p));
const NAR_FILES = { 'config.json': '{"model_type":"idefics3"}', 'tokenizer.json': '{}', 'tokenizer_config.json': '{}', 'chat_template.json': '{"chat_template":"x"}',
  'preprocessor_config.json': '{"do_resize":false,"do_image_splitting":false,"max_image_size":{"longest_edge":512}}', 'processor_config.json': '{}', 'generation_config.json': '{"eos_token_id":49279}',
  'onnx/decoder_model_merged_q4f16.onnx': 'x', 'onnx/embed_tokens_fp16.onnx': 'x', 'onnx/vision_encoder_quantized.onnx': 'x' };

test('the parity input is the formula the Python export uses: byte k of frame f is imul(k + f*n, 2654435761) >>> 24 over 255', () => {
  const a = nodeInput(4, 0), b = nodeInput(4, 1); assert.equal(a[1], Math.fround(158 / 255)); assert.equal(a[0], 0); assert.equal(b[0], Math.fround((Math.imul(4, 2654435761) >>> 24) / 255));
});
test('narrator check: the browser configs (no resize, 512, eos 49279, a chat template) and every deploy file', async () => {
  const d = tmp('apv-nc-'); for (const [f, s] of Object.entries(NAR_FILES)) { fs.mkdirSync(path.dirname(path.join(d, f)), { recursive: true }); fs.writeFileSync(path.join(d, f), s); }
  assert.equal((await checkNarrator(d, { deep: false })).files, 10);
  fs.writeFileSync(path.join(d, 'preprocessor_config.json'), '{"do_resize":true}'); await assert.rejects(checkNarrator(d, { deep: false }), /do_resize false/);
  fs.rmSync(path.join(d, 'onnx/embed_tokens_fp16.onnx')); await assert.rejects(checkNarrator(d, { deep: false }), /embed_tokens_fp16/);
  fs.rmSync(d, { recursive: true, force: true });
});
test('unpack refuses a zip whose files do not match MANIFEST.json (a truncated or edited download)', () => {
  const d = tmp('apv-uz-'), z = new AdmZip(); z.addFile('pilot-eye-v1/a.json', Buffer.from('{}'));
  z.addFile('MANIFEST.json', Buffer.from(JSON.stringify({ format: 'apv-models/1', eye: 'pilot-eye-v1', narrator: 'narrator-v1', files: { 'pilot-eye-v1/a.json': { sha256: '0'.repeat(64), bytes: 2 } } })));
  z.writeZip(path.join(d, 'm.zip')); assert.throws(() => unpack(path.join(d, 'm.zip'), path.join(d, 'x')), /does not match its manifest/);
  const y = new AdmZip(); y.addFile('a.txt', Buffer.from('a')); y.writeZip(path.join(d, 'n.zip')); assert.throws(() => unpack(path.join(d, 'n.zip'), path.join(d, 'y')), /no MANIFEST/);
  fs.rmSync(d, { recursive: true, force: true });
});
// about 250 MB zipped, hashed and unpacked: opt in with APV_E2E=1 (1-10 min on the USB disk)
const ready = process.env.APV_E2E === '1' && fs.existsSync(`${V0EYE}/encoder.onnx`) && fs.existsSync(`${V0NAR}/onnx/decoder_model_merged_q4f16.onnx`) && fs.existsSync(PY);
test('end to end: a fake v1 zip made from the v0 models by the notebook\'s zipper installs, passes the Node parity and the ONNX sessions, and switches current.json', { skip: !ready && 'set APV_E2E=1 (needs the v0 models and the LaCie venv)', timeout: 1200000 }, async () => {
  const d = tmp('apv-inst-'), eye = path.join(d, 'pilot-eye-v1'), zip = path.join(d, 'apv-models-v1.zip'), models = path.join(d, 'models');
  fs.mkdirSync(eye); for (const f of fs.readdirSync(V0EYE).filter((x) => !x.startsWith('._'))) fs.copyFileSync(path.join(V0EYE, f), path.join(eye, f));
  const code = `from vlm.train.colab import eye, package\neye.node_parity_sample(${JSON.stringify(eye)})\npackage.make_zip(${JSON.stringify(zip)}, ${JSON.stringify(eye)}, ${JSON.stringify(V0NAR)}, {'note': {'fake': 'v0 models'}}, dataset_id='fake-v1')\n`;
  const py = spawnSync(PY, ['-c', code], { cwd: ROOT, encoding: 'utf8', env: { ...process.env, HF_HUB_OFFLINE: '1' } }); assert.equal(py.status, 0, py.stderr);
  fs.mkdirSync(path.join(models, 'pilot-eye-v1'), { recursive: true }); fs.writeFileSync(path.join(models, 'current.json'), JSON.stringify({ eye: 'pilot-eye-v0/export', narrator: 'narrator-v0' }));
  const said = [], cur = await install(zip, { models, log: (s) => said.push(s) });
  assert.deepEqual([cur.eye, cur.narrator, cur.dataset], ['pilot-eye-v1', 'narrator-v1', 'fake-v1']); assert.ok(cur.checks.eye.max_abs_diff <= 1e-3, `${cur.checks.eye.max_abs_diff}`);
  assert.equal(cur.checks.eye.heads, 8); assert.ok(cur.checks.narrator.decoder_inputs > 3, 'the fused decoder session loaded');
  assert.deepEqual(JSON.parse(fs.readFileSync(path.join(models, 'current.json'), 'utf8')).previous, { eye: 'pilot-eye-v0/export', narrator: 'narrator-v0' });
  assert.ok(fs.existsSync(path.join(models, 'narrator-v1/onnx/vision_encoder_quantized.onnx')) && fs.existsSync(path.join(models, 'narrator-v1/metrics/note.json')));
  assert.ok(!fs.existsSync(path.join(models, 'narrator-v1/onnx/decoder_model_merged.onnx')), 'the fp32 reference stays out of the zip');
  assert.ok(cur.replaced.eye && fs.existsSync(cur.replaced.eye), 'the older pilot-eye-v1 folder is kept, renamed');
  assert.ok(!fs.readdirSync(models).some((f) => f.startsWith('.install-')), 'no staging folder left');
  assert.match(said.join('\n'), /parity in Node/);
  fs.rmSync(d, { recursive: true, force: true });
});
