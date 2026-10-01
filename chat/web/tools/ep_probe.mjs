// chat/web/tools/ep_probe.mjs — which execution provider runs each node of a CAPCOM graph in headless Chrome (ORT-web 1.30 through
// transformers.js 4.3, main thread, ORT log level verbose): prints the placement / fallback / Memcpy lines and any contrib-op errors.
//   node chat/web/tools/ep_probe.mjs [--model /__capcom/models/lfm350-base-web] [--device webgpu] [--dtype q4f16] [--out <log>]
import fs from 'node:fs';
import { startServer } from '../serve.mjs';
const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
const PW = process.env.PLAYWRIGHT || '/Volumes/LaCie/astro-pilot/test/node_modules/playwright/index.mjs';
const model = arg('model', '/__capcom/models/lfm350-base-web'), device = arg('device', 'webgpu'), dtype = arg('dtype', 'q4f16'), out = arg('out', null);
const { chromium } = await import(PW), srv = await startServer({ port: 0 });
const browser = await chromium.launch({ channel: 'chromium', headless: true, args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--use-angle=metal'] });
const lines = [];
try {
  const page = await browser.newPage();
  page.on('console', (m) => lines.push(`[${m.type()}] ${m.text()}`));
  page.on('pageerror', (e) => lines.push(`[pageerror] ${e.message}`));
  await page.route('**/__probe.html', (r) => r.fulfill({ contentType: 'text/html', body: '<!doctype html><meta charset="utf-8">' }));
  await page.goto(`http://127.0.0.1:${srv.port}/__probe.html`);
  const r = await page.evaluate(async ({ model, device, dtype }) => {
    const tf = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0'), { modelSource } = await import('/chat/web/capcom.js');
    const src = modelSource(model); Object.assign(tf.env, { localModelPath: src.localModelPath, allowRemoteModels: false, allowLocalModels: true });
    tf.env.backends.onnx.logLevel = 'verbose';
    const t0 = performance.now();
    try {
      const tok = await tf.AutoTokenizer.from_pretrained(src.modelId);
      const m = await tf.AutoModelForCausalLM.from_pretrained(src.modelId, { device, dtype, session_options: { logSeverityLevel: 0, logVerbosityLevel: 0 } });
      const load = performance.now() - t0, inputs = tok('<|startoftext|><|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n', { add_special_tokens: false });
      const out = await m.generate({ ...inputs, max_new_tokens: 8, do_sample: false });
      const text = tok.decode(Array.from(out.data, Number).slice(inputs.input_ids.dims.at(-1)));
      await m.dispose(); return { ok: true, load, text };
    } catch (e) { return { ok: false, error: String(e && e.message ? e.message : e) }; }
  }, { model, device, dtype });
  const keep = lines.filter((l) => /placed on|not assigned|CPUExecutionProvider|Memcpy|CausalConv|fallback|Unsupported|not supported|error|NaN/i.test(l));
  console.log(JSON.stringify(r)); console.log(`${lines.length} console lines; relevant:\n  ${keep.slice(0, 60).map((l) => l.slice(0, 400)).join('\n  ')}`);
  if (out) fs.writeFileSync(out, lines.join('\n'));
} finally { await browser.close(); await srv.close(); }
