// chat/web/bench.mjs — CAPCOM's browser bench (spec §7): ONE headless Chrome (WebGPU flags as test/shot.mjs) drives the real worker
// (capcom-client.js → capcom-worker.js → capcom.js) per config, one page at a time: load (wall / model / warm-up), then N visitor
// prompts with realistic retrieval and state lines → TTFT (first, warm median, p90), decode tok/s (median, p10, p90), adapter +
// shader-f16; then greedy parity: chat/web/ref_outputs.py re-tokenizes the same prompt texts (ids must match), re-renders the
// messages with the Python chat template (text must match) and greedy-decodes the q4 graph on ORT CPU from the page's ids — the
// first --parity-tokens generated ids are compared per config. Report: /Volumes/LaCie/astro-pilot/chat/logs/bench_<name>.json.
//   node chat/web/bench.mjs run  [--model /__capcom/models/lfm350-base-web] [--name base] [--configs webgpu:q4f16,wasm:q4] [--n 12]
//                                [--tokens 120] [--prompts <jsonl>] [--isolate] [--parity-tokens 32] [--ref-dtypes q4] [--no-parity]
//                                [--torch-src <hf dir>]   (adds PyTorch fp32 greedy + teacher-forced agreement: the gold reference)
//                                [--ref-model-dir <web folder>]   (the ORT CPU reference graph; default: the benched folder)
//   node chat/web/bench.mjs demo [--device webgpu|wasm|none] [--model …] [--asks "q1|q2|…"] [--out <png>]   (a demo.html screenshot)
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { startServer, CHAT_ROOT } from './serve.mjs';
import { HIGHLIGHTS } from './prompt.js';
const PW = process.env.PLAYWRIGHT || '/Volumes/LaCie/astro-pilot/test/node_modules/playwright/index.mjs';
const PY = process.env.PYTHON || `${CHAT_ROOT}/venv/bin/python`, LOGS = `${CHAT_ROOT}/logs`, HERE = fileURLToPath(new URL('.', import.meta.url));
const ARGS = ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--use-angle=metal', '--enable-gpu', '--disable-background-timer-throttling'];
const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
export const BUILTIN = ['How does it learn to dodge asteroids?', 'what am i looking at right now', 'How fast does the space station orbit Earth?',
  'is this real physics or just a game?', 'How do I try the airliner landing?', 'who flies the ship in the docking scenario?', 'What is PPO?',
  'can i fly it myself', 'why do the comets have two tails', 'How do I zoom in on Earth?', 'what does the Narrator do', 'does it train in my browser??'];
export const q = (a, p) => { if (!a.length) return null; const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.max(0, Math.round(p * (s.length - 1))))]; };
const r1 = (x) => (x == null ? null : Math.round(x * 10) / 10);

// visitor questions: --prompts, else the eval sets when present, else BUILTIN; fields q | user | question | prompt (string or messages)
export function loadPrompts(file, n) {
  const cands = file ? [file] : [`${CHAT_ROOT}/data/eval/single_raw.jsonl`, `${CHAT_ROOT}/data/capcom-dev/eval/single.jsonl`, `${CHAT_ROOT}/kb/questions.eval.jsonl`];
  for (const f of cands) {
    if (!fs.existsSync(f) || !fs.statSync(f).size) continue;
    const qs = fs.readFileSync(f, 'utf8').split('\n').filter((l) => l.trim()).map((l) => JSON.parse(l)).map((o) => o.user ?? o.q ?? o.question ??
      (typeof o.prompt === 'string' ? o.prompt : Array.isArray(o.prompt) ? o.prompt.filter((m) => m.role === 'user').at(-1)?.content : null)).filter(Boolean);
    if (qs.length) { const k = Math.max(1, Math.floor(qs.length / n)); return { source: f, prompts: qs.filter((_, i) => i % k === 0).slice(0, n) }; }
  }
  return { source: 'builtin', prompts: BUILTIN.slice(0, n) };
}

// first-k greedy agreement: shared prefix length per prompt, prompts matching all compared tokens, position-wise agreement
export function parity(js, py, k) {
  const rows = js.map((a, i) => {
    const b = py[i], x = a.slice(0, k), y = b.slice(0, k), m = Math.min(x.length, y.length);
    let p = 0; while (p < m && x[p] === y[p]) p++;
    let same = 0; for (let j = 0; j < m; j++) same += x[j] === y[j];
    return { prefix: p, compared: m, full: p === m && x.length === y.length, same };
  });
  const n = rows.reduce((s, r) => s + r.compared, 0);
  return { k, prompts: rows.length, full_match: rows.filter((r) => r.full).length, mean_prefix: r1(rows.reduce((s, r) => s + r.prefix, 0) / Math.max(1, rows.length)),
    token_agreement: n ? Math.round((1000 * rows.reduce((s, r) => s + r.same, 0)) / n) / 1000 : null, prefixes: rows.map((r) => r.prefix) };
}

async function launch(chromium) {
  const browser = await chromium.launch({ channel: 'chromium', headless: true, args: ARGS });
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 860 }, deviceScaleFactor: 2 });
  return { browser, ctx };
}
const blank = async (ctx, base) => {
  const page = await ctx.newPage(), logs = [];
  page.on('console', (m) => { if (['error', 'warning'].includes(m.type())) logs.push(`[${m.type()}] ${m.text()}`.slice(0, 600)); });
  page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`.slice(0, 600)));
  await page.route('**/__bench.html', (r) => r.fulfill({ contentType: 'text/html', body: '<!doctype html><meta charset="utf-8"><title>CAPCOM bench</title>' }));
  await page.goto(`${base}/__bench.html`);
  return { page, logs };
};

// one config in a fresh page: load, then every prompt (render + ask) through the worker
async function runConfig(ctx, base, { model, kb, device, dtype, prompts, tokens }) {
  const { page, logs } = await blank(ctx, base);
  try {
    return await page.evaluate(async ({ model, kb, device, dtype, prompts, tokens, scenes }) => {
      const { createCapcomClient } = await import('/chat/web/capcom-client.js');
      const t0 = performance.now(), phases = [];
      const c = createCapcomClient({ workerUrl: '/chat/web/capcom-worker.js', onProgress: (p) => { if (p.status !== 'download') phases.push({ ...p, t: Math.round(performance.now() - t0) }); } });
      const info = await c.load({ model, kb, device, dtype }), loadWallMs = Math.round(performance.now() - t0), rows = [];
      if (info.ready) {
        for (const [i, user] of prompts.entries()) {
          const state = { scene: scenes[i % scenes.length], seen: [] }, rd = await c.render(user, { state });
          const r = await c.ask(user, { state, maxNewTokens: tokens });
          rows.push({ user, scene: state.scene, text: rd.text, ids: rd.ids, messages: rd.messages, out: r.ids, answer: r.text, notes: r.notes,
            ttftMs: r.ttftMs, ms: r.ms, tokens: r.tokens, tokPerSec: r.tokPerSec, promptTokens: r.promptTokens });
        }
      }
      c.terminate();
      return { info, loadWallMs, phases, rows, coi: self.crossOriginIsolated, cores: navigator.hardwareConcurrency };
    }, { model, kb, device, dtype, prompts, tokens, scenes: HIGHLIGHTS.map(([k]) => k) });
  } finally { await page.close(); if (logs.length) console.log(`console (${device}/${dtype}):\n  ${logs.slice(0, 20).join('\n  ')}`); }
}

function summarize(res) {
  const rows = res.rows, ttft = rows.map((r) => r.ttftMs).filter((x) => x != null), warm = ttft.slice(1);
  const tps = rows.filter((r) => r.tokens >= 8 && r.tokPerSec).map((r) => r.tokPerSec);
  return { device: res.info.device, dtype: res.info.dtype, ready: res.info.ready, error: res.info.error, attempts: res.info.attempts, gpu: res.info.gpu,
    cross_origin_isolated: res.coi, load_wall_ms: res.loadWallMs, model_load_ms: res.info.loadMs, warmup_ms: res.info.warmupMs, phases: res.phases,
    prompts: rows.length, prompt_tokens_median: q(rows.map((r) => r.promptTokens), 0.5), out_tokens_median: q(rows.map((r) => r.tokens), 0.5),
    ttft_first_ms: r1(ttft[0]), ttft_warm_median_ms: r1(q(warm, 0.5)), ttft_warm_p90_ms: r1(q(warm, 0.9)),
    decode_tok_s_median: r1(q(tps, 0.5)), decode_tok_s_p10: r1(q(tps, 0.1)), decode_tok_s_p90: r1(q(tps, 0.9)),
    answer_ms_median: r1(q(rows.map((r) => r.ms), 0.5)) };
}

function reference(modelDir, rows, dtype, n, cands, torchSrc) {
  const inp = path.join(LOGS, `.bench_ref_in_${dtype}.json`), out = path.join(LOGS, `.bench_ref_out_${dtype}.json`);
  fs.writeFileSync(inp, JSON.stringify(rows.map((r, i) => ({ text: r.text, ids: r.ids, messages: r.messages, cands: Object.fromEntries(Object.entries(cands).map(([k, v]) => [k, v[i]])) }))));
  const extra = torchSrc ? ['--torch-src', torchSrc] : [];
  execFileSync(PY, [path.join(HERE, 'ref_outputs.py'), '--model-dir', modelDir, '--prompts', inp, '--out', out, '--dtype', dtype, '--n', String(n), ...extra], { stdio: 'inherit' });
  const ref = JSON.parse(fs.readFileSync(out, 'utf8')); fs.rmSync(inp); fs.rmSync(out);
  return ref;
}
// teacher-forced agreement with fp32: summed [agree, n] over prompts per candidate
const tfAgree = (rows) => { const t = {}; for (const r of rows) for (const [k, [a, n]] of Object.entries(r.tf || {})) { t[k] ||= [0, 0]; t[k][0] += a; t[k][1] += n; } return Object.fromEntries(Object.entries(t).map(([k, [a, n]]) => [k, n ? Math.round((1000 * a) / n) / 1000 : null])); };

export async function benchRun({ model, name, configs, n, tokens, promptsFile, isolate, parityTokens, refDtypes, doParity, torchSrc, refModelDir = null }) {
  const { chromium } = await import(PW), srv = await startServer({ port: 0, isolate }), base = `http://127.0.0.1:${srv.port}`;
  const { source, prompts } = loadPrompts(promptsFile, n), kb = '/__capcom/kb/kb.json', results = {};
  const modelDir = model.startsWith('/__capcom/') ? path.join(CHAT_ROOT, model.slice('/__capcom/'.length)) : null, refDir = refModelDir || modelDir;
  let b = null;
  try {
    b = await launch(chromium);
    for (const cfg of configs) {
      const [device, dtype] = cfg.split(':'); console.log(`== ${device}/${dtype}: ${prompts.length} prompts from ${source}`);
      const res = await runConfig(b.ctx, base, { model, kb, device, dtype, prompts, tokens });
      results[cfg] = { raw: res, summary: summarize(res) };
      console.log(JSON.stringify({ ...results[cfg].summary, phases: undefined, attempts: res.info.attempts.map((a) => ({ ...a, error: a.error && a.error.slice(0, 300) })) }));
    }
  } finally { if (b) await b.browser.close(); await srv.close(); }
  const report = { name, model, date: new Date().toISOString(), prompts_source: source, n: prompts.length, max_new_tokens: tokens, isolate,
    sizes_mb: modelDir ? Object.fromEntries(fs.readdirSync(path.join(modelDir, 'onnx')).filter((f) => !f.startsWith('._')).map((f) => [f, r1(fs.statSync(path.join(modelDir, 'onnx', f)).size / 2 ** 20)])) : null,
    configs: {}, parity: {} };
  for (const [cfg, { summary }] of Object.entries(results)) report.configs[cfg] = summary;
  const ran = Object.entries(results).filter(([, r]) => r.raw.rows.length);
  if (doParity && refDir && ran.length) {
    const rows0 = ran[0][1].raw.rows;
    const same = ran.filter(([, r]) => r.raw.rows.every((x, i) => JSON.stringify(x.ids) === JSON.stringify(rows0[i].ids)));
    const cands = Object.fromEntries(same.map(([cfg, r]) => [cfg, r.raw.rows.map((x) => x.out)]));
    for (const [k, rd] of refDtypes.entries()) {
      let ref; try { ref = reference(refDir, rows0, rd, parityTokens, cands, k === 0 ? torchSrc : null); } catch (e) { report.parity[rd] = { error: String(e.message).slice(0, 500) }; continue; }
      const p = { reference: `ORT CPU ${rd} greedy (python) of ${refDir}`, eos: ref.eos };
      p.prompt_ids_match = rows0.filter((r, i) => JSON.stringify(r.ids) === JSON.stringify(ref.rows[i].py_ids)).length + `/${rows0.length}`;
      p.template_text_match = rows0.filter((r, i) => r.text === ref.rows[i].template).length + `/${rows0.length}`;
      p.ref_ms_median = q(ref.rows.map((r) => r.ms), 0.5);
      for (const [cfg, r] of same) p[cfg] = parity(r.raw.rows.map((x) => x.out), ref.rows.map((x) => x.gen), parityTokens);
      if (ref.rows[0].torch_gen) {
        const g = ref.rows.map((x) => x.torch_gen);
        p.vs_torch_fp32 = { note: 'shared-prefix parity vs PyTorch fp32 greedy (same prompt ids, same penalty)',
          ...Object.fromEntries([...same.map(([cfg, r]) => [cfg, r.raw.rows.map((x) => x.out)]), [`ort_cpu_${rd}`, ref.rows.map((x) => x.gen)]].map(([k2, v]) => [k2, parity(v, g, parityTokens)])) };
        p.teacher_forced_vs_torch_fp32 = tfAgree(ref.rows);
      }
      report.parity[rd] = p; console.log(`parity (ref ${rd}):`, JSON.stringify(p));
    }
  }
  for (const [cfg, r] of Object.entries(results)) report.configs[cfg].samples = r.raw.rows.map((x) => ({ user: x.user, scene: x.scene, notes: x.notes, prompt_tokens: x.promptTokens, tokens: x.tokens, ttft_ms: r1(x.ttftMs), tok_s: r1(x.tokPerSec), answer: x.answer }));
  const out = path.join(LOGS, `bench_${name}.json`); fs.writeFileSync(out, JSON.stringify(report, null, 1)); console.log('wrote', out);
  return report;
}

// the demo panel in one browser: ask each question in turn through the UI (Enter), wait for its debug line, then a tall screenshot
// (the card is expanded so every exchange shows) and the replies with their TTFT / tok/s
export async function benchDemo({ model, device, dtype, asks, out, debug = true }) {
  const { chromium } = await import(PW), srv = await startServer({ port: 0 });
  let b = null;
  try {
    b = await launch(chromium); const page = await b.ctx.newPage(), logs = [];
    page.on('console', (m) => { if (['error', 'warning'].includes(m.type())) logs.push(`[${m.type()}] ${m.text()}`.slice(0, 400)); });
    page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));
    const qs = new URLSearchParams({ model, device, ...(dtype ? { dtype } : {}), ...(debug ? { debug: '1' } : {}) });
    await page.goto(`http://127.0.0.1:${srv.port}/chat/web/demo.html?${qs}`);
    const info = await page.evaluate(() => window.capcomDemo.ready), replies = [];
    for (const [i, ask] of asks.entries()) {
      await page.fill('#q', ask); await page.press('#q', 'Enter');
      replies.push(await page.evaluate(async (n) => {
        const done = () => [...document.querySelectorAll('.meta')].filter((m) => m.textContent).length; // one filled debug line per answer
        for (let k = 0; k < 1200 && done() <= n; k++) await new Promise((ok) => setTimeout(ok, 100));
        return { reply: [...document.querySelectorAll('.msg.bot')].at(-1)?.textContent, meta: [...document.querySelectorAll('.meta')].at(-1)?.textContent };
      }, i));
    }
    await page.addStyleTag({ content: 'body{overflow:visible;height:auto;padding:24px 0}.card{height:auto!important}#log{overflow:visible}' });
    await page.waitForTimeout(500); await page.screenshot({ path: out, fullPage: true });
    if (logs.length) console.log(logs.slice(0, 20).join('\n'));
    return { info: { device: info.device, dtype: info.dtype, ready: info.ready, loadMs: info.loadMs, error: info.error }, replies: replies.map((r, i) => ({ ask: asks[i], ...r })), out };
  } finally { if (b) await b.browser.close(); await srv.close(); }
}

if (process.argv[1] && import.meta.url === pathToFileURL(fs.realpathSync(process.argv[1])).href) {
  const cmd = process.argv[2], model = arg('model', '/__capcom/models/lfm350-base-web');
  if (cmd === 'run') {
    await benchRun({ model, name: arg('name', path.basename(model)), configs: arg('configs', 'webgpu:q4f16,wasm:q4').split(','), n: +arg('n', 12), tokens: +arg('tokens', 120),
      promptsFile: arg('prompts', null), isolate: process.argv.includes('--isolate'), parityTokens: +arg('parity-tokens', 32), refDtypes: arg('ref-dtypes', 'q4').split(','),
      doParity: !process.argv.includes('--no-parity'), torchSrc: arg('torch-src', null), refModelDir: arg('ref-model-dir', null) });
  } else if (cmd === 'demo') {
    const device = arg('device', 'webgpu');
    console.log(JSON.stringify(await benchDemo({ model, device, dtype: arg('dtype', null), asks: arg('asks', 'How does it learn to dodge asteroids?').split('|'), out: arg('out', `${LOGS}/demo_${device}.png`) })));
  } else { console.log('usage: bench.mjs run|demo [--model …] (see the header)'); process.exit(2); }
  process.exit(0);
}
