// chat/web/capcom.js — CAPCOM in the page (spec §7, R7): transformers.js 4.3 (injected as `tf`), BM25 notes from kb.json (retriever.js,
// kb.retrieval.min_score), the prompt contract (prompt.js) rendered by the tokenizer's chat template, greedy decoding with repetition
// penalty 1.1 until the end-of-turn token, token streaming, and a serial request queue where a new ask cancels the running one.
// Runtime ladder (ruling 2026-10-01): WebGPU + shader-f16 → q4f16; WebGPU → q4; then retrieval-only answer cards ('auto' falls through on
// a failed load or a degenerate warm-up). A WASM rung exists only when the model folder declares a WASM-safe graph (stock LFM2 int4
// graphs fail or crawl on ORT-web WASM). Without a runtime, info.ready is false and ask() answers from retrieval alone (the top note,
// or a deflection that leads to an unseen highlight).
import { BM25, TOP_K, MIN_SCORE } from './retriever.js';
import { messages, prevUser, unseen } from './prompt.js';
export const REPETITION_PENALTY = 1.1, MAX_NEW_TOKENS = 120;
const now = () => performance.now();
const errText = (e) => String(e && e.message ? e.message : e);

// the WebGPU adapter's facts: null without WebGPU; else { f16, vendor, architecture, description }
export async function probeGpu() {
  const gpu = globalThis.navigator && navigator.gpu;
  if (!gpu) return null;
  try {
    const a = await gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!a) return null;
    const i = a.info || {};
    return { f16: a.features.has('shader-f16'), vendor: i.vendor || '', architecture: i.architecture || '', description: i.description || '' };
  } catch { return null; }
}

// the [device, dtype] rungs to try, in order. 'auto': WebGPU q4f16 (shader-f16) → WebGPU q4 → the folder's WASM-safe graph only when its
// config.json declares one ({"capcom": {"wasm": "q4"}}, chat/web/tools/wasm_rewrite.py) → none (retrieval-only cards). 'none': cards only.
export function ladder(device = 'auto', dtype = null, gpu = null, wasm = null) {
  const best = gpu && gpu.f16 ? 'q4f16' : 'q4';
  if (device === 'none') return [];
  if (device === 'webgpu') return [['webgpu', dtype || best]];
  if (device !== 'auto') return [[device, dtype || wasm || 'q4']]; // 'wasm' (explicit), or 'cpu' under Node
  const l = gpu ? (dtype ? [['webgpu', dtype]] : best === 'q4f16' ? [['webgpu', 'q4f16'], ['webgpu', 'q4']] : [['webgpu', 'q4']]) : [];
  return wasm ? [...l, ['wasm', wasm]] : l;
}

// the answer without a model: the top note, or a deflection that leads to the first unseen highlight
export function retrievalAnswer(hits, byId, state = null) {
  if (hits.length) return byId.get(hits[0][0]).text;
  const next = unseen(state ? state.scene : null, state ? state.seen || [] : [])[0];
  return `That's not in my flight notes.${next ? ` Want to see ${next} next?` : ''}`;
}

// kb (kb.json object or URL) → the retriever and retrieval-only answer cards (usable while the model is still downloading)
export async function loadKb(kb) {
  const kbj = typeof kb === 'string' || kb instanceof URL ? await (await fetch(kb)).json() : kb;
  const bm = new BM25(kbj.facts), minScore = kbj.retrieval && kbj.retrieval.min_score != null ? kbj.retrieval.min_score : MIN_SCORE;
  const retrieve = (user, history = []) => bm.search(user, prevUser(history), TOP_K, minScore);
  const card = (user, { history = [], state = null } = {}) => {
    const hits = retrieve(user, history);
    return { text: retrievalAnswer(hits, bm.byId, state), notes: hits.map(([id]) => id), scores: hits.map(([, sc]) => sc), fallback: true };
  };
  return { kb: kbj, bm, minScore, retrieve, card };
}

// a model folder URL/path → where transformers.js should look: same origin → a root-relative localModelPath (4.3's get_file_metadata
// skips a local path that is an absolute http(s) URL, so the tokenizer "does not exist"); another origin → remoteHost + '{model}/';
// anything else is a Hub repo id (remote, default host)
export function modelSource(model, base = globalThis.location ? location.href : 'http://localhost/') {
  const m = String(model).replace(/\/+$/, '');
  if (!/^([a-z]+:)?\/\/|^\.{0,2}\//i.test(m)) return { modelId: m };
  const u = new URL(m, base), dir = u.pathname.slice(0, u.pathname.lastIndexOf('/') + 1), modelId = u.pathname.slice(dir.length);
  return u.origin === new URL(base).origin ? { localModelPath: dir, modelId } : { remoteHost: u.origin + dir, modelId };
}

export async function createCapcom({ tf, modelId, localModelPath = null, remoteHost = null, kb, device = 'auto', dtype = null, onProgress = null, warmup: doWarmup = true }) {
  const say = (e) => { try { onProgress && onProgress(e); } catch { /* the page's problem */ } };
  const { bm, minScore, retrieve } = await loadKb(kb);
  if (localModelPath) Object.assign(tf.env, { localModelPath, allowRemoteModels: false, allowLocalModels: true });
  else if (remoteHost) Object.assign(tf.env, { remoteHost, remotePathTemplate: '{model}/', allowRemoteModels: true, allowLocalModels: false });
  const gpu = await probeGpu(), info = { device: null, dtype: null, loadMs: null, warmupMs: null, ready: false, error: null, attempts: [], gpu, minScore };
  let tok = null, model = null;
  let files = {};
  const progress = (p) => {
    if (p.status === 'progress' && p.total) {
      files[p.file] = [p.loaded, p.total];
      const v = Object.values(files), loaded = v.reduce((a, x) => a + x[0], 0), total = v.reduce((a, x) => a + x[1], 0);
      say({ status: 'download', file: p.file, loaded, total, progress: total ? loaded / total : 0 });
    }
  };
  const t0 = now();
  let wasm = null; // the folder's own declaration of a WASM-safe graph (stock exports have none: GatherBlockQuantized, slow MatMulNBits)
  if (device === 'auto') try { wasm = ((await tf.AutoConfig.from_pretrained(modelId)) || {}).capcom?.wasm || null; } catch { /* no config: no WASM rung */ }
  info.wasm = wasm;
  for (const [dev, dt] of ladder(device, dtype, gpu, wasm)) {
    const t1 = now(); files = {}; say({ status: 'load', device: dev, dtype: dt });
    try {
      tok ||= await tf.AutoTokenizer.from_pretrained(modelId, { progress_callback: progress });
      // WASM: let ORT constant-fold DequantizeLinear weights (a no-op for MatMulNBits graphs; ORT-web's WASM MatMulNBits re-dequantizes
      // its weight every call, ~1 tok/s, so a WASM export should use DequantizeLinear+MatMul — chat/web/tools/wasm_rewrite.py --dq-matmul)
      const session_options = dev === 'wasm' ? { extra: { session: { disable_quant_qdq: '1' } } } : undefined;
      model = await tf.AutoModelForCausalLM.from_pretrained(modelId, { device: dev, dtype: dt, progress_callback: progress, session_options });
      Object.assign(info, { device: dev, dtype: dt, loadMs: Math.round(now() - t0) });
      if (doWarmup) { say({ status: 'warmup', device: dev, dtype: dt }); await warm(); }
      info.ready = true; info.attempts.push({ device: dev, dtype: dt, ok: true, ms: Math.round(now() - t1) });
      break;
    } catch (e) {
      info.attempts.push({ device: dev, dtype: dt, ok: false, ms: Math.round(now() - t1), error: errText(e) });
      say({ status: 'fallback', device: dev, dtype: dt, error: errText(e) });
      if (model) { try { await model.dispose(); } catch { /* already gone */ } model = null; }
    }
  }
  if (!info.ready) { info.device = info.dtype = null; info.error = info.attempts.map((a) => `${a.device}/${a.dtype}: ${a.error}`).join(' | ') || 'no runtime'; }

  // a 4-token greedy generate: compiles the shaders and rejects a graph whose logits are NaN (argmax 0 = <|pad|> every step)
  async function warm() {
    const t = now(), text = tok.apply_chat_template([{ role: 'user', content: 'Hi' }], { add_generation_prompt: true, tokenize: false });
    const inputs = tok(text, { add_special_tokens: false }), n = inputs.input_ids.dims.at(-1);
    const out = await model.generate({ ...inputs, max_new_tokens: 4, do_sample: false });
    const ids = Array.from(out.data, Number).slice(n);
    info.warmupMs = Math.round(now() - t);
    if (ids.length && ids.every((x) => x === 0)) throw new Error(`degenerate warm-up output ${JSON.stringify(ids)} (NaN logits?)`);
  }
  const template = (hits, user, history, state) =>
    tok.apply_chat_template(messages(history, user, hits.map(([id]) => bm.byId.get(id).text), state), { add_generation_prompt: true, tokenize: false });
  // the exact text the model sees (the chat template over the prompt contract) and the [id, score] notes behind it
  const render = (user, { history = [], state = null } = {}) => { const hits = retrieve(user, history); return { hits, text: template(hits, user, history, state) }; };

  async function run(req, user, { history = [], state = null, onToken = null, maxNewTokens = MAX_NEW_TOKENS } = {}) {
    const t0 = now(), hits = retrieve(user, history), notes = hits.map(([id]) => id);
    const base = { notes, scores: hits.map(([, s]) => s), device: info.device, dtype: info.dtype };
    if (req.aborted) return { ...base, text: '', ttftMs: null, ms: 0, tokens: 0, tokPerSec: null, aborted: true };
    if (!info.ready) {
      const text = retrievalAnswer(hits, bm.byId, state);
      if (onToken) onToken(text);
      return { ...base, text, fallback: true, ttftMs: 0, ms: now() - t0, tokens: 0, tokPerSec: null, aborted: false };
    }
    const inputs = tok(template(hits, user, history, state), { add_special_tokens: false }), n = inputs.input_ids.dims.at(-1);
    let first = null, last = null, count = 0;
    const streamer = new tf.TextStreamer(tok, { skip_prompt: true, skip_special_tokens: true,
      callback_function: (x) => { if (onToken && !req.aborted) onToken(x); },
      token_callback_function: () => { last = now(); if (first === null) first = last; count++; } });
    const stop = new tf.InterruptableStoppingCriteria(); req.stop = stop;
    if (req.aborted) stop.interrupt();
    const out = await model.generate({ ...inputs, max_new_tokens: maxNewTokens, do_sample: false, repetition_penalty: REPETITION_PENALTY, streamer, stopping_criteria: stop });
    const ids = Array.from(out.data, Number).slice(n), end = now();
    const text = tok.decode(ids, { skip_special_tokens: true }).trim();
    return { ...base, text, ids, promptTokens: n, ttftMs: first === null ? null : first - t0, ms: end - t0, tokens: ids.length,
      tokPerSec: count > 1 ? ((count - 1) * 1000) / (last - first) : null, aborted: !!req.aborted };
  }

  // the serial queue: every ask waits for the previous one, and a new ask cancels whatever is running or waiting
  let chain = Promise.resolve(), live = [];
  const cancel = (req) => { req.aborted = true; if (req.stop) req.stop.interrupt(); };
  function ask(user, opts = {}) {
    for (const r of live) cancel(r);
    const req = { aborted: false, stop: null };
    live.push(req);
    if (opts.signal) { if (opts.signal.aborted) cancel(req); else opts.signal.addEventListener('abort', () => cancel(req), { once: true }); }
    const p = chain.then(() => run(req, user, opts)).finally(() => { live = live.filter((r) => r !== req); });
    chain = p.catch(() => {});
    return p;
  }
  return {
    info, ask, retrieve, render, tokenizer: tok, model,
    abort: () => { for (const r of live) cancel(r); },
    warmup: async () => { if (info.ready) await (chain = chain.then(warm).catch(() => {})); },
    dispose: async () => { for (const r of live) cancel(r); await chain; if (model) await model.dispose(); model = null; info.ready = false; },
  };
}
