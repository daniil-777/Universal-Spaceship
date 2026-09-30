// engine.js — the two on-device models behind one small interface for the card (real here, canned in mock.js):
//   Pilot Eye: vlm/web/pilot-eye-worker.js (ORT-web WASM, 6 MB), started when the Narrator opens; frames arrive as bitmaps.
//   Narrator:  vlm/web/narrator.js (transformers.js 4.3 from jsdelivr, SmolVLM v0, 244 MB of ONNX), loaded only after
//              the user agrees to the download; the browser's Cache Storage keeps it for later visits.
// Both come from /__vlm/models/ (node vlm/web/serve.mjs); ?eye= and ?narrator= pick other folders there.
import { createNarrator, contextLine, NARRATOR_TASKS } from '../../vlm/web/narrator.js';

const TF_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0';
const STORE = 'astro.narrator';
export const DOWNLOAD_MB = 244;
// the trained prompts: caption_detail, safety, and the VQA "What action is recommended?" (80 training rows)
export const TASKS = Object.freeze({ describe: NARRATOR_TASKS.describe, safety: NARRATOR_TASKS.safety, now: 'What action is recommended?' });
export const TOKENS = Object.freeze({ describe: 90, safety: 64, now: 48, ask: 56 });

export function config(search = location.search) {
  const q = new URLSearchParams(search), clean = (s, d) => (s && /^[\w./-]+$/.test(s) && !s.includes('..') ? s : d);
  return { base: '/__vlm/models/', eye: clean(q.get('eye'), 'pilot-eye-v0/export'), narrator: clean(q.get('narrator'), 'narrator-v0'), mock: q.get('narratorMock') === '1' };
}
// consent and download state, remembered per browser (a blocked storage just asks again)
export const memory = {
  get() { try { return JSON.parse(localStorage.getItem(STORE)) || {}; } catch { return {}; } },
  set(patch) { try { localStorage.setItem(STORE, JSON.stringify({ ...memory.get(), ...patch })); } catch { /* private mode */ } },
};
export const hasWebGPU = () => typeof navigator !== 'undefined' && !!navigator.gpu;

function bitmapImage(tf, bitmap) {
  const c = new OffscreenCanvas(bitmap.width, bitmap.height), g = c.getContext('2d', { willReadFrequently: true });
  g.drawImage(bitmap, 0, 0);
  return new tf.RawImage(g.getImageData(0, 0, c.width, c.height).data, c.width, c.height, 4).rgb();
}

export function createEngine(cfg = config()) {
  let labels = null, worker = null, onResult = null, nid = 0, tf = null, narrator = null, loading = null;
  const eyeUrl = (f) => `${cfg.base}${cfg.eye}/${f}`;
  return {
    mock: false,
    get device() { return narrator ? narrator.device : null; },
    get ready() { return !!narrator; },
    get loadMs() { return narrator ? narrator.loadMs : null; },
    async probe() {
      try { const r = await fetch(eyeUrl('labels.json'), { cache: 'no-store' }); if (!r.ok) return { ok: false }; labels = await r.json(); return { ok: true, labels }; } catch { return { ok: false }; }
    },
    startEye(cb) {
      onResult = cb;
      if (worker || !labels) return;
      worker = new Worker(new URL('../../vlm/web/pilot-eye-worker.js', import.meta.url), { type: 'module' });
      worker.onmessage = (e) => { const m = e.data; if (onResult) onResult(m.type === 'result' ? m : m.type === 'error' ? { status: 'error', message: m.message, id: m.id } : { status: m.type }); };
      worker.onerror = (e) => onResult && onResult({ status: 'error', message: e.message || 'worker failed' });
      worker.postMessage({ type: 'init', encoderUrl: eyeUrl('encoder.onnx'), headsUrl: eyeUrl('heads.onnx'), labels });
    },
    nominal: () => (labels && labels.nominal_frame_dt) || null,
    postFrame(f) { if (worker) worker.postMessage({ type: 'frame', id: nid++, ...f }, [f.bitmap]); else f.bitmap.close(); },
    stopEye() { if (worker) worker.terminate(); worker = null; onResult = null; },
    // resolves once the Narrator is in memory; onProgress({loaded, total}) in bytes while it downloads or reads the cache
    loadNarrator(onProgress) {
      return (loading ||= (async () => {
        tf = await import(TF_URL);
        narrator = await createNarrator({ tf, modelId: cfg.narrator, localModelPath: cfg.base, onProgress });
        return narrator;
      })().catch((e) => { loading = null; throw e; }));
    },
    // bitmap: the 896x504 frame (closed here). Returns {text, ms, aborted}; onToken streams text pieces.
    async describe(bitmap, { family, heads = null, facts = {}, task, maxNewTokens = 90, onToken, signal }) {
      let image;
      try { image = bitmapImage(tf, bitmap); } finally { bitmap.close(); }
      const context = contextLine({ heads: heads && heads.status === 'ok' ? heads : null, facts, family });
      return narrator.describe(image, { context, task, maxNewTokens, onToken, signal });
    },
    cancel() { if (narrator && narrator.cancel) narrator.cancel(); },
  };
}
