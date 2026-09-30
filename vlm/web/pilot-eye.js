// vlm/web/pilot-eye.js — Pilot Eye at runtime (spec §10.3): ORT-web WASM, 1 thread (no COOP/COEP); frames posted on the sim-time
// grid of the family's nominal frame_dt; a ring of 3 {sim_t_s, map} reset on a family or episode change, a non-increasing time
// or a step outside frame_dt +- max(1/15 s, one family step); heads only when the ring is full. ORT is injected (browser or Node).
// The enums come from schema.js (browser-loadable); the reason order from labels.json when it lists one (Task 11 writes it).
import { boxResize } from '../gen/boxresize.js';
import { ACTIONS, FAMILY_ACTIONS, REASONS, ZOOM_TAGS, VERDICTS, NOMINAL_FRAME_DT, STEP_S } from '../gen/schema.js';
import { templateSentence } from './templates.js';
export { ACTIONS, FAMILY_ACTIONS };
const HEAD_NAMES = ['verdict', 'severity', 'reasons', 'actions', 'p_ref', 'reg', 'tags', 'range'];
const sig = (x) => 1 / (1 + Math.exp(-x)), argmax = (a) => a.reduce((b, v, i) => (v > a[b] ? i : b), 0);
const soft = (a) => { const m = Math.max(...a), e = a.map((v) => Math.exp(v - m)), s = e.reduce((x, y) => x + y, 0); return e.map((v) => v / s); };
export const gridIndex = (simT, frameDt) => Math.floor(simT / frameDt + 1e-9);
export function createFrameRing({ frameDt, stepS }) {
  let ring = [], fam = null, ep = null; const tol = Math.max(1 / 15, stepS);
  return {
    push(e) {
      const last = ring[ring.length - 1];
      if (e.family !== fam || e.episode_id !== ep || (last && (e.sim_t_s <= last.sim_t_s || Math.abs(e.sim_t_s - last.sim_t_s - frameDt) > tol + 1e-3))) ring = [];
      fam = e.family; ep = e.episode_id; ring.push(e); if (ring.length > 3) ring.shift();
      return ring.length === 3 ? { maps: ring.map((x) => x.map), dt: [(ring[1].sim_t_s - ring[0].sim_t_s) / frameDt, (ring[2].sim_t_s - ring[1].sim_t_s) / frameDt] } : null;
    },
    reset() { ring = []; fam = null; ep = null; },
  };
}
// argmax over the family's legal actions; a tie that includes CONTINUE goes to CONTINUE; NONE_SAFE when every sigmoid < 0.5
// (the same rule as evaluate.py pick_action)
export function pickAction(p, family) {
  const sub = FAMILY_ACTIONS[family].map((a) => ACTIONS.indexOf(a)), best = Math.max(...sub.map((i) => p[i]));
  if (!(best >= 0.5)) return 'NONE_SAFE';
  const ties = sub.filter((i) => Math.abs(p[i] - best) < 1e-6); return ties.includes(0) ? 'CONTINUE' : ACTIONS[ties[0]];
}
// o: the 8 head logits (typed arrays or plain arrays) → the decoded monitor fields; reg stays as the raw log1p regression row
export function decodeHeads(o, family, reasons = REASONS) {
  const v = soft([...o.verdict]), reg = Array.from(o.reg, Number);
  return { verdict: VERDICTS[argmax(v)], p_unsafe: v[2], severity: argmax([...o.severity]), reasons: reasons.filter((_, i) => sig(o.reasons[i]) > 0.5),
    action: pickAction(Array.from(o.actions, sig), family === 'Z' ? 'S' : family), p_ref: sig(o.p_ref[0]), reg, ttc_s: Math.expm1(reg[0]), clearance_u: Math.expm1(reg[1]),
    tags: ZOOM_TAGS.filter((_, i) => sig(o.tags[i]) > 0.5), range_bin: argmax([...o.range]) };
}
export const prepareFrame = (rgba, w, h, [W, H] = [160, 96]) => boxResize(rgba, w, h, W, H);
// encoderUrl / headsUrl: a URL (browser), a path (Node) or the model bytes; labels: the export's labels.json
export async function createPilotEye({ ort, encoderUrl, headsUrl, labels = {}, numThreads = 1 }) {
  ort.env.wasm.numThreads = numThreads;
  const opt = { executionProviders: ['wasm'] }, enc = await ort.InferenceSession.create(encoderUrl, opt), hd = await ort.InferenceSession.create(headsUrl, opt), rings = {};
  const [W, H] = labels.input || [160, 96], reasons = labels.reasons || REASONS, n = W * H;
  const frameDt = (f) => (labels.nominal_frame_dt && labels.nominal_frame_dt[f]) || NOMINAL_FRAME_DT[f].s;
  return {
    async push({ rgb, sim_t_s, family, episode_id }) {
      const t0 = performance.now(), x = new Float32Array(3 * n);
      if (!rgb || rgb.length !== 3 * n) throw new Error(`pilot-eye: rgb must hold ${W}x${H}x3 bytes, got ${rgb ? rgb.length : rgb}`);
      if (!NOMINAL_FRAME_DT[family] && family !== 'Z') throw new Error(`pilot-eye: unknown family ${family}`);
      for (let i = 0; i < n; i++) { x[i] = rgb[i * 3] / 255; x[n + i] = rgb[i * 3 + 1] / 255; x[2 * n + i] = rgb[i * 3 + 2] / 255; }
      const map = (await enc.run({ pixels: new ort.Tensor('float32', x, [1, 3, H, W]) })).map;
      // Z is one frame: the three slots hold it and dt is [1, 1] (as PilotEyeData feeds a single-frame record)
      const r = family === 'Z' ? { maps: [map, map, map], dt: [1, 1] } : (rings[family] ||= createFrameRing({ frameDt: frameDt(family), stepS: STEP_S[family] })).push({ sim_t_s, family, episode_id, map });
      if (family !== 'Z') for (const f of Object.keys(rings)) if (f !== family) rings[f].reset();
      if (!r) return { status: 'warming up', ms: performance.now() - t0 };
      const out = await hd.run({ m0: r.maps[0], m1: r.maps[1], m2: r.maps[2], dt: new ort.Tensor('float32', Float32Array.from(r.dt), [1, 2]) }), o = {};
      for (const k of HEAD_NAMES) o[k] = out[k].data;
      const h = decodeHeads(o, family, reasons);
      return { status: 'ok', ...h, sentence: templateSentence(h, family), ms: performance.now() - t0 };
    },
  };
}
