// mock.js — ?narratorMock=1: the engine interface with canned verdicts and text at realistic timing and no models
// (Pilot Eye warms up over two frames; the download runs ~2.6 s; the Narrator answers after ~380 ms, then a word every
// 45 ms, and the canned text repeats itself and ends mid-sentence as v0 does, so the sentence gate is exercised).
// window.__narrator.card.mock holds it for visual QA: holdAt (bytes) freezes the download, pauseAfter (words) the stream,
// heads = {family: {...}} overrides a verdict.
import { templateSentence } from '../../vlm/web/templates.js';
import { memory } from './engine.js';

const TOTAL = 243.6e6;
export const MOCK_HEADS = {
  S: { verdict: 'CAUTION', severity: 1, reasons: ['HAZARD_AHEAD'], action: 'CLIMB' },
  A: { verdict: 'UNSAFE', severity: 3, reasons: ['TERRAIN_CLOSE'], action: 'CLIMB' },
  L: { verdict: 'SAFE', severity: 0, reasons: [], action: 'CONTINUE' },
  D: { verdict: 'CAUTION', severity: 1, reasons: ['CLOSING_TOO_FAST'], action: 'SLOW_DOWN' },
  Z: { verdict: 'SAFE', severity: 0, reasons: [], action: 'NONE_SAFE', tags: ['MOUNTAINS', 'COASTLINE'], range_bin: 3 },
};
const TEXT = {
  describe: {
    S: 'The camera follows the ship through the asteroid belt from close behind. A large rock drifts to the upper left, clear of the path. The stars are bright against a black sky. The camera follows the ship through the belt.',
    A: 'The ship flies low over snow-covered mountains under a heavy sky. A ridge rises ahead and slightly to the right. Clouds hang low over the valley floor. The ship flies low over',
    Z: 'The view looks down on the Alps from about 400 km. Snow covers the high peaks while the valleys stay green. A long lake lies to the west of the range. The view looks down on the Alps.',
    L: 'The airliner is on final approach to runway 26 in clear weather. Fields stretch out on both sides of the extended centreline. The runway lights are visible ahead.',
    D: 'The orbiter approaches the station along the docking corridor. The docking port sits in the middle of the frame. Earth fills the lower part of the view.',
  },
  safety: {
    S: 'Monitor verdict: CAUTION. A rock sits ahead of the ship and slightly above the path. Climbing now keeps a wide margin. Climbing now keeps a wide margin.',
    A: 'Monitor verdict: UNSAFE. The ridge ahead is higher than the ship. The ship must climb to clear it.',
    L: 'Monitor verdict: SAFE. The approach is stable on the localizer and the glideslope. The wind is light.',
    D: 'Monitor verdict: CAUTION. The ship closes on the port a little too fast. Slowing down keeps it inside the corridor limit.',
  },
  now: { S: 'Climb a little to pass above the rock. Then hold the course.', A: 'Climb at once to clear the ridge. Keep the speed up.', L: 'Continue the approach. Keep the speed on target.', D: 'Slow down the closing rate. Then continue to the port.' },
  ask: { S: 'The ship is in the middle of the corridor. Nothing blocks the path right now.', A: 'The ship is over high ground. The ridge ahead is the main hazard.', Z: 'The view shows mountains and a coastline. It is daytime.', L: 'The runway is straight ahead. The weather is clear.', D: 'The station is straight ahead. The port is aligned.' },
};
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

export function createMockEngine() {
  const mock = { holdAt: null, pauseAfter: null, wordMs: 45, ttftMs: 380, heads: {} };
  let onResult = null, ready = false, loading = null; const seen = {};
  const until = async (cond) => { while (cond()) await wait(50); };
  return {
    mock, get device() { return ready ? 'webgpu' : null; }, get ready() { return ready; },
    probe: async () => { await wait(120); return { ok: true, labels: {} }; },
    startEye(cb) { onResult = cb; },
    nominal: () => null,
    accepting: () => !!onResult,
    async release() { ready = false; loading = null; },
    postFrame({ bitmap, family, episode_id }) {
      bitmap.close();
      const k = `${family}:${episode_id}`; seen[k] = (seen[k] || 0) + 1;
      const h = { ...MOCK_HEADS[family], ...(mock.heads[family] || {}) }, warm = family !== 'Z' && seen[k] < 3;
      setTimeout(() => onResult && onResult(warm ? { status: 'warming up', ms: 7 } : { status: 'ok', ...h, sentence: templateSentence(h, family), ms: 7.4 }), 8);
    },
    stopEye() { onResult = null; },
    loadNarrator(onProgress) {
      return (loading ||= (async () => {
        const dur = memory.get().downloaded ? 700 : 2600, t0 = performance.now();
        for (let x = 0; x < 1;) {
          x = Math.min(1, (performance.now() - t0) / dur); const loaded = TOTAL * (x < 0.5 ? 2 * x * x : 1 - 2 * (1 - x) * (1 - x));
          if (mock.holdAt !== null && loaded >= mock.holdAt) { if (onProgress) onProgress({ loaded: mock.holdAt, total: TOTAL }); const h = mock.holdAt; await until(() => mock.holdAt === h); continue; }
          if (onProgress) onProgress({ loaded, total: TOTAL }); await wait(50);
        }
        ready = true;
      })());
    },
    async describe(bitmap, { family, task, kind = 'describe', onToken, signal }) {
      bitmap.close();
      const t0 = performance.now(), text = (TEXT[kind] && (TEXT[kind][family] || TEXT[kind].S)) || TEXT.ask.S, words = text.split(' ');
      await wait(mock.ttftMs); let out = '';
      for (let i = 0; i < words.length; i++) {
        if (signal && signal.aborted) return { text: out.trim(), ms: performance.now() - t0, aborted: true };
        if (mock.pauseAfter !== null && i >= mock.pauseAfter) { await until(() => mock.pauseAfter !== null && !(signal && signal.aborted)); i--; continue; }
        const w = words[i] + (i < words.length - 1 ? ' ' : ''); out += w; if (onToken) onToken(w); await wait(mock.wordMs);
      }
      return { text: out.trim(), ms: performance.now() - t0, aborted: false, task };
    },
    cancel() {},
  };
}
