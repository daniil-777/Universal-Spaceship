// chat/tests/capcom.test.mjs — chat/web/capcom.js without a model (a mock transformers.js): the runtime ladder, model URL → loader
// paths, the fallback walk (load error, NaN-degenerate warm-up), retrieval-only answer cards, the serial queue (a new ask cancels the
// running one), AbortSignal; plus chat/web/serve.mjs's allow-list and chat/web/bench.mjs's parity/percentile helpers.
//   node --test chat/tests/capcom.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import { createCapcom, ladder, modelSource, retrievalAnswer, loadKb } from '../web/capcom.js';
import { resolveFile } from '../web/serve.mjs';
import { parity, q } from '../web/bench.mjs';

const KB = { retrieval: { min_score: 0.5 }, facts: [
  { id: 'RL-001', title: 'PPO pilot', text: 'The pilot is a PPO policy trained in the browser.', keywords: ['ppo', 'policy', 'learn'] },
  { id: 'ORBIT-001', title: 'Orbital speed', text: 'The station orbits at about 27,581 km/h.', keywords: ['orbit', 'speed', 'station'] },
  { id: 'HOWTO-001', title: 'Landing', text: 'Open ?scenario=landing to watch the airliner land.', keywords: ['landing', 'airliner'] }] };

// a mock tf: generate emits `gen(i)` token ids one per tick until max_new_tokens, eos (7) or the stopping criteria
function mockTf({ fail = [], zeros = [], gen = (i) => 10 + i, eosAt = null, tick = 2 } = {}) {
  const tok = (text) => { const n = text.length % 50 + 5; return { input_ids: { dims: [1, n], data: new BigInt64Array(n).fill(3n) }, attention_mask: {} }; };
  tok.apply_chat_template = (msgs) => JSON.stringify(msgs); tok.decode = (ids) => ids.filter((x) => x !== 7).map((x) => `t${x}`).join(' ');
  const calls = [];
  return { calls, env: {}, AutoTokenizer: { from_pretrained: async () => tok },
    AutoModelForCausalLM: { from_pretrained: async (id, { device, dtype }) => {
      calls.push(`${device}/${dtype}`);
      if (fail.includes(`${device}/${dtype}`)) throw new Error(`no ${device}/${dtype}`);
      const z = zeros.includes(`${device}/${dtype}`);
      return { dispose: async () => {}, generate: async ({ input_ids, max_new_tokens, streamer, stopping_criteria }) => {
        const out = [...input_ids.data];
        for (let i = 0; i < max_new_tokens; i++) {
          await new Promise((ok) => setTimeout(ok, tick));
          if (stopping_criteria && stopping_criteria.interrupted) break;
          const t = z ? 0 : eosAt === i ? 7 : gen(i); out.push(BigInt(t));
          if (streamer) { streamer.o.token_callback_function([BigInt(t)]); if (t !== 7) streamer.o.callback_function(`t${t} `); }
          if (t === 7) break;
        }
        return { data: BigInt64Array.from(out) };
      } };
    } },
    TextStreamer: class { constructor(t, o) { this.o = o; } },
    InterruptableStoppingCriteria: class { constructor() { this.interrupted = false; } interrupt() { this.interrupted = true; } } };
}
const withGpu = async (f16, fn) => {
  const d = Object.getOwnPropertyDescriptor(globalThis.navigator, 'gpu');
  Object.defineProperty(globalThis.navigator, 'gpu', { configurable: true, value: { requestAdapter: async () => ({ features: new Set(f16 ? ['shader-f16'] : []), info: { vendor: 'mock' } }) } });
  try { return await fn(); } finally { if (d) Object.defineProperty(globalThis.navigator, 'gpu', d); else delete globalThis.navigator.gpu; }
};

test('ladder: shader-f16 → q4f16, no f16 → q4, no WebGPU → WASM; explicit device / dtype', () => {
  assert.deepEqual(ladder('auto', null, { f16: true }), [['webgpu', 'q4f16'], ['webgpu', 'q4'], ['wasm', 'q4']]);
  assert.deepEqual(ladder('auto', null, { f16: false }), [['webgpu', 'q4'], ['wasm', 'q4']]);
  assert.deepEqual(ladder('auto', null, null), [['wasm', 'q4']]);
  assert.deepEqual(ladder('webgpu', null, { f16: true }), [['webgpu', 'q4f16']]);
  assert.deepEqual(ladder('webgpu', 'q4', { f16: true }), [['webgpu', 'q4']]);
  assert.deepEqual(ladder('wasm', null, { f16: true }), [['wasm', 'q4']]);
  assert.deepEqual(ladder('auto', 'q4f16', null), [['wasm', 'q4']]);
});
test('modelSource: same origin → root-relative local path; other origin → remoteHost; else a Hub id', () => {
  const b = 'http://127.0.0.1:5/chat/web/capcom-worker.js';
  assert.deepEqual(modelSource('/__capcom/models/m1/', b), { localModelPath: '/__capcom/models/', modelId: 'm1' });
  assert.deepEqual(modelSource('http://127.0.0.1:5/a/b/m2', b), { localModelPath: '/a/b/', modelId: 'm2' });
  assert.deepEqual(modelSource('https://huggingface.co/x/resolve/main/m3', b), { remoteHost: 'https://huggingface.co/x/resolve/main/', modelId: 'm3' });
  assert.deepEqual(modelSource('onnx-community/LFM2.5-350M-ONNX', b), { modelId: 'onnx-community/LFM2.5-350M-ONNX' });
});
test('retrieval-only cards: the top note, else a deflection that leads to an unseen highlight', async () => {
  const K = await loadKb(KB);
  assert.equal(K.card('how does the ppo policy learn').text, KB.facts[0].text);
  assert.deepEqual(K.card('how does the ppo policy learn').notes[0], 'RL-001');
  assert.equal(retrievalAnswer([], K.bm.byId, { scene: 'belt', seen: ['comets'] }), "That's not in my flight notes. Want to see Earth orbit next?");
});
test('every rung fails → notes only; ask answers with the top note', async () => {
  const tf = mockTf({ fail: ['wasm/q4'] }), c = await createCapcom({ tf, modelId: 'm', kb: KB, device: 'auto' });
  assert.equal(c.info.ready, false); assert.match(c.info.error, /wasm\/q4: no wasm\/q4/);
  const r = await c.ask('what speed does the station orbit at?');
  assert.equal(r.fallback, true); assert.equal(r.text, KB.facts[1].text); assert.deepEqual(r.notes.slice(0, 1), ['ORBIT-001']);
});
test('auto walks the ladder: q4f16 load error → q4 NaN warm-up (all <|pad|>) → WASM q4', async () => withGpu(true, async () => {
  const tf = mockTf({ fail: ['webgpu/q4f16'], zeros: ['webgpu/q4'] }), c = await createCapcom({ tf, modelId: 'm', kb: KB });
  assert.deepEqual(tf.calls, ['webgpu/q4f16', 'webgpu/q4', 'wasm/q4']);
  assert.equal(c.info.ready, true); assert.equal(c.info.device, 'wasm');
  assert.deepEqual(c.info.attempts.map((a) => a.ok), [false, false, true]); assert.match(c.info.attempts[1].error, /degenerate/);
}));
test('ask: streams, stops at eos, reports notes / TTFT / tok/s; history + state reach the prompt', async () => {
  const tf = mockTf({ eosAt: 5 }), c = await createCapcom({ tf, modelId: 'm', kb: KB, device: 'wasm' });
  let s = ''; const r = await c.ask('ppo policy?', { history: [{ role: 'user', content: 'hi' }, { role: 'assistant', content: 'hello' }], state: { scene: 'orbit' }, onToken: (x) => { s += x; } });
  assert.equal(r.tokens, 6); assert.equal(r.text, 't10 t11 t12 t13 t14'); assert.equal(s.trim(), r.text); assert.equal(r.aborted, false);
  assert.ok(r.ttftMs > 0 && r.tokPerSec > 0 && r.ms >= r.ttftMs); assert.equal(r.notes[0], 'RL-001');
  const { text } = c.render('ppo policy?', { history: [{ role: 'user', content: 'hi' }], state: { scene: 'orbit' } });
  assert.match(text, /State: Earth orbit; not seen yet/); assert.match(text, /PPO policy/);
});
test('serial queue: a new ask cancels the running one; results come back in order', async () => {
  const c = await createCapcom({ tf: mockTf({ tick: 3 }), modelId: 'm', kb: KB, device: 'wasm' }), done = [];
  const a = c.ask('first', { maxNewTokens: 200 }).then((r) => { done.push('a'); return r; });
  await new Promise((ok) => setTimeout(ok, 30));
  const b = c.ask('second', { maxNewTokens: 5 }).then((r) => { done.push('b'); return r; });
  const [ra, rb] = await Promise.all([a, b]);
  assert.equal(ra.aborted, true); assert.ok(ra.tokens < 200); assert.equal(rb.aborted, false); assert.equal(rb.tokens, 5); assert.deepEqual(done, ['a', 'b']);
  const x = c.ask('x', { maxNewTokens: 100 }), y = c.ask('y', { maxNewTokens: 100 }), z = c.ask('z', { maxNewTokens: 3 });
  const [rx, ry, rz] = await Promise.all([x, y, z]);
  assert.equal(rx.aborted, true); assert.equal(ry.aborted, true); assert.equal(ry.tokens, 0); assert.equal(rz.tokens, 3);
});
test('AbortSignal stops a running ask', async () => {
  const c = await createCapcom({ tf: mockTf({ tick: 3 }), modelId: 'm', kb: KB, device: 'wasm' }), ac = new AbortController();
  const p = c.ask('abort me', { maxNewTokens: 500, signal: ac.signal }); setTimeout(() => ac.abort(), 20);
  const r = await p; assert.equal(r.aborted, true); assert.ok(r.tokens < 500);
});
test('serve allow-list: site files, /__capcom/{models,kb,data} only, no dotfiles / ._ sidecars / traversal', () => {
  const o = { site: '/s', chat: '/c' };
  assert.equal(resolveFile('/chat/web/demo.html', o), '/s/chat/web/demo.html');
  assert.equal(resolveFile('/__capcom/models/m/onnx/model_q4.onnx', o), '/c/models/m/onnx/model_q4.onnx');
  assert.equal(resolveFile('/__capcom/kb/kb.json', o), '/c/kb/kb.json');
  for (const bad of ['/__capcom/logs/x.json', '/__capcom/venv/bin/python', '/__capcom/models/../logs/a', '/../etc/passwd', '/chat/web/._demo.html', '/.git/config', '/__capcom/kb/._kb.json'])
    assert.equal(resolveFile(bad, o), null, bad);
});
test('bench helpers: parity prefix / agreement, percentiles', () => {
  const p = parity([[1, 2, 3, 4], [1, 9, 3], [5]], [[1, 2, 3, 4], [1, 2, 3], [6, 7]], 32);
  assert.deepEqual(p.prefixes, [4, 1, 0]); assert.equal(p.full_match, 1); assert.equal(p.token_agreement, 0.75);
  assert.equal(q([5, 1, 3, 2, 4], 0.5), 3); assert.equal(q([], 0.5), null);
});
