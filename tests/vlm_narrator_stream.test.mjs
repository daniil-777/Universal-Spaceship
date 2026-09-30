// vlm/web/narrator.js, the in-site additions: download progress (transformers.js progress_callback), an AbortSignal that
// stops generation through a stopping criterion, and a serial queue in which a new request cancels the one before it.
// transformers.js is a stand-in: generate() streams one token per tick and asks the stopping criteria after each.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createNarrator, progressTracker } from '../vlm/web/narrator.js';

const tick = () => new Promise((r) => setTimeout(r, 1));
function streamingTf({ tokens = ['The ', 'sky ', 'is ', 'clear. ', 'The ', 'sea ', 'is ', 'calm. '], progress = [] } = {}) {
  const log = { runs: 0, started: [], stoppedAt: [], seen: [] };
  class InterruptableStoppingCriteria { constructor() { this.interrupted = false; } interrupt() { this.interrupted = true; } }
  const processor = Object.assign(async (text) => { log.seen.push(text); return { input_ids: { dims: [1, 5] } }; },
    { apply_chat_template: (m) => m[0].content.map((c) => c.text || '<image>').join('|'), batch_decode: () => [` ${log.last} `], tokenizer: {} });
  const mdl = {
    async generate(o) {
      const id = log.runs++; log.started.push(id); let out = '', n = 0;
      for (const t of tokens) {
        await tick();
        if (o.stopping_criteria && o.stopping_criteria.interrupted) break;
        out += t; n++; if (o.streamer) o.streamer.cb(t);
      }
      log.stoppedAt.push(n); log.last = out; return { slice: () => out };
    },
  };
  const tf = { env: {}, log, InterruptableStoppingCriteria, AutoProcessor: { from_pretrained: async () => processor },
    AutoModelForVision2Seq: { from_pretrained: async (id, o) => { for (const p of progress) o.progress_callback && o.progress_callback(p); return mdl; } },
    TextStreamer: class { constructor(tok, o) { this.cb = o.callback_function; } } };
  return tf;
}
const img = { width: 512, height: 512 };

test('progressTracker: sums per-file progress, prefers progress_total, ignores other events', () => {
  const seen = [], cb = progressTracker((p) => seen.push([p.loaded, p.total]));
  cb({ status: 'initiate', file: 'a.onnx' }); cb({ status: 'progress', file: 'a.onnx', loaded: 10, total: 100 }); cb({ status: 'progress', file: 'b.onnx', loaded: 5, total: 50 });
  cb({ status: 'progress', file: 'a.onnx', loaded: 40, total: 100 }); cb({ status: 'done', file: 'a.onnx' });
  assert.deepEqual(seen, [[10, 100], [15, 150], [45, 150]]);
  const tot = [], t2 = progressTracker((p) => tot.push([p.loaded, p.total]));
  t2({ status: 'progress_total', loaded: 84e6, total: 244e6, files: {} }); t2({ status: 'progress', file: 'x', loaded: 1, total: 2 });
  assert.deepEqual(tot, [[84e6, 244e6]], 'once an aggregate arrives, per-file events do not double count');
});
test('createNarrator passes onProgress to the model download as progress_callback', async () => {
  const seen = [], tf = streamingTf({ progress: [{ status: 'progress_total', loaded: 1, total: 4 }, { status: 'progress_total', loaded: 4, total: 4 }] });
  await createNarrator({ tf, modelId: 'm', onProgress: (p) => seen.push(p.loaded / p.total) });
  assert.deepEqual(seen, [0.25, 1]);
  await createNarrator({ tf: streamingTf({ progress: [{ status: 'progress_total', loaded: 1, total: 4 }] }), modelId: 'm' });
});
test('describe streams tokens and an AbortSignal stops generation through the stopping criterion', async () => {
  const tf = streamingTf(), nar = await createNarrator({ tf, modelId: 'm' }), ctl = new AbortController(), got = [];
  const d = await nar.describe(img, { task: 'Describe the image in detail.', signal: ctl.signal, onToken: (t) => { got.push(t); if (got.length === 3) ctl.abort(); } });
  assert.deepEqual(got, ['The ', 'sky ', 'is ']); assert.equal(d.aborted, true); assert.equal(tf.log.stoppedAt[0], 3, 'generation stopped on the next step');
  const full = await nar.describe(img, { onToken: () => {} }); assert.equal(full.aborted, false); assert.equal(full.text, 'The sky is clear. The sea is calm.');
});
test('describe with an already aborted signal never starts generating', async () => {
  const tf = streamingTf(), nar = await createNarrator({ tf, modelId: 'm' }), ctl = new AbortController(); ctl.abort();
  await assert.rejects(nar.describe(img, { signal: ctl.signal }), (e) => e.name === 'AbortError'); assert.equal(tf.log.runs, 0);
});
test('a serial queue: a new request cancels the running one and runs only after it has stopped', async () => {
  const tf = streamingTf(), nar = await createNarrator({ tf, modelId: 'm' }), a = [], b = [];
  const first = nar.describe(img, { task: 'first', onToken: (t) => a.push(t) });
  await tick(); await tick(); await tick();
  const second = nar.describe(img, { task: 'second', onToken: (t) => b.push(t) });
  const [r1, r2] = await Promise.all([first, second]);
  assert.equal(r1.aborted, true); assert.ok(a.length > 0 && a.length < 8, `first streamed ${a.length}`); assert.equal(r2.aborted, false); assert.equal(b.length, 8);
  assert.deepEqual(tf.log.seen.map((s) => s.split('|').pop()), ['first', 'second'], 'never two generations at once');
  const x = nar.describe(img, { task: 'x' }), y = nar.describe(img, { task: 'y' }), z = nar.describe(img, { task: 'z' });
  const drop = (e) => e.name === 'AbortError', rx = assert.rejects(x, drop), ry = assert.rejects(y, drop, 'a request replaced before it ran is dropped');
  await rx; await ry; assert.equal((await z).aborted, false);
  assert.deepEqual(tf.log.seen.slice(2).map((s) => s.split('|').pop()), ['z'], 'only the latest request runs');
});
