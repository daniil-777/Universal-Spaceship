// The in-site Narrator's runtime seams (src/narrator): Pilot Eye backpressure in the engine, the open/close lifecycle
// (a close or reopen mid-await never lets a stale open continue; the model is released after 5 minutes closed), the
// mock gate, the feed's prototype wrap (exact restore, one wrap, never throws, re-entrancy) and the grab fallback.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createEngine, config, resolveCurrent, V0 } from '../src/narrator/engine.js';
import { createLifecycle, IDLE_RELEASE_MS } from '../src/narrator/life.js';
import { createFeed, createGrabber } from '../src/narrator/feed.js';

const bitmap = () => ({ closed: 0, width: 896, height: 504, close() { this.closed++; } });
class FakeWorker {
  constructor() { FakeWorker.last = this; this.sent = []; this.dead = false; }
  postMessage(m) { this.sent.push(m); } terminate() { this.dead = true; }
  emit(data) { this.onmessage({ data }); }
}
const cfg = { base: '/m/', eye: 'e', narrator: 'n', mock: false };
const okFetch = async () => ({ ok: true, json: async () => ({ input: [160, 96], nominal_frame_dt: { S: 0.2 } }) });
const frame = (b) => ({ bitmap: b, sim_t_s: 1, family: 'S', episode_id: 1 });

test('engine: no frame reaches Pilot Eye before it is ready or while one is outstanding; every dropped bitmap is closed', async () => {
  const eng = createEngine(cfg, { Worker: FakeWorker, fetch: okFetch }), seen = [];
  assert.equal(eng.accepting(), false, 'no worker yet');
  const b0 = bitmap(); eng.postFrame(frame(b0)); assert.equal(b0.closed, 1);
  assert.equal((await eng.probe()).ok, true); eng.startEye((r) => seen.push(r));
  const w = FakeWorker.last; assert.equal(w.sent[0].type, 'init');
  const b1 = bitmap(); assert.equal(eng.accepting(), false); eng.postFrame(frame(b1));
  assert.equal(b1.closed, 1, 'dropped before ready'); assert.equal(w.sent.length, 1, 'nothing queued before ready');
  w.emit({ type: 'ready' }); assert.equal(eng.accepting(), true);
  const b2 = bitmap(); eng.postFrame(frame(b2)); assert.equal(w.sent.length, 2); assert.equal(w.sent[1].bitmap, b2); assert.equal(b2.closed, 0, 'the worker owns it now');
  assert.equal(eng.accepting(), false, 'one outstanding');
  const b3 = bitmap(); eng.postFrame(frame(b3)); assert.equal(b3.closed, 1, 'a slow result drops the next frame'); assert.equal(w.sent.length, 2);
  w.emit({ type: 'result', id: w.sent[1].id, status: 'ok', verdict: 'SAFE' }); assert.equal(eng.accepting(), true); assert.equal(seen.at(-1).verdict, 'SAFE');
  const b4 = bitmap(); eng.postFrame(frame(b4)); w.emit({ type: 'error', id: w.sent[2].id, message: 'x' }); assert.equal(eng.accepting(), true, 'an error frees the slot too');
  w.emit({ type: 'error', id: 99, message: 'late' }); eng.postFrame(frame(bitmap())); assert.equal(eng.accepting(), false, 'the count never goes below zero');
  eng.stopEye(); assert.ok(w.dead); assert.equal(eng.accepting(), false);
});
test('engine: the Narrator model is released on request and loads again (from the browser cache) next time', async () => {
  let loads = 0, disposed = 0;
  const tf = { env: {}, AutoProcessor: { from_pretrained: async () => Object.assign(async () => ({}), { tokenizer: {} }) },
    AutoModelForVision2Seq: { from_pretrained: async () => { loads++; return { dispose: async () => { disposed++; } }; } } };
  const eng = createEngine(cfg, { importTf: async () => tf });
  await eng.loadNarrator(() => {}); assert.equal(eng.ready, true); assert.equal(loads, 1);
  await eng.release(); assert.equal(eng.ready, false); assert.equal(disposed, 1);
  await eng.loadNarrator(() => {}); assert.equal(eng.ready, true); assert.equal(loads, 2);
});
test('config: ?narratorMock=1 only on localhost or 127.0.0.1; model folders are sanitised', () => {
  assert.equal(config('?narratorMock=1', 'localhost').mock, true); assert.equal(config('?narratorMock=1', '127.0.0.1').mock, true);
  assert.equal(config('?narratorMock=1', 'daniil-777.github.io').mock, false); assert.equal(config('', 'localhost').mock, false);
  assert.equal(config('?eye=../../etc', 'localhost').eye, 'pilot-eye-v0/export'); assert.equal(config('?narrator=nar-v1', 'x').narrator, 'nar-v1');
});

function deferred() { let ok; const p = new Promise((r) => { ok = r; }); return { p, ok }; }
function steps(log, d = {}) {
  return {
    build: () => { log.push('build'); return d.build ? d.build.p : Promise.resolve(); }, show: () => log.push('show'),
    install: () => { log.push('install'); return d.install ? d.install.p : Promise.resolve(); }, startTick: () => log.push('tick'),
    probe: () => { log.push('probe'); return d.probe ? d.probe.p : Promise.resolve({ ok: true }); }, ready: () => log.push('ready'),
    expand: () => log.push('expand'), teardown: () => log.push('teardown'), idle: () => log.push('idle'),
  };
}
test('lifecycle: a close during install stops the open there (no tick, no probe)', async () => {
  const log = [], install = deferred(), life = createLifecycle(steps(log, { install }));
  const opening = life.open(); await Promise.resolve(); await Promise.resolve();
  assert.deepEqual(log, ['build', 'show', 'install']);
  life.close(); install.ok(); await opening;
  assert.deepEqual(log, ['build', 'show', 'install', 'teardown']); assert.equal(life.on, false);
});
test('lifecycle: close then reopen while the first build is pending runs the rest once, for the newest open', async () => {
  const log = [], build = deferred(), life = createLifecycle(steps(log, { build }));
  const a = life.open(); life.close(); const b = life.open();
  assert.equal(life.live(1), false, 'the first open is stale'); assert.equal(life.live(life.seq), true);
  build.ok(); await Promise.all([a, b]);
  assert.deepEqual(log, ['build', 'teardown', 'show', 'install', 'tick', 'probe', 'ready'], 'built once, shown once');
  await life.open(); assert.equal(log.at(-1), 'expand', 'open while on only expands');
});
test('lifecycle: a guard taken before an await goes stale on close and on reopen', async () => {
  const life = createLifecycle(steps([])); await life.open(); const g = life.guard();
  assert.equal(g(), true); life.close(); assert.equal(g(), false); await life.open(); assert.equal(g(), false, 'a reopen is a new session');
});
test('lifecycle: the model is released after 5 minutes closed, and a reopen before that keeps it', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const log = [], life = createLifecycle(steps(log)); assert.equal(IDLE_RELEASE_MS, 5 * 60 * 1000);
  await life.open(); life.close(); t.mock.timers.tick(IDLE_RELEASE_MS - 1); assert.ok(!log.includes('idle'));
  await life.open(); t.mock.timers.tick(10 * 60 * 1000); assert.ok(!log.includes('idle'), 'reopened in time');
  life.close(); t.mock.timers.tick(IDLE_RELEASE_MS); assert.equal(log.filter((x) => x === 'idle').length, 1);
});

function feedRig({ grab = null, accepting = () => true } = {}) {
  const calls = { orig: 0, grab: 0, frames: [] }, canvas = { id: 'view', width: 1920, height: 1080 };
  const proto = { render(dt) { calls.orig++; if (this.reenter) { this.reenter = false; this.render(dt); } return 'r'; } };
  const orig = proto.render;
  const g = { __ap: { ready: true, zoom: { active: false }, state: { atmo: false }, env: { t: 0 } } };
  const feed = createFeed({ proto, g, doc: { hidden: false }, accepting, onFrame: (f) => calls.frames.push(f),
    grab: grab || (() => { calls.grab++; return Promise.resolve(bitmap()); }) });
  const composer = Object.create(proto, { renderer: { value: { domElement: canvas } }, renderToScreen: { value: true, writable: true } });
  return { proto, orig, g, feed, composer, calls };
}
test('feed: install wraps the injected prototype, uninstall restores the exact function, and install twice wraps once', async () => {
  const r = feedRig();
  await r.feed.install(); assert.notEqual(r.proto.render, r.orig); assert.equal(r.feed.installed, true);
  r.feed.uninstall(); assert.equal(r.proto.render, r.orig, 'restored exactly'); assert.equal(r.feed.installed, false);
  await r.feed.install(); await r.feed.install(); r.g.__ap.env.t = 0.2;
  assert.equal(r.composer.render(0), 'r'); assert.equal(r.calls.orig, 1); assert.equal(r.calls.grab, 1, 'one wrap: one grab per render');
  r.feed.uninstall(); assert.equal(r.proto.render, r.orig);
});
test('feed: a throwing grab never escapes the render; a re-entrant render grabs once', async () => {
  const bad = feedRig({ grab: () => { throw new Error('boom'); } });
  await bad.feed.install(); bad.g.__ap.env.t = 0.2;
  assert.doesNotThrow(() => bad.composer.render(0)); assert.equal(bad.calls.orig, 1); assert.equal(bad.feed.errors, 1);
  const re = feedRig(); await re.feed.install(); re.g.__ap.env.t = 0.4; re.composer.reenter = true;
  re.composer.render(0); assert.equal(re.calls.orig, 2, 'both renders ran'); assert.equal(re.calls.grab, 1, 'the grab ran once, after the outer render');
});
test('feed: nothing is grabbed while the engine is not accepting frames', async () => {
  let open = false; const r = feedRig({ accepting: () => open });
  await r.feed.install(); r.g.__ap.env.t = 0.2; r.composer.render(0); assert.equal(r.calls.grab, 0);
  open = true; r.g.__ap.env.t = 0.4; r.composer.render(0); assert.equal(r.calls.grab, 1);
});
test('grabber: after createImageBitmap rejects once it copies synchronously through a 2D canvas from then on', async () => {
  const drawn = [], cib = (src) => (src.kind === '2d' ? Promise.resolve({ from: '2d' }) : Promise.reject(new Error('unsupported')));
  const makeCanvas = () => ({ kind: '2d', width: 0, height: 0, getContext: () => ({ drawImage: (...a) => drawn.push(a.length) }) });
  const grab = createGrabber({ cib, makeCanvas }), cv = { width: 1920, height: 1080 };
  assert.equal(await grab(cv), null, 'the first frame is lost'); assert.equal(grab.useCanvas, true);
  const p = grab(cv); assert.deepEqual(drawn, [9], 'drawImage ran inside the call, before any await'); assert.deepEqual(await p, { from: '2d' });
});

test('models: the engine uses the folders models/current.json names (install_models.mjs), a URL pin wins, v0 without it', async () => {
  const served = (cur) => async (u) => (u.endsWith('current.json') ? (cur ? { ok: true, json: async () => cur } : { ok: false }) : { ok: true, json: async () => ({ input: [160, 96] }) });
  const v1 = { eye: 'pilot-eye-v1', narrator: 'narrator-v1' };
  const eng = createEngine(config('', 'x'), { Worker: FakeWorker, fetch: served(v1) }); await eng.probe(); eng.startEye(() => {});
  assert.equal(FakeWorker.last.sent[0].encoderUrl, '/__vlm/models/pilot-eye-v1/encoder.onnx'); assert.deepEqual(eng.models, v1);
  let asked = null;
  const tf = { env: {}, AutoProcessor: { from_pretrained: async (id) => { asked = id; return Object.assign(async () => ({}), { tokenizer: {} }); } }, AutoModelForVision2Seq: { from_pretrained: async () => ({ dispose: async () => {} }) } };
  await createEngine(config('', 'x'), { fetch: served(v1), importTf: async () => tf }).loadNarrator(() => {}); assert.equal(asked, 'narrator-v1', 'the Narrator loads from the installed folder even before a probe');
  assert.deepEqual(await resolveCurrent(config('', 'x'), served(null)).then((c) => [c.eye, c.narrator]), [V0.eye, V0.narrator], 'no current.json: v0');
  assert.deepEqual(await resolveCurrent(config('?narrator=nar-x', 'x'), served(v1)).then((c) => [c.eye, c.narrator]), ['pilot-eye-v1', 'nar-x'], 'a pinned folder wins');
  assert.equal((await resolveCurrent(config('', 'x'), served({ eye: '../../etc', narrator: 'n v1' }))).eye, V0.eye, 'a bad folder name is ignored');
  assert.equal((await resolveCurrent(config('', 'x'), async () => { throw new Error('offline'); })).narrator, V0.narrator);
  assert.deepEqual(config('?eye=pe-x', 'x').pinned, { eye: true, narrator: false });
});
