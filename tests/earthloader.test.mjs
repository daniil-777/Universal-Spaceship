import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createTileLoader, sha1Js } from '../src/earthloader.js';

// A fake network: every request waits until the test answers it; a body's byte length stands for its content.
function fakeNet() {
  const pending = new Map(), calls = [];
  const fetchImpl = (url) => { calls.push(url); return new Promise((resolve) => pending.set(url, resolve)); };
  const reply = (url, { ok = true, status = 200, bytes = 10 } = {}) => { pending.get(url)({ ok, status, arrayBuffer: async () => new Uint8Array(bytes).buffer }); pending.delete(url); };
  return { fetchImpl, reply, calls };
}
const decode = async (blob) => ({ bitmap: blob.size });
const settle = () => new Promise((r) => setTimeout(r, 5));
const job = (url, prio, log, wanted = () => true) => ({ url, prio, raw: false, wanted, done: (bmp, why) => log.push([url, bmp ? 'bitmap' : null, why]) });
const quiet = () => { const w = console.warn; console.warn = () => {}; return () => { console.warn = w; }; };

test('at most maxInFlight requests at once; the queue is served lowest priority number first', async () => {
  const net = fakeNet(), L = createTileLoader({ maxInFlight: 2, fetchImpl: net.fetchImpl, decode }), log = [];
  for (const [u, p] of [['a', 5], ['b', 1], ['c', 3], ['d', 0]]) L.request(job(u, p, log));
  assert.equal(L.inFlight, 2); assert.equal(L.queued, 2); assert.deepEqual(net.calls, ['a', 'b']);
  net.reply('a'); await settle(); assert.deepEqual(net.calls, ['a', 'b', 'd'], 'd (0) before c (3)');
  net.reply('b'); net.reply('d'); await settle(); net.reply('c'); await settle();
  assert.deepEqual(log.map((l) => l[1]), ['bitmap', 'bitmap', 'bitmap', 'bitmap']); assert.equal(L.stats.loaded, 4);
});

test('a cached tile is served at once without a new request', async () => {
  const net = fakeNet(), L = createTileLoader({ fetchImpl: net.fetchImpl, decode }), log = [];
  L.request(job('a', 0, log)); net.reply('a'); await settle();
  L.request(job('a', 0, log)); assert.deepEqual(net.calls, ['a']); assert.deepEqual(log[1], ['a', 'bitmap', 'ok']);
});

test('prune drops queued requests nobody wants any more, before they are fetched', async () => {
  const net = fakeNet(), L = createTileLoader({ maxInFlight: 1, fetchImpl: net.fetchImpl, decode }), log = [];
  let keep = true; L.request(job('a', 0, log)); L.request(job('b', 1, log, () => keep));
  keep = false; L.prune(); assert.deepEqual(log, [['b', null, 'dropped']]); assert.equal(L.queued, 0);
  net.reply('a'); await settle(); assert.deepEqual(net.calls, ['a']);
});

test('a failed tile is not asked for again until the retry time has passed', async () => {
  const restore = quiet(); let t = 0;
  const net = fakeNet(), L = createTileLoader({ fetchImpl: net.fetchImpl, decode, retryMs: 1000, now: () => t }), log = [];
  L.request(job('a', 0, log)); net.reply('a', { ok: false, status: 404 }); await settle();
  assert.deepEqual(log[0], ['a', null, 'failed']); L.request(job('a', 0, log)); assert.deepEqual(log[1], ['a', null, 'failed']); assert.equal(net.calls.length, 1);
  t = 1500; L.request(job('a', 0, log)); assert.equal(net.calls.length, 2); restore();
});

test('Esri’s placeholder (its size and hash) counts as no imagery and is never fetched again', async () => {
  const net = fakeNet(), L = createTileLoader({ fetchImpl: net.fetchImpl, decode, blank: { bytes: 7, sha1: 'cafe' }, hashHex: async () => 'cafe' }), log = [];
  L.request(job('a', 0, log)); net.reply('a', { bytes: 7 }); await settle();
  assert.deepEqual(log[0], ['a', null, 'blank']); assert.equal(L.stats.blank, 1);
  L.request(job('a', 0, log)); assert.deepEqual(log[1], ['a', null, 'blank']); assert.equal(net.calls.length, 1);
  L.request(job('b', 0, log)); net.reply('b', { bytes: 8 }); await settle(); assert.deepEqual(log[2], ['b', 'bitmap', 'ok']);
});

test('a server that never answers times out and frees its slot', async () => {
  const restore = quiet(), calls = [], log = [];
  const fetchImpl = (url, opts) => { calls.push(url); return new Promise((resolve, reject) => { if (opts && opts.signal) opts.signal.addEventListener('abort', () => reject(new Error('timed out'))); }); };
  const L = createTileLoader({ maxInFlight: 1, fetchImpl, decode, timeoutMs: 20 });
  L.request(job('a', 0, log)); L.request(job('b', 1, log)); assert.deepEqual(calls, ['a']);
  await new Promise((r) => setTimeout(r, 60)); assert.deepEqual(log[0], ['a', null, 'failed']); assert.deepEqual(calls, ['a', 'b'], 'the next request got the slot');
  restore();
});

test('suspend stops a closed view fetching: the queue is dropped, running answers are discarded, nothing new starts until resume', async () => {
  const net = fakeNet(), L = createTileLoader({ maxInFlight: 1, fetchImpl: net.fetchImpl, decode }), log = [];
  L.request(job('a', 0, log)); L.request(job('b', 1, log)); L.suspend();
  assert.deepEqual(log, [['b', null, 'dropped']]); assert.equal(L.queued, 0);
  net.reply('a'); await settle(); assert.deepEqual(log[1], ['a', null, 'dropped']);
  L.request(job('c', 0, log)); assert.deepEqual(log[2], ['c', null, 'dropped']); assert.deepEqual(net.calls, ['a']);
  L.resume(); L.request(job('c', 0, log)); assert.deepEqual(net.calls, ['a', 'c']);
});

test('in-flight dedup: a second request for the same URL attaches to the first; one fetch, two done("ok")', async () => {
  const net = fakeNet(), L = createTileLoader({ fetchImpl: net.fetchImpl, decode }), log = [];
  L.request(job('a', 0, log)); L.request(job('a', 1, log));
  assert.equal(net.calls.length, 1, 'only one fetch for the two requests'); assert.equal(L.stats.deduped, 1);
  net.reply('a'); await settle();
  assert.deepEqual(log.map((l) => l[2]), ['ok', 'ok']);
});

test('in-flight dedup: an attached job no longer wanted gets dropped, the other still gets the result', async () => {
  const net = fakeNet(), L = createTileLoader({ fetchImpl: net.fetchImpl, decode }), log = [];
  let keep = true;
  L.request(job('a', 0, log)); L.request(job('a', 1, log, () => keep));
  keep = false; net.reply('a'); await settle();
  assert.deepEqual(log[0], ['a', 'bitmap', 'ok']); assert.deepEqual(log[1], ['a', null, 'dropped']);
});

test('perHost limits: at most one request per host in flight at once, with two hosts running concurrently', async () => {
  const net = fakeNet(), L = createTileLoader({ maxInFlight: 8, perHost: { default: 1 }, fetchImpl: net.fetchImpl, decode }), log = [];
  L.request(job('https://a.example/1', 0, log)); L.request(job('https://a.example/2', 0, log));
  L.request(job('https://b.example/1', 0, log)); L.request(job('https://b.example/2', 0, log));
  assert.equal(L.inFlight, 2, 'one per host, two hosts'); assert.deepEqual(net.calls, ['https://a.example/1', 'https://b.example/1']);
  net.reply('https://a.example/1'); net.reply('https://b.example/1'); await settle();
  assert.deepEqual(net.calls, ['https://a.example/1', 'https://b.example/1', 'https://a.example/2', 'https://b.example/2']);
  net.reply('https://a.example/2'); net.reply('https://b.example/2'); await settle();
  assert.equal(L.stats.loaded, 4);
});

test('a pinned URL is kept forever in its own map, even under a tiny keep limit', async () => {
  const net = fakeNet(), L = createTileLoader({ keep: 2, pin: (url) => url === 'pinned', fetchImpl: net.fetchImpl, decode }), log = [];
  L.request(job('pinned', 0, log)); net.reply('pinned'); await settle();
  for (const u of ['x1', 'x2', 'x3', 'x4', 'x5']) { L.request(job(u, 0, log)); net.reply(u); await settle(); }
  L.request(job('pinned', 0, log));
  assert.deepEqual(log[log.length - 1], ['pinned', 'bitmap', 'ok']);
  assert.equal(net.calls.filter((c) => c === 'pinned').length, 1, 'never refetched though 5 more loads passed keep:2');
});

test('sha1Js: the pure-JS SHA-1 matches the standard test vectors', () => {
  const enc = (s) => new TextEncoder().encode(s).buffer;
  assert.equal(sha1Js(enc('abc')), 'a9993e364706816aba3e25717850c26c9cd0d89d');
  assert.equal(sha1Js(new ArrayBuffer(0)), 'da39a3ee5e6b4b0d3255bfef95601890afd80709');
});

test('without crypto.subtle, blank detection falls back to sha1Js', async () => {
  const restore = Object.getOwnPropertyDescriptor(globalThis, 'crypto');
  Object.defineProperty(globalThis, 'crypto', { value: {}, configurable: true });
  try {
    const abc = new TextEncoder().encode('abc').buffer, blank = { bytes: 3, sha1: sha1Js(abc) };
    const fetchImpl = () => Promise.resolve({ ok: true, arrayBuffer: async () => abc });
    const L = createTileLoader({ fetchImpl, decode, blank }), log = [];
    L.request(job('a', 0, log)); await settle();
    assert.deepEqual(log[0], ['a', null, 'blank']);
  } finally { Object.defineProperty(globalThis, 'crypto', restore); }
});

test('keepIf holds a wanted URL through repeated evictions, and releases it once it is no longer wanted', async () => {
  const net = fakeNet(), held = new Set(['wanted']);
  const L = createTileLoader({ keep: 2, keepIf: (url) => held.has(url), fetchImpl: net.fetchImpl, decode }), log = [];
  L.request(job('wanted', 0, log)); net.reply('wanted'); await settle();
  for (let i = 0; i < 12; i++) {
    const u = `x${i}`; L.request(job(u, 0, log)); net.reply(u); await settle();
    assert.ok(L.peek('wanted'), `wanted still cached after eviction ${i}`);
  }
  held.delete('wanted');
  L.request(job('x12', 0, log)); net.reply('x12'); await settle();
  assert.equal(L.peek('wanted'), null, 'released once no longer held by keepIf');
});

test('prune drops every unwanted job on a shared entry exactly once, and keeps the entry queued for the wanted job', async () => {
  const net = fakeNet(), L = createTileLoader({ maxInFlight: 1, fetchImpl: net.fetchImpl, decode }), log = [];
  L.request(job('busy', 0, log));   // occupies the only in-flight slot, so 'a' below stays queued
  let w1 = true, w2 = true, w3 = true;
  L.request(job('a', 1, log, () => w1)); L.request(job('a', 2, log, () => w2)); L.request(job('a', 3, log, () => w3));
  assert.equal(L.queued, 1, 'one shared queue entry for the three attached jobs'); assert.equal(L.stats.deduped, 2);
  w1 = false; w2 = false; L.prune();
  const dropped = log.filter((l) => l[0] === 'a' && l[2] === 'dropped');
  assert.equal(dropped.length, 2, 'each unwanted job dropped exactly once'); assert.equal(L.queued, 1, 'the entry stays queued for the still-wanted job');
  net.reply('busy'); await settle(); net.reply('a'); await settle();
  const finishedA = log.filter((l) => l[0] === 'a');
  assert.equal(finishedA.length, 3, 'no job forgotten: 2 dropped earlier plus 1 ok now'); assert.deepEqual(finishedA[2], ['a', 'bitmap', 'ok']);
  assert.equal(L.inFlight, 0, 'no in-flight count leak');
});

test('suspend drops every job on a shared queued entry, once each, with no in-flight count leak', async () => {
  const net = fakeNet(), L = createTileLoader({ maxInFlight: 1, fetchImpl: net.fetchImpl, decode }), log = [];
  L.request(job('busy', 0, log));
  L.request(job('b', 1, log)); L.request(job('b', 2, log)); L.request(job('b', 3, log));
  assert.equal(L.queued, 1); assert.equal(L.stats.deduped, 2);
  L.suspend();
  const dropped = log.filter((l) => l[0] === 'b' && l[2] === 'dropped');
  assert.equal(dropped.length, 3, 'all three jobs dropped, once each'); assert.equal(L.queued, 0);
  net.reply('busy'); await settle();
  assert.deepEqual(log.find((l) => l[0] === 'busy'), ['busy', null, 'dropped'], 'the pre-suspend in-flight job resolves discarded');
  assert.equal(L.inFlight, 0, 'no in-flight count leak');
});

test('cross-host starvation: a saturated host does not block a lower-priority job on a free host', async () => {
  const net = fakeNet(), L = createTileLoader({ maxInFlight: 8, perHost: { default: 1 }, fetchImpl: net.fetchImpl, decode }), log = [];
  L.request(job('https://a.example/busy', 5, log));     // host A: takes its one slot
  L.request(job('https://a.example/urgent', 0, log));   // host A: higher priority (lower number), but A is saturated
  L.request(job('https://b.example/1', 9, log));         // host B: free; lower priority than 'urgent', runs anyway
  assert.deepEqual(net.calls, ['https://a.example/busy', 'https://b.example/1'], 'B is not starved by a higher-priority job stuck on a saturated host');
  assert.equal(L.inFlight, 2); assert.equal(L.queued, 1, 'only the A-urgent job is left queued');
});
