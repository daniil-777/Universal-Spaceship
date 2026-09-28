import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createTileLoader } from '../src/earthloader.js';

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

test('suspend stops a closed view fetching: the queue is dropped, running answers are discarded, nothing new starts until resume', async () => {
  const net = fakeNet(), L = createTileLoader({ maxInFlight: 1, fetchImpl: net.fetchImpl, decode }), log = [];
  L.request(job('a', 0, log)); L.request(job('b', 1, log)); L.suspend();
  assert.deepEqual(log, [['b', null, 'dropped']]); assert.equal(L.queued, 0);
  net.reply('a'); await settle(); assert.deepEqual(log[1], ['a', null, 'dropped']);
  L.request(job('c', 0, log)); assert.deepEqual(log[2], ['c', null, 'dropped']); assert.deepEqual(net.calls, ['a']);
  L.resume(); L.request(job('c', 0, log)); assert.deepEqual(net.calls, ['a', 'c']);
});
