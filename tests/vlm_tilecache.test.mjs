import { test, after } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { openTileCache, createUpstream, StopDrive } from '../vlm/capture/tilecache.mjs';

const dirs = [], tmp = () => { const d = fs.mkdtempSync(path.join(os.tmpdir(), 'apv-cache-')); dirs.push(d); return d; };
after(() => { for (const d of dirs) fs.rmSync(d, { recursive: true, force: true }); });
const jpg = Buffer.concat([Buffer.from([0xff, 0xd8, 0xff]), Buffer.alloc(500, 7)]);
const res = (status, body, type, extra = {}) => ({ status, headers: { get: (k) => ({ 'content-type': type, 'content-length': String(extra.len ?? body.length) })[k] ?? null }, arrayBuffer: async () => body.buffer.slice(body.byteOffset, body.byteOffset + body.length) });
const U = 'https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless_3857/default/g/9/170/268.jpg';
test('put/get round trip in WAL mode, one file per host', () => {
  const d = tmp(), c = openTileCache(d, 'tiles.maps.eox.at'); c.put(U, jpg, 'image/jpeg');
  const g = c.get(U); assert.deepEqual(Buffer.from(g.body), jpg); assert.equal(g.ctype, 'image/jpeg'); assert.equal(c.count(), 1); c.close();
  assert.ok(fs.existsSync(path.join(d, 'tiles.maps.eox.at.sqlite')));
});
test('upstream: a miss fetches once and caches; the next get is a hit', async () => {
  let n = 0; const up = createUpstream({ dir: tmp(), contact: 'test@example.invalid', fetchImpl: async () => { n++; return res(200, jpg, 'image/jpeg'); } });
  assert.equal((await up.get(U)).outcome, 'fetch'); const h = await up.get(U); assert.equal(h.outcome, 'hit'); assert.equal(n, 1); assert.equal(h.status, 200);
});
test('REVIEW FOCUS 5: upstream: a 200 HTML body or a short body is not cached', async () => {
  let n = 0; const up = createUpstream({ dir: tmp(), contact: 'x', fetchImpl: async (u) => { n++; return u.endsWith('a.jpg') ? res(200, Buffer.from('<html>captcha</html>'), 'text/html') : res(200, jpg.subarray(0, 100), 'image/jpeg', { len: 5000 }); } });
  for (const u of [U.replace('268.jpg', 'a.jpg'), U.replace('268.jpg', 'b.jpg')]) { const r = await up.get(u); assert.equal(r.outcome, 'fetch-error'); assert.equal(r.status, 502); assert.equal((await up.get(u)).outcome, 'fetch-error'); }
  assert.equal(n, 4);
});
test('403/429 and the EOX budget stop the drive; at most 4 requests in flight per host', async () => {
  await assert.rejects(createUpstream({ dir: tmp(), contact: 'x', fetchImpl: async () => res(429, Buffer.alloc(0), 'text/plain') }).get(U), StopDrive);
  const up = createUpstream({ dir: tmp(), contact: 'x', budgets: { 'tiles.maps.eox.at': 2 }, fetchImpl: async () => res(200, jpg, 'image/jpeg') });
  await up.get(U + '?1'); await up.get(U + '?2'); await assert.rejects(up.get(U + '?3'), /budget/);
  let live = 0, peak = 0; const slow = createUpstream({ dir: tmp(), contact: 'x', maxPerSec: 1000, fetchImpl: async () => { live++; peak = Math.max(peak, live); await new Promise((r) => setTimeout(r, 30)); live--; return res(200, jpg, 'image/jpeg'); } });
  await Promise.all(Array.from({ length: 12 }, (_, i) => slow.get(U + `?p${i}`))); assert.ok(peak <= 4, `peak ${peak}`);
});
test('upstream: a network error is a fetch-error, never a crash', async () => {
  const up = createUpstream({ dir: tmp(), contact: 'x', fetchImpl: async () => { throw new TypeError('fetch failed'); } }); const r = await up.get(U); assert.equal(r.outcome, 'fetch-error'); assert.equal(r.status, 502);
});

// Beyond the brief: the WAL file on disk, the signature check, compressed bodies, in-flight sharing, the budget under
// concurrency, the per-second limit and the optional contact.
const withEnc = (r, enc) => ({ ...r, headers: { get: (k) => (k === 'content-encoding' ? enc : r.headers.get(k)) } });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
test('the cache file is in WAL mode on disk while open', () => {
  const d = tmp(), c = openTileCache(d, 'gibs.earthdata.nasa.gov'); c.put('https://gibs.earthdata.nasa.gov/x.jpeg', jpg, 'image/jpeg');
  assert.ok(fs.existsSync(path.join(d, 'gibs.earthdata.nasa.gov.sqlite-wal'))); assert.equal(c.get('https://gibs.earthdata.nasa.gov/y.jpeg'), null); c.close();
});
test('REVIEW FOCUS 5: a 200 image/* tile whose bytes are not a JPEG or PNG is not cached', async () => {
  let n = 0; const html = Buffer.from('<!DOCTYPE html><html><body>rate limited</body></html>'.padEnd(600, ' '));
  const up = createUpstream({ dir: tmp(), fetchImpl: async () => { n++; return res(200, html, 'image/jpeg'); } });
  const r = await up.get(U); assert.deepEqual([r.outcome, r.status, r.body.length], ['fetch-error', 502, 0]);
  assert.equal((await up.get(U)).outcome, 'fetch-error'); assert.equal(n, 2); assert.equal(up.stats().byHost['tiles.maps.eox.at'].error, 2);
  const png = Buffer.concat([Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]), Buffer.alloc(300, 1)]), T = 'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/9/268/170.png';
  const p = createUpstream({ dir: tmp(), fetchImpl: async () => res(200, png, 'image/png') }); assert.equal((await p.get(T)).outcome, 'fetch'); assert.equal((await p.get(T)).outcome, 'hit');
});
test('a compressed text body is not a length mismatch; the cache persists across upstream instances', async () => {
  const d = tmp(), js = Buffer.from('export const x = 1;\n'.repeat(50)), T = 'https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js';
  const a = createUpstream({ dir: d, fetchImpl: async () => withEnc(res(200, js, 'application/javascript', { len: 97 }), 'br') });
  assert.equal((await a.get(T)).outcome, 'fetch'); a.close();
  const b = createUpstream({ dir: d, fetchImpl: async () => { throw new Error('must not fetch'); } });
  const h = await b.get(T); assert.equal(h.outcome, 'hit'); assert.deepEqual(h.body, js); b.close();
  const c = createUpstream({ dir: tmp(), fetchImpl: async () => res(200, js, 'application/javascript', { len: 97 }) }); assert.equal((await c.get(T)).outcome, 'fetch-error', 'an uncompressed short body is truncated');
});
test('concurrent gets of one URL share one upstream request; the budget is reserved before queueing', async () => {
  let n = 0; const up = createUpstream({ dir: tmp(), fetchImpl: async () => { n++; await sleep(20); return res(200, jpg, 'image/jpeg'); } });
  const rs = await Promise.all(Array.from({ length: 5 }, () => up.get(U))); assert.equal(n, 1); assert.ok(rs.every((r) => r.status === 200 && r.body.length === jpg.length));
  let m = 0; const b = createUpstream({ dir: tmp(), budgets: { 'tiles.maps.eox.at': 2 }, maxPerSec: 1000, fetchImpl: async () => { m++; await sleep(20); return res(200, jpg, 'image/jpeg'); } });
  const out = await Promise.allSettled(Array.from({ length: 6 }, (_, i) => b.get(U + `?b${i}`)));
  assert.equal(m, 2); assert.equal(out.filter((o) => o.status === 'rejected' && o.reason instanceof StopDrive).length, 4);
});
test('at most maxPerSec requests sent per host in any 1 s window, even after the event loop was blocked', async () => {
  const starts = []; const up = createUpstream({ dir: tmp(), maxPerSec: 3, fetchImpl: async (u) => { starts.push([new URL(u).host, Date.now()]); return res(200, jpg, 'image/jpeg'); } });
  const G = 'https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/BlueMarble_NextGeneration/default/GoogleMapsCompatible_Level8/3/2/4.jpeg';
  const all = [...Array.from({ length: 5 }, (_, i) => up.get(U + `?r${i}`)), up.get(G)];
  const t0 = Date.now();
  while (Date.now() - t0 < 1100);
  await Promise.all(all);
  const eox = starts.filter(([h]) => h === 'tiles.maps.eox.at').map(([, t]) => t), gibs = starts.find(([h]) => h === 'gibs.earthdata.nasa.gov')[1];
  assert.equal(eox.length, 5); assert.ok(gibs < eox[3], 'another host is not held back');
  for (let i = 0; i + 3 < eox.length; i++) assert.ok(eox[i + 3] - eox[i] >= 990, `send ${i + 4} came ${eox[i + 3] - eox[i]} ms after send ${i + 1}`);
});
test('the User-Agent carries a contact only when one is given (APV_CONTACT is optional)', async () => {
  const seen = [], f = async (u, o) => { seen.push(o.headers['user-agent']); return res(200, jpg, 'image/jpeg'); };
  await createUpstream({ dir: tmp(), fetchImpl: f }).get(U); await createUpstream({ dir: tmp(), contact: 'ops@example.invalid', fetchImpl: f }).get(U);
  assert.ok(seen[0].startsWith('AstroPilotVision/') && !/contact/.test(seen[0]), seen[0]); assert.match(seen[1], /contact ops@example\.invalid/);
});

// Fix round 1
const G0 = 'https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/BlueMarble_NextGeneration/default/GoogleMapsCompatible_Level8/3/2/';
test('IMPORTANT: a 403 or 429 latches the host: queued and later gets reject before sending, the body is cancelled', async () => {
  for (const code of [429, 403]) {
    let n = 0, cancelled = 0;
    const up = createUpstream({ dir: tmp(), fetchImpl: async (u) => { if (u.startsWith(G0)) return res(200, jpg, 'image/jpeg'); n++; await sleep(10); return { ...res(code, Buffer.alloc(0), 'text/html'), body: { cancel: async () => { cancelled++; } } }; } });
    const out = await Promise.allSettled(Array.from({ length: 20 }, (_, i) => up.get(U + `?l${i}`)));
    assert.deepEqual([n, cancelled], [1, 1], `${code}: upstream requests and cancelled bodies`); assert.ok(out.every((o) => o.status === 'rejected' && o.reason instanceof StopDrive));
    await assert.rejects(up.get(U + '?later'), StopDrive); assert.equal(n, 1); assert.equal((await up.get(G0 + '4.jpeg')).outcome, 'fetch', 'another host is not latched');
  }
});
test('each upstream request carries a timeout signal; a timed-out request is a fetch-error', async () => {
  const keep = setTimeout(() => {}, 5000); let sig = null;
  const up = createUpstream({ dir: tmp(), timeoutMs: 50, fetchImpl: (u, o) => { sig = o.signal; return new Promise((_, rej) => o.signal.addEventListener('abort', () => rej(o.signal.reason))); } });
  const t = Date.now(), r = await up.get(U); clearTimeout(keep);
  assert.ok(sig instanceof AbortSignal && sig.aborted); assert.deepEqual([r.outcome, r.status], ['fetch-error', 502]); assert.ok(Date.now() - t < 2000);
});
test('budgets merge with the EOX default, a resumed run passes the requests already used, and the fetch counts agree', async () => {
  let n = 0; const f = async (u) => { n++; return u.includes('bad') ? res(200, Buffer.from('<html></html>'), 'text/html') : res(200, jpg, 'image/jpeg'); };
  const up = createUpstream({ dir: tmp(), budgets: { 'gibs.earthdata.nasa.gov': 2 }, used: { 'tiles.maps.eox.at': 79999 }, fetchImpl: f });
  assert.equal((await up.get(U)).outcome, 'fetch'); await assert.rejects(up.get(U + '?2'), /budget of 80000/);
  assert.equal((await up.get(G0 + '1.jpeg')).outcome, 'fetch'); assert.equal((await up.get(G0 + 'bad.jpeg')).outcome, 'fetch-error'); await assert.rejects(up.get(G0 + '3.jpeg'), /budget of 2 /);
  const s = up.stats(); assert.equal(s.fetch, n); assert.equal(s.fetch, Object.values(s.byHost).reduce((a, h) => a + h.fetch, 0));
  assert.deepEqual([s.byHost['tiles.maps.eox.at'].used, s.byHost['gibs.earthdata.nasa.gov'].used, s.error, s.hit], [80000, 2, 1, 0]);
});
