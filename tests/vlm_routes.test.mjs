import { test } from 'node:test';
import assert from 'node:assert/strict';
import { classify, installRoutes, imageryFrom, discardReason, ledgerCounts } from '../vlm/capture/routes.mjs';

const o = { port: 5555, profile: 'open' };
test('default-deny classification per spec §6', () => {
  assert.equal(classify('http://127.0.0.1:5555/index.html', o).action, 'continue');
  assert.equal(classify('https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js', o).action, 'cache');
  assert.equal(classify('https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/BlueMarble_NextGeneration/default/GoogleMapsCompatible_Level8/8/100/200.jpeg', o).layer, 'gibs');
  const esri = classify('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/12/1400/2130', o);
  assert.equal(esri.action, 'rewrite'); assert.equal(esri.upstream, 'https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless_3857/default/g/12/1400/2130.jpg');
  assert.equal(classify('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/12/1400/2130', { ...o, profile: 'nc' }).layer, 's2cloudless-2025_3857');
  assert.equal(classify('https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2025_3857/default/g/9/1/2.jpg', o).upstream, 'https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless_3857/default/g/9/1/2.jpg');
  assert.equal(classify('https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2021_3857/default/g/9/1/2.jpg', { ...o, profile: 'nc' }).action, 'abort');
  assert.equal(classify('https://example.com/x.js', o).action, 'abort'); assert.equal(classify('http://127.0.0.1:8791/', o).action, 'abort');
});
test('installRoutes: Esri is fulfilled from EOX and never continued; unknown hosts are aborted; imagery[] comes from served layers', async () => {
  let handler; const ctx = { route: async (_, h) => { handler = h; } }, ledger = [], asked = [];
  const upstream = { get: async (u) => { asked.push(u); return { status: 200, body: Buffer.from([1]), ctype: 'image/jpeg', outcome: 'fetch' }; } };
  await installRoutes(ctx, { port: 5555, profile: 'open', upstream, ledger, stop: {} });
  const run = async (url) => { const r = { did: null, request: () => ({ url: () => url }), continue: async () => { r.did = 'continue'; }, abort: async () => { r.did = 'abort'; }, fulfill: async (x) => { r.did = 'fulfill'; r.x = x; } }; await handler(r); return r; };
  const e = await run('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/9/170/268');
  assert.equal(e.did, 'fulfill'); assert.equal(e.x.headers['access-control-allow-origin'], '*'); assert.ok(asked[0].startsWith('https://tiles.maps.eox.at/'));
  assert.equal((await run('https://evil.example/tile.png')).did, 'abort');
  assert.equal((await run('http://127.0.0.1:5555/src/app.js')).did, 'continue');
  assert.deepEqual(imageryFrom(ledger, 0, ledger.length), [{ level: 9, layer_id: 's2cloudless_3857', source: 'EOX Sentinel-2 cloudless', year: 2016, licence: 'CC BY 4.0' }]);
  assert.equal(discardReason(ledger, 0, 1, 'open'), null);
  assert.match(discardReason([{ host: 'tiles.maps.eox.at', outcome: 'aborted' }], 0, 1, 'open'), /aborted tile/);
});
test('a rewrite whose upstream fetch failed is a fetch error: never imagery, always a discard; Esri never reaches the network', async () => {
  let handler; const ctx = { route: async (_, h) => { handler = h; } }, ledger = [];
  const upstream = { get: async (u) => (u.includes('/9/') ? { status: 502, body: Buffer.alloc(0), ctype: 'text/plain', outcome: 'fetch-error' } : { status: 200, body: Buffer.from([1]), ctype: 'image/jpeg', outcome: 'fetch' }) };
  await installRoutes(ctx, { port: 5555, profile: 'open', upstream, ledger, stop: {} });
  const run = async (url) => { const r = { request: () => ({ url: () => url }), continue: async () => {}, abort: async () => {}, fulfill: async () => {} }; await handler(r); };
  await run('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/9/170/268');
  await run('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/10/340/536');
  assert.equal(ledger[0].outcome, 'fetch error'); assert.equal(ledger[1].outcome, 'rewritten');
  assert.deepEqual(imageryFrom(ledger, 0, 2).map((m) => m.level), [10]);
  assert.match(discardReason(ledger, 0, 2, 'open'), /aborted tile/);
  const c = ledgerCounts(ledger);
  assert.equal(c.esriReachedNetwork, 0); assert.deepEqual(c.upstreamFetchesByHost, { 'tiles.maps.eox.at': 1 }); assert.deepEqual(c.byHost['server.arcgisonline.com'], { 'fetch error': 1, rewritten: 1 });
});
