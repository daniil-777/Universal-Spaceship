import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { createRing } from '../vlm/capture/probe/ring.js';
import { fnv1a32, renderSeed, scanRun, writeEpisode, assemble, startGate, stopOf, writeStop, readStop, exitCodeOf, STOP_EXIT } from '../vlm/capture/drive.mjs';
import { shrink, SIZES } from '../vlm/capture/report.mjs';
import { createUpstream, StopDrive } from '../vlm/capture/tilecache.mjs';
import { withSpares, nextSpare } from '../vlm/capture/zplan.mjs';
import { zLocation, replaceDiscarded } from '../vlm/capture/families.mjs';
import { settleView, SLOT } from '../vlm/capture/episode_zoom.mjs';
import { groundOf, CAM_CLEAR_KM } from '../vlm/capture/probe/zoom.js';
import { createLocalFrame, cameraPose } from '../src/earthtiles.js';
import { fact } from '../vlm/gen/schema.js';

// scratch dirs live on LaCie (nothing on the Mac disk) when it is mounted, and are always removed
const ROOT = fileURLToPath(new URL('..', import.meta.url)), TMP = fs.existsSync('/Volumes/LaCie/astro-pilot/vlm/tmp') ? '/Volumes/LaCie/astro-pilot/vlm/tmp' : os.tmpdir();
async function withTmp(prefix, fn) { const d = fs.mkdtempSync(path.join(TMP, prefix)); try { return await fn(d); } finally { fs.rmSync(d, { recursive: true, force: true }); } }

test('REVIEW FOCUS 3: ring drops frames across a reset or a crash', () => {
  const r = createRing(3); assert.equal(r.push({ step: 10 }), null); assert.equal(r.push({ step: 13 }), null);
  assert.equal(r.push({ step: 2 }), null, 'a smaller step (env.reset after a crash) clears the ring'); assert.equal(r.size, 1);
  r.push({ step: 5 }); assert.equal(r.push({ step: 8, reset: true }), null); assert.equal(r.size, 1);
  r.push({ step: 11 }); const clip = r.push({ step: 14 }); assert.deepEqual(clip.map((f) => f.step), [8, 11, 14]);
});
test('render_seed depends only on run, family and episode seed (FNV-1a 32)', () => {
  assert.equal(fnv1a32('a'), 0xe40c292c); assert.equal(renderSeed('apv0', 'S', 101), renderSeed('apv0', 'S', 101)); assert.notEqual(renderSeed('apv0', 'S', 101), renderSeed('apv0', 'S', 102));
});
test('REVIEW FOCUS 4: resume scan removes unfinished episodes and counts only finished ones', () => withTmp('apv-raw-', (d) => {
  const png = Buffer.from('89504e47', 'hex');
  writeEpisode(d, [{ key: 'S_r_00001_000100', frames: ['S_r_00001_000100.f0.png', 'S_r_00001_000100.f1.png', 'S_r_00001_000100.f2.png'] }], { 'S_r_00001_000100.f0.png': png, 'S_r_00001_000100.f1.png': png, 'S_r_00001_000100.f2.png': png }, 1);
  fs.writeFileSync(path.join(d, 'S_r_00002_000050.f0.png'), png); fs.writeFileSync(path.join(d, 'S_r_00002_000050.json'), '{}');
  const s = scanRun(d);
  assert.equal(s.samples, 1); assert.deepEqual([...s.episodes], [1]); assert.deepEqual(s.removed.sort(), ['S_r_00002_000050.f0.png', 'S_r_00002_000050.json']);
  assert.ok(!fs.existsSync(path.join(d, 'S_r_00002_000050.f0.png')));
}));
test('R17 shrink: within budget the sizes stay; over budget the over-share families shrink first, never below 50 %', () => {
  const fast = { Z: 1, S: 1, A: 1, L: 1, D: 1 }, a = shrink(fast, 12);
  assert.deepEqual(a.sizes, SIZES); assert.ok(a.hours < 12);
  const slowZ = { Z: 40, S: 5, A: 5, L: 5, D: 5 }, b = shrink(slowZ, 12);
  assert.ok(b.hours <= 12, `projected ${b.hours} h`); assert.ok(b.sizes.Z < SIZES.Z && b.sizes.Z >= SIZES.Z / 2); assert.equal(b.sizes.S, SIZES.S); assert.equal(b.sizes.A, SIZES.A);
  const hopeless = { Z: 600, S: 600, A: 600, L: 600, D: 600 }, c = shrink(hopeless, 12);
  for (const f of Object.keys(SIZES)) assert.equal(c.sizes[f], Math.ceil(SIZES[f] / 2)); assert.ok(c.hours > 12);
});
test('Z steady state: the zoom rings\' per-frame easing (earthrings.js, dt = 16 ms) ends at one fixed point from any start', () => {
  const e = 1 - Math.exp(-(16 / 1000) * 3), settle = (x) => { for (let n = 0; n < 5000; n++) { const y = x + (1 - x) * e; if (y === x) return x; x = y; } return NaN; };
  const ends = [0, 0.123, 0.5, 0.9, 0.99, 0.995, 0.9999999].map(settle);
  assert.ok(ends.every((x) => x === ends[0]), 'a cold and a warm run reach the same blend and relief factors'); assert.equal(ends[0], 1 - 10 * 2 ** -53);
});
test('fix 1: a 403/429 StopDrive persists to stop.json and exits 3; a budget stop exits 0 and writes no stop', () => withTmp('apv-stop-', async (d) => {
  const eox = 'https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless_3857/default/g/9/1/2.jpg';
  const banned = createUpstream({ dir: path.join(d, 'c1'), fetchImpl: async () => new Response(null, { status: 429 }) }), broke = createUpstream({ dir: path.join(d, 'c2'), budgets: { 'tiles.maps.eox.at': 0 } });
  const e429 = await banned.get(eox).catch((e) => e), eBudget = await broke.get(eox).catch((e) => e); banned.close(); broke.close();
  assert.ok(e429 instanceof StopDrive && eBudget instanceof StopDrive);
  assert.deepEqual(stopOf(e429), { kind: 'http', host: 'tiles.maps.eox.at', status: 429 }); assert.equal(stopOf(eBudget).kind, 'budget');
  assert.equal(exitCodeOf(e429), STOP_EXIT); assert.equal(STOP_EXIT, 3); assert.equal(exitCodeOf(eBudget), 0);
  const run = path.join(d, 'run'); assert.equal(readStop(run), null); assert.equal(startGate(run).refuse, false);
  writeStop(run, stopOf(e429), 1790000000000); writeStop(run, { host: 'gibs.earthdata.nasa.gov', status: 403 }, 1790000001000);
  const s = readStop(run);
  assert.deepEqual({ host: s.host, status: s.status, utc_ms: s.utc_ms }, { host: 'tiles.maps.eox.at', status: 429, utc_ms: 1790000000000 }); assert.equal(s.stops.length, 2);
  const g = startGate(run); assert.equal(g.refuse, true); assert.equal(g.code, 3); assert.match(g.why, /429 from tiles\.maps\.eox\.at/);
}));
test('fix 1: drive.mjs refuses to start (exit 3, before any preflight or browser) while raw/<run>/stop.json exists', () => withTmp('apv-gate-', (d) => {
  const drive = (run) => spawnSync(process.execPath, [path.join(ROOT, 'vlm/capture/drive.mjs'), '--family', 'S', '--n', '1', '--run', run], { env: { ...process.env, APV_RAW_ROOT: d }, encoding: 'utf8', timeout: 60000 });
  writeStop(path.join(d, 'banned'), { host: 'tiles.maps.eox.at', status: 429 });
  const r = drive('banned'); assert.equal(r.status, 3, r.stderr); assert.match(r.stderr, /refusing to start: .*429 from tiles\.maps\.eox\.at/);
  const ok = drive('clean'); assert.equal(ok.status, 2, ok.stderr); assert.match(ok.stderr, /start gate passed; stopping before the preflight/);
}));
test('fix 2: spare Z locations are taken in order, same split first, once per discard, and a resume replays the order', () => withTmp('apv-spares-', (d) => {
  const L = [0, 1, 2, 3, 4].map((id) => ({ id, split: ['val', 'train', 'train', 'val', 'test'][id] })), { locations, spares } = withSpares(L, 2);
  assert.deepEqual(locations.map((l) => l.id), [0, 1]); assert.deepEqual(spares.map((l) => l.id), [2, 3, 4]);
  assert.equal(nextSpare(spares, new Set(), 'val'), 1); assert.equal(nextSpare(spares, new Set([1]), 'val'), 0); assert.equal(nextSpare(spares, new Set([0, 1, 2]), 'val'), null);
  const D = { dir: d, zplan: { locations, spares } };
  assert.equal(zLocation(D, 2), null, 'no spare runs before a discard');
  assert.equal(replaceDiscarded(D, 0), 1); assert.equal(zLocation(D, 2).id, 3, 'the val location is replaced by the val spare');
  assert.equal(replaceDiscarded(D, 0), null, 'a discard is replaced once');
  assert.equal(replaceDiscarded(D, 1), 0); assert.equal(zLocation(D, 3).id, 2);
  const R = { dir: d, zplan: { locations, spares } }; assert.deepEqual([2, 3, 4].map((e) => zLocation(R, e) && zLocation(R, e).id), [3, 2, null]);
}));
function zSample(ledger, window) {
  const D = { run: 'r', mode: 'clock', licence: 'open', gitSha: 'abc1234', ledger }, ep = { family: 'Z', episode: 0, seed: 0, url: '/vlm/capture/zoom.html?rs=1', renderSeed: 1, utcMs: 999 };
  const s = { step: 2, v: { range_bin: 0, utcMs: 1234 }, ledger0: 1, frame: { png: 'data:image/png;base64,iVBORw0KGgo=', cam: { mode: 'zoom' }, clock: { date_ms: 1, perf_ms: 2 }, ledger: window },
    label: { facts: { 'view.rings': fact([13, 12, 11, 10, 8], null, 'visual'), 'view.lat_deg': fact(10.42, 'deg', 'visual') }, lighting: { mode: 'utc', sun_elev_deg: 3.1, sun_az_deg: 290.1, class: 'golden' }, ground: { km: 1.1181, src: 'pose' } } };
  return assemble(D, ep, s);
}
test('fix 9: assemble (Z) keeps ring-level imagery from the page\'s ledger, discards on the view window only, passes ground and Sun time', () => {
  const t = (level, outcome, layer = 's2cloudless_3857', host = 'tiles.maps.eox.at') => ({ host, url: `u${level}`, layer_id: layer, level, outcome });
  const ledger = [t(10, 'cache hit'), t(8, 'cache hit', 'gibs', 'gibs.earthdata.nasa.gov'), t(9, 'aborted'), t(13, 'upstream fetch'), t(12, 'cache hit', 'terrarium', 's3.amazonaws.com'), t(12, 'rewritten'), t(14, 'cache hit')];
  const a = zSample(ledger, [3, 7]);
  assert.equal(a.error, null, 'the abort at index 2 belongs to an earlier view');
  assert.deepEqual(a.rec.render.imagery.map((m) => `${m.level}|${m.layer_id}`), ['8|gibs', '12|s2cloudless_3857', '13|s2cloudless_3857'], 'from ledger0 on, ring levels only, no Terrarium');
  assert.equal(a.rec.key, 'Z_r_00000_000002'); assert.deepEqual(a.rec.zoom.lighting, { mode: 'utc', sun_elev_deg: 3.1, sun_az_deg: 290.1, class: 'golden', utc_ms: 1234 });
  assert.equal(a.rec.zoom.ground_km, 1.1181); assert.equal(a.rec.zoom.ground_src, 'pose'); assert.equal(a.rec.provenance.utc_ms, 999);
  assert.match(zSample(ledger.concat([t(11, 'aborted')]), [3, 8]).error, /aborted tile/, 'an abort inside the view window discards it');
});
test('fix 9: groundOf reads zoom.info first, then the rendered pose (exact against cameraPose), and is null under the clearance lift', () => {
  const src = fs.readFileSync(path.join(ROOT, 'src/earthzoom.js'), 'utf8'); assert.equal(CAM_CLEAR_KM, +/camClearKm: ([\d.]+)/.exec(src)[1]);
  assert.deepEqual(groundOf({ groundKm: 0.5, clearKm: 3, rangeKm: 10, tilt: 0 }, { position: { y: 99 } }), { km: 0.5, src: 'info' });
  assert.deepEqual(groundOf({ clearKm: CAM_CLEAR_KM, rangeKm: 10, tilt: 0 }, { position: { y: 99 } }), { km: null, src: null });
  for (const v of [{ lat: 10.42, lon: -67.02, rangeKm: 19.18, tilt: 0.546, heading: 1.1, groundKm: 1.1181 }, { lat: 46.1, lon: 11.05, rangeKm: 619.2, tilt: 0, heading: 4, groundKm: 2.9 }]) {
    const pose = cameraPose(createLocalFrame(v.lat, v.lon), v), g = groundOf({ clearKm: 5, rangeKm: v.rangeKm, tilt: v.tilt }, { position: { y: pose.pos[1] } });
    assert.equal(g.src, 'pose'); assert.ok(Math.abs(g.km - v.groundKm) < 1e-9, `${g.km} vs ${v.groundKm}`);
  }
});
function fakeIO({ gateAt = 150, stableAfter = 628, lostAtEnd = false, msPerFrame = 0, stallAt = Infinity } = {}) {
  let n = 0;
  return { frame: async () => { if (n >= stallAt) throw new Error('stall'); n++; }, used: () => n, elapsedMs: () => n * msPerFrame,
    gate: async () => ({ ok: n >= gateAt && !(lostAtEnd && n >= SLOT), why: n >= gateAt ? ['lost'] : ['ring 13 incomplete'] }), steady: async () => Math.max(0, n - (gateAt + stableAfter)) };
}
test('fix 9: settleView captures at the end of the fixed slot and drops with a distinct reason for each failure', async () => {
  const io = fakeIO(), ok = await settleView(io);
  assert.deepEqual({ ok: ok.ok, why: ok.why, gate: ok.gateFrames, steady: ok.steadyFrames, used: io.used() }, { ok: true, why: null, gate: 150, steady: 630, used: SLOT });
  const early = fakeIO({ gateAt: 10 }), e = await settleView(early); assert.equal(e.gateFrames, 94, 'the 94 frames of §3.4 always run first'); assert.equal(early.used(), SLOT);
  const slot = fakeIO({ gateAt: Infinity }), s = await settleView(slot); assert.match(s.why, /^gate: the 1600-frame slot was used without the gate \(ring 13 incomplete\)/); assert.equal(slot.used(), SLOT - 1);
  const slow = fakeIO({ gateAt: Infinity, msPerFrame: 100 }), t = await settleView(slow); assert.match(t.why, /^gate timeout: 120 s of real time/); assert.equal(slow.used(), 1200);
  assert.match((await settleView(fakeIO({ stableAfter: 5000 }))).why, /^not steady within the 1600-frame slot/);
  assert.match((await settleView(fakeIO({ lostAtEnd: true }))).why, /^gate lost at the slot end/);
  await assert.rejects(settleView(fakeIO({ stallAt: 300 })), /stall/, 'a stall propagates to runZoomLocation, which logs it as its own reason');
});
