import { test } from 'node:test';
import assert from 'node:assert/strict';
import { corridorState, landingState, dockingState, sceneOf, rawOf, RAW_VERSION } from '../vlm/gen/labels/rawstate.js';
import { lookAtCamera } from '../vlm/gen/labels/camera.js';
import { assemble } from '../vlm/capture/records.mjs';
import { validateRecord } from '../vlm/gen/schema.js';
import { RWY, AIRPORT } from '../src/landing/airport.js';
import { STATION_PORT, SHIP_PORT } from '../vlm/gen/labels/docking.js';

const cam = { mode: 'chase', ...lookAtCamera({ eye: [-10, 2, 0], target: [10, 0, 0], fovDeg: 50, aspect: 896 / 504, near: 0.1, far: 9000 }) };
const f64 = (a) => Float64Array.from(a);
const env = { ship: { p: f64([0, 0, 0]), v: f64([12, 0, 0]), q: f64([0, 0, 0, 1]) }, nearest: [2],
  asteroids: [{ p: f64([30, 0, 0]), v: f64([0, 0, 0]), r: 2, kind: 0 }, { p: f64([-40, 0, 0]), v: f64([0, 1, 0]), r: 1, kind: 2 }, { p: f64([0, 0, 17]), v: f64([0, 0, 0]), r: 1.5, kind: 3 }] };

test('V1-2 S/A raw state: ship pose and every hazard in frame or among the nearest, with world centre, radius and kind', () => {
  const s = corridorState(env, cam);
  assert.deepEqual(s.ship.p, [0, 0, 0]); assert.deepEqual(s.ship.v, [12, 0, 0]); assert.deepEqual(s.ship.q, [0, 0, 0, 1]);
  const byI = Object.fromEntries(s.hazards.map((h) => [h.i, h]));
  assert.ok(byI[0], 'the hazard ahead is in frame'); assert.deepEqual(byI[0].c, [30, 0, 0]); assert.equal(byI[0].r, 2); assert.equal(byI[0].kind, 'rock'); assert.equal(byI[0].in_frame, true); assert.equal(byI[0].visible, true); assert.equal(byI[0].fogged, false);
  assert.equal(byI[1], undefined, 'a hazard behind the camera and not among the nearest is left out');
  assert.ok(byI[2], 'a nearest hazard is kept even out of frame'); assert.equal(byI[2].in_frame, false); assert.equal(byI[2].kind, 'airliner');
  assert.ok(Array.isArray(byI[0].box_px) && byI[0].box_px.length === 4);
});
test('V1-2 S/A raw state: hazard centres use the wrapped corridor x (the rendered position, as the facts project it)', () => {
  const e = { ...env, nearest: [0], asteroids: [{ p: f64([115, 0, 0]), v: f64([0, 0, 0]), r: 1, kind: 0 }] };
  assert.deepEqual(corridorState(e, cam).hazards[0].c, [-5, 0, 0], 'x 115 wraps to -5 in the 120 u corridor');
});
test('V1-2 L and D raw state and scene geometry are plain finite arrays', () => {
  const flight = { p: f64([-3000, 150, 2]), v: f64([70, -3.6, 0]), q: f64([0, 0.1, 0, 0.995]), w: f64([0, 0, 0]), wow: false, gear: 1 };
  const l = landingState({ flight }, 0.0004);
  assert.deepEqual(l.aircraft.p, [-3000, 150, 2]); assert.equal(l.aircraft.wow, false); assert.equal(l.fog_density, 0.0004);
  const s = { x: f64([-40, 1, 0.5, 0.1, 0, 0]), q: f64([0, 0, 0, 1]) }, d = dockingState(s);
  assert.deepEqual(d.ship.x, [-40, 1, 0.5]); assert.deepEqual(d.ship.v, [0.1, 0, 0]); assert.deepEqual(d.ship_port_w, [-40 + SHIP_PORT[0], 1 + SHIP_PORT[1], 0.5].map((x) => +x.toFixed(4)));
  const L = sceneOf('L'), { corners, ...rw } = L.runway; assert.deepEqual(rw, { ...RWY }); assert.equal(corners.length, 4); assert.equal(L.papi.length, AIRPORT.papi.length); assert.deepEqual(L.windsock, AIRPORT.windsock);
  assert.deepEqual(L.runway.corners[2], [RWY.length, RWY.elevation, RWY.width / 2]); assert.deepEqual(d.port_rel.length, 3); assert.equal(d.att_err_rad.length, 3);
  const D = sceneOf('D'); assert.deepEqual(D.station_port, [...STATION_PORT]); assert.deepEqual(D.ship_port, [...SHIP_PORT]);
  assert.equal(sceneOf('S'), null); assert.equal(sceneOf('Z'), null);
});
test('V1-2 records carry snapshot.frames keyed by frame name (plus the chase frame) and still validate; old-style frames give no raw block', () => {
  const png = 'data:image/png;base64,iVBORw0KGgo=', st = (k) => ({ aircraft: { p: [k, 0, 0] } }), clock = { date_ms: 1, perf_ms: 1 };
  const D0 = { run: 'r', mode: 'clock', licence: 'open', gitSha: 'abc', siteDirty: false, ledger: [] };
  const ep = { family: 'L', episode: 3, seed: 5, url: '/x', renderSeed: 1, utcMs: 0 };
  const facts = { mode: { v: 'LAND', unit: null, obs: 'context' } }, safety = { verdict: 'SAFE', severity: 0, reasons: [], risk: {}, outcome: {}, best: 'CONTINUE' };
  const frames = [0, 1, 2].map((k) => ({ png, cam, clock, step: 100 + 30 * k, ledger: [0, 0], state: st(k) }));
  const r = assemble(D0, ep, { step: 160, n: [30, 30], frames, label: { facts, safety, safety_eye: safety } }, { eyeView: 'chase' }).rec;
  assert.equal(r.snapshot.v, RAW_VERSION); assert.deepEqual(Object.keys(r.snapshot.frames), r.frames); assert.deepEqual(r.snapshot.frames[r.frames[2]], st(2)); assert.deepEqual(r.snapshot.scene, sceneOf('L'));
  const old = assemble(D0, ep, { step: 160, n: [30, 30], frames: frames.map(({ state, ...f }) => f), label: { facts, safety, safety_eye: safety } }, { eyeView: 'chase' }).rec;
  assert.equal(old.snapshot, undefined, 'no state, no snapshot block (v0 layout unchanged)');
  const dD = assemble(D0, { ...ep, family: 'D' }, { step: 160, n: [20, 20], frames, chase: { png, cam, state: st(9) }, label: { facts, safety, safety_eye: safety } }, { eyeView: 'centreline' }).rec;
  assert.deepEqual(dD.snapshot.frames[dD.narrator_frame], st(9), 'the D chase frame has its own state');
  assert.equal(rawOf('Z', [], null), null);
});
test('V1-2 raw state never carries a non-finite number (validateRecord rejects them)', () => {
  const s = corridorState({ ...env, ship: { ...env.ship, v: f64([NaN, Infinity, 1]) } }, cam);
  assert.deepEqual(s.ship.v, [null, null, 1]);
  const errs = []; JSON.stringify(s, (k, v) => { if (typeof v === 'number' && !Number.isFinite(v)) errs.push(k); return v; }); assert.deepEqual(errs, []);
  assert.equal(typeof validateRecord, 'function');
});
