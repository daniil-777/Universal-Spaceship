// chat/tests/launcher.test.mjs — chat/web/launcher.js's pure parts: the scene from the page URL and from the main app's Playbox state,
// the per-tab seen list, ?capcom config resolution, the kb probe, the iframe URL, and the button's free-spot search.
//   node --test chat/tests/launcher.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import { sceneFromUrl, sceneFromDom, parseSeen, addSeen, seenFor, resolveConfig, pickKb, frameUrl, freeSpot, blocked, CONFIG, KEYS } from '../web/launcher.js';
import { HIGHLIGHTS } from '../web/prompt.js';

test('scene from the URL flags: scenarios, atmospheric flight, moon, training, default belt', () => {
  const cases = {
    '': 'belt', '?scenario=landing': 'landing', '?scenario=real': 'docking', '?real=1': 'docking', '?scenario=landing&atmo=1': 'landing',
    '?orbit=moon': 'moon', '?atmo=1': 'atmo', '?atmo=1&route=alps': 'alps', '?atmo=1&route=china': 'pillars', '?atmo=1&sky=storm': 'weather',
    '?route=dubai': 'cities', '?route=newyork&atmo=1': 'cities', '?skyline=1': 'cities', '?route=dubai&skyline=0': 'belt',
    '?route=dubai&skyline=0&atmo=1': 'atmo', '?route=alps': 'belt', '?orbit=moon&atmo=1': 'atmo', '?orbit=moon&route=london': 'cities',
    '?train=1': 'training', '?scenario=nope': 'belt', '?hud=0&capcom=1': 'belt',
  };
  for (const [q, want] of Object.entries(cases)) assert.equal(sceneFromUrl(q), want, q);
  for (const q of Object.keys(cases)) assert.ok(KEYS.includes(sceneFromUrl(q)), `${q} → a highlight key`);
});

test('scene from the main app state refines the URL; scenarios and pages without the Playbox keep the URL scene', () => {
  const d = { zooming: false, atmo: false, skyline: false, route: 'alps', sky: 'fair', orbit: 'earth' };
  assert.equal(sceneFromDom(null, 'moon'), 'moon');
  assert.equal(sceneFromDom(d, 'landing'), 'landing');
  assert.equal(sceneFromDom({ ...d, atmo: true }, 'docking'), 'docking');
  assert.equal(sceneFromDom(d, 'belt'), 'belt');
  assert.equal(sceneFromDom(d, 'atmo'), 'belt', 'the visitor switched atmospheric flight off');
  assert.equal(sceneFromDom(d, 'training'), 'training');
  assert.equal(sceneFromDom({ ...d, zooming: true, atmo: true }, 'belt'), 'zoom');
  assert.equal(sceneFromDom({ ...d, atmo: true }, 'belt'), 'atmo');
  assert.equal(sceneFromDom({ ...d, atmo: true }, 'alps'), 'alps');
  assert.equal(sceneFromDom({ ...d, atmo: true, route: 'china' }, 'belt'), 'pillars');
  assert.equal(sceneFromDom({ ...d, atmo: true, skyline: true, route: 'dubai' }, 'belt'), 'cities');
  assert.equal(sceneFromDom({ ...d, atmo: true, sky: 'storm' }, 'belt'), 'weather');
  assert.equal(sceneFromDom({ ...d, orbit: 'moon' }, 'belt'), 'moon');
});

test('seen list: highlight keys only, first-seen order, no repeats; the state line gets everything but the scene', () => {
  assert.deepEqual(parseSeen('["belt","landing","belt"]'), ['belt', 'landing']);
  assert.deepEqual(parseSeen('belt, moon,,evil<script>,moon'), ['belt', 'moon']);
  assert.deepEqual(parseSeen(['docking', 'x', 'atmo']), ['docking', 'atmo']);
  assert.deepEqual(parseSeen('{"a":1}'), []);
  assert.deepEqual(parseSeen(null), []);
  assert.deepEqual(parseSeen(''), []);
  let s = [];
  for (const k of ['belt', 'landing', 'belt', 'nope', 'docking']) s = addSeen(s, k);
  assert.deepEqual(s, ['belt', 'landing', 'docking']);
  assert.deepEqual(seenFor(s, 'landing'), ['belt', 'docking']);
  assert.deepEqual(seenFor(s, 'moon'), s);
  const all = HIGHLIGHTS.reduce((a, [k]) => addSeen(a, k), []);
  assert.equal(all.length, HIGHLIGHTS.length);
});

test('config: ?capcom=0 disables, ?capcom=<folder|Hub id> picks the model, defaults otherwise; device/dtype validated', () => {
  for (const q of ['?capcom=0', '?capcom=off', '?capcom=false', '?capcom=NO']) assert.equal(resolveConfig(q), null, q);
  const d = resolveConfig('');
  assert.equal(d.model, CONFIG.model);
  assert.equal(d.model, '/__capcom/models/capcom-pilot-web-acc1');
  assert.deepEqual(d.kbs, CONFIG.kb);
  assert.match(d.kbs[0], /\/chat\/kb\/kb\.json$/);
  assert.equal(d.kbs[1], '/__capcom/kb/kb.json');
  assert.deepEqual([d.device, d.dtype, d.debug], ['auto', null, false]);
  assert.equal(resolveConfig('?capcom=1').model, CONFIG.model);
  assert.equal(resolveConfig('?capcom=onnx-community/LFM2-350M-ONNX').model, 'onnx-community/LFM2-350M-ONNX');
  assert.equal(resolveConfig('?capcom=' + encodeURIComponent('https://cdn.example.org/m/capcom/')).model, 'https://cdn.example.org/m/capcom/');
  assert.deepEqual(resolveConfig('?capcomkb=/kb2.json').kbs, ['/kb2.json']);
  assert.deepEqual(resolveConfig('', CONFIG, '127.0.0.1').kbs, [CONFIG.kb[1], CONFIG.kb[0]], 'local dev: the LaCie kb first');
  assert.deepEqual(resolveConfig('', CONFIG, 'localhost').kbs[0], '/__capcom/kb/kb.json');
  assert.deepEqual(resolveConfig('', CONFIG, 'daniil-777.github.io').kbs, CONFIG.kb, 'a static host: the repo copy first');
  assert.deepEqual(resolveConfig('?capcomkb=/kb2.json', CONFIG, '127.0.0.1').kbs, ['/kb2.json']);
  const c = resolveConfig('?capcomdevice=none&capcomdtype=q4&capcomdebug=1');
  assert.deepEqual([c.device, c.dtype, c.debug], ['none', 'q4', true]);
  const bad = resolveConfig('?capcomdevice=gpu&capcomdtype=fp32');
  assert.deepEqual([bad.device, bad.dtype], ['auto', null]);
});

test('kb probe: the first candidate answering HEAD, else the last', async () => {
  const calls = [], fake = (ok) => async (u, o) => { calls.push([u, o.method]); return { ok: ok.includes(u) }; };
  assert.equal(await pickKb(['a', 'b'], fake(['a'])), 'a');
  assert.deepEqual(calls, [['a', 'HEAD']]);
  assert.equal(await pickKb(['a', 'b'], fake([])), 'b');
  assert.equal(await pickKb(['a', 'b'], async () => { throw new Error('offline'); }), 'b');
  assert.equal(await pickKb(['only'], fake([])), 'only');
});

test('iframe URL carries embed, scene, seen, model, kb, device (+ dtype, debug)', () => {
  const u = new URL(frameUrl({ model: '/__capcom/models/m', device: 'auto', dtype: null, debug: false }, '/__capcom/kb/kb.json', 'landing', ['belt', 'atmo'], 'http://x/chat/web/demo.html'));
  assert.equal(u.pathname, '/chat/web/demo.html');
  const p = Object.fromEntries(u.searchParams);
  assert.deepEqual(p, { embed: '1', scene: 'landing', seen: 'belt,atmo', model: '/__capcom/models/m', kb: '/__capcom/kb/kb.json', device: 'auto' });
  const v = new URL(frameUrl({ model: 'a&b=c', device: 'none', dtype: 'q4', debug: true }, 'k', 'belt', [], 'http://x/d.html')).searchParams;
  assert.deepEqual([v.get('model'), v.get('dtype'), v.get('debug'), v.get('seen')], ['a&b=c', 'q4', '1', '']);
  assert.match(CONFIG.demo, /\/chat\/web\/demo\.html$/);
});

test('free spot: bottom-right when clear, else beside or above the HUD boxes, never over them', () => {
  const R = (left, top, right, bottom) => ({ left, top, right, bottom });
  const over = (s, w, h, obs, vw, vh) => obs.some((o) => vw - s.right - w < o.right && vw - s.right > o.left && vh - s.bottom - h < o.bottom && vh - s.bottom > o.top);
  assert.deepEqual(freeSpot(120, 40, [], 1280, 860), { right: 22, bottom: 22 });
  // the main app at 1280×860 (measured): Playbox, flight board, credit line, HUD → left of the board, on the bottom row
  const main = [R(946, 76, 1258, 632), R(719, 629, 1258, 820), R(808, 828, 1258, 844), R(22, 718, 563, 838), R(0, 0, 1280, 82)];
  const s1 = freeSpot(120, 40, main, 1280, 860);
  assert.equal(s1.bottom, 22); assert.ok(!over(s1, 120, 40, main, 1280, 860)); assert.ok(1280 - s1.right <= 719);
  // the landing (ATC radio box bottom-right, nav strip bottom-left) → between them
  const landing = [R(16, 730, 453, 844), R(704, 14, 1264, 93), R(834, 792, 1264, 844)];
  const s2 = freeSpot(120, 40, landing, 1280, 860);
  assert.equal(s2.bottom, 22); assert.ok(!over(s2, 120, 40, landing, 1280, 860));
  // a phone with the docking bar across the bottom → above it
  const phone = [R(16, 750, 374, 828), R(16, 610, 274, 828), R(16, 14, 374, 76)];
  const s3 = freeSpot(46, 46, phone, 390, 844, 16);
  assert.ok(s3.bottom > 844 - 750); assert.ok(!over(s3, 46, 46, phone, 390, 844));
  assert.equal(freeSpot(46, 46, [R(0, 0, 390, 844)], 390, 844, 16), null, 'nothing free');
  // a parked spot stays until a box covers it or it falls off screen
  assert.equal(blocked(s1, 120, 40, main, 1280, 860), false);
  assert.equal(blocked({ right: 22, bottom: 22 }, 120, 40, main, 1280, 860), true);
  assert.equal(blocked({ right: 2000, bottom: 22 }, 120, 40, [], 1280, 860), true);
});
