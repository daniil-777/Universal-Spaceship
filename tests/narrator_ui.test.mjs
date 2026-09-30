// The in-site Narrator's pure parts (src/narrator): the streaming sentence gate, the pill and chip words, the placement
// solver, family / clock / episode mapping, verdict hysteresis, the 16:9 crop, and a boot that stays inert until asked.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createSentenceGate, jaccard, words, CAPS } from '../src/narrator/clean.js';
import { pillLabel, contextChip, skyName, humanTemplate, fmtKm } from '../src/narrator/words.js';
import { placeBox } from '../src/narrator/place.js';
import { familyOf, clockOf, createEpisodes, createHysteresis, centerCrop, factsOf, frameDtOf } from '../src/narrator/feed.js';
import { installBoot } from '../src/narrator/boot.js';

const feedAll = (gate, text, step = 3) => { const out = []; let stop = false; for (let i = 0; i < text.length && !stop; i += step) { const r = gate.push(text.slice(i, i + step)); out.push(...r.add); stop = r.stop; } return { out, stop }; };

test('clean: words and Jaccard ignore case and punctuation', () => {
  assert.deepEqual(words('The Nearest feature, is a town.'), ['the', 'nearest', 'feature', 'is', 'a', 'town']);
  assert.equal(jaccard('The nearest ground is a town.', 'The nearest ground is a city.'), 5 / 7);
  assert.equal(jaccard('', ''), 0);
  assert.deepEqual(CAPS, { describe: 3, safety: 4, ask: 2 });
});
test('clean: commits only finished sentences while streaming; a decimal point is not an end', () => {
  const g = createSentenceGate({ max: 3 });
  assert.deepEqual(g.push('The ship flies at 3.5 km'), { add: [], stop: false });
  assert.deepEqual(g.push(' above the sea.'), { add: [], stop: false }, 'the end is final only once the next word begins');
  assert.deepEqual(g.push(' The'), { add: ['The ship flies at 3.5 km above the sea.'], stop: false });
  assert.deepEqual(g.push(' sky is clear! Then'), { add: ['The sky is clear!'], stop: false });
  assert.deepEqual(g.sentences, ['The ship flies at 3.5 km above the sea.', 'The sky is clear!']);
});
test('clean: stops at the first near-repeat (Jaccard >= 0.6) and never shows it', () => {
  const g = createSentenceGate({ max: 4 });
  const { out, stop } = feedAll(g, 'The nearest ground is a town. The weather is clear. The nearest ground is a city. The weather is clear. ');
  assert.deepEqual(out, ['The nearest ground is a town.', 'The weather is clear.']); assert.equal(stop, true); assert.equal(g.reason, 'repeat');
  assert.deepEqual(g.push('More text. And more. ').add, [], 'nothing after a stop');
});
test('clean: a sentence that opens like a kept one (3 words) and shares >= 0.4 of its words is a repeat too', () => {
  const g = createSentenceGate({ max: 3 });
  const { out, stop } = feedAll(g, "The camera follows the ship from about 100 km. The nearest hazard is a flock of birds at 1 o'clock in the image, near the middle. The nearest hazard is a flock of birds, and the nearest one is a hazard in the frame. ");
  assert.deepEqual(out, ['The camera follows the ship from about 100 km.', "The nearest hazard is a flock of birds at 1 o'clock in the image, near the middle."]); assert.equal(stop, true); assert.equal(g.reason, 'repeat');
  const k = createSentenceGate({ max: 3 }); assert.deepEqual(feedAll(k, 'The ship flies over the sea. The ship turns left toward a rocky coast. ').out, ['The ship flies over the sea.', 'The ship turns left toward a rocky coast.'], 'a shared two-word opening alone is fine');
});
test('clean: caps at 3 sentences for Describe, 4 for safety (perception, prediction, verdict with its reason, advice) and 2 for ask', () => {
  const text = 'One is here. Two goes there. Three sits low. Four flies high. Five lands soon. ';
  const d = createSentenceGate({ max: CAPS.describe }), a = feedAll(d, text);
  assert.deepEqual(a.out, ['One is here.', 'Two goes there.', 'Three sits low.']); assert.equal(a.stop, true); assert.equal(d.reason, 'cap');
  const s = createSentenceGate({ max: CAPS.safety }); assert.deepEqual(feedAll(s, text).out, ['One is here.', 'Two goes there.', 'Three sits low.', 'Four flies high.']);
  const q = createSentenceGate({ max: CAPS.ask }); assert.deepEqual(feedAll(q, text).out, ['One is here.', 'Two goes there.']);
});
test('clean: finish() trims a dangling fragment and keeps a finished last sentence', () => {
  const g = createSentenceGate({ max: 3 }); feedAll(g, 'The frame shows the ship in space. The frame shows the Moon at');
  assert.deepEqual(g.finish().add, []); assert.equal(g.text, 'The frame shows the ship in space.'); assert.equal(g.reason, 'fragment');
  const h = createSentenceGate({ max: 3 }); feedAll(h, 'A calm sea lies below. Clouds drift east.');
  assert.deepEqual(h.finish().add, ['Clouds drift east.']); assert.equal(h.text, 'A calm sea lies below. Clouds drift east.');
  const r = createSentenceGate({ max: 3 }); feedAll(r, 'Clouds drift east. Clouds drift east.'); assert.deepEqual(r.finish().add, [], 'a repeat at the very end is dropped too');
});
test('clean: dropEcho drops a bare restated verdict (the pill shows it) but keeps the monitor\'s reasons, prediction and advice (V1-10)', () => {
  const g = createSentenceGate({ max: 3, dropEcho: true });
  const { out } = feedAll(g, 'Monitor verdict: SAFE. Reason: the corridor edge. The ship holds the middle of the corridor. Rocks drift far to the left. ');
  assert.deepEqual(out, ['Reason: the corridor edge.', 'The ship holds the middle of the corridor.', 'Rocks drift far to the left.']);
  const k = createSentenceGate({ max: 2 }); assert.deepEqual(feedAll(k, 'The verdict is SAFE. Then more. ').out, ['The verdict is SAFE.', 'Then more.'], 'off by default');
  for (const bare of ['The monitor rates this UNSAFE.', 'For the monitor this is CAUTION.', 'Verdict from the monitor: SAFE.', 'The monitor judges the situation SAFE.', 'The verdict is caution.']) {
    const b = createSentenceGate({ max: 4, dropEcho: true }); assert.deepEqual(feedAll(b, `${bare} The pilot should climb. `).out, ['The pilot should climb.'], bare);
  }
});
test('clean: a trained safety answer keeps its "The monitor rates this … because …" sentence and its advice (V1-10)', () => {
  const answer = "I see a rock at 10 o'clock in the image. If nothing changes, the monitor expects that a crash is possible. The monitor rates this UNSAFE because of a hazard ahead. The pilot should turn right. ";
  const g = createSentenceGate({ max: CAPS.safety, dropEcho: true }), { out } = feedAll(g, answer); g.finish();
  assert.deepEqual(out, ["I see a rock at 10 o'clock in the image.", 'If nothing changes, the monitor expects that a crash is possible.', 'The monitor rates this UNSAFE because of a hazard ahead.', 'The pilot should turn right.']);
  for (const s of ['With a hazard ahead flagged, the monitor rates this UNSAFE; the pilot should turn right.', 'The monitor flags a hazard ahead and the corridor edge and rates the situation CAUTION.',
    'For the monitor this is SAFE, with no reason listed, and the pilot should continue.', 'According to the monitor the approach is CAUTION, driven by an unstable approach.', 'Advice: continue.']) {
    const x = createSentenceGate({ max: CAPS.safety, dropEcho: true }); assert.deepEqual(feedAll(x, `${s} `).out.concat(x.finish().add), [s], s);
  }
  const r = createSentenceGate({ max: CAPS.safety, dropEcho: true });
  const rep = feedAll(r, 'The monitor rates this UNSAFE because of a hazard ahead. The monitor rates this UNSAFE because of a hazard ahead and a fast closing rock. The pilot should climb. ');
  assert.deepEqual(rep.out, ['The monitor rates this UNSAFE because of a hazard ahead.']); assert.equal(r.reason, 'repeat', 'a restated reason is still a repeat');
});

test('words: pill labels from decoded heads; Watching while warming up', () => {
  assert.equal(pillLabel(null, 'S'), 'Watching…'); assert.equal(pillLabel({ status: 'warming up' }, 'S'), 'Watching…');
  assert.equal(pillLabel({ status: 'ok', verdict: 'SAFE', reasons: ['HAZARD_AHEAD'], action: 'CLIMB' }, 'S'), 'Safe');
  assert.equal(pillLabel({ status: 'ok', verdict: 'CAUTION', reasons: ['HAZARD_AHEAD'], action: 'CLIMB' }, 'S'), 'Caution · rock ahead');
  assert.equal(pillLabel({ status: 'ok', verdict: 'CAUTION', reasons: ['HAZARD_AHEAD'], action: 'CONTINUE' }, 'A'), 'Caution · obstacle ahead');
  assert.equal(pillLabel({ status: 'ok', verdict: 'CAUTION', reasons: [], action: 'SLOW_DOWN' }, 'D'), 'Caution · slow down');
  assert.equal(pillLabel({ status: 'ok', verdict: 'CAUTION', reasons: [], action: 'NONE_SAFE' }, 'L'), 'Caution');
  assert.equal(pillLabel({ status: 'ok', verdict: 'UNSAFE', reasons: ['TERRAIN_CLOSE'], action: 'CLIMB' }, 'A'), 'Unsafe · climb');
  assert.equal(pillLabel({ status: 'ok', verdict: 'UNSAFE', reasons: ['CLOSING_TOO_FAST'], action: 'CONTINUE' }, 'D'), 'Unsafe · closing fast');
});
test('words: context chips per family, the sky from the weather knobs, km formatting', () => {
  assert.equal(contextChip('S', {}), 'Deep space'); assert.equal(contextChip('L', {}), 'Final approach'); assert.equal(contextChip('D', {}), 'Docking');
  assert.equal(contextChip('Z', { rangeKm: 420.4 }), 'Earth · 420 km'); assert.equal(contextChip('Z', { rangeKm: 3.25 }), 'Earth · 3.3 km'); assert.equal(contextChip('Z', {}), 'Earth');
  assert.equal(contextChip('A', { route: 'alps', sky: 'Storm' }), 'Alps · Storm'); assert.equal(contextChip('A', { route: 'newyork' }), 'New York');
  assert.equal(contextChip('A', { route: 'zz', sky: 'Fair' }), 'Zz · Fair');
  assert.deepEqual([[0.2, 0], [0.5, 1], [0.55, 2], [1, 1.6]].map(([s, c]) => skyName(s, c)), ['Clear', 'Fair', 'Cloudy', 'Storm']);
  assert.equal(skyName(undefined, undefined), null);
  assert.deepEqual([fmtKm(12400), fmtKm(420), fmtKm(3.25), fmtKm(0.42)], ['12,400 km', '420 km', '3.3 km', '420 m']);
});
test('words: the template sentence reads in sentence case', () => {
  assert.equal(humanTemplate('UNSAFE: a hazard ahead. Advice: climb.'), 'Unsafe: a hazard ahead. Advice: climb.');
  assert.equal(humanTemplate('SAFE. Advice: continue.'), 'Safe. Advice: continue.');
  assert.equal(humanTemplate('A view from under 20 km showing hills.'), 'A view from under 20 km showing hills.');
  assert.equal(humanTemplate(''), '');
});

test('place: centres in the free band, drops below chrome that leaves too little room, and caps at the viewport', () => {
  const app = [{ l: 22, t: 18, r: 372, b: 64 }, { l: 984, t: 18, r: 1418, b: 51 }, { l: 1106, t: 76, r: 1418, b: 672 }, { l: 22, t: 76, r: 394, b: 726 }];
  assert.deepEqual(placeBox({ vw: 1440, vh: 900, obstacles: app, h: 300 }), { x: 720, y: 17, w: 496 }, 'the tools leave 264 px each side of the centre');
  assert.deepEqual(placeBox({ vw: 1440, vh: 900, obstacles: app, h: 34, minW: 120, maxW: 240 }), { x: 720, y: 17, w: 240 });
  const narrow = [{ l: 22, t: 18, r: 372, b: 64 }, { l: 824, t: 18, r: 1258, b: 51 }, { l: 946, t: 76, r: 1258, b: 672 }, { l: 22, t: 76, r: 394, b: 726 }];
  assert.deepEqual(placeBox({ vw: 1280, vh: 800, obstacles: narrow, h: 300 }), { x: 640, y: 63, w: 460 }, 'below the header tools, between the panels');
  const land = [{ l: 864, t: 14, r: 1424, b: 93 }];
  assert.deepEqual(placeBox({ vw: 1440, vh: 900, obstacles: land, h: 34, minW: 120, maxW: 240 }), { x: 720, y: 17, w: 240 });
  assert.deepEqual(placeBox({ vw: 1440, vh: 900, obstacles: land, h: 300 }), { x: 720, y: 105, w: 560 });
  const straddle = [{ l: 0, t: 0, r: 1440, b: 60 }];
  assert.deepEqual(placeBox({ vw: 1440, vh: 900, obstacles: straddle, h: 300 }), { x: 720, y: 72, w: 560 });
  const full = [{ l: 0, t: 0, r: 1440, b: 900 }];
  assert.deepEqual(placeBox({ vw: 1440, vh: 900, obstacles: full, h: 300 }), { x: 720, y: 17, w: 560 }, 'nowhere free: an overlay at the top');
  assert.deepEqual(placeBox({ vw: 390, vh: 844, obstacles: [], h: 34, minW: 120, maxW: 240 }), { x: 195, y: 17, w: 240 });
});

test('feed: the family follows zoom, page and atmosphere (Z > L > D > A > S)', () => {
  const ap = (o = {}) => ({ ready: true, zoom: { active: false }, state: { atmo: false }, ...o });
  assert.equal(familyOf({}), null); assert.equal(familyOf({ __ap: { ready: false } }), null);
  assert.equal(familyOf({ __ap: ap() }), 'S'); assert.equal(familyOf({ __ap: ap({ state: { atmo: true } }) }), 'A');
  assert.equal(familyOf({ __ap: ap({ zoom: { active: true }, state: { atmo: true } }) }), 'Z');
  assert.equal(familyOf({ __landing: { sim: {} } }), 'L'); assert.equal(familyOf({ __real: { scene: {}, sim: {} } }), 'D');
  assert.equal(familyOf({ __real: { headless: true } }), null, 'the headless run has no view');
  assert.deepEqual(['S', 'A', 'L', 'D'].map((f) => frameDtOf(f)), [0.2, 0.2, 0.25, 2]); assert.equal(frameDtOf('L', { L: 0.5 }), 0.5);
});
test('feed: sim clocks and episode keys come from the globals; episodes bump on reset, family or run change', () => {
  const env = { t: 4.2, ship: { v: [3, 4, 0] } }, sim = { flight: { t: 12.5 } }, rs = { t: 30 };
  assert.deepEqual(clockOf('S', { __ap: { env, state: { route: 'alps' } } }), { t: 4.2, key: 'S' });
  assert.deepEqual(clockOf('A', { __ap: { env, state: { route: 'alps' } } }), { t: 4.2, key: 'A:alps' });
  assert.deepEqual(clockOf('L', { __landing: { sim } }), { t: 12.5, key: sim }); assert.deepEqual(clockOf('D', { __real: { sim: rs } }), { t: 30, key: rs });
  const e = createEpisodes(); const a = e.id('S', 1, 'S'); assert.equal(e.id('S', 1.2, 'S'), a); assert.notEqual(e.id('S', 0.1, 'S'), a, 'the clock went back: a reset');
  const b = e.id('S', 0.3, 'S'); assert.notEqual(e.id('A', 0.5, 'A:alps'), b); const c = e.id('A', 0.7, 'A:alps'); assert.notEqual(e.id('A', 0.9, 'A:dubai'), c);
  assert.deepEqual(factsOf('S', { __ap: { env } }), { 'ship.speed_m_s': { v: 95, unit: 'm/s' } });
  assert.deepEqual(factsOf('Z', { __ap: { zoom: { info: { rangeKm: 420.123 } } } }), { 'view.range_km': { v: 420.12, unit: 'km' } });
  assert.deepEqual(factsOf('L', {}), {});
});
test('feed: verdict hysteresis holds a new verdict for 2 decisions, UNSAFE for 1', () => {
  const h = createHysteresis();
  assert.equal(h.push('SAFE'), 'SAFE', 'the first verdict shows at once');
  assert.equal(h.push('CAUTION'), 'SAFE'); assert.equal(h.push('SAFE'), 'SAFE'); assert.equal(h.push('CAUTION'), 'SAFE'); assert.equal(h.push('CAUTION'), 'CAUTION');
  assert.equal(h.push('UNSAFE'), 'UNSAFE', 'UNSAFE needs one decision');
  assert.equal(h.push('SAFE'), 'UNSAFE'); assert.equal(h.push('CAUTION'), 'UNSAFE', 'a changing candidate restarts the count'); assert.equal(h.push('CAUTION'), 'CAUTION');
  h.reset(); assert.equal(h.value, null); assert.equal(h.push('CAUTION'), 'CAUTION');
});
test('feed: the centre 16:9 crop keeps the vertical field on wide windows and the width on tall ones', () => {
  assert.deepEqual(centerCrop(1920, 1080), [0, 0, 1920, 1080]); assert.deepEqual(centerCrop(2160, 900), [280, 0, 1600, 900]);
  assert.deepEqual(centerCrop(390, 844), [0, 312, 390, 219]); assert.deepEqual(centerCrop(0, 0), [0, 0, 1, 1]);
});

function fakeDom(search = '') {
  const listeners = {}, els = {}, appended = [];
  const el = (tag) => ({ tagName: tag.toUpperCase(), style: {}, children: [], attrs: {}, listeners: {}, className: '', id: '', textContent: '', innerHTML: '',
    addEventListener(k, f) { (this.listeners[k] ||= []).push(f); }, setAttribute(k, v) { this.attrs[k] = v; }, appendChild(c) { this.children.push(c); appended.push(c); return c; }, append(...c) { c.forEach((x) => this.appendChild(x)); } });
  const doc = { head: el('head'), body: el('body'), createElement: el, getElementById: (id) => els[id] || null };
  els.btnNarrator = el('button');
  const win = { location: { search }, addEventListener(k, f) { (listeners[k] ||= []).push(f); } };
  const key = (k, extra = {}) => { const e = { key: k, target: { tagName: 'CANVAS' }, preventDefault() { e.prevented = true; }, ...extra }; for (const f of listeners.keydown || []) f(e); return e; };
  return { doc, win, els, key, listeners, appended };
}
test('boot: inert until asked; N (not while typing, not with modifiers) and the header button load the card once', async () => {
  const d = fakeDom(); let loads = 0; const calls = [];
  const card = { toggle: () => calls.push('toggle'), open: () => calls.push('open'), close: () => calls.push('close') };
  const api = installBoot({ win: d.win, doc: d.doc, load: async () => { loads++; return card; } });
  assert.equal(loads, 0, 'nothing loads at boot'); assert.equal(api.loaded, false); assert.equal(d.win.__narrator, api);
  d.key('n', { target: { tagName: 'INPUT' } }); d.key('n', { metaKey: true }); d.key('n', { repeat: true }); d.key('m');
  assert.equal(loads, 0, 'typing, modifiers, repeats and other keys do nothing');
  const e = d.key('N'); await api.ready; assert.equal(loads, 1); assert.ok(e.prevented); assert.deepEqual(calls, ['toggle']);
  for (const f of d.els.btnNarrator.listeners.click) f(); await api.ready; assert.equal(loads, 1, 'the module is imported once'); assert.deepEqual(calls, ['toggle', 'toggle']);
  const launch = d.appended.find((x) => x.id === 'nr-launch'); assert.ok(launch, 'a launcher for pages without the header'); for (const f of launch.listeners.click) f(); await api.ready; assert.deepEqual(calls, ['toggle', 'toggle', 'open']);
  assert.equal(installBoot({ win: fakeDom('?vlm=0').win, doc: fakeDom().doc, load: async () => card }), null, '?vlm=0 turns the Narrator off');
});
test('boot: the boot module imports nothing at load time and never touches the renderer', () => {
  const src = fs.readFileSync(new URL('../src/narrator/boot.js', import.meta.url), 'utf8');
  assert.ok(!/^\s*import\s/m.test(src), 'no static imports'); assert.ok(!/EffectComposer|createImageBitmap|Worker\(/.test(src));
  assert.deepEqual([...src.matchAll(/import\(\s*['"]([^'"]+)['"]\s*\)/g)].map((m) => m[1]), ['./card.js']);
  const html = fs.readFileSync(new URL('../index.html', import.meta.url), 'utf8');
  assert.equal(html.split('<script type="module" src="./src/narrator/boot.js"></script>').length, 2); assert.equal(html.split('id="btnNarrator"').length, 2);
});
test('src/narrator files stay under 500 lines and app.js stays at 499', () => {
  const dir = new URL('../src/narrator/', import.meta.url);
  for (const f of fs.readdirSync(dir)) { const n = fs.readFileSync(new URL(f, dir), 'utf8').split('\n').length; assert.ok(n < 500, `${f}: ${n}`); }
  assert.equal(fs.readFileSync(new URL('../src/app.js', import.meta.url), 'utf8').split('\n').length - 1, 499);
});
