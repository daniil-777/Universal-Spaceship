// Real spacecraft S1: the flag is isolated (spec section 8, durable invariants only; the SHA-256 manifest of the
// pre-existing files is an execution-time check in the LaCie harness, real_manifest.mjs, not a permanent test).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('..', import.meta.url)), REAL = path.join(ROOT, 'src/real');
const HEADLESS = ['consts', 'cw', 'rigid', 'jets', 'nav', 'passive', 'guidance', 'control', 'safety', 'sim'];
const BROWSER = ['scene', 'shipview', 'hud', 'main'];
const ALLOW_HEADLESS = ['../mathx.js', '../orbit.js'], ALLOW_BROWSER = ['three', '../ship.js', '../space.js', '../satellites.js'];
// the three index.html additions (Task 10); everything else in index.html stays byte-identical
export const INDEX_ADDS = [
  '<button class="btn" id="btnReal" title="Real spacecraft: rendezvous and docking with a space station, flown by a classical GNC autopilot (?scenario=real)">Real spacecraft</button>',
  "  else if (q.get('scenario') === 'real' || q.get('real') === '1') import('./src/real/main.js').then((m) => m.start());\n",
  " document.getElementById('btnReal')?.addEventListener('click', () => { location.search = '?scenario=real'; });",
];
const importsOf = (src) => [...src.matchAll(/(?:^|\n)\s*import\s[^'"]*?from\s+['"]([^'"]+)['"]|import\(\s*['"]([^'"]+)['"]\s*\)/g)].map((m) => m[1] || m[2]);
const walk = (d) => fs.readdirSync(d, { withFileTypes: true }).flatMap((e) => (e.isDirectory() ? walk(path.join(d, e.name)) : [path.join(d, e.name)]));

test('src/real imports only the allow-list: headless modules need no three and no DOM', () => {
  const files = fs.readdirSync(REAL).filter((f) => f.endsWith('.js')).map((f) => f.slice(0, -3)).sort();
  assert.deepEqual(files, [...HEADLESS, ...BROWSER].sort());
  for (const f of files) {
    const own = (s) => s.startsWith('./') && files.includes(s.slice(2, -3));
    for (const s of importsOf(fs.readFileSync(path.join(REAL, f + '.js'), 'utf8'))) {
      const ok = own(s) || ALLOW_HEADLESS.includes(s) || (BROWSER.includes(f) && (ALLOW_BROWSER.includes(s) || s.startsWith('three/addons/')));
      assert.ok(ok, `${f}.js imports ${s}`);
      if (HEADLESS.includes(f) && own(s)) assert.ok(HEADLESS.includes(s.slice(2, -3)), `headless ${f}.js imports browser module ${s}`);
    }
  }
});

test('nothing outside src/real, tests/real_* and the index.html additions references the real scenario', () => {
  const hits = [];
  for (const f of [...walk(path.join(ROOT, 'src')), ...walk(path.join(ROOT, 'tests'))]) {
    const rel = path.relative(ROOT, f);
    if (rel.startsWith('src/real/') || rel.includes('node_modules') || /^tests\/real_/.test(rel) || !/\.(m?js|json)$/.test(rel)) continue;
    if (/src\/real|['"]\.\/real\/|scenario=real|btnReal/.test(fs.readFileSync(f, 'utf8'))) hits.push(rel);
  }
  assert.deepEqual(hits, []);
  let html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');
  for (const add of INDEX_ADDS) { assert.equal(html.split(add).length, 2, `index.html has the addition once: ${add.trim().slice(0, 60)}`); html = html.replace(add, ''); }
  assert.ok(!/src\/real|scenario=real|btnReal|'real'/.test(html), 'no other real-mode edit in index.html');
  assert.ok(/if \(q\.get\('scenario'\) === 'landing'\)/.test(html) && /else \{ import\('\.\/src\/app\.js'\)/.test(html), 'landing first, the game by default');
});

test('a 600 s 6-DOF real run in-process leaves the game bit-for-bit: space_golden values, ENV and PPO_DEFAULTS', async () => {
  const { ENV, SpaceEnv, ACT_DIM } = await import('../src/env.js'), { PPO_DEFAULTS } = await import('../src/ppo.js');
  const before = JSON.stringify([ENV, PPO_DEFAULTS]);
  const { createRealSim } = await import('../src/real/sim.js');
  const sim = createRealSim({ seed: 3, start: 'near' }); sim.run(600);
  assert.ok(sim.t >= 599.9);
  assert.equal(JSON.stringify([ENV, PPO_DEFAULTS]), before);
  const env = new SpaceEnv(7, { level: 0.5 }), a = new Float32Array(ACT_DIM); let ret = 0, done = false;
  for (let t = 0; t < 120 && !done; t++) { a[0] = Math.sin(t * 0.13) * 0.8; a[1] = Math.cos(t * 0.07) * 0.5; a[2] = Math.sin(t * 0.05) * 0.6; a[3] = 0.3; const r = env.step(a); ret += r.reward; done = r.done; }
  const near = (x, y, m) => assert.ok(Math.abs(x - y) < 1e-7, `${m}: ${x} vs ${y}`);
  [36.410279475, 20.998879435, 11.661418056].forEach((v, i) => near(env.ship.p[i], v, 'p' + i));
  near(ret, -4.134203638, 'return');
});
