// Space flight (the belt, Earth orbit, low passes) must stay exactly today's model when the atmosphere engine lands.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { SpaceEnv, ACT_DIM } from '../src/env.js';

test('space flight is bit-for-bit the pre-weather model', () => {
  const env = new SpaceEnv(7, { level: 0.5 }), a = new Float32Array(ACT_DIM); let ret = 0, done = false;
  for (let t = 0; t < 120 && !done; t++) { a[0] = Math.sin(t * 0.13) * 0.8; a[1] = Math.cos(t * 0.07) * 0.5; a[2] = Math.sin(t * 0.05) * 0.6; a[3] = 0.3; const r = env.step(a); ret += r.reward; done = r.done; }
  const s = env.ship, near = (x, y, m) => assert.ok(Math.abs(x - y) < 1e-7, `${m}: ${x} vs ${y}`);
  assert.equal(env.steps, 120); assert.equal(done, false);
  [36.410279475, 20.998879435, 11.661418056].forEach((v, i) => near(s.p[i], v, 'p' + i));
  [11.533262314, -0.024777674, -10.205649179].forEach((v, i) => near(s.v[i], v, 'v' + i));
  [-0.704790226, 0.302849907, 0.469334376, 0.437353307].forEach((v, i) => near(s.q[i], v, 'q' + i));
  near(ret, -4.134203638, 'return');
  [0, 0.0152455, 0, 0, 0].forEach((v, i) => assert.ok(Math.abs(env.obs[i] - v) < 1e-6, 'obs' + i));
});
