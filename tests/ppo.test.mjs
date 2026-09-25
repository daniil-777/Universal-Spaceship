import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as tf from '@tensorflow/tfjs';
import { PPOAgent, computeGAE, setTF, f32ToB64, b64ToF32 } from '../src/ppo.js';
import { Trainer } from '../src/trainer.js';
import { OBS_DIM, ACT_DIM } from '../src/env.js';
import { orthogonal } from '../src/actor.js';
import { mulberry32, randn } from '../src/mathx.js';

setTF(tf);
const near = (a, b, eps, msg) => assert.ok(Math.abs(a - b) <= eps, `${msg || ''} ${a} vs ${b}`);

test('GAE matches a hand computation with a terminal and a truncation', () => {
  const T = 4, N = 1, gamma = 0.9, lambda = 0.8;
  const rew = Float32Array.from([1, 1, 1, 1]), values = Float32Array.from([0.5, 0.6, 0.7, 0.8]), lastV = Float32Array.from([0.9]);
  const done = Uint8Array.from([0, 1, 0, 0]), trunc = Uint8Array.from([0, 0, 0, 1]), bootV = Float32Array.from([0, 0, 0, 2.0]);
  const { adv, ret } = computeGAE({ T, N, rew, done, trunc, values, lastV, bootV, gamma, lambda });
  const d3 = 1 + gamma * 2.0 - 0.8, a3 = d3;                       // truncated: bootstrap with V(final obs), chain reset
  const d2 = 1 + gamma * 0.8 - 0.7, a2 = d2 + gamma * lambda * a3;
  const d1 = 1 + 0 - 0.6, a1 = d1;                                  // terminal: next value 0, chain reset
  const d0 = 1 + gamma * 0.6 - 0.5, a0 = d0 + gamma * lambda * a1;
  [a0, a1, a2, a3].forEach((a, i) => { near(adv[i], a, 1e-6, 'adv' + i); near(ret[i], a + values[i], 1e-6, 'ret' + i); });
});

test('orthogonal init has orthonormal columns scaled by gain', () => {
  const rng = mulberry32(1), W = orthogonal(91, 64, 2, () => randn(rng));
  for (let a = 0; a < 64; a += 21) for (let b = 0; b < 64; b += 17) { let d = 0; for (let i = 0; i < 91; i++) d += W[i * 64 + a] * W[i * 64 + b]; near(d, a === b ? 4 : 0, 1e-5, `col ${a}·${b}`); }
});

test('JS actor/critic mirror equals the TF forward pass; log-probs agree', async () => {
  const ag = new PPOAgent(OBS_DIM, ACT_DIM, {}, 5); await ag.initTF();
  const n = 7, obs = new Float32Array(n * OBS_DIM); for (let i = 0; i < obs.length; i++) obs[i] = Math.sin(i * 0.37) * 2;
  const meanJS = ag.actor.forward(obs, n), meanTF = tf.tidy(() => ag.forwardTF(tf.tensor2d(obs, [n, OBS_DIM]), ag.tfv.actor).dataSync());
  for (let i = 0; i < n * ACT_DIM; i++) near(meanJS[i], meanTF[i], 1e-5, 'actor ' + i);
  const vJS = ag.critic.forward(obs, n), vTF = await ag.valuesTF(obs, n);
  for (let i = 0; i < n; i++) near(vJS[i], vTF[i], 1e-5, 'critic ' + i);
  const act = new Float32Array(n * ACT_DIM), logp = new Float32Array(n); ag.sample(obs, n, act, logp);
  const lpTF = tf.tidy(() => { const mean = tf.tensor2d(meanTF, [n, ACT_DIM]), ls = tf.tensor1d(ag.logStd), z = tf.div(tf.sub(tf.tensor2d(act, [n, ACT_DIM]), mean), tf.exp(ls)); return tf.sub(tf.sub(tf.mul(tf.sum(tf.square(z), 1), -0.5), tf.sum(ls)), 0.5 * ACT_DIM * Math.log(2 * Math.PI)).dataSync(); });
  for (let i = 0; i < n; i++) near(logp[i], lpTF[i], 1e-4, 'logp ' + i);
  ag.dispose();
});

test('export → import round trip preserves the policy and the normaliser', async () => {
  const ag = new PPOAgent(OBS_DIM, ACT_DIM, {}, 8); ag.obsNorm.mean[3] = 1.5; ag.obsNorm.var[3] = 4; ag.logStd[1] = -1.2; ag.steps = 12345;
  const json = JSON.parse(JSON.stringify(ag.toJSON({ note: 'x' }))), ag2 = PPOAgent.fromJSON(json);
  const obs = new Float32Array(OBS_DIM).map((_, i) => Math.cos(i)), a = ag.actMean(ag.normalize(obs, 1)), b = ag2.actMean(ag2.normalize(obs, 1));
  for (let i = 0; i < ACT_DIM; i++) near(a[i], b[i], 1e-7);
  assert.equal(ag2.steps, 12345); assert.equal(ag2.logStd[1], Float32Array.from([-1.2])[0]); assert.equal(ag2.meta.note, 'x');
  const f = Float32Array.from([1.5, -2.25, 3e-7]); assert.deepEqual(Array.from(b64ToF32(f32ToB64(f))), Array.from(f));
});

test('Trainer runs two PPO updates end to end with finite stats and the mirror in sync', async () => {
  const ag = new PPOAgent(OBS_DIM, ACT_DIM, { minibatch: 32, epochs: 2 }, 3);
  const tr = new Trainer(ag, { nEnvs: 4, T: 24, seed: 50, level: 0.2 });
  tr.mode = 'max';
  let n = 0; tr.onMetrics = (m) => { n++; if (n >= 2) tr.stop(); };
  await tr.run();
  const m = tr.metrics;
  assert.equal(m.updates, 2); assert.equal(m.step, 2 * 4 * 24); assert.equal(ag.steps, m.step);
  for (const k of ['pgLoss', 'vLoss', 'entropy', 'kl', 'clipFrac', 'explainedVar']) assert.ok(Number.isFinite(m[k]), k + ' finite');
  assert.ok(m.kl >= 0 && m.clipFrac >= 0 && m.clipFrac <= 1);
  const obs = ag.normalize(tr.envs[0].obs, 1), js = ag.actor.forward(obs, 1), tfm = tf.tidy(() => ag.forwardTF(tf.tensor2d(obs, [1, OBS_DIM]), ag.tfv.actor).dataSync());
  for (let i = 0; i < ACT_DIM; i++) near(js[i], tfm[i], 1e-5, 'mirror synced ' + i);
  assert.equal(tr.history.step.length, 2);
  ag.dispose();
  assert.equal(tf.memory().numTensors, 0, 'no tensor leak: ' + tf.memory().numTensors);
});
