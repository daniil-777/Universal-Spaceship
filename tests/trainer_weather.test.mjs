import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Trainer } from '../src/trainer.js';
import { PPOAgent } from '../src/ppo.js';
import { OBS_DIM, ACT_DIM } from '../src/env.js';

test('trainer: weather only in atmospheric envs, severities drawn up to the current maximum', () => {
  const tr = new Trainer(new PPOAgent(OBS_DIM, ACT_DIM, {}, 3), { nEnvs: 8, T: 4, mixed: true, weather: 0.4 });
  assert.equal(tr.weatherMax, 0.4);
  for (const e of tr.envs) { if (!e.atmosphere) { assert.equal(e.weather, null); continue; } assert.ok(e.weatherSeverity >= 0 && e.weatherSeverity <= 0.4); assert.ok(e.weather); }
  assert.ok(tr.envs.some((e) => e.atmosphere) && tr.envs.some((e) => !e.atmosphere));
});
test('trainer: the weather gate raises the storm ceiling when the pilot copes as well as in calm air, at most every 2 M steps', () => {
  const tr = new Trainer(new PPOAgent(OBS_DIM, ACT_DIM, {}, 3), { nEnvs: 4, T: 4, mixed: true, weather: 0.2 });
  assert.equal(tr.weatherGate(1e6, 300, 250), 0.3); assert.equal(tr.weatherGate(2e6, 300, 300), 0.3, 'too soon');
  assert.equal(tr.weatherGate(3.1e6, 300, 200), 0.3, 'worse in weather'); assert.equal(tr.weatherGate(3.2e6, 300, 260), 0.4);
});
