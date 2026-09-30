import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
test('Narrator loss mask (spec §10.2): Python unittest on the real SmolVLM tokenizer', () => {
  const r = spawnSync('/Volumes/LaCie/astro-pilot/vlm/venv/bin/python', ['-m', 'unittest', 'vlm.train.tests.test_common'], { cwd: fileURLToPath(new URL('..', import.meta.url)), encoding: 'utf8', env: { ...process.env, HF_HUB_OFFLINE: '1' } });
  assert.equal(r.status, 0, r.stderr.slice(-2000));
});
