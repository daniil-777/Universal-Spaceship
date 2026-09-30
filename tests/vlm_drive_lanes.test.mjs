import { test } from 'node:test';
import assert from 'node:assert/strict';
import { lanesOf, minFreePct, mainBrowsers, resourceVerdict } from '../vlm/capture/drive.mjs';

const PS = [
  '101 /Users/x/Library/Caches/ms-playwright/chromium_headless_shell-1200/chrome-mac/chrome-headless-shell --disable-field-trial-config --headless --remote-debugging-pipe',
  '102 /Users/x/Library/Caches/ms-playwright/chromium_headless_shell-1200/chrome-mac/chrome-headless-shell --type=renderer --lang=en-US',
  '103 /Users/x/Library/Caches/ms-playwright/chromium_headless_shell-1200/chrome-mac/chrome-headless-shell --type=gpu-process',
  '104 /Users/x/.cache/hyperframes/chrome/mac_arm-1/chrome-headless-shell --headless --remote-debugging-port=9222'].join('\n');
test('APV_BROWSER_LANES: default 1, a positive integer otherwise; the memory floor is 25 % for one lane and 35 % for more', () => {
  assert.equal(lanesOf({}), 1); assert.equal(lanesOf({ APV_BROWSER_LANES: '2' }), 2); assert.equal(lanesOf({ APV_BROWSER_LANES: 'x' }), 1); assert.equal(lanesOf({ APV_BROWSER_LANES: '0' }), 1);
  assert.equal(minFreePct(1), 25); assert.equal(minFreePct(2), 35);
});
test('only main browser processes count: no --type= flag, the hyperframes preview browser excluded', () => {
  assert.deepEqual(mainBrowsers(PS).map((l) => l.split(' ')[0]), ['101']); assert.deepEqual(mainBrowsers(''), []);
});
test('the preflight passes while fewer than N main browsers run and memory is above the lane floor', () => {
  const v = (o) => resourceVerdict({ free: 60, pgrep: '', train: '', mc: '', lanes: 1, ...o });
  assert.equal(v({}).ok, true); assert.equal(v({ pgrep: PS }).ok, false, 'one lane: another main browser blocks');
  assert.equal(v({ pgrep: PS, lanes: 2 }).ok, true, 'two lanes: one other main browser is allowed');
  assert.equal(v({ pgrep: `${PS}\n${PS.split('\n')[0].replace('101', '105')}`, lanes: 2 }).ok, false, 'two lanes: two main browsers block');
  assert.equal(v({ pgrep: PS, lanes: 2, free: 34 }).ok, false); assert.equal(v({ free: 26 }).ok, true); assert.equal(v({ free: 24 }).ok, false);
  assert.equal(v({ train: '999 python -m vlm.train.pilot_eye.train', lanes: 2 }).ok, false); assert.equal(v({ mc: '998 node real_mc.mjs' }).ok, false);
});
test('zplan --blocks-from: a second run reuses the first run\'s level-3 blocks, so one place keeps one split across runs', async () => {
  const { blocksFor } = await import('../vlm/capture/zplan.mjs'), w = new Map([['3/0/0', 1], ['3/1/0', 1], ['3/2/0', 1]]);
  assert.deepEqual([...blocksFor({ seed: 29, weights: w, from: { blocks: { '3/0/0': 'val', '3/1/0': 'train' } } })], [['3/0/0', 'val'], ['3/1/0', 'train']]);
  const { assignBlocks } = await import('../vlm/gen/sampler_z.js'); assert.deepEqual([...blocksFor({ seed: 29, weights: w })], [...assignBlocks(29, w)]);
});
