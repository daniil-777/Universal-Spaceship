import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { nextStopTime, parseKV, countFamily, pickFamily, decide, cmdlineClean, orphanBrowsers, LANES, STOP_EXIT, RESOURCE_EXIT } from '../vlm/tools/overnight.mjs';

test('overnight: the hard stop is the next local 07:00 (tonight -> tomorrow morning, early morning -> the same day)', () => {
  const eve = new Date(2026, 8, 30, 23, 40).getTime(), t = new Date(nextStopTime(eve, '07:00'));
  assert.deepEqual([t.getDate(), t.getHours(), t.getMinutes()], [1, 7, 0]);
  const early = new Date(2026, 9, 1, 3, 0).getTime(), u = new Date(nextStopTime(early, '07:00'));
  assert.deepEqual([u.getDate(), u.getHours()], [1, 7]);
});
test('overnight: targets and seeds parse; lane SALD takes A first, lane Z takes Z then helps from the other end', () => {
  assert.deepEqual(parseKV('A=2500,S=3200'), { A: 2500, S: 3200 });
  assert.deepEqual(LANES.SALD, ['A', 'S', 'L', 'D']); assert.equal(LANES.Z[0], 'Z');
  const st = { targets: { A: 1, S: 1, L: 1, D: 1, Z: 1 }, done: {}, held: {} };
  assert.equal(pickFamily(LANES.SALD, st), 'A'); st.held.A = 'SALD'; assert.equal(pickFamily(LANES.Z, st), 'Z');
  st.done.Z = true; assert.equal(pickFamily(LANES.Z, st), 'D', 'a family held by the other lane is never shared'); st.held.D = 'Z';
  st.done.A = true; delete st.held.A; assert.equal(pickFamily(LANES.SALD, st), 'S');
  st.done.S = st.done.L = true; assert.equal(pickFamily(LANES.SALD, st), null);
});
test('overnight: the watchdog restarts a crash, waits on a resource stop, never restarts after the 403/429 latch, and ends on a clean exit', () => {
  const base = { signal: null, added: 5, count: 50, target: 100, stopping: false, fails: 0 };
  assert.equal(decide({ ...base, code: STOP_EXIT }), 'stop-latch');
  assert.equal(decide({ ...base, code: 1 }), 'restart'); assert.equal(decide({ ...base, code: null, signal: 'SIGINT' }), 'restart');
  assert.equal(decide({ ...base, code: RESOURCE_EXIT }), 'wait-restart');
  assert.equal(decide({ ...base, code: 0, count: 100 }), 'done'); assert.equal(decide({ ...base, code: 0, added: 0 }), 'done', 'a clean exit that adds nothing: plan or EOX budget exhausted');
  assert.equal(decide({ ...base, code: 0 }), 'rerun'); assert.equal(decide({ ...base, code: 1, added: 0, fails: 5 }), 'failed');
  assert.equal(decide({ ...base, code: 1, stopping: true }), 'stop');
});
test('overnight: finished records are counted without deleting anything (unlike the drive resume scan)', () => {
  const d = fs.mkdtempSync(path.join(fs.existsSync('/Volumes/LaCie/astro-pilot/vlm/tmp') ? '/Volumes/LaCie/astro-pilot/vlm/tmp' : os.tmpdir(), 'apv-ovn-'));
  try {
    for (const n of ['L_r_00001_000100.json', 'L_r_00001_000100.f0.png', 'L_r_50001_000100.json', 'L_r_00002_000050.json', 'episode_00001.done', 'episode_50001.done', '._L_r_00001_000100.json']) fs.writeFileSync(path.join(d, n), '{}');
    assert.deepEqual(countFamily(d), { records: 2, episodes: 2, twins: 1 });
    assert.ok(fs.existsSync(path.join(d, 'L_r_00002_000050.json')), 'the unfinished episode file is left in place');
  } finally { fs.rmSync(d, { recursive: true, force: true }); }
});
test('ruling T10-i: no launcher-started command line carries the preflight patterns; orphan sweep only takes our profile', () => {
  assert.ok(cmdlineClean(['vlm/capture/drive.mjs', '--family', 'A', '--n', '2500', '--run', 'apv-open-v1']));
  assert.ok(!cmdlineClean(['pkill', '-f', 'chromium'])); assert.ok(!cmdlineClean(['python', 'vlm/train/x.py']));
  const src = fs.readFileSync(new URL('../vlm/tools/overnight.mjs', import.meta.url), 'utf8');
  assert.ok(!/spawn\([^)]*(chromium|Chrome for Testing|chrome-headless-shell)/.test(src));
  const ps = ['  101     1 /x/chrome-headless-shell --user-data-dir=/Volumes/LaCie/astro-pilot/vlm/tmp/playwright_x --headless', '  102     1 /x/chrome-headless-shell --type=renderer --user-data-dir=/Volumes/LaCie/astro-pilot/vlm/tmp/playwright_x',
    '  103   500 /x/chrome-headless-shell --user-data-dir=/Volumes/LaCie/astro-pilot/vlm/tmp/playwright_y', '  104     1 /Users/u/.cache/hyperframes/chrome/chrome-headless-shell --user-data-dir=/var/folders/tn/p'].join('\n');
  assert.deepEqual(orphanBrowsers(ps), [101]);
});
