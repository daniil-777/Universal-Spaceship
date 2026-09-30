import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import zlib from 'node:zlib';
import { licenceOf, attributionOf, oofPreds } from '../vlm/gen/build/licence.js';
import { licenceGate } from '../vlm/capture/records.mjs';

const rec = (prof, layers = []) => ({ key: `S_${prof}_${layers.join('+')}`, render: { licence_profile: prof, imagery: layers.map((layer_id) => ({ layer_id, level: 9 })) } });

test('a dataset has one licence profile: mixed runs are refused, an explicit --licence must match every record', () => {
  assert.equal(licenceOf([rec('open'), rec('open')]), 'open');
  assert.equal(licenceOf([rec('nc')], 'nc'), 'nc');
  assert.throws(() => licenceOf([rec('open'), rec('nc')]), /licence profiles nc, open/);
  assert.throws(() => licenceOf([rec('open')], 'nc'), /--licence nc/);
  assert.throws(() => licenceOf([]), /no records/);
});

test('ATTRIBUTION.txt names the EOX layer actually served and the NC-SA terms of the 2025 layer', () => {
  const open = attributionOf([rec('open', ['s2cloudless_3857', 'gibs'])], 'open');
  assert.match(open, /Copernicus Sentinel data 2016 & 2017/); assert.doesNotMatch(open, /2025|NC-SA/);
  const nc = attributionOf([rec('nc', ['s2cloudless-2025_3857'])], 'nc');
  assert.match(nc, /Copernicus Sentinel data 2025\), CC BY-NC-SA 4\.0/); assert.match(nc, /non-commercial/); assert.doesNotMatch(nc, /2016 & 2017/);
});

test('oof Context: duplicate keys and in-fold train predictions are refused', () => {
  const half = (g) => zlib.crc32(g) % 2, g0 = ['S:1', 'S:2', 'S:3', 'S:4'].find((g) => half(g) === 0);
  const ctx = { groupOf: () => g0, splitOf: (k) => (k.startsWith('t') ? 'train' : 'val') }, line = (o) => JSON.stringify({ monitor: { verdict: 'SAFE' }, ...o });
  assert.equal(oofPreds([line({ key: 't1', fold: 1 }), line({ key: 'v1', fold: null })].join('\n'), ctx).size, 2);
  assert.throws(() => oofPreds([line({ key: 'v1' }), line({ key: 'v1' })].join('\n'), ctx), /duplicate/);
  assert.throws(() => oofPreds(line({ key: 't1', fold: 0 }), ctx), /trained on/);
  assert.throws(() => oofPreds(line({ key: 't1' }), ctx), /fold/);
});

test('a run keeps the licence profile it started with: a resume with another --licence is refused', () => {
  const d = fs.mkdtempSync(path.join(os.tmpdir(), 'apv-lic-'));
  assert.equal(licenceGate(d, 'open').refuse, false); assert.equal(JSON.parse(fs.readFileSync(path.join(d, 'licence.json'), 'utf8')).licence, 'open');
  assert.equal(licenceGate(d, 'open').refuse, false);
  const g = licenceGate(d, 'nc'); assert.equal(g.refuse, true); assert.match(g.why, /open/);
  fs.rmSync(d, { recursive: true, force: true });
});
