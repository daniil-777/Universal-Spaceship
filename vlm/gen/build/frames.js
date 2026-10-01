// vlm/gen/build/frames.js — a record's image work in one pass (build v1): every PNG it needs is read once from the disk (the
// raw frames live on a USB hard disk, so reads, not decodes, are what a big build waits on) and decoded once, then the image
// facts of the narrator frame (imagefacts.js), the dHash of the last frame (dedupe.js), the Pilot Eye rows of every frame
// (boxresize.js; a one-frame Z record repeats its frame to 3 rows) and the SHA-256 of each frame file (the pixel-identical
// pair check compares those instead of reading the files again). Each result equals the v0 code's on the file (sharp gives
// the same bytes from the decoded pixels; tests/vlm_build_scale.test.mjs). Runs in the build's worker pool (pool.js calls
// job()), or in-process.
import crypto from 'node:crypto';
import fsp from 'node:fs/promises';
import path from 'node:path';
import sharp from 'sharp';
import { imageFactsOf } from '../imagefacts.js';
import { dhashOf } from './dedupe.js';
import { boxResize } from '../boxresize.js';

// one libvips thread per operation and no operation cache: the pool's workers are the parallelism (a PNG decode is serial
// anyway), so a build is `workers` cores of load, never workers x cores threads
sharp.concurrency(1); sharp.cache(false);

// task: {root, narrator, frames[], want: {facts, hash, eye, files}, eyeSize: [W, H]} (paths relative to root, or absolute)
export async function frameJob({ root = '', narrator = null, frames = [], want = {}, eyeSize = [160, 96] }) {
  const bytes = new Map(), decoded = new Map(), at = (p) => (path.isAbsolute(p) ? p : path.join(root, p));
  const read = (p) => { if (!bytes.has(p)) bytes.set(p, fsp.readFile(at(p))); return bytes.get(p); };
  const decode = (p) => { if (!decoded.has(p)) decoded.set(p, read(p).then((b) => sharp(b).raw().toBuffer({ resolveWithObject: true }))); return decoded.get(p); };
  const img = async (p) => { const { data, info } = await decode(p); return sharp(data, { raw: { width: info.width, height: info.height, channels: info.channels } }); };
  const out = {};
  if (want.facts && narrator) {
    const { data, info } = await (await img(narrator)).removeAlpha().resize(224, 126, { kernel: 'cubic' }).raw().toBuffer({ resolveWithObject: true });
    out.facts = imageFactsOf(data, info.width, info.height, info.channels);
  }
  if (want.hash && frames.length) out.hash = (await dhashOf(await img(frames[frames.length - 1]))).toString(16);
  if (want.files) out.files = await Promise.all(frames.map(async (f) => crypto.createHash('sha256').update(await read(f)).digest('hex')));
  if (want.eye && frames.length) {
    const [EW, EH] = eyeSize, rows = [];
    for (const f of frames) { const { data, info } = await decode(f); rows.push(boxResize(data, info.width, info.height, EW, EH, info.channels)); }
    while (rows.length < 3) rows.push(rows[rows.length - 1]);
    const eye = new Uint8Array(rows.reduce((n, r) => n + r.length, 0)); let o = 0;
    for (const r of rows) { eye.set(r, o); o += r.length; }
    out.eye = eye;
  }
  return out;
}
export async function job(task) { const r = await frameJob(task); return { result: r, transfer: r.eye ? [r.eye.buffer] : [] }; }
