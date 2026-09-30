// vlm/gen/build/dedupe.js — 64-bit dHash of each f2 frame; cross-split pairs within Hamming 4 are dropped from val/test/ood
// (spec §8), never from train. The pair scan compares the two 32-bit halves, so a full dataset (~5k records) takes well
// under a second.
import sharp from 'sharp';

export async function dhash(file) {
  const px = await sharp(file).resize(9, 8, { fit: 'fill', kernel: 'lanczos3' }).greyscale().raw().toBuffer();
  let h = 0n;
  for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) h = (h << 1n) | (px[y * 9 + x + 1] > px[y * 9 + x] ? 1n : 0n);
  return h;
}
const pop32 = (x) => { x -= (x >>> 1) & 0x55555555; x = (x & 0x33333333) + ((x >>> 2) & 0x33333333); return (Math.imul((x + (x >>> 4)) & 0x0f0f0f0f, 0x01010101) >>> 24); };
const halves = (h) => [Number((h >> 32n) & 0xffffffffn), Number(h & 0xffffffffn)];
export const hamming = (a, b) => { const [ah, al] = halves(a), [bh, bl] = halves(b); return pop32((ah ^ bh) >>> 0) + pop32((al ^ bl) >>> 0); };
// hashes: Map key -> bigint; splits: Map key -> split. Returns the keys to drop: every non-train member of a pair from two
// different splits within maxDist.
export function crossSplitDrops(hashes, splits, { maxDist = 4 } = {}) {
  const keys = [...hashes.keys()], hi = new Uint32Array(keys.length), lo = new Uint32Array(keys.length), drop = new Set();
  keys.forEach((k, i) => { [hi[i], lo[i]] = halves(hashes.get(k)); });
  for (let i = 0; i < keys.length; i++) for (let j = i + 1; j < keys.length; j++) {
    const a = keys[i], b = keys[j];
    if (splits.get(a) === splits.get(b) || pop32((hi[i] ^ hi[j]) >>> 0) + pop32((lo[i] ^ lo[j]) >>> 0) > maxDist) continue;
    for (const k of [a, b]) if (splits.get(k) !== 'train') drop.add(k);
  }
  return drop;
}
