// vlm/gen/build/dedupe.js — 64-bit dHash of each f2 frame; cross-split pairs within Hamming 4 are dropped from val/test/ood
// (spec §8), never from train. The pairs are found by multi-index hashing (Norouzi et al. 2012): the 64 bits are cut into
// maxDist + 1 chunks, and two hashes within maxDist agree exactly on at least one chunk (pigeonhole), so only hashes sharing a
// chunk value are compared, each candidate verified exactly. Identical hashes are grouped first, so a flood of identical
// frames (black space, an untextured port face) is one group, not n² pairs. The drops equal the all-pairs scan's (tested).
import sharp from 'sharp';

// img: a sharp instance of the frame (the file, or its decoded pixels: both give the same bytes)
export async function dhashOf(img) {
  const px = await img.resize(9, 8, { fit: 'fill', kernel: 'lanczos3' }).greyscale().raw().toBuffer();
  let h = 0n;
  for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) h = (h << 1n) | (px[y * 9 + x + 1] > px[y * 9 + x] ? 1n : 0n);
  return h;
}
export const dhash = (file) => dhashOf(sharp(file));
const pop32 = (x) => { x -= (x >>> 1) & 0x55555555; x = (x & 0x33333333) + ((x >>> 2) & 0x33333333); return (Math.imul((x + (x >>> 4)) & 0x0f0f0f0f, 0x01010101) >>> 24); };
const halves = (h) => [Number((h >> 32n) & 0xffffffffn), Number(h & 0xffffffffn)];
export const hamming = (a, b) => { const [ah, al] = halves(a), [bh, bl] = halves(b); return pop32((ah ^ bh) >>> 0) + pop32((al ^ bl) >>> 0); };
// the chunk values of a hash: `parts` disjoint bit ranges covering all 64 bits (a chunk wider than 48 bits stays a BigInt)
export function chunksOf(h, parts) {
  const w = Math.ceil(64 / parts), out = [];
  for (let c = 0, lo = 0; c < parts; c++, lo += w) {
    const v = (h >> BigInt(lo)) & ((1n << BigInt(Math.max(0, Math.min(w, 64 - lo)))) - 1n);
    out.push(w > 48 ? v : Number(v));
  }
  return out;
}
// hashes: Map key -> bigint; splits: Map key -> split. Returns the keys to drop: every non-train member of a pair from two
// different splits within maxDist.
export function crossSplitDrops(hashes, splits, { maxDist = 4 } = {}) {
  const groups = new Map(), drop = new Set();
  for (const [k, h] of hashes) { let g = groups.get(h); if (!g) groups.set(h, (g = { h, keys: [], splits: new Set() })); g.keys.push(k); g.splits.add(splits.get(k)); }
  const G = [...groups.values()];
  // a key leaves when it is not train and the other group holds a key of another split
  const dropFrom = (g, other) => { for (const k of g.keys) { const s = splits.get(k); if (s !== 'train' && [...other.splits].some((x) => x !== s)) drop.add(k); } };
  for (const g of G) if (g.splits.size > 1) dropFrom(g, g);
  const parts = maxDist + 1, ch = G.map((g) => chunksOf(g.h, parts)), hl = G.map((g) => halves(g.h));
  for (let c = 0; c < parts; c++) {
    const buckets = new Map();
    G.forEach((_, i) => { const v = ch[i][c]; let b = buckets.get(v); if (!b) buckets.set(v, (b = [])); b.push(i); });
    for (const b of buckets.values()) {
      for (let x = 0; x < b.length; x++) for (let y = x + 1; y < b.length; y++) {
        const i = b[x], j = b[y];
        // a pair sharing an earlier chunk was already judged there
        let seen = false; for (let e = 0; e < c && !seen; e++) seen = ch[i][e] === ch[j][e];
        if (seen || pop32((hl[i][0] ^ hl[j][0]) >>> 0) + pop32((hl[i][1] ^ hl[j][1]) >>> 0) > maxDist) continue;
        dropFrom(G[i], G[j]); dropFrom(G[j], G[i]);
      }
    }
  }
  return drop;
}
