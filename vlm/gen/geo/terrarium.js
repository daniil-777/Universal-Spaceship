// vlm/gen/geo/terrarium.js — AWS Terrarium elevations for Z facts, decoded raw (R*256 + G + B/256 - 32768, so sea stays
// below 0; the site's decodeTerrarium clamps at 0) from the capture's tile cache only: the build never fetches (spec §4.2).
import sharp from 'sharp';
import { lonLatToTile } from '../../../src/earthtiles.js';
export const decodeTerrariumRaw = (r, g, b) => r * 256 + g + b / 256 - 32768;
export const terrariumUrl = (z, x, y) => `https://s3.amazonaws.com/elevation-tiles-prod/terrarium/${z}/${x}/${y}.png`;
export async function decodeTilePng(buf) {
  const { data, info } = await sharp(buf).removeAlpha().raw().toBuffer({ resolveWithObject: true }), elev = new Float32Array(info.width * info.height);
  for (let i = 0; i < elev.length; i++) elev[i] = decodeTerrariumRaw(data[i * 3], data[i * 3 + 1], data[i * 3 + 2]);
  return { w: info.width, h: info.height, elev };
}
// The decode promise is memoised, so concurrent reads of one tile decode it once; at most maxTiles decoded tiles are kept
// (least recently used first out; a 256x256 tile is 256 KB of Float32).
export function createElevationReader(cache, { maxTiles = 256 } = {}) {
  const memo = new Map();
  return { async at(lat, lon, z) {
    const t = lonLatToTile(lon, lat, z), n = 2 ** z, x = ((Math.floor(t.x) % n) + n) % n, y = Math.min(n - 1, Math.max(0, Math.floor(t.y))), url = terrariumUrl(z, x, y);
    let p = memo.get(url);
    if (p === undefined) { const hit = cache.get(url); p = hit ? decodeTilePng(Buffer.from(hit.body)) : null; if (memo.size >= maxTiles) memo.delete(memo.keys().next().value); } else memo.delete(url);
    memo.set(url, p);
    const tile = await p; if (!tile) return null;
    return tile.elev[Math.min(tile.h - 1, Math.floor((t.y - Math.floor(t.y)) * tile.h)) * tile.w + Math.min(tile.w - 1, Math.floor((t.x - Math.floor(t.x)) * tile.w))];
  } };
}
export function elevStats(points, { W = 896, H = 504 } = {}) {
  const v = points.filter((p) => p.elev !== null), sea = points.filter((p) => p.sea || (p.elev !== null && p.elev < 0 && p.coastRefine));
  const side = (f) => { const s = points.filter(f); return s.length ? s.filter((p) => p.sea).length / s.length : 0; };
  const sides = { left: side((p) => p.px < W / 2), right: side((p) => p.px >= W / 2), top: side((p) => p.py < H / 2), bottom: side((p) => p.py >= H / 2) };
  const seaFrac = points.length ? points.filter((p) => p.sea).length / points.length : 0, coast = seaFrac > 0.05 && seaFrac < 0.95 ? Object.entries(sides).sort((a, b) => b[1] - a[1])[0][0] : null;
  if (!v.length) return { n: 0, min: null, max: null, mean: null, relief: null, sea_frac: seaFrac, coast_side: coast, sea_pts: sea.length };
  const e = v.map((p) => p.elev), min = Math.min(...e), max = Math.max(...e);
  return { n: v.length, min, max, mean: e.reduce((a, b) => a + b, 0) / e.length, relief: max - min, sea_frac: seaFrac, coast_side: coast, sea_pts: sea.length };
}
