// vlm/gen/geo/terrarium.js — AWS Terrarium elevations for Z facts, decoded raw (R*256 + G + B/256 - 32768, so sea stays
// below 0; the site's decodeTerrarium clamps at 0) from the capture's tile cache only: the build never fetches (spec §4.2).
import sharp from 'sharp';
import { lonLatToTile, MAX_LAT } from '../../../src/earthtiles.js';
export const decodeTerrariumRaw = (r, g, b) => r * 256 + g + b / 256 - 32768;
export const terrariumUrl = (z, x, y) => `https://s3.amazonaws.com/elevation-tiles-prod/terrarium/${z}/${x}/${y}.png`;
// A Terrarium tile is RGB: a source with fewer colour channels (sharp would expand grey to RGB) is refused, never guessed
// at; the raw stride comes from the decoder's info.channels.
export async function decodeTilePng(buf) {
  const meta = await sharp(buf).metadata(), colour = meta.channels - (meta.hasAlpha ? 1 : 0);
  if (!(colour >= 3)) throw new Error(`Terrarium tile has ${colour} colour channels; RGB expected`);
  const { data, info } = await sharp(buf).removeAlpha().raw().toBuffer({ resolveWithObject: true }), c = info.channels;
  const elev = new Float32Array(info.width * info.height);
  for (let i = 0; i < elev.length; i++) elev[i] = decodeTerrariumRaw(data[i * c], data[i * c + 1], data[i * c + 2]);
  return { w: info.width, h: info.height, elev };
}
// Only successful decodes are memoised (as the promise, so concurrent reads of one tile decode it once); a cache miss or
// a failed decode is looked up again next time. At most maxTiles decoded tiles are kept (least recently used first out;
// a 256x256 tile is 256 KB of Float32). Web Mercator tiles end at MAX_LAT: beyond it there is no height (null).
export function createElevationReader(cache, { maxTiles = 256 } = {}) {
  const memo = new Map();
  return { async at(lat, lon, z) {
    if (!(Math.abs(lat) <= MAX_LAT)) return null;
    const t = lonLatToTile(lon, lat, z), n = 2 ** z, x = ((Math.floor(t.x) % n) + n) % n, y = Math.min(n - 1, Math.max(0, Math.floor(t.y))), url = terrariumUrl(z, x, y);
    let p = memo.get(url);
    if (p === undefined) {
      const hit = cache.get(url); if (!hit) return null;
      p = decodeTilePng(Buffer.from(hit.body)); if (memo.size >= maxTiles) memo.delete(memo.keys().next().value);
    } else memo.delete(url);
    memo.set(url, p);
    let tile;
    try { tile = await p; } catch { if (memo.get(url) === p) memo.delete(url); return null; }
    return tile.elev[Math.min(tile.h - 1, Math.floor((t.y - Math.floor(t.y)) * tile.h)) * tile.w + Math.min(tile.w - 1, Math.floor((t.x - Math.floor(t.x)) * tile.w))];
  } };
}
// sea_frac is Natural Earth only; the coast side also counts Terrarium < 0 m at points flagged coastRefine (spec §4.2)
export function elevStats(points, { W = 896, H = 504 } = {}) {
  const isSea = (p) => p.sea || (p.coastRefine && p.elev !== null && p.elev < 0), v = points.filter((p) => p.elev !== null), sea = points.filter(isSea);
  const side = (f) => { const s = points.filter(f); return s.length ? s.filter(isSea).length / s.length : 0; };
  const sides = { left: side((p) => p.px < W / 2), right: side((p) => p.px >= W / 2), top: side((p) => p.py < H / 2), bottom: side((p) => p.py >= H / 2) };
  const seaFrac = points.length ? points.filter((p) => p.sea).length / points.length : 0, coast = seaFrac > 0.05 && seaFrac < 0.95 ? Object.entries(sides).sort((a, b) => b[1] - a[1])[0][0] : null;
  if (!v.length) return { n: 0, min: null, max: null, mean: null, relief: null, sea_frac: seaFrac, coast_side: coast, sea_pts: sea.length };
  const e = v.map((p) => p.elev), min = Math.min(...e), max = Math.max(...e);
  return { n: v.length, min, max, mean: e.reduce((a, b) => a + b, 0) / e.length, relief: max - min, sea_frac: seaFrac, coast_side: coast, sea_pts: sea.length };
}
