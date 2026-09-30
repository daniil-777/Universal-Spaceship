// node vlm/capture/report.mjs --run <name> [--out sizes.json]: seconds per kept record per family from throughput.jsonl (samples and
// wall_s both include the twins, which count toward a family's size, §8), projected hours for the §3.2 sizes, and the R17 shrink:
// over-share families first, >= 50 %. Also prints the §7.6 detail (ms per frame, capture and PNG encode ms and bytes, label and
// rollout ms, Z settle seconds and tiles per view). A --measure run's extra encode-only pass (measure_extra_s) is not capture work
// and is taken off the wall time.
import fs from 'node:fs';
import path from 'node:path';
export const SIZES = Object.freeze({ Z: 800, S: 1200, A: 1600, L: 800, D: 600 });
const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; }, L = '/Volumes/LaCie/astro-pilot/vlm';
export function shrink(secPer, budgetH = 12) {
  const size = { ...SIZES }, hours = () => Object.entries(size).reduce((a, [f, n]) => a + (n * secPer[f]) / 3600, 0), total0 = Object.values(SIZES).reduce((a, b) => a + b, 0);
  while (hours() > budgetH) {
    const T = hours(), over = Object.keys(size).filter((f) => size[f] > Math.ceil(SIZES[f] / 2)).map((f) => [f, (size[f] * secPer[f] / 3600 / T) / (SIZES[f] / total0)]).sort((a, b) => b[1] - a[1]);
    if (!over.length) break; const f = over[0][0]; size[f] = Math.max(Math.ceil(SIZES[f] / 2), Math.floor(size[f] * 0.95));
  }
  return { sizes: size, hours: hours() };
}
const sum = (rows, f) => rows.reduce((a, r) => a + (f(r) || 0), 0), r1 = (x) => (Number.isFinite(x) ? +x.toFixed(1) : null);
export function detail(fam, rows) {
  const samples = sum(rows, (r) => r.samples), wall = sum(rows, (r) => r.wall_s - (r.measure_extra_s || 0)), frames = sum(rows, (r) => r.frames), frameMs = sum(rows, (r) => r.frameMs ?? r.frame_ms);
  const d = { episodes: rows.length, samples, wall_s: r1(wall), sec_per_record: samples ? +(wall / samples).toFixed(2) : null, samples_per_s: wall ? +(samples / wall).toFixed(4) : null, ms_per_frame: frames ? +(frameMs / frames).toFixed(2) : null, frames };
  if (fam === 'Z') {
    const settle = rows.flatMap((r) => r.settle_s || []).sort((a, b) => a - b), tiles = rows.flatMap((r) => r.tiles || []), views = settle.length;
    Object.assign(d, { views, dropped_views: sum(rows, (r) => (r.dropped || []).length + (r.split_dropped || []).length), settle_s_mean: views ? r1(settle.reduce((a, b) => a + b, 0) / views) : null, settle_s_median: views ? r1(settle[views >> 1]) : null, settle_s_max: views ? r1(settle[views - 1]) : null,
      tile_requests_per_view: views ? r1(sum(tiles, (t) => t.requests) / views) : null, upstream_tiles_per_view: views ? r1(sum(tiles, (t) => t.upstream) / views) : null,
      capture_host_ms: views ? r1(sum(rows, (r) => (r.cap_host_ms || 0) * (r.views || 0)) / views) : null, png_bytes: views ? Math.round(sum(rows, (r) => (r.png_bytes || 0) * (r.views || 0)) / views) : null, boot_s_mean: r1(sum(rows, (r) => r.boot_ms) / rows.length / 1000) });
  } else {
    const caps = sum(rows, (r) => r.caps), labels = sum(rows, (r) => r.labels), encN = sum(rows, (r) => r.encN);
    Object.assign(d, { captures: caps, capture_host_ms: caps ? r1(sum(rows, (r) => r.capHostMs) / caps) : null, render_encode_ms: caps ? r1(sum(rows, (r) => r.capPageMs) / caps) : null, png_encode_ms: encN ? r1(sum(rows, (r) => r.encMs) / encN) : null,
      png_bytes: caps ? Math.round(sum(rows, (r) => r.pngBytes) / caps) : null, labels, label_host_ms: labels ? r1(sum(rows, (r) => r.labelHostMs) / labels) : null, rollout_ms: labels ? r1(sum(rows, (r) => r.rolloutMs) / labels) : null,
      facts_ms: labels ? r1(sum(rows, (r) => r.factsMs) / labels) : null, boot_s_mean: r1(sum(rows, (r) => r.bootMs) / rows.length / 1000), warmup_s_mean: r1(sum(rows, (r) => r.warmMs) / rows.length / 1000), crashed_episodes: rows.filter((r) => r.crashed).length });
  }
  // the licence ledger over the family's episodes (§6): outcomes per requested host and requests that reached an upstream host
  const led = { by_host: {}, upstream_by_host: {}, esri_reached_network: 0 };
  for (const r of rows.filter((x) => x.ledger)) {
    for (const [h, o] of Object.entries(r.ledger.byHost)) for (const [k, v] of Object.entries(o)) { const hh = /^127\.0\.0\.1:/.test(h) ? 'site' : h; (led.by_host[hh] ||= {})[k] = (led.by_host[hh][k] || 0) + v; }
    for (const [h, v] of Object.entries(r.ledger.upstreamFetchesByHost)) led.upstream_by_host[h] = (led.upstream_by_host[h] || 0) + v;
    led.esri_reached_network += r.ledger.esriReachedNetwork;
  }
  d.ledger = led;
  return d;
}
if (process.argv[1] && process.argv[1].endsWith('report.mjs')) {
  const sec = {}, det = {}, run = arg('run');
  for (const f of Object.keys(SIZES)) {
    const p = path.join(L, 'raw', run, f, 'throughput.jsonl'); if (!fs.existsSync(p)) continue;
    const rows = fs.readFileSync(p, 'utf8').trim().split('\n').filter(Boolean).map((l) => JSON.parse(l)); det[f] = detail(f, rows); sec[f] = det[f].sec_per_record;
  }
  for (const [f, d] of Object.entries(det)) console.log(`${f}:`, JSON.stringify(d));
  console.log('seconds per kept record (twins included, spec §8):', JSON.stringify(sec));
  const part = Object.fromEntries(Object.entries(sec).filter(([, s]) => s).map(([f, s]) => [f, +(SIZES[f] * s / 3600).toFixed(2)])), used = Object.values(part).reduce((a, b) => a + b, 0);
  console.log(`hours at the §3.2 sizes for the measured families: ${JSON.stringify(part)} = ${used.toFixed(2)} h of 12 h; ${(12 - used).toFixed(2)} h left for ${Object.keys(SIZES).filter((f) => !part[f]).join(', ') || 'nothing'}`);
  let out = { secPer: sec, detail: det, partialHours: part, date: new Date().toISOString() };
  if (Object.keys(SIZES).every((f) => sec[f])) { const r = shrink(sec); console.log(`projected ${r.hours.toFixed(1)} h for ${JSON.stringify(r.sizes)}`); out = { ...r, ...out }; }
  if (arg('out')) fs.writeFileSync(arg('out'), JSON.stringify(out, null, 1));
}
