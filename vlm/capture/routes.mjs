// vlm/capture/routes.mjs — default-deny licence routes (spec §6): the site continues; jsdelivr, GIBS <= z8, Terrarium and
// the profile's EOX layer come from the tile cache; Esri is always fulfilled from the profile's EOX layer at the same
// z/y/x and never continued; everything else is aborted. The ledger records host, served layer and outcome per request.
import { LAYERS, PROFILE_LAYERS } from '../gen/schema.js';
export const PROFILE_EOX = Object.freeze({ open: 's2cloudless_3857', nc: 's2cloudless-2025_3857' }), LMAX = Object.freeze({ open: 15, nc: 14 });
const EOX = (layer, z, y, x) => `https://tiles.maps.eox.at/wmts/1.0.0/${layer}/default/g/${z}/${y}/${x}.jpg`;
const TILE_HOSTS = ['tiles.maps.eox.at', 'gibs.earthdata.nasa.gov', 's3.amazonaws.com', 'server.arcgisonline.com'];
export function classify(url, { port, profile }) {
  const u = new URL(url), h = u.host, p = u.pathname; let m;
  if (h === `127.0.0.1:${port}`) return { action: 'continue' };
  if (h === 'cdn.jsdelivr.net') return { action: 'cache', upstream: url, layer: null };
  if (h === 'gibs.earthdata.nasa.gov' && (m = /\/(\d+)\/\d+\/\d+\.jpe?g$/.exec(p))) return +m[1] <= 8 ? { action: 'cache', upstream: url, layer: 'gibs', level: +m[1] } : { action: 'abort', why: 'GIBS above z8' };
  if (h === 's3.amazonaws.com' && (m = /^\/elevation-tiles-prod\/terrarium\/(\d+)\/\d+\/\d+\.png$/.exec(p))) return { action: 'cache', upstream: url, layer: 'terrarium', level: +m[1] };
  if (h === 'server.arcgisonline.com' && (m = /\/tile\/(\d+)\/(\d+)\/(\d+)$/.exec(p))) return +m[1] <= LMAX[profile] ? { action: 'rewrite', upstream: EOX(PROFILE_EOX[profile], m[1], m[2], m[3]), layer: PROFILE_EOX[profile], level: +m[1] } : { action: 'abort', why: `Esri z${m[1]} above Lmax` };
  if (h === 'tiles.maps.eox.at' && (m = /\/wmts\/1\.0\.0\/(s2cloudless(?:-\d{4})?_3857)\/default\/g\/(\d+)\/(\d+)\/(\d+)\.jpg$/.exec(p))) {
    if (PROFILE_LAYERS[profile].includes(m[1])) return { action: 'cache', upstream: url, layer: m[1], level: +m[2] };
    if (profile === 'open' && /^s2cloudless-20(1[89]|2[0-5])_3857$/.test(m[1])) return { action: 'rewrite', upstream: EOX('s2cloudless_3857', m[2], m[3], m[4]), layer: 's2cloudless_3857', level: +m[2] };
    return { action: 'abort', why: `EOX ${m[1]} not allowed in ${profile}` };
  }
  return { action: 'abort', why: 'not on the allow-list' };
}
// The drive's stop latch: the first 403/429 wins over any other stop, earlier or later (a budget stop never hides a ban, so
// the ban is persisted and the drive exits 3); otherwise the first stop stays.
const httpStop = (e) => /^(403|429) from /.test(String(e && e.message));
export function latchStop(stop, err) { if (!stop.error || (httpStop(err) && !httpStop(stop.error))) stop.error = err; return stop.error; }
// A response the tile cache did not serve (a failed upstream fetch: 502, empty body) is ledgered as 'fetch error' even on
// a rewrite, so it is never counted as imagery and always discards its sample. `via` keeps the cache's own outcome.
export async function installRoutes(ctx, { port, profile, upstream, ledger, stop }) {
  await ctx.route('**/*', async (route) => {
    const url = route.request().url(), c = classify(url, { port, profile }), e = { t: Date.now(), host: new URL(url).host, url, layer_id: c.layer ?? null, level: c.level ?? null };
    if (c.action === 'continue') { ledger.push({ ...e, outcome: 'continued' }); return route.continue(); }
    if (c.action === 'abort') { ledger.push({ ...e, outcome: 'aborted', why: c.why }); return route.abort('blockedbyclient'); }
    let r;
    try { r = await upstream.get(c.upstream); } catch (err) { latchStop(stop, err); ledger.push({ ...e, outcome: 'aborted', why: String(err.message) }); return route.abort('failed'); }
    const served = r.outcome === 'hit' || r.outcome === 'fetch';
    ledger.push({ ...e, outcome: !served ? 'fetch error' : c.action === 'rewrite' ? 'rewritten' : r.outcome === 'hit' ? 'cache hit' : 'upstream fetch', via: r.outcome, served: c.upstream });
    return route.fulfill({ status: r.status, body: r.body, headers: { 'content-type': r.ctype || 'application/octet-stream', 'access-control-allow-origin': '*' } });
  });
}
export function imageryFrom(ledger, i0, i1) {
  const seen = new Map();
  for (const e of ledger.slice(i0, i1)) if (e.layer_id && e.level !== null && e.layer_id !== 'terrarium' && ['rewritten', 'cache hit', 'upstream fetch'].includes(e.outcome)) seen.set(`${e.level}|${e.layer_id}`, { level: e.level, layer_id: e.layer_id, ...LAYERS[e.layer_id] });
  return [...seen.values()].sort((a, b) => a.level - b.level || a.layer_id.localeCompare(b.layer_id));
}
export function discardReason(ledger, i0, i1, profile) {
  for (const e of ledger.slice(i0, i1)) {
    if (e.host === 'server.arcgisonline.com' && e.outcome === 'upstream fetch') return 'an upstream fetch reached an Esri host';
    if (TILE_HOSTS.includes(e.host) && (e.outcome === 'aborted' || e.outcome === 'fetch error')) return `an aborted tile request: ${e.url}`;
    if (e.layer_id && !PROFILE_LAYERS[profile].includes(e.layer_id)) return `served layer ${e.layer_id} not allowed in ${profile}`;
  }
  return null;
}
// ledger totals for the report (spec §6): requests per host and outcome, and how many reached an upstream network
export function ledgerCounts(ledger) {
  const byHost = {}, net = {};
  for (const e of ledger) { const h = (byHost[e.host] ||= {}); h[e.outcome] = (h[e.outcome] || 0) + 1; if (e.via === 'fetch') { const up = new URL(e.served).host; net[up] = (net[up] || 0) + 1; } }
  return { byHost, upstreamFetchesByHost: net, esriReachedNetwork: net['server.arcgisonline.com'] || 0 };
}
