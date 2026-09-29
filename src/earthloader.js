// earthloader.js — the Earth zoom's tile loader (no three.js; Node-testable with a fake fetch and decode): at most
// `maxInFlight` requests, the lowest priority number first, requests nobody wants any more dropped before they start
// (prune), a failed tile retried only after `retryMs`, Esri's "no imagery here" placeholder recognised by its size and
// SHA-1 and never fetched again, and suspend() / resume() so a closed view stops fetching.
// A job is { url, prio, raw, wanted(), done(bitmap | null, why) }, why = 'ok' | 'failed' | 'blank' | 'dropped'.
import { ESRI_BLANK } from './earthtiles.js';

const hex = (buf) => Array.from(new Uint8Array(buf), (b) => b.toString(16).padStart(2, '0')).join('');
const sha1Hex = async (buf) => (globalThis.crypto && crypto.subtle ? hex(await crypto.subtle.digest('SHA-1', buf)) : '');
const hostOf = (url) => { try { return new URL(url).host; } catch (e) { return url; } };

export function createTileLoader({ maxInFlight = 8, keep = 192, retryMs = 30000, timeoutMs = 15000, blank = ESRI_BLANK, fetchImpl = (url, opts) => fetch(url, opts),
  decode = (blob, opts) => createImageBitmap(blob, opts), hashHex = sha1Hex, now = () => performance.now() } = {}) {
  const cache = new Map(), failed = new Map(), warned = new Set(), queue = [], recent = [], stats = { requested: 0, loaded: 0, failed: 0, blank: 0 };
  let inFlight = 0, suspended = false;
  const isBlank = async (buf) => !!blank && buf.byteLength === blank.bytes && (await hashHex(buf)) === blank.sha1;
  const remember = (url, bmp) => { cache.delete(url); cache.set(url, bmp); if (cache.size > keep) cache.delete(cache.keys().next().value); };
  const note = (bad) => { recent.push(bad ? 1 : 0); if (recent.length > 24) recent.shift(); };
  async function run(job) {
    inFlight++; let bmp = null, why = 'failed';
    // a server that never answers must not hold a slot for ever
    const ctl = typeof AbortController === 'function' ? new AbortController() : null, timer = ctl && setTimeout(() => ctl.abort(), timeoutMs);
    try {
      const res = await fetchImpl(job.url, { mode: 'cors', signal: ctl ? ctl.signal : undefined });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const buf = await res.arrayBuffer();
      if (await isBlank(buf)) { stats.blank++; failed.set(job.url, Infinity); why = 'blank'; }
      else { bmp = await decode(new Blob([buf]), job.raw ? { colorSpaceConversion: 'none', premultiplyAlpha: 'none' } : {}); remember(job.url, bmp); stats.loaded++; why = 'ok'; }
      note(false);
    } catch (e) {
      stats.failed++; failed.set(job.url, now()); note(true);
      const host = hostOf(job.url);
      if (!warned.has(host)) { warned.add(host); console.warn(`earth zoom: tiles from ${host} are not loading (${e.message})`); }
    }
    if (timer) clearTimeout(timer);
    inFlight--;
    if (suspended) job.done(null, 'dropped'); else job.done(bmp, why);
    pump();
  }
  function pump() {
    if (suspended) return;
    if (queue.length > 1) queue.sort((a, b) => a.prio - b.prio);
    while (inFlight < maxInFlight && queue.length) { const job = queue.shift(); if (job.wanted()) run(job); else job.done(null, 'dropped'); }
  }
  return {
    stats, get inFlight() { return inFlight; }, get queued() { return queue.length; },
    get failRate() { return recent.length < 8 ? 0 : recent.reduce((a, b) => a + b, 0) / recent.length; },
    peek: (url) => cache.get(url) || null,
    request(job) {
      if (suspended) { job.done(null, 'dropped'); return; }
      const hit = cache.get(job.url);
      if (hit) { remember(job.url, hit); job.done(hit, 'ok'); return; }
      const t = failed.get(job.url);
      if (t !== undefined && now() - t < retryMs) { job.done(null, t === Infinity ? 'blank' : 'failed'); return; }
      stats.requested++; queue.push(job); pump();
    },
    prune() { for (let i = queue.length - 1; i >= 0; i--) if (!queue[i].wanted()) { queue[i].done(null, 'dropped'); queue.splice(i, 1); } },
    suspend() { suspended = true; for (const job of queue.splice(0)) job.done(null, 'dropped'); },
    resume() { suspended = false; },
  };
}
