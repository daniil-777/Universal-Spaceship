// earthloader.js — the Earth zoom's tile loader (no three.js; Node-testable with a fake fetch and decode): at most
// `maxInFlight` requests overall and `perHost` per host, the lowest priority number first, requests nobody wants any
// more dropped before they start (prune), a failed tile retried only after `retryMs`, Esri's "no imagery here"
// placeholder recognised by its size and SHA-1 and never fetched again, `pin(url)` tiles cached for good in their own
// map (GIBS sends `Cache-Control: no-store`, so its low levels must not be re-fetched), `keepIf(url)` protects a tile
// still in use by the caller from LRU eviction even past `keep` (a shared height tile at the loader's size cap does not
// get evicted out from under a ring that still wants it), a second request for a URL already queued or in flight joins
// the first instead of firing a second fetch, and suspend() / resume() so a closed view stops fetching.
// A job is { url, prio, raw, wanted(), done(bitmap | null, why) }, why = 'ok' | 'failed' | 'blank' | 'dropped'.
import { ESRI_BLANK } from './earthtiles.js';

const hex = (buf) => Array.from(new Uint8Array(buf), (b) => b.toString(16).padStart(2, '0')).join('');
// Pure-JS SHA-1 (RFC 3174), used only when crypto.subtle is unavailable (e.g. plain http on the LAN).
export function sha1Js(buf) {
  const data = new Uint8Array(buf), padded = new Uint8Array((data.length + 9 + 63) & ~63);
  padded.set(data); padded[data.length] = 0x80;
  const view = new DataView(padded.buffer), bits = data.length * 8;
  view.setUint32(padded.length - 8, Math.floor(bits / 2 ** 32)); view.setUint32(padded.length - 4, bits >>> 0);
  let h0 = 0x67452301, h1 = 0xefcdab89, h2 = 0x98badcfe, h3 = 0x10325476, h4 = 0xc3d2e1f0;
  const w = new Uint32Array(80);
  for (let chunk = 0; chunk < padded.length; chunk += 64) {
    for (let i = 0; i < 16; i++) w[i] = view.getUint32(chunk + i * 4);
    for (let i = 16; i < 80; i++) { const v = w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]; w[i] = (v << 1) | (v >>> 31); }
    let a = h0, b = h1, c = h2, d = h3, e = h4;
    for (let i = 0; i < 80; i++) {
      const f = i < 20 ? (b & c) | (~b & d) : i < 40 ? b ^ c ^ d : i < 60 ? (b & c) | (b & d) | (c & d) : b ^ c ^ d;
      const k = i < 20 ? 0x5a827999 : i < 40 ? 0x6ed9eba1 : i < 60 ? 0x8f1bbcdc : 0xca62c1d6;
      const t = (((a << 5) | (a >>> 27)) + f + e + k + w[i]) >>> 0;
      e = d; d = c; c = (b << 30) | (b >>> 2); b = a; a = t;
    }
    h0 = (h0 + a) >>> 0; h1 = (h1 + b) >>> 0; h2 = (h2 + c) >>> 0; h3 = (h3 + d) >>> 0; h4 = (h4 + e) >>> 0;
  }
  const be = (n) => (n >>> 0).toString(16).padStart(8, '0');
  return be(h0) + be(h1) + be(h2) + be(h3) + be(h4);
}
const sha1Hex = async (buf) => (globalThis.crypto && crypto.subtle ? hex(await crypto.subtle.digest('SHA-1', buf)) : sha1Js(buf));
const hostOf = (url) => { try { return new URL(url).host; } catch (e) { return url; } };

export function createTileLoader({ maxInFlight = 8, perHost = { default: 6, 'gibs.earthdata.nasa.gov': 10, 'tiles.maps.eox.at': 10 },
  pin = () => false, keepIf = () => false, keep = 256, retryMs = 30000, timeoutMs = 15000, blank = ESRI_BLANK, fetchImpl = (url, opts) => fetch(url, opts),
  decode = (blob, opts) => createImageBitmap(blob, opts), hashHex = sha1Hex, now = () => performance.now() } = {}) {
  // pin(url) is bounded on its own: z ≤ 8 is at most Σ 4^z (z = 0..8) ≈ 87k URLs per source (GIBS colour, Terrarium
  // height), and a real session's rings only ever touch a few hundred of them.
  const cache = new Map(), pinned = new Map(), failed = new Map(), warned = new Set(), queue = [], pending = new Map(), recent = [];
  const stats = { requested: 0, loaded: 0, failed: 0, blank: 0, deduped: 0 };
  const hostLoad = new Map();
  let inFlight = 0, suspended = false;
  const isBlank = async (buf) => !!blank && buf.byteLength === blank.bytes && (await hashHex(buf)) === blank.sha1;
  const remember = (url, bmp) => {
    if (pin(url)) { pinned.set(url, bmp); return; }
    cache.delete(url); cache.set(url, bmp);
    // evict the oldest entry that keepIf does not claim; if every cached entry is claimed, keep is exceeded for now
    if (cache.size > keep) for (const k of cache.keys()) if (!keepIf(k)) { cache.delete(k); break; }
  };
  const note = (bad) => { recent.push(bad ? 1 : 0); if (recent.length > 24) recent.shift(); };
  const hostCap = (host) => perHost[host] ?? perHost.default ?? Infinity;
  const finish = (jobs, bmp, why) => { for (const job of jobs) job.done(job.wanted() ? bmp : null, job.wanted() ? why : 'dropped'); };
  async function run(entry) {
    inFlight++; const host = hostOf(entry.url); hostLoad.set(host, (hostLoad.get(host) || 0) + 1);
    let bmp = null, why = 'failed';
    // a server that never answers must not hold a slot for ever
    const ctl = typeof AbortController === 'function' ? new AbortController() : null, timer = ctl && setTimeout(() => ctl.abort(), timeoutMs);
    try {
      const res = await fetchImpl(entry.url, { mode: 'cors', signal: ctl ? ctl.signal : undefined });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const buf = await res.arrayBuffer();
      if (await isBlank(buf)) { stats.blank++; failed.set(entry.url, Infinity); why = 'blank'; }
      else { bmp = await decode(new Blob([buf]), entry.raw ? { colorSpaceConversion: 'none', premultiplyAlpha: 'none' } : {}); remember(entry.url, bmp); stats.loaded++; why = 'ok'; }
      note(false);
    } catch (e) {
      stats.failed++; failed.set(entry.url, now()); note(true);
      if (!warned.has(host)) { warned.add(host); console.warn(`earth zoom: tiles from ${host} are not loading (${e.message})`); }
    }
    if (timer) clearTimeout(timer);
    inFlight--; hostLoad.set(host, hostLoad.get(host) - 1); pending.delete(entry.url);
    if (suspended) finish(entry.jobs, null, 'dropped'); else finish(entry.jobs, bmp, why);
    pump();
  }
  function pump() {
    if (suspended) return;
    if (queue.length > 1) queue.sort((a, b) => a.prio - b.prio);
    for (let i = 0; inFlight < maxInFlight && i < queue.length; ) {
      const entry = queue[i], host = hostOf(entry.url);
      if ((hostLoad.get(host) || 0) >= hostCap(host)) { i++; continue; }
      queue.splice(i, 1);
      entry.jobs = entry.jobs.filter((job) => job.wanted() || (job.done(null, 'dropped'), false));
      if (!entry.jobs.length) { pending.delete(entry.url); continue; }
      run(entry);
    }
  }
  return {
    stats, get inFlight() { return inFlight; }, get queued() { return queue.length; },
    get failRate() { return recent.length < 8 ? 0 : recent.reduce((a, b) => a + b, 0) / recent.length; },
    peek: (url) => pinned.get(url) || cache.get(url) || null,
    request(job) {
      if (suspended) { job.done(null, 'dropped'); return; }
      const hit = pinned.get(job.url) || cache.get(job.url);
      if (hit) { if (cache.has(job.url)) remember(job.url, hit); job.done(hit, 'ok'); return; }
      const t = failed.get(job.url);
      if (t !== undefined && now() - t < retryMs) { job.done(null, t === Infinity ? 'blank' : 'failed'); return; }
      const entry = pending.get(job.url);
      if (entry) { entry.jobs.push(job); entry.prio = Math.min(entry.prio, job.prio); stats.deduped++; return; }
      stats.requested++; // requested counts only the fetch that started it; an attached duplicate is deduped instead
      const fresh = { url: job.url, prio: job.prio, raw: job.raw, jobs: [job] };
      pending.set(job.url, fresh); queue.push(fresh); pump();
    },
    prune() {
      for (let i = queue.length - 1; i >= 0; i--) {
        const entry = queue[i];
        entry.jobs = entry.jobs.filter((job) => job.wanted() || (job.done(null, 'dropped'), false));
        if (!entry.jobs.length) { pending.delete(entry.url); queue.splice(i, 1); }
      }
    },
    suspend() { suspended = true; for (const entry of queue.splice(0)) { pending.delete(entry.url); finish(entry.jobs, null, 'dropped'); } },
    resume() { suspended = false; },
  };
}
