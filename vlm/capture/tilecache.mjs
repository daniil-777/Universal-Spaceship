// vlm/capture/tilecache.mjs — the tile cache of spec §6 / R15: one WAL SQLite file per upstream host on LaCie (WAL checked
// on the exFAT drive), keyed by the served URL, and the politeness limiter: <= 4 requests in flight and <= 8 requests
// sent per second per host (one at a time until the host has answered once), a descriptive User-Agent that carries a
// contact only if one is given (the optional APV_CONTACT), a 30 s timeout per request, and per-host budgets (the EOX
// 80,000 by default, merged with any given; `used` carries a resumed run's count) reserved before queueing. A 403 or 429
// latches its host: every queued and later request to it rejects with StopDrive before sending. Only valid responses are
// cached: a 200 with the whole body and, for tile hosts, an image type whose bytes start with a JPEG or PNG signature
// (an HTML error page or captcha served as 200 is a failed fetch, never imagery).
import { DatabaseSync } from 'node:sqlite';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

export class StopDrive extends Error {}
export function openTileCache(dir, host) {
  fs.mkdirSync(dir, { recursive: true });
  const db = new DatabaseSync(path.join(dir, `${host}.sqlite`));
  db.exec('PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; CREATE TABLE IF NOT EXISTS tiles (url TEXT PRIMARY KEY, body BLOB NOT NULL, ctype TEXT NOT NULL, sha1 TEXT NOT NULL, bytes INTEGER NOT NULL, fetched_ms INTEGER NOT NULL)');
  const get = db.prepare('SELECT body, ctype, sha1 FROM tiles WHERE url = ?'), put = db.prepare('INSERT OR REPLACE INTO tiles VALUES (?, ?, ?, ?, ?, ?)'), cnt = db.prepare('SELECT COUNT(*) AS n FROM tiles');
  return { get: (url) => get.get(url) || null, put: (url, body, ctype) => put.run(url, body, ctype, crypto.createHash('sha1').update(body).digest('hex'), body.length, Date.now()), count: () => cnt.get().n, close: () => db.close() };
}
const TILE_HOSTS = new Set(['tiles.maps.eox.at', 'gibs.earthdata.nasa.gov', 's3.amazonaws.com']);
export const DEFAULT_BUDGETS = Object.freeze({ 'tiles.maps.eox.at': 80000 });
const isImage = (b) => b.length >= 4 && ((b[0] === 0xff && b[1] === 0xd8 && b[2] === 0xff) || (b[0] === 0x89 && b[1] === 0x50 && b[2] === 0x4e && b[3] === 0x47));
const failed = () => ({ status: 502, body: Buffer.alloc(0), ctype: 'text/plain', outcome: 'fetch-error' });
// stats: hit (served from the cache), fetch (requests sent upstream), error (sent requests that failed); byHost[h] has
// fetch and error as above plus used (the budget count: the given used[h] plus this run's fetch)
// Lanes (other drive processes) share hosts: elsewhere[h]() is the budget the other lanes used (added to the cap check),
// sharedStop(h) a stop message another lane latched (checked before every send), onStop(h, status) persists this lane's latch.
export function createUpstream({ dir, contact, budgets = {}, used = {}, elsewhere = {}, sharedStop = () => null, onStop = () => {}, fetchImpl = fetch, maxInFlight = 4, maxPerSec = 8, timeoutMs = 30000 }) {
  const limits = { ...DEFAULT_BUDGETS, ...budgets }, caches = new Map(), lanes = new Map(), pending = new Map(), stats = { hit: 0, fetch: 0, error: 0, byHost: {} };
  const ua = contact ? `AstroPilotVision/0 (research dataset capture; low-rate, cached; contact ${contact})` : 'AstroPilotVision/0 (research dataset capture; low-rate, cached)';
  const cache = (h) => { if (!caches.has(h)) caches.set(h, openTileCache(dir, h)); return caches.get(h); };
  const hostStats = (h) => (stats.byHost[h] ||= { fetch: 0, error: 0, used: used[h] || 0 });
  // per host: live (in flight), starts (send times in the last second), queued (admitted, not sent; they hold budget),
  // open (the host has answered once), stop (the StopDrive message once a 403/429 latched the host)
  const lane = (h) => { if (!lanes.has(h)) lanes.set(h, { live: 0, starts: [], queued: 0, open: false, stop: null }); return lanes.get(h); };
  const checkShared = (h, L) => { if (!L.stop) { const m = sharedStop(h); if (m) L.stop = m; } return L.stop; };
  // send() runs under the host's limits. The start is stamped in the same tick as the send, so a blocked event loop (a
  // slow synchronous cache open or write on the USB drive) cannot bunch granted-but-unsent requests into a burst.
  function limited(h, send) {
    const L = lane(h), hs = hostStats(h);
    return new Promise((resolve, reject) => {
      const attempt = () => {
        if (checkShared(h, L)) { L.queued--; reject(new StopDrive(L.stop)); return; }
        const now = Date.now(), cap = L.open ? maxInFlight : 1; L.starts = L.starts.filter((t) => now - t < 1000);
        if (L.live >= cap || L.starts.length >= maxPerSec) { setTimeout(attempt, L.live >= cap ? 5 : 1000 - (now - L.starts[0]) + 1); return; }
        L.queued--; L.live++; L.starts.push(now); hs.fetch++; hs.used++; stats.fetch++;
        send().finally(() => { L.live--; }).then(resolve, reject);
      };
      attempt();
    });
  }
  async function fetchOnce(url, h, c) {
    const L = lane(h), hs = hostStats(h); let r, body;
    try {
      ({ r, body } = await limited(h, async () => {
        const res = await fetchImpl(url, { headers: { 'user-agent': ua }, signal: AbortSignal.timeout(timeoutMs) });
        if (res.status === 403 || res.status === 429) {
          if (!L.stop) { L.stop = `${res.status} from ${h}: stopping without retry`; try { onStop(h, res.status); } catch { /* the latch holds in memory either way */ } }
          try { await res.body?.cancel(); } catch { /* the body is dropped either way */ }
          return { r: res, body: null };
        }
        L.open = true;
        return { r: res, body: Buffer.from(await res.arrayBuffer()) };
      }));
    } catch (err) { if (err instanceof StopDrive) throw err; hs.error++; stats.error++; return failed(); }
    if (body === null) throw new StopDrive(L.stop);
    // fetch decodes gzip/br bodies while content-length stays the encoded size, so the length is compared only unencoded
    const ctype = r.headers.get('content-type') || '', len = r.headers.get('content-length'), enc = r.headers.get('content-encoding');
    const whole = len === null || (enc !== null && enc !== 'identity') || +len === body.length;
    const ok = r.status === 200 && body.length > 0 && whole && (!TILE_HOSTS.has(h) || (ctype.startsWith('image/') && isImage(body)));
    if (!ok) { hs.error++; stats.error++; return failed(); }
    c.put(url, body, ctype);
    return { status: 200, body, ctype, outcome: 'fetch' };
  }
  async function get(url) {
    const h = new URL(url).host, L = lane(h);
    if (checkShared(h, L)) throw new StopDrive(L.stop);
    const c = cache(h), hit = c.get(url);
    if (hit) { stats.hit++; return { status: 200, body: Buffer.from(hit.body), ctype: hit.ctype, outcome: 'hit' }; }
    // a second request for a URL already on its way shares that one upstream request (served as a hit if it succeeds)
    if (pending.has(url)) return pending.get(url).then((r) => (r.outcome === 'fetch' ? (stats.hit++, { ...r, outcome: 'hit' }) : r));
    const hs = hostStats(h);
    if (limits[h] !== undefined && hs.used + L.queued + (elsewhere[h] ? elsewhere[h]() : 0) >= limits[h]) throw new StopDrive(`upstream budget of ${limits[h]} requests reached for ${h}`);
    L.queued++;
    const p = fetchOnce(url, h, c).finally(() => pending.delete(url));
    pending.set(url, p);
    return p;
  }
  return { get, stats: () => JSON.parse(JSON.stringify(stats)), close: () => { for (const c of caches.values()) c.close(); caches.clear(); } };
}
