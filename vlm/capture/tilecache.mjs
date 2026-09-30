// vlm/capture/tilecache.mjs — the tile cache of spec §6 / R15: one WAL SQLite file per upstream host on LaCie (WAL checked
// on the exFAT drive), keyed by the served URL, and the politeness limiter: <= 4 requests in flight and <= 8 requests
// sent per second per host, a descriptive User-Agent that carries a contact only if one is given (the optional
// APV_CONTACT), an 80,000 EOX budget reserved before queueing; 403/429 or the budget stop the drive. Only valid responses
// are cached: a 200 with the whole body and, for tile hosts, an image type whose bytes start with a JPEG or PNG signature
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
const isImage = (b) => b.length >= 4 && ((b[0] === 0xff && b[1] === 0xd8 && b[2] === 0xff) || (b[0] === 0x89 && b[1] === 0x50 && b[2] === 0x4e && b[3] === 0x47));
const failed = () => ({ status: 502, body: Buffer.alloc(0), ctype: 'text/plain', outcome: 'fetch-error' });
export function createUpstream({ dir, contact, budgets = { 'tiles.maps.eox.at': 80000 }, fetchImpl = fetch, maxInFlight = 4, maxPerSec = 8 }) {
  const caches = new Map(), lanes = new Map(), pending = new Map(), stats = { hit: 0, fetch: 0, error: 0, byHost: {} };
  const ua = contact ? `AstroPilotVision/0 (research dataset capture; low-rate, cached; contact ${contact})` : 'AstroPilotVision/0 (research dataset capture; low-rate, cached)';
  const cache = (h) => { if (!caches.has(h)) caches.set(h, openTileCache(dir, h)); return caches.get(h); };
  const lane = (h) => { if (!lanes.has(h)) lanes.set(h, { live: 0, starts: [] }); return lanes.get(h); };
  // send() runs under the host's limits. The start is stamped in the same tick as the send, so a blocked event loop (a
  // slow synchronous cache open or write on the USB drive) cannot bunch granted-but-unsent requests into a burst.
  function limited(h, send) {
    const L = lane(h);
    return new Promise((resolve, reject) => {
      const attempt = () => {
        const now = Date.now(); L.starts = L.starts.filter((t) => now - t < 1000);
        if (L.live >= maxInFlight || L.starts.length >= maxPerSec) { setTimeout(attempt, L.live >= maxInFlight ? 5 : 1000 - (now - L.starts[0]) + 1); return; }
        L.live++; L.starts.push(now);
        send().finally(() => { L.live--; }).then(resolve, reject);
      };
      attempt();
    });
  }
  async function fetchOnce(url, h, c, hs) {
    let r, body;
    try {
      ({ r, body } = await limited(h, async () => {
        const res = await fetchImpl(url, { headers: { 'user-agent': ua } });
        return { r: res, body: res.status === 403 || res.status === 429 ? null : Buffer.from(await res.arrayBuffer()) };
      }));
    } catch { hs.error++; stats.error++; return failed(); }
    if (r.status === 403 || r.status === 429) throw new StopDrive(`${r.status} from ${h}: stopping without retry`);
    // fetch decodes gzip/br bodies while content-length stays the encoded size, so the length is compared only unencoded
    const ctype = r.headers.get('content-type') || '', len = r.headers.get('content-length'), enc = r.headers.get('content-encoding');
    const whole = len === null || (enc !== null && enc !== 'identity') || +len === body.length;
    const ok = r.status === 200 && body.length > 0 && whole && (!TILE_HOSTS.has(h) || (ctype.startsWith('image/') && isImage(body)));
    if (!ok) { hs.error++; stats.error++; return failed(); }
    c.put(url, body, ctype); stats.fetch++;
    return { status: 200, body, ctype, outcome: 'fetch' };
  }
  async function get(url) {
    const h = new URL(url).host, c = cache(h), hit = c.get(url);
    if (hit) { stats.hit++; return { status: 200, body: Buffer.from(hit.body), ctype: hit.ctype, outcome: 'hit' }; }
    // a second request for a URL already on its way shares that one upstream request (served as a hit if it succeeds)
    if (pending.has(url)) return pending.get(url).then((r) => (r.outcome === 'fetch' ? (stats.hit++, { ...r, outcome: 'hit' }) : r));
    const hs = (stats.byHost[h] ||= { fetch: 0, error: 0 });
    if (budgets[h] !== undefined && hs.fetch >= budgets[h]) throw new StopDrive(`upstream budget of ${budgets[h]} requests reached for ${h}`);
    hs.fetch++;
    const p = fetchOnce(url, h, c, hs).finally(() => pending.delete(url));
    pending.set(url, p);
    return p;
  }
  return { get, stats: () => JSON.parse(JSON.stringify(stats)), close: () => { for (const c of caches.values()) c.close(); caches.clear(); } };
}
