// vlm/capture/session.mjs — one page load = one fresh context (896x504, DPR 1, service workers blocked) with the licence
// routes, the seeded Math.random, the L/D preserveDrawingBuffer wrapper, a paused fake clock (install(utc - 5000) then
// pauseAt(utc)), an all-request in-flight counter, the deterministic boot and the readiness waits of spec §7.1-§7.2.
import { installRoutes } from './routes.mjs';
export class Discard extends Error {}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const SEED = `(() => { let a = (+new URLSearchParams(location.search).get('rs')) >>> 0; Math.random = function () { a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; })();`;
const PDB = `(() => { const g = HTMLCanvasElement.prototype.getContext; HTMLCanvasElement.prototype.getContext = function (type, attrs) { if (type === 'webgl' || type === 'webgl2') attrs = Object.assign({}, attrs, { preserveDrawingBuffer: true }); return g.call(this, type, attrs); }; })();`;
const FREEZE = (utc) => `(() => { const D = Date, T = ${utc}; class P extends D { constructor(...a) { super(...(a.length ? a : [T])); } static now() { return T; } } window.Date = P; })();`;
// runs after Playwright 1.63's clock init scripts (pinned internals): their replay advances ticks by the host gap between the
// install and pauseAt log stamps, so performance.now() would start at a few host ms that differ between runs; reset it to 0
const CLOCK0 = '(() => { const c = globalThis.__pwClock && globalThis.__pwClock.controller; if (c) { c.now(); c._now.ticks = 0; } })();';
export async function openPage(browser, { utcMs, family, mode, routes }) {
  const ctx = await browser.newContext({ baseURL: `http://127.0.0.1:${routes.port}`, viewport: { width: 896, height: 504 }, deviceScaleFactor: 1, serviceWorkers: 'block' });
  await installRoutes(ctx, routes); await ctx.addInitScript(SEED);
  if (family === 'L' || family === 'D') await ctx.addInitScript(PDB);
  if (mode === 'freeze') await ctx.addInitScript(FREEZE(utcMs));
  const page = await ctx.newPage(), S = { ctx, page, n: 0, done: 0, logs: [], nFrames: 0, frameMs: 0, get inflight() { return this.n; } };
  page.on('request', () => { S.n++; }); page.on('requestfinished', () => { S.n--; S.done++; }); page.on('requestfailed', () => { S.n--; S.done++; });
  page.on('pageerror', (e) => S.logs.push(String(e.message)));
  if (mode === 'clock') { await page.clock.install({ time: utcMs - 5000 }); await page.clock.pauseAt(utcMs); await ctx.addInitScript(CLOCK0); }
  return S;
}
// one page frame; S.nFrames / S.frameMs count them and their host wall time for the throughput report (§7.6)
export async function frame(S, mode) { const t = performance.now(); if (mode === 'clock') await S.page.clock.runFor(16); else await sleep(16); S.nFrames++; S.frameMs += performance.now() - t; }
// progress-based: a cold A page queues ~712 strip tiles behind the 8 req/s EOX lane (~90 s), so the timeout counts real time
// since the last finished request, not since the call
export async function waitIdle(S, check = null, { timeoutMs = 10000, stop = null } = {}) {
  let t0 = Date.now(), seen = S.done;
  for (;;) {
    if (stop && stop.error) throw stop.error;
    if (S.inflight === 0 && (!check || (await check()))) return;
    if (S.done !== seen) { seen = S.done; t0 = Date.now(); }
    if (Date.now() - t0 > timeoutMs) throw new Discard(`no request finished and not ready for ${timeoutMs} ms of real time`);
    await sleep(5);
  }
}
// Boot (§7.1; Task 9 review, step 0 of Task 10): deterministic, with no real-time quiet heuristic. A page is pumped a frame only
// while its pump predicate holds, i.e. while it waits on a fake timer; everything else before ready is real-time work
// (module loads, the policy fetch, parse and decode), awaited with the clock paused. S/A (app.js): main() runs synchronously
// to `ui.loading('building the solar system…')` and awaits a 30 ms timer (app.js:404), which fires on the 2nd frame; from
// there it runs to `ui.loading('loading the policy…')` and only real-time work remains. So a frame is pumped only while
// #loadingMsg reads 'building the solar system…' (a still-loading module reads 'warming up the engines…' and is waited for,
// never pumped), and exactly 2 frames must have run at ready (Z, L and D set ready without a frame: 0). A mismatch, a page
// error (__ap.errors or a pageerror), more than maxFrames frames or no ready within readyMs of real time discards the page.
export const BOOT = Object.freeze({
  app: Object.freeze({ ready: '!!(window.__ap && window.__ap.ready)', pump: "(() => { const l = document.getElementById('loadingMsg'); return !!l && l.textContent === 'building the solar system…'; })()", frames: 2 }),
  zoom: Object.freeze({ ready: '!!(window.__zoomPage && window.__zoomPage.ready)', pump: null, frames: 0 }),
  landing: Object.freeze({ ready: '!!(window.__landing && window.__landing.ready)', pump: null, frames: 0 }),
  real: Object.freeze({ ready: '!!(window.__real && window.__real.ready)', pump: null, frames: 0 }),
});
export const BOOT_READY_MS = 60000, BOOT_MAX_FRAMES = 8;
const bootState = (b) => `(() => { const a = window.__ap; return { ready: ${b.ready}, errors: a && a.errors && a.errors.length ? a.errors.slice(0, 3).map(String) : null, pump: ${b.pump ? `!!${b.pump}` : 'false'}, perf: performance.now() }; })()`;
// `stop` (the drive's latch) makes a 403/429 during boot surface as StopDrive; returns {frames, perf} (page performance.now()
// at ready, the fake clock's, for the log)
export async function boot(S, url, b, mode, stop = null, { readyMs = BOOT_READY_MS, maxFrames = BOOT_MAX_FRAMES } = {}) {
  await S.page.goto(url, { waitUntil: 'load', timeout: 180000 }); await waitIdle(S, null, { stop });
  const expr = bootState(b); let n = 0, since = Date.now();
  for (;;) {
    if (stop && stop.error) throw stop.error;
    const s = await S.page.evaluate(expr), errs = s.errors || (S.logs.length ? S.logs.slice(0, 3) : null);
    if (errs) throw new Discard(`boot: page error: ${errs.join(' | ')}`);
    if (s.ready) {
      if (mode === 'clock' && n !== b.frames) throw new Discard(`boot: ready after ${n} frames, expected ${b.frames}`);
      return { frames: n, perf: s.perf };
    }
    if (s.pump) {
      if (n >= maxFrames) throw new Discard(`boot: ${maxFrames} boot frames without leaving the fake-timer wait`);
      await waitIdle(S, null, { stop }); await frame(S, mode); n++; since = Date.now(); continue;
    }
    if (Date.now() - since > readyMs) throw new Discard(`boot: not ready after ${readyMs / 1000} s of real time with the clock paused (${n} frames)`);
    await sleep(10);
  }
}
export async function loadProbe(S, family, params) { return S.page.evaluate(async ([f, p]) => { const m = await import('/vlm/capture/probe/core.js'); window.__apv = m.createProbe(); return window.__apv.setup(f, p); }, [family, params]); }
export const call = (S, name, arg = null) => S.page.evaluate(([n, a]) => window.__apv.call(n, a), [name, arg]);
