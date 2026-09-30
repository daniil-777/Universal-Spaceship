// vlm/capture/session.mjs — one page load = one fresh context (896x504, DPR 1, service workers blocked) with the licence
// routes, the seeded Math.random, the L/D preserveDrawingBuffer wrapper, a paused fake clock (install(utc - 5000) then
// pauseAt(utc)), an all-request in-flight counter, and the readiness waits of spec §7.1-§7.2.
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
// Waits, with the clock paused, until the page is ready (true) or has been quiet for quietMs of real time: no request in
// flight and none finished (false). Network idle alone is not enough: the page keeps working after a response arrives
// (app.js parses and decodes the policy before __ap.ready), and a frame pumped during that work lands ready one fake tick
// later in some runs (fix round 1: seen as a +16 ms provenance.clock and different pixels in one of four S runs).
export async function settleReal(S, readyExpr, { stop = null, quietMs = BOOT_QUIET_MS } = {}) {
  let seen = S.done, since = Date.now();
  for (;;) {
    if (await S.page.evaluate(readyExpr)) return true;
    if (stop && stop.error) throw stop.error;
    if (S.inflight > 0) { await waitIdle(S, null, { stop }); seen = S.done; since = Date.now(); continue; }
    if (S.done !== seen) { seen = S.done; since = Date.now(); } else if (Date.now() - since >= quietMs) return false;
    await sleep(10);
  }
}
export const BOOT_QUIET_MS = 1000;
// the §7.2 per-frame wait holds from boot on, and a boot frame is pumped only when the page has settled without becoming
// ready (it waits on a fake timer, app.js:404's 30 ms), so ready lands on the same fake tick in every run; `stop` (the
// drive's latch) makes a 403/429 during boot surface as StopDrive
export async function boot(S, url, readyExpr, mode, stop = null) {
  await S.page.goto(url, { waitUntil: 'load', timeout: 180000 }); await waitIdle(S, null, { stop });
  for (let i = 0; i < 4000; i++) { if (await settleReal(S, readyExpr, { stop })) return; await frame(S, mode); }
  if (stop && stop.error) throw stop.error;
  throw new Discard('boot: the ready flag never set');
}
export async function loadProbe(S, family, params) { return S.page.evaluate(async ([f, p]) => { const m = await import('/vlm/capture/probe/core.js'); window.__apv = m.createProbe(); return window.__apv.setup(f, p); }, [family, params]); }
export const call = (S, name, arg = null) => S.page.evaluate(([n, a]) => window.__apv.call(n, a), [name, arg]);
