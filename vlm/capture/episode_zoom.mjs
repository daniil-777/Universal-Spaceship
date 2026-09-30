// vlm/capture/episode_zoom.mjs — one Z location = one fresh page with 4 fresh views (spec §3.4): goTo, 94 frames of page time,
// then frames until the Z gate holds and the ledger is clean, frames until the rendered state is steady, then frames to the end
// of the view's page-time slot (the last one is the gate's "one more runFor(16)"), capture. A view not passing the gate in
// 120 s of real time, or not steady within its slot, is dropped and logged.
//
// Determinism fixes (G1; no tolerance changed): the §3.4 pacing is made strict, so page time stands still while any page
// request or zoom-loader job is pending (the loader's 15 s fake-clock abort still never fires, and the frames a view needs no
// longer depend on network speed); the capture waits for the steady state (probe/zoom.js steady()), whose eased blend and
// relief factors are path-independent; and every view is captured at the end of a fixed slot of SLOT frames, so
// provenance.clock is the same in a cold and a warm run.
import { openPage, boot, loadProbe, call, frame, waitIdle } from './session.mjs';
import { discardReason } from './routes.mjs';
export const SLOT = 1600;
const TILE = /^(tiles\.maps\.eox\.at|gibs\.earthdata\.nasa\.gov|s3\.amazonaws\.com|server\.arcgisonline\.com)$/;
async function zframe(S, D) { await waitIdle(S, () => call(S, 'loaderIdle'), { timeoutMs: 60000, stop: D.stop }); await frame(S, D.mode); }
// per view: tile requests the page made in the view's ledger window, and how many of them went to an upstream host
const tilesOf = (ledger, i0, i1) => { const w = ledger.slice(i0, i1).filter((e) => TILE.test(e.host)); return { requests: w.length, upstream: w.filter((e) => e.via === 'fetch').length }; };
export async function runZoomLocation(D, loc) {
  const ledger0 = D.ledger.length, S = await openPage(D.browser, { utcMs: loc.utcMs, family: 'Z', mode: D.mode, routes: D.routes(loc) }), views = [], t0 = Date.now(), drops = [];
  try {
    await boot(S, loc.url, '!!(window.__zoomPage && window.__zoomPage.ready)', D.mode); await loadProbe(S, 'Z', { licence: D.licence });
    const bootMs = Date.now() - t0;
    for (const [index, v] of loc.views.entries()) {
      const i0 = D.ledger.length, tv = Date.now(), f0 = S.nFrames, used = () => S.nFrames - f0; await call(S, 'goView', v);
      for (let i = 0; i < 94; i++) await zframe(S, D);
      let g = await call(S, 'gate');
      while (!g.ok && Date.now() - tv < 120000 && used() < SLOT - 1) { await zframe(S, D); g = await call(S, 'gate'); }
      const gateFrames = used(); let same = 0;
      while (g.ok && same < 2 && used() < SLOT - 1) { await zframe(S, D); same = await call(S, 'steady'); }
      const steadyFrames = used() - gateFrames;
      while (used() < SLOT) await zframe(S, D);
      if (g.ok) { g = await call(S, 'gate'); same = await call(S, 'steady'); }
      const bad = discardReason(D.ledger, i0, D.ledger.length, D.licence), why = bad || (!g.ok ? g.why.join('; ') : same < 2 ? `not steady within the ${SLOT}-frame slot` : null);
      if (why) { drops.push(index); D.log({ drop: 'Z view', loc: loc.id, view: index, why }); continue; }
      const tc = performance.now(), f = await call(S, 'capture'), capHostMs = performance.now() - tc, lab = await call(S, 'label'); f.ledger = [i0, D.ledger.length];
      views.push({ index, v, frame: f, label: lab, ledger: [i0, D.ledger.length], ledger0, settle_s: (Date.now() - tv) / 1000, settle_frames: used(), gate_frames: gateFrames, steady_frames: steadyFrames, tiles: tilesOf(D.ledger, i0, D.ledger.length), capHostMs });
    }
    return { views, drops, logs: S.logs, wallMs: Date.now() - t0, bootMs, frames: S.nFrames, frameMs: S.frameMs };
  } finally { await S.ctx.close(); }
}
