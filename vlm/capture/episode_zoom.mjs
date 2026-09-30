// vlm/capture/episode_zoom.mjs — one Z location = one fresh page with 4 fresh views (spec §3.4): goTo, 94 frames of page time,
// then frames until the Z gate holds and the ledger is clean, frames until the rendered state is steady, then frames to the end
// of the view's page-time slot (the last one is the gate's "one more runFor(16)"), capture.
//
// Determinism fixes (G1; no tolerance changed): the §3.4 pacing is made strict, so page time stands still while any page
// request or zoom-loader job is pending (the loader's 15 s fake-clock abort still never fires, and the frames a view needs no
// longer depend on network speed); the capture waits for the steady state (probe/zoom.js steady()), whose eased blend and
// relief factors are path-independent; and every view is captured at the end of a fixed slot of SLOT frames, so
// provenance.clock is the same in a cold and a warm run.
//
// Drops, each with its own logged reason: 'gate timeout' (120 s of real time without the gate), 'gate: slot used' (the page-
// time slot ran out first), 'gate lost' (it held, then failed at the slot end), 'not steady', a dirty ledger, and 'stall' (no
// request finished for 60 s; the location ends there, the views already captured are kept). A dropped view still fills its
// slot, so the later views of that page keep their page time (step 0 of Task 10).
import { openPage, boot, BOOT, loadProbe, call, frame, waitIdle, Discard } from './session.mjs';
import { discardReason } from './routes.mjs';
export const SLOT = 1600, GATE_REAL_MS = 120000, STALL_MS = 60000;
const TILE = /^(tiles\.maps\.eox\.at|gibs\.earthdata\.nasa\.gov|s3\.amazonaws\.com|server\.arcgisonline\.com)$/;
async function zframe(S, D) { await waitIdle(S, () => call(S, 'loaderIdle'), { timeoutMs: STALL_MS, stop: D.stop }); await frame(S, D.mode); }
// per view: tile requests the page made in the view's ledger window, and how many of them went to an upstream host
const tilesOf = (ledger, i0, i1) => { const w = ledger.slice(i0, i1).filter((e) => TILE.test(e.host)); return { requests: w.length, upstream: w.filter((e) => e.via === 'fetch').length }; };
// The per-view control flow on injected io (the tests drive it with fakes): io.frame() one paced frame, io.gate() -> {ok, why},
// io.steady() -> frames the rendered state has not changed, io.used() -> frames since goView, io.elapsedMs() -> real ms since goView.
// Every outcome, a drop included, ends at the end of the view's slot (step 0 of Task 10), so the next view of the page starts at
// the same page time whether this one was kept or dropped; a stall (io.frame throws) still ends the location.
export async function settleView(io, { slot = SLOT, realMs = GATE_REAL_MS } = {}) {
  const fill = async (r) => { while (io.used() < slot) await io.frame(); return r; };
  for (let i = 0; i < 94; i++) await io.frame();
  let g = await io.gate();
  while (!g.ok && io.used() < slot - 1) {
    if (io.elapsedMs() >= realMs) return fill({ ok: false, why: `gate timeout: ${realMs / 1000} s of real time (${g.why.join('; ')})`, gateFrames: io.used(), steadyFrames: 0 });
    await io.frame(); g = await io.gate();
  }
  if (!g.ok) return fill({ ok: false, why: `gate: the ${slot}-frame slot was used without the gate (${g.why.join('; ')})`, gateFrames: io.used(), steadyFrames: 0 });
  const gateFrames = io.used(); let same = 0;
  while (same < 2 && io.used() < slot - 1) { await io.frame(); same = await io.steady(); }
  const steadyFrames = io.used() - gateFrames;
  while (io.used() < slot) await io.frame();
  g = await io.gate(); same = await io.steady();
  const why = !g.ok ? `gate lost at the slot end (${g.why.join('; ')})` : same < 2 ? `not steady within the ${slot}-frame slot` : null;
  return { ok: !why, why, gateFrames, steadyFrames };
}
export async function runZoomLocation(D, loc) {
  const ledger0 = D.ledger.length, S = await openPage(D.browser, { utcMs: loc.utcMs, family: 'Z', mode: D.mode, routes: D.routes(loc) }), views = [], t0 = Date.now(), drops = [];
  try {
    await boot(S, loc.url, BOOT.zoom, D.mode, D.stop); await loadProbe(S, 'Z', { licence: D.licence });
    const bootMs = Date.now() - t0;
    for (const [index, v] of loc.views.entries()) {
      const i0 = D.ledger.length, tv = Date.now(), f0 = S.nFrames, used = () => S.nFrames - f0; await call(S, 'goView', v);
      let r;
      try { r = await settleView({ frame: () => zframe(S, D), gate: () => call(S, 'gate'), steady: () => call(S, 'steady'), used, elapsedMs: () => Date.now() - tv }); } catch (e) {
        if (!(e instanceof Discard)) throw e;
        drops.push(index); D.log({ drop: 'Z view', loc: loc.id, view: index, why: `stall: ${e.message}` });
        for (const j of loc.views.keys()) if (j > index) { drops.push(j); D.log({ drop: 'Z view', loc: loc.id, view: j, why: 'skipped: the location ended at a stall' }); }
        break;
      }
      const why = discardReason(D.ledger, i0, D.ledger.length, D.licence) || r.why;
      if (why) { drops.push(index); D.log({ drop: 'Z view', loc: loc.id, view: index, why }); continue; }
      const tc = performance.now(), f = await call(S, 'capture'), capHostMs = performance.now() - tc, lab = await call(S, 'label'); f.ledger = [i0, D.ledger.length];
      views.push({ index, v, frame: f, label: lab, ledger: [i0, D.ledger.length], ledger0, settle_s: (Date.now() - tv) / 1000, settle_frames: used(), gate_frames: r.gateFrames, steady_frames: r.steadyFrames, tiles: tilesOf(D.ledger, i0, D.ledger.length), capHostMs });
    }
    return { views, drops, logs: S.logs, wallMs: Date.now() - t0, bootMs, frames: S.nFrames, frameMs: S.frameMs };
  } finally { await S.ctx.close(); }
}
