// vlm/capture/episode_docking.mjs — one D run = one fresh page, never begun (spec §3.2 D, §3.3, §7.3-§7.4 D, R8): one CYCLE per
// rendered frame; a skip with the probe's skip() (whole cycles, headless), the slot selection (probe/frame.js keepProb: 12
// slots spread over the times left before the clean run's end, w = 0.4 beyond 30 m; sampler_weight = 1/p), then
// 63 rendered cycles (>= 1 s) before f0 whenever a skip happened since the last clip; f0-f2 at 19-21 cycles in the centreline
// view (R8), then the override removed for one render-only frame (chase.png, for Narrator) and restored; f2 snapshots. The one
// runtime kick is armed in the page and lands at its trigger cycle, which the probe's oracle run predicts: a scheduled clip that
// would reach that time yields to the injection clip, pre-rolled to the kick and taken right after it (a clip straddling the
// kick is dropped: the ring is cleared at the injection). Branches after the run ends; a sample whose continued run ends in
// fail:timeout or fail:keep_in is dropped (isDroppedDockingOutcome, ruling T4-d). As in episode_landing.mjs, every page-time
// operation is logged in `ops` and a twin replays them with no kick armed (a failed-P6 twin flies the probe-owned clean sim).
import sharp from 'sharp';
import { openPage, boot, BOOT, loadProbe, call, frame, waitIdle, Discard } from './session.mjs';
import { frameStats, saneFrame, straddles, keepProb, freeSlots, dockingW } from './probe/frame.js';
import { isDroppedDockingOutcome } from '../gen/labels/docking.js';
export const CYCLE_S = 0.1, PRE = 63, MAX_SAMPLES = 12;
const chk = (S) => S.page.evaluate(() => window.__apv.check());
async function cyc(S, D, ops) { await call(S, 'cycle', 1); await waitIdle(S, null, { stop: D.stop }); await frame(S, D.mode); if (ops) ops.push(['cycle']); }
async function render(S, D, ops) { await waitIdle(S, null, { stop: D.stop }); await frame(S, D.mode); if (ops) ops.push(['frame']); }
// a frame failing the blank-buffer guard (alpha 0 or luminance variance <= 4) is returned as null and drops its clip only
async function grab(S, D, mode, T) {
  const t = performance.now(), f = await call(S, 'capture', { mode }), px = await sharp(Buffer.from(f.png.split(',')[1], 'base64')).ensureAlpha().raw().toBuffer();
  if (!saneFrame(frameStats(px))) { if (T) T.drops.blank = (T.drops.blank || 0) + 1; return null; }
  f.ledger = [D.ledger.length, D.ledger.length];
  if (T) { T.grabs++; T.grabMs += performance.now() - t; T.pngBytes += Math.floor(((f.png.length - f.png.indexOf(',') - 1) * 3) / 4); }
  return f;
}
async function replay(S, D, ep, out, T) {
  const buf = {};
  for (const op of ep.ops) {
    if (op[0] === 'skip') await call(S, 'skip', op[1]);
    else if (op[0] === 'cycle') await cyc(S, D, null);
    else if (op[0] === 'frame') await render(S, D, null);
    else if (op[0] === 'centreline') await call(S, 'centreline', op[1]);
    else if (!ep.keep.has(op[1])) continue;
    else if (op[0] === 'clip') buf[op[1]] = { fr: [] };
    else if (op[0] === 'grab') buf[op[1]].fr.push(await grab(S, D, 'centreline', T));
    else if (op[0] === 'chase') buf[op[1]].chase = await grab(S, D, 'chase', T);
    else if (op[0] === 'snap') {
      const b = buf[op[1]], fr = b.fr; if (!clean(fr.concat([b.chase]), T)) continue;
      // M-4: a twin sample sits at its original's cycle (the replayed ops keep the page time and the cycle count)
      const st = await call(S, 'snap'); if (st !== op[2] || st !== fr[2].step) { T.drops.twin_step = (T.drops.twin_step || 0) + 1; continue; }
      out.push({ idx: op[1], step: st, srcStep: op[2], n: [fr[1].step - fr[0].step, fr[2].step - fr[1].step], frames: fr, chase: b.chase, weight: op[4] ?? 1, injection: null });
    }
  }
}
// a clip is usable when its frames (and chase.png) passed the blank guard and its steps increase (an ended run stops the counter)
function clean(fr, T) {
  if (fr.some((f) => !f)) return false;
  if (fr.some((f, i) => i && i < 3 && f.step <= fr[i - 1].step)) { T.drops.done_in_clip = (T.drops.done_in_clip || 0) + 1; return false; }
  return true;
}
async function clip(S, D, ep, out, ops, T, weight = 1) {
  const n01 = ep.rng.int(19, 21), n12 = ep.rng.int(19, 21), fr = [], idx = out.length; ops.push(['clip', idx]);
  for (const k of [1, n01, n12]) { for (let i = 0; i < k; i++) await cyc(S, D, ops); fr.push(await grab(S, D, 'centreline', T)); ops.push(['grab', idx]); }
  await call(S, 'centreline', false); ops.push(['centreline', false]); await render(S, D, ops);
  const chase = await grab(S, D, 'chase', T); ops.push(['chase', idx]); await call(S, 'centreline', true); ops.push(['centreline', true]);
  if (!clean(fr.concat([chase]), T)) return;
  const inj = (await chk(S)).injected;
  if (straddles(inj, fr)) { T.drops.straddle = (T.drops.straddle || 0) + 1; return; }
  const st = await call(S, 'snap'), n = [fr[1].step - fr[0].step, fr[2].step - fr[1].step];
  if (st !== fr[2].step) { T.drops.snap_step = (T.drops.snap_step || 0) + 1; return; }
  ops.push(['snap', idx, st, n, weight]);
  out.push({ idx, step: st, n, frames: fr, chase, weight, injection: inj && inj.step < fr[0].step ? inj : null });
}
async function sampleRun(S, D, ep, out, ops, c0, T) {
  const tInj = c0.injAt === null || c0.injAt === undefined ? null : c0.injAt * CYCLE_S, endT = c0.endT ?? Infinity;
  let pending = tInj !== null, k = 0, needPre = false;
  const skip = async (sec) => { await call(S, 'skip', sec); ops.push(['skip', sec]); needPre = true; };
  const preroll = async () => { if (needPre) for (let i = 0; i < PRE; i++) await cyc(S, D, ops); needPre = false; };
  for (;;) {
    let c = await chk(S); if (c.done) break;
    // one slot stays free for a pending injection; the scheduled times left before the clean run's end share the others
    const slots = freeSlots(MAX_SAMPLES, out.length, pending, ep.preKick ?? MAX_SAMPLES), tS = slots > 0 ? ep.schedule[k] : undefined;
    // a clip spans 1 + 38-42 cycles (4.1 s) after its start; one that would reach the kick yields to the injection clip
    if (pending && (tS === undefined || Math.max(tS, c.t) + 4.5 >= tInj)) {
      pending = false;
      if (tInj - 6.4 > c.t + 1) await skip(tInj - 6.4 - c.t);
      await preroll(); c = await chk(S);
      for (let i = 0; i < 200 && !c.injected && !c.done; i++) { await cyc(S, D, ops); c = await chk(S); }
      if (c.done) break;
      if (!c.injected) throw new Discard(`the armed D kick did not land at its oracle cycle ${c0.injAt} (now ${c.cycle})`);
      await clip(S, D, ep, out, ops, T); T.injClips++; continue;
    }
    if (tS === undefined) break;
    k++;
    // a clip starts at its scheduled time: a skip to tS - 6.3 then the 63-cycle pre-roll, or (tS within 7.3 s) cycles up to tS;
    // a time that passed during the previous clip is dropped, so clips never run back to back
    if (tS < c.t) continue;
    const m = ep.schedule.slice(k - 1).filter((t) => t < endT - 5).length;
    if (tS - 6.3 > c.t + 1) { await skip(tS - 6.3 - c.t); c = await chk(S); if (c.done) break; }
    const p = keepProb(dockingW(c), slots, m);
    if (ep.rng.float() >= p) continue;
    await preroll(); c = await chk(S);
    for (let i = 0; i < 200 && c.t < tS - 1e-6 && !c.done; i++) { await cyc(S, D, ops); c = await chk(S); }
    if (c.done) break;
    await clip(S, D, ep, out, ops, T, 1 / p);
  }
}
export async function runDocking(D, ep) {
  const S = await openPage(D.browser, { utcMs: ep.utcMs, family: 'D', mode: D.mode, routes: D.routes(ep) }), out = [], ops = [], t0 = Date.now();
  const T = { bootMs: 0, grabs: 0, grabMs: 0, pngBytes: 0, labels: 0, labelMs: 0, injClips: 0, drops: {} }, stats = () => ({ ...T, frames: S.nFrames, frameMs: S.frameMs });
  const drop = (why, s) => { T.drops[why] = (T.drops[why] || 0) + 1; D.log({ drop: `D sample ${s.step}`, episode: ep.episode, why }); };
  try {
    T.boot = await boot(S, ep.url, BOOT.real, D.mode, D.stop); T.bootMs = Date.now() - t0;
    const armed = !ep.twin && ep.inject && ep.inject.at === 'runtime' ? ep.inject : null, c0 = await loadProbe(S, 'D', { ...ep.params, inject: armed });
    if (ep.twin && ep.inject && ep.inject.kind === 'failed_p6') await call(S, 'ownSim', { failed: [] });
    await call(S, 'centreline', true);
    if (ep.ops) await replay(S, D, ep, out, T); else await sampleRun(S, D, ep, out, ops, c0, T);
    const live = await call(S, 'finish'), kept = [];
    for (const s of out) {
      // §3.3: every entry leaves >= 1 s between f2 and the terminal event; samples that do not are dropped
      if (live.t - s.step * CYCLE_S < 1) { drop('under 1 s before the end of the run', s); continue; }
      const tl = performance.now(), b = await call(S, 'branches', { step: s.step }); T.labels++; T.labelMs += performance.now() - tl;
      if (b.discard) { drop(b.discard, s); continue; }
      if (isDroppedDockingOutcome(b.outcome)) { drop(`CONTINUE fail:${b.outcome.failKind}`, s); continue; }
      s.label = b; kept.push(s);
    }
    return { samples: kept, ops, wallMs: Date.now() - t0, stats: { ...stats(), live: `${live.result}${live.reason ? ': ' + live.reason : ''}`, inj_at: c0.injAt ?? null } };
  } finally { await S.ctx.close(); }
}
