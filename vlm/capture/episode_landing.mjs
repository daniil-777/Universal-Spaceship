// vlm/capture/episode_landing.mjs — one L run = one fresh page (spec §3.3, §3.4 L, §7.3-§7.4 L). Scheduled times: a skip with
// __landing.advance, the slot selection (probe/frame.js keepProb: the run's 12 slots spread over the times left before the clean
// run's end, w = 0.35 above 1000 ft), then >= 1 s (63 frames) of rendered frames before f0 whenever a skip happened since the
// last clip; f0-f2 at 22-38 steps via runFor(16) until the step counter reaches the target; f2 snapshots. The one runtime kick is armed in the page (probe/landing.js) and lands at its
// trigger step, which the probe's oracle run predicts: a scheduled clip that would reach that time yields to the injection
// clip, pre-rolled to the kick and taken right after it, so no skip or clip crosses it (a clip with f0 <= kick <= f2 would be
// dropped: the ring is cleared at the injection). After the last sample the run is finished and branches are replayed.
// Every page-time operation is logged in `ops` (advance, frame, clip/grab/snap markers). A twin (§8) replays `ops` verbatim
// with no kick armed and grabs only the kept samples, so its render history (chase lerp, clouds, sky clock) is the original's
// and frames before the fault shows are byte-identical (build.mjs flags them pixel_identical_pair).
import sharp from 'sharp';
import { openPage, boot, BOOT, loadProbe, call, frame, waitIdle, Discard } from './session.mjs';
import { frameStats, saneFrame, straddles, keepProb } from './probe/frame.js';
export const H = 1 / 120, PRE = 63, MAX_SAMPLES = 12;
const chk = (S) => S.page.evaluate(() => window.__apv.check());
async function step1(S, D, ops) { await waitIdle(S, null, { stop: D.stop }); await frame(S, D.mode); if (ops) ops.push(['frame']); }
async function framesUntil(S, D, ops, pred, max = 20000) { for (let i = 0; i < max; i++) { const c = await chk(S); if (pred(c)) return c; await step1(S, D, ops); } throw new Discard('landing target not reached'); }
// a frame failing the blank-buffer guard (alpha 0 or luminance variance <= 4: a blank buffer, or a uniform frame inside a
// cloud deck) is returned as null and drops its clip only (counted in stats.drops.blank)
async function grab(S, D, T) {
  const t = performance.now(), f = await call(S, 'capture'), px = await sharp(Buffer.from(f.png.split(',')[1], 'base64')).ensureAlpha().raw().toBuffer();
  if (!saneFrame(frameStats(px))) { if (T) T.drops.blank = (T.drops.blank || 0) + 1; return null; }
  f.ledger = [D.ledger.length, D.ledger.length];
  if (T) { T.grabs++; T.grabMs += performance.now() - t; T.pngBytes += Math.floor(((f.png.length - f.png.indexOf(',') - 1) * 3) / 4); }
  return f;
}
async function replay(S, D, ep, out, T) {
  const buf = {};
  for (const op of ep.ops) {
    if (op[0] === 'adv') await S.page.evaluate((dt) => window.__landing.advance(dt), op[1]);
    else if (op[0] === 'frame') await step1(S, D, null);
    else if (!ep.keep.has(op[1])) continue;
    else if (op[0] === 'clip') buf[op[1]] = [];
    else if (op[0] === 'grab') buf[op[1]].push(await grab(S, D, T));
    else if (op[0] === 'snap') { const fr = buf[op[1]]; if (!clean(fr, T)) continue; out.push({ idx: op[1], step: await call(S, 'snap'), srcStep: op[2], n: [fr[1].step - fr[0].step, fr[2].step - fr[1].step], frames: fr, injection: null }); }
  }
}
// a clip is usable when its 3 frames passed the blank guard and their steps increase (a run that ended stops the counter)
function clean(fr, T) {
  if (fr.some((f) => !f)) return false;
  if (fr.some((f, i) => i && f.step <= fr[i - 1].step)) { T.drops.done_in_clip = (T.drops.done_in_clip || 0) + 1; return false; }
  return true;
}
// one clip from the state c: f0 8 steps on, then n01 and n12 steps (22-38 each once a frame's 1-2 steps round up)
async function clip(S, D, ep, out, ops, c, T) {
  const n01 = ep.rng.int(23, 37), n12 = ep.rng.int(23, 37), fr = [], idx = out.length; ops.push(['clip', idx]);
  for (const k of [0, n01, n01 + n12]) {
    const target = c.step + 8 + k, x = await framesUntil(S, D, ops, (y) => y.step >= target || y.done);
    if (x.done) { T.drops.done_in_clip = (T.drops.done_in_clip || 0) + 1; return; }
    fr.push(await grab(S, D, T)); ops.push(['grab', idx]);
  }
  if (!clean(fr, T)) return;
  const inj = (await chk(S)).injected;
  if (straddles(inj, fr)) { T.drops.straddle = (T.drops.straddle || 0) + 1; return; }
  const st = await call(S, 'snap'), n = [fr[1].step - fr[0].step, fr[2].step - fr[1].step]; ops.push(['snap', idx, st, n]);
  out.push({ idx, step: st, n, frames: fr, injection: inj && inj.step < fr[0].step ? inj : null });
}
async function sampleRun(S, D, ep, out, ops, c0, T) {
  const tInj = c0.injAt === null || c0.injAt === undefined ? null : c0.injAt * H, endT = c0.endT ?? 900;
  let pending = tInj !== null, k = 0, needPre = false;
  const adv = async (dt) => { await S.page.evaluate((x) => window.__landing.advance(x), dt); ops.push(['adv', dt]); needPre = true; };
  const preroll = async () => { if (needPre) for (let i = 0; i < PRE; i++) await step1(S, D, ops); needPre = false; };
  for (;;) {
    let c = await chk(S); if (c.done) break;
    // one slot stays free for a pending injection; the scheduled times left before the clean run's end share the others
    const slots = MAX_SAMPLES - out.length - (pending ? 1 : 0), tS = slots > 0 ? ep.schedule[k] : undefined;
    if (pending && (tS === undefined || Math.max(tS, c.t) + 1 >= tInj)) {
      pending = false;
      if (tInj - 1.1 > c.t + 2) await adv(tInj - 1.1 - c.t);
      await preroll(); c = await framesUntil(S, D, ops, (x) => x.injected || x.done, 600);
      if (c.done) break;
      await clip(S, D, ep, out, ops, c, T); T.injClips++; continue;
    }
    if (tS === undefined) break;
    k++;
    // a clip starts at its scheduled time: a skip to tS - 1 then the 1-s pre-roll, or (tS within 3 s) frames up to tS; a time
    // that passed during the previous clip is dropped, so clips never run back to back
    if (tS < c.t) continue;
    const m = ep.schedule.slice(k - 1).filter((t) => t < endT - 1).length;
    if (tS - 1 > c.t + 2) { await adv(tS - 1 - c.t); c = await chk(S); if (c.done) break; }
    if (ep.rng.float() >= keepProb(c.hRAft > 1000 ? 0.35 : 1, slots, m)) continue;
    if (ep.force === 'below_base' && c.hRAft >= 280) continue;
    await preroll(); c = await framesUntil(S, D, ops, (x) => x.t >= tS || x.done, 400); if (c.done) break;
    await clip(S, D, ep, out, ops, c, T);
  }
}
export async function runLanding(D, ep) {
  const S = await openPage(D.browser, { utcMs: ep.utcMs, family: 'L', mode: D.mode, routes: D.routes(ep) }), out = [], ops = [], t0 = Date.now();
  const T = { bootMs: 0, grabs: 0, grabMs: 0, pngBytes: 0, labels: 0, labelMs: 0, injClips: 0, drops: {} }, stats = () => ({ ...T, frames: S.nFrames, frameMs: S.frameMs });
  const drop = (why) => { T.drops[why] = (T.drops[why] || 0) + 1; };
  try {
    T.boot = await boot(S, ep.url, BOOT.landing, D.mode, D.stop); T.bootMs = Date.now() - t0;
    const armed = !ep.twin && ep.inject && ep.inject.at === 'runtime' ? ep.inject : null, c0 = await loadProbe(S, 'L', { ...ep.params, inject: armed });
    if (ep.ops) await replay(S, D, ep, out, T); else await sampleRun(S, D, ep, out, ops, c0, T);
    const fin = await call(S, 'finish'); if (fin.result === 'timeout') throw new Discard('landing timeout: run dropped');
    const kept = [];
    for (const s of out) {
      // §3.3: every entry leaves >= 1 s between f2 and the terminal event; samples that do not are dropped
      if (fin.t - s.step * H < 1) { drop('under 1 s before the end of the run'); continue; }
      const tl = performance.now(), b = await call(S, 'branches', { step: s.step }); T.labels++; T.labelMs += performance.now() - tl;
      if (b.discard) { drop(b.discard); D.log({ drop: `L sample ${s.step}`, episode: ep.episode, why: b.discard }); continue; }
      s.label = b; kept.push(s);
    }
    return { samples: kept, ops, liveResult: fin.result, wallMs: Date.now() - t0, stats: { ...stats(), live: fin.result, inj_at: c0.injAt ?? null } };
  } finally { await S.ctx.close(); }
}
