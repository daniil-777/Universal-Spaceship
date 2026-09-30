// vlm/capture/episode_corridor.mjs — one S or A episode = one fresh page (spec §3.4, §7.1-§7.4): boot, the S warm-up or the A
// gate, candidates every 1.5 s of sim time accepted with p = 1/3 or 1 on the CPA/clearance trigger, clips at a jittered
// 2-4 step spacing captured speculatively (f0, f1 are dropped when f2 is rejected), labels at f2; the episode ends at a
// crash, 60 s or 12 samples. drive.mjs keeps at most 2 severity-4 samples per episode (weight x candidates / kept).
import { openPage, boot, BOOT, loadProbe, call, frame, waitIdle, Discard } from './session.mjs';
import { createRing } from './probe/ring.js';
export class Crash extends Error {}
const chk = (S) => S.page.evaluate(() => window.__apv.check()), sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const pngBytes = (d) => Math.floor(((d.length - d.indexOf(',') - 1) * 3) / 4);
export async function stepTo(S, target, mode, stop) {
  for (let k = 0; k < 40 * (target + 10); k++) {
    let c = null; await waitIdle(S, async () => (c = await chk(S)).ok, { stop });
    if (c.crashed) throw new Crash('crash'); if (c.steps >= target) return c;
    await frame(S, mode);
  }
  throw new Discard(`step ${target} not reached`);
}
// T (optional) collects the throughput numbers of §7.6: host ms per capture call, the in-page render + encode and
// encode-only ms (Event.timeStamp clock), PNG bytes, and the frames spent waiting for the no-op
// ready (A): the full per-frame readiness check (strips, Meshy) before every frame; S passes null (always ready in space)
export async function captureExact(S, mode, stop, T = null, ready = null) {
  for (let tries = 0; tries < 3; tries++) {
    const n0 = await call(S, 'requestNoop');
    for (let k = 0; k < 12; k++) { await waitIdle(S, ready, { stop }); await frame(S, mode); if ((await chk(S)).noopDone > n0) break; }
    const t0 = performance.now(), f = await call(S, 'capture');
    if (f.why) throw new Discard(f.why);
    if (f.exact) { if (T) { T.caps++; T.capHostMs += performance.now() - t0; T.capPageMs += f.ms.capture; if (f.ms.encode !== null) { T.encMs += f.ms.encode; T.encN++; } T.pngBytes += pngBytes(f.png); } return f; }
  }
  throw new Discard('the drawn ship never matched env.ship.p');
}
// The A warm-up gate (§7.2). Real-time work first, with the clock paused (as in boot): the Earth textures, the atmo policy's
// fetch, parse and decode, and on mega and china the long grid's import and applyWorld (env.hf.lap); a frame pumped during that
// work would land it on a run-dependent fake tick (the belt pilot flies until the atmo policy arrives, app.js:131). Then frames,
// each after the full readiness wait (strips, Meshy), until atmo, route, atmosphere > 0.99, not space, the long grid and (A-PPO)
// the atmo policy hold; then 94 more frames, so the atmosphere is >= 0.9965 at every sample.
export async function aWarmup(S, D, ep) {
  const long = ['mega', 'china'].includes(ep.params.route), ready = async () => (await chk(S)).ok, t0 = Date.now();
  let c = await chk(S);
  while (!(c.textures && (c.policyAtmo || c.policyError) && (!long || c.lap))) {
    if (Date.now() - t0 > 120000) throw new Discard(`A warm-up: real-time loads not done in 120 s (textures ${c.textures}, atmo policy ${c.policyAtmo || c.policyError}, lap ${c.lap})`);
    await waitIdle(S, null, { stop: D.stop, timeoutMs: 120000 }); await sleep(10); c = await chk(S);
  }
  if (ep.policyId === 'atmo_ppo' && !c.policyAtmo) throw new Discard(`A-PPO without the atmo policy: ${c.policyError}`);
  for (let n = 0; ; n++) {
    await waitIdle(S, ready, { stop: D.stop }); c = await chk(S);
    if (c.atmo && c.route === ep.params.route && c.atmosphere > 0.99 && c.world !== 'space' && (!long || c.lap) && (ep.policyId !== 'atmo_ppo' || (c.policyAtmo && !c.policyError)) && c.ok) break;
    if (n >= 4000) throw new Discard(`A warm-up gate not passed in 4000 frames (atmosphere ${c.atmosphere}, world ${c.world}, route ${c.route})`);
    await frame(S, D.mode);
  }
  for (let i = 0; i < 94; i++) { await waitIdle(S, ready, { stop: D.stop }); await frame(S, D.mode); }
  return chk(S);
}
export const MAX_RESETS = 10;
// One segment: candidates every 23 steps from the segment's start; returns 'crash' or 'end' (60 s, 12 samples, the injected
// sample). The original applies the collision course at f0 - 1 of candidate injCi; a twin steps to the same f0 - 1 and does not.
async function segment(S, D, ep, { c, seg, injCi, seen, out, ring, T, ready, ledger0 }) {
  const gateStep = c.steps, plan = Array.from({ length: 40 }, (_, k) => ({ cand: gateStep + 23 * (k + 1), n: [ep.rng.int(2, 4), ep.rng.int(2, 4)] }));
  try {
    for (const { cand, n: [n01, n12] } of plan) {
      if (cand > gateStep + 900 || out.length >= 12) return 'end';
      const ci = ++seen.ci, f0 = cand - n01 - n12, fr = [], isInj = ci === injCi;
      if (f0 <= (await chk(S)).steps) continue;
      ring.clear(); let applied = null;
      if (isInj) { await stepTo(S, f0 - 1, D.mode, D.stop); if (!ep.twin) applied = await call(S, 'inject', { ...ep.inject, leadSteps: n01 + n12 + 1 }); ring.clear(); }
      for (const target of [f0, f0 + n01, cand]) { await stepTo(S, target, D.mode, D.stop); const i0 = D.ledger.length, f = await captureExact(S, D.mode, D.stop, T, ready); f.ledger = [i0, D.ledger.length]; fr.push(f); ring.push(f); }
      if (ring.size !== 3) continue;
      if (fr[2].step !== cand) throw new Discard(`f2 captured at step ${fr[2].step}, not the candidate ${cand}`);
      if (ep.force === 'cloud' && !((await call(S, 'camInCloud')) > 0)) continue;
      const trig = await call(S, 'trigger', { cNear: D.cNear }), forced = isInj && (ep.twin || !!applied);
      if (!trig && !forced && ep.rng.float() >= 1 / 3) continue;
      const tl = performance.now(), label = await call(S, 'label', { cNear: D.cNear });
      T.labels++; T.labelHostMs += performance.now() - tl; T.rolloutMs += label.ms.rollout; T.factsMs += label.ms.facts;
      out.push({ step: cand, n: [n01, n12], ci, seg, weight: trig || forced ? 1 : 3, label, frames: fr, ledger0, gate_step: gateStep, atmosphere: (await chk(S)).atmosphere, injection: forced && !ep.twin ? applied : null });
      if (forced) return 'end';
    }
    return 'end';
  } catch (e) { if (e instanceof Crash) return 'crash'; throw e; }
}
export async function runCorridorEpisode(D, ep) {
  const ledger0 = D.ledger.length, S = await openPage(D.browser, { utcMs: ep.utcMs, family: ep.family, mode: D.mode, routes: D.routes(ep) }), out = [], t0 = Date.now(), ring = createRing(3);
  const T = { caps: 0, capHostMs: 0, capPageMs: 0, encMs: 0, encN: 0, pngBytes: 0, labels: 0, labelHostMs: 0, rolloutMs: 0, factsMs: 0, bootMs: 0, warmMs: 0 };
  const stats = () => ({ ...T, frames: S.nFrames, frameMs: S.frameMs }), ready = ep.family === 'A' ? async () => (await chk(S)).ok : null;
  try {
    T.boot = await boot(S, ep.url, BOOT.app, D.mode, D.stop); T.bootMs = Date.now() - t0;
    let c = await loadProbe(S, ep.family, ep.params);
    if (c.bad) throw new Discard(c.bad);
    if (c.renderScale !== 1 || c.pixelRatio !== 1) throw new Discard(`renderScale ${c.renderScale} pixelRatio ${c.pixelRatio}`);
    // A: a crash during the descent (before the gate) ends the episode at its first step, with no record (crash statistics)
    if (ep.warmup) { c = await ep.warmup(S, D); T.crashedBeforeGate = !!c.crashed; }
    else {
      // S warm-up (§7.2): __ap.texturesReady, then 94 frames. The Earth textures resolve from image loads, which need no fake
      // ticks, so they are awaited in real time with the clock paused: the first sampled step never depends on load timing.
      for (const t = Date.now(); !c.textures && Date.now() - t < 60000; c = await chk(S)) { await waitIdle(S, null, { stop: D.stop }); await sleep(10); }
      if (!c.textures) throw new Discard('S warm-up: the Earth textures did not load in 60 s of real time');
      for (let i = 0; i < 94; i++) { await waitIdle(S, null, { stop: D.stop }); await frame(S, D.mode); } c = await chk(S);
    }
    T.warmMs = Date.now() - t0 - T.bootMs; T.resets = 0;
    // A samples across the page's own crash resets (ruling T10-a): each reset starts a new segment (a new episode index, see
    // families.mjs), the ring is cleared, and the warm-up is paid once per page; S ends at its first crash. Candidate numbers
    // (ci) run on across segments, so the one injection lands at its candidate in whichever segment reaches it.
    const across = ep.family === 'A', injCi = ep.inject ? ep.inject.candidate : null, seen = { ci: 0 };
    let crashed = false;
    for (let seg = 0; ; seg++) {
      const end = await segment(S, D, ep, { c, seg, injCi, seen, out, ring, T, ready, ledger0 });
      if (end !== 'crash') break;
      crashed = true;
      if (!across || out.length >= 12 || T.resets >= MAX_RESETS) break;
      // the page resets the env 1.6 s after a crash (app.js:158; for a crash in the descent it has already happened); wait for
      // that reset, then clear the sticky crash flag
      const reset = (x) => x.resetsAtCrash !== null && x.resets > x.resetsAtCrash && x.crashTimer <= 0;
      for (let k = 0; k < 400; k++) { c = await chk(S); if (reset(c)) break; await waitIdle(S, ready, { stop: D.stop }); await frame(S, D.mode); }
      if (!reset(c)) break;
      await call(S, 'clearCrash'); c = await chk(S); T.resets++;
    }
    // a twin (§8) runs this same loop, so its page-time history (frames, no-ops, captures, label calls) is the original's up to
    // the injected candidate; it keeps only that sample (families.mjs asserts equal steps and clocks)
    const samples = ep.twin ? out.filter((x) => x.ci === injCi) : out;
    return { samples, crashed, resets: T.resets, logs: S.logs, wallMs: Date.now() - t0, stats: stats() };
  } finally { await S.ctx.close(); }
}
