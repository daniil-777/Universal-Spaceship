// vlm/capture/episode_corridor.mjs — one S (and, from Task 10, A) episode = one fresh page (spec §3.4, §7.1-§7.4): boot,
// warm-up, candidates every 1.5 s of sim time accepted with p = 1/3 or 1 on the CPA/clearance trigger, clips at a jittered
// 2-4 step spacing captured speculatively (f0, f1 are dropped when f2 is rejected), labels at f2; the episode ends at a
// crash, 60 s or 12 samples. drive.mjs keeps at most 2 severity-4 samples per episode (weight x candidates / kept).
import { openPage, boot, loadProbe, call, frame, waitIdle, Discard } from './session.mjs';
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
export async function captureExact(S, mode, stop, T = null) {
  for (let tries = 0; tries < 3; tries++) {
    const n0 = await call(S, 'requestNoop');
    for (let k = 0; k < 12; k++) { await waitIdle(S, null, { stop }); await frame(S, mode); if ((await chk(S)).noopDone > n0) break; }
    const t0 = performance.now(), f = await call(S, 'capture');
    if (f.exact) { if (T) { T.caps++; T.capHostMs += performance.now() - t0; T.capPageMs += f.ms.capture; if (f.ms.encode !== null) { T.encMs += f.ms.encode; T.encN++; } T.pngBytes += pngBytes(f.png); } return f; }
  }
  throw new Discard('the drawn ship never matched env.ship.p');
}
export async function runCorridorEpisode(D, ep) {
  const ledger0 = D.ledger.length, S = await openPage(D.browser, { utcMs: ep.utcMs, family: ep.family, mode: D.mode, routes: D.routes(ep) }), out = [], t0 = Date.now(), ring = createRing(3);
  const T = { caps: 0, capHostMs: 0, capPageMs: 0, encMs: 0, encN: 0, pngBytes: 0, labels: 0, labelHostMs: 0, rolloutMs: 0, factsMs: 0, bootMs: 0, warmMs: 0 };
  const stats = () => ({ ...T, frames: S.nFrames, frameMs: S.frameMs });
  try {
    await boot(S, ep.url, '!!(window.__ap && window.__ap.ready)', D.mode); T.bootMs = Date.now() - t0;
    let c = await loadProbe(S, ep.family, ep.params);
    if (c.renderScale !== 1 || c.pixelRatio !== 1) throw new Discard(`renderScale ${c.renderScale} pixelRatio ${c.pixelRatio}`);
    if (ep.warmup) c = await ep.warmup(S, D);
    else {
      // S warm-up (§7.2): __ap.texturesReady, then 94 frames. The Earth textures resolve from image loads, which need no fake
      // ticks, so they are awaited in real time with the clock paused: the first sampled step never depends on load timing.
      for (const t = Date.now(); !c.textures && Date.now() - t < 60000; c = await chk(S)) { await waitIdle(S, null, { stop: D.stop }); await sleep(10); }
      if (!c.textures) throw new Discard('S warm-up: the Earth textures did not load in 60 s of real time');
      for (let i = 0; i < 94; i++) { await waitIdle(S, null, { stop: D.stop }); await frame(S, D.mode); } c = await chk(S);
    }
    T.warmMs = Date.now() - t0 - T.bootMs;
    const gateStep = c.steps;
    const plan = ep.only ? ep.only.map((o) => ({ cand: o.step, n: o.n, ci: o.ci })) : Array.from({ length: 40 }, (_, k) => ({ cand: c.steps + 23 * (k + 1), n: [ep.rng.int(2, 4), ep.rng.int(2, 4)], ci: k + 1 }));
    for (const { cand, n: [n01, n12], ci } of plan) {
      if (cand > 900 || out.length >= 12) break;
      const f0 = cand - n01 - n12, fr = [];
      if (f0 <= (await chk(S)).steps) continue;
      ring.clear();
      if (ep.inject && !ep.twin && ep.inject.candidate === ci) { await stepTo(S, f0 - 1, D.mode, D.stop); ep.inject.applied = await call(S, 'inject', { ...ep.inject, leadSteps: n01 + n12 + 1 }); ring.clear(); }
      for (const target of [f0, f0 + n01, cand]) { await stepTo(S, target, D.mode, D.stop); const i0 = D.ledger.length, f = await captureExact(S, D.mode, D.stop, T); f.ledger = [i0, D.ledger.length]; fr.push(f); ring.push(f); }
      if (ring.size !== 3) continue;
      if (ep.force === 'cloud' && !((await call(S, 'camInCloud')) > 0)) continue;
      const trig = await call(S, 'trigger', { cNear: D.cNear }), forced = !!ep.only || !!(ep.inject && ep.inject.applied && ep.inject.candidate === ci);
      if (!trig && !forced && ep.rng.float() >= 1 / 3) continue;
      const tl = performance.now(), label = await call(S, 'label', { cNear: D.cNear });
      T.labels++; T.labelHostMs += performance.now() - tl; T.rolloutMs += label.ms.rollout; T.factsMs += label.ms.facts;
      out.push({ step: cand, n: [n01, n12], ci, weight: trig || forced ? 1 : 3, label, frames: fr, ledger0, gate_step: gateStep, atmosphere: (await chk(S)).atmosphere, injection: forced && !ep.only ? ep.inject.applied : null });
      if (forced && !ep.only) break;
    }
    return { samples: out, crashed: false, logs: S.logs, wallMs: Date.now() - t0, stats: stats() };
  } catch (e) { if (e instanceof Crash) return { samples: out, crashed: true, logs: S.logs, wallMs: Date.now() - t0, stats: stats() }; throw e; } finally { await S.ctx.close(); }
}
