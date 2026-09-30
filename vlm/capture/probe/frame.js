// vlm/capture/probe/frame.js — blank-buffer guard for L/D captures (spec §3.2): blank frames hash identically, so the
// determinism gate would not catch them; alpha must be > 0 and the luminance variance above a floor.
export function frameStats(rgba) {
  let aMax = 0, s = 0, s2 = 0, n = 0;
  for (let i = 0; i < rgba.length; i += 4) { aMax = Math.max(aMax, rgba[i + 3]); const y = 0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2]; s += y; s2 += y * y; n++; }
  return { alphaMax: aMax, variance: n ? s2 / n - (s / n) ** 2 : 0 };
}
export const saneFrame = (st, floor = 4) => st.alphaMax > 0 && st.variance > floor;
// the ring rule of §3.3/§7.4 for L/D clips: a clip straddles the injection when it lands at or between f0's and f2's steps
// (f2's own step included: the f2 label would be pre-kick while the kick lands before the next step); such a clip is dropped
export const straddles = (inj, fr) => !!inj && inj.step >= fr[0].step && inj.step <= fr[fr.length - 1].step;
// L/D slot selection (§3.4 "spread over the phases, oversampling below 1000 ft (L) and within 30 m (D)"): a scheduled time is
// kept with p = min(1, w * slots / m), slots = the free sample slots, m = the scheduled times left before the run's end (from
// the probe's oracle run), w = 0.35 above 1000 ft (L) or 0.4 beyond 30 m (D), else 1; with w = 1 this is selection sampling,
// which keeps exactly the free slots, spread uniformly over the run
export const keepProb = (w, slots, m) => (slots <= 0 ? 0 : Math.min(1, (w * slots) / Math.max(1, m)));
// the phase weights w: L 0.35 above 1000 ft and on the ground roll (WOW, ruling M-6), else 1; D 0.4 beyond 30 m, else 1. A kept
// sample stores sampler_weight = 1 / keepProb (ruling I-3); an injection clip is forced (weight 1)
export const landingW = (c) => (c.hRAft > 1000 || c.wow ? 0.35 : 1);
export const dockingW = (c) => (c.rho > 30 ? 0.4 : 1);
// V1-3: the free sample slots. While a runtime kick is pending, v0 kept one slot back for it (max - 1 clean samples before the
// kick); v1 caps the clean samples before the kick at PRE_KICK (L 4, D 0), so an injected run spends its slots on the kick and
// after it (D's contact-range kicks leave room for one clip only). With no pending kick the whole run is open.
export const PRE_KICK = Object.freeze({ L: 4, D: 0 });
export const freeSlots = (max, taken, pending, preKick = max) => (pending ? Math.min(max - 1, preKick) : max) - taken;
