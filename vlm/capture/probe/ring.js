// vlm/capture/probe/ring.js — the clip ring of spec §7.4: 3 frames at the family spacing; a frame whose step does not
// increase (the page resets the env 1.6 s after a crash) or that is marked reset (an injection) clears the ring first.
export function createRing(n = 3) {
  let frames = [];
  return { push(f) { const last = frames[frames.length - 1]; if (f.reset || (last && f.step <= last.step)) frames = []; frames.push(f); if (frames.length > n) frames.shift(); return frames.length === n ? frames.slice() : null; },
    clear() { frames = []; }, get size() { return frames.length; } };
}
