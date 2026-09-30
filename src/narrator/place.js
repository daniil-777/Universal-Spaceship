// place.js — where the pill and the card go (pure; Node-tested). Both hang from the top centre of the viewport. The page's
// own chrome (the header wordmark and tools, open panels, the landing and docking bars) are obstacles: the box keeps the
// centre and takes the widest symmetric band free of every obstacle that shares its rows, minus a gutter. When that band
// is narrower than minW, the box drops below the obstacles that squeezed it and tries again; when nothing fits, it
// overlays at the top at full width. obstacles: [{l, t, r, b}] in CSS px. Returns {x (centre), y (top), w}.
export function placeBox({ vw, vh, obstacles = [], h, minW = 420, maxW = 560, gutter = 16, top = 17, gap = 12 }) {
  const x = vw / 2, full = Math.min(maxW, vw - 2 * gutter);
  let y = top;
  for (let i = 0; i < 8 && y + h <= vh - gutter; i++) {
    const rows = obstacles.filter((o) => o.t < y + h && o.b > y && o.r > o.l);
    const room = rows.map((o) => ({ o, half: o.r <= x ? x - o.r : o.l >= x ? o.l - x : 0 }));
    const half = Math.min(Infinity, ...room.map((q) => q.half)), w = Math.min(full, 2 * (half - gutter));
    if (w >= minW || (!rows.length && full >= 0)) return { x, y, w: Math.max(0, Math.round(w)) };
    const squeeze = room.filter((q) => 2 * (q.half - gutter) < minW);
    y = Math.max(...squeeze.map((q) => q.o.b)) + gap;
  }
  return { x, y: top, w: full };
}
