// vlm/gen/boxresize.js — the one Pilot Eye resize (spec §10.1, R13): exact area (box) average from any size to 160x96 RGB,
// separable, fractional pixel coverage, rounded once. The build's cache and the browser runtime both call this function.
function axis(n, m) { const s = n / m, w = []; for (let o = 0; o < m; o++) { const a = o * s, b = a + s, parts = []; for (let i = Math.floor(a); i < Math.min(n, Math.ceil(b)); i++) parts.push([i, (Math.min(b, i + 1) - Math.max(a, i)) / s]); w.push(parts); } return w; }
export function boxResize(src, sw, sh, dw = 160, dh = 96, ch = 4) {
  const wx = axis(sw, dw), wy = axis(sh, dh), tmp = new Float64Array(sh * dw * 3), out = new Uint8Array(dw * dh * 3);
  for (let y = 0; y < sh; y++) for (let x = 0; x < dw; x++) for (const [i, f] of wx[x]) { const s = (y * sw + i) * ch, t = (y * dw + x) * 3; tmp[t] += f * src[s]; tmp[t + 1] += f * src[s + 1]; tmp[t + 2] += f * src[s + 2]; }
  for (let y = 0; y < dh; y++) for (let x = 0; x < dw; x++) { let r = 0, g = 0, b = 0; for (const [j, f] of wy[y]) { const t = (j * dw + x) * 3; r += f * tmp[t]; g += f * tmp[t + 1]; b += f * tmp[t + 2]; } const o = (y * dw + x) * 3; out[o] = Math.round(r); out[o + 1] = Math.round(g); out[o + 2] = Math.round(b); }
  return out;
}
