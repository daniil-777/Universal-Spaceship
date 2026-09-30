// vlm/gen/imagefacts.js — the §4.2 image facts from narrator_frame (Node, sharp), all obs: visual: mean luminance and
// contrast (in [0, 1]), the 3 dominant of the 12 PALETTE_NAMES with their fractions, near-white/near-black fractions, Sobel
// edge density and its bin. The names, bins and edges are schema.js's (ruling T7-b): brightness is binned on the mean
// luminance in [0, 1] at BRIGHTNESS_EDGES, the colour names are PALETTE_NAMES.
// Each name has one or more RGB anchors and a pixel takes the name of its nearest anchor. The brief's single-anchor table
// had "navy" where PALETTE_NAMES has "pink"; with navy gone a dark sea (about 12/32/70) would be nearest to black, so
// blue also anchors a navy and a sky blue, grey a dark grey, green a dark green and brown a tan (T7 dropped "tan").
import sharp from 'sharp';
import { fact, PALETTE_NAMES, BRIGHTNESS_BINS, BRIGHTNESS_EDGES, EDGE_BINS } from './schema.js';

export const PALETTE = Object.freeze({ black: [[15, 15, 15]], grey: [[128, 128, 128], [80, 80, 80]], white: [[240, 240, 240]], red: [[200, 40, 40]], orange: [[230, 130, 30]],
  yellow: [[230, 210, 60]], green: [[60, 150, 60], [30, 80, 40]], teal: [[40, 150, 150]], blue: [[40, 80, 200], [20, 30, 80], [120, 160, 220]], purple: [[120, 60, 150]],
  pink: [[235, 150, 190]], brown: [[120, 80, 40], [180, 140, 100]] });
const ANCHORS = PALETTE_NAMES.flatMap((name, k) => PALETTE[name].map((rgb) => [k, ...rgb]));
export const EDGE_EDGES = Object.freeze([0.05, 0.15]);
export const SOBEL_THRESHOLD = 160;
export const brightnessBin = (lum01) => BRIGHTNESS_BINS[BRIGHTNESS_EDGES.filter((e) => lum01 >= e).length];
export const edgeBin = (density) => EDGE_BINS[EDGE_EDGES.filter((e) => density >= e).length];
const nearest = (r, g, b) => {
  let best = 0, bd = Infinity;
  for (const [k, R, G, B] of ANCHORS) { const d = (R - r) ** 2 + (G - g) ** 2 + (B - b) ** 2; if (d < bd) { bd = d; best = k; } }
  return best;
};
// the facts of raw RGB(A) pixels (ch channels per pixel); imageFacts() feeds it the 224x126 working copy of a frame
export function imageFactsOf(data, W, H, ch = 3) {
  const n = W * H, Y = new Float64Array(n), hist = new Array(PALETTE_NAMES.length).fill(0);
  let s = 0, s2 = 0, white = 0, black = 0;
  for (let i = 0; i < n; i++) {
    const r = data[i * ch], g = data[i * ch + 1], b = data[i * ch + 2], y = 0.299 * r + 0.587 * g + 0.114 * b;
    Y[i] = y; s += y; s2 += y * y; if (r > 230 && g > 230 && b > 230) white++; if (r < 25 && g < 25 && b < 25) black++; hist[nearest(r, g, b)]++;
  }
  let edges = 0;
  for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
    const k = y * W + x, gx = Y[k - W + 1] + 2 * Y[k + 1] + Y[k + W + 1] - Y[k - W - 1] - 2 * Y[k - 1] - Y[k + W - 1], gy = Y[k + W - 1] + 2 * Y[k + W] + Y[k + W + 1] - Y[k - W - 1] - 2 * Y[k - W] - Y[k - W + 1];
    if (Math.hypot(gx, gy) > SOBEL_THRESHOLD) edges++;
  }
  const mean = s / n / 255, sd = Math.sqrt(Math.max(0, s2 / n - (s / n) ** 2)) / 255, inner = Math.max(1, (W - 2) * (H - 2)), ed = W > 2 && H > 2 ? edges / inner : 0;
  const top = hist.map((c, i) => [PALETTE_NAMES[i], c / n, i]).sort((a, b) => b[1] - a[1] || a[2] - b[2]).slice(0, 3), F = {}, put = (id, v, unit = null) => { F[id] = fact(v, unit, 'visual'); };
  put('image.mean_lum', +mean.toFixed(4)); put('image.contrast', +sd.toFixed(4)); put('image.brightness_bin', brightnessBin(mean));
  top.forEach(([name, f], i) => { put(`image.palette_${i}`, name); put(`image.palette_${i}_frac`, +f.toFixed(3)); });
  put('image.white_frac', +(white / n).toFixed(4)); put('image.black_frac', +(black / n).toFixed(4)); put('image.edge_density', +ed.toFixed(4)); put('image.edge_bin', edgeBin(ed));
  return F;
}
export async function imageFacts(file) {
  const { data, info } = await sharp(file).removeAlpha().resize(224, 126, { kernel: 'cubic' }).raw().toBuffer({ resolveWithObject: true });
  return imageFactsOf(data, info.width, info.height, info.channels);
}
