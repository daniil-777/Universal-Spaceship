// Turbulence for atmospheric flight (never in space): the Dryden model of MIL-F-8785C — gusts along the ship's track
// through the air (u), across it (v) and vertical (w), plus the rolling gust p (the vertical gust's spanwise gradient).
// u is a first-order shaping filter; v and w are Dryden's second-order ones, H(s) ∝ (1 + √3·Ls/V)/(1 + Ls/V)², built from
// two cascaded lags. The filter states have unit variance and the intensity σ(p) scales the output, so a cloud's
// roughness is felt the moment the ship enters it and gone when it leaves. Length scales and σ_u/σ_w follow the low-
// altitude model (1 unit = 19 m = 62.3 ft): L_w = h, L_u = L_v = h/(0.177 + 0.000823h)^1.2, σ_u = σ_v = σ_w/(0.177 +
// 0.000823h)^0.4 up to 1000 ft, 1750 ft and isotropic from 2000 ft; inside a cloud its own scale (0.4 R) takes over in
// proportion to the cloud's share of σ². σ, h and the cloud come from weather.turbAt().
import { randn } from './mathx.js';

export const SPAN = 3.4;                                  // the wing span the rolling gust acts across (units)
const SQ3 = Math.sqrt(3), K16 = 0.177 + 0.0513 * 16, LU16 = 16 / K16 ** 1.2, RU16 = K16 ** -0.4, A0 = Math.exp(-(1 / 60) * 14 / 20);
export function turbScales(hA, share, Lc, o, minH = 1.5) { // → o.Lw, o.Lu (= L_v), o.ru = σ_u/σ_w; minH: the lowest height the scales follow (units)
  const h = Math.max(minH, hA); let Lw, Lu, ru;
  if (h <= 16) { const k = 0.177 + 0.0513 * h; Lw = h; Lu = h / k ** 1.2; ru = Math.min(1.7, k ** -0.4); }
  else { const t = Math.min(1, (h - 16) / 16); Lw = 16 + 12 * t; Lu = LU16 + (28 - LU16) * t; ru = RU16 + (1 - RU16) * t; }
  const c = Math.max(1.5, Lc || 0); o.Lw = Lw + share * (c - Lw); o.Lu = Lu + share * (c - Lu); o.ru = ru + share * (1 - ru); return o;
}
// the second-order channel's output, normalised to unit variance for the step's pole a (x2 lags x1: cov a/(1+a), var (1+a²)/(1+a)²)
const shaped = (x1, x2, a) => (SQ3 * x1 + (1 - SQ3) * x2) / Math.sqrt(3 + (4 - 2 * SQ3) * (1 + a * a) / ((1 + a) * (1 + a)) + 2 * SQ3 * (1 - SQ3) * a / (1 + a));
const _s = { Lw: 0, Lu: 0, ru: 1 };

export class Dryden {
  constructor(rng) { this.rng = rng; this.x = new Float64Array(6); this.p = 0; }   // x: u, v1, v2, w1, w2, p — unit variance each
  reset() {                                               // a stationary draw: the gusts are already there when the episode starts
    const r = this.rng, x = this.x; x[0] = randn(r); x[1] = randn(r); x[2] = (A0 * x[1] + randn(r)) / (1 + A0); x[3] = randn(r); x[4] = (A0 * x[3] + randn(r)) / (1 + A0); x[5] = randn(r); this.p = 0;
  }
  step(h, V, T, hx, hz, out) {                            // T: { sigma, hA, share, Lc }; (hx, hz): the level heading of the ship's motion through the air
    const x = this.x, r = this.rng; turbScales(T.hA, T.share, T.Lc, _s, T.minH ?? 1.5);
    const au = Math.exp(-h * V / _s.Lu), aw = Math.exp(-h * V / _s.Lw), ap = Math.exp(-h * Math.PI * V / (4 * SPAN)), bu = Math.sqrt(1 - au * au), bw = Math.sqrt(1 - aw * aw);
    x[0] = au * x[0] + bu * randn(r);
    x[2] = au * x[2] + (1 - au) * x[1]; x[1] = au * x[1] + bu * randn(r);
    x[4] = aw * x[4] + (1 - aw) * x[3]; x[3] = aw * x[3] + bw * randn(r);
    x[5] = ap * x[5] + Math.sqrt(1 - ap * ap) * randn(r);
    const sw = T.sigma, su = _s.ru * sw, gu = su * x[0], gv = su * shaped(x[1], x[2], au);
    out[0] = gu * hx - gv * hz; out[1] = sw * shaped(x[3], x[4], aw); out[2] = gu * hz + gv * hx;
    this.p = sw * Math.sqrt(0.8 * Math.PI * Math.PI * Math.cbrt(Math.PI / (4 * SPAN)) / (8 * SPAN * _s.Lw ** (2 / 3))) * x[5];   // first order, corner πV/4b
    return out;
  }
}
