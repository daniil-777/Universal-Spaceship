// Weather for the landing scenario (SI, runway frame: x along the runway heading, z right, y up): a METAR-style mean wind
// — the reported 10-m wind on a log profile (z₀ 5 cm, up to 600 m) veering 20° through the boundary layer — slow gusts
// (the wind speed swinging between lulls and the reported gust), and MIL-F-8785C Dryden turbulence: the corridor's filters
// (src/turbulence.js) rescaled from its 19-m units to metres, σ_w = 0.1·W20 below 1000 ft (W20 = 15/30/45 kt for light/
// moderate/severe), fading to half by 3000 ft. The turbulence is frozen in the air the ship flies through: step() once
// per physics step with its airspeed, level track and height; at() is then the wind at any point.
import { Dryden } from '../turbulence.js';
import { mulberry32, randn, clamp } from '../mathx.js';

const KT = 0.514444, U = 19, Z0 = 0.05, LN10 = Math.log(10 / Z0), PMAX = Math.log(600 / Z0) / LN10, DEG = Math.PI / 180;
export const TURB_W20 = Object.freeze({ none: 0, light: 15, moderate: 30, severe: 45 });
export function parseMetarWind(s) {                       // 'dddssGggKT' / 'VRBss' / '00000'
  const m = /^(\d{3}|VRB)(\d{2,3})(?:G(\d{2,3}))?(?:KT)?$/.exec(String(s || '').trim().toUpperCase());
  return m ? { dir: m[1] === 'VRB' ? null : +m[1], kt: +m[2], gust: m[3] ? +m[3] : 0 } : null;
}

export function createWind({ dir = 260, kt = 10, gust = 0, turb = 'light', seed = 1, rwyHdg = 260, span = 30.6 } = {}) {   // span: gusts shorter than the wings average out
  const rng = mulberry32((seed ^ 0x9e3779b9) >>> 0), d = dir == null ? rwyHdg + (rng() * 2 - 1) * 60 : dir;   // variable: within 60° of the runway
  const to = (d + 180 - rwyHdg) * DEG;                   // where the air goes, from the runway heading (clockwise = right, +z)
  const W20 = TURB_W20[turb] ?? 0, dry = new Dryden(mulberry32((seed ^ 0x51ed270b) >>> 0)), T = { sigma: 0, hA: 1, share: 0, Lc: 0, minH: 3 / U }, g = new Float64Array(3), gf = new Float64Array(3);
  dry.reset();
  const gsSig = Math.max(0, gust - kt) * KT / 2.5, gsTau = 6; let gs = 0;   // the gust swing: an OU process in the wind speed
  const prof = (h) => clamp(Math.log(Math.max(h, 0.2) / Z0) / LN10, 0, PMAX), veer = (h) => clamp((h - 10) / 590, 0, 1) * 20 * DEG;
  function mean(h, out) { const sp = Math.max(0, kt * KT + gs) * prof(h), a = to + veer(h); out[0] = sp * Math.cos(a); out[1] = 0; out[2] = sp * Math.sin(a); return out; }
  return {
    dir: d, kt, gust, turb, W20, mean,
    components(h) { const a = to + veer(h), sp = kt * prof(h); return { head: -sp * Math.cos(a), cross: -sp * Math.sin(a) }; },   // kt; head > 0 into the nose, cross > 0 from the right
    step(h, V, hx, hz, hAGL) {
      gs += -gs * h / gsTau + gsSig * Math.sqrt(2 * h / gsTau) * randn(rng);
      const ft = hAGL / 0.3048, sw = 0.1 * W20 * KT * (ft <= 1000 ? 1 : Math.max(0.5, 1 - 0.5 * (ft - 1000) / 2000));
      T.sigma = sw / U; T.hA = Math.max(hAGL, 3) / U; dry.step(h, Math.max(V, 10) / U, T, hx, hz, g);   // MIL: L_w = h right down to the runway (3 m floor)
      const k = 1 - Math.exp(-h * 2 * Math.max(V, 10) / span); for (let i = 0; i < 3; i++) gf[i] += k * (g[i] * U - gf[i]);   // what the airframe feels: the gust averaged over its span
    },
    at(p, t, out) { mean(Math.max(p[1], 0), out); out[0] += gf[0]; out[1] += gf[1]; out[2] += gf[2]; return out; },
  };
}
