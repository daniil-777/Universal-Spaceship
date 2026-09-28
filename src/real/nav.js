// Relative navigation (spec section 6): an alpha-beta filter per axis on lidar position (sigma = 2 cm + 0.1 % of the
// port range) and range-rate (2 mm/s), predicted with the exact CW step and the measured thrust acceleration.
// Attitude and body rates come from the star tracker and gyros as truth in S1. mode 'truth' passes the state through.
import { CYCLE } from './consts.js';
import { phi, gamma, stepWith } from './cw.js';
import { randn } from '../mathx.js';

export const NAV = Object.freeze({ sigma0: 0.02, sigmaRel: 0.001, sigmaV: 0.002, alpha: 0.1, beta: 0.02 });
const P = phi(CYCLE), G = gamma(CYCLE);

export function createNav({ mode = 'noisy', rng = Math.random, scale = 1 } = {}) {
  const xh = new Float64Array(6); let init = false;
  return {
    xh, mode, valid: true,
    // x: truth state; rho: port range (m); aThrust: LVLH thrust acceleration over the last cycle (m/s^2)
    update(x, rho, aThrust = null) {
      if (mode === 'truth' || !init) { xh.set(x); init = true; return xh; }
      stepWith(P, G, xh, aThrust, 0, xh);
      const sp = scale * (NAV.sigma0 + NAV.sigmaRel * rho), sv = scale * NAV.sigmaV;
      for (let i = 0; i < 3; i++) {
        xh[i] += NAV.alpha * (x[i] + sp * randn(rng) - xh[i]);
        xh[3 + i] += NAV.beta * (x[3 + i] + sv * randn(rng) - xh[3 + i]);
      }
      return xh;
    },
  };
}
