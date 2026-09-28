// Real spacecraft S1: the CW kernel (spec sections 4 and 10). Axes: X along-track, Y up, Z = X x Y.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { circularOrbit } from '../src/orbit.js';
import { N, T_ORB, DAY } from '../src/real/consts.js';
import { phi, gamma, step, propagate, deriv, jacobi, stepWith } from '../src/real/cw.js';

const near = (a, b, tol, m) => assert.ok(Math.abs(a - b) <= tol, `${m}: ${a} vs ${b} (tol ${tol})`);
const X = (...v) => Float64Array.from(v);
function rk4(x, h, steps, a = null, ad = 0) {
  const k = [0, 1, 2, 3].map(() => new Float64Array(6)), y = new Float64Array(6);
  x = Float64Array.from(x);
  for (let s = 0; s < steps; s++) {
    deriv(x, a, ad, k[0]);
    for (let i = 0; i < 6; i++) y[i] = x[i] + (h / 2) * k[0][i]; deriv(y, a, ad, k[1]);
    for (let i = 0; i < 6; i++) y[i] = x[i] + (h / 2) * k[1][i]; deriv(y, a, ad, k[2]);
    for (let i = 0; i < 6; i++) y[i] = x[i] + h * k[2][i]; deriv(y, a, ad, k[3]);
    for (let i = 0; i < 6; i++) x[i] += (h / 6) * (k[0][i] + 2 * k[1][i] + 2 * k[2][i] + k[3][i]);
  }
  return x;
}

test('n and T come from orbit.js circularOrbit({h: 420})', () => {
  const o = circularOrbit({ h: 420 });
  near(N, o.n, 1e-12, 'n'); near(N, 1.1281538e-3, 1e-8, 'n value'); near(T_ORB, 5569.44, 0.01, 'T');
});

test('ZOH steps match the closed form to < 1 mm per orbit, and RK4 to < 1e-6 m over 600 s', () => {
  const x0 = X(-250, 30, -120, 0.05, -0.02, 0.01), a = X(1e-4, -2e-4, 5e-5);
  let x = Float64Array.from(x0); const P = phi(10), G = gamma(10);
  for (let i = 0; i < 557; i++) stepWith(P, G, x, null, 0, x);
  const ref = propagate(x0, 5570, 0);
  for (let i = 0; i < 3; i++) near(x[i], ref[i], 1e-3, 'orbit pos ' + i);
  x = Float64Array.from(x0);
  for (let i = 0; i < 6000; i++) step(x, 0.1, a, -1e-6, x);
  const r = rk4(x0, 0.01, 60000, a, -1e-6);
  for (let i = 0; i < 3; i++) near(x[i], r[i], 1e-6, 'rk4 pos ' + i);
});

test('the Jacobi integral drifts < 1e-9 (relative) over one orbit without drag', () => {
  const x = X(-400, 80, 150, 0.1, -0.05, 0.03), C0 = jacobi(x);
  for (let i = 0; i < 557; i++) step(x, 10, null, 0, x);
  assert.ok(Math.abs(jacobi(x) - C0) / Math.abs(C0) < 1e-9, `drift ${(jacobi(x) - C0) / C0}`);
});

test('100 m above V-bar drifts at Xd = -1.5 n Y = -0.169223 m/s', () => {
  const vd = -1.5 * N * 100; near(vd, -0.169223, 1e-4, 'Xd');
  const x = propagate(X(0, 100, 0, vd, 0, 0), T_ORB);
  near(x[1], 100, 1e-6, 'stays at Y = 100'); near(x[3], vd, 1e-9, 'constant Xd'); near(x[0], vd * T_ORB, 1e-6, 'X');
});

test('posigrade +X 0.1 m/s: Y(T/2) = 354.6 m, X(T) = -1670.8 m; radial +Y 0.1 m/s closes the ellipse', () => {
  const pos = X(0, 0, 0, 0.1, 0, 0);
  near(propagate(pos, T_ORB / 2)[1], 354.6, 0.1, 'Y(T/2)'); near(propagate(pos, T_ORB)[0], -1670.8, 0.5, 'X(T)');
  const rad = X(0, 0, 0, 0, 0.1, 0);
  near(propagate(rad, T_ORB / 2)[1], 0, 1e-3, 'radial Y(T/2)'); near(propagate(rad, T_ORB / 2)[0], -354.6, 0.1, 'radial X(T/2)');
  near(propagate(rad, T_ORB)[0], 0, 1e-3, 'radial X(T)');
});

test('drag: a_d < 0 sinks the ship and moves it ahead (X > 0, Y < 0), within 0.1 %', () => {
  for (const [ad, x24, y24] of [[-1e-6, 11191.2, -153.30], [-2e-7, 2238.2, -30.66]]) {
    const x = propagate(X(0, 0, 0, 0, 0, 0), DAY, ad);
    assert.ok(x[0] > 0 && x[1] < 0, 'ahead and below');
    near(x[0], x24, Math.abs(x24) * 1e-3, 'X(24 h)'); near(x[1], y24, Math.abs(y24) * 1e-3, 'Y(24 h)');
  }
});
