// Procedural Eiffel Tower for Astro Pilot — the real structure as iron lacework, plus five stylised variations.
// eiffelParts(variant, { height, lod }) → [{ geo, mat, name }] (+ .height .radius .footprint .floors .tris .ms): one indexed
// BufferGeometry (position / normal / uv / color) per material; local frame: centred on the y axis, base on y = 0, top at `height`.
//
// Structure (u = y / H, H = height to the antenna tip): the legs' outer arêtes follow Eiffel's wind-load curve, half-width
// o(u) = (base − core)·e^(−u/decay) + core − taper·u (for H = 330 m: a 125 m square base, ≈ 67 m at the 1st floor, 39 m at the
// 2nd, 15 m at the 3rd). Each leg is a lattice box girder whose share of the half-width grows from leg0 at the ground to 1 at
// `join`, so the legs converge above the 2nd floor and close the pointed slit; above it the shaft is one braced square tube
// with mid-face chords round the lift core. Chords are horizontal square sections swept along the curves (their sides lie in
// the tower's faces); every face panel carries a St Andrew's cross and a strut — lattice girders (twin flanges + zigzag web) at
// level 0, single bars at level 1, solid legs at level 2. The great arches (braced bands with spandrel hangers) span the faces
// under the 1st floor, whose frieze carries the arcade; three platforms with decks, railings and pavilions; the 3rd-floor cabin
// and a crown per variant (campanile + lantern + antenna, Art Deco steps + needle, gilded spire, ruby star, LED spire).
// Only one eighth is built — the wedge 0 ≤ x ≤ z: S parts are mirrored and turned (8 copies), A parts lie on a mirror plane
// (turned: 4), C parts on the axis (1) — so even the near level builds in a few tens of milliseconds.
// Materials: frame (MeshStandardMaterial, vertex colours = the paint graded dark → light up the tower, as the real one is, × a
// cheap occlusion term), accent (glass, stone, gilding), lights (MeshBasicMaterial, HDR vertex colours: luminance > 1.1 blooms
// in the host). Closed single-sided beams, no onBeforeCompile (the host adds its curvature). Deterministic (seeded sparkle).
import * as THREE from 'three';
import { mulberry32 } from './mathx.js';

const lin = (hex) => { const c = new THREE.Color(hex); return [c.r, c.g, c.b]; };                      // sRGB hex → linear
const hdr = (hex, lum) => { const c = lin(hex), l = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]; return c.map((v) => v * lum / Math.max(1e-4, l)); };
const sstep = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const off = (p, n, k) => [p[0] + n[0] * k, p[1] + n[1] * k, p[2] + n[2] * k];
const mix3 = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];

// ---- variants (all lengths are fractions of the height; colours sRGB) --------------------------------------------------------
const DEF = {
  label: '', hScale: 1,                                     // default height = 64 × hScale (an explicit `height` always wins)
  base: 0.189, core: 0.0373, decay: 0.2097, taper: 0.02,     // outer profile o(u); base = half the footprint
  leg0: 0.41, legP: 2.25, join: 0.576,                       // leg width share of o(u): leg0 at the ground → 1 at the join
  floors: [0.1745, 0.3506, 0.837], bands: [0.022, 0.012, 0.008],   // platform levels and the girder bands under them
  spring: 0.04, archDepth: 0.014, panel: 0.5, twist: 0, twistSpan: [0, 1], top: 'campanile', trimBands: false, seed: 1889,
  paint: [0x5c4431, 0x9a7a5a], metal: 0.15, rough: 0.6, glow: null,            // frame: paint at the foot → at the 3rd floor
  accent: { glass: 0x7c93a8, stone: 0xb9ab92, trim: 0x9a7a5a }, aMetal: 0.1, aRough: 0.3,
  lamps: { beacon: 0xffe2b0, air: 0xff2a1a },
};
const VARIANTS = {
  classic: { label: 'Paris 1889: wrought iron in Eiffel brown, graded dark to light, faithful proportions and campanile' },
  gold: { label: 'Champagne gold lit from within: warm strings on the arêtes and decks, sparkle bulbs, glowing lantern',
    paint: [0x9a7438, 0xdcbd7a], metal: 0.55, rough: 0.46, glow: [0xffa040, 0.07], accent: { glass: 0x7a6038, stone: 0xcbb68c, trim: 0xf0d08a },
    aMetal: 0.5, aRough: 0.38, lamps: { beacon: 0xfff0c8, air: 0xff3020, strings: 0xffc070, decks: 0xffd8a0, sparkle: 440 } },
  chrome: { label: 'New York Art Deco: satin-polished steel, straighter legs, a stepped crown with lit sunburst windows and a needle spire',
    hScale: 1.1, base: 0.158, core: 0.036, decay: 0.26, taper: 0.018, join: 0.54, floors: [0.16, 0.33, 0.78], bands: [0.02, 0.012, 0.008],
    top: 'deco', paint: [0x8d969f, 0xd2d8de], metal: 0.5, rough: 0.46, accent: { glass: 0x2c3844, stone: 0x9aa0a6, trim: 0xe4e9ee },
    aMetal: 0.7, aRough: 0.32, lamps: { air: 0xff2a1a, decks: 0xe8f2ff, crown: 0xfff1d6 } },
  ivory: { label: 'Dubai: a slender white lattice with gilded floors and a tall gold-ringed spire',
    hScale: 1.22, base: 0.142, core: 0.03, decay: 0.25, taper: 0.013, join: 0.56, floors: [0.15, 0.31, 0.79], bands: [0.02, 0.011, 0.008],
    top: 'needle', trimBands: true, paint: [0xe0d9ca, 0xf2eee4], metal: 0.03, rough: 0.55, accent: { glass: 0x6a7c8a, stone: 0xe8dcc0, trim: 0xd9ad4f },
    aMetal: 0.75, aRough: 0.42, lamps: { beacon: 0xfff4de, strings: 0xfff0d0 } },
  red: { label: 'Moscow: constructivist red steel, graphite floors and a ruby star on a gilded spire',
    hScale: 1.06, top: 'star', paint: [0x7a1712, 0xb02a1e], metal: 0.22, rough: 0.5, accent: { glass: 0x1b1c20, stone: 0x8e8a84, trim: 0xd2a24c },
    aMetal: 0.65, aRough: 0.4, lamps: { star: 0xff0a28, air: 0xff2a1a, decks: 0xffe0b0 } },
  twisted: { label: 'Futurist: white steel, teal glass rings, the upper shaft turned through a quarter twist, cyan LED lines on the arêtes',
    hScale: 1.1, base: 0.168, twist: Math.PI / 2, twistSpan: [0.15, 0.84], top: 'glass', glassRing: true, paint: [0xcfd5db, 0xeef1f4], metal: 0.05, rough: 0.52,
    accent: { glass: 0x2b5f70, stone: 0x5a646e, trim: 0xe8ecf0 }, aMetal: 0.45, aRough: 0.28, lamps: { beacon: 0xc8f4ff, air: 0xff2a1a, lines: 0x38e0ff } },
};
const merge = (v) => ({ ...DEF, ...v, accent: { ...DEF.accent, ...(v.accent || {}) }, lamps: { ...(v.lamps || DEF.lamps) } });
export const EIFFEL_VARIANTS = Object.fromEntries(Object.entries(VARIANTS).map(([k, v]) => [k, Object.freeze({ name: k, ...merge(v) })]));

function profile(V) {                                     // outer half-width o, its slope, leg inner offset i, local face width w
  const { base, core, decay, taper, leg0, legP, join: J } = V, A = base - core;
  const o = (u) => A * Math.exp(-u / decay) + core - taper * u, dO = (u) => -A / decay * Math.exp(-u / decay) - taper;
  const fr = (u) => (u >= J ? 1 : leg0 + (1 - leg0) * Math.pow(Math.max(0, u) / J, legP));
  return { o, dO, i: (u) => o(u) * (1 - fr(u)), w: (u) => o(u) * fr(u), J };
}

// ---- geometry buffers and primitives ---------------------------------------------------------------------------------------
const grow = (a) => { const r = new a.constructor(a.length * 2); r.set(a); return r; };
const buf = () => ({ p: new Float32Array(12288), n: new Float32Array(12288), c: new Float32Array(12288), t: new Float32Array(8192), i: new Uint32Array(12288), nv: 0, ni: 0, col: [1, 1, 1] });
function vert(b, x, y, z, nx, ny, nz, s, t) {
  if (3 * b.nv + 3 > b.p.length) { b.p = grow(b.p); b.n = grow(b.n); b.c = grow(b.c); b.t = grow(b.t); }
  const j = 3 * b.nv, k = 2 * b.nv; b.p[j] = x; b.p[j + 1] = y; b.p[j + 2] = z; b.n[j] = nx; b.n[j + 1] = ny; b.n[j + 2] = nz;
  b.c[j] = b.col[0]; b.c[j + 1] = b.col[1]; b.c[j + 2] = b.col[2]; b.t[k] = s; b.t[k + 1] = t; return b.nv++;
}
function tri(b, a, c, d) { if (b.ni + 3 > b.i.length) b.i = grow(b.i); b.i[b.ni++] = a; b.i[b.ni++] = c; b.i[b.ni++] = d; }

// Swept prism: n rings of m corners (counter-clockwise seen from the next ring). Faces are flat across and smooth along the
// sweep; each normal is the ring edge × the local tangent of its corner pair, so tapers and bends shade correctly.
function tube(b, R, n, m = 4) {
  const q = 3 * m;
  for (let j = 0; j < m; j++) {
    const a0 = 3 * j, c0 = 3 * ((j + 1) % m), v0 = b.nv;
    for (let k = 0; k < n; k++) {
      const P = q * Math.min(n - 1, k + 1), Q = q * Math.max(0, k - 1), a = q * k + a0, c = q * k + c0;
      const tx = R[P + a0] + R[P + c0] - R[Q + a0] - R[Q + c0], ty = R[P + a0 + 1] + R[P + c0 + 1] - R[Q + a0 + 1] - R[Q + c0 + 1], tz = R[P + a0 + 2] + R[P + c0 + 2] - R[Q + a0 + 2] - R[Q + c0 + 2];
      const ex = R[c] - R[a], ey = R[c + 1] - R[a + 1], ez = R[c + 2] - R[a + 2];
      let nx = ey * tz - ez * ty, ny = ez * tx - ex * tz, nz = ex * ty - ey * tx; const l = Math.hypot(nx, ny, nz) || 1; nx /= l; ny /= l; nz /= l;
      const s = k / (n - 1); vert(b, R[a], R[a + 1], R[a + 2], nx, ny, nz, s, j / m); vert(b, R[c], R[c + 1], R[c + 2], nx, ny, nz, s, (j + 1) / m);
    }
    for (let k = 0; k < n - 1; k++) { const v = v0 + 2 * k; tri(b, v, v + 1, v + 3); tri(b, v, v + 3, v + 2); }
  }
}
function rect(R, o, p, e, n, w, t) {                      // w × t rectangle round p (w in the face ⊥ axis e, t along n), CCW seen from ahead
  let nx = n[0], ny = n[1], nz = n[2]; const d = nx * e[0] + ny * e[1] + nz * e[2]; nx -= d * e[0]; ny -= d * e[1]; nz -= d * e[2];
  let l = Math.hypot(nx, ny, nz); if (l < 1e-5) { nx = e[2]; ny = 0; nz = -e[0]; l = Math.hypot(nx, nz); if (l < 1e-5) { nx = 1; nz = 0; l = 1; } }
  nx /= l; ny /= l; nz /= l;
  const sx = ny * e[2] - nz * e[1], sy = nz * e[0] - nx * e[2], sz = nx * e[1] - ny * e[0], hw = w / 2, ht = t / 2;
  for (let q = 0; q < 4; q++) { const cs = q === 1 || q === 2 ? hw : -hw, cn = q > 1 ? ht : -ht, k = o + 3 * q; R[k] = p[0] + sx * cs + nx * cn; R[k + 1] = p[1] + sy * cs + ny * cn; R[k + 2] = p[2] + sz * cs + nz * cn; }
}
const RB = new Float64Array(24);
function beam(b, a, c, n, w, t) {                          // straight box a → c without end caps (ends sit in joints)
  const d = sub(c, a); if (Math.abs(d[0]) + Math.abs(d[1]) + Math.abs(d[2]) < 1e-9) return;
  const e = unit(d); rect(RB, 0, a, e, n, w, t); rect(RB, 12, c, e, n, w, t); tube(b, RB, 2);
}
function sweep(b, pts, nf, w, t) {                         // rectangular section along a polyline; nf(p) → the face normal there
  const n = pts.length, R = new Float64Array(12 * n);
  for (let k = 0; k < n; k++) rect(R, 12 * k, pts[k], unit(sub(pts[Math.min(n - 1, k + 1)], pts[Math.max(0, k - 1)])), nf(pts[k]), w, t);
  tube(b, R, n);
}
function chord(b, f, u0, u1, n, h) {                      // horizontal square sections of half-size h(u) round f(u)
  const R = new Float64Array(12 * (n + 1));
  for (let k = 0; k <= n; k++) { const u = u0 + (u1 - u0) * k / n, p = f(u), s = h(u); R.set([p[0] + s, p[1], p[2] + s, p[0] + s, p[1], p[2] - s, p[0] - s, p[1], p[2] - s, p[0] - s, p[1], p[2] + s], 12 * k); }
  tube(b, R, n + 1);
}
function girder(b, a, c, n, d, r, wb, pitch = 1.15) {      // lattice girder: twin flanges d apart in the face + a zigzag web
  const ac = sub(c, a), s = unit(cross(n, ac)), h = d / 2, k = Math.max(2, Math.round(Math.hypot(ac[0], ac[1], ac[2]) / (pitch * d)));
  beam(b, off(a, s, h), off(c, s, h), n, r, r); beam(b, off(a, s, -h), off(c, s, -h), n, r, r);
  for (let j = 0; j < k; j++) { const g = j & 1 ? h : -h; beam(b, off(mix3(a, c, j / k), s, g), off(mix3(a, c, (j + 1) / k), s, -g), n, wb, wb); }
}
function quad(b, p, q, r, s, n) { const v = b.nv; for (const x of [p, q, r, s]) vert(b, x[0], x[1], x[2], n[0], n[1], n[2], 0, 0); tri(b, v, v + 1, v + 2); tri(b, v, v + 2, v + 3); }
function box(b, x0, x1, y0, y1, z0, z1) {                 // axis-aligned closed box
  tube(b, [x1, y0, z1, x1, y0, z0, x0, y0, z0, x0, y0, z1, x1, y1, z1, x1, y1, z0, x0, y1, z0, x0, y1, z1], 2);
  quad(b, [x1, y1, z1], [x1, y1, z0], [x0, y1, z0], [x0, y1, z1], [0, 1, 0]); quad(b, [x0, y0, z1], [x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [0, -1, 0]);
}
function cyl(b, y0, y1, r0, r1, m, caps = true) {         // m-sided frustum on the axis, flat sides, optional caps
  const R = new Float64Array(6 * m), a0 = Math.PI / m, ang = (j) => a0 - 2 * Math.PI * j / m;
  for (let j = 0; j < m; j++) { const c = Math.cos(ang(j)), s = Math.sin(ang(j)); R.set([r0 * c, y0, r0 * s], 3 * j); R.set([r1 * c, y1, r1 * s], 3 * m + 3 * j); }
  tube(b, R, 2, m);
  if (caps) for (const [y, r, up] of [[y1, r1, 1], [y0, r0, -1]]) {
    if (r < 2e-4) continue; const v = b.nv; vert(b, 0, y, 0, 0, up, 0, 0.5, 0.5);
    for (let j = 0; j < m; j++) vert(b, r * Math.cos(ang(j)), y, r * Math.sin(ang(j)), 0, up, 0, 0, 0);
    for (let j = 0; j < m; j++) { const p = v + 1 + j, q = v + 1 + (j + 1) % m; if (up > 0) tri(b, v, p, q); else tri(b, v, q, p); }
  }
}
function tri3(b, p, q, r, f) {                             // one flat triangle facing f
  let n = cross(sub(q, p), sub(r, p)); if (n[0] * f[0] + n[1] * f[1] + n[2] * f[2] < 0) { [q, r] = [r, q]; n = n.map((x) => -x); }
  n = unit(n); const v = b.nv; for (const x of [p, q, r]) vert(b, x[0], x[1], x[2], n[0], n[1], n[2], 0.5, 0.5); tri(b, v, v + 1, v + 2);
}
const OCT = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], OCTF = [0, 2, 4, 4, 2, 1, 1, 2, 5, 5, 2, 0, 4, 3, 0, 1, 3, 4, 5, 3, 1, 0, 3, 5];
function lamp(b, p, r) { const v = b.nv; for (const d of OCT) vert(b, p[0] + d[0] * r, p[1] + d[1] * r, p[2] + d[2] * r, d[0], d[1], d[2], 0.5, 0.5); for (let k = 0; k < 24; k += 3) tri(b, v + OCTF[k], v + OCTF[k + 1], v + OCTF[k + 2]); }
function star(b, rim, c, R, hex) {                        // faceted ruby star (both faces, ±z) with a gilded rim; kept below ~1.3 so ACES leaves it red
  const h = lin(hex), m = Math.max(...h), hi = h.map((v) => 1.25 * v / m), lo = h.map((v) => 0.55 * v / m), pts = [];
  for (let k = 0; k < 10; k++) { const a = Math.PI / 2 + k * Math.PI / 5, r = k & 1 ? 0.42 * R : R; pts.push([c[0] + r * Math.cos(a), c[1] + r * Math.sin(a), c[2]]); }
  for (const sd of [1, -1]) for (let k = 0; k < 10; k++) { b.col = (k & 1) ^ (sd > 0 ? 1 : 0) ? lo : hi; tri3(b, [c[0], c[1], c[2] + sd * 0.3 * R], pts[k], pts[(k + 1) % 10], [0, 0, sd]); }
  for (let k = 0; k < 10; k++) beam(rim, pts[k], pts[(k + 1) % 10], [0, 0, 1], 0.0011, 0.0018);
}
function levels(u0, u1, W, k) {                           // panel boundaries in [u0, u1]: each panel ≈ k × the local face width tall
  const M = 48, acc = [0]; for (let m = 1; m <= M; m++) acc.push(acc[m - 1] + (u1 - u0) / M / (k * W(u0 + (u1 - u0) * (m - 0.5) / M)));
  const n = Math.max(1, Math.round(acc[M])), out = [u0]; let m = 0;
  for (let j = 1; j < n; j++) { const t = acc[M] * j / n; while (acc[m + 1] < t) m++; out.push(u0 + (u1 - u0) * (m + (t - acc[m]) / (acc[m + 1] - acc[m])) / M); }
  out.push(u1); return out;
}

// ---- the tower (normalised: H = 1) ---------------------------------------------------------------------------------------------
function build(V, lod) {
  const M = {}; for (const k of ['frame', 'accent', 'lights']) M[k] = { S: buf(), A: buf(), C: buf() };
  const F = M.frame, Ac = M.accent, Li = M.lights, near = lod === 0, L2 = lod === 2, Lm = V.lamps, sides = L2 ? 6 : 8;
  const { o, dO, i, w, J } = profile(V), [f1, f2, f3] = V.floors, [b1, b2, b3] = V.bands;
  const hOO = (u) => Math.max(0.0012, 0.05 * w(u)), hIO = (u) => 0.8 * hOO(u), hII = (u) => 0.7 * hOO(u), hM = (u) => 0.9 * hOO(u);
  const Coo = (u) => [o(u), u, o(u)], Cio = (u) => [i(u), u, o(u)], Cii = (u) => [i(u), u, i(u)], Cm = (u) => [0, u, o(u)];
  const nF = (u) => unit([0, -dO(u), 1]), zo = (u) => 1.3 * hOO(u), edge = (u) => o(u) + zo(u), P = (x, u) => [x, u, edge(u)], PX = (f, u) => P(f * edge(u), u);
  const paint = (L, hex) => { L.S.col = L.A.col = L.C.col = hex == null ? [1, 1, 1] : lin(hex); };
  const glow = (hex, lum) => { Li.S.col = Li.A.col = Li.C.col = hdr(hex, lum); };
  let uT = J; for (let k = 0; k <= 120; k++) { const u = f2 + (J - f2) * k / 120; if (i(u) < 1.2 * hIO(u)) { uT = u; break; } }   // neighbouring inner chords touch

  // legs and the upper shaft: chords + braced panels (the girder bands under the floors are drawn by band())
  const u0 = 0.0075, segs = [[u0, f1 - b1], [f1 - b1, f1, 1], [f1, f2 - b2], [f2 - b2, f2, 1], [f2, J], [J, f3 - b3], [f3 - b3, f3, 1]];
  const LV = [u0]; for (const [a, c, bd] of segs) LV.push(...(bd ? [c] : levels(a, c, w, V.panel * (near ? 1 : 1.15)).slice(1)));
  const bandAt = (u) => segs.some(([a, , bd]) => bd && Math.abs(a - u) < 1e-9);
  const panel = (b, Lf, Rf, ua, ub, wd, inner) => {                // ua on the piers: the base frame strut too
    const BL = Lf(ua), BR = Rf(ua), TL = Lf(ub), TR = Rf(ub), n = unit(cross(sub(BR, BL), sub(TL, BL))), foot = ua === u0;
    if (near) {                                            // lattice girders: strut + St Andrew's cross; outer faces add the inscribed diamond
      const k = inner ? 0.8 : 1, gd = Math.max(0.0021, 0.088 * wd * k), gh = Math.max(0.0019, 0.07 * wd * k), r = Math.max(0.0006, 0.17 * gd), wb = Math.max(0.00032, 0.55 * r), e = 0.62 * r, p = inner ? 2.2 : 1.15;
      girder(b, TL, TR, n, gh, r, wb, p); girder(b, off(BL, n, e), off(TR, n, e), n, gd, r, wb, p); girder(b, off(BR, n, -e), off(TL, n, -e), n, gd, r, wb, p); if (foot) girder(b, BL, BR, n, gh, r, wb, p);
      if (!inner) { const m = [mix3(BL, TL, 0.5), mix3(TL, TR, 0.5), mix3(TR, BR, 0.5), mix3(BR, BL, 0.5)], t = 0.8 * r; for (let q = 0; q < 4; q++) beam(b, m[q], m[(q + 1) & 3], n, t, t); }
    } else {
      const t = Math.max(0.001, (inner ? 0.024 : 0.03) * wd), e = 0.6 * t;
      beam(b, TL, TR, n, t, t); beam(b, off(BL, n, e), off(TR, n, e), n, t, t); beam(b, off(BR, n, -e), off(TL, n, -e), n, t, t); if (foot) beam(b, BL, BR, n, t, t);
    }
  };
  if (!L2) {
    const d = near ? 1 : 0.4;
    chord(F.A, Coo, 0, f3, Math.round(72 * d), hOO); chord(F.S, Cio, 0, uT, Math.round(44 * d), hIO);
    chord(F.A, Cii, 0, uT, Math.round(40 * d), hII); chord(F.A, Cm, uT, f3, Math.round(28 * d), hM);
    for (let j = 0; j < LV.length - 1; j++) {
      const ua = LV[j], ub = LV[j + 1], wd = w((ua + ub) / 2);
      if (ua < J - 1e-9) { if (!bandAt(ua)) panel(F.S, Cio, Coo, ua, ub, wd); if (ub <= uT + 1e-9) panel(F.S, Cii, Cio, ua, ub, 0.9 * wd, true); }
      else if (!bandAt(ua)) panel(F.S, Cm, Coo, ua, ub, wd);
    }
    const hc = 0.0062, du = 0.02;                           // the lift core inside the shaft
    chord(F.A, (u) => [hc, u, hc], f2, f3, 4, () => 0.0006);
    for (let u = f2 + du; u < f3 - 0.002; u += du) {
      beam(F.S, [0, u, hc], [hc, u, hc], [0, 0, 1], 0.0005, 0.0005);
      if (near) { beam(F.S, [0, u - du, hc], [hc, u, hc], [0, 0, 1], 0.0004, 0.0004); beam(F.S, [hc, u - du, hc], [0, u, hc], [0, 0, 1], 0.0004, 0.0004); }
    }
    paint(Ac, V.accent.stone); const pier = (b, p, h) => { box(b, p[0] - h, p[0] + h, 0, 0.0055, p[2] - h, p[2] + h); box(b, p[0] - 0.8 * h, p[0] + 0.8 * h, 0.0055, 0.0078, p[2] - 0.8 * h, p[2] + 0.8 * h); };   // masonry pier + cap
    pier(Ac.A, Coo(0), 1.9 * hOO(0)); pier(Ac.S, Cio(0), 1.9 * hIO(0)); pier(Ac.A, Cii(0), 1.9 * hII(0));
  } else {                                                  // solid legs, drawn in a little (the lattice lets light through), and the shaft
    const n = 14, R = new Float64Array(12 * (n + 1)), m = 12, S2 = new Float64Array(12 * (m + 1));
    for (let k = 0; k <= n; k++) { const u = J * k / n, c = (o(u) + i(u)) / 2, h = 0.4 * (o(u) - i(u)); R.set([c + h, u, c + h, c + h, u, c - h, c - h, u, c - h, c - h, u, c + h], 12 * k); }
    for (let k = 0; k <= m; k++) { const u = J + (f3 - J) * k / m, h = 0.86 * o(u); S2.set([h, u, h, h, u, -h, -h, u, -h, -h, u, h], 12 * k); }
    tube(F.A, R, n + 1); tube(F.C, S2, m + 1);
  }

  // the great arches: an elliptic braced band from the leg's inner chord (spring) to the crown under the 1st-floor frieze
  {
    const ufb = f1 - b1, ad = V.archDepth, us = V.spring, ax = i(us) + 0.5 * hIO(us), ay = ufb - ad / 2 - us, N = near ? 26 : L2 ? 8 : 12, In = [], Ex = [], Cn = [];
    for (let k = 0; k <= N; k++) {
      const th = Math.PI / 2 * k / N, x = ax * Math.cos(th), u = us + ay * Math.sin(th);
      let nx = Math.cos(th) / ax, nu = Math.sin(th) / ay; const l = Math.hypot(nx, nu) || 1; nx /= l; nu /= l;
      In.push(P(x - nx * ad / 2, u - nu * ad / 2)); Ex.push(P(x + nx * ad / 2, u + nu * ad / 2)); Cn.push(P(x, u));
    }
    const nf = (p) => nF(p[1]);
    if (L2) sweep(F.S, Cn, nf, ad, 0.0026);
    else {
      sweep(F.S, In, nf, 0.0026, 0.0034); sweep(F.S, Ex, nf, 0.0022, 0.0034);
      for (let k = 0; k < N; k++) {
        if (near) beam(F.S, k & 1 ? Ex[k] : In[k], k & 1 ? In[k + 1] : Ex[k + 1], nf(In[k]), 0.0009, 0.0016);
        if (k && k % (near ? 3 : 2) === 0) beam(F.S, In[k], Ex[k], nf(In[k]), 0.0011, 0.0018);
      }
      let prev = null;                                      // spandrel hangers up to the frieze, braced in a zigzag
      for (let q = 1; q < 8; q++) {
        const x = ax * q / 8; let k = 0; while (k < N - 1 && Ex[k + 1][0] > x) k++;
        const t = (Ex[k][0] - x) / Math.max(1e-9, Ex[k][0] - Ex[k + 1][0]), ue = Ex[k][1] + (Ex[k + 1][1] - Ex[k][1]) * t;
        if (ufb - ue < 0.003) { prev = null; continue; }
        beam(F.S, P(x, ue), P(x, ufb), nF(ufb), 0.0011, 0.0016);
        if (near && prev) beam(F.S, P(prev[0], q & 1 ? prev[1] : ufb), P(x, q & 1 ? ufb : ue), nF(ufb), 0.0007, 0.0012);
        prev = [x, ue];
      }
    }
  }

  // girder bands under the floors (the 1st with its arcade), decks, railings, pavilions
  const band = (u0, u1, arcade) => {
    const T = V.trimBands ? Ac : F, um = arcade ? u0 + 0.45 * (u1 - u0) : u0, mid = (u0 + u1) / 2; if (V.trimBands) paint(Ac, V.accent.trim);
    if (L2) beam(T.S, P(0, mid), P(edge(mid), mid), nF(mid), u1 - u0, 0.0024);
    else {
      const rail = (u, r) => beam(T.S, P(0, u), P(edge(u), u), nF(u), r, 0.0019);
      rail(u0 + 0.0009, 0.0018); rail(u1 - 0.0009, 0.0018); if (arcade) rail(um, 0.0013);
      const bw0 = arcade ? um - u0 : 1.15 * (u1 - u0), nb = Math.max(2, Math.round(edge(u1) / bw0)), bw = edge(um) / nb, top = arcade ? um - bw / 2 : u1;
      for (let k = 0; k <= nb; k += near ? 1 : 2) beam(k && k < nb ? T.S : T.A, PX(k / nb, u0), PX(k / nb, top), nF(u0), 0.0011, 0.0014);
      if (arcade && near) for (let k = 0; k < nb; k++) {
        const pts = []; for (let q = 0; q <= 8; q++) { const a = Math.PI * q / 8, u = top + bw / 2 * Math.sin(a); pts.push(PX((k + 0.5 + 0.5 * Math.cos(a)) / nb, u)); }
        sweep(T.S, pts, (p) => nF(p[1]), 0.0008, 0.0013);
      }
      if (arcade) beam(F.S, PX(0, (um + u1) / 2), PX(1, (um + u1) / 2), nF(um), u1 - um - 0.002, 0.0008);   // the fascia that carries the names
      if (near || !arcade) { const lo = arcade ? um : u0; for (let k = 0; k < nb; k++) { beam(T.S, PX(k / nb, lo), PX((k + 1) / nb, u1), nF(lo), 0.0008, 0.0012); beam(T.S, PX((k + 1) / nb, lo), PX(k / nb, u1), nF(lo), 0.0008, 0.0012); } }
    }
    if (V.trimBands) paint(Ac, null);
  };
  const deck = (u, Po, Pi, th = 0.0022) => {             // pinwheel slabs (four turned copies tile the ring) on floor girders
    box(F.A, -Po, Pi, u - th, u, Pi, Po);
    if (near) for (let x = 0.003; x < Pi - 0.001; x += 0.006) beam(F.S, [x, u - th - 0.001, Pi], [x, u - th - 0.001, Po], [0, 1, 0], 0.0011, 0.002);
  };
  const railing = (u, Po, h) => {
    if (L2) return; const R = V.trimBands ? Ac : F; if (V.trimBands) paint(Ac, V.accent.trim);
    beam(R.S, [0, u + h, Po], [Po, u + h, Po], [0, 0, 1], 0.0007, 0.0007);
    if (near) {
      beam(R.S, [0, u + h / 2, Po], [Po, u + h / 2, Po], [0, 0, 1], 0.00045, 0.00045); beam(R.A, [Po, u, Po], [Po, u + h, Po], [0, 0, 1], 0.0006, 0.0006);
      for (let x = 0.00225; x < Po - 0.001; x += 0.0045) beam(R.S, [x, u, Po], [x, u + h, Po], [0, 0, 1], 0.0005, 0.0005);
    }
    if (V.trimBands) paint(Ac, null);
  };
  const P1 = edge(f1) + 0.0035, I1 = i(f1), P2 = edge(f2) + 0.003, I2 = i(f2), u2 = f2 + 0.0165, P2u = 0.74 * P2, o3 = o(f3), P3 = edge(f3) + 0.003, eh = 0.82 * o3, ut = f3 + 0.0125;
  band(f1 - b1, f1, true); deck(f1, P1, I1); railing(f1, P1, 0.0055);
  band(f2 - b2, f2, false); deck(f2, P2, I2); railing(f2, P2, 0.005); deck(u2, P2u, 0.8 * I2, 0.0018); railing(u2, P2u, 0.0045);
  band(f3 - b3, f3, false); box(F.C, -P3, P3, f3 - 0.002, f3, -P3, P3); railing(f3, P3, 0.0042);
  paint(Ac, V.accent.glass);
  if (V.glassRing) for (const [u, Po] of [[f1, P1], [f2, P2]]) box(Ac.A, 0.0045 - Po, Po - 0.014, u, u + 0.012, Po - 0.014, Po - 0.0045);   // glazed rings
  else {                                                   // 1st-floor pavilions: glass, mullions, a thin roof
    const ph = 0.0085, pz = P1 - 0.0075; box(Ac.A, -0.78 * I1, 0.78 * I1, f1, f1 + ph, pz - 0.013, pz); box(F.A, -0.8 * I1, 0.8 * I1, f1 + ph, f1 + ph + 0.0011, pz - 0.0135, pz + 0.0006);
    if (!L2) for (let x = 0.002; x < 0.78 * I1; x += 0.004) beam(F.S, [x, f1, pz + 0.0002], [x, f1 + ph, pz + 0.0002], [0, 0, 1], 0.0005, 0.0005);
  }
  box(Ac.C, -0.8 * I2, 0.8 * I2, f2, u2 - 0.0018, -0.8 * I2, 0.8 * I2);                                                                                           // 2nd-floor lift hall
  if (V.top === 'needle') paint(Ac, V.accent.trim);
  box(Ac.C, -eh, eh, f3, ut - 0.0016, -eh, eh); box(F.C, -eh - 0.0018, eh + 0.0018, ut - 0.0016, ut, -eh - 0.0018, eh + 0.0018); railing(ut, eh + 0.0018, 0.004);   // 3rd-floor cabin
  if (!L2) { for (let x = 0.004; x < eh - 0.001; x += 0.004) beam(F.S, [x, f3, eh + 0.0003], [x, ut - 0.0016, eh + 0.0003], [0, 0, 1], 0.0006, 0.0006); beam(F.A, [eh, f3, eh], [eh, ut, eh], [0, 0, 1], 0.001, 0.001); }

  // crowns
  const air = (b, p, r = 0.0016) => { if (Lm.air && !L2) { glow(Lm.air, 0.33); lamp(b, p, r); } };   // aviation red: bright reds above ~0.4 turn salmon under ACES
  if (V.top === 'campanile') {                              // four lattice ribs to the lantern, the beacon, the antenna mast
    const rw = 0.74 * eh, rt = 0.0048, ul = ut + 0.042, N = near ? 14 : 6, rib = [], at = (s) => [rt + (rw - rt) * Math.cos(s * Math.PI / 2), ut + (ul - ut) * Math.sin(s * Math.PI / 2)];
    const g = near ? 0.0016 : 0, ribO = [], ribI = [], nd = () => [1, 0, -1];
    for (let k = 0; k <= N; k++) {                        // rib curve in the diagonal plane, offset ± g along its in-plane normal
      const [r, u] = at(k / N), [r1, u1] = at(Math.min(1, (k + 1) / N)), [r0, u0] = at(Math.max(0, (k - 1) / N)), dr = r1 - r0, du = u1 - u0, l = Math.hypot(dr * Math.SQRT2, du);
      const nr = du / l / Math.SQRT2, nu = -dr * Math.SQRT2 / l; rib.push([r, u, r]); ribO.push([r + nr * g, u + nu * g, r + nr * g]); ribI.push([r - nr * g, u - nu * g, r - nr * g]);
    }
    if (near) { sweep(F.A, ribO, nd, 0.0011, 0.0022); sweep(F.A, ribI, nd, 0.0011, 0.0022); for (let k = 0; k < N; k++) beam(F.A, k & 1 ? ribO[k] : ribI[k], k & 1 ? ribI[k + 1] : ribO[k + 1], [1, 0, -1], 0.0006, 0.0012); }
    else sweep(F.A, rib, nd, 0.0026, 0.0022);
    if (!L2) for (const s of [0.33, 0.66, 1]) { const [r, u] = at(s); beam(F.S, [0, u, r], [r, u, r], [0, 0, 1], 0.0011, 0.0011); }
    if (Lm.beacon) { glow(Lm.beacon, 1.15); cyl(Li.C, ul, ul + 0.009, 0.0048, 0.0048, sides); } else { paint(Ac, V.accent.glass); cyl(Ac.C, ul, ul + 0.009, 0.0048, 0.0048, sides); }
    if (!L2) for (let k = 0; k < 8; k++) { const a = Math.PI * (k + 0.5) / 4; beam(F.C, [0.0052 * Math.cos(a), ul, 0.0052 * Math.sin(a)], [0.0052 * Math.cos(a), ul + 0.009, 0.0052 * Math.sin(a)], [Math.cos(a), 0, Math.sin(a)], 0.0007, 0.0007); }
    cyl(F.C, ul - 0.0012, ul, 0.0068, 0.0068, sides); cyl(F.C, ul + 0.009, ul + 0.0165, 0.0064, 0.0012, sides);
    const ua = ul + 0.016; cyl(F.C, ua, 0.95, 0.0021, 0.0016, sides); cyl(F.C, 0.95, 0.984, 0.0011, 0.0008, 6); cyl(F.C, 0.984, 0.997, 0.0005, 0.0003, 6);
    if (!L2) for (const u of [0.93, 0.966]) box(F.A, -0.0007, 0.0007, u - 0.005, u + 0.005, 0.0012, 0.0027);   // antenna panels
    air(Li.C, [0, 1 - 0.0016, 0]); air(Li.C, [0, 0.95, 0], 0.0022);
  } else if (V.top === 'deco') {                            // stepped tiers with arched hoods and sunburst windows, then the needle
    let y = ut, wk = 0.9 * eh, hk = 0.015; paint(Ac, V.accent.trim);
    for (let k = 0; k < 5; k++) {
      const wn = 0.8 * wk, hn = hk * 0.88; box(Ac.C, -wk, wk, y, y + hk, -wk, wk);
      if (!L2 && k < 4) {
        const pts = []; for (let q = 0; q <= 12; q++) { const a = Math.PI / 2 * q / 12; pts.push([wk * Math.cos(a), y + hk + hn * Math.sin(a), wk + 0.0009]); }
        sweep(Ac.S, pts, () => [0, 0, 1], 0.0021, 0.0018);
        if (Lm.crown) { glow(Lm.crown, 1.25); for (let q = 0; q < 3; q++) { const a = Math.PI / 2 * (q + 0.5) / 3, e = (r, da) => [wk * r * Math.cos(a + da), y + hk + hn * r * Math.sin(a + da), wn + 0.0004];
          tri3(Li.S, e(0.45, 0), e(0.86, -0.12), e(0.86, 0.12), [0, 0, 1]); } paint(Ac, V.accent.trim); }
      }
      y += hk; wk = wn; hk = hn;
    }
    cyl(Ac.C, y, 1, 0.0042, 0.00015, sides, false);
  } else if (V.top === 'needle') {                          // a Burj-like tapering spire, gold rings at the joints
    for (const [a, c, r0, r1] of [[ut, 0.87, 0.0095, 0.0068], [0.87, 0.922, 0.0056, 0.0039], [0.922, 0.963, 0.0031, 0.0019], [0.963, 0.9968, 0.0013, 0.0004]]) {
      cyl(F.C, a, c, r0, r1, sides); paint(Ac, V.accent.trim); cyl(Ac.C, a - 0.0014, a + 0.0014, r0 * 1.3, r0 * 1.3, sides);
    }
    if (Lm.beacon && !L2) { glow(Lm.beacon, 1.3); lamp(Li.C, [0, 1 - 0.0016, 0], 0.0016); }
  } else if (V.top === 'star') {                            // gilded spire, a ball and the faceted ruby star (faces ±z)
    paint(Ac, V.accent.trim); cyl(Ac.C, ut, 0.948, 0.0068, 0.0013, sides); cyl(Ac.C, 0.946, 0.956, 0.0026, 0.0026, sides);
    const R = 0.022; star(Li.C, Ac.C, [0, 1 - R - 0.0006, 0], R, Lm.star);
  } else {                                                  // 'glass': a slim white spire ringed with LEDs
    cyl(F.C, ut, 0.9968, 0.0064, 0.0005, 10);
    if (Lm.lines && !L2) { glow(Lm.lines, 1.15); for (let k = 1; k <= 7; k++) { const t = k / 8, u = ut + (0.9968 - ut) * t, r = 0.0064 + (0.0005 - 0.0064) * t + 0.0005; cyl(Li.C, u - 0.0007, u + 0.0007, r, r, 10); } }
    if (Lm.beacon && !L2) { glow(Lm.beacon, 1.3); lamp(Li.C, [0, 1 - 0.0016, 0], 0.0016); }
  }
  air(Li.A, [eh + 0.0018, ut + 0.0012, eh + 0.0018], 0.0013);

  // light strings, deck lights, sparkle bulbs, LED lines
  if (!L2) {
    const lr = near ? 0.0011 : 0.0019, edges = [[f1, P1], [f2, P2], [u2, P2u], [f3, P3]];
    if (Lm.strings) { glow(Lm.strings, 1.14); const n = near ? 150 : 64; for (let k = 0; k < n; k++) { const u = 0.012 + (f3 - 0.02) * (k + 0.5) / n, h = 0.8 * hOO(u) + lr; lamp(Li.A, [o(u) + h, u, o(u) + h], lr); } }
    if (Lm.decks) { glow(Lm.decks, 1.12); const s = near ? 0.006 : 0.012; for (const [u, Po] of edges) { for (let x = s / 2; x < Po - 0.002; x += s) lamp(Li.S, [x, u + 0.0012, Po + 0.0012], 0.9 * lr); lamp(Li.A, [Po + 0.0012, u + 0.0012, Po + 0.0012], 0.9 * lr); } }
    if (Lm.sparkle) {
      glow(0xeef4ff, 1.5); const rng = mulberry32(V.seed), n = Math.round(Lm.sparkle / 8 * (near ? 1 : 0.5)), w0 = w(0);
      for (let k = 0; k < n;) { const u = 0.01 + rng() * (f3 - 0.02); if (rng() * w0 > w(u)) continue; const lo = u < J ? i(u) : 0, x = lo + rng() * (o(u) - lo); lamp(Li.S, [x, u, o(u) + 0.8 * zo(u)], near ? 0.0008 : 0.0014); k++; }
    }
    if (Lm.lines) {
      glow(Lm.lines, 1.12); chord(Li.A, (u) => { const h = hOO(u); return [o(u) + h, u, o(u) + h]; }, 0.004, f3, near ? 96 : 40, () => 0.0006);
      for (const [u, Po] of edges) beam(Li.S, [0, u + 0.0004, Po + 0.0006], [Po + 0.0006, u + 0.0004, Po + 0.0006], [0, 0, 1], 0.0008, 0.0008);
    }
  }
  return M;
}

// ---- symmetric expansion, shading, twist, scale → BufferGeometries -------------------------------------------------------
function materials(V) {
  const frame = new THREE.MeshStandardMaterial({ name: `eiffel-${V.name}-frame`, vertexColors: true, metalness: V.metal, roughness: V.rough });
  if (V.glow) { frame.emissive.set(V.glow[0]); frame.emissiveIntensity = V.glow[1]; }
  const accent = new THREE.MeshStandardMaterial({ name: `eiffel-${V.name}-accent`, vertexColors: true, metalness: V.aMetal, roughness: V.aRough });
  const lights = new THREE.MeshBasicMaterial({ name: `eiffel-${V.name}-lights`, vertexColors: true });
  return { frame, accent, lights };
}
const ROT = [[1, 0, 0, 1], [0, 1, -1, 0], [-1, 0, 0, -1], [0, -1, 1, 0]];   // x' = a x + b z, z' = c x + d z: the four quarter turns
function finalize(M, V, lod, H) {
  const { o } = profile(V), f3 = V.floors[2], pg = V.paint.map(lin), mats = materials(V), out = [];
  for (const key of ['frame', 'accent', 'lights']) {
    const { S, A, C } = M[key], nv = 8 * S.nv + 4 * A.nv + C.nv, ni = 8 * S.ni + 4 * A.ni + C.ni; if (!ni) continue;
    const pos = new Float32Array(3 * nv), nor = new Float32Array(3 * nv), col = new Float32Array(3 * nv), uv = new Float32Array(2 * nv), idx = nv > 65535 ? new Uint32Array(ni) : new Uint16Array(ni);
    let ov = 0, oi = 0;
    const put = (b, [ta, tb, tc, td], flip) => {
      for (let v = 0; v < b.nv; v++) {
        const j = 3 * v, k = 3 * (ov + v), x = b.p[j], z = b.p[j + 2], nx = b.n[j], nz = b.n[j + 2];
        pos[k] = ta * x + tb * z; pos[k + 1] = b.p[j + 1]; pos[k + 2] = tc * x + td * z; nor[k] = ta * nx + tb * nz; nor[k + 1] = b.n[j + 1]; nor[k + 2] = tc * nx + td * nz;
        col[k] = b.c[j]; col[k + 1] = b.c[j + 1]; col[k + 2] = b.c[j + 2]; uv[2 * (ov + v)] = b.t[2 * v]; uv[2 * (ov + v) + 1] = b.t[2 * v + 1];
      }
      for (let t = 0; t < b.ni; t += 3) { idx[oi++] = b.i[t] + ov; idx[oi++] = b.i[t + (flip ? 2 : 1)] + ov; idx[oi++] = b.i[t + (flip ? 1 : 2)] + ov; }
      ov += b.nv;
    };
    for (const r of ROT) { put(S, r, false); put(S, [r[1], r[0], r[3], r[2]], true); put(A, r, false); }   // wedge, its mirror image, the plane parts
    put(C, ROT[0], false);
    if (key !== 'lights') for (let v = 0; v < nv; v++) {   // paint gradient × occlusion: members deep inside the legs and undersides darker
      const k = 3 * v, y = pos[k + 1], r = y < f3 + 0.004 ? Math.min(1, Math.max(Math.abs(pos[k]), Math.abs(pos[k + 2])) / Math.max(1e-4, o(Math.max(0, y)))) : 1;
      const s = (key === 'frame' ? 0.5 + 0.5 * sstep(0.25, 0.97, r) : 0.8 + 0.2 * sstep(0.25, 0.97, r)) * (nor[k + 1] < -0.6 ? 0.8 : 1), g = sstep(0, f3, y);
      for (let c = 0; c < 3; c++) col[k + c] *= s * (key === 'frame' ? (pg[0][c] + (pg[1][c] - pg[0][c]) * g) * (lod === 2 ? 0.72 : 1) : 1);
    }
    if (V.twist) for (let v = 0; v < nv; v++) {             // twist about the axis, eased in over twistSpan (the flared legs stay classic)
      const k = 3 * v, a = V.twist * sstep(V.twistSpan[0], V.twistSpan[1], pos[k + 1]), c = Math.cos(a), s = Math.sin(a), x = pos[k], z = pos[k + 2], nx = nor[k], nz = nor[k + 2];
      pos[k] = c * x - s * z; pos[k + 2] = s * x + c * z; nor[k] = c * nx - s * nz; nor[k + 2] = s * nx + c * nz;
    }
    for (let k = 0; k < 3 * nv; k++) pos[k] = (k % 3 === 1 && pos[k] < 0 ? 0 : pos[k]) * H;   // the lowest members' ends sit on the ground
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
    g.setAttribute('uv', new THREE.BufferAttribute(uv, 2)); g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    g.setIndex(new THREE.BufferAttribute(idx, 1)); g.computeBoundingBox(); g.computeBoundingSphere(); g.userData = { variant: V.name, lod, part: key };
    out.push({ geo: g, mat: mats[key], name: key });
  }
  return out;
}

// variant: a name from EIFFEL_VARIANTS or a parameter object (e.g. { ...EIFFEL_VARIANTS.gold, twist: 0.3 }).
// height: tip of the antenna / spire (default 64 × the variant's hScale). lod: 0 near, 1 far, 2 extra-far.
export function eiffelParts(variant = 'classic', { height, lod = 0 } = {}) {
  const t0 = performance.now(), V = typeof variant === 'object' ? { name: 'custom', ...merge(variant) } : EIFFEL_VARIANTS[variant] || EIFFEL_VARIANTS.classic;
  const L = Math.max(0, Math.min(2, lod | 0)), H = height ?? 64 * V.hScale, parts = finalize(build(V, L), V, L, H);
  return Object.assign(parts, { variant: V.name, lod: L, height: H, radius: Math.SQRT2 * (V.base + 0.008) * H, footprint: 2 * V.base * H,
    floors: V.floors.map((f) => f * H), tris: parts.reduce((s, p) => s + p.geo.index.count / 3, 0), ms: performance.now() - t0 });
}
