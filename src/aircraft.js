// Procedural Airbus A320-class airliner, shared by the corridor hazards (planes.js) and the jets over the Earth (below).
// Everything is built from lofted rings and lathes (no model files): a fuselage with the drooped, rounded Airbus nose and
// an up-swept tail cone; 25°-swept tapered wings with 5° dihedral and blended sharklets; a swept fin with dorsal fillet;
// a low-set swept tailplane; belly fairing; two under-wing engines (painted cowl, bare-metal intake lip, dark fan face
// with spinner, core cowl and exhaust cone) on short pylons. The livery (grey belly, blue cheat line, cabin window row,
// doors, six cockpit panes, titles, radome joint, blue fin with stripes and rudder line) is one 1024×512 canvas texture
// shared by every instance. Two materials → two instanced draw calls per consumer, ≈ 4.2k triangles per aircraft.
// Nose along +x, up +y, wings span z (starboard = +z); length 3.2 units, span ≈ 3.1.
import * as THREE from 'three';

const TRAIL = 56, TRAIL_DT = 0.35;
const L = 3.2, R = 0.168, XT = -1.6;                      // length, cabin radius, tail x (nose at +1.6)
const TW = 1024, TH = 512, FROWS = 384;                    // texture: fuselage band (rows 0..383: v = angle/2π · 0.75), fin livery + white swatch below
const SW = [0.992, 0.875];                                 // uv of the white swatch (right edge of the lower half): parts coloured only by vertex colour
const WUV = (sp) => (u) => [(520 + sp * 480) / TW, (386 + u * 124) / TH];   // wing region: span fraction → column, chord fraction → row
const lin = (hex) => { const c = new THREE.Color(hex); return [c.r, c.g, c.b]; };
const COL = { white: lin(0xffffff), belly: lin(0xb9bfc8), wing: lin(0xd2d6dc), blue: lin(0x1f55b0), lip: lin(0xc0c4cb), duct: lin(0x8e939b), fan: lin(0x14171c), spinner: lin(0x9a9ea6), dark: lin(0x24272d), core: lin(0x54565e), cone: lin(0x33353b) };
const US = [0, 0.012, 0.04, 0.09, 0.17, 0.28, 0.42, 0.6, 0.8, 1];   // chordwise samples of a foil section, LE → TE
const yt = (u) => 5 * (0.2969 * Math.sqrt(u) - 0.126 * u - 0.3516 * u * u + 0.2843 * u ** 3 - 0.1036 * u ** 4);   // NACA 00xx half-thickness (t/c = 1, closed TE)
let TEX = null;

// ---- geometry helpers --------------------------------------------------------------------------------------------
// Loft consecutive rings (equal point counts) into one indexed surface. ring = { p: [[x,y,z]…], uv: [u,v] | [[u,v]…], c: [r,g,b] }.
// wrap closes each ring; flip reverses the winding (rings that advance along +y with sections traced LE → +z → TE).
function loft(rings, wrap, flip = false) {
  const M = rings[0].p.length, N = rings.length, pos = [], uv = [], col = [], idx = [];
  for (const r of rings) for (let j = 0; j < M; j++) { pos.push(r.p[j][0], r.p[j][1], r.p[j][2]); const t = typeof r.uv[0] === 'number' ? r.uv : r.uv[j]; uv.push(t[0], t[1]); col.push(r.c[0], r.c[1], r.c[2]); }
  const S = wrap ? M : M - 1;
  for (let i = 0; i < N - 1; i++) for (let j = 0; j < S; j++) { const a = i * M + j, b = i * M + (j + 1) % M, c = a + M, d = b + M; if (flip) idx.push(a, c, b, b, c, d); else idx.push(a, b, c, b, d, c); }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2)); g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  g.setIndex(idx); g.computeVertexNormals(); return g;
}
// Closed foil section: LE → upper surface → TE → lower surface. p: LE, ud: thickness direction, c: chord (along −x), tc: t/c, k: upper share of the thickness.
function foil(p, ud, c, tc, k, uvf) {
  const pts = [], uvs = [], put = (u, s) => { const t = yt(u) * tc * c * (s > 0 ? k : k - 1); pts.push([p[0] - u * c + ud[0] * t, p[1] + ud[1] * t, p[2] + ud[2] * t]); uvs.push(uvf ? uvf(u) : SW); };
  for (const u of US) put(u, 1); for (let i = US.length - 2; i > 0; i--) put(US[i], -1);
  return { pts, uvs };
}
const ring = (st, k, uvf) => { const f = foil(st.p, st.ud, st.c, st.tc, k, uvf); return { p: f.pts, uv: uvf ? f.uvs : SW, c: st.col }; };
const cap = (st, col) => ({ p: US.map(() => [st.p[0] - st.c * 0.5, st.p[1], st.p[2]]).concat(US.slice(1, -1).map(() => [st.p[0] - st.c * 0.5, st.p[1], st.p[2]])), uv: SW, c: col });
function mirrorZ(g) {                                      // port-side copy of a starboard part: z → −z with the winding reversed
  const m = g.clone(), p = m.attributes.position.array, n = m.attributes.normal.array, ix = m.index.array;
  for (let i = 2; i < p.length; i += 3) { p[i] = -p[i]; n[i] = -n[i]; }
  for (let i = 0; i < ix.length; i += 3) { const t = ix[i + 1]; ix[i + 1] = ix[i + 2]; ix[i + 2] = t; }
  m.attributes.position.needsUpdate = true; return m;
}
function lathe(profile, segs, x0, y0, z0) {                // profile [[r, d, colour]] revolved about the engine axis, d = distance aft of the intake plane at x0
  const g = new THREE.LatheGeometry(profile.map(([r, d]) => new THREE.Vector2(r, d)), segs); g.rotateZ(Math.PI / 2); g.translate(x0, y0, z0);
  const n = g.attributes.position.count, col = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) col.set(profile[i % profile.length][2], i * 3);
  g.setAttribute('color', new THREE.BufferAttribute(col, 3)); return g;
}
function merge(list) {                                     // indexed geometries (position / normal / uv / color) → one; [{ geo, c?, uvc? }] fill missing colour / uv
  let nv = 0; const idx = [];
  for (const { geo } of list) nv += geo.attributes.position.count;
  const pos = new Float32Array(nv * 3), nor = new Float32Array(nv * 3), uv = new Float32Array(nv * 2), col = new Float32Array(nv * 3); let ov = 0;
  for (const { geo, c, uvc } of list) {
    const n = geo.attributes.position.count; pos.set(geo.attributes.position.array, ov * 3); nor.set(geo.attributes.normal.array, ov * 3);
    if (uvc || !geo.attributes.uv) for (let i = 0; i < n; i++) uv.set(uvc || SW, (ov + i) * 2); else uv.set(geo.attributes.uv.array, ov * 2);
    if (geo.attributes.color) col.set(geo.attributes.color.array, ov * 3); else for (let i = 0; i < n; i++) col.set(c, (ov + i) * 3);
    if (geo.index) for (let i = 0; i < geo.index.count; i++) idx.push(geo.index.array[i] + ov); else for (let i = 0; i < n; i++) idx.push(ov + i);
    ov += n; geo.dispose();
  }
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
  g.setAttribute('uv', new THREE.BufferAttribute(uv, 2)); g.setAttribute('color', new THREE.BufferAttribute(col, 3)); g.setIndex(idx); return g;
}

// ---- airframe ----------------------------------------------------------------------------------------------------
function fuselage() {                                      // rings of a circle r(x) centred at yc(x); u = x along the texture, v = angle from the belly
  const M = 28, rings = [];
  const station = (x, r, yc) => { const p = [], uv = []; for (let j = 0; j <= M; j++) { const a = 2 * Math.PI * j / M; p.push([x, yc - r * Math.cos(a), r * Math.sin(a)]); uv.push([(x - XT) / L, 0.75 * j / M]); } rings.push({ p, uv, c: COL.white }); };
  for (const s of [1, 0.995, 0.98, 0.95, 0.9, 0.82, 0.72, 0.6, 0.47, 0.34, 0.2, 0.08, 0]) station(1.05 + 0.55 * s, R * Math.pow(1 - Math.pow(s, 2.4), 0.62), -0.05 * Math.pow(s, 2.5));   // ogive nose, tip drooped 0.3 R
  for (const x of [0.5, -0.1]) station(x, R, 0);
  for (const s of [0, 0.12, 0.25, 0.38, 0.5, 0.62, 0.73, 0.83, 0.91, 0.97, 1]) { const top = R - 0.046 * s ** 3, bot = -R + 0.264 * Math.pow(s, 1.25); station(-0.5 - 1.1 * s, (top - bot) / 2, (top + bot) / 2); }   // tail cone: top line stays, belly sweeps up
  station(-1.615, 0, 0.109);
  const g = loft(rings, false), n = g.attributes.normal.array;
  for (let i = 0; i < rings.length; i++) { const a = i * (M + 1) * 3, b = (i * (M + 1) + M) * 3; for (let k = 0; k < 3; k++) { const v = n[a + k] + n[b + k]; n[a + k] = v; n[b + k] = v; } }   // stitch the seam normals
  g.normalizeNormals(); return g;
}
function wing() {                                          // starboard wing: root inside the fuselage → kink → tip → blended sharklet (blue)
  const yz = (z) => -0.1 + 0.0875 * z, st = [], add = (z, le, c, tc, ud, col, y = yz(z)) => st.push({ p: [le, y, z], c, tc, ud, col });
  add(0, 0.36, 0.72, 0.12, [0, 1, 0], COL.wing); add(0.17, 0.36, 0.7, 0.12, [0, 1, 0], COL.wing); add(0.5, 0.192, 0.43, 0.1, [0, 1, 0], COL.wing);
  add(0.85, 0.013, 0.323, 0.095, [0, 1, 0], COL.wing); add(1.2, -0.165, 0.215, 0.09, [0, 1, 0], COL.wing); add(1.45, -0.293, 0.14, 0.09, [0, 1, 0], COL.wing);
  const y0 = yz(1.45), rho = 0.075;
  for (const [deg, c, dle, col] of [[25, 0.135, -0.012, COL.wing], [50, 0.125, -0.028, COL.blue], [72, 0.11, -0.045, COL.blue]]) { const b = deg * Math.PI / 180; add(1.45 + rho * Math.sin(b), -0.293 + dle, c, 0.085, [0, Math.cos(b), -Math.sin(b)], col, y0 + rho * (1 - Math.cos(b))); }
  const b = 72 * Math.PI / 180; add(1.45 + rho * Math.sin(b) + 0.15 * Math.cos(b), -0.41, 0.05, 0.08, [0, Math.cos(b), -Math.sin(b)], COL.blue, y0 + rho * (1 - Math.cos(b)) + 0.15 * Math.sin(b));
  return loft(st.map((s, i) => ring(s, 0.62, i < 6 ? WUV(s.p[2] / 1.45) : null)).concat([cap(st[st.length - 1], COL.blue)]), true);
}
function tailplane() {                                     // starboard stabiliser: low on the tail cone, 30° sweep, 6° dihedral
  const st = [{ p: [-1.03, 0.07, 0], c: 0.3, tc: 0.1, ud: [0, 1, 0], col: COL.wing }, { p: [-1.1, 0.082, 0.12], c: 0.25, tc: 0.09, ud: [0, 1, 0], col: COL.wing }, { p: [-1.35, 0.125, 0.54], c: 0.1, tc: 0.08, ud: [0, 1, 0], col: COL.wing }];
  return loft(st.map((s) => ring(s, 0.55)).concat([cap(st[2], COL.wing)]), true);
}
function fin() {                                           // swept fin with a dorsal fillet; uv → the blue fin livery region of the texture
  const st = [{ p: [-0.55, 0.12, 0], c: 0.82, tc: 0.05 }, { p: [-0.84, 0.2, 0], c: 0.5, tc: 0.08 }, { p: [-1.0, 0.42, 0], c: 0.34, tc: 0.08 }, { p: [-1.235, 0.66, 0], c: 0.19, tc: 0.08 }];
  const uvf = (h) => (u) => [(20 + u * 472) / TW, (500 - h * 108) / TH];
  const rings = st.map((s) => ring({ ...s, ud: [0, 0, 1], col: COL.white }, 0.5, uvf(Math.max(0, (s.p[1] - 0.16) / 0.5))));
  const top = cap(st[3], COL.white); top.uv = top.p.map(() => uvf(1)(0.5)); rings.push(top);
  return loft(rings, true, true);
}
function pylon(z) {                                        // short swept strut between the cowl top and the wing underside
  const st = [{ p: [0.25, -0.115, z], c: 0.22, tc: 0.11, ud: [0, 0, 1], col: COL.wing }, { p: [0.14, -0.06, z], c: 0.3, tc: 0.085, ud: [0, 0, 1], col: COL.wing }];
  return loft(st.map((s) => ring(s, 0.5)), true, true);
}
const ENG = { x: 0.38, y: -0.215, z: 0.52 };
function canoe(z) {                                        // flap-track fairing under the trailing edge
  const te = -0.24 - (Math.abs(z) - 0.5) * 0.203, g = new THREE.SphereGeometry(1, 8, 6);
  g.applyMatrix4(new THREE.Matrix4().makeScale(0.1, 0.015, 0.016).setPosition(te + 0.025, -0.1 + 0.0875 * Math.abs(z) - 0.025, z)); return g;
}
const COWL = [[0.072, 0.066, COL.duct], [0.075, 0.02, COL.duct], [0.079, 0.004, COL.lip], [0.087, 0, COL.lip], [0.095, 0.005, COL.lip], [0.1, 0.025, COL.white], [0.101, 0.09, COL.white], [0.098, 0.19, COL.white], [0.088, 0.29, COL.white], [0.074, 0.35, COL.white], [0.064, 0.352, COL.dark]];
const FAN = [[0, 0.03, COL.spinner], [0.02, 0.07, COL.spinner], [0.072, 0.07, COL.fan]];
const EXH = [[0.064, 0.352, COL.dark], [0.05, 0.3, COL.dark], [0.05, 0.44, COL.core], [0.044, 0.47, COL.core], [0.018, 0.52, COL.cone], [0, 0.55, COL.cone]];

// ---- livery texture (shared, built once per page) -----------------------------------------------------------------
function livery() {
  if (TEX || typeof document === 'undefined') return TEX;
  const cv = document.createElement('canvas'); cv.width = TW; cv.height = TH; const g = cv.getContext('2d');
  const X = (x) => (x - XT) / L * TW, A = (deg) => deg / 360 * FROWS;   // model x → column; angle from the belly (deg, starboard side up to 180° = top) → row
  const fill = (x0, x1, a0, a1, s) => { g.fillStyle = s; g.fillRect(X(x0), A(a0), X(x1) - X(x0), A(a1) - A(a0)); };
  const poly = (pts, s, line) => { g.beginPath(); pts.forEach(([x, a], i) => (i ? g.lineTo(X(x), A(a)) : g.moveTo(X(x), A(a)))); g.closePath(); g.fillStyle = s; g.fill(); if (line) { g.strokeStyle = line; g.lineWidth = 1.5; g.stroke(); } };
  g.fillStyle = '#f3f5f7'; g.fillRect(0, 0, TW, TH);
  fill(-1.7, 1.7, 0, 50, '#b9bfc8'); fill(-1.7, 1.7, 310, 360, '#b9bfc8'); fill(-1.7, 1.7, 49.5, 51.5, '#2b5fb8'); fill(-1.7, 1.7, 308.5, 310.5, '#2b5fb8');   // grey belly, cheat lines
  fill(1.38, 1.7, 0, 360, '#e3e6ea'); fill(1.376, 1.384, 0, 360, '#a6abb3'); fill(-1.7, -1.52, 0, 360, '#3d4046');                                 // radome + joint, APU tip
  for (const side of [1, -1]) {
    const S = (a) => (side > 0 ? a : 360 - a);
    g.fillStyle = '#1b2029'; for (let x = 1.02; x > -1.0; x -= 0.0453) { g.beginPath(); g.roundRect(X(x) - 4.5, A(S(105)) - 3.5, 9, 7, 2); g.fill(); }   // cabin windows
    g.strokeStyle = '#9ea3ab'; g.lineWidth = 1;
    for (const [x0, x1, a0, a1] of [[1.045, 1.125, 62, 126], [-1.115, -1.035, 62, 126], [0.155, 0.205, 96, 122], [0.055, 0.105, 96, 122]]) g.strokeRect(X(x0), A(S(a0)), X(x1) - X(x0), A(S(a1)) - A(S(a0)));   // doors, over-wing exits
    poly([[1.215, S(108)], [1.32, S(118)], [1.345, S(150)], [1.30, S(171)], [1.215, S(150)]], '#0c1017', '#4a4f58');                              // cockpit: windshield
    poly([[1.13, S(106)], [1.21, S(108)], [1.21, S(148)], [1.15, S(146)]], '#0c1017', '#4a4f58'); poly([[1.06, S(108)], [1.125, S(106)], [1.14, S(144)], [1.07, S(140)]], '#0c1017', '#4a4f58');   // side panes
    g.save(); g.translate(X(1.0), A(S(113))); g.scale(side > 0 ? 1 : -1, side > 0 ? -1 : 1); g.fillStyle = '#1f55b0'; g.font = '700 15px Helvetica, Arial, sans-serif'; g.textAlign = side > 0 ? 'right' : 'left'; g.fillText('ASTRO PILOT', 0, 0); g.restore();   // titles, upright on both sides
  }
  g.fillStyle = '#1f55b0'; g.fillRect(0, FROWS, 512, TH - FROWS);                                                                                      // fin livery: blue, two stripes, rudder hinge line
  g.fillStyle = '#ffffff'; g.beginPath(); g.moveTo(40, 512); g.lineTo(120, 512); g.lineTo(330, 392); g.lineTo(250, 392); g.closePath(); g.fill();
  g.fillStyle = '#6fb1ff'; g.beginPath(); g.moveTo(130, 512); g.lineTo(175, 512); g.lineTo(385, 392); g.lineTo(340, 392); g.closePath(); g.fill();
  g.fillStyle = '#163f85'; g.fillRect(326, FROWS, 2, TH - FROWS);
  g.fillStyle = '#ff3030'; g.fillRect(X(0.3) - 2, A(180) - 2, 4, 4); g.fillRect(X(0.3) - 2, 0, 4, 2); g.fillRect(X(0.3) - 2, FROWS - 2, 4, 2);                   // beacons (top, belly)
  g.fillStyle = '#ffffff'; g.fillRect(512, FROWS, 512, TH - FROWS);                                                                                    // lower right: wing region + white swatch at the right edge
  const C = (sp) => 520 + sp * 480, Rw = (u) => 386 + u * 124, ln = (x0, y0, x1, y1) => { g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke(); };
  g.fillStyle = '#d9dde3'; g.fillRect(C(0.1), Rw(0.2), C(0.24) - C(0.1), Rw(0.62) - Rw(0.2));                                                           // walkway near the root
  g.strokeStyle = 'rgba(60,66,76,0.6)'; g.lineWidth = 1;
  ln(C(0.1), Rw(0.13), C(0.97), Rw(0.13)); for (const sp of [0.36, 0.56, 0.76]) ln(C(sp), Rw(0), C(sp), Rw(0.13));                                     // slats
  ln(C(0.1), Rw(0.7), C(0.73), Rw(0.7)); ln(C(0.75), Rw(0.72), C(0.97), Rw(0.72)); for (const sp of [0.1, 0.36, 0.73, 0.75, 0.97]) ln(C(sp), Rw(0.7), C(sp), Rw(1));   // flaps, aileron
  for (let i = 0; i < 5; i++) g.strokeRect(C(0.15 + i * 0.11), Rw(0.58), C(0.11) - C(0), Rw(0.7) - Rw(0.58));                                           // spoiler panels
  TEX = new THREE.CanvasTexture(cv); TEX.colorSpace = THREE.SRGBColorSpace; TEX.flipY = false; TEX.anisotropy = 4; return TEX;
}

// Parts for one aircraft: [{ geo, mat, name }] — one InstancedMesh each (2 draw calls) — plus dispose() (the livery texture is shared and kept).
export function airlinerParts({ glow = false } = {}) {
  const belly = new THREE.SphereGeometry(1, 16, 10); belly.applyMatrix4(new THREE.Matrix4().makeScale(0.6, 0.13, 0.2).setPosition(-0.15, -0.07, 0));
  const wingR = wing(), tailR = tailplane(), body = [
    { geo: fuselage() }, { geo: wingR.clone() }, { geo: mirrorZ(wingR) }, { geo: tailR.clone() }, { geo: mirrorZ(tailR) }, { geo: fin() }, { geo: belly, c: COL.belly, uvc: SW },
    { geo: pylon(ENG.z) }, { geo: pylon(-ENG.z) }, { geo: lathe(COWL, 20, ENG.x, ENG.y, ENG.z), uvc: SW }, { geo: lathe(COWL, 20, ENG.x, ENG.y, -ENG.z), uvc: SW },
    ...[0.62, 0.92, -0.62, -0.92].map((z) => ({ geo: canoe(z), c: COL.wing, uvc: SW }))];
  const metal = [{ geo: lathe(FAN, 20, ENG.x, ENG.y, ENG.z) }, { geo: lathe(FAN, 20, ENG.x, ENG.y, -ENG.z) }, { geo: lathe(EXH, 20, ENG.x, ENG.y, ENG.z) }, { geo: lathe(EXH, 20, ENG.x, ENG.y, -ENG.z) }];
  wingR.dispose(); tailR.dispose();
  const paint = new THREE.MeshStandardMaterial({ map: livery(), vertexColors: true, roughness: 0.42, metalness: 0.18, emissive: glow ? 0x2a3340 : 0x000000, emissiveIntensity: 1.0 });
  const dark = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.4, metalness: 0.75 });
  const parts = [{ geo: merge(body), mat: paint, name: 'paint' }, { geo: merge(metal), mat: dark, name: 'metal' }];
  return { parts, dispose() { for (const p of parts) { p.geo.dispose(); p.mat.dispose(); } } };
}

// Jets over the Earth for the low passes: procedural airliners on great-circle routes just above the cloud deck, exaggerated
// in size so they read from orbit, each trailing a fading contrail. Instanced (2 draw calls) + one LineSegments for all
// contrails. Attached to the Earth's tilted group and rotated with the ground.
export function createAircraft(parent, { R, n = 7, spin = 0.004, spin0 = 1.1, seed = 3, hub = null } = {}) {   // hub: direction (parent frame) of the surface point below the ship — all routes cross it
  let t = seed * 1013; const rng = () => { t = (t * 1664525 + 1013904223) >>> 0; return t / 4294967296; };
  const group = new THREE.Group(); group.rotation.y = spin0; parent.add(group);
  const model = airlinerParts({ glow: true });
  const meshes = model.parts.map(({ geo, mat }) => { const im = new THREE.InstancedMesh(geo, mat, n); im.instanceMatrix.setUsage(THREE.DynamicDrawUsage); im.frustumCulled = false; group.add(im); return im; });
  // contrails
  const trailGeo = new THREE.BufferGeometry(), tp = new Float32Array(n * TRAIL * 2 * 3), tc = new Float32Array(n * TRAIL * 2 * 3);
  trailGeo.setAttribute('position', new THREE.BufferAttribute(tp, 3)); trailGeo.setAttribute('color', new THREE.BufferAttribute(tc, 3));
  const trails = new THREE.LineSegments(trailGeo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false })); trails.frustumCulled = false; group.add(trails);
  const planes = [];
  const hubG = hub ? hub.clone().normalize().applyAxisAngle(new THREE.Vector3(0, 1, 0), -spin0) : null;
  for (let i = 0; i < n; i++) {
    let k = new THREE.Vector3(rng() * 2 - 1, rng() * 2 - 1, rng() * 2 - 1).normalize();                   // orbit-plane normal
    if (hubG) { const r = new THREE.Vector3(rng() * 2 - 1, rng() * 2 - 1, rng() * 2 - 1); k = new THREE.Vector3().crossVectors(hubG, r).normalize(); }   // great circle through the hub
    const u = new THREE.Vector3(1, 0, 0); if (Math.abs(k.x) > 0.9) u.set(0, 1, 0); u.cross(k).normalize(); const v = new THREE.Vector3().crossVectors(k, u);
    let th = rng() * Math.PI * 2;
    if (hubG) { th = Math.atan2(hubG.dot(v), hubG.dot(u)) + (rng() * 2 - 1) * 0.35; }                        // start within ±20° of the hub along the route
    planes.push({ k, u, v, th, w: (0.0045 + rng() * 0.0025) * (rng() < 0.5 ? 1 : -1), hist: new Float32Array(TRAIL * 3), hn: 0, hh: 0, acc: rng() * TRAIL_DT, alt: R * (1.009 + rng() * 0.003) });
  }
  const m = new THREE.Matrix4(), q = new THREE.Quaternion(), pos = new THREE.Vector3(), fwd = new THREE.Vector3(), up = new THREE.Vector3(), right = new THREE.Vector3(), basis = new THREE.Matrix4(), sc = new THREE.Vector3();
  let time = 0;
  return {
    group,
    setHub(h) {                                             // re-route every jet through a new surface point (parent frame): the ground track has moved since creation
      const hubL = h.clone().normalize().applyAxisAngle(new THREE.Vector3(0, 1, 0), -group.rotation.y);
      for (const P of planes) {
        const r = new THREE.Vector3(rng() * 2 - 1, rng() * 2 - 1, rng() * 2 - 1); P.k = new THREE.Vector3().crossVectors(hubL, r).normalize();
        P.u = new THREE.Vector3(1, 0, 0); if (Math.abs(P.k.x) > 0.9) P.u.set(0, 1, 0); P.u.cross(P.k).normalize(); P.v = new THREE.Vector3().crossVectors(P.k, P.u);
        P.th = Math.atan2(hubL.dot(P.v), hubL.dot(P.u)) + (rng() * 2 - 1) * 0.35; P.hn = 0; P.hh = 0;
      }
    },
    update(dt, vis) {                                       // vis 0..1 (low-pass altitude): planes scale in, contrails fade in
      time += dt; group.rotation.y += dt * spin; group.visible = vis > 0.02;
      if (!group.visible) return;
      for (let i = 0; i < n; i++) {
        const P = planes[i]; P.th += P.w * dt;
        pos.copy(P.u).multiplyScalar(Math.cos(P.th)).addScaledVector(P.v, Math.sin(P.th)).multiplyScalar(P.alt);
        fwd.copy(P.u).multiplyScalar(-Math.sin(P.th)).addScaledVector(P.v, Math.cos(P.th)).multiplyScalar(Math.sign(P.w)).normalize();
        up.copy(pos).normalize(); right.crossVectors(fwd, up).normalize();
        basis.makeBasis(fwd, up, right); q.setFromRotationMatrix(basis); sc.setScalar(vis * 1.6);
        m.compose(pos, q, sc); for (const im of meshes) im.setMatrixAt(i, m);
        P.acc += dt; if (P.acc >= TRAIL_DT) { P.acc -= TRAIL_DT; P.hist[P.hh * 3] = pos.x; P.hist[P.hh * 3 + 1] = pos.y; P.hist[P.hh * 3 + 2] = pos.z; P.hh = (P.hh + 1) % TRAIL; P.hn = Math.min(TRAIL, P.hn + 1); }
        for (let j = 0; j < TRAIL - 1; j++) {                 // segments from the plane back along the history, fading
          const o = (i * TRAIL + j) * 6, valid = j < P.hn - 1, a0 = (P.hh - 1 - j + 2 * TRAIL) % TRAIL, a1 = (P.hh - 2 - j + 2 * TRAIL) % TRAIL, f = valid ? (1 - j / TRAIL) * vis * 0.9 : 0;
          const x0 = j === 0 ? pos.x : P.hist[a0 * 3], y0 = j === 0 ? pos.y : P.hist[a0 * 3 + 1], z0 = j === 0 ? pos.z : P.hist[a0 * 3 + 2];
          tp[o] = x0; tp[o + 1] = y0; tp[o + 2] = z0; tp[o + 3] = valid ? P.hist[a1 * 3] : x0; tp[o + 4] = valid ? P.hist[a1 * 3 + 1] : y0; tp[o + 5] = valid ? P.hist[a1 * 3 + 2] : z0;
          tc[o] = tc[o + 1] = tc[o + 2] = f * 1.7; tc[o + 3] = tc[o + 4] = tc[o + 5] = f * 1.45;
        }
      }
      for (const im of meshes) im.instanceMatrix.needsUpdate = true; trailGeo.attributes.position.needsUpdate = true; trailGeo.attributes.color.needsUpdate = true;
    },
    dispose() { parent.remove(group); model.dispose(); trailGeo.dispose(); trails.material.dispose(); },
  };
}
