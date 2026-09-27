// Astro Pilot — ship.js — procedural spacecraft for three.js (no external assets, no fetches).
//
// Design ("if Apple designed a spacecraft"): ONE seamless lofted hull — a long drooped nose, flattened
// underside, a low dark-glass canopy blister that fairs back into a spine, and an aft body that swells into
// two blended engine lobes (the "nacelles" are part of the same surface, not bolted on) ending in a flat
// engine face with two recessed nozzle cups. Blended 55° delta wings with strakes and 5° dihedral, twin fins
// canted 35°. Pearl-white clearcoat hull, warm dark-grey engine section, ONE accent colour: ice-blue emissive
// piping along the wing leading edges and around the nozzles. Effects: HDR additive exhaust cones + nozzle
// glows + a single PointLight, 12 RCS puffs, a 1 Hz tail strobe, explosion (80 particles, flash, tumbling
// fragments). Every glow / puff / particle is a single instanced-billboard draw call. Moving parts (shipsurfaces.js):
// 4 elevons, Shuttle-style split rudders / speed brake, a body flap, thrust-vectored plumes and wingtip vapour; the
// wings and fins carry the matching dark hinge coves.
//
// Model frame: nose +X, up +Y, right wing +Z. Nose x = +2.10, tail x = −1.90, span 3.4, height ≈ 0.9.
import * as THREE from 'three';
import { WING_COVE, FIN_COVE, FIN_MOUNT, splitCove, buildSurfaceGeometry, createSurfaces, createVapour } from './shipsurfaces.js';

const DEG = Math.PI / 180, TAU = Math.PI * 2;
const clamp = (v, a, b) => Math.min(Math.max(v, a), b);
const V = (x, y, z) => new THREE.Vector3(x, y, z);
const ACCENT = new THREE.Color(0x8ec5ff);
const ICE = [0.55, 0.8, 1.0];
const GLOW_N = 96, PART_N = 80, FRAG_N = 7; // glow slots: 0-1 nozzles, 2-13 RCS, 14 strobe, 15 flash, 16.. particles

// ---------- hull profile keys: [x, halfWidth, heightAboveCentre, depthBelowCentre, centreY, twinLobeDip] -------
const HULL_KEYS = [
  [2.10, 0.000, 0.000, 0.000, -0.110, 0.00], [1.95, 0.060, 0.050, 0.045, -0.105, 0.00],
  [1.60, 0.160, 0.130, 0.100, -0.085, 0.00], [1.20, 0.250, 0.200, 0.150, -0.050, 0.00],
  [0.70, 0.330, 0.270, 0.190, -0.020, 0.00], [0.20, 0.390, 0.300, 0.220, 0.000, 0.00],
  [-0.30, 0.420, 0.300, 0.230, 0.000, 0.05], [-0.80, 0.520, 0.290, 0.240, 0.000, 0.30],
  [-1.30, 0.660, 0.280, 0.250, 0.000, 0.52], [-1.60, 0.715, 0.275, 0.250, 0.000, 0.55],
  [-1.86, 0.720, 0.270, 0.250, 0.000, 0.55],
];
const CANOPY_KEYS = [ // [x, halfWidth, height above the seat line]
  [1.10, 0.005, 0.004], [0.95, 0.10, 0.07], [0.70, 0.19, 0.135], [0.40, 0.235, 0.165],
  [0.05, 0.235, 0.155], [-0.30, 0.19, 0.105], [-0.62, 0.09, 0.04], [-0.82, 0.0, 0.0],
];

// Cubic Hermite through keys sorted by descending x (finite-difference tangents → C1, gentle curves).
function sampleKeys(keys, x) {
  const n = keys.length; let i = 0;
  while (i < n - 2 && x < keys[i + 1][0]) i++;
  const a = keys[i], b = keys[i + 1], p = keys[Math.max(i - 1, 0)], q = keys[Math.min(i + 2, n - 1)];
  const h = b[0] - a[0], t = clamp((x - a[0]) / h, 0, 1), t2 = t * t, t3 = t2 * t;
  const h00 = 2 * t3 - 3 * t2 + 1, h10 = t3 - 2 * t2 + t, h01 = 3 * t2 - 2 * t3, h11 = t3 - t2;
  const out = [x];
  for (let c = 1; c < a.length; c++) {
    const m0 = (b[c] - p[c]) / (b[0] - p[0]), m1 = (q[c] - a[c]) / (q[0] - a[0]);
    const v = h00 * a[c] + h10 * h * m0 + h01 * b[c] + h11 * h * m1;
    out[c] = c === 4 ? v : Math.max(v, 0); // widths / heights / dip never negative, centreY may be
  }
  return out;
}
const hullAt = (x) => { const s = sampleKeys(HULL_KEYS, x); return { x, w: s[1], ht: s[2], hb: s[3], cy: s[4], m: s[5] }; };
const hullTop = (x) => { const s = hullAt(x); return s.cy + s.ht * (1 - s.m); };

// One cross-section point: slightly super-elliptic (fuller) section; aft of the canopy a Gaussian valley on
// the centreline turns the section into two blended engine lobes.
function sectionPoint(st, th, out) {
  const c = Math.cos(th), s = Math.sin(th);
  const cs = Math.sign(c) * Math.pow(Math.abs(c), 0.9), ss = Math.sign(s) * Math.pow(Math.abs(s), 0.9);
  const dip = st.m * Math.exp(-(c * c) / 0.16);
  const h = s >= 0 ? st.ht * (1 - dip) : st.hb * (1 - 0.45 * dip);
  out.set(st.x, st.cy + h * ss, st.w * cs);
}

// Loft rings into one smooth indexed surface; optional chamfered flat end cap and a material split at x.
function loft(stations, segs, { cap = false, chamfer = 0, splitX = null } = {}) {
  const pos = [], idx = [], p = new THREE.Vector3(), rings = stations.slice();
  if (chamfer > 0) { const l = rings[rings.length - 1]; rings.push({ ...l, x: l.x - chamfer, w: l.w * 0.93, ht: l.ht * 0.93, hb: l.hb * 0.93 }); }
  for (const st of rings) for (let j = 0; j < segs; j++) { sectionPoint(st, (j / segs) * TAU, p); pos.push(p.x, p.y, p.z); }
  let split = -1;
  for (let i = 0; i < rings.length - 1; i++) {
    if (splitX !== null && split < 0 && rings[i].x < splitX) split = idx.length;
    for (let j = 0; j < segs; j++) {
      const a = i * segs + j, b = i * segs + ((j + 1) % segs), c = a + segs, d = b + segs;
      idx.push(a, b, c, b, d, c);
    }
  }
  if (cap) { // separate vertices → flat shading on the engine face
    const l = rings[rings.length - 1], base = pos.length / 3;
    for (let j = 0; j < segs; j++) { sectionPoint(l, (j / segs) * TAU, p); pos.push(p.x, p.y, p.z); }
    pos.push(l.x, l.cy, 0);
    for (let j = 0; j < segs; j++) idx.push(base + j, base + ((j + 1) % segs), base + segs);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(idx); g.computeVertexNormals();
  if (split >= 0) { g.addGroup(0, split, 0); g.addGroup(split, idx.length - split, 1); }
  return g;
}
function hullStations(n = 46) {
  const out = [];
  for (let i = 0; i < n; i++) out.push(hullAt(2.10 - 3.96 * Math.pow(i / (n - 1), 1.25)));
  return out;
}
function canopyStations(n = 18) {
  const out = [];
  for (let i = 0; i < n; i++) {
    const x = 1.10 - 1.92 * (i / (n - 1)), s = sampleKeys(CANOPY_KEYS, x);
    out.push({ x, w: s[1], ht: s[2], hb: 0.12, cy: hullTop(x) - 0.06, m: 0 });
  }
  return out;
}

// Concatenate non-indexed geometries (position / normal / uv) after applying per-part matrices.
function mergeGeoms(parts) {
  const gs = parts.map(({ geom, matrix }) => { const g = geom.index ? geom.toNonIndexed() : geom; if (matrix) g.applyMatrix4(matrix); return g; });
  const n = gs.reduce((s, g) => s + g.attributes.position.count, 0), out = new THREE.BufferGeometry();
  for (const name of ['position', 'normal', 'uv']) {
    if (!gs.every((g) => g.attributes[name])) continue;
    const size = gs[0].attributes[name].itemSize, arr = new Float32Array(n * size); let o = 0;
    for (const g of gs) { arr.set(g.attributes[name].array, o); o += g.attributes[name].array.length; }
    out.setAttribute(name, new THREE.BufferAttribute(arr, size));
  }
  for (const g of gs) g.dispose();
  return out;
}

function wingGeometry() { // one planform through the fuselage: strake → 55° leading edge → raked tip; elevon coves in the trailing edge
  const s = new THREE.Shape(), cove = (sg) => (sg > 0 ? WING_COVE : WING_COVE.slice().reverse()).forEach(([x, z]) => s.lineTo(x, sg * z));
  s.moveTo(1.25, 0); s.quadraticCurveTo(0.55, 0.42, 0.18, 0.62); s.lineTo(-1.30, 1.66);
  s.quadraticCurveTo(-1.56, 1.75, -1.74, 1.66); cove(1); s.lineTo(-1.66, 0.55); s.lineTo(-1.66, -0.55); cove(-1); s.lineTo(-1.74, -1.66);
  s.quadraticCurveTo(-1.56, -1.75, -1.30, -1.66); s.lineTo(0.18, -0.62); s.quadraticCurveTo(0.55, -0.42, 1.25, 0);
  const g = new THREE.ExtrudeGeometry(s, { depth: 0.024, bevelEnabled: true, bevelThickness: 0.018, bevelSize: 0.022, bevelSegments: 3, curveSegments: 8 });
  g.translate(0, 0, -0.012); g.rotateX(Math.PI / 2); // shape y → world z, thickness → world y
  const a = g.attributes.position, tan5 = Math.tan(5 * DEG);
  for (let i = 0; i < a.count; i++) a.setY(i, a.getY(i) - 0.06 + tan5 * Math.abs(a.getZ(i))); // dihedral shear
  g.computeVertexNormals();
  return g;
}
function finGeometry() { // one fin in its own frame (FIN_MOUNT cants it 35° and places it), rudder cove in the trailing edge
  const s = new THREE.Shape();
  s.moveTo(-0.94, -0.14); s.lineTo(-1.0, 0.0); s.lineTo(-1.60, 0.50); s.quadraticCurveTo(-1.74, 0.535, -1.86, 0.48);
  for (const [x, y] of FIN_COVE) s.lineTo(x, y);
  s.lineTo(-1.86, -0.14); s.closePath();
  const g = new THREE.ExtrudeGeometry(s, { depth: 0.016, bevelEnabled: true, bevelThickness: 0.012, bevelSize: 0.014, bevelSegments: 2, curveSegments: 6 });
  g.translate(0, 0, -0.008); return g;
}
function airframeGeometry() { // wings + both fins in one mesh: group 0 pearl, group 1 the dark hinge coves
  const [wp, wd] = splitCove(wingGeometry(), [WING_COVE, WING_COVE.map(([x, z]) => [x, -z])], 0.036, 1);
  const [fp, fd] = splitCove(finGeometry(), [FIN_COVE], 0.025, 2);
  const fins = (f) => [1, -1].map((side) => ({ geom: f.clone(), matrix: FIN_MOUNT(side) }));
  const nP = wp.attributes.position.count + 2 * fp.attributes.position.count;
  const g = mergeGeoms([{ geom: wp }, ...fins(fp), { geom: wd }, ...fins(fd)]);
  fp.dispose(); fd.dispose(); g.addGroup(0, nP, 0); g.addGroup(nP, g.attributes.position.count - nP, 1);
  return g;
}
function nozzleGeometry() { // lathe: outer skirt + recessed inner cup, axis along X
  const prof = [[0, 0.03], [0.08, 0.02], [0.13, -0.02], [0.155, -0.06], [0.185, -0.085], [0.2, -0.05], [0.205, 0.02], [0.195, 0.09]].map(([r, h]) => new THREE.Vector2(r, h));
  const l = new THREE.LatheGeometry(prof, 28); l.rotateZ(-Math.PI / 2);
  const g = mergeGeoms([1, -1].map((s) => ({ geom: l.clone(), matrix: new THREE.Matrix4().makeTranslation(-1.90, -0.01, s * 0.46) })));
  l.dispose(); return g;
}
function accentGeometry() { // emissive piping: wing leading edges + nozzle rings
  const parts = [], wy = (z) => -0.026 + Math.tan(5 * DEG) * Math.abs(z);
  for (const s of [1, -1]) {
    const path = new THREE.CurvePath();
    path.add(new THREE.QuadraticBezierCurve3(V(1.05, wy(0.06), s * 0.06), V(0.56, wy(0.44), s * 0.44), V(0.19, wy(0.65), s * 0.65)));
    path.add(new THREE.LineCurve3(V(0.19, wy(0.65), s * 0.65), V(-1.29, wy(1.69), s * 1.69)));
    parts.push({ geom: new THREE.TubeGeometry(path, 28, 0.019, 5, false) });
  }
  const ring = new THREE.TorusGeometry(0.205, 0.012, 6, 36); ring.rotateY(Math.PI / 2);
  for (const s of [1, -1]) parts.push({ geom: ring.clone(), matrix: new THREE.Matrix4().makeTranslation(-1.955, -0.01, s * 0.46) });
  const g = mergeGeoms(parts); ring.dispose(); return g;
}
function exhaustGeometry() { // two unit cones, base at x = 0, tip at x = −1 (scale.x = plume length)
  const cone = new THREE.CylinderGeometry(0.0, 0.15, 1, 24, 1, true); cone.rotateZ(Math.PI / 2); cone.translate(-0.5, 0, 0);
  const g = mergeGeoms([1, -1].map((s) => ({ geom: cone.clone(), matrix: new THREE.Matrix4().makeTranslation(0, 0, s * 0.46) })));
  cone.dispose(); return g;
}

// ---------- RCS layout: { p: surface point, d: thrust direction }. Pairs: pitch [1,2]/[0,3], yaw [4,7]/[5,6], roll [8,11]/[9,10]
const RCS = (() => {
  const n = (x, y, z, dx, dy, dz) => ({ p: V(x, y, z), d: V(dx, dy, dz) });
  const nose = hullAt(1.45), tail = hullAt(-1.5), tailS = hullAt(-1.8);   // tail yaw pair sits behind the elevons
  const top = (s) => s.cy + s.ht * (1 - s.m), bot = (s) => s.cy - s.hb * (1 - 0.45 * s.m), tipY = -0.03 + Math.tan(5 * DEG) * 1.6;
  return [
    n(1.45, top(nose), 0, 0, 1, 0), n(1.45, bot(nose), 0, 0, -1, 0), n(-1.5, top(tail), 0, 0, 1, 0), n(-1.5, bot(tail), 0, 0, -1, 0),
    n(1.45, nose.cy, nose.w, 0, 0, 1), n(1.45, nose.cy, -nose.w, 0, 0, -1), n(-1.8, tailS.cy, tailS.w, 0, 0, 1), n(-1.8, tailS.cy, -tailS.w, 0, 0, -1),
    n(-1.5, tipY + 0.03, 1.6, 0, 1, 0), n(-1.5, tipY - 0.03, 1.6, 0, -1, 0), n(-1.5, tipY + 0.03, -1.6, 0, 1, 0), n(-1.5, tipY - 0.03, -1.6, 0, -1, 0),
  ];
})();
const PAIRS = [['pitch', 1, 2, 0, 3], ['yaw', 4, 7, 5, 6], ['roll', 8, 11, 9, 10]];
function rcsGeometry() {
  const cone = new THREE.CylinderGeometry(0.03, 0.014, 0.06, 10, 1, true), up = V(0, 1, 0), one = V(1, 1, 1);
  const g = mergeGeoms(RCS.map((n) => ({ geom: cone.clone(), matrix: new THREE.Matrix4().compose(n.p.clone().addScaledVector(n.d, 0.012), new THREE.Quaternion().setFromUnitVectors(up, n.d), one) })));
  cone.dispose(); return g;
}

// ---------- shaders --------------------------------------------------------------------------------------------
const GLOW_VS = /* glsl */`
  attribute vec3 iPos; attribute vec4 iCol; varying vec2 vUv; varying vec3 vCol;
  void main() { vUv = uv; vCol = iCol.rgb; vec4 mv = modelViewMatrix * vec4(iPos, 1.0); mv.xy += position.xy * iCol.w; gl_Position = projectionMatrix * mv; }`;
const GLOW_FS = /* glsl */`
  varying vec2 vUv; varying vec3 vCol;
  void main() { vec2 p = vUv - 0.5; float d = dot(p, p) * 4.0;
    float a = max(exp(-d * 3.5) - exp(-3.5), 0.0) / (1.0 - exp(-3.5)) + 0.7 * exp(-d * 16.0);
    gl_FragColor = vec4(vCol * a, 1.0); }`;
const EX_VS = /* glsl */`
  varying float vT; varying float vRim;
  void main() { vT = uv.y; vec3 n = normalize(normalMatrix * normal); vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vRim = abs(dot(n, normalize(-mv.xyz))); gl_Position = projectionMatrix * mv; }`;
const EX_FS = /* glsl */`
  uniform float uI; uniform vec3 uCol; varying float vT; varying float vRim;
  void main() { float fall = pow(1.0 - vT, 1.6), rim = pow(vRim, 1.4);
    float core = pow(vRim, 4.0) * pow(1.0 - vT, 3.0);
    vec3 c = mix(uCol, vec3(1.0), core * 0.85) * (fall * rim * uI);
    gl_FragColor = vec4(c, 1.0); }`;

function makeGlowCloud(n) { // n billboards, one additive draw call; iCol = (r, g, b, size)
  const g = new THREE.InstancedBufferGeometry(), q = new THREE.PlaneGeometry(1, 1);
  g.setIndex(q.index); g.setAttribute('position', q.attributes.position); g.setAttribute('uv', q.attributes.uv);
  const pos = new THREE.InstancedBufferAttribute(new Float32Array(n * 3), 3).setUsage(THREE.DynamicDrawUsage);
  const col = new THREE.InstancedBufferAttribute(new Float32Array(n * 4), 4).setUsage(THREE.DynamicDrawUsage);
  g.setAttribute('iPos', pos); g.setAttribute('iCol', col); g.instanceCount = n;
  const m = new THREE.ShaderMaterial({ vertexShader: GLOW_VS, fragmentShader: GLOW_FS, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });
  const mesh = new THREE.Mesh(g, m); mesh.frustumCulled = false; mesh.renderOrder = 2;
  return {
    mesh,
    set(i, x, y, z, size, r, gg, b, k) { pos.setXYZ(i, x, y, z); col.setXYZW(i, r * k, gg * k, b * k, size); },
    dirty() { pos.needsUpdate = col.needsUpdate = true; },
    dispose() { g.dispose(); m.dispose(); },
  };
}
function mulberry32(a) {
  return () => { a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

// Bound specular glints: a delta DirectionalLight on clearcoat / glass is unbounded in the GGX model and would
// flash the bloom pass (values of 50–1000) whenever the attitude lines up with the sun's mirror angle. Soft knee
// on the outgoing radiance: untouched below 1.0 (sun-lit white diffuse stays), asymptote 1.8 — highlights keep
// their shape and still bloom softly, but never flashbulb.
const KNEE_GLSL = `{ float apMx = max(max(outgoingLight.r, outgoingLight.g), outgoingLight.b);
  if (apMx > 1.0) outgoingLight *= (1.0 + 0.8 * (1.0 - exp(-(apMx - 1.0) / 0.8))) / apMx; }`;
function physical(params) {
  const mt = new THREE.MeshPhysicalMaterial(params);
  mt.onBeforeCompile = (sh) => { sh.fragmentShader = sh.fragmentShader.replace('#include <opaque_fragment>', KNEE_GLSL + '\n#include <opaque_fragment>'); };
  return mt;
}
const PEARL = { color: 0xf3f2ee, roughness: 0.32, metalness: 0.15, clearcoat: 1, clearcoatRoughness: 0.08 };

// ---------- shared, ref-counted geometry + static materials (a ghost copy is created near the wrap edge) --------
let SHARED = null;
function acquireShared() {
  if (SHARED) { SHARED.refs++; return SHARED; }
  const g = {}, m = {};
  g.hull = loft(hullStations(), 40, { cap: true, chamfer: 0.04, splitX: -1.45 });
  g.canopy = loft(canopyStations(), 24);
  g.airframe = airframeGeometry(); g.nozzles = nozzleGeometry(); g.accent = accentGeometry();
  const surf = buildSurfaceGeometry(); g.surfaces = surf.geom;
  g.rcs = rcsGeometry(); g.exhaust = exhaustGeometry();
  g.frag = new THREE.IcosahedronGeometry(0.17, 0); g.frag.scale(1.7, 0.35, 1.1);
  m.pearl = physical(PEARL);
  m.dark = physical({ color: 0x6b6d70, roughness: 0.5, metalness: 0.6, side: THREE.DoubleSide });
  m.glass = physical({ color: 0x06080c, roughness: 0.05, metalness: 0, clearcoat: 1, clearcoatRoughness: 0.03 });
  m.accent = new THREE.MeshBasicMaterial({ color: ACCENT.clone().multiplyScalar(1.6) });
  SHARED = { g, m, parts: surf.parts, refs: 1 };
  return SHARED;
}
function releaseShared() {
  if (!SHARED || --SHARED.refs > 0) return;
  for (const k in SHARED.g) SHARED.g[k].dispose();
  for (const k in SHARED.m) SHARED.m[k].dispose();
  SHARED = null;
}

const _q = new THREE.Quaternion(), _m = new THREE.Matrix4(), _one = V(1, 1, 1), _v = new THREE.Vector3();

export function createShip({ seed = 1 } = {}) {
  const { g, m, parts: surfParts } = acquireShared();
  const rnd = mulberry32(seed | 0);
  const group = new THREE.Group(); group.name = 'ship';
  const hull = new THREE.Group(); hull.name = 'hull';
  const meshes = { body: new THREE.Mesh(g.hull, [m.pearl, m.dark]), canopy: new THREE.Mesh(g.canopy, m.glass), wing: new THREE.Mesh(g.airframe, [m.pearl, m.dark]),
    nozzles: new THREE.Mesh(g.nozzles, m.dark), accent: new THREE.Mesh(g.accent, m.accent), rcs: new THREE.Mesh(g.rcs, m.dark) };
  for (const k in meshes) { meshes[k].name = k; hull.add(meshes[k]); }
  group.add(hull);
  const exMat = new THREE.ShaderMaterial({ uniforms: { uI: { value: 1 }, uCol: { value: V(...ICE) } }, vertexShader: EX_VS, fragmentShader: EX_FS,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });
  const exhaust = new THREE.Mesh(g.exhaust, exMat); exhaust.position.set(-1.9, -0.01, 0); exhaust.frustumCulled = false; exhaust.renderOrder = 1;
  const glow = makeGlowCloud(GLOW_N);
  const fragMat = physical({ ...PEARL, transparent: true });
  const frags = new THREE.InstancedMesh(g.frag, fragMat, FRAG_N); frags.visible = false; frags.frustumCulled = false;
  frags.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  const light = new THREE.PointLight(0x9fd0ff, 0, 12, 2); light.position.set(-2.3, 0, 0);
  exhaust.name = 'exhaust'; glow.mesh.name = 'glow'; frags.name = 'fragments'; light.name = 'engineLight';
  group.add(exhaust, glow.mesh, frags, light);

  let time = rnd() * 10, mono = 0, exploding = false, exAge = 0, air = 0;
  const ph = [rnd() * TAU, rnd() * TAU, rnd() * TAU], rcsV = new Float32Array(12), rcsT = new Float32Array(12);
  const surf = createSurfaces({ geom: g.surfaces, parts: surfParts }, [m.pearl, m.dark], rnd), vapour = createVapour();   // after the phases above: same look per seed
  group.add(surf.mesh, vapour.mesh);   // outside `hull`: the surfaces tumble away as debris when the hull is gone
  const parts = Array.from({ length: PART_N }, () => ({ p: V(0, 0, 0), v: V(0, 0, 0), life: 1, size: 0.1, warm: true, age: 0 }));
  const shards = Array.from({ length: FRAG_N }, () => ({ p: V(0, 0, 0), v: V(0, 0, 0), ax: V(0, 1, 0), rate: 1, q: new THREE.Quaternion() }));
  const rdir = (out) => { const z = rnd() * 2 - 1, a = rnd() * TAU, r = Math.sqrt(1 - z * z); return out.set(r * Math.cos(a), z, r * Math.sin(a)); };

  function clearParticles() { for (let i = 15; i < GLOW_N; i++) glow.set(i, 0, 0, 0, 0, 0, 0, 0, 0); glow.dirty(); }

  function update(dt, cmd = {}) {
    dt = clamp(Number.isFinite(dt) ? dt : 0, 0, 0.1); time += dt;
    const thr = clamp(cmd.throttle ?? 0, 0, 1), spd = clamp(cmd.speed ?? 0, 0, 30), alive = hull.visible, inkK = mono === 2 ? 0.75 : 1;
    const fl = 0.5 * Math.sin(37 * time + ph[0]) + 0.3 * Math.sin(53 * time + ph[1]) + 0.2 * Math.sin(91 * time + ph[2]);
    // engines: plume length / brightness / nozzle glow / point light all follow the throttle, with flicker
    exhaust.visible = alive;
    exhaust.scale.x = (1.2 + 2.6 * thr + 0.01 * spd) * (1 + 0.07 * fl);
    exMat.uniforms.uI.value = (0.55 + 2.4 * thr) * (1 + 0.12 * thr * fl) * inkK;
    const gk = alive ? (0.8 + 2.4 * thr) * (1 + 0.1 * fl) * inkK : 0;
    glow.set(0, -1.94, -0.01, 0.46, alive ? 0.32 + 0.4 * thr : 0, ICE[0], ICE[1], ICE[2], gk);
    glow.set(1, -1.94, -0.01, -0.46, alive ? 0.32 + 0.4 * thr : 0, ICE[0], ICE[1], ICE[2], gk);
    if (!exploding) light.intensity = alive ? (0.4 + 2.2 * thr) * (1 + 0.2 * thr * fl) : 0;
    // RCS: sign picks the nozzle pair, brightness ∝ |command|; fast rise (18 ms), quick decay (80 ms)
    rcsT.fill(0);
    if (alive) for (const [k, p0, p1, n0, n1] of PAIRS) {
      const v = clamp(Number(cmd[k]) || 0, -1, 1), a = Math.abs(v);
      if (a < 0.02) continue;
      if (v > 0) { rcsT[p0] = rcsT[p1] = a; } else { rcsT[n0] = rcsT[n1] = a; }
    }
    for (let i = 0; i < 12; i++) {
      const v = rcsV[i], nv = v + (rcsT[i] - v) * (1 - Math.exp(-dt / (rcsT[i] > v ? 0.018 : 0.08)));
      rcsV[i] = nv;
      const n = RCS[i], d = 0.05 + 0.15 * nv;
      glow.set(2 + i, n.p.x + n.d.x * d, n.p.y + n.d.y * d, n.p.z + n.d.z * d, nv > 0.003 ? 0.1 + 0.24 * nv : 0, 0.85, 0.93, 1, 2.4 * nv * inkK);
    }
    // navigation strobe: 1 Hz, 40 ms, on the tail spine
    const on = alive && (time % 1) < 0.04;
    glow.set(14, -1.72, hullTop(-1.72) + 0.03, 0, on ? 0.13 : 0, 1, 1, 1, 3.5 * inkK);
    // control surfaces (fly-by-wire mixing + actuators), thrust-vectored plumes, wingtip vapour in air. Air density:
    // cmd.air (0..1) when the app provides it, else inferred from the scene fog that atmospheric flight switches on.
    const tv = surf.update(dt, cmd, group.quaternion, spd, alive);
    exhaust.rotation.set(0, tv.yaw, tv.pitch);
    air = Number.isFinite(cmd.air) ? clamp(cmd.air, 0, 1) : air + ((group.parent && group.parent.fog ? 1 : 0) - air) * (1 - Math.exp(-dt / 2));
    vapour.update(dt, group, alive ? air * tv.vap : 0, inkK);
    // accent breathing (±10 %), toned down in mono / ink
    m.accent.color.copy(ACCENT).multiplyScalar((mono ? 1.2 : 1.6) * (1 + 0.1 * Math.sin(TAU * 0.45 * time)));
    if (exploding) stepExplosion(dt);
    glow.dirty();
  }

  function explode() {
    exploding = true; exAge = 0; hull.visible = false; rcsV.fill(0);
    for (const q of parts) {
      q.p.set((rnd() - 0.5) * 2.4, (rnd() - 0.5) * 0.5, (rnd() - 0.5) * 1.6);
      rdir(q.v).multiplyScalar(2 + 7 * Math.pow(rnd(), 1.6));
      q.life = 0.7 + 0.7 * rnd(); q.size = 0.07 + 0.16 * rnd(); q.warm = rnd() < 0.7; q.age = 0;
    }
    for (const s of shards) {
      s.p.set((rnd() - 0.5) * 2.4, (rnd() - 0.5) * 0.3, (rnd() - 0.5) * 1.4);
      rdir(s.v).multiplyScalar(1.5 + 3 * rnd()); rdir(s.ax); s.rate = 3 + 6 * rnd(); s.q.setFromAxisAngle(rdir(_v), rnd() * TAU);
    }
    frags.visible = true; fragMat.opacity = 1; surf.explode(rdir);
    light.position.set(0, 0, 0); light.color.setHex(0xffc898);
  }
  function stepExplosion(dt) {
    exAge += dt;
    const drag = Math.exp(-1.4 * dt);
    for (let i = 0; i < PART_N; i++) {
      const q = parts[i]; q.age += dt;
      const k = 1 - q.age / q.life;
      if (k <= 0) { glow.set(16 + i, 0, 0, 0, 0, 0, 0, 0, 0); continue; }
      q.p.addScaledVector(q.v, dt); q.v.multiplyScalar(drag);
      const kk = Math.pow(k, 0.7);
      glow.set(16 + i, q.p.x, q.p.y, q.p.z, q.size * (0.35 + 0.65 * k), q.warm ? 1 : 0.6, q.warm ? 0.3 + 0.55 * kk : 0.85, q.warm ? 0.08 + 0.45 * kk : 1, 3 * k);
    }
    const f = Math.max(0, 1 - exAge / 0.22);
    glow.set(15, 0, 0, 0, f > 0 ? 1.6 + 1.6 * (1 - f) : 0, 1, 0.95, 0.85, 5 * f * f * f);
    light.intensity = 25 * Math.pow(Math.max(0, 1 - exAge / 0.4), 2);
    for (let i = 0; i < FRAG_N; i++) {
      const s = shards[i]; s.p.addScaledVector(s.v, dt); s.v.multiplyScalar(drag);
      _q.setFromAxisAngle(s.ax, s.rate * dt); s.q.premultiply(_q); _m.compose(s.p, s.q, _one); frags.setMatrixAt(i, _m);
    }
    frags.instanceMatrix.needsUpdate = true;
    fragMat.opacity = 1 - clamp((exAge - 0.9) / 0.5, 0, 1);
    if (exAge >= 1.4) { exploding = false; frags.visible = false; light.intensity = 0; clearParticles(); }
  }
  function reset() {
    exploding = false; hull.visible = true; frags.visible = false; rcsV.fill(0); clearParticles(); surf.reset(); vapour.reset(); exhaust.rotation.set(0, 0, 0);
    light.position.set(-2.3, 0, 0); light.color.setHex(0x9fd0ff); light.intensity = 0;
  }
  function dispose() {
    group.removeFromParent(); glow.dispose(); exMat.dispose(); fragMat.dispose(); surf.dispose(); vapour.dispose(); releaseShared();
  }
  update(0, {});
  return { group, update, explode, reset, setMono(mode) { mono = mode | 0; }, setVisible(v) { if (v && !group.visible) { surf.resync(); vapour.reset(); } group.visible = !!v; }, dispose };
}
