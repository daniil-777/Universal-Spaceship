// Astro Pilot — shipsurfaces.js — the hero ship's moving aerodynamic parts and their flight-control laws (used by ship.js).
//
// Parts: ONE SkinnedMesh, one bone per part, two draw calls (pearl + dark); the geometry is shared by the ship and its ghost.
//   · 4 elevons (inboard + outboard per wing, Shuttle / Dream Chaser layout), 4 split-rudder panels (each canted fin's
//     rudder is a clamshell pair that also splays open as the speed brake), 1 body flap under the engine face.
//   · Every part is a swept profile with a round nose centred on its hinge line, so it turns in place inside a dark
//     cove cut into the wing / fin (ship.js cuts the coves from WING_COVE / FIN_COVE, derived from the same numbers):
//     the hinge gap stays constant and the inside of the wing is never exposed.
// Control laws — rate-command fly-by-wire in the ship's axes (+pitch = nose up, +yaw = nose left, +roll = right wing down):
//   · axis demand u = 0.8·cmd + 0.8·(cmd − ω/ωmax). ω is measured by differencing group.quaternion. The rate-error
//     term makes the surfaces lead a manoeuvre (full throw before the ship starts to turn), settle to a trim deflection
//     while it turns, and flick briefly the other way to check the rotation when the command is released.
//   · elevons (+ = trailing edge down): inboard −18°·pitch ∓ 12°·roll, outboard −12°·pitch ∓ 22°·roll, clamp ±25°.
//   · rudders: −18°·yaw + 4°·roll (aileron–rudder interconnect); braking (throttle < 0.5) splays each pair ±32°.
//   · body flap: slow pitch trim (τ 0.9 s) −10°·pitch, +4° with the speed brake. Thrust vectoring: plumes gimbal 6°.
//   · actuators: first-order lag τ = 60 ms + rate limit 90°/s (speed brake 45°/s, body flap 30°/s, TVC 60°/s), and a
//     ±0.4° fly-by-wire trim hunt.
// Wingtip vapour: two camera-facing ribbons trail from the tips when the normal load (speed × pitch rate) is high in air.
import * as THREE from 'three';

const DEG = Math.PI / 180, TAN5 = Math.tan(5 * DEG);
const clamp = (v, a, b) => Math.min(Math.max(v, a), b);
const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
const V = (x, y, z) => new THREE.Vector3(x, y, z);

// ---------- layout (model frame: nose +X, up +Y, right wing +Z; wing planform points [x, z], fin points [x, y_fin]) -----
// Wing: 0.06-thick slab (ExtrudeGeometry lids ±0.030, bevel 0.022 wide × 0.018), mid-plane y = −0.06 + tan5°·|z|,
// trailing edge (−1.66, 0.55) → (−1.74, 1.66). Hinge lines run parallel to it, W_CH ahead of the trailing-edge contour.
const W_SIG = -0.08 / 1.11, W_N = Math.hypot(1, W_SIG), W_P = [1 / W_N, -W_SIG / W_N];   // sweep dx/dz; unit ⊥ (forward)
const W_R = 0.030, W_BEV = 0.022, W_WALL = 0.012, W_CH = 0.27, W_GAP = 0.006, W_SIDE = 0.045;
const ELEVONS = [[0.706, 1.125], [1.150, 1.465]];                                          // hinge-line z spans: in, out
const W_ROOT = 0.63, ROOT_SLOPE = 0.138;   // cove root wall (hidden in the hull); the inboard end cap follows the hull flank (dz/d−x)
const W_FRONT = W_R + W_GAP + W_BEV;                                                       // cove wall contour ahead of the hinge
const wHinge = (z, off = 0) => [-1.66 + W_SIG * (z - 0.55) + W_CH * W_N + off * W_P[0], z + off * W_P[1]];
export const WING_COVE = [wHinge(ELEVONS[1][1] + W_SIDE, -W_CH), wHinge(ELEVONS[1][1] + W_SIDE, W_FRONT),
  wHinge(W_ROOT, W_FRONT), wHinge(W_ROOT, -W_CH)];                                          // right wing, tip → root order
// Fin (its own frame; mounted by FIN_MOUNT): 0.04 thick (lids ±0.020, bevel 0.014 × 0.012), trailing edge x = −1.86.
const F_R = 0.020, F_BEV = 0.014, F_WALL = 0.008, F_GAP = 0.005, F_END = 0.022, F_TE = -1.86;
const F_HB = [-1.630, 0.170], F_HT = [-1.690, 0.420];                                      // rudder hinge line, bottom → top
const F_LEN = Math.hypot(F_HT[0] - F_HB[0], F_HT[1] - F_HB[1]);
const F_H = [(F_HT[0] - F_HB[0]) / F_LEN, (F_HT[1] - F_HB[1]) / F_LEN], F_P = [F_H[1], -F_H[0]];
const F_FRONT = F_R + F_GAP + F_BEV;
const fAt = (b, s, off) => [b[0] + s * F_H[0] + off * F_P[0], b[1] + s * F_H[1] + off * F_P[1]];
const fTE = (b, s) => { const p = fAt(b, s, 0); return [F_TE, p[1] + (F_TE - p[0]) / F_P[0] * F_P[1]]; };
export const FIN_COVE = [fTE(F_HT, F_END), fAt(F_HT, F_END, F_FRONT), fAt(F_HB, -F_END, F_FRONT), fTE(F_HB, -F_END)];
export const FIN_MOUNT = (side) => new THREE.Matrix4().makeTranslation(0, 0.17, side * 0.44).multiply(new THREE.Matrix4().makeRotationX(side * 35 * DEG));
const BF = { x: -1.925, y: -0.192, half: 0.24, R: 0.018, rte: 0.008, L: 0.20 };             // body flap hinge, span, chord
const TIPS = [V(-1.70, 0.088, 1.69), V(-1.70, 0.088, -1.69)];                               // vortex cores leave the tips here

// ---------- profiles: patches { p: [[u, v]], n: [[nu, nv]], dark } in a station plane (u aft of the hinge, v "up") ------
function arc(cu, cv, r, a0, a1, segs) {
  const p = [], n = [];
  for (let i = 0; i <= segs; i++) { const a = (a0 + (a1 - a0) * i / segs) * DEG, c = Math.cos(a), s = Math.sin(a); p.push([cu + r * c, cv + r * s]); n.push([c, s]); }
  return { p, n };
}
function bevel(cu, cv, a, b, sgn, rev) { // ExtrudeGeometry's 3-step bevel: lid edge (t = 0) → side wall (t = 1)
  const p = [], n = [];
  for (let i = 0; i <= 3; i++) { const t = i * Math.PI / 6, s = Math.sin(t), c = Math.cos(t), l = Math.hypot(s / a, c / b); p.push([cu + a * s, cv + sgn * b * c]); n.push([s / a / l, sgn * c / b / l]); }
  if (rev) { p.reverse(); n.reverse(); }
  return { p, n };
}
function elevonProfile() { // round nose (dark) + the wing's own trailing-edge bevel, so it sits flush at rest
  const L = W_CH + W_BEV, e = L - W_BEV, b = W_R - W_WALL, lo = bevel(e, -W_WALL, W_BEV, b, -1, false), hi = bevel(e, W_WALL, W_BEV, b, 1, true);
  return [{ ...arc(0, 0, W_R, 90, 270, 10), dark: true }, { p: [[0, -W_R], [e, -W_R]], n: [[0, -1], [0, -1]] },
    { p: [...lo.p, ...hi.p], n: [...lo.n, ...hi.n] }, { p: [[e, W_R], [0, W_R]], n: [[0, 1], [0, 1]] }];
}
function rudderProfile(L, sgn) { // one clamshell panel (sgn = +1: the +z_fin skin); the twin's nose is 3 % smaller so splayed noses never z-fight
  const r = sgn > 0 ? F_R : 0.97 * F_R, e = L - F_BEV, bv = bevel(e, F_WALL, F_BEV, F_R - F_WALL, 1, false);
  const prof = [{ ...arc(0, 0, r, 180, 90, 5), dark: true }, { p: [[0, r], [e, F_R]], n: [[0, 1], [0, 1]] },
    { p: [...bv.p, [L, 0]], n: [...bv.n, [1, 0]] }, { p: [[L, 0], [-r, 0]], n: [[0, -1], [0, -1]], dark: true }];
  if (sgn < 0) for (const pt of prof) for (const q of [...pt.p, ...pt.n]) q[1] = -q[1];
  return prof;
}
function flapProfile() { // tapered plate, round nose and trailing edge — all engine-section dark
  const { R, rte, L } = BF, c = L - rte, len = Math.hypot(c, R - rte), nu = (R - rte) / len, nv = c / len;
  return [arc(0, 0, R, 90, 270, 8), { p: [[0, -R], [c, -rte]], n: [[nu, -nv], [nu, -nv]] }, arc(c, 0, rte, -90, 90, 6),
    { p: [[c, rte], [0, R]], n: [[nu, nv], [nu, nv]] }].map((pt) => ({ ...pt, dark: true }));
}

// ---------- skinned geometry: every part lofted between two stations on its hinge line, flat end caps ⟂ the hinge ------
export function buildSurfaceGeometry() {
  const P = [[], []], N = [[], []], B = [[], []], parts = [];
  const e1 = new THREE.Vector3(), e2 = new THREE.Vector3(), cr = new THREE.Vector3(), nsum = new THREE.Vector3(), ld = new THREE.Vector3();
  function tri(k, bone, v0, v1, v2) { // v = [position, normal]; the winding follows the normals (mirrored parts need no care)
    e1.subVectors(v1[0], v0[0]); e2.subVectors(v2[0], v0[0]); cr.crossVectors(e1, e2); nsum.copy(v0[1]).add(v1[1]).add(v2[1]);
    if (cr.dot(nsum) < 0) { const s = v1; v1 = v2; v2 = s; }
    for (const v of [v0, v1, v2]) { P[k].push(v[0].x, v[0].y, v[0].z); N[k].push(v[1].x, v[1].y, v[1].z); B[k].push(bone); }
  }
  const station = (s) => s.prof.map((pt) => pt.p.map((q, i) => [s.o.clone().addScaledVector(s.u, q[0]).addScaledVector(s.v, q[1]), s.u.clone().multiplyScalar(pt.n[i][0]).addScaledVector(s.v, pt.n[i][1])]));
  function part(kind, sA, sB, capDark = false) {
    const bone = parts.length + 1, A = station(sA), Bs = station(sB), dir = sB.o.clone().sub(sA.o).normalize();
    const chord = sB.u.clone().addScaledVector(dir, -sB.u.dot(dir)).normalize();   // a station's u may be skewed (tilted end cap)
    sA.prof.forEach((pt, k) => {
      const ra = A[k], rb = Bs[k], m = pt.dark ? 1 : 0;
      for (let i = 0; i < ra.length; i++) { // normals ⟂ the loft line (ruled surfaces between unequal profiles)
        ld.subVectors(rb[i][0], ra[i][0]).normalize();
        for (const r of [ra[i], rb[i]]) r[1].addScaledVector(ld, -r[1].dot(ld)).normalize();
      }
      for (let i = 0; i < ra.length - 1; i++) { tri(m, bone, ra[i], ra[i + 1], rb[i + 1]); tri(m, bone, ra[i], rb[i + 1], rb[i]); }
    });
    for (const [S, sgn] of [[A, -1], [Bs, 1]]) { // end caps turn in their own plane → constant side gaps
      const ring = []; for (const r of S) for (let i = 0; i < r.length - 1; i++) ring.push(r[i][0]);
      const St = sgn < 0 ? sA : sB, n = new THREE.Vector3().crossVectors(St.u, St.v).normalize(), c = ring.reduce((s, p) => s.add(p), new THREE.Vector3()).multiplyScalar(1 / ring.length);
      if (n.dot(dir) * sgn < 0) n.negate();
      for (let i = 0; i < ring.length; i++) tri(capDark ? 1 : 0, bone, [c, n], [ring[i], n], [ring[(i + 1) % ring.length], n]);
    }
    parts.push({ kind, pivot: sA.o.clone(), axis: new THREE.Vector3().crossVectors(chord, sB.v).normalize() }); // +angle: trailing edge → +v
  }
  const mirror = (s) => ({ ...s, o: s.o.clone().setZ(-s.o.z), u: s.u.clone().setZ(-s.u.z), v: s.v.clone().setZ(-s.v.z) });
  // elevons: straight prisms along the dihedral hinge line (bones 1-4: right in, right out, left in, left out)
  const d = V(W_SIG, TAN5, 1).normalize(), eu = V(-1, 0, 0).addScaledVector(d, d.x).normalize(), ev = new THREE.Vector3().crossVectors(eu, d);
  const hinge3 = (z) => { const [x] = wHinge(z); return V(x, -0.06 + TAN5 * z, z); };
  const skew = eu.clone().addScaledVector(d, (-ROOT_SLOPE * eu.x - eu.z) / (d.z + ROOT_SLOPE * d.x));   // in-plane u whose planform slope is ROOT_SLOPE
  const wingSt = ELEVONS.map(([z0, z1], k) => [{ o: hinge3(z0), u: k ? eu : skew, v: ev, prof: elevonProfile() }, { o: hinge3(z1), u: eu, v: ev, prof: elevonProfile() }]);
  for (const [a, b] of wingSt) part('elevon', a, b);
  for (const [a, b] of wingSt) part('elevon', mirror(a), mirror(b));
  // split rudders (bones 5-8: right +z_fin, right −z_fin, left +z_fin, left −z_fin)
  for (const side of [1, -1]) {
    const M = FIN_MOUNT(side), R = new THREE.Matrix3().setFromMatrix4(M), u = V(-F_P[0], -F_P[1], 0).applyMatrix3(R), v = V(0, 0, 1).applyMatrix3(R);
    const st = (h, sgn) => { const L = (F_TE - F_BEV - h[0]) / -F_P[0]; return { o: V(h[0], h[1], 0).applyMatrix4(M), u, v, prof: rudderProfile(L, sgn) }; };
    for (const sgn of [1, -1]) part('rudder', st(F_HB, sgn), st(F_HT, sgn));
  }
  // body flap (bone 9)
  part('flap', { o: V(BF.x, BF.y, -BF.half), u: V(-1, 0, 0), v: V(0, 1, 0), prof: flapProfile() }, { o: V(BF.x, BF.y, BF.half), u: V(-1, 0, 0), v: V(0, 1, 0), prof: flapProfile() }, true);
  const g = new THREE.BufferGeometry(), n0 = P[0].length / 3, n = n0 + P[1].length / 3;
  g.setAttribute('position', new THREE.Float32BufferAttribute(P[0].concat(P[1]), 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute(N[0].concat(N[1]), 3));
  const si = new Uint16Array(n * 4), sw = new Float32Array(n * 4);
  B[0].concat(B[1]).forEach((bn, i) => { si[i * 4] = bn; sw[i * 4] = 1; });
  g.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(si, 4)); g.setAttribute('skinWeight', new THREE.Float32BufferAttribute(sw, 4));
  g.addGroup(0, n0, 0); g.addGroup(n0, n - n0, 1); g.computeBoundingSphere();
  return { geom: g, parts };
}

// Split a non-indexed extrusion into [pearl, dark]: side-wall facets whose three corners hug a cove polyline become the
// dark hinge cove. axis = the slab's thickness axis (1: wing, planform x/z; 2: fin, planform x/y).
export function splitCove(g, polys, band, axis) {
  const pos = g.attributes.position.array, nor = g.attributes.normal.array, out = [[[], []], [[], []]], j = axis === 1 ? 2 : 1;
  const segD = (x, y, a, b) => { const dx = b[0] - a[0], dy = b[1] - a[1], t = clamp(((x - a[0]) * dx + (y - a[1]) * dy) / (dx * dx + dy * dy), 0, 1); return Math.hypot(x - a[0] - t * dx, y - a[1] - t * dy); };
  const near = (x, y) => polys.some((pl) => pl.some((p, i) => i > 0 && segD(x, y, pl[i - 1], p) < band));
  for (let t = 0; t < pos.length / 9; t++) {
    let dark = Math.abs(nor[t * 9 + axis] + nor[t * 9 + 3 + axis] + nor[t * 9 + 6 + axis]) / 3 < 0.995;
    for (let k = 0; k < 3 && dark; k++) dark = near(pos[t * 9 + k * 3], pos[t * 9 + k * 3 + j]);
    const o = out[dark ? 1 : 0];
    for (let k = 0; k < 9; k++) { o[0].push(pos[t * 9 + k]); o[1].push(nor[t * 9 + k]); }
  }
  g.dispose();
  return out.map(([p, n]) => { const r = new THREE.BufferGeometry(); r.setAttribute('position', new THREE.Float32BufferAttribute(p, 3)); r.setAttribute('normal', new THREE.Float32BufferAttribute(n, 3)); return r; });
}

// ---------- per-ship controller: fly-by-wire mixing, actuators, debris ------------------------------------------------
const RATE_MAX = [2.0, 1.2, 1.6];                          // = ENV.ship.rateMax [roll, yaw, pitch] rad/s (env.js)
const K_FF = 0.8, K_LEAD = 0.8, TAU = 0.06;
const LIM = [25, 25, 25, 25, 45, 45, 45, 45, 14], RATE = [90, 90, 90, 90, 90, 90, 90, 90, 30], SIGN = [-1, -1, -1, -1, 1, 1, 1, 1, -1];
const _dq = new THREE.Quaternion(), _w = new THREE.Vector3(), _q = new THREE.Quaternion();

export function createSurfaces(shared, materials, rnd) {
  const { geom, parts } = shared, n = parts.length;
  const mesh = new THREE.SkinnedMesh(geom, materials); mesh.name = 'surfaces'; mesh.userData.parts = parts;
  mesh.boundingSphere = geom.boundingSphere.clone(); mesh.boundingSphere.radius += 0.45;   // rest pose + the largest throw (culled like the hull)
  const root = new THREE.Bone(), bones = [root];
  for (const p of parts) { const b = new THREE.Bone(); b.position.copy(p.pivot); root.add(b); bones.push(b); }
  mesh.add(root); mesh.updateMatrixWorld(true); mesh.bind(new THREE.Skeleton(bones));
  const cur = new Float32Array(n), tgt = new Float32Array(n), w = new THREE.Vector3(), qPrev = new THREE.Quaternion(), cPrev = new Float64Array(4);
  const ph = [rnd() * 6.3, rnd() * 6.3, rnd() * 6.3, rnd() * 6.3];
  const deb = parts.map(() => ({ p: new THREE.Vector3(), v: new THREE.Vector3(), ax: new THREE.Vector3(), q: new THREE.Quaternion(), rate: 0 }));
  let hasPrev = false, fresh = true, snap = true, sb = 0, trim = 0, tvP = 0, tvY = 0, time = 0, exploding = false, age = 0;
  const act = (c, t, rate, dt) => c + clamp((t - c) * (1 - Math.exp(-dt / TAU)), -rate * dt, rate * dt);   // lag + rate limit
  const tv = { pitch: 0, yaw: 0, load: 0, vap: 0 };

  function update(dt, cmd, q, speed, alive) {
    if (exploding) stepDebris(dt);
    if (exploding || !alive) { tv.load = tv.vap = 0; return tv; }
    const cmdSame = cPrev[0] === cmd.pitch && cPrev[1] === cmd.yaw && cPrev[2] === cmd.roll && cPrev[3] === cmd.throttle;
    cPrev[0] = cmd.pitch; cPrev[1] = cmd.yaw; cPrev[2] = cmd.roll; cPrev[3] = cmd.throttle;
    if (!snap && hasPrev && cmdSame && qPrev.equals(q)) return tv;                     // sim paused (the app still ticks the visuals): hold
    time += dt;
    if (hasPrev && dt > 1e-4) { // body rates from the attitude the app just set (body frame: x roll, y yaw, z pitch)
      _dq.copy(qPrev).invert().multiply(q); if (_dq.w < 0) _dq.set(-_dq.x, -_dq.y, -_dq.z, -_dq.w);
      const s = Math.hypot(_dq.x, _dq.y, _dq.z), k = s > 1e-9 ? 2 * Math.atan2(s, _dq.w) / s / dt : 2 / dt;
      _w.set(_dq.x * k, _dq.y * k, _dq.z * k); if (_w.length() > 12) _w.set(0, 0, 0);   // a respawn / seam jump, not a rotation
      if (fresh) { w.copy(_w); fresh = false; } else w.lerp(_w, 1 - Math.exp(-dt / 0.05));
    }
    qPrev.copy(q); hasPrev = true;
    const cR = clamp(Number(cmd.roll) || 0, -1, 1), cY = clamp(Number(cmd.yaw) || 0, -1, 1), cP = clamp(Number(cmd.pitch) || 0, -1, 1);
    const lead = fresh ? 0 : K_LEAD;                                                    // no measured rate yet → assume steady state
    const uR = K_FF * cR + lead * (cR - w.x / RATE_MAX[0]), uY = K_FF * cY + lead * (cY - w.y / RATE_MAX[1]), uP = K_FF * cP + lead * (cP - w.z / RATE_MAX[2]);
    const thr = Number.isFinite(cmd.throttle) ? cmd.throttle : 0.5;
    const hunt = 0.4 * Math.sin(0.9 * time + ph[0]) * Math.sin(0.31 * time + ph[1]), huntR = 0.3 * Math.sin(0.7 * time + ph[2]) * Math.sin(0.23 * time + ph[3]);
    tgt[0] = -18 * uP - 12 * uR + hunt; tgt[1] = -12 * uP - 22 * uR + hunt;            // right elevons (+ = trailing edge down)
    tgt[2] = -18 * uP + 12 * uR + hunt; tgt[3] = -12 * uP + 22 * uR + hunt;            // left elevons
    const brake = smooth(0.03, 0.4, 0.47 - thr), sbT = 32 * brake;
    sb = snap ? sbT : sb + clamp(sbT - sb, -45 * dt, 45 * dt);
    const rud = -18 * uY + 4 * uR + huntR;                                              // + = trailing edge to the right
    tgt[4] = rud + sb; tgt[5] = rud - sb; tgt[6] = rud + sb; tgt[7] = rud - sb;        // clamshell pairs splay ±sb
    trim += (cP - trim) * (snap ? 1 : 1 - Math.exp(-dt / 0.9));
    tgt[8] = clamp(-10 * trim + 4 * brake, -10, 14);
    for (let i = 0; i < n; i++) {
      const T = clamp(tgt[i], -LIM[i], LIM[i]);
      cur[i] = snap ? T : act(cur[i], T, RATE[i], dt);
      bones[i + 1].quaternion.setFromAxisAngle(parts[i].axis, SIGN[i] * cur[i] * DEG);
    }
    const pT = 6 * clamp(uP, -1, 1), yT = 6 * clamp(uY, -1, 1);                     // thrust vectoring (plume up on pitch-up)
    tvP = snap ? pT : act(tvP, pT, 60, dt); tvY = snap ? yT : act(tvY, yT, 60, dt);
    snap = false;
    tv.pitch = -tvP * DEG; tv.yaw = -tvY * DEG;                                         // exhaust.rotation z / y
    tv.load = (Number(speed) || 0) * Math.hypot(w.z, 0.5 * w.y);                        // ≈ normal acceleration
    tv.vap = smooth(9, 17, tv.load);                                                    // wingtip condensation onset
    return tv;
  }
  function explode(rdir) { // the real surfaces tumble away with the fragments
    exploding = true; age = 0; mesh.visible = true; mesh.frustumCulled = false;
    parts.forEach((p, i) => {
      const s = deb[i], b = bones[i + 1];
      s.p.copy(b.position); s.q.copy(b.quaternion); rdir(s.v).multiplyScalar(1.5 + 3 * rnd()).addScaledVector(p.pivot, 1.4);
      rdir(s.ax); s.rate = 4 + 8 * rnd();
    });
  }
  function stepDebris(dt) {
    age += dt;
    const drag = Math.exp(-1.4 * dt), k = Math.max(1e-3, 1 - clamp((age - 0.8) / 0.5, 0, 1));
    parts.forEach((p, i) => {
      const s = deb[i], b = bones[i + 1];
      s.p.addScaledVector(s.v, dt); s.v.multiplyScalar(drag); _q.setFromAxisAngle(s.ax, s.rate * dt); s.q.premultiply(_q);
      b.position.copy(s.p); b.quaternion.copy(s.q); b.scale.setScalar(k);
    });
    if (age >= 1.3) { exploding = false; mesh.visible = false; }
  }
  function reset() {
    exploding = false; mesh.visible = true; mesh.frustumCulled = true; resync(); sb = 0; trim = 0; tvP = tvY = 0; cur.fill(0);
    parts.forEach((p, i) => { const b = bones[i + 1]; b.position.copy(p.pivot); b.quaternion.identity(); b.scale.setScalar(1); });
  }
  function resync() { snap = true; hasPrev = false; fresh = true; w.set(0, 0, 0); } // next update jumps straight to the commanded state
  return { mesh, update, explode, reset, resync, get deflections() { return Array.from(cur); }, dispose() { mesh.skeleton.dispose(); } };
}

// ---------- wingtip vapour: two camera-facing ribbons (one draw call, only while visible) -------------------------------
const VAP_N = 22, VAP_DT = 0.02, VAP_LIFE = 0.45;
const VAP_VS = /* glsl */`
  attribute float aSide; attribute float aK; attribute float aAge; attribute vec3 aDir; varying float vK; varying float vS;
  void main() { vec4 mv = modelViewMatrix * vec4(position, 1.0); vec3 t = mat3(modelViewMatrix) * aDir;
    vec3 s = cross(t, normalize(-mv.xyz)); float l = length(s);
    mv.xyz += (l > 1e-6 ? s / l : vec3(0.0)) * aSide * (0.016 + 0.2 * aAge);
    vK = aK; vS = aSide; gl_Position = projectionMatrix * mv; }`;
const VAP_FS = /* glsl */`
  uniform float uI; varying float vK; varying float vS;
  void main() { float e = 1.0 - vS * vS; gl_FragColor = vec4(0.93, 0.96, 1.0, clamp(vK * e * uI, 0.0, 1.0)); }`;

export function createVapour() {
  const P = VAP_N + 1, nv = 2 * P * 2, g = new THREE.BufferGeometry();
  const pos = new Float32Array(nv * 3), dir = new Float32Array(nv * 3), kk = new Float32Array(nv), ag = new Float32Array(nv), side = new Float32Array(nv), idx = [];
  for (let t = 0; t < 2; t++) for (let i = 0; i < P; i++) {
    const v = (t * P + i) * 2; side[v] = -1; side[v + 1] = 1;
    if (i < P - 1) idx.push(v, v + 1, v + 2, v + 1, v + 3, v + 2);
  }
  const attr = (a, s) => new THREE.BufferAttribute(a, s).setUsage(THREE.DynamicDrawUsage);
  const aPos = attr(pos, 3), aDir = attr(dir, 3), aK = attr(kk, 1), aAge = attr(ag, 1);
  g.setAttribute('position', aPos); g.setAttribute('aDir', aDir); g.setAttribute('aK', aK); g.setAttribute('aAge', aAge);
  g.setAttribute('aSide', new THREE.BufferAttribute(side, 1)); g.setIndex(idx);
  const mat = new THREE.ShaderMaterial({ uniforms: { uI: { value: 1 } }, vertexShader: VAP_VS, fragmentShader: VAP_FS, transparent: true, depthWrite: false });
  const mesh = new THREE.Mesh(g, mat); mesh.name = 'vapour'; mesh.frustumCulled = false; mesh.visible = false; mesh.renderOrder = 3;
  const hp = [new Float32Array(VAP_N * 3), new Float32Array(VAP_N * 3)], hk = [new Float32Array(VAP_N), new Float32Array(VAP_N)], ha = [new Float32Array(VAP_N), new Float32Array(VAP_N)];
  let count = 0, head = 0, acc = 0;
  const inv = new THREE.Matrix4(), tw = [new THREE.Vector3(), new THREE.Vector3()], a = new THREE.Vector3(), b = new THREE.Vector3();

  function update(dt, group, strength, inkK = 1) {
    if (strength <= 0.01 && !mesh.visible) { count = 0; return; }                         // nothing formed, nothing left to fade
    group.updateMatrixWorld(); inv.copy(group.matrixWorld).invert();
    for (let t = 0; t < 2; t++) tw[t].copy(TIPS[t]).applyMatrix4(group.matrixWorld);
    if (count && Math.hypot(tw[0].x - hp[0][head * 3], tw[0].y - hp[0][head * 3 + 1], tw[0].z - hp[0][head * 3 + 2]) > 6) count = 0;   // seam wrap / respawn
    for (let t = 0; t < 2; t++) for (let i = 0; i < VAP_N; i++) ha[t][i] += dt;
    acc += dt;
    if (acc >= VAP_DT || !count) {
      acc = count ? acc % VAP_DT : 0; head = (head + 1) % VAP_N; count = Math.min(count + 1, VAP_N);
      for (let t = 0; t < 2; t++) { const h = hp[t]; h[head * 3] = tw[t].x; h[head * 3 + 1] = tw[t].y; h[head * 3 + 2] = tw[t].z; hk[t][head] = strength; ha[t][head] = 0; }
    }
    let any = strength > 0.01;
    for (let t = 0; t < 2; t++) for (let i = 0; i < P; i++) {
      const j = (head - i + 1 + VAP_N) % VAP_N, live = i === 0 || i <= count;           // i = 0: the tip itself, then history
      if (i === 0) a.copy(TIPS[t]); else a.set(hp[t][j * 3], hp[t][j * 3 + 1], hp[t][j * 3 + 2]).applyMatrix4(inv);
      const age = i === 0 ? 0 : ha[t][j], k0 = i === 0 ? strength : hk[t][j], life = 1 - age / VAP_LIFE;
      const k = live && life > 0 ? k0 * life * life * smooth(0, 0.03, age + (i === 0 ? 0 : 0.01)) : 0;
      if (k > 0.01) any = true;
      const jn = (j - 1 + VAP_N) % VAP_N;                                                   // direction toward the older neighbour
      const o = (i === 0 ? head : jn) * 3;
      if (i < count) b.set(hp[t][o], hp[t][o + 1], hp[t][o + 2]).applyMatrix4(inv).sub(a); else b.set(-1, 0, 0);
      if (b.lengthSq() < 1e-8) b.set(-1, 0, 0);
      for (let s = 0; s < 2; s++) { const v = (t * P + i) * 2 + s; pos[v * 3] = a.x; pos[v * 3 + 1] = a.y; pos[v * 3 + 2] = a.z; dir[v * 3] = b.x; dir[v * 3 + 1] = b.y; dir[v * 3 + 2] = b.z; kk[v] = k; ag[v] = age; }
    }
    mesh.visible = any; mat.uniforms.uI.value = 0.55 * inkK;
    if (any) aPos.needsUpdate = aDir.needsUpdate = aK.needsUpdate = aAge.needsUpdate = true;
  }
  return { mesh, update, reset() { count = 0; acc = 0; mesh.visible = false; }, dispose() { g.dispose(); mat.dispose(); } };
}
