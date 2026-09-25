// Bird flocks for atmospheric flight — one instanced draw call for every bird. Procedural silhouette (fusiform body, neck and
// head, fanned tail, cambered wings whose trailing edge is slotted into primaries) morphed per instance between a goose/crane
// (wingspan 0.45) and a starling (0.24). The wing stroke runs in the vertex shader: fast downstroke, slower flexed upstroke,
// the hand lagging the arm, pronation/supination twist, and glides from a slow per-bird envelope (large birds glide more).
// Large birds fly a wobbling V with staggered wingbeats, small birds a coherent tilting swirl; every bird banks into its turn.
import * as THREE from 'three';

const VERT = /* glsl */`attribute vec3 aAlt; attribute float aWing, aChord; attribute vec4 aInst;   // aInst: stroke phase, frequency, flap amplitude (0 = glide), type
  varying vec3 vN, vP; varying float vSpan, vType;
  float stroke(float th) { float u = fract(th * 0.15915494), d = u < 0.38 ? 0.5 + 0.5 * cos(3.14159265 * u / 0.38) : 0.5 - 0.5 * cos(3.14159265 * (u - 0.38) / 0.62);
    return mix(-0.6, 0.85, d); }                                                                  // dihedral -34° … +49°; the downstroke takes 38% of the beat
  void main(){
    vec3 p = mix(position, aAlt, aInst.w), n = normal; float s = abs(aWing), sg = sign(aWing), th = aInst.x, flap = aInst.z, u = fract(th * 0.15915494);
    float a = stroke(th), lag = stroke(th - 0.7) - a;                                             // the hand trails the arm by 0.7 rad of the cycle (≤ ~20° of bend)
    float a1 = mix(0.12, a, flap), a2 = a1 + mix(-0.15, 0.55 * lag, flap);                        // gliding: a shallow dihedral with the hand drooped
    float fold = flap * (u > 0.38 ? sin(3.14159265 * (u - 0.38) / 0.62) : 0.0);                    // upstroke: the hand flexes and sweeps back
    if (s > 0.0) {
      float z = abs(p.z), W = 0.22, arm = min(z, W), hand = max(z - W, 0.0) * (1.0 - 0.3 * fold), ang = z > W ? a2 : a1;
      float tw = (0.35 * lag * flap - 0.1 * fold) * s * s, y = p.y + (aChord - 0.3) * 0.14 * sin(tw);   // twist: leading edge down on the downstroke
      p = vec3(p.x - 0.09 * fold * hand / 0.28 - 0.03 * fold * s, arm * sin(a1) + hand * sin(a2) + y * cos(ang), sg * (arm * cos(a1) + hand * cos(a2) - y * sin(ang)));
      n = normalize(vec3(sin(tw), cos(ang), -sg * sin(ang)));
    } else p.y += 0.012 * flap * (0.3 - a);                                                       // the body bobs against the stroke
    vec4 w = modelMatrix * instanceMatrix * vec4(p, 1.0); vP = w.xyz; vN = normalize(mat3(modelMatrix) * (mat3(instanceMatrix) * n)); vSpan = s; vType = aInst.w;
    gl_Position = projectionMatrix * viewMatrix * w; }`;
const FRAG = /* glsl */`uniform vec3 uSun, uHaze; varying vec3 vN, vP; varying float vSpan, vType;
  void main(){
    vec3 V = normalize(cameraPosition - vP), N = normalize(vN); if (dot(N, V) < 0.0) N = -N; float dist = length(cameraPosition - vP);
    float under = smoothstep(0.35, -0.35, N.y);                                                  // pale breast and underwing, dark back
    vec3 alb = mix(mix(vec3(0.11, 0.09, 0.08), vec3(0.06, 0.06, 0.07), vType), mix(vec3(0.42, 0.40, 0.37), vec3(0.15, 0.14, 0.15), vType), under);
    alb *= 1.0 - 0.55 * (1.0 - vType) * smoothstep(0.62, 0.9, vSpan);                             // black primaries on the big birds
    float ndl = max(dot(N, uSun), 0.0), ndv = max(dot(N, V), 0.0);
    vec3 sunCol = vec3(1.0, 0.95, 0.86), skyCol = vec3(0.45, 0.58, 0.82) * 0.62, bounce = vec3(0.34, 0.30, 0.24) * 0.35;
    vec3 col = alb * (sunCol * ndl + skyCol * (0.55 + 0.45 * N.y) + bounce * (1.0 - N.y));
    col += sunCol * 0.14 * pow(1.0 - ndv, 4.0) * smoothstep(-0.2, 0.8, dot(-V, uSun));            // faint sun rim when looking toward the sun
    col += vType * vec3(0.05, 0.09, 0.07) * pow(max(dot(reflect(-V, N), uSun), 0.0), 8.0);         // starling sheen
    col = mix(col, uHaze, 0.85 * (1.0 - exp(-dist / 500.0)));
    gl_FragColor = vec4(col, 1.0); }`;

function shape(big) {                                       // one size class → flat vertex arrays; same triangle order for both classes so the shader can blend them
  const R = big ? [[0.44, 0.04, 0.003, 0.003], [0.41, 0.04, 0.013, 0.013], [0.36, 0.041, 0.019, 0.017], [0.30, 0.032, 0.011, 0.011], [0.20, 0.012, 0.017, 0.016], [0.11, 0, 0.032, 0.034], [0, 0, 0.036, 0.038], [-0.10, 0.004, 0.028, 0.03], [-0.19, 0.01, 0.012, 0.028], [-0.31, 0.015, 0.003, 0.06]]
    : [[0.26, 0.02, 0.003, 0.003], [0.24, 0.02, 0.02, 0.02], [0.20, 0.022, 0.028, 0.026], [0.17, 0.018, 0.026, 0.024], [0.13, 0.01, 0.03, 0.03], [0.07, 0, 0.042, 0.044], [0, 0, 0.046, 0.048], [-0.08, 0.003, 0.034, 0.036], [-0.15, 0.008, 0.014, 0.03], [-0.24, 0.012, 0.003, 0.05]];   // body stations nose → tail: x, y, ry, rz
  const SP = [0.07, 0.22, 0.44, 0.58, 0.68, 0.76, 0.82, 0.88, 0.93, 0.97, 1.0];                    // wing stations (span fraction; the wrist at 0.44)
  const LE = big ? [0.115, 0.11, 0.10, 0.09, 0.075, 0.06, 0.045, 0.03, 0.012, -0.01, -0.045] : [0.13, 0.125, 0.11, 0.09, 0.07, 0.05, 0.03, 0.01, -0.01, -0.03, -0.06];
  const TE = big ? [-0.06, -0.075, -0.08, -0.075, -0.065, -0.025, -0.08, -0.03, -0.085, -0.04, -0.06] : [-0.10, -0.11, -0.10, -0.09, -0.075, -0.065, -0.055, -0.045, -0.04, -0.038, -0.062];   // slotted primaries vs a pointed tip
  const camb = big ? 0.014 : 0.012, yw = big ? 0.02 : 0.025, pos = [], nrm = [], wing = [], chord = [];
  const V = (p, n, w, c) => { pos.push(...p); nrm.push(...n); wing.push(w); chord.push(c); };
  const ring = (j, k) => { const [x, y, ry, rz] = R[j], a = (k % 6 + 0.5) * Math.PI / 3; return [[x, y + ry * Math.cos(a), rz * Math.sin(a)], [0, Math.cos(a), Math.sin(a)], 0, 0.5]; };
  for (let j = 0; j < R.length - 1; j++) for (let k = 0; k < 6; k++) { const a = ring(j, k), b = ring(j + 1, k), c = ring(j + 1, k + 1), d = ring(j, k + 1); for (const q of [a, b, c, a, c, d]) V(...q); }
  for (const sg of [1, -1]) {
    const W = (j, r) => { const s = SP[j], x = r === 0 ? LE[j] : r === 2 ? TE[j] : 0.5 * (LE[j] + TE[j]); return [[x, yw + (r === 1 ? camb * (1 - 0.6 * s) : 0), sg * 0.5 * s], [0, 1, 0], sg * s, r * 0.5]; };
    for (let j = 0; j < SP.length - 1; j++) for (let r = 0; r < 2; r++) { const a = W(j, r), b = W(j + 1, r), c = W(j + 1, r + 1), d = W(j, r + 1); for (const q of [a, b, c, a, c, d]) V(...q); }
  }
  return { pos, nrm, wing, chord };
}
function birdGeometry() {                                   // 188 triangles per bird, non-indexed
  const A = shape(true), B = shape(false), g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(A.pos, 3)); g.setAttribute('aAlt', new THREE.Float32BufferAttribute(B.pos, 3)); g.setAttribute('normal', new THREE.Float32BufferAttribute(A.nrm, 3));
  g.setAttribute('aWing', new THREE.Float32BufferAttribute(A.wing, 1)); g.setAttribute('aChord', new THREE.Float32BufferAttribute(A.chord, 1)); return g;
}

const H = (n) => { const x = Math.sin(n * 12.9898 + 78.233) * 43758.5453; return x - Math.floor(x); };
const ss = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
const TAU = 6.2831853, G = 3;                              // g in scene units (1 unit ≈ 3.3 m)

export function createBirds(scene, { max = 8, perFlock = 14 } = {}) {
  const N = max * perFlock, geo = birdGeometry(), inst = new Float32Array(N * 4);
  geo.setAttribute('aInst', new THREE.InstancedBufferAttribute(inst, 4).setUsage(THREE.DynamicDrawUsage));
  const mat = new THREE.ShaderMaterial({ vertexShader: VERT, fragmentShader: FRAG, side: THREE.DoubleSide,
    uniforms: { uSun: { value: new THREE.Vector3(0.6, 0.35, -0.7).normalize() }, uHaze: { value: new THREE.Color(0.62, 0.74, 0.9) } } });
  const mesh = new THREE.InstancedMesh(geo, mat, N); mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage); mesh.count = 0; mesh.frustumCulled = false; scene.add(mesh);
  const slots = Array.from({ length: max }, () => ({ gen: -1, hd: 0, bank: 0, spc: 0, cosO: 1, sinO: 0, asym: 1, b: new Float32Array(perFlock * 10) }));
  const m = new THREE.Matrix4(), o0 = [0, 0, 0], o1 = [0, 0, 0], o2 = [0, 0, 0], kMax = Math.max(2, perFlock >> 1);          // birds per arm of the V
  function reseed(S, f, j) {                                // per-bird parameters from a hash of (flock, bird, generation): phase, frequency, glide period/threshold/phase, size, swirl angle/rate/radius/height
    S.gen = f.gen; S.hd = Math.atan2(f.v[2], f.v[0]); S.bank = 0; const big = f.type === 0, b = S.b, base = 131 * f.gen + 17 * j, Om = 1.2 + 0.7 * H(base), open = 0.45 + 0.25 * H(base + 1);
    S.spc = 1.2 * f.r / kMax; S.cosO = Math.cos(open); S.sinO = Math.sin(open); S.asym = 0.8 + 0.35 * H(base + 2);
    for (let i = 0; i < perFlock; i++) {
      const n = base + 7 * i + 3, o = i * 10;
      b[o] = TAU * H(n); b[o + 1] = TAU * (big ? 2.5 + 1.0 * H(n + 1) : 8.0 + 3.0 * H(n + 1)); b[o + 2] = 5 + 9 * H(n + 2);
      b[o + 3] = -Math.cos(Math.PI * (big ? 0.3 + 0.35 * H(n + 3) : 0.08 + 0.2 * H(n + 3))); b[o + 4] = TAU * H(n + 4);   // glide duty: 30–65% for the big birds, 8–28% for the small
      b[o + 5] = (big ? 0.45 : 0.24) * (0.88 + 0.24 * H(n + 5)); b[o + 6] = TAU * H(n + 6); b[o + 7] = Om * (0.95 + 0.1 * H(n + 7)); b[o + 8] = 0.35 + 0.3 * H(n + 8); b[o + 9] = 0.5 * (H(n + 9) - 0.5);
    }
  }
  function swirl(f, b, o, t, out) {                         // starling swirl: an undulating ribbon on a precessing, tilting orbit with a travelling ripple — neighbours move together
    const a = b[o + 6] + b[o + 7] * t, rho = f.r * b[o + 8] * (1 + 0.18 * Math.sin(2 * a - 1.4 * t + f.phase));
    const lx = rho * Math.cos(a), ly = rho * (0.4 * Math.sin(a + 0.6 * t + f.phase) + b[o + 9]), lz = rho * Math.sin(a), psi = 0.35 * t + f.phase, chi = 0.7 * Math.sin(0.27 * t + 0.5 * f.phase);
    const cx = Math.cos(chi), sx = Math.sin(chi), cy = Math.cos(psi), sy = Math.sin(psi), y1 = ly * cx - lz * sx, z1 = ly * sx + lz * cx;
    out[0] = lx * cy + z1 * sy; out[1] = y1; out[2] = -lx * sy + z1 * cy;
  }
  function pose(k, px, py, pz, fx, fy, fz, roll, sc) {     // instance matrix: nose along the velocity, rolled about it
    let l = Math.hypot(fx, fy, fz) || 1; fx /= l; fy /= l; fz /= l;
    let rx = -fz, rz = fx; l = Math.hypot(rx, rz); if (l < 1e-4) { rx = 0; rz = 1; l = 1; } rx /= l; rz /= l;   // horizontal right vector (any, if flying straight up/down)
    const ux = -rz * fy, uy = rz * fx - rx * fz, uz = rx * fy, c = Math.cos(roll), s = Math.sin(roll);
    m.set(fx * sc, (ux * c + rx * s) * sc, (rx * c - ux * s) * sc, px, fy * sc, uy * c * sc, -uy * s * sc, py, fz * sc, (uz * c + rz * s) * sc, (rz * c - uz * s) * sc, pz, 0, 0, 0, 1);
    mesh.setMatrixAt(k, m);
  }
  return {
    mesh,
    update(list, dt, time, sunDir) {                        // list: [{ p, v, r, type, phase, gen }] in render space; time in seconds
      if (sunDir) mat.uniforms.uSun.value.copy(sunDir).normalize();
      const h = Math.min(0.1, Math.max(1e-3, dt || 0)); let k = 0;
      for (let j = 0; j < Math.min(max, list.length); j++) {
        const f = list[j], S = slots[j], b = S.b, big = f.type === 0, vx = f.v[0], vy = f.v[1], vz = f.v[2], sp = Math.hypot(vx, vz) || 1e-3, hd = Math.atan2(vz, vx);
        if (S.gen !== f.gen) reseed(S, f, j);
        let dh = hd - S.hd; dh -= TAU * Math.round(dh / TAU); S.hd = hd;                          // flock yaw rate → the whole V banks into the turn
        S.bank += (Math.max(-0.9, Math.min(0.9, Math.atan(sp * dh / h / G))) - S.bank) * Math.min(1, h * 3);
        const ca = Math.cos(hd), sa = Math.sin(hd);
        for (let i = 0; i < perFlock; i++, k++) {
          const o = i * 10, flap = ss(b[o + 3] - 0.15, b[o + 3] + 0.15, Math.sin(TAU * time / b[o + 2] + b[o + 4]));
          let px, py, pz, fx, fy, fz, roll;
          if (big) {                                        // V formation: leader ahead of the centre, arms trailing at the flock's opening angle, each bird drifting a little
            const kk = (i + 1) >> 1, side = i & 1 ? -1 : 1, d = kk * S.spc, w1 = time * (0.5 + 0.4 * b[o + 8]) + b[o + 6], w2 = time * 0.4 + b[o + 4];
            const bx = 0.55 * f.r - d * S.cosO + 0.05 * f.r * Math.sin(w1), by = 0.05 * f.r * Math.sin(w2) + 0.02 * Math.sin(1.7 * w1), bz = side * d * S.sinO * (side > 0 ? S.asym : 1) + 0.04 * f.r * Math.sin(0.8 * w1 + 1);
            px = f.p[0] + ca * bx - sa * bz; py = f.p[1] + by; pz = f.p[2] + sa * bx + ca * bz;
            const yaw = hd + 0.05 * Math.sin(w1); fx = Math.cos(yaw) * sp; fy = vy + 0.1 * Math.cos(w2); fz = Math.sin(yaw) * sp; roll = S.bank + 0.06 * Math.sin(1.3 * w1);
          } else {                                          // swarm: position, velocity and acceleration of the swirl by finite differences → heading and bank
            swirl(f, b, o, time - 0.03, o0); swirl(f, b, o, time, o1); swirl(f, b, o, time + 0.03, o2);
            px = f.p[0] + o1[0]; py = f.p[1] + o1[1]; pz = f.p[2] + o1[2];
            fx = vx + (o2[0] - o0[0]) / 0.06; fy = vy + (o2[1] - o0[1]) / 0.06; fz = vz + (o2[2] - o0[2]) / 0.06;
            const ax = (o2[0] - 2 * o1[0] + o0[0]) / 9e-4, ay = (o2[1] - 2 * o1[1] + o0[1]) / 9e-4, az = (o2[2] - 2 * o1[2] + o0[2]) / 9e-4, l = Math.hypot(fx, fz) || 1e-3;
            roll = Math.max(-1.25, Math.min(1.25, Math.atan2((ax * -fz + az * fx) / l, Math.max(1, G + ay))));
          }
          pose(k, px, py, pz, fx, fy, fz, roll, b[o + 5]);
          inst[k * 4] = (b[o] + time * b[o + 1]) % TAU; inst[k * 4 + 1] = b[o + 1]; inst[k * 4 + 2] = flap; inst[k * 4 + 3] = big ? 0 : 1;   // phase folded on the CPU: no float drift in the shader
        }
      }
      mesh.count = k; mesh.visible = k > 0; mesh.instanceMatrix.needsUpdate = true; geo.attributes.aInst.needsUpdate = true;
    },
    setMono() {},
    dispose() { scene.remove(mesh); geo.dispose(); mat.dispose(); },
  };
}
