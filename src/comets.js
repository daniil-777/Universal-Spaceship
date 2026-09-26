// Comets (Hale–Bopp / NEOWISE look). Per comet: a tiny dark irregular nucleus; a billboard coma (shader: 1/ρ column-
// density halo compressed into a parabolic hood on the sunward side and drawn out anti-sunward, an HDR core the app's
// bloom softens, a slow pulse); a broad curved dust tail of soft instanced particles (born at the nucleus with a small
// "fountain" launch toward the sun, then pushed anti-sunward by radiation pressure while the nucleus flies on, so the
// tail curves behind the orbit — positions are integrated in the vertex shader from the birth time, the CPU only writes
// the few particles born each frame); and an ion tail of three thin camera-facing ribbon filaments (straight
// anti-sunward, fanning slightly, slow kinks, flowing streaks, an occasional disconnection knot). All comets share four
// draw calls: nucleus InstancedMesh, coma quads, dust particles (InstancedBufferGeometry) and ion ribbons.
// Render-space jumps: a `gen` change restarts a comet's tails; an x wrap shifts its dust so the tail stays attached.
import * as THREE from 'three';

const NP = 480, LIFE = 2.6, RATE = NP / LIFE, FIL = 3, ST = 22, VTX = FIL * ST * 2, IDX = FIL * (ST - 1) * 6, DEAD = -1e9, TAU = Math.PI * 2;
const ION_GAIN = 0.12, DUST_GAIN = 0.62, ACC = 3.2;                 // ACC: anti-sunward radiation-pressure acceleration (units/s²)
const AXES = /* glsl */`
  vec3 camRight() { return vec3(viewMatrix[0][0], viewMatrix[1][0], viewMatrix[2][0]); }
  vec3 camUp() { return vec3(viewMatrix[0][1], viewMatrix[1][1], viewMatrix[2][1]); }`;
const NOISE = /* glsl */`
  float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
  float noise(vec2 p) { vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y); }`;

const comaVert = AXES + /* glsl */`
  attribute vec2 aCorner; attribute vec2 aSG; attribute float aPhase;       // position = comet centre; aSG = (half size, gain)
  uniform vec3 uSun; uniform float uTime;
  varying vec2 vUv; varying vec2 vSun; varying float vGain;
  void main() {
    vec3 R = camRight(), U = camUp();
    vUv = aCorner; vSun = vec2(dot(uSun, R), dot(uSun, U));
    vGain = aSG.y * (1.0 + 0.035 * sin(uTime * 1.9 + aPhase) + 0.02 * sin(uTime * 4.7 + aPhase * 1.7));
    vec3 c = position + normalize(cameraPosition - position) * (aSG.x * 0.12);   // in front of the nucleus
    gl_Position = projectionMatrix * viewMatrix * vec4(c + (R * aCorner.x + U * aCorner.y) * aSG.x, 1.0);
  }`;
const comaFrag = /* glsl */`
  uniform vec3 uCore; uniform vec3 uHalo;
  varying vec2 vUv; varying vec2 vSun; varying float vGain;
  void main() {
    float l = length(vUv), d = l * 3.0; vec2 dir = vUv / max(l, 1e-4);  // d: distance in units of r (half size = 3r)
    float sw = dot(dir, vSun);                                          // +1 toward the sun in the image plane
    float re = d * max(0.6, 1.0 + 0.55 * sw);                           // compressed sunward, drawn out anti-sunward
    float core = 1.5 * exp(-d * d * 45.0) + 0.5 * exp(-re * re * 6.0);  // HDR centre (-> bloom) + inner coma
    float halo = 0.045 * exp(-re * 1.6);                                // soft outer coma
    float hood = 0.06 * exp(-pow((re - 0.9) * 3.0, 2.0)) * smoothstep(-0.2, 0.8, sw);   // sunward shell
    float edge = 1.0 - smoothstep(2.4, 3.0, d);
    gl_FragColor = vec4((uCore * core + uHalo * (halo + hood)) * edge * vGain, 1.0);
  }`;

const dustVert = AXES + /* glsl */`
  attribute vec3 iP0; attribute vec3 iV; attribute vec4 iB;             // iB = (birth time, size, alpha, tint)
  uniform vec3 uAnti; uniform float uTime; uniform float uAcc;
  varying vec2 vUv; varying float vA; varying float vTint;
  void main() {
    float s = uTime - iB.x, age = s / ${LIFE};
    vUv = position.xy; vTint = iB.w;
    if (age < 0.0 || age >= 1.0) { vA = 0.0; gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
    float grow = 0.4 + 1.2 * age;                                       // dust spreads as it ages (kept modest so a close pass does not fog the frame)
    vec3 p = iP0 + iV * s + uAnti * (0.5 * uAcc * s * s);               // fountain launch + radiation pressure
    vA = iB.z * smoothstep(0.0, 0.04, age) * pow(1.0 - age, 1.3) / grow;
    vec3 R = camRight(), U = camUp();
    gl_Position = projectionMatrix * viewMatrix * vec4(p + (R * position.x + U * position.y) * (iB.y * grow), 1.0);
  }`;
const dustFrag = /* glsl */`
  uniform vec3 uColA; uniform vec3 uColB; uniform float uGain;
  varying vec2 vUv; varying float vA; varying float vTint;
  void main() {
    float a = pow(max(0.0, 1.0 - dot(vUv, vUv)), 1.8) * vA * uGain;
    gl_FragColor = vec4(mix(uColA, uColB, vTint) * a, 1.0);
  }`;

const ionVert = /* glsl */`
  attribute vec4 aInfo; attribute float aFade;                          // aInfo = (t along tail, side ±1, filament, phase)
  varying vec4 vInfo; varying float vFade;
  void main() { vInfo = aInfo; vFade = aFade; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;
const ionFrag = NOISE + /* glsl */`
  uniform vec3 uHead; uniform vec3 uTail; uniform float uTime; uniform float uGain;
  varying vec4 vInfo; varying float vFade;
  void main() {
    float t = vInfo.x, side = vInfo.y, f = vInfo.z, ph = vInfo.w;
    float edge = 1.0 - side * side; edge *= edge;
    edge = edge * 0.7 + exp(-side * side * 10.0) * 0.6;                                   // faint band + thin bright spine
    float len = pow(1.0 - t, 1.35) * smoothstep(0.0, 0.05, t);
    float flow = 0.25 + 0.75 * noise(vec2(t * 18.0 - uTime * 2.4, side * 1.5 + f * 3.7 + ph));   // streaks flowing tailward
    flow *= 0.55 + 0.45 * noise(vec2(t * 5.0 - uTime * 0.9 + 7.0, side * 0.8 + ph + f));
    flow *= 0.75 + 0.25 * noise(vec2(t * 60.0 - uTime * 5.0, side * 3.0 + ph));         // fine strands (near passes)
    float th = mod(uTime * 0.19 + ph, 6.2832), kp = (th - 0.67) / 1.8;                    // disconnection event: a knot
    float kon = step(0.0, kp) * step(kp, 1.0) * smoothstep(0.0, 0.2, kp) * (1.0 - smoothstep(0.8, 1.0, kp));   // drifts down the tail
    float knot = kon * exp(-pow((t - kp) * 12.0, 2.0)) * 1.5 * (1.0 - t);
    float gap = kon * exp(-pow((t - kp + 0.06) * 25.0, 2.0)) * 0.7;
    vec3 col = mix(uHead, uTail, smoothstep(0.0, 0.5, t));
    float a = edge * (len * flow * (1.0 - gap) + knot) * vFade * uGain;
    gl_FragColor = vec4(col * a, 1.0);
  }`;

const mark = (attr, start, count) => { attr.addUpdateRange(start, count); attr.needsUpdate = true; };

export function createComets(scene, { max = 8, sunDir = new THREE.Vector3(-0.48, 0.08, -0.88) } = {}) {
  const sunW = sunDir.clone().normalize(), anti = sunW.clone().negate(), P1 = new THREE.Vector3(), P2 = new THREE.Vector3();
  const setBasis = () => { P1.set(0, 1, 0).cross(anti); if (P1.lengthSq() < 1e-4) P1.set(1, 0, 0).cross(anti); P1.normalize(); P2.crossVectors(anti, P1).normalize(); };
  setBasis();
  // ---- nucleus: one InstancedMesh of a small dark irregular body
  const nGeo = new THREE.IcosahedronGeometry(1, 1), np = nGeo.attributes.position, nv = new THREE.Vector3();
  for (let i = 0; i < np.count; i++) { nv.fromBufferAttribute(np, i); nv.multiplyScalar(0.72 + 0.28 * Math.sin(nv.x * 5.1 + nv.y * 3.7) * Math.cos(nv.z * 4.3 + nv.x * 2.2)); np.setXYZ(i, nv.x, nv.y, nv.z); }
  nGeo.computeVertexNormals();
  const nMat = new THREE.MeshStandardMaterial({ color: 0x3a3632, roughness: 0.95, metalness: 0.0 });
  const nucleus = new THREE.InstancedMesh(nGeo, nMat, max); nucleus.count = 0; nucleus.frustumCulled = false; nucleus.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  // ---- coma: one quad per comet, billboarded in the vertex shader
  const cGeo = new THREE.BufferGeometry(), cCenter = new Float32Array(max * 12), cCorner = new Float32Array(max * 8), cSG = new Float32Array(max * 8), cPh = new Float32Array(max * 4), cIdx = [];
  for (let i = 0; i < max; i++) { for (let k = 0; k < 4; k++) { cCorner[(i * 4 + k) * 2] = k & 1 ? 1 : -1; cCorner[(i * 4 + k) * 2 + 1] = k & 2 ? 1 : -1; cPh[i * 4 + k] = i * 2.39; } cIdx.push(i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 1, i * 4 + 3, i * 4 + 2); }
  const cCenterA = new THREE.BufferAttribute(cCenter, 3).setUsage(THREE.DynamicDrawUsage), cSGA = new THREE.BufferAttribute(cSG, 2).setUsage(THREE.DynamicDrawUsage);
  cGeo.setAttribute('position', cCenterA); cGeo.setAttribute('aCorner', new THREE.BufferAttribute(cCorner, 2)); cGeo.setAttribute('aSG', cSGA); cGeo.setAttribute('aPhase', new THREE.BufferAttribute(cPh, 1)); cGeo.setIndex(cIdx);
  const cMat = new THREE.ShaderMaterial({ vertexShader: comaVert, fragmentShader: comaFrag, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
    uniforms: { uSun: { value: sunW }, uTime: { value: 0 }, uCore: { value: new THREE.Color(0.85, 1.0, 0.9) }, uHalo: { value: new THREE.Color(0.6, 1.0, 0.8) } } });
  const coma = new THREE.Mesh(cGeo, cMat); coma.frustumCulled = false; coma.renderOrder = 3;
  // ---- dust: instanced soft particles, NP per comet in a ring buffer
  const dGeo = new THREE.InstancedBufferGeometry();
  dGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 1, -1, 0, -1, 1, 0, 1, 1, 0]), 3)); dGeo.setIndex([0, 1, 2, 1, 3, 2]);
  const dP0 = new THREE.InstancedBufferAttribute(new Float32Array(max * NP * 3), 3).setUsage(THREE.DynamicDrawUsage);
  const dV = new THREE.InstancedBufferAttribute(new Float32Array(max * NP * 3), 3).setUsage(THREE.DynamicDrawUsage);
  const dB = new THREE.InstancedBufferAttribute(new Float32Array(max * NP * 4), 4).setUsage(THREE.DynamicDrawUsage);
  for (let i = 0; i < max * NP; i++) dB.array[i * 4] = DEAD;
  dGeo.setAttribute('iP0', dP0); dGeo.setAttribute('iV', dV); dGeo.setAttribute('iB', dB); dGeo.instanceCount = 0;
  const dMat = new THREE.ShaderMaterial({ vertexShader: dustVert, fragmentShader: dustFrag, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
    uniforms: { uAnti: { value: anti }, uTime: { value: 0 }, uAcc: { value: ACC }, uGain: { value: DUST_GAIN }, uColA: { value: new THREE.Color(1.0, 0.93, 0.78) }, uColB: { value: new THREE.Color(1.0, 0.88, 0.7) } } });
  const dust = new THREE.Mesh(dGeo, dMat); dust.frustumCulled = false; dust.renderOrder = 2;
  // ---- ion tail: FIL camera-facing ribbons × ST stations per comet, one geometry
  const iGeo = new THREE.BufferGeometry(), iPos = new Float32Array(max * VTX * 3), iInfo = new Float32Array(max * VTX * 4), iFade = new Float32Array(max * VTX), iIdx = [];
  for (let i = 0; i < max; i++) for (let f = 0; f < FIL; f++) for (let j = 0; j < ST; j++) {
    const v = ((i * FIL + f) * ST + j) * 2;
    for (let k = 0; k < 2; k++) { iInfo[(v + k) * 4] = j / (ST - 1); iInfo[(v + k) * 4 + 1] = k ? 1 : -1; iInfo[(v + k) * 4 + 2] = f; iInfo[(v + k) * 4 + 3] = i * 2.39 + 0.7; }
    if (j < ST - 1) iIdx.push(v, v + 1, v + 2, v + 1, v + 3, v + 2);
  }
  const iPosA = new THREE.BufferAttribute(iPos, 3).setUsage(THREE.DynamicDrawUsage), iFadeA = new THREE.BufferAttribute(iFade, 1).setUsage(THREE.DynamicDrawUsage);
  iGeo.setAttribute('position', iPosA); iGeo.setAttribute('aInfo', new THREE.BufferAttribute(iInfo, 4)); iGeo.setAttribute('aFade', iFadeA); iGeo.setIndex(iIdx);
  const iMat = new THREE.ShaderMaterial({ vertexShader: ionVert, fragmentShader: ionFrag, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
    uniforms: { uHead: { value: new THREE.Color(0.6, 0.78, 1.0) }, uTail: { value: new THREE.Color(0.42, 0.62, 1.0) }, uTime: { value: 0 }, uGain: { value: ION_GAIN } } });
  const ion = new THREE.Mesh(iGeo, iMat); ion.frustumCulled = false; ion.renderOrder = 1;
  scene.add(nucleus, coma, dust, ion);

  const slots = []; for (let i = 0; i < max; i++) slots.push({ gen: -1, head: 0, acc: 0, fade: 0, px: 0, py: 0, pz: 0, ph: i * 2.39 + 0.7, seed: (i + 1) * 7919 });
  const rnd = (s) => { s.seed = (s.seed * 1664525 + 1013904223) >>> 0; return s.seed / 4294967296; };
  const _p = new THREE.Vector3(), _q = new THREE.Quaternion(), _s = new THREE.Vector3(), _m = new THREE.Matrix4(), _pt = new THREE.Vector3(), _view = new THREE.Vector3(), _side = new THREE.Vector3();
  let T = 0;
  function restart(s, base, c) {
    s.fade = 0; s.head = 0; s.acc = 0; for (let k = 0; k < NP; k++) dB.array[(base + k) * 4] = DEAD; mark(dB, base * 4, NP * 4);
    s.px = c.p[0]; s.py = c.p[1]; s.pz = c.p[2];
  }
  function emit(s, base, c, r, dt, vx, vy, vz) {
    s.acc += RATE * Math.max(0, dt); const ne = Math.min(24, s.acc | 0); s.acc -= ne; if (ne <= 0) return;
    const off = 0.25 * r;
    for (let k = 0; k < ne; k++) {
      const j = base + s.head; s.head = (s.head + 1) % NP; const f = (k + 1) / (ne + 1);
      const cz = 2 * rnd(s) - 1, sz = Math.sqrt(1 - cz * cz), ang = TAU * rnd(s), ex = sz * Math.cos(ang), ey = sz * Math.sin(ang), ez = cz;
      const spread = 0.5 + 1.2 * rnd(s), vs = 1.6 * (0.6 + 0.4 * rnd(s)), u = rnd(s);
      dP0.array[j * 3] = s.px + (c.p[0] - s.px) * f + ex * off; dP0.array[j * 3 + 1] = s.py + (c.p[1] - s.py) * f + ey * off; dP0.array[j * 3 + 2] = s.pz + (c.p[2] - s.pz) * f + ez * off;
      dV.array[j * 3] = ex * spread + sunW.x * vs + vx * 0.05; dV.array[j * 3 + 1] = ey * spread + sunW.y * vs + vy * 0.05; dV.array[j * 3 + 2] = ez * spread + sunW.z * vs + vz * 0.05;
      const szr = 0.35 + 1.6 * u * u;                                          // mostly small grains, a few big faint puffs
      dB.array[j * 4] = T - dt * (1 - f); dB.array[j * 4 + 1] = r * szr; dB.array[j * 4 + 2] = 0.07 * Math.pow(0.35 / szr, 0.8); dB.array[j * 4 + 3] = rnd(s);
    }
    const st = (s.head - ne + NP) % NP, n1 = Math.min(ne, NP - st);              // ring ranges (elements)
    mark(dP0, (base + st) * 3, n1 * 3); mark(dV, (base + st) * 3, n1 * 3); mark(dB, (base + st) * 4, n1 * 4);
    if (n1 < ne) { mark(dP0, base * 3, (ne - n1) * 3); mark(dV, base * 3, (ne - n1) * 3); mark(dB, base * 4, (ne - n1) * 4); }
  }
  return {
    max,
    setSunDir(v) { sunW.copy(v).normalize(); anti.copy(sunW).negate(); setBasis(); },
    setMono(m) { const g = m === 2 ? 0.6 : 1; dMat.uniforms.uGain.value = DUST_GAIN * g; iMat.uniforms.uGain.value = ION_GAIN * g; },
    // list: [{ p: [x,y,z], q: [x,y,z,w], v: [vx,vy,vz], r, gen }] in render space; dt in seconds; t: animation clock
    update(list, dt, camera, t) {
      T += dt; const n = Math.min(max, list.length), cam = camera.position;
      nucleus.visible = coma.visible = dust.visible = ion.visible = n > 0;
      for (let i = 0; i < n; i++) {
        const c = list[i], s = slots[i], r = c.r, vx = c.v[0], vy = c.v[1], vz = c.v[2], speed = Math.hypot(vx, vy, vz) || 1, base = i * NP;
        if (c.gen !== s.gen) { s.gen = c.gen; restart(s, base, c); }
        else {
          const jx = c.p[0] - s.px - vx * dt, jy = c.p[1] - s.py - vy * dt, jz = c.p[2] - s.pz - vz * dt;   // unexpected jump?
          if (Math.abs(jx) > 30 && Math.abs(jy) < 6 && Math.abs(jz) < 6) { for (let k = 0; k < NP; k++) dP0.array[(base + k) * 3] += jx; mark(dP0, base * 3, NP * 3); s.px += jx; }
          else if (jx * jx + jy * jy + jz * jz > 36) restart(s, base, c);
        }
        s.fade = Math.min(1, s.fade + dt * 2.5);
        emit(s, base, c, r, dt, vx, vy, vz);
        s.px = c.p[0]; s.py = c.p[1]; s.pz = c.p[2];
        // nucleus + coma
        _m.compose(_p.set(c.p[0], c.p[1], c.p[2]), _q.set(c.q[0], c.q[1], c.q[2], c.q[3]), _s.setScalar(0.3 * r)); nucleus.setMatrixAt(i, _m);
        for (let k = 0; k < 4; k++) { const v = i * 4 + k; cCenter[v * 3] = c.p[0]; cCenter[v * 3 + 1] = c.p[1]; cCenter[v * 3 + 2] = c.p[2]; cSG[v * 2] = 3 * r; cSG[v * 2 + 1] = s.fade; }
        // ion filaments: straight anti-sunward, fanning + slow kinks; ribbon faces the camera
        const L = 38 + 1.6 * speed;
        for (let f = 0; f < FIL; f++) {
          const th = s.ph + f * 2.094, ct = Math.cos(th), sn = Math.sin(th), ph1 = s.ph * 0.37 + f, ph2 = s.ph * 0.61 + f * 1.3;
          for (let j = 0; j < ST; j++) {
            const u = j / (ST - 1), d = u * L, fan = r * (0.4 + 1.3 * u) * Math.sqrt(u);
            const k1 = Math.sin(TAU * (1.3 * u + ph1 + t * 0.06)) * 0.9 * r * u, k2 = Math.sin(TAU * (2.1 * u + ph2 - t * 0.045)) * 0.6 * r * u;
            const ox = fan * ct + k1, oy = fan * sn + k2;
            _pt.set(c.p[0] + anti.x * d + P1.x * ox + P2.x * oy, c.p[1] + anti.y * d + P1.y * ox + P2.y * oy, c.p[2] + anti.z * d + P1.z * ox + P2.z * oy);
            _view.copy(cam).sub(_pt); _side.crossVectors(anti, _view).normalize();
            const w = r * (0.08 + 0.32 * Math.sin(Math.PI * Math.pow(u, 0.6))), v = ((i * FIL + f) * ST + j) * 2;
            iPos[v * 3] = _pt.x - _side.x * w; iPos[v * 3 + 1] = _pt.y - _side.y * w; iPos[v * 3 + 2] = _pt.z - _side.z * w;
            iPos[v * 3 + 3] = _pt.x + _side.x * w; iPos[v * 3 + 4] = _pt.y + _side.y * w; iPos[v * 3 + 5] = _pt.z + _side.z * w;
            iFade[v] = iFade[v + 1] = s.fade;
          }
        }
      }
      for (let i = n; i < max; i++) slots[i].gen = -1;                  // a slot that empties restarts when reused
      nucleus.count = n; nucleus.instanceMatrix.needsUpdate = true;
      cCenterA.needsUpdate = true; cSGA.needsUpdate = true; cGeo.setDrawRange(0, n * 6);
      dGeo.instanceCount = n * NP;
      if (n) { mark(iPosA, 0, n * VTX * 3); mark(iFadeA, 0, n * VTX); } iGeo.setDrawRange(0, n * IDX);
      cMat.uniforms.uTime.value = t; iMat.uniforms.uTime.value = t; dMat.uniforms.uTime.value = T;
    },
    dispose() { scene.remove(nucleus, coma, dust, ion); nGeo.dispose(); nMat.dispose(); cGeo.dispose(); cMat.dispose(); dGeo.dispose(); dMat.dispose(); iGeo.dispose(); iMat.dispose(); },
  };
}
