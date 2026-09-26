// Airliners as corridor hazards for atmospheric flight: instanced jets oriented along their velocity (level, with a
// gentle bank into turns) and world-space contrails; positions come from the environment (kind = 3 bodies).
// The aircraft is the procedural Airbus from aircraft.js: one InstancedMesh per part (2 draw calls) + one LineSegments.
import * as THREE from 'three';
import { airlinerParts } from './aircraft.js';

const TRAIL = 40, TRAIL_DT = 0.12;
export function createPlanes(scene, { max = 10 } = {}) {
  const model = airlinerParts();
  const meshes = model.parts.map(({ geo, mat }) => { const im = new THREE.InstancedMesh(geo, mat, max); im.instanceMatrix.setUsage(THREE.DynamicDrawUsage); im.count = 0; im.frustumCulled = false; scene.add(im); return im; });
  const trailGeo = new THREE.BufferGeometry(), tp = new Float32Array(max * TRAIL * 2 * 3), tc = new Float32Array(max * TRAIL * 2 * 3);
  trailGeo.setAttribute('position', new THREE.BufferAttribute(tp, 3)); trailGeo.setAttribute('color', new THREE.BufferAttribute(tc, 3));
  const trails = new THREE.LineSegments(trailGeo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false })); trails.frustumCulled = false; scene.add(trails);
  const slots = Array.from({ length: max }, () => ({ hist: new Float32Array(TRAIL * 3), hn: 0, hh: 0, acc: 0, gen: -1, bank: 0, lastVz: 0 }));
  const m = new THREE.Matrix4(), q = new THREE.Quaternion(), qb = new THREE.Quaternion(), pos = new THREE.Vector3(), fwd = new THREE.Vector3(), up = new THREE.Vector3(0, 1, 0), right = new THREE.Vector3(), upv = new THREE.Vector3(), basis = new THREE.Matrix4(), sc = new THREE.Vector3();
  let gain = 1;
  return {
    update(list, dt) {                                        // list: [{ p, v, r, gen }] in render space
      const n = Math.min(max, list.length);
      for (let i = 0; i < n; i++) {
        const P = list[i], s = slots[i];
        pos.set(P.p[0], P.p[1], P.p[2]); fwd.set(P.v[0], P.v[1], P.v[2]); if (fwd.lengthSq() < 0.01) fwd.set(1, 0, 0); fwd.normalize();
        right.crossVectors(fwd, up).normalize(); upv.crossVectors(right, fwd).normalize();
        basis.makeBasis(fwd, upv, right); q.setFromRotationMatrix(basis);
        const turn = (P.v[2] - s.lastVz); s.lastVz = P.v[2]; s.bank += (Math.max(-0.5, Math.min(0.5, -turn * 0.6)) - s.bank) * Math.min(1, dt * 2);   // bank into z-turns
        qb.setFromAxisAngle(fwd, s.bank); q.premultiply(qb);
        const k = (P.r / 1.6) * 1.25; sc.set(k, k, k); m.compose(pos, q, sc); for (const im of meshes) im.setMatrixAt(i, m);
        if (P.gen !== s.gen) { s.gen = P.gen; s.hn = 0; s.hh = 0; }
        const hx = s.hist[((s.hh - 1 + TRAIL) % TRAIL) * 3];
        if (s.hn > 0 && Math.abs(pos.x - hx) > 30) { s.hn = 0; s.hh = 0; }                    // seam wrap → restart the trail
        s.acc += dt; if (s.acc >= TRAIL_DT) { s.acc -= TRAIL_DT; s.hist[s.hh * 3] = pos.x; s.hist[s.hh * 3 + 1] = pos.y; s.hist[s.hh * 3 + 2] = pos.z; s.hh = (s.hh + 1) % TRAIL; s.hn = Math.min(TRAIL, s.hn + 1); }
        for (let j = 0; j < TRAIL - 1; j++) {
          const o = (i * TRAIL + j) * 6, valid = j < s.hn - 1, a0 = (s.hh - 1 - j + 2 * TRAIL) % TRAIL, a1 = (s.hh - 2 - j + 2 * TRAIL) % TRAIL, f = valid ? (1 - j / TRAIL) * 0.7 * gain : 0;
          const x0 = j === 0 ? pos.x - fwd.x * 1.6 : s.hist[a0 * 3], y0 = j === 0 ? pos.y - fwd.y * 1.6 : s.hist[a0 * 3 + 1], z0 = j === 0 ? pos.z - fwd.z * 1.6 : s.hist[a0 * 3 + 2];
          tp[o] = x0; tp[o + 1] = y0; tp[o + 2] = z0; tp[o + 3] = valid ? s.hist[a1 * 3] : x0; tp[o + 4] = valid ? s.hist[a1 * 3 + 1] : y0; tp[o + 5] = valid ? s.hist[a1 * 3 + 2] : z0;
          tc[o] = tc[o + 1] = tc[o + 2] = f; tc[o + 3] = tc[o + 4] = tc[o + 5] = f * 0.8;
        }
      }
      for (let i = n; i < max; i++) for (let j = 0; j < TRAIL - 1; j++) { const o = (i * TRAIL + j) * 6; tc[o] = tc[o + 1] = tc[o + 2] = tc[o + 3] = tc[o + 4] = tc[o + 5] = 0; }
      for (const im of meshes) { im.count = n; im.visible = n > 0; im.instanceMatrix.needsUpdate = true; }
      trails.visible = n > 0; trailGeo.attributes.position.needsUpdate = true; trailGeo.attributes.color.needsUpdate = true;
    },
    setMono(mode) { gain = mode === 2 ? 0.6 : 1; },
    dispose() { scene.remove(...meshes, trails); model.dispose(); trailGeo.dispose(); trails.material.dispose(); },
  };
}
