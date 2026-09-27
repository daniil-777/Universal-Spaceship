// The Avatar valley's waterfalls (src/avatarlayout.js set.falls): a curtain of water from a ledge on the formations beside
// the corridor and on a few giants far out, arcing off the lip and widening as it falls, streaks racing down faster as
// they drop, soft edges, and a churning cloud of mist where it lands. Two instanced draws (curtains, mist sprites), placed
// in map space at the copies near the camera; all motion is in the shaders. Same curvature and fog as the rest.
import * as THREE from 'three';

const P = 960, R = 460, MIST = 3;                        // period, range, mist sprites per fall
const NOISE = /* glsl */`
  float fh(vec2 p) { p = fract(p * vec2(0.1031, 0.1030)); p += dot(p, p.yx + 33.33); return fract((p.x + p.y) * p.x); }
  float fn(vec2 q) { vec2 i = floor(q), f = fract(q), u = f * f * (3.0 - 2.0 * f);
    return mix(mix(fh(i), fh(i + vec2(1.0, 0.0)), u.x), mix(fh(i + vec2(0.0, 1.0)), fh(i + vec2(1.0, 1.0)), u.x), u.y); }`;
const CURVE = `{ float cdx = wp.x - uApexX; wp.y -= (cdx * cdx + wp.z * wp.z) / 2800.0; }`;

function curtainMaterial(U, T) {
  return new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, side: THREE.DoubleSide, fog: true,
    uniforms: THREE.UniformsUtils.merge([THREE.UniformsLib.fog, { uTime: { value: 0 }, uApexX: { value: 0 } }]),
    vertexShader: /* glsl */`
      #include <fog_pars_vertex>
      uniform float uApexX; attribute float aSeed, aDrop, aVeil; varying vec2 vUv; varying float vSeed, vDrop, vVeil, vW;
      void main() {
        vec3 p = position; float v = clamp(-p.y, 0.0, 1.0);
        p.x *= 1.0 + 0.8 * v; p.z += 0.08 * sqrt(v) + 0.012;          // widens as it falls; leaves the lip and arcs out (a free fall is a parabola)
        vec4 wp = modelMatrix * instanceMatrix * vec4(p, 1.0); ${CURVE}
        vec4 mvPosition = viewMatrix * wp; gl_Position = projectionMatrix * mvPosition;
        vUv = vec2(position.x + 0.5, v); vSeed = aSeed; vDrop = aDrop; vVeil = aVeil; vW = length(instanceMatrix[0].xyz) * (1.0 + 0.8 * v);   // the curtain's width here (streaks keep a fixed size)
        #include <fog_vertex>
      }`,
    fragmentShader: /* glsl */`
      #include <fog_pars_fragment>
      uniform float uTime; varying vec2 vUv; varying float vSeed, vDrop, vVeil, vW;` + NOISE + `
      void main() {
        float y = vUv.y * vDrop, t = uTime * (2.6 + 3.0 * vUv.y) * (1.0 - 0.45 * vVeil);   // falling faster lower down; the spray veil drifts slower
        float ux = vUv.x * vW;                                                     // across the fall in units: ribbons ~0.4 wide however wide the fall
        vec2 q = vec2(ux * 2.2 + vSeed * 17.0, y * 0.3 - t); q.x += 0.4 * sin(q.y * 0.9 + vSeed * 6.0) + 0.2 * sin(q.y * 2.3);   // wavy ribbons, no square blocks
        float s = 0.5 * fn(q) + 0.3 * fn(q * vec2(2.6, 2.2) + 3.1) + 0.2 * fn(vec2(ux * 7.0 - vSeed, y * 1.1 - 1.8 * t));
        float edge = smoothstep(0.0, 0.22, vUv.x) * smoothstep(1.0, 0.78, vUv.x), top = smoothstep(0.0, 0.025, vUv.y), foot = 1.0 - smoothstep(0.7, 1.0, vUv.y);
        float a = edge * top * foot * (0.18 + 0.82 * smoothstep(0.4, 0.85, s)) * (0.9 - 0.3 * vUv.y) * mix(1.0, 0.16 + 0.2 * vUv.y, vVeil);   // gaps between the ribbons; the veil: faint at the lip, thickening into spray
        vec3 c = mix(vec3(0.58, 0.68, 0.72), vec3(0.96, 0.98, 1.0), smoothstep(0.3, 0.85, s));
        gl_FragColor = vec4(c, a);
        #include <fog_fragment>
      }`,
  });
}
function mistMaterial() {
  return new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, fog: true,
    uniforms: THREE.UniformsUtils.merge([THREE.UniformsLib.fog, { uTime: { value: 0 }, uApexX: { value: 0 } }]),
    vertexShader: /* glsl */`
      #include <fog_pars_vertex>
      uniform float uApexX, uTime; attribute float aSeed; varying vec2 vUv; varying float vSeed, vLife;
      void main() {
        float life = fract(uTime * 0.09 + aSeed), size = length(instanceMatrix[0].xyz) * (0.75 + 0.6 * life);   // each puff swells and rises, then starts over
        vec4 wp = modelMatrix * instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0); wp.y += life * size * 0.45; ${CURVE}
        vec4 mvPosition = viewMatrix * wp; mvPosition.xy += position.xy * size;   // facing the camera
        gl_Position = projectionMatrix * mvPosition; vUv = position.xy + 0.5; vSeed = aSeed; vLife = life;
        #include <fog_vertex>
      }`,
    fragmentShader: /* glsl */`
      #include <fog_pars_fragment>
      uniform float uTime; varying vec2 vUv; varying float vSeed, vLife;` + NOISE + `
      void main() {
        vec2 q = vUv - 0.5; float r = length(q) * 2.0, n = fn(q * 4.0 + vSeed * 9.0 + uTime * 0.25) * 0.6 + fn(q * 9.0 - vSeed * 3.0 - uTime * 0.4) * 0.4;
        float a = (1.0 - smoothstep(0.35, 1.0, r + 0.25 * (n - 0.5))) * smoothstep(0.0, 0.15, vLife) * (1.0 - smoothstep(0.6, 1.0, vLife)) * 0.34;
        gl_FragColor = vec4(mix(vec3(0.74, 0.8, 0.84), vec3(0.95, 0.97, 1.0), n), a);
        #include <fog_fragment>
      }`,
  });
}

export function createValleyFalls(scene, set, { y0 = -26, U } = {}) {
  const falls = set.falls || [], n = falls.length, T0 = performance.now();
  const curtainGeo = new THREE.PlaneGeometry(1, 1, 1, 40).translate(0, -0.5, 0), mistGeo = new THREE.PlaneGeometry(1, 1);
  for (const k of ['aSeed', 'aDrop', 'aVeil']) curtainGeo.setAttribute(k, new THREE.InstancedBufferAttribute(new Float32Array(n * 4), 1));   // two copies of the map at most, two layers each
  mistGeo.setAttribute('aSeed', new THREE.InstancedBufferAttribute(new Float32Array(n * 2 * MIST), 1));
  const cMat = curtainMaterial(), mMat = mistMaterial(), curtains = new THREE.InstancedMesh(curtainGeo, cMat, Math.max(1, n * 4)), mist = new THREE.InstancedMesh(mistGeo, mMat, Math.max(1, n * 2 * MIST));
  for (const im of [curtains, mist]) { im.frustumCulled = false; im.count = 0; im.renderOrder = 4; scene.add(im); }
  const m = new THREE.Matrix4(), q = new THREE.Quaternion(), v = new THREE.Vector3(), sc = new THREE.Vector3(), Y = new THREE.Vector3(0, 1, 0); let lastKey = '';
  return {
    update(camera, on, shift = 0) {
      const t = (performance.now() - T0) / 1000; cMat.uniforms.uTime.value = t; mMat.uniforms.uTime.value = t; cMat.uniforms.uApexX.value = mMat.uniforms.uApexX.value = U.uApexX.value;
      curtains.visible = mist.visible = on && n > 0; if (!curtains.visible) return;
      const cx = camera.position.x, cz = camera.position.z, key = Math.round((cx + shift) / 8) + ',' + Math.round(cz / 8) + ',' + shift; if (key === lastKey) return; lastKey = key;
      let nc = 0, nm = 0;
      for (let i = 0; i < n; i++) {
        const f = falls[i];
        for (let k = Math.ceil((cx + shift - R - f.x) / P); k <= Math.floor((cx + shift + R - f.x) / P); k++) {
          const x = f.x + k * P - shift; if (Math.hypot(x - cx, f.z - cz) > R) continue;
          q.setFromAxisAngle(Y, Math.PI / 2 - f.dir);                  // local +z → the fall's outward direction (dir = atan2(z, x))
          for (let L = 0; L < 2; L++) {                                  // the falling water, then a wider veil of spray around it
            curtains.setMatrixAt(nc, m.compose(v.set(x + Math.cos(f.dir) * (0.4 + 0.3 * L), y0 + f.y, f.z + Math.sin(f.dir) * (0.4 + 0.3 * L)), q, sc.set(f.w * (1 + 0.6 * L), f.drop, f.drop * (1 + 0.4 * L))));
            curtainGeo.attributes.aSeed.setX(nc, ((i + 0.37 * L) * 0.6180339) % 1); curtainGeo.attributes.aDrop.setX(nc, f.drop); curtainGeo.attributes.aVeil.setX(nc, L); nc++;
          }
          const out = 0.08 * f.drop, fx = x + Math.cos(f.dir) * (out + 0.4), fz = f.z + Math.sin(f.dir) * (out + 0.4);
          for (let j = 0; j < MIST; j++) { const r = f.w * (f.mouth ? 1.2 + j * 0.4 : 2.2 + j * 0.9);
            mist.setMatrixAt(nm, m.compose(v.set(fx + Math.cos(f.dir + j * 2.1) * f.w * 0.6, y0 + f.y - f.drop * 0.93 + j * 0.8, fz + Math.sin(f.dir + j * 2.1) * f.w * 0.6), q.identity(), sc.set(r, r, r)));
            mistGeo.attributes.aSeed.setX(nm, ((i * 7 + j) * 0.3819660) % 1); nm++; }
        }
      }
      curtains.count = nc; mist.count = nm; curtains.instanceMatrix.needsUpdate = mist.instanceMatrix.needsUpdate = true; for (const k of ['aSeed', 'aDrop', 'aVeil']) curtainGeo.attributes[k].needsUpdate = true; mistGeo.attributes.aSeed.needsUpdate = true;
    },
    dispose() { for (const im of [curtains, mist]) { scene.remove(im); im.dispose(); } curtainGeo.dispose(); mistGeo.dispose(); cMat.dispose(); mMat.dispose(); },
  };
}
