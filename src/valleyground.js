// The Avatar valley's floor (src/avatarlayout.js): the ground the formations grow out of — forest canopy over rolling
// hills, a flood plain, gravel banks, scree and contact shade around every formation's foot, the foothills that bury
// their rims — and the river down the middle. Built in map space (x ∈ [0, 960)) as eight 120-unit chunks, each a fine
// band over the corridor and coarser bands out to the horizon (skirts hide the seams between bands); a chunk is built
// the first time it comes into view and drawn at its copy nearest the camera. Bent by the shared Earth curvature, shaded
// by the shadow field over the corridor. No texture files: the canopy is procedural (crown clumps fade out with distance).
import * as THREE from 'three';
import { valleySampler, riverZ, riverW } from './avatarlayout.js';
import { SHADOW_GLSL } from './shadowfield.js';
import { createValleyTrees } from './valleytrees.js';
import { createValleyFalls } from './valleyfalls.js';

const P = 960, CH = 120, WATER = -0.62;                   // period, chunk length, the river's surface above y0
const BANDS = [[-60, 60, 1], [60, 180, 2.5], [-180, -60, 2.5], [180, 600, 6], [-600, -180, 6]];   // z range and grid spacing
const CURVE = `
      vec4 wp = modelMatrix * vec4(transformed, 1.0); vFlatW = wp.xyz;
      { float cdx = wp.x - uApexX; wp.y -= (cdx * cdx + wp.z * wp.z) / 2800.0; }
      vec4 mvPosition = viewMatrix * wp; gl_Position = projectionMatrix * mvPosition;`;
const NOISE = /* glsl */`
  float hsh(vec2 p) { p = fract(p * vec2(0.1031, 0.1030)); p += dot(p, p.yx + 33.33); return fract((p.x + p.y) * p.x); }
  float vn(vec2 q, float n) { vec2 i = floor(q), f = fract(q), u = f * f * (3.0 - 2.0 * f); float i0 = mod(i.x, n), i1 = mod(i.x + 1.0, n);   // value noise, periodic in x over n cells
    return mix(mix(hsh(vec2(i0, i.y)), hsh(vec2(i1, i.y)), u.x), mix(hsh(vec2(i0, i.y + 1.0)), hsh(vec2(i1, i.y + 1.0)), u.x), u.y); }
  float rZ(float x) { return 6.0 * sin(6.2831853 * x / 240.0) + 2.5 * sin(6.2831853 * x / 96.0 + 1.3); }
  float rW(float x) { return 3.4 + 0.9 * sin(6.2831853 * x / 160.0 + 0.4); }`;

function chunkGeometry(c, sample) {                        // positions in map space, normals from the grid, aNear = foot proximity
  const pos = [], nrm = [], near = [], idx = [], out = [0, 0], x0 = c * CH;
  for (const [za, zb, d] of BANDS) {
    const nx = Math.round(CH / d) + 1, nz = Math.round((zb - za) / d) + 1, base = pos.length / 3, H = new Float32Array(nx * nz), N = new Float32Array(nx * nz);
    for (let j = 0; j < nz; j++) for (let i = 0; i < nx; i++) { sample(x0 + i * d, za + j * d, out); H[j * nx + i] = out[0]; N[j * nx + i] = out[1]; }
    for (let j = 0; j < nz; j++) for (let i = 0; i < nx; i++) {
      const h = H[j * nx + i], hx = (H[j * nx + Math.min(nx - 1, i + 1)] - H[j * nx + Math.max(0, i - 1)]) / ((Math.min(nx - 1, i + 1) - Math.max(0, i - 1)) * d);
      const hz = (H[Math.min(nz - 1, j + 1) * nx + i] - H[Math.max(0, j - 1) * nx + i]) / ((Math.min(nz - 1, j + 1) - Math.max(0, j - 1)) * d), l = Math.hypot(hx, 1, hz);
      pos.push(x0 + i * d, h, za + j * d); nrm.push(-hx / l, 1 / l, -hz / l); near.push(N[j * nx + i]);
    }
    for (let j = 0; j < nz - 1; j++) for (let i = 0; i < nx - 1; i++) { const a = base + j * nx + i, b = a + 1, e = a + nx, f = e + 1; idx.push(a, e, b, b, e, f); }
    for (const j of [0, nz - 1]) {                         // skirts along the band's z edges hide the cracks against a coarser neighbour
      const s0 = pos.length / 3;
      for (let i = 0; i < nx; i++) { const k = base + j * nx + i; pos.push(pos[3 * k], pos[3 * k + 1] - 2, pos[3 * k + 2]); nrm.push(nrm[3 * k], nrm[3 * k + 1], nrm[3 * k + 2]); near.push(near[k]); }
      for (let i = 0; i < nx - 1; i++) { const a = base + j * nx + i, b = a + 1, e = s0 + i, f = e + 1; if (j === 0) idx.push(a, b, e, b, f, e); else idx.push(a, e, b, b, e, f); }
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.setAttribute('normal', new THREE.Float32BufferAttribute(nrm, 3)); g.setAttribute('aNear', new THREE.Float32BufferAttribute(near, 1));
  g.setIndex(pos.length / 3 > 65535 ? new THREE.Uint32BufferAttribute(idx, 1) : new THREE.Uint16BufferAttribute(idx, 1));
  g.boundingSphere = new THREE.Sphere(new THREE.Vector3(x0 + CH / 2, 0, 0), 1e5);   // placed and culled by hand
  return g;
}
function riverGeometry(c) {                                // the water surface: a strip following the channel
  const pos = [], idx = [], x0 = c * CH, n = CH + 1, across = 5;
  for (let i = 0; i < n; i++) { const x = x0 + i, z = riverZ(x), w = riverW(x) + 1.6; for (let k = 0; k < across; k++) pos.push(x, WATER, z - w + 2 * w * k / (across - 1)); }
  for (let i = 0; i < n - 1; i++) for (let k = 0; k < across - 1; k++) { const a = i * across + k, b = a + 1, e = a + across, f = e + 1; idx.push(a, b, e, b, f, e); }
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.setIndex(idx); g.computeVertexNormals();
  g.boundingSphere = new THREE.Sphere(new THREE.Vector3(x0 + CH / 2, 0, 0), 1e5); return g;
}

function groundMaterial(U, S) {
  const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.95, metalness: 0, envMapIntensity: 0.9, side: THREE.DoubleSide });   // double-sided: the skirts face either way
  mat.onBeforeCompile = (sh) => {
    sh.uniforms.uApexX = U.uApexX; if (S) for (const k of ['uHeight', 'uSunDir', 'uY0', 'uShadowsOn', 'uPeriod', 'uShift']) sh.uniforms[k] = S[k];
    sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nuniform float uApexX; attribute float aNear; varying vec2 vMap; varying float vNear, vUp; varying vec3 vFlatW;')
      .replace('#include <project_vertex>', 'vMap = position.xz; vNear = aNear; vUp = normal.y;' + CURVE);
    sh.fragmentShader = sh.fragmentShader.replace('#include <common>', '#include <common>\nvarying vec2 vMap; varying float vNear, vUp; varying vec3 vFlatW;' + NOISE + (S ? SHADOW_GLSL : '') + `
      float crownLit;
      vec3 canopy(vec2 m) {                                                  // forest seen from the air: patches of tone, the odd autumn stand, clumps of crowns up close
        float n1 = vn(m / 40.0, 24.0), n2 = vn(m / 12.0 + 7.0, 80.0), n3 = vn(m / 96.0 + 3.0, 10.0);
        vec3 c = mix(vec3(0.034, 0.072, 0.024), vec3(0.085, 0.15, 0.045), smoothstep(0.15, 0.85, n1));
        c = mix(c, vec3(0.16, 0.22, 0.065), 0.5 * smoothstep(0.55, 0.92, n2));
        c = mix(c, vec3(0.3, 0.14, 0.04), 0.5 * smoothstep(0.74, 0.92, n3) * smoothstep(0.4, 0.8, n2));
        float fw = max(length(fwidth(m)), 1e-4), k = 0.5;                  // three octaves of crown clumps, each fading out as it shrinks below a few pixels
        k += (vn(m / 3.2 + 1.7, 300.0) - 0.5) * 0.9 * smoothstep(1.5, 4.0, 3.2 / fw);
        k += (vn(m / 1.2 + 9.1, 800.0) - 0.5) * 0.7 * smoothstep(1.5, 4.0, 1.2 / fw);
        k += (vn(m / 0.4 + 4.3, 2400.0) - 0.5) * 0.5 * smoothstep(1.5, 4.0, 0.4 / fw);
        crownLit = mix(0.66, 1.2, clamp(k, 0.0, 1.0));
        return c;
      }`)
      .replace('#include <color_fragment>', `#include <color_fragment>
      {
        float dr = abs(vMap.y - rZ(vMap.x)) - rW(vMap.x), steep = 1.0 - vUp;
        vec3 c = canopy(vMap);
        c = mix(c, vec3(0.16, 0.13, 0.09), smoothstep(0.25, 0.55, steep) * 0.7);                       // bare earth on steep slopes
        c = mix(c, vec3(0.17, 0.16, 0.13) * (0.75 + 0.5 * vn(vMap / 1.5, 640.0)), 1.0 - smoothstep(-0.4, 1.1, dr));   // a narrow strip of stones and sand along the water
        c = mix(c, vec3(0.12, 0.11, 0.09) * (0.75 + 0.5 * vn(vMap / 2.0, 480.0)), smoothstep(0.55, 1.0, vNear) * 0.55);   // scree at the rock's foot
        c *= crownLit * (1.0 - 0.45 * vNear * vNear);                                                   // crowns and gaps; contact shade under the formations
        diffuseColor.rgb = c;
      }`);
    if (S) sh.fragmentShader = sh.fragmentShader.replace('#include <aomap_fragment>', `#include <aomap_fragment>
      { float inField = 1.0 - smoothstep(34.0, 44.0, abs(vFlatW.z)), sv = inField > 0.0 ? mix(1.0, sunVisSoft(vFlatW), inField) : 1.0;   // the field covers the corridor: fade out before its edge (no lookups beyond)
        reflectedLight.directDiffuse *= sv; reflectedLight.directSpecular *= sv; }`);
  };
  mat.customProgramCacheKey = () => 'valley-ground' + (S ? '-lit' : '');
  return mat;
}
function waterMaterial(U, T) {                             // clear jade water: sky reflections through three's PBR, flowing ripples, shallows at the banks
  const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.14, metalness: 0, envMapIntensity: 1.0 });
  mat.onBeforeCompile = (sh) => {
    sh.uniforms.uApexX = U.uApexX; sh.uniforms.uTime = T;
    sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nuniform float uApexX; varying vec2 vMap; varying vec3 vFlatW;').replace('#include <project_vertex>', 'vMap = position.xz;' + CURVE);
    sh.fragmentShader = sh.fragmentShader.replace('#include <common>', '#include <common>\nuniform float uTime; varying vec2 vMap; varying vec3 vFlatW;' + NOISE)
      .replace('#include <color_fragment>', `#include <color_fragment>
      float t = clamp(abs(vMap.y - rZ(vMap.x)) / rW(vMap.x), 0.0, 1.0);
      diffuseColor.rgb = mix(vec3(0.012, 0.045, 0.045), vec3(0.06, 0.14, 0.11), t * t);`)
      .replace('#include <normal_fragment_maps>', `#include <normal_fragment_maps>
      { vec2 p = vMap; float t = uTime;                                       // flowing ripples: a few travelling sine waves (smooth, so the sun's glints stay round)
        float gx = 0.055 * cos(p.x * 1.7017 + p.y * 0.5 - t * 2.2) + 0.035 * cos(p.x * 3.29867 - p.y * 1.9 - t * 3.1) + 0.02 * cos(p.x * 6.09993 + p.y * 4.3 - t * 4.7);   // x frequencies: whole cycles per 960
        float gz = 0.04 * cos(p.y * 2.1 + p.x * 0.79849 - t * 1.6) + 0.03 * cos(p.y * 4.7 - p.x * 1.30245 - t * 2.6);
        normal = normalize(normal + (viewMatrix * vec4(-gx, 0.0, -gz, 0.0)).xyz); }`);
  };
  mat.customProgramCacheKey = () => 'valley-water';
  return mat;
}

export function createValleyGround(scene, set, { y0 = -26, U, shadow = null } = {}) {
  const { sample } = valleySampler(set), gMat = groundMaterial(U, shadow), T = { value: 0 }, wMat = waterMaterial(U, T);
  const chunks = Array.from({ length: P / CH }, () => null), fwd = new THREE.Vector3(), trees = createValleyTrees(scene, set, { y0, U, shadow }), falls = createValleyFalls(scene, set, { y0, U });
  function build(c) {
    const g = new THREE.Mesh(chunkGeometry(c, sample), gMat), w = new THREE.Mesh(riverGeometry(c), wMat);
    for (const m of [g, w]) { m.frustumCulled = false; m.renderOrder = -6; m.visible = false; scene.add(m); }
    chunks[c] = { g, w };
  }
  return {
    // cx: camera x; shift: the lap shift (render x = map x − shift); builds at most one missing chunk per call unless `all`
    update(dt, camera, on, shift = 0, all = false) {
      T.value += dt; camera.getWorldDirection(fwd); const cx = camera.position.x, camAbs = cx + shift; let built = false; trees.update(camera, on, shift); falls.update(camera, on, shift);
      for (let c = 0; c < chunks.length; c++) {
        const k = Math.round((camAbs - (c * CH + CH / 2)) / P), x = c * CH + CH / 2 + k * P - shift, dx = x - cx;
        const want = on && Math.abs(dx) < 620 + CH / 2 && !(fwd.x * (dx + (fwd.x > 0 ? CH / 2 : -CH / 2)) < -40 && Math.abs(fwd.z) < 0.9);   // within the fog, not wholly behind
        if (want && !chunks[c] && (all || !built)) { build(c); built = true; }
        const ch = chunks[c]; if (!ch) continue;
        ch.g.visible = ch.w.visible = want; if (want) { ch.g.position.set(k * P - shift, y0, 0); ch.w.position.set(k * P - shift, y0, 0); }
      }
    },
    dispose() { for (const ch of chunks) if (ch) for (const m of [ch.g, ch.w]) { scene.remove(m); m.geometry.dispose(); } gMat.dispose(); wMat.dispose(); trees.dispose(); falls.dispose(); },
    get trees() { return trees.count; },
  };
}
