// The Avatar valley's forest up close: real tree crowns on the valley floor within ~70 units of the camera (beyond, the
// ground's procedural canopy carries on). Broadleaf crowns (three lumpy shapes) and pines (cones), one instanced mesh
// per shape, placed on a jittered grid in map space by 24-unit tiles that are generated once and kept — so trees never
// swim — on the ground height itself, out of the river, thinning at the formations' feet (the rock rises straight
// out of the woods) and toward the edge of the range. Same curvature, shadow field and fog as the ground.
import * as THREE from 'three';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import { valleySampler, riverZ, riverW } from './avatarlayout.js';
import { SHADOW_GLSL, SHADOW_KEYS } from './shadowfield.js';

const P = 960, TILE = 24, STEP = 1.6, R = 62, MAX = 6000;
const CURVE = `
      vec4 wp = modelMatrix * instanceMatrix * vec4(transformed, 1.0); vFlatW = wp.xyz;
      { float cdx = wp.x - uApexX; wp.y -= (cdx * cdx + wp.z * wp.z) / 2800.0; }
      vec4 mvPosition = viewMatrix * wp; gl_Position = projectionMatrix * mvPosition;`;
const hash = (i, j, s) => { let h = Math.imul(i, 374761393) + Math.imul(j, 668265263) + Math.imul(s, 1442695041); h = Math.imul(h ^ (h >>> 13), 1274126177); return ((h ^ (h >>> 16)) >>> 0) / 4294967296; };

function treeMaterial(U, S) {
  const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.92, metalness: 0, envMapIntensity: 0.8 });
  mat.onBeforeCompile = (sh) => {
    sh.uniforms.uApexX = U.uApexX; if (S) for (const k of SHADOW_KEYS) sh.uniforms[k] = S[k];
    sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nuniform float uApexX; varying vec3 vFlatW, vLeaf; varying float vUpL;')
      .replace('#include <project_vertex>', 'vUpL = position.y; vLeaf = position * 4.5 + instanceMatrix[3].xyz * 1.7;' + CURVE);
    sh.fragmentShader = sh.fragmentShader.replace('#include <common>', `#include <common>
      varying vec3 vFlatW, vLeaf; varying float vUpL;` + (S ? SHADOW_GLSL : '') + `
      float h3(vec3 p) { p = fract(p * 0.1031); p += dot(p, p.zyx + 31.32); return fract((p.x + p.y) * p.z); }
      float n3(vec3 p) { vec3 i = floor(p), f = fract(p), u = f * f * (3.0 - 2.0 * f);
        return mix(mix(mix(h3(i), h3(i + vec3(1, 0, 0)), u.x), mix(h3(i + vec3(0, 1, 0)), h3(i + vec3(1, 1, 0)), u.x), u.y),
                   mix(mix(h3(i + vec3(0, 0, 1)), h3(i + vec3(1, 0, 1)), u.x), mix(h3(i + vec3(0, 1, 1)), h3(i + vec3(1, 1, 1)), u.x), u.y), u.z); }`)
      .replace('#include <color_fragment>', `#include <color_fragment>
      { float leaf = 0.6 * n3(vLeaf) + 0.4 * n3(vLeaf * 2.3 + 5.0);                       // clumps of leaves with dark gaps between them
        diffuseColor.rgb *= mix(0.5, 1.12, smoothstep(-0.6, 0.8, vUpL)) * mix(0.55, 1.3, smoothstep(0.25, 0.75, leaf)); }`);   // darker underneath, sunlit tops
    if (S) sh.fragmentShader = sh.fragmentShader.replace('#include <aomap_fragment>', `#include <aomap_fragment>
      { float inField = 1.0 - smoothstep(34.0, 44.0, abs(vFlatW.z)), sv = inField > 0.0 ? mix(1.0, sunVisSoft(vFlatW), inField) : 1.0; reflectedLight.directDiffuse *= sv; reflectedLight.directSpecular *= sv; }`);
  };
  mat.customProgramCacheKey = () => 'valley-trees' + (S ? '-lit' : '');
  return mat;
}

export function createValleyTrees(scene, set, { y0 = -26, U, shadow = null } = {}) {
  const { sample } = valleySampler(set), out = [0, 0], tiles = new Map(), v0 = new THREE.Vector3();
  // three lumpy crown shapes (a subdivided icosahedron pushed in and out by noise, flattened a little) and a pine
  const crowns = [0, 1, 2].map((vi) => {
    const g = mergeVertices(new THREE.IcosahedronGeometry(1, 1).deleteAttribute('normal').deleteAttribute('uv')), p = g.attributes.position;
    for (let i = 0; i < p.count; i++) { v0.fromBufferAttribute(p, i); const k = 0.82 + 0.36 * hash(Math.round(v0.x * 9) + vi * 31, Math.round(v0.y * 9), Math.round(v0.z * 9)); v0.multiplyScalar(k); p.setXYZ(i, v0.x, v0.y * 0.78, v0.z); }
    g.computeVertexNormals(); return g; });
  const pine = new THREE.ConeGeometry(0.5, 2.2, 7).translate(0, 0.7, 0);
  const mats = [treeMaterial(U, shadow), treeMaterial(U, shadow)], kinds = [...crowns, pine].map((g, i) => {
    const im = new THREE.InstancedMesh(g, mats[i < 3 ? 0 : 1], MAX); im.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(MAX * 3), 3);
    im.frustumCulled = false; im.count = 0; im.renderOrder = -5; scene.add(im); return im; });
  const TINT = [[0.045, 0.09, 0.03], [0.065, 0.12, 0.035], [0.09, 0.15, 0.045], [0.055, 0.1, 0.04], [0.12, 0.13, 0.045], [0.16, 0.075, 0.03]];   // forest greens, olive, the rare maple turning red
  function tile(ti, tj) {                                  // → Float32Array of trees: x, y, z, size, kind, tint (map space)
    const key = ti * 100000 + tj; if (tiles.has(key)) return tiles.get(key);
    const list = [], n = TILE / STEP;
    for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) {
      const i = ti * n + a, j = tj * n + b, r1 = hash(i, j, 1), r2 = hash(i, j, 2), r3 = hash(i, j, 3);
      const x = (i + 0.15 + 0.7 * r1) * STEP, z = (j + 0.15 + 0.7 * r2) * STEP, xm = ((x % P) + P) % P;
      if (Math.abs(z - riverZ(xm)) < riverW(xm) + 0.9) continue;          // not in the water
      sample(xm, z, out); if (out[1] > 0.8 || r3 < 0.08) continue;           // not under the rock's skirt; a few clearings
      const pine = hash(i, j, 4) < 0.1 + 0.2 * out[1], size = (0.85 + 0.55 * hash(i, j, 5)) * (pine ? 0.8 : 1), tint = hash(i, j, 6);
      list.push(x, out[0] + (pine ? -0.15 : 0.25 * size), z, size, pine ? 3 : Math.floor(hash(i, j, 8) * 3), tint > 0.97 ? TINT.length - 1 : Math.floor(tint * (TINT.length - 1)));   // crowns ride on (unseen) trunks; autumn stays rare
    }
    const arr = Float32Array.from(list); tiles.set(key, arr); if (tiles.size > 400) tiles.delete(tiles.keys().next().value);
    return arr;
  }
  const m = new THREE.Matrix4(), q = new THREE.Quaternion(), v = new THREE.Vector3(), sc = new THREE.Vector3(), Y = new THREE.Vector3(0, 1, 0), c = new THREE.Color();
  let lastKey = '';
  return {
    update(camera, on, shift = 0) {
      for (const im of kinds) im.visible = on; if (!on) return;
      const cx = camera.position.x, cz = camera.position.z, ax = cx + shift, key = Math.round(ax / 10) + ',' + Math.round(cz / 10) + ',' + shift;
      if (key === lastKey) return; lastKey = key;          // rebuilt when the camera has moved ~10 units (tiles are cached)
      const cnt = [0, 0, 0, 0];
      for (let ti = Math.floor((ax - R) / TILE); ti <= Math.floor((ax + R) / TILE); ti++) for (let tj = Math.floor((cz - R) / TILE); tj <= Math.floor((cz + R) / TILE); tj++) {
        const T = tile(ti, tj);
        for (let k = 0; k < T.length; k += 6) {
          const x = T[k] - shift, z = T[k + 2], d = Math.hypot(x - cx, z - cz); if (d > R) continue;
          const kind = T[k + 4], im = kinds[kind]; if (cnt[kind] >= MAX) continue;
          if (d > R - 18 && hash(Math.round(T[k] * 7), Math.round(z * 7), 9) < (d - (R - 18)) / 18) continue;   // thin out toward the edge of the range
          const s = T[k + 3], w = 0.85 + 0.3 * hash(Math.round(T[k] * 13), Math.round(z * 13), 7); sc.set(s * w, s, s * (2 - w)); q.setFromAxisAngle(Y, T[k] * 3.7 + z);
          im.setMatrixAt(cnt[kind], m.compose(v.set(x, y0 + T[k + 1], z), q, sc));
          const t = TINT[T[k + 5]], b = 0.85 + 0.3 * hash(Math.round(T[k] * 5), Math.round(z * 5), 11); c.setRGB(t[0] * b, t[1] * b, t[2] * b); if (kind === 3) c.multiplyScalar(0.62); im.setColorAt(cnt[kind], c); cnt[kind]++;
        }
      }
      kinds.forEach((im, i) => { im.count = cnt[i]; im.instanceMatrix.needsUpdate = true; im.instanceColor.needsUpdate = true; });
    },
    get count() { return kinds.reduce((n, im) => n + im.count, 0); },
    dispose() { for (const im of kinds) { scene.remove(im); im.dispose(); } for (const g of crowns) g.dispose(); pine.dispose(); for (const mm of mats) mm.dispose(); },
  };
}
