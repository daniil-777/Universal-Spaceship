// Meshy-generated models placed along the periodic corridor: the China route's pillar clusters and the city routes'
// district tiles — square dioramas joined edge to edge on a street grid — with bigger district plates beyond them.
// Every model is an InstancedMesh per part and level of detail (near / far) holding the copies of the layout the camera
// can see; tiles are trimmed to their square plate, sunk so the streets along their edges meet the ground, and every
// vertex bends with the shared Earth curvature (as the imagery and the mountains do), so nothing floats at a distance.
// Loaded only when the route is chosen. Same update/setMono/dispose API as the mountains; `ready` resolves when loaded.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { MESHY_SETS } from './meshylayout.js';
import { SHADOW_GLSL, SHADOW_KEYS } from './shadowfield.js';
import { createValleyGround } from './valleyground.js';

const PERIOD = 120, BANDS = [45, 220], RANGE0 = 480;   // detail levels: near within 45 units of the footprint, far to 220, extra-far beyond (a set may bring its own, with a mid level); past 480 the fog has it
export function meshyLoader() {                            // GLB with Draco-compressed meshes and WebP textures (the decoder comes from the same CDN as three)
  const draco = new DRACOLoader(); draco.setDecoderPath('https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/libs/draco/');
  return new GLTFLoader().setDRACOLoader(draco);
}

// Street level of a tile: the area-weighted height of the flat, low surfaces in the plate's outer band (the streets
// along its edges), so the plate's slab can be sunk below the ground.
function streetLevel(parts, W, H) {
  const bins = new Float64Array(240), a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3(), e = new THREE.Vector3(), n = new THREE.Vector3(), top = 0.3 * H;
  for (const { geo } of parts) {
    const pos = geo.attributes.position, idx = geo.index, cnt = idx ? idx.count : pos.count;
    for (let t = 0; t < cnt; t += 3) {
      a.fromBufferAttribute(pos, idx ? idx.getX(t) : t); b.fromBufferAttribute(pos, idx ? idx.getX(t + 1) : t + 1); c.fromBufferAttribute(pos, idx ? idx.getX(t + 2) : t + 2);
      const cy = (a.y + b.y + c.y) / 3; if (cy > top) continue;
      const ex = Math.max(Math.abs(a.x + b.x + c.x), Math.abs(a.z + b.z + c.z)) / 3; if (ex < 0.36 * W || ex > 0.5 * W) continue;
      n.subVectors(b, a).cross(e.subVectors(c, a)); const area = n.length() / 2; if (area < 1e-9 || n.y / (2 * area) < 0.9) continue;   // upward faces only (not the slab's underside)
      bins[Math.min(239, Math.floor(cy / top * 240))] += area;
    }
  }
  let best = 0; for (let i = 1; i < 240; i++) if (bins[i] > bins[best]) best = i;
  return (best + 1) / 240 * top;                           // the bin's top: the street surface sits at y = 0 after sinking
}

// The model's meshes merged into world-ready parts. fit: a number or { summit } (the tallest point → summit), or
// { width, inner, rot, trim } (a tile: the plate — `inner` × the model's footprint, turned by `rot` — spans `width`, street level → y = 0,
// `clip` = the half-width kept after trimming the rim and anything outside the plate)
export function normalisedModel(gltf, fit = 40) {
  if (typeof fit === 'number') fit = { summit: fit };
  if (fit.rot) gltf.scene.rotation.y = fit.rot;               // a plate generated at an angle: square it to the axes first
  gltf.scene.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(gltf.scene, true), height = box.max.y - box.min.y, W = Math.max(box.max.x - box.min.x, box.max.z - box.min.z);   // precise: a turned mesh's local box would overstate it
  const k = fit.width ? fit.width / (W * (fit.inner || 1)) : fit.summit / height;
  const norm = new THREE.Matrix4().makeScale(k, k, k).multiply(new THREE.Matrix4().makeTranslation(-(box.min.x + box.max.x) / 2, -box.min.y, -(box.min.z + box.max.z) / 2));   // centred in x/z, base on y = 0
  const parts = [];
  gltf.scene.traverse((o) => { if (o.isMesh) { const g = o.geometry.clone().applyMatrix4(new THREE.Matrix4().multiplyMatrices(norm, o.matrixWorld)); parts.push({ geo: g, mat: o.material }); } });
  let sink = 0, clip = 0;
  if (fit.width) { sink = streetLevel(parts, fit.width, height * k); clip = fit.width / 2 * (fit.trim ?? 0.985); for (const p of parts) p.geo.translate(0, -sink, 0); }
  return Object.assign(parts, { sink, clip, height: height * k - sink, radius: Math.SQRT1_2 * W * k });
}
const _t = new THREE.Vector3(), _r = new THREE.Quaternion(), _k = new THREE.Vector3(), _Y = new THREE.Vector3(0, 1, 0);   // scratch: called for every copy in view, every frame (no garbage)
export function placementMatrix(p, y0, apexX, xOffset = 0, out = new THREE.Matrix4()) {   // world matrix of one placement (copy offset along x; apexX = null: flat — the shader bends it)
  const x = p.x + xOffset, drop = apexX == null ? 0 : ((x - apexX) * (x - apexX) + p.z * p.z) / 2800;
  return out.compose(_t.set(x, y0 - drop + (p.y || 0), p.z), _r.setFromAxisAngle(_Y, p.ry), _k.set(p.s, p.s * (p.sy || 1), p.s * (p.sd || 1)));   // p.y: a formation's base, sunk into its foothill; sd: a squeezed depth (a tunnel rock)
}

const CURVE = `
      vec4 wp = vec4(transformed, 1.0);
      #ifdef USE_INSTANCING
        wp = instanceMatrix * wp;
      #endif
      wp = modelMatrix * wp; FLATW { float cdx = wp.x - uApexX; wp.y -= (cdx * cdx + wp.z * wp.z) / 2800.0; }
      vec4 mvPosition = viewMatrix * wp; gl_Position = projectionMatrix * mvPosition;`;
// Matte (Meshy's metalness maps turn glass into sky mirrors), a soft highlight shoulder (baked-white roofs would bloom),
// the shared Earth curvature per vertex, and the tile's square trim.
// The district look: matte where Meshy's roughness map says rough, a dielectric sky reflection where it says smooth (glass,
// polished stone — metalness stays 0: Meshy's metalness maps turned whole towers into sky mirrors), real sun shadows and
// street-canyon occlusion from the shadow field (S: its shared uniforms; inside the corridor's height grid), darker street
// level, a gentle highlight shoulder (baked-white roofs would bloom), the shared Earth curvature, and the per-instance clip.
export function meshyMaterial(mat, U, clip = 0, keepLook = false, S = null, canyon = 0.45) {   // canyon: how much taller neighbours darken (a pillar forest wants little)   // keepLook: a procedural model's own metal and lights (curvature and clip only)
  if (mat.map) mat.map.anisotropy = 16;
  if (!keepLook) { mat.metalness = 0; mat.metalnessMap = null; mat.roughness = 1; mat.envMapIntensity = 0.9; mat.color.setRGB(1.0, 0.96, 0.9); if (mat.roughnessMap) mat.roughnessMap.anisotropy = 8; }
  if (clip > 0) { mat.polygonOffset = true; mat.polygonOffsetFactor = -2; mat.polygonOffsetUnits = -2; }   // a tile's streets win over the urban base under them, even far away
  const lit = !keepLook && S;
  mat.onBeforeCompile = (sh) => {                          // aClip: the part of the model an instance keeps (x0, z0, x1, z1 in its own frame) — the tile's square, or half of it in a remix
    sh.uniforms.uApexX = U.uApexX;
    if (lit) for (const k of SHADOW_KEYS) sh.uniforms[k] = S[k];
    sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nuniform float uApexX; attribute vec4 aClip; varying vec2 vTileXZ; varying vec4 vClip; varying vec3 vFlatW;')
      .replace('#include <project_vertex>', 'vTileXZ = transformed.xz;\n#ifdef USE_INSTANCING\n vClip = aClip;\n#else\n vClip = vec4(-1e5, -1e5, 1e5, 1e5);\n#endif\n' + CURVE.replace('FLATW', 'vFlatW = wp.xyz;'));
    let fs = sh.fragmentShader.replace('#include <common>', '#include <common>\nvarying vec2 vTileXZ; varying vec4 vClip; varying vec3 vFlatW;' + (lit ? SHADOW_GLSL : ''));
    if (clip > 0) fs = fs.replace('void main() {', 'void main() {\n  if (vTileXZ.x < vClip.x || vTileXZ.x > vClip.z || vTileXZ.y < vClip.y || vTileXZ.y > vClip.w) discard;');   // only tiles clip: a discard switches off the early depth test, and overlapping rock would shade every hidden layer
    if (lit) fs = fs.replace('#include <aomap_fragment>', `#include <aomap_fragment>
      {
        float inField = 1.0 - step(44.0, abs(vFlatW.z)), sv = 1.0, cc = 0.0;                   // the height grid covers the corridor's rows: outside it, no lookups at all
        if (inField > 0.5) { sv = sunVis(vFlatW); for (int k = 0; k < 4; k++) { float a = float(k) * 1.5708 + 0.785; cc += clamp((fieldH(vFlatW.xz + vec2(cos(a), sin(a)) * 4.5) - vFlatW.y) / 12.0, 0.0, 1.0); } }
        float hg = max(vFlatW.y - uY0, 0.0), ao = mix(0.5, 1.0, smoothstep(0.0, 10.0, hg));       // street level sits in the city's shade
        ao *= 1.0 - ${canyon.toFixed(2)} * inField * uShadowsOn * cc * 0.25;                  // canyons: taller neighbours close in the sky
        reflectedLight.directDiffuse *= sv; reflectedLight.directSpecular *= sv;
        reflectedLight.indirectDiffuse *= ao; reflectedLight.indirectSpecular *= mix(1.0, ao, 0.7);
      }`);
    sh.fragmentShader = fs.replace('#include <opaque_fragment>', keepLook ? '#include <opaque_fragment>' : 'outgoingLight /= 1.0 + 0.3 * outgoingLight;\n#include <opaque_fragment>');
  };
  mat.customProgramCacheKey = () => (keepLook ? 'meshy-curved-own' : lit ? 'meshy-curved-lit-' + canyon.toFixed(2) : 'meshy-curved') + (clip > 0 ? '-clip' : '');
  mat.needsUpdate = true;
  return mat;
}

// The urban base: one flat asphalt plane under the tile rows (follows the camera, bent per vertex like the rest), so no
// satellite ground shows between districts or where a tile's plate is trimmed; drawn above the imagery layers.
function createBase(grid, U, y0) {
  const zMin = Math.min(...grid.rows) - grid.tile / 2 - 2, zMax = Math.max(...grid.rows) + grid.tile / 2 + 2, L = 2 * RANGE0 + 240;
  const geo = new THREE.PlaneGeometry(L, zMax - zMin, 96, 12).rotateX(-Math.PI / 2);
  const mat = new THREE.MeshStandardMaterial({ color: 0x3a3c41, roughness: 0.95, metalness: 0, polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1 });
  mat.onBeforeCompile = (sh) => { sh.uniforms.uApexX = U.uApexX; sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nuniform float uApexX;').replace('#include <project_vertex>', CURVE.replace('FLATW', '')); };
  mat.customProgramCacheKey = () => 'meshy-base';
  const mesh = new THREE.Mesh(geo, mat); mesh.frustumCulled = false; mesh.renderOrder = -5; mesh.position.set(0, y0 + 0.45, (zMin + zMax) / 2);   // above the imagery layers (lifted 0.15 / 0.3) so no satellite ground shows through the city
  return { mesh, place(cx) { mesh.position.x = Math.round(cx / 10) * 10; } };
}

// levels: 0 near, 1 mid (the rock formations only), 2 far, 3 extra-far. The coarse levels load first (the world appears
// at once); a finer level of a model loads as soon as one of its placements comes within reach of it (well before the
// camera gets there), a set with `preload` then streams every remaining finer level in the background, nearest model
// first, one at a time, and textures go to the GPU when a level arrives (not on the frame it is first drawn). A near level
// unused for a while is dropped only past the set's `maxNear` (GPU memory stays bounded on a long flight through the megacity).
const MAX_NEAR = 6, NEAR_IDLE = 20, NL = 4, PREFETCH = 120;
const isLoaded = (L) => !!L && typeof L === 'object';
export function createMeshyWorld(scene, { y0 = -26, base = '', set = 'china', shadow = null, renderer = null } = {}) {   // shadow: the shadow field's uniforms
  const S = MESHY_SETS[set], P = S.period || PERIOD, loader = meshyLoader(), meshes = [], U = { uApexX: { value: 0 } };
  const urls = [S.models, S.mid || [], S.far || [], S.xfar || []], fit = (mi) => (S.fits ? S.fits[mi] : S.summits[mi]);
  const bands = S.bands || [BANDS[0], BANDS[0], BANDS[1]];  // distance from the footprint beyond which each finer level gives way (no mid level: its band is empty)
  const RANGE = S.range || RANGE0;                           // how far copies are drawn (a hazier world can stop sooner)
  const byModel = S.models.map((_, mi) => S.placements.filter((p) => p.m === mi)), radius = S.models.map(() => 30);
  const lift = S.models.map((u, mi) => (typeof u === 'string' && fit(mi) && typeof fit(mi) === 'object' && fit(mi).width ? 0.5 : S.grid ? 0.45 : 0));   // tiles stand a hair above the base; towers on it
  const levels = S.models.map(() => [null, null, null, null]), cap = Math.ceil((2 * RANGE + 200) / P) + 2, clipOf = S.models.map(() => 0), closest = S.models.map(() => Infinity);
  const maxNear = S.maxNear ?? MAX_NEAR, prefetch = S.prefetch ?? PREFETCH; let lowReady = false, queued = 0;
  let vis = 0, disposed = false, clock = 0, firstFrame = true; const baseMesh = S.grid ? createBase(S.grid, U, y0) : null, heightOf = S.models.map(() => 40);
  if (baseMesh) { scene.add(baseMesh.mesh); meshes.push(baseMesh.mesh); }
  const ground = S.valley ? createValleyGround(scene, S, { y0, U, shadow }) : null;   // the Avatar valley brings its own ground (forest, river, foothills)
  function load(mi, l) {                                   // → promise; a level that fails stays null (the others stand in)
    const u = urls[l][mi]; if (!u || levels[mi][l]) return Promise.resolve();
    levels[mi][l] = 'loading';
    const proc = typeof u === 'object';                    // { proc: 'eiffel', variant, height }: geometry built here, per level of detail
    const got = proc ? import('./eiffel.js').then(({ eiffelParts }) => Object.assign(eiffelParts(u.variant, { height: u.height, lod: Math.max(0, l - 1) }), { clip: 0, radius: Math.max(u.height * 0.55, 20) }))
      : loader.loadAsync(base + u).then((gltf) => normalisedModel(gltf, fit(mi)));
    return got.then((parts) => {
      if (disposed) return;
      radius[mi] = parts.radius; heightOf[mi] = parts.height || 40;
      const n = Math.max(1, byModel[mi].length * cap);
      const list = parts.map(({ geo, mat }) => { geo.setAttribute('aClip', new THREE.InstancedBufferAttribute(new Float32Array(n * 4), 4)); const mm = meshyMaterial(mat, U, parts.clip, proc, shadow, S.canyon ?? 0.45); if (S.sky && !proc) mm.envMapIntensity = S.sky; const im = new THREE.InstancedMesh(geo, mm, n); im.frustumCulled = false; im.renderOrder = -7 + l; im.count = 0; im.visible = false; scene.add(im); meshes.push(im); return im; });   // finer levels (nearer copies) draw first: the depth test then skips rock hidden behind them
      levels[mi][l] = { list, used: clock }; clipOf[mi] = parts.clip;
      if (renderer) for (const { mat } of parts) for (const t of [mat.map, mat.normalMap, mat.roughnessMap]) if (t) renderer.initTexture(t);   // upload now, not mid-flight on first draw
    }, (e) => { levels[mi][l] = 'failed'; throw e; });   // a missing level stays missing (the others stand in); no retry every frame
  }
  function drop(mi, l) { const L = levels[mi][l]; if (!isLoaded(L)) return; for (const im of L.list) { scene.remove(im); meshes.splice(meshes.indexOf(im), 1); im.geometry.dispose(); im.material.dispose(); if (im.material.map) im.material.map.dispose(); if (im.material.normalMap) im.material.normalMap.dispose(); im.dispose(); } levels[mi][l] = null; }
  const first = (mi) => (urls[3][mi] ? 3 : urls[2][mi] ? 2 : urls[1][mi] ? 1 : 0);
  const ready = Promise.all(S.models.map((_, mi) => load(mi, first(mi)))).then(() => Promise.all(S.models.map((_, mi) => (first(mi) === 3 && urls[2][mi] ? load(mi, 2) : null)).map((p) => p && p.catch(() => {})))).then(() => { lowReady = true; });
  function background() {                                  // preload: the next missing finer level (all mids first, then the nears), nearest model first
    if (!S.preload || !lowReady || queued) return;
    for (const l of [1, 0]) { let best = -1; for (let mi = 0; mi < levels.length; mi++) if (urls[l][mi] && !levels[mi][l] && (best < 0 || closest[mi] < closest[best])) best = mi;
      if (best >= 0) { queued++; load(best, l).catch(() => {}).finally(() => { queued--; }); return; } }
  }
  const pool = [], order = [], byDist = (a, b) => a.d - b.d;   // this frame's copies of one model (pooled records)
  const PREF = [[0, 1, 2, 3], [1, 0, 2, 3], [2, 1, 3, 0], [3, 2, 1, 0]], m = new THREE.Matrix4(), fwd = new THREE.Vector3(), counts = [0, 0, 0, 0], frustum = new THREE.Frustum(), pv = new THREE.Matrix4(), sph = new THREE.Sphere();
  return {
    ready, get meshes() { return meshes; }, ownGround: !!ground,
    get stats() { return levels.map((Lv) => Lv.map((L) => (!L ? '-' : typeof L === 'string' ? L[0].toUpperCase() : L.list[0].count)).join('/')).join(' '); },   // per model: instances drawn at near/mid/far/extra-far (L loading, F failed)
    // shift: the long city's lap shift (render x = map x − shift); the copies in view, each at the best detail loaded
    update(dt, camera, v, sunDirWorld, apexX, shift = 0) {
      clock += dt; vis = v; const on = vis > 0.01; U.uApexX.value = apexX ?? camera.position.x + 10;
      for (const im of meshes) im.visible = on; if (ground) ground.update(dt, camera, on, shift, firstFrame); firstFrame = false; if (!on) return;
      const cx = camera.position.x, cz = camera.position.z, camAbs = cx + shift, apex = U.uApexX.value; camera.getWorldDirection(fwd);
      camera.updateMatrixWorld(); pv.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse); frustum.setFromProjectionMatrix(pv);
      if (baseMesh) baseMesh.place(cx);
      let nearCount = 0;
      for (let mi = 0; mi < byModel.length; mi++) {
        const Lv = levels[mi]; counts[0] = counts[1] = counts[2] = counts[3] = 0; let need = NL, nc = 0; order.length = 0; closest[mi] = Infinity;   // need: the finest level any copy will want soon
        const hw = S.halves && S.halves[mi];                 // the footprint's half-extents: detail follows the distance to the footprint, not to a bounding circle (a long ridge would stay at full detail far too long)
        for (const p of byModel[mi]) {
          const r = radius[mi] * p.s, k0 = Math.ceil((camAbs - RANGE - r - p.x) / P), k1 = Math.floor((camAbs + RANGE + r - p.x) / P), cr = Math.cos(p.ry), sr = Math.sin(p.ry);
          for (let k = k0; k <= k1; k++) {
            const x = p.x + k * P - shift, dx = x - cx, dz = p.z - cz, d = Math.hypot(dx, dz);
            if (d > RANGE + r || dx * fwd.x + dz * fwd.z < -r - 30) continue;   // past the fog, or behind the camera
            const db = hw ? Math.hypot(Math.max(0, Math.abs(-cr * dx + sr * dz) - hw[0] * p.s), Math.max(0, Math.abs(-sr * dx - cr * dz) - hw[1] * p.s * (p.sd || 1))) : Math.max(0, d - r);
            if (d > 90 + r) { const hh = heightOf[mi] * p.s * (p.sy || 1) / 2; sph.center.set(x, y0 + (p.y || 0) + hh - ((x - apex) * (x - apex) + p.z * p.z) / 2800, p.z); sph.radius = Math.hypot(r, hh) + 4; if (!frustum.intersectsSphere(sph)) continue; }   // off screen (near ones stay: the onboard camera looks elsewhere)
            const want = db > bands[2] ? 3 : db > bands[1] ? 2 : db > bands[0] ? 1 : 0; need = Math.min(need, db < bands[0] + prefetch ? 0 : db < bands[1] + prefetch ? 1 : NL); if (db < closest[mi]) closest[mi] = db;
            let l = -1; for (const c of PREF[want]) if (isLoaded(Lv[c])) { l = c; break; } if (l < 0) continue;
            const e = pool[nc] || (pool[nc] = { d: 0, p: null, k: 0, l: 0 }); e.d = d; e.p = p; e.k = k; e.l = l; order.push(e); nc++;
          }
        }
        order.sort(byDist);                                  // nearest copies first within each draw: the depth test then skips what they hide
        for (const { p, k, l } of order) {
          placementMatrix(p, y0 + lift[mi], null, k * P - shift, m); const c = clipOf[mi] || 1e5, r4 = p.clip;   // a remix keeps only its half of the tile
          for (const im of Lv[l].list) { im.setMatrixAt(counts[l], m); im.geometry.attributes.aClip.setXYZW(counts[l], r4 ? Math.max(-c, r4[0]) : -c, r4 ? Math.max(-c, r4[1]) : -c, r4 ? Math.min(c, r4[2]) : c, r4 ? Math.min(c, r4[3]) : c); }
          counts[l]++; Lv[l].used = clock;
        }
        for (let l = 0; l < 2; l++) if (need <= l && urls[l][mi] && !Lv[l]) load(mi, l).catch(() => {});
        for (let l = 0; l < NL; l++) if (isLoaded(Lv[l])) for (const im of Lv[l].list) { im.count = counts[l]; im.instanceMatrix.needsUpdate = true; im.geometry.attributes.aClip.needsUpdate = true; }
        if (isLoaded(Lv[0])) nearCount++;
      }
      background();
      if (nearCount > maxNear) {                           // too many near levels resident: drop the longest idle ones (never the only level a model has)
        const idle = levels.map((Lv, mi) => ({ mi, t: isLoaded(Lv[0]) && (isLoaded(Lv[1]) || isLoaded(Lv[2]) || isLoaded(Lv[3])) ? clock - Lv[0].used : -1 })).filter((e) => e.t > NEAR_IDLE).sort((a, b) => b.t - a.t);
        for (const e of idle.slice(0, nearCount - maxNear)) drop(e.mi, 0);
      }
    },
    setMono() {},
    dispose() { disposed = true; if (ground) ground.dispose(); for (let mi = 0; mi < levels.length; mi++) for (let l = 0; l < NL; l++) drop(mi, l); for (const im of meshes.slice()) { scene.remove(im); im.geometry.dispose(); im.material.dispose(); } meshes.length = 0; },
  };
}
