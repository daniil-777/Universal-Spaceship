// Streamed satellite-imagery terrain for atmospheric flight — seamless version. Each layer is ONE curved mesh with ONE
// texture atlas (cols × rows tiles of 256 px); tiles are copied into the atlas on the GPU as they arrive, the ground
// scrolls by shifting the atlas u-offset, and the atlas wraps horizontally so trailing columns are refilled with the
// tiles two columns ahead of the visible window. Continuous filtering across tile borders, no geometry seams, no
// loading holes (a not-yet-loaded column keeps the last imagery and is behind the window's leading edge anyway).
import * as THREE from 'three';
import { SHADOW_GLSL } from './shadowfield.js';

export const TERRAIN_ATTRIBUTION = 'imagery © Esri, Maxar, Earthstar Geographics and the GIS user community';
const KM_PER_DEG = 40075.016 / 360, T = 256;
const VERT = /* glsl */`uniform float uApexX; varying vec2 vUv; varying vec3 vP; varying vec2 vLocal;
  void main(){ vUv = uv; vLocal = position.xz; vec4 w = modelMatrix * vec4(position, 1.0);
    w.y -= ((w.x - uApexX) * (w.x - uApexX) + w.z * w.z) / 2800.0;                             // one shared Earth curvature (R = 1400) for every strip, the mountains and the towers
    vP = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`;
const FRAG = /* glsl */`uniform sampler2D tAtlas; uniform float uUScale, uUOffset, uVis, uHalfW, uHalfH, uFadeX, uFadeZ, uSea; uniform vec3 uHaze;
  ${SHADOW_GLSL}
  varying vec2 vUv; varying vec3 vP; varying vec2 vLocal;
  void main(){
    vec3 c = texture2D(tAtlas, vec2(vUv.x * uUScale + uUOffset, 1.0 - vUv.y)).rgb;
    float lum = dot(c, vec3(0.3, 0.59, 0.11)), seaLike = smoothstep(0.02, 0.09, c.b - max(c.r, c.g)) * (1.0 - smoothstep(0.25, 0.45, lum));
    c = mix(c, vec3(0.055, 0.11, 0.22) * (0.7 + 0.6 * lum / 0.3), seaLike * uSea);                // open-sea tiles vary in tone between tiles: even them out
    float dist = length(cameraPosition - vP);
    c = mix(c, uHaze, 0.85 * (1.0 - exp(-dist / 700.0)));                                          // aerial perspective (same law as the globe's shader)
    float ax = 1.0 - smoothstep(uHalfW - uFadeX, uHalfW, abs(vLocal.x)), az = 1.0 - smoothstep(uHalfH - uFadeZ, uHalfH, abs(vLocal.y));
    if (dist < 240.0) c *= mix(0.5, 1.0, sunVis(vec3(vP.x, uY0, vP.z)));                     // towers and ridges shade the nearby ground (too small to see farther out)
    gl_FragColor = vec4(c, uVis * ax * az); }`;

export function createTerrainLayer(scene, renderer, { zoom = 9, lat = 41.5, lon0 = 6.0, rows = 13, cols = 18, behind = 4, kmPerUnit = 2.8, groundSpeed = 0.45, y = -26, R = 1400, lift = 0, order = -7, shadow = null,
  fadeX = 0.1, fadeZ = 0.18, sea = 0.7, url = (z, x, yy) => `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/${z}/${yy}/${x}` } = {}) {
  const n = 1 << zoom, phi = lat * Math.PI / 180, cosL = Math.cos(phi);
  const side = KM_PER_DEG * 360 * cosL / n / kmPerUnit, vcols = cols - 2;                       // visible columns; 2 spare columns load ahead
  const mRef = (1 - Math.log(Math.tan(phi) + 1 / Math.cos(phi)) / Math.PI) / 2;
  const tx0 = Math.floor((lon0 + 180) / 360 * n) - behind, ty0 = Math.floor(mRef * n) - Math.floor(rows / 2);
  const width = vcols * side, height = rows * side;
  // atlas
  const atlas = new THREE.DataTexture(new Uint8Array(cols * T * rows * T * 4).fill(18), cols * T, rows * T, THREE.RGBAFormat); atlas.colorSpace = THREE.SRGBColorSpace;
  atlas.wrapS = THREE.RepeatWrapping; atlas.wrapT = THREE.ClampToEdgeWrapping; atlas.minFilter = THREE.LinearMipmapLinearFilter; atlas.magFilter = THREE.LinearFilter; atlas.anisotropy = 8; atlas.generateMipmaps = true; atlas.flipY = false; atlas.needsUpdate = true;
  renderer.initTexture(atlas);
  // mesh on the sphere approximation; the mesh stays put (relative to the camera x), only the texture scrolls. Tile rows run
  // north → south down the atlas, and the plane's v runs toward −z, so the shader samples 1 − v: north lies at −z, on the
  // left of a ship flying east (+x) — as seen from a real aircraft (sampling v directly mirrored the map).
  const geo = new THREE.PlaneGeometry(width, height, 48, 24); geo.rotateX(-Math.PI / 2);
  const mat = new THREE.ShaderMaterial({ vertexShader: VERT, fragmentShader: FRAG, transparent: true, depthWrite: true,
    uniforms: { tAtlas: { value: atlas }, uUScale: { value: vcols / cols }, uUOffset: { value: 0 }, uVis: { value: 0 }, uHalfW: { value: width / 2 }, uHalfH: { value: height / 2 }, uFadeX: { value: fadeX * width }, uFadeZ: { value: fadeZ * height }, uSea: { value: sea }, uHaze: { value: new THREE.Color(0.62, 0.74, 0.9) }, uApexX: { value: 0 }, ...(shadow ? shadow.uniforms : {}) } });
  const mesh = new THREE.Mesh(geo, mat); mesh.renderOrder = order; mesh.frustumCulled = false; mesh.visible = false; scene.add(mesh);
  const loader = new THREE.TextureLoader(); loader.setCrossOrigin('anonymous');
  let loads = 0, fails = 0, copies = 0, scroll = 0, filled = 0; const pending = new Map();      // atlas column → tx being loaded
  const dst = new THREE.Vector2();
  function fillColumn(col, tx) {                                                                 // fetch the rows of tile column tx into atlas column col
    pending.set(col, tx);
    for (let r = 0; r < rows; r++) {
      const ty = ty0 + r; if (ty < 0 || ty >= n) continue;
      const myTx = tx; loads++;
      loader.load(url(zoom, ((tx % n) + n) % n, ty), (tex) => {
        if (pending.get(col) !== myTx) { tex.dispose(); return; }                                // column was recycled again meanwhile
        tex.flipY = false; tex.colorSpace = THREE.SRGBColorSpace; tex.generateMipmaps = false; tex.minFilter = THREE.LinearFilter;
        try { renderer.copyTextureToTexture(tex, atlas, null, dst.set(col * T, r * T)); copies++; } catch (e) { fails++; }
        tex.dispose();
      }, undefined, () => { fails++; });
    }
  }
  for (let c = 0; c < cols; c++) fillColumn(c, tx0 + c);
  return { apexOffset: (vcols / 2 - behind) * side,
    mesh, side, zoom, attribution: TERRAIN_ATTRIBUTION, get stats() { return { zoom, loads, copies, fails, scroll: +scroll.toFixed(1), filled }; },
    update(dt, camera, vis, apexX) {
      mat.uniforms.uApexX.value = apexX ?? camera.position.x;
      mesh.visible = vis > 0.01; mat.uniforms.uVis.value = vis; if (!mesh.visible) return;
      mesh.position.set(camera.position.x + (vcols / 2 - behind) * side, y + lift, 0);   // the strip's curvature apex sits (vcols/2 − behind) tiles ahead of the camera
      const before = Math.floor(scroll / side); scroll += groundSpeed * dt * vis; const after = Math.floor(scroll / side);
      for (let m = before + 1; m <= after; m++) { const col = ((m - 1) % cols + cols) % cols; fillColumn(col, tx0 + (m - 1) + cols); filled++; }   // the column that just left the window gets the tile two columns ahead
      mat.uniforms.uUOffset.value = (scroll / side) / cols;
    },
    dispose() { scene.remove(mesh); geo.dispose(); mat.dispose(); atlas.dispose(); },
  };
}
// Three aligned layers: a wide coarse strip (zoom 9), a sharper inner strip (zoom 11) and a city-detail strip (zoom 12)
// right under the flight path. All scroll together; each fades into the next at its edges.
export const ROUTES = {                                     // parallels to fly, starting just west of the sight so it arrives in ~15 s; terrain: 'alps' | 'pillars' | none (flat imagery); city: skyline
  alps: { name: 'Alps', lat: 46.55, lon0: 7.3, terrain: 'alps' },
  china: { name: 'China', lat: 29.33, lon0: 109.9, terrain: 'meshy' },   // Zhangjiajie: Meshy-generated pillar clusters (baked collision grid)
  newyork: { name: 'New York', lat: 40.74, lon0: -74.5, city: 'newyork' },    // Manhattan ~16 units ahead at start (≈ 35 s at 0.45 units/s); `city` = skyline available
  london: { name: 'London', lat: 51.5, lon0: -0.7, city: 'london' },
  moscow: { name: 'Moscow', lat: 55.75, lon0: 36.85, city: 'moscow' },
  dubai: { name: 'Dubai', lat: 25.2, lon0: 54.75, city: 'dubai' },
  mega: { name: 'Megacity', lat: 48.86, lon0: 2.1, city: 'mega' },   // every city's districts on one long map, over the Paris basin
};
export function createTerrain(scene, renderer, opts = {}) {
  // the three zoom levels stack 0.15 apart, all BELOW the ground level y0, so mountain valley floors and city streets
  // (which sit on y0) always cover the imagery instead of the map showing through them
  const far = createTerrainLayer(scene, renderer, { zoom: 9, rows: 13, cols: 18, behind: 4, lift: -0.35, order: -7, fadeX: 0.08, fadeZ: 0.2, ...opts });
  const near = createTerrainLayer(scene, renderer, { zoom: 11, rows: 13, cols: 26, behind: 7, lift: -0.2, order: -6, fadeX: 0.12, fadeZ: 0.25, ...opts });
  const city = createTerrainLayer(scene, renderer, { zoom: 12, rows: 7, cols: 20, behind: 5, lift: -0.05, order: -5, fadeX: 0.15, fadeZ: 0.35, sea: 0.5, ...opts });
  return { far, near, city, attribution: TERRAIN_ATTRIBUTION, get stats() { return { far: far.stats, near: near.stats, city: city.stats }; },
    update(dt, camera, vis, apexX) { far.update(dt, camera, vis, apexX); near.update(dt, camera, vis, apexX); city.update(dt, camera, vis, apexX); }, dispose() { far.dispose(); near.dispose(); city.dispose(); } };
}
