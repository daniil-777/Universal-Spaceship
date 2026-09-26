// Skyline renderer: every building of the city field is one instance of a unit shape — box (also tapered in the vertex
// shader), round tower, octagonal tower — three copies along the periodic corridor, drawn by a facade shader: curtain-wall
// styles (grid / ribbon / piers), glass reflecting sky, horizon and ground, stone with recessed windows, spandrels, dark
// lobby plinths, cornices, helipads, canyon shading from the height field, real cast shadows (shadow field), haze.
// Three instanced draw calls per city, no textures. Same update/setMono/dispose API as the mountains.
import * as THREE from 'three';
import { SHADOW_GLSL } from './shadowfield.js';
import { createFacadeAtlas, TILE_UNITS } from './facades.js';
import { T } from './cityfield.js';

const VERT = /* glsl */`attribute vec3 aTint; attribute vec4 aInfo; attribute vec3 aExtra; uniform float uApexX;   // aInfo: width, height, depth, kind; aExtra: taper, style, facade tile
  varying vec3 vP, vN, vL, vTint, vExtra; varying vec4 vInfo; varying float vFlatY, vAng;
  void main(){ vec4 mp = instanceMatrix * vec4(position, 1.0); vec4 w = modelMatrix * mp; vFlatY = w.y;
    w.y -= ((w.x - uApexX) * (w.x - uApexX) + w.z * w.z) / 2800.0; vP = w.xyz;                   // follow the Earth's curvature like the imagery strips
    vN = normalize(mat3(modelMatrix) * normalize(normal / aInfo.xyz));                          // normals under a non-uniform scale: inverse scale, not the scale
    vL = vec3(position.x + 0.5, position.y, position.z + 0.5); vAng = atan(position.z, position.x); vTint = aTint; vInfo = aInfo; vExtra = aExtra;
    gl_Position = projectionMatrix * viewMatrix * w; }`;
const FRAG = /* glsl */`uniform vec3 uSun, uHaze; uniform float uVis, uShape; uniform sampler2DArray uAtlas; uniform vec2 uTileUnits[8];
  ${SHADOW_GLSL}
  varying vec3 vP, vN, vL, vTint, vExtra; varying vec4 vInfo; varying float vFlatY, vAng;
  float h21(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
  void main(){
    vec3 N = normalize(vN); float w = vInfo.x, h = vInfo.y, d = vInfo.z, kind = vInfo.w, style = vExtra.y;
    float dist = length(cameraPosition - vP);
    float perim = 3.1416 * (w + d) * 0.5, up = vL.y * h;
    float along = uShape == 1.0 || uShape == 2.0 ? (vAng / 6.28318 + 0.5) * perim : (abs(N.x) > 0.5 ? (vL.z - 0.5) * d : (vL.x - 0.5) * w) / (uShape == 3.0 ? mix(1.0, vExtra.x, vL.y) : 1.0);   // facade coordinate in world units (around round towers; rescaled up tapered ones)
    int ti = int(vExtra.z + 0.5); vec2 tu = uTileUnits[ti];                                        // the building's facade tile and its repeat size in world units
    vec4 tx = texture(uAtlas, vec3(along / tu.x + style * 3.0, (h - up) / tu.y, float(ti)));       // drawn facade: frames, sills, mortar, panels… A = glass
    float win = tx.a;
    vec3 alb = tx.rgb * mix(vec3(1.0), vTint * 1.7, kind == 1.0 ? 0.25 : 0.55);                   // the building's tint colours the drawn facade
    float sv = dist < 300.0 ? sunVis(vec3(vP.x, vFlatY, vP.z) + N * 0.4) : 1.0;                 // cast shadows from the neighbours
    float ndl = max(dot(N, uSun), 0.0) * sv;
    vec3 V = normalize(cameraPosition - vP), R = reflect(-V, N); float spec = pow(max(dot(R, uSun), 0.0), 70.0) * sv, fres = pow(1.0 - max(dot(N, V), 0.0), 3.0);
    vec3 zen = vec3(0.36, 0.55, 0.90), hor = vec3(0.70, 0.78, 0.88), gnd = vec3(0.40, 0.38, 0.32);
    vec3 env = R.y > 0.0 ? mix(hor, zen, pow(R.y, 0.6)) : mix(hor, gnd, min(1.0, -R.y * 3.0));   // what the glass reflects: sky above, horizon, ground below
    float nb = fieldH(vP.xz + N.xz * 3.0) - vFlatY;                                              // a taller neighbour right next door shades this face (canyon AO)
    float ao = 1.0 - 0.35 * clamp(nb / 12.0, 0.0, 1.0);
    vec3 tint = vTint, col;
    if (N.y > 0.5) {                                                                           // roofs: matte, a faint service grid, darker rim, sometimes a helipad
      col = tint * 0.42 * (0.4 + 0.6 * ndl) + vec3(0.05);
      col *= 1.0 - 0.22 * min(1.0, step(0.93, fract(vL.x * w / 2.0)) + step(0.93, fract(vL.z * d / 2.0)));
      col *= 0.75 + 0.25 * smoothstep(0.0, 0.08, min(min(vL.x, 1.0 - vL.x) * w, min(vL.z, 1.0 - vL.z) * d) * 2.0);
      if (kind != 1.0 && style > 0.72 && min(w, d) > 4.0) { float rr = length((vL.xz - 0.5) * vec2(w, d)); col = mix(col, vec3(0.85), 0.8 * smoothstep(0.1, 0.0, abs(rr - 1.1)) + 0.6 * smoothstep(0.06, 0.0, abs(rr - 0.45)) * step(abs(atan(vL.z - 0.5, vL.x - 0.5)), 0.5)); }
    } else if (kind == 3.0) { col = tint * (0.35 + 0.75 * ndl) + spec * 0.6; }                  // spires and antennas: bright metal
    else {                                                                                     // walls from the drawn facade; panes add the reflection of sky, horizon, ground and sun
      vec3 wall = alb * (0.30 + 0.70 * ndl) + alb * 0.20 * (0.5 + 0.5 * N.y);
      vec3 pane = alb * (0.35 + 0.45 * ndl) + env * (0.22 + 0.55 * fres) * (kind == 1.0 ? 0.45 : 0.85) + spec * vec3(1.0, 0.95, 0.85) * (kind == 4.0 ? 1.2 : 0.9);
      col = mix(wall, pane, win);
      if (kind == 1.0) col = mix(col, wall * 1.15, smoothstep(0.7, 0.0, h - up) * 0.6);          // cornice band on stone and brick
      else col = mix(col, tint * 0.22 * (0.5 + 0.5 * ndl) + env * 0.15 * fres, smoothstep(1.8, 1.2, up));   // dark glass lobby plinth
    }
    col *= mix(0.55, 1.0, smoothstep(0.0, 6.0, up)) * ao;                                      // street level and canyon walls sit in shade
    col = mix(col, uHaze, 0.85 * (1.0 - exp(-dist / 700.0)));
    gl_FragColor = vec4(col, uVis); }`;

const TAPERS = [0.4, 0.6, 0.8];
function unitShapes() {                                    // all stand on y = 0 and fit the unit cube (real geometry, so CPU raycasts and the field agree)
  const box = new THREE.BoxGeometry(1, 1, 1).translate(0, 0.5, 0);
  const round = new THREE.CylinderGeometry(0.5, 0.5, 1, 28, 1).translate(0, 0.5, 0);
  const oct = new THREE.CylinderGeometry(0.5412, 0.5412, 1, 8, 1).rotateY(Math.PI / 8).translate(0, 0.5, 0);   // flats at ±0.5: |x|,|z| ≤ 0.5 and |x|+|z| ≤ 0.707
  const frusta = TAPERS.map((t) => {                       // square frusta from a box: the top ring scaled in, side normals tilted to match
    const g = new THREE.BoxGeometry(1, 1, 1).translate(0, 0.5, 0), pos = g.attributes.position, nor = g.attributes.normal, tilt = 0.5 * (1 - t), n = new THREE.Vector3();
    for (let i = 0; i < pos.count; i++) {
      if (pos.getY(i) > 0.5) { pos.setX(i, pos.getX(i) * t); pos.setZ(i, pos.getZ(i) * t); }
      n.fromBufferAttribute(nor, i); if (Math.abs(n.y) < 0.5) { n.y = tilt * (Math.abs(n.x) + Math.abs(n.z)); n.normalize(); nor.setXYZ(i, n.x, n.y, n.z); }
    }
    return g;
  });
  return [box, round, oct, ...frusta];
}

export function createCity(scene, cf, { sunDir = new THREE.Vector3(0.6, 0.35, -0.7), shadow = null } = {}) {
  const { buildings, y0, PERIOD } = cf, COPIES = 3, geos = unitShapes();
  if (!buildings.length) return { mesh: null, meshes: [], update() {}, setMono() {}, dispose() {} };   // a route whose Meshy districts cover the whole period has no procedural blocks
  const facades = createFacadeAtlas(), TILES = facades.TILES;
  const tileFor = (b) => {                                 // which drawn facade a building wears
    if (b.kind === 4) return TILES.premium;
    if (b.kind === 0) { const s = ((b.x * 12.9898 + b.z * 78.233 + b.h) % 1 + 1) % 1; return s < 0.4 ? TILES.grid : s < 0.7 ? TILES.ribbon : TILES.piers; }
    if (b.tint === T.brick) return TILES.brick; if (b.tint === T.stone || b.tint === T.cream) return TILES.stone; if (b.tint === T.slab) return TILES.panel; return TILES.concrete;
  };
  const shared = { uSun: { value: sunDir.clone() }, uHaze: { value: new THREE.Color(0.62, 0.74, 0.9) }, uVis: { value: 0 }, uApexX: { value: 0 }, uAtlas: { value: facades.texture }, uTileUnits: { value: TILE_UNITS.map(([w, h]) => new THREE.Vector2(w, h)) }, ...(shadow ? shadow.uniforms : {}) };
  const groups = geos.map(() => []);                       // buildings per geometry: box, round, octagon, frusta by taper
  for (const b of buildings) groups[b.shape === 3 ? 3 + Math.max(0, TAPERS.indexOf(b.taper)) : (b.shape || 0)].push(b);
  const meshes = [], mats = [], m = new THREE.Matrix4(), p = new THREE.Vector3(), q = new THREE.Quaternion(), s = new THREE.Vector3();
  groups.forEach((list, gi) => {
    if (!list.length) return;
    const n = list.length, geo = geos[gi].clone(), tint = new Float32Array(n * COPIES * 3), info = new Float32Array(n * COPIES * 4), extra = new Float32Array(n * COPIES * 3);
    const mat = new THREE.ShaderMaterial({ vertexShader: VERT, fragmentShader: FRAG, transparent: true, depthWrite: true, uniforms: { ...shared, uShape: { value: Math.min(3, gi) } } });
    const mesh = new THREE.InstancedMesh(geo, mat, n * COPIES); mesh.frustumCulled = false; mesh.renderOrder = -4; mesh.visible = false;
    for (let c = 0; c < COPIES; c++) for (let i = 0; i < n; i++) {
      const b = list[i], k = c * n + i;
      p.set(b.x + (c - 1) * PERIOD, y0 + b.y, b.z); s.set(b.w, b.h, b.d); m.compose(p, q, s); mesh.setMatrixAt(k, m);
      tint[k * 3] = b.tint[0]; tint[k * 3 + 1] = b.tint[1]; tint[k * 3 + 2] = b.tint[2];
      info[k * 4] = b.w; info[k * 4 + 1] = b.h; info[k * 4 + 2] = b.d; info[k * 4 + 3] = b.kind;
      extra[k * 3] = b.shape === 3 ? b.taper : 1; extra[k * 3 + 1] = ((b.x * 12.9898 + b.z * 78.233 + b.h) % 1 + 1) % 1; extra[k * 3 + 2] = tileFor(b);   // taper, style, facade tile
    }
    geo.setAttribute('aTint', new THREE.InstancedBufferAttribute(tint, 3)); geo.setAttribute('aInfo', new THREE.InstancedBufferAttribute(info, 4)); geo.setAttribute('aExtra', new THREE.InstancedBufferAttribute(extra, 3));
    mesh.instanceMatrix.needsUpdate = true; scene.add(mesh); meshes.push(mesh); mats.push(mat);
  });
  for (const g of geos) g.dispose();
  return {
    mesh: meshes[0], meshes, city: cf.city,
    update(dt, camera, vis, sunDirWorld, apexX) {          // vis 0..1; the copies follow the camera's period so the wrap is invisible; apexX = the ground's curvature apex
      shared.uVis.value = vis; shared.uApexX.value = apexX ?? camera.position.x;
      const on = vis > 0.01, base = Math.round(camera.position.x / PERIOD) * PERIOD;
      for (const mesh of meshes) { mesh.visible = on; mesh.position.x = base; }
      if (on && sunDirWorld) shared.uSun.value.copy(sunDirWorld).normalize();
    },
    setMono() {},
    dispose() { for (const mesh of meshes) { scene.remove(mesh); mesh.geometry.dispose(); } for (const mat of mats) mat.dispose(); facades.texture.dispose(); },
  };
}
