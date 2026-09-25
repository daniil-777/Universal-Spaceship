// Soviet space stations (Mir / Salyut class) for the low passes over the Earth: a core module with the conical transition
// and the spherical docking node, a Kvant-1 style aft module, four radial modules on the node (Kvant-2 / Kristall /
// Spektr / Priroda), flat solar arrays with the dark-blue cell grid, a docked Soyuz at the forward port, whip antennas,
// dishes and a Sofora-like girder. Procedural geometry merged per material and instanced: four draw calls for any number.
import * as THREE from 'three';

function cellTexture() {                                  // solar-cell grid, dark blue with thin silver seams
  const c = document.createElement('canvas'); c.width = 128; c.height = 64; const x = c.getContext('2d');
  x.fillStyle = '#132a5e'; x.fillRect(0, 0, 128, 64); x.strokeStyle = 'rgba(200,215,235,0.75)'; x.lineWidth = 1;
  for (let i = 0; i <= 128; i += 16) { x.beginPath(); x.moveTo(i + 0.5, 0); x.lineTo(i + 0.5, 64); x.stroke(); }
  for (let j = 0; j <= 64; j += 16) { x.beginPath(); x.moveTo(0, j + 0.5); x.lineTo(128, j + 0.5); x.stroke(); }
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.wrapS = t.wrapT = THREE.RepeatWrapping; t.anisotropy = 4; return t;
}

const Q = Math.PI / 2;
const cyl = (rt, rb, h, seg = 20, open = false) => new THREE.CylinderGeometry(rt, rb, h, seg, 1, open);   // along y
const rod = (h, r = 0.011) => cyl(r, r, h, 5, true);
const cap = (r) => new THREE.SphereGeometry(r, 16, 5, 0, Math.PI * 2, 0, Math.PI * 0.27);              // shallow dish: cap around +y
const T = (g, x = 0, y = 0, z = 0, rx = 0, ry = 0, rz = 0) => g.rotateX(rx).rotateY(ry).rotateZ(rz).translate(x, y, z);   // rotate X→Y→Z, then move
function panel(len, wid, cell = 0.11) {                   // flat solar array: len along x, wid along z, cells tiled every `cell` units
  const g = new THREE.BoxGeometry(len, 0.024, wid), uv = g.attributes.uv, su = len / (8 * cell), sv = wid / (4 * cell);
  for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) * su, uv.getY(i) * sv);
  return g;
}
function truss(len, w = 0.05, step = 0.16) {              // three-rail lattice girder along +y (Sofora style)
  const parts = [], corner = (i) => [w * Math.cos(i * Math.PI * 2 / 3), w * Math.sin(i * Math.PI * 2 / 3)];
  for (let i = 0; i < 3; i++) { const [x, z] = corner(i); parts.push(T(rod(len), x, len / 2, z)); }
  for (let y = step; y < len; y += step) for (let i = 0; i < 3; i++) {
    const [x0, z0] = corner(i), [x1, z1] = corner(i + 1);
    parts.push(T(rod(w * Math.sqrt(3)), (x0 + x1) / 2, y, (z0 + z1) / 2, Q, Math.atan2(x1 - x0, z1 - z0)));
  }
  return parts;
}

function buildStation() {                                 // y is the long axis: +y forward (node, Soyuz), −y aft (Kvant-1)
  const G = { hull: [], blanket: [], foil: [], cells: [] }, add = (k, g) => G[k].push(g), NODE = 1.0;
  // core module: small-diameter work compartment, conical transition, large-diameter compartment, aft assembly section
  add('hull', T(cyl(0.21, 0.21, 0.56), 0, 0.5)); add('hull', T(cyl(0.21, 0.30, 0.28), 0, 0.08)); add('foil', T(cyl(0.31, 0.31, 0.05), 0, -0.07));
  add('hull', T(cyl(0.30, 0.30, 1.0), 0, -0.55)); add('blanket', T(cyl(0.306, 0.306, 0.28), 0, -0.86)); add('foil', T(cyl(0.23, 0.25, 0.16), 0, -1.13));
  // Kvant-1 aft module with its blanket band, gold aft cap and docking port
  add('hull', T(cyl(0.30, 0.30, 0.64), 0, -1.53)); add('blanket', T(cyl(0.306, 0.306, 0.2), 0, -1.62));
  add('foil', T(cyl(0.30, 0.18, 0.08), 0, -1.89)); add('hull', T(cyl(0.12, 0.12, 0.14), 0, -1.99));
  // docking node with the forward axial port
  add('hull', T(new THREE.SphereGeometry(0.28, 24, 16), 0, NODE)); add('hull', T(cyl(0.14, 0.14, 0.14), 0, NODE + 0.31));
  for (const s of [-1, 1]) {
    add('cells', T(panel(2.15, 0.55), s * 1.44, -0.6, 0, Q)); add('hull', T(rod(0.2, 0.03), s * 0.36, -0.6, 0, 0, 0, Q));   // core arrays (±x)
    add('cells', T(panel(1.2, 0.42), 0, -1.5, s * 0.96, 0, Q, Q)); add('hull', T(rod(0.2, 0.03), 0, -1.5, s * 0.36, Q));   // Kvant-1 arrays (±z)
    add('hull', T(rod(0.6), 0, 0.5, s * 0.45, Q));                                                                         // whip antennas
  }
  // Soyuz at the forward port: collar, orbital module, descent module, service module, gold aft rim, nozzle, two arrays
  add('blanket', T(cyl(0.1, 0.1, 0.1), 0, 1.43)); add('blanket', T(new THREE.SphereGeometry(0.17, 18, 12), 0, 1.62));
  add('blanket', T(cyl(0.13, 0.19, 0.28), 0, 1.92)); add('blanket', T(cyl(0.19, 0.19, 0.4), 0, 2.26));
  add('foil', T(cyl(0.19, 0.15, 0.05), 0, 2.48)); add('hull', T(cyl(0.05, 0.09, 0.08), 0, 2.54));
  for (const s of [-1, 1]) { add('cells', T(panel(0.7, 0.26), s * 0.57, 2.26, 0, Q)); add('hull', T(rod(0.34, 0.008), s * 0.3, 1.62, 0, 0, 0, Q)); }
  // four radial modules on the node: docking adapter, instrument section, step, main compartment with a blanket band and a
  // gold end cap, two arrays fore and aft; Kristall (+x) carries the docking module, Priroda (−x) a dish antenna
  for (const [rx, rz, alen, kind] of [[0, -Q, 0.65, 1], [0, Q, 0.65, 2], [Q, 0, 0.85, 0], [-Q, 0, 0.85, 0]]) {
    const put = (k, g) => add(k, T(g, 0, NODE, 0, rx, 0, rz));
    put('hull', T(cyl(0.15, 0.15, 0.2), 0, 0.32)); put('hull', T(cyl(0.21, 0.21, 0.4), 0, 0.6)); put('hull', T(cyl(0.28, 0.21, 0.12), 0, 0.86));
    put('hull', T(cyl(0.28, 0.28, 0.72), 0, 1.26)); put('blanket', T(cyl(0.286, 0.286, 0.22), 0, 1.06)); put('foil', T(cyl(0.14, 0.28, 0.1), 0, 1.67));
    for (const s of [-1, 1]) {
      const c = 0.34 + alen / 2, pn = panel(alen, 0.42), boom = rod(0.24, 0.02);
      put('cells', rz ? T(pn, s * c, 1.3, 0, Q) : T(pn, 0, 1.3, s * c, 0, Q, Q));
      put('hull', rz ? T(boom, s * 0.3, 1.3, 0, 0, 0, Q) : T(boom, 0, 1.3, s * 0.3, Q));
    }
    if (kind === 1) { put('blanket', T(cyl(0.16, 0.16, 0.26), 0, 1.84)); put('foil', T(cyl(0.18, 0.18, 0.05), 0, 1.99)); }
    if (kind === 2) { put('hull', T(rod(0.3, 0.02), 0, 1.78)); put('hull', T(cap(0.3), 0, 2.2, 0, Math.PI)); }
  }
  // Sofora girder on Kvant-1 (+x, raked aft) with the thruster block at its tip; the Altair dish on the −x side
  for (const g of truss(1.35)) add('hull', T(g, 0.28, -1.45, 0, 0, 0, -Q - 0.26));
  add('foil', T(new THREE.BoxGeometry(0.16, 0.1, 0.1), 1.6, -1.8, 0, 0, 0, -0.26));
  add('hull', T(cap(0.3), -0.86, -1.45, 0, 0, 0, -Q)); add('hull', T(rod(0.3, 0.02), -0.43, -1.45, 0, 0, 0, Q));
  const out = {}; for (const k in G) out[k] = mergeInto(G[k]).translate(0, -0.25, 0);     // centre the long axis on the origin
  return out;
}

export function createSatellites(scene, { max = 8 } = {}) {
  const geo = buildStation(), cellMap = cellTexture();
  const mats = {
    hull: new THREE.MeshStandardMaterial({ color: 0xe2e5e0, metalness: 0.3, roughness: 0.45, side: THREE.DoubleSide }),   // double-sided for the dishes
    blanket: new THREE.MeshStandardMaterial({ color: 0x7e8a78, metalness: 0.25, roughness: 0.75 }),                       // green-grey thermal blankets, Soyuz
    foil: new THREE.MeshStandardMaterial({ color: 0xc9a227, metalness: 0.85, roughness: 0.35 }),
    cells: new THREE.MeshStandardMaterial({ map: cellMap, metalness: 0.45, roughness: 0.35 }),
  };
  const meshes = Object.keys(geo).map((k) => { const im = new THREE.InstancedMesh(geo[k], mats[k], max); im.instanceMatrix.setUsage(THREE.DynamicDrawUsage); im.count = 0; im.frustumCulled = false; scene.add(im); return im; });
  const m = new THREE.Matrix4(), q = new THREE.Quaternion(), p = new THREE.Vector3(), sc = new THREE.Vector3();
  return {
    update(list) {                                         // list: [{ p: [x,y,z], q: [x,y,z,w], r }]; span ≈ 5 units at r = 1.1
      const n = Math.min(max, list.length);
      for (let i = 0; i < n; i++) { const s = list[i], k = s.r / 2.2; p.set(s.p[0], s.p[1], s.p[2]); q.set(s.q[0], s.q[1], s.q[2], s.q[3]); sc.set(k, k, k); m.compose(p, q, sc); for (const im of meshes) im.setMatrixAt(i, m); }
      for (const im of meshes) { im.count = n; im.instanceMatrix.needsUpdate = true; im.visible = n > 0; }
    },
    setMono() {},
    dispose() { for (const im of meshes) { scene.remove(im); im.geometry.dispose(); im.material.dispose(); } cellMap.dispose(); },
  };
}

function mergeInto(geos) {                                 // minimal non-indexed merge (positions, normals, uvs)
  const parts = geos.map((g) => g.toNonIndexed()), n = parts.reduce((s, g) => s + g.attributes.position.count, 0);
  const pos = new Float32Array(n * 3), nor = new Float32Array(n * 3), uv = new Float32Array(n * 2); let o = 0;
  for (const g of parts) { const c = g.attributes.position.count; pos.set(g.attributes.position.array, o * 3); nor.set(g.attributes.normal.array, o * 3); if (g.attributes.uv) uv.set(g.attributes.uv.array, o * 2); o += c; g.dispose(); }
  const out = new THREE.BufferGeometry(); out.setAttribute('position', new THREE.BufferAttribute(pos, 3)); out.setAttribute('normal', new THREE.BufferAttribute(nor, 3)); out.setAttribute('uv', new THREE.BufferAttribute(uv, 2)); return out;
}
