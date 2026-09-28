// The land around the spaceport: 240 km of farmland (its edge beyond the haze) drawn by a shader in world space — an irregular patchwork of fields
// (wheat, barley, plough, meadow, rape in flower) with hedgerows along their edges, two roads and a river — and the mown
// airfield inside its perimeter, striped by the mowers; instanced trees along the nearer hedges and a wood give the eye
// the depth cues a pilot judges the flare by.
import * as THREE from 'three';
import { surface } from './airportmesh.js';
import { mulberry32 } from '../mathx.js';

const FIELDS = /* glsl */`{
  vec2 p = vWP.xz;
  vec2 q = p / 520.0 + 0.35 * vec2(fbm(p / 900.0), fbm(p / 900.0 + 7.3));   // warped cells: fields are not a grid
  vec2 c = floor(q), f = fract(q); float id = h21(c), e = min(min(f.x, 1.0 - f.x), min(f.y, 1.0 - f.y));
  vec3 crop = id < 0.25 ? vec3(0.42, 0.52, 0.2) : id < 0.45 ? vec3(0.7, 0.62, 0.32) : id < 0.6 ? vec3(0.43, 0.33, 0.22) : id < 0.8 ? vec3(0.34, 0.47, 0.2) : id < 0.88 ? vec3(0.78, 0.74, 0.25) : vec3(0.5, 0.55, 0.28);
  crop *= 0.85 + 0.25 * fbm(p / 18.0 + id * 40.0);
  crop *= 1.0 - 0.08 * step(0.5, fract((dot(p, vec2(cos(id * 6.3), sin(id * 6.3)))) / 9.0));   // drill rows
  float hedge = 1.0 - smoothstep(0.004, 0.014, e); crop = mix(crop, vec3(0.16, 0.24, 0.1), hedge * 0.85);
  float road = min(abs(p.y - 1400.0 - 60.0 * sin(p.x / 1700.0)), abs(p.x + 3000.0 + 90.0 * sin(p.y / 2100.0)));
  crop = mix(crop, vec3(0.3, 0.3, 0.31), 1.0 - smoothstep(4.0, 6.0, road));
  float river = abs(p.y + 2600.0 + 380.0 * sin(p.x / 2600.0) + 120.0 * sin(p.x / 700.0));
  crop = mix(crop, vec3(0.16, 0.23, 0.26), 1.0 - smoothstep(22.0, 30.0, river));
  bool field = p.x > -1500.0 && p.x < 5600.0 && p.y > -420.0 && p.y < 820.0;                  // the airfield: mown grass
  if (field) { vec3 grass = vec3(0.33, 0.45, 0.19) * (0.9 + 0.2 * fbm(p / 6.0)); grass *= 1.0 + 0.025 * sign(sin(p.y / 7.0)); crop = grass; }
  diffuseColor.rgb = pow(crop, vec3(2.2));                        // the palette above is in display (sRGB) terms; lighting is linear
}`;

export function createGround(AP) {
  const group = new THREE.Group(), g = new THREE.PlaneGeometry(240000, 240000, 1, 1); g.rotateX(-Math.PI / 2);
  const mat = surface(0xffffff, 0.95, FIELDS), ground = new THREE.Mesh(g, mat); ground.position.y = -0.03; ground.receiveShadow = true; group.add(ground);
  const rng = mulberry32(21), n = 2600, trunk = new THREE.CylinderGeometry(0.25, 0.35, 3, 5), crown = new THREE.IcosahedronGeometry(3.2, 0);
  const tm = new THREE.InstancedMesh(trunk, new THREE.MeshStandardMaterial({ color: 0x4a3a2a, roughness: 1 }), n), cm = new THREE.InstancedMesh(crown, new THREE.MeshStandardMaterial({ color: 0x2f4a22, roughness: 1, flatShading: true }), n);
  const o = new THREE.Object3D(); let k = 0;
  const place = (x, z, s) => { if (k >= n || (x > -1500 && x < 5600 && z > -420 && z < 820)) return; o.position.set(x, 1.5 * s, z); o.rotation.set(0, rng() * 6.3, 0); o.scale.set(s, s, s); o.updateMatrix(); tm.setMatrixAt(k, o.matrix);
    o.position.y = (3 + 2.4 * rng()) * s; o.scale.set(s * (0.8 + 0.5 * rng()), s * (0.9 + 0.6 * rng()), s * (0.8 + 0.5 * rng())); o.updateMatrix(); cm.setMatrixAt(k++, o.matrix); };
  for (let i = 0; i < 1500; i++) { const side = rng() < 0.5 ? -1 : 1, x = -6000 + rng() * 14000; place(x, side * (460 + rng() * 1800) + (side > 0 ? 380 : 0), 0.8 + 0.6 * rng()); }   // hedgerow trees either side
  for (let i = 0; i < 1100; i++) { const a = rng() * 6.28, r = Math.sqrt(rng()) * 700; place(-4200 + Math.cos(a) * r, -1300 + Math.sin(a) * r * 0.6, 0.9 + 0.7 * rng()); }   // a wood under the approach
  tm.count = cm.count = k; tm.castShadow = cm.castShadow = true; group.add(tm, cm);
  return { group };
}
