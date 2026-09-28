// The airport in three.js from its surveyed layout (src/landing/airport.js): asphalt with rubber in the touchdown zones,
// the ICAO paint as merged quads (and the runway numbers drawn as decals), shoulders, blast pads, taxiways with their
// yellow lines and hold-short bars, a jointed concrete apron with stand lead-in lines, the terminal's curtain wall and
// jet bridges, the control tower, hangars, fire station, cargo shed, fuel farm, the turning ASR radar, the ILS arrays,
// PAPI boxes, approach-light bars, a windsock that flies with the wind and parked airliners.
import * as THREE from 'three';
import { airlinerParts } from '../aircraft.js';

const NOISE = /* glsl */`
  float h21(vec2 p) { p = fract(p * vec2(233.34, 851.73)); p += dot(p, p + 23.45); return fract(p.x * p.y); }
  float vn(vec2 p) { vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f); return mix(mix(h21(i), h21(i + vec2(1, 0)), f.x), mix(h21(i + vec2(0, 1)), h21(i + vec2(1, 1)), f.x), f.y); }
  float fbm(vec2 p) { float s = 0.0, a = 0.5; for (int i = 0; i < 4; i++) { s += a * vn(p); p *= 2.03; a *= 0.5; } return s; }`;
export function surface(color, rough, pattern, glow = '') {   // a standard material whose albedo (and, optionally, glow) is patterned in world space
  const m = new THREE.MeshStandardMaterial({ color, roughness: rough, metalness: 0 });
  m.onBeforeCompile = (sh) => {
    sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nvarying vec3 vWP;').replace('#include <worldpos_vertex>', '#include <worldpos_vertex>\nvWP = (modelMatrix * vec4(transformed, 1.0)).xyz;');
    sh.fragmentShader = sh.fragmentShader.replace('#include <common>', '#include <common>\nvarying vec3 vWP;' + NOISE).replace('#include <color_fragment>', '#include <color_fragment>\n' + pattern).replace('#include <emissivemap_fragment>', '#include <emissivemap_fragment>\n' + glow);
  };
  m.customProgramCacheKey = () => 'apx-surface:' + pattern + glow;   // same onBeforeCompile source, different pattern: without a key three.js would share one program
  return m;
}
const ASPHALT = /* glsl */`{ vec2 p = vWP.xz; float g = fbm(p * 0.9) * 0.18 + fbm(p * 7.0) * 0.06;
  float tz = max(1.0 - smoothstep(900.0, 1300.0, p.x) - (1.0 - smoothstep(60.0, 150.0, p.x)), 1.0 - smoothstep(900.0, 1300.0, 4000.0 - p.x) - (1.0 - smoothstep(60.0, 150.0, 4000.0 - p.x)));
  float track = exp(-pow((abs(p.y) - 4.6) / 2.2, 2.0)) + 0.6 * exp(-p.y * p.y / 3.0);                  // tyre tracks: main gear and nose wheel
  float rubber = clamp(tz, 0.0, 1.0) * track * (0.55 + 0.45 * fbm(p * vec2(0.05, 0.6)));
  diffuseColor.rgb *= (0.86 + g) * (1.0 - 0.72 * rubber); }`;
const CONCRETE = /* glsl */`{ vec2 p = vWP.xz; float g = fbm(p * 1.3) * 0.1 + fbm(p * 9.0) * 0.04;
  vec2 j = abs(fract(p / 7.5 + 0.5) - 0.5) * 7.5; float joint = (1.0 - smoothstep(0.0, 0.06, min(j.x, j.y))) * 0.25;
  diffuseColor.rgb *= (0.9 + g - joint) * (0.93 + 0.07 * h21(floor(p / 7.5))); }`;
const PAINT = /* glsl */`{ vec2 p = vWP.xz; diffuseColor.rgb *= 0.82 + 0.18 * fbm(p * 1.7) - 0.25 * step(0.8, fbm(p * 4.0)); }`;
const GLASS_GLOW = (k) => /* glsl */`{ vec2 p = vec2(vWP.x + vWP.z, vWP.y), cell = floor(vec2(p.x / 3.0, p.y / 4.2));   // at night: lit panes, a random half
  float frame = max(step(0.92, fract(p.x / 3.0)), step(0.9, fract(p.y / 4.2)));
  totalEmissiveRadiance += vec3(1.0, 0.78, 0.5) * ${k.toFixed(3)} * step(0.5, h21(cell + 7.0)) * (1.0 - frame); }`;
const GLASS = /* glsl */`{ vec2 p = vec2(vWP.x + vWP.z, vWP.y); float mx = step(0.92, fract(p.x / 3.0)), my = step(0.9, fract(p.y / 4.2));
  diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.22, 0.23, 0.25), max(mx, my)); }`;

function quads(rects, y) {                                 // axis-aligned rectangles in the ground plane → one geometry
  const pos = new Float32Array(rects.length * 12), idx = [];
  rects.forEach((r, i) => { pos.set([r.x0, y, r.z0, r.x1, y, r.z0, r.x1, y, r.z1, r.x0, y, r.z1], i * 12); const k = i * 4; idx.push(k, k + 2, k + 1, k, k + 3, k + 2); });
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setIndex(idx); g.computeVertexNormals(); return g;
}
function ribbon(pts, width, y) {                            // a straight strip (taxiway, line) of the given width, facing up
  const [a, b] = pts, dx = b[0] - a[0], dz = b[1] - a[1], L = Math.hypot(dx, dz), nx = -dz / L * width / 2, nz = dx / L * width / 2;
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(new Float32Array([a[0] + nx, y, a[1] + nz, b[0] + nx, y, b[1] + nz, b[0] - nx, y, b[1] - nz, a[0] - nx, y, a[1] - nz]), 3));
  g.setIndex([0, 1, 2, 0, 2, 3]); g.computeVertexNormals(); if (g.attributes.normal.array[1] < 0) { g.setIndex([0, 2, 1, 0, 3, 2]); g.computeVertexNormals(); } return g;
}
function digits(text, up) {                                 // the runway designation as a painted decal
  const c = document.createElement('canvas'); c.width = 256; c.height = 256; const g = c.getContext('2d');
  g.fillStyle = '#fff'; g.font = '700 236px "DIN Alternate", "Arial Narrow", Arial, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(text, 128, 138);
  const t = new THREE.CanvasTexture(c); t.anisotropy = 8; t.colorSpace = THREE.SRGBColorSpace;
  const m = new THREE.Mesh(new THREE.PlaneGeometry(9, 9), new THREE.MeshStandardMaterial({ map: t, transparent: true, roughness: 0.7, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2 }));
  m.rotation.x = -Math.PI / 2; m.rotation.z = up > 0 ? -Math.PI / 2 : Math.PI / 2; return m;   // the digits' tops point down the runway, read on approach
}

export function createAirport(AP, { night = 0, wet = false } = {}) {   // wet: rain on the runway — darker, glossy asphalt
  const group = new THREE.Group(), R = AP.runway, L = R.length, HW = R.width / 2;
  const asphalt = surface(wet ? 0x333538 : 0x4a4c4f, wet ? 0.34 : 0.92, ASPHALT), shoulder = surface(wet ? 0x46484a : 0x5b5d5e, wet ? 0.5 : 0.95, ASPHALT), concrete = surface(0x9d9c97, 0.88, CONCRETE);
  const white = surface(0xf1f0ea, 0.7, PAINT), yellow = surface(0xe2b43a, 0.7, PAINT);
  for (const m of [white, yellow]) { m.polygonOffset = true; m.polygonOffsetFactor = -2; m.polygonOffsetUnits = -2; }
  const add = (geo, mat) => { const m = new THREE.Mesh(geo, mat); m.receiveShadow = true; group.add(m); return m; };
  add(quads([{ x0: 0, x1: L, z0: -HW, z1: HW }], 0.0), asphalt);
  add(quads([{ x0: -R.blast, x1: 0, z0: -HW, z1: HW }, { x0: L, x1: L + R.blast, z0: -HW, z1: HW }, { x0: -R.blast, x1: L + R.blast, z0: HW, z1: HW + R.shoulder }, { x0: -R.blast, x1: L + R.blast, z0: -HW - R.shoulder, z1: -HW }], -0.01), shoulder);
  add(quads(AP.markings.filter((m) => m.color === 'white' && m.kind !== 'designation'), 0.02), white);
  for (const c of AP.markings.filter((m) => m.kind === 'chevron')) {   // blast pads: yellow chevrons pointing at the threshold
    const dir = c.end === 26 ? 1 : -1, xr = dir > 0 ? c.x0 : c.x1, e = HW - 1.5;
    for (const s of [1, -1]) add(ribbon([[xr, s * e], [xr + dir * e * 0.5, 0]], 0.9, 0.02), yellow);
  }
  for (const d of AP.markings.filter((m) => m.kind === 'designation')) { const m = digits(d.text, d.up); m.position.set((d.x0 + d.x1) / 2, 0.025, (d.z0 + d.z1) / 2); group.add(m); }
  for (const t of AP.taxiways) {                            // asphalt, a yellow centreline, hold-short bars 90 m from the runway centreline
    add(ribbon(t.pts, t.width, -0.005), asphalt); add(ribbon(t.pts, 0.3, 0.015), yellow);
    if (t.kind === 'link' || t.kind === 'rapid') { const [a, b] = t.pts, d = Math.hypot(b[0] - a[0], b[1] - a[1]), u = [(b[0] - a[0]) / d, (b[1] - a[1]) / d], s = 90 / Math.abs(u[1]);
      for (const off of [0, 0.9, 2.4, 3.3]) { const c = [a[0] + u[0] * (s + off), a[1] + u[1] * (s + off)]; add(ribbon([[c[0] - u[1] * t.width / 2, c[1] + u[0] * t.width / 2], [c[0] + u[1] * t.width / 2, c[1] - u[0] * t.width / 2]], 0.3, 0.016), yellow); } }
  }
  const ap = AP.apron; add(quads([{ x0: ap.x0, x1: ap.x1, z0: ap.z0, z1: ap.z1 }], -0.004), concrete);
  for (const s of AP.stands) add(ribbon([[s.x, ap.z0 + 10], [s.x, s.z + 20]], 0.3, 0.012), yellow);
  // buildings
  const box = (w, h, d, mat, x, y, z) => { const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat); m.position.set(x, y, z); m.castShadow = m.receiveShadow = true; group.add(m); return m; };
  const wall = new THREE.MeshStandardMaterial({ color: 0xc9c6bd, roughness: 0.85 }), roof = new THREE.MeshStandardMaterial({ color: 0x6d7174, roughness: 0.7, metalness: 0.2 });
  const glass = surface(0x5a7390, 0.15, GLASS, night > 0.05 ? GLASS_GLOW(1.8 * night) : ''); glass.metalness = 0.55;
  const metal = new THREE.MeshStandardMaterial({ color: 0xb8bcbf, roughness: 0.45, metalness: 0.6 }), red = new THREE.MeshStandardMaterial({ color: 0xb3261e, roughness: 0.6 }), dark = new THREE.MeshStandardMaterial({ color: 0x2b2b2b, roughness: 0.6 });
  let radar = null;
  for (const b of AP.buildings) {
    if (b.kind === 'terminal') { box(b.w, b.h, b.d, glass, b.x, b.h / 2, b.z); box(b.w + 6, 1.6, b.d + 6, roof, b.x, b.h + 0.8, b.z);
      for (const s of AP.stands) if (AP.stands.indexOf(s) % 3 !== 2) box(3.5, 3.2, 28, metal, s.x + 5, 5, b.z - b.d / 2 - 14); }   // jet bridges to the forward left doors
    else if (b.kind === 'tower') { const shaft = new THREE.Mesh(new THREE.CylinderGeometry(3.2, 4.2, b.h - 8, 20), wall); shaft.position.set(b.x, (b.h - 8) / 2, b.z); group.add(shaft);
      const cab = new THREE.Mesh(new THREE.CylinderGeometry(7.5, 6.2, 5, 8), glass); cab.position.set(b.x, b.h - 5.5, b.z); group.add(cab);
      const cap = new THREE.Mesh(new THREE.CylinderGeometry(8.2, 8.2, 1.5, 8), roof); cap.position.set(b.x, b.h - 2.2, b.z); group.add(cap);
      box(0.4, 6, 0.4, metal, b.x, b.h + 1.5, b.z); shaft.castShadow = cab.castShadow = true; }
    else if (b.kind === 'hangar') { const arch = new THREE.Mesh(new THREE.CylinderGeometry(b.w / 2, b.w / 2, b.d, 24, 1, false, -Math.PI / 2, Math.PI), roof);   // a barrel vault spanning x, its length along z
      arch.rotation.x = -Math.PI / 2; arch.scale.set(1, 1, b.h / (b.w / 2)); arch.position.set(b.x, 0, b.z); arch.castShadow = arch.receiveShadow = true; group.add(arch);
      box(b.w * 0.6, b.h * 0.75, 0.6, metal, b.x, b.h * 0.375, b.z - b.d / 2 - 0.3); }   // the doors in the gable that faces the runway
    else if (b.kind === 'radar') { box(2, b.h, 2, metal, b.x, b.h / 2, b.z); radar = box(9, 2.2, 0.5, metal, b.x, b.h + 1.5, b.z); }
    else if (b.kind === 'tanks') { for (let k = 0; k < 4; k++) { const t = new THREE.Mesh(new THREE.CylinderGeometry(11, 11, b.h, 28), wall); t.position.set(b.x - 25 + (k % 2) * 50, b.h / 2, b.z - 15 + (k >> 1) * 30); t.castShadow = true; group.add(t); } }
    else if (b.kind === 'fire') { box(b.w, b.h, b.d, red, b.x, b.h / 2, b.z); box(b.w + 2, 0.8, b.d + 2, roof, b.x, b.h + 0.4, b.z); }
    else { box(b.w, b.h, b.d, wall, b.x, b.h / 2, b.z); box(b.w + 2, 0.8, b.d + 2, roof, b.x, b.h + 0.4, b.z); }
  }
  // ILS: the localizer array beyond the far end, the glideslope mast and shelter; PAPI boxes; approach-light bars
  for (let k = -10; k <= 10; k++) box(0.25, 3.2, 0.25, metal, AP.ils.loc.x, 1.6, AP.ils.loc.z + k * 2.2);
  box(1.2, 1.2, 46, red, AP.ils.loc.x - 0.8, 0.6, AP.ils.loc.z);
  box(0.6, 16, 0.6, red, AP.ils.gs.x, 8, AP.ils.gs.z); box(4, 3, 3, wall, AP.ils.gs.x + 4, 1.5, AP.ils.gs.z + 3);
  for (const u of AP.papi) box(1.6, 0.9, 1.1, dark, u.x, 0.45, u.z);
  const bars = [...new Set(AP.lights.filter((l) => l.kind === 'approach').map((l) => l.x))], bar = new THREE.InstancedMesh(new THREE.BoxGeometry(0.25, 0.3, 4.6), metal, bars.length), m4 = new THREE.Matrix4();
  bars.forEach((x, i) => bar.setMatrixAt(i, m4.makeTranslation(x, 0.2, 0))); group.add(bar);
  // apron floodlight masts (25 m) along the apron's taxiway edge: lamp heads, and at night the pools of light they throw
  const head = new THREE.MeshStandardMaterial({ color: 0x333333, emissive: 0xffe2b0, emissiveIntensity: 4 * night }), pool = night > 0.05 ? (() => {
    const c = document.createElement('canvas'); c.width = c.height = 128; const g = c.getContext('2d'), r = g.createRadialGradient(64, 64, 0, 64, 64, 64);
    r.addColorStop(0, 'rgba(255,226,180,0.3)'); r.addColorStop(0.55, 'rgba(255,214,160,0.1)'); r.addColorStop(1, 'rgba(255,210,150,0)'); g.fillStyle = r; g.fillRect(0, 0, 128, 128);
    return new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, opacity: Math.min(1, night), polygonOffset: true, polygonOffsetFactor: -3 }); })() : null;
  for (let k = 0; k < 6; k++) { const x = ap.x0 + 40 + k * (ap.x1 - ap.x0 - 80) / 5, z = ap.z0 + 12; box(0.5, 25, 0.5, metal, x, 12.5, z); box(3.2, 0.8, 1.2, head, x, 25.2, z + 0.6);
    if (pool) { const q = new THREE.Mesh(new THREE.PlaneGeometry(120, 120), pool); q.rotation.x = -Math.PI / 2; q.position.set(x, 0.06, z + 45); group.add(q); } }
  // the windsock: a pole and a frustum that swings round and fills with the wind (the mouth at the pole)
  const sock = new THREE.Group(), cone = new THREE.Mesh(new THREE.CylinderGeometry(0.45, 0.9, 4.5, 16, 1, true), new THREE.MeshStandardMaterial({ color: 0xff6a1a, roughness: 0.8, side: THREE.DoubleSide }));
  cone.rotation.z = -Math.PI / 2; cone.position.x = 2.25; sock.add(cone); sock.position.set(AP.windsock.x, AP.windsock.h, AP.windsock.z); group.add(sock); box(0.25, AP.windsock.h, 0.25, metal, AP.windsock.x, AP.windsock.h / 2, AP.windsock.z);
  // parked airliners at the stands (the corridor's procedural Airbus, ×11.75 = 37.6 m) on their gear, nose to the terminal
  const parts = airlinerParts().parts, parked = AP.stands.filter((_, i) => i % 3 !== 2), o = new THREE.Object3D();
  for (const { geo, mat } of parts) { const im = new THREE.InstancedMesh(geo, mat, parked.length); parked.forEach((s, i) => { o.position.set(s.x, 3.4, s.z - 6); o.rotation.set(0, -Math.PI / 2, 0); o.scale.setScalar(11.75); o.updateMatrix(); im.setMatrixAt(i, o.matrix); }); im.castShadow = true; group.add(im); }
  const legs = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.35, 0.35, 1.6, 10), dark, parked.length * 3);
  parked.forEach((s, i) => { [[0, 11], [-3.2, -1.5], [3.2, -1.5]].forEach(([dx, dz], k) => legs.setMatrixAt(i * 3 + k, m4.makeTranslation(s.x + dx, 0.8, s.z - 6 + dz))); }); group.add(legs);
  return {
    group,
    update(t, wind) {                                        // wind: the air's velocity near the ground (runway frame, m/s)
      if (radar) radar.rotation.y = t * 1.3;
      const sp = Math.hypot(wind[0], wind[2]); sock.rotation.y = -Math.atan2(wind[2], wind[0]); sock.rotation.z = -Math.max(0, 1 - sp / 7.7) * 1.2 + 0.05 * Math.sin(t * 3.1);   // 15 kt fills it
    },
  };
}
