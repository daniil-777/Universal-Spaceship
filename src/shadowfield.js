// Real shadows without shadow maps: the active world's height field (mountain range or skyline) lives on the GPU as one
// R32F texture, and any shader can ask how much of the sun a point sees by marching ten samples up-sun through it.
// Towers shade each other and the streets, ridges shade their valleys — for ~10 texture fetches per pixel and no extra
// render passes. All materials share one set of uniform objects, so swapping the world or moving the sun updates them all.
import * as THREE from 'three';

export const SHADOW_GLSL = /* glsl */`
  uniform sampler2D uHeight; uniform vec3 uSunDir; uniform float uY0, uShadowsOn, uPeriod, uShift;
  float fieldH(vec2 xz) { return uY0 + texture2D(uHeight, vec2((xz.x + uShift) / uPeriod, xz.y / 90.0 + 0.5)).r; }   // periodic in x (a long city's grid slides by the lap shift), the corridor's z span
  float sunVis(vec3 P) {                                                                        // 1 = full sun … 0.15 = deep shadow
    if (uShadowsOn < 0.5) return 1.0;
    vec2 dxz = normalize(uSunDir.xz + vec2(1e-4, 0.0)); float rise = max(uSunDir.y, 0.05) / max(length(uSunDir.xz), 1e-3), occ = 0.0;
    for (int k = 1; k <= 10; k++) { float t = float(k) * 3.2; float d = fieldH(P.xz + dxz * t) - (P.y + rise * t); occ = max(occ, clamp(d / 1.5, 0.0, 1.0)); }
    return 1.0 - 0.85 * occ; }
  float fieldHs(vec2 xz) {                                                                      // the same field, bilinear (soft shadow edges on smooth ground)
    vec2 sz = vec2(textureSize(uHeight, 0)), p = vec2((xz.x + uShift) / uPeriod, xz.y / 90.0 + 0.5) * sz - 0.5, f = fract(p), i0 = (floor(p) + 0.5) / sz, d = 1.0 / sz;
    return uY0 + mix(mix(texture2D(uHeight, i0).r, texture2D(uHeight, i0 + vec2(d.x, 0.0)).r, f.x), mix(texture2D(uHeight, i0 + vec2(0.0, d.y)).r, texture2D(uHeight, i0 + d).r, f.x), f.y); }
  float sunVisSoft(vec3 P) {
    if (uShadowsOn < 0.5) return 1.0;
    vec2 dxz = normalize(uSunDir.xz + vec2(1e-4, 0.0)); float rise = max(uSunDir.y, 0.05) / max(length(uSunDir.xz), 1e-3), occ = 0.0;
    for (int k = 1; k <= 10; k++) { float t = float(k) * 3.2; float d = fieldHs(P.xz + dxz * t) - (P.y + rise * t); occ = max(occ, clamp(d / 2.5, 0.0, 1.0)); }
    return 1.0 - 0.8 * occ; }`;

export function createShadowField() {
  const empty = new THREE.DataTexture(new Float32Array([0]), 1, 1, THREE.RedFormat, THREE.FloatType); empty.needsUpdate = true;
  const uniforms = { uHeight: { value: empty }, uSunDir: { value: new THREE.Vector3(0.6, 0.35, -0.7) }, uY0: { value: -26 }, uShadowsOn: { value: 0 }, uPeriod: { value: 120 }, uShift: { value: 0 } };
  let tex = null;
  return {
    uniforms,
    setField(hf) {                                        // the environment's height field → texture (shared grid, no copy)
      if (tex) tex.dispose();
      tex = new THREE.DataTexture(hf.grid, hf.NX, hf.NZ, THREE.RedFormat, THREE.FloatType);
      tex.wrapS = THREE.RepeatWrapping; tex.wrapT = THREE.ClampToEdgeWrapping; tex.magFilter = THREE.NearestFilter; tex.minFilter = THREE.NearestFilter; tex.needsUpdate = true;
      uniforms.uHeight.value = tex; uniforms.uY0.value = hf.y0; uniforms.uShadowsOn.value = 1; uniforms.uPeriod.value = hf.PERIOD || 120; uniforms.uShift.value = hf.lap ? hf.lap.shift : 0;
    },
    setShift(s) { uniforms.uShift.value = s; },
    setOn(on) { uniforms.uShadowsOn.value = on && tex ? 1 : 0; },
    setSun(d) { uniforms.uSunDir.value.copy(d).normalize(); },
    dispose() { if (tex) tex.dispose(); empty.dispose(); },
  };
}
