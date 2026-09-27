// The corridor's edges made visible when the ship meets them: the side walls (and in space the ceiling and floor) are
// invisible until the ship's beams find one close by — then a faint grid shimmers on it around the point the ship is
// heading for, rippling outward, and fades as the edge guard (src/env.js) turns the ship away. Additive, no depth writes.
import * as THREE from 'three';

const VERT = /* glsl */`varying vec2 vP; void main() { vP = position.xy; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;
const FRAG = /* glsl */`
  uniform vec3 uColor; uniform vec2 uHit; uniform float uA, uTime; varying vec2 vP;
  void main() {
    float r = length(vP - uHit), fall = 1.0 - smoothstep(2.0, 15.0, r);
    vec2 q = vP / 2.0, g = abs(fract(q - 0.5) - 0.5) / max(fwidth(q), vec2(1e-4)); float line = 1.0 - min(min(g.x, g.y), 1.0);   // a 2-unit grid, lines one pixel wide
    float ring = 0.5 + 0.5 * sin(r * 1.4 - uTime * 5.0), a = uA * fall * (0.12 + 0.6 * line) * (0.7 + 0.3 * ring);
    gl_FragColor = vec4(uColor * a, a);
  }`;

export function createEdgeFence(scene, { color = new THREE.Color(0.55, 0.85, 1.0) } = {}) {
  const T = { value: 0 }, walls = [];
  // n: which edge (0 z+, 1 z−, 2 ceiling, 3 floor); each plane lies in its local xy, turned onto the edge
  for (let n = 0; n < 4; n++) {
    const mat = new THREE.ShaderMaterial({ vertexShader: VERT, fragmentShader: FRAG, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
      uniforms: { uColor: { value: color }, uHit: { value: new THREE.Vector2() }, uA: { value: 0 }, uTime: T } });
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(44, 44), mat); mesh.frustumCulled = false; mesh.visible = false; mesh.renderOrder = 5;
    if (n >= 2) mesh.rotation.x = -Math.PI / 2;           // the ceiling / floor: local (x, y) = world (x, −z)
    scene.add(mesh); walls.push({ mesh, mat, n });
  }
  return {
    // p: the ship's render position; env: its environment (zHalf, yHalf, ceiling, the guard); atmo: over terrain (side walls only, between the ground and the ceiling)
    update(dt, p, f, { zHalf, yHalf, top, guard, atmo, on = true }) {
      T.value += dt;
      for (const w of walls) {
        const side = w.n < 2, sgn = w.n === 0 || w.n === 2 ? 1 : -1, edge = side ? sgn * zHalf : sgn > 0 ? top : -yHalf;
        const dist = side ? sgn * (edge - p.z) : sgn * (edge - p.y), heading = side ? sgn * f.z : sgn * f.y;   // how far, and how much the nose points at it
        let a = 0;
        if (on && (side || !atmo)) { a = Math.min(1, Math.max(0, 1 - (dist - 3) / 9)) * (0.35 + 0.65 * Math.max(0, heading)); if (guard > 0 && heading > 0) a = Math.min(1, a + 0.5 * guard); }
        const U = w.mat.uniforms; U.uA.value += (a * 0.55 - U.uA.value) * (1 - Math.exp(-dt * 8)); w.mesh.visible = U.uA.value > 0.004; if (!w.mesh.visible) continue;
        const s = heading > 0.05 ? Math.min(dist / heading, 20) : 0, hx = Math.max(-15, Math.min(15, f.x * s)), hy = Math.max(-15, Math.min(15, (side ? f.y : f.z) * s));   // where the nose's line meets the edge
        if (side) { w.mesh.position.set(p.x, p.y, edge); U.uHit.value.set(hx, hy); }
        else { w.mesh.position.set(p.x, edge, p.z); U.uHit.value.set(hx, -hy); }   // turned plane: local y = −world z
      }
    },
    dispose() { for (const w of walls) { scene.remove(w.mesh); w.mesh.geometry.dispose(); w.mat.dispose(); } },
  };
}
