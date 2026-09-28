// A cloud deck for the landing scene, as a METAR reports it (FEW / SCT / BKN / OVC at a base height): two surfaces, the
// base and the top, whose coverage is patterned in world space and drifts with the wind aloft — ragged grey bases seen
// from below, sunlit tops from above, lit by the time of day — and, for the scene's fog, how far the camera is inside
// the layer (src/landing/scene.js closes the fog in there and dims the sun beneath a broken or overcast deck).
import * as THREE from 'three';

const NOISE = /* glsl */`
  float h21(vec2 p) { p = fract(p * vec2(233.34, 851.73)); p += dot(p, p + 23.45); return fract(p.x * p.y); }
  float vn(vec2 p) { vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f); return mix(mix(h21(i), h21(i + vec2(1, 0)), f.x), mix(h21(i + vec2(0, 1)), h21(i + vec2(1, 1)), f.x), f.y); }
  float fbm(vec2 p) { float s = 0.0, a = 0.5; for (int i = 0; i < 5; i++) { s += a * vn(p); p = p * 2.03 + 17.1; a *= 0.5; } return s; }`;
export const COVER = { FEW: 0.25, SCT: 0.45, BKN: 0.75, OVC: 1 };   // oktas / 8, roughly

export function createCloudDeck({ cover = 0.45, base = 900, top = 1250 } = {}) {
  const group = new THREE.Group(), mats = [];
  for (const isTop of [false, true]) {
    const mat = new THREE.ShaderMaterial({
      uniforms: THREE.UniformsUtils.merge([THREE.UniformsLib.fog, { uCover: { value: cover }, uOff: { value: new THREE.Vector2() }, uSun: { value: new THREE.Vector3(0, 1, 0) },
        uSunCol: { value: new THREE.Color(1, 1, 1) }, uLight: { value: 1 }, uTop: { value: isTop ? 1 : 0 } }]),
      vertexShader: /* glsl */`
        varying vec3 vW;
        #include <common>
        #include <fog_pars_vertex>
        #include <logdepthbuf_pars_vertex>
        void main() {
          vec4 wp = modelMatrix * vec4(position, 1.0); vW = wp.xyz; vec4 mvPosition = viewMatrix * wp; gl_Position = projectionMatrix * mvPosition;
          #include <logdepthbuf_vertex>
          #include <fog_vertex>
        }`,
      fragmentShader: /* glsl */`
        uniform float uCover, uLight, uTop; uniform vec2 uOff; uniform vec3 uSun, uSunCol; varying vec3 vW;
        #include <common>
        #include <fog_pars_fragment>
        #include <logdepthbuf_pars_fragment>
        ${NOISE}
        void main() {
          #include <logdepthbuf_fragment>
          vec2 p = vW.xz + uOff; float c = fbm(p / 2400.0) * 0.6 + fbm(p / 520.0) * 0.4, thr = mix(0.62, 0.32, uCover);   // big cells, ragged edges
          float a = uCover > 0.99 ? 0.985 : smoothstep(thr, thr + 0.07, c); if (a < 0.01) discard;
          float d = fbm(p / 110.0), sunUp = clamp(uSun.y * 3.0, 0.0, 1.0);
          vec3 lit = uTop > 0.5 ? mix(vec3(0.78, 0.8, 0.85), vec3(1.0), d) * mix(vec3(0.6), uSunCol, sunUp)   // tops: white where the sun is on them
                                : mix(vec3(0.38, 0.4, 0.44), vec3(0.7, 0.72, 0.76), d) * (1.05 - 0.45 * uCover);   // bases: greyer the thicker the deck
          gl_FragColor = vec4(lit * uLight, a);
          #include <fog_fragment>
        }`,
      transparent: true, depthWrite: false, side: THREE.DoubleSide, fog: true,
    });
    const m = new THREE.Mesh(new THREE.PlaneGeometry(160000, 160000).rotateX(-Math.PI / 2), mat);
    m.position.y = isTop ? top : base; m.frustumCulled = false; m.renderOrder = 3; group.add(m); mats.push(mat);
  }
  const drift = new THREE.Vector2();
  return {
    group, cover, base, top,
    inside(y) { return Math.min(1, Math.max(0, Math.min(y - base, top - y) / 25 + 0.5)); },   // 0 outside … 1 once ~12 m in
    update(dt, sunDir, sunCol, light, wind) {               // wind: the air's velocity aloft (runway frame, m/s) carries the pattern
      drift.x -= wind[0] * dt; drift.y -= wind[2] * dt;
      for (const m of mats) { const u = m.uniforms; u.uOff.value.copy(drift); u.uSun.value.copy(sunDir); u.uSunCol.value.copy(sunCol); u.uLight.value = light; }
    },
    dispose() { for (const m of mats) m.dispose(); group.traverse((o) => o.geometry && o.geometry.dispose()); },
  };
}
