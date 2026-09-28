// The airfield's lights as one point cloud with its own shader: every runway, approach, taxiway and PAPI light at its
// surveyed position (src/landing/airport.js), sized by distance, bright enough by night to bloom, dimmer by day. The
// shader does what a real installation does optically: the approach lights shine only toward the approach, the runway
// lights both ways along the runway, the sequenced flashers run toward the threshold twice a second, and each PAPI box
// shows white above its setting angle and red below it — from the viewer's actual elevation, a 3′ transition.
import * as THREE from 'three';

const COLORS = { white: [1, 0.96, 0.86], yellow: [1, 0.78, 0.25], red: [1, 0.12, 0.08], green: [0.2, 1, 0.45], blue: [0.25, 0.45, 1] };
const KIND = { edge: [1.1, 2], centerline: [0.8, 2], tdz: [0.9, 1], threshold: [1.3, 1], wingbar: [1.3, 1], end: [1.1, 3], approach: [1.3, 1], crossbar: [1.3, 1],
  siderow: [1.2, 1], flasher: [2.6, 1], papi: [2.2, 1], taxi: [0.55, 0], taxicl: [0.45, 0], obstacle: [0.9, 0] };   // [size m, direction: 0 omni, 1 toward the approach, 2 along the runway, 3 toward the far end]

export function createAirfieldLights(AP) {
  const list = AP.lights.map((l) => ({ ...l }));
  for (const u of AP.papi) list.push({ kind: 'papi', x: u.x, y: u.y, z: u.z, color: 'white', angle: u.angle });
  for (const t of AP.taxiways) {                            // blue edges every 30 m, green centreline on the exits
    const [a, b] = t.pts, len = Math.hypot(b[0] - a[0], b[1] - a[1]), ux = (b[0] - a[0]) / len, uz = (b[1] - a[1]) / len, h = t.width / 2 + 1;
    for (let s = 20; s < len - 20; s += 30) { for (const side of [1, -1]) list.push({ kind: 'taxi', x: a[0] + ux * s - uz * h * side, y: 0.3, z: a[1] + uz * s + ux * h * side, color: 'blue' });
      if (t.kind !== 'parallel') list.push({ kind: 'taxicl', x: a[0] + ux * s, y: 0.1, z: a[1] + uz * s, color: 'green' }); }
  }
  for (const b of AP.buildings) if (b.h > 15) list.push({ kind: 'obstacle', x: b.x, y: b.h + 0.5, z: b.z, color: 'red' });
  const n = list.length, pos = new Float32Array(n * 3), col = new Float32Array(n * 3), prm = new Float32Array(n * 4);
  list.forEach((l, i) => { pos.set([l.x, l.y ?? 0.4, l.z], i * 3); col.set(COLORS[l.color] || COLORS.white, i * 3);
    const [size, dir] = KIND[l.kind] || [1, 0]; prm.set([size, dir, l.kind === 'flasher' ? l.seq : l.kind === 'papi' ? 100 + l.angle * 1000 : -1, l.kind === 'obstacle' ? 1 : 0], i * 4); });
  const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(pos, 3)); geo.setAttribute('color', new THREE.BufferAttribute(col, 3)); geo.setAttribute('prm', new THREE.BufferAttribute(prm, 4));
  const mat = new THREE.ShaderMaterial({
    uniforms: { uTime: { value: 0 }, uNight: { value: 0 }, uScale: { value: 800 }, uFogDensity: { value: 0 } },
    vertexShader: /* glsl */`
      attribute vec3 color; attribute vec4 prm; uniform float uTime, uNight, uScale, uFogDensity; varying vec3 vCol; varying float vI, vFog;
      #include <common>
      #include <logdepthbuf_pars_vertex>
      void main() {
        vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_Position = projectionMatrix * mv;
        #include <logdepthbuf_vertex>
        vec3 toCam = cameraPosition - position; float d = length(toCam); vec3 v = toCam / d;
        float dirK = 1.0;
        if (prm.y == 1.0) dirK = smoothstep(-0.2, 0.25, -v.x);              // faces the approach (toward −x)
        else if (prm.y == 2.0) dirK = smoothstep(0.35, 0.75, abs(v.x));     // along the runway, both ways
        else if (prm.y == 3.0) dirK = smoothstep(-0.2, 0.25, -v.x) * 0.9;    // the end bar, seen landing
        vCol = color; float I = 1.0;
        if (prm.z >= 100.0) {                                              // PAPI: white above the setting angle, red below
          float el = atan(toCam.y, length(toCam.xz)), setA = (prm.z - 100.0) / 1000.0;
          vCol = mix(vec3(1.0, 0.1, 0.06), vec3(1.0, 0.97, 0.9), smoothstep(setA - 0.00045, setA + 0.00045, el)); I = 1.6;
        } else if (prm.z >= 0.0) {                                         // sequenced flashers: the rabbit runs in, twice a second
          float ph = fract(uTime * 2.0) * 22.0; I = (abs(ph - prm.z) < 0.6) ? 4.0 : 0.0;
        }
        if (prm.w > 0.5) I *= 0.6 + 0.4 * step(0.5, fract(uTime * 0.75));   // obstacle lights blink
        float lowVis = clamp(uFogDensity * 900.0, 0.0, 1.0);                // by day in fog the lights run at full intensity (step 5)
        vI = I * dirK * mix(0.55, 1.6, max(uNight, lowVis));
        vFog = exp(-uFogDensity * uFogDensity * d * d);
        gl_PointSize = clamp(prm.x * uScale / max(d, 1.0), 1.2, 26.0) * mix(1.0, 1.35, uNight) * (1.0 + 2.2 * lowVis) * min(1.0, vI + 0.2);   // fog spreads each light into a halo
      }`,
    fragmentShader: /* glsl */`
      varying vec3 vCol; varying float vI, vFog;
      #include <logdepthbuf_pars_fragment>
      void main() {
        #include <logdepthbuf_fragment>
        vec2 c = gl_PointCoord - 0.5; float r = length(c); if (r > 0.5 || vI < 0.01) discard;
        float core = smoothstep(0.5, 0.0, r); gl_FragColor = vec4(vCol * vI * (0.6 + 2.4 * core * core), core * vFog);
      }`,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  });
  const points = new THREE.Points(geo, mat); points.frustumCulled = false; points.renderOrder = 5;
  return {
    points, count: n,
    update(t, night, fogDensity, viewportH) { mat.uniforms.uTime.value = t; mat.uniforms.uNight.value = night; mat.uniforms.uFogDensity.value = fogDensity; mat.uniforms.uScale.value = viewportH * 0.9; },
    dispose() { geo.dispose(); mat.dispose(); },
  };
}
