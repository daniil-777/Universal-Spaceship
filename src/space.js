// space.js — the backdrop for Astro Pilot (three.js): low Earth orbit at sunrise.
// Draws: a 7000-star sky sphere centred on the camera (colour temperature + brightness spread, a faint Milky Way band),
// the Earth (see earth.js: Blue Marble textures, clouds, analytic atmosphere), the sun (HDR disc + animated corona in one
// billboard) with a subtle screen-space lens flare, Jupiter and Saturn as small far discs, Venus as a bright point.
// Also owns the scene lights: a warm DirectionalLight from the sun and a faint hemisphere fill.
// The action corridor (x∈[-60,60], |y|≤22, |z|≤18) is kept empty — every object sits 1000+ units away.
// Everything is written in linear HDR; the app's post chain (bloom → tone mapping → sRGB) finishes the look.
// Composition is designed for the chase view (root yawed by −45°): the Earth's limb runs across the lower half of the
// frame, the sun stands in the upper right above it. The root follows the camera in x (the corridor is periodic).
import * as THREE from 'three';
import { createEarth } from './earth.js';
import { createAircraft } from './aircraft.js';

const STAR_R = 4000;                  // star sphere radius (camera far plane must exceed it)
const CHASE_YAW = -Math.PI / 4, VIEW_PITCH = -13.05;    // chase camera: (-13, 3.6, 0) → (9, -1.5, 0), 50° vertical FOV
const Y_AXIS = new THREE.Vector3(0, 1, 0);
// A direction as seen from the chase camera (absolute elevation / azimuth in degrees, +az = right) in root-local
// coordinates: the root is yawed by CHASE_YAW in that view, so local = R_y(−CHASE_YAW) · world.
const fromChase = (el, az, dist) => { const e = el * Math.PI / 180, a = az * Math.PI / 180;
  return new THREE.Vector3(Math.cos(e) * Math.cos(a), Math.sin(e), Math.cos(e) * Math.sin(a)).applyAxisAngle(Y_AXIS, -CHASE_YAW).multiplyScalar(dist); };
const EARTH_R = 1400, EARTH_D = 2100, EARTH_RHO = Math.asin(EARTH_R / EARTH_D) * 180 / Math.PI;   // angular radius ≈ 42°
const EARTH_POS = fromChase(VIEW_PITCH - 4.5 - EARTH_RHO, 0, EARTH_D);                           // upper limb 4.5° below the frame centre
const EARTH_POS_LOW = fromChase(VIEW_PITCH + 7 - Math.asin(EARTH_R / 1600) * 180 / Math.PI, 20, 1600);   // low pass: closer, limb above the frame centre, shifted toward the day side
const EARTH_POS_ATMO = new THREE.Vector3(0, -(EARTH_R + 26), 0);                        // atmospheric flight: the surface 26 units under the corridor centre
const TRACK = { spin: 0.016, sweepAmp: 0.85, sweepPeriod: 170 };                        // orbit ground track: the globe turns (6.5 min/rev) and the track swings ±49° in latitude every 170 s, so every continent passes below
const SUN_POS = fromChase(VIEW_PITCH + 19, 22, 1250);                                           // upper right, ~30° above the limb below it
const SUN_R = 27;                     // disc radius (world units) → ~22 px at 1600×900 in the chase view

// ---------------------------------------------------------------- GLSL helpers (shared by every shader)
const NOISE = /* glsl */`
float hash3(vec3 p){ p = fract(p * 0.3183099 + vec3(0.11, 0.17, 0.23)); p *= 17.0;
  return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }
float vnoise(vec3 p){ vec3 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(hash3(i), hash3(i + vec3(1,0,0)), f.x), mix(hash3(i + vec3(0,1,0)), hash3(i + vec3(1,1,0)), f.x), f.y),
             mix(mix(hash3(i + vec3(0,0,1)), hash3(i + vec3(1,0,1)), f.x), mix(hash3(i + vec3(0,1,1)), hash3(i + vec3(1,1,1)), f.x), f.y), f.z); }
float fbm(vec3 p){ float a = 0.5, s = 0.0;
  for (int i = 0; i < 5; i++) { s += a * vnoise(p); p = p * 2.03 + vec3(1.7, 9.2, 3.1); a *= 0.5; } return s; }
float hash1(float x){ return fract(sin(x * 127.1) * 43758.5453); }
float noise1(float x){ float i = floor(x), f = fract(x); f = f * f * (3.0 - 2.0 * f); return mix(hash1(i), hash1(i + 1.0), f); }
float dither(){ return (fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453) - 0.5) * 0.003; }
float gband(float y, float c, float w){ float d = (y - c) / w; return exp(-d * d); }
`;

// Saturn ring optical depth as a function of x = (r - Rin) / (Rout - Rin). Shared by the ring and the globe (shadow).
const RING_DENSITY = /* glsl */`
float ringDensity(float x, float fw){
  float ringlet = mix(1.0, 0.72 + 0.28 * noise1(x * 38.0), clamp(1.0 - fw * 76.0, 0.0, 1.0))
                * mix(1.0, 0.86 + 0.14 * noise1(x * 130.0), clamp(1.0 - fw * 260.0, 0.0, 1.0));
  float c = 0.22 * smoothstep(0.0, 0.03, x) * smoothstep(0.22, 0.17, x);                        // C ring (faint)
  float b = 0.95 * smoothstep(0.17, 0.22, x) * smoothstep(0.585, 0.565, x);                      // B ring (dense)
  float a = 0.66 * smoothstep(0.615, 0.635, x) * smoothstep(0.905, 0.885, x)                   // A ring, with the Encke gap
          * (1.0 - 0.9 * (1.0 - smoothstep(0.0, 0.006, abs(x - 0.845))));
  float f = 0.30 * (1.0 - smoothstep(0.0, 0.010, abs(x - 0.955)));                                // F ring (thin)
  return (c + b + a) * ringlet + f; }
`;

function makeStarTexture() {
  const c = document.createElement('canvas'); c.width = c.height = 64;
  const g = c.getContext('2d'), grd = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grd.addColorStop(0, 'rgba(255,255,255,1)'); grd.addColorStop(0.12, 'rgba(255,255,255,0.9)');
  grd.addColorStop(0.3, 'rgba(255,255,255,0.35)'); grd.addColorStop(0.55, 'rgba(255,255,255,0.07)');
  grd.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grd; g.fillRect(0, 0, 64, 64);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.NoColorSpace; return t;
}

function mulberry32(seed) { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

// Star colour by temperature parameter u∈[0,1] (warm → white → blue), linear RGB.
function starColor(u, out) {
  const stops = [[1.0, 0.60, 0.32], [1.0, 0.86, 0.66], [1.0, 0.97, 0.92], [0.80, 0.87, 1.0], [0.64, 0.74, 1.0]];
  const s = u * 4, i = Math.min(3, Math.floor(s)), f = s - i, a = stops[i], b = stops[i + 1];
  out[0] = a[0] + (b[0] - a[0]) * f; out[1] = a[1] + (b[1] - a[1]) * f; out[2] = a[2] + (b[2] - a[2]) * f; return out;
}

// ---------------------------------------------------------------- sky: stars + Milky Way (one group centred on the camera)
function createSky(rng, bandN, bandT, extra) {
  const N = 7000 + extra.length, pos = new Float32Array(N * 3), col = new Float32Array(N * 3), size = new Float32Array(N);
  const b1 = new THREE.Vector3().crossVectors(bandN, new THREE.Vector3(1, 0, 0)).normalize(), b2 = new THREE.Vector3().crossVectors(bandN, b1);
  const gauss = () => { const u = Math.max(rng(), 1e-9), v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(6.2831853 * v); };
  const d = new THREE.Vector3(), c = [0, 0, 0];
  for (let i = 0; i < N; i++) {
    const e = i < extra.length ? extra[i] : null;                          // planets as bright points (Venus): fixed direction, colour, size
    if (e) d.copy(e.dir).normalize();
    else if (rng() < 0.36) {   // stars concentrated toward the galactic plane
      const a = rng() * 6.2831853, off = gauss() * 0.16;
      d.copy(b1).multiplyScalar(Math.cos(a)).addScaledVector(b2, Math.sin(a)).addScaledVector(bandN, off).normalize();
    } else {
      const z = rng() * 2 - 1, a = rng() * 6.2831853, r = Math.sqrt(1 - z * z);
      d.set(r * Math.cos(a), z, r * Math.sin(a));
    }
    pos[i * 3] = d.x * STAR_R; pos[i * 3 + 1] = d.y * STAR_R; pos[i * 3 + 2] = d.z * STAR_R;
    const bright = rng() < 0.05;
    const br = e ? e.bright : bright ? 0.7 + 1.1 * rng() : 0.14 + 0.5 * Math.pow(rng(), 2.0);
    if (e) c[0] = e.col[0], c[1] = e.col[1], c[2] = e.col[2]; else starColor(Math.pow(rng(), 1.15), c);
    col[i * 3] = c[0] * br; col[i * 3 + 1] = c[1] * br; col[i * 3 + 2] = c[2] * br;
    size[i] = e ? e.size : bright ? 6 + 5 * Math.pow(rng(), 2) : 2.6 + 3.2 * Math.pow(rng(), 1.5);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('aColor', new THREE.BufferAttribute(col, 3));
  geo.setAttribute('aSize', new THREE.BufferAttribute(size, 1));
  const tex = makeStarTexture();
  const starMat = new THREE.ShaderMaterial({
    uniforms: { uMap: { value: tex }, uPR: { value: Math.min(window.devicePixelRatio || 1, 2) }, uGain: { value: 1 } },
    vertexShader: /* glsl */`attribute float aSize; attribute vec3 aColor; varying vec3 vColor; uniform float uPR;
      void main(){ vColor = aColor; vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = aSize * uPR * (${STAR_R.toFixed(1)} / max(-mv.z, 1.0)); gl_Position = projectionMatrix * mv; }`,
    fragmentShader: /* glsl */`uniform sampler2D uMap; uniform float uGain; varying vec3 vColor;
      void main(){ float a = texture2D(uMap, gl_PointCoord).a; gl_FragColor = vec4(vColor * a * uGain, 1.0); }`,
    blending: THREE.AdditiveBlending, transparent: true, depthWrite: false,
  });
  const stars = new THREE.Points(geo, starMat); stars.renderOrder = -6; stars.frustumCulled = false;

  const bandMat = new THREE.ShaderMaterial({
    uniforms: { uN: { value: bandN }, uT: { value: bandT }, uGain: { value: 0.009 } },
    vertexShader: /* glsl */`varying vec3 vDir; void main(){ vDir = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: /* glsl */`uniform vec3 uN, uT; uniform float uGain; varying vec3 vDir; ${NOISE}
      void main(){ vec3 d = normalize(vDir); float h = dot(d, uN), along = dot(d, uT), side = dot(d, cross(uN, uT));
        float w = 0.075 + 0.05 * (fbm(d * 2.5) - 0.5);
        float core = exp(-h * h / (2.0 * w * w)), wing = 0.28 * exp(-h * h / 0.045);
        float cloud = fbm(vec3(along * 5.0, h * 9.0, side * 5.0) + 7.0), dust = fbm(vec3(along * 7.0, h * 22.0, side * 7.0) + 2.0);
        float bulge = exp(-(along - 0.9) * (along - 0.9) * 4.0);
        float lanes = smoothstep(0.46, 0.72, dust) * smoothstep(0.2, 0.0, abs(h));                // dust lanes stretched along the band
        float i = (core * (0.4 + 1.2 * cloud) + wing * (0.7 + 0.6 * cloud)) * (1.0 - 0.6 * lanes) * (1.0 + 0.8 * bulge);
        vec3 col = mix(vec3(0.62, 0.72, 0.98), vec3(0.98, 0.88, 0.74), 0.2 + 0.6 * bulge) * i * uGain;
        gl_FragColor = vec4(col + dither(), 1.0); }`,
    blending: THREE.AdditiveBlending, transparent: true, depthWrite: false, side: THREE.BackSide,
  });
  const band = new THREE.Mesh(new THREE.SphereGeometry(STAR_R * 1.06, 48, 32), bandMat); band.renderOrder = -7; band.frustumCulled = false;
  const group = new THREE.Group(); group.add(band, stars);
  return { group, starMat, dispose() { geo.dispose(); starMat.dispose(); tex.dispose(); band.geometry.dispose(); bandMat.dispose(); } };
}

// ---------------------------------------------------------------- sun: photosphere disc + corona + wide glow as one additive billboard
function createSun() {
  const plane = SUN_R * 9;   // half-size of the billboard
  const mat = new THREE.ShaderMaterial({
    uniforms: { uTime: { value: 0 }, uRd: { value: SUN_R / plane }, uGlare: { value: 1 }, uCol: { value: new THREE.Color(1.0, 0.90, 0.74) } },
    vertexShader: /* glsl */`varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: /* glsl */`uniform float uTime, uRd, uGlare; uniform vec3 uCol; varying vec2 vUv; ${NOISE}
      void main(){ vec2 p = (vUv - 0.5) * 2.0; float r = length(p), rd = uRd;
        float q = clamp(r / rd, 0.0, 1.0), ld = 0.78 + 0.22 * sqrt(1.0 - q * q);                // limb darkening
        float disc = 1.0 - smoothstep(rd - 0.004, rd + 0.004, r);
        vec3 col = uCol * 5.0 * ld * disc;
        float ang = atan(p.y, p.x + 1e-6), x = max(r - rd, 0.0) / rd;                       // x: distance from the limb in radii
        float st = fbm(vec3(cos(ang) * 5.0, sin(ang) * 5.0, r * 4.0 - uTime * 0.03));
        float st2 = fbm(vec3(cos(ang) * 11.0, sin(ang) * 11.0, r * 8.0 + uTime * 0.02));
        float sp = pow(max(0.6 * st + 0.4 * st2 - 0.22, 0.0) / 0.78, 2.2);              // soft-edged radial streamers
        float rim = exp(-x * 20.0) * 1.1;                                                // chromosphere
        float cor = (exp(-x * 1.6) * 0.30 + exp(-x * 0.55) * 0.08) * (0.25 + 1.6 * sp);     // corona, 4–6 R wide, very soft
        float wide = pow(max(1.0 - r, 0.0), 3.2) * 0.03;                                 // soft warm haze
        col += (vec3(1.0, 0.80, 0.55) * rim + vec3(1.0, 0.72, 0.42) * cor + vec3(1.0, 0.80, 0.55) * wide) * uGlare * (1.0 - disc);
        gl_FragColor = vec4(col + dither(), 1.0); }`,
    blending: THREE.AdditiveBlending, transparent: true, depthWrite: false,
  });
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(plane * 2, plane * 2), mat);
  mesh.position.copy(SUN_POS); mesh.renderOrder = -1; mesh.frustumCulled = false;
  return { mesh, mat, dispose() { mesh.geometry.dispose(); mat.dispose(); } };
}

// ---------------------------------------------------------------- lens flare: a screen-space quad (ghosts along the sun→centre axis + streak)
function createFlare() {
  const mat = new THREE.ShaderMaterial({
    uniforms: { uSun: { value: new THREE.Vector2(9, 9) }, uAspect: { value: 16 / 9 }, uVis: { value: 0 } },
    vertexShader: /* glsl */`varying vec2 vQ; void main(){ vQ = position.xy; gl_Position = vec4(position.xy, 0.0, 1.0); }`,
    fragmentShader: /* glsl */`uniform vec2 uSun; uniform float uAspect, uVis; varying vec2 vQ;
      float ghost(vec2 p, vec2 c, float r){ float d = length(p - c) / r;                          // soft disc with a slightly brighter rim
        return smoothstep(1.0, 0.5, d) * 0.5 + smoothstep(0.6, 0.93, d) * (1.0 - smoothstep(0.93, 1.0, d)) * 0.6; }
      void main(){ vec2 p = vec2(vQ.x * uAspect, vQ.y), s = vec2(uSun.x * uAspect, uSun.y);
        vec3 col = vec3(0.55, 0.75, 1.0) * 0.020 * ghost(p, s * 0.55, 0.055)
                 + vec3(1.0, 0.72, 0.45) * 0.014 * ghost(p, s * 0.18, 0.09)
                 + vec3(0.70, 1.0, 0.80) * 0.010 * ghost(p, -s * 0.28, 0.14)
                 + vec3(1.0, 0.60, 0.70) * 0.010 * ghost(p, -s * 0.62, 0.05)
                 + vec3(0.60, 0.80, 1.0) * 0.0035 * ghost(p, -s * 1.05, 0.26);
        vec2 q = p - s; col += vec3(1.0, 0.85, 0.7) * exp(-q.y * q.y * 3500.0) * exp(-abs(q.x) * 2.2) * 0.08;   // anamorphic streak
        gl_FragColor = vec4(col * uVis, 1.0); }`,
    blending: THREE.AdditiveBlending, transparent: true, depthTest: false, depthWrite: false,
  });
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat); mesh.renderOrder = 10; mesh.frustumCulled = false;
  return { mesh, mat, dispose() { mesh.geometry.dispose(); mat.dispose(); } };
}

// ---------------------------------------------------------------- planets: shared lit shader with a pluggable surface function
const PLANET_VS = /* glsl */`varying vec3 vN, vP, vO;
  void main(){ vO = position / uR; vN = normalize(mat3(modelMatrix) * normal); vec4 w = modelMatrix * vec4(position, 1.0); vP = w.xyz;
    gl_Position = projectionMatrix * viewMatrix * w; }`;
const SURFACES = {
  jupiter: /* glsl */`vec3 surface(vec3 n){
    float lat = asin(clamp(n.y, -1.0, 1.0)), lon = atan(n.z, n.x + 1e-7);
    vec3 q = vec3(n.x, n.y * 3.0, n.z);                                                    // features stretched along longitude
    float turb = fbm(q * 2.6 + 5.0) - 0.5, t2 = fbm(q * 6.0 + 11.0) - 0.5;
    float y = lat + 0.10 * turb + 0.04 * t2;
    float belts = gband(y, 0.22, 0.11) + gband(y, -0.24, 0.12) + 0.8 * gband(y, 0.45, 0.035) + 0.8 * gband(y, -0.50, 0.035);   // NEB, SEB, NTB, STB
    float dark = smoothstep(0.1, 0.8, clamp(belts + 0.4 * t2, 0.0, 1.0));
    vec3 col = mix(mix(vec3(0.82, 0.76, 0.64), vec3(0.74, 0.64, 0.50), 0.5 + 0.5 * sin(y * 14.0 + 1.0)), mix(vec3(0.60, 0.33, 0.16), vec3(0.36, 0.20, 0.11), smoothstep(-0.2, 0.3, t2)), dark);
    col = mix(col, vec3(0.58, 0.55, 0.50), smoothstep(0.85, 1.3, abs(lat)));                   // polar hoods
    float dl = mod(lon - 2.3 + 3.14159, 6.28318) - 3.14159;                                    // Great Red Spot at 22°S
    float d = length(vec2(dl * 0.93 / 0.30, (lat + 0.38) / 0.14));
    return mix(col, vec3(0.84, 0.46, 0.30), 1.0 - smoothstep(0.72, 1.0, d)); }`,
  saturn: /* glsl */`${RING_DENSITY} uniform vec3 uLightL; uniform vec2 uRing;
    vec3 surface(vec3 n){
    float lat = asin(clamp(n.y, -1.0, 1.0));
    float t2 = fbm(vec3(n.x * 2.0, n.y * 7.0, n.z * 2.0) + 3.0) - 0.5;
    float y = lat + 0.04 * t2;
    float belts = gband(y, 0.16, 0.06) + gband(y, -0.14, 0.05) + 0.8 * gband(y, 0.40, 0.05) + 0.8 * gband(y, -0.42, 0.05);
    float dark = smoothstep(0.1, 0.9, clamp(belts + 0.25 * t2, 0.0, 1.0));
    vec3 col = mix(mix(vec3(0.94, 0.90, 0.78), vec3(0.88, 0.78, 0.58), 0.5 + 0.5 * sin(y * 12.0)), vec3(0.74, 0.62, 0.42), dark * 0.7);
    return mix(col, vec3(0.66, 0.70, 0.72), smoothstep(1.0, 1.45, abs(lat))); }
    float shadowAt(vec3 o){ float t = -o.y / (abs(uLightL.y) < 1e-4 ? 1e-4 : uLightL.y); vec3 h = o + t * uLightL;   // ring shadow on the globe
      float x = (length(h.xz) - uRing.x) / (uRing.y - uRing.x); return step(0.0, t) * ringDensity(clamp(x, 0.0, 1.0), 0.03) * step(0.0, x) * step(x, 1.0); }`,
};
const PLANET_FS = (surface, hasRingShadow) => /* glsl */`uniform vec3 uLight, uSunCol, uRimCol; uniform float uR, uSpin, uWrap, uLimb, uRim, uEdge, uGain;
  varying vec3 vN, vP, vO; ${NOISE} ${surface}
  void main(){ float c = cos(uSpin), s = sin(uSpin); vec3 o = normalize(vO), n = vec3(c * o.x - s * o.z, o.y, s * o.x + c * o.z);
    vec3 N = normalize(vN), V = normalize(cameraPosition - vP);
    float ndl = dot(N, uLight), ndv = max(dot(N, V), 0.0);
    float diff = clamp((ndl + uWrap) / (1.0 + uWrap), 0.0, 1.0) * mix(1.0 - uLimb, 1.0, pow(ndv, 0.55));
    ${hasRingShadow ? 'diff *= 1.0 - 0.92 * shadowAt(o * uR);' : ''}
    vec3 alb = surface(n);
    vec3 col = alb * uSunCol * diff * uGain + alb * vec3(0.10, 0.12, 0.16) * 0.06 * (0.6 + 0.4 * N.y);
    col += uRimCol * pow(1.0 - ndv, 4.0) * clamp(ndl * 2.0 + 0.5, 0.0, 1.0) * uRim;
    gl_FragColor = vec4(col + dither(), smoothstep(0.0, uEdge, ndv)); }`;

function createPlanet({ kind, pos, r, segs, spin, tilt, sunCol, wrap = 0.08, limb = 0.45, rim = 0.06, rimCol = [1, 0.9, 0.75], pxRadius }) {
  const light = new THREE.Vector3(0, 0, 1);   // world-space light direction, refreshed by applyYaw()
  const uniforms = {
    uLight: { value: light }, uSunCol: { value: sunCol }, uRimCol: { value: new THREE.Color(...rimCol) }, uR: { value: r },
    uSpin: { value: 0 }, uWrap: { value: wrap }, uLimb: { value: limb }, uRim: { value: rim }, uGain: { value: 0.80 },
    uEdge: { value: Math.sqrt(1 - Math.pow(1 - 1.5 / pxRadius, 2)) },   // fade the last ~1.5 px of the limb (shader anti-aliasing)
  };
  const mat = new THREE.ShaderMaterial({ uniforms, vertexShader: `uniform float uR; ${PLANET_VS}`, fragmentShader: PLANET_FS(SURFACES[kind], kind === 'saturn'), transparent: true });
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(r, segs, Math.round(segs * 0.7)), mat);
  const group = new THREE.Group(); group.position.copy(pos); if (tilt) group.rotation.set(...tilt); group.add(mesh);
  mesh.renderOrder = -4;   // whole backdrop sorts before the app's transparent effects (renderOrder 0)
  return { group, mesh, mat, light, spin, dispose() { mesh.geometry.dispose(); mat.dispose(); } };
}

function createRings(planet, r) {
  const inner = r * 1.25, outer = r * 2.3;
  planet.mat.uniforms.uLightL = { value: new THREE.Vector3() }; planet.mat.uniforms.uRing = { value: new THREE.Vector2(inner, outer) };
  const mat = new THREE.ShaderMaterial({
    uniforms: { uLight: { value: planet.light }, uCenter: { value: new THREE.Vector3() }, uR: { value: r }, uSunCol: planet.mat.uniforms.uSunCol },
    vertexShader: /* glsl */`varying float vX; varying vec3 vP, vNw;
      void main(){ vX = (length(position.xy) - ${inner.toFixed(2)}) / ${(outer - inner).toFixed(2)}; vec4 w = modelMatrix * vec4(position, 1.0); vP = w.xyz;
        vNw = normalize(mat3(modelMatrix) * vec3(0.0, 0.0, 1.0)); gl_Position = projectionMatrix * viewMatrix * w; }`,
    fragmentShader: /* glsl */`uniform vec3 uLight, uCenter, uSunCol; uniform float uR; varying float vX; varying vec3 vP, vNw; ${NOISE} ${RING_DENSITY}
      void main(){ float fw = fwidth(vX); float dens = ringDensity(vX, fw) * smoothstep(0.0, 0.012, vX) * smoothstep(1.0, 0.985, vX);
        vec3 v = vP - uCenter; float b = dot(v, uLight); float dperp = sqrt(max(dot(v, v) - b * b, 0.0));   // planet shadow (analytic)
        float shadow = step(b, 0.0) * (1.0 - smoothstep(uR * 0.985, uR * 1.03, dperp));
        vec3 V = normalize(cameraPosition - vP); float mu0 = dot(vNw, uLight), mu = dot(vNw, V);
        float lit = abs(mu0), same = step(0.0, mu0 * mu);
        float shade = mix(0.42 * (1.0 - 0.5 * dens), 1.0, same) * (0.55 + 0.45 * lit);
        vec3 tint = mix(vec3(0.72, 0.70, 0.66), vec3(0.90, 0.84, 0.72), smoothstep(0.15, 0.35, vX)) * (0.9 + 0.1 * noise1(vX * 17.0));
        vec3 col = tint * uSunCol * (shade * (1.0 - 0.94 * shadow) * 0.80 + 0.03);
        gl_FragColor = vec4(col + dither(), dens); }`,
    transparent: true, depthWrite: false, side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(new THREE.RingGeometry(inner, outer, 128, 1), mat);
  mesh.rotation.x = -Math.PI / 2; mesh.renderOrder = -3; planet.group.add(mesh);
  return { mesh, mat, dispose() { mesh.geometry.dispose(); mat.dispose(); } };
}

// ---------------------------------------------------------------- public API
export function createSpace(scene, { seed = 1, texturePath = 'textures/' } = {}) {
  const rng = mulberry32(seed * 7919 + 17);
  const sunDir = SUN_POS.clone().normalize();                     // root-local; the sun is parallel light for everything
  const sunDirWorld = sunDir.clone();                              // follows the yaw; shared by every planet/Earth shader
  const sunCol = new THREE.Color(1.0, 0.95, 0.88), earthSun = sunCol.clone().multiplyScalar(2.2);
  // Milky Way great circle: a diagonal band high in the chase frame; the bulge sits off toward the upper right.
  const bandN = new THREE.Vector3(0.42, -0.89, 0.19).normalize();
  const bandT = new THREE.Vector3(0.906, 0.414, -0.087).normalize();
  const venus = { dir: fromChase(VIEW_PITCH + 12, 17, 1), col: [1.0, 0.96, 0.9], bright: 6, size: 10 };
  const sky = createSky(rng, bandN, bandT, [venus]);
  const sun = createSun(), flare = createFlare();
  // Earth: axis tilted 23.4° away from the camera so the disc is centred near 25° N; the spin phase puts the Atlantic in view.
  const axis = new THREE.Vector3(0, 1, 0).applyAxisAngle(new THREE.Vector3(0.707, 0, 0.707), -23.4 * Math.PI / 180);   // → (0.28, 0.92, −0.28)
  const earth = createEarth({ texturePath, R: EARTH_R, position: EARTH_POS, axis, sunDir: sunDirWorld, sunCol: earthSun, spin: TRACK.spin, spin0: 1.1 });
  const tiltedQ = earth.group.children[0].quaternion.clone().invert(), hub = EARTH_POS_LOW.clone().multiplyScalar(-1).applyQuaternion(tiltedQ);   // surface point under the ship (tilted frame)
  const aircraft = createAircraft(earth.group.children[0], { R: EARTH_R, n: 14, spin: TRACK.spin, spin0: 1.1, hub });
  let trackAngle = 0; const X = new THREE.Vector3(1, 0, 0);   // jets over the surface, shown during low passes
  // Outer planets are always near-full from here: light each from between the sun and the viewer (gibbous, lit toward the sun).
  const gibbous = (pos) => pos.clone().normalize().negate().addScaledVector(sunDir, 0.6).normalize();
  const planets = [
    createPlanet({ kind: 'jupiter', pos: fromChase(VIEW_PITCH + 3, -22, 3200), r: 24, segs: 32, spin: 0.009, tilt: [0, 0, 0.06], sunCol, limb: 0.5, rim: 0.05, pxRadius: 7.7 }),
    createPlanet({ kind: 'saturn', pos: fromChase(VIEW_PITCH + 9, -33, 3400), r: 17, segs: 32, spin: 0.011, tilt: [0.42, 0, 0.30], sunCol, limb: 0.42, rim: 0.05, pxRadius: 6 }),
  ];
  for (const p of planets) p.light0 = gibbous(p.group.position);
  const rings = createRings(planets[1], 17);

  const sunLight = new THREE.DirectionalLight(new THREE.Color(1.0, 0.95, 0.88), 2.4);
  sunLight.position.copy(sunDir).multiplyScalar(1000); sunLight.target.position.set(0, 0, 0);
  const fill = new THREE.HemisphereLight(0x2a3a5a, 0x0e1626, 0.25);              // faint sky fill, a hint of earthshine from below
  const root = new THREE.Group();                        // the whole backdrop yaws (chase view) and follows the camera in x
  root.add(sky.group, earth.group, sun.mesh, flare.mesh, ...planets.map((p) => p.group), sunLight, sunLight.target, fill); scene.add(root);
  const _v = new THREE.Vector3(), _q = new THREE.Quaternion(), _m = new THREE.Matrix4(), _s = new THREE.Vector3(), _d = new THREE.Vector3();
  let yaw = 0, yawTarget = 0, dirty = true, flareVis = 0, flareGain = 1, trackedCamera = null, alt = 0, altTarget = 0, atmoK = 0, atmoTarget = 0;
  // sky dome for atmospheric flight: blue zenith → pale horizon, sun glow, dark on the night side; drawn behind the ground
  const domeMat = new THREE.ShaderMaterial({ transparent: true, depthWrite: false, side: THREE.BackSide, uniforms: { uSun: { value: new THREE.Vector3() }, uAlpha: { value: 0 } },
    vertexShader: 'varying vec3 vD; void main(){ vD = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: `uniform vec3 uSun; uniform float uAlpha; varying vec3 vD;
      void main(){ vec3 d = normalize(vD); float h = d.y; vec3 zen = vec3(0.03, 0.14, 0.45), hor = vec3(0.5, 0.64, 0.86);
        vec3 c = mix(hor, zen, smoothstep(-0.02, 0.5, h)); float day = smoothstep(-0.18, 0.25, uSun.y); c *= 0.06 + 0.94 * day;
        c += vec3(1.0, 0.85, 0.6) * pow(max(dot(d, uSun), 0.0), 60.0) * 0.5 * day + vec3(1.0, 0.7, 0.45) * pow(max(dot(d, uSun), 0.0), 6.0) * 0.12 * day;
        gl_FragColor = vec4(c, uAlpha); }` });
  const dome = new THREE.Mesh(new THREE.SphereGeometry(2600, 32, 16), domeMat); dome.renderOrder = -9; dome.frustumCulled = false; dome.visible = false; root.add(dome);
  const fillSpace = { sky: new THREE.Color(0x2a3a5a), ground: new THREE.Color(0x0e1626), i: 0.25 }, fillAir = { sky: new THREE.Color(0x9cc0ee), ground: new THREE.Color(0x5a4a38), i: 0.6 }, Z = new THREE.Vector3(0, 0, 1);
  const applyYaw = () => {
    root.rotation.y = yaw;
    sunDirWorld.copy(sunDir).applyQuaternion(root.quaternion);
    for (const p of planets) p.light.copy(p.light0).applyQuaternion(root.quaternion);              // shared Vector3 → planet + ring shaders
    planets[1].mat.uniforms.uLightL.value.copy(planets[1].light).applyQuaternion(_q.copy(planets[1].group.quaternion).invert());   // Saturn-local, ring shadow
  };
  applyYaw();
  flare.mesh.onBeforeRender = (r, sc, cam) => { flare.mat.uniforms.uVis.value = cam === trackedCamera ? flareVis * flareGain : 0; };   // not in env-map bakes

  // Lens flare: project the sun; fade when it leaves the frame or when the camera→sun ray passes through the Earth.
  const updateFlare = (camera) => {
    trackedCamera = camera; camera.updateMatrixWorld(); _m.copy(camera.matrixWorld).invert();
    sun.mesh.getWorldPosition(_s); _v.copy(_s).applyMatrix4(_m);
    let vis = 0;
    if (_v.z < 0) {
      _v.applyMatrix4(camera.projectionMatrix);
      vis = THREE.MathUtils.clamp((1.6 - Math.max(Math.abs(_v.x), Math.abs(_v.y))) / 0.5, 0, 1);
      _d.copy(_s).sub(camera.position); const dist = _d.length(); _d.divideScalar(dist);
      const oc = _s.copy(camera.position).sub(earth.centre), b = oc.dot(_d), perp = Math.sqrt(Math.max(oc.lengthSq() - b * b, 0));
      if (b < 0 && dist > -b) vis *= THREE.MathUtils.clamp((perp - earth.R * 0.995) / (earth.R * 0.03), 0, 1);
      flare.mat.uniforms.uSun.value.set(_v.x, _v.y); flare.mat.uniforms.uAspect.value = camera.aspect;
    }
    flareVis = vis;
  };

  return {
    sunDir, get sunDirWorld() { return sunDirWorld; }, lights: { sun: sunLight, fill }, root, ready: earth.ready,
    setYaw(target) { yawTarget = target; dirty = true; },
    setAltitude(a) {                                                  // 0 = orbit, 1 = low pass (the Earth comes ~27 % closer, its limb rises, airliners appear)
      altTarget = Math.max(0, Math.min(1, a));
      if (a > 0.5) aircraft.setHub(EARTH_POS_LOW.clone().negate().applyQuaternion(_q.copy(earth.group.quaternion).multiply(earth.group.children[0].quaternion).invert()));   // jets cross the point now under the ship
    },
    get altitude() { return alt; },
    setAtmosphere(on) { atmoTarget = on ? 1 : 0; },                   // atmospheric flight: the surface right below, sky dome, haze, ground scrolling at flight speed
    get atmosphere() { return atmoK; },
    update(t, dt, camera) {
      const moving = Math.abs(altTarget - alt) > 1e-4 || Math.abs(atmoTarget - atmoK) > 1e-4;
      if (moving) {
        alt += (altTarget - alt) * (1 - Math.exp(-dt * 0.9)); atmoK += (atmoTarget - atmoK) * (1 - Math.exp(-dt * 0.7));
        earth.group.position.lerpVectors(EARTH_POS, EARTH_POS_LOW, alt).lerp(EARTH_POS_ATMO, atmoK);
        earth.setHaze(atmoK); domeMat.uniforms.uAlpha.value = atmoK; dome.visible = atmoK > 0.01; sky.group.visible = atmoK < 0.97;
        for (const p of planets) p.group.visible = atmoK < 0.5;
        fill.color.copy(fillSpace.sky).lerp(fillAir.sky, atmoK); fill.groundColor.copy(fillSpace.ground).lerp(fillAir.ground, atmoK); fill.intensity = fillSpace.i + (fillAir.i - fillSpace.i) * atmoK;
      }
      if (atmoK < 0.99) { const target = TRACK.sweepAmp * Math.sin(2 * Math.PI * t / TRACK.sweepPeriod), d = (target - trackAngle) * (1 - atmoK); if (Math.abs(d) > 1e-7) { earth.group.rotateOnAxis(X, d); trackAngle += d; } }   // the ground track swings north and south
      if (atmoK > 0.01) { earth.group.rotateOnAxis(Z, -dt * atmoK * 0.45 / EARTH_R); dome.position.copy(camera.position).sub(root.position).applyQuaternion(_q.copy(root.quaternion).invert()); domeMat.uniforms.uSun.value.copy(sunDir); }   // the far ground turns at the terrain strip's speed
      aircraft.update(dt, alt * (1 - atmoK));                                                   // sphere jets only on low passes; corridor airliners take over in the atmosphere
      if (dirty) { yaw += (yawTarget - yaw) * (1 - Math.exp(-dt * 2.5)); if (Math.abs(yawTarget - yaw) < 1e-4) { yaw = yawTarget; dirty = false; } applyYaw(); }
      root.position.x = camera.position.x;                                                   // no parallax along the periodic corridor
      sky.group.position.copy(camera.position).sub(root.position).applyQuaternion(_q.copy(root.quaternion).invert());
      sun.mesh.quaternion.copy(root.quaternion).invert().multiply(camera.quaternion); sun.mat.uniforms.uTime.value = t;
      for (const p of planets) p.mat.uniforms.uSpin.value += dt * p.spin;
      planets[1].group.getWorldPosition(rings.mat.uniforms.uCenter.value);
      earth.update(dt); updateFlare(camera);
    },
    setMono(mode) {   // greyscale / ink: tame the glare so the two-tone pass keeps its detail; no flare ghosts in ink
      sun.mat.uniforms.uGlare.value = mode === 2 ? 0.45 : mode ? 0.6 : 1.0; flareGain = mode === 2 ? 0 : mode === 1 ? 0.6 : 1;
      sky.starMat.uniforms.uGain.value = mode === 2 ? 1.25 : 1.0; earth.setMono(mode);
    },
    dispose() {
      scene.remove(root);
      sky.dispose(); sun.dispose(); flare.dispose(); rings.dispose(); planets.forEach((p) => p.dispose()); earth.dispose(); aircraft.dispose();
    },
  };
}
