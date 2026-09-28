// earth.js — the Earth of Astro Pilot's backdrop (helper for space.js). NASA Blue Marble textures on three shaders:
//   ground  — day/night blend with a soft terminator, normal-mapped relief, ocean sun glint (roughness from the packed
//             map), cloud shadows, warm city lights; the sunlight is reddened near the terminator by the atmosphere.
//   clouds  — a slightly larger sphere, alpha from the packed cloud channel, lit by the same sun with a soft self-shadow,
//             rotating a little faster than the ground.
//   atmosphere — a front-side shell whose fragment shader integrates single scattering analytically along the view ray
//             (Chapman-function optical depths → Rayleigh in-scatter + extinction of what is behind, Mie forward glow).
// All lighting is done in world space: the shaders get the sun direction, the Earth centre and the camera in world units.
// Textures load asynchronously; until they arrive (or if they fail) the globe renders as a plain blue-grey ball.
import * as THREE from 'three';

// Shared GLSL: column density through an exponential atmosphere along a ray (Schüler's Chapman approximation; X = R/H,
// h = height in scale heights, cz = cos of the zenith angle), the sunlight transmittance, and an output dither.
const ATMO = /* glsl */`uniform float uR, uH; uniform vec3 uBeta;
float chapman(float X, float h, float cz){ float c = sqrt(1.5708 * (X + h)); if (cz >= 0.0) return c / (c * cz + 1.0) * exp(-h);
  float x0 = sqrt(1.0 - cz * cz) * (X + h), c0 = sqrt(1.5708 * x0); return 2.0 * c0 * exp(X - x0) - c / (1.0 - c * cz) * exp(-h); }
vec3 sunTrans(float h, float cz){ return exp(-uBeta * 0.6 * uH * chapman(uR / uH, h, max(cz, -0.2))); }   // ×0.6: multiple scattering fills in some blue
float dither(){ return (fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453) - 0.5) * 0.003; }
`;

// Sphere vertex shader: world normal, world tangent (direction of increasing u — the UV sphere's parallels) and cos(latitude).
const VS = /* glsl */`varying vec3 vN, vP, vT, vO; varying vec2 vUv; varying float vCl;
  void main(){ vUv = uv; vO = position; vec3 t = cross(vec3(0.0, 1.0, 0.0), normal); float l = length(t); vCl = l; t = l > 1e-5 ? t / l : vec3(1.0, 0.0, 0.0);
    mat3 m = mat3(modelMatrix); vN = normalize(m * normal); vT = normalize(m * t);
    vec4 w = modelMatrix * vec4(position, 1.0); vP = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`;

const GROUND_FS = /* glsl */`uniform sampler2D tDay, tNight, tPack, tNormal, tSpec; uniform vec3 uL, uSunCol, uNightCol, uHazeCol;
  uniform float uHasTex, uNight, uCloudShift, uEdge, uNormalScale, uSpec, uHaze; varying vec3 vN, vP, vT, vO; varying vec2 vUv; varying float vCl; ${ATMO}
  float h3(vec3 p) { return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453); }
  float vn3(vec3 p) { vec3 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(h3(i), h3(i + vec3(1, 0, 0)), f.x), mix(h3(i + vec3(0, 1, 0)), h3(i + vec3(1, 1, 0)), f.x), f.y),
               mix(mix(h3(i + vec3(0, 0, 1)), h3(i + vec3(1, 0, 1)), f.x), mix(h3(i + vec3(0, 1, 1)), h3(i + vec3(1, 1, 1)), f.x), f.y), f.z); }
  float fbm3(vec3 p) { return 0.5 * vn3(p) + 0.25 * vn3(p * 2.03) + 0.125 * vn3(p * 4.11) + 0.0625 * vn3(p * 8.3); }
  void main(){
    vec3 N0 = normalize(vN), T = normalize(vT), B = cross(N0, T), V = normalize(cameraPosition - vP);
    vec3 alb = vec3(0.07, 0.11, 0.22), nm = vec3(0.0, 0.0, 1.0), night = vec3(0.0); float ocean = 1.0, rough = 0.3, cloudSh = 0.0;
    vec3 Lt = vec3(dot(uL, T), dot(uL, B), dot(uL, N0));                                      // sun in tangent space
    if (uHasTex > 0.5) {
      alb = texture2D(tDay, vUv).rgb; night = max(texture2D(tNight, vUv).rgb - vec3(0.003, 0.004, 0.012), 0.0);   // drop the map's blue base
      vec3 pk = texture2D(tPack, vUv).rgb; ocean = texture2D(tSpec, vUv).r; nm = texture2D(tNormal, vUv).xyz * 2.0 - 1.0;
      rough = mix(0.32, 0.7, pk.g);
      vec2 sh = vec2(Lt.x / (6.2832 * max(vCl, 0.2)), Lt.y / 3.1416) * (0.006 / max(Lt.z, 0.3));   // where the sun ray meets the cloud deck
      cloudSh = texture2D(tPack, vUv + vec2(uCloudShift, 0.0) + sh).b;
    }
    vec3 N = normalize(T * nm.x * uNormalScale + B * nm.y * uNormalScale + N0 * nm.z);
    N = normalize(mix(N, N0, ocean * 0.85));                                                   // the sea is flat; relief is for land
    float ndl0 = dot(N0, uL), ndv = max(dot(N0, V), 0.0);
    float ndl = max(dot(N, uL), 0.0) * smoothstep(-0.04, 0.06, ndl0);
    vec3 Ts = sunTrans(0.0, ndl0) * uSunCol;                                                   // sunlight at the ground, warm near the terminator
    if (uHaze > 0.001) {                                                                      // close range (atmospheric flight): procedural land texture and sea shimmer, attached to the ground
      float land = 1.0 - ocean, dt0 = fbm3(vO * 0.05) - 0.47, dt1 = fbm3(vO * 0.22 + 7.0) - 0.47, dt2 = fbm3(vO * 1.1 + 3.1) - 0.47;   // 20-, 5- and 1-unit features
      float relief = 0.9 * dt0 + 0.6 * dt1 + 0.35 * dt2;
      alb *= 1.0 + uHaze * land * relief + uHaze * ocean * 0.1 * (vn3(vO * 9.0) - 0.5);
      alb = mix(alb, alb * vec3(0.78, 1.0, 0.7), uHaze * land * smoothstep(0.0, 0.2, -dt0) * 0.6);      // greener lowlands
      alb = mix(alb, alb * vec3(1.15, 1.05, 0.9) + vec3(0.06), uHaze * land * smoothstep(0.12, 0.3, dt0 + 0.5 * dt1) * 0.7);   // pale rocky heights
      float ridge = smoothstep(0.35, 0.7, abs(vn3(vO * 0.35 + 11.0) - 0.5) * 2.0);                     // shaded ridges
      alb *= 1.0 - uHaze * land * 0.35 * ridge;
    }
    vec3 col = alb * Ts * ndl * (1.0 - 0.6 * smoothstep(0.1, 0.9, cloudSh));
    vec3 Hh = normalize(V + uL); float ndh = max(dot(N, Hh), 0.0), sh = 2.0 / (rough * rough) - 2.0;
    float fres = 0.03 + 0.97 * pow(1.0 - max(dot(V, Hh), 0.0), 5.0);
    col += Ts * ocean * uSpec * pow(ndh, sh) * (sh + 2.0) * 0.125 * fres * ndl;                  // sun glint on the ocean
    float dark = 1.0 - smoothstep(-0.12, 0.04, ndl0);
    col += night * uNightCol * uNight * dark + alb * vec3(0.05, 0.07, 0.12) * 0.03;               // city lights; faint starlight
    if (uHaze > 0.001) { float dist = length(cameraPosition - vP); col = mix(col, uHazeCol * (0.25 + 0.75 * max(ndl0, 0.0)), uHaze * 0.85 * (1.0 - exp(-dist / 700.0))); }   // aerial perspective
    gl_FragColor = vec4(col + dither(), smoothstep(0.0, uEdge, ndv)); }`;

const CLOUD_FS = /* glsl */`uniform sampler2D tPack; uniform vec3 uL, uSunCol; uniform float uHasTex, uAlpha;
  varying vec3 vN, vP, vT; varying vec2 vUv; varying float vCl; ${ATMO}
  void main(){ vec3 N = normalize(vN), T = normalize(vT), B = cross(N, T);
    float dens = uHasTex > 0.5 ? texture2D(tPack, vUv).b : 0.0, a = smoothstep(0.1, 0.9, dens) * uAlpha; if (a < 0.003) discard;
    float ndl0 = dot(N, uL);
    vec2 sh = vec2(dot(uL, T) / (6.2832 * max(vCl, 0.2)), dot(uL, B) / 3.1416) * 0.02;         // density toward the sun → soft self-shadow
    float selfSh = clamp((texture2D(tPack, vUv + sh).b - dens) * 1.5, 0.0, 0.5);
    vec3 Ts = sunTrans(0.5, ndl0) * uSunCol; float diff = clamp((ndl0 + 0.12) / 1.12, 0.0, 1.0) * smoothstep(-0.06, 0.06, ndl0);
    vec3 col = Ts * diff * (1.0 - selfSh) * (0.98 - 0.3 * dens) + vec3(0.001, 0.0012, 0.002);
    gl_FragColor = vec4(col + dither(), a); }`;

const ATMO_VS = /* glsl */`varying vec3 vP; void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vP = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`;
const ATMO_FS = /* glsl */`uniform vec3 uC, uL, uSunCol; uniform float uGain, uMie; varying vec3 vP; ${ATMO}
  void main(){ vec3 d = normalize(vP - cameraPosition), oc = cameraPosition - uC;
    float b = dot(oc, d), rm2 = max(dot(oc, oc) - b * b, 0.0), rm = sqrt(rm2), X = uR / uH, hg = uR * uR - rm2;
    vec3 rep; float col, h;                                                                    // representative point, column density, height
    if (hg > 0.0 && b < 0.0) { rep = (oc + (-b - sqrt(hg)) * d) / uR; h = 1.0; float cz = max(dot(-d, rep), 0.0);                    // ray ends on the ground (light sampled 1 H up);
      col = uH * chapman(X, 0.0, cz) * (2.0 - smoothstep(0.0, 0.12, cz)); }                                                       // grazing → full chord (continuous at the limb)
    else { h = (rm - uR) / uH; rep = (oc - b * d) / max(rm, 1e-3); col = 2.0 * uH * chapman(X, h, 0.0); }                          // ray grazes through the shell
    float czs = dot(rep, uL), lit = smoothstep(-0.18, 0.1, czs), mu = dot(d, uL);
    vec3 Tv = exp(-uBeta * col), Ts = sunTrans(h, czs);
    float phR = 0.75 * (1.0 + mu * mu), phM = 0.4224 / pow(1.5776 - 1.52 * mu, 1.5);            // Rayleigh / Henyey-Greenstein (g = 0.76) phases
    vec3 sky = mix(vec3(1.0), vec3(0.62, 0.8, 1.0), smoothstep(0.0, 0.3, czs));                    // day-side Rayleigh bias (keeps the terminator orange)
    vec3 ins = ((1.0 - Tv) * phR * sky + (1.0 - exp(-col * uMie)) * phM * 0.03) * Ts * lit * uSunCol * uGain;
    gl_FragColor = vec4(ins + dither(), 1.0 - dot(Tv, vec3(0.3333))); }`;                      // rgb added, background × (1 − alpha)

/** @param {{ texturePath: string, R: number, position: THREE.Vector3, axis: THREE.Vector3, sunDir: THREE.Vector3, sunCol: THREE.Color, spin?: number, spin0?: number }} o */
export function createEarth({ texturePath, R, position, axis, sunDir, sunCol, spin = 0.004, spin0 = 0 }) {
  const H = R * 0.007;                                                                          // scale height (visual: ~0.7 % of R)
  const beta = new THREE.Vector3(0.04, 0.10, 0.25).divideScalar(H);                             // Rayleigh extinction per unit (vertical optical depth 0.04/0.10/0.25)
  const beta0 = beta.clone(), mie0 = 0.5 / (2 * Math.sqrt(Math.PI * R * H / 2)); let scale = 1;               // setScale() rescales the air's world-space lengths
  const common = () => ({ uR: { value: R }, uH: { value: H }, uBeta: { value: beta }, uL: { value: sunDir }, uSunCol: { value: sunCol } });
  const groundMat = new THREE.ShaderMaterial({
    uniforms: { ...common(), tDay: { value: null }, tNight: { value: null }, tPack: { value: null }, tNormal: { value: null }, tSpec: { value: null },
      uNightCol: { value: new THREE.Color(1.1, 0.9, 0.65) }, uHasTex: { value: 0 }, uNight: { value: 1 }, uCloudShift: { value: 0 }, uEdge: { value: 0.06 },
      uNormalScale: { value: 0.9 }, uSpec: { value: 1.5 }, uHaze: { value: 0 }, uHazeCol: { value: new THREE.Color(0.62, 0.74, 0.9) } },
    vertexShader: VS, fragmentShader: GROUND_FS, transparent: true });
  const cloudMat = new THREE.ShaderMaterial({ uniforms: { ...common(), tPack: { value: null }, uHasTex: { value: 0 }, uAlpha: { value: 1 } },
    vertexShader: VS, fragmentShader: CLOUD_FS, transparent: true, depthWrite: false });
  const atmoMat = new THREE.ShaderMaterial({
    uniforms: { ...common(), uC: { value: new THREE.Vector3() }, uGain: { value: 0.4 }, uMie: { value: mie0 } },
    vertexShader: ATMO_VS, fragmentShader: ATMO_FS, transparent: true, depthWrite: false,
    blending: THREE.CustomBlending, blendSrc: THREE.OneFactor, blendDst: THREE.OneMinusSrcAlphaFactor, blendEquation: THREE.AddEquation });
  const ground = new THREE.Mesh(new THREE.SphereGeometry(R, 256, 160), groundMat); ground.renderOrder = -8;      // 256 segments: the limb chord error is < 0.1 px
  const clouds = new THREE.Mesh(new THREE.SphereGeometry(R * 1.006, 160, 100), cloudMat); clouds.renderOrder = -5;
  const atmo = new THREE.Mesh(new THREE.SphereGeometry(R * 1.05, 96, 60), atmoMat); atmo.renderOrder = -2;
  ground.rotation.y = spin0; clouds.rotation.y = spin0;
  const tilted = new THREE.Group(); tilted.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), axis.clone().normalize()); tilted.add(ground, clouds);
  const group = new THREE.Group(); group.position.copy(position); group.add(tilted, atmo);

  // Textures: any that fails is replaced by a neutral 1×1 fallback; the textured look needs the day and the packed map.
  const loader = new THREE.TextureLoader(), textures = []; let disposed = false;
  const load = (name, srgb) => new Promise((res) => loader.load(texturePath + name,
    (t) => { t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace; t.anisotropy = 8; t.wrapS = THREE.RepeatWrapping; res(t); }, undefined,
    () => { console.warn(`space: texture ${name} failed to load`); res(null); }));
  const flat = (r, g, b) => { const t = new THREE.DataTexture(new Uint8Array([r, g, b, 255]), 1, 1); t.needsUpdate = true; return t; };
  const ready = Promise.all([load('earth_day_2048.jpg', true), load('earth_night_2048.jpg', true), load('earth_bump_roughness_clouds_2048.jpg'),
    load('earth_normal_2048.jpg'), load('earth_specular_2048.jpg')]).then(([day, night, pack, normal, spec]) => {
    if (disposed) { [day, night, pack, normal, spec].forEach((t) => t && t.dispose()); return; }
    if (!day || !pack) { textures.push(...[day, night, pack, normal, spec].filter(Boolean)); return; }
    night = night || flat(0, 0, 0); normal = normal || flat(128, 128, 255); spec = spec || flat(0, 0, 0);
    textures.push(day, night, pack, normal, spec);
    const u = groundMat.uniforms; u.tDay.value = day; u.tNight.value = night; u.tPack.value = pack; u.tNormal.value = normal; u.tSpec.value = spec;
    u.uHasTex.value = 1; cloudMat.uniforms.tPack.value = pack; cloudMat.uniforms.uHasTex.value = 1;
  });

  return {
    group, ready, R, centre: atmoMat.uniforms.uC.value,
    update(dt) {
      ground.rotation.y += dt * spin; clouds.rotation.y += dt * spin * 1.12;
      groundMat.uniforms.uCloudShift.value = (ground.rotation.y - clouds.rotation.y) / (2 * Math.PI);   // cloud deck → ground UV offset
      atmo.getWorldPosition(atmoMat.uniforms.uC.value);
    },
    setMono(mode) { groundMat.uniforms.uNight.value = mode === 2 ? 0 : mode === 1 ? 0.5 : 1; },   // ink: no city-light speckle
    setHaze(h) { groundMat.uniforms.uHaze.value = h; cloudMat.uniforms.uAlpha.value = 1 - h; },   // in the atmosphere the tile imagery carries its own clouds
    setScale(k) {                                                    // the far Earth of a lunar orbit: the globe k× smaller, its atmosphere's lengths with it
      if (k === scale) return; scale = k; group.scale.setScalar(k); beta.copy(beta0).divideScalar(k); atmoMat.uniforms.uMie.value = mie0 / k;
      for (const m of [groundMat, cloudMat, atmoMat]) { m.uniforms.uR.value = R * k; m.uniforms.uH.value = H * k; }
      for (const m of [cloudMat, atmoMat]) { m.polygonOffset = k < 1; m.polygonOffsetFactor = 0; m.polygonOffsetUnits = -4; }   // far away the shells sit within a depth step of the ground: pull them forward
    },
    dispose() { disposed = true; [ground, clouds, atmo].forEach((m) => { m.geometry.dispose(); m.material.dispose(); }); textures.forEach((t) => t.dispose()); },
  };
}
