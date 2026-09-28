// Volumetric clouds for atmospheric flight (never in space: the pass is off unless env.atmosphere): the cells of
// src/weather.js ray-marched at half resolution between the camera and the scene's depth — flat-based cumulus, towering
// cells with rain shafts beneath, valley fog banks — carved by a tileable 3D noise (src/cloudnoise.js) and lit by the sun
// (Beer–Lambert extinction, a dual-lobe Henyey–Greenstein phase for the silver lining, a powder term, a short march toward
// the sun for self-shadowing) and the sky, fading into the scene's haze. The world bends with the Earth's curvature; the
// march undoes it, because the weather lives in the flat frame. A temporal pass reprojects last frame's clouds and blends
// them in (clamped to this frame's neighbourhood) to hide the jitter; the composite upsamples with depth weights over the
// scene and adds god rays. Pass order: RenderPass → clouds → bloom → output. Budget ≈ 5.5 ms at a 1536-px canvas.
import * as THREE from 'three';
import { Pass, FullScreenQuad } from 'three/addons/postprocessing/Pass.js';
import { CELL_GLSL } from './weatherglsl.js';
import { SHADOW_GLSL } from './shadowfield.js';
import { cloudNoiseData } from './cloudnoise.js';
import { cellShape, CELL, LOBE } from './weather.js';

export const MAX_CELLS = 64;
const NOISE_N = 64, _sun = new THREE.Vector3(), _fwd = new THREE.Vector3();
const VS = /* glsl */`varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;
const MARCH_FS = /* glsl */`
  precision highp float; precision highp sampler3D;
  uniform sampler2D tDepth, uCells; uniform sampler3D uNoise; uniform int uCount, uSteps;
  uniform mat4 uInvProj, uCamWorld; uniform vec3 uCamPos, uSunCol, uAmbTop, uAmbBot, uFogCol;   // uSunDir comes with the shared shadow field
  uniform float uNear, uFar, uApexX, uTime, uFogNear, uFogFar, uDensity, uFrame, uGroundY, uExt, uBaseScale, uDetScale, uSunScale, uLobe, uLobeMid, uDrift, uEdge, uEro, uNearFade;
  varying vec2 vUv;
  layout(location = 0) out vec4 oLight; layout(location = 1) out vec4 oDist;
  ${SHADOW_GLSL}
  ${CELL_GLSL}
  float terrainSun(vec3 P) {                                // the valley's walls shade low cloud and fog from a low sun (the shared height field, flat frame)
    if (uShadowsOn < 0.5) return 1.0;
    vec2 dxz = normalize(uSunDir.xz + vec2(1e-4, 0.0)); float rise = max(uSunDir.y, 0.05) / max(length(uSunDir.xz), 1e-3), occ = 0.0;
    for (int k = 1; k <= 6; k++) { float t = float(k * k) * 1.4; occ = max(occ, clamp((fieldH(P.xz + dxz * t) - (P.y + rise * t)) / 2.0, 0.0, 1.0)); }
    return 1.0 - 0.9 * occ;
  }
  float viewZ(float d) { return (uNear * uFar) / ((uFar - uNear) * d - uFar); }
  float ign(vec2 p) { return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715)))); }
  float hg(float c, float g) { float g2 = g * g; return (1.0 - g2) / (12.566371 * pow(max(1e-4, 1.0 + g2 - 2.0 * g * c), 1.5)); }
  float remap(float x, float a, float b, float c, float d) { return c + (x - a) / max(b - a, 1e-4) * (d - c); }
  vec4 cA[8]; vec4 cB[8]; vec2 cT[8]; int nC = 0, gI = 0; float gH = 0.5, gN = 0.5;   // at the last sample: height within the densest cell (bases dark, tops bright), the lobe noise (thick parts darker)
  // the cloud at a point: the Perlin–Worley noise pushes each cell's edge out or in (cauliflower lobes on the sides, a lumpy
  // dome on top, the flat base kept), the edge is crisp and the core solid; Worley detail then erodes the rim — little at the
  // base, more up the tops. Stratus is a thin mist. detail = false is the cheap version for the march toward the sun.
  float dens(vec3 pw, bool detail) {
    vec3 p = vec3(pw.x, pw.y + ((pw.x - uApexX) * (pw.x - uApexX) + pw.z * pw.z) / 2800.0, pw.z);   // undo the Earth's curvature
    vec3 pc = vec3(p.x - uDrift, p.y, p.z);                    // the noise drifts with the cells (and is periodic in 120: lap slides are seamless)
    float nr = texture(uNoise, pc * uBaseScale + vec3(uTime * 0.004, 0.0, 0.0)).r, n = clamp((nr - 0.3) / 0.62, 0.0, 1.0), g = lobeG(nr, uLobe, uLobeMid);   // the lobes the physics flies through too (weather.js)
    float s = 0.0, rain = 0.0, typ = 0.0, hh = 0.5;
    for (int i = 0; i < 8; i++) { if (i >= nC) break;
      float H = cB[i].y - cB[i].x, c = cellLobed(cA[i].w, cA[i].z, cB[i].x, cB[i].y, p.x - cA[i].x, p.y, p.z - cA[i].y, g);   // cauliflower cumulus; fog sheets with ragged edges
      if (c > s) { s = c; typ = cA[i].w; hh = (p.y - cB[i].x) / H; if (detail) gI = i; }
      if (detail) rain = max(rain, rainShape(cA[i].w, cA[i].z, cB[i].x, uGroundY, p.x - cA[i].x, p.y, p.z - cA[i].y)); }
    float d = typ > 1.5 ? s : smoothstep(0.0, uEdge, s); if (detail) { gH = hh; gN = n; }   // a cumulus edge (uEdge: its width in shape units); stratus fades out softly
    if (detail && d > 0.0) { float det = texture(uNoise, pc * uDetScale + vec3(0.0, uTime * 0.01, 0.0)).g; d = clamp(remap(d, det * uEro * (typ > 1.5 ? 0.75 : mix(0.25, 0.6, clamp(hh, 0.0, 1.0))), 1.0, 0.0, 1.0), 0.0, 1.0); }
    d *= typ > 1.5 ? 0.35 : 1.0;                               // stratus: a fog bank, soft-edged, near-zero visibility inside
    if (rain > 0.0) d = max(d, rain * 0.1 * smoothstep(0.35, 0.8, texture(uNoise, vec3(p.x * 0.05, p.y * 0.012 + uTime * 0.35, p.z * 0.05)).g));   // falling streaks
    return d * uDensity;
  }
  float densSun(vec3 pw, int i) {                          // the cheap density of one cell (the one a sample belongs to) for the march toward the sun
    vec3 p = vec3(pw.x, pw.y + ((pw.x - uApexX) * (pw.x - uApexX) + pw.z * pw.z) / 2800.0, pw.z);
    float g = lobeG(texture(uNoise, vec3(p.x - uDrift, p.y, p.z) * uBaseScale + vec3(uTime * 0.004, 0.0, 0.0)).r, uLobe, uLobeMid);
    float s = cellLobed(cA[i].w, cA[i].z, cB[i].x, cB[i].y, p.x - cA[i].x, p.y, p.z - cA[i].y, g);
    return (cA[i].w > 1.5 ? s * 0.35 : smoothstep(0.0, uEdge, s)) * uDensity;
  }
  void main() {
    vec4 vp = uInvProj * vec4(vUv * 2.0 - 1.0, 1.0, 1.0); vec3 vdir = normalize(vp.xyz / vp.w);
    vec3 rd = normalize((uCamWorld * vec4(vdir, 0.0)).xyz), ro = uCamPos;
    float depth = texture(tDepth, vUv).r, sceneT = depth >= 0.999999 ? 1e6 : -viewZ(depth) / max(1e-4, -vdir.z), tEnd = min(sceneT, uFogFar);
    for (int i = 0; i < ${MAX_CELLS}; i++) { if (i >= uCount) break;            // the cells this ray crosses, nearest first, at most 8
      vec4 a = texelFetch(uCells, ivec2(2 * i, 0), 0), b = texelFetch(uCells, ivec2(2 * i + 1, 0), 0);
      float drop = ((a.x - uApexX) * (a.x - uApexX) + a.y * a.y) / 2800.0, bot = (a.w > 0.5 && a.w < 1.5 ? uGroundY : b.x) - drop - 4.0, topY = b.y + 0.15 * uLobe * (b.y - b.x) - drop + 4.0;
      float ex = (1.02 + 0.5 * uLobe) * a.z;                  // room for the lobes (the noise grows a cell by up to uLobe/2)
      vec3 bmin = vec3(a.x - ex, bot, a.y - (a.w > 1.5 ? 0.5 : 1.0) * ex), bmax = vec3(a.x + ex, topY, a.y + (a.w > 1.5 ? 0.5 : 1.0) * ex);
      vec3 inv = 1.0 / rd, t0 = (bmin - ro) * inv, t1 = (bmax - ro) * inv, tl = min(t0, t1), th = max(t0, t1);
      float tn = max(max(tl.x, tl.y), max(tl.z, 0.0)), tf = min(min(th.x, th.y), min(th.z, tEnd));
      if (tn >= tf || (nC == 8 && tn >= cT[7].x)) continue;
      int j = min(nC, 7);
      while (j > 0 && cT[j - 1].x > tn) { cA[j] = cA[j - 1]; cB[j] = cB[j - 1]; cT[j] = cT[j - 1]; j--; }
      cA[j] = a; cB[j] = b; cT[j] = vec2(tn, tf); nC = min(nC + 1, 8);
    }
    float T = 1.0, first = -1.0, jit = ign(gl_FragCoord.xy + uFrame * 5.588238), done = 0.0; vec3 L = vec3(0.0); int used = 0;
    float cosT = dot(rd, uSunDir), iso = 0.0796;             // the phase of each scattering octave, part isotropic: thick cloud returns light diffusely
    float ph0 = mix(mix(hg(cosT, 0.6), hg(cosT, -0.2), 0.3), iso, 0.5), ph1 = mix(hg(cosT, 0.3), iso, 0.6), ph2 = mix(hg(cosT, 0.15), iso, 0.7);
    for (int k = 0; k < 8; k++) { if (k >= nC || T < 0.02 || used >= uSteps) break;
      float t0 = max(cT[k].x, done), t1 = cT[k].y; if (t1 <= t0) continue; done = t1;
      float n = clamp(ceil((t1 - t0) / 2.0), 4.0, 32.0), dt = (t1 - t0) / n;
      for (int q = 0; q < 32; q++) { if (float(q) >= n || T < 0.02 || used >= uSteps) break; used++;
        float t = t0 + (float(q) + jit) * dt; vec3 p = ro + rd * t; float d = dens(p, true) * mix(uNearFade, 1.0, smoothstep(2.0, 12.0, t)); if (d < 0.003) continue;   // thinned near the camera: inside a cloud the ship (5–6 units ahead) shows through the mist, beyond it is white
        if (first < 0.0) first = t;
        float h = gH, tau = 0.0; for (int m = 1; m <= 4; m++) tau += densSun(p + uSunDir * float(m * m) * 1.6, gI) * float(2 * m - 1) * 1.6; tau *= uExt;   // optical depth toward the sun, through the sample's own cell
        float sun = exp(-tau) * ph0 + 0.5 * exp(-0.4 * tau) * ph1 + 0.25 * exp(-0.16 * tau) * ph2   // multiple scattering: weaker, wider octaves …
          + iso / (1.0 + 0.11 * tau);                            // … and the diffuse light a thick cloud lets through (two-stream, g 0.85): a cloud's inside is bright
        sun *= terrainSun(vec3(p.x, p.y + ((p.x - uApexX) * (p.x - uApexX) + p.z * p.z) / 2800.0, p.z));   // in the flat frame, like the terrain
        float powder = 1.0 - exp(-d * uExt * 8.0), sigma = d * uExt;
        vec3 amb = mix(uAmbBot, uAmbTop, smoothstep(-0.1, 0.9, h)) * (0.35 + 0.65 * exp(-0.12 * tau)) * mix(1.2, 0.7, gN);   // skylight: dim at the base, and a deep point sees less sky (its sun-ward depth as the proxy)
        vec3 S = sigma * (uSunCol * sun * mix(0.5, 1.0, powder) * uSunScale + amb);
        S = mix(S, sigma * uFogCol, clamp((t - uFogNear) / (uFogFar - uFogNear), 0.0, 1.0));   // into the haze
        float Tr = exp(-sigma * dt); L += T * (S - S * Tr) / max(sigma, 1e-5); T *= Tr;      // energy-conserving step
      }
    }
    oLight = vec4(L, T); oDist = vec4(first, min(sceneT, 6e4), 0.0, 1.0);   // a half-float target: at most 65504
  }`;
const RESOLVE_FS = /* glsl */`
  uniform sampler2D tCur, tDist, tHist; uniform mat4 uPrevVP, uInvProj, uCamWorld; uniform vec3 uCamPos; uniform vec2 uTexel; uniform float uHistOk;
  varying vec2 vUv;
  void main() {
    vec4 cur = texture2D(tCur, vUv), mn = cur, mx = cur;
    for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) { vec4 c = texture2D(tCur, vUv + vec2(float(i), float(j)) * uTexel); mn = min(mn, c); mx = max(mx, c); }
    vec4 outc = cur;
    if (uHistOk > 0.5) {
      float d = texture2D(tDist, vUv).x, dist = d > 0.0 ? d : 400.0;          // an empty pixel reprojects at a typical cloud distance
      vec4 vp = uInvProj * vec4(vUv * 2.0 - 1.0, 1.0, 1.0); vec3 rd = normalize((uCamWorld * vec4(normalize(vp.xyz / vp.w), 0.0)).xyz);
      vec4 pc = uPrevVP * vec4(uCamPos + rd * dist, 1.0); vec2 puv = pc.xy / pc.w * 0.5 + 0.5;
      if (pc.w > 0.0 && puv.x >= 0.0 && puv.y >= 0.0 && puv.x <= 1.0 && puv.y <= 1.0) outc = mix(cur, clamp(texture2D(tHist, puv), mn, mx), 0.85);
    }
    gl_FragColor = outc;
  }`;
const COMPOSITE_FS = /* glsl */`
  uniform sampler2D tScene, tDepth, tClouds, tDist; uniform mat4 uInvProj; uniform vec2 uHalfTexel, uSunUv; uniform float uNear, uFar, uGod; uniform vec3 uSunCol;
  varying vec2 vUv;
  float viewZ(float d) { return (uNear * uFar) / ((uFar - uNear) * d - uFar); }
  void main() {
    vec4 scene = texture2D(tScene, vUv); float depth = texture2D(tDepth, vUv).r;
    vec4 vp = uInvProj * vec4(vUv * 2.0 - 1.0, 1.0, 1.0); vec3 vdir = normalize(vp.xyz / vp.w);
    float here = depth >= 0.999999 ? 6e4 : -viewZ(depth) / max(1e-4, -vdir.z);
    vec4 acc = vec4(0.0); float wsum = 0.0;                  // depth-aware upsample: half-res texels whose scene depth matches this pixel's count most
    for (int j = 0; j < 2; j++) for (int i = 0; i < 2; i++) { vec2 o = (vec2(float(i), float(j)) - 0.5) * uHalfTexel; float sd = texture2D(tDist, vUv + o).y;
      float w = 1.0 / (0.02 + abs(sd - here) / max(here, 1.0)); acc += texture2D(tClouds, vUv + o) * w; wsum += w; }
    vec4 c = acc / wsum; vec3 col = scene.rgb * c.a + c.rgb;
    if (uGod > 0.0) {                                        // god rays: open sky past the clouds, smeared toward the sun
      vec2 st = (uSunUv - vUv) / 24.0, uv = vUv; float g = 0.0, w = 1.0;
      for (int k = 0; k < 24; k++) { uv += st; g += (texture2D(tDepth, uv).r >= 0.999999 ? 1.0 : 0.0) * texture2D(tClouds, uv).a * w; w *= 0.94; }
      col += uSunCol * g / 24.0 * uGod;
    }
    gl_FragColor = vec4(col, scene.a);
  }`;

const u = (v) => ({ value: v });
export class CloudPass extends Pass {
  constructor(camera) {
    super(); this.camera = camera; this.needsSwap = true; this.enabled = false; this.forceOff = false; this.scale = 0.5; this.steps = 48; this.frame = 0; this.god = 0;
    this.cells = new THREE.DataTexture(new Float32Array(MAX_CELLS * 8), MAX_CELLS * 2, 1, THREE.RGBAFormat, THREE.FloatType); this.cells.needsUpdate = true;
    this.noise = null; this.noiseData = null; this.noiseSync = false; this.list = []; this.prevVP = new THREE.Matrix4(); this.prevCam = new THREE.Vector3(1e9, 0, 0);
    const hf = { type: THREE.HalfFloatType, depthBuffer: false };
    this.rtMarch = new THREE.WebGLRenderTarget(1, 1, { ...hf, count: 2 }); this.rtHist = [new THREE.WebGLRenderTarget(1, 1, hf), new THREE.WebGLRenderTarget(1, 1, hf)];
    this.march = new FullScreenQuad(new THREE.ShaderMaterial({ glslVersion: THREE.GLSL3, vertexShader: VS, fragmentShader: MARCH_FS, uniforms: {
      tDepth: u(null), uCells: u(this.cells), uNoise: u(null), uCount: u(0), uSteps: u(48), uInvProj: u(new THREE.Matrix4()), uCamWorld: u(new THREE.Matrix4()), uCamPos: u(new THREE.Vector3()),
      uSunDir: u(new THREE.Vector3(0.6, 0.35, -0.7).normalize()), uSunCol: u(new THREE.Color(1.0, 0.95, 0.88)), uAmbTop: u(new THREE.Color(0.30, 0.36, 0.46)), uAmbBot: u(new THREE.Color(0.12, 0.13, 0.15)),
      uFogCol: u(new THREE.Color(0x9fbbd8)), uNear: u(0.1), uFar: u(9000), uApexX: u(0), uTime: u(0), uFogNear: u(70), uFogFar: u(560), uDensity: u(1), uFrame: u(0), uGroundY: u(-26),
      uExt: u(0.9), uBaseScale: u(LOBE.scale), uDetScale: u(0.1), uSunScale: u(6.0), uLobe: u(LOBE.amount), uLobeMid: u(LOBE.median), uDrift: u(0), uEdge: u(0.45), uEro: u(1.3), uNearFade: u(0.06),
      uHeight: u(null), uY0: u(-26), uShadowsOn: u(0), uPeriod: u(120), uShift: u(0) } }));   // shadow-field stand-ins until setShadow()
    this.resolve = new FullScreenQuad(new THREE.ShaderMaterial({ vertexShader: VS, fragmentShader: RESOLVE_FS, uniforms: {
      tCur: u(null), tDist: u(null), tHist: u(null), uPrevVP: u(new THREE.Matrix4()), uInvProj: u(new THREE.Matrix4()), uCamWorld: u(new THREE.Matrix4()), uCamPos: u(new THREE.Vector3()), uTexel: u(new THREE.Vector2()), uHistOk: u(0) } }));
    this.composite = new FullScreenQuad(new THREE.ShaderMaterial({ vertexShader: VS, fragmentShader: COMPOSITE_FS, uniforms: {
      tScene: u(null), tDepth: u(null), tClouds: u(null), tDist: u(null), uInvProj: u(new THREE.Matrix4()), uHalfTexel: u(new THREE.Vector2()), uSunUv: u(new THREE.Vector2()), uNear: u(0.1), uFar: u(9000), uGod: u(0), uSunCol: u(new THREE.Color(1.0, 0.92, 0.8)) } }));
    this.buildNoise();
  }
  buildNoise() {                                           // the 64³ noise takes ~0.5 s: built in a worker at start-up, long before the first descent
    try {
      const src = `import { cloudNoiseData } from '${new URL('./cloudnoise.js', import.meta.url).href}'; onmessage = (e) => { const d = cloudNoiseData(e.data); postMessage(d, [d.buffer]); };`;
      const w = new Worker(URL.createObjectURL(new Blob([src], { type: 'text/javascript' })), { type: 'module' });
      w.onmessage = (e) => { this.noiseData = e.data; w.terminate(); }; w.onerror = () => { this.noiseSync = true; w.terminate(); }; w.postMessage(NOISE_N);
    } catch (e) { this.noiseSync = true; }                 // no module workers: built here on the first descent
  }
  setShadow(U) { const M = this.march.material.uniforms; for (const k of Object.keys(U)) M[k] = U[k]; }   // share the terrain's shadow field (and its sun)
  setSize(w, h) { this.W = w; this.H = h; const hw = Math.max(1, Math.round(w * this.scale)), hh = Math.max(1, Math.round(h * this.scale)); this.rtMarch.setSize(hw, hh); for (const t of this.rtHist) t.setSize(hw, hh); this.frame = 0; }
  quality(renderScale) { this.steps = renderScale >= 0.9 ? 48 : 32; const s = renderScale <= 0.6 ? 0.25 : 0.5; if (s !== this.scale) { this.scale = s; if (this.W) this.setSize(this.W, this.H); } }
  update({ env, atmosphere = 0, sunDir = null, fog = null, time = 0 }) {
    const now = env && env.atmosphere ? env.weather : null; if (now) this.lastW = now; else if (atmosphere <= 0.02) this.lastW = null;   // climbing out the sky fades with the air instead of vanishing
    const W = now || this.lastW; this.enabled = !this.forceOff && !!W && atmosphere > 0.02; if (!this.enabled) return;
    if (!this.noise) {
      const data = this.noiseData || (this.noiseSync ? cloudNoiseData(NOISE_N) : null); if (!data) { this.enabled = false; return; }   // the worker is still busy
      const N = NOISE_N, t = new THREE.Data3DTexture(data, N, N, N); t.format = THREE.RGBAFormat; t.type = THREE.UnsignedByteType; t.minFilter = t.magFilter = THREE.LinearFilter; t.wrapS = t.wrapT = t.wrapR = THREE.RepeatWrapping; t.unpackAlignment = 1; t.needsUpdate = true; this.noise = t;
    }
    const M = this.march.material.uniforms, cam = this.camera.position, list = W.cellsNear(cam.x, 480, this.list);
    list.sort((a, b) => (a.x - cam.x) ** 2 + (a.z - cam.z) ** 2 - (b.x - cam.x) ** 2 - (b.z - cam.z) ** 2);
    const d = this.cells.image.data, n = Math.min(MAX_CELLS, list.length);
    for (let i = 0; i < n; i++) { const c = list[i], o = i * 8; d[o] = c.x; d[o + 1] = c.z; d[o + 2] = c.R; d[o + 3] = c.type; d[o + 4] = c.base; d[o + 5] = c.top; d[o + 6] = c.sig; d[o + 7] = c.seed; }
    this.cells.needsUpdate = true;
    M.uCount.value = n; M.uNoise.value = this.noise; M.uDensity.value = Math.min(1, atmosphere); M.uApexX.value = cam.x + 10; M.uTime.value = W.time; M.uDrift.value = W.drift; M.uGroundY.value = W.floor;   // the weather's own clock: the lobes match the physics
    if (sunDir) M.uSunDir.value.copy(sunDir).normalize();
    if (fog) { M.uFogCol.value.copy(fog.color); M.uFogNear.value = fog.near; M.uFogFar.value = fog.far; }
    const C = this.composite.material.uniforms; _sun.copy(M.uSunDir.value).multiplyScalar(1000).add(cam).project(this.camera);   // where the sun is on screen
    C.uSunUv.value.set(_sun.x * 0.5 + 0.5, _sun.y * 0.5 + 0.5); const facing = this.camera.getWorldDirection(_fwd).dot(M.uSunDir.value);
    this.god = facing > 0 ? 0.35 * Math.min(1, 2 * facing) * M.uDensity.value : 0;   // only when looking toward the sun
  }
  render(renderer, writeBuffer, readBuffer) {
    const cam = this.camera, M = this.march.material.uniforms, R = this.resolve.material.uniforms, C = this.composite.material.uniforms;
    cam.updateMatrixWorld(); for (const U of [M, R, C]) U.uInvProj.value.copy(cam.projectionMatrixInverse);
    M.uCamWorld.value.copy(cam.matrixWorld); R.uCamWorld.value.copy(cam.matrixWorld); M.uCamPos.value.setFromMatrixPosition(cam.matrixWorld); R.uCamPos.value.copy(M.uCamPos.value);
    M.tDepth.value = readBuffer.depthTexture; M.uNear.value = C.uNear.value = cam.near; M.uFar.value = C.uFar.value = cam.far; M.uFrame.value = this.frame % 64; M.uSteps.value = this.steps;
    renderer.setRenderTarget(this.rtMarch); this.march.render(renderer);
    const hist = this.rtHist[this.frame & 1], prev = this.rtHist[(this.frame + 1) & 1], jump = M.uCamPos.value.distanceTo(this.prevCam) > 20;   // a lap wrap moves the camera by 120: no history then
    R.tCur.value = this.rtMarch.textures[0]; R.tDist.value = this.rtMarch.textures[1]; R.tHist.value = prev.texture; R.uHistOk.value = jump || this.frame === 0 ? 0 : 1;
    R.uPrevVP.value.copy(this.prevVP); R.uTexel.value.set(1 / this.rtMarch.width, 1 / this.rtMarch.height);
    renderer.setRenderTarget(hist); this.resolve.render(renderer);
    C.tScene.value = readBuffer.texture; C.tDepth.value = readBuffer.depthTexture; C.tClouds.value = hist.texture; C.tDist.value = this.rtMarch.textures[1];
    C.uHalfTexel.value.set(1 / this.rtMarch.width, 1 / this.rtMarch.height); C.uGod.value = this.god;
    renderer.setRenderTarget(this.renderToScreen ? null : writeBuffer); this.composite.render(renderer);
    this.prevVP.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse); this.prevCam.copy(M.uCamPos.value); this.frame++;
  }
  dispose() { this.rtMarch.dispose(); for (const t of this.rtHist) t.dispose(); this.cells.dispose(); if (this.noise) this.noise.dispose(); for (const q of [this.march, this.resolve, this.composite]) { q.material.dispose(); q.dispose(); } }
}

// Cloud shadows: the sun's transmittance through the cells over a 480 × 128 box around the camera (128 × 32 texels), read
// by every shader that uses the shadow field (sunVis, sunVisSoft). Each cell is rasterised only where its shadow falls (its
// footprint at six heights, shifted down-sun), so the whole map is rebuilt cheaply, every fourth frame. 8-bit texels: a
// float texture would need an optional extension to filter.
export function createCloudShadows(U) {
  const NX = 128, NZ = 32, data = new Uint8Array(NX * NZ).fill(255), od = new Float32Array(NX * NZ), tex = new THREE.DataTexture(data, NX, NZ, THREE.RedFormat, THREE.UnsignedByteType);
  tex.minFilter = tex.magFilter = THREE.LinearFilter; tex.unpackAlignment = 1; tex.needsUpdate = true; U.uCloudTex.value = tex;
  const box = [0, -64, 480, 128], list = []; let frame = 0, lastW = null;
  return {
    update(env, camX, sunDir, atmosphere) {
      const now = env.atmosphere ? env.weather : null; if (now) lastW = now; else if (atmosphere <= 0.5) lastW = null;
      const W = now || lastW; U.uCloudOn.value = W && atmosphere > 0.5 ? 1 : 0; if (!U.uCloudOn.value || frame++ % 4) return;
      box[0] = camX - 160; U.uCloudBox.value.set(box[0], box[1], box[2], box[3]); od.fill(0);
      const cells = W.cellsNear(camX + 80, 330, list), sy = Math.max(0.2, sunDir.y), sx = box[2] / NX, sz = box[3] / NZ;
      for (let k = 0; k < 6; k++) {
        const y = W.floor + 12 + 12 * k, t = (y - W.floor) / sy, ox = sunDir.x * t, oz = sunDir.z * t;   // a texel's sun ray meets height y at (gx + ox, gz + oz)
        for (const c of cells) { if (y < c.base || y > c.top) continue; const rz = c.type === CELL.STRATUS ? c.R / 2 : c.R;
          const i0 = Math.max(0, Math.floor((c.x - ox - c.R - box[0]) / sx)), i1 = Math.min(NX - 1, Math.ceil((c.x - ox + c.R - box[0]) / sx));
          const j0 = Math.max(0, Math.floor((c.z - oz - rz - box[1]) / sz)), j1 = Math.min(NZ - 1, Math.ceil((c.z - oz + rz - box[1]) / sz));
          for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) od[j * NX + i] += cellShape(c.type, c.R, c.base, c.top, box[0] + (i + 0.5) * sx + ox - c.x, y, box[1] + (j + 0.5) * sz + oz - c.z); }
      }
      for (let i = 0; i < NX * NZ; i++) data[i] = Math.round(255 * Math.exp(-od[i] * 0.25));
      tex.needsUpdate = true;
    },
  };
}
