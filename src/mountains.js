// Alpine range renderer for atmospheric flight: the environment's height field as one BatchedMesh (one draw call) of
// half-period slots that follow the camera — three fine slots (0.25 units along the corridor, a rock relief of ≤ 0.20 units
// on top; the physics field stays authoritative) and five slots of the physics grid itself farther out — shaded per pixel.
// Procedural rock detail bends the normal (bedding planes that also notch the silhouette, couloirs and erosion streaks down
// the faces, grain), material zones follow altitude, slope, curvature and exposure (conifer forest → alpine meadow → strata
// rock and scree fans → snow that fills hollows and lee slopes and lets the rock poke through), lit by a warm low sun with
// cast shadows from the shared shadow field, blue sky light with baked ambient occlusion, ground/snow bounce, rim light on
// backlit crests, drifting cloud shadows, valley mist and the scene's haze.
import * as THREE from 'three';
import { SHADOW_GLSL } from './shadowfield.js';

const NOISE = /* glsl */`
  float h21(vec2 p) { vec3 q = fract(vec3(p.xyx) * 0.1031); q += dot(q, q.yzx + 33.33); return fract((q.x + q.y) * q.z); }
  float h31(vec3 p) { p = fract(p * 0.1031); p += dot(p, p.zyx + 31.32); return fract((p.x + p.y) * p.z); }
  vec3 nd2(vec2 p, float px) {                                                                 // value noise + gradient; the lattice repeats every px cells in x (= the corridor period)
    vec2 i = floor(p), f = fract(p), u = f * f * (3.0 - 2.0 * f), du = 6.0 * f * (1.0 - f); float x0 = mod(i.x, px), x1 = mod(i.x + 1.0, px);
    float a = h21(vec2(x0, i.y)), b = h21(vec2(x1, i.y)), c = h21(vec2(x0, i.y + 1.0)), d = h21(vec2(x1, i.y + 1.0)), k1 = b - a, k2 = c - a, k3 = a - b - c + d;
    return vec3(a + k1 * u.x + k2 * u.y + k3 * u.x * u.y, du.x * (k1 + k3 * u.y), du.y * (k2 + k3 * u.x)); }
  vec4 nd3(vec3 p, float px) {                                                                 // 3-D value noise + gradient, periodic in x
    vec3 i = floor(p), f = fract(p), u = f * f * (3.0 - 2.0 * f), du = 6.0 * f * (1.0 - f); float x0 = mod(i.x, px), x1 = mod(i.x + 1.0, px);
    float a = h31(vec3(x0, i.yz)), b = h31(vec3(x1, i.yz)), c = h31(vec3(x0, i.y + 1.0, i.z)), d = h31(vec3(x1, i.y + 1.0, i.z));
    float e = h31(vec3(x0, i.y, i.z + 1.0)), g = h31(vec3(x1, i.y, i.z + 1.0)), h = h31(vec3(x0, i.yz + 1.0)), k = h31(vec3(x1, i.yz + 1.0));
    float k1 = b - a, k2 = c - a, k3 = e - a, k4 = a - b - c + d, k5 = a - c - e + h, k6 = a - b - e + g, k7 = -a + b + c - d + e - g - h + k;
    return vec4(a + k1 * u.x + k2 * u.y + k3 * u.z + k4 * u.x * u.y + k5 * u.y * u.z + k6 * u.z * u.x + k7 * u.x * u.y * u.z,
      du.x * (k1 + k4 * u.y + k6 * u.z + k7 * u.y * u.z), du.y * (k2 + k4 * u.x + k5 * u.z + k7 * u.z * u.x), du.z * (k3 + k5 * u.y + k6 * u.x + k7 * u.x * u.y)); }`;

const VERT = /* glsl */`attribute vec4 aBake; attribute vec3 aBlotch; attribute float aS; uniform float uApexX;   // aBake: sky visibility, small / large curvature, bedding mask; aBlotch: slow patchiness; aS: strata coordinate
  varying vec3 vP, vN, vL, vBlotch; varying vec4 vBake; varying float vS, vFlatY, vSunVis;
  ${SHADOW_GLSL}
  #include <batching_pars_vertex>
  void main(){ vec4 mp = vec4(position, 1.0);
  #ifdef USE_BATCHING
    #include <batching_vertex>
    mp = batchingMatrix * mp;                                                                    // the copy's translation along the corridor
  #endif
  #ifdef USE_INSTANCING
    mp = instanceMatrix * mp;
  #endif
    vec4 w = modelMatrix * mp; vFlatY = w.y; vL = position;                                   // local coordinates repeat with the period: the copies share one seamless detail
    w.y -= ((w.x - uApexX) * (w.x - uApexX) + w.z * w.z) / 2800.0; vP = w.xyz;                // the shared Earth curvature (R = 1400), on the world position
    vN = normalize(mat3(modelMatrix) * normal); vBake = aBake; vS = aS;
    vSunVis = sunVis(vec3(w.x, vFlatY, w.z) + vN * 0.7);                                       // cast shadows from the shared field, per vertex (the field is coarser than the mesh)
    vBlotch = aBlotch;
    gl_Position = projectionMatrix * viewMatrix * w; }`;

const FRAG = /* glsl */`uniform vec3 uSun, uHaze; uniform float uAmp, uVis, uSnow, uTime, uStyle;   // uAmp = summit height above the floor; uStyle 0 = Alps, 1 = Zhangjiajie pillars
  ${SHADOW_GLSL}
  ${NOISE}
  varying vec3 vP, vN, vL, vBlotch; varying vec4 vBake; varying float vS, vFlatY, vSunVis;
  #define TAU 6.2831853
  void main(){
    vec3 Ng = normalize(vN), toCam = cameraPosition - vP; float dist = length(toCam); vec3 V = toCam / dist, P = vL;
    float rel = clamp((vFlatY - uY0) / uAmp, 0.0, 1.0), slopeG = 1.0 - Ng.y, ao = vBake.x, curvS = vBake.y, curvL = vBake.z, massive = vBake.w;
    float near = exp(-dist / 45.0), mid = exp(-dist / 200.0);                                   // the finer the detail, the sooner it fades with distance (no shimmer)
    // ---- zones from the large-scale shape: slow patchiness drives the tree line, snow drifts and rock tones
    vec3 pt = vBlotch, pt2 = vBlotch.yyy; float blotch = 0.65 * pt.x + 0.35 * pt2.x, tone = vBlotch.z;
    float treeLine = 0.44 + 0.18 * (blotch - 0.5);
    float pil = uStyle;
    float forestK = mix((1.0 - smoothstep(treeLine - 0.06, treeLine + 0.02, rel)) * (1.0 - smoothstep(0.40, 0.60, slopeG)),   // Alps: conifers up to the tree line, not on cliffs
      (1.0 - smoothstep(0.28, 0.55, slopeG)) * (0.7 + 0.3 * blotch), pil);                        // pillars: woods wherever the ground is gentle — the valley floor and the pillar tops
    // ---- relief: a height function of position; its tangential gradient bends the normal
    vec3 dp1 = dFdx(vP), dp2 = dFdy(vP); vec3 gS = (dFdx(vS) * cross(dp2, Ng) + dFdy(vS) * cross(Ng, dp1)) / (dot(dp1, cross(dp2, Ng)) + 1e-12);   // ∇ of the baked strata coordinate on the surface
    gS /= max(1.0, length(gS) / 2.5);
    float s = vS, band = 0.7 * sin(s * TAU) + 0.3 * sin(2.0 * s * TAU + 1.0), dband = TAU * (0.7 * cos(s * TAU) + 0.6 * cos(2.0 * s * TAU + 1.0));   // bedding planes (the coordinate counts beds)
    float rockDetail = (1.0 - 0.85 * forestK) * mid;
    vec3 grad = (0.07 * smoothstep(0.15, 0.55, slopeG) * massive * rockDetail) * dband * gS;
    vec3 e0 = nd2(P.xz * 0.5 + 2.0, 60.0); float rg = 1.0 - abs(2.0 * e0.x - 1.0), steepK = smoothstep(0.18, 0.55, slopeG);   // couloirs: ridged noise, constant in y, creases the faces top to bottom
    grad += (-1.2 * rg * -2.0 * sign(2.0 * e0.x - 1.0)) * 0.5 * vec3(e0.y, 0.0, e0.z) * steepK * rockDetail * 0.7;
    float gully = smoothstep(0.55, 0.95, rg) * steepK;
    vec3 e1 = nd2(P.xz * 0.9 + 0.3 * P.y, 108.0); vec4 e2 = vec4(0.5, 0.0, 0.0, 0.0); float fE2 = 1.0 - smoothstep(70.0, 120.0, dist);   // ribs (leaning slightly) and, closer, erosion streaks: vertically stretched noise runs down the faces
    if (fE2 > 0.0) e2 = mix(e2, nd3(vec3(P.x * 2.5, P.y * 0.6, P.z * 2.5) + 7.0, 300.0), fE2);
    grad += (0.15 + 0.45 * smoothstep(0.15, 0.6, slopeG)) * (0.6 + 0.8 * pt.x) * rockDetail * (0.9 * 0.9 * vec3(e1.y, 0.3 * (e1.y + e1.z), e1.z) + 0.5 * vec3(2.5, 0.6, 2.5) * e2.yzw);   // some faces are more gullied than others
    vec4 g1 = vec4(0.5, 0.0, 0.0, 0.0); float fG1 = 1.0 - smoothstep(140.0, 220.0, dist), fG2 = 1.0 - smoothstep(30.0, 60.0, dist);   // rock grain, three octaves, each only where it spans pixels
    if (fG1 > 0.0) { g1 = mix(g1, nd3(P * 2.2 + 3.0, 264.0), fG1); grad += 0.16 * 2.2 * g1.yzw * rockDetail; }
    if (fG2 > 0.0) { vec4 g2 = nd3(P * 6.0 + 9.0, 720.0); grad += 0.10 * 6.0 * g2.yzw * rockDetail * fG2; }
    if (dist < 35.0) { vec4 g3 = nd3(P * 15.0 + 1.0, 1800.0); grad += 0.035 * 15.0 * g3.yzw * rockDetail * (1.0 - smoothstep(15.0, 35.0, dist)); }
    float rockH = 0.5 * band * massive + 1.2 * (e1.x - 0.5) + 0.8 * (e2.x - 0.5) + 0.8 * (g1.x - 0.5) - 1.2 * gully;   // rock relief height (≈ −1..1): dirt darkens the hollows
    float snowFill = (0.6 + 0.7 * steepK) * (0.4 + 0.6 * mid) * (e1.x - 0.5) + (0.8 * (e2.x - 0.5) + 0.4 * (g1.x - 0.5)) * mid - 1.2 * gully - 0.6 * smoothstep(0.2, 0.8, band) * massive * steepK - 0.8 * (pt.x - 0.5) * (1.0 - steepK);   // what the snow settles into: gullies, streaks and ledges on the faces, broad fields on gentle ground
    float canopy = 0.5;
    float fC = (1.0 - smoothstep(90.0, 150.0, dist)) * forestK;
    if (fC > 0.01) { vec3 c1 = nd2(P.xz * 1.3, 156.0), c2 = nd2(P.xz * 3.5 + 5.0, 420.0);   // canopy: lumpy crowns
      grad += fC * mid * (0.45 * 1.3 * vec3(c1.y, 0.0, c1.z) + 0.20 * 3.5 * vec3(c2.y, 0.0, c2.z)); canopy = mix(0.5, 0.6 * c1.x + 0.4 * c2.x, fC / forestK); }
    vec3 gt = grad - Ng * dot(grad, Ng), N = normalize(Ng - gt); float slope = 1.0 - N.y;
    // ---- materials
    float meadowK = (1.0 - pil) * smoothstep(treeLine - 0.04, treeLine + 0.06, rel) * (1.0 - smoothstep(0.48, 0.60, rel + 0.3 * (blotch - 0.5))) * (1.0 - smoothstep(0.28, 0.48, slopeG));   // alpine grass on the gentle ground above the trees
    float screeK = (1.0 - 0.7 * pil) * smoothstep(0.22, 0.42, rel) * smoothstep(0.10, 0.26, slopeG) * (1.0 - smoothstep(0.40, 0.58, slopeG)) * (1.0 - smoothstep(-0.35, 0.05, curvL));   // fans below the cliffs: concave, moderate slopes
    float lee = 0.5 - 0.5 * N.x, snowLine = uSnow + 0.20 * (blotch - 0.5);                  // wind from the +x: drifts settle on faces looking away from it
    float snowAmt = 2.5 * (rel - snowLine) + 0.7 * (0.5 - slopeG) - 0.25 * curvS + 0.25 * (lee - 0.5) + 0.2 * (pt2.x - 0.5) - 0.30 * snowFill - 20.0 * pil;   // no snow on the subtropical pillars
    float sw = 0.08 + 0.3 * (1.0 - mid); float snowK = smoothstep(-sw, sw, snowAmt) * (1.0 - smoothstep(0.66, 0.88, slope)) * (1.0 - 0.9 * forestK);   // no snow on the sheer bits or under the trees
    N = normalize(Ng - gt * (1.0 - 0.75 * snowK));                                             // snow blankets the relief
    vec3 conifer = mix(vec3(0.007, 0.018, 0.006), vec3(0.055, 0.10, 0.028), canopy);
    conifer = mix(conifer, vec3(0.10, 0.12, 0.035), smoothstep(0.6, 0.9, blotch) * smoothstep(treeLine - 0.14, treeLine, rel));   // larches and clearings toward the tree line
    conifer = mix(conifer, mix(vec3(0.030, 0.085, 0.022), vec3(0.15, 0.24, 0.07), canopy), pil);   // pillars: broadleaf woods, brighter and lusher
    vec3 grass = mix(vec3(0.10, 0.15, 0.04), vec3(0.22, 0.21, 0.08), pt2.x);
    float warm = smoothstep(0.35, 0.7, 0.5 + 0.5 * band) * massive;
    vec3 rock = mix(mix(vec3(0.20, 0.195, 0.19), vec3(0.34, 0.33, 0.31), pt.x), vec3(0.31, 0.26, 0.20), smoothstep(0.35, 0.75, pt2.x) * 0.6);   // gneiss greys and brown
    rock = mix(rock, mix(vec3(0.46, 0.43, 0.38), vec3(0.62, 0.58, 0.49), pt.x), pil);            // pillars: pale quartz sandstone
    rock = mix(rock, vec3(0.38, 0.33, 0.26), 0.3 * warm) * (0.72 + 0.56 * tone);                  // ochre beds; whole faces lighter or darker
    rock = mix(rock, vec3(0.34, 0.24, 0.16), 0.35 * smoothstep(0.55, 0.85, pt.x) * smoothstep(0.5, 0.75, rel));   // rusty high beds
    rock *= 0.45 + 0.55 * smoothstep(-0.7, 0.7, rockH);                                       // cracks and gullies stay dark (dirt, moisture)
    rock = mix(rock, vec3(0.16, 0.18, 0.09), 0.5 * smoothstep(0.25, 0.6, pt2.x) * (1.0 - smoothstep(0.45, 0.65, rel)) * (1.0 - smoothstep(0.5, 0.8, slopeG)));   // lichen and turf on the lower, gentler rock
    vec3 scree = mix(vec3(0.36, 0.35, 0.33), vec3(0.45, 0.43, 0.40), g1.x), snow = vec3(0.88, 0.91, 0.96);
    vec3 alb = mix(rock, scree, screeK); alb = mix(alb, grass, meadowK); alb = mix(alb, conifer, forestK); alb = mix(alb, snow, snowK);
    // ---- light: warm low sun with cast shadows and cloud shadows, blue sky with AO, the bright sky around the sun, ground bounce
    vec3 L = normalize(uSun); float el = clamp(L.y, 0.0, 1.0);
    vec3 sunCol = mix(vec3(1.0, 0.74, 0.55), vec3(1.0, 0.95, 0.88), smoothstep(0.0, 0.3, el)) * (1.0 + 0.4 * smoothstep(0.0, 0.4, el));
    float vis = vSunVis;
    if (vis > 0.2) vis *= 1.0 - 0.55 * smoothstep(0.50, 0.68, 0.6 * nd2(P.xz * 0.05 + uTime * vec2(0.010, 0.004), 6.0).x + 0.4 * nd2(P.xz * 0.1 + uTime * vec2(0.014, 0.006) + 3.0, 12.0).x);   // drifting cloud shadows
    float ndl = max(dot(N, L), 0.0) * smoothstep(-0.05, 0.3, dot(Ng, L));                    // bumps only light up where the face itself sees the sun
    vec3 direct = sunCol * ndl * vis * mix(0.45 + 0.55 * smoothstep(-0.8, 0.4, rockH), 1.0, max(snowK, forestK));   // the relief shadows its own hollows
    vec3 skyCol = mix(vec3(0.60, 0.70, 0.86), vec3(0.28, 0.42, 0.78), 0.5 + 0.5 * N.y) * 0.80 * (0.5 + 0.5 * N.y) * ao;
    vec3 Lh = normalize(vec3(L.x, 0.25, L.z)); vec3 glow = sunCol * 0.20 * max(dot(N, Lh), 0.0) * ao;   // the sky is brightest around the sun: backlit faces stay readable
    float snowAround = smoothstep(uSnow - 0.15, uSnow + 0.05, rel);
    vec3 bounce = mix(vec3(0.12, 0.11, 0.07), vec3(0.55, 0.58, 0.66), snowAround) * sunCol * 0.35 * (0.5 - 0.5 * N.y) * (0.4 + 0.6 * vis);
    vec3 col = alb * (direct * (1.0 - 0.25 * forestK) + (skyCol + glow + bounce) * (1.0 - 0.5 * forestK));   // a canopy shades itself
    float spec = pow(max(dot(reflect(-L, N), V), 0.0), 48.0) * vis; col += snowK * sunCol * spec * 0.25;   // snow sheen
    col += snowK * sunCol * vis * ndl * step(0.975, h21(floor(P.xz * 60.0) + floor(P.y * 60.0))) * 2.0 * near;   // glints on sunlit snow, close up
    float rim = pow(1.0 - max(dot(N, V), 0.0), 3.0) * smoothstep(0.0, 0.7, dot(-V, L)); col += sunCol * rim * vis * (0.12 + 0.3 * snowK);   // backlit crests
    // ---- atmosphere: mist pooled in the valleys (exponential height fog), then the scene's aerial perspective
    float cy = cameraPosition.y - uY0, py = vFlatY - uY0, dy = cy - py, Hm = mix(4.0, 9.0, pil);   // the pillars stand in deep mist
    float path = abs(dy) > 0.05 ? dist * Hm * (exp(-py / Hm) - exp(-cy / Hm)) / dy : dist * exp(-py / Hm);
    col = mix(col, uHaze * vec3(0.92, 0.96, 1.0), min(mix(0.55, 0.80, pil), 1.0 - exp(-mix(0.009, 0.016, pil) * path)));
    col = mix(col, uHaze, 0.85 * (1.0 - exp(-dist / 700.0)));
    gl_FragColor = vec4(col, uVis); }`;                                                        // opaque to the ground: the range stands on the imagery, its floor hidden beneath it

const SKIRT = 1.5, DROP = 4;                                // skirt rows 1.5 units outside the field and 4 units below the floor close the range's sides
const sm = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
function makeNoise(period) {                                // periodic-in-x value noise for the vertex-level relief (any frequency with an integer number of cells per period)
  const hash = (ix, iz) => { let n = Math.imul(ix, 374761393) + Math.imul(iz, 668265263); n = Math.imul(n ^ (n >>> 13), 1274126177); return ((n ^ (n >>> 16)) >>> 0) / 4294967296; };
  return (x, z, k) => { const px = Math.round(period * k), X = x * k, Z = z * k, xi = Math.floor(X), zi = Math.floor(Z), fx = X - xi, fz = Z - zi, sx = fx * fx * (3 - 2 * fx), sz = fz * fz * (3 - 2 * fz);
    const x0 = ((xi % px) + px) % px, x1 = (x0 + 1) % px, a = hash(x0, zi), b = hash(x1, zi), c = hash(x0, zi + 1), d = hash(x1, zi + 1);
    return a + (b - a) * sx + (c - a) * sz + (a - b - c + d) * sx * sz; };
}
function bakeAmbient(hf) {                                  // on the physics grid: sky visibility above the local tangent plane (8 directions × 3 radii) and two curvatures
  const { NX, NZ, PERIOD, ZSPAN } = hf, H = hf.height, ao = new Float32Array(NX * NZ), cs = new Float32Array(NX * NZ), cl = new Float32Array(NX * NZ);
  const dirs = []; for (let k = 0; k < 8; k++) dirs.push([Math.cos(k * Math.PI / 4), Math.sin(k * Math.PI / 4)]);
  for (let j = 0; j < NZ; j++) for (let i = 0; i < NX; i++) {
    const x = i / NX * PERIOD, z = -ZSPAN / 2 + j / (NZ - 1) * ZSPAN, h = H(x, z), gx = H(x + 0.5, z) - H(x - 0.5, z), gz = H(x, z + 0.5) - H(x, z - 0.5);
    let occ = 0;
    for (const [dx, dz] of dirs) { let t = 0; for (const r of [2.5, 5, 10]) { const e = (H(x + dx * r, z + dz * r) - h - (gx * dx + gz * dz) * r) / r; if (e > t) t = e; } occ += t / (1 + t); }
    ao[j * NX + i] = 1 - 0.8 * occ / 8;
    const m1 = (H(x + 1, z) + H(x - 1, z) + H(x, z + 1) + H(x, z - 1)) / 4, m3 = (H(x + 3, z) + H(x - 3, z) + H(x, z + 3) + H(x, z - 3)) / 4;
    cs[j * NX + i] = Math.tanh((h - m1) / 0.8); cl[j * NX + i] = Math.tanh((h - m3) / 3);   // convex on ridges, concave in gullies and cirques
  }
  const at = (arr, x, z) => {                               // bilinear, periodic in x, clamped in z (same indexing as hf.height)
    const fx = ((x % PERIOD) + PERIOD) % PERIOD / PERIOD * NX, fz = Math.min(NZ - 1, Math.max(0, (z / ZSPAN + 0.5) * (NZ - 1)));
    const i0 = Math.floor(fx) % NX, i1 = (i0 + 1) % NX, j0 = Math.min(NZ - 2, Math.floor(fz)), j1 = j0 + 1, tx = fx - Math.floor(fx), tz = fz - j0;
    const a = arr[j0 * NX + i0], b = arr[j0 * NX + i1], c = arr[j1 * NX + i0], d = arr[j1 * NX + i1];
    return (a + (b - a) * tx) + ((c + (d - c) * tx) - (a + (b - a) * tx)) * tz; };
  return (x, z) => [at(ao, x, z), at(cs, x, z), at(cl, x, z)];
}

export function createMountains(scene, hf, { sunDir = new THREE.Vector3(0.6, 0.35, -0.7), shadow = null, style = 'alps' } = {}) {
  const pillars = style === 'pillars';
  const { NX, NZ, PERIOD, ZSPAN, y0 } = hf, amp = hf.peak - y0, H = hf.height, noise = makeNoise(PERIOD), bakeAt = bakeAmbient(hf);
  const relief = (x, z, h, g, massive) => {                 // vertex-level rock relief (≤ 0.20): bedding ledges on the rock (not on the sheerest faces, where the sampling could not follow them), a soft grain everywhere; nothing on the valley floor, which lies under the imagery
    const rel = (h - y0) / amp, foot = sm(0, 1.2, h - y0), rockK = pillars ? sm(0.3, 1.2, g) : sm(0.30, 0.50, rel + 0.35 * Math.min(1, g / 1.5)), ledge = 1 - sm(1.4, 2.6, g);
    const s = (h + 0.16 * x + 0.10 * z + 3.0 * (noise(x, z, 0.2) - 0.5) + 6.0 * (noise(x + 17, z + 9, 0.075) - 0.5)) / (1.6 * (0.7 + 0.7 * noise(x + 31, z + 23, 0.05)));   // in beds: tilted, folded, of varying thickness
    const band = 0.7 * Math.sin(s * 2 * Math.PI) + 0.3 * Math.sin(s * 4 * Math.PI + 1);
    const grain = (noise(x, z, 0.35) - 0.5) * 0.8 + (noise(x, z, 0.7) - 0.5) * 1.2;
    return [s, foot * (0.04 * band * rockK * ledge * massive + 0.16 * grain * (0.3 + 0.7 * rockK))]; };
  const build = (subX, fine) => {                         // full-period grid: PERIOD / (NX·subX) in x, the physics rows in z, a skirt row on each side; the last column repeats the first (periodic)
    const NXf = Math.round(NX * subX), NZf = NZ - 1, cols = NXf + 1, rows = NZf + 3, nV = cols * rows, dx = PERIOD / NXf, dz = ZSPAN / NZf;
    const pos = new Float32Array(nV * 3), bake = new Float32Array(nV * 4), blotch = new Float32Array(nV * 3), strata = new Float32Array(nV);
    for (let r = 0; r < rows; r++) {
      const j = r - 1, inside = j >= 0 && j <= NZf, z = inside ? -ZSPAN / 2 + j * dz : (j < 0 ? -ZSPAN / 2 - SKIRT : ZSPAN / 2 + SKIRT);
      for (let i = 0; i < cols; i++) {
        const k = r * cols + i, x = -PERIOD / 2 + i * dx; let y = y0 - DROP, b = [0.6, 0, 0], s = 0, massive = 0;
        if (inside) { const h = H(x, z), g = Math.hypot(H(x + 0.25, z) - H(x - 0.25, z), H(x, z + 0.25) - H(x, z - 0.25)) * 2; massive = pillars ? 1 : sm(0.50, 0.80, noise(x + 53.3, z + 4, 0.075)) * sm(0.35, 0.7, noise(x + 7, z + 41, 0.25)); b = bakeAt(x, z); const rl = relief(x, z, h, g, massive); s = rl[0]; y = h + (fine ? rl[1] : 0); }
        pos[k * 3] = x; pos[k * 3 + 1] = y; pos[k * 3 + 2] = z; bake[k * 4] = b[0]; bake[k * 4 + 1] = b[1]; bake[k * 4 + 2] = b[2]; bake[k * 4 + 3] = massive; strata[k] = s;
        blotch[k * 3] = noise(x + 5.7, z + 2, 0.35); blotch[k * 3 + 1] = noise(x + 1.1, z + 8, 0.9); blotch[k * 3 + 2] = noise(x + 9.3, z + 3, 0.075);   // slow patchiness: tree line, drifts, rock tones
      }
    }
    const idx = new Uint32Array((cols - 1) * (rows - 1) * 6); let n = 0;
    for (let r = 0; r < rows - 1; r++) for (let i = 0; i < cols - 1; i++) {                  // split each quad along the diagonal that best matches the field at the quad's centre (clean crests)
      const a = r * cols + i, b = a + 1, c = a + cols, d = c + 1, ya = pos[a * 3 + 1], yb = pos[b * 3 + 1], yc = pos[c * 3 + 1], yd = pos[d * 3 + 1];
      const centre = r >= 1 && r < rows - 2 ? H(pos[a * 3] + dx / 2, pos[a * 3 + 2] + dz / 2) : (ya + yd) / 2;
      if (Math.abs((ya + yd) / 2 - centre) <= Math.abs((yb + yc) / 2 - centre)) { idx[n++] = a; idx[n++] = c; idx[n++] = d; idx[n++] = a; idx[n++] = d; idx[n++] = b; }
      else { idx[n++] = a; idx[n++] = c; idx[n++] = b; idx[n++] = b; idx[n++] = c; idx[n++] = d; }
    }
    const geo = new THREE.BufferGeometry();
    geo.setIndex(new THREE.BufferAttribute(idx, 1)); geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('aBake', new THREE.BufferAttribute(bake, 4)); geo.setAttribute('aBlotch', new THREE.BufferAttribute(blotch, 3)); geo.setAttribute('aS', new THREE.BufferAttribute(strata, 1));
    geo.computeVertexNormals();
    { const nr = geo.attributes.normal.array;                // the seam columns see only one side's faces: give both the shared average so the copies join without a crease
      for (let r = 0; r < rows; r++) { const a = r * cols * 3, b = (r * cols + cols - 1) * 3, x = nr[a] + nr[b], y = nr[a + 1] + nr[b + 1], z = nr[a + 2] + nr[b + 2], l = Math.hypot(x, y, z) || 1;
        nr[a] = nr[b] = x / l; nr[a + 1] = nr[b + 1] = y / l; nr[a + 2] = nr[b + 2] = z / l; } }
    return { geo, cols, rows }; };
  const half = ({ geo, cols, rows }, c0, c1) => {          // columns c0..c1 of a grid geometry as a geometry of its own (normals and diagonals as computed on the whole)
    const nc = c1 - c0 + 1, out = new THREE.BufferGeometry();
    for (const [name, size] of [['position', 3], ['normal', 3], ['aBake', 4], ['aBlotch', 3], ['aS', 1]]) { const src = geo.attributes[name].array, dst = new Float32Array(nc * rows * size);
      for (let r = 0; r < rows; r++) dst.set(src.subarray((r * cols + c0) * size, (r * cols + c1 + 1) * size), r * nc * size); out.setAttribute(name, new THREE.BufferAttribute(dst, size)); }
    const src = geo.index.array, idx = new Uint32Array((nc - 1) * (rows - 1) * 6); let n = 0;
    for (let r = 0; r < rows - 1; r++) for (let i = c0; i < c1; i++) { const q = (r * (cols - 1) + i) * 6; for (let k = 0; k < 6; k++) { const v = src[q + k]; idx[n++] = Math.floor(v / cols) * nc + (v % cols) - c0; } }
    out.setIndex(new THREE.BufferAttribute(idx, 1)); return out; };
  // Level of detail in half-period (60-unit) slots: three fine slots (0.25 units in x, with the relief) always reach ≥ 60 units behind
  // and ahead of the camera; two slots behind and three ahead use the physics grid itself (every summit exact, no relief). A half-
  // period slot shows the field's first or second half depending on where it sits, so one BatchedMesh holds four geometries.
  const fineFull = build(2, true), farFull = build(1, false);
  const parts = [half(fineFull, 0, (fineFull.cols - 1) / 2), half(fineFull, (fineFull.cols - 1) / 2, fineFull.cols - 1), half(farFull, 0, (farFull.cols - 1) / 2), half(farFull, (farFull.cols - 1) / 2, farFull.cols - 1)];
  fineFull.geo.dispose(); farFull.geo.dispose();
  const sh = shadow ? shadow.uniforms : { uHeight: { value: new THREE.DataTexture(new Float32Array([0]), 1, 1, THREE.RedFormat, THREE.FloatType) }, uSunDir: { value: sunDir.clone() }, uY0: { value: y0 }, uShadowsOn: { value: 0 } };
  const mat = new THREE.ShaderMaterial({ vertexShader: VERT, fragmentShader: FRAG, transparent: true, depthWrite: true,
    uniforms: { uSun: { value: sunDir.clone().normalize() }, uHaze: { value: new THREE.Color(0.62, 0.74, 0.9) }, uAmp: { value: amp }, uVis: { value: 0 }, uSnow: { value: 0.60 }, uTime: { value: 0 }, uApexX: { value: 0 }, uStyle: { value: pillars ? 1 : 0 }, ...sh } });
  const SLOTS = [0, 1, 2, -1, -2, 3, 4, 5], U = mat.uniforms;   // slot offsets in half periods from the fine window's start: fine, fine, fine, far behind ×2, far ahead ×3
  const mesh = new THREE.BatchedMesh(SLOTS.length, parts.reduce((a, g) => a + g.attributes.position.count, 0), parts.reduce((a, g) => a + g.index.count, 0), mat);   // one object, one (multi-)draw call
  const ids = parts.map((g) => mesh.addGeometry(g)), inst = SLOTS.map(() => mesh.addInstance(ids[0])); for (const g of parts) g.dispose();
  mesh.frustumCulled = false; mesh.perObjectFrustumCulled = false; mesh.sortObjects = false; mesh.renderOrder = -4; mesh.visible = false; scene.add(mesh);
  const m = new THREE.Matrix4(), HALF = PERIOD / 2;
  return {
    mesh,
    update(dt, camera, vis, sunDirWorld, apexX) {          // vis 0..1; the slots follow the camera so the wrap is invisible; apexX = the ground's curvature apex
      mesh.visible = vis > 0.01; U.uVis.value = vis; U.uApexX.value = apexX ?? camera.position.x;
      if (!mesh.visible) return;
      if (sunDirWorld) U.uSun.value.copy(sunDirWorld).normalize(); U.uTime.value += dt;
      const W = HALF * Math.floor(camera.position.x / HALF) - HALF;                           // the fine window [W, W + 3·HALF): the camera is 60…120 units into it
      for (let k = 0; k < SLOTS.length; k++) {
        const p = W + SLOTS[k] * HALF, first = ((p % PERIOD) + PERIOD) % PERIOD === HALF;      // a slot at p shows the field's first half (local x < 0, placed at p + HALF) or its second (placed at p)
        mesh.setGeometryIdAt(inst[k], ids[(k < 3 ? 0 : 2) + (first ? 0 : 1)]); m.makeTranslation(first ? p + HALF : p, 0, 0); mesh.setMatrixAt(inst[k], m);
      }
    },
    setMono(mode) { U.uSnow.value = mode === 2 ? 0.54 : 0.60; },   // ink: a lower snow line keeps the two-tone range legible
    dispose() { scene.remove(mesh); mesh.dispose(); mat.dispose(); },
  };
}
