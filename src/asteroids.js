// asteroids.js — procedural asteroid shapes + InstancedMesh field for Astro Pilot (three.js, no assets).
//
// Model (per shape, fully seeded):
//   icosphere (2562 verts / 5120 tris, or 10242 / 20480 for the first `hiDetail` shapes = three.js detail 15 / 31;
//   shared vertices, uv-seam duplicates merged logically) with radial radius
//     ρ(n̂) = base(n̂) · (1 + A₁·fbm(f₁n̂+o₁) + A₂·fbm(5.7n̂+o₂) + A₃·fbm(f₃n̂+o₃) + ridges + boulders − craters)
//   base(n̂):  1 (irregular, Eros/Ida/Gaspra)  |  soft union of two offset spheres (contact binary, Itokawa/Arrokoth)
//             |  sphere↔bicone blend with an equatorial ridge (spinning top, Bennu/Ryugu)
//   craters:  ρ ← ρ − δ·smoothstep(ρc, 0.5ρc, ‖n̂−c‖) + 0.45δ·rim(‖n̂−c‖); 3–8 large (ρc ~ U(0.15,0.45)) plus a
//             population of small ones (ρc ≥ 1.5 × mesh edge) so the surface is crater-saturated like real rocks
//   boulders: flat-topped bumps h·(1−(d/a)²)^0.7, dense on rubble-pile tops; fbm = Perlin gradient noise
//   then anisotropic ellipsoid scaling (axis ratios U(0.62,1); elongated variants down to 0.5) and normalisation
//   so the maximum vertex radius is exactly 1 (an asteroid of collision radius r is drawn at scale r·1.1).
//   Vertex colour = grey/umber albedo × low-frequency noise (±15 %) × cavity term (ρ minus its Laplacian-smoothed
//   version at two scales: concave dark, convex light — cheap AO) × crater-floor darkening × rim/boulder lightening,
//   plus sparse bright "fresh" specks. A contrast-raised grey copy is swapped in for ink mode (setMono(2)).
//   A small noise texture, sampled triplanar in object space (onBeforeCompile), adds sub-vertex bump + albedo grain.
//   Rendering: one InstancedMesh per shape (DynamicDrawUsage), instance rotation = q, uniform scale = r·1.1.
import * as THREE from 'three';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';

const GREY = [0.42, 0.40, 0.38], UMBER = [0.36, 0.30, 0.25], BLUEGREY = [0.40, 0.41, 0.42];
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const sstep = (e0, e1, x) => { const t = clamp((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); };

// ---------------------------------------------------------------- seeded RNG (mulberry32) + Perlin noise / fbm
function mulberry32(a) {
  return () => {
    a = (a + 0x6D2A79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function randUnit(R) { const z = 2 * R() - 1, p = 2 * Math.PI * R(), s = Math.sqrt(1 - z * z); return [s * Math.cos(p), s * Math.sin(p), z]; }

function makeNoise(R) {
  const p = new Uint8Array(512), src = new Uint8Array(256);
  for (let i = 0; i < 256; i++) src[i] = i;
  for (let i = 255; i > 0; i--) { const j = (R() * (i + 1)) | 0; const t = src[i]; src[i] = src[j]; src[j] = t; }
  for (let i = 0; i < 512; i++) p[i] = src[i & 255];
  const grad = (h, x, y, z) => { const g = h & 15, u = g < 8 ? x : y, v = g < 4 ? y : (g === 12 || g === 14 ? x : z); return ((g & 1) ? -u : u) + ((g & 2) ? -v : v); };
  const lerp = (t, a, b) => a + t * (b - a);
  function noise(x, y, z) {
    const X = Math.floor(x), Y = Math.floor(y), Z = Math.floor(z);
    x -= X; y -= Y; z -= Z;
    const xi = X & 255, yi = Y & 255, zi = Z & 255;
    const u = x * x * x * (x * (x * 6 - 15) + 10), v = y * y * y * (y * (y * 6 - 15) + 10), w = z * z * z * (z * (z * 6 - 15) + 10);
    const A = p[xi] + yi, AA = p[A] + zi, AB = p[A + 1] + zi, B = p[xi + 1] + yi, BA = p[B] + zi, BB = p[B + 1] + zi;
    return lerp(w,
      lerp(v, lerp(u, grad(p[AA], x, y, z), grad(p[BA], x - 1, y, z)), lerp(u, grad(p[AB], x, y - 1, z), grad(p[BB], x - 1, y - 1, z))),
      lerp(v, lerp(u, grad(p[AA + 1], x, y, z - 1), grad(p[BA + 1], x - 1, y, z - 1)), lerp(u, grad(p[AB + 1], x, y - 1, z - 1), grad(p[BB + 1], x - 1, y - 1, z - 1))));
  }
  function fbm(x, y, z, oct) {
    let s = 0, a = 1, n = 0;
    for (let i = 0; i < oct; i++) { s += a * noise(x, y, z); n += a; a *= 0.5; x = x * 2 + 17.3; y = y * 2 + 9.1; z = z * 2 + 31.7; }
    return s / n;
  }
  return { noise, fbm };
}

// ---------------------------------------------------------------- icosphere template: indexed mesh + canonical ids + adjacency
const templates = new Map();
function getTemplate(detail) {
  if (templates.has(detail)) return templates.get(detail);
  const merged = mergeVertices(new THREE.IcosahedronGeometry(1, detail));
  const geo = new THREE.BufferGeometry().copy(merged);
  merged.dispose();
  const pos = geo.attributes.position.array, nv = pos.length / 3, idx = geo.index.array;
  // seam / pole duplicates (different uv) share a canonical id so displacement, normals and colours stay continuous
  const canonOf = new Int32Array(nv), map = new Map(); let nc = 0;
  for (let i = 0; i < nv; i++) {
    const k = `${Math.round(pos[3 * i] * 1e5)},${Math.round(pos[3 * i + 1] * 1e5)},${Math.round(pos[3 * i + 2] * 1e5)}`;
    let c = map.get(k); if (c === undefined) { c = nc++; map.set(k, c); }
    canonOf[i] = c;
  }
  const dir = new Float32Array(nc * 3);
  for (let i = 0; i < nv; i++) { const c = canonOf[i]; dir[3 * c] = pos[3 * i]; dir[3 * c + 1] = pos[3 * i + 1]; dir[3 * c + 2] = pos[3 * i + 2]; }
  const sets = Array.from({ length: nc }, () => new Set());
  for (let t = 0; t < idx.length; t += 3) {
    const a = canonOf[idx[t]], b = canonOf[idx[t + 1]], c = canonOf[idx[t + 2]];
    sets[a].add(b); sets[a].add(c); sets[b].add(a); sets[b].add(c); sets[c].add(a); sets[c].add(b);
  }
  const adjOff = new Int32Array(nc + 1);
  for (let i = 0; i < nc; i++) adjOff[i + 1] = adjOff[i] + sets[i].size;
  const adj = new Int32Array(adjOff[nc]);
  for (let i = 0, k = 0; i < nc; i++) for (const j of sets[i]) adj[k++] = j;
  const i0 = idx[0] * 3, i1 = idx[1] * 3;
  const edge = Math.hypot(pos[i0] - pos[i1], pos[i0 + 1] - pos[i1 + 1], pos[i0 + 2] - pos[i1 + 2]);
  const tpl = { geo, canonOf, nc, dir, adjOff, adj, edge };
  templates.set(detail, tpl);
  return tpl;
}

// ---------------------------------------------------------------- one asteroid shape (kind 0 irregular, 1 contact binary, 2 spinning top)
function makeShape(tpl, noise, R, si, kind) {
  const { nc, dir, canonOf, adjOff, adj, edge } = tpl;
  const geo = tpl.geo.clone();
  const fbm = noise.fbm;
  const o = new Float32Array(18); for (let i = 0; i < 18; i++) o[i] = R() * 200 - 100;
  const f1 = 1.9 + 0.4 * R(), f3 = 0.35 / edge, fr2 = edge < 0.05 ? 8.4 : 0;   // finest octaves tied to the mesh edge length
  const A1 = kind === 2 ? 0.08 + 0.04 * R() : kind === 1 ? 0.16 + 0.06 * R() : 0.22 + 0.10 * R();
  const A2 = kind === 2 ? 0.025 : 0.03, A3 = 0.008, AR = kind === 2 ? 0.015 : kind === 1 ? 0.03 : 0.04 + 0.03 * R();
  // anisotropic scaling
  let sx = 1, sy = 1, sz = 1;
  if (kind === 0) {
    const a = [1, 0.62 + 0.38 * R(), 0.62 + 0.38 * R()];
    if (si % 4 === 2) { a[1] = 0.5 + 0.15 * R(); a[2] = 0.6 + 0.2 * R(); }   // elongated (Eros-like)
    const pm = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]][(R() * 6) | 0];
    sx = a[pm[0]]; sy = a[pm[1]]; sz = a[pm[2]];
  } else if (kind === 1) { sy = 0.68 + 0.2 * R(); sz = 0.68 + 0.2 * R(); }
  else { sx = 0.94 + 0.06 * R(); sz = 0.94 + 0.06 * R(); sy = 0.86 + 0.08 * R(); }
  // contact-binary lobes (unit sphere A at +dA·â, sphere B (radius RB) at −dB·â, origin inside both) and top profile
  let ax = 1, ay = 0.3 * (R() - 0.5), az = 0.3 * (R() - 0.5); { const l = Math.hypot(ax, ay, az); ax /= l; ay /= l; az /= l; }
  const dA = 0.25 + 0.2 * R(), RB = 0.55 + 0.25 * R(), dB = RB * (0.75 + 0.2 * R()), kSoft = 0.08 + 0.1 * R();
  const topMix = 0.4 + 0.25 * R(), topSlope = 1.05 + 0.2 * R(), ridge = 0.015 + 0.02 * R();
  // craters: x,y,z, rc, depth, rim width (large population + small population)
  const nBig = kind === 2 ? 2 + ((R() * 3) | 0) : kind === 1 ? 3 + ((R() * 4) | 0) : 4 + ((R() * 5) | 0);
  const nSmall = kind === 2 ? 8 : kind === 1 ? 14 : 16, nCr = nBig + nSmall, cr = new Float32Array(nCr * 7);
  for (let i = 0; i < nCr; i++) {
    const c = randUnit(R), big = i === 0 && kind === 0 && R() < 0.5;
    const rc = i < nBig ? (big ? 0.45 + 0.2 * R() : 0.15 + 0.3 * R()) : 1.5 * edge + 0.12 * R() * R();
    cr.set([c[0], c[1], c[2], rc, rc * (0.16 + 0.14 * R()), Math.max(0.16 * rc, 1.2 * edge), i < nBig ? 0.45 : 0.25], i * 7);
  }
  // boulders: x,y,z, a², h, brightness
  const nB = kind === 2 ? 60 : kind === 1 ? 12 : 8, bo = new Float32Array(nB * 6);
  for (let i = 0; i < nB; i++) {
    const c = randUnit(R), a = 1.4 * edge + 0.05 * R() * R();
    bo.set([c[0], c[1], c[2], a * a, a * (kind === 2 ? 0.35 + 0.35 * R() : 0.25 + 0.3 * R()), 0.5 * (R() - 0.35)], i * 6);
  }
  // ---- radius field per canonical vertex
  const rho = new Float32Array(nc), fl = new Float32Array(nc), rm = new Float32Array(nc), bb = new Float32Array(nc);
  for (let c = 0; c < nc; c++) {
    const nx = dir[3 * c], ny = dir[3 * c + 1], nz = dir[3 * c + 2];
    let base = 1;
    if (kind === 1) {
      const b = nx * ax + ny * ay + nz * az;
      const tA = b * dA + Math.sqrt(b * b * dA * dA - dA * dA + 1);
      const cB = -b * dB, tB = cB + Math.sqrt(cB * cB - dB * dB + RB * RB);
      base = kSoft * Math.log(Math.exp(tA / kSoft) + Math.exp(tB / kSoft));
    } else if (kind === 2) {
      const s = Math.sqrt(ny * ny + 0.0016), cl = Math.sqrt(Math.max(0, 1 - ny * ny));
      base = 1 + topMix * (1 / (cl + topSlope * s) - 1) + ridge * Math.exp(-(ny * ny) / 0.012);
    }
    // terrain mask: smooth regolith "seas" vs rough, blocky terrain (Itokawa-like); ridged noise gives sharp crests
    const mask = sstep(-0.15, 0.35, fbm(1.3 * nx + o[12], 1.3 * ny + o[13], 1.3 * nz + o[14], 2));
    const rg = mask * (0.5 - Math.abs(noise.noise(4.2 * nx + o[15], 4.2 * ny + o[16], 4.2 * nz + o[17]))
             + (fr2 ? 0.5 * (0.5 - Math.abs(noise.noise(fr2 * nx + o[16], fr2 * ny + o[17], fr2 * nz + o[15]))) : 0));
    let r = 1 + A1 * fbm(f1 * nx + o[0], f1 * ny + o[1], f1 * nz + o[2], 3)
              + A2 * fbm(5.7 * nx + o[3], 5.7 * ny + o[4], 5.7 * nz + o[5], 2)
              + A3 * (0.4 + 0.6 * mask) * fbm(f3 * nx + o[6], f3 * ny + o[7], f3 * nz + o[8], 2)
              + AR * rg;
    for (let j = 0; j < nB; j++) {
      const k = j * 6, dx = nx - bo[k], dy = ny - bo[k + 1], dz = nz - bo[k + 2], d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < bo[k + 3]) { const t = Math.pow(1 - d2 / bo[k + 3], 0.7); r += bo[k + 4] * t; bb[c] += t * bo[k + 5]; }
    }
    let f = 0, m = 0;
    for (let j = 0; j < nCr; j++) {
      const k = j * 7, dx = nx - cr[k], dy = ny - cr[k + 1], dz = nz - cr[k + 2], d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const rc = cr[k + 3], w = cr[k + 5];
      if (d > rc + 2.5 * w) continue;
      const floor = sstep(rc, 0.6 * rc, d), e = (d - rc) / w, rim = Math.exp(-e * e), dep = cr[k + 4];
      r += -dep * floor + cr[k + 6] * dep * rim; f += floor; m += rim;
    }
    fl[c] = f; rm[c] = m; rho[c] = base * r;
  }
  // ---- positions (ellipsoid scaling, normalise max radius to 1)
  const pos = geo.attributes.position.array, nv = pos.length / 3; let maxR = 0;
  for (let i = 0; i < nv; i++) {
    const c = canonOf[i], r = rho[c], x = dir[3 * c] * r * sx, y = dir[3 * c + 1] * r * sy, z = dir[3 * c + 2] * r * sz;
    pos[3 * i] = x; pos[3 * i + 1] = y; pos[3 * i + 2] = z;
    const rr = x * x + y * y + z * z; if (rr > maxR) maxR = rr;
  }
  const inv = 1 / Math.sqrt(maxR);
  for (let i = 0; i < pos.length; i++) pos[i] *= inv;
  // ---- cavity (ρ − smoothed ρ) at a narrow and a wide scale → cheap ambient occlusion
  const smooth = (src, iters) => {
    let a = Float32Array.from(src), b = new Float32Array(nc);
    for (let it = 0; it < iters; it++) {
      for (let c = 0; c < nc; c++) { let s = 0; const k0 = adjOff[c], k1 = adjOff[c + 1]; for (let k = k0; k < k1; k++) s += a[adj[k]]; b[c] = s / (k1 - k0); }
      const t = a; a = b; b = t;
    }
    return a;
  };
  const sN = smooth(rho, 2), sW = smooth(rho, 14), cN = new Float32Array(nc), cW = new Float32Array(nc);
  let vN = 0, vW = 0;
  for (let c = 0; c < nc; c++) { cN[c] = (rho[c] - sN[c]) / rho[c]; cW[c] = (rho[c] - sW[c]) / rho[c]; vN += cN[c] * cN[c]; vW += cW[c] * cW[c]; }
  const sigN = Math.sqrt(vN / nc) + 1e-6, sigW = Math.sqrt(vW / nc) + 1e-6;
  // ---- smooth normals accumulated per canonical id (continuous across the uv seam)
  const idx = geo.index.array, nrm = new Float32Array(nc * 3);
  for (let t = 0; t < idx.length; t += 3) {
    const j0 = idx[t], j1 = idx[t + 1], j2 = idx[t + 2], i0 = j0 * 3, i1 = j1 * 3, i2 = j2 * 3;
    const ex = pos[i1] - pos[i0], ey = pos[i1 + 1] - pos[i0 + 1], ez = pos[i1 + 2] - pos[i0 + 2];
    const gx = pos[i2] - pos[i0], gy = pos[i2 + 1] - pos[i0 + 1], gz = pos[i2 + 2] - pos[i0 + 2];
    const nx = ey * gz - ez * gy, ny = ez * gx - ex * gz, nz = ex * gy - ey * gx;
    let c = canonOf[j0] * 3; nrm[c] += nx; nrm[c + 1] += ny; nrm[c + 2] += nz;
    c = canonOf[j1] * 3; nrm[c] += nx; nrm[c + 1] += ny; nrm[c + 2] += nz;
    c = canonOf[j2] * 3; nrm[c] += nx; nrm[c + 1] += ny; nrm[c + 2] += nz;
  }
  const nor = geo.attributes.normal.array;
  for (let i = 0; i < nv; i++) {
    const c = canonOf[i] * 3, l = Math.hypot(nrm[c], nrm[c + 1], nrm[c + 2]) || 1;
    nor[3 * i] = nrm[c] / l; nor[3 * i + 1] = nrm[c + 1] / l; nor[3 * i + 2] = nrm[c + 2] / l;
  }
  // ---- vertex colours (albedo × noise × cavity × craters × specks) and an ink (high-contrast) variant
  const tMix = R(), tBlue = R() < 0.35 ? 0.5 + 0.5 * R() : 0, bright = kind === 2 ? 0.8 : kind === 1 ? 1.05 : 1.0;
  const cb = [0, 1, 2].map((i) => { const g = GREY[i] + (UMBER[i] - GREY[i]) * tMix; return (g + (BLUEGREY[i] - g) * tBlue) * bright; });
  const speck = new Uint8Array(nc);
  for (let i = 0, n = (nc * 0.012) | 0; i < n; i++) speck[(R() * nc) | 0] = 1;
  const ccol = new Float32Array(nc * 3), cink = new Float32Array(nc);
  for (let c = 0; c < nc; c++) {
    const nx = dir[3 * c], ny = dir[3 * c + 1], nz = dir[3 * c + 2];
    const lf = fbm(1.6 * nx + o[9], 1.6 * ny + o[10], 1.6 * nz + o[11], 2), hf = noise.noise(9 * nx + o[12], 9 * ny + o[13], 9 * nz + o[14]);
    const alb = 1 + 0.3 * lf + 0.08 * hf;
    let sh = (1 + 0.32 * Math.tanh(cN[c] / sigN) + 0.14 * Math.tanh(cW[c] / sigW))
           * (1 - 0.3 * Math.min(1, fl[c])) * (1 + 0.10 * Math.min(1, rm[c])) * (1 + clamp(bb[c], -0.25, 0.3));
    sh = clamp(sh, 0.45, 1.35);
    let r = cb[0] * alb * sh, g = cb[1] * alb * sh, b = cb[2] * alb * sh;
    if (fl[c] > 0.5) { r *= 1.04; b *= 0.94; }            // warm dusty crater floors
    if (speck[c]) { r *= 1.3; g *= 1.33; b *= 1.4; }      // fresh, slightly bluish exposures
    ccol[3 * c] = clamp(r, 0, 0.62); ccol[3 * c + 1] = clamp(g, 0, 0.62); ccol[3 * c + 2] = clamp(b, 0, 0.62);
    cink[c] = clamp(0.42 * Math.pow(sh * alb, 2.4) * (speck[c] ? 1.5 : 1), 0.03, 0.62);
  }
  const col = new Float32Array(nv * 3), ink = new Float32Array(nv * 3);
  for (let i = 0; i < nv; i++) {
    const c = canonOf[i];
    col[3 * i] = ccol[3 * c]; col[3 * i + 1] = ccol[3 * c + 1]; col[3 * i + 2] = ccol[3 * c + 2];
    ink[3 * i] = ink[3 * i + 1] = ink[3 * i + 2] = cink[c];
  }
  geo.userData.color = new THREE.BufferAttribute(col, 3);
  geo.userData.ink = new THREE.BufferAttribute(ink, 3);
  geo.setAttribute('color', geo.userData.color);
  geo.computeBoundingSphere();
  return geo;
}

// ---------------------------------------------------------------- regolith grain: noise texture + triplanar bump/albedo shader patch
function makeGrainTexture(noise, size) {
  const data = new Uint8Array(size * size);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const u = x / size * 4, v = y / size * 4;
    const h = 0.5 * noise.fbm(u, v, 0.37, 3) + 0.3 * noise.noise(u * 3.1, v * 3.1, 5.2) + 0.2 * noise.noise(u * 7.3, v * 7.3, 9.1);
    data[y * size + x] = clamp(128 + 200 * h, 0, 255);
  }
  const tex = new THREE.DataTexture(data, size, size, THREE.RedFormat, THREE.UnsignedByteType);
  tex.wrapS = tex.wrapT = THREE.MirroredRepeatWrapping;               // seamless without a tileable texture
  tex.minFilter = THREE.LinearMipmapLinearFilter; tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = true; tex.needsUpdate = true;
  return tex;
}
// Samples bumpMap triplanar in object space (no uv seams / pole stretch) and modulates the albedo with the same grain.
function patchTriplanar(shader) {
  shader.vertexShader = shader.vertexShader
    .replace('#include <common>', '#include <common>\nvarying vec3 vObjPos;\nvarying vec3 vObjNrm;')
    .replace('#include <begin_vertex>', '#include <begin_vertex>\nvObjPos = position;\nvObjNrm = normal;');
  const tri = THREE.ShaderChunk.bumpmap_pars_fragment.replace(/vec2 dHdxy_fwd\(\) \{[\s\S]*?\n\t\}/, `
    varying vec3 vObjPos; varying vec3 vObjNrm;
    float triH( vec3 p ) {
      vec3 w = abs( vObjNrm ); w = w * w; w /= ( w.x + w.y + w.z );
      return texture2D( bumpMap, p.yz ).x * w.x + texture2D( bumpMap, p.xz ).x * w.y + texture2D( bumpMap, p.xy ).x * w.z;
    }
    float grainH() { float a = 0.45 + 1.1 * triH( vObjPos * 0.9 ); return a * ( triH( vObjPos * 6.0 ) + 0.5 * triH( vObjPos * 15.0 ) ); }
    vec2 dHdxy_fwd() { float h = bumpScale * grainH(); return vec2( dFdx( h ), dFdy( h ) ); }`);
  shader.fragmentShader = shader.fragmentShader
    .replace('#include <bumpmap_pars_fragment>', tri)
    .replace('#include <color_fragment>', '#include <color_fragment>\n#ifdef USE_BUMPMAP\n\tdiffuseColor.rgb *= 0.86 + 0.19 * grainH();\n#endif')
    // airless bodies: a hard terminator — the camera-side fill keeps the night side readable, not lit (the sun is the first directional light)
    .replace('#include <opaque_fragment>', '#if NUM_DIR_LIGHTS > 0\n\toutgoingLight *= mix( 0.55, 1.0, smoothstep( -0.06, 0.28, dot( normal, directionalLights[ 0 ].direction ) ) );\n#endif\n#include <opaque_fragment>');
}
// Albedo classes of real asteroids, as tints on the vertex colours: carbonaceous (dark), stony (warm), metallic (cool, bright),
// primitive D-type (reddish) and plain grey; each rock keeps its class for as long as it lives.
const TINTS = [[0.66, 0.66, 0.68], [1.05, 0.93, 0.8], [1.0, 1.02, 1.08], [0.98, 0.8, 0.7], [0.86, 0.86, 0.86], [0.76, 0.73, 0.7]].map((c) => new THREE.Color(...c));
const WHITE = new THREE.Color(1, 1, 1);

// ---------------------------------------------------------------- public API
export function createAsteroidField(scene, { nShapes = 12, maxInstances = 80, seed = 7, hiDetail = 4, bump = true } = {}) {
  const R = mulberry32(seed | 0), noise = makeNoise(R);
  const material = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.92, metalness: 0.02 });
  let bumpTex = null;
  if (bump) { bumpTex = makeGrainTexture(noise, 128); material.bumpMap = bumpTex; material.bumpScale = 2.0; material.onBeforeCompile = patchTriplanar; }
  const geos = [], meshes = [], arrays = [];
  for (let s = 0; s < nShapes; s++) {
    const kind = s % 4 === 1 ? 1 : s % 4 === 3 ? 2 : 0;
    const g = makeShape(getTemplate(s < hiDetail ? 31 : 15), noise, R, s, kind);   // 16/32 segments per icosahedron edge = 5120 / 20480 triangles
    const m = new THREE.InstancedMesh(g, material, maxInstances);
    m.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    m.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(maxInstances * 3).fill(1), 3); m.instanceColor.setUsage(THREE.DynamicDrawUsage);
    m.count = 0; m.frustumCulled = false; m.matrixAutoUpdate = false; m.name = `asteroid-shape-${s}`;
    scene.add(m); geos.push(g); meshes.push(m); arrays.push(m.instanceMatrix.array);
  }
  const counts = new Int32Array(nShapes);
  let inkOn = false;

  function update(asteroids) {
    counts.fill(0);
    const n = asteroids ? asteroids.length : 0;
    for (let i = 0; i < n; i++) {
      const a = asteroids[i], p = a.p, q = a.q;
      let s = a.shape | 0; if (s < 0 || s >= nShapes) s = ((s % nShapes) + nShapes) % nShapes;
      const k = counts[s]; if (k >= maxInstances) continue;
      let x = q[0], y = q[1], z = q[2], w = q[3];
      const l2 = x * x + y * y + z * z + w * w;
      if (!(l2 > 1e-12)) { x = y = z = 0; w = 1; } else if (Math.abs(l2 - 1) > 1e-6) { const il = 1 / Math.sqrt(l2); x *= il; y *= il; z *= il; w *= il; }
      const sc = (a.r > 0 ? a.r : 1) * 1.1, m = arrays[s], o = k * 16;
      const x2 = x + x, y2 = y + y, z2 = z + z, xx = x * x2, xy = x * y2, xz = x * z2, yy = y * y2, yz = y * z2, zz = z * z2, wx = w * x2, wy = w * y2, wz = w * z2;
      m[o] = (1 - (yy + zz)) * sc; m[o + 1] = (xy + wz) * sc; m[o + 2] = (xz - wy) * sc; m[o + 3] = 0;
      m[o + 4] = (xy - wz) * sc; m[o + 5] = (1 - (xx + zz)) * sc; m[o + 6] = (yz + wx) * sc; m[o + 7] = 0;
      m[o + 8] = (xz + wy) * sc; m[o + 9] = (yz - wx) * sc; m[o + 10] = (1 - (xx + yy)) * sc; m[o + 11] = 0;
      m[o + 12] = p[0]; m[o + 13] = p[1]; m[o + 14] = p[2]; m[o + 15] = 1;
      const h = (Math.imul(i + 1, 2654435761) ^ Math.imul(s + 7, 40503) ^ Math.round(a.r * 97)) >>> 0;   // a stable class per rock
      meshes[s].setColorAt(k, inkOn ? WHITE : TINTS[h % TINTS.length]);
      counts[s] = k + 1;
    }
    for (let s = 0; s < nShapes; s++) {
      const c = counts[s], mesh = meshes[s];
      mesh.count = c;
      if (c > 0) { const attr = mesh.instanceMatrix; attr.clearUpdateRanges(); attr.addUpdateRange(0, c * 16); attr.needsUpdate = true; mesh.instanceColor.needsUpdate = true; }
    }
  }

  function setMono(mode) {
    const ink = mode === 2;
    if (ink === inkOn) return;
    inkOn = ink;
    for (const g of geos) g.setAttribute('color', ink ? g.userData.ink : g.userData.color);
  }

  function dispose() {
    for (const m of meshes) { scene.remove(m); m.dispose(); }
    for (const g of geos) g.dispose();
    material.dispose();
    if (bumpTex) bumpTex.dispose();
  }

  return { update, setMono, nShapes, dispose };
}
