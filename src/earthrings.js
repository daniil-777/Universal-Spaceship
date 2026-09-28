// earthrings.js — the Earth zoom view's ground: five nested rings of real imagery with real relief around the view's
// target, streamed the way src/terrain.js streams its strips (one curved mesh + one GPU atlas per ring, tiles copied in
// as they arrive, an atlas addressed toroidally so re-centring fetches only the new row or column). Rings sit on levels
// L0, L0−1, L0−2, L0−3, L0−5 (src/earthtiles.js); a ring shows once 90 % of its tiles are in and a per-slot mask hides
// any tile not yet (or never) loaded, so the coarser ring or the globe covers it — no holes. Heights: Terrarium tiles
// one level coarser in a 4 × 4 atlas, decoded in the vertex shader. Units: km in the view's local frame.
import * as THREE from 'three';
import { RING_TILES, RING_COUNT, HEIGHT_TILES, MAX_LEVEL, ESRI_BLANK, HEIGHT_SOURCE, sourceForLevel, tileUrl, ringLevels, ringWindow, heightWindow, windowTiles, tileToLonLat, lonLatToTile, tileSizeKm, decodeTerrarium, mod } from './earthtiles.js';

const T = 256, GRID = 128, SHOW_AT = 0.9, HIDE_AT = 0.5, UPLOADS_PER_FRAME = 6, RETRY_MS = 30000;

const VERT = /* glsl */`uniform sampler2D tHeight; uniform vec2 uHOff; uniform float uHScale, uHSize, uHeightK, uStepKm;
  attribute vec3 aUp; varying vec2 vUv; varying vec3 vW, vN, vUp;
  float hAt(vec2 t) { vec3 c = floor(texture2D(tHeight, t).rgb * 255.0 + 0.5); return max(c.r * 256.0 + c.g + c.b / 256.0 - 32768.0, 0.0); }
  float hBil(vec2 uv) {                                   // Terrarium RGB must not be filtered by the hardware: 4 nearest taps
    vec2 p = (uHOff + uv * uHScale) * uHSize - 0.5, f = fract(p), b = (floor(p) + 0.5) / uHSize; float d = 1.0 / uHSize;
    return mix(mix(hAt(b), hAt(b + vec2(d, 0.0)), f.x), mix(hAt(b + vec2(0.0, d)), hAt(b + vec2(d)), f.x), f.y); }
  void main() {
    vUv = uv; float k = 0.001 * uHeightK, du = ${(1 / GRID).toFixed(7)};
    float h = hBil(uv) * k, hx = hBil(uv + vec2(du, 0.0)) * k, hy = hBil(uv + vec2(0.0, du)) * k;
    vUp = normalize(aUp); vN = normalize(vUp + vec3(-(hx - h), 0.0, -(hy - h)) / uStepKm);   // uv.y runs south = +z
    vec4 w = modelMatrix * vec4(position + vUp * h, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`;

// A coarser ring leaves a hole where the next finer ring is fully in (its interior, not its fade band): each ring's relief
// comes from its own height level, so where the rings overlap a coarse ring's smoothed valleys would sit above the fine
// ring's real ones and hide them from the depth test. Haze grows with the air crossed (the path below uAirTop km), not
// with the distance, so a view straight down from orbit stays clear while a low view's horizon fades.
const FRAG = /* glsl */`uniform sampler2D tColor, tMask, tInner; uniform vec2 uCOff, uVRange, uInnerMin, uInnerCOff; uniform vec3 uSun, uHazeCol;
  uniform float uVis, uHazeK, uHazeL, uInnerOn, uInnerScale, uAirTop;
  varying vec2 vUv; varying vec3 vW, vN, vUp;
  void main() {
    vec3 col = texture2D(tColor, uCOff + vUv).rgb; float m = texture2D(tMask, uCOff + vUv).r;
    vec2 iu = (vUv - uInnerMin) * uInnerScale; float inner = texture2D(tInner, uInnerCOff + iu).r;
    if (vUv.y < uVRange.x || vUv.y > uVRange.y || m < 0.5) discard;          // beyond ±85° or a tile not (yet) in
    if (uInnerOn > 0.5 && min(iu.x, iu.y) > 0.1 && max(iu.x, iu.y) < 0.9 && inner > 0.5) discard;
    float sunUp = dot(vUp, uSun), day = smoothstep(-0.08, 0.12, sunUp);
    float relief = clamp(mix(1.0, dot(vN, uSun) / max(sunUp, 0.15), 0.8), 0.35, 1.6);   // slopes relative to flat ground: the photo keeps its own light
    col *= mix(0.03, 1.0, day) * mix(1.0, relief, day);
    float dist = length(cameraPosition - vW), air = dist * min(1.0, uAirTop / max(cameraPosition.y - vW.y, uAirTop));
    col = mix(col, uHazeCol * (0.2 + 0.8 * day), uHazeK * (1.0 - exp(-air / uHazeL)));
    vec2 e = smoothstep(vec2(0.0), vec2(0.08), vUv) * smoothstep(vec2(0.0), vec2(0.08), 1.0 - vUv);
    gl_FragColor = vec4(col, uVis * e.x * e.y * day); }`;          // the night side fades to the globe's city lights

export function createTileLoader({ maxInFlight = 8, keep = 192 } = {}) {
  const cache = new Map(), failed = new Map(), warned = new Set(), queue = [], recent = [], stats = { requested: 0, loaded: 0, failed: 0, blank: 0 };
  let inFlight = 0;
  const hex = (buf) => Array.from(new Uint8Array(buf), (b) => b.toString(16).padStart(2, '0')).join('');
  const isBlank = async (buf) => !!ESRI_BLANK && buf.byteLength === ESRI_BLANK.bytes && !!globalThis.crypto?.subtle && hex(await crypto.subtle.digest('SHA-1', buf)) === ESRI_BLANK.sha1;
  const remember = (url, bmp) => { cache.delete(url); cache.set(url, bmp); if (cache.size > keep) cache.delete(cache.keys().next().value); };
  const note = (bad) => { recent.push(bad ? 1 : 0); if (recent.length > 24) recent.shift(); };
  async function run(job) {
    inFlight++; let bmp = null;
    try {
      const res = await fetch(job.url, { mode: 'cors' });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const buf = await res.arrayBuffer();
      if (await isBlank(buf)) { stats.blank++; failed.set(job.url, Infinity); }
      else { bmp = await createImageBitmap(new Blob([buf]), job.raw ? { colorSpaceConversion: 'none', premultiplyAlpha: 'none' } : {}); remember(job.url, bmp); stats.loaded++; }
      note(false);
    } catch (e) {
      stats.failed++; failed.set(job.url, performance.now()); note(true);
      const host = new URL(job.url).host;
      if (!warned.has(host)) { warned.add(host); console.warn(`earth zoom: tiles from ${host} are not loading (${e.message})`); }
    }
    inFlight--; job.done(bmp); pump();
  }
  function pump() {
    if (queue.length > 1) queue.sort((a, b) => a.prio - b.prio);
    while (inFlight < maxInFlight && queue.length) { const job = queue.shift(); if (job.wanted()) run(job); else job.done(null); }
  }
  return {
    stats, get inFlight() { return inFlight; }, get queued() { return queue.length; },
    get failRate() { return recent.length < 8 ? 0 : recent.reduce((a, b) => a + b, 0) / recent.length; },
    peek: (url) => cache.get(url) || null,
    request(job) {
      const hit = cache.get(job.url);
      if (hit) { remember(job.url, hit); job.done(hit); return; }
      const t = failed.get(job.url);
      if (t !== undefined && performance.now() - t < RETRY_MS) { job.done(null); return; }
      stats.requested++; queue.push(job); pump();
    },
    prune() { for (let i = queue.length - 1; i >= 0; i--) if (!queue[i].wanted()) { queue[i].done(null); queue.splice(i, 1); } },
  };
}

export function createEarthRings(scene, renderer, { loader = createTileLoader() } = {}) {
  const uploads = [], dst = new THREE.Vector2(), cpu = new Map(), _p = [0, 0, 0], _u = [0, 0, 0];
  const zero = new THREE.DataTexture(new Uint8Array(T * T * 4), T, T, THREE.RGBAFormat); zero.needsUpdate = true;
  let frame = null, scratch = null;
  const count = (w, k) => { let c = 0; for (let i = 0; i < w.length; i++) if (w[i] && w[i] === k[i]) c++; return c; };
  function atlas(size, colorSpace, mips) {
    const t = new THREE.DataTexture(new Uint8Array(size * size * 4), size, size, THREE.RGBAFormat);
    t.colorSpace = colorSpace; t.wrapS = t.wrapT = THREE.RepeatWrapping; t.flipY = false; t.generateMipmaps = mips; t.anisotropy = mips ? 8 : 1;
    t.minFilter = mips ? THREE.LinearMipmapLinearFilter : THREE.NearestFilter; t.magFilter = mips ? THREE.LinearFilter : THREE.NearestFilter; t.needsUpdate = true;
    renderer.initTexture(t); return t;
  }
  function makeRing() {
    const n = GRID + 1, geo = new THREE.BufferGeometry(), uv = new Float32Array(n * n * 2), idx = new Uint32Array(GRID * GRID * 6);
    for (let j = 0, k = 0; j < n; j++) for (let i = 0; i < n; i++, k += 2) { uv[k] = i / GRID; uv[k + 1] = j / GRID; }
    for (let j = 0, k = 0; j < GRID; j++) for (let i = 0; i < GRID; i++, k += 6) {
      const a = j * n + i; idx[k] = a; idx[k + 1] = a + n; idx[k + 2] = a + 1; idx[k + 3] = a + 1; idx[k + 4] = a + n; idx[k + 5] = a + n + 1;
    }
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * n * 3), 3)); geo.setAttribute('aUp', new THREE.BufferAttribute(new Float32Array(n * n * 3), 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2)); geo.setIndex(new THREE.BufferAttribute(idx, 1));
    const color = atlas(RING_TILES * T, THREE.SRGBColorSpace, true), height = atlas(HEIGHT_TILES * T, THREE.NoColorSpace, false);
    const mask = new THREE.DataTexture(new Uint8Array(RING_TILES * RING_TILES), RING_TILES, RING_TILES, THREE.RedFormat);
    mask.wrapS = mask.wrapT = THREE.RepeatWrapping; mask.minFilter = mask.magFilter = THREE.NearestFilter; mask.needsUpdate = true;
    const mat = new THREE.ShaderMaterial({ vertexShader: VERT, fragmentShader: FRAG, transparent: true, depthWrite: true, polygonOffset: true,
      uniforms: { tColor: { value: color }, tHeight: { value: height }, tMask: { value: mask }, tInner: { value: mask }, uInnerMin: { value: new THREE.Vector2() }, uInnerCOff: { value: new THREE.Vector2() }, uInnerOn: { value: 0 }, uInnerScale: { value: 2 }, uAirTop: { value: 8 },
        uCOff: { value: new THREE.Vector2() }, uHOff: { value: new THREE.Vector2() },
        uHScale: { value: 1 }, uHSize: { value: HEIGHT_TILES * T }, uHeightK: { value: 0 }, uStepKm: { value: 1 }, uVRange: { value: new THREE.Vector2(0, 1) }, uVis: { value: 0 },
        uSun: { value: new THREE.Vector3(0, 1, 0) }, uHazeCol: { value: new THREE.Color(0.62, 0.74, 0.9) }, uHazeK: { value: 0 }, uHazeL: { value: 60 } } });
    const mesh = new THREE.Mesh(geo, mat); mesh.frustumCulled = false; mesh.visible = false; scene.add(mesh);
    const S = RING_TILES * RING_TILES, H = HEIGHT_TILES * HEIGHT_TILES;
    return { level: -1, win: null, hwin: null, mesh, mat, color, height, mask, vis: 0, hk: 0, shown: false, hShown: false, valid: 0, hValid: 0, stats: null,
      want: Array(S).fill(''), key: Array(S).fill(''), hwant: Array(H).fill(''), hkey: Array(H).fill(''), hbmp: Array(H).fill(null) };
  }
  const rings = Array.from({ length: RING_COUNT }, () => makeRing());

  function copy(src, target, slot, size) { dst.set(mod(slot, size) * T, Math.floor(slot / size) * T); renderer.copyTextureToTexture(src, target, null, dst); }
  function build(r) {
    const { level, x0, y0 } = r.win, n = 2 ** level, pos = r.mesh.geometry.attributes.position, up = r.mesh.geometry.attributes.aUp, lats = [], lons = [];
    for (let i = 0; i <= GRID; i++) {
      lons.push(tileToLonLat(x0 + RING_TILES * i / GRID, 0, level).lon);
      lats.push(tileToLonLat(0, Math.min(n, Math.max(0, y0 + RING_TILES * i / GRID)), level).lat);
    }
    for (let j = 0, k = 0; j <= GRID; j++) for (let i = 0; i <= GRID; i++, k += 3) {
      frame.toLocal(lats[j], lons[i], 0, _p); frame.upAt(lats[j], lons[i], _u);
      pos.array[k] = _p[0]; pos.array[k + 1] = _p[1]; pos.array[k + 2] = _p[2]; up.array[k] = _u[0]; up.array[k + 1] = _u[1]; up.array[k + 2] = _u[2];
    }
    pos.needsUpdate = true; up.needsUpdate = true;
  }
  function place(r, level, lat, lon) {
    const win = ringWindow(lon, lat, level);
    if (r.level === level && r.win && r.win.x0 === win.x0 && r.win.y0 === win.y0) return;
    if (r.level !== level) {
      r.level = level; r.vis = 0; r.hk = 0; r.shown = r.hShown = false; r.key.fill(''); r.hkey.fill(''); r.hbmp.fill(null); r.mask.image.data.fill(0);
      for (let s = 0; s < HEIGHT_TILES * HEIGHT_TILES; s++) copy(zero, r.height, s, HEIGHT_TILES);
    }
    r.win = win; r.hwin = heightWindow(win); build(r);
    const u = r.mat.uniforms, hw = r.hwin, n = 2 ** level, centreLat = tileToLonLat(0, win.y0 + RING_TILES / 2, level).lat;
    u.uCOff.value.set(mod(win.x0, RING_TILES) / RING_TILES, mod(win.y0, RING_TILES) / RING_TILES);
    u.uHOff.value.set(mod(hw.x0, HEIGHT_TILES) / HEIGHT_TILES + hw.off[0], mod(hw.y0, HEIGHT_TILES) / HEIGHT_TILES + hw.off[1]); u.uHScale.value = hw.scale;
    u.uStepKm.value = tileSizeKm(level, centreLat) * RING_TILES / GRID; u.uVRange.value.set(-win.y0 / RING_TILES, (n - win.y0) / RING_TILES);
    r.mesh.renderOrder = 2 + level; r.mat.polygonOffsetFactor = MAX_LEVEL - level; r.mat.polygonOffsetUnits = 4 * (MAX_LEVEL - level);
    const src = sourceForLevel(level), prio = (MAX_LEVEL - level) * 100;
    r.want.fill('');
    const tiles = windowTiles(win, RING_TILES); r.valid = tiles.length;
    for (const t of tiles) {
      r.want[t.slot] = t.key;
      if (r.key[t.slot] === t.key) continue;
      loader.request({ url: tileUrl(src, level, t.x, t.y), prio: prio + t.d, raw: false, wanted: () => r.level === level && r.want[t.slot] === t.key,
        done: (bmp) => { if (bmp) uploads.push({ r, slot: t.slot, key: t.key, bmp, height: false }); } });
    }
    for (let s = 0; s < r.want.length; s++) if (r.want[s] !== r.key[s]) r.mask.image.data[s] = 0;
    r.mask.needsUpdate = true;
    r.hwant.fill('');
    const htiles = windowTiles({ level: hw.level, x0: hw.x0, y0: hw.y0 }, HEIGHT_TILES); r.hValid = htiles.length;
    for (const t of htiles) {
      r.hwant[t.slot] = t.key;
      if (r.hkey[t.slot] === t.key) continue;
      if (r.hkey[t.slot]) { copy(zero, r.height, t.slot, HEIGHT_TILES); r.hkey[t.slot] = ''; r.hbmp[t.slot] = null; }
      loader.request({ url: tileUrl(HEIGHT_SOURCE, hw.level, t.x, t.y), prio: prio + 50 + t.d, raw: true, wanted: () => r.level === level && r.hwant[t.slot] === t.key,
        done: (bmp) => { if (bmp) uploads.push({ r, slot: t.slot, key: t.key, bmp, height: true }); } });
    }
  }
  function flush() {
    for (let done = 0; done < UPLOADS_PER_FRAME && uploads.length; ) {
      const u = uploads.shift(), r = u.r;
      if ((u.height ? r.hwant : r.want)[u.slot] !== u.key) continue;
      const src = new THREE.Texture(u.bmp); src.flipY = false; src.generateMipmaps = false; src.colorSpace = u.height ? THREE.NoColorSpace : THREE.SRGBColorSpace;
      try {
        copy(src, u.height ? r.height : r.color, u.slot, u.height ? HEIGHT_TILES : RING_TILES);
        if (u.height) { r.hkey[u.slot] = u.key; r.hbmp[u.slot] = u.bmp; }
        else { r.key[u.slot] = u.key; r.mask.image.data[u.slot] = 255; r.mask.needsUpdate = true; }
      } catch (e) {
        console.warn('earth zoom: a tile could not be uploaded', e.message);
      }
      src.dispose(); done++;
    }
  }
  function decoded(key, bmp) {
    let h = cpu.get(key);
    if (h) return h;
    try {
      scratch = scratch || new OffscreenCanvas(T, T).getContext('2d', { willReadFrequently: true });
      scratch.clearRect(0, 0, T, T); scratch.drawImage(bmp, 0, 0); const d = scratch.getImageData(0, 0, T, T).data;
      h = new Float32Array(T * T); for (let i = 0; i < T * T; i++) h[i] = decodeTerrarium(d[i * 4], d[i * 4 + 1], d[i * 4 + 2]);
    } catch (e) { return null; }
    cpu.set(key, h); if (cpu.size > 16) cpu.delete(cpu.keys().next().value);
    return h;
  }

  return {
    loader,
    rebase(f) { frame = f; for (const r of rings) if (r.win) build(r); },
    update(dt, view) {
      const want = ringLevels(view.L0);
      for (const L of want) place(rings.find((q) => q.level === L) || rings.find((q) => !want.includes(q.level)), L, view.lat, view.lon);
      loader.prune(); flush();
      const ease = 1 - Math.exp(-dt * 3);
      for (const r of rings) {
        const used = want.includes(r.level), have = used ? count(r.want, r.key) : 0, hHave = used ? count(r.hwant, r.hkey) : 0;
        r.shown = used && have >= (r.shown ? HIDE_AT : SHOW_AT) * r.valid;
        r.hShown = used && hHave >= (r.hShown ? HIDE_AT : SHOW_AT) * r.hValid;
        r.vis += ((r.shown ? 1 : 0) - r.vis) * ease; r.hk += ((r.hShown ? 1 : 0) - r.hk) * ease;
        const u = r.mat.uniforms; u.uVis.value = r.vis * view.vis; u.uHeightK.value = r.hk; u.uSun.value.copy(view.sun); u.uHazeK.value = view.hazeK; u.uHazeL.value = view.hazeL;
        r.mesh.visible = u.uVis.value > 0.003; r.stats = { have, valid: r.valid, hHave, hValid: r.hValid };
      }
      for (const r of rings) {
        const finer = want.filter((L) => L > r.level), inner = want.includes(r.level) && finer.length && rings.find((q) => q.level === Math.min(...finer)), u = r.mat.uniforms;
        u.uInnerOn.value = inner && inner.vis > 0.98 ? 1 : 0;
        if (!u.uInnerOn.value) continue;
        const s = 2 ** (inner.level - r.level);
        u.tInner.value = inner.mask; u.uInnerCOff.value.copy(inner.mat.uniforms.uCOff.value); u.uInnerScale.value = s;
        u.uInnerMin.value.set((inner.win.x0 / s - r.win.x0) / RING_TILES, (inner.win.y0 / s - r.win.y0) / RING_TILES);
      }
    },
    heightAt(latDeg, lonDeg) {
      for (const r of rings.filter((q) => q.hwin).sort((a, b) => b.level - a.level)) {
        const hl = r.hwin.level, n = 2 ** hl, t = lonLatToTile(lonDeg, latDeg, hl), hx = Math.floor(t.x), hy = Math.floor(t.y);
        if (hy < 0 || hy >= n) continue;
        const slot = mod(hy, HEIGHT_TILES) * HEIGHT_TILES + mod(hx, HEIGHT_TILES), key = `${hl}/${mod(hx, n)}/${hy}`;
        if (r.hkey[slot] !== key || !r.hbmp[slot]) continue;
        const h = decoded(key, r.hbmp[slot]);
        if (!h) continue;
        const fx = Math.min(T - 1.001, Math.max(0, (t.x - hx) * T - 0.5)), fy = Math.min(T - 1.001, Math.max(0, (t.y - hy) * T - 0.5));
        const ix = Math.floor(fx), iy = Math.floor(fy), ax = fx - ix, ay = fy - iy, i = iy * T + ix;
        return ((h[i] * (1 - ax) + h[i + 1] * ax) * (1 - ay) + (h[i + T] * (1 - ax) + h[i + T + 1] * ax) * ay) / 1000;
      }
      return 0;
    },
    get settled() { return loader.inFlight === 0 && loader.queued === 0 && uploads.length === 0; },
    get info() {
      return { rings: rings.map((r) => ({ level: r.level, vis: +r.vis.toFixed(2), height: +r.hk.toFixed(2), ...(r.stats || {}) })),
        inFlight: loader.inFlight, queued: loader.queued, uploads: uploads.length, ...loader.stats, failRate: +loader.failRate.toFixed(2) };
    },
  };
}
