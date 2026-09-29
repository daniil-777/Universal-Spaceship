// earthrings.js — the Earth zoom view's ground: five nested rings of real imagery with real relief around the view's
// target, streamed the way src/terrain.js streams its strips (one curved mesh + one GPU atlas per ring, tiles copied in
// as they arrive, an atlas addressed toroidally so re-centring fetches only the new row or column). Rings sit on levels
// L0, L0−1, L0−2 and two even outer levels (src/earthtiles.js ringLevels); a ring shows once 90 % of its tiles are in, missing
// tiles are asked for again every RETRY_S seconds (a network blip heals without panning), and a per-slot mask hides
// any tile that is not (yet) in together with the height tile under it, so the coarser ring or the globe covers it —
// no holes, and no imagery drawn at sea level while its relief is still on the way. Heights: Terrarium tiles one level
// coarser in a 4 × 4 atlas, decoded in the vertex shader. Tiles come from src/earthloader.js. Units: km in the view's
// local frame.
import * as THREE from 'three';
import { RING_TILES, RING_COUNT, HEIGHT_TILES, MAX_LEVEL, HEIGHT_SOURCE, sourceForLevel, tileUrl, ringLevels, ringWindow, heightWindow, heightTileFor, windowTiles, innerWindow, innerState, tileToLonLat, lonLatToTile, tileSizeKm, decodeTerrarium, mod } from './earthtiles.js';
import { createTileLoader } from './earthloader.js';

const T = 256, GRID = 128, SHOW_AT = 0.9, HIDE_AT = 0.5, UPLOADS_PER_FRAME = 6, RETRY_S = 10;

const VERT = /* glsl */`uniform sampler2D tHeight; uniform vec2 uHOff; uniform float uHScale, uHSize, uHeightK, uStepKm;
  attribute vec3 aUp; varying vec2 vUv; varying vec3 vW, vN, vUp; varying float vSea;
  float hRaw(vec2 t) { vec3 c = floor(texture2D(tHeight, t).rgb * 255.0 + 0.5); return c.r * 256.0 + c.g + c.b / 256.0 - 32768.0; }
  float hAt(vec2 t) { return max(hRaw(t), 0.0); }
  float hBil(vec2 uv) {                                   // Terrarium RGB must not be filtered by the hardware: 4 nearest taps
    vec2 p = (uHOff + uv * uHScale) * uHSize - 0.5, f = fract(p), b = (floor(p) + 0.5) / uHSize; float d = 1.0 / uHSize;
    return mix(mix(hAt(b), hAt(b + vec2(d, 0.0)), f.x), mix(hAt(b + vec2(0.0, d)), hAt(b + vec2(d)), f.x), f.y); }
  void main() {
    vUv = uv; float k = 0.001 * uHeightK, du = ${(1 / GRID).toFixed(7)};
    float h = hBil(uv) * k, hx = hBil(uv + vec2(du, 0.0)) * k, hy = hBil(uv + vec2(0.0, du)) * k;
    vUp = normalize(aUp); vN = normalize(vUp + vec3(-(hx - h), 0.0, -(hy - h)) / uStepKm);   // uv.y runs south = +z
    float raw = hRaw((floor((uHOff + uv * uHScale) * uHSize) + 0.5) / uHSize);
    vSea = raw < -1.0 && raw > -12000.0 ? 1.0 : 0.0;          // below sea level (an empty atlas slot decodes to -32768: unknown, not sea)
    vec4 w = modelMatrix * vec4(position + vUp * h, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`;

// A coarser ring never competes in depth with the next finer ring (diagnosis 2026-09-29, fix A): each ring's relief comes
// from its own height level, so where the rings overlap a coarse ring's smoothed valleys would sit above the fine ring's
// real ones and win the depth test. Under the finer ring's footprint (where its tiles are in) the coarse ring is pushed to
// the far plane, so it only ever shows THROUGH the finer ring (while that one fades in, and in its edge band), and once
// the finer ring's imagery and relief are fully in, the coarse ring leaves a hole in its interior. The test wraps in
// longitude (uInnerWrap): a level-2 ring spans the globe twice, and a copy of it left at its true depth would hide the
// pushed rings. Haze grows with the air crossed (the path below uAirTop km), not with the distance, so a view straight
// down from orbit stays clear while a low view's horizon fades.
const FRAG = /* glsl */`uniform sampler2D tColor, tMask, tInner; uniform vec2 uCOff, uVRange, uInnerMin, uInnerCOff; uniform vec3 uSun, uHazeCol;
  uniform float uVis, uHazeK, uHazeL, uInnerOn, uInnerDone, uInnerScale, uInnerWrap, uAirTop; uniform vec3 uSeaCol;
  uniform vec3 uTint; uniform float uTintOn;
  varying vec2 vUv; varying vec3 vW, vN, vUp; varying float vSea;
  void main() {
    vec3 col = texture2D(tColor, uCOff + vUv).rgb; float m = texture2D(tMask, uCOff + vUv).r;
    vec2 iu = vec2(mod(vUv.x - uInnerMin.x, uInnerWrap), vUv.y - uInnerMin.y) * uInnerScale; float inner = texture2D(tInner, uInnerCOff + iu).r;
    if (vUv.y < uVRange.x || vUv.y > uVRange.y || m < 0.5) discard;          // beyond ±85° or a tile not (yet) in
    bool under = uInnerOn > 0.5 && iu.x > 0.0 && iu.y > 0.0 && iu.x < 1.0 && iu.y < 1.0 && inner > 0.5;
    if (under && uInnerDone > 0.5 && min(iu.x, iu.y) > 0.1 && max(iu.x, iu.y) < 0.9) discard;
    float lum = dot(col, vec3(0.2126, 0.7152, 0.0722));
    col = mix(col, uSeaCol, vSea * (1.0 - smoothstep(0.004, 0.03, lum)));   // Sentinel-2 cloudless draws the open sea near-black
    float sunUp = dot(vUp, uSun), day = smoothstep(-0.08, 0.12, sunUp);
    float relief = clamp(mix(1.0, dot(vN, uSun) / max(sunUp, 0.15), 0.8), 0.35, 1.6);   // slopes relative to flat ground: the photo keeps its own light
    col *= mix(0.03, 1.0, day) * mix(1.0, relief, day);
    float dist = length(cameraPosition - vW), air = dist * min(1.0, uAirTop / max(cameraPosition.y - vW.y, uAirTop));
    col = mix(col, uHazeCol * (0.2 + 0.8 * day), uHazeK * (1.0 - exp(-air / uHazeL)));
    vec2 e = smoothstep(vec2(0.0), vec2(0.08), vUv) * smoothstep(vec2(0.0), vec2(0.08), 1.0 - vUv);
    gl_FragDepth = under ? 0.99999 : gl_FragCoord.z;
    col = mix(col, uTint, uTintOn);
    gl_FragColor = vec4(col, uVis * e.x * e.y * day); }`;          // the night side fades to the globe's city lights

// Debug false colour for the leak check (diagnosis §1(a)): one tint per ring, keyed off the ring's own level so it
// tracks a level change and never hard-codes which colour is the finest ring (MAX_LEVEL may change later).
const DEBUG_TINTS = [0xff0000, 0x0080ff, 0x00c000, 0xffd000, 0xffffff];
// The atlases start empty and are only ever written on the GPU: all of them share one zero buffer per size.
const ZEROS = new Map();
const zeros = (n) => { if (!ZEROS.has(n)) ZEROS.set(n, new Uint8Array(n)); return ZEROS.get(n); };

export function createEarthRings(scene, renderer, { loader = createTileLoader() } = {}) {
  const uploads = [], dst = new THREE.Vector2(), cpu = new Map(), _p = [0, 0, 0], _u = [0, 0, 0];
  const zero = new THREE.DataTexture(zeros(T * T * 4), T, T, THREE.RGBAFormat); zero.needsUpdate = true;
  let frame = null, scratch = null, retryT = 0, debugTint = false;
  const count = (w, k) => { let c = 0; for (let i = 0; i < w.length; i++) if (w[i] && w[i] === k[i]) c++; return c; };
  function atlas(size, colorSpace, mips) {
    const t = new THREE.DataTexture(zeros(size * size * 4), size, size, THREE.RGBAFormat);
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
      uniforms: { tColor: { value: color }, tHeight: { value: height }, tMask: { value: mask }, tInner: { value: mask }, uInnerMin: { value: new THREE.Vector2() }, uInnerCOff: { value: new THREE.Vector2() }, uInnerOn: { value: 0 }, uInnerDone: { value: 0 }, uInnerScale: { value: 2 }, uInnerWrap: { value: 1 }, uAirTop: { value: 8 }, uSeaCol: { value: new THREE.Color(0.012, 0.035, 0.1) },
        uCOff: { value: new THREE.Vector2() }, uHOff: { value: new THREE.Vector2() },
        uHScale: { value: 1 }, uHSize: { value: HEIGHT_TILES * T }, uHeightK: { value: 0 }, uStepKm: { value: 1 }, uVRange: { value: new THREE.Vector2(0, 1) }, uVis: { value: 0 },
        uSun: { value: new THREE.Vector3(0, 1, 0) }, uHazeCol: { value: new THREE.Color(0.62, 0.74, 0.9) }, uHazeK: { value: 0 }, uHazeL: { value: 60 }, uTint: { value: new THREE.Color(1, 1, 1) }, uTintOn: { value: 0 } } });
    const mesh = new THREE.Mesh(geo, mat); mesh.frustumCulled = false; mesh.visible = false; scene.add(mesh);
    const S = RING_TILES * RING_TILES, H = HEIGHT_TILES * HEIGHT_TILES;
    return { level: -1, win: null, hwin: null, mesh, mat, color, height, mask, vis: 0, hk: 0, shown: false, hShown: false, valid: 0, hValid: 0, stats: null,
      want: Array(S).fill(''), key: Array(S).fill(''), pend: Array(S).fill(''), tile: Array(S).fill(null),
      hwant: Array(H).fill(''), hkey: Array(H).fill(''), hpend: Array(H).fill(''), hfail: Array(H).fill(''), hbmp: Array(H).fill(null), htile: Array(H).fill(null) };
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
      r.level = level; r.vis = 0; r.hk = 0; r.shown = r.hShown = false; r.key.fill(''); r.pend.fill(''); r.hkey.fill(''); r.hpend.fill(''); r.hfail.fill(''); r.hbmp.fill(null);
      r.mask.image.data.fill(0); r.mask.needsUpdate = true;
      for (let s = 0; s < HEIGHT_TILES * HEIGHT_TILES; s++) copy(zero, r.height, s, HEIGHT_TILES);
    }
    r.win = win; r.hwin = heightWindow(win); build(r);
    const u = r.mat.uniforms, hw = r.hwin, n = 2 ** level, centreLat = tileToLonLat(0, win.y0 + RING_TILES / 2, level).lat;
    u.uCOff.value.set(mod(win.x0, RING_TILES) / RING_TILES, mod(win.y0, RING_TILES) / RING_TILES);
    u.uHOff.value.set(mod(hw.x0, HEIGHT_TILES) / HEIGHT_TILES + hw.off[0], mod(hw.y0, HEIGHT_TILES) / HEIGHT_TILES + hw.off[1]); u.uHScale.value = hw.scale;
    u.uStepKm.value = tileSizeKm(level, centreLat) * RING_TILES / GRID; u.uVRange.value.set(-win.y0 / RING_TILES, (n - win.y0) / RING_TILES);
    r.mesh.renderOrder = 2 + level; r.mat.polygonOffsetFactor = MAX_LEVEL - level; r.mat.polygonOffsetUnits = 4 * (MAX_LEVEL - level);
    r.want.fill('');
    const tiles = windowTiles(win, RING_TILES); r.valid = tiles.length;
    for (const t of tiles) {
      r.want[t.slot] = t.key; r.tile[t.slot] = t;
      if (r.key[t.slot] !== t.key && r.pend[t.slot] !== t.key) requestColour(r, t);
    }
    r.hwant.fill('');
    const htiles = windowTiles({ level: hw.level, x0: hw.x0, y0: hw.y0 }, HEIGHT_TILES); r.hValid = htiles.length;
    for (const t of htiles) {
      r.hwant[t.slot] = t.key; r.htile[t.slot] = t;
      if (r.hkey[t.slot] === t.key || r.hfail[t.slot] === t.key || r.hpend[t.slot] === t.key) continue;
      if (r.hkey[t.slot]) { copy(zero, r.height, t.slot, HEIGHT_TILES); r.hkey[t.slot] = ''; r.hbmp[t.slot] = null; }
      r.hfail[t.slot] = '';
      requestHeight(r, t);
    }
  }
  function requestColour(r, t) {
    const level = r.level; r.pend[t.slot] = t.key;
    loader.request({ url: tileUrl(sourceForLevel(level), level, t.x, t.y), prio: (MAX_LEVEL - level) * 100 + t.d, raw: false, wanted: () => r.level === level && r.want[t.slot] === t.key,
      done: (bmp) => {
        if (r.pend[t.slot] === t.key) r.pend[t.slot] = '';
        if (bmp) uploads.push({ r, slot: t.slot, key: t.key, bmp, height: false });
      } });
  }
  function requestHeight(r, t) {
    const level = r.level, hl = r.hwin.level, wanted = () => r.level === level && r.hwant[t.slot] === t.key; r.hpend[t.slot] = t.key;
    loader.request({ url: tileUrl(HEIGHT_SOURCE, hl, t.x, t.y), prio: (MAX_LEVEL - level) * 100 + 50 + t.d, raw: true, wanted,
      done: (bmp, why) => {
        if (r.hpend[t.slot] === t.key) r.hpend[t.slot] = '';
        if (bmp) uploads.push({ r, slot: t.slot, key: t.key, bmp, height: true });
        else if ((why === 'failed' || why === 'blank') && wanted()) r.hfail[t.slot] = t.key;   // no relief to wait for here: show the imagery flat
      } });
  }
  // Ask again for whatever is still missing and not on its way (the loader's retry time gates real refetches).
  function retryMissing(r) {
    for (let s = 0; s < r.want.length; s++) if (r.want[s] && r.key[s] !== r.want[s] && r.pend[s] !== r.want[s]) requestColour(r, r.tile[s]);
    for (let s = 0; s < r.hwant.length; s++) if (r.hwant[s] && r.hkey[s] !== r.hwant[s] && r.hpend[s] !== r.hwant[s]) requestHeight(r, r.htile[s]);
  }
  function flush() {
    for (let done = 0; done < UPLOADS_PER_FRAME && uploads.length; ) {
      const u = uploads.shift(), r = u.r;
      if ((u.height ? r.hwant : r.want)[u.slot] !== u.key) continue;
      const src = new THREE.Texture(u.bmp); src.flipY = false; src.generateMipmaps = false; src.colorSpace = u.height ? THREE.NoColorSpace : THREE.SRGBColorSpace;
      try {
        copy(src, u.height ? r.height : r.color, u.slot, u.height ? HEIGHT_TILES : RING_TILES);
        if (u.height) { r.hkey[u.slot] = u.key; r.hbmp[u.slot] = u.bmp; } else r.key[u.slot] = u.key;
      } catch (e) {
        console.warn('earth zoom: a tile could not be uploaded', e.message);
      }
      src.dispose(); done++;
    }
  }
  // A colour slot shows once its imagery AND the height tile under it are in (or that height tile failed for good).
  function refreshMask(r) {
    const m = r.mask.image.data; let changed = false;
    for (let s = 0; s < m.length; s++) {
      let on = 0;
      if (r.want[s] && r.key[s] === r.want[s]) { const t = r.tile[s], h = heightTileFor(t.x, t.y, r.level, r.hwin); on = r.hkey[h.slot] === h.key || r.hfail[h.slot] === h.key ? 255 : 0; }
      if (m[s] !== on) { m[s] = on; changed = true; }
    }
    if (changed) r.mask.needsUpdate = true;
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
    // A closed view stops fetching; on reopening every ring re-requests what it is still missing.
    suspend() { loader.suspend(); uploads.length = 0; },
    resume() { loader.resume(); for (const r of rings) r.win = null; },
    setDebugTint(on) { debugTint = !!on; },
    update(dt, view) {
      const want = ringLevels(view.L0);
      for (const L of want) place(rings.find((q) => q.level === L) || rings.find((q) => !want.includes(q.level)), L, view.lat, view.lon);
      loader.prune(); flush();
      retryT += dt;
      if (retryT > RETRY_S) { retryT = 0; for (const r of rings) if (r.win && want.includes(r.level)) retryMissing(r); }
      const ease = 1 - Math.exp(-dt * 3);
      for (const r of rings) {
        // a height tile that failed for good counts as done: the rest of the ring's relief still shows (that tile stays flat)
        const used = want.includes(r.level), have = used ? count(r.want, r.key) : 0, hHave = used ? count(r.hwant, r.hkey) + count(r.hwant, r.hfail) : 0;
        if (used) refreshMask(r);
        r.shown = used && have >= (r.shown ? HIDE_AT : SHOW_AT) * r.valid;
        r.hShown = used && hHave >= (r.hShown ? HIDE_AT : SHOW_AT) * r.hValid;
        r.vis += ((r.shown ? 1 : 0) - r.vis) * ease; r.hk += ((r.hShown ? 1 : 0) - r.hk) * ease;
        const u = r.mat.uniforms; u.uVis.value = r.vis * view.vis; u.uHeightK.value = r.hk; u.uSun.value.copy(view.sun); u.uHazeK.value = view.hazeK; u.uHazeL.value = view.hazeL;
        u.uTintOn.value = debugTint ? 1 : 0; if (debugTint) u.uTint.value.setHex(DEBUG_TINTS[(MAX_LEVEL - r.level) % 5]);
        r.mesh.visible = u.uVis.value > 0.003; r.stats = { have, valid: r.valid, hHave, hValid: r.hValid };
      }
      // fix A: every used ring with a finer used ring tracks that ring's window every frame, so it is behind the finer
      // ring as soon as that one is drawn at all (on) and leaves its hole once the finer ring is fully in (done)
      for (const r of rings) {
        const finer = want.filter((L) => L > r.level), inner = want.includes(r.level) && finer.length && rings.find((q) => q.level === Math.min(...finer)), u = r.mat.uniforms;
        const st = inner ? innerState(inner) : { on: false, done: false };
        u.uInnerOn.value = st.on ? 1 : 0; u.uInnerDone.value = st.done ? 1 : 0;
        if (!inner) continue;
        const w = innerWindow(inner.win, r.win);
        u.tInner.value = inner.mask; u.uInnerCOff.value.copy(inner.mat.uniforms.uCOff.value); u.uInnerScale.value = w.scale; u.uInnerMin.value.set(w.min[0], w.min[1]);
        u.uInnerWrap.value = w.wrap;
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
        // scaled like the drawn relief while it rises, so the camera and the marker never jump ahead of the ground
        return ((h[i] * (1 - ax) + h[i + 1] * ax) * (1 - ay) + (h[i + T] * (1 - ax) + h[i + T + 1] * ax) * ay) / 1000 * r.hk;
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
