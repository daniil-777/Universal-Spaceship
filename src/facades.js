// Facade texture array for the skylines: eight 512×512 tiles drawn on a canvas at load (no files), stored as layers of
// one 2D texture array so every tile repeats seamlessly with hardware mipmaps — three curtain-wall styles,
// premium floor-to-ceiling glass, brick, ashlar stone, precast panels with balconies, raw concrete. RGB = albedo with the
// frames, sills, mortar, panel joints and dirt already shaded in; A = how much of the pixel is glass (the shader puts the
// sky/sun reflection only there). Each tile repeats every `w` × `h` world units of facade (1 unit ≈ 8 m).
import * as THREE from 'three';

const S = 512;                                            // tile size in px
const TILES = { grid: 0, ribbon: 1, piers: 2, premium: 3, brick: 4, stone: 5, panel: 6, concrete: 7 };
export const TILE_UNITS = [[4, 4.05], [4, 4.05], [4, 4.05], [4, 4.05], [3, 3.15], [4, 4.05], [4, 4.05], [4, 4.05]];   // world units per repeat (w, h); h = 9 floors of 0.45 (7 of 0.45 for brick)

let seed = 7; const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
const rgb = (r, g, b) => `rgb(${r | 0},${g | 0},${b | 0})`;
const vary = (c, k) => c.map((v) => Math.max(0, Math.min(255, v + (rnd() - 0.5) * k)));

function paneRows(ctx, { floors, cols, sill, head, mullion, spandrel, paneCol, frameCol, litFrac, gloss = 1, dark = false }) {
  // a curtain-wall grid: `floors` rows of `cols` panes; spandrel = dark band fraction of a floor; mullion = frame width px
  const fh = S / floors, cw = S / cols;
  ctx.fillStyle = rgb(...frameCol); ctx.fillRect(0, 0, S, S);
  for (let r = 0; r < floors; r++) {
    const y0 = r * fh, spH = fh * spandrel;
    ctx.fillStyle = rgb(...vary(frameCol.map((v) => v * 0.58), 10)); ctx.fillRect(0, y0, S, spH);                           // spandrel band (clearly darker than the frames)
    ctx.fillStyle = 'rgba(0,0,0,0.25)'; ctx.fillRect(0, y0 + spH, S, 2);                                                  // shadow under the band
    for (let c = 0; c < cols; c++) {
      const x0 = c * cw, lit = rnd() < litFrac;
      const pc = vary(paneCol, 22).map((v) => v * (lit ? 1.35 : 1) * (0.9 + 0.2 * rnd()));
      const g = ctx.createLinearGradient(x0, y0 + spH, x0, y0 + fh); g.addColorStop(0, rgb(...pc.map((v) => v * 1.12))); g.addColorStop(1, rgb(...pc.map((v) => v * 0.8)));
      ctx.fillStyle = g; ctx.fillRect(x0 + mullion, y0 + spH + sill, cw - 2 * mullion, fh - spH - sill - head);
      if (dark && rnd() < 0.3) { ctx.fillStyle = 'rgba(255,255,255,0.10)'; ctx.fillRect(x0 + mullion, y0 + spH + sill, cw - 2 * mullion, (fh - spH) * 0.35); }   // blinds half down
      ctx.fillStyle = 'rgba(255,255,255,0.22)'; ctx.fillRect(x0 + mullion, y0 + spH + sill, 1, fh - spH - sill - head);      // frame highlight
      ctx.fillStyle = 'rgba(0,0,0,0.35)'; ctx.fillRect(x0 + cw - mullion - 1, y0 + spH + sill, 1, fh - spH - sill - head);   // frame shadow
    }
  }
  return gloss;
}
function glassMask({ floors, cols, sill, head, mullion, spandrel, amount = 1 }) {   // glass mask: 255 on the panes
  const fh = S / floors, cw = S / cols, m = new Uint8Array(S * S);
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const r = Math.floor(y / fh), c = Math.floor(x / cw), yy = y - r * fh, xx = x - c * cw, spH = fh * spandrel;
    m[y * S + x] = yy >= spH + sill && yy < fh - head && xx >= mullion && xx < cw - mullion ? Math.round(255 * amount) : 0;
  }
  return m;
}
function bricks(ctx, base, mortar, courseH, brickW) {     // running bond with per-brick colour variation and mortar joints
  ctx.fillStyle = rgb(...mortar); ctx.fillRect(0, 0, S, S);
  for (let r = 0, y = 0; y < S; r++, y += courseH) for (let x = (r % 2) * -brickW / 2; x < S; x += brickW) {
    ctx.fillStyle = rgb(...vary(base, 34)); ctx.fillRect(x + 1, y + 1, brickW - 2, courseH - 2);
    ctx.fillStyle = 'rgba(0,0,0,0.18)'; ctx.fillRect(x + 1, y + courseH - 2, brickW - 2, 1);                             // shadow at the bottom of each brick
  }
}
function punched(ctx, { floors, cols, wFrac, hFrac, sillCol, paneCol, litFrac, reveal = 3 }) {   // openings in a wall: reveal shadow, sill, lintel, dark pane
  const fh = S / floors, cw = S / cols, ww = cw * wFrac, wh = fh * hFrac;
  for (let r = 0; r < floors; r++) for (let c = 0; c < cols; c++) {
    const x0 = c * cw + (cw - ww) / 2, y0 = r * fh + fh * 0.18;
    ctx.fillStyle = 'rgba(0,0,0,0.45)'; ctx.fillRect(x0 - reveal, y0 - reveal, ww + 2 * reveal, wh + 2 * reveal);            // reveal
    const lit = rnd() < litFrac, pc = paneCol.map((v) => v * (lit ? 2.2 : 1));
    const g = ctx.createLinearGradient(x0, y0, x0, y0 + wh); g.addColorStop(0, rgb(...pc.map((v) => v * 1.3))); g.addColorStop(1, rgb(...pc.map((v) => v * 0.7)));
    ctx.fillStyle = g; ctx.fillRect(x0, y0, ww, wh);
    ctx.fillStyle = 'rgba(255,255,255,0.35)'; ctx.fillRect(x0 + ww * 0.48, y0, 2, wh); ctx.fillRect(x0, y0 + wh * 0.42, ww, 2);   // sash bars
    ctx.fillStyle = rgb(...sillCol); ctx.fillRect(x0 - reveal - 2, y0 + wh + reveal, ww + 2 * reveal + 4, 3);               // sill
    ctx.fillStyle = 'rgba(0,0,0,0.25)'; ctx.fillRect(x0 - reveal - 2, y0 + wh + reveal + 3, ww + 2 * reveal + 4, 2);       // sill shadow
  }
}
function punchedMask({ floors, cols, wFrac, hFrac, amount = 0.85 }) {
  const fh = S / floors, cw = S / cols, ww = cw * wFrac, wh = fh * hFrac, m = new Uint8Array(S * S);
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const r = Math.floor(y / fh), c = Math.floor(x / cw), x0 = c * cw + (cw - ww) / 2, y0 = r * fh + fh * 0.18;
    m[y * S + x] = x >= x0 && x < x0 + ww && y >= y0 && y < y0 + wh ? Math.round(255 * amount) : 0;
  }
  return m;
}
function grime(ctx, strength) {                            // vertical rain streaks and a little grime, everywhere
  for (let i = 0; i < 40; i++) { const x = rnd() * S, w = 2 + rnd() * 12, a = strength * rnd() * 0.35; const g = ctx.createLinearGradient(0, 0, 0, S); g.addColorStop(0, `rgba(0,0,0,${a})`); g.addColorStop(1, 'rgba(0,0,0,0)'); ctx.fillStyle = g; ctx.fillRect(x, 0, w, S); }
}

export function createFacadeAtlas() {
  const tile = document.createElement('canvas'); tile.width = tile.height = S; const ctx = tile.getContext('2d', { willReadFrequently: true }), data = new Uint8Array(S * S * 4 * 8);
  const put = (index, mask) => { const d = ctx.getImageData(0, 0, S, S).data, o = index * S * S * 4; for (let i = 0, n = S * S; i < n; i++) { d[i * 4 + 3] = mask[i]; } data.set(d, o); };   // RGB from the canvas, A = the glass mask
  const gridSpec = { floors: 9, cols: 16, sill: 0, head: 0, mullion: 2, spandrel: 0.3 };
  paneRows(ctx, { ...gridSpec, paneCol: [96, 118, 140], frameCol: [178, 182, 188], litFrac: 0.2, dark: true }); grime(ctx, 0.5); put(TILES.grid, glassMask(gridSpec));
  const ribbonSpec = { floors: 9, cols: 8, sill: 0, head: 0, mullion: 1, spandrel: 0.42 };
  paneRows(ctx, { ...ribbonSpec, paneCol: [84, 104, 124], frameCol: [150, 154, 160], litFrac: 0.15 }); grime(ctx, 0.4); put(TILES.ribbon, glassMask(ribbonSpec));
  const pierSpec = { floors: 9, cols: 11, sill: 0, head: 0, mullion: 7, spandrel: 0.22 };
  paneRows(ctx, { ...pierSpec, paneCol: [70, 88, 108], frameCol: [206, 200, 188], litFrac: 0.18, dark: true }); grime(ctx, 0.6); put(TILES.piers, glassMask(pierSpec));
  const premiumSpec = { floors: 9, cols: 18, sill: 0, head: 0, mullion: 2, spandrel: 0.14 };
  paneRows(ctx, { ...premiumSpec, paneCol: [92, 120, 150], frameCol: [150, 156, 166], litFrac: 0.22, dark: true }); put(TILES.premium, glassMask(premiumSpec));
  bricks(ctx, [146, 84, 62], [178, 168, 152], 8, 26); grime(ctx, 0.9); punched(ctx, { floors: 7, cols: 6, wFrac: 0.42, hFrac: 0.55, sillCol: [214, 206, 190], paneCol: [30, 36, 46], litFrac: 0.25 }); put(TILES.brick, punchedMask({ floors: 7, cols: 6, wFrac: 0.42, hFrac: 0.55 }));
  ctx.fillStyle = rgb(206, 196, 176); ctx.fillRect(0, 0, S, S);                                                          // ashlar stone
  for (let r = 0, y = 0; y < S; r++, y += 32) for (let x = (r % 2) * -32; x < S; x += 64) { ctx.fillStyle = rgb(...vary([206, 196, 176], 18)); ctx.fillRect(x + 1, y + 1, 62, 30); ctx.fillStyle = 'rgba(0,0,0,0.2)'; ctx.fillRect(x + 1, y + 30, 62, 1); }
  grime(ctx, 0.8); punched(ctx, { floors: 9, cols: 7, wFrac: 0.45, hFrac: 0.6, sillCol: [190, 180, 160], paneCol: [28, 32, 40], litFrac: 0.2, reveal: 4 }); put(TILES.stone, punchedMask({ floors: 9, cols: 7, wFrac: 0.45, hFrac: 0.6 }));
  ctx.fillStyle = rgb(196, 192, 184); ctx.fillRect(0, 0, S, S);                                                          // precast panels with balconies
  for (let y = 0; y < S; y += S / 9) { ctx.fillStyle = 'rgba(0,0,0,0.3)'; ctx.fillRect(0, y, S, 2); ctx.fillStyle = rgb(150, 148, 144); ctx.fillRect(0, y + S / 9 - 9, S, 7); ctx.fillStyle = 'rgba(0,0,0,0.35)'; ctx.fillRect(0, y + S / 9 - 2, S, 2); }   // floor joints, balcony slabs
  for (let x = 0; x < S; x += S / 4) { ctx.fillStyle = 'rgba(0,0,0,0.22)'; ctx.fillRect(x, 0, 2, S); }
  grime(ctx, 0.7); punched(ctx, { floors: 9, cols: 8, wFrac: 0.6, hFrac: 0.5, sillCol: [170, 168, 160], paneCol: [34, 40, 50], litFrac: 0.3, reveal: 2 }); put(TILES.panel, punchedMask({ floors: 9, cols: 8, wFrac: 0.6, hFrac: 0.5 }));
  ctx.fillStyle = rgb(168, 166, 160); ctx.fillRect(0, 0, S, S);                                                          // raw concrete
  for (let y = 0; y < S; y += 64) { ctx.fillStyle = 'rgba(0,0,0,0.25)'; ctx.fillRect(0, y, S, 2); } for (let x = 0; x < S; x += 128) { ctx.fillStyle = 'rgba(0,0,0,0.2)'; ctx.fillRect(x, 0, 2, S); }
  grime(ctx, 1.2); punched(ctx, { floors: 9, cols: 10, wFrac: 0.4, hFrac: 0.4, sillCol: [150, 148, 144], paneCol: [30, 34, 42], litFrac: 0.15, reveal: 2 }); put(TILES.concrete, punchedMask({ floors: 9, cols: 10, wFrac: 0.4, hFrac: 0.4 }));
  const tex = new THREE.DataArrayTexture(data, S, S, 8); tex.format = THREE.RGBAFormat; tex.type = THREE.UnsignedByteType; tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping; tex.anisotropy = 8; tex.generateMipmaps = true; tex.minFilter = THREE.LinearMipmapLinearFilter; tex.magFilter = THREE.LinearFilter; tex.needsUpdate = true;
  return { texture: tex, TILES };
}
