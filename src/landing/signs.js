// The airfield's signs, in the standard colours (FAA AC 150/5340-18, ICAO Annex 14): distance-remaining boards every
// 1000 ft along the right side of the runway (white numerals on black — the thousands of feet left, each way), exit
// direction signs before the taxiways (black on yellow, the arrow toward the turn-off) and mandatory hold-position signs
// at every taxiway's hold-short bar (white "26-08" on red, the left runway end first) — lit from inside at night.
import * as THREE from 'three';

function face(text, fg, bg, w, h) {
  const c = document.createElement('canvas'); c.width = w; c.height = h; const g = c.getContext('2d');
  g.fillStyle = bg; g.fillRect(0, 0, w, h); g.fillStyle = fg; g.font = `700 ${Math.round(h * 0.7)}px "DIN Alternate", "Arial Narrow", Arial, sans-serif`;
  g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(text, w / 2, h * 0.55);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4; return t;
}

export function createAirfieldSigns(AP, { night = 0 } = {}) {
  const group = new THREE.Group(), L = AP.runway.length, HW = AP.runway.width / 2, KFT = 304.8;
  const black = new THREE.MeshStandardMaterial({ color: 0x141414, roughness: 0.8 }), post = new THREE.MeshStandardMaterial({ color: 0x9a9da0, roughness: 0.5, metalness: 0.5 });
  const lit = (tex) => new THREE.MeshStandardMaterial({ map: tex, roughness: 0.6, emissive: 0xffffff, emissiveMap: tex, emissiveIntensity: 0.12 + 0.9 * night });
  let count = 0;
  function sign(x, z, yaw, w, h, front, back = null) {       // a panel on two frangible posts; yaw 0: the front faces −x (the approach)
    const g = new THREE.Group(), y = 0.45 + h / 2;
    const m = new THREE.Mesh(new THREE.BoxGeometry(0.22, h, w), [back ? lit(back) : black, lit(front), black, black, black, black]);   // faces +x, −x, ±y, ±z
    m.position.y = y; g.add(m);
    for (const s of [-1, 1]) { const p = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, y), post); p.position.set(0, y / 2, s * w * 0.35); g.add(p); }
    g.position.set(x, 0, z); g.rotation.y = yaw; group.add(g); count++;
  }
  for (let k = 1; k * KFT < L - KFT + 1; k++) {             // distance remaining: ahead for runway 26 on the front, for 08 on the back
    const x = k * KFT; sign(x, HW + 20, 0, 1.7, 1.25, face(String(Math.floor((L - x) / KFT)), '#fff', '#111', 256, 192), face(String(k), '#fff', '#111', 256, 192));
  }
  for (const t of AP.taxiways) {
    if (t.kind !== 'link' && t.kind !== 'rapid') continue;
    const [a, b] = t.pts, d = Math.hypot(b[0] - a[0], b[1] - a[1]), u = [(b[0] - a[0]) / d, (b[1] - a[1]) / d], s = 90 / Math.abs(u[1]);
    const c = [a[0] + u[0] * s, a[1] + u[1] * s], left = [-u[1], u[0]], off = t.width / 2 + 6;   // to the left of a ship taxiing toward the runway
    sign(c[0] + left[0] * off, c[1] + left[1] * off, Math.atan2(u[1], -u[0]), 2.4, 1.0, face('26-08', '#fff', '#c8102e', 320, 128));
    if (a[0] > 0 && a[0] < L) sign(a[0] - 120, HW + 12, 0, 2.4, 1.0, face(`${t.name} ${t.kind === 'rapid' ? '↗' : '→'}`, '#111', '#f2c230', 320, 128));   // the exit ahead, on the right
  }
  return { group, count };
}
