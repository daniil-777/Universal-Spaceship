// Astro Pilot Spaceport (APX), runway 26/08 — the airport as data, laid out to ICAO Annex 14 for a 60-m precision-approach
// (CAT III) runway: paint (threshold, designation, centerline, aiming point, touchdown zone, side stripes, blast pads),
// lights (edge, centerline with the end colour coding, touchdown-zone barrettes, threshold/wing bars, end, ALSF-2-style
// approach lights with the crossbar, red side rows and sequenced flashers), the PAPI, the ILS, taxiways, apron stands and
// buildings. Runway frame: x along runway 26 from its landing threshold (0 … 4000), z to the right, y up. Pure data —
// src/landing/airportmesh.js and lights.js build the scene from it, nav.js flies the ILS.
const DEG = Math.PI / 180;
export const RWY = Object.freeze({ length: 4000, width: 60, heading: 260, shoulder: 7.5, blast: 120, elevation: 0 });
const L = RWY.length, HW = RWY.width / 2;

function paint() {
  const m = [], rect = (kind, end, x0, x1, z0, z1, extra = {}) => m.push({ kind, end, x0, x1, z0, z1, color: 'white', ...extra });
  for (const end of [26, 8]) {                            // both ends: 08 is 26 turned end for end (x → L − x, z → −z)
    const put = (kind, x0, x1, z0, z1, extra) => (end === 26 ? rect(kind, end, x0, x1, z0, z1, extra) : rect(kind, end, L - x1, L - x0, -z1, -z0, extra));
    for (const side of [1, -1]) for (let i = 0; i < 8; i++) { const a = 1.8 + i * (27 - 3.6) / 7; put('threshold', 6, 36, side > 0 ? a : -a - 1.8, side > 0 ? a + 1.8 : -a); }
    put('designation', 48, 57, -4.5, 4.5, { text: end === 26 ? '26' : '08', up: end === 26 ? 1 : -1 });
    for (const side of [1, -1]) put('aiming', 400, 460, side > 0 ? 10 : -20, side > 0 ? 20 : -10);
    [[150, 3], [300, 3], [600, 2], [750, 2], [900, 1], [1050, 1]].forEach(([x, n]) => {
      for (const side of [1, -1]) for (let k = 0; k < n; k++) { const a = 10 + k * 4.5; put('tdz', x, x + 22.5, side > 0 ? a : -a - 3, side > 0 ? a + 3 : -a); }
    });
    for (let k = 0; k < 6; k++) put('chevron', -RWY.blast + 12 + k * 18, -RWY.blast + 12 + k * 18 + 6, -HW, HW, { color: 'yellow' });   // blast pad
  }
  for (let x0 = 63; x0 + 30 <= L - 63; x0 += 50) rect('centerline', 26, x0, x0 + 30, -0.45, 0.45);
  for (const s of [1, -1]) rect('side', 26, 0, L, s > 0 ? HW - 0.9 : -HW, s > 0 ? HW : -HW + 0.9);
  return m;
}

function lamps() {
  const l = [], add = (kind, x, z, color, extra = {}) => l.push({ kind, x, y: 0.4, z, color, ...extra });
  for (let x = 0; x <= L + 1e-9; x += 60) for (const s of [1, -1]) add('edge', x, s * (HW + 1.5), x >= L - 600 ? 'yellow' : 'white');
  for (let x = 15, i = 0; x < L; x += 15, i++) add('centerline', x, 0.6, x < L - 900 ? 'white' : x > L - 300 ? 'red' : i % 2 ? 'red' : 'white');
  for (let x = 60; x <= 900 + 1e-9; x += 30) for (const s of [1, -1]) for (const z of [9, 10.5, 12]) add('tdz', x, s * z, 'white');
  for (let z = -HW; z <= HW + 1e-9; z += 3) { add('threshold', -1.5, z, 'green'); add('end', L + 1.5, z, 'red'); }
  for (const s of [1, -1]) for (let k = 1; k <= 5; k++) add('wingbar', -1.5, s * (HW + 1.5 + 2 * k), 'green');
  for (let x = -30; x >= -900 - 1e-9; x -= 30) for (let z = -2; z <= 2; z++) add('approach', x, z, 'white');
  for (let z = 3; z <= 15 + 1e-9; z += 1.5) for (const s of [1, -1]) add('crossbar', -300, s * z, 'white');
  for (let x = -30; x >= -270 - 1e-9; x -= 30) for (const s of [1, -1]) for (const z of [9, 10.5, 12]) add('siderow', x, s * z, 'red');
  for (let x = -900, k = 0; x <= -330 + 1e-9; x += 30, k++) add('flasher', x, 0, 'white', { seq: k });   // the rabbit: runs toward the runway twice a second
  return l;
}

function airside() {
  const tw = [{ kind: 'parallel', name: 'A', pts: [[-40, 200], [L + 40, 200]], width: 23 }];
  for (const [name, x] of [['A1', 0], ['A4', 1500], ['A7', L]]) tw.push({ kind: 'link', name, pts: [[x, 0], [x, 200]], width: 23 });
  for (const [name, x] of [['A5', 2100], ['A6', 2700]]) tw.push({ kind: 'rapid', name, pts: [[x, 0], [x + 200 / Math.tan(30 * DEG), 200]], width: 23 });
  tw.push({ kind: 'apron', name: 'L', pts: [[1500, 200], [1500, 230]], width: 30 });
  const stands = Array.from({ length: 8 }, (_, k) => ({ id: String(k + 1).padStart(2, '0'), x: 1290 + k * 60, z: 452, heading: 90 }));   // nose-in toward the terminal
  const buildings = [
    { kind: 'terminal', x: 1500, z: 525, w: 520, d: 70, h: 22 },
    { kind: 'tower', x: 2060, z: 470, w: 12, d: 12, h: 62 },
    { kind: 'hangar', x: 2350, z: 330, w: 110, d: 80, h: 28 }, { kind: 'hangar', x: 2480, z: 330, w: 110, d: 80, h: 28 },
    { kind: 'fire', x: 2060, z: 280, w: 60, d: 30, h: 10 }, { kind: 'radar', x: 2900, z: 560, w: 10, d: 10, h: 18 },
    { kind: 'tanks', x: 2900, z: 700, w: 80, d: 60, h: 14 }, { kind: 'cargo', x: 900, z: 470, w: 160, d: 90, h: 16 },
  ];
  return { taxiways: tw, apron: { x0: 1200, x1: 1800, z0: 215, z1: 480 }, stands, buildings };
}

const DOT_LOC = Math.atan(105 / (L + 300)) / 2;           // full scale (2 dots) = 105 m either side at the threshold
export const AIRPORT = Object.freeze({
  runway: RWY, markings: paint(), lights: lamps(),
  papi: [3.5, 3 + 10 / 60, 2 + 50 / 60, 2.5].map((a, i) => ({ x: 300, y: 1.0, z: -(HW + 15 + 9 * i), angle: a * DEG })),   // inner unit highest
  ils: { loc: { x: L + 300, z: 0, dot: DOT_LOC, freq: '110.30', ident: 'IAPX' }, gs: { x: 300, z: 120, angle: 3, dot: 0.35 * DEG } },
  windsock: { x: 350, z: -110, h: 10 },
  ...airside(),
});
