// The ship's senses → env.obs (body frame): soft beams that see hazards, terrain and tunnel roofs, the corridor's edges
// in the hit distances, the K most urgent hazards, and the ship's own state. Moved out of env.js unchanged.
import { wrapX, qRotate, qInvRotate } from './mathx.js';
import { ENV, N_RAYS, RAY_DIRS_BODY, OBS_DIM } from './envconst.js';
import { AERO } from './aero.js';
import { cellShape, CELL } from './weather.js';

const _rel = new Float64Array(3), _tmp = new Float64Array(3);

export function sense(env) {                                             // ray sensors + nearest asteroids + own state -> env.obs (body frame)
  const s = env.ship, o = env.obs, range = ENV.rays.range, list = env.asteroids, K = ENV.nearestK;
  for (let i = 0; i < N_RAYS; i++) {                  // world directions of the body-frame rays
    _tmp[0] = RAY_DIRS_BODY[i * 3]; _tmp[1] = RAY_DIRS_BODY[i * 3 + 1]; _tmp[2] = RAY_DIRS_BODY[i * 3 + 2];
    qRotate(s.q, _tmp, _rel); env.rayDirWorld[i * 3] = _rel[0]; env.rayDirWorld[i * 3 + 1] = _rel[1]; env.rayDirWorld[i * 3 + 2] = _rel[2];
    env.rayHit[i] = range; env.rayEdge[i] = 0;
  }
  const dist = new Float64Array(list.length), ttc = new Float64Array(list.length), cone = ENV.rays.cone, cosEdgeMin = Math.cos(cone + Math.PI / 2);
  env.rayVal.fill(0);
  for (let j = 0; j < list.length; j++) {
    const a = list[j], dx = wrapX(a.p[0] - s.p[0], ENV.xHalf), dy = a.p[1] - s.p[1], dz = a.p[2] - s.p[2];
    const c2 = dx * dx + dy * dy + dz * dz, cd = Math.sqrt(c2);
    dist[j] = cd - a.r - ENV.ship.radius;
    const closing = -((a.v[0] - s.v[0]) * dx + (a.v[1] - s.v[1]) * dy + (a.v[2] - s.v[2]) * dz) / Math.max(1e-6, cd);   // approach speed along the line of sight
    ttc[j] = closing > 0 ? Math.max(0, dist[j]) / Math.max(1, closing) : Math.max(0, dist[j]) + 20;                   // receding rocks rank last
    if (cd - a.r > range) continue;                   // beyond sensor range
    if (dx * s.f[0] + dy * s.f[1] + dz * s.f[2] < -a.r) continue;   // entirely behind the ship
    // soft beams: a rock contributes in proportion to how much of it lies inside the cone (continuous in angle), so the
    // observation — and therefore the policy's output — changes smoothly as rocks drift across beam boundaries
    const alpha = Math.asin(Math.min(1, a.r / cd)), surf = Math.max(0, cd - a.r), edge = cone + alpha;
    for (let i = 0; i < N_RAYS; i++) {
      const d = env.rayDirWorld, cosT = (dx * d[i * 3] + dy * d[i * 3 + 1] + dz * d[i * 3 + 2]) / cd;
      if (cosT < cosEdgeMin) continue;
      const theta = Math.acos(Math.min(1, cosT)); if (theta >= edge) continue;
      const w = Math.min(1, (edge - theta) / cone), v = w * (1 - surf / range);          // 0 = clear … 1 = touching
      if (v > env.rayVal[i]) { env.rayVal[i] = v; env.rayHit[i] = Math.min(env.rayHit[i], surf); }
    }
  }
  if (env.hf) {                                     // terrain: march each beam that can reach the ground
    const hf = env.hf, step = ENV.mountains.marchStep, above = s.p[1] - ENV.ship.radius;
    for (let i = 0; i < N_RAYS; i++) {
      const dx = env.rayDirWorld[i * 3], dy = env.rayDirWorld[i * 3 + 1], dz = env.rayDirWorld[i * 3 + 2];
      if (dy >= 0 && above > hf.peak) continue;                                                // a level or climbing beam above the highest peak sees no ground
      for (let t = step; t < range; t += step) {
        const px = s.p[0] + dx * t, py = s.p[1] + dy * t, pz = s.p[2] + dz * t;
        if (py < hf.height(px, pz) || (hf.ceiling && py > hf.ceiling(px, pz))) { const v = 1 - (t - step * 0.5) / range; if (v > env.rayVal[i]) env.rayVal[i] = v; if (t < env.rayHit[i]) env.rayHit[i] = t; break; }   // ground, rock, or a tunnel's roof
      }
    }
  }
  {                                                  // the corridor's edges are surfaces the beams see — the side walls, the ceiling (thin air over the mountains and cities), in space the floor — in
                                                     // their hit distances (the sensor display, the edge guard); the policy's beam values stay terrain and hazards (it knows the edges from its position inputs)
    const top = env.hf ? ENV.mountains.ceiling : ENV.yHalf, zw = ENV.zHalf, yw = ENV.yHalf;
    for (let i = 0; i < N_RAYS; i++) {
      const dy = env.rayDirWorld[i * 3 + 1], dz = env.rayDirWorld[i * 3 + 2]; let t = range;
      if (dz > 1e-4) t = Math.min(t, (zw - s.p[2]) / dz); else if (dz < -1e-4) t = Math.min(t, (-zw - s.p[2]) / dz);
      if (dy > 1e-4) t = Math.min(t, (top - s.p[1]) / dy); else if (!env.hf && dy < -1e-4) t = Math.min(t, (-yw - s.p[1]) / dy);
      if (t < range) { t = Math.max(0, t - ENV.ship.radius); if (t < env.rayHit[i]) { env.rayHit[i] = t; env.rayEdge[i] = 1; } }
    }
  }
  for (let i = 0; i < N_RAYS; i++) o[i] = env.rayVal[i];
  // the K most urgent asteroids by time-to-contact
  const idx = Array.from(list.keys()).sort((x, y) => ttc[x] - ttc[y]).slice(0, K);
  env.nearest = idx;
  let k = N_RAYS;
  for (let n = 0; n < K; n++) {
    if (n < idx.length) {
      const a = list[idx[n]];
      _rel[0] = wrapX(a.p[0] - s.p[0], ENV.xHalf); _rel[1] = a.p[1] - s.p[1]; _rel[2] = a.p[2] - s.p[2];
      qInvRotate(s.q, _rel, _tmp); o[k++] = _tmp[0] / range; o[k++] = _tmp[1] / range; o[k++] = _tmp[2] / range;
      _rel[0] = a.v[0] - s.v[0]; _rel[1] = a.v[1] - s.v[1]; _rel[2] = a.v[2] - s.v[2];
      qInvRotate(s.q, _rel, _tmp); o[k++] = _tmp[0] / 20; o[k++] = _tmp[1] / 20; o[k++] = _tmp[2] / 20;
      o[k++] = a.r / 5;
    } else { o[k++] = 1; o[k++] = 0; o[k++] = 0; o[k++] = 0; o[k++] = 0; o[k++] = 0; o[k++] = 0; }   // "nothing, far ahead"
  }
  qInvRotate(s.q, s.v, _tmp); o[k++] = _tmp[0] / ENV.ship.maxSpeed; o[k++] = _tmp[1] / ENV.ship.maxSpeed; o[k++] = _tmp[2] / ENV.ship.maxSpeed;
  o[k++] = s.w[0] / 2; o[k++] = s.w[1] / 2; o[k++] = s.w[2] / 2;
  o[k++] = s.f[0]; o[k++] = s.f[1]; o[k++] = s.f[2]; o[k++] = s.u[0]; o[k++] = s.u[1]; o[k++] = s.u[2];
  o[k++] = s.p[1] / ENV.yHalf; o[k++] = s.p[2] / ENV.zHalf;
  o[k++] = env.cmd[0]; o[k++] = env.cmd[1]; o[k++] = env.cmd[2]; o[k++] = env.cmd[3];
  return senseAir(env, o, k);
}

const _cells = [], _tc = new Float64Array(4); let _ex = new Float64Array(32), _ez = new Float64Array(32), _cc = new Float64Array(32), _kz = new Float64Array(32);
// after the space-era inputs (115): a weather radar value per beam — the roughest cloud along it, up to what the beam
// hits first (it does not see through rock) — then the air data. All zero in space. Like a scanning radar it refreshes a
// third of the beams per decision (the whole sky every 0.2 s) and holds the rest; a new sky (reset, new weather) is swept
// whole at once.
const SWEEP = 3;
function senseAir(env, o, k) {
  if (!env.atmosphere) { env.radarSky = null; for (; k < OBS_DIM; k++) o[k] = 0; return o; }
  const s = env.ship, range = ENV.rays.range, cells = env.weather ? env.weather.cellsNear(s.p[0], range + 10, _cells) : (_cells.length = 0, _cells);   // no sky yet: an empty radar, the air data still read
  let n = 0; for (const c of cells) { const dh = Math.max(0, Math.sqrt((s.p[0] - c.x) ** 2 + (s.p[2] - c.z) ** 2) - c.R), dv = Math.max(0, c.base - s.p[1], s.p[1] - c.top); if (dh * dh + dv * dv < range * range) cells[n++] = c; }
  cells.length = n; cells.sort((a, b) => b.sig - a.sig);      // only cells within the beams' reach; the roughest first: once one echoes, weaker ones are skipped
  if (_ex.length < n) { _ex = new Float64Array(2 * n); _ez = new Float64Array(2 * n); _cc = new Float64Array(2 * n); _kz = new Float64Array(2 * n); }
  for (let j = 0; j < n; j++) { const c = cells[j], kz = c.type === CELL.STRATUS ? 2 : 1, ex = s.p[0] - c.x, ez = (s.p[2] - c.z) * kz; _kz[j] = kz; _ex[j] = ex; _ez[j] = ez; _cc[j] = ex * ex + ez * ez - c.R * c.R; }
  const full = env.radarSky !== env.weather, ph = env.radarPhase; env.radarSky = env.weather; env.radarPhase = (ph + 1) % SWEEP;
  for (let i = 0; i < N_RAYS; i++) if (full || i % SWEEP === ph) env.radar[i] = n ? radarBeam(s.p, env.rayDirWorld[3 * i], env.rayDirWorld[3 * i + 1], env.rayDirWorld[3 * i + 2], env.rayHit[i], cells, n, range) : 0;
  for (let i = 0; i < N_RAYS; i++) o[k++] = env.radar[i];
  const ai = env.air, A = AERO; qInvRotate(s.q, ai.wind, _tmp);
  o[k++] = ai.V / A.cruise - 1; o[k++] = ai.alpha / A.alphaStall; o[k++] = ai.beta / A.betaMax; o[k++] = ai.n - 1; o[k++] = ai.climb / 5;
  o[k++] = _tmp[0] / 10; o[k++] = _tmp[1] / 10; o[k++] = _tmp[2] / 10; o[k++] = ai.sigmaFelt; o[k++] = ai.rho;
  return o;
}
// one beam: only where it passes through a cell's slab (between base and top) and footprint (an ellipse in x/z — stratus
// sheets are half as wide in z) is the cell sampled — ten times across its width or twenty across its depth, whichever is
// finer along the beam, plus the corners of the dome's profile where the loudest echo sits: the top of the base ramp
// (h = 0.06), the start of the top fade (h = 0.85), the closest approach to the cell's axis and the edge of the core
// there. A cell (or the rest of one) that could not echo louder than what the beam already sees is skipped.
const echoAt = (c, cx, py, cz, dx, dy, dz, t, cap, range) => cellShape(c.type, c.R, c.base, c.top, cx + dx * t, py + dy * t, cz + dz * t) * cap * (1 - 0.5 * t / range);
function radarBeam(p, dx, dy, dz, tMax, cells, n, range) {
  let best = 0; const py = p[1], steep = dy > 1e-9 || dy < -1e-9;
  for (let j = 0; j < n; j++) {
    const c = cells[j], cap = c.sig / 4; let t0 = 0, t1 = tMax;
    if (steep) { const ya = (c.base - py) / dy, yb = (c.top - py) / dy; if (ya < yb) { if (ya > t0) t0 = ya; if (yb < t1) t1 = yb; } else { if (yb > t0) t0 = yb; if (ya < t1) t1 = ya; } }
    else if (py < c.base || py > c.top) continue;
    if (t1 <= t0 || cap * (1 - 0.5 * t0 / range) <= best) continue;
    const ddz = dz * _kz[j], a = dx * dx + ddz * ddz; let b = 0;
    if (a < 1e-9) { if (_cc[j] > 0) continue; }
    else { b = 2 * (dx * _ex[j] + ddz * _ez[j]); const disc = b * b - 4 * a * _cc[j]; if (disc <= 0) continue; const sq = Math.sqrt(disc), ia = 0.5 / a, ta = (-b - sq) * ia, tb = (-b + sq) * ia; if (ta > t0) t0 = ta; if (tb < t1) t1 = tb; if (t1 <= t0) continue; }
    const H = c.top - c.base, cx = p[0] - c.x, cz = p[2] - c.z;
    const dt = Math.max(0.25, Math.min(0.2 * c.R / Math.sqrt(Math.max(a, 1e-12)), 0.05 * H / Math.max(Math.abs(dy), 1e-12)));
    const m = Math.max(4, Math.ceil((t1 - t0) / dt)), st = (t1 - t0) / m;
    for (let q = 0; q <= m; q++) {                                // both ends too: the loudest point is often where the beam enters or meets rock
      const t = t0 + q * st; if (cap * (1 - 0.5 * t / range) <= best) break;
      const e = echoAt(c, cx, py, cz, dx, dy, dz, t, cap, range); if (e > best) best = e;
    }
    _tc[0] = steep ? (c.base + 0.06 * H - py) / dy : -1; _tc[1] = steep ? (c.base + 0.85 * H - py) / dy : -1; _tc[2] = -1; _tc[3] = -1;
    if (a >= 1e-9) {
      const ts = -b / (2 * a), dmin = _ex[j] * _ex[j] + _ez[j] * _ez[j] - b * b / (4 * a), h = (py + dy * ts - c.base) / H;
      const rc = 0.6 * c.R * (c.type === CELL.STRATUS ? 1 : 1 - 0.55 * h * h); _tc[2] = ts; if (rc > 0 && dmin < rc * rc) _tc[3] = ts - Math.sqrt((rc * rc - dmin) / a);
    }
    for (let q = 0; q < 4; q++) { const t = _tc[q]; if (t > t0 && t < t1) { const e = echoAt(c, cx, py, cz, dx, dy, dz, t, cap, range); if (e > best) best = e; } }
  }
  return Math.min(1, best);
}
