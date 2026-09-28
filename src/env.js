// SpaceEnv — the flight environment used both for training (64 copies) and for the displayed ship.
// Pure JS, no rendering. Units: 1 = one "metre"-ish, seconds. Right-handed, y up, x = east, z toward the camera.
// The corridor is periodic in x (period 2·xHalf) and soft-walled in y and z.
// Ship: the policy commands body rates (roll, yaw, pitch) and thrust at 15 Hz; a fly-by-wire controller tracks the rates
// and pulls the velocity toward the cruise speed along the nose (aircraft-like feel, smooth by construction). Asteroids: power-law sizes, mean westward drift with Keplerian-like shear, an
// Ornstein–Uhlenbeck random wander, free spin and elastic sphere collisions (see docs/superpowers/specs/…-design.md §4).
// Comets: fast bodies on straight lines that enter through a corridor face, cross, and respawn elsewhere; they live in the
// same `asteroids` list (kind = 1) so sensors, collisions and rewards treat them like any other hazard.
// Satellites (kind = 2): slow small bodies that appear during low passes over the Earth; same treatment.
// Airliners (kind = 3): the hazards of atmospheric flight — level flight, head-on, crossing or overtaken.
// Mountains (optional): a procedural height field under the corridor; the beams see it, touching it is a crash.
import { mulberry32, randn, wrapX, clamp, qIdentity, qRotate, qInvRotate, qIntegrate, qRandom, qAxes, qFromAxisAngle, qMul, qNormalize, randomUnit } from './mathx.js';
import { createHeightField } from './heightfield.js';
import { sense } from './sensors.js';
import { edgeGuard, tunnelGuide } from './guards.js';
import { createCityField, createLongField, hasLongGrid } from './cityfield.js';

import { ENV, ACT_DIM, N_RAYS, OBS_DIM, OBS_BASE, N_AIR, RAY_DIRS_BODY } from './envconst.js';
export { ENV, ACT_DIM, N_RAYS, OBS_DIM, OBS_BASE, N_AIR, RAY_DIRS_BODY };
import { stepAir, newAirState, trimAttitude, autoThrottle, rateToStick, AERO } from './aero.js';
import { Dryden } from './turbulence.js';
import { createWeather } from './weather.js';

const _rel = new Float64Array(3), _tmp = new Float64Array(3), _f = new Float64Array(3), _u = new Float64Array(3), _r = new Float64Array(3), _q = new Float64Array(4);

export class SpaceEnv {
  constructor(seed = 1, opts = {}) {
    this.seed = seed; this.rng = mulberry32(seed);
    this.level = opts.level ?? 1;                       // curriculum 0..1: asteroid count, drift speed and max size
    this.countOverride = opts.count ?? null;            // playbox: fixed asteroid count
    this.speedScale = opts.speedScale ?? 1;             // playbox: asteroid speed multiplier
    this.cometsOverride = opts.comets ?? null;          // playbox: fixed comet count (else nMax·level)
    this.cometSpeed = opts.cometSpeed ?? 1;             // playbox: comet speed multiplier
    this.satellitesCount = opts.satellites ?? 0;        // low passes: satellites in orbit (0 = none)
    this.planesCount = opts.planes ?? 0;                // atmospheric flight: airliners (0 = none)
    this.autoPlanes = opts.autoPlanes ?? false;         // training: planes = 2 + 6·level
    this.birdsCount = opts.birds ?? 0;                  // atmospheric flight: bird flocks (0 = none)
    this.autoBirds = opts.autoBirds ?? false;           // training: flocks = 1 + 4·level in any atmospheric world
    this.hf = null; this.world = 'space'; if (opts.mountains) this.setMountains(true, opts.mountainSeed); if (opts.pillars) this.setPillars(true, opts.pillarSeed); if (opts.meshy) this.setMeshy(true, typeof opts.meshy === 'string' ? opts.meshy : null); if (opts.city) this.setCity(opts.city, opts.citySeed);
    this.obs = new Float32Array(OBS_DIM);
    this.rayHit = new Float32Array(N_RAYS);             // nearest surface distance per beam (range when clear), for the sensor display
    this.rayVal = new Float32Array(N_RAYS);             // beam value 0..1 (soft overlap × proximity) = the observation
    this.guard = 0; this.guardCmd = new Float64Array(3);  // the edge guard's blend and its commands (pitch, yaw, roll)
    this.guardOn = opts.edgeGuard ?? true;               // evaluation can switch it off (a baseline)
    this.guideOn = opts.tunnelGuide ?? true; this.guide = 0; this.guideCmd = new Float64Array(4);   // the tunnel guide (off when a person flies)
    this.rayEdge = new Uint8Array(N_RAYS);              // 1 where a beam's nearest hit is a corridor edge
    this.rayDirWorld = new Float64Array(N_RAYS * 3);
    this.radar = new Float32Array(N_RAYS); this.radarPhase = 0; this.radarSky = null;   // the weather radar's last sweep per beam (src/sensors.js)
    this.ship = { p: new Float64Array(3), v: new Float64Array(3), q: qIdentity(), w: new Float64Array(3), f: new Float64Array(3), u: new Float64Array(3), r: new Float64Array(3) };
    this.prevA = new Float32Array(ACT_DIM);                // previous raw (tanh'd) action, for the jerk penalty
    this.cmd = new Float32Array(ACT_DIM);                  // low-pass filtered command actually applied (what the observation reports)
    this.asteroids = [];
    this.nearest = [];                                  // indices of the K nearest asteroids after the last step
    this.atmosphere = opts.atmosphere ?? this.world !== 'space';   // the air (lift, drag, gravity, wind, clouds): never in space — the belt, Earth orbit and the low passes keep the rate model
    this.weatherSeverity = opts.weather ?? 0; this.weatherWind = 1; this.weatherCover = 1; this.weatherTurb = 1; this.weather = null; this.air = newAirState();
    this.dryden = new Dryden(mulberry32(seed ^ 0x5bd1e995)); this.rngAir = mulberry32(seed ^ 0x27d4eb2f);   // own streams: space never draws from them
    this.rngSky = mulberry32(seed ^ 0x165667b1); this.skySeed = 1;   // one sky per episode (and, in training, its severity): drawn once per reset, whatever else the air drew
    this.autoThrottle = false; this.speedTarget = AERO.cruise; this.crashCause = '';
    this.reset();
  }

  setLevel(l) { this.level = clamp(l, 0, 1); }
  setCount(n) { this.countOverride = n; }
  setSpeedScale(s) { this.speedScale = s; }
  setComets(n) { this.cometsOverride = n; }
  setCometSpeed(s) { this.cometSpeed = s; }
  get cometCount() { return this.cometsOverride ?? Math.round(ENV.comets.nMax * this.level); }
  get rockCount() { return this.asteroids.reduce((n, a) => n + (a.kind ? 0 : 1), 0); }
  setSatellites(n) {                                    // add or remove satellites immediately (kept away from the ship)
    this.satellitesCount = n;
    const have = this.asteroids.filter((a) => a.kind === 2);
    for (let i = have.length; i < n; i++) { const sat = { p: new Float64Array(3), v: new Float64Array(3), q: qRandom(this.rng), axis: randomUnit(this.rng), spin: 0.15, r: 1, m: 0.5, drift: 0, shape: 0, kind: 2, gen: 0 }; this.spawnSatellite(sat); this.asteroids.push(sat); }
    if (have.length > n) { const drop = new Set(have.slice(n)); this.asteroids = this.asteroids.filter((a) => !drop.has(a)); }
  }
  setMountains(on, seed = 5) { this.hf = on ? createHeightField({ seed, amplitude: ENV.mountains.amplitude, y0: ENV.mountains.y0 }) : null; this.world = on ? 'mountains' : 'space'; }
  setPillars(on, seed = 5) { this.hf = on ? createHeightField({ seed, amplitude: ENV.mountains.amplitude, y0: ENV.mountains.y0, style: 'pillars' }) : null; this.world = on ? 'pillars' : 'space'; }   // Zhangjiajie: sandstone pillars in the flight band
  setMeshy(on, set = null) {                          // the Meshy pillar clusters (baked grid); set 'avatar': the long Avatar valley, once its grid is registered
    this.hf = !on ? null : set && hasLongGrid(set) ? createLongField(set, ENV.mountains.y0) : createHeightField({ amplitude: ENV.mountains.amplitude, y0: ENV.mountains.y0, style: 'meshy' }); this.world = on ? 'meshy' : 'space'; }
  setCity(city, seed = 1) { this.hf = city ? createCityField({ city, seed, y0: ENV.mountains.y0 }) : null; this.world = city || 'space'; }   // skyline flight: the city's towers replace the range
  setAtmosphere(on) { this.atmosphere = on; this.air.lagOn = false; if (!on) this.weather = null; else if (!this.weather) this.buildWeather(); }   // into the air mid-flight: the sky is there at once
  setWeather(severity, { wind = this.weatherWind, cover = this.weatherCover, turb = this.weatherTurb } = {}) { this.weatherSeverity = severity; this.weatherWind = wind; this.weatherCover = cover; this.weatherTurb = turb; if (this.atmosphere) this.buildWeather(this.weather); }   // live from the Playbox: the same sky goes on
  buildWeather(carry = null) {
    this.weather = createWeather({ seed: this.skySeed, carry, orographic: this.world === 'mountains' || this.world === 'pillars' || this.world === 'meshy', severity: this.weatherSeverity, wind: this.weatherWind, cover: this.weatherCover, turb: this.weatherTurb, period: this.hf && this.hf.PERIOD ? this.hf.PERIOD : 2 * ENV.xHalf,
      hf: this.hf, ceiling: ENV.mountains.ceiling, floor: this.hf ? ENV.mountains.y0 : -ENV.yHalf, zHalf: ENV.zHalf });
  }
  slideLap() { const hf = this.hf; if (hf && hf.lap) { const n = hf.segments, k = (((this.lapBase + this.laps) % n) + n) % n; hf.lap.shift = k * 2 * ENV.xHalf; } }   // a long city: each wrap moves the corridor on to the next 120 units of the map
  slideAlong(top, ly, lz) {                            // in the air an edge is slid along: the velocity loses its part into the edge but keeps its speed, and the ship turns with it (α and β unchanged — no instant stall)
    const s = this.ship, v = s.v, sp = Math.hypot(v[0], v[1], v[2]), ox = v[0] / (sp || 1), oy = v[1] / (sp || 1), oz = v[2] / (sp || 1);
    if (s.p[1] > top) { s.p[1] = top; if (v[1] > 0) v[1] = 0; } else if (s.p[1] < -ly) { s.p[1] = -ly; if (v[1] < 0) v[1] = 0; }
    if (s.p[2] > lz) { s.p[2] = lz; if (v[2] > 0) v[2] = 0; } else if (s.p[2] < -lz) { s.p[2] = -lz; if (v[2] < 0) v[2] = 0; }
    const nl = Math.hypot(v[0], v[1], v[2]); if (sp < 1e-6 || nl < 1e-3 * sp) return;
    for (let i = 0; i < 3; i++) v[i] *= sp / nl;
    const nx = v[0] / sp, ny = v[1] / sp, nz = v[2] / sp, ax = oy * nz - oz * ny, ay = oz * nx - ox * nz, az = ox * ny - oy * nx, sn = Math.hypot(ax, ay, az);
    if (sn < 1e-9) return;
    qMul(qFromAxisAngle(ax / sn, ay / sn, az / sn, Math.atan2(sn, ox * nx + oy * ny + oz * nz), _q), s.q, s.q); qNormalize(s.q); qAxes(s.q, s.f, s.u, s.r);
  }
  groundClearance() { const s = this.ship; return this.hf ? s.p[1] - this.hf.height(s.p[0], s.p[2]) - ENV.ship.radius : Infinity; }
  setPlanes(n) {
    this.planesCount = n;
    const have = this.asteroids.filter((a) => a.kind === 3);
    for (let i = have.length; i < n; i++) { const pl = { p: new Float64Array(3), v: new Float64Array(3), q: qIdentity(), axis: new Float64Array([0, 1, 0]), spin: 0, r: ENV.planes.r, m: 2, drift: 0, shape: 0, kind: 3, gen: 0 }; this.spawnPlane(pl); this.asteroids.push(pl); }
    if (have.length > n) { const drop = new Set(have.slice(n)); this.asteroids = this.asteroids.filter((a) => !drop.has(a)); }
  }
  spawnPlane(pl) {                                      // level flight: 55 % head-on from the east, 25 % crossing, 20 % same direction (overtaken)
    const P = ENV.planes, rng = this.rng, sp = this.ship.p, s = P.speedMin + rng() * (P.speedMax - P.speedMin);
    pl.gen++;
    for (let tries = 0; tries < 20; tries++) {
      pl.p[0] = wrapX(sp[0] + 30 + rng() * 70, ENV.xHalf); pl.p[1] = this.hf ? rng() * 9 : (rng() * 2 - 1) * (ENV.yHalf - 4); pl.p[2] = (rng() * 2 - 1) * (ENV.zHalf - 4);   // airliners stay in the flight band above the valleys
      const dx = wrapX(pl.p[0] - sp[0], ENV.xHalf), dy = pl.p[1] - sp[1], dz = pl.p[2] - sp[2];
      if (Math.hypot(dx, dy, dz) > 20) break;
    }
    const kind = rng();
    if (kind < 0.55) { pl.v[0] = -s; pl.v[1] = 0; pl.v[2] = (rng() * 2 - 1) * 0.6; }
    else if (kind < 0.8) { pl.v[0] = -2 - rng() * 2; pl.v[1] = 0; pl.v[2] = (rng() < 0.5 ? 1 : -1) * s * 0.8; }
    else { pl.v[0] = s * 0.75; pl.v[1] = 0; pl.v[2] = (rng() * 2 - 1) * 0.4; }
  }
  setBirds(n) {
    this.birdsCount = n;
    const have = this.asteroids.filter((a) => a.kind === 4);
    for (let i = have.length; i < n; i++) { const f = { p: new Float64Array(3), v: new Float64Array(3), q: qIdentity(), axis: new Float64Array([0, 1, 0]), spin: 0, r: 1.5, m: 0.3, drift: 0, shape: 0, kind: 4, gen: 0, type: 0, phase: 0, w: 0 }; this.spawnFlock(f); this.asteroids.push(f); }
    if (have.length > n) { const drop = new Set(have.slice(n)); this.asteroids = this.asteroids.filter((a) => !drop.has(a)); }
  }
  tunnelZone(x) {                                        // corridor x inside a tunnel's approach, bore or exit (birds keep out of the cliffs' mouths)
    const hf = this.hf; if (!hf || !hf.tunnels || !hf.tunnels.length) return false; const X = x + (hf.lap ? hf.lap.shift : 0), P = hf.PERIOD;
    for (const t of hf.tunnels) { const d0 = t.x0 - 60, d1 = t.x1 + 12, xx = X - P * Math.round((X - (d0 + d1) / 2) / P); if (xx > d0 && xx < d1) return true; }
    return false;
  }
  spawnFlock(f) {                                       // a flock wanders at a random heading somewhere ahead, above whatever stands there
    const B = ENV.birds, rng = this.rng, sp = this.ship.p, s = B.speedMin + rng() * (B.speedMax - B.speedMin);
    f.r = B.rMin + rng() * (B.rMax - B.rMin); f.type = rng() < 0.5 ? 0 : 1; f.phase = rng() * 100; f.w = 0; f.gen++;
    for (let tries = 0; tries < 20; tries++) {
      f.p[0] = wrapX(sp[0] + 25 + rng() * 70, ENV.xHalf); f.p[1] = -4 + rng() * 12; f.p[2] = (rng() * 2 - 1) * 13; if (tries < 19 && this.tunnelZone(f.p[0])) continue;
      if (this.hf) { const g = this.hf.height(f.p[0], f.p[2]) + 4; if (g > ENV.mountains.ceiling - 2) continue; f.p[1] = Math.max(f.p[1], g); }   // not over the tallest roofs and summits
      const dx = wrapX(f.p[0] - sp[0], ENV.xHalf), dy = f.p[1] - sp[1], dz = f.p[2] - sp[2]; if (Math.hypot(dx, dy, dz) > 20) break;
    }
    const ang = rng() * Math.PI * 2; f.v[0] = s * Math.cos(ang); f.v[1] = 0; f.v[2] = s * Math.sin(ang);
  }
  spawnSatellite(sat) {
    const S = ENV.satellites, rng = this.rng, sp = this.ship.p;
    sat.r = S.rMin + rng() * (S.rMax - S.rMin); sat.m = 0.5; sat.spin = 0.1 + rng() * 0.25; sat.gen++;
    for (let tries = 0; tries < 20; tries++) {
      sat.p[0] = wrapX(sp[0] + 25 + rng() * 70, ENV.xHalf); sat.p[1] = (rng() * 2 - 1) * (ENV.yHalf - 3); sat.p[2] = (rng() * 2 - 1) * (ENV.zHalf - 3);
      const dx = wrapX(sat.p[0] - sp[0], ENV.xHalf), dy = sat.p[1] - sp[1], dz = sat.p[2] - sp[2];
      if (Math.hypot(dx, dy, dz) > 16) break;
    }
    const s = S.speedMin + rng() * (S.speedMax - S.speedMin);
    sat.v[0] = -s; sat.v[1] = (rng() * 2 - 1) * 0.4; sat.v[2] = (rng() * 2 - 1) * 0.4;   // orbital drift: slow, mostly westward like the belt
  }

  get asteroidCount() { const b = ENV.belt; return this.countOverride ?? Math.round(b.nMin + (b.nMax - b.nMin) * this.level); }

  reset() {
    const s = this.ship, rng = this.rng;
    this.laps = 0; this.lapBase = this.hf && this.hf.lap ? Math.floor(rng() * this.hf.segments) : 0; this.slideLap();   // a long city: every episode starts somewhere else on the map (before the spawn looks at the ground)
    s.p[0] = -ENV.xHalf + 6; s.p[1] = (rng() * 2 - 1) * 5; s.p[2] = (rng() * 2 - 1) * 5;
    if (this.hf) {                                      // mountains: start over a valley, low in the flight band, well clear of the ground
      let best = Infinity; for (let tries = 0; tries < 12; tries++) {                       // pick the lane with the lowest ground over the first 25 units
        const z = (rng() * 2 - 1) * 8; let g = -Infinity; for (let d = 0; d <= 25; d += 5) g = Math.max(g, this.hf.height(s.p[0] + d, z));
        if (g < best) { best = g; s.p[2] = z; } }
      s.p[1] = Math.max(1 + rng() * 6, best + 5);
    }
    qFromAxisAngle(0, 1, 0, (rng() * 2 - 1) * 0.15, s.q); qMul(s.q, qFromAxisAngle(0, 0, 1, (rng() * 2 - 1) * 0.15, _q), s.q);
    qAxes(s.q, s.f, s.u, s.r);
    for (let i = 0; i < 3; i++) { s.v[i] = ENV.ship.cruise * s.f[i]; s.w[i] = 0; }
    this.prevA.fill(0); this.cmd.fill(0);
    this.air = newAirState(); this.dryden.reset(); this.crashCause = '';
    if (this.atmosphere) {                              // the air: level flight at cruise on the drawn heading (the normal law holds whatever path it starts on), then trimmed
      qFromAxisAngle(0, 1, 0, Math.atan2(-s.f[2], s.f[0]), s.q); qAxes(s.q, s.f, s.u, s.r); for (let i = 0; i < 3; i++) s.v[i] = ENV.ship.cruise * s.f[i];
      this.skySeed = Math.floor(this.rngSky() * 2 ** 31); this.buildWeather(); trimAttitude(this);
    } else this.weather = null;   // a fresh sky every episode; the nose starts at the 1-g angle of attack
    this.steps = 0; this.t = 0; this.laps = 0; this.done = false; this.collided = false; this.episodeReturn = 0; this.radarPhase = 0; this.radarSky = null;
    this.spawnAsteroids();
    this.sense();
    return this.obs;
  }

  spawnAsteroids() {
    const b = ENV.belt, rng = this.rng, n = this.asteroidCount, rMax = b.rMaxBase + b.rMaxLevel * this.level, drift = 0.5 + 0.5 * this.level;
    const list = [], sp = this.ship.p;
    for (let i = 0; i < n; i++) {
      const r = Math.min(rMax, b.rMin * Math.pow(1 - rng(), -1 / b.alpha));           // collisional-cascade size law N(>r) ∝ r^-alpha
      const a = { p: new Float64Array(3), v: new Float64Array(3), q: qRandom(rng), axis: randomUnit(rng), spin: 0, r, m: r * r * r,
        drift: (b.driftMin + rng() * (b.driftMax - b.driftMin)) * drift, shape: r > 3 ? Math.floor(rng() * 4) : 4 + Math.floor(rng() * (b.nShapes - 4)) };   // the four high-detail shapes only for big rocks (triangle budget)
      a.spin = (b.spinMin + rng() * (b.spinMax - b.spinMin)) / Math.sqrt(r);           // bigger rocks tumble more slowly
      let ok = false;
      for (let tries = 0; tries < 60 && !ok; tries++) {
        a.p[0] = (rng() * 2 - 1) * ENV.xHalf; a.p[1] = (rng() * 2 - 1) * (ENV.yHalf - r); a.p[2] = (rng() * 2 - 1) * (ENV.zHalf - r);
        const dx = wrapX(a.p[0] - sp[0], ENV.xHalf), dy = a.p[1] - sp[1], dz = a.p[2] - sp[2];
        if (Math.hypot(dx, dy, dz) < 12 + r || (dx > 0 && dx < 24 && dy * dy + dz * dz < 49)) continue;   // clear zone around and ahead of the ship
        ok = true;
        for (const o of list) { const ex = wrapX(o.p[0] - a.p[0], ENV.xHalf), ey = o.p[1] - a.p[1], ez = o.p[2] - a.p[2]; if (Math.hypot(ex, ey, ez) < o.r + r + 1) { ok = false; break; } }
      }
      if (!ok) continue;
      const vx = -a.drift + b.shear * a.p[2];
      a.v[0] = vx + randn(rng) * 0.6; a.v[1] = randn(rng) * 0.6; a.v[2] = randn(rng) * 0.6;
      list.push(a);
    }
    for (let i = 0; i < this.cometCount; i++) { const c = { p: new Float64Array(3), v: new Float64Array(3), q: qRandom(rng), axis: randomUnit(rng), spin: 0.3, r: 1, m: 1, drift: 0, shape: 0, kind: 1, gen: 0 }; this.respawnComet(c, true); list.push(c); }
    this.asteroids = list;
    if (this.satellitesCount) { const n = this.satellitesCount; this.satellitesCount = 0; this.setSatellites(n); }
    if (this.autoPlanes) this.planesCount = this.world === 'mountains' ? Math.round(2 + 6 * this.level) : 0;   // no airliners at rooftop height over a city
    if (this.planesCount) { const n = this.planesCount; this.planesCount = 0; this.setPlanes(n); }
    if (this.autoBirds) this.birdsCount = this.world !== 'space' ? Math.round(1 + 4 * this.level) : 0;
    if (this.birdsCount) { const n = this.birdsCount; this.birdsCount = 0; this.setBirds(n); }
  }

  respawnComet(c, initial = false) {                    // enter through a random corridor face on a straight line that crosses the corridor
    const C = ENV.comets, rng = this.rng, sp = this.ship.p;
    for (let tries = 0; tries < 12; tries++) {
      c.r = C.rMin + rng() * (C.rMax - C.rMin); c.m = c.r * c.r * c.r; c.shape = Math.floor(rng() * ENV.belt.nShapes);
      const face = Math.floor(rng() * 4), s = (C.speedMin + rng() * (C.speedMax - C.speedMin)) * this.cometSpeed;
      c.p[0] = initial ? (rng() * 2 - 1) * ENV.xHalf : sp[0] + 20 + rng() * 60;            // ahead of the ship (in x) when respawning mid-flight
      c.p[0] = wrapX(c.p[0], ENV.xHalf);
      const dir = new Float64Array([(rng() * 2 - 1) * 0.9 - 0.3, (rng() * 2 - 1) * 0.5, (rng() * 2 - 1) * 0.5]);   // mostly westward, always crossing
      if (face === 0) { c.p[1] = ENV.yHalf + c.r; c.p[2] = (rng() * 2 - 1) * ENV.zHalf * 0.8; dir[1] = -0.35 - rng() * 0.6; }
      else if (face === 1) { c.p[1] = -ENV.yHalf - c.r; c.p[2] = (rng() * 2 - 1) * ENV.zHalf * 0.8; dir[1] = 0.35 + rng() * 0.6; }
      else if (face === 2) { c.p[2] = ENV.zHalf + c.r; c.p[1] = (rng() * 2 - 1) * ENV.yHalf * 0.8; dir[2] = -0.35 - rng() * 0.6; }
      else { c.p[2] = -ENV.zHalf - c.r; c.p[1] = (rng() * 2 - 1) * ENV.yHalf * 0.8; dir[2] = 0.35 + rng() * 0.6; }
      const n = Math.hypot(dir[0], dir[1], dir[2]) || 1; c.v[0] = dir[0] / n * s; c.v[1] = dir[1] / n * s; c.v[2] = dir[2] / n * s;
      c.spin = 0.2 + rng() * 0.5; c.gen++;
      const dx = wrapX(c.p[0] - sp[0], ENV.xHalf), dy = c.p[1] - sp[1], dz = c.p[2] - sp[2];
      if (Math.hypot(dx, dy, dz) > 14) return;
    }
  }

  edgeGuard() { return edgeGuard(this); }
  tunnelGuide() { return tunnelGuide(this); }

  step(action) {
    const s = this.ship, S = ENV.ship, R = ENV.reward, h = ENV.dt / ENV.substeps, rng = this.rng;
    let a0 = Math.tanh(action[0]), a1 = Math.tanh(action[1]), a2 = Math.tanh(action[2]), a3 = Math.tanh(action[3]);
    const w0 = s.w[0], w1 = s.w[1], w2 = s.w[2];                // body rates before this decision (the air's smoothness term)
    const air = this.atmosphere, tg = this.guideOn ? this.tunnelGuide() : 0; if (tg > 0) { const e = this.guideCmd, e0 = air ? rateToStick(this, e[0]) : e[0], e3 = air ? 2 * autoThrottle(this, 0) - 1 : e[3]; a0 += (e0 - a0) * tg; a1 += (e[1] - a1) * tg; a2 += (e[2] - a2) * tg; a3 += (e3 - a3) * tg; }   // in the air: a load-factor stick and an auto-throttle   // lined up with a tunnel's bore and through it
    const g = this.guardOn && tg < 0.5 ? this.edgeGuard() : 0; if (g > 0) { const e = this.guardCmd, e0 = air ? rateToStick(this, e[0]) : e[0]; a0 += (e0 - a0) * g; a1 += (e[1] - a1) * g; a2 += (e[2] - a2) * g; }   // near a corridor edge the fly-by-wire turns away
    const c = this.cmd, kAct = 1 - Math.exp(-h / S.actuatorTau);
    let progress = 0, collided = false;
    const kRate = h / S.rateTau;
    for (let k = 0; k < ENV.substeps; k++) {
      c[0] += (a0 - c[0]) * kAct; c[1] += (a1 - c[1]) * kAct; c[2] += (a2 - c[2]) * kAct; c[3] += (a3 - c[3]) * kAct;   // actuator lag
      if (air) stepAir(this, h, k);               // the air: forces, fly-by-wire, wind and gusts (src/aero.js)
      else {
        const thrust = S.thrust * c[3];
        s.w[0] += kRate * (c[2] * S.rateMax[0] - s.w[0]); s.w[1] += kRate * (c[1] * S.rateMax[1] - s.w[1]); s.w[2] += kRate * (c[0] * S.rateMax[2] - s.w[2]);   // rate controller; body rates: x = roll, y = yaw, z = pitch
        qIntegrate(s.q, s.w, h);
        qAxes(s.q, s.f, s.u, s.r);
        for (let i = 0; i < 3; i++) s.v[i] += h * (s.f[i] * thrust + S.assist * (S.cruise * s.f[i] - s.v[i]));
        const sp = Math.hypot(s.v[0], s.v[1], s.v[2]);
        if (sp > S.maxSpeed) for (let i = 0; i < 3; i++) s.v[i] *= S.maxSpeed / sp;
      }
      s.p[0] += h * s.v[0]; s.p[1] += h * s.v[1]; s.p[2] += h * s.v[2];
      progress += h * s.v[0];
      if (s.p[0] >= ENV.xHalf) { s.p[0] -= 2 * ENV.xHalf; this.laps++; this.slideLap(); } else if (s.p[0] < -ENV.xHalf) { s.p[0] += 2 * ENV.xHalf; this.laps--; this.slideLap(); }
      const ly = ENV.yHalf + S.wallClamp, lz = ENV.zHalf + S.wallClamp, top = this.hf ? ENV.mountains.ceiling + S.wallClamp : ly;
      if (air) { if (s.p[1] > top || s.p[1] < -ly || s.p[2] > lz || s.p[2] < -lz) this.slideAlong(top, ly, lz); }
      else {
        if (s.p[1] > top) { s.p[1] = top; if (s.v[1] > 0) s.v[1] = 0; } else if (s.p[1] < -ly) { s.p[1] = -ly; if (s.v[1] < 0) s.v[1] = 0; }
        if (s.p[2] > lz) { s.p[2] = lz; if (s.v[2] > 0) s.v[2] = 0; } else if (s.p[2] < -lz) { s.p[2] = -lz; if (s.v[2] < 0) s.v[2] = 0; }
      }
      if (air && this.weather) this.weather.advance(h);
      this.stepAsteroids(h, rng);
      for (const a of this.asteroids) {                 // collision with the ship, checked every physics substep
        const dx = wrapX(a.p[0] - s.p[0], ENV.xHalf), dy = a.p[1] - s.p[1], dz = a.p[2] - s.p[2];
        if (dx * dx + dy * dy + dz * dz < (a.r + S.radius) * (a.r + S.radius)) { collided = true; break; }
      }
      if (!collided && this.hf) {                       // flew into terrain or a wall: the column under the centre and the sphere's four flanks
        const hf = this.hf, r = S.radius, e = 0.7 * r, x = s.p[0], y = s.p[1], z = s.p[2];
        if (y - r < hf.height(x, z) || y - e < Math.max(hf.height(x + e, z), hf.height(x - e, z), hf.height(x, z + e), hf.height(x, z - e))) collided = true;
        else if (hf.ceiling && (y + r > hf.ceiling(x, z) || y + e > Math.min(hf.ceiling(x + e, z), hf.ceiling(x - e, z), hf.ceiling(x, z + e), hf.ceiling(x, z - e)))) collided = true;   // a tunnel's roof
      }
      if (!collided && air && this.air.overstressed) { collided = true; this.crashCause = 'overstress'; }   // a gust at speed broke the airframe
      if (collided) break;
    }
    this.collideAsteroids();
    this.steps++; this.t += ENV.dt;
    this.sense();

    // --- reward
    let reward = R.progress * clamp(s.v[0] / S.cruise, -1, 1) + R.heading * s.f[0] + R.level * s.u[1];   // progress is capped at cruise speed: no reward for racing
    reward -= R.rate * (s.w[0] * s.w[0] + s.w[1] * s.w[1] + s.w[2] * s.w[2]);
    const p = this.prevA;
    reward -= R.jerk * ((a0 - p[0]) ** 2 + (a1 - p[1]) ** 2 + (a2 - p[2]) ** 2 + (a3 - p[3]) ** 2);
    reward -= R.effort * (a0 * a0 + a1 * a1 + a2 * a2 + 0.5 * a3 * a3);
    if (!air) reward -= R.speed * Math.max(0, (Math.hypot(s.v[0], s.v[1], s.v[2]) - S.cruise) / (S.maxSpeed - S.cruise));
    else { const Ra = R.air, ai = this.air, dev2 = ai.dev2 / ENV.substeps; ai.dev2 = 0;
      reward -= Ra.stall * (ai.stalled ? ENV.dt : 0) + Ra.comfort * dev2 + Ra.rough * ai.sigmaFelt * ai.sigmaFelt + Ra.power * ai.throttle + Ra.overspeed * Math.max(0, ai.V - AERO.vne) / 2;
      reward -= Ra.jerk * ((a0 - p[0]) ** 2 + (a1 - p[1]) ** 2 + (a2 - p[2]) ** 2 + (a3 - p[3]) ** 2) + Ra.angAcc * ((s.w[0] - w0) ** 2 + (s.w[1] - w1) ** 2 + (s.w[2] - w2) ** 2) / (ENV.dt * ENV.dt); }   // smooth flying: gentle commands, no rate kicks   // speed limit: boosting costs reward
    let prox = 0;
    for (const i of this.nearest) { const d = this.surfaceDist(this.asteroids[i]); if (d < R.dSafe) { const e = 1 - d / R.dSafe; prox += e * e; } }
    if (this.hf) { const g = this.groundClearance(); if (g < R.dSafe) { const e = 1 - Math.max(0, g) / R.dSafe; prox += e * e; } }
    reward -= R.proximity * prox;
    const top = this.hf ? ENV.mountains.ceiling : ENV.yHalf;                                   // thin air: the ceiling is lower over the mountains
    const wy = Math.max(0, s.p[1] > 0 ? s.p[1] - (top - R.wallMargin) : -s.p[1] - (ENV.yHalf - R.wallMargin)) / R.wallMargin, wz = Math.max(0, Math.abs(s.p[2]) - (ENV.zHalf - R.wallMargin)) / R.wallMargin;
    reward -= R.wall * (wy * wy + wz * wz);
    if (collided) reward -= R.collision;
    p[0] = a0; p[1] = a1; p[2] = a2; p[3] = a3;
    this.collided = collided; this.done = collided; this.truncated = !collided && this.steps >= ENV.maxSteps;
    this.episodeReturn += reward;
    this.lastAction = p;
    return { reward, done: collided, truncated: this.truncated, progress };
  }

  stepAsteroids(h, rng) {
    const b = ENV.belt, sq = Math.sqrt(h) * b.ouSigma, ss = this.speedScale;
    for (const a of this.asteroids) {
      if (a.kind === 3) {                               // airliner: level flight, gentle turns back at the walls (kept level), wraps in x
        a.p[0] += h * a.v[0]; a.p[1] += h * a.v[1]; a.p[2] += h * a.v[2];
        if (a.p[0] >= ENV.xHalf) a.p[0] -= 2 * ENV.xHalf; else if (a.p[0] < -ENV.xHalf) a.p[0] += 2 * ENV.xHalf;
        const lz = ENV.zHalf - 2, ly = ENV.yHalf - 3;
        if (a.p[2] > lz) { a.p[2] = lz; a.v[2] = -Math.abs(a.v[2]); } else if (a.p[2] < -lz) { a.p[2] = -lz; a.v[2] = Math.abs(a.v[2]); }
        if (a.p[1] > ly) a.p[1] = ly; else if (a.p[1] < -ly) a.p[1] = -ly;
        continue;
      }
      if (a.kind === 2) {                               // satellite: slow straight drift, reflects at the walls, spins slowly
        a.p[0] += h * a.v[0]; a.p[1] += h * a.v[1]; a.p[2] += h * a.v[2];
        if (a.p[0] >= ENV.xHalf) a.p[0] -= 2 * ENV.xHalf; else if (a.p[0] < -ENV.xHalf) a.p[0] += 2 * ENV.xHalf;
        const ly = ENV.yHalf - 1, lz = ENV.zHalf - 1;
        if (a.p[1] > ly) { a.p[1] = ly; a.v[1] = -Math.abs(a.v[1]); } else if (a.p[1] < -ly) { a.p[1] = -ly; a.v[1] = Math.abs(a.v[1]); }
        if (a.p[2] > lz) { a.p[2] = lz; a.v[2] = -Math.abs(a.v[2]); } else if (a.p[2] < -lz) { a.p[2] = -lz; a.v[2] = Math.abs(a.v[2]); }
        _tmp[0] = a.axis[0] * a.spin; _tmp[1] = a.axis[1] * a.spin; _tmp[2] = a.axis[2] * a.spin; qIntegrate(a.q, _tmp, h);
        continue;
      }
      if (a.kind === 4) {                               // bird flock: wandering, mostly level flight above whatever is below; turns back at the corridor sides
        const B = ENV.birds; a.w += h * (-B.turnTheta * a.w) + Math.sqrt(h) * B.turnSigma * randn(rng);
        const sp = Math.hypot(a.v[0], a.v[2]) || 1, ang = Math.atan2(a.v[2], a.v[0]) + a.w * h; a.v[0] = sp * Math.cos(ang); a.v[2] = sp * Math.sin(ang);
        a.v[1] += h * (-0.6 * a.v[1]) + Math.sqrt(h) * 0.35 * randn(rng);
        a.p[0] += h * a.v[0]; a.p[1] += h * a.v[1]; a.p[2] += h * a.v[2];
        if (a.p[0] >= ENV.xHalf) a.p[0] -= 2 * ENV.xHalf; else if (a.p[0] < -ENV.xHalf) a.p[0] += 2 * ENV.xHalf;
        const lz = ENV.zHalf - 2, top = (this.hf ? ENV.mountains.ceiling : ENV.yHalf) - 1.5;
        if (a.p[2] > lz) { a.p[2] = lz; a.v[2] = -Math.abs(a.v[2]); } else if (a.p[2] < -lz) { a.p[2] = -lz; a.v[2] = Math.abs(a.v[2]); }
        if (a.p[1] > top) { a.p[1] = top; a.v[1] = -Math.abs(a.v[1]); } else if (a.p[1] < -6) { a.p[1] = -6; a.v[1] = Math.abs(a.v[1]); }
        if (this.hf) { const g = this.hf.height(a.p[0], a.p[2]) + 3; if (a.p[1] < g) { a.p[1] = g; a.v[1] = Math.abs(a.v[1]) + 0.3; } if (this.tunnelZone(a.p[0])) this.spawnFlock(a); }
        continue;
      }
      if (a.kind) {                                     // comet: straight line, wraps in x, respawns after leaving the corridor
        a.p[0] += h * a.v[0]; a.p[1] += h * a.v[1]; a.p[2] += h * a.v[2];
        if (a.p[0] >= ENV.xHalf) a.p[0] -= 2 * ENV.xHalf; else if (a.p[0] < -ENV.xHalf) a.p[0] += 2 * ENV.xHalf;
        if (Math.abs(a.p[1]) > ENV.yHalf + a.r + 0.5 || Math.abs(a.p[2]) > ENV.zHalf + a.r + 0.5) this.respawnComet(a);
        _tmp[0] = a.axis[0] * a.spin; _tmp[1] = a.axis[1] * a.spin; _tmp[2] = a.axis[2] * a.spin; qIntegrate(a.q, _tmp, h);
        continue;
      }
      const vbx = ss * (-a.drift + b.shear * a.p[2]);
      a.v[0] += h * b.ouTheta * (vbx - a.v[0]) + sq * randn(rng);      // Ornstein–Uhlenbeck wander around the mean drift
      a.v[1] += h * b.ouTheta * (0 - a.v[1]) + sq * randn(rng);
      a.v[2] += h * b.ouTheta * (0 - a.v[2]) + sq * randn(rng);
      a.p[0] += h * a.v[0]; a.p[1] += h * a.v[1]; a.p[2] += h * a.v[2];
      if (a.p[0] >= ENV.xHalf) a.p[0] -= 2 * ENV.xHalf; else if (a.p[0] < -ENV.xHalf) a.p[0] += 2 * ENV.xHalf;
      const ly = ENV.yHalf, lz = ENV.zHalf;                                                // centres reflect at the walls, so rocks fill the whole corridor (no safe lane)
      if (a.p[1] > ly) { a.p[1] = ly; a.v[1] = -Math.abs(a.v[1]); } else if (a.p[1] < -ly) { a.p[1] = -ly; a.v[1] = Math.abs(a.v[1]); }
      if (a.p[2] > lz) { a.p[2] = lz; a.v[2] = -Math.abs(a.v[2]); } else if (a.p[2] < -lz) { a.p[2] = -lz; a.v[2] = Math.abs(a.v[2]); }
      _tmp[0] = a.axis[0] * a.spin; _tmp[1] = a.axis[1] * a.spin; _tmp[2] = a.axis[2] * a.spin;   // free rotation about a fixed axis
      qIntegrate(a.q, _tmp, h);
    }
  }

  collideAsteroids() {                                  // elastic sphere–sphere collisions, mass ∝ r³, with positional separation
    const list = this.asteroids, e = ENV.belt.restitution;
    for (let i = 0; i < list.length; i++) {
      const A = list[i]; if (A.kind) continue;
      for (let j = i + 1; j < list.length; j++) {
        const B = list[j], rs = A.r + B.r; if (B.kind) continue;
        const dx = wrapX(B.p[0] - A.p[0], ENV.xHalf), dy = B.p[1] - A.p[1], dz = B.p[2] - A.p[2];
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 >= rs * rs || d2 < 1e-9) continue;
        const d = Math.sqrt(d2), nx = dx / d, ny = dy / d, nz = dz / d;
        const vn = (B.v[0] - A.v[0]) * nx + (B.v[1] - A.v[1]) * ny + (B.v[2] - A.v[2]) * nz;
        const invA = 1 / A.m, invB = 1 / B.m, pen = rs - d, corr = pen / (invA + invB);
        A.p[0] -= corr * invA * nx; A.p[1] -= corr * invA * ny; A.p[2] -= corr * invA * nz;
        B.p[0] += corr * invB * nx; B.p[1] += corr * invB * ny; B.p[2] += corr * invB * nz;
        if (vn >= 0) continue;
        const jn = -(1 + e) * vn / (invA + invB);
        A.v[0] -= jn * invA * nx; A.v[1] -= jn * invA * ny; A.v[2] -= jn * invA * nz;
        B.v[0] += jn * invB * nx; B.v[1] += jn * invB * ny; B.v[2] += jn * invB * nz;
      }
    }
  }

  surfaceDist(a) {
    const s = this.ship.p, dx = wrapX(a.p[0] - s[0], ENV.xHalf), dy = a.p[1] - s[1], dz = a.p[2] - s[2];
    return Math.sqrt(dx * dx + dy * dy + dz * dz) - a.r - ENV.ship.radius;
  }

  sense() { return sense(this); }

}
