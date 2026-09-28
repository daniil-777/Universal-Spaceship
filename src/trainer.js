// Trainer — runs N SpaceEnv copies in lock-step, collects a time-major rollout [T][N] with the JS actor mirror, then
// drives PPOAgent.update() one minibatch at a time. Everything is chunked by a wall-clock budget and yields to the
// browser between chunks, so the 3D view keeps rendering while the fleet learns. Also owns the automatic curriculum
// (asteroid count / speed / size grow once the fleet survives 75 % of the episode cap) and the metrics history.
import { SpaceEnv, OBS_DIM, ACT_DIM, ENV } from './env.js';
import { hasLongGrid } from './cityfield.js';

export const HIST_KEYS = ['step', 'epRet', 'epLen', 'collPer1k', 'evalLen', 'evalColl', 'pgLoss', 'vLoss', 'entropy', 'kl', 'clipFrac', 'explainedVar', 'level', 'lr', 'sps'];
const EVAL_EVERY = 8, EVAL_EPISODES = 6;
const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

const CITIES = ['dubai', 'newyork', 'moscow', 'london'];
export class Trainer {
  constructor(agent, { nEnvs = 128, T = 64, seed = 1000, curriculum = true, level = 0, speedScale = 1, mixed = false, weather = 0 } = {}) {
    this.agent = agent; this.N = nEnvs; this.T = T; this.curriculum = curriculum; this.level = level; this.mixed = mixed;
    this.weatherMax = weather; this.wxRaisedAt = -Infinity;   // the atmospheric envs fly weather up to this severity (raised by the check-ups); 0 = calm air
    // mixed: true → half asteroid belts, a quarter mountain worlds (airliners + flocks), a quarter city skylines (flocks);
  // 'cities' → a quarter belts, a quarter mountains, half cities. One policy learns all of them.
    const cityHeavy = mixed === 'cities';
    this.envs = Array.from({ length: nEnvs }, (_, i) => {
      if (!mixed || (cityHeavy ? i % 4 === 0 : i % 2 === 0)) return new SpaceEnv(seed + i, { level, speedScale });
      if (i % 4 === 1) return (i >> 2) % 2 ? new SpaceEnv(seed + i, { level, count: 0, comets: 0, meshy: hasLongGrid('avatar') && (i >> 3) % 2 ? 'avatar' : true, autoBirds: true })   // the Avatar valley joins once its grid is registered
        : new SpaceEnv(seed + i, { level, count: 0, comets: 0, mountains: true, mountainSeed: 5 + (i % 7), autoPlanes: true, autoBirds: true });
      const cities = hasLongGrid('mega') ? [...CITIES, 'mega'] : CITIES;   // the megacity joins the mix once its grid is registered
      return new SpaceEnv(seed + i, { level, count: 0, comets: 0, city: cities[(cityHeavy ? (i >> 1) : (i >> 2)) % cities.length], citySeed: 1 + (i % 5), autoBirds: true });
    });
    for (const e of this.envs) if (e.atmosphere && this.weatherMax > 0) { e.weatherSeverity = this.weatherMax * e.rngSky(); e.reset(); }
    const TN = T * nEnvs, D = OBS_DIM, A = ACT_DIM;
    this.buf = { T, N: nEnvs, obs: new Float32Array(TN * D), act: new Float32Array(TN * A), logp: new Float32Array(TN), rew: new Float32Array(TN),
      done: new Uint8Array(TN), trunc: new Uint8Array(TN), lastObs: new Float32Array(nEnvs * D), bootObs: new Float32Array(nEnvs * D), bootIdx: new Int32Array(nEnvs) };
    this.nBoot = 0;
    this.rawObs = new Float32Array(nEnvs * D); this.actions = new Float32Array(nEnvs * A); this.logp = new Float32Array(nEnvs); this.retBatch = new Float32Array(nEnvs);
    this.retAcc = new Float64Array(nEnvs); this.epRet = new Float64Array(nEnvs); this.epLen = new Int32Array(nEnvs);
    this.recent = [];                                       // last 100 finished episodes: { ret, len, collided, level }
    this.sinceLevelChange = 0;
    this.metrics = { step: agent.steps || 0, episodes: 0, collisions: 0, updates: agent.updates || 0, level, epRet: NaN, epLen: NaN, collPer1k: NaN,
      evalLen: NaN, evalColl: NaN, pgLoss: NaN, vLoss: NaN, entropy: NaN, kl: NaN, clipFrac: NaN, explainedVar: NaN, lr: agent.currentLR(), sps: 0, elapsed: 0, phase: 'idle' };
    this.history = Object.fromEntries(HIST_KEYS.map((k) => [k, []]));
    this.levelChanges = [];                                 // steps at which the curriculum level rose (chart annotations)
    this.running = false; this.mode = 'balanced'; this.onMetrics = null; this.onEpisode = null;
    this.trainedMs = 0;
  }
  get budgetMs() { return this.mode === 'max' ? 40 : 8; }

  yieldFrame() {
    if (this.mode !== 'max' && typeof requestAnimationFrame === 'function') return new Promise((r) => requestAnimationFrame(() => r()));
    if (this.mode !== 'max' && typeof document === 'undefined') return new Promise((r) => setTimeout(r, 6));   // worker, balanced: leave CPU for the page
    if (typeof setImmediate === 'function') return new Promise((r) => setImmediate(r));                       // Node
    if (typeof MessageChannel !== 'undefined') {                                                                // browser, max mode: a fast macrotask yield
      if (!this._mc) this._mc = new MessageChannel();
      return new Promise((r) => { this._mc.port1.onmessage = () => r(); this._mc.port2.postMessage(0); });
    }
    return new Promise((r) => setTimeout(r, 0));
  }

  rolloutStep(t) {
    const { N, buf, agent: ag, envs } = this, D = OBS_DIM, A = ACT_DIM;
    for (let i = 0; i < N; i++) this.rawObs.set(envs[i].obs, i * D);
    ag.obsNorm.update(this.rawObs, N);
    const obsN = buf.obs.subarray(t * N * D, (t + 1) * N * D);
    ag.normalize(this.rawObs, N, obsN);
    ag.sample(obsN, N, this.actions, this.logp);
    buf.act.set(this.actions, t * N * A); buf.logp.set(this.logp, t * N);
    const g = ag.cfg.gamma;
    for (let i = 0; i < N; i++) { this.retAcc[i] = g * this.retAcc[i]; }
    for (let i = 0; i < N; i++) {
      const env = envs[i], res = env.step(this.actions.subarray(i * A, (i + 1) * A)), k = t * N + i;
      this.epRet[i] += res.reward; this.epLen[i]++;
      this.retAcc[i] += res.reward; this.retBatch[i] = this.retAcc[i];
      buf.rew[k] = res.reward; buf.done[k] = res.done ? 1 : 0; buf.trunc[k] = res.truncated ? 1 : 0;
      if (res.done || res.truncated) {
        if (res.truncated && this.nBoot < N) { ag.normalize(env.obs, 1, buf.bootObs.subarray(this.nBoot * D, (this.nBoot + 1) * D)); buf.bootIdx[this.nBoot++] = k; }
        this.finishEpisode(i, res.done);
        if (env.atmosphere && this.weatherMax > 0) env.weatherSeverity = this.weatherMax * env.rngSky();   // every episode its own sky
        env.reset(); this.retAcc[i] = 0;
      }
    }
    ag.retNorm.update(this.retBatch, N);
    const rs = 1 / Math.sqrt(ag.retNorm.var[0] + 1e-8);
    for (let i = 0; i < N; i++) buf.rew[t * N + i] *= rs;          // return-based reward scaling
    this.metrics.step += N; ag.steps += N;
  }

  weatherGate(step, calmLen, roughLen) {                  // +0.1 when the weather costs the pilot less than a fifth of its flight time, at most every 2 M steps
    if (roughLen >= 0.8 * calmLen && step - this.wxRaisedAt >= 2e6) { this.weatherMax = Math.min(1, Math.round((this.weatherMax + 0.1) * 10) / 10); this.wxRaisedAt = step; this.levelChanges.push(step); }
    return this.weatherMax;
  }
  finishEpisode(i, collided) {
    const ep = { ret: this.epRet[i], len: this.epLen[i], collided, level: this.level, step: this.metrics.step };
    this.recent.push(ep); if (this.recent.length > 100) this.recent.shift();
    this.metrics.episodes++; if (collided) this.metrics.collisions++;
    this.epRet[i] = 0; this.epLen[i] = 0; this.sinceLevelChange++;
    if (this.onEpisode) this.onEpisode(ep);
  }

  async run() {
    if (this.running) return;
    this.running = true;
    const { N, T, buf, envs, agent } = this, D = OBS_DIM;
    let frameStart = now(); const runStart = now();
    while (this.running) {
      this.nBoot = 0; this.metrics.phase = 'rollout';
      const tStart = now(), step0 = this.metrics.step;
      let busy = 0, mbN = 0;
      for (let t = 0; t < T && this.running; t++) {
        const t1 = now(); this.rolloutStep(t); busy += now() - t1;
        if (now() - frameStart > this.budgetMs) { await this.yieldFrame(); frameStart = now(); }
      }
      if (!this.running) break;
      this.metrics.rolloutMs = now() - tStart; this.metrics.rolloutBusyMs = busy;
      for (let i = 0; i < N; i++) agent.normalize(envs[i].obs, 1, buf.lastObs.subarray(i * D, (i + 1) * D));
      this.metrics.phase = 'update';
      const tU = now(), gen = agent.update({ ...buf, bootIdx: buf.bootIdx.subarray(0, this.nBoot) });
      let r;
      while (!(r = await gen.next()).done) { mbN++; if (now() - frameStart > this.budgetMs) { await this.yieldFrame(); frameStart = now(); } }
      this.metrics.updateMs = now() - tU; this.metrics.minibatchMs = mbN ? (now() - tU) / mbN : 0;
      this.metrics.sps = (this.metrics.step - step0) / ((now() - tStart) / 1000);
      this.trainedMs += now() - tStart;
      this.afterUpdate(r.value);
      await this.yieldFrame(); frameStart = now();
    }
    this.metrics.phase = 'idle'; this.running = false; this.metrics.elapsed = this.trainedMs / 1000;
    if (this.onMetrics) this.onMetrics(this.metrics);
  }
  stop() { this.running = false; }

  afterUpdate(stats) {
    const m = this.metrics; Object.assign(m, stats); m.updates++; m.elapsed = this.trainedMs / 1000;
    const rec = this.recent;
    if (rec.length) {
      let sr = 0, sl = 0, sc = 0; for (const e of rec) { sr += e.ret; sl += e.len; sc += e.collided ? 1 : 0; }
      m.epRet = sr / rec.length; m.epLen = sl / rec.length; m.collPer1k = 1000 * sc / sl;
    }
    if (m.updates % EVAL_EVERY === 0) {                        // deterministic check-up (mean action) at the current level
      const ev = evaluatePolicy(this.agent, { episodes: EVAL_EPISODES, seed: 5000 + Math.round(this.level * 10), level: this.level });
      m.evalLen = ev.meanLen; m.evalColl = ev.collPer1k;
      if (this.curriculum && this.level < 1 - 1e-9 && ev.meanLen >= 0.8 * ENV.maxSteps) {   // the fleet flies the whole episode without crashing → harder belt
        this.level = Math.min(1, Math.round((this.level + 0.1) * 10) / 10);
        for (const e of this.envs) e.setLevel(this.level);
        this.sinceLevelChange = 0; this.levelChanges.push(m.step);
      }
    }
    if (m.updates % EVAL_EVERY === 0 && this.weatherMax > 0 && this.weatherMax < 1 - 1e-9) {   // rougher weather once the pilot flies the current maximum as long as calm air (the same Alps episodes)
      const o = { episodes: 4, seed: 7000, level: this.level, count: 0, comets: 0, mountains: true, planes: 4, birds: 2 };
      this.weatherGate(m.step, evaluatePolicy(this.agent, { ...o, weather: 0 }).meanLen, evaluatePolicy(this.agent, { ...o, weather: this.weatherMax }).meanLen);
    }
    m.level = this.level; m.weather = this.weatherMax;
    for (const k of HIST_KEYS) this.history[k].push(m[k]);
    if (this.onMetrics) this.onMetrics(m);
  }

  setLevel(l) { this.level = l; for (const e of this.envs) e.setLevel(l); this.sinceLevelChange = 0; }
  fleet(out) {                                              // positions of all training ships (for the fleet scatter)
    out = out || new Float32Array(this.N * 3);
    for (let i = 0; i < this.N; i++) { const p = this.envs[i].ship.p; out[i * 3] = p[0]; out[i * 3 + 1] = p[1]; out[i * 3 + 2] = p[2]; }
    return out;
  }
}

// Deterministic evaluation (mean action, no exploration noise) for the harness and the README numbers.
export function evaluatePolicy(agent, { episodes = 20, seed = 999, level = 1, count = null, comets = null, speedScale = 1, maxSteps = ENV.maxSteps, mountains = false, planes = 0, city = null, birds = 0, pillars = false, meshy = false, edgeGuard = true, weather = 0 } = {}) {
  const env = new SpaceEnv(seed, { level, count, comets, speedScale, mountains, planes, city, birds, pillars, meshy, edgeGuard, weather }), obsN = new Float32Array(OBS_DIM), out = new Float32Array(ACT_DIM), prev = new Float32Array(ACT_DIM);
  let collisions = 0, totLen = 0, totRet = 0, sumSpeed = 0, sumSpeed2 = 0, sumW = 0, sumJerk = 0, sumProg = 0, minLen = Infinity, spMin = Infinity, spMax = 0, brake = 0, boost = 0, guarded = 0, edge = 0, ceil = 0;
  const lz = ENV.zHalf + ENV.ship.wallClamp - 0.02, top = (mountains || pillars || meshy || city ? ENV.mountains.ceiling : ENV.yHalf) + ENV.ship.wallClamp - 0.02;   // at the clamp = scraping an edge
  for (let e = 0; e < episodes; e++) {
    env.reset(); prev.fill(0); let len = 0, ret = 0;
    while (len < maxSteps) {
      agent.normalize(env.obs, 1, obsN); agent.actMean(obsN, out);
      const r = env.step(out); ret += r.reward; len++;
      const s = env.ship, sp = Math.hypot(s.v[0], s.v[1], s.v[2]); sumSpeed += sp; sumSpeed2 += sp * sp; spMin = Math.min(spMin, sp); spMax = Math.max(spMax, sp); sumW += Math.hypot(s.w[0], s.w[1], s.w[2]); sumProg += r.progress;
      if (env.cmd[3] < -0.25) brake++; else if (env.cmd[3] > 0.25) boost++;
      if (env.guard > 0) guarded++; if (Math.abs(s.p[2]) >= lz) edge++; if (s.p[1] >= top) ceil++;
      let j = 0; for (let k = 0; k < ACT_DIM; k++) { const a = Math.tanh(out[k]); j += (a - prev[k]) ** 2; prev[k] = a; } sumJerk += Math.sqrt(j);
      if (r.done) { collisions++; break; }
    }
    totLen += len; totRet += ret; minLen = Math.min(minLen, len);
  }
  return { episodes, collisions, collPer1k: 1000 * collisions / totLen, crashesPerMinute: collisions / (totLen * ENV.dt / 60), meanLen: totLen / episodes, minLen, meanReturn: totRet / episodes,
    meanSpeed: sumSpeed / totLen, speedStd: Math.sqrt(Math.max(0, sumSpeed2 / totLen - (sumSpeed / totLen) ** 2)), speedMin: spMin, speedMax: spMax, brakeFrac: brake / totLen, boostFrac: boost / totLen,
    meanAngRate: sumW / totLen, meanActionJerk: sumJerk / totLen, progressPerSec: sumProg / (totLen * ENV.dt), level, count: env.asteroidCount, comets: env.cometCount,
    guardFrac: guarded / totLen, edgeFrac: edge / totLen, ceilFrac: ceil / totLen };   // how often the edge guard steered, how often the ship scraped a side edge, and the ceiling
}
