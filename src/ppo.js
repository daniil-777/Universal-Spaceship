// PPO-Clip agent (TensorFlow.js) with the standard implementation details: separate actor/critic MLPs with orthogonal
// init, state-independent log-std, GAE, per-batch advantage normalisation, observation + return normalisation,
// value-loss clipping, entropy bonus, global-norm gradient clipping, Adam(eps 1e-5) with linear LR decay, KL early
// stopping and proper time-limit bootstrapping. Inference runs on the plain-JS mirror (actor.js); TF.js is only needed
// for update(), and is attached lazily via setTF(tf) so the demo page can run without loading it.
import { MLP, orthogonal } from './actor.js';
import { RunningMeanStd, mulberry32, randn } from './mathx.js';
import { OBS_DIM } from './envconst.js';

let tf = null;
export function setTF(t) { tf = t; }
export function getTF() { return tf; }

export const PPO_DEFAULTS = Object.freeze({
  hidden: [128, 128], lr: 4e-4, lrMin: 5e-5, totalSteps: 3e6, gamma: 0.99, lambda: 0.95, clip: 0.2, clipValue: true,
  entCoef: 1e-3, vfCoef: 0.5, maxGradNorm: 0.5, epochs: 5, minibatch: 2048, targetKL: 0.03,
  logStdInit: -0.7, logStdMin: -2.5, logStdMax: 0.5, obsClip: 5, adamEps: 1e-5,
  adaptiveLR: true, klThreshold: 0.01, lrMinAdaptive: 1e-5, lrMaxAdaptive: 2e-3,   // KL-adaptive learning rate (rl_games style); falls back to the linear schedule when off
});
const LOG2PI = Math.log(2 * Math.PI);
let instanceCounter = 0;

export function f32ToB64(arr) {
  const u8 = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength); let s = '';
  for (let i = 0; i < u8.length; i += 8192) s += String.fromCharCode.apply(null, u8.subarray(i, i + 8192));
  return btoa(s);
}
export function b64ToF32(s) { const bin = atob(s), u8 = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i); return new Float32Array(u8.buffer); }
// float16 packing halves the policy file (weights tolerate it: relative error ~1e-3)
function bytesToB64(u8) { let s = ''; for (let i = 0; i < u8.length; i += 8192) s += String.fromCharCode.apply(null, u8.subarray(i, i + 8192)); return btoa(s); }
function b64ToBytes(s) { const bin = atob(s), u8 = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i); return u8; }
export function f32ToF16B64(arr) { const u16 = new Uint16Array(arr.length); for (let i = 0; i < arr.length; i++) u16[i] = toHalf(arr[i]); return bytesToB64(new Uint8Array(u16.buffer)); }
export function f16B64ToF32(s) { const u8 = b64ToBytes(s), u16 = new Uint16Array(u8.buffer, 0, u8.length >> 1), out = new Float32Array(u16.length); for (let i = 0; i < u16.length; i++) out[i] = fromHalf(u16[i]); return out; }

// An older policy (fewer inputs) extended to newDim: the new inputs get zero weights and a neutral normaliser (mean 0,
// var 1), so the network acts exactly as before until training teaches it to use them (the weather spec's warm start).
export function padPolicyInputs(o, newDim) {
  if (o.obsDim >= newDim) return o;
  const dec = o.dtype === 'f16' ? f16B64ToF32 : b64ToF32, enc = o.dtype === 'f16' ? f32ToF16B64 : f32ToB64, add = newDim - o.obsDim;
  const pad = (L) => { const [, nout] = L.shape, W = dec(L.W), W2 = new Float32Array(newDim * nout); W2.set(W); return { ...L, shape: [newDim, nout], W: enc(W2) }; };   // TF layout [in][out]: new rows at the end
  const on = o.obsNorm;
  return { ...o, obsDim: newDim, actor: [pad(o.actor[0]), ...o.actor.slice(1)], critic: [pad(o.critic[0]), ...o.critic.slice(1)],
    obsNorm: { ...on, mean: [...on.mean, ...new Array(add).fill(0)], var: [...on.var, ...new Array(add).fill(1)] } };
}
function toHalf(v) { const f = new Float32Array([v]), b = new Uint32Array(f.buffer)[0], sign = (b >>> 16) & 0x8000, e = ((b >>> 23) & 0xff) - 112, m = b & 0x7fffff; if (e <= 0) return sign; if (e >= 31) return sign | 0x7c00; return sign | (e << 10) | (m >>> 13); }
function fromHalf(h) { const s = h & 0x8000 ? -1 : 1, e = (h >>> 10) & 0x1f, m = h & 0x3ff; if (e === 0) return s * m * Math.pow(2, -24); if (e === 31) return m ? NaN : s * Infinity; return s * (1 + m / 1024) * Math.pow(2, e - 15); }

export class PPOAgent {
  constructor(obsDim, actDim, cfg = {}, seed = 42) {
    this.obsDim = obsDim; this.actDim = actDim; this.cfg = { ...PPO_DEFAULTS, ...cfg };
    const sizes = [obsDim, ...this.cfg.hidden];
    this.actor = new MLP([...sizes, actDim]); this.critic = new MLP([...sizes, 1]);
    this.logStd = new Float32Array(actDim).fill(this.cfg.logStdInit);
    this.obsNorm = new RunningMeanStd(obsDim); this.retNorm = new RunningMeanStd(1);
    this.steps = 0; this.updates = 0; this.id = instanceCounter++; this.lrAdaptive = this.cfg.lr;
    this.tfv = null; this.optimizer = null;
    this.rng = mulberry32(seed); const rn = () => randn(this.rng);
    for (const [net, outGain] of [[this.actor, 0.01], [this.critic, 1.0]])
      for (let l = 0; l < net.Wt.length; l++) { const nin = net.sizes[l], nout = net.sizes[l + 1], last = l === net.Wt.length - 1; net.setLayer(l, orthogonal(nin, nout, last ? outGain : Math.SQRT2, rn), new Float32Array(nout)); }
  }
  get nParams() { return this.actor.nParams + this.critic.nParams + this.actDim; }

  // ---- inference (plain JS) ----
  normalize(raw, n, out) {                                      // (x - mean) / sqrt(var + 1e-8), clipped
    const d = this.obsDim, m = this.obsNorm.mean, v = this.obsNorm.var, c = this.cfg.obsClip;
    out = out || new Float32Array(n * d);
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) { let x = (raw[i * d + j] - m[j]) / Math.sqrt(v[j] + 1e-8); out[i * d + j] = x < -c ? -c : x > c ? c : x; }
    return out;
  }
  sample(obsN, n, actions, logp) {                              // Gaussian sample around the actor mean; logp of the raw (pre-tanh) sample
    const A = this.actDim, mean = this.actor.forward(obsN, n), ls = this.logStd;
    for (let i = 0; i < n; i++) {
      let lp = -0.5 * A * LOG2PI;
      for (let j = 0; j < A; j++) { const z = randn(this.rng); actions[i * A + j] = mean[i * A + j] + Math.exp(ls[j]) * z; lp += -0.5 * z * z - ls[j]; }
      logp[i] = lp;
    }
  }
  actMean(obsN, out) { return this.actor.forward(obsN, 1, out); }
  value(obsN) { return this.critic.forward(obsN, 1)[0]; }

  // ---- TF.js side ----
  async initTF() {
    if (this.tfv) return;
    if (!tf) throw new Error('TensorFlow.js not attached (call setTF)');
    const mkVar = (init, name) => { const v = tf.variable(init, true, name); init.dispose(); return v; };   // the variable keeps the buffer; drop the init tensor's handle
    const mk = (net, tag) => net.Wt.map((_, l) => { const { W, b } = net.getLayer(l); return [
      mkVar(tf.tensor2d(W, [net.sizes[l], net.sizes[l + 1]]), `ap${this.id}_${tag}W${l}`),
      mkVar(tf.tensor1d(b), `ap${this.id}_${tag}b${l}`)]; }).flat();
    this.tfv = { actor: mk(this.actor, 'a'), critic: mk(this.critic, 'c'), logStd: mkVar(tf.tensor1d(this.logStd), `ap${this.id}_logStd`) };
    this.optimizer = tf.train.adam(this.currentLR(), 0.9, 0.999, this.cfg.adamEps);
  }
  currentLR() { const c = this.cfg; if (c.adaptiveLR) return this.lrAdaptive; const f = Math.min(1, this.steps / c.totalSteps); return c.lr + (c.lrMin - c.lr) * f; }
  forwardTF(x, vars) {
    let h = x;
    for (let l = 0; l < vars.length / 2; l++) { h = tf.add(tf.matMul(h, vars[2 * l]), vars[2 * l + 1]); if (l < vars.length / 2 - 1) h = tf.tanh(h); }
    return h;
  }
  async valuesTF(obsN, n) {                                     // critic on a big batch (used once per update for GAE)
    const v = tf.tidy(() => tf.squeeze(this.forwardTF(tf.tensor2d(obsN, [n, this.obsDim]), this.tfv.critic), [1]));
    const out = new Float32Array(await v.data()); v.dispose(); return out;
  }
  async syncToJS() {
    const pull = async (net, vars) => { for (let l = 0; l < net.Wt.length; l++) net.setLayer(l, new Float32Array(await vars[2 * l].data()), new Float32Array(await vars[2 * l + 1].data())); };
    await pull(this.actor, this.tfv.actor); await pull(this.critic, this.tfv.critic);
    this.logStd.set(await this.tfv.logStd.data());
  }
  dispose() { if (this.tfv) { for (const v of [...this.tfv.actor, ...this.tfv.critic, this.tfv.logStd]) v.dispose(); this.tfv = null; } if (this.optimizer) { this.optimizer.dispose(); this.optimizer = null; } }

  // ---- learning ----
  // buf: { T, N, obs (T*N*obsDim, normalised), act (T*N*A), logp (T*N), rew (T*N, scaled), done (Uint8, terminal), trunc (Uint8, time limit),
  //        lastObs (N*obsDim), bootObs (nBoot*obsDim, the obs after each truncation), bootIdx (Int32 t*N+n) }
  // Async generator: yields after every minibatch so the caller can spread the update over animation frames; returns stats.
  async *update(buf) {
    await this.initTF();
    const c = this.cfg, { T, N, obs, act, logp, rew, done, trunc, lastObs, bootObs, bootIdx } = buf, TN = T * N, D = this.obsDim, A = this.actDim, nBoot = bootIdx.length;
    const all = new Float32Array((TN + N + nBoot) * D); all.set(obs); all.set(lastObs, TN * D); if (nBoot) all.set(bootObs.subarray(0, nBoot * D), (TN + N) * D);
    const vals = await this.valuesTF(all, TN + N + nBoot);
    const values = vals.subarray(0, TN), lastV = vals.subarray(TN, TN + N), bootV = new Float32Array(TN);
    for (let i = 0; i < nBoot; i++) bootV[bootIdx[i]] = vals[TN + N + i];
    const { adv, ret } = computeGAE({ T, N, rew, done, trunc, values, lastV, bootV, gamma: c.gamma, lambda: c.lambda });
    let m = 0, s2 = 0; for (let i = 0; i < TN; i++) m += adv[i]; m /= TN; for (let i = 0; i < TN; i++) s2 += (adv[i] - m) ** 2; const sd = Math.sqrt(s2 / TN) + 1e-8;
    const advN = new Float32Array(TN); for (let i = 0; i < TN; i++) advN[i] = (adv[i] - m) / sd;
    let rm = 0, rv = 0, ev = 0; for (let i = 0; i < TN; i++) rm += ret[i]; rm /= TN; for (let i = 0; i < TN; i++) { rv += (ret[i] - rm) ** 2; ev += (ret[i] - values[i]) ** 2; }
    const explainedVar = rv > 0 ? 1 - ev / rv : 0;
    const t = { obs: tf.tensor2d(obs, [TN, D]), act: tf.tensor2d(act, [TN, A]), logp: tf.tensor1d(logp), adv: tf.tensor1d(advN), ret: tf.tensor1d(ret), val: tf.tensor1d(values) };
    const vars = [...this.tfv.actor, ...this.tfv.critic, this.tfv.logStd];
    this.optimizer.learningRate = this.currentLR();
    const order = new Int32Array(TN); for (let i = 0; i < TN; i++) order[i] = i;
    const acc = { pgLoss: 0, vLoss: 0, entropy: 0, kl: 0, clipFrac: 0, n: 0 }; let earlyStop = false, epochsDone = 0;
    for (let ep = 0; ep < c.epochs && !earlyStop; ep++) {
      for (let i = TN - 1; i > 0; i--) { const j = Math.floor(this.rng() * (i + 1)); const tmp = order[i]; order[i] = order[j]; order[j] = tmp; }
      for (let start = 0; start < TN; start += c.minibatch) {
        const idx = tf.tensor1d(order.subarray(start, Math.min(TN, start + c.minibatch)), 'int32');
        const stats = tf.tidy(() => {
          const o = tf.gather(t.obs, idx), a = tf.gather(t.act, idx), olp = tf.gather(t.logp, idx), ad = tf.gather(t.adv, idx), rt = tf.gather(t.ret, idx), ov = tf.gather(t.val, idx);
          const kept = [];
          const lossFn = () => {
            const mean = this.forwardTF(o, this.tfv.actor), ls = this.tfv.logStd, std = tf.exp(ls);
            const z = tf.div(tf.sub(a, mean), std);
            const lp = tf.sub(tf.sub(tf.mul(tf.sum(tf.square(z), 1), -0.5), tf.sum(ls)), 0.5 * A * LOG2PI);
            const logRatio = tf.sub(lp, olp), ratio = tf.exp(logRatio);
            const pg = tf.neg(tf.mean(tf.minimum(tf.mul(ad, ratio), tf.mul(ad, tf.clipByValue(ratio, 1 - c.clip, 1 + c.clip)))));
            const v = tf.squeeze(this.forwardTF(o, this.tfv.critic), [1]);
            const vl = c.clipValue
              ? tf.mul(0.5, tf.mean(tf.maximum(tf.square(tf.sub(v, rt)), tf.square(tf.sub(tf.add(ov, tf.clipByValue(tf.sub(v, ov), -c.clip, c.clip)), rt)))))
              : tf.mul(0.5, tf.mean(tf.square(tf.sub(v, rt))));
            const ent = tf.sum(tf.add(ls, 0.5 + 0.5 * LOG2PI));
            const kl = tf.mean(tf.sub(tf.sub(ratio, 1), logRatio));                  // unbiased approx KL (Schulman)
            const cf = tf.mean(tf.cast(tf.greater(tf.abs(tf.sub(ratio, 1)), c.clip), 'float32'));
            kept.push(tf.keep(pg), tf.keep(vl), tf.keep(ent), tf.keep(kl), tf.keep(cf));
            return tf.add(tf.sub(pg, tf.mul(c.entCoef, ent)), tf.mul(c.vfCoef, vl));
          };
          const { value, grads } = tf.variableGrads(lossFn, vars);
          const names = Object.keys(grads), gn = tf.sqrt(tf.addN(names.map((k) => tf.sum(tf.square(grads[k])))));
          const scale = tf.minimum(1, tf.div(c.maxGradNorm, tf.add(gn, 1e-6)));
          const clipped = {}; for (const k of names) clipped[k] = tf.mul(grads[k], scale);
          this.optimizer.applyGradients(clipped);
          value.dispose();
          const st = tf.stack(kept); for (const k of kept) k.dispose();
          return st;
        });
        idx.dispose();
        const sv = await stats.data(); stats.dispose();
        acc.pgLoss += sv[0]; acc.vLoss += sv[1]; acc.entropy += sv[2]; acc.kl += sv[3]; acc.clipFrac += sv[4]; acc.n++;
        if (!Number.isFinite(sv[0]) || !Number.isFinite(sv[1])) throw new Error('non-finite loss');
        yield { epoch: ep, kl: sv[3] };
        if (c.targetKL && sv[3] > 1.5 * c.targetKL) { earlyStop = true; break; }
      }
      epochsDone++;
    }
    for (const k in t) t[k].dispose();
    tf.tidy(() => this.tfv.logStd.assign(tf.clipByValue(this.tfv.logStd, c.logStdMin, c.logStdMax)));
    await this.syncToJS();
    this.updates++;
    const n = Math.max(1, acc.n);
    if (c.adaptiveLR) { const kl = acc.kl / n; if (kl > 2 * c.klThreshold) this.lrAdaptive /= 1.5; else if (kl < 0.5 * c.klThreshold) this.lrAdaptive *= 1.5; this.lrAdaptive = Math.min(c.lrMaxAdaptive, Math.max(c.lrMinAdaptive, this.lrAdaptive)); }
    return { pgLoss: acc.pgLoss / n, vLoss: acc.vLoss / n, entropy: acc.entropy / n, kl: acc.kl / n, clipFrac: acc.clipFrac / n, explainedVar, lr: this.currentLR(), epochs: epochsDone, earlyStop, minibatches: acc.n };
  }

  // ---- snapshots (typed arrays, transferable between the page and the training worker) ----
  snapshot() {
    const L = (net) => net.Wt.map((_, l) => { const { W, b } = net.getLayer(l); return [W, b]; }).flat();
    const o = this.obsNorm, r = this.retNorm;
    return { obsDim: this.obsDim, actDim: this.actDim, hidden: this.cfg.hidden, actor: L(this.actor), critic: L(this.critic), logStd: Float32Array.from(this.logStd),
      obsMean: Float64Array.from(o.mean), obsVar: Float64Array.from(o.var), obsCount: o.count, retMean: Float64Array.from(r.mean), retVar: Float64Array.from(r.var), retCount: r.count,
      steps: this.steps, updates: this.updates, lr: this.lrAdaptive, meta: this.meta || {} };
  }
  applySnapshot(s) {
    for (let l = 0; l < this.actor.Wt.length; l++) { this.actor.setLayer(l, s.actor[2 * l], s.actor[2 * l + 1]); this.critic.setLayer(l, s.critic[2 * l], s.critic[2 * l + 1]); }
    this.logStd.set(s.logStd); this.obsNorm.mean.set(s.obsMean); this.obsNorm.var.set(s.obsVar); this.obsNorm.count = s.obsCount;
    this.retNorm.mean.set(s.retMean); this.retNorm.var.set(s.retVar); this.retNorm.count = s.retCount; this.steps = s.steps; this.updates = s.updates; this.meta = s.meta || {}; if (s.lr) this.lrAdaptive = s.lr;
    if (this.tfv) this.syncToTF();
    return this;
  }
  static fromSnapshot(s, cfg = {}) { return new PPOAgent(s.obsDim, s.actDim, { ...cfg, hidden: s.hidden }).applySnapshot(s); }
  static transferables(s) { return [...s.actor, ...s.critic, s.logStd, s.obsMean, s.obsVar, s.retMean, s.retVar].map((a) => a.buffer); }
  syncToTF() {                                                  // JS mirror -> TF variables (after a load); resets the optimiser state
    tf.tidy(() => {
      const push = (net, vars) => { for (let l = 0; l < net.Wt.length; l++) { const { W, b } = net.getLayer(l); vars[2 * l].assign(tf.tensor2d(W, [net.sizes[l], net.sizes[l + 1]])); vars[2 * l + 1].assign(tf.tensor1d(b)); } };
      push(this.actor, this.tfv.actor); push(this.critic, this.tfv.critic); this.tfv.logStd.assign(tf.tensor1d(this.logStd));
    });
    if (this.optimizer) { this.optimizer.dispose(); this.optimizer = tf.train.adam(this.currentLR(), 0.9, 0.999, this.cfg.adamEps); }
  }

  // ---- persistence ----
  toJSON(meta = {}, { half = false } = {}) {
    const enc = half ? f32ToF16B64 : f32ToB64;
    const layers = (net) => net.Wt.map((_, l) => { const { W, b } = net.getLayer(l); return { shape: [net.sizes[l], net.sizes[l + 1]], W: enc(W), b: enc(b) }; });
    return { format: 'astro-pilot-policy-v1', dtype: half ? 'f16' : 'f32', obsDim: this.obsDim, actDim: this.actDim, hidden: this.cfg.hidden, steps: this.steps, updates: this.updates,
      logStd: Array.from(this.logStd), obsNorm: this.obsNorm.toJSON(), retNorm: this.retNorm.toJSON(), actor: layers(this.actor), critic: layers(this.critic), lr: this.lrAdaptive, meta };
  }
  static fromJSON(o, cfg = {}) {
    o = padPolicyInputs(o, OBS_DIM);                          // an older policy (fewer inputs) acts unchanged on the new observation
    if (o.format !== 'astro-pilot-policy-v1') throw new Error('unknown policy format');
    const ag = new PPOAgent(o.obsDim, o.actDim, { ...cfg, hidden: o.hidden }), dec = o.dtype === 'f16' ? f16B64ToF32 : b64ToF32;
    o.actor.forEach((L, l) => ag.actor.setLayer(l, dec(L.W), dec(L.b)));
    o.critic.forEach((L, l) => ag.critic.setLayer(l, dec(L.W), dec(L.b)));
    ag.logStd.set(o.logStd); ag.obsNorm = RunningMeanStd.fromJSON(o.obsNorm); ag.retNorm = RunningMeanStd.fromJSON(o.retNorm);
    ag.steps = o.steps || 0; ag.updates = o.updates || 0; ag.meta = o.meta || {}; if (o.lr) ag.lrAdaptive = o.lr;
    return ag;
  }
}

// Generalised advantage estimation over a time-major rollout [T][N]. done = terminal (value 0 after), trunc = time limit
// (bootstrap with the value of the observation the episode ended in). Both reset the advantage chain.
export function computeGAE({ T, N, rew, done, trunc, values, lastV, bootV, gamma, lambda }) {
  const adv = new Float32Array(T * N), ret = new Float32Array(T * N);
  for (let n = 0; n < N; n++) {
    let nextV = lastV[n], gae = 0;
    for (let t = T - 1; t >= 0; t--) {
      const i = t * N + n;
      if (done[i]) { nextV = 0; gae = 0; } else if (trunc[i]) { nextV = bootV[i]; gae = 0; }
      const delta = rew[i] + gamma * nextV - values[i];
      gae = delta + gamma * lambda * gae;
      adv[i] = gae; ret[i] = gae + values[i];
      nextV = values[i];
    }
  }
  return { adv, ret };
}
