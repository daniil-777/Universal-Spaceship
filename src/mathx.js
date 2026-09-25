// Small numeric helpers shared by the environment, the trainer and the renderer glue.
// Quaternions are plain arrays / typed arrays in [x, y, z, w] order (three.js convention).

export function mulberry32(seed) {                     // tiny seeded PRNG, uniform in [0, 1)
  let t = seed >>> 0;
  return function () {
    t = (t + 0x6D2B79F5) >>> 0;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

export function randn(rng) {                           // standard normal (Box–Muller, one value per call)
  let u = 0;
  while (u === 0) u = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

export function wrapX(x, half) {                       // periodic coordinate into [-half, half)
  const p = 2 * half;
  x = (x + half) % p;
  if (x < 0) x += p;
  return x - half;
}

export function clamp(x, lo, hi) { return x < lo ? lo : x > hi ? hi : x; }

export function qIdentity(out = new Float64Array(4)) { out[0] = 0; out[1] = 0; out[2] = 0; out[3] = 1; return out; }

export function qNormalize(q) {
  const n = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  q[0] /= n; q[1] /= n; q[2] /= n; q[3] /= n;
  return q;
}

export function qMul(a, b, out = new Float64Array(4)) {   // out = a ⊗ b (apply b first, then a, when rotating vectors)
  const ax = a[0], ay = a[1], az = a[2], aw = a[3], bx = b[0], by = b[1], bz = b[2], bw = b[3];
  out[0] = aw * bx + ax * bw + ay * bz - az * by;
  out[1] = aw * by - ax * bz + ay * bw + az * bx;
  out[2] = aw * bz + ax * by - ay * bx + az * bw;
  out[3] = aw * bw - ax * bx - ay * by - az * bz;
  return out;
}

export function qRotate(q, v, out = new Float64Array(3)) {    // out = q v q*  (body -> world)
  const qx = q[0], qy = q[1], qz = q[2], qw = q[3], vx = v[0], vy = v[1], vz = v[2];
  const tx = 2 * (qy * vz - qz * vy), ty = 2 * (qz * vx - qx * vz), tz = 2 * (qx * vy - qy * vx);
  out[0] = vx + qw * tx + (qy * tz - qz * ty);
  out[1] = vy + qw * ty + (qz * tx - qx * tz);
  out[2] = vz + qw * tz + (qx * ty - qy * tx);
  return out;
}

export function qInvRotate(q, v, out = new Float64Array(3)) { // out = q* v q  (world -> body)
  const qx = -q[0], qy = -q[1], qz = -q[2], qw = q[3], vx = v[0], vy = v[1], vz = v[2];
  const tx = 2 * (qy * vz - qz * vy), ty = 2 * (qz * vx - qx * vz), tz = 2 * (qx * vy - qy * vx);
  out[0] = vx + qw * tx + (qy * tz - qz * ty);
  out[1] = vy + qw * ty + (qz * tx - qx * tz);
  out[2] = vz + qw * tz + (qx * ty - qy * tx);
  return out;
}

export function qFromAxisAngle(ax, ay, az, angle, out = new Float64Array(4)) {   // axis must be unit length
  const s = Math.sin(angle / 2);
  out[0] = ax * s; out[1] = ay * s; out[2] = az * s; out[3] = Math.cos(angle / 2);
  return out;
}

const _dq = new Float64Array(4);
export function qIntegrate(q, w, h, out = q) {          // q <- q ⊗ exp(h w / 2), w = body-frame angular rate
  const wx = w[0] * h, wy = w[1] * h, wz = w[2] * h;
  const th = Math.sqrt(wx * wx + wy * wy + wz * wz);
  if (th < 1e-12) return out === q ? q : (out.set(q), out);
  const s = Math.sin(th / 2) / th;
  _dq[0] = wx * s; _dq[1] = wy * s; _dq[2] = wz * s; _dq[3] = Math.cos(th / 2);
  qMul(q, _dq, out);
  return qNormalize(out);
}

export function qRandom(rng, out = new Float64Array(4)) {    // uniformly random rotation (Shoemake)
  const u1 = rng(), u2 = rng() * 2 * Math.PI, u3 = rng() * 2 * Math.PI;
  const a = Math.sqrt(1 - u1), b = Math.sqrt(u1);
  out[0] = a * Math.sin(u2); out[1] = a * Math.cos(u2); out[2] = b * Math.sin(u3); out[3] = b * Math.cos(u3);
  return out;
}

export function qAxes(q, f, u, r) {                    // body axes in world coordinates
  const x = q[0], y = q[1], z = q[2], w = q[3];
  f[0] = 1 - 2 * (y * y + z * z); f[1] = 2 * (x * y + w * z); f[2] = 2 * (x * z - w * y);
  u[0] = 2 * (x * y - w * z); u[1] = 1 - 2 * (x * x + z * z); u[2] = 2 * (y * z + w * x);
  if (r) { r[0] = 2 * (x * z + w * y); r[1] = 2 * (y * z - w * x); r[2] = 1 - 2 * (x * x + y * y); }
}

export function randomUnit(rng, out = new Float64Array(3)) {  // uniform direction on the sphere
  const z = 2 * rng() - 1, a = 2 * Math.PI * rng(), s = Math.sqrt(1 - z * z);
  out[0] = s * Math.cos(a); out[1] = s * Math.sin(a); out[2] = z;
  return out;
}

export class RunningMeanStd {                          // Welford / Chan parallel update, used for observation and return normalisation
  constructor(dim) { this.dim = dim; this.mean = new Float64Array(dim); this.var = new Float64Array(dim).fill(1); this.count = 1e-4; }
  update(batch, n) {                                   // batch: Float32Array(n * dim)
    const d = this.dim, bm = new Float64Array(d), bv = new Float64Array(d);
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) bm[j] += batch[i * d + j];
    for (let j = 0; j < d; j++) bm[j] /= n;
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) { const e = batch[i * d + j] - bm[j]; bv[j] += e * e; }
    for (let j = 0; j < d; j++) bv[j] /= n;
    const tot = this.count + n;
    for (let j = 0; j < d; j++) {
      const delta = bm[j] - this.mean[j];
      const m2 = this.var[j] * this.count + bv[j] * n + delta * delta * this.count * n / tot;
      this.mean[j] += delta * n / tot;
      this.var[j] = m2 / tot;
    }
    this.count = tot;
  }
  toJSON() { return { mean: Array.from(this.mean), var: Array.from(this.var), count: this.count }; }
  static fromJSON(o) { const r = new RunningMeanStd(o.mean.length); r.mean.set(o.mean); r.var.set(o.var); r.count = o.count; return r; }
}
