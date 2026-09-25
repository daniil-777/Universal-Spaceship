// Plain-JS multilayer perceptron used for all inference (rollouts and the displayed ship), mirrored from the TF.js
// variables after every PPO update. Keeps the render loop free of GPU readbacks. Layers: tanh hidden, linear output.
export class MLP {
  constructor(sizes) {
    this.sizes = sizes;
    this.Wt = []; this.b = [];                                            // Wt[l] is stored transposed: [out][in], contiguous per output unit
    for (let l = 0; l < sizes.length - 1; l++) { this.Wt.push(new Float32Array(sizes[l] * sizes[l + 1])); this.b.push(new Float32Array(sizes[l + 1])); }
    this.bufA = new Float32Array(0); this.bufB = new Float32Array(0);
  }
  get inDim() { return this.sizes[0]; }
  get outDim() { return this.sizes[this.sizes.length - 1]; }
  get nParams() { return this.Wt.reduce((s, w, l) => s + w.length + this.b[l].length, 0); }

  setLayer(l, W, b) {                                                    // W: Float32Array in TF layout [in][out]
    const nin = this.sizes[l], nout = this.sizes[l + 1], Wt = this.Wt[l];
    if (W.length !== nin * nout || b.length !== nout) throw new Error(`layer ${l}: bad shapes ${W.length} ${b.length}`);
    for (let i = 0; i < nin; i++) for (let j = 0; j < nout; j++) Wt[j * nin + i] = W[i * nout + j];
    this.b[l].set(b);
  }
  getLayer(l) {                                                          // back to TF layout
    const nin = this.sizes[l], nout = this.sizes[l + 1], W = new Float32Array(nin * nout), Wt = this.Wt[l];
    for (let i = 0; i < nin; i++) for (let j = 0; j < nout; j++) W[i * nout + j] = Wt[j * nin + i];
    return { W, b: Float32Array.from(this.b[l]) };
  }

  forward(x, n = 1, out) {                                              // x: Float32Array(n * inDim) -> Float32Array(n * outDim)
    const L = this.Wt.length, maxH = Math.max(...this.sizes.slice(1));
    if (this.bufA.length < n * maxH) { this.bufA = new Float32Array(n * maxH); this.bufB = new Float32Array(n * maxH); }
    let src = x, dst = null;
    for (let l = 0; l < L; l++) {
      const nin = this.sizes[l], nout = this.sizes[l + 1], Wt = this.Wt[l], b = this.b[l], last = l === L - 1;
      dst = last ? (out && out.length >= n * nout ? out : new Float32Array(n * nout)) : (l % 2 === 0 ? this.bufA : this.bufB);   // hidden activations alternate between two buffers
      for (let i = 0; i < n; i++) {
        const xo = i * nin, yo = i * nout;
        for (let j = 0; j < nout; j++) {
          let s = b[j]; const wo = j * nin;
          for (let k = 0; k < nin; k++) s += src[xo + k] * Wt[wo + k];
          dst[yo + j] = last ? s : Math.tanh(s);
        }
      }
      src = dst;
    }
    return dst;
  }
}

// Orthogonal initialisation (Saxe et al.), gain-scaled, in TF layout [rows=in][cols=out]. Modified Gram–Schmidt on the
// longer dimension so the result has orthonormal rows or columns like tf.initializers.orthogonal.
export function orthogonal(rows, cols, gain, randn) {
  const tall = rows >= cols, n = tall ? rows : cols, m = tall ? cols : rows;   // n x m with n >= m, orthonormal columns
  const A = new Float64Array(n * m);
  for (let i = 0; i < n * m; i++) A[i] = randn();
  for (let j = 0; j < m; j++) {
    for (let k = 0; k < j; k++) {
      let d = 0; for (let i = 0; i < n; i++) d += A[i * m + j] * A[i * m + k];
      for (let i = 0; i < n; i++) A[i * m + j] -= d * A[i * m + k];
    }
    let nrm = 0; for (let i = 0; i < n; i++) nrm += A[i * m + j] ** 2;
    nrm = Math.sqrt(nrm) || 1;
    for (let i = 0; i < n; i++) A[i * m + j] /= nrm;
  }
  const W = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) W[r * cols + c] = gain * (tall ? A[r * m + c] : A[c * m + r]);
  return W;
}
