// Minimal canvas charts for the training panel and the HUD. DPR aware, colours from CSS custom properties so the
// colour / mono / ink modes restyle them. LineChart: one series, 2 px line, 10 % area wash, hairline axes, end marker,
// crosshair + tooltip on hover. HeatGrid: single-hue sensor grid. Bars: centred action bars. FleetMap: the training fleet.
const css = (el, name, fallback) => (getComputedStyle(el).getPropertyValue(name).trim() || fallback);
export const fmtSteps = (x) => (x >= 1e6 ? (x / 1e6).toFixed(2) + 'M' : x >= 1e3 ? (x / 1e3).toFixed(x >= 1e5 ? 0 : 1) + 'k' : String(Math.round(x)));
export const fmtNum = (v, d = 2) => (Number.isFinite(v) ? (Math.abs(v) >= 1000 ? fmtSteps(v) : v.toFixed(d)) : '–');

function fit(canvas) {                                   // size the backing store to the CSS box × DPR
  const dpr = Math.min(3, window.devicePixelRatio || 1), w = canvas.clientWidth || 200, h = canvas.clientHeight || 60;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
  const ctx = canvas.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

export class LineChart {
  constructor(canvas, { format = (v) => fmtNum(v), tip = null, logY = false } = {}) {
    this.c = canvas; this.format = format; this.tip = tip; this.logY = logY;
    this.xs = []; this.ys = []; this.marks = []; this.hover = -1;
    canvas.addEventListener('pointermove', (e) => { const r = canvas.getBoundingClientRect(); this.hover = this.nearest(e.clientX - r.left); this.draw(); this.showTip(e); });
    canvas.addEventListener('pointerleave', () => { this.hover = -1; this.draw(); if (this.tip) this.tip.style.opacity = '0'; });
  }
  setData(xs, ys, marks = []) { this.xs = xs; this.ys = ys; this.marks = marks; }
  layout() { const { w, h } = fit(this.c); return { w, h, l: 34, r: 8, t: 6, b: 4 }; }
  nearest(px) {
    const L = this.layout(), n = this.xs.length; if (n < 1) return -1;
    const x0 = this.xs[0], x1 = this.xs[n - 1] || x0 + 1, sx = (L.w - L.l - L.r) / Math.max(1e-9, x1 - x0);
    let best = -1, bd = 1e9; for (let i = 0; i < n; i++) { if (!Number.isFinite(this.ys[i])) continue; const d = Math.abs(L.l + (this.xs[i] - x0) * sx - px); if (d < bd) { bd = d; best = i; } }
    return best;
  }
  showTip(e) {
    if (!this.tip || this.hover < 0) return;
    const wrap = this.tip.offsetParent || document.body, wr = wrap.getBoundingClientRect();
    this.tip.querySelector('.v').textContent = this.format(this.ys[this.hover]);
    this.tip.querySelector('.k').textContent = fmtSteps(this.xs[this.hover]) + ' steps';
    this.tip.style.opacity = '1';
    const x = e.clientX - wr.left + 12, y = e.clientY - wr.top - 30;
    this.tip.style.transform = `translate(${Math.min(x, wr.width - 120)}px, ${Math.max(0, y)}px)`;
  }
  draw() {
    const { ctx, w, h, l, r, t, b } = { ...this.layout(), ctx: this.c.getContext('2d') };
    const body = document.body, accent = css(body, '--accent', '#9fd0ff'), text2 = css(body, '--text-2', '#9a9a9f'), hair = css(body, '--hair', 'rgba(255,255,255,.14)'), surface = css(body, '--panel-solid', '#0b0b0d');
    ctx.clearRect(0, 0, w, h);
    const pts = []; for (let i = 0; i < this.xs.length; i++) if (Number.isFinite(this.ys[i])) pts.push(i);
    ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace'; ctx.textBaseline = 'middle'; ctx.fillStyle = text2;
    if (pts.length < 2) { ctx.textAlign = 'center'; ctx.fillStyle = text2; ctx.fillText(pts.length ? this.format(this.ys[pts[0]]) : 'waiting for data', w / 2, h / 2); return; }
    let lo = Infinity, hi = -Infinity; for (const i of pts) { lo = Math.min(lo, this.ys[i]); hi = Math.max(hi, this.ys[i]); }
    if (this.logY) { lo = Math.log10(Math.max(1e-6, lo)); hi = Math.log10(Math.max(1e-6, hi)); }
    if (hi - lo < 1e-9) { hi += 1; lo -= 1; }
    const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
    const x0 = this.xs[pts[0]], x1 = this.xs[pts[pts.length - 1]], pw = w - l - r, ph = h - t - b;
    const X = (x) => l + (x - x0) / Math.max(1e-9, x1 - x0) * pw, Y = (y) => t + (1 - ((this.logY ? Math.log10(Math.max(1e-6, y)) : y) - lo) / (hi - lo)) * ph;
    ctx.strokeStyle = hair; ctx.lineWidth = 1;                                            // baseline + mid hairline
    for (const yy of [t + ph, t + ph / 2]) { ctx.beginPath(); ctx.moveTo(l, Math.round(yy) + 0.5); ctx.lineTo(w - r, Math.round(yy) + 0.5); ctx.stroke(); }
    ctx.textAlign = 'right'; ctx.fillStyle = text2;                                        // y ticks: max / min
    const inv = (v) => (this.logY ? Math.pow(10, v) : v);
    ctx.fillText(this.format(inv(hi - pad)), l - 4, t + 4); ctx.fillText(this.format(inv(lo + pad)), l - 4, t + ph - 4);
    for (const m of this.marks) { if (m < x0 || m > x1) continue; ctx.strokeStyle = hair; ctx.beginPath(); ctx.moveTo(Math.round(X(m)) + 0.5, t); ctx.lineTo(Math.round(X(m)) + 0.5, t + ph); ctx.stroke(); }
    ctx.beginPath(); pts.forEach((i, k) => (k ? ctx.lineTo(X(this.xs[i]), Y(this.ys[i])) : ctx.moveTo(X(this.xs[i]), Y(this.ys[i]))));
    ctx.lineTo(X(x1), t + ph); ctx.lineTo(X(x0), t + ph); ctx.closePath(); ctx.globalAlpha = 0.1; ctx.fillStyle = accent; ctx.fill(); ctx.globalAlpha = 1;   // area wash
    ctx.beginPath(); pts.forEach((i, k) => (k ? ctx.lineTo(X(this.xs[i]), Y(this.ys[i])) : ctx.moveTo(X(this.xs[i]), Y(this.ys[i]))));
    ctx.strokeStyle = accent; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.lineCap = 'round'; ctx.stroke();
    const last = pts[pts.length - 1], hi_ = this.hover >= 0 && Number.isFinite(this.ys[this.hover]) ? this.hover : last;
    if (this.hover >= 0) { ctx.strokeStyle = text2; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(Math.round(X(this.xs[hi_])) + 0.5, t); ctx.lineTo(Math.round(X(this.xs[hi_])) + 0.5, t + ph); ctx.stroke(); }
    ctx.beginPath(); ctx.arc(X(this.xs[hi_]), Y(this.ys[hi_]), 6, 0, Math.PI * 2); ctx.fillStyle = surface; ctx.fill();   // end marker with a surface ring
    ctx.beginPath(); ctx.arc(X(this.xs[hi_]), Y(this.ys[hi_]), 4, 0, Math.PI * 2); ctx.fillStyle = accent; ctx.fill();
  }
}

export class HeatGrid {                                  // rows = elevation (top = up), cols = azimuth (left = port), value 0..1
  constructor(canvas, cols, rows) { this.c = canvas; this.cols = cols; this.rows = rows; this.v = new Float32Array(cols * rows); }
  set(values) { this.v.set(values); }
  draw() {
    const { ctx, w, h } = fit(this.c), accent = css(document.body, '--accent', '#9fd0ff'), gap = 2, cw = (w - gap * (this.cols - 1)) / this.cols, ch = (h - gap * (this.rows - 1)) / this.rows;
    ctx.clearRect(0, 0, w, h);
    for (let r = 0; r < this.rows; r++) for (let c = 0; c < this.cols; c++) {
      const v = this.v[(this.rows - 1 - r) * this.cols + c], x = c * (cw + gap), y = r * (ch + gap);
      ctx.globalAlpha = 0.07 + 0.93 * Math.pow(Math.max(0, Math.min(1, v)), 0.8); ctx.fillStyle = accent;
      ctx.beginPath(); ctx.roundRect(x, y, cw, ch, 2); ctx.fill();
    }
    ctx.globalAlpha = 1;
  }
}

export class Bars {                                      // centred bars in [-1, 1]
  constructor(canvas, labels) { this.c = canvas; this.labels = labels; this.v = new Float32Array(labels.length); }
  set(values) { this.v.set(values); }
  draw() {
    const { ctx, w, h } = fit(this.c), accent = css(document.body, '--accent', '#9fd0ff'), text2 = css(document.body, '--text-2', '#9a9a9f'), hair = css(document.body, '--hair', 'rgba(255,255,255,.14)');
    ctx.clearRect(0, 0, w, h);
    const n = this.labels.length, rowH = h / n, barH = Math.min(6, rowH * 0.45), lw = 44, mid = lw + (w - lw) / 2;
    ctx.font = '9px -apple-system, system-ui, sans-serif'; ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
    for (let i = 0; i < n; i++) {
      const y = i * rowH + rowH / 2, v = Math.max(-1, Math.min(1, this.v[i]));
      ctx.fillStyle = text2; ctx.fillText(this.labels[i], 0, y);
      ctx.strokeStyle = hair; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(lw, Math.round(y) + 0.5); ctx.lineTo(w, Math.round(y) + 0.5); ctx.stroke();
      ctx.fillStyle = accent; const x0 = v >= 0 ? mid : mid + v * (w - lw) / 2, bw = Math.abs(v) * (w - lw) / 2;
      ctx.beginPath(); ctx.roundRect(x0, y - barH / 2, Math.max(2, bw), barH, 2); ctx.fill();
    }
  }
}

export class FleetMap {                                  // top view of the corridor: every training ship as a dot, size by depth
  constructor(canvas, xHalf, yHalf) { this.c = canvas; this.xHalf = xHalf; this.yHalf = yHalf; this.pts = new Float32Array(0); }
  set(pts) { this.pts = pts; }
  draw() {
    const { ctx, w, h } = fit(this.c), accent = css(document.body, '--accent', '#9fd0ff'), hair = css(document.body, '--hair', 'rgba(255,255,255,.14)'), surface = css(document.body, '--panel-solid', '#0b0b0d');
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = hair; ctx.lineWidth = 1; ctx.beginPath(); ctx.roundRect(0.5, 0.5, w - 1, h - 1, 6); ctx.stroke();
    for (let i = 0; i < this.pts.length; i += 3) {
      const x = (this.pts[i] / this.xHalf * 0.5 + 0.5) * w, y = (0.5 - this.pts[i + 1] / this.yHalf * 0.5) * h, r = 2.2 + 1.6 * (this.pts[i + 2] / 18 + 1) / 2;
      ctx.beginPath(); ctx.arc(x, y, r + 1.5, 0, Math.PI * 2); ctx.fillStyle = surface; ctx.fill();
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fillStyle = accent; ctx.fill();
    }
  }
}
