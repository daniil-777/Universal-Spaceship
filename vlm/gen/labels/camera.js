// vlm/gen/labels/camera.js — camera matrices as three.js stores them (column-major matrixWorldInverse and
// projectionMatrix), built in Node for tests or read from the page at capture; projection to pixels and pixel rays.
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]], dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l]; };
const mul = (m, v) => [0, 1, 2, 3].map((r) => m[r] * v[0] + m[4 + r] * v[1] + m[8 + r] * v[2] + m[12 + r] * v[3]);
export function lookAtCamera({ eye, target, up = [0, 1, 0], fovDeg, aspect, near = 0.1, far = 10000 }) {
  const z = norm(sub(eye, target)), x = norm(cross(up, z)), y = cross(z, x), t = 1 / Math.tan(fovDeg * Math.PI / 360), nf = 1 / (near - far);
  return { matrixWorldInverse: [x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dot(x, eye), -dot(y, eye), -dot(z, eye), 1],
    projectionMatrix: [t / aspect, 0, 0, 0, 0, t, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0], fov_deg: fovDeg, aspect, near, far };
}
export function project(cam, p, W, H) {
  const c = mul(cam.projectionMatrix, mul(cam.matrixWorldInverse, [p[0], p[1], p[2], 1]));
  return c[3] <= 1e-9 ? { x: NaN, y: NaN, w: c[3], front: false } : { x: (c[0] / c[3] + 1) / 2 * W, y: (1 - c[1] / c[3]) / 2 * H, w: c[3], front: true };
}
export const cameraPosition = (cam) => { const m = cam.matrixWorldInverse; return [0, 1, 2].map((i) => -(m[i * 4] * m[12] + m[i * 4 + 1] * m[13] + m[i * 4 + 2] * m[14])); };
export function pixelRay(cam, px, py, W, H) {
  const m = cam.matrixWorldInverse, P = cam.projectionMatrix, d = norm([(2 * px / W - 1) / P[0], (1 - 2 * py / H) / P[5], -1]);
  return { origin: cameraPosition(cam), dir: norm([0, 1, 2].map((i) => m[i * 4] * d[0] + m[i * 4 + 1] * d[1] + m[i * 4 + 2] * d[2])) };
}
export function projectSphere(cam, c, r, W, H) {
  const p = project(cam, c, W, H);
  if (!p.front) return { inFrame: false, box: null, cx: NaN, cy: NaN, rPx: 0 };
  const rPx = r * cam.projectionMatrix[5] * H / 2 / p.w, box = [p.x - rPx, p.y - rPx, p.x + rPx, p.y + rPx];
  return { inFrame: box[2] >= 0 && box[0] <= W && box[3] >= 0 && box[1] <= H, box, cx: p.x, cy: p.y, rPx };
}
