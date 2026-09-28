// Rain for the landing scene: drops falling at 9 m/s with the wind in a box that travels with the camera, drawn as
// streaks along their velocity relative to the camera over one frame (so at approach speed they race past nearly
// level), and only below the cloud base they fall from.
import * as THREE from 'three';

const N = 3000, BX = 70, BY = 40, BZ = 70, FALL = 9;
export function createRain() {
  const drops = new Float32Array(N * 3), pos = new Float32Array(N * 6), geo = new THREE.BufferGeometry(), wrap = (v, c, h) => c + ((((v - c + h) % (2 * h)) + 2 * h) % (2 * h)) - h;
  for (let i = 0; i < N; i++) { drops[i * 3] = (Math.random() - 0.5) * 2 * BX; drops[i * 3 + 1] = (Math.random() - 0.5) * 2 * BY; drops[i * 3 + 2] = (Math.random() - 0.5) * 2 * BZ; }
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const lines = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: 0xaeb8c4, transparent: true, opacity: 0.32, depthWrite: false }));
  lines.frustumCulled = false; lines.renderOrder = 4;
  return {
    lines,
    update(dt, cam, camVel, wind, ceiling) {                // cam: the camera's position; camVel, wind: m/s (runway frame); ceiling: the cloud base (m)
      lines.visible = cam.y < ceiling + 30; if (!lines.visible) return;
      const vx = wind[0] - camVel[0], vy = -FALL - camVel[1], vz = wind[2] - camVel[2], k = Math.min(0.05, Math.max(1 / 60, dt));   // the streak: one frame of relative motion
      for (let i = 0; i < N; i++) {
        const j = i * 3; let x = drops[j] + wind[0] * dt, y = drops[j + 1] - FALL * dt, z = drops[j + 2] + wind[2] * dt;
        x = wrap(x, cam.x, BX); y = wrap(y, cam.y, BY); z = wrap(z, cam.z, BZ); if (y < 0) y += 2 * BY;   // keep the box around the camera, above the ground
        drops[j] = x; drops[j + 1] = y; drops[j + 2] = z;
        pos[i * 6] = x; pos[i * 6 + 1] = y; pos[i * 6 + 2] = z; pos[i * 6 + 3] = x - vx * k; pos[i * 6 + 4] = y - vy * k; pos[i * 6 + 5] = z - vz * k;
      }
      geo.attributes.position.needsUpdate = true;
    },
  };
}
