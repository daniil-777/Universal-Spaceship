// Droplets streaking past the camera while it is inside a cloud (atmosphere only): short line segments in a box around the
// camera, stretched along the air's motion relative to the ship, faded in by the cloud density at the camera.
import * as THREE from 'three';

export function createDroplets(scene, n = 260) {
  const pos = new Float32Array(n * 6), off = Float32Array.from({ length: n * 3 }, () => (Math.random() * 2 - 1) * 6), geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const mat = new THREE.LineBasicMaterial({ color: 0xe8f0ff, transparent: true, opacity: 0, depthWrite: false }), lines = new THREE.LineSegments(geo, mat), vel = [0, 0, 0];
  lines.frustumCulled = false; lines.renderOrder = 6; lines.visible = false; scene.add(lines);
  return {
    update(dt, camera, env, atmosphere) {
      const W = env.atmosphere ? env.weather : null, c = camera.position, dens = W ? W.cloudAt(c.x, c.y, c.z) * atmosphere : 0;
      mat.opacity += (Math.min(0.55, dens * 0.8) - mat.opacity) * (1 - Math.exp(-dt * 4)); lines.visible = mat.opacity > 0.01; if (!lines.visible) return;
      for (let k = 0; k < 3; k++) vel[k] = env.ship.v[k] - env.air.wind[k];
      for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) {
        let o = off[3 * i + k] - vel[k] * dt; if (o < -6) o += 12; else if (o > 6) o -= 12; off[3 * i + k] = o;
        pos[6 * i + k] = c.getComponent(k) + o; pos[6 * i + 3 + k] = pos[6 * i + k] + vel[k] * 0.06;
      }
      geo.attributes.position.needsUpdate = true;
    },
    dispose() { scene.remove(lines); geo.dispose(); mat.dispose(); },
  };
}
