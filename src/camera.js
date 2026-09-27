// Camera rig: Side (fixed side view whose visible width equals the wrap period, gentle parallax), Chase (damped
// follow in the ship's frame, banks with the ship) and Orbit (OrbitControls around the ship). All three shift with
// the ship when it wraps across the x seam so the jump is never visible.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { ENV } from './env.js';

export function createCameraRig(camera, canvas) {
  const controls = new OrbitControls(camera, canvas);
  controls.enabled = false; controls.enableDamping = true; controls.dampingFactor = 0.08; controls.minDistance = 5; controls.maxDistance = 160; controls.enablePan = false;
  const pos = new THREE.Vector3(), look = new THREE.Vector3(), up = new THREE.Vector3(0, 1, 0), tP = new THREE.Vector3(), tL = new THREE.Vector3(), tU = new THREE.Vector3(), delta = new THREE.Vector3(), gP = new THREE.Vector3(), gL = new THREE.Vector3();
  const EL = 12 * Math.PI / 180, WORLD_UP = new THREE.Vector3(0, 1, 0);
  let mode = 'side', lastX = 0, init = true;
  const damp = (cur, target, rate, dt) => cur.lerp(target, 1 - Math.exp(-rate * dt));
  const rig = {
    pitchExtra: 0,                                        // 0..1, set by the app during low passes
    groundLock: 0,                                        // 0..1: atmospheric flight — a world-level following camera that keeps the ground in view whatever the ship's attitude
    terrain: null,                                        // (x, z) → ground height; keeps the chase camera out of the mountains
    ceiling: null,                                        // (x, z) → a tunnel roof's underside (Infinity in the open); keeps the camera inside the bore
    tunnel: 0,                                            // 0..1 while the tunnel guide flies a bore: a low, level view that frames the cave mouth
    get mode() { return mode; },
    controls,
    setMode(m) { mode = m; controls.enabled = m === 'orbit'; init = true; },
    sideDistance(aspect) { return Math.min(150, (ENV.xHalf + 2.5) / (Math.tan(camera.fov * Math.PI / 360) * aspect)); },
    update(dt, ship, aspect) {                              // ship: { p, q, f, u } as three.js objects (render-space, wrapped x)
      if (Math.abs(ship.p.x - lastX) > ENV.xHalf) { const dx = ship.p.x - lastX; pos.x += dx; look.x += dx; camera.position.x += dx; controls.target.x += dx; }
      lastX = ship.p.x;
      if (mode === 'side') {
        const d = rig.sideDistance(aspect);
        tL.set(0, 0.14 * ship.p.y, 0.1 * ship.p.z); tP.set(0, d * Math.sin(EL) + 0.14 * ship.p.y, d * Math.cos(EL) + 0.06 * ship.p.z); tU.set(0, 1, 0);
        if (init) { pos.copy(tP); look.copy(tL); up.copy(tU); init = false; }
        damp(pos, tP, 3, dt); damp(look, tL, 3, dt); damp(up, tU, 3, dt);
        camera.position.copy(pos); camera.up.copy(up).normalize(); camera.lookAt(look);
      } else if (mode === 'chase') {
        tP.set(-13, 3.6, 0).applyQuaternion(ship.q).add(ship.p); tL.copy(ship.f).multiplyScalar(9).add(ship.p).addScaledVector(ship.u, -1.5 - 1.2 * rig.pitchExtra); tU.copy(ship.u).lerp(WORLD_UP, 0.65).normalize();   // a touch nose-down; the camera follows only a third of the ship's bank so the horizon stays readable
        if (rig.groundLock > 0) {                             // world-aligned chase: behind and above, looking ahead and down at the landscape
          const g = rig.groundLock, tn = rig.tunnel; gP.set(ship.p.x - 13 + 2 * tn, ship.p.y + 5 - 3.6 * tn, ship.p.z); gL.set(ship.p.x + 10 + 14 * tn, ship.p.y - 6.2 + 6.6 * tn, ship.p.z);
          if (rig.terrain) {                                  // keep the camera out of the terrain and keep the ship in sight: pull in when a tower or ridge is in between
            gP.y = Math.max(gP.y, rig.terrain(gP.x, gP.z) + 3.5);
            if (rig.ceiling) { const c = rig.ceiling(gP.x, gP.z); if (c < Infinity) gP.y = Math.max(Math.min(gP.y, c - 1.5), ship.p.y); }   // under a tunnel roof: no higher than the roof allows
            const L = gP.distanceTo(ship.p); let tBlock = L;
            for (let t = 3; t < L; t += 1) { const f = t / L, px = ship.p.x + (gP.x - ship.p.x) * f, py = ship.p.y + (gP.y - ship.p.y) * f, pz = ship.p.z + (gP.z - ship.p.z) * f; if (rig.terrain(px, pz) + 1.2 > py || (rig.ceiling && rig.ceiling(px, pz) - 1 < py)) { tBlock = t - 1; break; } }
            if (tBlock < L) gP.lerpVectors(ship.p, gP, Math.max(3, tBlock) / L);
          }
          tP.lerp(gP, g); tL.lerp(gL, g); tU.lerp(WORLD_UP, g).normalize();
        }
        if (init) { pos.copy(tP); look.copy(tL); up.copy(tU); init = false; }
        damp(pos, tP, 5, dt); damp(look, tL, 7, dt); damp(up, tU, 4, dt);
        camera.position.copy(pos); camera.up.copy(up).normalize(); camera.lookAt(look);
      } else {
        if (init) { camera.up.set(0, 1, 0); camera.position.copy(ship.p).add(delta.set(-13, 6, 19)); controls.target.copy(ship.p); init = false; }
        else { delta.copy(ship.p).sub(controls.target); controls.target.copy(ship.p); camera.position.add(delta); }
        controls.update();
        if (rig.terrain) { const floor = rig.terrain(camera.position.x, camera.position.z) + 2.5; if (camera.position.y < floor) { camera.position.y = floor; camera.lookAt(controls.target); } }   // never inside a mountain
      }
    },
  };
  return rig;
}
