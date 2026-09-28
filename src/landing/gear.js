// The spaceplane as the landing scene draws it: the Astro Pilot ship ×9 on a rig posed from the flight state, with its
// landing gear (oleo legs that compress on the ground and swing up into their wells, doors, twin wheels that spin up
// at touchdown with a puff of tyre smoke), the drag chute (streams, inflates over 1.5 s, is released) and a nose landing
// light for dusk and night.
import * as THREE from 'three';
import { VEH, SCALE } from './vehicle.js';

function smokeTexture() {
  const c = document.createElement('canvas'); c.width = c.height = 64; const g = c.getContext('2d'), r = g.createRadialGradient(32, 32, 2, 32, 32, 31);
  r.addColorStop(0, 'rgba(235,235,235,0.9)'); r.addColorStop(0.5, 'rgba(210,210,210,0.35)'); r.addColorStop(1, 'rgba(200,200,200,0)'); g.fillStyle = r; g.fillRect(0, 0, 64, 64);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}

export function createRig(ship) {
  const rig = new THREE.Group(); ship.scale.setScalar(SCALE); rig.add(ship);
  const metal = new THREE.MeshStandardMaterial({ color: 0xb9bdc1, roughness: 0.35, metalness: 0.8 }), tyre = new THREE.MeshStandardMaterial({ color: 0x1c1c1e, roughness: 0.9 }), hub = new THREE.MeshStandardMaterial({ color: 0x8e9296, roughness: 0.5, metalness: 0.6 });
  const legs = VEH.legs.map((L, k) => {
    const pivot = new THREE.Group(); pivot.position.set(L.attach[0], L.attach[1], L.attach[2]); rig.add(pivot);
    const len = L.attach[1] - L.contact[1], r = k === 0 ? 0.55 : 0.72;
    const strut = new THREE.Mesh(new THREE.CylinderGeometry(k === 0 ? 0.16 : 0.24, k === 0 ? 0.16 : 0.24, 1, 12), metal); pivot.add(strut);
    const axle = new THREE.Group(); pivot.add(axle);
    const wheels = [-1, 1].map((s) => { const w = new THREE.Group(), t = new THREE.Mesh(new THREE.CylinderGeometry(r, r, 0.42, 22), tyre), h = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.55, r * 0.55, 0.44, 14), hub);
      t.rotation.x = h.rotation.x = Math.PI / 2; w.add(t, h); w.position.z = s * 0.34; axle.add(w); t.castShadow = true; return w; });
    const door = new THREE.Mesh(new THREE.BoxGeometry(k === 0 ? 3 : 4, 0.06, 1.2), new THREE.MeshStandardMaterial({ color: 0x2a2d31, roughness: 0.6 }));
    door.position.set(L.attach[0], L.attach[1] + 0.03, L.attach[2] + (k === 0 ? 0.9 : Math.sign(L.attach[2]) * 1.4)); rig.add(door);
    strut.castShadow = true; return { pivot, strut, axle, wheels, door, len, r, spin: 0 };
  });
  // drag chute: risers and a ribbon canopy behind the tail (its mouth to the airflow)
  const cg = new THREE.SphereGeometry(6, 24, 12, 0, Math.PI * 2, 0, Math.PI / 2), cp = cg.attributes.position, cc = new Float32Array(cp.count * 3);   // a ribbon canopy: orange and white gores, banded
  for (let i = 0; i < cp.count; i++) { const gore = Math.floor((Math.atan2(cp.getZ(i), cp.getX(i)) + Math.PI) / (Math.PI / 6)) % 2, band = Math.floor(Math.acos(Math.min(1, cp.getY(i) / 6)) / 0.26) % 2, k = band ? 0.8 : 1;
    cc.set(gore ? [0.9 * k, 0.28 * k, 0.07 * k] : [0.86 * k, 0.85 * k, 0.8 * k], i * 3); }
  cg.setAttribute('color', new THREE.BufferAttribute(cc, 3));
  const chute = new THREE.Group(), canopy = new THREE.Mesh(cg, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.9, side: THREE.DoubleSide }));
  canopy.rotation.z = Math.PI / 2; canopy.position.x = -32; chute.add(canopy);
  const riserGeo = new THREE.BufferGeometry().setFromPoints([...Array(8)].flatMap((_, i) => { const a = i / 8 * Math.PI * 2; return [new THREE.Vector3(0, 0, 0), new THREE.Vector3(-32, 6 * Math.sin(a), 6 * Math.cos(a))]; }));
  const risers = new THREE.LineSegments(riserGeo, new THREE.LineBasicMaterial({ color: 0xcfcfcf })); chute.add(risers); chute.position.set(VEH.tail[0], VEH.tail[1] + 1.5, 0); chute.visible = false; rig.add(chute);
  // tyre smoke and the landing light
  const smokeMat = new THREE.SpriteMaterial({ map: smokeTexture(), transparent: true, depthWrite: false }), puffs = [];
  const light = new THREE.SpotLight(0xfff2dd, 0, 1400, 0.22, 0.5, 1.2); light.position.set(12, -4, 0); light.target.position.set(80, -12, 0); rig.add(light, light.target);
  let wasDown = false; const _v = new THREE.Vector3();
  return {
    rig, legs,
    update(s, dt, wind, scene, night) {
      rig.position.set(s.p[0], s.p[1], s.p[2]); rig.quaternion.set(s.q[0], s.q[1], s.q[2], s.q[3]);
      const g = s.gear, gs = Math.hypot(s.v[0], s.v[2]);
      legs.forEach((l, k) => {
        const comp = s.legs[k].comp, ext = Math.max(0.5, l.len - Math.min(comp, 0.45));     // the oleo shortens under load
        l.pivot.rotation.z = (1 - g) * (k === 0 ? 1.45 : -1.45); l.pivot.visible = g > 0.02; l.door.rotation.x = g > 0.02 ? Math.sign(VEH.legs[k].attach[2] || 1) * 1.3 : 0;
        const sl = ext - l.r; l.strut.scale.y = sl; l.strut.position.y = -sl / 2; l.axle.position.y = -sl;   // the tyre's bottom on the contact point
        if (s.legs[k].onGround) l.spin += gs / l.r * dt; for (const w of l.wheels) w.rotation.z = -l.spin;
      });
      const down = s.legs[1].onGround || s.legs[2].onGround;                                // spin-up: the tyres smoke for a moment
      if (down && !wasDown && s.touchdown) for (const k of [1, 2]) for (let i = 0; i < 6; i++) {
        const sp = new THREE.Sprite(smokeMat.clone()); const L = VEH.legs[k].contact; _v.set(L[0] - i * 1.5, L[1] + 0.6, L[2]).applyQuaternion(rig.quaternion).add(rig.position);
        sp.position.copy(_v); sp.scale.setScalar(3 + i); sp.userData = { age: 0, life: 1.6 + 0.3 * i, vx: s.v[0] * 0.3 + wind[0], vz: s.v[2] * 0.3 + wind[2] }; scene.add(sp); puffs.push(sp);
      }
      wasDown = down;
      for (let i = puffs.length - 1; i >= 0; i--) { const p = puffs[i], u = p.userData; u.age += dt; p.position.x += u.vx * dt; p.position.z += u.vz * dt; p.position.y += 0.8 * dt; p.scale.setScalar(p.scale.x + 6 * dt);
        p.material.opacity = Math.max(0, 0.7 * (1 - u.age / u.life)); if (u.age > u.life) { scene.remove(p); p.material.dispose(); puffs.splice(i, 1); } }
      chute.visible = s.chute === 1; if (chute.visible) { const o = Math.max(0.05, s.chuteOpen); canopy.scale.set(o, 1.6 - 0.6 * o, o); risers.scale.set(1, o, o); }   // streaming long and thin, then the canopy blossoms
      light.intensity = night > 0.3 && g > 0.5 ? 3e3 * night : 0;   // candela-scale: a spot on the runway ahead, not a flood
    },
  };
}
