// The ship at full scale (x9, as landing/vehicle.js) posed from the simulation, with its RCS plumes driven by the
// fired jets: after every ship.update() the main-engine exhaust is hidden, the engine light zeroed, and glow slots 0-13
// rewritten for P13, P14, P1-P12 (consts.GLOW_SLOTS), billboards x9 (the glow shader adds its size in view space,
// after the model scale). P15/P16 and V1-V6 have no glow slot: the HUD's 22-jet map shows them.
import { createShip } from '../ship.js';
import { JETS, GLOW_SLOTS, CYCLE } from './consts.js';

export const SCALE = 9;
const REST = Object.freeze({ throttle: 0, speed: 0, air: 0, pitch: 0, yaw: 0, roll: 0 });
// visual only: the P13/P14 plumes leave the model's nozzle bells (ship.js glow 0/1 x 9); physics keeps the spec's arms
const PLUME_AT = Object.freeze({ P13: Object.freeze([-17.46, -0.09, 4.14]), P14: Object.freeze([-17.46, -0.09, -4.14]) });

export function createShipView(scene, { seed = 1 } = {}) {
  const ship = createShip({ seed });
  ship.group.scale.setScalar(SCALE);
  scene.add(ship.group);
  const exhaust = ship.group.getObjectByName('exhaust'), light = ship.group.getObjectByName('engineLight'), glow = ship.group.getObjectByName('glow');
  const iPos = glow.geometry.getAttribute('iPos'), iCol = glow.geometry.getAttribute('iCol');
  const level = new Float32Array(JETS.length);
  return {
    group: ship.group, level,
    // sim: { x, q, onTimes }; dt: real seconds since the last frame (the puffs animate in real time)
    update(sim, dt) {
      ship.group.position.set(sim.x[0], sim.x[1], sim.x[2]);
      ship.group.quaternion.set(sim.q[0], sim.q[1], sim.q[2], sim.q[3]);
      ship.update(dt, REST);
      exhaust.visible = false;
      light.intensity = 0;
      const on = sim.onTimes;
      for (let j = 0; j < JETS.length; j++) {
        const want = Math.min(1, (on[j] || 0) / CYCLE), k = 1 - Math.exp(-dt / (want > level[j] ? 0.018 : 0.08));
        level[j] += (want - level[j]) * k;
      }
      GLOW_SLOTS.forEach((j, slot) => {
        const jet = JETS[j], p = PLUME_AT[jet.name] || jet.pos, v = level[j], off = 0.05 + 0.15 * v, k = 2.4 * v;
        iPos.setXYZ(slot, p[0] / SCALE + jet.d[0] * off, p[1] / SCALE + jet.d[1] * off, p[2] / SCALE + jet.d[2] * off);
        iCol.setXYZW(slot, 0.85 * k, 0.93 * k, k, v > 0.003 ? (0.1 + 0.24 * v) * SCALE : 0);
      });
      iPos.needsUpdate = true;
      iCol.needsUpdate = true;
    },
    dispose() { ship.dispose(); },
  };
}
