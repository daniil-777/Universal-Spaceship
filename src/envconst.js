// The environment's constants: corridor, ship, hazards, sensors, reward weights, observation layout. Shared by env.js,
// sensors.js, guards.js and the atmosphere modules (aero.js, weather.js) without import cycles.
const DEG = Math.PI / 180;
export const ENV = Object.freeze({
  dt: 1 / 15, substeps: 4,                           // one policy decision per 1/15 s, physics at 60 Hz
  xHalf: 60, yHalf: 22, zHalf: 18,
  ship: { radius: 1.2, cruise: 14, maxSpeed: 22, thrust: 18, assist: 1.5, rateMax: [2.0, 1.2, 1.6], rateTau: 0.15, wallClamp: -1, actuatorTau: 0.1 },   // rateMax = [roll, yaw, pitch] rad/s commanded by the policy, tracked with a 0.15 s time constant; the ship is held 1 unit inside the walls
  belt: { nMin: 10, nMax: 40, rMin: 1.0, rMaxBase: 3.0, rMaxLevel: 2.0, alpha: 2.3, driftMin: 2, driftMax: 7, shear: 0.06,
    ouTheta: 0.4, ouSigma: 1.2, spinMin: 0.1, spinMax: 0.9, restitution: 0.8, nShapes: 12 },
  comets: { nMax: 4, speedMin: 10, speedMax: 20, rMin: 1.0, rMax: 1.8 },
  satellites: { speedMin: 1.0, speedMax: 3.5, rMin: 1.3, rMax: 1.7 },   // space stations (kind 2): the hazard sphere encloses the whole station
  planes: { speedMin: 5, speedMax: 9, r: 1.6 },
  birds: { speedMin: 2.5, speedMax: 5.5, rMin: 1.3, rMax: 2.0, turnTheta: 1.2, turnSigma: 0.9 },   // flocks (kind 4): slow wandering bodies; a bird strike is a crash
  mountains: { amplitude: 40, y0: -26, marchStep: 1.5, ceiling: 12 },   // summits reach y = +14: they cross the flight band, and thin air caps the ship at the ceiling
  rays: { nAz: 11, nEl: 5, azSpan: 150 * DEG, elSpan: 80 * DEG, range: 50, cone: 8 * DEG },   // each ray is a narrow cone so nothing slips between rays
  nearestK: 6, maxSteps: 900,                      // the K asteroids with the shortest time-to-contact
  reward: { progress: 0.10, heading: 0.02, level: 0.02, rate: 0.04, jerk: 0.05, effort: 0.003, speed: 0.05, proximity: 0.05, dSafe: 6,
    wall: 0.10, wallMargin: 6, collision: 20,
    air: { stall: 0.5, comfort: 0.02, rough: 0.02, power: 0.005, overspeed: 0.05, jerk: 0.07, angAcc: 0.0015 } },   // atmosphere only: stall time, a bumpy ride, rough air, fuel, overspeed, and smooth flying (commands and body rates that change gently)
});
export const ACT_DIM = 4;                               // [pitch, yaw, roll, throttle], raw Gaussian samples; the env applies tanh
export const N_RAYS = ENV.rays.nAz * ENV.rays.nEl;
export const OBS_BASE = N_RAYS + ENV.nearestK * 7 + 18;   // 115: the space-era observation
export const N_AIR = 10;                                  // air data: airspeed, α, β, n − 1, climb through the air, wind (3, body), felt turbulence, density
export const OBS_DIM = OBS_BASE + N_RAYS + N_AIR;         // 180: + weather radar per beam + air data (zeros in space)

// Ray directions in the body frame (forward +x, up +y, right +z), elevation-major order.
export const RAY_DIRS_BODY = (() => {
  const { nAz, nEl, azSpan, elSpan } = ENV.rays, out = new Float64Array(nAz * nEl * 3);
  for (let e = 0; e < nEl; e++) for (let a = 0; a < nAz; a++) {
    const el = -elSpan / 2 + (nEl > 1 ? e * elSpan / (nEl - 1) : 0), az = -azSpan / 2 + (nAz > 1 ? a * azSpan / (nAz - 1) : 0), i = (e * nAz + a) * 3;
    out[i] = Math.cos(el) * Math.cos(az); out[i + 1] = Math.sin(el); out[i + 2] = Math.cos(el) * Math.sin(az);
  }
  return out;
})();
