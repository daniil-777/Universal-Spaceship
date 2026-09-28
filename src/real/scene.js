// The real-spacecraft view. Pass 1: the space.js backdrop (Earth, Sun, Moon, stars) through a rotation-only camera at
// its origin (setYaw(0): its axes are LVLH). Pass 2 (clear false, clearDepth true): the metric LVLH scene in metres,
// the ship x9 and the Mir-class station; without the split, anything 92 m below V-bar would render inside the globe.
// The sun light copies the backdrop's true Sun (and its eclipse); the stars and planets turn by +n t about Z, so they
// stay inertial while the LVLH frame turns at -n (space.update leaves their rotation and positions alone).
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { createSpace } from '../space.js';
import { createSatellites } from '../satellites.js';
import { createShipView } from './shipview.js';
import { N, SHIP_PORT, STATION_PORT } from './consts.js';

// station long axis (model +y) along LVLH +X, aft port at -X; r 13.2 = model scale 6 (satellites.js: k = r / 2.2)
export const STATION_Q = Object.freeze([0, 0, -Math.SQRT1_2, Math.SQRT1_2]);
export const STATION_R = 13.2;
export const VIEWS = Object.freeze(['chase', 'port', 'wide']);
const Z = new THREE.Vector3(0, 0, 1);

// resolved by content, not by child index (space.js is edited by other work): the star Points' group, the planets
export const findSky = (root) => root.children.find((c) => c.isGroup && c.children.some((o) => o.isPoints)) || null;
export const findPlanets = (root) => root.children.filter((c) => c.isGroup && c.children.some((o) => o.material && o.material.uniforms && o.material.uniforms.uSpin));

export function createRealScene(canvas, { seed = 1 } = {}) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(1.5, window.devicePixelRatio || 1));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  const bgScene = new THREE.Scene(), scene = new THREE.Scene();
  const bgCam = new THREE.PerspectiveCamera(45, 1, 0.5, 9000), camera = new THREE.PerspectiveCamera(45, 1, 0.5, 20000);
  const space = createSpace(bgScene, { seed, texturePath: 'textures/' });
  space.setYaw(0);
  const sky = findSky(space.root), planets = findPlanets(space.root), planet0 = planets.map((g) => g.position.clone());
  const sun = new THREE.DirectionalLight(0xfff2e0, 2.4), fill = new THREE.HemisphereLight(0x2a3a5a, 0x1c2c48, 0.35);
  scene.add(sun, sun.target, fill);
  const before = new Set(scene.children);
  const sats = createSatellites(scene, { max: 1 });
  sats.update([{ p: [0, 0, 0], q: STATION_Q, r: STATION_R }]);
  const stationMeshes = scene.children.filter((c) => !before.has(c));
  const ship = createShipView(scene, { seed });
  const composer = new EffectComposer(renderer), fg = new RenderPass(scene, camera);
  fg.clear = false;
  fg.clearDepth = true;
  composer.addPass(new RenderPass(bgScene, bgCam));
  composer.addPass(fg);
  composer.addPass(new UnrealBloomPass(new THREE.Vector2(256, 256), 0.45, 0.4, 1.1));
  composer.addPass(new OutputPass());

  let view = 'chase', tReal = 0, warp = 1;
  const eye = new THREE.Vector3(), look = new THREE.Vector3(), sp = new THREE.Vector3(), q = new THREE.Quaternion(), portW = new THREE.Vector3();
  const stationPort = new THREE.Vector3(...STATION_PORT);
  function placeCamera(sim) {
    sp.set(sim.x[0], sim.x[1], sim.x[2]);
    q.set(sim.q[0], sim.q[1], sim.q[2], sim.q[3]);
    portW.set(...SHIP_PORT).applyQuaternion(q).add(sp);
    const r = sp.length();
    if (view === 'port') { eye.copy(stationPort).add(new THREE.Vector3(-3, 2.2, 1.5)); look.copy(portW); }
    else if (view === 'wide') { eye.set(sp.x * 0.5, Math.max(400, r * 0.6), Math.max(600, r * 0.9)); look.set(sp.x * 0.5, 0, 0); }
    else {
      eye.set(-58, 18, 24).applyQuaternion(q).add(sp);
      if (r < 320) look.copy(sp).lerp(stationPort, 0.5); else look.copy(sp).addScaledVector(sp.clone().negate().normalize(), 80);
    }
    camera.position.copy(eye);
    camera.up.set(0, 1, 0);
    camera.lookAt(look);
    bgCam.quaternion.copy(camera.quaternion);
    bgCam.position.set(0, 0, 0);
    bgCam.fov = camera.fov;
    bgCam.updateProjectionMatrix();
  }
  return {
    renderer, camera, space, ship, debug: { sky, planets, planet0, stationMeshes, sun, space },
    setView(v) { if (VIEWS.includes(v)) view = v; return view; },
    get view() { return view; },
    // the backdrop's sky clock runs at the sim's warp, so the Sun and the Earth keep sim time
    setWarp(w) { if (w !== warp) { warp = w; space.setWarp(w); } },
    resize(w, h) {
      renderer.setSize(w, h, false); composer.setSize(w, h);
      camera.aspect = bgCam.aspect = w / h; camera.updateProjectionMatrix(); bgCam.updateProjectionMatrix();
    },
    // sim: the real sim; dtReal: real seconds of this frame; tSim: sim seconds since the start
    frame(sim, dtReal, tSim) {
      tReal += dtReal;
      if (sky) sky.rotation.z = N * tSim;
      planets.forEach((g, i) => g.position.copy(planet0[i]).applyAxisAngle(Z, N * tSim));
      placeCamera(sim);
      space.update(tReal, dtReal, bgCam);
      sun.position.copy(space.sunDirWorld).multiplyScalar(1000);
      sun.intensity = 2.4 * space.sunLit;
      ship.update(sim, dtReal);
      composer.render();
    },
    dispose() { composer.dispose(); ship.dispose(); sats.dispose(); space.dispose(); renderer.dispose(); },
  };
}
