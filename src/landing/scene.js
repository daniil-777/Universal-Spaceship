// The landing scene: its own renderer (logarithmic depth — 80 km of land and centimetre paint), a physical sky with the
// sun where the time of day puts it (runway 26 points WSW: the dusk approach flies into the sunset), light and haze or
// fog by the visibility, the ground, the airport, its lights, the spaceplane on its rig with the sun's shadow following it
// down to the runway, the planned approach drawn as a magenta path with gates, bloom for the lights, and the cameras:
// chase, cockpit, tower cab, runway side, under the approach, and a cinematic cut between them.
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { Sky } from 'three/addons/objects/Sky.js';
import { createShip } from '../ship.js';
import { AIRPORT, RWY } from './airport.js';
import { createAirport } from './airportmesh.js';
import { createGround } from './ground.js';
import { createAirfieldLights } from './lights.js';
import { createRig } from './gear.js';
import { createCloudDeck } from './clouds.js';
import { createAirfieldSigns } from './signs.js';
import { createRain } from './rain.js';
import { VEH, FT } from './vehicle.js';
import { poseAt } from './dubins.js';
import { glidePathHeight } from './nav.js';

const DEG = Math.PI / 180, clamp1 = (v) => Math.max(-1, Math.min(1, v));
const TIMES = { day: { el: 38, az: 160, sun: 3.2, hemi: 1.0, night: 0, cloud: 1 }, dusk: { el: 2.5, az: 250, sun: 1.3, hemi: 0.35, night: 0.6, cloud: 0.62 }, night: { el: -16, az: 300, sun: 0.04, hemi: 0.06, night: 1, cloud: 0.05 } };
const VIS = { cavok: 0.000028, haze: 0.00008, fog: 0.0016 };
const dirFromCompass = (az, el, out) => { const a = (az - RWY.heading) * DEG, e = el * DEG; return out.set(Math.cos(e) * Math.cos(a), Math.sin(e), Math.cos(e) * Math.sin(a)); };

export function createLandingScene(canvas, { time = 'day', vis = 'cavok', clouds = null, rain = false } = {}) {   // clouds: { cover, base, top } (m) or null
  const T = TIMES[time] || TIMES.day, fogD = VIS[vis] ?? VIS.cavok;
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, logarithmicDepthBuffer: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1)); renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = time === 'night' ? 1.1 : time === 'dusk' ? 0.9 : 1.0;
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(50, 1, 0.5, 150000);
  // sky, sun, environment
  const sunDir = dirFromCompass(T.az, T.el, new THREE.Vector3()), sky = new Sky(), U = sky.material.uniforms; sky.scale.setScalar(60000);
  U.turbidity.value = vis === 'cavok' ? 3 : 8; U.rayleigh.value = time === 'dusk' ? 2.5 : 1.2; U.mieCoefficient.value = 0.005; U.mieDirectionalG.value = 0.8; U.sunPosition.value.copy(sunDir); scene.add(sky);
  const pm = new THREE.PMREMGenerator(renderer), envScene = new THREE.Scene(), envSky = new Sky(); envSky.scale.setScalar(1000); envSky.material.uniforms.sunPosition.value.copy(sunDir); envScene.add(envSky);
  scene.environment = pm.fromScene(envScene).texture; scene.environmentIntensity = time === 'night' ? 0.05 : time === 'dusk' ? 0.45 : 0.9;
  const fogColor = new THREE.Color(time === 'night' ? 0x0b1220 : time === 'dusk' ? 0xc79a7a : vis === 'fog' ? 0xb9bec4 : 0xb6c9dc);
  scene.fog = new THREE.FogExp2(fogColor, fogD); if (vis === 'fog' || time === 'night') { scene.background = fogColor; sky.visible = false; }
  const sun = new THREE.DirectionalLight(time === 'dusk' ? 0xffb37a : time === 'night' ? 0x8aa6ff : 0xfff4e6, T.sun); sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048); Object.assign(sun.shadow.camera, { left: -70, right: 70, top: 70, bottom: -70, near: 1, far: 900 }); sun.shadow.camera.updateProjectionMatrix(); sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.6;
  scene.add(sun, sun.target, new THREE.HemisphereLight(time === 'night' ? 0x223355 : 0xbfd7ff, 0x3b3a2c, T.hemi));
  if (time === 'night') { const n = 1800, p = new Float32Array(n * 3); for (let i = 0; i < n; i++) { const u = Math.random() * 2 - 1, a = Math.random() * 6.28, r = Math.sqrt(1 - u * u); p.set([r * Math.cos(a) * 50000, Math.abs(u) * 50000, r * Math.sin(a) * 50000], i * 3); }
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(p, 3)); scene.add(new THREE.Points(g, new THREE.PointsMaterial({ color: 0x6f7788, size: 1.0, sizeAttenuation: false, fog: false }))); }
  // the world
  const ground = createGround(AIRPORT), airport = createAirport(AIRPORT, { night: T.night, wet: rain }), lights = createAirfieldLights(AIRPORT);
  scene.add(ground.group, airport.group, lights.points, createAirfieldSigns(AIRPORT, { night: T.night }).group);
  const deck = clouds ? createCloudDeck(clouds) : null, cloudFog = new THREE.Color(0xc4c8cf).multiplyScalar(Math.max(0.08, T.cloud)), _wa = new Float64Array(3); if (deck) scene.add(deck.group);
  const rainFx = rain ? createRain() : null, camPrev = new THREE.Vector3(), camVel = new Float64Array(3), _wr = new Float64Array(3); if (rainFx) scene.add(rainFx.lines);
  const ship = createShip({ seed: 1 }), rig = createRig(ship.group); scene.add(rig.rig); rig.rig.traverse((o) => { if (o.isMesh) o.castShadow = true; });
  // the planned approach: the Dubins leg (descending at ≤ 3.5° to 2000 ft), level, then the glideslope — magenta, gates every 400 m
  const pathMat = new THREE.LineBasicMaterial({ color: 0xff3df2, transparent: true, opacity: 0.85, fog: false }), gateMat = new THREE.LineBasicMaterial({ color: 0xff3df2, transparent: true, opacity: 0.55, fog: false });
  const pathGroup = new THREE.Group(); scene.add(pathGroup);
  function setPath(gnc, h0) {
    pathGroup.clear(); const st = gnc.st, P = gnc.params, pts = [], q = {}, prof = (s, x, z) => Math.min(Math.max(P.hInt, h0 - s * Math.tan(3.5 * DEG)), glidePathHeight(x, z));
    for (let s = 0; s <= st.dub.length; s += 60) { poseAt(st.dub, s, q); pts.push(new THREE.Vector3(q.x, prof(s, q.x, q.z) + 8, q.z)); }
    for (let x = st.xJ, s = st.dub.length; x <= AIRPORT.ils.gs.x; x += 60, s += 60) pts.push(new THREE.Vector3(x, prof(s, x, 0) + 8, 0));
    pathGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), pathMat));
    for (let x = -400; x >= st.xJ; x -= 400) { const h = Math.min(P.hInt, glidePathHeight(x, 0)) + 8, w = 18 + (-x) * 0.006, e = 12 + (-x) * 0.004;
      pathGroup.add(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(x, h - e, -w), new THREE.Vector3(x, h - e, w), new THREE.Vector3(x, h + e, w), new THREE.Vector3(x, h + e, -w)]), gateMat)); }
  }
  // post
  const composer = new EffectComposer(renderer); composer.addPass(new RenderPass(scene, camera));
  composer.addPass(new UnrealBloomPass(new THREE.Vector2(256, 256), time === 'night' ? 0.7 : time === 'dusk' ? 0.3 : 0.15, 0.4, time === 'night' ? 0.55 : time === 'dusk' ? 0.85 : 0.8)); composer.addPass(new OutputPass());   // only lamps and the sun bloom
  function resize(w, h) { renderer.setSize(w, h, false); composer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix(); }
  // cameras
  let view = 'cinema', cut = 'chase', cutT = 0; const look = new THREE.Vector3(), eye = new THREE.Vector3(), fwd = new THREE.Vector3(), tmp = new THREE.Vector3(), yawFix = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, -Math.PI / 2, 0));
  const FIXED = { tower: [2060, 58, 470], runway: [470, 2.6, -95], approach: [-1900, 1.8, 42] }, _w = new Float64Array(3);
  function place(s, dt, which) {
    const P = rig.rig.position; fwd.set(s.v[0], 0, s.v[2]); if (fwd.lengthSq() < 1) fwd.set(1, 0, 0); fwd.normalize();
    if (which === 'cockpit') { eye.set(VEH.eye[0], VEH.eye[1], VEH.eye[2]).applyQuaternion(rig.rig.quaternion).add(P); camera.position.copy(eye); camera.quaternion.copy(rig.rig.quaternion).multiply(yawFix); camera.fov = 62; }
    else if (FIXED[which]) { camera.position.set(...FIXED[which]); camera.lookAt(P); const d = camera.position.distanceTo(P); camera.fov = THREE.MathUtils.clamp(2 * Math.atan(70 / d) / DEG, 3, 60); }
    else { tmp.copy(P).addScaledVector(fwd, -78); tmp.y += 16; if (camera.userData.v !== which || camera.position.distanceToSquared(tmp) > 160000) camera.position.copy(tmp); camera.position.lerp(tmp, 1 - Math.exp(-dt * 2.5)); if (camera.position.y < 2) camera.position.y = 2; look.copy(P).addScaledVector(fwd, 30); camera.lookAt(look); camera.fov = 50; }   // a new view, or the ship far off (a restart): cut, don't fly across
    camera.userData.v = which; camera.updateProjectionMatrix();
  }
  function cinematic(s, dt) {                               // cut by phase: chase → under the approach → runway side → tower → chase
    const ft = s.air.hRA / FT, want = s.wow ? (Math.hypot(s.v[0], s.v[2]) > 25 ? 'runway' : 'tower') : ft > 1400 ? 'chase' : s.p[0] > -1700 ? 'runway' : 'approach';
    cutT += dt; if (want !== cut && cutT > 4) { cut = want; cutT = 0; } return cut;
  }
  return {
    renderer, scene, camera, rig, ship, lights, airport, resize, setPath, setView(v) { view = v; }, get view() { return view; }, showPath(v) { pathGroup.visible = v; },
    frame(sim, dt, t) {
      const s = sim.flight, w = sim.wind.mean(10, _w);
      rig.update(s, dt, w, scene, T.night); ship.update(dt, { throttle: s.spool, speed: 10, pitch: clamp1(s.w[2] / 0.08), roll: clamp1(s.w[0] / 0.2), yaw: clamp1(s.w[1] / 0.08), air: 1 });
      airport.update(t, w); sun.position.copy(rig.rig.position).addScaledVector(sunDir, 400); sun.target.position.copy(rig.rig.position);
      place(s, dt, view === 'cinema' ? cinematic(s, dt) : view);
      let fogNow = fogD;
      if (deck) {                                             // the deck drifts with the wind aloft; inside it the fog closes in; beneath a broken or overcast deck the light is diffuse
        deck.update(dt, sunDir, sun.color, T.cloud, sim.wind.mean(deck.base, _wa));
        const k = deck.inside(camera.position.y) * (deck.cover >= 0.7 ? 1 : 0.35 * deck.cover / 0.45), shade = deck.cover >= 0.7 ? deck.cover * THREE.MathUtils.smoothstep(deck.top - rig.rig.position.y, -30, 30) : 0;
        fogNow = fogD + (0.012 - fogD) * k; scene.fog.density = fogNow; scene.fog.color.copy(fogColor).lerp(cloudFog, k);
        sun.intensity = T.sun * (1 - 0.85 * shade); sun.shadow.intensity = 1 - 0.9 * shade;
      }
      if (rainFx) {                                         // the streaks follow the camera's own motion (the ship's, or none on a fixed camera)
        for (let i = 0; i < 3; i++) camVel[i] = THREE.MathUtils.clamp((camera.position.getComponent(i) - camPrev.getComponent(i)) / Math.max(dt, 1e-3), -150, 150);
        camPrev.copy(camera.position); rainFx.update(dt, camera.position, camVel, sim.wind.mean(20, _wr), deck ? deck.base : 3000);
      }
      lights.update(t, T.night, fogNow, renderer.domElement.height);
      composer.render();
    },
  };
}
