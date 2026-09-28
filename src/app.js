// Astro Pilot — page glue: three.js scene + post chain, the displayed flight (a SpaceEnv driven by the policy or the
// keyboard), the camera rig, HUD, and the in-browser PPO training loop. Test hooks live on window.__ap.
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { SMAAPass } from 'three/addons/postprocessing/SMAAPass.js';
import { CloudPass, createCloudShadows } from './clouds.js';
import { createDroplets } from './droplets.js';
import { createSpace } from './space.js';
import { createAsteroidField } from './asteroids.js';
import { createShip } from './ship.js';
import { createComets } from './comets.js';
import { createSatellites } from './satellites.js';
import { createPlanes } from './planes.js';
import { createBirds } from './birds.js';
import { createTerrain, TERRAIN_ATTRIBUTION, ROUTES } from './terrain.js';
import { createMountains } from './mountains.js';
import { createCity } from './city.js';
import { CITY_NAMES, registerLongGrid, hasLongGrid } from './cityfield.js';
const LONG_GRIDS = { mega: () => import('./mega_grid.js'), avatar: () => import('./avatar_grid.js') };   // long worlds' baked grids load when their route is chosen (~460 KB each)
import { createShadowField } from './shadowfield.js';
import { createMeshyWorld } from './meshyworld.js';
import { MESHY_SETS } from './meshylayout.js';
import { createCameraRig } from './camera.js';
import { MonoShader } from './post.js';
import { SpaceEnv, ENV, OBS_DIM, OBS_BASE, ACT_DIM, N_RAYS } from './env.js';
import { SKIES } from './weather.js';
import { PPOAgent } from './ppo.js';
import { evaluatePolicy } from './trainer.js';
import { createTrainClient } from './train-client.js';
import { createUI } from './ui.js';
import { createBoard } from './board.js';
import { createPipCam } from './pipcam.js';
import { createEdgeFence } from './edgefence.js'; import { attachEarthZoom } from './earthzoom.js';
import { wrapX } from './mathx.js';

const TFJS = { url: 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js', sri: 'sha384-vE8hbVJ4lezako5rlvE7bY0BVzWlFhZncPlckrqNwcUQpVtgbENTgZ8TBbnPjZre' };
const qs = new URLSearchParams(location.search);
const num = (k, d) => (qs.has(k) ? parseFloat(qs.get(k)) : d);
const ap = (window.__ap = { ready: false, frames: 0, fps: 0, errors: [], policy: null, crashes: 0, trainer: null, renderer: '' });
window.addEventListener('error', (e) => ap.errors.push(String(e.message || e)));
window.addEventListener('unhandledrejection', (e) => ap.errors.push(String((e.reason && e.reason.message) || e.reason)));
if (qs.get('hud') === '0') document.body.classList.add('nohud');

// ---------- state ----------
const state = { playing: true, simSpeed: num('speed', 1), density: Math.round(num("density", 25)), astSpeed: 1, comets: Math.round(num("comets", 2)), cometSpeed: 1, manual: false, trail: true, sensors: false, mode: 0, invert: false, camera: qs.get('camera') || 'chase',
  crashTimer: 0, flightTime: 0, laps: 0, fps: 60, training: false, trainSpeed: 'balanced', curriculum: qs.get('curriculum') !== '0',
  lowPasses: qs.get('lowpass') !== '0', lowPass: qs.get('lowpass') === '1', phaseTimer: qs.get('lowpass') === '1' ? 0 : 40, atmo: false, route: ROUTES[qs.get('route')] ? qs.get('route') : 'alps', skyline: qs.get('skyline') === '1' || (qs.get('skyline') !== '0' && !!(ROUTES[qs.get('route')] && ROUTES[qs.get('route')].city)) };   // skyline = fly the route's city among its towers   // orbit ↔ low pass cycle; atmo = atmospheric flight flag
let agent = null, ui = null, train = null;
let board = null, pip = null, zoom = null;
const envMaps = { space: null, day: null, bake: null, isDay: false };   // image-based light: space, or a daylight sky once the descent is half done                                 // the flight board and its onboard camera
let atmoAgent = null, atmoLoad = null, singlePilot = false;   // the atmospheric pilot (model/policy_atmo.json) flies the atmosphere; one pilot only once the user trains, loads or resets a policy
const env = new SpaceEnv(Math.round(num('seed', 11)), { level: 1, count: state.density, speedScale: state.astSpeed, comets: state.comets, cometSpeed: state.cometSpeed });
const knob = (k, d, lo, hi) => { const v = parseFloat(qs.get(k)); return Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : d; }, sky0 = SKIES[Object.hasOwn(SKIES, qs.get('sky')) ? qs.get('sky') : 'fair'];
const skyOf = () => Object.keys(SKIES).find((k) => SKIES[k].every((v, i) => Math.abs(v - [env.weatherSeverity, env.weatherWind, env.weatherCover, env.weatherTurb][i]) < 1e-6)) || '', wxSet = (s, o) => { env.setWeather(s, o); ui.setControls({ sky: skyOf() }); };
env.setWeather(knob('weather', sky0[0], 0, 1), { wind: knob('wind', sky0[1], 0, 2), cover: knob('cover', sky0[2], 0, 2), turb: knob('turb', sky0[3], 0, 3) });   // ?sky=clear|fair|cloudy|storm, then ?weather ?wind ?cover ?turb
 state.autoThr = qs.get('autothr') !== '0';
const action = new Float32Array(ACT_DIM), obsN = new Float32Array(OBS_DIM), actTanh = new Float32Array(ACT_DIM);

// ---------- renderer / scene ----------
const canvas = document.getElementById('view');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(1.5, window.devicePixelRatio || 1));   // 1.5× is visually enough on Retina and ~45 % fewer pixels than 2×
renderer.outputColorSpace = THREE.SRGBColorSpace; renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.0;
try { const gl = renderer.getContext(), ext = gl.getExtension('WEBGL_debug_renderer_info'); ap.renderer = ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : 'webgl'; } catch (e) { /* ignore */ }
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x000000);
const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 9000);
const sceneTarget = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType }); sceneTarget.depthTexture = new THREE.DepthTexture(1, 1);   // the clouds read the scene's depth
const composer = new EffectComposer(renderer, sceneTarget); composer.renderTarget2.depthTexture = new THREE.DepthTexture(1, 1);   // its own: the composer's clone shares the first one's image — one GL texture, a feedback loop   // no MSAA target: multisampled HalfFloat targets render black bands on ANGLE/Metal; SMAA below instead
const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.55, 0.4, 1.1);   // threshold above sunlit white paint: only the sun, exhaust, comas and accents bloom
const monoPass = new ShaderPass(MonoShader); monoPass.enabled = false;
const smaa = new SMAAPass(1, 1); smaa.enabled = qs.get('smaa') !== '0' && renderer.getPixelRatio() < 1.4;   // at 1.5× the supersampling already smooths edges
composer.addPass(new RenderPass(scene, camera)); composer.addPass(bloom); composer.addPass(new OutputPass()); composer.addPass(smaa); composer.addPass(monoPass);
const clouds = new CloudPass(camera); clouds.forceOff = qs.get('clouds') === '0'; composer.insertPass(clouds, 1);   // after the scene, before bloom: silver linings glow; atmosphere only
const droplets = createDroplets(scene);                   // streaks past the camera inside a cloud
function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h, false); composer.setSize(w, h); bloom.resolution.set(w, h);
  camera.aspect = w / h; camera.updateProjectionMatrix();
  monoPass.uniforms.cell.value = 2 * Math.max(1, Math.round(renderer.getPixelRatio()));   // ink dither cells of 2 CSS px
}
window.addEventListener('resize', resize); resize();

// ---------- world ----------
let space, field, ship, ghost, rig, comets, fill, sats, planes, birds, terrain, mountains, city, chunks, mountainsStyle = null;
const shadow = createShadowField(); const cloudShadows = createCloudShadows(shadow.uniforms);                        // the active world's height field on the GPU: towers and ridges cast real shadows
clouds.setShadow(shadow.uniforms);                         // the valley's walls shade the clouds too
const LOW_PASS = { orbitSeconds: 45, lowSeconds: 28, satellites: 5, planes: 6, pitch: -3.5 * Math.PI / 180 }, ATMO_BIRDS = 3;   // the atmosphere's hazard density = the pilot's evaluation setting
const CAM_FOV = { side: 42, chase: 50, orbit: 46, moon: 1.2 }, CAM_YAW = { side: 0, chase: -Math.PI / 4, orbit: null, moon: null }, FILL_DIR = { side: [0.35, 0.6, 1.0], chase: [-0.8, 0.55, 0.25], orbit: [-0.3, 0.7, 0.8], moon: [-0.8, 0.55, 0.25] };   // moon: a 1.2° lens, the Moon (0.52°) fills ~40 % of the frame
const fillTarget = new THREE.Vector3(0.35, 0.6, 1.0);
function setCameraMode(c) { state.camera = c; rig.setMode(c); camera.fov = CAM_FOV[c]; camera.updateProjectionMatrix(); if (CAM_YAW[c] !== null) space.setYaw(CAM_YAW[c]); fillTarget.set(...FILL_DIR[c]); if (c === 'moon' && ui) { const tl = space.telescope; ui.toast(`the ${tl.name} through a ${tl.lens || tl.fov}° lens — true size and direction, ${Math.round(100 * (tl.lit ?? 0))} % lit (in orbit)`, 4000); } }
const trailN = 220, trailPos = new Float32Array(trailN * 3); let trailLen = 0, trailHead = 0;
const trailGeom = new THREE.BufferGeometry(), trailArr = new Float32Array(trailN * 2 * 3), trailCol = new Float32Array(trailN * 2 * 3);
trailGeom.setAttribute('position', new THREE.BufferAttribute(trailArr, 3)); trailGeom.setAttribute('color', new THREE.BufferAttribute(trailCol, 3));
const trailLine = new THREE.LineSegments(trailGeom, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }));
const rayGeom = new THREE.BufferGeometry(), rayArr = new Float32Array(N_RAYS * 2 * 3), rayCol = new Float32Array(N_RAYS * 2 * 3);
rayGeom.setAttribute('position', new THREE.BufferAttribute(rayArr, 3)); rayGeom.setAttribute('color', new THREE.BufferAttribute(rayCol, 3));
const rayLines = new THREE.LineSegments(rayGeom, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.85, blending: THREE.AdditiveBlending, depthWrite: false }));
rayLines.visible = false; trailLine.frustumCulled = false; rayLines.frustumCulled = false;
const accent = new THREE.Color(0x8ec5ff), edgeTint = new THREE.Color(1.0, 0.62, 0.22), stormTint = new THREE.Color(1.0, 0.42, 0.36);   // beams that meet a corridor edge turn amber
let fence = null;                                           // the corridor's edges, shown where the ship meets one (src/edgefence.js)

// interpolation buffers
const shipPrev = { p: new Float64Array(3), q: new Float64Array([0, 0, 0, 1]) };
let astPrev = new Float64Array(0), astPrevQ = new Float64Array(0), astPrevGen = new Int32Array(0);
const renderList = [], cometList = [], satList = [], planeList = [], birdList = [];
const shipRender = { p: new THREE.Vector3(), q: new THREE.Quaternion(), f: new THREE.Vector3(1, 0, 0), u: new THREE.Vector3(0, 1, 0) };

function snapshotPrev() {
  shipPrev.p.set(env.ship.p); shipPrev.q.set(env.ship.q);
  const n = env.asteroids.length;
  if (astPrev.length !== n * 3) { astPrev = new Float64Array(n * 3); astPrevQ = new Float64Array(n * 4); astPrevGen = new Int32Array(n); }
  for (let i = 0; i < n; i++) { const a = env.asteroids[i]; astPrev[i * 3] = a.p[0]; astPrev[i * 3 + 1] = a.p[1]; astPrev[i * 3 + 2] = a.p[2]; astPrevQ[i * 4] = a.q[0]; astPrevQ[i * 4 + 1] = a.q[1]; astPrevQ[i * 4 + 2] = a.q[2]; astPrevQ[i * 4 + 3] = a.q[3]; astPrevGen[i] = a.gen || 0; }
}
function pushTrail() { const i = trailHead * 3; trailPos[i] = env.ship.p[0]; trailPos[i + 1] = env.ship.p[1]; trailPos[i + 2] = env.ship.p[2]; trailHead = (trailHead + 1) % trailN; trailLen = Math.min(trailN, trailLen + 1); }
function clearTrail() { trailLen = 0; trailHead = 0; }

// ---------- policy ----------
async function loadPolicy() {
  singlePilot = qs.get('fresh') === '1' || !!qs.get('policy');
  if (qs.get('fresh') === '1') { agent = new PPOAgent(OBS_DIM, ACT_DIM, {}, Math.round(num('seed', 11))); ap.policy = 'fresh'; return; }
  try {
    const r = await fetch(qs.get('policy') || 'model/policy.json', { cache: 'no-cache' }); if (!r.ok) throw new Error('policy ' + r.status);
    const j = await r.json(); agent = PPOAgent.fromJSON(j); ap.policy = `pretrained ${(agent.steps / 1e6).toFixed(1)}M steps`;
  } catch (e) { agent = new PPOAgent(OBS_DIM, ACT_DIM, {}, 11); ap.policy = 'fresh'; ap.policyError = String(e.message || e); }
}
function policyText() { return agent.steps > 0 ? `policy: ${(agent.steps / 1e6).toFixed(2)}M training steps · ${agent.updates} updates${agent.meta && agent.meta.trainedIn ? ' · trained in ' + agent.meta.trainedIn : ''}${atmoAgent && !singlePilot ? ` · atmospheric pilot: ${(atmoAgent.steps / 1e6).toFixed(2)}M steps` : ''}` : 'policy: untrained (random init) — press Train and watch it learn'; }
function activeAgent() { return state.atmo && atmoAgent && !state.training ? atmoAgent : agent; }   // the atmospheric pilot flies among mountains and skylines; the belt pilot flies orbit, and everything while the user trains
function loadAtmoPolicy() {                                // fetched once, when atmospheric flight is first switched on (187 KB, float16)
  if (singlePilot || atmoLoad) return;
  atmoLoad = fetch('model/policy_atmo.json', { cache: 'no-cache' }).then((r) => { if (!r.ok) throw new Error('policy_atmo ' + r.status); return r.json(); })
    .then((j) => { if (singlePilot) return; atmoAgent = PPOAgent.fromJSON(j); ap.policyAtmo = `pretrained ${(atmoAgent.steps / 1e6).toFixed(1)}M steps`; ui.policyInfo(policyText()); })
    .catch((e) => { ap.policyError = String(e.message || e); });
}
function dropAtmoPilot() { singlePilot = true; if (atmoAgent) { atmoAgent.dispose(); atmoAgent = null; } }
function showPretrainingCurves() {                         // the curves of the run that produced the loaded policy, until live training continues them
  const meta = agent && agent.meta; if (!meta || !meta.history || !meta.history.step) return;
  const hist = {}; for (const k in meta.history) hist[k] = meta.history[k].map((v) => (v === null ? NaN : v));
  const m = { ...(meta.finalMetrics || {}), phase: 'pretrained' }; for (const k in hist) if (!(k in m)) m[k] = hist[k][hist[k].length - 1];
  ui.metrics(m, hist, meta.levelChanges || []);
}

// ---------- simulation step (30 Hz) ----------
function manualAction(out) {
  const k = ui.keys, P = 1.6, has = (...c) => c.some((x) => k.has(x));
  out[0] = (has('KeyW', 'ArrowUp') ? P : 0) - (has('KeyS', 'ArrowDown') ? P : 0);
  out[1] = (has('KeyA', 'ArrowLeft') ? P : 0) - (has('KeyD', 'ArrowRight') ? P : 0);
  out[2] = (has('KeyE') ? P : 0) - (has('KeyQ') ? P : 0);
  out[3] = (has('ShiftLeft', 'ShiftRight') ? P : 0) - (has('KeyX', 'ControlLeft', 'ControlRight') ? P : 0);
  if (env.autoThrottle) { env.speedTarget = Math.min(22, Math.max(10, env.speedTarget + out[3] * 0.05)); out[3] = 0; }
}
let lastValue = 0;
function simStep() {
  snapshotPrev();
  if (state.crashTimer > 0) { state.crashTimer -= ENV.dt; action.fill(0); env.step(action); if (state.crashTimer <= 0) { env.reset(); ship.reset(); clearTrail(); snapshotPrev(); } return; }
  const pilot = activeAgent(); pilot.normalize(env.obs, 1, obsN);
  env.autoThrottle = state.manual && state.autoThr && env.atmosphere;   // a person flying through the air: the throttle holds a speed (Shift / Ctrl change it)
  if (state.manual) manualAction(action); else pilot.actMean(obsN, action);
  lastValue = pilot.value(obsN);
  const lapsBefore = env.laps, res = env.step(action);
  if (env.laps !== lapsBefore) state.laps += env.laps - lapsBefore;
  state.flightTime += ENV.dt; pushTrail();
  for (let i = 0; i < ACT_DIM; i++) actTanh[i] = Math.tanh(action[i]);
  if (res.done) { ap.crashes++; state.crashTimer = 1.6; ship.explode(); ghost.setVisible(false); }
}

// ---------- adaptive quality: drop the render scale on slow machines, restore when there is headroom ----------
let slowTicks = 0, fastTicks = 0, renderScale = 1;
function adaptQuality(fps) {
  if (fps < 40) { slowTicks++; fastTicks = 0; } else if (fps > 57) { fastTicks++; slowTicks = 0; } else { slowTicks = 0; fastTicks = 0; }
  if (slowTicks >= 4 && renderScale > 0.6) { renderScale = Math.max(0.6, renderScale - 0.15); slowTicks = 0; applyScale(); }
  else if (fastTicks >= 12 && renderScale < 1) { renderScale = Math.min(1, renderScale + 0.15); fastTicks = 0; applyScale(); }
}
function applyScale() { const base = Math.min(1.5, window.devicePixelRatio || 1); renderer.setPixelRatio(base * renderScale); resize(); ap.renderScale = renderScale; clouds.quality(renderScale); }

// ---------- rendering ----------
const _q1 = new THREE.Quaternion(), _q2 = new THREE.Quaternion();
function updateVisuals(alpha, dt) {
  const s = env.ship, n = env.asteroids.length, side = rig.mode === 'side';
  // ship pose (interpolated; no interpolation across the seam)
  const wrapped = Math.abs(s.p[0] - shipPrev.p[0]) > ENV.xHalf, a = wrapped ? 1 : alpha;
  shipRender.p.set(shipPrev.p[0] + (s.p[0] - shipPrev.p[0]) * a, shipPrev.p[1] + (s.p[1] - shipPrev.p[1]) * alpha, shipPrev.p[2] + (s.p[2] - shipPrev.p[2]) * alpha);
  _q1.set(shipPrev.q[0], shipPrev.q[1], shipPrev.q[2], shipPrev.q[3]); _q2.set(s.q[0], s.q[1], s.q[2], s.q[3]); shipRender.q.copy(_q1).slerp(_q2, alpha);
  shipRender.f.set(1, 0, 0).applyQuaternion(shipRender.q); shipRender.u.set(0, 1, 0).applyQuaternion(shipRender.q);
  ship.group.position.copy(shipRender.p); ship.group.quaternion.copy(shipRender.q);
  const eff = { throttle: state.crashTimer > 0 ? 0 : 0.5 + 0.5 * env.cmd[3], pitch: env.cmd[0], yaw: env.cmd[1], roll: env.cmd[2], speed: Math.hypot(s.v[0], s.v[1], s.v[2]), air: space.atmosphere };   // air: wingtip vapour only in the atmosphere
  ship.update(dt, eff);
  const nearSeam = side && Math.abs(shipRender.p.x) > ENV.xHalf - 14 && state.crashTimer <= 0;
  ghost.setVisible(nearSeam);
  if (nearSeam) { ghost.group.position.copy(shipRender.p); ghost.group.position.x -= Math.sign(shipRender.p.x) * 2 * ENV.xHalf; ghost.group.quaternion.copy(shipRender.q); ghost.update(dt, eff); }
  // asteroids: interpolated, wrapped in side view, minimum-image relative to the ship otherwise
  const sx = shipRender.p.x; let nr = 0, nc = 0, ns = 0, np = 0, nb = 0;
  for (let i = 0; i < n; i++) {
    const A = env.asteroids[i]; let o;
    if (A.kind === 3) { o = planeList[np] || (planeList[np] = { p: [0, 0, 0], q: [0, 0, 0, 1], v: [0, 0, 0], r: 1, gen: 0 }); np++; o.v[0] = A.v[0]; o.v[1] = A.v[1]; o.v[2] = A.v[2]; o.gen = A.gen; }
    else if (A.kind === 4) { o = birdList[nb] || (birdList[nb] = { p: [0, 0, 0], q: [0, 0, 0, 1], v: [0, 0, 0], r: 1, type: 0, phase: 0, gen: 0 }); nb++; o.v[0] = A.v[0]; o.v[1] = A.v[1]; o.v[2] = A.v[2]; o.type = A.type; o.phase = A.phase; o.gen = A.gen; }
    else if (A.kind === 2) { o = satList[ns] || (satList[ns] = { p: [0, 0, 0], q: [0, 0, 0, 1], r: 1 }); ns++; }
    else if (A.kind) { o = cometList[nc] || (cometList[nc] = { p: [0, 0, 0], q: [0, 0, 0, 1], v: [0, 0, 0], r: 1, gen: 0 }); nc++; o.v[0] = A.v[0]; o.v[1] = A.v[1]; o.v[2] = A.v[2]; o.gen = A.gen; }
    else { o = renderList[nr] || (renderList[nr] = { p: [0, 0, 0], q: [0, 0, 0, 1], r: 1, shape: 0 }); nr++; }
    const px = astPrev[i * 3], wr = Math.abs(A.p[0] - px) > ENV.xHalf || (A.gen || 0) !== astPrevGen[i], aa = wr ? 1 : alpha;
    let x = px + (A.p[0] - px) * aa; if (!side) x = sx + wrapX(x - sx, ENV.xHalf);
    o.p[0] = x; o.p[1] = astPrev[i * 3 + 1] + (A.p[1] - astPrev[i * 3 + 1]) * alpha; o.p[2] = astPrev[i * 3 + 2] + (A.p[2] - astPrev[i * 3 + 2]) * alpha;
    let qx = astPrevQ[i * 4] + (A.q[0] - astPrevQ[i * 4]) * alpha, qy = astPrevQ[i * 4 + 1] + (A.q[1] - astPrevQ[i * 4 + 1]) * alpha, qz = astPrevQ[i * 4 + 2] + (A.q[2] - astPrevQ[i * 4 + 2]) * alpha, qw = astPrevQ[i * 4 + 3] + (A.q[3] - astPrevQ[i * 4 + 3]) * alpha;
    const qn = Math.hypot(qx, qy, qz, qw) || 1; o.q[0] = qx / qn; o.q[1] = qy / qn; o.q[2] = qz / qn; o.q[3] = qw / qn; o.r = A.r; o.shape = A.shape;
  }
  renderList.length = nr; field.update(renderList); cometList.length = nc; comets.setSunDir(space.sunDirWorld); comets.update(cometList, dt, camera, performance.now() / 1000); satList.length = ns; sats.update(satList); planeList.length = np; planes.update(planeList, dt); birdList.length = nb; birds.update(birdList, dt, performance.now() / 1000, space.sunDirWorld);
  const apexX = camera.position.x + 10;                                                     // one curvature apex just ahead of the camera, shared by the imagery strips, the mountains and the towers
  const lapShift = env.hf && env.hf.lap ? env.hf.lap.shift : 0; if (env.hf && env.hf.lap) shadow.setShift(lapShift);   // a long world: the corridor's place on its map
  if (terrain) terrain.update(dt, camera, mountains && mountains.ownGround && !city ? 0 : space.atmosphere, apexX);   // the Avatar valley has its own ground: no satellite photo under it
  if (mountains) mountains.update(dt, camera, city ? 0 : space.atmosphere, space.sunDirWorld, apexX, lapShift); if (city) city.update(dt, camera, space.atmosphere, space.sunDirWorld, apexX); if (chunks) chunks.update(dt, camera, space.atmosphere, space.sunDirWorld, apexX, lapShift);
  // trail
  trailLine.visible = state.trail && trailLen > 1 && !(rig.mode === 'chase' || rig.mode === 'moon');   // from the chase seat the trail is behind you (the Moon view falls back to it in the air)
  if (trailLine.visible) {
    const rel = (x) => (side ? x : sx + wrapX(x - sx, ENV.xHalf)); let k = 0;
    for (let j = 0; j < trailLen - 1; j++) {
      const i0 = ((trailHead - trailLen + j + trailN) % trailN) * 3, i1 = ((trailHead - trailLen + j + 1 + trailN) % trailN) * 3;
      const x0 = rel(trailPos[i0]), x1 = rel(trailPos[i1]), broken = Math.abs(x1 - x0) > ENV.xHalf, f = (j + 1) / trailLen, c = 0.55 * f * f;
      trailArr[k] = x0; trailArr[k + 1] = trailPos[i0 + 1]; trailArr[k + 2] = trailPos[i0 + 2];
      trailArr[k + 3] = broken ? x0 : x1; trailArr[k + 4] = broken ? trailPos[i0 + 1] : trailPos[i1 + 1]; trailArr[k + 5] = broken ? trailPos[i0 + 2] : trailPos[i1 + 2];
      for (let m = 0; m < 2; m++) { trailCol[k + m * 3] = accent.r * c; trailCol[k + m * 3 + 1] = accent.g * c; trailCol[k + m * 3 + 2] = accent.b * c; }
      k += 6;
    }
    trailGeom.setDrawRange(0, (trailLen - 1) * 2); trailGeom.attributes.position.needsUpdate = true; trailGeom.attributes.color.needsUpdate = true;
  }
  // sensor rays
  rayLines.visible = state.sensors && state.crashTimer <= 0;
  if (rayLines.visible) {
    const p = env.ship.p, range = ENV.rays.range;
    for (let i = 0; i < N_RAYS; i++) {
      const d = env.rayHit[i], k = i * 6, prox = 1 - d / range, len = Math.min(d, range), dx = env.rayDirWorld[i * 3], dy = env.rayDirWorld[i * 3 + 1], dz = env.rayDirWorld[i * 3 + 2];
      rayArr[k] = shipRender.p.x; rayArr[k + 1] = shipRender.p.y; rayArr[k + 2] = shipRender.p.z;
      rayArr[k + 3] = shipRender.p.x + dx * len; rayArr[k + 4] = shipRender.p.y + dy * len; rayArr[k + 5] = shipRender.p.z + dz * len;
      const c0 = 0.05 + 0.3 * prox, c1 = 0.08 + 0.9 * prox, col = env.rayEdge[i] && state.mode === 0 ? edgeTint : env.obs[OBS_BASE + i] > 0.15 && state.mode === 0 ? stormTint : accent;   // rough cloud on the weather radar: red
      rayCol[k] = col.r * c0; rayCol[k + 1] = col.g * c0; rayCol[k + 2] = col.b * c0; rayCol[k + 3] = col.r * c1; rayCol[k + 4] = col.g * c1; rayCol[k + 5] = col.b * c1;
    }
    rayGeom.attributes.position.needsUpdate = true; rayGeom.attributes.color.needsUpdate = true;
  }
  if (fence) fence.update(dt, shipRender.p, shipRender.f, { zHalf: ENV.zHalf, yHalf: ENV.yHalf, top: env.hf ? ENV.mountains.ceiling : ENV.yHalf, guard: env.guard, atmo: !!env.hf, on: state.crashTimer <= 0 && !side });
  fill.position.lerp(_fillPos.copy(fillTarget).multiplyScalar(500), 1 - Math.exp(-dt * 3));
  rig.pitchExtra = Math.max(space.altitude, 1.6 * space.atmosphere); rig.groundLock = space.atmosphere; rig.shake = env.atmosphere ? Math.min(0.12, env.air.sigmaFelt * 0.1) * space.atmosphere : 0; rig.terrain = env.hf ? (x, z) => env.hf.height(x, z) : null; rig.ceiling = env.hf && env.hf.ceiling ? env.hf.ceiling : null; env.guideOn = !state.manual; rig.tunnel += ((env.guide || 0) - rig.tunnel) * (1 - Math.exp(-dt * 2)); monoPass.uniforms.contrast.value = space.atmosphere; if (space.atmosphere > 0) shadow.setSun(space.sunDirWorld);
  const fovNow = state.camera === 'moon' && space.atmosphere < 0.5 ? space.telescope.fov : CAM_FOV[state.camera === 'moon' ? 'chase' : state.camera] - 9 * space.atmosphere; if (Math.abs(camera.fov - fovNow) > 0.01) { camera.fov = fovNow; camera.updateProjectionMatrix(); }   // a slightly longer lens in the atmosphere: cities read larger
  rig.update(dt, shipRender, camera.aspect);
  const day = space.atmosphere > 0.5;                      // swap the reflections of every PBR surface (glass, hull, airliners, towers) with the sky
  if (envMaps.bake && day !== envMaps.isDay) { envMaps.isDay = day; envMaps.at = -Infinity; envMaps.sun.set(0, 0, 0); scene.environment = day ? envMaps.bake() : envMaps.space.texture; scene.environmentIntensity = day ? 0.6 : 0.7; }
  else if (envMaps.rebake) envMaps.rebake(day);   // the real Sun circles the ship each orbit and hides behind the Earth; the air's sun turns with the camera mode
}
const _fillPos = new THREE.Vector3(), _aim = new THREE.Vector3();

// ---------- low passes: every ~45 s the ship dives toward the Earth for ~28 s and meets satellites ----------
function setLowPass(on) {
  if (state.lowPass === on) return;
  state.lowPass = on; space.setAltitude(on ? 1 : 0); env.setSatellites(on ? LOW_PASS.satellites : 0); snapshotPrev();
  ui.setPhase(on ? 'low pass' : 'orbit'); ui.toast(on ? 'low pass — dropping toward the Earth · satellites ahead' : 'climbing back to orbit', 3000);
  state.phaseTimer = on ? LOW_PASS.lowSeconds : LOW_PASS.orbitSeconds;
}
function setBody(b) {                                    // the orbit's body: 'earth' (the ISS's orbit) or 'moon' (a polar lunar orbit, the Earth a far globe); routes fly over the Earth
  if ((b === 'moon' ? 'moon' : 'earth') === space.body) { ui.setControls({ orbit: space.body }); return; }
  if (b === 'moon') { if (state.atmo) setAtmo(false); setLowPass(false); } space.setBody(b); ui.setControls({ orbit: space.body }); envMaps.pending = true; ui.setPhase(space.body === 'moon' ? 'lunar orbit' : 'orbit');
  ui.toast(space.body === 'moon' ? `lunar orbit · ${Math.round(space.moonAltitude).toLocaleString('en-US')} km above the Moon · the Earth rising ahead` : 'back in the Earth orbit', 3200);
}
function setAtmo(on) {                                    // atmospheric flight: half an airliner's altitude (in this scene's scale) with a bird's-eye view of the planet
  if (state.atmo === on) return;
  state.atmo = on; if (on && state.lowPass) setLowPass(false); if (on && space.body === 'moon') setBody('earth'); ui.setControls({ atmo: on }); env.setAtmosphere(on);   // the air's physics only below the sky
  if (on) space.setGround(ROUTES[state.route].lat, ROUTES[state.route].lon0); space.setAtmosphere(on); scene.fog = on ? new THREE.Fog(0x9fbbd8, 70, 560) : null;   // the orbit brings the ship over the route first
  if (on) {                                                 // the sky belongs to airliners: no rocks, no comets, real planes to avoid
    state.saved = { density: state.density, comets: state.comets };
    env.setComets(0); env.setCount(0); applyWorld(); loadAtmoPolicy();
    if (!terrain) terrain = createTerrain(scene, renderer, { ...ROUTES[state.route], shadow }); ui.setCredit('three.js · TensorFlow.js · ' + TERRAIN_ATTRIBUTION);
  } else {
    if (city) { city.dispose(); city = null; } if (chunks) { chunks.dispose(); chunks = null; }
    shadow.setOn(false); env.setPlanes(0); env.setBirds(0); env.setMountains(false); env.setCount(state.saved ? state.saved.density : state.density); env.setComets(state.saved ? state.saved.comets : state.comets); env.spawnAsteroids(); snapshotPrev(); clearTrail();
    ui.setCredit(null);
  }
  if (!on) ui.setPhase(state.lowPass ? 'low pass' : 'orbit'); ui.toast(on ? 'descending into the atmosphere' : 'climbing back to orbit', 3000);
}
function cityKey() { return state.skyline ? (ROUTES[state.route].city || null) : null; }
let avatarFailed = false;                                  // the Avatar valley's grid would not load: the China route keeps its first pillar clusters
function longKey() { const key = cityKey(); return key || (!key && ROUTES[state.route].terrain === 'meshy' && !avatarFailed ? 'avatar' : null); }
function applyWorld() {                                   // atmospheric hazards by route: a skyline (Skyline flight on), the Alps, the Zhangjiajie pillars, or just the imagery from above with airliners and birds
  const key = cityKey(), route = ROUTES[state.route], terrain = key ? null : (route.terrain || null), long = longKey();
  if (long && LONG_GRIDS[long] && !hasLongGrid(long)) {     // a long world (the megacity, the Avatar valley): fetch its map first, then build the world
    ui.toast(long === 'mega' ? 'loading the megacity…' : 'loading the Avatar valley…', 4000);
    LONG_GRIDS[long]().then((m) => { registerLongGrid(long, m); if (longKey() === long && state.atmo) applyWorld(); },
      (e) => { console.warn(long + ' grid unavailable', e); if (long === 'avatar') { avatarFailed = true; if (state.atmo) applyWorld(); } else ui.toast('the megacity could not be loaded', 4000); });
    return;
  }
  if (key) { env.setCity(key, 1); env.setPlanes(0); }
  else if (terrain === 'pillars') { env.setMountains(false); env.setPillars(true); env.setPlanes(0); }
  else if (terrain === 'meshy') { env.setMountains(false); env.setMeshy(true, long); env.setPlanes(0); }
  else if (terrain === 'alps') { env.setMountains(true); env.setPlanes(LOW_PASS.planes); }
  else { env.setMountains(false); env.setPlanes(LOW_PASS.planes); }
  env.setBirds(terrain === 'pillars' || terrain === 'meshy' ? 4 : ATMO_BIRDS);   // flocks live only in the atmosphere
  env.spawnAsteroids(); env.reset(); ship.reset(); snapshotPrev(); clearTrail();
  if (city) { city.dispose(); city = null; } if (chunks) { chunks.dispose(); chunks = null; }
  const style = terrain === 'meshy' ? 'meshy:' + (env.hf.chunks || 'china') : terrain;   // the China route: the Avatar valley, or the first clusters as a fallback
  if (mountains && mountainsStyle !== style) { mountains.dispose(); mountains = null; }
  if (key) { city = createCity(scene, env.hf, { sunDir: space.sunDirWorld, shadow }); if (env.hf.chunks) chunks = createMeshyWorld(scene, { y0: ENV.mountains.y0, set: env.hf.chunks, shadow: shadow.uniforms, renderer }); }   // the Meshy districts stand among the blocks
  else if (terrain === 'meshy' && !mountains) {              // the Meshy clusters; if they cannot load, draw the baked field as pillars so the picture still matches the obstacles
    const world = createMeshyWorld(scene, { y0: ENV.mountains.y0, set: env.hf.chunks || 'china', shadow: shadow.uniforms, renderer }); mountains = world; mountainsStyle = style;
    world.ready.catch((e) => { console.warn('Meshy models unavailable, drawing the baked field', e); if (mountains === world) { world.dispose(); mountains = createMountains(scene, env.hf, { sunDir: space.sunDirWorld, shadow, style: 'pillars' }); } });
  }
  else if (terrain && !mountains) { mountains = createMountains(scene, env.hf, { sunDir: space.sunDirWorld, shadow, style: terrain }); mountainsStyle = style; }
  if (env.hf) shadow.setField(env.hf); else shadow.setOn(false);
  if (scene.fog) { const valley = terrain === 'meshy' && !!(env.hf && env.hf.lap); scene.fog.near = valley ? 30 : 70; scene.fog.far = valley ? 500 : 560; }   // the Avatar valley: denser haze, ridge behind ridge
  ui.setPhase(key ? 'skyline · ' + CITY_NAMES[key] : terrain === 'meshy' && env.hf && env.hf.lap ? 'Avatar valley' : terrain === 'pillars' || terrain === 'meshy' ? 'Zhangjiajie' : terrain === 'alps' ? 'Alps' : route.name + ' from above');
}
function setSkyline(on) {                                 // the flag: fly low along the skyscrapers of Dubai, New York or Moscow (the towers are obstacles the pilot has learned)
  state.skyline = on; ui.setControls({ skyline: on });
  if (on && !ROUTES[state.route].city) { setRoute('dubai'); return; }
  if (!state.atmo) { if (on) setAtmo(true); return; }
  applyWorld(); ui.toast(on ? 'skyline flight — ' + ROUTES[state.route].name + "'s towers ahead" : 'back over the mountains', 3000);
}
function setRoute(key) {                                  // choose the parallel to fly; a city route shows its skyline straight away; switches atmospheric flight on
  if (!ROUTES[key]) return;
  state.route = key; ui.setRoute(key); state.skyline = !!ROUTES[key].city; ui.setControls({ skyline: state.skyline });
  if (terrain) { terrain.dispose(); terrain = null; }
  if (!state.atmo) setAtmo(true); else { space.setGround(ROUTES[key].lat, ROUTES[key].lon0); terrain = createTerrain(scene, renderer, { ...ROUTES[key], shadow }); applyWorld(); ui.toast('route: ' + ROUTES[key].name, 2500); }
}
function tickLowPass(dt) {
  if (!state.lowPasses || !state.playing || state.atmo || space.body === 'moon') return;
  state.phaseTimer -= dt;
  if (state.phaseTimer <= 0) setLowPass(!state.lowPass);
}

// ---------- modes & controls ----------
function setMode(m) {
  state.mode = m; monoPass.enabled = m > 0; monoPass.uniforms.mode.value = m; accent.set(m > 0 ? 0xffffff : 0x8ec5ff);
  for (const o of [space, field, ship, ghost, comets, planes, birds, mountains, city]) if (o && o.setMono) o.setMono(m);
  ui.setMode(m);
}
function setInvert(b) { state.invert = b; monoPass.uniforms.invert.value = b ? 1 : 0; scene.background.set(b && state.mode === 2 ? 0x000000 : 0x000000); ui.setInvert(b); }
function resetFlight() { env.reset(); ship.reset(); ghost.setVisible(false); clearTrail(); snapshotPrev(); state.crashTimer = 0; state.flightTime = 0; state.laps = 0; ap.crashes = 0; }
function setDensity(n) { state.density = n; env.setCount(n); env.spawnAsteroids(); snapshotPrev(); }

// ---------- training (in a worker when possible, see train-client.js) ----------
const N_ENVS = Math.round(num('nenvs', 128));
function makeTrainClient() {
  const stopAt = num('steps', 0);
  return createTrainClient({ tfUrl: TFJS.url, tfSri: TFJS.sri, backend: qs.get('backend') || 'webgl', getAgent: () => agent, forceInThread: qs.get('worker') === '0', seed: agent && agent.meta && agent.meta.history ? agent.meta : null,
    onMetrics: (m, history, levelChanges) => { ui.metrics(m, history, levelChanges); ui.policyInfo(policyText()); if (stopAt && m.step >= stopAt && train.running) { train.stop(); ap.trainingDone = true; } },
    onFleet: (pts) => { if (ui.trainingVisible()) ui.fleet(pts); },
    onStatus: (st, msg) => {
      if (st === 'running') { state.training = true; ui.setTrainingActive(true); ui.toast(`training ${train.view.where === 'worker' ? 'in a background worker' : 'on the main thread'} · ${train.view.backend} · ${N_ENVS} ships in parallel`); }
      else { state.training = false; ui.setTrainingActive(false); if (st === 'error') { ap.errors.push('training: ' + msg); ui.toast('training stopped: ' + msg, 5000); } }
    } });
}
async function startTraining(opts = {}) {
  if (state.training) return;
  if (!train) train = makeTrainClient(); ap.trainer = train.view;
  const level = qs.has('level') ? num('level', 0) : (agent.steps > 0 ? 1 : 0);
  ui.toast('starting training…', 3000);
  try { await train.start({ nEnvs: N_ENVS, T: 64, seed: 1000 + Math.round(num('seed', 11)), curriculum: state.curriculum, level, speedScale: 1 }, opts.speed || state.trainSpeed); }
  catch (e) { ap.errors.push('training: ' + e.message); ui.toast('cannot train: ' + e.message, 5000); }
}
function stopTraining() { if (train) train.stop(); }
function resetPolicy() {
  if (train) { train.dispose(); train = null; state.training = false; ui.setTrainingActive(false); } dropAtmoPilot();
  if (agent) agent.dispose();
  agent = new PPOAgent(OBS_DIM, ACT_DIM, {}, (Math.random() * 1e9) >>> 0); ap.policy = 'fresh';
  ui.policyInfo(policyText()); ui.metrics({ step: 0, episodes: 0, updates: 0, level: NaN, elapsed: 0, phase: 'idle' }, Object.fromEntries(['step'].map((k) => [k, []])), []); ui.toast('new random policy — press Train to watch it learn');
}
function savePolicy() {
  const blob = new Blob([JSON.stringify(agent.toJSON({ exportedAt: new Date().toISOString() }))], { type: 'application/json' });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = `astro-pilot-policy-${Math.round(agent.steps / 1000)}k.json`; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}
function loadPolicyJSON(j) { try { const ag = PPOAgent.fromJSON(j); if (train) { train.dispose(); train = null; state.training = false; ui.setTrainingActive(false); } dropAtmoPilot(); if (agent) agent.dispose(); agent = ag; ap.policy = 'loaded'; ui.policyInfo(policyText()); showPretrainingCurves(); ui.toast('policy loaded'); } catch (e) { ui.toast('not a valid policy file'); } }

// ---------- flight board ----------
const boardData = { q: null, f: null, u: null, v: null, y: 0, ground: null, y0: ENV.mountains.y0, atmo: 0, orbitAlt: 0, throttle: 0, prox: Infinity, crashed: false, manual: false, world: '', edge: 0 };
function worldLabel() {                                   // the route, and in the megacity the zone under the ship (map x = corridor x + lap shift)
  const name = (ROUTES[state.route].name || '').toUpperCase(), set = env.hf && env.hf.lap && MESHY_SETS[env.hf.city];
  if (!set || !set.zones) return name;
  const X = (((env.ship.p[0] + env.hf.lap.shift) % set.period) + set.period) % set.period, zone = set.zones[Math.floor(X / (set.period / set.zones.length))];
  return name + ' · ' + zone.toUpperCase();
}
function drawBoard(dt) {
  const s = env.ship; let near = Infinity, nearEdge = false; for (let i = 0; i < N_RAYS; i++) if (env.rayHit[i] < near) { near = env.rayHit[i]; nearEdge = !!env.rayEdge[i]; }
  Object.assign(boardData, { q: shipRender.q, f: shipRender.f, u: shipRender.u, v: s.v, y: s.p[1], ground: env.hf ? env.hf.height(s.p[0], s.p[2]) : null, atmo: space.atmosphere, orbitAlt: space.altitude, orbit: space.orbitInfo,
    throttle: env.cmd[3], prox: near < ENV.rays.range - 1e-3 ? near : Infinity, crashed: state.crashTimer > 0, manual: state.manual, edge: env.guard, proxEdge: nearEdge,
    world: space.atmosphere > 0.5 ? worldLabel() : space.body === 'moon' ? 'LUNAR ORBIT' : state.lowPass ? 'LOW PASS' : 'ORBIT',
    airOn: env.atmosphere, ias: env.air.V, aoa: env.air.alpha, g: env.air.n, wind: env.air.wind, turb: env.air.sigma, stall: env.atmosphere && env.air.stalled, over: env.crashCause === 'overstress' });
  board.update(dt, boardData);
  if (pip.active) pip.render(scene, shipRender, dt, { hide: [ship.group, rayLines], hideAlways: [ghost.group],
    style: { mode: monoPass.enabled ? monoPass.uniforms.mode.value : 0, invert: monoPass.uniforms.invert.value, cell: monoPass.uniforms.cell.value, contrast: monoPass.uniforms.contrast.value }, fps: state.fps });
}

// ---------- main ----------
async function main() {
  ui = createUI({
    play: () => { state.playing = !state.playing; ui.setPlaying(state.playing); }, reset: resetFlight,
    simSpeed: (v) => { state.simSpeed = v; }, density: setDensity, astSpeed: (v) => { state.astSpeed = v; env.setSpeedScale(v); },
    camera: (c) => setCameraMode(c), trail: (b) => { state.trail = b; }, sensors: (b) => { state.sensors = b; },
    comets: (n) => { state.comets = n; env.setComets(n); env.spawnAsteroids(); snapshotPrev(); }, cometSpeed: (v) => { const k = v / state.cometSpeed; state.cometSpeed = v; env.setCometSpeed(v); for (const a of env.asteroids) if (a.kind) { a.v[0] *= k; a.v[1] *= k; a.v[2] *= k; } },
    manual: (b) => { state.manual = b; ui.setManual(b); }, invert: setInvert, colorMode: setMode, cycleMode: () => setMode((state.mode + 1) % 3),
    lowPass: (b) => { state.lowPasses = b; if (!b) setLowPass(false); }, orbit: (b) => setBody(b), moonAlt: (km) => space.setMoonAltitude(km), moonScale: (k) => space.setMoonScale(k), atmo: (b) => setAtmo(b), route: (k) => setRoute(k), warp: (w) => space.setWarp(w), skyline: (b) => setSkyline(b), weather: (v) => wxSet(v), wind: (v) => wxSet(env.weatherSeverity, { wind: v }), cover: (v) => wxSet(env.weatherSeverity, { cover: v }), turb: (v) => wxSet(env.weatherSeverity, { turb: v }), sky: (k) => { const p = SKIES[k]; env.setWeather(p[0], { wind: p[1], cover: p[2], turb: p[3] }); ui.setControls({ weather: p[0], wind: p[1], cover: p[2], turb: p[3] }); }, autoThr: (b) => { state.autoThr = b; },
    hideUI: () => document.body.classList.toggle('nohud'), board: (b) => { board.setVisible(b); },
    train: () => (state.training ? stopTraining() : startTraining()), resetPolicy, save: savePolicy, load: loadPolicyJSON,
    trainSpeed: (s) => { state.trainSpeed = s; if (train) train.setSpeed(s); }, curriculum: (b) => { state.curriculum = b; if (train) train.setCurriculum(b); },
  });
  ui.setControls({ density: state.density, simSpeed: state.simSpeed, comets: state.comets, weather: env.weatherSeverity, wind: env.weatherWind, cover: env.weatherCover, turb: env.weatherTurb, sky: skyOf(), autoThr: state.autoThr });
  ui.loading('building the solar system…'); await new Promise((r) => setTimeout(r, 30));
  space = createSpace(scene, { seed: 1, texturePath: 'textures/', start: ['dawn', 'moon'].includes(qs.get('start')) ? qs.get('start') : 'auto', lowPass: state.lowPass, halfFov: Math.atan(Math.tan(CAM_FOV.chase * Math.PI / 360) * innerWidth / innerHeight) });   // ?start=dawn|moon: the orbit's opening view
  // Reflection environment for the hull and rocks: a small procedural sky (space, a sun spot, earthshine from below).
  // Baking it from the live scene is not safe: the atmosphere/sun shaders can emit NaN in the cube render, and one NaN
  // texel turns every reflective material black (and bloom then spreads it over the whole frame).
  const pm = new THREE.PMREMGenerator(renderer), bakeEnv = (day, target = null) => {                                // day = false: space (a sun spot, earthshine from below); true: a daylight sky over the city for atmospheric flight
    const W = 128, H = 64, data = new Float32Array(W * H * 4), sd = day ? space.airSunDirWorld : space.sunDirWorld;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const th = (0.5 - y / H) * Math.PI, ph = (x / W) * 2 * Math.PI - Math.PI, dx = Math.cos(th) * Math.sin(ph), dy = Math.sin(th), dz = Math.cos(th) * Math.cos(ph);   // equirect direction
      const cosSun = dx * sd.x + dy * sd.y + dz * sd.z, i = (y * W + x) * 4; let r, g, b;
      if (!day) {
        const sun = (Math.exp(-Math.max(0, 1 - cosSun) * 120) * 1.8 + Math.exp(-Math.max(0, 1 - cosSun) * 10) * 0.2) * space.sunLit;   // soft, not blinding: the hull's clearcoat mirrors it
        const earth = Math.max(0, -dy - 0.15) * 0.35, gr = space.body === 'moon' ? 0.6 : 0; r = 0.012 + sun * 1.0 + earth * (gr || 0.55); g = 0.016 + sun * 0.95 + earth * (gr || 0.72); b = 0.03 + sun * 0.85 + earth * (gr || 1.0);   // lunar orbit: grey moonshine
      } else if (dy >= 0) {                                   // sky: deep blue overhead, pale haze at the horizon, a bright sun and its aureole
        const t = Math.pow(1 - dy, 3), sun = Math.exp(-Math.max(0, 1 - cosSun) * 900) * 14 + Math.exp(-Math.max(0, 1 - cosSun) * 18) * 0.45;
        r = 0.08 + 0.36 * t + sun; g = 0.16 + 0.32 * t + sun * 0.92; b = 0.38 + 0.2 * t + sun * 0.78;
      } else {                                                // below the horizon: the city and the land, lit from above
        const t = Math.min(1, -dy * 3); r = 0.2 - 0.08 * t; g = 0.19 - 0.08 * t; b = 0.18 - 0.07 * t;
      }
      data[i] = r; data[i + 1] = g; data[i + 2] = b; data[i + 3] = 1;
    }
    const tex = new THREE.DataTexture(data, W, H, THREE.RGBAFormat, THREE.FloatType); tex.mapping = THREE.EquirectangularReflectionMapping; tex.needsUpdate = true;
    const rt = pm.fromEquirectangular(tex, target); tex.dispose(); return rt;   // one generator for the page: a re-bake reuses its shaders and writes into the same target
  };
  envMaps.space = bakeEnv(false); scene.environment = envMaps.space.texture; scene.environmentIntensity = 0.7;
  envMaps.bake = () => { if (!envMaps.day) envMaps.day = bakeEnv(true); return envMaps.day.texture; };
  envMaps.sun = new THREE.Vector3(); envMaps.lit = -1; envMaps.at = -Infinity; envMaps.rebake = (day) => {   // in place, when the Sun moved > 3.6° or went into or out of the Earth's shadow; the first frame bakes with the real Sun
    const dir = day ? space.airSunDirWorld : space.sunDirWorld, lit = day ? 1 : space.sunLit; if ((dir.dot(envMaps.sun) > 0.998 && Math.abs(lit - envMaps.lit) < 0.3) || performance.now() - envMaps.at < 2000 || (day && !envMaps.day)) return;
    envMaps.sun.copy(dir); envMaps.lit = lit; envMaps.at = performance.now(); bakeEnv(day, day ? envMaps.day : envMaps.space); };
  if (space.ready) space.ready.then(() => { ap.texturesReady = true; }).catch(() => {});
  fill = new THREE.DirectionalLight(0xdfe8ff, 1.15); fill.position.set(0.35, 0.6, 1.0).multiplyScalar(500); scene.add(fill, fill.target);   // camera-side fill light (direction follows the camera mode)
  comets = createComets(scene, { max: 8, sunDir: space.sunDir }); sats = createSatellites(scene, { max: 8 }); planes = createPlanes(scene, { max: 10 }); birds = createBirds(scene, { max: 8 });
  field = createAsteroidField(scene, { nShapes: ENV.belt.nShapes, maxInstances: 80, seed: 7 });
  ship = createShip({ seed: 1 }); ghost = createShip({ seed: 1 }); ghost.setVisible(false);
  scene.add(ship.group, ghost.group, trailLine, rayLines); fence = createEdgeFence(scene, { color: accent });
  rig = createCameraRig(camera, canvas); Object.defineProperty(rig, 'moon', { get: () => space.telescope.target });   // the Moon, or the Earth from lunar orbit
  setCameraMode(state.camera); ui.setCamera(state.camera);
  pip = createPipCam(renderer); if (qs.has('cam2')) pip.setMode(qs.get('cam2'));
  board = createBoard(document.getElementById('board'), { onCamera: () => pip.cycle() }); board.setCamera(pip.mode);
  board.onRect((r) => pip.setRect(r)); if (qs.get('board') === '0') { board.setVisible(false); ui.setControls({ board: false }); }
  zoom = attachEarthZoom({ renderer, space, host: { get manual() { return state.manual; }, setManual: (b) => { state.manual = b; ui.setManual(b); }, toast: (m) => ui.toast(m, 3000), credit: () => document.getElementById('credit').textContent, setCredit: (t) => ui.setCredit(t), flightControls: (on) => { rig.controls.enabled = on && rig.mode === 'orbit'; }, entry: () => { const o = space.orbitInfo, r = ROUTES[state.route]; return state.atmo && terrain ? { lat: r.lat, lon: r.lon0 + terrain.far.stats.scroll * 2.8 / (111.32 * Math.cos(r.lat * Math.PI / 180)), altKm: 8 } : { lat: o.lat, lon: o.lon, altKm: o.altKm }; } } });
  ui.loading('loading the policy…');
  await loadPolicy(); ui.policyInfo(policyText()); showPretrainingCurves();
  if (qs.has('mode')) { const named = { color: 0, colour: 0, mono: 1, ink: 2 }[qs.get('mode')]; setMode(named !== undefined ? named : (+qs.get('mode') || 0)); }
  if (qs.get('invert') === '1') setInvert(true);
  if (qs.get('sensors') === '1') { state.sensors = true; ui.setControls({ sensors: true }); }
  if (qs.get('manual') === '1') { state.manual = true; ui.setManual(true); }
  if (qs.get('panels') === '0' || window.innerWidth < 820) ui.showPanels(false, false);
  else if (window.innerWidth >= 1280) ui.showPanels(true, true);            // wide screens: playbox and training side by side
  snapshotPrev(); resetFlight();
  if (state.lowPass) { state.lowPass = false; setLowPass(true); }
  ui.setRoute(state.route);
  space.setMoonAltitude(knob('moonalt', 1500, 500, 60000)); space.setMoonScale(knob('moonscale', 4, 1, 8)); ui.setControls({ moonAlt: space.moonAltitude, moonScale: space.moonScale }); if (qs.get('orbit') === 'moon' && qs.get('atmo') !== '1' && !state.skyline) setBody('moon');   // ?orbit=moon&moonalt=km
  if (qs.get('atmo') === '1' || state.skyline) { ui.setControls({ skyline: state.skyline }); if (state.skyline && !ROUTES[state.route].city) state.route = 'dubai'; setAtmo(true); ui.setControls({ atmo: true }); ui.setRoute(state.route); }
  ui.ready(); ap.ready = true;
  if (ap.policyError) ui.toast('no pretrained policy found — press Train', 4000);
  if (qs.get('train') === '1') startTraining({ speed: qs.get('trainspeed') || 'max' });
  let last = performance.now(), acc = 0, t = 0, fpsAcc = 0, fpsN = 0, hudTimer = 0;
  function frame(now) {
    requestAnimationFrame(frame);
    let dtReal = Math.min(0.1, Math.max(0, (now - last) / 1000)); last = now;   // a frame's timestamp can precede the clock read at start-up: never a negative step
    fpsAcc += dtReal; fpsN++; if (fpsAcc >= 0.5) { state.fps = ap.fps = fpsN / fpsAcc; fpsAcc = 0; fpsN = 0; adaptQuality(state.fps); }
    const dt = state.playing ? dtReal * state.simSpeed : 0; t += dt; acc += dt;
    let steps = 0; while (acc >= ENV.dt && steps < 8) { simStep(); acc -= ENV.dt; steps++; }
    if (steps === 8) acc = 0;
    tickLowPass(dt); updateVisuals(Math.min(1, acc / ENV.dt), dtReal);
    space.update(t, dt, camera); if (envMaps.pending) { envMaps.pending = false; envMaps.at = -Infinity; envMaps.sun.set(0, 0, 0); } if (state.camera === 'moon' && space.atmosphere < 0.5 && rig.moon.visible) camera.lookAt(rig.moon.getWorldPosition(_aim));   // a body switch re-bakes the reflections (next frame); the telescope re-aims at the sky just moved
    clouds.update({ env, atmosphere: space.atmosphere, sunDir: space.sunDirWorld, fog: scene.fog, time: state.flightTime }); cloudShadows.update(env, camera.position.x, space.sunDirWorld, space.atmosphere); droplets.update(dtReal, camera, env, space.atmosphere);
    if (zoom.active) zoom.render(dtReal); else if (state.bypass) renderer.render(scene, camera); else composer.render();
    if (board.visible && !zoom.active) { drawBoard(dtReal); }
    ap.frames++;
    hudTimer += dtReal;
    if (hudTimer >= 0.1) {
      hudTimer = 0; const s = env.ship;
      ui.hud({ speed: Math.hypot(s.v[0], s.v[1], s.v[2]), value: lastValue, laps: state.laps, flight: state.flightTime, crashes: ap.crashes, fps: state.fps });
      ui.sensors(env.rayHit, ENV.rays.range); ui.radar(env.obs.subarray(OBS_BASE, OBS_BASE + N_RAYS)); ui.actions(actTanh);
    }
  }
  requestAnimationFrame(frame);
}

Object.defineProperties(ap, { zoom: { get: () => zoom }, clouds: { get: () => clouds }, shadow: { get: () => shadow }, board: { get: () => board }, pip: { get: () => pip }, agent: { get: () => agent }, atmoAgent: { get: () => atmoAgent }, pilot: { get: () => activeAgent() }, space: { get: () => space }, mountains: { get: () => mountains }, city: { get: () => city }, chunks: { get: () => chunks }, camera: { get: () => camera }, info: { get: () => renderer.info.render } }); ap.THREE = THREE;   // live getters (Object.assign would copy the values once)
Object.assign(ap, { env, setMode, setInvert, setCamera: (c) => { setCameraMode(c); ui.setCamera(c); }, setSensors: (b) => { state.sensors = b; ui.setControls({ sensors: b }); },
  setManual: (b) => { state.manual = b; ui.setManual(b); }, setPlaying: (b) => { state.playing = b; ui.setPlaying(b); }, startTraining, stopTraining, resetPolicy, showTraining: (b) => ui.showPanels(!b, b), setLowPass, setAtmo, setRoute, setSkyline, terrainStats: () => (terrain ? terrain.stats : null),
  exportPolicy: (meta) => agent.toJSON({ trainedIn: ((train && train.view.backend) || 'browser') + ' · ' + (navigator.platform || ''), ...(meta || {}) }), loadPolicyJSON,
  evaluate: (o) => evaluatePolicy(agent, o), state, scene,
  brightness: (direct) => { if (direct) renderer.render(scene, camera); else composer.render(); const gl = renderer.getContext(), w = gl.drawingBufferWidth, h = gl.drawingBufferHeight, buf = new Uint8Array(w * 4); let sum = 0, n = 0; for (let r = 0; r < 12; r++) { gl.readPixels(0, Math.floor(h * (r + 0.5) / 12), w, 1, gl.RGBA, gl.UNSIGNED_BYTE, buf); for (let x = 0; x < w; x += 8) { sum += buf[x * 4] + buf[x * 4 + 1] + buf[x * 4 + 2]; n += 3; } } return sum / n; },
  debug: () => ({ size: renderer.getSize(new THREE.Vector2()).toArray(), viewport: renderer.getViewport(new THREE.Vector4()).toArray(), pixelRatio: renderer.getPixelRatio(), target: [composer.renderTarget1.width, composer.renderTarget1.height], canvas: [canvas.width, canvas.height, canvas.clientWidth, canvas.clientHeight], inner: [innerWidth, innerHeight], aspect: camera.aspect, scissorTest: renderer.getScissorTest(), scissor: renderer.getScissor(new THREE.Vector4()).toArray() }),
  renderDirect: () => { renderer.render(scene, camera); }, bypassComposer: (b) => { state.bypass = b; }, setBloom: (b) => { bloom.enabled = b; },
  capture: () => { composer.render(); return canvas.toDataURL('image/png'); }, hideShip: (b) => { ship.group.visible = !b; },
  blackRun: () => { composer.render(); const gl = renderer.getContext(), w = gl.drawingBufferWidth, h = gl.drawingBufferHeight, rows = 9, buf = new Uint8Array(w * 4), black = new Uint8Array(w).fill(1);
    for (let r = 0; r < rows; r++) { gl.readPixels(0, Math.floor(h * (r + 0.5) / rows), w, 1, gl.RGBA, gl.UNSIGNED_BYTE, buf); for (let x = 0; x < w; x++) if (buf[x * 4] + buf[x * 4 + 1] + buf[x * 4 + 2] > 12) black[x] = 0; }
    let best = 0, run = 0; for (let x = 0; x < w; x++) { run = black[x] ? run + 1 : 0; best = Math.max(best, run); } return best; },
  probe: (direct) => { if (direct) renderer.render(scene, camera); else composer.render(); const gl = renderer.getContext(), w = gl.drawingBufferWidth, h = gl.drawingBufferHeight, row = new Uint8Array(w * 4); let maxx = 0;
    for (const y of [Math.floor(h * 0.3), Math.floor(h * 0.5), Math.floor(h * 0.7)]) { gl.readPixels(0, y, w, 1, gl.RGBA, gl.UNSIGNED_BYTE, row); for (let x = w - 1; x > maxx; x--) if (row[x * 4] + row[x * 4 + 1] + row[x * 4 + 2] > 20) { maxx = x; break; } }
    return { w, h, maxx, vp: renderer.getViewport(new THREE.Vector4()).toArray(), sc: renderer.getScissorTest(), rt1: [composer.renderTarget1.width, composer.renderTarget1.height, composer.renderTarget1.viewport.toArray()], rt2: [composer.renderTarget2.width, composer.renderTarget2.height, composer.renderTarget2.viewport.toArray()] }; } });
main().catch((e) => { ap.errors.push('boot: ' + e.message); const l = document.getElementById('loadingMsg'); if (l) l.textContent = 'something went wrong: ' + e.message; console.error(e); });
