// Training client for the page: prefers a module Web Worker (TF.js + Trainer off the main thread), falls back to running
// the Trainer on the main thread when workers/OffscreenCanvas are unavailable. Keeps a mirror of metrics/history for the
// UI and pushes every new policy snapshot into the page's inference agent.
import { Trainer, HIST_KEYS } from './trainer.js';
import { setTF } from './ppo.js';

export function createTrainClient({ tfUrl, tfSri, backend = 'webgl', getAgent, onMetrics, onFleet, onStatus, forceInThread = false, seed = null }) {
  const view = { metrics: null, history: Object.fromEntries(HIST_KEYS.map((k) => [k, (seed && seed.history && seed.history[k]) ? seed.history[k].map((v) => (v === null ? NaN : v)) : []])), levelChanges: (seed && seed.levelChanges) ? seed.levelChanges.slice() : [], running: false, backend: null, mode: 'balanced', where: null, stop: () => stop() };
  if (seed && seed.history && seed.history.step) for (const k of HIST_KEYS) if (view.history[k].length !== view.history.step.length) view.history[k] = view.history.step.map(() => NaN);   // keys missing from an older run
  let worker = null, trainer = null, starting = false;
  const canWorker = !forceInThread && typeof Worker !== 'undefined' && typeof OffscreenCanvas !== 'undefined';
  const record = (m, levelChanges) => { view.metrics = m; for (const k of HIST_KEYS) view.history[k].push(m[k]); if (levelChanges) view.levelChanges = levelChanges; onMetrics && onMetrics(m, view.history, view.levelChanges); };

  function loadScript(url, sri) { return new Promise((res, rej) => { if (window.tf) return res(); const s = document.createElement('script'); s.src = url; if (sri) { s.integrity = sri; s.crossOrigin = 'anonymous'; } s.onload = res; s.onerror = () => rej(new Error('failed to load TensorFlow.js')); document.head.appendChild(s); }); }

  async function startInThread(opts, speed) {
    await loadScript(tfUrl, tfSri); setTF(window.tf);
    if (!(await tf.setBackend(backend))) await tf.setBackend('cpu'); await tf.ready();
    view.backend = tf.getBackend(); view.where = 'page';
    trainer = new Trainer(getAgent(), opts); trainer.mode = speed;
    trainer.onMetrics = (m) => record({ ...m }, trainer.levelChanges.slice());
    view.running = true; onStatus && onStatus('running');
    let fleetTimer = setInterval(() => { if (trainer.running && onFleet) onFleet(trainer.fleet()); }, 200);
    trainer.run().then(() => { clearInterval(fleetTimer); view.running = false; onStatus && onStatus('stopped'); }).catch((e) => { clearInterval(fleetTimer); view.running = false; onStatus && onStatus('error', e.message); });
  }

  function startWorker(opts, speed, snapshot, cfg) {
    return new Promise((resolve, reject) => {
      let w;
      try { w = new Worker(new URL('./train-worker.js', import.meta.url), { type: 'module' }); } catch (e) { return reject(e); }
      let ready = false;
      w.onerror = (e) => { if (!ready) reject(new Error(e.message || 'worker failed')); else onStatus && onStatus('error', e.message); };
      w.onmessage = (e) => {
        const m = e.data;
        if (m.type === 'ready') { ready = true; worker = w; view.backend = m.backend; view.where = 'worker'; w.postMessage({ type: 'start', opts, speed }); view.running = true; onStatus && onStatus('running'); resolve(); }
        else if (m.type === 'metrics') { getAgent().applySnapshot(m.snapshot); record(m.metrics, m.levelChanges); }
        else if (m.type === 'fleet') { onFleet && onFleet(m.pts); }
        else if (m.type === 'stopped') { view.running = false; view.metrics = m.metrics; onStatus && onStatus('stopped'); }
        else if (m.type === 'error') { view.running = false; if (!ready) reject(new Error(m.message)); else onStatus && onStatus('error', m.message); }
      };
      w.postMessage({ type: 'init', tfUrl, backend, snapshot, cfg }, snapshot ? [] : []);
    });
  }

  async function start(opts, speed = 'balanced') {
    if (view.running || starting) return; starting = true; view.mode = speed;
    try {
      if (canWorker) {
        try { await startWorker(opts, speed, getAgent().snapshot(), getAgent().cfg); }
        catch (e) { console.warn('training worker unavailable, training on the main thread:', e.message); if (worker) { worker.terminate(); worker = null; } await startInThread(opts, speed); }
      } else await startInThread(opts, speed);
    } finally { starting = false; }
  }
  function stop() { if (worker) worker.postMessage({ type: 'stop' }); if (trainer) trainer.stop(); }
  function setSpeed(s) { view.mode = s; if (worker) worker.postMessage({ type: 'speed', speed: s }); if (trainer) trainer.mode = s; }
  function setCurriculum(on) { if (worker) worker.postMessage({ type: 'curriculum', on }); if (trainer) trainer.curriculum = on; }
  function dispose() { stop(); if (worker) { worker.terminate(); worker = null; } trainer = null; view.running = false; view.metrics = null; for (const k of HIST_KEYS) view.history[k] = []; view.levelChanges = []; }
  return { view, start, stop, setSpeed, setCurriculum, dispose, get running() { return view.running; } };
}
