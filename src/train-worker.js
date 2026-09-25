// Training worker: owns the TF.js runtime and the Trainer so the page's render loop never blocks. Protocol (postMessage):
//   in : { type:'init', tfUrl, backend, snapshot? }  → { type:'ready', backend } | { type:'error', message }
//        { type:'start', opts, speed }               → { type:'metrics', metrics, levelChanges, snapshot } after every PPO update, { type:'fleet', pts } at ~5 Hz
//        { type:'stop' } → { type:'stopped', metrics } ; { type:'speed', speed } ; { type:'curriculum', on } ; { type:'level', level } ; { type:'load', snapshot }
import { OBS_DIM, ACT_DIM } from './env.js';
import { PPOAgent, setTF } from './ppo.js';
import { Trainer } from './trainer.js';

let agent = null, trainer = null, fleetTimer = 0;
const post = (m, t) => self.postMessage(m, t || []);

async function loadTF(tfUrl, backend) {
  if (!self.tf) await import(/* @vite-ignore */ tfUrl);
  const tf = self.tf; if (!tf) throw new Error('TensorFlow.js did not load in the worker');
  setTF(tf);
  if (!(await tf.setBackend(backend))) await tf.setBackend('cpu');
  await tf.ready();
  return tf.getBackend();
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      const backend = await loadTF(m.tfUrl, m.backend || 'webgl');
      agent = m.snapshot ? PPOAgent.fromSnapshot(m.snapshot, m.cfg || {}) : new PPOAgent(OBS_DIM, ACT_DIM, m.cfg || {}, m.seed || 1);
      post({ type: 'ready', backend });
    } else if (m.type === 'start') {
      if (trainer && trainer.running) return;
      trainer = new Trainer(agent, m.opts); trainer.mode = m.speed || 'max';
      trainer.onMetrics = (met) => { const snap = agent.snapshot(); post({ type: 'metrics', metrics: { ...met }, levelChanges: trainer.levelChanges.slice(), snapshot: snap }, PPOAgent.transferables(snap)); };
      clearInterval(fleetTimer); fleetTimer = setInterval(() => { if (trainer && trainer.running) { const pts = trainer.fleet(); post({ type: 'fleet', pts }, [pts.buffer]); } }, 200);
      trainer.run().then(() => { clearInterval(fleetTimer); post({ type: 'stopped', metrics: { ...trainer.metrics } }); }).catch((err) => { clearInterval(fleetTimer); post({ type: 'error', message: err.message }); });
    } else if (m.type === 'stop') { if (trainer) trainer.stop(); }
    else if (m.type === 'speed') { if (trainer) trainer.mode = m.speed; }
    else if (m.type === 'curriculum') { if (trainer) trainer.curriculum = m.on; }
    else if (m.type === 'level') { if (trainer) trainer.setLevel(m.level); }
    else if (m.type === 'load') { if (trainer) trainer.stop(); agent.dispose(); agent = PPOAgent.fromSnapshot(m.snapshot, m.cfg || {}); post({ type: 'loaded' }); }
  } catch (err) { post({ type: 'error', message: err.message || String(err) }); }
};
