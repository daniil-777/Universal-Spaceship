// vlm/web/pilot-eye-worker.js — module worker (spec §10.3) running Pilot Eye off the main thread, ORT-web WASM, 1 thread.
// in:  {type: 'init', encoderUrl, headsUrl, labels} then {type: 'frame', id, rgb (Uint8Array W*H*3), sim_t_s, family, episode_id}
// out: {type: 'ready'} | {type: 'result', id, status: 'warming up'|'ok', …heads, sentence, ms} | {type: 'error', id?, message}
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs';
import { createPilotEye } from './pilot-eye.js';
let eye = null;
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') { eye = await createPilotEye({ ort, encoderUrl: m.encoderUrl, headsUrl: m.headsUrl, labels: m.labels, numThreads: 1 }); self.postMessage({ type: 'ready' }); }
    else if (m.type === 'frame') {
      if (!eye) throw new Error('frame before init');
      self.postMessage({ type: 'result', id: m.id, ...(await eye.push(m)) });
    }
  } catch (err) { self.postMessage({ type: 'error', id: m.id, message: String(err && err.message ? err.message : err) }); }
};
