// vlm/web/pilot-eye-worker.js — module worker (spec §10.3) running Pilot Eye off the main thread, ORT-web WASM, 1 thread.
// in:  {type: 'init', encoderUrl, headsUrl, labels} then {type: 'frame', id, rgb (Uint8Array W*H*3), sim_t_s, family, episode_id}
//      (or bitmap: an ImageBitmap in place of rgb, e.g. the site's 896x504 frame, box-resized here to labels.input)
// out: {type: 'ready'} | {type: 'result', id, status: 'warming up'|'ok', …heads, sentence, ms} | {type: 'error', id?, message}
// Messages run one at a time, in order: the frame ring and the ORT sessions never see two frames at once. A bitmap is
// always closed, used or not (a frame before init, an error).
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs';
import { createPilotEye, prepareFrame } from './pilot-eye.js';
let eye = null, input = [160, 96], chain = Promise.resolve();
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
function pixels(bitmap) {
  const c = new OffscreenCanvas(bitmap.width, bitmap.height), g = c.getContext('2d', { willReadFrequently: true });
  g.drawImage(bitmap, 0, 0); bitmap.close();
  return prepareFrame(g.getImageData(0, 0, c.width, c.height).data, c.width, c.height, input);
}
async function handle(m) {
  try {
    if (m.type === 'init') { input = (m.labels && m.labels.input) || input; eye = await createPilotEye({ ort, encoderUrl: m.encoderUrl, headsUrl: m.headsUrl, labels: m.labels, numThreads: 1 }); self.postMessage({ type: 'ready' }); }
    else if (m.type === 'frame') {
      if (!eye) throw new Error('frame before init');
      const rgb = m.bitmap ? pixels(m.bitmap) : m.rgb;
      self.postMessage({ type: 'result', id: m.id, ...(await eye.push({ rgb, sim_t_s: m.sim_t_s, family: m.family, episode_id: m.episode_id })) });
    }
  } catch (err) { self.postMessage({ type: 'error', id: m.id, message: String(err && err.message ? err.message : err) }); }
  finally { if (m.bitmap && typeof m.bitmap.close === 'function') m.bitmap.close(); }
}
self.onmessage = (e) => { chain = chain.then(() => handle(e.data)); };
