// vlm/gen/build/pool_worker.js — the worker side of pool.js: loads workerData.module and answers {id, task} messages with
// {id, result} (its transfer list moved, not copied) or {id, error}. Jobs run one at a time in arrival order (the pool's
// queued jobs wait here, so the worker never idles on the main thread, and a worker is one core's worth of load).
import { parentPort, workerData } from 'node:worker_threads';

const mod = await import(workerData.module);
let chain = Promise.resolve();
parentPort.on('message', ({ id, task }) => {
  chain = chain.then(async () => {
    try {
      const { result, transfer = [] } = await mod.job(task);
      parentPort.postMessage({ id, result }, transfer);
    } catch (e) {
      parentPort.postMessage({ id, error: String((e && e.stack) || e) });
    }
  });
});
