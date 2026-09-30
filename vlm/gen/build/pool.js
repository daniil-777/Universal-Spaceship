// vlm/gen/build/pool.js — a fixed pool of worker threads, each running the `job(task) -> {result, transfer}` export of one
// module (pool_worker.js loads it). run(task) resolves with that job's result, whatever order the workers finish in; a
// thrown job rejects its own promise only. Each worker holds up to `depth` queued jobs, so it keeps working while the main
// thread is busy (a worker's result waits in the port; only a new dispatch needs the main thread). The workers share the
// process's libuv threadpool, where sharp decodes: build.mjs sizes it (UV_THREADPOOL_SIZE) before any work starts.
import os from 'node:os';
import { Worker } from 'node:worker_threads';
import { pathToFileURL } from 'node:url';

export const poolSize = (env = process.env) => {
  const n = parseInt(env.APV_BUILD_WORKERS || '', 10);
  return Number.isInteger(n) && n >= 1 ? n : Math.max(1, Math.min(8, (os.availableParallelism ? os.availableParallelism() : os.cpus().length) - 2));
};
export function createPool(moduleUrl, size = poolSize(), { depth = 4 } = {}) {
  const href = moduleUrl instanceof URL ? moduleUrl.href : /^file:/.test(String(moduleUrl)) ? String(moduleUrl) : pathToFileURL(String(moduleUrl)).href;
  const queue = [], pending = new Map(), all = [];
  let seq = 0, broken = null;
  const pump = () => {
    for (let more = true; more && queue.length;) {
      more = false;
      for (const w of all) if (w.held.size < depth && queue.length) { const q = queue.shift(); w.held.add(q.id); w.postMessage({ id: q.id, task: q.task }); more = true; }
    }
  };
  for (let i = 0; i < size; i++) {
    const w = new Worker(new URL('./pool_worker.js', import.meta.url), { workerData: { module: href } });
    w.held = new Set();
    w.on('message', (m) => {
      const p = pending.get(m.id); pending.delete(m.id); w.held.delete(m.id);
      if (p) (m.error ? p.no(new Error(m.error)) : p.ok(m.result));
      pump();
    });
    // an uncaught worker error fails its jobs and every later one: the build stops on it
    w.on('error', (e) => { broken = e; for (const id of w.held) { pending.get(id)?.no(e); pending.delete(id); } for (const q of queue.splice(0)) pending.get(q.id)?.no(e); });
    all.push(w);
  }
  return {
    size,
    run(task) {
      if (broken) return Promise.reject(broken);
      return new Promise((ok, no) => { const id = seq++; pending.set(id, { ok, no }); queue.push({ id, task }); pump(); });
    },
    async close() { await Promise.all(all.map((w) => w.terminate())); },
  };
}
