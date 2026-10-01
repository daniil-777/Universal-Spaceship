// chat/web/capcom-client.js — the page side of capcom-worker.js as promises: load() resolves with the runtime info, ask() streams
// tokens through onToken and resolves with the result ({text, notes, ttftMs, ms, tokens, tokPerSec, aborted, fallback?}); a new ask
// cancels the running one inside the worker (its promise still resolves, aborted: true). Used by demo.html and bench.mjs.
export function createCapcomClient({ workerUrl = new URL('./capcom-worker.js', import.meta.url), onProgress = null } = {}) {
  const w = new Worker(workerUrl, { type: 'module' }), pending = new Map();
  let nid = 0, loading = null;
  w.onmessage = (e) => {
    const m = e.data, p = m.id !== undefined ? pending.get(m.id) : null;
    if (m.type === 'progress' || m.type === 'kb') { if (onProgress) onProgress(m); return; }
    if (m.type === 'ready' && loading) { loading.ok(m.info); loading = null; return; }
    if (m.type === 'token' && p) { if (p.onToken) p.onToken(m.text); return; }
    if ((m.type === 'done' || m.type === 'rendered') && p) { pending.delete(m.id); p.ok(m.type === 'done' ? m.result : m); return; }
    if (m.type === 'error') {
      if (p) { pending.delete(m.id); p.no(new Error(m.message)); } else if (loading) { loading.no(new Error(m.message)); loading = null; } else console.error('capcom worker:', m.message);
    }
  };
  w.onerror = (e) => { const x = new Error(e.message || 'worker failed'); if (loading) loading.no(x); for (const p of pending.values()) p.no(x); pending.clear(); };
  const call = (msg, onToken = null) => new Promise((ok, no) => { const id = nid++; pending.set(id, { ok, no, onToken }); w.postMessage({ ...msg, id }); });
  return {
    load({ model, kb, device = 'auto', dtype = null }) {
      const p = new Promise((ok, no) => { loading = { ok, no }; });
      w.postMessage({ type: 'load', model, kb, device, dtype });
      return p;
    },
    ask: (user, { history = [], state = null, maxNewTokens, onToken = null } = {}) => call({ type: 'ask', user, history, state, maxNewTokens }, onToken),
    render: (user, { history = [], state = null } = {}) => call({ type: 'render', user, history, state }),
    abort: () => w.postMessage({ type: 'abort' }),
    terminate: () => w.terminate(),
  };
}
