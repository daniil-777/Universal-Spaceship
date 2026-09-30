// chat/web/capcom-worker.js — module worker running CAPCOM (capcom.js) off the main thread, so generation never blocks three.js.
// in:  {type:'load', model, kb, device, dtype} · {type:'ask', id, user, history, state, maxNewTokens} · {type:'render', id, user, history, state}
//      · {type:'abort'} · {type:'dispose'}
// out: {type:'kb', facts, minScore} · {type:'progress', …} · {type:'ready', info} · {type:'token', id, text} · {type:'done', id, result}
//      · {type:'rendered', id, text, ids, hits, messages} · {type:'error', id?, message}
// Before the model is ready, asks are answered from the notes (result.fallback = 'loading'); if every runtime rung fails, from the
// notes for good (result.fallback = true, info.ready = false).
import * as tf from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0';
import { createCapcom, loadKb, modelSource } from './capcom.js';
import { messages } from './prompt.js';
let K = null, capcom = null;
const post = (m) => self.postMessage(m), err = (e) => String(e && e.message ? e.message : e);
const titled = (r) => ({ ...r, noteTitles: r.notes.map((id) => K.bm.byId.get(id).title || '') });

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'load') {
      K = await loadKb(new URL(m.kb, self.location.href).href);
      post({ type: 'kb', facts: K.kb.facts.length, minScore: K.minScore });
      capcom = await createCapcom({ tf, ...modelSource(m.model, self.location.href), kb: K.kb, device: m.device || 'auto', dtype: m.dtype || null,
        onProgress: (p) => post({ type: 'progress', ...p }) });
      post({ type: 'ready', info: capcom.info });
    } else if (m.type === 'ask') {
      const opts = { history: m.history || [], state: m.state || null, maxNewTokens: m.maxNewTokens };
      if (!capcom) {
        if (!K) throw new Error('ask before load');
        const r = { ...K.card(m.user, opts), fallback: 'loading', ttftMs: 0, ms: 0, tokens: 0, tokPerSec: null, aborted: false };
        post({ type: 'token', id: m.id, text: r.text }); return post({ type: 'done', id: m.id, result: titled(r) });
      }
      const result = await capcom.ask(m.user, { ...opts, onToken: (text) => post({ type: 'token', id: m.id, text }) });
      post({ type: 'done', id: m.id, result: titled(result) });
    } else if (m.type === 'render') {
      if (!capcom || !capcom.tokenizer) throw new Error('render before the tokenizer is loaded');
      const { hits, text } = capcom.render(m.user, { history: m.history || [], state: m.state || null });
      const ids = Array.from(capcom.tokenizer(text, { add_special_tokens: false }).input_ids.data, Number);
      post({ type: 'rendered', id: m.id, text, ids, hits,
        messages: messages(m.history || [], m.user, hits.map(([id]) => K.bm.byId.get(id).text), m.state || null) });
    } else if (m.type === 'abort') { if (capcom) capcom.abort();
    } else if (m.type === 'dispose') { if (capcom) await capcom.dispose(); capcom = null; post({ type: 'disposed' }); }
  } catch (x) { post({ type: 'error', id: m.id, message: err(x) }); }
};
