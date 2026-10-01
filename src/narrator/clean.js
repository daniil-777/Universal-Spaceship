// clean.js — the Narrator's streaming sentence gate (pure; Node-tested). The v0 model repeats itself and runs to its
// token cap on the deploy path, so the card shows only what survives this gate:
//   - a sentence is committed only once it has ended (. ! ? then the next word has begun), so a decimal point is no end;
//   - generation stops at the first near-repeat: word Jaccard >= 0.6 against any kept sentence, or the same first three
//     words with Jaccard >= 0.4 ("The nearest hazard is … / The nearest hazard is …"); a repeat is never shown;
//   - a cap of sentences per task (Describe 3, safety and ask 2);
//   - finish() drops a dangling fragment left when the model hit its token cap;
//   - dropEcho (safety): sentences restating the monitor are skipped, because the pill already shows the verdict.
export const JACCARD_STOP = 0.6, OPENING_STOP = 0.4;
export const CAPS = Object.freeze({ describe: 3, safety: 2, ask: 2 });
const END = /^\s*([\s\S]*?[.!?]+["')\]]*)(?=\s)/;
const ECHO_WORDS = /\b(monitor|verdict)\b|\b(reason|advice)\s*:/i, ECHO_CAPS = /\b(SAFE|CAUTION|UNSAFE)\b/;

export const words = (s) => (String(s).toLowerCase().match(/[a-z0-9']+/g) || []);
const opening = (s) => words(s).slice(0, 3).join(' ');
export function jaccard(a, b) {
  const A = new Set(words(a)), B = new Set(words(b));
  if (!A.size && !B.size) return 0;
  let n = 0; for (const w of A) if (B.has(w)) n++;
  return n / (A.size + B.size - n);
}

export function createSentenceGate({ max = CAPS.describe, dropEcho = false } = {}) {
  let buf = '', stopped = false, reason = null; const kept = [];
  function take(s, add) {
    if (stopped || !words(s).length) return;
    if (dropEcho && (ECHO_WORDS.test(s) || ECHO_CAPS.test(s))) return;
    const o = opening(s), again = (k) => { const j = jaccard(k, s); return j >= JACCARD_STOP || (j >= OPENING_STOP && words(s).length >= 3 && opening(k) === o); };
    if (kept.some(again)) { stopped = true; reason = 'repeat'; return; }
    kept.push(s); add.push(s);
    if (kept.length >= max) { stopped = true; reason = 'cap'; }
  }
  return {
    push(chunk) {
      const add = [];
      if (stopped) return { add, stop: true };
      buf += chunk;
      for (let m = END.exec(buf); m && !stopped; m = END.exec(buf)) { buf = buf.slice(m[0].length); take(m[1].trim(), add); }
      return { add, stop: stopped };
    },
    finish() {
      const add = [], rest = buf.trim(); buf = '';
      if (!stopped && rest) { if (/[.!?]["')\]]*$/.test(rest)) take(rest, add); else reason = 'fragment'; }
      if (!stopped) { stopped = true; reason ||= 'end'; }
      return { add };
    },
    get sentences() { return [...kept]; },
    get text() { return kept.join(' '); },
    get stopped() { return stopped; },
    get reason() { return reason; },
  };
}
