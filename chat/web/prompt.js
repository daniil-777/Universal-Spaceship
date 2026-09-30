// chat/web/prompt.js — CAPCOM's prompt contract v1 in the page, the same text as chat/prompt.py (spec §4): a fixed preamble, up to
// MAX_NOTES retrieved notes, a state line (scene + <= 3 not-yet-seen highlights in tour order), then the last HISTORY messages (two
// exchanges) and the visitor's message. Parity: chat/tests/prompt_parity.test.mjs renders both on the same cases.
export const PREAMBLE = 'You are CAPCOM, the guide inside Astro Pilot, a spaceflight demo running in this browser. ' +
  'Answer briefly from the notes, then offer one next step.';
export const MAX_NOTES = 3, HISTORY = 4, MAX_UNSEEN = 3;
// the demo's highlights in tour order: [key, the name the state line uses]
export const HIGHLIGHTS = Object.freeze([
  ['belt', 'the asteroid belt'], ['comets', 'comets'], ['orbit', 'Earth orbit'], ['atmo', 'atmospheric flight'],
  ['cities', 'city skylines'], ['alps', 'the Alps'], ['pillars', 'the Zhangjiajie pillars'], ['weather', 'storm weather'],
  ['landing', 'the airliner landing'], ['docking', 'station docking'], ['moon', 'lunar orbit'], ['zoom', 'the Earth-zoom telescope'],
  ['training', 'live training'], ['narrator', 'the Narrator'],
]);
export const NAMES = Object.freeze(Object.fromEntries(HIGHLIGHTS));

// the first MAX_UNSEEN highlight names in tour order that are neither the current scene nor already seen (keys)
export function unseen(scene, seen = []) {
  const skip = new Set([scene, ...seen]);
  return HIGHLIGHTS.filter(([k]) => !skip.has(k)).map(([, n]) => n).slice(0, MAX_UNSEEN);
}

// 'State: <scene name>; not seen yet: a, b, c.' from {scene: key, seen: [keys]} (null: no state line)
export function stateLine(state) {
  if (!state || !Object.keys(state).length) return null; // Python: `not state` (None or {})
  const left = unseen(state.scene, state.seen || []);
  const tail = left.length ? `; not seen yet: ${left.join(', ')}.` : '; everything seen.';
  return `State: ${Object.hasOwn(NAMES, state.scene) ? NAMES[state.scene] : state.scene}${tail}`;
}

// notes: fact texts, all rendered (the retriever returns at most MAX_NOTES)
export function systemText(notes, state = null) {
  const lines = [PREAMBLE, ...(notes && notes.length ? ['Notes:', ...notes.map((n) => `- ${n}`)] : ['Notes: none.'])];
  const s = stateLine(state);
  if (s) lines.push(s);
  return lines.join('\n');
}

// [system, <= HISTORY previous messages, the user message]; history: [{role, content}] alternating, oldest first
export function messages(history, user, notes, state = null) {
  const h = history.map((m) => ({ role: m.role, content: m.content })).slice(-HISTORY);
  return [{ role: 'system', content: systemText(notes, state) }, ...h, { role: 'user', content: user }];
}

// the previous visitor message (the retriever's follow-up context), or ''
export function prevUser(history) {
  for (let i = history.length - 1; i >= 0; i--) if (history[i].role === 'user') return history[i].content;
  return '';
}
