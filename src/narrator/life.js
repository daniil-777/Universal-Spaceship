// life.js — the Narrator's open/close sequence (pure; Node-tested). Opening awaits the card's build, the frame-grab install
// and the model probe; a close or a reopen during any of them must stop the stale open where it stands. Every open takes
// a sequence number and continues after each await only while it is still the live one. Closing also schedules the
// release of the Narrator model after IDLE_RELEASE_MS; a reopen before then keeps it in memory.
// steps: build() once ever, show(), install(), startTick(), probe(), ready(probeResult), expand(), teardown(), idle()
export const IDLE_RELEASE_MS = 5 * 60 * 1000;

export function createLifecycle(steps, { idleMs = IDLE_RELEASE_MS } = {}) {
  let seq = 0, on = false, built = null, idleT = null;
  const live = (my) => on && my === seq;
  return {
    get on() { return on; },
    get seq() { return seq; },
    live,
    // a check for async work started now (a download, a snapshot): false once this session has closed or reopened
    guard() { const my = seq; return () => live(my); },
    async open() {
      if (on) { steps.expand(); return; }
      on = true; const my = ++seq;
      if (idleT !== null) { clearTimeout(idleT); idleT = null; }
      built ||= steps.build();
      await built; if (!live(my)) return;
      steps.show();
      await steps.install(); if (!live(my)) return;
      steps.startTick();
      const pr = await steps.probe(); if (!live(my)) return;
      steps.ready(pr);
    },
    close() {
      if (!on) return;
      on = false; seq++;
      steps.teardown();
      if (idleT !== null) clearTimeout(idleT);
      idleT = setTimeout(() => { idleT = null; if (!on) steps.idle(); }, idleMs);
      if (idleT && typeof idleT.unref === 'function') idleT.unref();
    },
  };
}
