// The landing's sound, made on the fly with WebAudio (no files): the engines' roar and whine following the spool, the wind
// rising with airspeed (and the gear's drag), the gear motor and its lock clunk, the tyres' squeal and thump at touchdown
// (louder the firmer it is), the spoilers' whoosh, the chute's crack as it blossoms, the rumble of the rollout and the brakes. Browsers only allow sound after a click, so it starts from the HUD's "voice" button.
export function createLandingAudio() {
  let ctx = null, n = null, last = { wow: false, gear: null, spoil: 0, chute: 0 };
  function noiseBuffer(c) { const b = c.createBuffer(1, c.sampleRate * 2, c.sampleRate), d = b.getChannelData(0); for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1; return b; }
  function start() {
    if (ctx) return; const C = window.AudioContext || window.webkitAudioContext; if (!C) return; ctx = new C(); const buf = noiseBuffer(ctx), master = ctx.createGain(); master.gain.value = 0.5; master.connect(ctx.destination);
    const src = (loop = true) => { const s = ctx.createBufferSource(); s.buffer = buf; s.loop = loop; return s; };
    const chain = (source, type, freq, q, gain) => { const f = ctx.createBiquadFilter(), g = ctx.createGain(); f.type = type; f.frequency.value = freq; f.Q.value = q; g.gain.value = gain; source.connect(f); f.connect(g); g.connect(master); return { f, g }; };
    const roarSrc = src(), windSrc = src(), rumbleSrc = src(), whine = ctx.createOscillator(); whine.type = 'sawtooth';
    const motorSrc = src(), roar = chain(roarSrc, 'lowpass', 220, 0.7, 0), wind = chain(windSrc, 'bandpass', 600, 0.6, 0), rumble = chain(rumbleSrc, 'lowpass', 90, 1.2, 0), wh = chain(whine, 'bandpass', 2400, 8, 0), motor = chain(motorSrc, 'bandpass', 140, 3, 0);
    for (const s of [roarSrc, windSrc, rumbleSrc, whine, motorSrc]) s.start();
    n = { roar, wind, rumble, wh, whine, motor, master, buf };
  }
  function burst(freq, q, gain, dur) {                       // a filtered noise burst (a whoosh), shaped to fade
    const t = ctx.currentTime, s = ctx.createBufferSource(), f = ctx.createBiquadFilter(), g = ctx.createGain(); s.buffer = n.buf; f.type = 'bandpass'; f.frequency.value = freq; f.Q.value = q;
    g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(gain, t + dur * 0.15); g.gain.exponentialRampToValueAtTime(0.0001, t + dur); s.connect(f); f.connect(g); g.connect(n.master); s.start(t); s.stop(t + dur + 0.05);
  }
  function clunk(f0, gain) {                                 // a mechanical thud
    const t = ctx.currentTime, o = ctx.createOscillator(), g = ctx.createGain(); o.frequency.setValueAtTime(f0, t); o.frequency.exponentialRampToValueAtTime(f0 * 0.5, t + 0.15); g.gain.setValueAtTime(gain, t); g.gain.exponentialRampToValueAtTime(0.0001, t + 0.25); o.connect(g); g.connect(n.master); o.start(t); o.stop(t + 0.3);
  }
  function thump(firm) {                                   // the tyres spin up: a squeal and a thump, scaled by the sink rate
    const t = ctx.currentTime, s = ctx.createBufferSource(), f = ctx.createBiquadFilter(), g = ctx.createGain(); s.buffer = n.buf; f.type = 'bandpass'; f.frequency.value = 1800; f.Q.value = 4;
    g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.25 + 0.5 * firm, t + 0.02); g.gain.exponentialRampToValueAtTime(0.0001, t + 0.45); s.connect(f); f.connect(g); g.connect(n.master); s.start(t); s.stop(t + 0.5);
    const o = ctx.createOscillator(), og = ctx.createGain(); o.frequency.setValueAtTime(70, t); o.frequency.exponentialRampToValueAtTime(35, t + 0.25); og.gain.setValueAtTime(0.4 + firm, t); og.gain.exponentialRampToValueAtTime(0.0001, t + 0.35); o.connect(og); og.connect(n.master); o.start(t); o.stop(t + 0.4);
  }
  return {
    start, get on() { return !!ctx; },
    stop() { if (ctx) { ctx.close(); ctx = null; n = null; } },
    update(sim) {
      if (!ctx || !n) return; const s = sim.flight, a = s.air, t = ctx.currentTime, k = (p, v) => p.setTargetAtTime(v, t, 0.15);
      k(n.roar.g.gain, 0.08 + 0.5 * s.spool); k(n.roar.f.frequency, 160 + 380 * s.spool); k(n.wh.g.gain, 0.012 * s.spool); k(n.whine.frequency, 900 + 1500 * s.spool);
      k(n.wind.g.gain, Math.min(0.45, (a.V / 90) ** 2 * 0.3 * (1 + 0.4 * s.gear + 0.5 * s.spoil))); k(n.wind.f.frequency, 300 + a.V * 6);   // the gear and spoilers roar in the airflow
      k(n.motor.g.gain, s.gear > 0.01 && s.gear < 0.99 ? 0.06 : 0);
      if (last.gear !== null && last.gear < 0.99 && s.gear >= 0.99) clunk(70, 0.5); last.gear = s.gear;   // down and locked
      if (s.spoil > 0.05 && last.spoil <= 0.05) burst(900, 0.8, 0.3, 1.2); last.spoil = s.spoil;
      if (s.chute === 1 && last.chute !== 1) { burst(420, 0.7, 0.45, 1.6); clunk(55, 0.6); } last.chute = s.chute;
      k(n.rumble.g.gain, s.wow ? Math.min(0.6, a.gs / 60) * (0.6 + 0.8 * s.brake) : 0);
      if (s.wow && !last.wow && s.touchdown) thump(Math.min(1, s.touchdown.sinkFps / 6)); last.wow = s.wow;
    },
  };
}
