// DOM wiring for the page: panels, controls, HUD readouts, training charts and keyboard state. The app passes a set
// of handlers; the UI never touches the simulation directly. All text goes through textContent.
import { LineChart, HeatGrid, Bars, FleetMap, fmtSteps, fmtNum } from './charts.js';
import { ENV, N_RAYS } from './env.js';

const CHARTS = [
  ['epRet', 'Episode return', (v) => fmtNum(v, 1)], ['epLen', 'Episode length (steps, exploring)', (v) => fmtNum(v, 0)], ['evalLen', 'Episode length (steps, deterministic check-up)', (v) => fmtNum(v, 0)], ['collPer1k', 'Collisions per 1k steps', (v) => fmtNum(v, 2)],
  ['pgLoss', 'Policy loss', (v) => fmtNum(v, 4)], ['vLoss', 'Value loss', (v) => fmtNum(v, 3)], ['entropy', 'Policy entropy', (v) => fmtNum(v, 3)],
  ['kl', 'Approx. KL', (v) => fmtNum(v, 4)], ['clipFrac', 'Clip fraction', (v) => fmtNum(v, 3)], ['explainedVar', 'Explained variance', (v) => fmtNum(v, 3)],
];

const moonKm = (v) => { const x = 500 * 120 ** (v / 100), p = 10 ** (Math.floor(Math.log10(x)) - 1); return Math.round(x / p) * p; }, fmtKm = (km) => km.toLocaleString('en-US') + ' km';   // the lunar orbit's height slider (0–100, two significant digits)
export function createUI(h) {
  const $ = (id) => document.getElementById(id);
  const ui = { keys: new Set(), curriculum: true, trainSpeed: 'balanced' };
  const seg = (el, attr, cb) => el.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => { el.querySelectorAll('button').forEach((x) => x.classList.toggle('on', x === b)); cb(b.dataset[attr]); }));
  const setSeg = (el, attr, v) => el.querySelectorAll('button').forEach((x) => x.classList.toggle('on', x.dataset[attr] === String(v)));

  // header
  seg($('modeSeg'), 'mode', (m) => h.colorMode(+m));
  $('btnPlaybox').addEventListener('click', () => { $('playbox').classList.toggle('hidden'); $('btnPlaybox').classList.toggle('on', !$('playbox').classList.contains('hidden')); });
  $('btnTraining').addEventListener('click', () => { $('training').classList.toggle('hidden'); $('btnTraining').classList.toggle('on', !$('training').classList.contains('hidden')); ui.drawCharts(); });

  // playbox
  $('btnPlay').addEventListener('click', () => h.play());
  $('btnReset').addEventListener('click', () => h.reset());
  const range = (id, vid, fmt, cb) => { const el = $(id); const upd = () => { $(vid).textContent = fmt(+el.value); }; el.addEventListener('input', () => { upd(); cb(+el.value); }); upd(); };
  range('simSpeed', 'simSpeedV', (v) => v.toFixed(2) + '×', (v) => h.simSpeed(v));
  range('density', 'densityV', (v) => String(v), (v) => h.density(v));
  range('astSpeed', 'astSpeedV', (v) => v.toFixed(2) + '×', (v) => h.astSpeed(v));
  range('comets', 'cometsV', (v) => String(v), (v) => h.comets(v));
  range('cometSpeed', 'cometSpeedV', (v) => v.toFixed(2) + '×', (v) => h.cometSpeed(v));
  range('weather', 'weatherV', (v) => v.toFixed(2), (v) => h.weather(v)); range('warp', 'warpV', (v) => [1, 10, 60, 300][v] + '×', (v) => h.warp && h.warp([1, 10, 60, 300][v])); range('wind', 'windV', (v) => v.toFixed(1) + '×', (v) => h.wind(v)); range('cover', 'coverV', (v) => v.toFixed(1) + '×', (v) => h.cover(v)); range('turb', 'turbV', (v) => v.toFixed(1) + '×', (v) => h.turb(v)); seg($('skySeg'), 'sky', (k) => h.sky(k)); seg($('orbitSeg'), 'orbit', (b) => h.orbit(b));
  range('moonAlt', 'moonAltV', (v) => fmtKm(moonKm(v)), (v) => h.moonAlt(moonKm(v)));   // log scale: 500 km … 60,000 km
  seg($('camSeg'), 'cam', (c) => h.camera(c));
  seg($('routeSeg'), 'route', (r) => h.route(r));
  const sw = (id, cb) => { const el = $(id); el.addEventListener('change', () => cb(el.checked)); return el; };
  sw('swAutoThr', (b) => h.autoThr(b));
  sw('swBoard', (b) => h.board(b)); sw('swTrail', (b) => h.trail(b)); sw('swSensors', (b) => h.sensors(b)); sw('swManual', (b) => h.manual(b)); sw('swInvert', (b) => h.invert(b)); sw('swLowPass', (b) => h.lowPass(b)); sw('swAtmo', (b) => h.atmo(b)); sw('swSkyline', (b) => h.skyline(b));

  // training
  $('btnTrain').addEventListener('click', () => h.train());
  $('btnResetPolicy').addEventListener('click', () => h.resetPolicy());
  $('btnSave').addEventListener('click', () => h.save());
  $('btnLoad').addEventListener('click', () => $('fileLoad').click());
  $('fileLoad').addEventListener('change', async (e) => { const f = e.target.files[0]; if (!f) return; try { h.load(JSON.parse(await f.text())); } catch (err) { ui.toast('could not read that file'); } e.target.value = ''; });
  seg($('trainSpeedSeg'), 'ts', (s) => { ui.trainSpeed = s; h.trainSpeed(s); });
  sw('swCurriculum', (b) => { ui.curriculum = b; h.curriculum(b); });

  // charts
  const chartsEl = $('charts'), tip = $('tip'), charts = {};
  for (const [key, title, fmt] of CHARTS) {
    const box = document.createElement('div'); box.className = 'chart';
    const head = document.createElement('div'); head.className = 'head';
    const l = document.createElement('span'); l.className = 'lbl'; l.textContent = title;
    const cur = document.createElement('span'); cur.className = 'cur'; cur.textContent = '–';
    head.append(l, cur); const cv = document.createElement('canvas'); box.append(head, cv); chartsEl.append(box);
    charts[key] = { chart: new LineChart(cv, { format: fmt, tip }), cur, fmt };
  }
  const fleet = new FleetMap($('fleet'), ENV.xHalf, ENV.yHalf);
  const sensor = new HeatGrid($('sensorGrid'), ENV.rays.nAz, ENV.rays.nEl), radar = new HeatGrid($('radarGrid'), ENV.rays.nAz, ENV.rays.nEl, '#ff8a6b');
  const bars = new Bars($('actionBars'), ['pitch', 'yaw', 'roll', 'thrust']);
  ui.drawCharts = () => { if ($('training').classList.contains('hidden')) return; for (const k in charts) charts[k].chart.draw(); fleet.draw(); };

  // keyboard
  window.addEventListener('keydown', (e) => {
    if (e.target && /INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) return;
    ui.keys.add(e.code);
    if (e.code === 'Space') { e.preventDefault(); h.play(); }
    if (e.code === 'KeyM') h.cycleMode();
    if (e.code === 'KeyH') h.hideUI();
  });
  window.addEventListener('keyup', (e) => ui.keys.delete(e.code));
  window.addEventListener('blur', () => ui.keys.clear());

  // ---- update API ----
  ui.setPlaying = (b) => { $('btnPlay').textContent = b ? 'Pause' : 'Play'; };
  ui.setTrainingActive = (b) => { $('btnTrain').textContent = b ? 'Stop' : 'Train'; $('btnTrain').classList.toggle('primary', !b); };
  ui.setMode = (m) => { document.body.dataset.mode = String(m); setSeg($('modeSeg'), 'mode', m); ui.drawCharts(); };
  ui.setInvert = (b) => { document.body.dataset.invert = b ? '1' : '0'; $('swInvert').checked = b; };
  ui.setCamera = (c) => setSeg($('camSeg'), 'cam', c);
  ui.setRoute = (r) => setSeg($('routeSeg'), 'route', r);
  let manualTag = 'autopilot', phaseTag = 'orbit';
  const tag = () => { $('pilotTag').textContent = manualTag + ' · ' + phaseTag; };
  ui.setManual = (b) => { $('swManual').checked = b; manualTag = b ? 'manual' : 'autopilot'; tag(); };
  ui.setPhase = (t) => { phaseTag = t; tag(); };
  ui.setControls = ({ trail, sensors, density, astSpeed, simSpeed, comets, atmo, skyline, board, weather, wind, cover, turb, sky, autoThr, orbit, moonAlt }) => {
    if (orbit != null) setSeg($('orbitSeg'), 'orbit', orbit); if (moonAlt != null) { $('moonAlt').value = Math.round(100 * Math.log(moonAlt / 500) / Math.log(120)); $('moonAltV').textContent = fmtKm(Math.round(moonAlt)); }
    if (sky != null) $('skySeg').querySelectorAll('button').forEach((b) => b.classList.toggle('on', b.dataset.sky === sky));
    for (const [k, v] of [['wind', wind], ['cover', cover], ['turb', turb]]) if (v != null) { $(k).value = v; $(k + 'V').textContent = v.toFixed(1) + '×'; }
    if (weather != null) { $('weather').value = weather; $('weatherV').textContent = weather.toFixed(2); } if (autoThr != null) $('swAutoThr').checked = autoThr;
    if (board != null) $('swBoard').checked = board;
    if (atmo != null) { $('swAtmo').checked = atmo; } if (skyline != null) { $('swSkyline').checked = skyline; }
    if (comets != null) { $('comets').value = comets; $('cometsV').textContent = String(comets); }
    if (trail != null) $('swTrail').checked = trail; if (sensors != null) $('swSensors').checked = sensors;
    if (density != null) { $('density').value = density; $('densityV').textContent = String(density); }
    if (astSpeed != null) { $('astSpeed').value = astSpeed; $('astSpeedV').textContent = astSpeed.toFixed(2) + '×'; }
    if (simSpeed != null) { $('simSpeed').value = simSpeed; $('simSpeedV').textContent = simSpeed.toFixed(2) + '×'; }
  };
  ui.showPanels = (playbox, training) => { $('playbox').classList.toggle('hidden', !playbox); $('btnPlaybox').classList.toggle('on', playbox); $('training').classList.toggle('hidden', !training); $('btnTraining').classList.toggle('on', training); if (training) requestAnimationFrame(() => ui.drawCharts()); };
  ui.hud = ({ speed, value, laps, flight, crashes, fps }) => {
    $('hudSpeed').textContent = fmtNum(speed, 1); $('hudValue').textContent = fmtNum(value, 2); $('hudLaps').textContent = String(laps);
    $('hudFlight').textContent = Math.round(flight) + ' s'; $('hudCrashes').textContent = String(crashes); $('hudFps').textContent = fmtNum(fps, 0);
  };
  const sensorVals = new Float32Array(N_RAYS);
  ui.radar = (vals) => { radar.set(vals); radar.draw(); };
  ui.sensors = (rayHit, range) => { for (let i = 0; i < N_RAYS; i++) sensorVals[i] = 1 - rayHit[i] / range; sensor.set(sensorVals); sensor.draw(); };
  ui.actions = (a) => { bars.set(a); bars.draw(); };
  ui.metrics = (m, hist, marks) => {
    $('stSteps').textContent = fmtSteps(m.step); $('stEpisodes').textContent = fmtSteps(m.episodes); $('stUpdates').textContent = String(m.updates);
    $('stSps').textContent = m.sps ? fmtSteps(m.sps) : '–'; $('stLevel').textContent = Number.isFinite(m.level) ? Math.round(m.level * 100) + '%' : '–';
    $('stElapsed').textContent = ''; $('stElapsed').append(m.elapsed >= 90 ? (m.elapsed / 60).toFixed(1) : String(Math.round(m.elapsed))); const sm = document.createElement('small'); sm.textContent = m.elapsed >= 90 ? 'min' : 's'; $('stElapsed').append(sm);
    $('stPhase').textContent = m.phase === 'pretrained' ? 'pretrained · press Train to continue' : (m.phase || 'idle');
    for (const k in charts) { charts[k].chart.setData(hist.step || [], hist[k] || [], marks); charts[k].cur.textContent = charts[k].fmt(m[k]); }
    ui.drawCharts();
  };
  ui.fleet = (pts) => { fleet.set(pts); if (!$('training').classList.contains('hidden')) fleet.draw(); };
  ui.policyInfo = (t) => { $('policyInfo').textContent = t; };
  let toastTimer = 0;
  ui.toast = (msg, ms = 2600) => { const t = $('toast'); t.textContent = msg; t.classList.add('show'); clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove('show'), ms); };
  const creditDefault = document.getElementById('credit') ? document.getElementById('credit').textContent : '';
  ui.setCredit = (t) => { const c = document.getElementById('credit'); if (c) c.textContent = t || creditDefault; };
  ui.loading = (msg) => { const l = document.getElementById('loadingMsg'); if (l) l.textContent = msg; };
  ui.ready = () => { document.getElementById('loading').classList.add('hidden'); };
  ui.trainingVisible = () => !$('training').classList.contains('hidden');
  window.addEventListener('resize', () => { sensor.draw(); bars.draw(); ui.drawCharts(); });
  sensor.draw(); bars.draw();
  return ui;
}
