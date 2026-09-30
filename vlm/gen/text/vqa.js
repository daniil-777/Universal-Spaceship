// vlm/gen/text/vqa.js — question families (spec §5.2) and LRV-style negatives (§5.5). Each family reads facts only (the
// monitor-attributed is_safe, why, what_to_do, monitor_ttc and monitor_clearance read the row's Context monitor) and returns
// the bank template ids, the prompt and answer slots, hidden fact checks (the polarity of a yes/no answer), the cited fact
// ids and an answerKey for rejection sampling; null when its facts are missing or the answer would be ambiguous.
import { fmtSlot, factsOf, catSlot, entSlot, countSlot, clockSlot, regionSlot, compassSlot, rangeSlot, rangeText, roundNice, verifyTemplateItem, parseClaims, checkClaim, ROUTE_NAMES, NOUNS, countPhrase, numText, unitText, W, outcomeSlot, prefOutcome, outcomeAgrees } from './verify.js';
import { mulberry32 } from '../../../src/mathx.js';
export { W };
import { REASON_TEXT, ACTION_TEXT, TTC_RANGE, CLR_RANGE_U, normContext, contextSupplies } from './context.js';
import { CAUSE_REASONS } from './verify_words.js';
export { CAUSE_REASONS };
import { defaultBank, pickForm, render, polish } from './paraphrase.js';

const val = (F, id) => (F[id] && F[id].v !== null && F[id].v !== undefined ? F[id].v : null);
const list = (xs) => (xs.length < 2 ? xs.join('') : `${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}`);
export const KIND_A = Object.freeze({ rock: 'a rock', comet: 'a comet', satellite: 'a satellite', airliner: 'an airliner', birds: 'a flock of birds' });
export const KIND_N = Object.freeze({ rock: 'rock', comet: 'comet', satellite: 'satellite', airliner: 'airliner', birds: 'flock of birds' });
export const reasonsText = (rs) => list(rs.map((r) => REASON_TEXT[r]));
const BIN_U = [5, 15, 40], BIN_M = [95, 285, 760];
const binOf = (x, edges) => edges.filter((e) => x >= e).length;
const binSlot = (id, f, x, edges, unit, derive) => { const b = binOf(x, edges); return { b, s: rangeSlot(id, f, b ? edges[b - 1] : 0, b < edges.length ? edges[b] : Infinity, unit, derive) }; };
const uBin = (F, id) => { const v = val(F, id); if (v === null) return null; const { b } = binSlot(id, F[id], Math.abs(v), BIN_U, 'u'); return { b, s: rangeSlot(id, F[id], b ? BIN_M[b - 1] : 0, b < 3 ? BIN_M[b] : Infinity, 'm') }; };
const firstInFrame = (F) => { for (let i = 0; i < 6; i++) if (val(F, `hazard.${i}.in_frame`) === true) return i; return -1; };
const sys = (rng) => (rng() < 0.2 ? 'imperial' : 'metric');
const yes = (id, v, d) => catSlot(id, v, '', d);
const Q = (qid, aid, a, fact_ids, answerKey, extra = {}) => ({ qid, aid, q: extra.q || {}, a, checks: extra.checks || [], fact_ids, answerKey: String(answerKey), monitor: !!extra.monitor });
const mon = (ctx) => { const c = normContext(ctx); return c && c.monitor ? c.monitor : null; };
const reasonSlot = (m) => ({ text: reasonsText(m.reasons), fact_id: 'safety.reasons', value: m.reasons, unit: null, lo: null, hi: null, type: 'reason' });
const verdictSlot = (m) => ({ text: m.verdict, fact_id: 'safety.verdict', value: m.verdict, unit: null, lo: null, hi: null, type: 'verdict' });
const actionSlot = (m) => ({ text: ACTION_TEXT[m.action], fact_id: 'safety.best_action', value: m.action, unit: null, lo: null, hi: null, type: 'action' });
const monRange = (id, [lo, hi], unit) => ({ text: rangeText(lo, hi, unit), fact_id: id, value: null, unit, lo, hi, type: 'monitor_range', shownLo: lo, shownHi: hi });
const SA = ['S', 'A'], FLIGHT = ['S', 'A', 'L', 'D'];
const fam = (id, group, families, ask) => ({ id, group, families, ask });

export const QFAMILIES = [
  // perception (S/A; the image clock is screen-relative, 12 = up)
  fam('count_kind', 'perception', SA, (r) => { const F = factsOf(r), k = val(F, 'kinds_in_frame'), n = val(F, 'hazards.count_in_frame'); if (!k || k.length !== 1 || !(n >= 1)) return null;
    return Q('count_kind_q', 'count_kind_a', { kind_count: countSlot('hazards.count_in_frame', n, k[0]) }, ['hazards.count_in_frame', 'kinds_in_frame'], Math.min(n, 4), { q: { kind_pl: NOUNS[k[0]][1] }, checks: [yes('kinds_in_frame', k)] }); }),
  fam('presence_kind', 'perception', SA, (r, rng) => { const F = factsOf(r), k = val(F, 'kinds_in_frame'); if (!k) return null;
    const present = k.length && rng() < 0.5, pool = Object.keys(KIND_A).filter((x) => !k.includes(x)), kind = present ? k[Math.floor(rng() * k.length)] : pool[Math.floor(rng() * pool.length)];
    return present ? Q('presence_q', 'presence_yes_a', { a_kind: catSlot('kinds_in_frame', true, KIND_A[kind], ['has', kind]) }, ['kinds_in_frame'], 'yes', { q: { a_kind: KIND_A[kind] } })
      : Q('presence_q', 'presence_no_a', { kind_none: countSlot('kinds_in_frame', 0, kind, ['count_of', kind]) }, ['kinds_in_frame'], 'no', { q: { a_kind: KIND_A[kind] } }); }),
  fam('nearest_kind', 'perception', SA, (r) => { const F = factsOf(r), i = firstInFrame(F); if (i < 0) return null; const k = val(F, `hazard.${i}.kind`);
    return Q('nearest_kind_q', 'nearest_kind_a', { a_kind: catSlot(`hazard.${i}.kind`, k, KIND_A[k]) }, [`hazard.${i}.kind`, `hazard.${i}.in_frame`], k); }),
  fam('nearest_side', 'perception', SA, (r) => { const F = factsOf(r), i = firstInFrame(F), s = i < 0 ? null : val(F, `hazard.${i}.side`); if (!s) return null;
    return Q('nearest_side_q', 'nearest_side_a', { side_at: catSlot(`hazard.${i}.side`, s, W.side[s]) }, [`hazard.${i}.side`], s, { q: { kind: KIND_N[val(F, `hazard.${i}.kind`)] } }); }),
  fam('nearest_clock', 'perception', SA, (r) => { const F = factsOf(r), i = firstInFrame(F), c = i < 0 ? null : val(F, `hazard.${i}.clock`); if (!c) return null;
    return Q('nearest_clock_q', 'nearest_clock_a', { clock: clockSlot(`hazard.${i}.clock`, c) }, [`hazard.${i}.clock`, `hazard.${i}.kind`], c, { q: { kind: KIND_N[val(F, `hazard.${i}.kind`)] } }); }),
  fam('size_nearest', 'perception', SA, (r) => { const F = factsOf(r), i = firstInFrame(F), s = i < 0 ? null : val(F, `hazard.${i}.size_bin`); if (!s) return null;
    return Q('size_q', 'size_a', { size: catSlot(`hazard.${i}.size_bin`, s, W.size[s]) }, [`hazard.${i}.size_bin`, `hazard.${i}.kind`], s, { q: { kind: KIND_N[val(F, `hazard.${i}.kind`)] } }); }),
  fam('hazards_in_frame', 'perception', SA, (r) => { const n = val(factsOf(r), 'hazards.count_in_frame'); if (n === null) return null;
    return Q('hazards_q', 'hazards_a', { hazards: countSlot('hazards.count_in_frame', n, 'hazard') }, ['hazards.count_in_frame'], Math.min(n, 4)); }),
  ...['earth', 'moon'].map((b) => fam(`${b}_visible`, 'perception', ['S', 'D'], (r) => { const v = val(factsOf(r), `${b}_in_frame`); if (typeof v !== 'boolean') return null;
    return Q(`${b}_q`, `${b}_${v ? 'yes' : 'no'}_a`, {}, [`${b}_in_frame`], v ? 'yes' : 'no', { checks: [yes(`${b}_in_frame`, v)] }); })),
  fam('sun_lit', 'perception', ['S', 'D'], (r) => { const v = val(factsOf(r), 'sun.lit'); if (v === null || (v > 0.1 && v < 0.9)) return null;
    return Q('sun_q', v >= 0.9 ? 'sun_lit_a' : 'sun_shadow_a', {}, ['sun.lit'], v >= 0.9 ? 'lit' : 'shadow', { checks: [yes('sun.lit', v >= 0.9, ['gt', 0.5])] }); }),
  // distance (bins < 5, 5-15, 15-40, > 40 u, stated in metres) and motion
  fam('dist_nearest_bin', 'distance', SA, (r) => { const x = uBin(factsOf(r), 'hazard.0.dist_u'); return x && Q('dist_bin_q', 'dist_bin_a', { dist_bin: x.s }, ['hazard.0.dist_u'], x.b); }),
  fam('dist_nearest_m', 'distance', SA, (r, rng) => { const F = factsOf(r); if (val(F, 'hazard.0.dist_u') === null) return null; const s = fmtSlot('hazard.0.dist_u', F['hazard.0.dist_u'], { system: sys(rng) });
    return Q('dist_q', 'dist_a', { dist: s }, ['hazard.0.dist_u'], binOf(F['hazard.0.dist_u'].v, BIN_U)); }),
  fam('clearance_ground_bin', 'distance', ['A'], (r) => { const x = uBin(factsOf(r), 'clearance.ground_u'); return x && Q('clr_bin_q', 'clr_bin_a', { clr_bin: x.s }, ['clearance.ground_u'], x.b); }),
  fam('ceiling_margin', 'distance', SA, (r, rng) => { const F = factsOf(r), v = val(F, 'edges.ceiling_u'); if (!(v > 0.05)) return null;
    return Q('ceiling_q', 'ceiling_a', { ceiling: fmtSlot('edges.ceiling_u', F['edges.ceiling_u'], { system: sys(rng) }) }, ['edges.ceiling_u'], binOf(v, BIN_U)); }),
  fam('closing_or_receding', 'motion', SA, (r, rng) => { const F = factsOf(r), v = val(F, 'hazard.0.closing_u_s'); if (v === null || Math.abs(v) < 0.5) return null;
    return Q('closing_q', v > 0 ? 'closing_yes_a' : 'closing_no_a', { rate: fmtSlot('hazard.0.closing_u_s', F['hazard.0.closing_u_s'], { abs: true, system: sys(rng) }) }, ['hazard.0.closing_u_s'], v > 0 ? 'closing' : 'receding',
      { checks: [yes('hazard.0.closing_u_s', Math.sign(v), ['sign'])] }); }),
  fam('ttc_bin', 'motion', SA, (r) => { const F = factsOf(r), t = val(F, 'hazard.0.ttc_s'); if (t === null || t < 0) return null; const { b, s } = binSlot('hazard.0.ttc_s', F['hazard.0.ttc_s'], t, [1, 3, 6], 's');
    return Q('ttc_q', 'ttc_a', { ttc: s }, ['hazard.0.ttc_s'], b); }),
  fam('cpa_bin', 'motion', SA, (r) => { const x = uBin(factsOf(r), 'hazard.0.cpa_u'); return x && Q('cpa_q', 'cpa_a', { cpa: x.s }, ['hazard.0.cpa_u'], x.b); }),
  fam('speed_band', 'motion', SA, (r, rng) => { const F = factsOf(r), v = val(F, 'ship.speed_m_s'); if (!(v > 0.5)) return null;
    return Q('speed_q', 'speed_a', { speed: fmtSlot('ship.speed_m_s', F['ship.speed_m_s'], { system: sys(rng) }) }, ['ship.speed_m_s'], roundNice(v).value); }),
  fam('pitch_updown', 'motion', SA, (r) => { const F = factsOf(r), p = val(F, 'ship.pitch_deg'); if (p === null) return null;
    if (Math.abs(p) < 3) return Q('pitch_q', 'pitch_level_a', {}, ['ship.pitch_deg'], 'level', { checks: [yes('ship.pitch_deg', true, ['in_band', -3, 3])] });
    return Q('pitch_q', p > 0 ? 'pitch_up_a' : 'pitch_down_a', { pitch: fmtSlot('ship.pitch_deg', F['ship.pitch_deg'], { abs: true }) }, ['ship.pitch_deg'], p > 0 ? 'up' : 'down', { checks: [yes('ship.pitch_deg', Math.sign(p), ['sign'])] }); }),
  // terrain and weather (A)
  fam('terrain_type', 'terrain', ['A'], (r) => { const w = val(factsOf(r), 'world'); return W.world[w] ? Q('terrain_q', 'terrain_a', { terrain: catSlot('world', w, W.world[w]) }, ['world'], W.world[w]) : null; }),
  fam('city_type', 'terrain', ['A'], (r) => { const k = val(factsOf(r), 'route'); return ROUTE_NAMES[k] ? Q('route_q', 'route_a', { route: entSlot('route', ROUTE_NAMES[k], ['map', ROUTE_NAMES]) }, ['route'], k) : null; }),
  fam('in_tunnel', 'terrain', ['A'], (r) => { const v = val(factsOf(r), 'in_tunnel'); if (typeof v !== 'boolean') return null; return Q('tunnel_q', v ? 'tunnel_yes_a' : 'tunnel_no_a', {}, ['in_tunnel'], v ? 'yes' : 'no', { checks: [yes('in_tunnel', v)] }); }),
  fam('sky_preset', 'weather', ['A'], (r) => { const p = val(factsOf(r), 'weather.preset'); return W.sky[p] ? Q('sky_q', 'sky_a', { sky: catSlot('weather.preset', p, W.sky[p]) }, ['weather.preset'], p) : null; }),
  fam('turbulence_class', 'weather', ['A'], (r) => { const t = val(factsOf(r), 'air.turbulence'); return W.turb[t] ? Q('turb_q', 'turb_a', { turb: catSlot('air.turbulence', t, W.turb[t]) }, ['air.turbulence'], t) : null; }),
  fam('storm_cell_near', 'weather', ['A'], (r, rng) => { const F = factsOf(r), cells = val(F, 'weather.cells'); if (!Array.isArray(cells)) return null; const storms = cells.filter((c) => c.type === 1);
    if (!storms.length) return Q('storm_q', 'storm_no_a', {}, ['weather.cells'], 'no', { checks: [yes('weather.cells', 0, ['count_type', 1])] });
    const d = Math.min(...storms.map((c) => c.dist_u)), s = { ...fmtSlot('weather.cells', { v: d, unit: 'u' }, { system: sys(rng) }), derive: ['min_dist_type', 1] };
    return Q('storm_q', 'storm_yes_a', { cell_dist: s }, ['weather.cells'], 'yes'); }),
  fam('in_cloud', 'weather', ['A'], (r) => { const v = val(factsOf(r), 'air.in_cloud'); if (v === null || (v > 0.02 && v < 0.5)) return null;
    return Q('cloud_q', v >= 0.5 ? 'cloud_yes_a' : 'cloud_no_a', {}, ['air.in_cloud'], v >= 0.5 ? 'yes' : 'no', { checks: [yes('air.in_cloud', v >= 0.5, ['gt', 0.25])] }); }),
  // air data (A)
  fam('stall_margin_bin', 'air', ['A'], (r) => { const F = factsOf(r), m = val(F, 'air.stall_margin_deg'); if (m === null || m < 0) return null; const { b, s } = binSlot('air.stall_margin_deg', F['air.stall_margin_deg'], m, [2, 5, 10], 'deg');
    return Q('stall_q', 'stall_a', { stall_margin: s }, ['air.stall_margin_deg'], b); }),
  fam('g_load_bin', 'air', ['A'], (r) => { const F = factsOf(r), g = val(F, 'air.n_g'); if (!(g > 0.05)) return null; return Q('g_q', 'g_a', { g: fmtSlot('air.n_g', F['air.n_g']) }, ['air.n_g'], binOf(g, [0.8, 1.2, 2])); }),
  fam('airspeed_band', 'air', ['A'], (r) => { const b = val(factsOf(r), 'air.speed_band'); return W.band[b] ? Q('band_q', 'band_a', { band: catSlot('air.speed_band', b, W.band[b]) }, ['air.speed_band'], b) : null; }),
  // landing (L; aviation units)
  ...[['on_glideslope', 'ils.gs_dots', 'gs', ['above', 'below']], ['on_localizer', 'ils.loc_dots', 'loc', ['right', 'left']]].map(([id, fid, k, sides]) => fam(id, 'landing', ['L'], (r) => {
    const F = factsOf(r), d = val(F, fid); if (d === null) return null;
    if (Math.abs(d) <= 1) return Q(`${k}_q`, `${k}_on_a`, {}, [fid], 'on', { checks: [yes(fid, true, ['in_band', -1, 1])] });
    return Q(`${k}_q`, `${k}_off_a`, { [`${k}_dev`]: fmtSlot(fid, F[fid], { abs: true }), [`${k}_side`]: catSlot(fid, Math.sign(d), d > 0 ? sides[0] : sides[1], ['sign']) }, [fid], d > 0 ? sides[0] : sides[1]); })),
  fam('papi_reading', 'landing', ['L'], (r) => { const w = val(factsOf(r), 'papi_whites_cam'); if (w === null) return null;
    return Q('papi_q', 'papi_a', { papi_white: countSlot('papi_whites_cam', w, 'white'), papi_red: countSlot('papi_whites_cam', 4 - w, 'red', ['papi_red']), papi_path: catSlot('papi_whites_cam', w, W.papi[w]) }, ['papi_whites_cam'], w); }),
  fam('stabilized', 'landing', ['L'], (r) => { const F = factsOf(r), g = val(F, 'gates'); if (!g || val(F, 'wow') === true || !['ALT', 'GS'].includes(val(F, 'vert_mode'))) return null;
    const bad = Object.keys(g).filter((k) => !g[k]).sort();
    return bad.length ? Q('stab_q', 'stab_no_a', { gates_failed: catSlot('gates', bad.join(','), list(bad.map((k) => W.gate[k] || k)), ['false_keys']) }, ['gates'], 'no')
      : Q('stab_q', 'stab_yes_a', {}, ['gates'], 'yes', { checks: [yes('gates', true, ['all_true'])] }); }),
  fam('gear_down', 'landing', ['L'], (r) => { const g = val(factsOf(r), 'cfg.gear'); if (!g) return null;
    return g === 'down' ? Q('gear_q', 'gear_yes_a', {}, ['cfg.gear'], 'yes', { checks: [yes('cfg.gear', 'down')] }) : Q('gear_q', 'gear_no_a', { gear: catSlot('cfg.gear', g, g === 'up' ? 'up' : 'still in transit') }, ['cfg.gear'], 'no'); }),
  fam('head_cross_wind', 'landing', ['L'], (r) => { const F = factsOf(r), h = val(F, 'wind.head_kt'), c = val(F, 'wind.cross_kt'); if (h === null || c === null || Math.abs(h) < 1 || Math.abs(c) < 1) return null;
    return Q('wind_q', 'wind_a', { head: fmtSlot('wind.head_kt', F['wind.head_kt'], { abs: true, system: 'aviation' }), head_kind: catSlot('wind.head_kt', Math.sign(h), h > 0 ? 'headwind' : 'tailwind', ['sign']),
      cross: fmtSlot('wind.cross_kt', F['wind.cross_kt'], { abs: true, system: 'aviation' }), cross_side: catSlot('wind.cross_kt', Math.sign(c), c > 0 ? 'from the right' : 'from the left', ['sign']) }, ['wind.head_kt', 'wind.cross_kt'], h > 0 ? 'head' : 'tail'); }),
  fam('phase_landing', 'landing', ['L'], (r) => { const m = val(factsOf(r), 'vert_mode'); return W.vert[m] ? Q('lphase_q', 'lphase_a', { lphase: catSlot('vert_mode', m, W.vert[m]) }, ['vert_mode'], m) : null; }),
  fam('windsock_dir', 'landing', ['L'], (r) => { const F = factsOf(r); return val(F, 'windsock.from_deg') === null ? null : Q('sock_q', 'sock_a', { sock: fmtSlot('windsock.from_deg', F['windsock.from_deg']) }, ['windsock.from_deg'], Math.round(F['windsock.from_deg'].v / 45) % 8); }),
  fam('landing_weather', 'landing', ['L'], (r) => { const F = factsOf(r), t = val(F, 'scene.time'), v = val(F, 'scene.vis'); if (!W.time[t] || !W.vis[v]) return null;
    return Q('lweather_q', 'lweather_a', { light: catSlot('scene.time', t, W.time[t]), visibility: catSlot('scene.vis', v, W.vis[v]) }, ['scene.time', 'scene.vis'], `${t}/${v}`); }),
  // docking (D)
  fam('docking_phase', 'docking', ['D'], (r) => { const p = val(factsOf(r), 'phase'); return W.phase[p] ? Q('dphase_q', 'dphase_a', { dphase: catSlot('phase', p, W.phase[p]) }, ['phase'], p) : null; }),
  fam('station_distance', 'docking', ['D'], (r) => { const F = factsOf(r), s = stationSlot(F); return s ? Q('station_q', 'station_a', { station_dist: s }, ['station_distance_bin'], val(F, 'station_distance_bin')) : null; }),
  fam('closing_vs_limit', 'docking', ['D'], (r) => { const F = factsOf(r), c = val(F, 'closing_cms'), l = val(F, 'corridor_limit_cms'); if (c === null || l === null || !(c > 0.05) || !(l > 0)) return null;
    const k = c > l ? 'above' : 'below';
    return Q('dclosing_q', `dclosing_${k}_a`, { closing: fmtSlot('closing_cms', F.closing_cms, { unit: 'cm/s' }), limit: fmtSlot('corridor_limit_cms', F.corridor_limit_cms, { unit: 'cm/s' }) }, ['closing_cms', 'corridor_limit_cms'], k,
      { checks: [yes('closing_cms', k, ['cmp_fact', 'corridor_limit_cms'])] }); }),
  fam('aligned', 'docking', ['D'], (r) => { const F = factsOf(r), a = val(F, 'att_err_deg'); if (a === null || Math.abs(a - 2) < 0.05) return null;
    return Q('aligned_q', a < 2 ? 'aligned_yes_a' : 'aligned_no_a', { att: fmtSlot('att_err_deg', F.att_err_deg) }, ['att_err_deg'], a < 2 ? 'yes' : 'no', { checks: [yes('att_err_deg', a < 2, ['lt', 2])] }); }),
  fam('in_corridor', 'docking', ['D'], (r) => { const F = factsOf(r), v = val(F, 'in_cone'); if (typeof v !== 'boolean' || !['CORRIDOR', 'H2', 'FINAL'].includes(val(F, 'phase'))) return null;
    return Q('cone_q', v ? 'cone_yes_a' : 'cone_no_a', {}, ['in_cone', 'phase'], v ? 'yes' : 'no', { checks: [yes('in_cone', v)] }); }),
  fam('fuel_left', 'docking', ['D'], (r) => { const F = factsOf(r), v = val(F, 'fuel_frac'); if (!(v > 0.005)) return null; return Q('fuel_q', 'fuel_a', { fuel: fmtSlot('fuel_frac', F.fuel_frac) }, ['fuel_frac'], binOf(v, [0.4, 0.7])); }),
  fam('failed_jets', 'docking', ['D'], (r) => { const j = val(factsOf(r), 'jets_failed'); if (!Array.isArray(j)) return null;
    return j.length ? Q('jets_q', 'jets_some_a', { jets: countSlot('jets_failed', j.length, 'jet', ['len']), jet_names: catSlot('jets_failed', j, list(j)) }, ['jets_failed'], 'some')
      : Q('jets_q', 'jets_none_a', { jets: countSlot('jets_failed', 0, 'jet', ['len']) }, ['jets_failed'], 'none'); }),
  fam('breakout_ready', 'docking', ['D'], (r) => { const v = val(factsOf(r), 'breakout_available'); if (typeof v !== 'boolean') return null;
    return Q('breakout_q', v ? 'breakout_yes_a' : 'breakout_no_a', {}, ['breakout_available'], v ? 'yes' : 'no', { checks: [yes('breakout_available', v)] }); }),
  fam('speed_limit', 'docking', ['D'], (r) => { const F = factsOf(r), v = val(F, 'speed_limit_cms'); return v > 0 ? Q('dlimit_q', 'dlimit_a', { limit: fmtSlot('speed_limit_cms', F.speed_limit_cms, { unit: 'cm/s' }) }, ['speed_limit_cms'], roundNice(v).value) : null; }),
  // zoom (Z; every fact visual). place.*, geo.* and image.* are written by the build (vlm/gen/build/zoomgeo.js, imagefacts.js)
  fam('zoom_country', 'zoom', ['Z'], (r) => { const c = val(factsOf(r), 'place.country'); return typeof c === 'string' ? Q('country_q', 'country_a', { country: entSlot('place.country', c) }, ['place.country'], c) : null; }),
  fam('zoom_region', 'zoom', ['Z'], (r) => { const F = factsOf(r), a = val(F, 'place.admin1'); return typeof a === 'string' && a !== val(F, 'place.country') ? Q('admin_q', 'admin_a', { region_name: entSlot('place.admin1', a) }, ['place.admin1'], a) : null; }),
  fam('zoom_nearest_place', 'zoom', ['Z'], (r, rng) => { const F = factsOf(r), p = val(F, 'place.nearest'); if (!p || !p.name || !(p.km > 0) || !p.compass) return null;
    return Q('nplace_q', 'nplace_a', { place: entSlot('place.nearest', p.name, ['field', 'name']), place_km: { ...fmtSlot('place.nearest', { v: p.km, unit: 'km' }, { system: sys(rng) }), derive: ['field', 'km'] },
      compass: compassSlot('place.nearest', p.compass, ['field', 'compass']) }, ['place.nearest'], p.compass); }),
  fam('zoom_range_bin', 'zoom', ['Z'], (r, rng) => { const F = factsOf(r), km = val(F, 'view.range_km'); if (!(km > 0)) return null; const b = binOf(km, [20, 100, 400, 1500]);
    if (rng() < 0.5) return Q('zrange_q', 'zrange_a', { range: fmtSlot('view.range_km', F['view.range_km'], { system: sys(rng) }) }, ['view.range_km'], b);
    return Q('zrange_q', 'zrange_bin_a', { range_bin: binSlot('view.range_km', F['view.range_km'], km, [20, 100, 400, 1500], 'km').s }, ['view.range_km'], b); }),
  fam('zoom_mountains_flat', 'zoom', ['Z'], (r) => { const t = val(factsOf(r), 'zoom.tags'), k = t && t.find((x) => W.relief[x]); return k ? Q('relief_q', 'relief_a', { relief: catSlot('zoom.tags', k, W.relief[k], ['relief']) }, ['zoom.tags'], k) : null; }),
  fam('zoom_sea_fraction', 'zoom', ['Z'], (r) => { const F = factsOf(r), s = val(F, 'geo.sea_frac'); if (s === null) return null;
    if (s < 0.02) return Q('sea_q', 'sea_none_a', {}, ['geo.sea_frac'], 'none', { checks: [yes('geo.sea_frac', true, ['lt', 0.02])] });
    if (s > 0.98) return Q('sea_q', 'sea_all_a', {}, ['geo.sea_frac'], 'all', { checks: [yes('geo.sea_frac', true, ['gt', 0.98])] });
    return Q('sea_q', 'sea_part_a', { sea: fmtSlot('geo.sea_frac', F['geo.sea_frac']) }, ['geo.sea_frac'], binOf(s, [0.25, 0.5, 0.75])); }),
  fam('zoom_coast_side', 'zoom', ['Z'], (r) => { const s = val(factsOf(r), 'geo.coast_side'); return W.coast[s] ? Q('coast_q', 'coast_a', { coast_side: catSlot('geo.coast_side', s, W.coast[s]) }, ['geo.coast_side'], s) : null; }),
  fam('zoom_feature_where', 'zoom', ['Z'], (r, rng) => { const v = val(factsOf(r), 'place.in_view'); if (!Array.isArray(v)) return null; const idx = v.map((x, i) => i).filter((i) => v[i].name && v[i].region && v.filter((y) => y.name === v[i].name).length === 1);
    if (!idx.length) return null; const i = idx[Math.floor(rng() * idx.length)], name = entSlot('place.in_view', v[i].name, ['idx', i, 'name']);
    return Q('feature_q', 'feature_a', { feature: name, region: regionSlot('place.in_view', v[i].region, ['idx', i, 'region']) }, ['place.in_view'], v[i].region, { q: { feature: name } }); }),
  fam('zoom_day_night', 'zoom', ['Z'], (r) => { const c = val(factsOf(r), 'sun.class'); return W.daylight[c] ? Q('daynight_q', 'daynight_a', { daylight: catSlot('sun.class', c, W.daylight[c]) }, ['sun.class'], c) : null; }),
  fam('zoom_colours', 'zoom', ['Z'], (r) => { const F = factsOf(r), a = val(F, 'image.palette_0'), b = val(F, 'image.palette_1'); if (!a || !b || a === b) return null;
    return Q('colours_q', 'colours_a', { colour_a: catSlot('image.palette_0', a, a), colour_b: catSlot('image.palette_1', b, b) }, ['image.palette_0', 'image.palette_1'], a); }),
  fam('zoom_detail_gsd', 'zoom', ['Z'], (r) => { const F = factsOf(r), g = val(F, 'view.gsd_m'); return g > 0 ? Q('gsd_q', 'gsd_a', { gsd: fmtSlot('view.gsd_m', F['view.gsd_m']) }, ['view.gsd_m'], binOf(g, [20, 100, 1000])) : null; }),
  // safety: the verdict, reasons and action are the monitor's (the Context line); the outcome and cause come from safety.*
  fam('is_safe', 'safety', FLIGHT, (r, rng, ctx) => { const m = mon(ctx); if (!r.safety || !m) return null;
    return m.reasons.length ? Q('is_safe_q', 'is_safe_a', { verdict: verdictSlot(m), reason: reasonSlot(m) }, ['safety.verdict', 'safety.reasons'], m.verdict, { monitor: true })
      : Q('is_safe_q', 'is_safe_noreason_a', { verdict: verdictSlot(m) }, ['safety.verdict', 'safety.reasons'], m.verdict, { monitor: true }); }),
  fam('why', 'safety', FLIGHT, (r, rng, ctx) => { const m = mon(ctx); if (!r.safety || !m) return null;
    return m.reasons.length ? Q('why_q', 'why_a', { reason: reasonSlot(m) }, ['safety.reasons'], m.reasons[0], { monitor: true }) : Q('why_q', 'why_none_a', { verdict: verdictSlot(m) }, ['safety.reasons', 'safety.verdict'], 'none', { monitor: true }); }),
  fam('what_to_do', 'safety', FLIGHT, (r, rng, ctx) => { const m = mon(ctx); if (!r.safety || !m) return null;
    return Q('what_to_do_q', m.action === 'NONE_SAFE' ? 'what_to_do_none_a' : 'what_to_do_a', { action: actionSlot(m) }, ['safety.best_action'], m.action, { monitor: true }); }),
  // §5.7 target rule (controller ruling): S/A read the outcome off the monitor's p_ref and say it is the monitor's; L/D state
  // the safety block's outcome only when it agrees with the monitor's verdict; a cause is named only when it is one of the
  // monitor's reasons (and a hazard only when its kind is in the frame)
  fam('if_nothing_changes', 'safety', FLIGHT, (r, rng, ctx) => { const p = predictionOf(r, mon(ctx)); return p ? Q('what_if_q', p.aid, { outcome: p.slot }, [p.slot.fact_id], p.slot.value, { monitor: true }) : null; }),
  fam('most_dangerous', 'safety', FLIGHT, (r, rng, ctx) => { const c = r.safety && r.safety.cause, m = mon(ctx), k = val(factsOf(r), 'kinds_in_frame');
    if (!c || !W.cause[c] || !m || !(CAUSE_REASONS[c] || []).some((x) => m.reasons.includes(x))) return null;
    // a hazard only when it is the one kind in the frame (with two kinds in view, which one the danger is is not visible)
    if (KIND_A[c] && !(Array.isArray(k) && k.length === 1 && k[0] === c)) return null;
    return Q('danger_q', 'danger_a', { threat: catSlot('safety.cause', c, W.cause[c]) }, ['safety.cause'], c, { monitor: true }); }),
  fam('monitor_ttc', 'safety', SA, (r, rng, ctx) => { const m = mon(ctx), t = m && TTC_RANGE[m.ttc_bin]; return t ? Q('mttc_q', 'mttc_a', { ttc: monRange('safety.ttc_s', t, 's') }, ['safety.ttc_s'], m.ttc_bin, { monitor: true }) : null; }),
  fam('monitor_clearance', 'safety', SA, (r, rng, ctx) => { const m = mon(ctx), c = m && CLR_RANGE_U[m.clr_bin];
    return c ? Q('mclr_q', 'mclr_a', { clr_bin: monRange('safety.clearance', [c[0] * 19, c[1] * 19], 'm') }, ['safety.clearance'], m.clr_bin, { monitor: true }) : null; }),
];
// the prediction a row may state next to monitor m: {aid, slot} or null (S/A from p_ref, L/D the agreeing safety outcome)
export function predictionOf(r, m) {
  if (!r.safety || !m) return null;
  if (r.family === 'S' || r.family === 'A') { const o = prefOutcome(m.p_ref); return o ? { aid: 'what_if_mon_a', pred: 'pred_mon', slot: outcomeSlot(o, 'monitor') } : null; }
  const o = r.safety.action_outcome && r.safety.action_outcome.CONTINUE;
  return o && W.outcome[o] && outcomeAgrees(o, m.verdict) ? { aid: 'what_if_a', pred: 'what_if_a', slot: outcomeSlot(o, 'gt') } : null;
}
export function stationSlot(F) {
  const b = val(F, 'station_distance_bin'), m = typeof b === 'string' ? /^(<|>)?(\d+)(?:-(\d+))? m$/.exec(b) : null; if (!m) return null;
  const [lo, hi] = m[1] === '<' ? [0, +m[2]] : m[1] === '>' ? [+m[2], Infinity] : [+m[2], +m[3]];
  return { ...rangeSlot('station_distance_bin', F.station_distance_bin, lo, hi, 'm'), value: b };
}
// rejection sampling per question family (spec §5.2): each answer keeps at most the count that holds its share at or under
// maxShare (a seeded random subset of each answer), and a family with a single answer is dropped as degenerate
export function balanceAnswers(items, { maxShare = 0.5, rng = mulberry32(1) } = {}) {
  const byQ = new Map(); for (const it of items) (byQ.get(it.family_q) || byQ.set(it.family_q, []).get(it.family_q)).push(it);
  const out = [];
  for (const group of byQ.values()) {
    const byA = new Map(); for (const it of group) (byA.get(it.answerKey) || byA.set(it.answerKey, []).get(it.answerKey)).push(it);
    if (byA.size < 2) continue;
    const n = [...byA.values()].map((l) => l.length), kept = (M) => n.reduce((a, x) => a + Math.min(x, M), 0);
    let M = Math.max(...n); while (M > 1 && M / kept(M) > maxShare + 1e-12) M--;
    for (const l of byA.values()) { const x = [...l]; for (let i = x.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [x[i], x[j]] = [x[j], x[i]]; } out.push(...x.slice(0, M)); }
  }
  return out;
}

// every context-class fact an item cites is supplied by the row's Context: the one §5.7 rule, contextSupplies (telemetry
// ids; a monitor supplies SAFETY_TEXT_IDS)
export function exportable(factIds, rec, context) {
  const F = factsOf(rec);
  return contextSupplies(factIds.filter((id) => id.startsWith('safety.') || (F[id] && F[id].obs !== 'visual')), context);
}
// LRV-style negatives (spec §5.5): an absent hazard (checked against kinds_in_frame), a wrong number (|stated/true - 1| >= 0.3
// and outside the rounding interval) or a wrong place (not in view, not the nearest place, not the country; for A a wrong
// route). The premise must be stated in the prompt and false; the answer opens with "No", states the true value and never
// repeats the premise (an absent kind appears only inside its own negation, "no airliners").
export const NEGATIVE_TYPES = Object.freeze({ S: ['absent', 'number'], A: ['absent', 'number', 'place'], L: ['number'], D: ['number'], Z: ['place', 'number'] });
const FAMOUS = ['Tokyo', 'New York', 'Paris', 'London', 'Cairo', 'Sydney', 'Rio de Janeiro', 'Moscow', 'Beijing', 'Mumbai', 'Mexico City', 'Istanbul', 'Buenos Aires', 'Singapore', 'Rome', 'Madrid',
  'Berlin', 'Toronto', 'Los Angeles', 'Italy', 'Japan', 'Brazil', 'Canada', 'Australia', 'Egypt', 'India', 'China', 'Kenya', 'Peru', 'Norway', 'Iceland', 'Mexico', 'Argentina', 'Chile', 'South Africa',
  'Indonesia', 'Greenland', 'Mongolia', 'Spain', 'France'];
const pick = (xs, rng) => xs[Math.floor(rng() * xs.length)];
const lowerSet = (xs) => new Set(xs.filter(Boolean).map((x) => String(x).toLowerCase()));
function wrongNumber(F, id, rng, opts) {
  const f = F[id]; if (!f || typeof f.v !== 'number' || !(Math.abs(f.v) > 1e-6)) return null;
  const truth = fmtSlot(id, f, opts), shownTrue = truth.shown, factor = rng() < 0.5 ? 0.5 : 2, stated = roundNice(shownTrue * factor).value, iv = roundNice(shownTrue);
  if (!(stated > 0) || Math.abs(stated / shownTrue - 1) < 0.3 || (stated >= iv.lo && stated <= iv.hi)) return null;
  return { truth, stated, statedText: `about ${numText(stated)} ${unitText(truth.shownUnit, stated)}`, isPremise: (c) => c.type === 'number' && c.value === stated };
}
const V = {
  absent: (F, rng, { forceKind }) => { const k = val(F, 'kinds_in_frame'), n = val(F, 'hazards.count_in_frame'); if (!Array.isArray(k) || n === null) return null;
    const pool = Object.keys(KIND_A).filter((x) => !k.includes(x)), kind = forceKind ?? pick(pool, rng); if (!kind || k.includes(kind)) return null;
    return { variant: 'absent', qid: 'neg_absent_q', aid: 'neg_absent_a', q: { kind: KIND_N[kind], side: rng() < 0.5 ? 'left' : 'right' }, a: { kind_none: countSlot('kinds_in_frame', 0, kind, ['count_of', kind]), hazards: countSlot('hazards.count_in_frame', n, 'hazard') },
      fact_ids: ['kinds_in_frame', 'hazards.count_in_frame'], premise: { fact_id: 'kinds_in_frame', stated: kind },
      isPremise: (cl) => (cl.type === 'kind' || cl.type === 'cause') && cl.value === kind && !cl.negated }; },
  count: (F, rng) => { const k = val(F, 'kinds_in_frame'), t = val(F, 'hazards.count_in_frame'); if (!Array.isArray(k) || t === null) return null;
    const noun = k.length === 1 ? k[0] : 'hazard', c = [1, 2, 3, 4, 5, 6].filter((s) => s !== t && (t === 0 || Math.abs(s / t - 1) >= 0.3)), s = pick(c, rng);
    return { variant: 'count', qid: 'neg_count_q', aid: 'neg_count_a', q: { count_stated: countPhrase(s, noun) }, a: { count_true: countSlot('hazards.count_in_frame', t, noun) }, fact_ids: ['hazards.count_in_frame', 'kinds_in_frame'],
      premise: { fact_id: 'hazards.count_in_frame', stated: s }, isPremise: (cl) => cl.type === 'count' && cl.value === s }; },
  dist: (F, rng) => { const w = wrongNumber(F, 'hazard.0.dist_u', rng, { system: sys(rng) }); if (!w) return null;
    return { variant: 'dist', qid: 'neg_dist_q', aid: 'neg_dist_a', q: { dist_stated: w.statedText }, a: { dist_true: w.truth }, fact_ids: ['hazard.0.dist_u'], premise: { fact_id: 'hazard.0.dist_u', stated: w.statedText }, isPremise: w.isPremise }; },
  speed: (F, rng, { family }) => {
    const [id, what, opts] = family === 'L' ? ['ias_kt', 'the indicated airspeed', { system: 'aviation' }] : family === 'D' ? ['closing_cms', 'the closing rate', { unit: 'cm/s' }] : ['ship.speed_m_s', "the ship's speed", { system: sys(rng) }];
    if (!(val(F, id) > 0.05)) return null; const w = wrongNumber(F, id, rng, opts); if (!w) return null;
    return { variant: 'speed', qid: 'neg_speed_q', aid: 'neg_speed_a', q: { what, speed_stated: w.statedText }, a: { what: catSlot(id, F[id].v, what), speed_true: w.truth }, fact_ids: [id], premise: { fact_id: id, stated: w.statedText }, isPremise: w.isPremise }; },
  papi: (F, rng) => { const w = val(F, 'papi_whites_cam'); if (w === null) return null; const s = pick([0, 1, 2, 3, 4].filter((x) => Math.abs(x - w) >= 2), rng);
    return { variant: 'papi', qid: 'neg_papi_q', aid: 'neg_papi_a', q: { papi_stated: countPhrase(s, 'white') }, a: { papi_white: countSlot('papi_whites_cam', w, 'white'), papi_red: countSlot('papi_whites_cam', 4 - w, 'red', ['papi_red']) },
      fact_ids: ['papi_whites_cam'], premise: { fact_id: 'papi_whites_cam', stated: s }, isPremise: (cl) => cl.type === 'count' && cl.noun === 'white' && cl.value === s }; },
  station: (F, rng) => { const s = stationSlot(F); if (!s) return null; const mids = [1, 10, 50, 200, 800], far = mids.filter((m) => m < s.lo / 1.3 || m > (s.hi === Infinity ? Infinity : s.hi * 1.3));
    const m = pick(far, rng); if (m === undefined) return null;
    return { variant: 'station', qid: 'neg_station_q', aid: 'neg_station_a', q: { station_stated: `about ${m} m` }, a: { station_dist: s }, fact_ids: ['station_distance_bin'], premise: { fact_id: 'station_distance_bin', stated: `about ${m} m` }, premiseFacts: ['station_distance_bin', 'rho_m'],
      isPremise: (cl) => cl.type === 'number' && cl.value === m }; },
  range: (F, rng) => { const w = wrongNumber(F, 'view.range_km', rng, { system: sys(rng) }); if (!w) return null;
    return { variant: 'range', qid: 'neg_range_q', aid: 'neg_range_a', q: { range_stated: w.statedText }, a: { range_true: w.truth }, fact_ids: ['view.range_km'], premise: { fact_id: 'view.range_km', stated: w.statedText }, isPremise: w.isPremise }; },
  place: (F, rng, { gaz }) => { const p = val(F, 'place.nearest'), c = val(F, 'place.country'), names = lowerSet([p && p.name, c, ...(val(F, 'place.in_view') || []).map((x) => x.name), val(F, 'place.admin1')]);
    const truth = p && p.name ? entSlot('place.nearest', p.name, ['field', 'name']) : typeof c === 'string' ? entSlot('place.country', c) : null, wrong = pick(FAMOUS.filter((n) => gaz.has(n) && !names.has(n.toLowerCase())), rng);
    if (!truth || !wrong) return null;
    return { variant: 'place', qid: 'neg_place_q', aid: 'neg_place_a', q: { wrong }, a: { place: truth }, fact_ids: [truth.fact_id], premise: { fact_id: truth.fact_id, stated: wrong }, isPremise: (cl) => cl.type === 'entity' && cl.value === gaz.canonical(wrong) }; },
  route: (F, rng, { gaz }) => { const k = val(F, 'route'); if (!ROUTE_NAMES[k]) return null; const wrong = pick(Object.values(ROUTE_NAMES).filter((n) => n !== ROUTE_NAMES[k] && gaz.has(n)), rng); if (!wrong) return null;
    return { variant: 'route', qid: 'neg_route_q', aid: 'neg_route_a', q: { wrong_route: wrong }, a: { route: entSlot('route', ROUTE_NAMES[k], ['map', ROUTE_NAMES]) }, fact_ids: ['route'], premise: { fact_id: 'route', stated: wrong },
      isPremise: (cl) => cl.type === 'entity' && cl.value === gaz.canonical(wrong) }; } };
const VARIANTS = { absent: { S: ['absent'], A: ['absent'] }, number: { S: ['count', 'dist', 'speed'], A: ['count', 'dist', 'speed'], L: ['papi', 'speed'], D: ['station', 'speed'], Z: ['range'] }, place: { A: ['route'], Z: ['place'] } };
const textOf = (slots) => Object.fromEntries(Object.entries(slots).map(([k, s]) => [k, typeof s === 'string' ? s : s.text]));
export function makeNegative(r, type, rng, gaz, { forceKind = null, bank = defaultBank(), split = 'train', context = null } = {}) {
  const F = factsOf(r), names = ((VARIANTS[type] || {})[r.family] || []).map((n) => [n, rng()]).sort((a, b) => a[1] - b[1]).map(([n]) => n);
  const built = names.map((n) => V[n](F, rng, { forceKind, gaz, family: r.family })).filter((v) => v && bank[v.qid] && bank[v.aid]);
  const v = built.find((b) => exportable(b.fact_ids, r, context)) || built[0]; if (!v) return null;
  const qf = pickForm(bank, v.qid, rng, split), af = pickForm(bank, v.aid, rng, split), prompt = polish(render(qf.form, textOf(v.q))), answer = polish(render(af.form, textOf(v.a)));
  const it = { task: 'negative', prompt, answer, slots: Object.values(v.a).filter((s) => typeof s === 'object'), fact_ids: v.fact_ids, template_id: `${v.qid}+${v.aid}`, paraphrase_id: `${qf.paraphrase_id}+${af.paraphrase_id}`,
    generator: 'template', verified: false, false_premise: v.premise, family_q: `neg_${v.variant}`, answerKey: 'no' };
  const errors = verifyTemplateItem(it, r, { gaz, context, premise: v.isPremise }).errors;
  if (!/^no\b/i.test(answer)) errors.push('the answer does not negate the premise');
  if (v.isPremise) {
    const stated = parseClaims(prompt, gaz).filter(v.isPremise);
    if (!stated.length) errors.push('the premise is not stated in the prompt'); else if (stated.some((c) => checkClaim(c, r, { context, factIds: v.premiseFacts || [v.premise.fact_id] }))) errors.push('the premise is true');
    if (parseClaims(answer, gaz).some(v.isPremise)) errors.push('the answer repeats the premise');
  }
  it.verified = errors.length === 0;
  return Object.defineProperty(it, 'errors', { value: errors, enumerable: false });
}
