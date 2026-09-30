// words.js — the Narrator's short UI words (pure; Node-tested): the verdict pill's label from Pilot Eye's decoded heads,
// the card's context chip per family, and the template sentence in sentence case. The long forms (REASON_TEXT,
// ACTION_TEXT) belong to the Context line the model was trained on; the pill needs two or three words.
const SHORT = Object.freeze({
  HAZARD_AHEAD: 'hazard ahead', HAZARD_CLOSING_FAST: 'closing fast', TERRAIN_CLOSE: 'terrain', BUILDING_CLOSE: 'buildings', CORRIDOR_EDGE: 'corridor edge',
  STALL: 'stall', OVERSTRESS: 'overstress', SEVERE_TURBULENCE: 'turbulence', STORM_CELL: 'storm cell', PULL_UP: 'pull up',
  UNSTABLE_APPROACH: 'unstable approach', LOCALIZER_DEVIATION: 'off localizer', GLIDESLOPE_DEVIATION: 'off glideslope', SPEED_OUT_OF_BAND: 'speed',
  HIGH_SINK_RATE: 'sink rate', STRONG_CROSSWIND: 'crosswind', TAILWIND: 'tailwind', RUNWAY_EDGE: 'runway edge', CANNOT_STOP: 'runway short',
  KOS_VIOLATION: 'keep-out sphere', CLOSING_TOO_FAST: 'closing fast', LATERAL_MISALIGNMENT: 'misaligned', ATTITUDE_ERROR: 'attitude',
  JET_FAILURE: 'jet failure', LOW_FUEL: 'low propellant', NO_BREAKOUT_AVAILABLE: 'no breakout',
});
const HAZARD = { S: 'rock ahead', A: 'obstacle ahead' };
const ACT = Object.freeze({ CLIMB: 'climb', DESCEND: 'descend', TURN_LEFT: 'turn left', TURN_RIGHT: 'turn right', SPEED_UP: 'speed up', SLOW_DOWN: 'slow down',
  GO_AROUND: 'go around', HOLD_POSITION: 'hold position', BREAKOUT: 'break out' });
const VERDICT = { SAFE: 'Safe', CAUTION: 'Caution', UNSAFE: 'Unsafe' };
export const FAMILY_NAMES = Object.freeze({ S: 'Deep space', A: 'Atmosphere', Z: 'Earth', L: 'Final approach', D: 'Docking' });
const ROUTE_NAMES = { alps: 'Alps', china: 'China', newyork: 'New York', london: 'London', moscow: 'Moscow', dubai: 'Dubai', mega: 'Megacity' };

const reasonWord = (r, family) => (r === 'HAZARD_AHEAD' && HAZARD[family]) || SHORT[r] || null;
// "Safe" | "Caution · rock ahead" | "Unsafe · climb": UNSAFE leads with the action, CAUTION with the first reason
export function pillLabel(h, family) {
  if (!h || h.status !== 'ok' || !VERDICT[h.verdict]) return 'Watching…';
  if (h.verdict === 'SAFE') return 'Safe';
  const reason = (h.reasons || []).map((r) => reasonWord(r, family)).find(Boolean), act = ACT[h.action];
  const detail = h.verdict === 'UNSAFE' ? act || reason : reason || act;
  return detail ? `${VERDICT[h.verdict]} · ${detail}` : VERDICT[h.verdict];
}
export const fmtKm = (km) => (km >= 10 ? `${Math.round(km).toLocaleString('en-US')} km` : km >= 1 ? `${km.toFixed(1)} km` : `${Math.round(km * 1000)} m`);
// the Playbox skies (src/weather.js SKIES: clear, fair, cloudy, storm) from the weather severity and cloud cover
export function skyName(severity, cover) {
  if (!Number.isFinite(severity)) return null;
  return severity >= 0.9 ? 'Storm' : cover >= 1.5 ? 'Cloudy' : severity <= 0.3 ? 'Clear' : 'Fair';
}
// g: { rangeKm } for Z, { route, sky } for A
export function contextChip(family, g = {}) {
  if (family === 'Z') return Number.isFinite(g.rangeKm) ? `Earth · ${fmtKm(g.rangeKm)}` : 'Earth';
  if (family === 'A') {
    const route = ROUTE_NAMES[g.route] || (g.route ? g.route[0].toUpperCase() + g.route.slice(1) : FAMILY_NAMES.A);
    return g.sky ? `${route} · ${g.sky}` : route;
  }
  return FAMILY_NAMES[family] || 'Narrator';
}
export const humanTemplate = (s) => String(s || '').replace(/^(SAFE|CAUTION|UNSAFE)\b/, (v) => VERDICT[v]);
