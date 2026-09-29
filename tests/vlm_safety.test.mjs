import { test } from 'node:test';
import assert from 'node:assert/strict';
import { corridorSafety, landingSafety, dockingSafety, eyeView, CORRIDOR_ACTIONS, C_NEAR_DEFAULT } from '../vlm/gen/safety.js';
import { FAMILY_ACTIONS, OUTCOMES } from '../vlm/gen/schema.js';

const seenCorridor = [], seenL = [], seenD = [];
const br = (over = {}) => Object.fromEntries(CORRIDOR_ACTIONS.map((a) => [a, { k: 4, crashes: 0, minClr: 10, medClr: 10, causes: [], ...(over[a] || {}) }]));
const now0 = { overstressed: false, stalled: false, pullUp: false, turbSevere: false, edge: false, stormCell: false, closingFast: false, clrSource: 'hazard' };
const C = (branches, now = {}, x = {}) => { const s = corridorSafety({ branches, now: { ...now0, ...now }, ttc_s: null, cpa_m: null, p_pilot: null, pulse: 8, ...x }, { cNear: C_NEAR_DEFAULT }); seenCorridor.push(s); return s; };

test('corridor SAFE: all clear, CONTINUE first, all actions safe', () => {
  const s = C(br());
  assert.equal(s.verdict, 'SAFE'); assert.equal(s.severity, 0); assert.equal(s.escapable, null); assert.equal(s.best_action, 'CONTINUE');
  assert.deepEqual(s.safe_actions, CORRIDOR_ACTIONS); assert.equal(s.action_outcome.CONTINUE, 'clear'); assert.equal(s.ref_policy, 'HANDS_OFF');
  assert.deepEqual(s.macro, { pulse_steps: 8, then: 'hands_off' });
});
test('corridor CAUTION: p_ref in (0, 0.5), or median clearance <= c_near, or severe turbulence; EDGE never raises', () => {
  assert.equal(C(br({ CONTINUE: { crashes: 1, causes: ['rock'] } })).verdict, 'CAUTION');
  const near = C(br({ CONTINUE: { medClr: 2.5 } })); assert.equal(near.verdict, 'CAUTION'); assert.deepEqual(near.reasons, ['HAZARD_AHEAD']);
  assert.equal(C(br({ CONTINUE: { medClr: 2.51 } })).verdict, 'SAFE');
  assert.equal(C(br(), { turbSevere: true }).verdict, 'CAUTION');
  const edge = C(br(), { edge: true }); assert.equal(edge.verdict, 'SAFE'); assert.deepEqual(edge.reasons, ['CORRIDOR_EDGE']);
  assert.deepEqual(C(br(), { closingFast: true, stormCell: true }).reasons, []);
});
test('corridor UNSAFE severities, escapable, ties and NONE_SAFE', () => {
  const two = C(br({ CONTINUE: { crashes: 2, causes: ['rock', 'rock'] }, CLIMB: { minClr: 3 }, DESCEND: { minClr: 12 } }));
  assert.equal(two.verdict, 'UNSAFE'); assert.equal(two.severity, 2); assert.equal(two.escapable, true); assert.equal(two.cause, 'rock');
  assert.ok(!two.safe_actions.includes('CONTINUE'));
  assert.equal(two.best_action, 'DESCEND', 'the larger min-over-draws clearance wins among the tied safe actions');
  const tie = C(br({ CONTINUE: { crashes: 4, causes: ['terrain', 'terrain', 'terrain', 'terrain'] } }));
  assert.equal(tie.best_action, 'CLIMB', 'equal clearances fall back to CLIMB, TURN_LEFT, TURN_RIGHT, SLOW_DOWN, SPEED_UP, DESCEND');
  assert.deepEqual(tie.reasons, ['TERRAIN_CLOSE']);
  const d3 = C(br(Object.fromEntries(CORRIDOR_ACTIONS.map((a) => [a, { crashes: a === 'CONTINUE' ? 4 : 1 }]))));
  assert.equal(d3.severity, 3); assert.equal(d3.escapable, true); assert.equal(d3.p_best, 0.25); assert.equal(d3.regret_ref, 0.75);
  const all = C(br(Object.fromEntries(CORRIDOR_ACTIONS.map((a) => [a, { crashes: 2 }]))));
  assert.equal(all.severity, 4); assert.equal(all.best_action, 'NONE_SAFE'); assert.deepEqual(all.safe_actions, []); assert.equal(all.escapable, false);
  assert.equal(all.action_outcome.CONTINUE, 'crash_possible');
});
test('corridor now-criteria make UNSAFE at p_ref 0; c_near never moves UNSAFE', () => {
  for (const k of ['overstressed', 'stalled', 'pullUp']) { const s = C(br(), { [k]: true }); assert.equal(s.verdict, 'UNSAFE'); assert.equal(s.severity, 2); }
  const u = { CONTINUE: { crashes: 2, causes: ['rock', 'comet'] } };
  for (const cNear of [1.5, 4]) { const s = corridorSafety({ branches: br(u), now: now0, pulse: 15 }, { cNear }); seenCorridor.push(s); assert.equal(s.verdict, 'UNSAFE'); }
});
test('safety_eye equals safety for S and A', () => {
  const inp = { branches: br({ CONTINUE: { crashes: 1, causes: ['birds'] } }), now: now0, pulse: 15 };
  const a = eyeView('A', inp), b = corridorSafety(inp); seenCorridor.push(a, b);
  assert.deepEqual(a, b);
});

const gate = { lateral: true, vertical: true, loc: true, gs: true, speed: true, vs: true, gear: true };
const lnow = (o = {}) => ({ airborne: true, wow: false, vert: 'GS', lat: 'LOC', hRAft: 800, thrNm: 2.5, locDots: 0, gsDots: 0, crossing500: false, gate, speedErrKt: 0, crossKt: 0, tailKt: 0, turbSevere: false, mainGearZ: 0, halfWidthM: 30, stopNeedM: 0, runwayLeftM: 4000, ...o });
const L = (now, outcome = { CONTINUE: 'landed', GO_AROUND: 'go_around' }, eye = false) => { const s = landingSafety({ now: lnow(now), outcome }, { eye }); seenL.push(s); return s; };

test('landing truth table', () => {
  assert.equal(L({}).verdict, 'SAFE');
  for (const bad of ['crash', 'excursion', 'overrun', 'short', 'tailstrike', 'hard']) assert.equal(L({}, { CONTINUE: bad, GO_AROUND: 'go_around' }).verdict, 'UNSAFE');
  const ga = L({}, { CONTINUE: 'go_around', GO_AROUND: 'go_around' }); assert.equal(ga.verdict, 'CAUTION'); assert.ok(ga.reasons.includes('UNSTABLE_APPROACH'));
  const cross = L({ hRAft: 480, crossing500: true, gate: { ...gate, gs: false } }); assert.equal(cross.verdict, 'UNSAFE'); assert.equal(cross.best_action, 'GO_AROUND');
  const dev = L({ hRAft: 200, locDots: 1.6 }); assert.equal(dev.verdict, 'UNSAFE'); assert.ok(dev.reasons.includes('LOCALIZER_DEVIATION'));
  assert.equal(L({ hRAft: 200, locDots: 1.6 }, { CONTINUE: 'landed', GO_AROUND: 'crash' }).best_action, 'NONE_SAFE');
  const c = L({ hRAft: 700, gate: { ...gate, speed: false } }); assert.equal(c.verdict, 'CAUTION'); assert.ok(c.reasons.includes('SPEED_OUT_OF_BAND'));
  assert.equal(L({ crossKt: 15 }).verdict, 'CAUTION'); assert.equal(L({ tailKt: 10 }).verdict, 'CAUTION');
  const band = L({ hRAft: 300, gsDots: 1.2 }); assert.equal(band.verdict, 'CAUTION'); assert.equal(band.best_action, 'DESCEND');
  assert.equal(L({ hRAft: 300, locDots: -1.2 }).best_action, 'TURN_RIGHT');
  assert.equal(L({ vert: 'FLARE', hRAft: 30, gsDots: 1.4, speedErrKt: 20 }).verdict, 'SAFE', 'FLARE frames never use dot or speed rules');
});
test('landing ground rules and severities', () => {
  const g = (o) => L({ airborne: false, wow: true, vert: 'ROLLOUT', lat: 'ROLLOUT', hRAft: 0, ...o }, { CONTINUE: 'landed', GO_AROUND: null });
  const edge = g({ mainGearZ: 23 }); assert.equal(edge.verdict, 'UNSAFE'); assert.equal(edge.best_action, 'TURN_LEFT'); assert.ok(!('GO_AROUND' in edge.action_risk));
  assert.equal(g({ mainGearZ: -23 }).best_action, 'TURN_RIGHT');
  assert.equal(g({ mainGearZ: 23, lat: 'EXIT' }).verdict, 'SAFE');
  const stop = g({ stopNeedM: 900, runwayLeftM: 800 }); assert.equal(stop.best_action, 'SLOW_DOWN'); assert.ok(stop.reasons.includes('CANNOT_STOP'));
  const only = g({}); const bad = landingSafety({ now: lnow({ airborne: false, wow: true, vert: 'ROLLOUT', hRAft: 0 }), outcome: { CONTINUE: 'overrun', GO_AROUND: null } }); seenL.push(bad);
  assert.equal(only.verdict, 'SAFE'); assert.equal(bad.best_action, 'NONE_SAFE'); assert.equal(bad.severity, 4);
  assert.equal(L({ hRAft: 480, crossing500: true, gate: { ...gate, gear: false } }).severity, 2);
});
test('landing safety_eye removes TAILWIND, every GS-dot rule and LOC-dot rules beyond 1 NM', () => {
  assert.equal(L({ tailKt: 12 }, undefined, true).verdict, 'SAFE');
  assert.equal(L({ hRAft: 200, gsDots: 1.8 }, undefined, true).verdict, 'SAFE');
  assert.equal(L({ hRAft: 200, locDots: 1.8, thrNm: 1.2 }, undefined, true).verdict, 'SAFE');
  assert.equal(L({ hRAft: 200, locDots: 1.8, thrNm: 0.8 }, undefined, true).verdict, 'UNSAFE');
  assert.equal(L({ crossKt: 16 }, undefined, true).verdict, 'CAUTION', 'STRONG_CROSSWIND stays (the crab is visible)');
});

const dnow = (o = {}) => ({ phase: 'CORRIDOR', rho: 60, closing: 0.1, limit: 0.2, inCorridor: true, kosViolation: false, attDeg: 0.5, rateDps: 0.01, fuelFrac: 0.8, failedJets: 0, breakoutAvailable: true, holdPhase: false, outsideHoldBox: false, ttc_s: null, ...o });
const D = (now, outcome = { CONTINUE: 'capture', BREAKOUT: 'breakout', failReason: null }, eye = false) => { const s = dockingSafety({ now: dnow(now), outcome }, { eye }); seenD.push(s); return s; };
test('docking truth table', () => {
  assert.equal(D({}).verdict, 'SAFE');
  const f = D({}, { CONTINUE: 'fail', BREAKOUT: 'breakout', failReason: 'KOS_VIOLATION' }); assert.equal(f.verdict, 'UNSAFE'); assert.equal(f.best_action, 'BREAKOUT'); assert.equal(f.severity, 2);
  assert.equal(D({}, { CONTINUE: 'fail', BREAKOUT: 'fail', failReason: null }).best_action, 'NONE_SAFE');
  assert.equal(D({}, { CONTINUE: 'fail', BREAKOUT: null, failReason: null }).severity, 4);
  assert.equal(D({ kosViolation: true }).verdict, 'UNSAFE');
  assert.equal(D({ rho: 15, closing: 0.31, limit: 0.2 }).verdict, 'UNSAFE');
  const fast = D({ rho: 15, closing: 0.25, limit: 0.2 }); assert.equal(fast.verdict, 'CAUTION'); assert.equal(fast.best_action, 'SLOW_DOWN');
  assert.equal(D({}, { CONTINUE: 'breakout', BREAKOUT: 'breakout', failReason: null }).verdict, 'CAUTION');
  for (const o of [{ attDeg: 2.1 }, { rateDps: 0.11 }, { fuelFrac: 0.39 }, { failedJets: 1 }, { breakoutAvailable: false }]) assert.equal(D(o).verdict, 'CAUTION');
  assert.equal(D({ phase: 'H2', holdPhase: true, outsideHoldBox: true }).best_action, 'HOLD_POSITION');
});
test('docking safety_eye removes fuel, jets, breakout availability and closing rules beyond 20 m', () => {
  for (const o of [{ fuelFrac: 0.3 }, { failedJets: 1 }, { breakoutAvailable: false }, { rho: 60, closing: 0.5, limit: 0.2 }]) assert.equal(D(o, undefined, true).verdict, 'SAFE');
  assert.equal(D({ rho: 5, closing: 0.31, limit: 0.2 }, undefined, true).verdict, 'UNSAFE', 'within the eye scope (EYE.closingRhoM may be lowered by Task 4)');
  assert.equal(D({ attDeg: 3 }, undefined, true).verdict, 'CAUTION');
});

test('safe_actions and best_action stay inside FAMILY_ACTIONS[family] (or NONE_SAFE), and action_outcome stays inside OUTCOMES[family], for every truth-table case', () => {
  assert.ok(seenCorridor.length >= 15, `expected many corridor cases, got ${seenCorridor.length}`);
  assert.ok(seenL.length >= 15, `expected many landing cases, got ${seenL.length}`);
  assert.ok(seenD.length >= 10, `expected many docking cases, got ${seenD.length}`);
  const check = (fam, outcomesEnum, list, label) => {
    for (const s of list) {
      for (const a of s.safe_actions) assert.ok(FAMILY_ACTIONS[fam].includes(a), `${label} safe_actions has ${a}, outside FAMILY_ACTIONS.${fam}`);
      assert.ok(s.best_action === 'NONE_SAFE' || FAMILY_ACTIONS[fam].includes(s.best_action), `${label} best_action ${s.best_action}, outside FAMILY_ACTIONS.${fam}`);
      for (const [k, o] of Object.entries(s.action_outcome)) assert.ok(outcomesEnum.includes(o), `${label} action_outcome.${k} = ${o}, outside OUTCOMES.${fam}`);
    }
  };
  check('S', OUTCOMES.S, seenCorridor, 'corridor');
  check('L', OUTCOMES.L, seenL, 'landing');
  check('D', OUTCOMES.D, seenD, 'docking');
});
