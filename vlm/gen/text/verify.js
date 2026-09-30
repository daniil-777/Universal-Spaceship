// vlm/gen/text/verify.js — the §5.4 verifier. One rounding table (nice steps 1, 2, 2.5, 5) for every number; slot formatters
// return the interval their wording stands for; the claim parser reads numbers+units (bound to the nearest quantity noun),
// ranges, number words with a counted noun, o'clock (image; "your N o'clock" = pilot bearing), 3x3 regions, compass words,
// verdict/action/reason words, outcomes, category words (size, side, sky, light, colour, phase, PAPI path, terrain...),
// ungrounded descriptors (R9) and gazetteer entities. Failures are rejected, never repaired.
import { REASON_TEXT, ACTION_TEXT, TTC_RANGE, CLR_RANGE_U, normContext } from './context.js';
import { PALETTE_NAMES, TAG_WORDS, COAST_SIDES } from '../schema.js';

const NICE = [1, 2, 2.5, 5];
export function niceStep(x) { if (!(x > 0)) return 1; const e = 10 ** Math.floor(Math.log10(x)); let b = e; for (const n of NICE) if (n * e <= x) b = n * e; return b; }
export function roundNice(v) { const st = niceStep(Math.abs(v) / 3), r = +(Math.round(v / st) * st).toPrecision(12); return { value: r, lo: +(r - st / 2).toPrecision(12), hi: +(r + st / 2).toPrecision(12) }; }
export const UNITS = Object.freeze({ m: ['len', 1], km: ['len', 1000], ft: ['len', 0.3048], mi: ['len', 1609.344], NM: ['len', 1852], u: ['len', 19], 'm/s': ['spd', 1], 'km/h': ['spd', 1 / 3.6],
  mph: ['spd', 0.44704], kt: ['spd', 0.514444], 'u/s': ['spd', 19], 'cm/s': ['spd', 0.01], fpm: ['spd', 0.00508], s: ['time', 1], deg: ['ang', 1], 'deg/s': ['angrate', 1], g: ['g', 1], dots: ['dots', 1],
  frac: ['frac', 1], '%': ['frac', 0.01] });
// unitless fractions in the label modules (0..1), read and written as per cent
const FRAC_IDS = new Set(['fuel_frac', 'geo.sea_frac', 'sun.lit', 'air.in_cloud', 'cfg.spoilers', 'thrust', 'image.white_frac', 'image.black_frac']);
export const unitOf = (id, f) => f.unit || (FRAC_IDS.has(id) ? 'frac' : null);
const DISPLAY = { metric: { len: (m) => (m < 1000 ? 'm' : 'km'), spd: () => 'm/s', frac: () => '%' }, imperial: { len: (m) => (m < 1609 ? 'ft' : 'mi'), spd: () => 'mph', frac: () => '%' },
  aviation: { len: (m) => (m < 1852 ? 'ft' : 'NM'), spd: () => 'kt', frac: () => '%' } };
const SHOW = { s: ['second', 'seconds'], deg: ['degree', 'degrees'], 'deg/s': ['degree per second', 'degrees per second'], dots: ['dot', 'dots'] };
const WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty'];
export const numText = (x) => { const s = Number.isInteger(x) && Math.abs(x) >= 1000 ? x.toLocaleString('en-US') : String(x); return /e/.test(s) ? x.toFixed(12).replace(/0+$/, '') : s; };
export const unitText = (unit, x) => (SHOW[unit] ? SHOW[unit][x === 1 ? 0 : 1] : unit);
export function fmtSlot(factId, fact, { system = 'metric', abs = false, unit = null } = {}) {
  const fu = unitOf(factId, fact), u = UNITS[fu], v = fact.v;
  if (fu === 'count') return { text: v <= 20 ? WORDS[v] : String(v), fact_id: factId, value: v, unit: 'count', lo: v, hi: v, type: 'count' };
  if (fu === 'clock') return { text: `${v} o'clock`, fact_id: factId, value: v, unit: 'clock', lo: v, hi: v, type: 'clock' };
  if (!u || typeof v !== 'number') return { text: String(v), fact_id: factId, value: v, unit: fu, lo: null, hi: null, type: 'category' };
  const x = abs ? Math.abs(v) : v, si = x * u[1], sys = DISPLAY[system] || DISPLAY.metric, dn = unit || (sys[u[0]] ? sys[u[0]](Math.abs(si)) : fu), k = UNITS[dn][1], r = roundNice(si / k);
  return { text: `about ${numText(r.value)} ${unitText(dn, r.value)}`, fact_id: factId, value: v, unit: fu, lo: r.lo * k / u[1], hi: r.hi * k / u[1], type: 'number', dim: u[0], shown: r.value, shownUnit: dn, ...(abs ? { derive: ['abs'] } : {}) };
}
// the other slot kinds; derive (below) maps the fact value to what the slot states
const opt = (derive) => (derive ? { derive } : {});
export const NOUNS = Object.freeze({ rock: ['rock', 'rocks'], comet: ['comet', 'comets'], satellite: ['satellite', 'satellites'], airliner: ['airliner', 'airliners'], birds: ['flock of birds', 'flocks of birds'],
  hazard: ['hazard', 'hazards'], jet: ['jet', 'jets'], white: ['white', 'white'], red: ['red', 'red'] });
export const countPhrase = (n, noun) => (n === 0 ? `no ${NOUNS[noun][1]}` : `${n <= 20 ? WORDS[n] : n} ${NOUNS[noun][n === 1 ? 0 : 1]}`);
export const catSlot = (id, value, text, derive) => ({ text, fact_id: id, value, unit: null, lo: null, hi: null, type: 'category', ...opt(derive) });
export const entSlot = (id, name, derive) => ({ text: name, fact_id: id, value: name, unit: null, lo: null, hi: null, type: 'entity', ...opt(derive) });
export const countSlot = (id, n, noun, derive) => ({ text: countPhrase(n, noun), fact_id: id, value: n, unit: 'count', lo: n, hi: n, type: 'count', noun, ...opt(derive) });
export const clockSlot = (id, v) => ({ text: `${v} o'clock`, fact_id: id, value: v, unit: 'clock', lo: v, hi: v, type: 'clock' });
export const regionSlot = (id, region, derive) => ({ text: `the ${region}`, fact_id: id, value: region, unit: null, lo: null, hi: null, type: 'region', ...opt(derive) });
export const compassSlot = (id, c, derive) => ({ text: c, fact_id: id, value: c, unit: null, lo: null, hi: null, type: 'compass', ...opt(derive) });
export const rangeText = (lo, hi, unit) => (lo <= 0 ? `under ${numText(hi)} ${unitText(unit, hi)}` : hi === Infinity ? `over ${numText(lo)} ${unitText(unit, lo)}` : `between ${numText(lo)} and ${numText(hi)} ${unitText(unit, hi)}`);
// a bin [lo, hi] stated in `unit`; the fact value (in its own unit) must lie inside it
export const rangeSlot = (id, fact, lo, hi, unit, derive) => ({ text: rangeText(lo, hi, unit), fact_id: id, value: fact.v, unit, lo, hi, type: 'range', shownLo: lo <= 0 ? 0 : lo, shownHi: hi, ...opt(derive) });
// an outcome: the safety block's CONTINUE outcome (L/D) or the monitor's p_ref reading (S/A, src 'monitor')
export const outcomeSlot = (code, src) => ({ text: W.outcome[code], fact_id: src === 'monitor' ? 'safety.p_ref' : 'safety.action_outcome', value: code, unit: null, lo: null, hi: null, type: 'outcome', src, ...(src === 'monitor' ? {} : { derive: ['field', 'CONTINUE'] }) });
// S/A: the monitor's p_ref as an outcome (0 = clear, 1 = crash certain, otherwise crash possible)
export const prefOutcome = (p) => (p === null || p === undefined ? null : p <= 0 ? 'clear' : p >= 1 ? 'crash_certain' : 'crash_possible');
const DERIVE = { abs: (v) => Math.abs(v), has: (v, _, k) => v.includes(k), count_of: (v, _, k) => (v.includes(k) ? null : 0), len: (v) => v.length, papi_red: (v) => 4 - v,
  gt: (v, _, x) => v > x, lt: (v, _, x) => v < x, field: (v, _, k) => (v ? v[k] : null), idx: (v, _, i, k) => (v && v[i] ? v[i][k] : null), sign: (v) => Math.sign(v),
  all_true: (v) => Object.values(v).every(Boolean), false_keys: (v) => Object.keys(v).filter((k) => !v[k]).sort().join(','), sorted: (v) => [...v].sort().join(','),
  map: (v, _, m) => (Object.prototype.hasOwnProperty.call(m, v) ? m[v] : null), cmp_fact: (v, F, id) => (F[id] && typeof F[id].v === 'number' ? (v > F[id].v ? 'above' : 'below') : null),
  in_list: (v, _, list) => list.includes(v), relief: (v) => (v || []).find((t) => ['MOUNTAINS', 'HILLS', 'FLAT'].includes(t)) ?? null, in_band: (v, _, a, b) => v >= a && v <= b,
  cloud_code: (v) => (/\b(FEW|SCT|BKN|OVC|NSC|SKC|NONE)\b/.exec(String(v)) || [])[1] ?? null, count_type: (v, _, t) => v.filter((c) => c.type === t).length, min_dist_type: (v, _, t) => { const d = v.filter((c) => c.type === t).map((c) => c.dist_u); return d.length ? Math.min(...d) : null; } };
export const derive = (d, v, facts) => (!d ? v : v === null || v === undefined ? null : DERIVE[d[0]](v, facts, ...d.slice(1)));
// the words text states facts with (slot values and the parser's category words come from the same tables)
export const W = Object.freeze({
  size: { tiny: 'tiny', small: 'small', medium: 'medium-sized', large: 'large' },
  side: { left: 'on the left side of the image', centre: 'near the middle of the image', right: 'on the right side of the image' },
  sky: { clear: 'clear', fair: 'fair', cloudy: 'cloudy', storm: 'stormy' }, turb: { LIGHT: 'light', MOD: 'moderate', SEVERE: 'severe' },
  world: { mountains: 'a mountain range', pillars: 'a field of tall rock pillars', meshy: 'a range of steep pillar-shaped peaks', newyork: 'a city skyline', london: 'a city skyline', moscow: 'a city skyline', dubai: 'a city skyline', mega: 'a city skyline' },
  band: { slow: 'below the normal speed band', normal: 'inside the normal speed band', overspeed: 'above the never-exceed speed' },
  vert: { ALT: 'holding altitude before glideslope capture', GS: 'descending with the glideslope captured', FLARE: 'in the flare just above the runway', ROLLOUT: 'rolling out on the runway', STOP: 'stopped on the runway', GA: 'climbing away in a go-around' },
  phase: { TRANSFER: 'the transfer', 'H1 ACQ': 'the approach to the first hold', H1: 'the first hold', CORRIDOR: 'the corridor approach', H2: 'the second hold', FINAL: 'the final approach', BREAKOUT: 'a breakout', DEPART: 'the departure' },
  clouds: { FEW: 'a few clouds', SCT: 'scattered clouds', BKN: 'broken cloud', OVC: 'an overcast layer', NSC: 'no significant cloud', SKC: 'no significant cloud', NONE: 'no significant cloud' },
  time: { day: 'daylight', dusk: 'dusk light', night: 'darkness' }, vis: { cavok: 'clear visibility', haze: 'haze', fog: 'fog' },
  papi: ['well below the glide path', 'slightly below the glide path', 'on the glide path', 'slightly above the glide path', 'well above the glide path'],
  gate: { lateral: 'the lateral mode', vertical: 'the vertical mode', loc: 'the localizer gate', gs: 'the glideslope gate', speed: 'the speed', vs: 'the sink rate', gear: 'the gear' },
  relief: { MOUNTAINS: 'mountainous', HILLS: 'hilly', FLAT: 'mostly flat' }, coast: { left: 'left', right: 'right', top: 'upper', bottom: 'lower' },
  daylight: { day: 'in full daylight', golden: 'in low sunlight', twilight: 'in twilight', night: 'at night' }, bright: { dark: 'dark', medium: 'of medium brightness', bright: 'bright' },
  outcome: { clear: 'the flight stays clear of everything', crash_possible: 'a crash is possible', crash_certain: 'a crash is certain', landed: 'the aircraft touches down and completes the landing',
    go_around: 'the approach ends in a go-around', hard: 'the aircraft lands hard', excursion: 'the aircraft runs off the side of the runway', overrun: 'the aircraft overruns the end of the runway',
    short: 'the aircraft touches down short of the runway', tailstrike: 'the tail strikes the runway', crash: 'the aircraft crashes', capture: 'the spacecraft docks with the station',
    breakout: 'the approach ends in a breakout', fail: 'the docking fails' },
  cause: { rock: 'the rock', comet: 'the comet', satellite: 'the satellite', airliner: 'the airliner', birds: 'the flock of birds', terrain: 'the terrain', building: 'a building', roof: 'a rooftop',
    overstress: 'overloading the airframe', ground: 'the ground', runway: 'running off the runway', station: 'the station itself' } });

// facts plus the pseudo-facts the text layer cites: the zoom block (visual) and the safety block (context-class)
export function factsOf(rec) {
  const F = { ...(rec.facts || {}) };
  if (rec.zoom) { F['zoom.tags'] = { v: rec.zoom.tags ?? null, unit: null, obs: 'visual' }; F['zoom.range_bin'] = { v: rec.zoom.range_bin ?? null, unit: null, obs: 'visual' }; }
  if (rec.safety) for (const k of ['verdict', 'severity', 'reasons', 'best_action', 'safe_actions', 'action_outcome', 'cause', 'p_ref']) F[`safety.${k}`] = { v: rec.safety[k] ?? null, unit: null, obs: 'context' };
  return F;
}
export const ROUTE_NAMES = Object.freeze({ alps: 'Alps', china: 'China', newyork: 'New York', london: 'London', moscow: 'Moscow', dubai: 'Dubai', mega: 'Megacity' });
// the fixed part of the §5.4 gazetteer; the build adds every Natural Earth name it loads
// terms (acronyms, docking words, bodies, the airport) are known words, never place claims; route names are places
export const TERMS = Object.freeze(['Astro Pilot Spaceport', 'APX', 'IDSS', 'KOS', 'V-bar', 'Earth', 'Moon', 'Sun', 'PAPI', 'ILS', 'LOC', 'GS', 'IAS', 'METAR', 'AGL', 'UTC', 'TTC']);
export const BASE_NAMES = Object.freeze([...Object.values(ROUTE_NAMES), ...TERMS]);
const norm = (s) => String(s).normalize('NFC').toLowerCase().replace(/[^\p{L}\p{N} ]+/gu, '').replace(/\s+/g, ' ').trim();
const TERM_KEYS = new Set(TERMS.map(norm));
export function makeGazetteer(names) {
  const m = new Map(); let maxWords = 1;
  for (const n of names) { if (!n) continue; const k = norm(n); if (!k) continue; if (!m.has(k)) m.set(k, String(n)); maxWords = Math.max(maxWords, Math.min(8, k.split(' ').length)); }
  return { has: (s) => m.has(norm(s)), canonical: (s) => m.get(norm(s)), maxWords, names: [...m.values()] };
}
// words that never make an unknown entity mid-sentence
const STOP = new Set(['I', 'Context', 'Earth', 'Moon', 'Sun', 'V-bar']);
// words that may open a sentence (or follow ":" / ";"). Such a word is never read as a place even when a gazetteer holds it
// (Natural Earth has Split, Point, Orange, Wind...); any other capitalised word there must be a gazetteer name. Every word
// that opens a sentence in the bank or in a slot value is listed (tests/vlm_text.test.mjs checks this).
export const SENTENCE_WORDS = new Set([...PALETTE_NAMES, 'gray', 'tan', ...WORDS, ...('a about above according across actions advice aerial after again against ahead air airspeed ' +
  'all along already also although always am an and another any anything are around as at away back be because been before behind being below beside best between beyond both ' +
  'but by can cannot chase chase-camera check clear clearance close closest closing cloud clouds cloudy colour colours colour-wise compare components conditions confirm contact ' +
  'correct could count counting crosswind current currently day daylight darkness deployed describe descending did direct distance do docking does down driven during dusk each ' +
  'either else estimate even every everything expect explain failed failures fair falling far few flight flying fog following for from fuel further g-load gates gear give given ' +
  'going good ground had has have haze hazard hazards he head headroom headwind her here high highly hilly his holding how however i identify if imagery in inside instead into ' +
  'is it its judging just landing large laterally least left less level light like likely little look looking low lower many margin may maybe meanwhile measured medium ' +
  'medium-sized might minimum moderate monitor more moreover most mostly mountain mountainous much must my name near nearby nearest neither never next no none nor not note ' +
  'nothing now of off on once one only onto open or other otherwise our out outside over overall overhead overloading part papi per perhaps phase pitch point possibly problem ' +
  'quite rain range rate rather read reason receding recommended remaining resolution right rolling roughly route running sea seen severe several shadow she should since ' +
  'skies slightly small smooth so some something space speed speed-wise split spoilers stabilized stall still stopped stormy such sunlight tailwind taking tell terrain than that ' +
  'the their them then there these they this those though through thus time tiny to today together tonally too toward towards turbulence under unless unlike until up upon upper ' +
  'using vertically very visibility visible was water we well were what when where whether which while whose why wind with within without would yes yet you your city climbing ' +
  'textured transit scattered broken rock comet satellite airliner flock beige cyan navy bright coastline dark desert flat hills ice mountains night nose-down nose-up verdict ' +
  'beneath underneath amid among amongst besides despite via past front rear back apart plenty lots whole entire half certain various different same snow-capped snow sand trees fields ' +
  'forests roads rivers lakes streets houses buildings stars lights shadows land cloudless hazy foggy rainy windy sunny calm first second third finally lastly later earlier tonight ' +
  'furthermore hence therefore likewise similarly altogether additionally again visibly beyond straight shown viewed pictured captured taken heading approaching behind front').split(' ')]);
// lower-case words text uses that are not places even when a gazetteer holds them (the bank's vocabulary; a test keeps it whole)
const BANK_VOCAB = ('abort aborted about above accident according account accounts across act acting action actions actually administrative advice advise advises aerial after against ahead air aircraft airfield airframe ' +
  'airliner airliners airspeed aligned alignment all allowed alone along also altitude am among an and angle angle-of-attack answer any anywhere apparent appear appears applies approach approaches ' +
  'approaching are area areas around arrives as at attached attack attitude available avoid away axis band bar be because been before behind being belong belongs below beneath beside best between beyond ' +
  'big biggest birds black blowing blows blue both boundary bounds break breaking breakout bright brightness brings broken brown building bumps bumpy but by call calls camera can cannot capture captured ' +
  'captures carries carry catches cause cautionary ceiling cell cells cent centred certain change changes chase chase-camera check chiefly cites citing city class clear clearance climbing clock close ' +
  'closed closer closes closest closest-approach closing closing-rate cloud clouds cloudy coast coastline colour colour-wise colours come comes comet comets coming compare compared completes components ' +
  'concern conditions cone configured confirm contact contain contains continue continues controls converge converging correct correctly corridor could count counting country course cover covered covers ' +
  'crash crashes crew criteria cross crosses crosswind current currently danger dangerous dark darkness day daylight decreasing degree degrees departure descending describe desert detail detailed ' +
  'deviation direct direction directly distance distant do docking docks does doing dominant dominate done dot dots down downwards draws driven driving drops due during dusk each earth edge enclosed ' +
  'encloses end ends enough entirely enveloped error estimate estimates every everything exceed exceeds exist exists expect expects explain extended face factor failed failing fails failure failures fair ' +
  'falling falls far farther fast faster feature feel feeling feels feet few field fills final find finds fine first first-level flagged flags flare flat flies flight flock flocks flown fly flying fog ' +
  'followed following follows for forward fraction frame from fuel full further g-load gap gate gates gear gear-down get getting give given gives glide glideslope go go-around going good greatest green ' +
  'grey ground grows hand happen happens hard has have hazard hazards haze head headroom headwind height held here hidden high higher hills hilly hold holding holds how however ice identify if image ' +
  'imaged imagery in include includes increases indicate indicates indication inside instead intervene intervenes intervention into involve is it its itself jet jets judges judging just keep kilometres ' +
  'kind kinds knots laid land landing lands landscape large largely lateral laterally layer lead leading led left less level lie lies light lighting lights like likely limit limits line lined listed ' +
  'lists lit load loaded localizer located long longer look looking looks low lower lowered lying main mainly make makes many margin matters maximum may me mean meaning means measured measures medium ' +
  'medium-sized meet meets met metres middle miles minimum miss misses mixes mode moderate moment monitor moon more most mostly mountain mountainous mountains move moves moving much must name named ' +
  'namely names naming near nearby nearest needed neither never-exceed next night no none nor normal nose nose-down nose-up not note nothing now number object obscured obstacle obstacles occurred ocean ' +
  'of off on one only onto open opening opens option or orange oriented our out outcome outside over overall overcast overhead overloading overruns overspeed owing palette papi part partway pass passes ' +
  'passing passively past path peaks per percent permitted phase picture pillar-shaped pillars pilot pink pitch pitched pixel place placed places placing plan planet play plus point pointing points port ' +
  'portion pose poses position positions possible prediction predicts premise present preset primary problem propellant province pulling pulls purple puts putting quick quickly raised range rate rated ' +
  'rates rather rating reach reaches reaching reaction read reading reads really reason reasons recedes receding reckons recommended recommends red region relative relief remaining remains rendezvous ' +
  'report reports resolution resolves respect respond response result reveals ride right rise risk risky rock rocks rolling rooftop room rough roughly route routes rugged ruled ruling running runs runway ' +
  'safe safely safety sample satellite satellites satisfied say says scattered scene scenery scheme screen sea second seconds section see seen sees separate separates separation sequence serious service ' +
  'severe shadow share sharp shining ship short should show showing shown shows shrinking shrinks side sight sign significant sink sit sits situation size skies sky skyline slightly slow slower small ' +
  'smooth so sock something soon sort source space spacecraft spaceport span spans specific speed speed-wise split spoilers spot stabilization stabilized stabilized-approach stable stage stall stalling ' +
  'stalls stand stands state stated station stay stayed stays steep still stopped storm stormy streams stretch strikes strong suggest suggests sun sunlight sunlit surface surrounded surrounds tail ' +
  'tailwind take taken takes taking tall tanks teal tell telling terms terrain than that the their them then there these they things third this those though threat through thruster thrusters thunderstorm ' +
  'tight time time-to-contact tiny to together tonally tone tones too top total touches toward towards towering town tracking tracks trailing trails transfer travelling travels true tunnel turbulence ' +
  'turbulent twelve twilight two type types unchanged under undercarriage unless unmet unsafe until up upper upwards us using value verdict vertical vertically view viewed viewing viewpoint views ' +
  'visibility visible was water way we weather well wet what when where which while white whole whose why will wind windsock wing wings with within without working works world worried worry would wrong ' +
  'yellow yes yet you ').trim();
const COMMON_LOWER = new Set([...SENTENCE_WORDS, ...BANK_VOCAB.split(' '), ...('north north-east east south-east south south-west west north-west northeast northwest southeast southwest northern southern eastern western ' +
  'centre center middle upper lower top bottom left right').split(' ')]);
const isAcronym = (w) => /^[\p{Lu}\d_/-]+$/u.test(w) && /\p{Lu}/u.test(w);
const cleanWord = (w) => w.replace(/^[^\p{L}\p{N}]+/u, '').replace(/[’']s$/u, '').replace(/[^\p{L}\p{N}'’-]+$/u, '').replace(/[’']s$/u, '');
const UNIT_WORDS = [['degrees per second', 'deg/s'], ['degree per second', 'deg/s'], ['deg/s', 'deg/s'], ['km/h', 'km/h'], ['m/s', 'm/s'], ['cm/s', 'cm/s'], ['u/s', 'u/s'], ['feet per minute', 'fpm'], ['fpm', 'fpm'],
  ['mph', 'mph'], ['knots', 'kt'], ['knot', 'kt'], ['kt', 'kt'], ['nautical miles', 'NM'], ['nautical mile', 'NM'], ['nm', 'NM'], ['kilometres', 'km'], ['kilometers', 'km'], ['kilometre', 'km'], ['kilometer', 'km'], ['km', 'km'],
  ['miles', 'mi'], ['mile', 'mi'], ['mi', 'mi'], ['feet', 'ft'], ['foot', 'ft'], ['ft', 'ft'], ['metres', 'm'], ['meters', 'm'], ['metre', 'm'], ['meter', 'm'], ['m', 'm'], ['seconds', 's'], ['second', 's'], ['sec', 's'],
  ['s', 's'], ['degrees', 'deg'], ['degree', 'deg'], ['deg', 'deg'], ['°', 'deg'], ['g', 'g'], ['dots', 'dots'], ['dot', 'dots'], ['per cent', '%'], ['percent', '%'], ['%', '%'], ['u', 'u']];
const UNIT_OF = Object.fromEntries(UNIT_WORDS), esc = (s) => s.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&');
// the same for patterns with the u flag, where "\-" outside a class is a syntax error
const escU = (s) => s.replace(/[/\\^$*+?.()|[\]{}]/g, '\\$&');
const UALT = UNIT_WORDS.map(([w]) => esc(w)).join('|'), UEND = '(?![\\p{L}\\d])', UN = `(${UALT})${UEND}`, UNOPT = `(?:(?:${UALT})${UEND})?`;
const NUM = '(\\d[\\d,]*(?:\\.\\d+)?)', toNum = (s) => +s.replace(/,/g, '');
const RANGE_RES = [[new RegExp(`\\bbetween\\s+(?:about\\s+)?${NUM}\\s*${UNOPT}\\s+and\\s+${NUM}\\s*${UN}`, 'giu'), (m) => [toNum(m[1]), toNum(m[2]), m[3]]],
  [new RegExp(`${NUM}\\s*(?:-|–|to)\\s*${NUM}\\s*${UN}`, 'giu'), (m) => [toNum(m[1]), toNum(m[2]), m[3]]],
  [new RegExp(`(?:\\b(?:under|below|less than|within|closer than|up to)\\s+(?:about\\s+)?|<\\s*)${NUM}\\s*${UN}`, 'giu'), (m) => [0, toNum(m[1]), m[2]]],
  [new RegExp(`(?:\\b(?:over|above|more than|beyond|farther than|further than)\\s+(?:about\\s+)?|>\\s*)${NUM}\\s*${UN}`, 'giu'), (m) => [toNum(m[1]), Infinity, m[2]]]];
const NUMBER_RE = new RegExp(`${NUM}\\s*${UN}`, 'giu');
const COUNT_WORDS = { no: 0, none: 0, zero: 0, 'a single': 1, single: 1, 'a pair of': 2, 'pair of': 2 };
const NW = `no|none|zero|a single|single|a pair of|pair of|${WORDS.slice(1).join('|')}|\\d+`, countOf = (k) => COUNT_WORDS[k] ?? (WORDS.indexOf(k) >= 0 ? WORDS.indexOf(k) : +k);
const COUNT_RE = new RegExp(`\\b(${NW})\\s+(rock|comet|satellite|airliner|flock|bird|hazard|jet|white|red)(?:e?s)?\\b`, 'g');
// PAPI phrasings the count pattern misses: "all four lights are red", "the lights are all white", "three of the lights show white"
const PAPI_RES = [[/\ball\s+(?:four\s+)?(?:of\s+the\s+)?(?:papi\s+)?(?:lights?\s+)?(?:are\s+|show\s+|read\s+|showing\s+)?(white|red)\b/g, (m) => [m[1], 4]],
  [/\blights?\s+(?:are|show|read)\s+all\s+(white|red)\b/g, (m) => [m[1], 4]], [new RegExp(`\\b(${NW})\\s+of\\s+the\\s+(?:four\\s+)?(?:papi\\s+)?lights?\\s+(?:show|are|read|showing|shows|is)\\s+(white|red)\\b`, 'g'), (m) => [m[2], countOf(m[1])]]];
const CLOCK_WORD = `(1[0-2]|[1-9]|${WORDS.slice(1, 13).join('|')})`, clockNum = (s) => (/\d/.test(s) ? +s : WORDS.indexOf(s));
const REGION_ALIAS = { 'upper center': 'upper centre', 'lower center': 'lower centre', 'top left': 'upper left', 'top right': 'upper right', 'bottom left': 'lower left', 'bottom right': 'lower right',
  'top centre': 'upper centre', 'bottom centre': 'lower centre', top: 'upper centre', bottom: 'lower centre', center: 'centre', middle: 'centre' };
const COMPASS = ['north', 'north-east', 'east', 'south-east', 'south', 'south-west', 'west', 'north-west'];
const REASON_CTX = /\b(unsafe|safe|caution|monitor|because|due to|reasons?|cites?|cited|citing|flags?|flagged|owing to|caused by|driven by)\b/i;
const REASON_PHRASES = Object.entries(REASON_TEXT).map(([r, w]) => [r, w.replace(/^(a|an|the) /, '')]).sort((a, b) => b[1].length - a[1].length);
const ACTION_PHRASES = Object.entries(ACTION_TEXT).filter(([a]) => a !== 'CONTINUE').sort((a, b) => b[1].length - a[1].length);
// "continue" is an action only after an advice cue ("should continue", "advises the pilot to continue", "best action: continue")
const CONTINUE_RE = /\b(?:should|advises?|advised|advice(?: is)?|suggests?|suggested|recommends?|recommended(?: action)?|best action|action(?: is)?|is to|to)\s*:?\s*(?:the (?:pilot|crew|spacecraft|aircraft|ship)\s+)?(?:(?:is\s+)?to\s+)?(continue)\b/g;
// category words: [dim, value, phrase]; the value is what the fact must hold (see CAT below)
const OUTCOME_PHRASES = Object.entries(W.outcome).map(([o, p]) => [o, p]).sort((a, b) => b[1].length - a[1].length);
const LIGHT = [['in full daylight', ['day']], ['in low sunlight', ['golden']], ['in twilight', ['twilight', 'dusk']], ['twilight', ['twilight', 'dusk']], ['at night', ['night']], ['night-time', ['night']],
  ['in darkness', ['night']], ['darkness', ['night']], ['dusk light', ['dusk', 'twilight', 'golden']], ['at dusk', ['dusk', 'twilight', 'golden']], ['in direct sunlight', ['lit', 'day']], ['full sunlight', ['lit', 'day']],
  ['sunlit', ['lit', 'day']], ['lit by the sun', ['lit', 'day']], ['in sunlight', ['lit', 'day']], ['sunlight', ['lit', 'day', 'golden']], ['daylight', ['day', 'golden']], ['in shadow', ['shadow']], ['shadow', ['shadow']]];
const TAGP = Object.entries(TAG_WORDS).filter(([t]) => t !== 'NIGHT' && t !== 'URBAN').flatMap(([t, ps]) => ps.filter((p) => p !== 'open sea').map((p) => [p, t])).concat([['city areas', 'URBAN']]);
const CATS = [...Object.entries(W.size).map(([k, p]) => ['size', k, p]), ...LIGHT.map(([p, v]) => ['light', v, p]), ...PALETTE_NAMES.map((c) => ['colour', c, c]), ['colour', 'grey', 'gray'],
  ...Object.entries(W.bright).map(([k, p]) => ['bright', k, p]), ['bright', 'dark', 'dim'], ...['highly textured', 'textured', 'smooth'].map((p) => ['texture', p, p]),
  ...Object.entries(W.vis).map(([k, p]) => ['vis', k, p]), ['vis', 'haze', 'hazy'], ['vis', 'fog', 'foggy'], ...Object.entries(W.phase).filter(([k]) => k !== 'BREAKOUT').map(([k, p]) => ['phase', k, p]),
  ...Object.entries(W.vert).map(([k, p]) => ['lphase', k, p]), ...W.papi.map((p, i) => ['papi', i, p]), ...Object.entries(W.band).map(([k, p]) => ['band', k, p]),
  ...[...new Set(Object.values(W.world))].map((p) => ['world', Object.keys(W.world).filter((k) => W.world[k] === p), p]), ...TAGP.map(([p, t]) => ['tag', t, p]),
  ...Object.entries(W.relief).map(([t, p]) => ['tag', t, p]), ...Object.entries(W.clouds).map(([k, p]) => ['clouds', k, p]), ...Object.entries(W.sky).map(([k, p]) => ['sky', k, `${p} (?:skies|sky|weather)`]),
  ...Object.entries(W.sky).map(([k, p]) => ['sky', k, `(?:sky|skies|weather)(?: here)? (?:is|are|looks|look) ${p}`]), ['sky', 'storm', 'stormy']]
  .map(([dim, value, p]) => [dim, value, new RegExp(`(?<![\\p{L}-])${p.includes('(?') ? p : escU(p)}(?![\\p{L}-])${p === 'large' ? '(?! towns?\\b)' : ''}`, 'gu'), p.length]).sort((a, b) => b[3] - a[3]);
// relations that need a direction word next to their subject
const RELS = [['gs', /\b(?:(?:well|slightly) )?(above|below) the glideslope\b/g, (m) => m[1]], ['gs', /\bon (?:the )?glideslope\b(?! capture| gate)/g, () => 'on'],
  ['loc', /\b(left|right) of (?:the )?(?:localizer|course)\b/g, (m) => m[1]], ['loc', /\bon (?:the )?localizer\b(?! gate)/g, () => 'on'], ['loc', /\b(?:localizer|course)\b[^.]{0,50}?\bto the (left|right)\b/g, (m) => m[1]],
  ['cross', /\bcrosswind\b[^.;]{0,40}?\bfrom the (left|right)\b/g, (m) => m[1]], ['headtail', /\b(head|tail)wind\b/g, (m) => m[1]],
  ['dclosing', /\b(?:closing )?(faster|slower) than (?:allowed|the (?:allowed rate|limit)|its limit)\b/g, (m) => (m[1] === 'faster' ? 'above' : 'below')],
  ['dclosing', /\b(above|below|over|under|within|inside|beyond|exceeds|higher than|lower than) (?:the |its )?limit\b/g, (m) => (['above', 'over', 'beyond', 'exceeds', 'higher than'].includes(m[1]) ? 'above' : 'below')],
  ['closing', /\b(closing in|closing fast|closing on|getting closer|approaching|approaches|converg\w*|gap shrinks)\b/g, () => 1], ['closing', /\b(receding|recedes|moving away|draws away|opening away|getting farther away|separate at|separating|gap grows)\b/g, () => -1],
  ['turb', /\b(light|moderate|severe)(?:ly)? turbulen(?:ce|t)\b/g, (m) => m[1]], ['turb', /\bturbulen(?:ce|t)\b[^.;]{0,25}?\b(light|moderate|severe)\b/g, (m) => m[1]],
  ['stall', /\b(?:close to|near|at|past|in|into) (?:a |the )?stall\b|\bstalled\b/g, () => true], ['gear', /\bgear(?:-| is | are | shows | appears )(?:currently |still |now )?(down|up|in transit|extended|lowered)\b/g, (m) => (m[1] === 'up' || m[1] === 'in transit' ? m[1].replace('in ', '') : 'down')],
  ['side', /\b(?:on|to|at|toward|towards|near) the (left|right|middle|upper|lower|top|bottom)(?: (?:side|edge|third|part))?(?: of the (?:image|frame|picture|view))?(?! of the (?:localizer|flight band|band))\b/g, (m) => ({ middle: 'centre', upper: 'top', lower: 'bottom' })[m[1]] || m[1]]];
// R9: descriptors no fact grounds
const UNGROUNDED = /\b(burning|on fire|ablaze|damaged|wrecked|destroyed|exploding|explodes|exploded|smoking|smouldering|beautiful|stunning|breathtaking|gorgeous|spectacular|majestic|dramatic|picturesque|scenic|magnificent|amazing|awesome|terrifying|menacing|ominous|eerie|glowing|lush|serene|peaceful|tranquil|vibrant|vast|massive|huge|gigantic|enormous|colossal|deadly)\b/g;
// the quantity nouns a number can be bound to, with the facts each one names
const QTY = [[/\b(?:head|tail)wind/g, [/^wind\.head_kt$/]], [/\bcrosswind/g, [/^wind\.cross_kt$/]], [/\b(?:airspeed|ias|indicated)\b/g, [/^ias_kt$/, /^air\.V_u_s$/]],
  [/\b(?:(?<!closing )speed(?! limit)|moving(?! away)|moves|flying|travell?ing|travels)\b/g, [/^ship\.speed_[mu]_s$/, /^air\.V_u_s$/, /^ias_kt$/]],
  [/\b(?:clos(?:ing|es)|closing speed|moving away|recedes|receding|approaches|converge|separate|gap)\b/g, [/^closing_cms$/, /^inward_cms$/, /^hazard\.\d\.closing_u_s$/]],
  [/\b(?:limit|permitted|allowed|exceed)\b/g, [/_limit_cms$/]], [/\b(?:hazard|rock|comet|satellite|airliner|flock)\b/g, [/^hazard\.\d\.dist_u$/]], [/\b(?:closest approach|miss|pass|passes|separation)\b/g, [/^hazard\.\d\.cpa_u$/]],
  [/\bcontact\b/g, [/^hazard\.\d\.ttc_s$/, /^ttc_s$/]], [/\b(?:station|port)\b/g, [/^rho_m$/, /^r_m$/, /^station_distance_bin$/]], [/\b(?:clearance|ground|terrain|surface)\b/g, [/^clearance\./, /^air\.agl_m$/]],
  [/\b(?:ceiling|headroom|upper limit|top of the (?:flight )?band)\b/g, [/^edges\.ceiling_u$/]], [/\bstall/g, [/^air\.stall_margin_deg$/]], [/\b(?:g-load|load|pulling|pulls)\b/g, [/^air\.n_g$/]],
  [/\b(?:propellant|fuel|tanks?)\b/g, [/^fuel_frac$/]], [/\b(?:sea|water|ocean)\b/g, [/^geo\.sea_frac$/]], [/\b(?:range|camera|viewpoint|from)\b/g, [/^view\.range_km$/]],
  [/\b(?:pixel|imagery|resolution|sample distance)\b/g, [/^view\.gsd_m$/]], [/\btown\b/g, [/^place\.nearest$/]], [/\b(?:storm|cell|thunderstorm)\b/g, [/^weather\.cells$/]],
  [/\b(?:windsock|sock)\b/g, [/^windsock\.from_deg$/]], [/\b(?:attitude|pointing|aligned|alignment|port axis)\b/g, [/^att_err_deg$/]], [/\b(?:pitch|nose|pitched)\b/g, [/^ship\.pitch_deg$/, /^att\.pitch_deg$/]],
  [/\bglideslope\b/g, [/^ils\.gs_dots$/]], [/\blocalizer\b/g, [/^ils\.loc_dots$/]], [/\b(?:sink|vertical speed|descent rate)\b/g, [/^vs_fpm$/]], [/\b(?:altitude|height)\b/g, [/^alt_ft$/, /^ra_ft$/]]];
const CLAUSE_CUT = /[;:]|\b(?:and|with|but|while|whereas|against|versus|than)\b/g;
// a sentence continues the previous one's monitor attribution when it opens with one of these
const CONT_CUE = /^(?:it|its|so|best action|advice|recommended|reason|the recommended action|the pilot should|the crew should|the spacecraft should)\b/i;

// lower-casing that keeps every index (a character whose lower case changes length is left as it is)
const lowerSame = (s) => s.replace(/./gsu, (ch) => { const l = ch.toLowerCase(); return l.length === ch.length ? l : ch; });
const SENT_END = /[.!?:;]["')\]]*$/;
const NEGATED = /\b(?:not|no|never|nor|neither)\s+(?:\S+\s+)?$/;
export function parseClaims(text, gaz) {
  const src = String(text).replace(/[’‘]/g, "'"), c = [];
  const sents = [...src.matchAll(/(?:[^.!?;]|\.(?=\d))+[.!?;]?/g)].map((m) => ({ at: m.index, text: m[0] })), sentAt = (at) => { let k = 0; while (k + 1 < sents.length && sents[k + 1].at <= at) k++; return k; };
  const push = (type, value, at, extra = {}) => c.push({ type, value, at, sent: sentAt(at), ...extra });
  let w = src;
  const blank = (at, len) => { w = w.slice(0, at) + ' '.repeat(len) + w.slice(at + len); };
  const scan = (re, fn, lower = true) => { for (const m of [...(lower ? lowerSame(w) : w).matchAll(re)]) { if (fn(m) !== false) blank(m.index, m[0].length); } };
  // entities first, so a place name's compass or number words are not read twice; a capitalised word that opens a sentence
  // (or follows ":" / ";") is a common word, a gazetteer name or an unknown name
  const words = [...src.matchAll(/\S+/g)], start = (i) => i === 0 || SENT_END.test(words[i - 1][0]);
  for (let i = 0; i < words.length; i++) {
    const clean = cleanWord(words[i][0]), low = clean.toLowerCase(), cap = /^\p{Lu}/u.test(clean);
    let hit = null;
    for (let n = Math.min(gaz.maxWords, words.length - i); n >= 1 && !hit; n--) {
      if (words.slice(i, i + n - 1).some((x) => SENT_END.test(x[0]))) continue;
      const parts = words.slice(i, i + n).map((x) => cleanWord(x[0])), span = parts.join(' ');
      if (!gaz.has(span)) continue;
      if (cap ? !(n === 1 && (STOP.has(clean) || (start(i) && COMMON_LOWER.has(low)))) : !/\d/.test(span) && parts.some((p) => !COMMON_LOWER.has(p.toLowerCase()))) hit = [span, n];
    }
    if (hit) { const a = words[i].index, e = words[i + hit[1] - 1]; if (!TERM_KEYS.has(norm(hit[0]))) push('entity', gaz.canonical(hit[0]), a); blank(a, e.index + e[0].length - a); i += hit[1] - 1; }
    else if (cap && !STOP.has(clean) && !isAcronym(clean) && !(start(i) && (SENTENCE_WORDS.has(low) || /^\p{L}+ly$/u.test(low)))) push('unknown_entity', clean, words[i].index);
  }
  const numAt = (m) => { const st = sents[sentAt(m.index)], lowS = lowerSame(st.text), off = m.index - st.at, cuts = [0, ...[...lowS.matchAll(CLAUSE_CUT)].map((x) => x.index + x[0].length), lowS.length + 1];
    const a = Math.max(...cuts.filter((x) => x <= off)), b = Math.min(...cuts.filter((x) => x > off)); return { clause: lowS.slice(a, b), off: off - a, len: m[0].length, sentText: lowS, sentOff: off }; };
  for (const [re, get] of RANGE_RES) scan(re, (m) => { const [lo, hi, uw] = get(m), unit = UNIT_OF[uw.toLowerCase()]; push('range', null, m.index, { lo, hi, unit, dim: UNITS[unit][0], ...numAt(m) }); }, false);
  scan(NUMBER_RE, (m) => { const v = toNum(m[1]), unit = UNIT_OF[m[2].toLowerCase()], r = roundNice(v); push('number', v, m.index, { unit, dim: UNITS[unit][0], lo: r.lo, hi: r.hi, ...numAt(m) }); }, false);
  for (const [re, get] of PAPI_RES) scan(re, (m) => { const [noun, n] = get(m); push('count', n, m.index, { noun }); });
  scan(COUNT_RE, (m) => { push('count', countOf(m[1]), m.index, { noun: m[2] === 'flock' || m[2] === 'bird' ? 'birds' : m[2] }); });
  scan(new RegExp(`\\b(?:at\\s+)?your\\s+${CLOCK_WORD}\\s*o'clock`, 'g'), (m) => push('bearing_clock', clockNum(m[1]), m.index));
  scan(new RegExp(`\\b${CLOCK_WORD}\\s*o'clock`, 'g'), (m) => push('clock', clockNum(m[1]), m.index));
  scan(/\b(upper left|upper centre|upper center|upper right|middle left|middle right|lower left|lower centre|lower center|lower right|top left|top right|top centre|bottom left|bottom right|bottom centre)\b/g, (m) => push('region', REGION_ALIAS[m[1]] || m[1], m.index));
  scan(/\b(?:in|at) the (centre|center|middle|top|bottom)\b/g, (m) => push('region', REGION_ALIAS[m[1]] || m[1], m.index));
  scan(/\b(north-east|north-west|south-east|south-west|northeast|northwest|southeast|southwest|north|south|east|west)\b/g, (m) => push('compass', m[1].replace(/^(north|south)(east|west)$/, '$1-$2'), m.index));
  for (const [a, p] of ACTION_PHRASES) scan(new RegExp(`\\b${esc(p)}\\b`, 'g'), (m) => push('action', a, m.index));
  scan(CONTINUE_RE, (m) => push('action', 'CONTINUE', m.index + m[0].lastIndexOf('continue')));
  scan(/(?<![\p{L}])(?<!no )(unsafe|safe|caution)(?![\p{L}])(?! (?:action|breakout|option|distance|speed|margin|limit))/gu, (m) => push('verdict', m[1].toUpperCase(), m.index));
  // reason words count only in a sentence that states a verdict, a cause or the monitor ("a tailwind of 5 kt" is a wind fact)
  for (const s of sents) {
    if (!REASON_CTX.test(s.text)) continue;
    const part = lowerSame(w.slice(s.at, s.at + s.text.length));
    for (const [r, p] of REASON_PHRASES) for (const m of [...part.matchAll(new RegExp(`\\b${esc(p)}\\b`, 'g'))]) { push('reason', r, s.at + m.index); blank(s.at + m.index, m[0].length); }
  }
  for (const [o, p] of OUTCOME_PHRASES) scan(new RegExp(`\\b${esc(p)}\\b`, 'g'), (m) => push('outcome', o, m.index));
  for (const [dim, re, get] of RELS) scan(re, (m) => { if (NEGATED.test(lowerSame(w.slice(Math.max(0, m.index - 16), m.index)))) return false; push('category', get(m), m.index, { dim }); });
  for (const [dim, value, re] of CATS) scan(re, (m) => { if (NEGATED.test(lowerSame(w.slice(Math.max(0, m.index - 16), m.index)))) return false; push('category', value, m.index, { dim }); });
  scan(UNGROUNDED, (m) => push('adjective', m[1], m.index));
  return c.sort((a, b) => a.at - b.at);
}

// numbers the facts hold: plain numbers with a unit, and km/dist fields of object facts (place.nearest.km, weather.cells)
const SUB_UNIT = { km: 'km', dist_u: 'u', dist_m: 'm' };
function numericEntries(F) {
  const out = [], sub = (id, o) => { for (const [k, u] of Object.entries(SUB_UNIT)) if (o && typeof o[k] === 'number') out.push([id, o[k], u]); };
  for (const [id, f] of Object.entries(F)) {
    const u = unitOf(id, f), v = f.v;
    if (typeof v === 'number' && UNITS[u]) out.push([id, v, u]);
    else if (Array.isArray(v)) v.forEach((x) => (x && typeof x === 'object' && !Array.isArray(x) ? sub(id, x) : null));
    else if (v && typeof v === 'object') sub(id, v);
  }
  return out;
}
// bin facts stored as text ('20-100 m', '<2 m', '>400 m'): the interval they name, in SI
const BIN_RE = /^(<|>)?(\d+(?:\.\d+)?)(?:-(\d+(?:\.\d+)?))? (m|km|s|u)$/;
function binEntries(F) {
  const out = [];
  for (const [id, f] of Object.entries(F)) {
    const m = typeof f.v === 'string' ? BIN_RE.exec(f.v) : null; if (!m) continue;
    const k = UNITS[m[4]][1], a = +m[2], b = m[3] === undefined ? null : +m[3];
    out.push([id, UNITS[m[4]][0], m[1] === '<' ? [0, a * k] : m[1] === '>' ? [a * k, Infinity] : [a * k, b * k]]);
  }
  return out;
}
function factNames(id, f) {
  const s = new Set(), add = (x) => { if (typeof x === 'string') s.add(norm(x)); }, v = f.v;
  if (Array.isArray(v)) v.forEach((x) => add(x && typeof x === 'object' ? x.name : x)); else if (v && typeof v === 'object') add(v.name); else add(v);
  if ((id === 'route' || id === 'world') && ROUTE_NAMES[v]) add(ROUTE_NAMES[v]);
  return s;
}
const inIv = (x, lo, hi) => x >= lo - 1e-9 * Math.max(1, Math.abs(lo)) && x <= hi + 1e-9 * Math.max(1, Math.abs(hi));
const regionOfBox = (b) => `${['upper', 'middle', 'lower'][Math.min(2, Math.floor(3 * ((b[1] + b[3]) / 2) / 504))]} ${['left', 'centre', 'right'][Math.min(2, Math.floor(3 * ((b[0] + b[2]) / 2) / 896))]}`.replace('middle centre', 'centre');
const sameBin = (lo, hi, [a, b]) => Math.abs(lo - a) <= 0.01 * Math.max(1, a) && (hi === Infinity ? b === Infinity : b !== Infinity && Math.abs(hi - b) <= 0.01 * Math.max(1, b));
// a number is bound to the nearest quantity noun of its sentence whose facts have its dimension; unbound, any fact may hold it
// (within its clause: before it, or up to four words after it, a noun after it counting half a word further)
const wordsIn = (s) => (s.match(/[\p{L}\d][\p{L}\d'-]*/gu) || []).length;
// (within its clause: before it, or up to four words after it, a noun after it counting half a word further; a clause
// without such a noun falls back to its sentence)
function nearestNoun(text, off, len, dim, num, bins, maxAfter) {
  const end = off + len; let best = null;
  for (const [re, pats] of QTY) {
    const ids = [...num.filter(([id, , u]) => UNITS[u][0] === dim && pats.some((p) => p.test(id))).map(([id]) => id), ...bins.filter(([id, d]) => d === dim && pats.some((p) => p.test(id))).map(([id]) => id)];
    if (!ids.length) continue;
    for (const m of text.matchAll(re)) {
      const d = m.index + m[0].length <= off ? wordsIn(text.slice(m.index + m[0].length, off)) : m.index >= end ? wordsIn(text.slice(end, m.index)) + 0.5 : 0;
      if (m.index >= end && d > maxAfter) continue;
      if (!best || d < best.d) best = { d, ids };
    }
  }
  return best;
}
function boundIds(cl, F) {
  if (cl.clause === undefined) return null;
  const num = numericEntries(F), bins = binEntries(F), best = nearestNoun(cl.clause, cl.off, cl.len, cl.dim, num, bins, 4.5) || nearestNoun(cl.sentText, cl.sentOff, cl.len, cl.dim, num, bins, 4.5);
  return best ? new Set(best.ids) : null;
}
const OUTCOME_OK = { SAFE: ['landed', 'capture', 'clear'], CAUTION: ['landed', 'capture', 'clear', 'go_around', 'breakout', 'crash_possible'] };
// an L/D outcome a Narrator may state next to the monitor's verdict (controller ruling (b)); UNSAFE admits any outcome
export const outcomeAgrees = (outcome, verdict) => verdict === 'UNSAFE' || (OUTCOME_OK[verdict] || []).includes(outcome);
const has = (F, id) => F[id] && F[id].v !== null && F[id].v !== undefined;
const idsWhere = (F, pred) => Object.entries(F).filter(([id, f]) => f.v !== null && f.v !== undefined && pred(id, f.v)).map(([id]) => id);
const CAT = { size: (F, v) => idsWhere(F, (id, x) => /^hazard\.\d\.size_bin$/.test(id) && x === v), side: (F, v) => idsWhere(F, (id, x) => (/^hazard\.\d\.side$/.test(id) || id === 'geo.coast_side') && x === v),
  sky: (F, v) => idsWhere(F, (id, x) => id === 'weather.preset' && x === v), colour: (F, v) => idsWhere(F, (id, x) => /^image\.palette_\d$/.test(id) && x === v),
  light: (F, vs) => idsWhere(F, (id, x) => ((id === 'sun.class' || id === 'scene.time') && vs.includes(x)) || (id === 'sun.lit' && ((vs.includes('lit') && x >= 0.9) || (vs.includes('shadow') && x <= 0.1)))),
  bright: (F, v) => idsWhere(F, (id, x) => id === 'image.brightness_bin' && x === v), texture: (F, v) => idsWhere(F, (id, x) => id === 'image.edge_bin' && x === v),
  vis: (F, v) => idsWhere(F, (id, x) => id === 'scene.vis' && x === v), phase: (F, v) => idsWhere(F, (id, x) => id === 'phase' && x === v), lphase: (F, v) => idsWhere(F, (id, x) => id === 'vert_mode' && x === v),
  papi: (F, v) => idsWhere(F, (id, x) => id === 'papi_whites_cam' && x === v), band: (F, v) => idsWhere(F, (id, x) => id === 'air.speed_band' && x === v),
  world: (F, vs) => idsWhere(F, (id, x) => id === 'world' && vs.includes(x)), tag: (F, v) => idsWhere(F, (id, x) => id === 'zoom.tags' && Array.isArray(x) && x.includes(v)),
  clouds: (F, v) => idsWhere(F, (id, x) => id === 'scene.clouds' && derive(['cloud_code'], x) === v), turb: (F, v) => idsWhere(F, (id, x) => id === 'air.turbulence' && W.turb[x] === v),
  stall: (F) => idsWhere(F, (id, x) => (id === 'air.stall_margin_deg' && x < 2) || (id === 'air.stalled' && x === true)), gear: (F, v) => idsWhere(F, (id, x) => id === 'cfg.gear' && (x === v || (v === 'transit' && x === 'transit'))),
  gs: (F, v) => idsWhere(F, (id, x) => id === 'ils.gs_dots' && (v === 'on' ? Math.abs(x) <= 1 : v === 'above' ? x > 1 : x < -1)), loc: (F, v) => idsWhere(F, (id, x) => id === 'ils.loc_dots' && (v === 'on' ? Math.abs(x) <= 1 : v === 'right' ? x > 1 : x < -1)),
  cross: (F, v) => idsWhere(F, (id, x) => id === 'wind.cross_kt' && (v === 'right' ? x > 0 : x < 0)), headtail: (F, v) => idsWhere(F, (id, x) => id === 'wind.head_kt' && (v === 'head' ? x > 0 : x < 0)),
  closing: (F, v) => idsWhere(F, (id, x) => /^hazard\.\d\.closing_u_s$/.test(id) && Math.sign(x) === v),
  dclosing: (F, v) => (has(F, 'closing_cms') && has(F, 'corridor_limit_cms') && (F.closing_cms.v > F.corridor_limit_cms.v) === (v === 'above') ? ['closing_cms', 'corridor_limit_cms'] : []) };
// the fact ids (or 'monitor' / 'safety') that support a claim; [] when nothing does
function support(cl, F, rec, ctx, attributed) {
  const S = rec.safety, v = cl.value, ids = (pred) => Object.entries(F).filter(([id, f]) => pred(id, f)).map(([id]) => id);
  if (cl.type === 'number' || cl.type === 'range') {
    const k = UNITS[cl.unit][1], lo = cl.lo * k, hi = cl.hi * k, bound = boundIds(cl, F), ok = (id) => !bound || bound.has(id);
    const out = numericEntries(F).filter(([id, x, u]) => ok(id) && UNITS[u][0] === cl.dim && inIv(Math.abs(x * UNITS[u][1]), lo, hi)).map(([id]) => id);
    if (cl.type === 'number') return out;
    for (const [id, dim, iv] of binEntries(F)) if (ok(id) && dim === cl.dim && sameBin(lo, hi, iv)) out.push(id);
    if (attributed && ctx && ctx.monitor) {
      const t = TTC_RANGE[ctx.monitor.ttc_bin], r = CLR_RANGE_U[ctx.monitor.clr_bin];
      if ((cl.dim === 'time' && t && sameBin(lo, hi, t)) || (cl.dim === 'len' && r && sameBin(lo, hi, [r[0] * 19, r[1] * 19]))) out.push('monitor');
    }
    return out;
  }
  if (cl.type === 'count') {
    const n = F['hazards.count_in_frame'], kinds = F.kinds_in_frame, papi = F.papi_whites_cam;
    if (cl.noun === 'hazard') return n && n.v === v ? ['hazards.count_in_frame'] : [];
    if (cl.noun === 'white') return papi && papi.v === v ? ['papi_whites_cam'] : [];
    if (cl.noun === 'red') return papi && typeof papi.v === 'number' && 4 - papi.v === v ? ['papi_whites_cam'] : [];
    if (cl.noun === 'jet') return ids((id, f) => (id === 'jets_failed' || id === 'jets_firing') && Array.isArray(f.v) && f.v.length === v);
    if (!kinds || !Array.isArray(kinds.v)) return [];
    if (v === 0) return kinds.v.includes(cl.noun) ? [] : ['kinds_in_frame'];
    return kinds.v.length === 1 && kinds.v[0] === cl.noun && n && n.v === v ? ['kinds_in_frame', 'hazards.count_in_frame'] : [];
  }
  if (cl.type === 'clock') return ids((id, f) => id.endsWith('.clock') && f.v === v);
  if (cl.type === 'bearing_clock') return ids((id, f) => id.endsWith('.bearing_clock') && f.v === v);
  if (cl.type === 'region') return ids((id, f) => JSON.stringify(f.v ?? null).includes(`"region":"${v}"`) || (id.endsWith('.box_px') && Array.isArray(f.v) && regionOfBox(f.v) === v));
  if (cl.type === 'compass') return ids((id, f) => f.v && typeof f.v === 'object' && Number.isFinite(f.v.bearing) && Math.abs(((f.v.bearing - COMPASS.indexOf(v) * 45 + 540) % 360) - 180) <= 22.5);
  if (cl.type === 'entity') return ids((id, f) => !id.startsWith('safety.') && factNames(id, f).has(norm(v)));
  if (cl.type === 'category') return CAT[cl.dim] ? CAT[cl.dim](F, v) : [];
  if (cl.type === 'adjective') return [];
  const mon = attributed && ctx && ctx.monitor ? ctx.monitor : null, gtOut = S && S.action_outcome ? S.action_outcome.CONTINUE : null;
  if (cl.type === 'outcome') {
    const out = [];
    if (mon && (rec.family === 'S' || rec.family === 'A') && prefOutcome(mon.p_ref) === v) out.push('monitor');
    if (gtOut === v && !(ctx && ctx.monitor && (rec.family === 'L' || rec.family === 'D') && !outcomeAgrees(v, ctx.monitor.verdict))) out.push('safety.action_outcome');
    return out;
  }
  if (mon) {
    if (cl.type === 'verdict') return mon.verdict === v ? ['monitor'] : [];
    if (cl.type === 'action') return mon.action === v ? ['monitor'] : [];
    if (cl.type === 'reason') return mon.reasons.includes(v) ? ['monitor'] : [];
  }
  if (!S) return [];
  if (cl.type === 'verdict') return S.verdict === v ? ['safety'] : [];
  if (cl.type === 'action') return S.best_action === v || (S.safe_actions || []).includes(v) ? ['safety'] : [];
  if (cl.type === 'reason') return S.reasons.includes(v) ? ['safety'] : [];
  return [];
}
const MONITOR_TYPES = ['verdict', 'action', 'reason'], SEMANTIC = ['category', 'outcome', 'adjective'];
// a claim is the monitor's when its sentence names the monitor, or opens with a continuation cue ("It advises...", "Best action:") after one that does
function attribution(text) {
  const sents = [...String(text).matchAll(/(?:[^.!?;]|\.(?=\d))+[.!?;]?/g)].map((m) => m[0].trim()), out = [];
  sents.forEach((s) => out.push(/monitor/i.test(s) || (out.some(Boolean) && CONT_CUE.test(s))));
  return out;
}
// one parsed claim against the record (and the row's Context): true when some fact, the safety block or the monitor holds it
export const checkClaim = (cl, rec, { context = null, attributed = false, factIds = null } = {}) => {
  const all = factsOf(rec), F = factIds ? Object.fromEntries(Object.entries(all).filter(([id]) => factIds.includes(id))) : all;
  return support(cl, F, rec, normContext(context), attributed).length > 0;
};
function judge(claims, rec, F, ctx, { obsRule, attr }) {
  const errors = [];
  for (const cl of claims) {
    const attributed = !!attr[cl.sent]; cl.monitor = attributed;
    if (cl.type === 'unknown_entity') { errors.push(`unknown entity ${cl.value}`); continue; }
    if (cl.type === 'adjective') { errors.push(`ungrounded descriptor "${cl.value}"`); continue; }
    let sup = support(cl, F, rec, ctx, attributed);
    if (obsRule) sup = sup.filter((id) => (id === 'monitor' || id === 'safety' ? !!(ctx && ctx.monitor) && id === 'monitor' : (F[id] && F[id].obs === 'visual') || (!!ctx && (ctx.ids.has(id) || (!!ctx.monitor && id.startsWith('safety.') && !(id === 'safety.action_outcome' && (rec.family === 'S' || rec.family === 'A')))))));
    if (!sup.length) errors.push(`${cl.type}${cl.dim ? `/${cl.dim}` : ''} ${JSON.stringify(cl.type === 'range' ? [cl.lo, cl.hi, cl.unit] : cl.type === 'number' ? [cl.value, cl.unit] : cl.value)} not supported`);
  }
  return errors;
}
// obsRule (Narrator rows, §5.7): a claim must rest on a visual fact, on a fact the row's Context supplies, or on its monitor
export function verifyFreeText(text, rec, { gaz, context = null, factIds = null, obsRule = false } = {}) {
  const ctx = normContext(context), all = factsOf(rec), F = factIds ? Object.fromEntries(Object.entries(all).filter(([id]) => factIds.includes(id))) : all;
  const claims = parseClaims(text, gaz), errors = judge(claims, rec, F, ctx, { obsRule, attr: attribution(text) });
  if (rec.family !== 'Z' && !ctx && claims.some((cl) => MONITOR_TYPES.includes(cl.type) || cl.type === 'outcome')) errors.push('a flight row without Context states a context-class fact');
  return { verified: errors.length === 0, errors, claims };
}
const deq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
function slotError(s, F) {
  if (MONITOR_TYPES.includes(s.type) || s.type === 'monitor_range' || (s.type === 'outcome' && s.src === 'monitor')) return null;
  const f = F[s.fact_id];
  if (!f || f.v === null || f.v === undefined) return `slot ${s.fact_id} has no fact`;
  const x = derive(s.derive, f.v, F);
  if (s.type === 'number') return typeof x === 'number' && inIv(x, s.lo, s.hi) ? null : `slot ${s.fact_id}: ${x} outside [${s.lo}, ${s.hi}]`;
  if (s.type === 'range') {
    if (typeof x === 'string') return x === s.value ? null : `slot ${s.fact_id}: ${x} is not ${s.value}`;
    const fu = UNITS[unitOf(s.fact_id, f)], y = typeof x === 'number' && fu ? Math.abs(x) * fu[1] / UNITS[s.unit][1] : NaN;
    return inIv(y, s.lo, s.hi) ? null : `slot ${s.fact_id}: ${x} outside ${s.text}`;
  }
  if (s.type === 'entity') return typeof x === 'string' && norm(x) === norm(s.value) ? null : `slot ${s.fact_id}: ${JSON.stringify(x)} is not ${s.value}`;
  return deq(x, s.value) ? null : `slot ${s.fact_id}: ${JSON.stringify(x)} is not ${JSON.stringify(s.value)}`;
}
const keyOfClaim = (c) => `${c.type}:${c.type === 'range' ? `${c.lo}-${c.hi === Infinity ? 'inf' : c.hi}` : c.value}`;
function slotKeys(s, gaz) {
  if (s.type === 'number') return [`number:${s.shown ?? +s.text.replace(/[^\d.]/g, '')}`];
  if (s.type === 'range' || s.type === 'monitor_range') return [`range:${s.shownLo}-${s.shownHi === Infinity ? 'inf' : s.shownHi}`];
  if (['count', 'clock', 'bearing_clock', 'region', 'compass'].includes(s.type)) return [`${s.type}:${s.value}`];
  if (s.type === 'entity') return [`entity:${gaz.canonical(String(s.value)) ?? s.value}`];
  if (s.type === 'category' && typeof s.value === 'string' && gaz.has(s.value) && /^\p{Lu}/u.test(String(s.text))) return [`entity:${gaz.canonical(s.value)}`];
  return [];
}
// template text: (a) every slot (and hidden check) holds for the facts; (b) the parser recovers exactly the slot set (category,
// outcome and descriptor words are checked by meaning, not by slot); plus the free-text check of the answer and of the prompt
// (verdict/action/reason/category words in a question are not claims; a negative's false premise, matched by `premise`, is exempt)
export function verifyTemplateItem(item, rec, { gaz, context = null, premise = null } = {}) {
  const F = factsOf(rec), errors = [];
  for (const s of [...(item.slots || []), ...(item.checks || [])]) { const e = slotError(s, F); if (e) errors.push(e); }
  const got = parseClaims(item.answer, gaz).filter((c) => !MONITOR_TYPES.includes(c.type) && !SEMANTIC.includes(c.type)).map(keyOfClaim).sort();
  const want = (item.slots || []).flatMap((s) => slotKeys(s, gaz)).sort();
  const parserOk = JSON.stringify(got) === JSON.stringify(want);
  if (!parserOk) errors.push(`parser recovered ${got} but the slots are ${want}`);
  for (const e of verifyFreeText(item.answer, rec, { gaz, context }).errors) if (!errors.includes(e)) errors.push(e);
  if (item.prompt) {
    const claims = parseClaims(item.prompt, gaz).filter((c) => !MONITOR_TYPES.includes(c.type) && c.type !== 'category' && c.type !== 'outcome' && !(premise && premise(c)));
    for (const e of judge(claims, rec, F, normContext(context), { obsRule: false, attr: attribution(item.prompt) })) errors.push(`prompt: ${e}`);
  }
  return { verified: errors.length === 0, parserOk, errors };
}
