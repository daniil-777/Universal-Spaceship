// vlm/gen/text/verify_words.js — the word tables of the §5.4 verifier: the words text states facts with (W; slot values and
// the parser's category words come from the same tables), the claim phrases (outcomes, causes, categories, relations,
// ungrounded descriptors, disagreements), the subjects a position claim binds to, and the common-word lexicon that keeps
// sentence openers and lower-case words from being read as places.
import { REASON_TEXT, ACTION_TEXT } from './context.js';
import { PALETTE_NAMES, TAG_WORDS, BRIGHTNESS_BINS, EDGE_BINS, COAST_SIDES } from '../schema.js';
import { escU } from './verify_numbers.js';

// a table keyed by a schema enum, so the text words and the enum cannot drift apart
const keyed = (keys, words) => Object.freeze(Object.fromEntries(keys.map((k, i) => [k, words[i]])));
export const W = Object.freeze({
  size: { tiny: 'tiny', small: 'small', medium: 'medium-sized', large: 'large' },
  side: { left: 'on the left side of the image', centre: 'near the middle of the image', right: 'on the right side of the image' },
  sky: { clear: 'clear', fair: 'fair', cloudy: 'cloudy', storm: 'stormy' },
  turb: { LIGHT: 'light', MOD: 'moderate', SEVERE: 'severe' },
  world: { mountains: 'a mountain range', pillars: 'a field of tall rock pillars', meshy: 'a range of steep pillar-shaped peaks',
    newyork: 'a city skyline', london: 'a city skyline', moscow: 'a city skyline', dubai: 'a city skyline', mega: 'a city skyline' },
  band: { slow: 'below the normal speed band', normal: 'inside the normal speed band', overspeed: 'above the never-exceed speed' },
  vert: { ALT: 'holding altitude before glideslope capture', GS: 'descending with the glideslope captured',
    FLARE: 'in the flare just above the runway', ROLLOUT: 'rolling out on the runway', STOP: 'stopped on the runway',
    GA: 'climbing away in a go-around' },
  phase: { TRANSFER: 'the transfer', 'H1 ACQ': 'the approach to the first hold', H1: 'the first hold', CORRIDOR: 'the corridor approach',
    H2: 'the second hold', FINAL: 'the final approach', BREAKOUT: 'a breakout', DEPART: 'the departure' },
  clouds: { FEW: 'a few clouds', SCT: 'scattered clouds', BKN: 'broken cloud', OVC: 'an overcast layer', NSC: 'no significant cloud',
    SKC: 'no significant cloud', NONE: 'no significant cloud' },
  time: { day: 'daylight', dusk: 'dusk light', night: 'darkness' },
  vis: { cavok: 'clear visibility', haze: 'haze', fog: 'fog' },
  papi: ['well below the glide path', 'slightly below the glide path', 'on the glide path', 'slightly above the glide path',
    'well above the glide path'],
  gate: { lateral: 'the lateral mode', vertical: 'the vertical mode', loc: 'the localizer gate', gs: 'the glideslope gate',
    speed: 'the speed', vs: 'the sink rate', gear: 'the gear' },
  relief: { MOUNTAINS: 'mountainous', HILLS: 'hilly', FLAT: 'mostly flat' },
  coast: keyed(COAST_SIDES, ['left', 'right', 'upper', 'lower']),
  bright: keyed(BRIGHTNESS_BINS, ['dark', 'of medium brightness', 'bright']),
  daylight: { day: 'in full daylight', golden: 'in low sunlight', twilight: 'in twilight', night: 'at night' },
  outcome: { clear: 'the flight stays clear of everything', crash_possible: 'a crash is possible', crash_certain: 'a crash is certain',
    landed: 'the aircraft touches down and completes the landing', go_around: 'the approach ends in a go-around',
    hard: 'the aircraft lands hard', excursion: 'the aircraft runs off the side of the runway',
    overrun: 'the aircraft overruns the end of the runway',
    short: 'the aircraft touches down short of the runway', tailstrike: 'the tail strikes the runway', crash: 'the aircraft crashes',
    capture: 'the spacecraft docks with the station', breakout: 'the approach ends in a breakout', fail: 'the docking fails' },
  cause: { rock: 'the rock', comet: 'the comet', satellite: 'the satellite', airliner: 'the airliner', birds: 'the flock of birds',
    terrain: 'the terrain', building: 'a building', roof: 'a rooftop', overstress: 'overloading the airframe', ground: 'the ground',
    runway: 'running off the runway', station: 'the station itself' } });
// a cause a text may name next to a monitor: one that explains one of the monitor's reasons
const HZ = ['HAZARD_AHEAD', 'HAZARD_CLOSING_FAST'];
export const CAUSE_REASONS = Object.freeze({ rock: HZ, comet: HZ, satellite: HZ, airliner: HZ, birds: HZ,
  terrain: ['TERRAIN_CLOSE', 'PULL_UP'], building: ['BUILDING_CLOSE'], roof: ['BUILDING_CLOSE'], overstress: ['OVERSTRESS'],
  ground: ['UNSTABLE_APPROACH', 'HIGH_SINK_RATE', 'GLIDESLOPE_DEVIATION'], runway: ['RUNWAY_EDGE', 'CANNOT_STOP'],
  station: ['KOS_VIOLATION', 'LATERAL_MISALIGNMENT', 'CLOSING_TOO_FAST', 'ATTITUDE_ERROR'] });
// terrain words a corridor world or route grounds (the zoom tags ground them for Z)
const PEAKS = ['MOUNTAINS', 'HILLS', 'HIGH_TERRAIN'], CITY = ['URBAN'];
export const WORLD_TAGS = Object.freeze({ mountains: PEAKS, pillars: PEAKS, meshy: PEAKS, alps: PEAKS, china: PEAKS, newyork: CITY,
  london: CITY, moscow: CITY, dubai: CITY, mega: CITY });
export const ROUTE_NAMES = Object.freeze({ alps: 'Alps', china: 'China', newyork: 'New York', london: 'London', moscow: 'Moscow',
  dubai: 'Dubai', mega: 'Megacity' });
// the fixed part of the §5.4 gazetteer; the build adds every Natural Earth name it loads. Terms (acronyms, docking words,
// bodies, the airport) are known words, never place claims; route names are places.
export const TERMS = Object.freeze(['Astro Pilot Spaceport', 'APX', 'IDSS', 'KOS', 'V-bar', 'Earth', 'Moon', 'Sun', 'PAPI', 'ILS', 'LOC',
  'GS', 'IAS', 'METAR', 'AGL', 'UTC', 'TTC']);
export const BASE_NAMES = Object.freeze([...Object.values(ROUTE_NAMES), ...TERMS]);
export const NOUNS = Object.freeze({ rock: ['rock', 'rocks'], comet: ['comet', 'comets'], satellite: ['satellite', 'satellites'],
  airliner: ['airliner', 'airliners'], birds: ['flock of birds', 'flocks of birds'], hazard: ['hazard', 'hazards'], jet: ['jet', 'jets'],
  white: ['white', 'white'], red: ['red', 'red'] });

// ---- claim phrases ----
const longestFirst = (a, b) => b[1].length - a[1].length;
export const REASON_CTX = new RegExp('\\b(unsafe|safe|caution|monitor|because|due to|reasons?|cites?|cited|citing|flags?|flagged'
  + '|owing to|caused by|driven by)\\b', 'i');
export const REASON_PHRASES = Object.entries(REASON_TEXT).map(([r, w]) => [r, w.replace(/^(a|an|the) /, '')]).sort(longestFirst);
export const ACTION_PHRASES = Object.entries(ACTION_TEXT).filter(([a]) => a !== 'CONTINUE').sort(longestFirst);
// "continue" is an action only after an advice cue ("should continue", "advises the pilot to continue", "best action: continue")
export const CONTINUE_RE = new RegExp('\\b(?:should|advises?|advised|advice(?: is)?|suggests?|suggested|recommends?|recommended(?: action)?'
  + '|best action|action(?: is)?|is to|to)\\s*:?\\s*(?:the (?:pilot|crew|spacecraft|aircraft|ship)\\s+)?(?:(?:is\\s+)?to\\s+)?'
  + '(continue)\\b', 'g');
export const OUTCOME_PHRASES = Object.entries(W.outcome).sort(longestFirst);
// a cause is named in a sentence about the threat or danger ("the primary threat is the comet")
export const CAUSE_CUE = /\b(threats?|dangers?|dangerous|risks?|worry|matters most)\b/i;
export const CAUSE_PHRASES = Object.entries(W.cause).sort(longestFirst);
// a sentence continues the previous one's monitor attribution when it opens with one of these
export const CONT_CUE = new RegExp('^(?:it|its|so|best action|advice|recommended|reason|the recommended action|the pilot should'
  + '|the crew should|the spacecraft should)\\b', 'i');
// disagreeing with the monitor is never grounded: a visual contradiction is stated as the fact itself ("But the frame shows ...")
export const DISAGREE = new RegExp('\\b(?:monitor\\b[^.;]{0,30}\\b(?:is wrong|is mistaken|is incorrect|overstates|understates|errs'
  + '|got it wrong)|(?:disagree|despite|ignore|contrary to|regardless of)\\b[^.;]{0,20}\\bmonitor|in reality)\\b', 'gi');
// R9: descriptors no fact grounds
export const UNGROUNDED = new RegExp('\\b(burning|on fire|ablaze|damaged|wrecked|destroyed|exploding|explodes|exploded|smoking'
  + '|smouldering|beautiful|stunning|breathtaking|gorgeous|spectacular|majestic|dramatic|picturesque|scenic|magnificent|amazing'
  + '|awesome|terrifying|menacing|ominous|eerie|glowing|lush|serene|peaceful|tranquil|vibrant|vast|massive|huge|gigantic|enormous'
  + '|colossal|deadly|spinning|glow|warm)\\b', 'g');

// ---- categories: [dim, value, phrase]; the value is what the fact must hold (verify_rules.js CAT) ----
const DUSK = ['dusk', 'twilight', 'golden'], LIT = ['lit', 'day'];
const LIGHT = [['in full daylight', ['day']], ['in low sunlight', ['golden']], ['in twilight', ['twilight', 'dusk']],
  ['twilight', ['twilight', 'dusk']], ['at night', ['night']], ['night-time', ['night']], ['in darkness', ['night']],
  ['darkness', ['night']], ['dusk light', DUSK], ['at dusk', DUSK], ['in direct sunlight', LIT], ['full sunlight', LIT], ['sunlit', LIT],
  ['lit by the sun', LIT], ['in sunlight', LIT], ['sunlight', [...LIT, 'golden']], ['daylight', ['day', 'golden']],
  ['in shadow', ['shadow']], ['shadow', ['shadow']], ['in deep shade', ['shadow']], ['in shade', ['shadow']], ['daytime', ['day']],
  ['broad daylight', ['day']], ['midday', ['day']]];
const COLOUR_SYN = [['gray', ['grey']], ['golden', ['yellow', 'orange']], ['gold', ['yellow', 'orange']], ['amber', ['orange']],
  ['crimson', ['red']], ['scarlet', ['red']], ['azure', ['blue']], ['navy', ['blue']], ['cyan', ['teal', 'blue']], ['turquoise', ['teal']],
  ['silver', ['grey', 'white']], ['beige', ['brown', 'white']], ['tan', ['brown', 'orange']], ['violet', ['purple']],
  ['magenta', ['pink', 'purple']]];
// weather and flight-condition words: [phrase, dim, value]; only a METAR report says cloudless, a clear preset or report says
// clear skies, a storm cell or thunderstorm is a storm-type weather cell, and a storm or storm clouds are a stormy preset or
// such a cell
const WEATHER = [['overcast', 'overcast', true], ['cloudless', 'clouds', ['NSC', 'SKC', 'NONE']], ['clear skies', 'skyclear', true],
  ['clear sky', 'skyclear', true], ['raining', 'rain', true], ['rainy', 'rain', true], ['rainfall', 'rain', true], ['rain', 'rain', true],
  ['drizzle', 'rain', true], ['storm cells?', 'storm', true], ['thunderstorm cells?', 'storm', true], ['thunderstorms?', 'storm', true],
  ['storm clouds?', 'stormcloud', true], ['thunderclouds?', 'stormcloud', true],
  ['storm', 'stormcloud', true], ['snow-capped', 'snow', true], ['snow-covered', 'snow', true], ['snowy', 'snow', true],
  ['snow', 'snow', true],
  ['stars', 'stars', true], ['starry', 'stars', true], ['stormy', 'sky', 'storm'], ['inside a tunnel', 'tunnel', true],
  ['in a tunnel', 'tunnel', true], ['through a tunnel', 'tunnel', true], ['inside cloud', 'incloud', true],
  ['inside a cloud', 'incloud', true], ['inside the cloud', 'incloud', true], ['in cloud', 'incloud', true]];
const TAGP = Object.entries(TAG_WORDS).filter(([t]) => t !== 'NIGHT' && t !== 'URBAN')
  .flatMap(([t, ps]) => ps.filter((p) => p !== 'open sea').map((p) => [p, t])).concat([['city areas', 'URBAN']]);
const pairs = (o, dim) => Object.entries(o).map(([k, p]) => [dim, k, p]);
const worlds = [...new Set(Object.values(W.world))].map((p) => ['world', Object.keys(W.world).filter((k) => W.world[k] === p), p]);
const RAW = [...pairs(W.size, 'size'), ...LIGHT.map(([p, v]) => ['light', v, p]), ...PALETTE_NAMES.map((c) => ['colour', [c], c]),
  ...COLOUR_SYN.map(([p, v]) => ['colour', v, p]), ...pairs(W.bright, 'bright'), ['bright', 'dark', 'dim'],
  ...[...EDGE_BINS].reverse().map((p) => ['texture', p, p]), ...pairs(W.vis, 'vis'), ['vis', 'haze', 'hazy'], ['vis', 'fog', 'foggy'],
  ...pairs(W.phase, 'phase').filter(([, k]) => k !== 'BREAKOUT'), ...pairs(W.vert, 'lphase'), ...W.papi.map((p, i) => ['papi', i, p]),
  ...pairs(W.band, 'band'), ...worlds, ...TAGP.map(([p, t]) => ['tag', t, p]), ...pairs(W.relief, 'tag'), ...pairs(W.clouds, 'clouds'),
  ...WEATHER.map(([p, dim, v]) => [dim, v, p]), ...Object.entries(W.sky).map(([k, p]) => ['sky', k, `${p} (?:skies|sky|weather)`]),
  ...Object.entries(W.sky).map(([k, p]) => ['sky', k, `(?:sky|skies|weather)(?: here)? (?:is|are|looks|look) ${p}`])];
// colour words may follow a hyphen ("golden-orange"); a size word is not one in "large town" (the Z caption's fixed phrase)
const pre = (dim) => (dim === 'colour' ? '(?<![\\p{L}])' : '(?<![\\p{L}-])');
const body = (p) => (p.includes('(?') || p.endsWith('?') ? p : escU(p));
export const CATS = RAW.map(([dim, value, p]) => [dim, value,
  new RegExp(`${pre(dim)}${body(p)}(?![\\p{L}-])${p === 'large' ? '(?! towns?\\b)' : ''}`, 'gu'), p.length]).sort((a, b) => b[3] - a[3]);
// gear state in its common forms: "the gear is (not) down", "with the gear up", "has its gear retracted", "the undercarriage is
// stowed", "gear-down"
const GEAR_STATE = { down: 'down', extended: 'down', lowered: 'down', deployed: 'down', up: 'up', retracted: 'up', stowed: 'up',
  raised: 'up',
  'in transit': 'transit', moving: 'transit' };
const GEAR_RE = new RegExp('\\b(?:landing gear|gear|undercarriage|wheels)(?:\\s+(?:is|are|shows|appears|looks|seems|remains|stays'
  + '|isn\'t|aren\'t|has been|have been|was|were|still|now|already|currently|fully|not))*(?:\\s+|-)'
  + '(down|up|extended|lowered|deployed|retracted|stowed|raised|in transit)\\b', 'g');
// relations that need a direction word next to their subject
const DCL = ['above', 'over', 'beyond', 'exceeds', 'higher than'];
const SIDE_RE = new RegExp('\\b(?:on|to|at|toward|towards|near) the (left|right|middle|upper|lower|top|bottom)'
  + '(?: (?:side|edge|third|part))?(?: of the (?:image|frame|picture|view))?(?! of the (?:localizer|flight band|band))\\b', 'g');
export const RELS = [['gs', /\b(?:(?:well|slightly) )?(above|below) the glideslope\b/g, (m) => m[1]],
  ['gs', /\bon (?:the )?glideslope\b(?! capture| gate)/g, () => 'on'],
  ['loc', /\b(left|right) of (?:the )?(?:localizer|course)\b/g, (m) => m[1]],
  ['loc', /\b(?:localizer|course)\b[^.]{0,50}?\bto the (left|right)\b/g, (m) => m[1]],
  ['loc', /\bon (?:the )?localizer\b(?! gate)/g, () => 'on'],
  ['cross', /\bcrosswind\b[^.;]{0,40}?\bfrom the (left|right)\b/g, (m) => m[1]], ['headtail', /\b(head|tail)wind\b/g, (m) => m[1]],
  ['dclosing', /\b(?:closing )?(faster|slower) than (?:allowed|the (?:allowed rate|limit)|its limit)\b/g,
    (m) => (m[1] === 'faster' ? 'above' : 'below')],
  ['dclosing', /\b(above|below|over|under|within|inside|beyond|exceeds|higher than|lower than) (?:the |its )?limit\b/g,
    (m) => (DCL.includes(m[1]) ? 'above' : 'below')],
  ['closing', /\b(closing in|closing fast|closing on|getting closer|approaching|approaches|converg\w*|gap shrinks)\b/g, () => 1],
  ['closing', new RegExp('\\b(receding|recedes|moving away|draws away|drawing away|drifting away|drifts away|backing off|backs off'
    + '|pulling away|opening away|getting farther away|separate at|separating|gap grows)\\b', 'g'),
    () => -1],
  ['turb', /\b(light|moderate|severe)(?:ly)? turbulen(?:ce|t)\b/g, (m) => m[1]],
  ['turb', /\bturbulen(?:ce|t)\b[^.;]{0,25}?\b(light|moderate|severe)\b/g, (m) => m[1]],
  ['stall', /\b(?:close to|near|at|past|in|into) (?:a |the )?stall\b|\bstalled\b/g, () => true],
  ['gear', GEAR_RE, (m) => GEAR_STATE[m[1]]],
  ['side', SIDE_RE, (m) => ({ middle: 'centre', upper: 'top', lower: 'bottom' })[m[1]] || m[1]]];
// a kind word is a hazard only as a noun: not inside a compound ("bird's-eye", "rock-strewn") or before another noun that it
// modifies ("rock pillars", "a rock face", "satellite imagery")
const HEADS = 'pillars?|faces?|walls?|fields?|formations?|spires?|outcrops?|surfaces?|slabs?|arch(?:es)?|towers?|columns?|gardens?|bands?'
  + '|belts?|layers?|ledges?|eye|views?|images?|imagery|photos?|maps?|dish(?:es)?|links?|tails?|strikes?|songs?|nests?|debris|dust';
export const KIND_END = `s?(?![\\p{L}'’-])(?!\\s+(?:${HEADS})\\b)`;
// hazard kinds named with an article are claims about the frame ("there is a comet in the frame")
export const KIND_RE = new RegExp('\\b(?:a|an|the|another|lone|that|this)\\s+(?:[a-z-]+\\s+){0,2}?'
  + `(rock|comet|satellite|airliner|flocks? of birds|flock|bird)${KIND_END}`, 'gu');
export const kindOf = (w) => (/^flock|^bird/.test(w) ? 'birds' : w);
// the objects a position claim (o'clock, side, region, compass) or a closing relation can be about
export const SUBJECTS = [[new RegExp(`\\b(rock|comet|satellite|airliner|flock|bird)${KIND_END}`, 'gu'),
  (m) => ({ kind: 'hazard', value: kindOf(m[1]) })],
  [/\b(?:hazards?|nearest one|closest one)\b/g, () => ({ kind: 'hazard', value: null })],
  [/\b(moon|earth|sun)\b/g, (m) => ({ kind: 'body', value: m[1] })],
  [/\b(station|port|runway|windsock|sock|airfield)\b/g, (m) => ({ kind: 'object', value: m[1] })],
  [/\b(sea|water|ocean|coast|coastline)\b/g, () => ({ kind: 'sea' })]];

// presence of a body or an object in the frame: a clause with one of these nouns and a presence cue ("the Earth is visible",
// "also in view are the Earth and the Moon", "the chase camera shows the station", "the Moon is not in view")
export const PRESENCE_NOUN = /\b(earth|moon|station|docking port|runway|windsock|papi)\b/g;
export const PRESENCE_CUE = new RegExp('\\b(?:visible|in view|in the frame|in the picture|in the image|in sight|appears?|shows?|showing'
  + '|includes?|contains?)\\b');

// ---- the common-word lexicon ----
const WORD_LIST = (s) => s.trim().split(/\s+/);
// words that never make an unknown entity mid-sentence
export const STOP = new Set(['I', 'Context', 'Earth', 'Moon', 'Sun', 'V-bar']);
// words that may open a sentence (or follow ":" / ";"): never read as a place even when a gazetteer holds them (Natural
// Earth has Split, Point, Orange, Wind...). Every word that opens a sentence in the bank or a slot value is here (a test
// checks it).
const NUMBER_WORDS = ['zero', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen',
  'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty'];
export const SENTENCE_WORDS = new Set([...PALETTE_NAMES, 'gray', 'tan', ...NUMBER_WORDS, ...WORD_LIST(`
  a about above according across actions advice aerial after again against ahead air airspeed all along already also although always am
  an and another any anything are around as at away back be because been before behind being below beside best between beyond both but
  by can cannot chase chase-camera check clear clearance close closest closing cloud clouds cloudy colour colours colour-wise compare
  components conditions confirm contact correct could count counting crosswind current currently day daylight darkness deployed
  describe descending did direct distance do docking does down driven during dusk each either else estimate even every everything
  expect explain failed failures fair falling far few flight flying fog following for from fuel further g-load gates gear give given
  going good ground had has have haze hazard hazards he head headroom headwind her here high highly hilly his holding how however i
  identify if imagery in inside instead into is it its judging just landing large laterally least left less level light like likely
  little look looking low lower many margin may maybe meanwhile measured medium medium-sized might minimum moderate monitor more
  moreover most mostly mountain mountainous much must my name near nearby nearest neither never next no none nor not note nothing now
  of off on once one only onto open or other otherwise our out outside over overall overhead overloading part papi per perhaps phase
  pitch point possibly problem quite rain range rate rather read reason receding recommended remaining resolution right rolling roughly
  route running sea seen severe several shadow she should since skies slightly small smooth so some something space speed speed-wise
  split spoilers stabilized stall still stopped stormy such sunlight tailwind taking tell terrain than that the their them then there
  these they this those though through thus time tiny to today together tonally too toward towards turbulence under unless unlike until
  up upon upper using vertically very visibility visible was water we well were what when where whether which while whose why wind with
  within without would yes yet you your city climbing textured transit scattered broken rock comet satellite airliner flock beige cyan
  navy bright coastline dark desert flat hills ice mountains night nose-down nose-up verdict beneath underneath amid among amongst
  besides despite via past front rear back apart plenty lots whole entire half certain various different same snow-capped snow sand
  trees fields forests roads rivers lakes streets houses buildings stars lights shadows land cloudless hazy foggy rainy windy sunny
  calm first second third finally lastly later earlier tonight furthermore hence therefore likewise similarly altogether additionally
  again visibly beyond straight shown viewed pictured captured taken heading approaching`)]);
// common English verbs, adjectives and irregular participles that open descriptions and instructions ("Watch the rock",
// "Keep the runway centred", "Lit by the Sun, ..."); words ending in -ing, -ed or -ly, and hyphenated compounds of common
// words ("Built-up"), count as common too
const COMMON_EN = new Set(WORD_LIST(`
  watch keep notice stay mind beware spot see note hold turn fly follow avoid consider imagine picture observe find let make take bring
  put set lit seen shown drawn cast held hung spun flown built lain made done gone kept found left sat stood caught led met dead alive
  visible invisible bright dim empty full open closed wide narrow deep shallow steep gentle broad thin thick heavy light centred
  centered dead-ahead framed bathed hanging floating rising glinting drifting sitting silhouetted big small large little long short
  tall high low near far close distant quiet busy sharp soft hard faint pale deep sunlit ahead aside apart abreast aloft astern alone
  alongside everywhere somewhere nowhere anywhere outward inward upward downward meanwhile indeed perhaps certainly clearly directly
  just only even still yet also too`));
// the nouns a description of these scenes opens with ("Runway", "Debris", "Thrusters", "Glaciers"), beyond the bank's own words
const SCENE_NOUNS = new Set(WORD_LIST(`
  debris glare haze drizzle mist gloom murk dusk dawn sunrise sunset horizon skyline landscape scenery vista backdrop foreground
  background silhouette blackness void starfield tarmac asphalt concrete grass taxiway apron hangar terminal threshold beacon marking
  centreline centerline cloudbank thundercloud lightning thunder sleet hail frost fog smoke steam vapour vapor dust sand gravel rubble
  boulder crag cliff ridge valley canyon gorge plateau plain meadow dune shore shoreline beach bay harbour harbor cape delta lagoon reef
  marsh swamp glacier icefield farmland field forest woodland orchard vineyard village suburb street avenue boulevard block skyscraper
  rooftop bridge highway railway park solar module panel hull truss antenna radiator hatch arm cockpit fuselage wingtip engine cabin
  relief summit crest slope hillside peak spire`));
// a common word: in the lexicon or the bank's vocabulary, a plural of one, -ing/-ed/-ly, or a hyphenated compound of them
const singular = (w) => [w.replace(/ies$/, 'y'), w.replace(/(?:es)$/, ''), w.replace(/s$/, '')];
const knownWord = (w) => COMMON_LOWER.has(w) || SCENE_NOUNS.has(w);
export const isCommonOpener = (low) => knownWord(low) || singular(low).some(knownWord) || /^\p{L}{3,}(?:ing|ed|ly)$/u.test(low)
  || (/^\p{L}+(?:-\p{L}+)+$/u.test(low) && low.split('-').every(isCommonOpener));
// pronouns and determiners are never names, whatever follows them ("It lies on the right")
export const PRONOUNS = new Set(WORD_LIST(`
  it its this that these those there here which what who one each both either neither another nothing everything something the a an
  they he she we you i`));
// a capitalised word followed by one of these is used as a name: a gazetteer name with any of PROPER_FRAME ("Split lies on
// the coast"), any other word only with a noun of NAME_FRAME ("Bright town")
export const NAME_FRAME = /^\s*(?:town|city|village|river|lake|island|province|county)\b/i;
export const PROPER_FRAME = new RegExp('^\\s*(?:lies|lay|is located|is situated|is a (?:large |small )?(?:town|city|village)'
  + '|town|city|river|lake|island|province|county|region)\\b', 'i');
// lower-case words text uses that are not places even when a gazetteer holds them (the bank's vocabulary; a test keeps it
// whole)
const BANK_VOCAB = WORD_LIST(`
  abort aborted about above accident according account accounts across act acting action actions actually administrative advice advise
  advises aerial after against ahead air aircraft airfield airframe airliner airliners airspeed aligned alignment all allowed allows
  alone along also altitude am among an and angle angle-of-attack answer any anywhere apparent appear appears applies approach
  approaches approaching are area areas around arrives as at attached attack attitude available avoid away axis band bar be because
  been before behind being belong belongs below beneath beside best between beyond big biggest birds black blowing blows blue both
  boundary bounds break breaking breakout bright brightness brings broken brown building bumps bumpy but by call calls camera can
  cannot capture captured captures carries carry catches cause cautionary ceiling cell cells cent centred certain change changes chase
  chase-camera check chiefly cites citing city class clear clearance climbing clock close closed closer closes closest closest-approach
  closing closing-rate cloud clouds cloudy coast coastline colour colour-wise colours come comes comet comets coming compare compared
  completes components concern conditions cone configured confirm contact contain contains continue continues controls converge
  converging correct correctly corridor could count counting country couple course cover covered covers crash crashes crew criteria
  cross crosses crossing crosswind current currently danger dangerous dark darkness day daylight decreasing degree degrees departure
  descending describe desert detail detailed deviation direct direction directly distance distant do docking docks does doing dominant
  dominate done dot dots down downwards draws driven driving drops due during dusk each earth edge enclosed encloses end ends enough
  entirely enveloped error estimate estimates every everything exceed exceeds exist exists expect expects explain extended face factor
  failed failing fails failure failures fair falling falls far farther fast faster feature feel feeling feels feet few field fills
  final find finds fine first first-level flagged flags flare flat flies flight flock flocks flown fly flying fog followed following
  follows for forward fraction frame from fuel full further g-load gap gate gates gear gear-down get getting give given gives glide
  glideslope go go-around going good greatest green grey ground grows hand happen happens hard has have hazard hazards haze head
  headroom headwind height held here hidden high higher hills hilly hold holding holds how however ice identify if image imaged imagery
  in include includes increases indicate indicates indication inside instead intervene intervenes intervention into involve is it its
  itself jet jets judges judging just keep kilometres kind kinds knots laid land landing lands landscape large largely lateral
  laterally layer lead leading led left less level lie lies light lighting lights like likely limit limits line lined listed lists lit
  load loaded localizer located long longer look looking looks low lower lowered lying main mainly make makes many margin matters
  maximum may me mean meaning means measured measures medium medium-sized meet meets met metres middle miles minimum miss misses mixes
  mode moderate moment monitor moon more most mostly mountain mountainous mountains move moves moving much must name named namely names
  naming near nearby nearest needed neither never-exceed next night no none nor normal nose nose-down nose-up not note nothing now
  number object obscured obstacle obstacles occurred ocean of off on one only onto open opening opens option or orange oriented our out
  outcome outside over overall overcast overhead overloading overruns overspeed owing palette papi part partway pass passes passing
  passively past path peaks per percent permitted phase picture pillar-shaped pillars pilot pink pitch pitched pixel place placed
  places placing plan planet play plus point pointing points port portion pose poses position positions possible prediction predicts
  premise present preset primary problem propellant province pulling pulls purple puts putting quick quickly raised range rate rated
  rates rather rating reach reaches reaching reaction read reading reads really reason reasons recedes receding reckons recommended
  recommends red region relative relief remaining remains rendezvous report reports resolution resolves respect respond response result
  reveals ride right rise risk risky rock rocks rolling rooftop room rough roughly route routes rugged ruled ruling running runs runway
  safe safely safety sample satellite satellites satisfied say says scattered scene scenery scheme screen sea second seconds section
  see seen sees separate separates separation sequence serious service severe shadow share sharp shining ship short should show showing
  shown shows shrinking shrinks side sight sign significant sink sit sits situation size skies sky skyline slightly slow slower small
  smooth so sock something soon sort source space spacecraft spaceport span spans specific speed speed-wise split spoilers spot
  stabilization stabilized stabilized-approach stable stage stall stalling stalls stand stands state stated station stay stayed stays
  steep still stopped storm stormy streams stretch strikes strong suggest suggests sun sunlight sunlit surface surrounded surrounds
  tail tailwind take taken takes taking tall tanks teal tell telling terms terrain than that the their them then there these they
  things third this those though threat through thruster thrusters thunderstorm tight time time-to-contact tiny to together tonally
  tone tones too top total touches toward towards towering town tracking tracks trailing trails transfer travelling travels true tunnel
  turbulence turbulent twelve twilight two type types unchanged under undercarriage unless unmet unsafe until up upper upwards us using
  value verdict vertical vertically view viewed viewing viewpoint views visibility visible was water way we weather well wet what when
  where which while white whole whose why will wind windsock wing wings with within without working works world worried worry would
  wrong yellow yes yet you`);
export const COMMON_LOWER = new Set([...SENTENCE_WORDS, ...COMMON_EN, ...BANK_VOCAB, ...WORD_LIST(`north north-east east south-east
  south south-west west north-west northeast northwest southeast southwest northern southern eastern western centre center middle upper
  lower top bottom left right`)]);
