// chat/tests/retrieval_parity.test.mjs — chat/web/retriever.js vs chat/retrieval.py on the same facts and queries: same tokens,
// same top-k ids, scores equal to 1e-9 relative (Math.log and libm log may differ in the last ulp). The fixture always runs;
// CAPCOM_KB=<kb.json> adds the real knowledge base and CAPCOM_QUERIES=<questions.jsonl> its questions.
//   node --test chat/tests/retrieval_parity.test.mjs   (from the repo root; PYTHON overrides python3)
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { BM25, tokens } from '../web/retriever.js';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const FIXTURE = [
  { id: 'RL-001', title: 'PPO pilot', text: 'The Astro Pilot spacecraft is flown by a PPO policy trained in the browser with TF.js.', keywords: ['ppo', 'reinforcement learning', 'policy'] },
  { id: 'RL-002', title: 'Body-rate actions', text: 'The policy commands body rates at 15 Hz; a fly-by-wire controller tracks them. Torque actions failed.', keywords: ['actions', 'rates', 'fly-by-wire'] },
  { id: 'HOWTO-001', title: 'Landing scenario', text: 'Open ?scenario=landing to watch an airliner autoland in wind, rain or fog.', keywords: ['landing', 'autoland', 'airliner'] },
  { id: 'HOWTO-002', title: 'Zoom in', text: "Press Z or the Playbox 'Zoom in' button to use the Earth-zoom telescope.", keywords: ['zoom', 'telescope', 'earth'] },
  { id: 'ORBIT-001', title: 'Orbital speed', text: 'At 420 km the station orbits Earth at about 27,581 km/h, one lap every ~93 minutes.', keywords: ['orbit', 'speed', 'iss'] },
  { id: 'SPACE-001', title: "Comets' tails", text: "A comet's tails point away from the Sun, pushed by sunlight and the solar wind.", keywords: ['comet', 'tail', 'sun'] },
];
const QUERIES = [['what is ppo?', ''], ['how does it fly', 'what is ppo?'], ['and how fast is it?', 'tell me about the orbit'], ['How do I land a plane??', ''],
  ["Where's the telescope", ''], ['hello', ''], ['COMETS tails direction', ''], ['torques vs rates', 'why body rates'], ['', ''], ['zzz qqq', '']];

function python(facts, queries) {
  const script = `import json,sys\nsys.path.insert(0, ${JSON.stringify(ROOT)})\nfrom chat.retrieval import BM25, tokens\nd=json.load(sys.stdin)\nb=BM25(d['facts'])\n` +
    `print(json.dumps({'tok':[tokens(q) for q,_ in d['queries']],'res':[b.scores(q,p)[:5] for q,p in d['queries']],'top':[b.search(q,p) for q,p in d['queries']]}))`;
  return JSON.parse(execFileSync(process.env.PYTHON || 'python3', ['-c', script], { input: JSON.stringify({ facts, queries }), maxBuffer: 1 << 28 }).toString());
}

function check(facts, queries) {
  const py = python(facts, queries), b = new BM25(facts);
  queries.forEach(([q, p], i) => {
    assert.deepEqual(tokens(q), py.tok[i], `tokens of ${JSON.stringify(q)}`);
    const js = b.scores(q, p).slice(0, 5);
    assert.deepEqual(js.map((x) => x[1]), py.res[i].map((x) => x[1]), `ranking of ${JSON.stringify(q)}`);
    js.forEach(([s], j) => assert.ok(Math.abs(s - py.res[i][j][0]) <= 1e-9 * Math.max(1, Math.abs(s)), `score ${j} of ${JSON.stringify(q)}`));
    assert.deepEqual(b.search(q, p).map((x) => x[0]), py.top[i].map((x) => x[0]), `top-k of ${JSON.stringify(q)}`);
  });
}

test('fixture: tokens, rankings, scores and top-k match Python', () => check(FIXTURE, QUERIES));
test('fixture: follow-ups borrow the previous message', () => {
  const b = new BM25(FIXTURE);
  // MIN_SCORE is calibrated on the real KB (chat/kb/build_kb.py); six facts give tiny idf, so no threshold here
  assert.equal(b.search('and how fast is it?', 'tell me about the orbit', 3, 0)[0][0], 'ORBIT-001');
  assert.deepEqual(b.search('hello'), []);
});
test('real KB (CAPCOM_KB) matches Python', { skip: !process.env.CAPCOM_KB }, () => {
  const facts = JSON.parse(fs.readFileSync(process.env.CAPCOM_KB, 'utf8')).facts;
  const qs = process.env.CAPCOM_QUERIES ? fs.readFileSync(process.env.CAPCOM_QUERIES, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l).q) : [];
  const queries = [...QUERIES, ...qs.slice(0, 400).map((q, i) => [q, i ? qs[i - 1] : ''])];
  check(facts, queries);
});
