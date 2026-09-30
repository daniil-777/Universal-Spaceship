// stdin {text, record, context, names, obs_rule} -> stdout {verified, errors}; used by vlm/train/narrator/check_parity.py and
// slot_eval. obs_rule applies the §5.7 Narrator-row rule (claims must rest on visual facts or on what the Context supplies).
import { verifyFreeText, makeGazetteer, BASE_NAMES } from './verify.js';

let buf = ''; for await (const c of process.stdin) buf += c;
const { text, record, context = null, names = [], obs_rule: obsRule = false } = JSON.parse(buf), r = verifyFreeText(text, record, { gaz: makeGazetteer([...BASE_NAMES, ...names]), context, obsRule });
process.stdout.write(JSON.stringify({ verified: r.verified, errors: r.errors }) + '\n');
