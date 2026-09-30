// vlm/capture/budget.mjs — the per-run EOX budget (spec §6: 80,000 per run, across families, lanes and resumes). Each lane
// owns its family directories, so each keeps its own count in raw/<run>/<F>/eox_used.json and the run total is their sum
// (plus eox_legacy.json: a total recorded before per-family counts existed). No lane ever overwrites another lane's count;
// raw/<run>/eox_budget.json is the summed total, rewritten on every save, for reports and the datasheet.
import fs from 'node:fs';
import path from 'node:path';

const FAMS = ['S', 'A', 'L', 'D', 'Z'];
const readUsed = (f, k = 'used') => { try { return +JSON.parse(fs.readFileSync(f, 'utf8'))[k] || 0; } catch { return 0; } };
export function runBudget(runDir, family, { maxAgeMs = 1000 } = {}) {
  const legacyFile = path.join(runDir, 'eox_legacy.json'), totalFile = path.join(runDir, 'eox_budget.json'), own = (f) => path.join(runDir, f, 'eox_used.json');
  if (!fs.existsSync(legacyFile)) {
    let legacy = 0; try { const b = JSON.parse(fs.readFileSync(totalFile, 'utf8')); if (!b.by_family) legacy = +b.used || 0; } catch { /* no total yet */ }
    fs.mkdirSync(runDir, { recursive: true }); fs.writeFileSync(legacyFile, JSON.stringify({ used: legacy }));
  }
  const legacy = readUsed(legacyFile), counts = () => Object.fromEntries(FAMS.map((f) => [f, readUsed(own(f))]));
  let cache = null, at = 0;
  // the other families' counts plus the legacy total, re-read at most once per maxAgeMs (the cap check runs per request)
  const others = () => { if (!cache || Date.now() - at > maxAgeMs) { const c = counts(); cache = legacy + FAMS.filter((f) => f !== family).reduce((s, f) => s + c[f], 0); at = Date.now(); } return cache; };
  function save(used) {
    fs.mkdirSync(path.join(runDir, family), { recursive: true }); fs.writeFileSync(own(family), JSON.stringify({ used }));
    const c = counts(); cache = null;
    fs.writeFileSync(totalFile, JSON.stringify({ used: legacy + Object.values(c).reduce((s, x) => s + x, 0), by_family: c, legacy }));
  }
  return { own0: readUsed(own(family)), others, save, total: (used) => used + others() };
}
