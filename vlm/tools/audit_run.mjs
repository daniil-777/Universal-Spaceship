// node vlm/tools/audit_run.mjs --run <run> [--sizes sizes.json] [--dataset <name>]: 0 Esri upstream fetches, 0 aborted tile-host
// requests, EOX upstream <= 80,000, raw PNGs <= 10 GB, wall-clock <= 12 h; the records per family against the §3.2 sizes are always
// reported (ruling T3H-1: v0 reports them, the size rule is not applied); with --sizes the §11.3 size rule (each family's captured
// records, twins included, >= 50 % of §3.2 and within +-10 % of its recorded target); with --dataset the cell and text gates.
import fs from 'node:fs';
import path from 'node:path';
const arg = (k) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : null; }, L = '/Volumes/LaCie/astro-pilot/vlm', run = arg('run'), bad = [];
const SPEC = { Z: 800, S: 1200, A: 1600, L: 800, D: 600 }, TILE_HOSTS = ['tiles.maps.eox.at', 'gibs.earthdata.nasa.gov', 's3.amazonaws.com'], sizes = {};
let esri = 0, aborted = 0, eox = 0, bytes = 0, wall = 0;
for (const f of ['Z', 'S', 'A', 'L', 'D']) {
  const d = path.join(L, 'raw', run, f); if (!fs.existsSync(d)) continue;
  const names = fs.readdirSync(d).filter((n) => !n.startsWith('._'));
  sizes[f] = { records: names.filter((x) => /^[SALDZ]_.*\.json$/.test(x)).length, spec: SPEC[f] }; sizes[f].frac_of_spec = +(sizes[f].records / SPEC[f]).toFixed(3);
  for (const n of names) {
    const p = path.join(d, n);
    if (n.endsWith('.png')) bytes += fs.statSync(p).size;
    if (n.startsWith('ledger_')) for (const l of fs.readFileSync(p, 'utf8').split('\n').filter(Boolean)) {
      const e = JSON.parse(l), up = e.via === 'fetch' && e.served ? new URL(e.served).host : null;
      if ((e.host === 'server.arcgisonline.com' && e.outcome === 'upstream fetch') || up === 'server.arcgisonline.com') esri++;
      if (TILE_HOSTS.includes(e.host) && e.outcome === 'aborted') aborted++;
      if (up === 'tiles.maps.eox.at') eox++;
    }
    if (n === 'throughput.jsonl') for (const l of fs.readFileSync(p, 'utf8').split('\n').filter(Boolean)) wall += JSON.parse(l).wall_s;
  }
}
if (esri) bad.push(`${esri} Esri upstream fetches`); if (aborted) bad.push(`${aborted} aborted tile-host requests`); if (eox > 80000) bad.push(`${eox} EOX upstream requests > 80,000`);
if (bytes > 10e9) bad.push(`raw PNGs ${(bytes / 1e9).toFixed(1)} GB > 10 GB`); if (wall > 12 * 3600) bad.push(`capture wall-clock ${(wall / 3600).toFixed(1)} h > 12 h`);
if (arg('sizes')) {
  const want = JSON.parse(fs.readFileSync(arg('sizes'), 'utf8')).sizes;
  for (const f of Object.keys(SPEC)) { const n = sizes[f] ? sizes[f].records : 0; if (n < SPEC[f] / 2) bad.push(`${f}: ${n} records < 50 % of the §3.2 size ${SPEC[f]}`); if (Math.abs(n / want[f] - 1) > 0.1) bad.push(`${f}: ${n} records not within 10 % of the recorded target ${want[f]}`); }
}
if (arg('dataset')) {
  const s = JSON.parse(fs.readFileSync(path.join(L, 'datasets', arg('dataset'), 'stats.json'), 'utf8'));
  for (const f of ['S', 'A', 'L', 'D']) for (const v of ['SAFE', 'CAUTION', 'UNSAFE']) if ((s.counts[`${f}|train|${v}`] || 0) < 50) bad.push(`train cell ${f}|${v} has ${s.counts[`${f}|train|${v}`] || 0} < 50 clips`);
  if (s.text.rejectRate >= 0.05) bad.push(`text reject rate ${s.text.rejectRate}`); if (s.text.parserFalseReject >= 0.05) bad.push(`parser false-reject ${s.text.parserFalseReject}`);
}
console.log(JSON.stringify({ run, esri, aborted, eox, raw_gb: +(bytes / 1e9).toFixed(2), wall_h: +(wall / 3600).toFixed(2), sizes, bad })); process.exit(bad.length ? 1 : 0);
