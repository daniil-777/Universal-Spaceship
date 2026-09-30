// vlm/gen/geo/fetch.mjs — one-off Natural Earth download to LaCie, verified against sources.json (bytes always; sha256
// once pinned). node vlm/gen/geo/fetch.mjs [--pin]   (--pin writes the sha256 of the first verified download)
import fs from 'node:fs';
import crypto from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const SRC = fileURLToPath(new URL('./sources.json', import.meta.url)), S = JSON.parse(fs.readFileSync(SRC, 'utf8')), pin = process.argv.includes('--pin');
fs.mkdirSync(S.dir, { recursive: true });
let bad = 0;
for (const L of S.layers) {
  const file = path.join(S.dir, `${L.name}.geojson`);
  let buf = fs.existsSync(file) ? fs.readFileSync(file) : null;
  if (!buf || buf.length !== L.bytes) { const r = await fetch(`${S.base}${L.name}.geojson`); if (!r.ok) throw new Error(`${L.name}: HTTP ${r.status}`); buf = Buffer.from(await r.arrayBuffer()); fs.writeFileSync(file, buf); }
  const h = crypto.createHash('sha256').update(buf).digest('hex');
  if (buf.length !== L.bytes) { console.log(`BAD ${L.name}: ${buf.length} bytes, expected ${L.bytes}`); bad++; continue; }
  if (L.sha256 === null && pin) L.sha256 = h;
  if (L.sha256 !== h) { console.log(`BAD ${L.name}: sha256 ${h} != ${L.sha256}`); bad++; continue; }
  console.log(`ok ${L.name} ${L.bytes} ${h.slice(0, 12)}`);
}
if (pin) fs.writeFileSync(SRC, JSON.stringify(S, null, 1) + '\n');
process.exit(bad ? 1 : 0);
