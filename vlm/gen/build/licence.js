// vlm/gen/build/licence.js — the dataset's licence (spec R2): one profile across every kept record (a mixed build is refused,
// and an explicit --licence must match), ATTRIBUTION.txt from the layers the records actually served, and the oof Context
// file's checks (spec §5.7: a train row's Context comes from the cross-fit model that did not train on its group).
import zlib from 'node:zlib';
import { LAYERS } from '../schema.js';

export function licenceOf(records, want = null) {
  if (!records.length) throw new Error('licence: no records');
  const profs = [...new Set(records.map((r) => r.render && r.render.licence_profile))].sort();
  if (profs.length !== 1) throw new Error(`licence profiles ${profs.join(', ')} in one build: build each profile on its own (an nc record makes the whole dataset NC, spec R2)`);
  if (want && want !== profs[0]) throw new Error(`--licence ${want}, but every record is ${profs[0]}`);
  return profs[0];
}

const EOX_LINES = {
  s2cloudless_3857: 'EOxCloudless https://cloudless.eox.at by EOX IT Services GmbH (Contains modified Copernicus Sentinel data 2016 & 2017), CC BY 4.0',
  's2cloudless-2025_3857': 'Sentinel-2 cloudless 2025 https://s2maps.eu by EOX IT Services GmbH (Contains modified Copernicus Sentinel data 2025), CC BY-NC-SA 4.0',
};
const PROFILE_EOX = { open: 's2cloudless_3857', nc: 's2cloudless-2025_3857' };

export function attributionOf(records, profile) {
  const served = new Set(records.flatMap((r) => ((r.render && r.render.imagery) || []).map((m) => m.layer_id)));
  served.add(PROFILE_EOX[profile]);
  const eox = Object.keys(EOX_LINES).filter((l) => served.has(l) && LAYERS[l]).map((l) => EOX_LINES[l]);
  const nc = served.has('s2cloudless-2025_3857') ? ['This dataset contains CC BY-NC-SA 4.0 imagery: it is for non-commercial use only, shared alike, and models trained on it are treated as NC (spec R2).'] : [];
  return [...nc, ...eox, 'Blue Marble: NASA Earth Observatory (GIBS), public domain',
    'AWS Terrain Tiles: SRTM, GMTED2010, ETOPO1 and others (see https://github.com/tilezen/joerd/blob/master/docs/attribution.md)', 'Made with Natural Earth. Free vector and raster map data @ naturalearthdata.com',
    'NASA textures: see textures/CREDITS.txt in the Astro Pilot repository', 'Meshy models (CC BY 4.0 credit to Meshy if made on the free plan; the user confirms the plan)'].join('\n') + '\n';
}

// text: evaluate.py --write-preds lines {key, fold, monitor}; groupOf(key) and splitOf(key) are the build's own
export function oofPreds(text, { groupOf, splitOf }) {
  const out = new Map();
  for (const l of text.split('\n')) {
    if (!l.trim()) continue;
    const p = JSON.parse(l);
    if (out.has(p.key)) throw new Error(`--context-from: duplicate prediction for ${p.key} (one model per row)`);
    if (splitOf(p.key) === 'train') {
      if (p.fold !== 0 && p.fold !== 1) throw new Error(`--context-from: train row ${p.key} has no fold (a cross-fit checkpoint writes it)`);
      const g = groupOf(p.key);
      if (g && zlib.crc32(g) % 2 === p.fold) throw new Error(`--context-from: train row ${p.key} was predicted by the fold-${p.fold} model, which trained on its group ${g}`);
    }
    out.set(p.key, p.monitor);
  }
  return out;
}
