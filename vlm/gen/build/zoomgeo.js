// vlm/gen/build/zoomgeo.js — Z geo facts at build time (spec §4.2 Z; Natural Earth and cached Terrarium only, never a fetch):
// sea/land/lake per grid point, elevation stats over LAND points only (ruling T6-a; sea points feed sea_frac and coast_side
// only), water_frac = NE sea + NE lakes (WATER_DOMINANT uses it, sea_frac stays sea only), country/admin-1 fractions, the
// features in view with their 3x3 regions (most salient first), the nearest place outside the view, admin-1 at the view
// centre, the zoom tags, and geo.terrain in schema.js TAG_WORDS (the same phrase the captions use, items.js terrainPhrase).
import path from 'node:path';
import { loadNaturalEarth, featuresAt, isLand, featuresInView, nearestPlace, gazetteerNames, nameOf } from '../geo/naturalearth.js';
import { createElevationReader, elevStats } from '../geo/terrarium.js';
import { openTileCache } from '../../capture/tilecache.mjs';
import { zoomTags, GRID_NX } from '../labels/zoom.js';
import { heightLevel } from '../../../src/earthtiles.js';
import { fact } from '../schema.js';
import { terrainPhrase } from '../text/items.js';
import { queryOf } from './split.js';

const LACIE = '/Volumes/LaCie/astro-pilot/vlm', CELL_PX = 56;
const rankOf = (x) => { const p = x.feature ? x.feature.props : {}; return p.SCALERANK ?? p.scalerank ?? 20; };
const salience = (a, b) => rankOf(a) - rankOf(b) || (b.pop ?? 0) - (a.pop ?? 0) || String(a.name).localeCompare(String(b.name));
// NE writes about 300 region labels in capitals ("NEW GUINEA HIGHLANDS"), which text would read as acronyms: they are
// title-cased for place.in_view and the gazetteer alike; short all-caps names (USA, UAE) are kept
export const neName = (s) => (typeof s === 'string' && s.length > 3 && s === s.toUpperCase() && /\p{Lu}{2}/u.test(s) ? s.toLowerCase().replace(/(^|[\s(/-])(\p{L})/gu, (m, a, b) => a + b.toUpperCase()) : s);
const r1 = (x) => (x === null || x === undefined ? null : +x.toFixed(1));
function originOf(r) {
  const q = queryOf(r), lat = parseFloat(q.get('lat')), lon = parseFloat(q.get('lon'));
  return Number.isFinite(lat) && Number.isFinite(lon) ? { lat, lon } : { lat: r.facts['view.lat_deg'].v, lon: r.facts['view.lon_deg'].v };
}
export const zoomGeo = {
  async load({ geoDir = path.join(LACIE, 'geo'), tileDir = path.join(LACIE, 'tilecache') } = {}) {
    const ne = await loadNaturalEarth(geoDir);
    return { ne, names: gazetteerNames(ne).map(neName), elev: createElevationReader(openTileCache(tileDir, 's3.amazonaws.com')) };
  },
  async apply(r, g) {
    const ne = g.ne, put = (id, v, unit = null) => { r.facts[id] = fact(v, unit, 'visual'); }, grid = r.facts['grid.latlon'].v, P = new Array(grid.length).fill(null);
    // §4.2: each point reads Terrarium at its ring's height level: the finest ring whose height tile was cached covers it
    const rings = [...(r.facts['view.rings'].v || [])].sort((a, b) => b - a), countries = {}, admin1 = {};
    const elevAt = async (lat, lon) => { for (const lv of rings) { const x = await g.elev.at(lat, lon, heightLevel(lv)); if (x !== null && x !== undefined) return x; } return null; };
    for (let k = 0; k < grid.length; k++) {
      if (!grid[k]) continue;
      const [lat, lon] = grid[k], sea = !isLand(ne, lon, lat), lake = !sea && featuresAt(ne, 'ne_10m_lakes', lon, lat).length > 0;
      P[k] = { lat, lon, sea, lake, elev: await elevAt(lat, lon), px: ((k % GRID_NX) + 0.5) * CELL_PX, py: (Math.floor(k / GRID_NX) + 0.5) * CELL_PX, coastRefine: true };
      const c = featuresAt(ne, 'ne_10m_admin_0_countries', lon, lat)[0]; if (c) countries[nameOf(c.props)] = (countries[nameOf(c.props)] || 0) + 1;
      const a = featuresAt(ne, 'ne_10m_admin_1_states_provinces', lon, lat)[0]; if (a && nameOf(a.props)) admin1[nameOf(a.props)] = (admin1[nameOf(a.props)] || 0) + 1;
    }
    const pts = P.filter(Boolean), n = Math.max(1, pts.length), all = elevStats(pts), land = elevStats(pts.filter((p) => !p.sea)), water = pts.filter((p) => p.sea || p.lake).length / n;
    const frac = (m) => Object.fromEntries(Object.entries(m).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).map(([k, c]) => [k, +(c / n).toFixed(3)]));
    const origin = originOf(r), cam = r.cameras[path.basename(r.frames[0])], view = featuresInView(ne, cam, origin), inView = new Set(view.map((x) => x.feature));
    put('geo.sea_frac', +all.sea_frac.toFixed(3)); put('geo.water_frac', +water.toFixed(3)); put('geo.coast_side', all.coast_side);
    put('geo.elev_min_m', r1(land.min), 'm'); put('geo.elev_max_m', r1(land.max), 'm'); put('geo.elev_mean_m', r1(land.mean), 'm'); put('geo.relief_m', r1(land.relief), 'm'); put('geo.elev_points', land.n);
    // place.country: the admin-0 country at the view centre, like place.admin1 (the bank's country templates state it); the
    // plurality over the grid only when the centre is sea, and always in geo.country_frac (review item 2)
    const centreCountry = featuresAt(ne, 'ne_10m_admin_0_countries', origin.lon, origin.lat)[0];
    put('place.country', centreCountry ? nameOf(centreCountry.props) : Object.entries(countries).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))[0]?.[0] ?? null); put('geo.country_frac', frac(countries)); put('geo.admin1_frac', frac(admin1));
    const centreAdmin = featuresAt(ne, 'ne_10m_admin_1_states_provinces', origin.lon, origin.lat)[0]; put('place.admin1', centreAdmin ? nameOf(centreAdmin.props) ?? null : null);
    put('place.in_view', view.filter((x) => typeof x.name === 'string' && x.name).sort(salience).map((x) => ({ name: neName(x.name), kind: x.kind, region: x.region })));
    const np = nearestPlace(ne, origin.lat, origin.lon, { exclude: inView });
    put('place.nearest', np ? { name: np.name, km: +np.km.toFixed(1), bearing: +np.bearing.toFixed(1), compass: np.compass } : null);
    const cellRelief = (thr) => {
      let cells = 0, hit = 0;
      for (let j = 0; j + 1 < grid.length / GRID_NX; j++) for (let i = 0; i + 1 < GRID_NX; i++) {
        const q = [P[j * GRID_NX + i], P[j * GRID_NX + i + 1], P[(j + 1) * GRID_NX + i], P[(j + 1) * GRID_NX + i + 1]].filter((p) => p && p.elev !== null && !p.sea);
        if (q.length < 4) continue; cells++; if (Math.max(...q.map((p) => p.elev)) - Math.min(...q.map((p) => p.elev)) > thr) hit++;
      }
      return cells ? hit / cells : 0;
    };
    const rangeKm = r.facts['view.range_km'].v, urban = rangeKm < 150 && view.some((x) => x.kind === 'place' && (x.pop ?? 0) >= 100000 && x.px > 224 && x.px < 672 && x.py > 126 && x.py < 378);
    const tags = zoomTags({ geo: { sea_frac: all.sea_frac, relief: land.relief, mean: land.mean, cellReliefFrac: { 300: cellRelief(300), 1500: cellRelief(1500) }, urban, desert: view.some((x) => /desert/i.test(x.featurecla || '')), ice: view.some((x) => x.kind === 'glacier') },
      view: { rangeKm, lat: r.facts['view.lat_deg'].v }, lighting: r.zoom.lighting || {} }).filter((t) => t !== 'WATER_DOMINANT');
    if (water > 0.6) tags.unshift('WATER_DOMINANT');
    r.zoom.tags = tags;
    put('geo.terrain', terrainPhrase(tags));
    return r;
  },
};
