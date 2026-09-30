// vlm/gen/labels/worlds.js — the page's corridor worlds rebuilt in Node (app.js 56-59, 264-296) for tests and the G3
// calibration: space at a density, or an atmospheric route with the URL sky. Long worlds need registerLongGrids() first.
import { SpaceEnv } from '../../../src/env.js';
import { SKIES } from '../../../src/weather.js';
import { registerLongGrid, hasLongGrid } from '../../../src/cityfield.js';

// src/terrain.js:78-86 (terrain.js imports three, so the route table is copied; tests/vlm_corridor.test.mjs checks it against the text of src/terrain.js)
export const ROUTES = Object.freeze({ alps: { name: 'Alps', lat: 46.55, lon0: 7.3, terrain: 'alps' }, china: { name: 'China', lat: 29.33, lon0: 109.9, terrain: 'meshy' },
  newyork: { name: 'New York', lat: 40.74, lon0: -74.5, city: 'newyork' }, london: { name: 'London', lat: 51.5, lon0: -0.7, city: 'london' },
  moscow: { name: 'Moscow', lat: 55.75, lon0: 36.85, city: 'moscow' }, dubai: { name: 'Dubai', lat: 25.2, lon0: 54.75, city: 'dubai' }, mega: { name: 'Megacity', lat: 48.86, lon0: 2.1, city: 'mega' } });

export async function registerLongGrids() {
  if (!hasLongGrid('mega')) registerLongGrid('mega', await import('../../../src/mega_grid.js'));
  if (!hasLongGrid('avatar')) registerLongGrid('avatar', await import('../../../src/avatar_grid.js'));
}
export function pageWorld({ route = null, sky = 'fair', seed = 11, density = 25, comets = 2 } = {}) {
  const e = new SpaceEnv(seed, { level: 1, count: density, speedScale: 1, comets, cometSpeed: 1 }), p = SKIES[sky];
  e.setWeather(p[0], { wind: p[1], cover: p[2], turb: p[3] });
  if (!route) return e;
  const R = ROUTES[route];
  e.setAtmosphere(true); e.setComets(0); e.setCount(0);
  if (R.city) { e.setCity(R.city, 1); e.setPlanes(0); } else if (R.terrain === 'meshy') { e.setMountains(false); e.setMeshy(true, hasLongGrid('avatar') ? 'avatar' : null); e.setPlanes(0); }
  else if (R.terrain === 'alps') { e.setMountains(true); e.setPlanes(6); } else { e.setMountains(false); e.setPlanes(6); }
  e.setBirds(R.terrain === 'pillars' || R.terrain === 'meshy' ? 4 : 3); e.spawnAsteroids(); e.reset();
  return e;
}
