// vlm/capture/probe/core.js — window.__apv (spec §7.3), loaded into the page after boot; each family module exports
// setup(family, params) -> state and plain functions (state, arg) the drive calls by name. Family modules are imported on
// demand, so a Z page never loads the corridor labellers (and their city grids) and an S page never loads the zoom's.
const FAMILIES = { S: () => import('./corridor.js'), A: () => import('./corridor.js'), Z: () => import('./zoom.js') };
export function createProbe() {
  let fam = null, st = null;
  return { async setup(family, params) { if (!FAMILIES[family]) throw new Error(`no probe for family ${family}`); fam = await FAMILIES[family](); st = await fam.setup(family, params); return fam.check(st); },
    check: () => fam.check(st), call: (name, arg) => fam[name](st, arg) };
}
