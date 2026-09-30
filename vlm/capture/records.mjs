// vlm/capture/records.mjs — record assembly and the on-disk layout shared by drive.mjs and families.mjs: render seeds, one
// validated record per sample (imagery[] from the ledger window, discard on a dirty ledger), writing a finished episode
// (episode_<n>.done last) and the resume scan that deletes the files of unfinished episodes (Review Focus 4).
import fs from 'node:fs';
import path from 'node:path';
import { validateRecord, recordKey, STEP_S } from '../gen/schema.js';
import { imageryFrom, discardReason } from './routes.mjs';
export const fnv1a32 = (s) => { let h = 0x811c9dc5; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; } return h >>> 0; };
export const renderSeed = (run, family, seed) => fnv1a32(`${run}|${family}|${seed}`);
export function writeEpisode(dir, records, files, episode) {
  fs.mkdirSync(dir, { recursive: true });
  for (const [name, buf] of Object.entries(files)) fs.writeFileSync(path.join(dir, name), buf);
  for (const r of records) fs.writeFileSync(path.join(dir, `${r.key}.json`), JSON.stringify(r));
  fs.writeFileSync(path.join(dir, `episode_${String(episode).padStart(5, '0')}.done`), String(records.length));
}
// A 403/429 (§6: stop with no retry) must outlive the process: tilecache latches the host only in memory, so the drive
// persists it to raw/<run>/stop.json ({host, status, utc_ms}; `stops` keeps every one), refuses to start while it exists,
// and exits with STOP_EXIT so a `&&` chain halts. A budget stop is not a ban: it persists via eox_budget.json and exits 0.
export const STOP_EXIT = 3, RESOURCE_EXIT = 4;
export function stopOf(err, host = null) {
  const m = /^(403|429) from (\S+):/.exec(String(err && err.message));
  if (m) return { kind: 'http', host: m[2], status: +m[1] };
  const b = /^upstream budget of \d+ requests reached for (\S+)/.exec(String(err && err.message));
  return b ? { kind: 'budget', host: b[1], status: null } : { kind: 'other', host, status: null };
}
export function readStop(runDir) { const f = path.join(runDir, 'stop.json'); return fs.existsSync(f) ? JSON.parse(fs.readFileSync(f, 'utf8')) : null; }
export function writeStop(runDir, { host, status }, utcMs = Date.now()) {
  const old = readStop(runDir), entry = { host, status, utc_ms: utcMs }, stops = [...((old && old.stops) || []), entry];
  fs.mkdirSync(runDir, { recursive: true }); fs.writeFileSync(path.join(runDir, 'stop.json'), JSON.stringify({ ...(old ? { host: old.host, status: old.status, utc_ms: old.utc_ms } : entry), stops }));
  return readStop(runDir);
}
// the start gate: a run with a recorded 403/429 stop refuses to start (before any browser or request); clearing it is a
// deliberate human step (delete stop.json once the host allows the traffic again)
export function startGate(runDir) {
  const s = readStop(runDir);
  return s ? { refuse: true, code: STOP_EXIT, why: `${runDir}/stop.json: ${s.status} from ${s.host} at ${new Date(s.utc_ms).toISOString()}; delete it only once the host allows traffic again` } : { refuse: false, code: 0, why: null };
}
export const exitCodeOf = (err) => (stopOf(err).kind === 'http' ? STOP_EXIT : 0);
// the stop the drive classifies at its end: a latched 403/429 (routes.latchStop) wins over the error that was thrown
export const pickStop = (thrown, latched) => (latched && stopOf(latched).kind === 'http' ? latched : thrown);
// N-1: how many consecutive episode indices from e are free (no .done), up to cap: an A page that samples across crash resets
// may span only those, so a resumed rerun never reaches, let alone overwrites, a finished page's episodes
export function freeSpan(done, e, cap) { let k = 0; while (k < cap && !done.has(e + k)) k++; return k; }
// M-a: the records of a list that are not on disk yet (a resumed rerun rewrites a page's twin; only new files count)
export const freshRecords = (dir, recs) => recs.filter((r) => !fs.existsSync(path.join(dir, `${r.rec.key}.json`)));
export function scanRun(dir) {
  const out = { samples: 0, episodes: new Set(), removed: [] }; if (!fs.existsSync(dir)) return out;
  const names = fs.readdirSync(dir).filter((n) => !n.startsWith('._'));
  for (const n of names) { const m = /^episode_(\d{5})\.done$/.exec(n); if (m) out.episodes.add(+m[1]); }
  for (const n of names) { const m = /^[SALDZ]_.+_(\d{5})_\d{6}\./.exec(n); if (!m) continue; if (!out.episodes.has(+m[1])) { fs.rmSync(path.join(dir, n)); out.removed.push(n); } else if (n.endsWith('.json')) out.samples++; }
  return out;
}
export function capSeverity4(samples) {
  const s4 = samples.filter((s) => s.label.safety && s.label.safety.severity === 4), keep = new Set(s4.slice(0, 2));
  for (const s of keep) s.weight *= s4.length / keep.size;
  return samples.filter((s) => !s4.includes(s) || keep.has(s));
}
const png = (dataUrl) => Buffer.from(dataUrl.slice(dataUrl.indexOf(',') + 1), 'base64');
export function assemble(D, ep, s, extra = {}) {
  const fam = ep.family, key = recordKey(fam, D.run, ep.episode, s.step), names = fam === 'Z' ? [`${key}.f0.png`] : ['f0', 'f1', 'f2'].map((f) => `${key}.${f}.png`), chase = s.chase ? `${key}.chase.png` : null;
  if (chase) Object.assign(extra, { narrator: chase, cameras: { [chase]: s.chase.cam }, files: [[chase, png(s.chase.png)]] });
  // A strip tiles load during boot and warm-up, and a Z location's later views reuse tiles of the page-lifetime loader cache, so
  // A reads the ledger from its page's first request (s.ledger0; imagery and discard) and Z imagery too, restricted to the
  // view's ring levels (§6: A strip levels 9/11/12, Z ring levels)
  const frames = fam === 'Z' ? [s.frame] : s.frames, cams = Object.fromEntries(names.map((n, k) => [n, frames[k].cam])), p0 = s.ledger0 ?? 0, i0 = fam === 'A' ? p0 : frames[0].ledger[0], i1 = frames[frames.length - 1].ledger[1];
  const imagery = imageryFrom(D.ledger, fam === 'Z' ? p0 : i0, i1).filter((m) => fam !== 'Z' || (s.label.facts['view.rings'].v || []).includes(m.level));
  // Z: the ground height under the target (ruling T6-d; zoom.info or the rendered pose, src says which) is kept with the record,
  // and lighting.utc_ms is the view's own Sun time (orbitInfo.utc = v.utcMs), which provenance.utc_ms (the page clock) is not
  const zoom = fam === 'Z' ? { tags: null, range_bin: s.v.range_bin, lighting: { ...s.label.lighting, utc_ms: s.v.utcMs ?? null }, ground_km: s.label.ground ? s.label.ground.km : null, ground_src: s.label.ground ? s.label.ground.src : null } : null;
  const rec = { key, family: fam, frames: names, narrator_frame: extra.narrator ?? names[fam === 'Z' ? 0 : 2], frame_dt_steps: fam === 'Z' ? null : s.n, frame_dt_s: fam === 'Z' ? null : s.n.map((n) => +(n * STEP_S[fam]).toFixed(4)),
    facts: s.label.facts, safety: s.label.safety ?? null, safety_eye: s.label.safety_eye ?? null, zoom, cameras: { ...cams, ...(extra.cameras || {}) },
    render: { viewport: [896, 504], dpr: 1, renderScale: 1, toneMapping: 'ACESFilmic', capture_mode: D.mode, view: { eye: extra.eyeView ?? (fam === 'Z' ? 'zoom' : 'chase'), narrator: fam === 'Z' ? 'zoom' : 'chase' }, path: fam === 'L' ? false : null, imagery, licence_profile: D.licence },
    provenance: { git_sha: D.gitSha, site_dirty: !!D.siteDirty, page_url: ep.url, seed: ep.seed, render_seed: ep.renderSeed, episode: ep.episode, step: s.step, sim_t_s: +(s.step * (STEP_S[fam] ?? 0)).toFixed(4), utc_ms: ep.utcMs,
      clock: frames.map((f) => f.clock), policy_id: ep.policyId ?? null, policy_sha: ep.policySha ?? null, injection: s.injection ?? null, twin_of: ep.twinOf !== undefined && ep.twinOf !== null ? recordKey(fam, D.run, ep.twinOf, s.srcStep ?? s.step) : null,
      sampler_weight: s.weight ?? 1, ...(fam === 'A' ? { atmosphere: s.atmosphere, page_episode: ep.pageEpisode ?? ep.episode, page_segment: s.seg ?? 0, t_since_reset_s: (s.seg ?? 0) > 0 ? +((s.step - (s.gate_step ?? 0)) / 15).toFixed(3) : null } : {}), ...(fam === 'S' || fam === 'A' ? { t_since_gate_s: +((s.step - (s.gate_step ?? 0)) / 15).toFixed(3) } : {}), generator: `vlm/capture@${D.gitSha}` } };
  const bad = discardReason(D.ledger, i0, i1, D.licence), v = validateRecord(rec);
  return { rec, files: Object.fromEntries(names.map((n, k) => [n, png(frames[k].png)]).concat(extra.files || [])), error: bad || (v.ok ? null : v.errors.join('; ')) };
}
