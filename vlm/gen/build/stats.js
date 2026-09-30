// vlm/gen/build/stats.js — stats.json (spec §9, amendment B): counts per family x split x verdict x reason, A's verdict mix
// per route x behaviour and its raw mix (T10-h), A reset counts and the t_since_reset_s histogram, Narrator rows per family x
// {with, without Context}, minimal pairs, the text reject rates per family and per template (the §11.3 re-measure on real
// records), distinct-n and self-BLEU, dedupe drops, Z per-split x range-bin counts and the discarded Z views.
import fs from 'node:fs';
import path from 'node:path';
import { queryOf } from './split.js';
import { detail } from '../../capture/report.mjs';

const bump = (m, k, n = 1) => { m[k] = (m[k] || 0) + n; return m; };
const rate = (a, b) => +(a / Math.max(1, b)).toFixed(4);
// the verifier tally over recordTexts() results: totals and rejections per family, per template component (a composite id
// "a+b+c" counts toward each of a, b and c) and per task; parser rejections are rejected items whose parser check failed
export function textTally() {
  const fam = {}, tpl = {}, task = {}, T = { items: 0, rejected: 0, parser: 0 };
  const parts = (it) => (it.template_id ? String(it.template_id).split('+') : [`ask:${it.family_q || 'unknown'}`]);
  const count = (f, it, bad) => {
    const F = (fam[f] ||= { total: 0, rejected: 0 }), K = (task[it.task || 'vqa'] ||= { total: 0, rejected: 0 }); F.total++; K.total++; if (bad) { F.rejected++; K.rejected++; }
    for (const p of new Set(parts(it))) { const P = (tpl[p] ||= { total: 0, rejected: 0 }); P.total++; if (bad) P.rejected++; }
  };
  return {
    add(family, res) {
      for (const it of res.texts) { T.items++; count(family, it, false); }
      for (const x of res.rejected) { T.rejected++; if (x.parserOk === false) T.parser++; count(family, x.item || {}, true); }
    },
    summary() {
      const withRate = (o) => Object.fromEntries(Object.entries(o).map(([k, x]) => [k, { ...x, rejectRate: rate(x.rejected, x.total) }]));
      const byTemplate = Object.fromEntries(Object.entries(withRate(tpl)).sort((a, b) => b[1].rejected - a[1].rejected || a[0].localeCompare(b[0])));
      return { items: T.items, rejected: T.rejected, rejectRate: rate(T.rejected, T.items + T.rejected), parserRejects: T.parser, parserFalseReject: rate(T.parser, T.items + T.rejected),
        byFamily: withRate(fam), byTask: withRate(task), byTemplate };
    },
  };
}
const tokens = (s) => String(s).toLowerCase().replace(/[^\p{L}\p{N}' ]+/gu, ' ').split(/\s+/).filter(Boolean);
const grams = (t, n) => { const out = []; for (let i = 0; i + n <= t.length; i++) out.push(t.slice(i, i + n).join(' ')); return out; };
export function distinctN(texts, n) { const all = texts.flatMap((s) => grams(tokens(s), n)); return all.length ? new Set(all).size / all.length : 0; }
// self-BLEU-4 (add-one smoothed for n > 1) of up to `max` evenly spaced texts, each against the others as references
export function selfBleu(texts, { max = 200 } = {}) {
  const step = Math.max(1, texts.length / max), pick = []; for (let i = 0; i < texts.length && pick.length < max; i += step) pick.push(tokens(texts[Math.floor(i)]));
  if (pick.length < 2) return null;
  const counts = pick.map((t) => [1, 2, 3, 4].map((n) => grams(t, n).reduce((m, g) => bump(m, g), {})));
  let sum = 0;
  for (let h = 0; h < pick.length; h++) {
    let logp = 0;
    for (let n = 1; n <= 4; n++) {
      const hc = counts[h][n - 1], total = Object.values(hc).reduce((a, b) => a + b, 0); let clip = 0;
      for (const [g, c] of Object.entries(hc)) { let best = 0; for (let r = 0; r < pick.length; r++) if (r !== h) best = Math.max(best, counts[r][n - 1][g] || 0); clip += Math.min(c, best); }
      logp += Math.log(n === 1 ? Math.max(clip, 1e-9) / Math.max(total, 1) : (clip + 1) / (total + 1)) / 4;
    }
    const len = pick[h].length, ref = pick.filter((_, r) => r !== h).map((t) => t.length).sort((a, b) => Math.abs(a - len) - Math.abs(b - len) || a - b)[0];
    sum += Math.exp(logp) * (len >= ref ? 1 : Math.exp(1 - ref / Math.max(1, len)));
  }
  return +(sum / pick.length).toFixed(4);
}
export function diversity(recs) {
  const by = {}; for (const r of recs) for (const t of r.texts) (by[t.task] ||= []).push(t.answer);
  return Object.fromEntries(Object.entries(by).map(([k, xs]) => [k, { n: xs.length, distinct_1: +distinctN(xs, 1).toFixed(4), distinct_2: +distinctN(xs, 2).toFixed(4), self_bleu4: selfBleu(xs) }]));
}
// upstream requests and cache hits per host over the runs' throughput ledgers (report.mjs detail) and the EOX budget used
export function upstreamOf(rawRoot, runs) {
  const out = {};
  for (const run of runs) {
    const R = (out[run] = { families: {}, eox_used_run: null });
    const b = path.join(rawRoot, run, 'eox_budget.json'); if (fs.existsSync(b)) R.eox_used_run = JSON.parse(fs.readFileSync(b, 'utf8')).used;
    for (const f of ['S', 'A', 'L', 'D', 'Z']) {
      const p = path.join(rawRoot, run, f, 'throughput.jsonl'); if (!fs.existsSync(p)) continue;
      const rows = fs.readFileSync(p, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l)), d = detail(f, rows);
      R.families[f] = { requests_by_host: d.ledger.by_host, upstream_by_host: d.ledger.upstream_by_host, esri_reached_network: d.ledger.esri_reached_network };
    }
  }
  return out;
}
const HIST = [0, 1, 2, 3, 4.6, 10];
const histBin = (x) => { const i = HIST.filter((e) => x >= e).length - 1; return i < HIST.length - 1 ? `${HIST[i]}-${HIST[i + 1]} s` : `>= ${HIST[HIST.length - 1]} s`; };
export function datasetStats({ kept, recs, splitOfKey, dropped, invalid, zDiscards, textSummary, rows, exportDrops, eye, upstream, textFactFails }) {
  const eyeOf = (r) => (r.safety_eye ? r.safety_eye.verdict : 'zoom'), s = { counts: {}, reasons: {} };
  for (const r of kept) { bump(s.counts, `${r.family}|${r.split}|${eyeOf(r)}`); if (r.safety_eye) for (const x of r.safety_eye.reasons) bump((s.reasons[`${r.family}|${r.split}`] ||= {}), x); }
  const A = kept.filter((r) => r.family === 'A'), aMix = {}, aRaw = {}, aNat = {};
  for (const r of A) { const q = queryOf(r); bump((aMix[`${q.get('route')}|${r.provenance.policy_id}`] ||= {}), eyeOf(r)); bump(aRaw, eyeOf(r)); if (r.natural) bump(aNat, eyeOf(r), r.provenance.sampler_weight ?? 1); }
  const pages = new Map(), seg = {}, hist = {};
  for (const r of A.filter((x) => !x.provenance.twin_of)) {
    const k = `${r.run}|${r.provenance.page_episode}`, g = r.provenance.page_segment ?? 0; pages.set(k, Math.max(pages.get(k) ?? 0, g)); bump(seg, g);
    if (r.provenance.t_since_reset_s !== null && r.provenance.t_since_reset_s !== undefined) bump(hist, histBin(r.provenance.t_since_reset_s));
  }
  s.a = { raw_mix: aRaw, natural_mix_sampler_weighted: aNat, by_route_policy: aMix, pages: pages.size, resets_seen: [...pages.values()].reduce((a, b) => a + b, 0), records_by_segment: seg,
    t_since_reset_hist: hist, post_reset_records: Object.values(hist).reduce((a, b) => a + b, 0) };
  s.naturalMix = {};
  for (const r of kept.filter((x) => x.natural && x.safety_eye)) { const k = `${r.family}|${r.facts.world ? r.facts.world.v : r.family}|${eyeOf(r)}`, c = (s.naturalMix[k] ||= { n: 0, sampler_weighted: 0 }); c.n++; c.sampler_weighted = +(c.sampler_weighted + (r.provenance.sampler_weight ?? 1)).toFixed(3); }
  const keys = new Set(kept.map((r) => r.key)), twins = kept.filter((r) => r.provenance.twin_of);
  s.pairs = { total: twins.filter((r) => keys.has(r.provenance.twin_of)).length, pixel_identical: twins.filter((r) => r.pixel_identical_pair).length, twin_without_original: twins.filter((r) => !keys.has(r.provenance.twin_of)).length,
    injected: kept.filter((r) => r.provenance.injection).length, injected_without_twin: kept.filter((r) => r.provenance.injection && !twins.some((t) => t.provenance.twin_of === r.key)).length };
  s.dedupeDrops = recs.filter((r) => dropped.has(r.key)).reduce((m, r) => bump(m, `${r.family}|${splitOfKey.get(r.key)}`), {});
  s.z = { range_bins: {}, discarded_views: zDiscards };
  for (const r of kept.filter((x) => x.family === 'Z')) { const b = (s.z.range_bins[r.split] ||= [0, 0, 0, 0, 0]); b[r.zoom.range_bin]++; }
  s.narrator = { rows: {}, by_task: {}, drops: exportDrops };
  for (const x of rows) { bump((s.narrator.rows[x.family] ||= { with: 0, without: 0 }), x.context ? 'with' : 'without'); bump((s.narrator.by_task[x.family] ||= {}), x.task); }
  s.eye = eye; s.invalidRecords = invalid; s.textFactFails = textFactFails; s.rejectedRecords = invalid.length + textFactFails.length;
  s.text = { ...textSummary, diversity: diversity(kept) }; s.upstream = upstream;
  return s;
}
