// vlm/gen/build/balance.js — Pilot Eye's class-balanced sampling weights per family x verdict (spec §8), summing to N: every
// cell gets the same total weight. A keeps its natural mix (about 65 % UNSAFE, ruling T10-h); the weights balance it at
// sampling time, and stats.json reports the raw mix.
export const cellOf = (r) => `${r.family}|${r.safety_eye ? r.safety_eye.verdict : 'zoom'}`;
export function classWeights(recs) {
  const n = new Map(); for (const r of recs) n.set(cellOf(r), (n.get(cellOf(r)) || 0) + 1);
  const raw = new Map(recs.map((r) => [r.key, 1 / n.get(cellOf(r))])), sum = [...raw.values()].reduce((a, b) => a + b, 0);
  return new Map([...raw].map(([k, w]) => [k, (w * recs.length) / sum]));
}
// the weights within each split (rec.split): train balances family x verdict inside train and sums to N_train (review item 1)
export function splitClassWeights(recs) {
  const by = new Map(); for (const r of recs) (by.get(r.split) || by.set(r.split, []).get(r.split)).push(r);
  return new Map([...by.values()].flatMap((rs) => [...classWeights(rs)]));
}
