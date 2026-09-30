"""vlm/train/pilot_eye/evaluate.py — the §10.1 evaluation and the §11.4 gates per family (S, A, L, D) on the full test split and its
natural subset: within-family UNSAFE-vs-rest AUROC with a 95 % bootstrap CI over seed groups, pooled AUROC and macro-F1, the
context-prior baseline (logistic regression on family, route, sky, start, phase only), ECE after temperature scaling fitted on
natural val, action top-1 in safe_actions and S/A regret, minimal pairs (d), regression MAE, zoom tag F1 and range accuracy.
It also writes <out>/confusion.json, the val-split verdict confusion matrix (counts, row-normalised, and the per-verdict rows
that build.mjs --confusion passes to corruptMonitor), since v0 has no cross-fit (T3H-3). --write-preds writes the monitor
tuples used as Narrator's Context (spec §5.7).
  python -m vlm.train.pilot_eye.evaluate --ckpt <model.pt> --data <dataset> --split test [--out <dir>]
  python -m vlm.train.pilot_eye.evaluate --ckpt <model.pt> --data <dataset> --write-preds <file> --keys-from <split>
(--keys-from train needs a train.py --fold k checkpoint and predicts only the other half's rows, tagged with the fold.)"""
import argparse, json, math, os, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from torch.utils.data import DataLoader
from vlm.train.common import read_jsonl, group_half
from vlm.train.pilot_eye.model import PilotEye
from vlm.train.pilot_eye.data import PilotEyeData, REG
FAM_ACTIONS = {'S': [0, 1, 2, 3, 4, 5, 6], 'A': [0, 1, 2, 3, 4, 5, 6], 'L': [0, 7, 3, 4, 1, 2, 5, 6], 'D': [0, 9, 8, 6]}
ACTIONS = ['CONTINUE', 'CLIMB', 'DESCEND', 'TURN_LEFT', 'TURN_RIGHT', 'SPEED_UP', 'SLOW_DOWN', 'GO_AROUND', 'HOLD_POSITION', 'BREAKOUT', 'NONE_SAFE']
VERDICTS = ['SAFE', 'CAUTION', 'UNSAFE']
REASONS = ['HAZARD_AHEAD', 'HAZARD_CLOSING_FAST', 'TERRAIN_CLOSE', 'BUILDING_CLOSE', 'CORRIDOR_EDGE', 'STALL', 'OVERSTRESS', 'SEVERE_TURBULENCE', 'STORM_CELL', 'PULL_UP', 'UNSTABLE_APPROACH', 'LOCALIZER_DEVIATION', 'GLIDESLOPE_DEVIATION', 'SPEED_OUT_OF_BAND', 'HIGH_SINK_RATE', 'STRONG_CROSSWIND', 'TAILWIND', 'RUNWAY_EDGE', 'CANNOT_STOP', 'KOS_VIOLATION', 'CLOSING_TOO_FAST', 'LATERAL_MISALIGNMENT', 'ATTITUDE_ERROR', 'JET_FAILURE', 'LOW_FUEL', 'NO_BREAKOUT_AVAILABLE']
DEFAULT_ROW = {'SAFE': 0.125, 'CAUTION': 0.125, 'UNSAFE': 0.125}  # corruptMonitor's own prior (0.75 on the diagonal), for verdicts val lacks

def softmax(x, T=1.0):
    e = np.exp((x - x.max(-1, keepdims=True)) / T); return e / e.sum(-1, keepdims=True)
def predict(m, ds, T=1.0, device='cpu', batch=64):
    out = {}; m.eval().to(device)
    with torch.no_grad():
        for f, dt, _, _, keys in DataLoader(ds, batch_size=batch):
            o = [x.float().cpu().numpy() for x in m(f.to(device), dt.to(device))]
            for j, key in enumerate(keys):
                out[key] = {'verdict': softmax(o[0][j], T), 'logits': o[0][j], 'severity': int(o[1][j].argmax()), 'reasons': 1 / (1 + np.exp(-o[2][j])), 'actions': 1 / (1 + np.exp(-o[3][j])),
                            'p_ref': float(1 / (1 + np.exp(-o[4][j, 0]))), 'reg': o[5][j], 'tags': 1 / (1 + np.exp(-o[6][j])), 'range': int(o[7][j].argmax())}
    return out
def pick_action(p, fam):
    sub = FAM_ACTIONS[fam]; best = max(p[i] for i in sub)
    if best < 0.5: return 'NONE_SAFE'
    ties = [i for i in sub if abs(p[i] - best) < 1e-6]; return 'CONTINUE' if 0 in ties else ACTIONS[ties[0]]
def boot_ci(y, s, groups, n=1000, seed=0):
    if len(y) == 0 or len(set(y.tolist())) < 2: return [None, None]
    rng, g = np.random.default_rng(seed), np.array(groups); u = np.unique(g); vals = []
    for _ in range(n):
        idx = np.concatenate([np.where(g == k)[0] for k in rng.choice(u, len(u))])
        if len(set(y[idx])) == 2: vals.append(roc_auc_score(y[idx], s[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [None, None]
def ece(conf, correct, bins=15):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (conf > lo) & (conf <= hi)
        if k.any(): e += k.mean() * abs(correct[k].mean() - conf[k].mean())
    return float(e)
def confusion(y_true, y_pred):
    """3x3 verdict confusion (rows = truth, cols = prediction): counts, row-normalised (None for an empty row) and the keyed rows
    corruptMonitor reads as confusion[truth][drawn] (an empty row falls back to its own 0.75 / 0.125 prior, listed in fallback_rows)."""
    c = [[0] * 3 for _ in range(3)]
    for t, p in zip(y_true, y_pred): c[int(t)][int(p)] += 1
    rn = [[x / sum(r) for x in r] if sum(r) else None for r in c]
    R = {'labels': VERDICTS, 'n': int(sum(map(sum, c))), 'counts': c, 'row_normalised': rn, 'fallback_rows': [VERDICTS[i] for i, r in enumerate(rn) if r is None]}
    for i, v in enumerate(VERDICTS): R[v] = dict(zip(VERDICTS, rn[i])) if rn[i] else {**DEFAULT_ROW, v: 0.75}
    return R
def meta_x(rows, cats=None):
    keys = ['family', 'route', 'sky', 'start', 'phase']; vals = [[str(r['family'] if k == 'family' else r['meta'].get(k)) for k in keys] for r in rows]
    cats = cats or [sorted({v[i] for v in vals}) for i in range(len(keys))]
    return np.array([[1.0 if v[i] == c else 0.0 for i in range(len(keys)) for c in cats[i]] for v in vals]).reshape(len(vals), -1), cats
def flight(rows): return [r for r in rows if r['family'] != 'Z' and r['masks']['verdict']]
def auroc(y, s, w=None): return float(roc_auc_score(y, s, sample_weight=w)) if len(set(np.asarray(y).tolist())) == 2 else None
def evaluate(pred, rows, train_rows, recs):
    R, fl = {'per_family': {}}, flight(rows)
    y = np.array([r['targets']['verdict'] == 2 for r in fl], dtype=int); s = np.array([pred[r['key']]['verdict'][2] for r in fl])
    for f in 'SALD':
        # a family can be absent from a split (the OOD split holds only A, D and Z; a small pilot's test split may lack families)
        k = np.array([r['family'] == f for r in fl], dtype=bool)
        if not k.any(): R['per_family'][f] = {'n': 0, 'auroc': None, 'ci95': [None, None], 'auroc_natural': None}; continue
        nat = k & np.array([r['natural'] for r in fl], dtype=bool)
        R['per_family'][f] = {'n': int(k.sum()), 'auroc': auroc(y[k], s[k]), 'ci95': boot_ci(y[k], s[k], [r['group'] for r, kk in zip(fl, k) if kk]),
                              'auroc_natural': auroc(y[nat], s[nat], [r['sampler_weight'] for r, kk in zip(fl, nat) if kk]) if nat.any() else None}
    aus = [v['auroc'] for v in R['per_family'].values() if v['auroc'] is not None]; R['mean_family_auroc'] = float(np.mean(aus)) if aus else None
    R['families_measured'] = [f for f in 'SALD' if R['per_family'][f]['auroc'] is not None]  # gate (a) needs all four
    R['pooled_auroc'] = auroc(y, s) if fl else None; yv = [r['targets']['verdict'] for r in fl]; pv = [int(np.argmax(pred[r['key']]['verdict'])) for r in fl]
    R['pooled_macro_f1'] = float(f1_score(yv, pv, average='macro')) if fl else None
    R['majority_macro_f1'] = float(f1_score(yv, [max(set(yv), key=yv.count)] * len(yv), average='macro')) if fl else None
    tr = flight(train_rows); R['baseline_auroc'] = R['baseline_macro_f1'] = None
    if fl and len({r['targets']['verdict'] for r in tr}) >= 2:
        Xtr, cats = meta_x(tr); Xte, _ = meta_x(fl, cats)
        lr = LogisticRegression(max_iter=2000).fit(Xtr, [r['targets']['verdict'] for r in tr]); pb = lr.predict_proba(Xte)
        R['baseline_auroc'] = auroc(y, pb[:, list(lr.classes_).index(2)]) if 2 in lr.classes_ else None; R['baseline_macro_f1'] = float(f1_score(yv, lr.predict(Xte), average='macro'))
    act = {r['key']: pick_action(pred[r['key']]['actions'], r['family']) for r in fl}
    top1 = [act[r['key']] in [ACTIONS[i] for i, v in enumerate(r['targets']['safe_actions']) if v] or (act[r['key']] == 'NONE_SAFE' and not any(r['targets']['safe_actions'])) for r in fl]
    R['action_top1_in_safe'] = float(np.mean(top1)) if top1 else None
    reg = [recs[r['key']]['safety']['action_risk'].get(act[r['key']], 1) - recs[r['key']]['safety']['p_best'] for r in fl if r['family'] in 'SA' and (recs.get(r['key']) or {}).get('safety')]
    R['regret_sa'] = float(np.mean(reg)) if reg else None
    pairs, wins, by = 0, 0, {r['key']: r for r in fl}
    for r in fl:
        tw = r.get('twin_of')
        if not tw or tw not in by: continue
        a, b = by[tw], r
        if a['targets']['verdict'] == b['targets']['verdict'] or (recs.get(a['key']) or {}).get('pixel_identical_pair'): continue
        inj, clean = (a, b) if a.get('injection') else (b, a); pairs += 1; wins += int(pred[inj['key']]['verdict'][2] > pred[clean['key']]['verdict'][2])
    p = wins / pairs if pairs else None; R['pairs'] = {'n': pairs, 'frac': p, 'ci95': [p - 1.96 * math.sqrt(p * (1 - p) / pairs), p + 1.96 * math.sqrt(p * (1 - p) / pairs)] if pairs else None}
    z = [r for r in rows if r['family'] == 'Z']
    R['zoom_tags_f1'] = float(f1_score(np.array([r['targets']['zoom_tags'] for r in z]), np.array([pred[r['key']]['tags'] > 0.5 for r in z]).astype(int), average='micro', zero_division=0)) if z else None
    R['zoom_range_acc'] = float(np.mean([pred[r['key']]['range'] == r['targets']['range_bin'] for r in z])) if z else None
    R['reg_mae'] = {}
    for i, k in enumerate(REG):
        e = [abs(float(pred[r['key']]['reg'][i]) - float(r['targets'][k])) for r in rows if r['masks']['reg'][i]]; R['reg_mae'][k] = float(np.mean(e)) if e else None
    return R
def gates(R):
    bad = []
    # (a) is the mean over the four families (§11.4): a family with no AUROC (one class or no rows) fails it, never drops out
    miss = [f for f in 'SALD' if (R['per_family'].get(f) or {}).get('auroc') is None]
    if miss: bad.append(f"(a) family {','.join(miss)} has no AUROC (one class or no rows in the split)")
    elif not (R['mean_family_auroc'] and R['mean_family_auroc'] >= 0.75): bad.append('(a) mean within-family AUROC < 0.75')
    if not (R['pooled_auroc'] is not None and R['baseline_auroc'] is not None and R['pooled_auroc'] >= 0.80 and R['pooled_auroc'] >= R['baseline_auroc'] + 0.10): bad.append('(b) pooled AUROC (None = one class only: a failure)')
    if not (R['pooled_macro_f1'] is not None and R['baseline_macro_f1'] is not None and R['pooled_macro_f1'] >= R['baseline_macro_f1'] + 0.10): bad.append('(c) macro-F1 vs baseline')
    if not (R['pairs']['frac'] is not None and R['pairs']['frac'] >= 0.80): bad.append('(d) minimal pairs < 80 %')
    return bad
def fit_temperature(pred, rows):
    """The temperature (0.5-5) minimising the verdict NLL on natural flight rows; 1.0 when there are none."""
    rows = [r for r in flight(rows) if r['natural']]
    if not rows: return 1.0
    L = np.array([pred[r['key']]['logits'] for r in rows]); y = np.array([r['targets']['verdict'] for r in rows])
    nll = lambda T: -np.mean(np.log(softmax(L, T)[np.arange(len(y)), y] + 1e-12))
    return float(min(np.linspace(0.5, 5, 91), key=nll))
def monitor(p, fam, reasons, allowed=None):
    ttc = math.expm1(p['reg'][0]); clr = math.expm1(p['reg'][1])
    tb = '<1 s' if ttc < 1 else '1-3 s' if ttc < 3 else '3-6 s' if ttc < 6 else '>6 s'; cb = '<5 u' if clr < 5 else '5-15 u' if clr < 15 else '15-40 u' if clr < 40 else '>40 u'
    rs = [reasons[i] for i, v in enumerate(p['reasons']) if v > 0.5 and (allowed is None or reasons[i] in allowed)]
    return {'verdict': VERDICTS[int(np.argmax(p['verdict']))], 'severity': p['severity'], 'reasons': rs, 'action': pick_action(p['actions'], fam), 'p_ref': round(p['p_ref'], 2) if fam in 'SA' else None,
            'ttc_bin': tb if fam in 'SA' else 'none', 'clr_bin': cb if fam in 'SA' else 'none'}
def ort_family_auroc(out, data, enc_file, H, W):
    """Mean within-family verdict AUROC of the ONNX pair on val (None when no family has both classes)."""
    import onnxruntime as ort
    enc, hd, ds = ort.InferenceSession(f'{out}/{enc_file}'), ort.InferenceSession(f'{out}/heads.onnx'), PilotEyeData(data, 'val', H=H, W=W); ys, ss, fs = [], [], []
    for i in range(len(ds)):
        f, dt, T, M, _ = ds[i]
        if M['verdict'] == 0: continue
        maps = [enc.run(None, {'pixels': f[k:k + 1].numpy()})[0] for k in range(3)]; v = hd.run(None, {'m0': maps[0], 'm1': maps[1], 'm2': maps[2], 'dt': dt[None].numpy()})[0][0]
        ys.append(int(T['verdict']) == 2); ss.append(softmax(v)[2]); fs.append(ds.rows[i]['family'])
    ys, ss, fs = np.array(ys), np.array(ss), np.array(fs)
    aus = [roc_auc_score(ys[fs == f], ss[fs == f]) for f in 'SALD' if len(set(ys[fs == f].tolist())) == 2]
    return float(np.mean(aus)) if aus else None
def load_recs(path, keys):
    """key -> {safety: {action_risk, p_best}, pixel_identical_pair} for the rows evaluated, streamed (records.jsonl is large)."""
    out = {}
    with open(path) as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            if r['key'] in keys:
                s = r.get('safety'); out[r['key']] = {'safety': {'action_risk': s.get('action_risk') or {}, 'p_best': s.get('p_best', 0)} if s else None, 'pixel_identical_pair': bool(r.get('pixel_identical_pair'))}
    return out
def preds_rows(rows, fold, keys_from):
    """The rows --write-preds predicts. Train rows need a cross-fit checkpoint and get only the half its model did not train on
    (spec §5.7: Context from a model that never saw the row); a checkpoint trained on every train row is refused."""
    if keys_from != 'train': return rows
    if fold is None: raise SystemExit('--keys-from train needs a cross-fit checkpoint (train.py --fold k): out-of-fold predictions only')
    return [r for r in rows if group_half(r['group']) != fold]
def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--ckpt', required=True); ap.add_argument('--data', required=True); ap.add_argument('--split', default='test'); ap.add_argument('--write-preds'); ap.add_argument('--keys-from')
    ap.add_argument('--size', default=None, help="default: the checkpoint's size"); ap.add_argument('--out', default=None, help="default: the checkpoint's folder"); ap.add_argument('--device', default='cpu')
    return ap.parse_args(argv)
def main(argv=None):
    a = parse(argv); ck = torch.load(a.ckpt, map_location='cpu'); W, H = map(int, (a.size or ck.get('size') or '160x96').split('x'))
    m = PilotEye(pretrained=False); m.load_state_dict(ck['state']); m.eval(); out = a.out or os.path.dirname(os.path.abspath(a.ckpt)); os.makedirs(out, exist_ok=True)
    with open(f'{a.data}/labels.json') as f: labels = json.load(f)
    reasons, allowed = labels.get('reasons') or REASONS, labels.get('reason_masks') or {}
    if a.write_preds:
        ds = PilotEyeData(a.data, a.keys_from or a.split, H=H, W=W); fold = ck.get('fold'); ds.rows = preds_rows(ds.rows, fold, a.keys_from or a.split)
        pred = predict(m, ds, device=a.device); n = 0
        with open(a.write_preds, 'a') as f:  # appends: the two fold models write disjoint train halves; build.mjs rejects duplicate keys
            for r in ds.rows:
                if r['family'] != 'Z': f.write(json.dumps({'key': r['key'], 'fold': fold, 'monitor': monitor(pred[r['key']], r['family'], reasons, allowed.get(r['family']))}) + '\n'); n += 1
        print(f'wrote {n} predictions'); return
    val = PilotEyeData(a.data, 'val', H=H, W=W); pval = predict(m, val, device=a.device); vfl = flight(val.rows)
    C = confusion([r['targets']['verdict'] for r in vfl], [int(np.argmax(pval[r['key']]['logits'])) for r in vfl]); C['split'] = 'val'
    C['per_family'] = {f: confusion([r['targets']['verdict'] for r in vfl if r['family'] == f], [int(np.argmax(pval[r['key']]['logits'])) for r in vfl if r['family'] == f])['counts'] for f in 'SALD'}
    with open(f'{out}/confusion.json', 'w') as f: json.dump(C, f, indent=1)
    T = fit_temperature(pval, val.rows); ds = val if a.split == 'val' else PilotEyeData(a.data, a.split, H=H, W=W); pred = pval if a.split == 'val' else predict(m, ds, device=a.device)
    for p in pred.values(): p['verdict'] = softmax(p['logits'], T)
    recs = load_recs(f'{a.data}/records.jsonl', {r['key'] for r in ds.rows})
    R = evaluate(pred, ds.rows, read_jsonl(f'{a.data}/pilot_eye/train.jsonl'), recs); R['temperature'] = T; R['split'] = a.split; R['n'] = len(ds.rows)
    nat = [r for r in flight(ds.rows) if r['natural']]
    R['ece_natural_test'] = ece(np.array([pred[r['key']]['verdict'].max() for r in nat]), np.array([int(np.argmax(pred[r['key']]['verdict'])) == r['targets']['verdict'] for r in nat])) if nat else None
    R['gates_failed'] = gates(R)
    with open(f'{out}/metrics_{a.split}.json', 'w') as f: json.dump(R, f, indent=1)
    print(json.dumps({k: R[k] for k in ['mean_family_auroc', 'pooled_auroc', 'baseline_auroc', 'pooled_macro_f1', 'baseline_macro_f1', 'pairs', 'gates_failed']}))
    print(f'confusion (val, n={C["n"]}): {C["counts"]}')
if __name__ == '__main__': main()
