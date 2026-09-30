"""Pilot Eye (Task 13): model shapes and loss masks, the data reader on a synthetic Task 11 dataset (pilot_eye/*.jsonl, the
uint8 cache and its index), training with --max-minutes and the best checkpoint, the ONNX export and its parity, the
evaluation metrics, confusion.json and the monitor predictions, and the pilot-eye latency bench (Node, when available)."""
import os
os.environ.setdefault('HF_HOME', '/Volumes/LaCie/astro-pilot/vlm/hf')
import json, shutil, subprocess, tempfile, unittest, zlib
from pathlib import Path
import numpy as np, torch
from vlm.train.common import read_jsonl, group_half
from vlm.train.pilot_eye.model import PilotEye, loss_fn, HEADS

LACIE_TMP = '/Volumes/LaCie/astro-pilot/vlm/tmp'
REPO = Path(__file__).resolve().parents[3]
H, W = 96, 160
REASONS = ['HAZARD_AHEAD', 'HAZARD_CLOSING_FAST', 'TERRAIN_CLOSE', 'BUILDING_CLOSE', 'CORRIDOR_EDGE', 'STALL', 'OVERSTRESS', 'SEVERE_TURBULENCE', 'STORM_CELL', 'PULL_UP',
           'UNSTABLE_APPROACH', 'LOCALIZER_DEVIATION', 'GLIDESLOPE_DEVIATION', 'SPEED_OUT_OF_BAND', 'HIGH_SINK_RATE', 'STRONG_CROSSWIND', 'TAILWIND', 'RUNWAY_EDGE', 'CANNOT_STOP',
           'KOS_VIOLATION', 'CLOSING_TOO_FAST', 'LATERAL_MISALIGNMENT', 'ATTITUDE_ERROR', 'JET_FAILURE', 'LOW_FUEL', 'NO_BREAKOUT_AVAILABLE']
ACTIONS = ['CONTINUE', 'CLIMB', 'DESCEND', 'TURN_LEFT', 'TURN_RIGHT', 'SPEED_UP', 'SLOW_DOWN', 'GO_AROUND', 'HOLD_POSITION', 'BREAKOUT', 'NONE_SAFE']
FAM_ACT = {'S': range(7), 'A': range(7), 'L': [0, 1, 2, 3, 4, 5, 6, 7], 'D': [0, 6, 8, 9]}
FAM_REA = {'S': range(10), 'A': range(10), 'L': [10, 11, 14, 15, 17, 18], 'D': [19, 20, 21, 22]}
DT = {'S': [0.1333, 0.2], 'A': [0.2, 0.2], 'L': [0.25, 0.25], 'D': [2.0, 1.9]}

def batch(B=4, H=96, W=160):
    T = {'verdict': torch.randint(0, 3, (B,)), 'severity': torch.randint(0, 5, (B,)), 'reasons': torch.zeros(B, 26), 'actions': torch.zeros(B, 11), 'p_ref': torch.rand(B), 'reg': torch.zeros(B, 6), 'tags': torch.zeros(B, 10), 'range': torch.randint(0, 5, (B,))}
    M = {'verdict': torch.ones(B), 'severity': torch.ones(B), 'reasons': torch.ones(B, 26), 'actions': torch.ones(B, 11), 'p_ref': torch.ones(B), 'reg': torch.ones(B, 6), 'tags': torch.zeros(B), 'range': torch.zeros(B)}
    return torch.rand(B, 3, 3, H, W), torch.ones(B, 2), T, M

def tmpdir(case):
    """A temporary directory on LaCie (never the Mac disk), removed after the test."""
    os.makedirs(LACIE_TMP, exist_ok=True); d = tempfile.TemporaryDirectory(dir=LACIE_TMP); case.addCleanup(d.cleanup)
    return Path(d.name)

def pe_row(key, fam, verdict=None, seed=0, twin_of=None, injection=None, zoom=None):
    """One pilot_eye/<split>.jsonl row in the shape build.mjs writes (Task 11), with the per-family target and mask rules."""
    g = f'{fam}:{seed}'
    if fam == 'Z':
        tags = [1 if i in zoom else 0 for i in range(10)]
        T = {'reasons': [], 'safe_actions': [], 'zoom_tags': tags, 'range_bin': seed % 5, 'log_ttc': 0, 'log_clear': 0, 'agl': 0, 'vs': 0, 'closing': 0, 'loc_dots': 0}
        M = {'verdict': 0, 'severity': 0, 'reasons': [0] * 26, 'actions': [0] * 11, 'p_ref': 0, 'reg': [0] * 6, 'zoom_tags': 1, 'range_bin': 1}
        return {'key': key, 'family': 'Z', 'frames': [f'raw/x/Z/{key}.png'], 'frame_dt_s': None, 'targets': T, 'masks': M, 'weight': 1.0, 'natural': True, 'sampler_weight': 1,
                'group': g, 'twin_of': None, 'injection': None, 'meta': {'route': None, 'sky': None, 'start': None, 'phase': None}}
    safe = [1 if (i == 0 and verdict == 0) or (i in (1, 7, 9) and verdict > 0) else 0 for i in range(11)]
    T = {'reasons': [1 if (i == FAM_REA[fam][0] and verdict == 2) else 0 for i in range(26)], 'safe_actions': safe, 'zoom_tags': [], 'range_bin': 0, 'verdict': verdict, 'severity': 2 * verdict,
         'log_ttc': 0.5 * verdict, 'log_clear': 1.0, 'agl': 0.2 if fam in 'AL' else 0, 'vs': -0.04 if fam == 'L' else 0, 'closing': 0, 'loc_dots': 0}
    if fam in 'SA': T['p_ref'] = verdict / 2
    M = {'verdict': 1, 'severity': 1, 'reasons': [1 if i in FAM_REA[fam] else 0 for i in range(26)], 'actions': [1 if i in FAM_ACT[fam] else 0 for i in range(11)], 'p_ref': int(fam in 'SA'),
         'reg': [int(fam in 'SA'), int(fam in 'SA'), int(fam in 'AL'), int(fam == 'L'), 0, 0], 'zoom_tags': 0, 'range_bin': 0}
    meta = {'route': 'alps' if fam == 'A' else None, 'sky': ['clear', 'fair'][seed % 2] if fam == 'A' else None, 'start': 'final' if fam in 'LD' else None, 'phase': 'H2' if fam == 'D' else None}
    return {'key': key, 'family': fam, 'frames': [f'raw/x/{fam}/{key}.f{i}.png' for i in range(3)], 'frame_dt_s': DT[fam], 'targets': T, 'masks': M, 'weight': 1.0 + verdict,
            'natural': not (twin_of or injection), 'sampler_weight': 1, 'group': g, 'twin_of': twin_of, 'injection': injection, 'meta': meta}

def split_rows(split):
    rows, n = [], {'train': 3, 'val': 2, 'test': 2}[split]
    for fam in 'SALD':
        for i in range(n):
            for v in ([0, 2] if fam != 'L' else [0, 1, 2]):
                rows.append(pe_row(f'{fam}_{split}_{i}_{v}', fam, v, seed=100 * i + v))
    rows.append(pe_row(f'S_{split}_orig', 'S', 2, seed=77, injection='hazard_spawn'))
    rows.append(pe_row(f'S_{split}_twin', 'S', 0, seed=77, twin_of=f'S_{split}_orig'))
    rows += [pe_row(f'Z_{split}_{i}', 'Z', seed=i, zoom=[i % 10, 3]) for i in range(2)]
    return rows

def make_dataset(root):
    """A small Task 11 dataset: UNSAFE frames are bright and SAFE ones dark (learnable), Z rows repeat one frame 3x in the cache,
    and the OOD split is empty (a zero-byte cache file, as the pilot's empty splits are)."""
    rng, root = np.random.default_rng(0), Path(root)
    for d in ['pilot_eye', 'cache']: (root / d).mkdir(parents=True, exist_ok=True)
    index, recs = {}, []
    for split in ['train', 'val', 'test', 'ood']:
        rows = split_rows(split) if split != 'ood' else []
        (root / 'pilot_eye' / f'{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        index[split], frames = {}, []
        for r in rows:
            index[split][r['key']] = len(frames)
            base = 40 + 80 * r['targets'].get('verdict', 1)
            for k in range(3): frames.append(np.clip(base + 10 * k + rng.normal(0, 8, (H, W, 3)), 0, 255).astype(np.uint8))
            recs.append({'key': r['key'], 'family': r['family'], 'split': split, 'pixel_identical_pair': False,
                         'safety': None if r['family'] == 'Z' else {'action_risk': {a: (0.0 if r['targets']['safe_actions'][i] else 1.0) for i, a in enumerate(ACTIONS[:7])}, 'p_best': 0.0},
                         'render': {'licence_profile': 'open'}})
        (root / 'cache' / f'pilot_eye_{split}.u8').write_bytes(np.stack(frames).tobytes() if frames else b'')
    (root / 'cache' / 'index.json').write_text(json.dumps({'size': [W, H], 'rows_per_record': 3, 'index': index}))
    (root / 'records.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in recs))
    (root / 'labels.json').write_text(json.dumps({'reasons': REASONS, 'actions': ACTIONS, 'verdicts': ['SAFE', 'CAUTION', 'UNSAFE'], 'nominal_frame_dt': {'S': 0.2, 'A': 0.2, 'L': 0.25, 'D': 2}, 'input': [W, H]}))
    (root / 'ATTRIBUTION.txt').write_text('synthetic test data\n')
    return root

class TestPilotEye(unittest.TestCase):
    def test_shapes_160x96_and_224x128(self):
        m = PilotEye(pretrained=False).eval()
        for H, W, h, w in [(96, 160, 3, 5), (128, 224, 4, 7)]:
            self.assertEqual(tuple(m.enc(torch.rand(1, 3, H, W)).shape), (1, 64, h, w))
            out = m(*batch(2, H, W)[:2]); self.assertEqual([o.shape[1] for o in out], list(HEADS.values()))
    def test_masked_heads_contribute_nothing(self):
        m = PilotEye(pretrained=False); f, dt, T, M = batch(); out = m(f, dt)
        zero = {k: torch.zeros_like(v) for k, v in M.items()}; self.assertEqual(loss_fn(out, T, zero).item(), 0.0)
        self.assertGreater(loss_fn(out, T, M).item(), 0.0)
    def test_heads_input_is_386(self):
        self.assertEqual(PilotEye(pretrained=False).heads.mlp[0].in_features, 386)

class TestCommon(unittest.TestCase):
    def test_read_jsonl_and_group_half(self):
        d = tmpdir(self); (d / 'a.jsonl').write_text('{"a": 1}\n\n{"a": 2}\n')
        self.assertEqual(read_jsonl(d / 'a.jsonl'), [{'a': 1}, {'a': 2}])
        self.assertEqual(group_half('S:1000'), zlib.crc32(b'S:1000') % 2); self.assertEqual({group_half(f'A:{i}') for i in range(20)}, {0, 1})

class TestData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs(LACIE_TMP, exist_ok=True); cls._d = tempfile.mkdtemp(dir=LACIE_TMP); cls.root = make_dataset(cls._d)
    @classmethod
    def tearDownClass(cls): shutil.rmtree(cls._d, ignore_errors=True)
    def test_item_shapes_frames_and_normalised_spacings(self):
        from vlm.train.pilot_eye.data import PilotEyeData
        ds = PilotEyeData(self.root, 'train'); i = next(k for k, r in enumerate(ds.rows) if r['family'] == 'D')
        f, dt, T, M, key = ds[i]
        self.assertEqual(tuple(f.shape), (3, 3, H, W)); self.assertEqual(key, ds.rows[i]['key'])
        mm = np.fromfile(self.root / 'cache' / 'pilot_eye_train.u8', dtype=np.uint8).reshape(-1, H, W, 3); s = ds.index[key]
        self.assertTrue(torch.allclose(f, torch.from_numpy(mm[s:s + 3].transpose(0, 3, 1, 2).astype(np.float32) / 255)))
        self.assertTrue(torch.allclose(dt, torch.tensor([1.0, 0.95])))
        self.assertEqual({k: tuple(v.shape) for k, v in T.items()}, {'verdict': (), 'severity': (), 'reasons': (26,), 'actions': (11,), 'p_ref': (), 'reg': (6,), 'tags': (10,), 'range': ()})
        self.assertEqual(float(M['p_ref']), 0.0); self.assertEqual(M['actions'].tolist(), [1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0])
    def test_zoom_rows_have_unit_spacing_and_only_zoom_masks(self):
        from vlm.train.pilot_eye.data import PilotEyeData
        ds = PilotEyeData(self.root, 'train'); i = next(k for k, r in enumerate(ds.rows) if r['family'] == 'Z')
        f, dt, T, M, _ = ds[i]
        self.assertEqual(dt.tolist(), [1.0, 1.0]); self.assertEqual(float(M['verdict']), 0.0); self.assertEqual(float(M['tags']), 1.0); self.assertEqual(float(M['range']), 1.0)
        self.assertEqual(T['tags'].sum().item(), 2.0); self.assertEqual(T['reasons'].tolist(), [0.0] * 26)
    def test_augment_keeps_shape_and_range(self):
        from vlm.train.pilot_eye.data import PilotEyeData
        ds = PilotEyeData(self.root, 'train', augment=True)
        for i in range(len(ds)):
            f = ds[i][0]; self.assertEqual(tuple(f.shape), (3, 3, H, W)); self.assertGreaterEqual(f.min().item(), 0.0); self.assertLessEqual(f.max().item(), 1.0)
    def test_fold_limit_empty_split_and_size_check(self):
        from vlm.train.pilot_eye.data import PilotEyeData
        full = PilotEyeData(self.root, 'train'); halves = [PilotEyeData(self.root, 'train', fold=k) for k in (0, 1)]
        self.assertEqual(sum(len(h) for h in halves), len(full)); self.assertTrue(all(group_half(r['group']) == 1 for r in halves[1].rows))
        self.assertEqual(len(PilotEyeData(self.root, 'train', limit=5)), 5); self.assertEqual(len(PilotEyeData(self.root, 'ood')), 0)
        with self.assertRaises(ValueError): PilotEyeData(self.root, 'train', H=128, W=224)
    def test_dataloader_collates_and_pickles_without_the_memmap(self):
        import pickle
        from torch.utils.data import DataLoader
        from vlm.train.pilot_eye.data import PilotEyeData
        ds = PilotEyeData(self.root, 'val'); ds[0]; self.assertIsNone(pickle.loads(pickle.dumps(ds)).mm)
        f, dt, T, M, keys = next(iter(DataLoader(ds, batch_size=4)))
        self.assertEqual(tuple(f.shape), (4, 3, 3, H, W)); self.assertEqual(tuple(T['reasons'].shape), (4, 26)); self.assertEqual(len(keys), 4)

class TestTrainExportEvaluate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs(LACIE_TMP, exist_ok=True); cls._d = tempfile.mkdtemp(dir=LACIE_TMP); cls.root = make_dataset(Path(cls._d) / 'ds')
        from vlm.train.pilot_eye import train
        cls.rundir = Path(cls._d) / 'run'
        train.main(['--data', str(cls.root), '--out', str(cls.rundir), '--steps', '4', '--batch', '4', '--workers', '0', '--device', 'cpu', '--no-pretrained', '--eval-every', '2'])
    @classmethod
    def tearDownClass(cls): shutil.rmtree(cls._d, ignore_errors=True)
    def test_train_writes_the_best_checkpoint_and_log(self):
        ck = torch.load(self.rundir / 'model.pt', map_location='cpu')
        self.assertEqual(ck['size'], '160x96'); self.assertIn('state', ck); self.assertIn(ck['best_step'], [2, 4])
        log = read_jsonl(self.rundir / 'train_log.jsonl')
        self.assertEqual([r['step'] for r in log if 'loss' in r], [0, 1, 2, 3]); self.assertEqual(sum('val_loss' in r for r in log), 2)
        self.assertEqual(log[-1]['stopped'], 'steps'); self.assertEqual(log[-1]['best_step'], ck['best_step'])
        m = PilotEye(pretrained=False); m.load_state_dict(ck['state'])
    def test_max_minutes_stops_early_and_still_saves(self):
        from vlm.train.pilot_eye import train
        out = tmpdir(self)
        train.main(['--data', str(self.root), '--out', str(out), '--steps', '50', '--batch', '4', '--workers', '0', '--device', 'cpu', '--no-pretrained', '--max-minutes', '0'])
        log = read_jsonl(out / 'train_log.jsonl'); self.assertEqual(log[-1]['stopped'], 'time'); self.assertEqual(sum('loss' in r for r in log), 1)
        self.assertTrue((out / 'model.pt').exists())
    def test_freeze_backbone_trains_only_the_reduction_and_heads(self):
        from vlm.train.pilot_eye import train
        out = tmpdir(self); torch.manual_seed(0); ref = PilotEye(pretrained=False).state_dict()  # main() seeds 0 before building the model
        train.main(['--data', str(self.root), '--out', str(out), '--steps', '3', '--batch', '4', '--workers', '0', '--device', 'cpu', '--no-pretrained', '--freeze-backbone'])
        ck = torch.load(out / 'model.pt', map_location='cpu'); st = ck['state']; self.assertTrue(ck['freeze_backbone'])
        bb = [k for k in st if k.startswith('enc.backbone.')]; self.assertTrue(bb)
        for k in bb: self.assertTrue(torch.equal(st[k], ref[k]), k)  # weights and BatchNorm running stats untouched
        self.assertFalse(all(torch.equal(st[k], ref[k]) for k in st if k.startswith('heads.')))
    def test_smoke_check_compares_the_first_and_last_losses(self):
        from vlm.train.pilot_eye.train import loss_fell
        self.assertTrue(loss_fell([3.0] * 20 + [1.0] * 20)[0]); self.assertFalse(loss_fell([1.0] * 20 + [3.0] * 20)[0]); self.assertTrue(loss_fell([2.0, 1.0])[0])
    def test_export_parity_card_and_bench(self):
        from vlm.train.pilot_eye import export_onnx
        import onnxruntime as ort
        out = tmpdir(self); export_onnx.main(['--ckpt', str(self.rundir / 'model.pt'), '--data', str(self.root), '--out', str(out)])
        for f in ['encoder.onnx', 'heads.onnx', 'labels.json', 'ATTRIBUTION.txt', 'MODEL_CARD.md', 'parity.json']: self.assertTrue((out / f).exists(), f)
        p = json.loads((out / 'parity.json').read_text()); self.assertEqual(p['size'], '160x96'); self.assertLessEqual(p['max_abs_diff'], 1e-3); self.assertEqual(p['parity_split'], 'val'); self.assertGreater(p['n'], 0)
        self.assertIn('mobilenetv4_conv_small', (out / 'MODEL_CARD.md').read_text())
        enc = ort.InferenceSession(str(out / 'encoder.onnx')); self.assertEqual(enc.run(None, {'pixels': np.zeros((1, 3, H, W), np.float32)})[0].shape, (1, 64, 3, 5))
        hd = ort.InferenceSession(str(out / 'heads.onnx')); self.assertEqual([o.name for o in hd.get_outputs()], list(HEADS))
        node = shutil.which('node')
        if node and (REPO / 'vlm' / 'node_modules' / 'onnxruntime-web').exists():
            r = subprocess.run([node, 'vlm/web/bench.mjs', 'pilot-eye', '--dir', str(out), '--runs', '5'], cwd=REPO, capture_output=True, text=True, timeout=180)
            self.assertEqual(r.returncode, 0, r.stderr); b = json.loads(r.stdout.strip().splitlines()[-1])
            self.assertEqual(b['size'], '160x96'); self.assertEqual(b['encoder']['runs'], 5)
            for k in ['median_ms', 'min_ms', 'p90_ms']: self.assertGreater(b['encoder'][k], 0)
            self.assertGreater(b['heads']['median_ms'], 0)
    def test_untrained_export_without_data_writes_size_only_parity(self):
        from vlm.train.pilot_eye import export_onnx
        out = tmpdir(self); export_onnx.main(['--untrained', '--no-pretrained', '--size', '224x128', '--out', str(out)])
        p = json.loads((out / 'parity.json').read_text()); self.assertEqual(p['size'], '224x128'); self.assertNotIn('max_abs_diff', p); self.assertLessEqual(p['max_abs_diff_random'], 1e-3)
    def test_int8_encoder_is_kept_only_within_the_auroc_budget(self):
        from vlm.train.pilot_eye import export_onnx
        out = tmpdir(self); export_onnx.main(['--ckpt', str(self.rundir / 'model.pt'), '--data', str(self.root), '--out', str(out), '--int8'])
        q = json.loads((out / 'parity.json').read_text())['int8']; self.assertEqual(set(q), {'fp32', 'int8', 'kept'}); self.assertIsNotNone(q['fp32'])
        self.assertEqual(q['kept'], q['int8'] is not None and q['fp32'] - q['int8'] <= 0.01); self.assertEqual((out / 'encoder_int8.onnx').exists(), q['kept'])
    def test_evaluate_metrics_and_val_confusion(self):
        from vlm.train.pilot_eye import evaluate
        out = tmpdir(self); evaluate.main(['--ckpt', str(self.rundir / 'model.pt'), '--data', str(self.root), '--split', 'test', '--out', str(out)])
        R = json.loads((out / 'metrics_test.json').read_text())
        for k in ['per_family', 'mean_family_auroc', 'pooled_auroc', 'baseline_auroc', 'pooled_macro_f1', 'baseline_macro_f1', 'action_top1_in_safe', 'regret_sa', 'pairs', 'zoom_tags_f1',
                  'zoom_range_acc', 'reg_mae', 'temperature', 'ece_natural_test', 'gates_failed']: self.assertIn(k, R)
        self.assertEqual(set(R['per_family']), set('SALD')); self.assertEqual(R['per_family']['L']['n'], 6); self.assertEqual(R['pairs']['n'], 1)
        C = json.loads((out / 'confusion.json').read_text()); V = ['SAFE', 'CAUTION', 'UNSAFE']
        self.assertEqual(C['split'], 'val'); self.assertEqual(C['labels'], V); self.assertEqual(sum(map(sum, C['counts'])), 20)
        self.assertEqual([sum(r) for r in C['counts']], [9, 2, 9])
        for i, v in enumerate(V):
            self.assertAlmostEqual(sum(C['row_normalised'][i]), 1.0, places=6); self.assertAlmostEqual(sum(C[v].values()), 1.0, places=6)
    def test_write_preds_writes_monitor_tuples_for_flight_rows(self):
        from vlm.train.pilot_eye import evaluate
        out = tmpdir(self); p = out / 'preds.jsonl'
        evaluate.main(['--ckpt', str(self.rundir / 'model.pt'), '--data', str(self.root), '--write-preds', str(p), '--keys-from', 'val'])
        rows = read_jsonl(p); self.assertEqual(len(rows), 20)
        for r in rows:
            m = r['monitor']; self.assertEqual(set(m), {'verdict', 'severity', 'reasons', 'action', 'p_ref', 'ttc_bin', 'clr_bin'}); self.assertIn(m['verdict'], ['SAFE', 'CAUTION', 'UNSAFE'])
            self.assertIn(m['action'], ACTIONS); self.assertTrue(all(x in REASONS for x in m['reasons']))
            if r['key'].startswith(('L', 'D')): self.assertEqual((m['p_ref'], m['ttc_bin'], m['clr_bin']), (None, 'none', 'none'))

class TestEvaluateHelpers(unittest.TestCase):
    def test_gate_a_needs_an_auroc_for_all_four_families(self):
        from vlm.train.pilot_eye.evaluate import gates
        fam = lambda d: {f: {'auroc': (None if f == d else 0.9)} for f in 'SALD'}
        R = {'per_family': fam('D'), 'mean_family_auroc': 0.9, 'pooled_auroc': 0.95, 'baseline_auroc': 0.5, 'pooled_macro_f1': 0.9, 'baseline_macro_f1': 0.3, 'pairs': {'frac': 0.9}}
        self.assertEqual([g for g in gates(R) if g.startswith('(a)')], ['(a) family D has no AUROC (one class or no rows in the split)'])
        R['per_family'] = fam(None); self.assertEqual(gates(R), [])
    def test_export_licence_is_the_build_profile_never_a_default(self):
        from vlm.train.pilot_eye.export_onnx import licence
        d = tmpdir(self); rec = lambda p: json.dumps({'render': {'licence_profile': p}}) + '\n'
        (d / 'records.jsonl').write_text(rec('nc') + rec('nc')); self.assertEqual(licence(str(d)), 'nc')
        (d / 'records.jsonl').write_text(rec('open') + rec('nc'))
        with self.assertRaises(SystemExit): licence(str(d))
        (d / 'stats.json').write_text(json.dumps({'build': {'licence': 'open'}})); self.assertEqual(licence(str(d)), 'open')
    def test_oof_rows_are_the_half_the_fold_model_did_not_train_on(self):
        from vlm.train.pilot_eye.evaluate import preds_rows
        rows = [{'key': str(i), 'group': f'S:{i}'} for i in range(20)]
        for k in (0, 1): self.assertTrue(all(group_half(r['group']) != k for r in preds_rows(rows, k, 'train')))
        self.assertEqual(len(preds_rows(rows, 0, 'train')) + len(preds_rows(rows, 1, 'train')), 20)
        with self.assertRaises(SystemExit): preds_rows(rows, None, 'train')
        self.assertEqual(preds_rows(rows, None, 'val'), rows)
    def test_pick_action_confusion_and_ece(self):
        from vlm.train.pilot_eye.evaluate import pick_action, confusion, ece
        p = np.zeros(11); self.assertEqual(pick_action(p, 'S'), 'NONE_SAFE'); p[[0, 3]] = 0.9; self.assertEqual(pick_action(p, 'S'), 'CONTINUE')
        p = np.zeros(11); p[9] = 0.8; p[1] = 0.95; self.assertEqual(pick_action(p, 'D'), 'BREAKOUT')
        C = confusion([0, 0, 2, 2], [0, 2, 2, 2]); self.assertEqual(C['counts'], [[1, 0, 1], [0, 0, 0], [0, 0, 2]]); self.assertEqual(C['row_normalised'][1], None)
        self.assertEqual(C['CAUTION'], {'SAFE': 0.125, 'CAUTION': 0.75, 'UNSAFE': 0.125}); self.assertEqual(C['fallback_rows'], ['CAUTION']); self.assertEqual(C['UNSAFE']['UNSAFE'], 1.0)
        self.assertAlmostEqual(ece(np.array([1.0, 1.0]), np.array([1, 1])), 0.0)

if __name__ == '__main__':
    unittest.main()
