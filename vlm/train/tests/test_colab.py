"""The Colab notebook's code (vlm/train/colab): the GPU plans, the stratified Narrator epoch, the package sync from Drive
(SHA-256 checks, cache parts joined, shards unpacked), Pilot Eye training with resume on a synthetic dataset and its export
with the Node parity sample, Narrator LoRA steps with resume on real rows, and the apv-models-v1.zip layout."""
import os
os.environ.setdefault('HF_HOME', '/Volumes/LaCie/astro-pilot/vlm/hf')
import hashlib, io, json, tarfile, unittest, zipfile
from collections import Counter
from pathlib import Path
from vlm.train.colab import env, pack_io, package
from vlm.train.tests.test_pilot_eye import make_dataset, tmpdir

CPU = {'gpu': {'kind': 'cpu', 'name': 'cpu', 'mem_gb': 0.0, 'bf16': False, 'device': 'cpu'}, 'dtype': 'fp32', 'workers': 0, 'narrator': {'batch': 1, 'accum': 2, 'lr': 2e-4}, 'eye': {'batch': 8}}
G2 = '/Volumes/LaCie/astro-pilot/vlm/models/narrator-base-g2-fused'
SMOKE = '/Volumes/LaCie/astro-pilot/vlm/datasets/smoke-t11'
quiet = lambda *a, **k: None

class TestPlans(unittest.TestCase):
    def test_gpu_kinds_get_their_precision_batch_and_workers(self):
        g = lambda kind, mem, bf16: {'kind': kind, 'name': kind, 'mem_gb': mem, 'bf16': bf16, 'device': 'cuda'}
        a, l, t = env.plan_for(g('A100', 40, True), 12), env.plan_for(g('L4', 22.5, True), 8), env.plan_for(g('T4', 15, False), 2)
        self.assertEqual((a['dtype'], a['narrator']['batch'], a['narrator']['accum']), ('bf16', 32, 1))
        self.assertEqual((l['dtype'], l['narrator']['batch']), ('bf16', 16)); self.assertEqual((t['dtype'], t['narrator']['batch'], t['narrator']['accum']), ('fp16', 8, 2))
        self.assertEqual((a['workers'], t['workers']), (8, 1)); self.assertEqual(env.plan_for(env.detect_gpu() if False else CPU['gpu'])['dtype'], 'fp32')
    def test_pip_install_falls_back_and_reports(self):
        calls = []
        def run(cmd, **kw):
            calls.append(cmd[-1]); return type('R', (), {'returncode': 0 if cmd[-1].startswith('b') else 1, 'stderr': 'no wheel'})()
        self.assertEqual(env.pip_install(['a==1|b>=1'], run=run, log=quiet), ['b>=1']); self.assertEqual(calls, ['a==1', 'b>=1'])
        with self.assertRaises(RuntimeError): env.pip_install(['a==1'], run=run, log=quiet)
        self.assertEqual(env.pip_install(['a==1'], run=run, log=quiet, required=False), [])

class TestStratified(unittest.TestCase):
    def test_allocation_caps_small_strata_and_sums_to_n(self):
        al = pack_io.allocate({'a': 1000, 'b': 10, 'c': 100}, 500)
        self.assertEqual(sum(al.values()), 500); self.assertLessEqual(al['b'], 40); self.assertTrue(all(v > 0 for v in al.values()))
        self.assertEqual(sum(pack_io.allocate({'a': 3, 'b': 2}, 100).values()), 20, 'at most max_repeat passes of every row')
    def test_every_prefix_keeps_the_mix(self):
        rows = [{'family': f, 'task': t} for f, t, n in [('S', 'vqa', 3000), ('S', 'grounding', 600), ('Z', 'vqa', 400), ('Z', 'caption_detail', 50)] for _ in range(n)]
        order = pack_io.stratified_order(rows, 2000, seed=1); full = Counter((rows[i]['family'], rows[i]['task']) for i in order)
        self.assertEqual(len(order), 2000); self.assertEqual(order, pack_io.stratified_order(rows, 2000, seed=1), 'deterministic')
        head = Counter((rows[i]['family'], rows[i]['task']) for i in order[:200])
        for k, n in full.items(): self.assertLess(abs(head[k] / 200 - n / 2000), 0.03, k)
        self.assertGreater(full[('Z', 'caption_detail')] / 2000, 50 / 4050, 'small strata count for more than their raw share')

def write_package(d):
    """A package in the colab_pack.mjs layout: manifest, SHA256SUMS, files, two cache parts and one shard."""
    d = Path(d); files = {'labels.json': b'{}', 'narrator/train.jsonl': b'{"images": ["narrator-train-00000/k1.jpg"]}\n', 'cache/index.json': b'{"index": {}}', 'code/x.py': b'print(1)\n'}
    parts = {'cache/parts/pilot_eye_train.u8.000': b'\x01' * 10, 'cache/parts/pilot_eye_train.u8.001': b'\x02' * 7}
    tb = io.BytesIO()
    with tarfile.open(fileobj=tb, mode='w') as t:
        for n, b in [('k1.jpg', b'JPEG1'), ('k1.json', b'{}'), ('k2.jpg', b'JPEG2')]: ti = tarfile.TarInfo(n); ti.size = len(b); t.addfile(ti, io.BytesIO(b))
    shards = {'shards/narrator-train-00000.tar': tb.getvalue()}
    for rel, b in {**files, **parts, **shards}.items(): (d / rel).parent.mkdir(parents=True, exist_ok=True); (d / rel).write_bytes(b)
    man = {'id': 'pkg1', 'files': sorted(files) + ['manifest.json'], 'eye_parts': {'train': sorted(parts)}, 'shards': [{'name': 'narrator-train-00000', 'path': 'shards/narrator-train-00000.tar', 'split': 'train', 'n': 2}]}
    (d / 'manifest.json').write_text(json.dumps(man))
    sums = {rel: hashlib.sha256((d / rel).read_bytes()).hexdigest() for rel in [*files, *parts, *shards, 'manifest.json']}
    (d / 'SHA256SUMS').write_text(''.join(f'{h}  {rel}\n' for rel, h in sorted(sums.items())))
    return d

class TestSync(unittest.TestCase):
    def test_copy_verify_join_unpack_and_skip_when_done(self):
        src, dst = write_package(tmpdir(self)), tmpdir(self) / 'local'
        man = pack_io.sync_package(str(src), str(dst), log=quiet); self.assertEqual(man['id'], 'pkg1')
        self.assertEqual((dst / 'narrator/train.jsonl').read_bytes(), (src / 'narrator/train.jsonl').read_bytes())
        self.assertEqual((dst / 'cache/pilot_eye_train.u8').read_bytes(), b'\x01' * 10 + b'\x02' * 7)
        self.assertEqual((dst / 'frames/narrator-train-00000/k1.jpg').read_bytes(), b'JPEG1'); self.assertFalse((dst / 'shards_tmp').exists())
        said = []; pack_io.sync_package(str(src), str(dst), log=said.append); self.assertIn('already', said[0])
    def test_a_corrupt_file_stops_the_sync(self):
        src, dst = write_package(tmpdir(self)), tmpdir(self) / 'local'
        (src / 'labels.json').write_bytes(b'{"x": 1}')
        with self.assertRaises(IOError): pack_io.sync_package(str(src), str(dst), log=quiet)
        self.assertFalse((dst / '.synced.json').exists())

class TestEye(unittest.TestCase):
    def test_train_resume_export_and_node_parity_sample(self):
        from vlm.train.colab import eye
        data, out, ex = make_dataset(tmpdir(self) / 'ds'), tmpdir(self) / 'run', tmpdir(self) / 'export'
        r = eye.train_eye(str(data), str(out), CPU, budget_min=5, max_steps=2, pretrained=False, log=quiet)
        self.assertEqual((r['steps'], r['stopped']), (2, 'max_steps')); self.assertTrue((out / 'model.pt').exists() and (out / 'state.pt').exists())
        self.assertEqual(eye.train_eye(str(data), str(out), CPU, log=quiet), r, 'a finished run is not trained again')
        os.remove(out / 'DONE.json'); r2 = eye.train_eye(str(data), str(out), CPU, budget_min=5, max_steps=4, pretrained=False, log=quiet)
        self.assertEqual(r2['steps'], 4, 'resumed from step 2')
        s = eye.evaluate_and_export(str(data), str(out), str(ex), log=quiet)
        for f in ('encoder.onnx', 'heads.onnx', 'labels.json', 'parity.json', 'parity_sample.json', 'metrics_test.json'): self.assertTrue((ex / f).exists(), f)
        ps = json.loads((ex / 'parity_sample.json').read_text()); self.assertEqual(len(ps['outputs']['verdict']), 3); self.assertLessEqual(s['parity_max_abs_diff'], 1e-3)
        self.assertAlmostEqual(float(eye.node_input(4, 0)[1]), 158 / 255, places=6, msg='imul(1, 2654435761) >>> 24 = 158, as install_models.mjs computes')

@unittest.skipUnless(Path(G2).exists() and Path(f'{SMOKE}/narrator/train.jsonl').exists(), 'needs the G2 processor folder and the smoke-t11 rows')
class TestNarrator(unittest.TestCase):
    def test_lora_steps_checkpoint_and_resume(self):
        from vlm.train.colab import narrator
        out = tmpdir(self) / 'nar'
        r = narrator.train_narrator(SMOKE, str(out), G2, CPU, budget_min=10, epoch_rows=6, max_steps=1, val_n=2, log=quiet)
        self.assertEqual((r['steps'], r['stopped']), (1, 'max_steps')); self.assertTrue((out / 'adapter/adapter_model.safetensors').exists() and (out / 'state.pt').exists())
        self.assertIsNotNone(r['last_val_loss'])
        os.remove(out / 'DONE.json'); r2 = narrator.train_narrator(SMOKE, str(out), G2, CPU, budget_min=10, epoch_rows=6, max_steps=2, val_n=2, log=quiet)
        self.assertEqual((r2['steps'], r2['rows_seen']), (2, 4), 'resumed at row 2 of the stratified epoch')

class TestZip(unittest.TestCase):
    def test_zip_has_the_models_the_metrics_and_a_manifest_without_the_fp32_reference(self):
        d = tmpdir(self); eye, nar = d / 'pe', d / 'nr'
        for f in package.REQUIRED['pilot-eye-v1']: (eye / f).parent.mkdir(parents=True, exist_ok=True); (eye / f).write_bytes(b'x')
        for f in package.REQUIRED['narrator-v1'] + ['onnx/decoder_model_merged.onnx', 'parity/0.png']: (nar / f).parent.mkdir(parents=True, exist_ok=True); (nar / f).write_bytes(b'y')
        man = package.make_zip(str(d / 'm.zip'), str(eye), str(nar), {'eye': {'auroc': 0.9}}, dataset_id='pkg1', log=quiet)
        z = zipfile.ZipFile(d / 'm.zip'); names = set(z.namelist())
        self.assertIn('narrator-v1/onnx/decoder_model_merged_q4f16.onnx', names); self.assertNotIn('narrator-v1/onnx/decoder_model_merged.onnx', names)
        self.assertIn('metrics/eye.json', names); m = json.loads(z.read('MANIFEST.json')); self.assertEqual(m['dataset'], 'pkg1')
        self.assertEqual(m['files']['pilot-eye-v1/encoder.onnx']['sha256'], hashlib.sha256(b'x').hexdigest()); self.assertEqual(man['files'], m['files'])
        os.remove(eye / 'heads.onnx')
        with self.assertRaises(FileNotFoundError): package.make_zip(str(d / 'n.zip'), str(eye), str(nar), {}, log=quiet)
        self.assertIn('install_models.mjs', package.instructions('/content/drive/MyDrive/apv-open-v1/apv-models-v1.zip'))

if __name__ == '__main__': unittest.main()
