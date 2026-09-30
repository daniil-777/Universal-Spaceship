import os
os.environ.setdefault('HF_HOME', '/Volumes/LaCie/astro-pilot/vlm/hf')
import contextlib, io, unittest
from pathlib import Path
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor
from vlm.train.common import EOU, PAD, IMAGE_IDS, check_row
from vlm.train.narrator.data import NarratorData, collate, frame_path
from vlm.train.narrator.lora_train import RX, lora_targets, parse_args, train_loop
from vlm.train.narrator.merge import frozen_hashes, frozen_mismatches
from vlm.train.narrator.generate import encode_prompt

G2 = os.environ.get('NARRATOR_G2_DIR', '/Volumes/LaCie/astro-pilot/vlm/models/narrator-base-g2-fused')
ROWS = os.environ.get('NARRATOR_ROWS', '/Volumes/LaCie/astro-pilot/vlm/datasets/smoke-t11/narrator/train.jsonl')

@unittest.skipUnless(Path(G2).exists() and Path(ROWS).exists(), 'the G2 folder or narrator rows are missing')
class TestNarratorData(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.p = AutoProcessor.from_pretrained(G2); cls.ds = NarratorData(ROWS, cls.p, limit=2)

    def test_item_is_one_512_image_with_the_prefix_mask(self):
        self.assertEqual(len(self.ds), 2); it = self.ds[1]; ids = it['input_ids'].tolist()
        self.assertEqual(tuple(it['pixel_values'].shape), (1, 3, 512, 512)); self.assertEqual(sum(t == 49190 for t in ids), 64)
        labels = it['labels'].tolist(); check_row(ids, labels, set(self.p.tokenizer.all_special_ids) | IMAGE_IDS)
        self.assertEqual([l for l in labels if l != -100][-1], EOU)
        answer = self.p.tokenizer.decode([l for l in labels if l != -100], skip_special_tokens=True).strip()
        self.assertEqual(answer, self.ds.rows[1]['messages'][1]['content'][0]['text'])

    def test_collate_pads_ids_masks_and_labels(self):
        a, b = self.ds[0], self.ds[1]; out = collate([a, b]); n = max(len(a['input_ids']), len(b['input_ids']))
        self.assertEqual(tuple(out['input_ids'].shape), (2, n)); self.assertEqual(tuple(out['pixel_values'].shape), (2, 1, 3, 512, 512))
        short = 0 if len(a['input_ids']) < len(b['input_ids']) else 1; k = len((a, b)[short]['input_ids'])
        if k < n:
            self.assertTrue((out['input_ids'][short, k:] == PAD).all()); self.assertTrue((out['attention_mask'][short, k:] == 0).all()); self.assertTrue((out['labels'][short, k:] == -100).all())

    def test_check_rows_reports_every_row(self):
        from vlm.train.narrator.data import check_rows
        r = check_rows(ROWS, self.p, limit=3); self.assertEqual((r['rows'], r['failed']), (3, [])); self.assertGreater(r['max_len'], 64)

    def test_generate_prompt_matches_the_training_prefix(self):
        from PIL import Image
        r = self.ds.rows[0]; inp = encode_prompt(self.p, r['messages'][0]['content'][1]['text'], Image.open(frame_path(r['images'][0])))
        self.assertEqual(int((inp['input_ids'] == 49190).sum()), 64)
        self.assertTrue(self.p.tokenizer.decode(inp['input_ids'][0]).rstrip().endswith('Assistant:'))

    def test_onnx_deploy_path_greedy(self):
        from PIL import Image
        from vlm.train.narrator.generate import OnnxNarrator
        r = self.ds.rows[0]; inp = encode_prompt(self.p, 'Describe the image in one sentence.', Image.open(frame_path(r['images'][0])))
        toks = OnnxNarrator(G2, decoder='q4f16', vision='q8').generate(inp, max_new=4)
        self.assertTrue(1 <= len(toks) <= 4 and all(isinstance(t, int) for t in toks)); self.assertTrue(len(toks) == 4 or toks[-1] == EOU)
        self.assertTrue(self.p.tokenizer.decode(toks, skip_special_tokens=True).strip())

class TestPaths(unittest.TestCase):
    def test_frame_path(self):
        self.assertEqual(frame_path('raw/r/Z/k.png'), '/Volumes/LaCie/astro-pilot/vlm/raw/r/Z/k.png'); self.assertEqual(frame_path('/a/b.png'), '/a/b.png')

@unittest.skipUnless(Path(G2).exists(), 'the G2 folder is missing')
class TestLoraTargets(unittest.TestCase):
    def test_regex_resolves_exactly_210_text_projections(self):
        with torch.device('meta'): m = AutoModelForVision2Seq.from_config(AutoConfig.from_pretrained(G2))
        names = lora_targets(m); self.assertEqual(len(names), 210)
        self.assertTrue(all(n.startswith('model.text_model.layers.') for n in names))
        self.assertFalse(any('vision_model' in n or 'connector' in n for n in names))

class TestArgs(unittest.TestCase):
    def test_smoke_defaults_and_overrides(self):
        a = parse_args(['--data', 'd', '--out', 'o', '--processor-dir', 'p', '--smoke'])
        self.assertEqual((a.steps, a.max_rows, a.max_minutes), (100, 400, None))
        a = parse_args(['--data', 'd', '--out', 'o', '--processor-dir', 'p', '--smoke', '--steps', '20', '--max-rows', '40', '--max-minutes', '4.5', '--device', 'cpu'])
        self.assertEqual((a.steps, a.max_rows, a.max_minutes, a.device), (20, 40, 4.5, 'cpu'))
        a = parse_args(['--data', 'd', '--out', 'o', '--processor-dir', 'p']); self.assertEqual((a.steps, a.max_rows), (3000, None))
        with contextlib.redirect_stderr(io.StringIO()):   # argparse prints its usage on the rejected values
            with self.assertRaises(SystemExit): parse_args(['--data', 'd', '--out', 'o', '--processor-dir', 'p', '--device', 'cuda'])
            with self.assertRaises(SystemExit): parse_args(['--data', 'd', '--out', 'o', '--processor-dir', 'p', '--batch', '3'])

class Toy(nn.Module):
    def __init__(self): super().__init__(); self.w = nn.Linear(2, 1)
    def forward(self, x, y): return type('Out', (), {'loss': ((self.w(x) - y) ** 2).mean()})()

class TestTrainLoop(unittest.TestCase):
    def batches(self, n=3): return [{'x': torch.randn(2, 2), 'y': torch.randn(2, 1)} for _ in range(n)]

    def test_runs_the_requested_steps_across_epochs(self):
        m = Toy(); opt = torch.optim.SGD(m.parameters(), lr=0.01); logs = []
        r = train_loop(m, self.batches(3), opt, steps=7, dev='cpu', log=logs.append)
        self.assertEqual((r['steps'], r['stop'], len(r['losses'])), (7, 'steps', 7)); self.assertTrue(any(l.startswith('step 7 ') for l in logs))

    def test_wall_clock_stop(self):
        t = [0.0]
        def clock(): t[0] += 30.0; return t[0]
        m = Toy(); opt = torch.optim.SGD(m.parameters(), lr=0.01)
        r = train_loop(m, self.batches(3), opt, steps=1000, dev='cpu', max_minutes=2, clock=clock, log=lambda s: None)
        self.assertEqual(r['stop'], 'time'); self.assertLess(r['steps'], 10)

    def test_empty_loader_fails(self):
        m = Toy(); opt = torch.optim.SGD(m.parameters(), lr=0.01)
        with self.assertRaises(AssertionError): train_loop(m, [], opt, steps=5, dev='cpu', log=lambda s: None)

class TestMergeChecks(unittest.TestCase):
    def test_frozen_hashes_and_mismatches(self):
        sd = {'model.vision_model.a': torch.ones(3), 'model.connector.b': torch.zeros(2), 'model.text_model.embed_tokens.weight': torch.ones(2, 2),
              'model.text_model.layers.0.self_attn.q_proj.weight': torch.ones(2, 2), 'lm_head.weight': torch.ones(2)}
        base = frozen_hashes(sd); self.assertEqual(sorted(base), ['model.connector.b', 'model.text_model.embed_tokens.weight', 'model.vision_model.a'])
        same = dict(sd, **{'model.text_model.layers.0.self_attn.q_proj.weight': torch.zeros(2, 2)})
        self.assertEqual(frozen_mismatches(base, frozen_hashes(same)), [])
        changed = dict(sd, **{'model.connector.b': torch.tensor([0.0, 1e-7])}); self.assertEqual(frozen_mismatches(base, frozen_hashes(changed)), ['model.connector.b'])
        missing = {k: v for k, v in sd.items() if k != 'model.vision_model.a'}; self.assertEqual(frozen_mismatches(base, frozen_hashes(missing)), ['model.vision_model.a'])

if __name__ == '__main__':
    unittest.main()
