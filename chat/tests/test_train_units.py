"""chat/tests/test_train_units.py — fast checks of the training plumbing (no model weights; the LFM2.5 tokenizer from the HF cache for the
label-mask test): hyper-parameter tables, the label mask on a real dev-package row, DPO pair selection, GRPO reward wrappers, data
loading / general-replay mixing, Run's DONE markers + Drive mirror + resume, eval metrics, the notebook generator.
  HF_HOME=/Volumes/LaCie/astro-pilot/chat/hf /Volumes/LaCie/astro-pilot/chat/venv/bin/python -m unittest chat.tests.test_train_units"""
import json, os, tempfile, unittest, zipfile
from pathlib import Path
from chat.train import common
from chat.train.common import (BASE, MODELS, SMOKE, TIER, Notes, Run, clean_visitor, find_package, hp, mix_general, prompt_rows,
                               read_jsonl, sft_rows, sim_messages, verify_manifest)

DEV = Path('/Volumes/LaCie/astro-pilot/chat/data/capcom-dev')
TIERS = ('t4', 'l4', 'a100', 'h100', 'big', 'mps', 'cpu')
CPU = {'tier': 'cpu', 'device': 'cpu', 'name': 'cpu', 'mem_gb': None, 'cc': None, 'bf16': False, 'fp16': False}

def conv(u, a):
    return {'prompt': [{'role': 'system', 'content': 'S'}, {'role': 'user', 'content': u}], 'completion': [{'role': 'assistant', 'content': a}]}

class TestHP(unittest.TestCase):
    def test_every_stage_and_tier(self):
        for stage in BASE:
            for tier in TIERS:
                for smoke in (False, True):
                    h = hp(stage, tier, smoke)
                    self.assertTrue(all(v is not None or k.startswith(('n_', 'max_rows')) for k, v in h.items()), (stage, tier, h))
    def test_spec_values(self):
        s2 = hp('s2', 'a100'); self.assertEqual((s2['lr'], s2['epochs'], s2['bs'] * s2['ga'], s2['max_len'], s2['warmup']), (3e-5, 3, 64, 1024, 0.03))
        for tier in ('t4', 'l4', 'h100'): self.assertEqual(hp('s2', tier)['bs'] * hp('s2', tier)['ga'], 64, tier)
        s1 = hp('s1', 'a100'); self.assertEqual((s1['lora_r'], s1['lora_alpha'], s1['lr'], s1['epochs']), (64, 128, 1e-4, 2))
        s3 = hp('s3', 'a100'); self.assertEqual((s3['beta'], s3['lr'], s3['temperature'], s3['max_completion']), (1.0, 1e-5, 1.0, 128))
        s4 = hp('s4', 'a100'); self.assertEqual((s4['k'], s4['temperature'], s4['margin'], s4['beta'], s4['lr'], s4['epochs'], s4['rounds']),
                                                (4, 0.8, 0.15, 0.3, 1e-6, 2, 2))
        self.assertEqual(hp('s4', 't4')['rounds'], 1)
        s5 = hp('s5', 'a100'); self.assertEqual((s5['num_generations'], s5['beta'], s5['epsilon_high'], s5['lr'], s5['max_steps']), (4, 0.02, 0.28, 1e-6, 200))
        self.assertEqual(hp('s5', 't4')['num_generations'], 2)
        self.assertEqual(hp('s2', 'a100', student='smol135')['lr'], 5e-5)
    def test_grpo_batches_divide(self):
        for tier in TIERS:
            for smoke in (False, True):
                h = hp('s5', tier, smoke); self.assertEqual(h['bs'] * h['ga'] % h['num_generations'], 0, (tier, smoke))
    def test_smoke_is_tiny(self):
        for stage, o in SMOKE.items():
            h = hp(stage, 'a100', True)
            for k in ('max_steps', 'n_prompts', 'n_ctx', 'max_rows', 'n_val'):
                if k in o: self.assertLessEqual(h[k], 32, (stage, k))
    def test_stage_defaults_and_models(self):
        self.assertFalse(common.default_stages('t4')['S5']); self.assertTrue(common.default_stages('a100')['S5'])
        self.assertEqual(MODELS['lfm350']['student'], 'LiquidAI/LFM2.5-350M'); self.assertEqual(set(MODELS), {'lfm350', 'smol135', 'granite350'})
        self.assertEqual(set(TIER), set(TIERS))

@unittest.skipUnless((DEV / 'sft/capcom_train.jsonl').exists(), 'dev package missing')
class TestLabelMask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('HF_HOME', '/Volumes/LaCie/astro-pilot/chat/hf'); os.environ.setdefault('HF_HUB_OFFLINE', '1')
        cls.tok = common.load_tok('LiquidAI/LFM2.5-350M', 'right')
        cls.rows = read_jsonl(DEV / 'sft/capcom_train.jsonl')
    def test_real_row(self):
        from chat.train.sft_stage import pc_tokens
        for r in (self.rows[0], self.rows[7], self.rows[-1]):
            t = pc_tokens(self.tok, r['prompt'], r['completion'])
            lab = [x for x in t['labels'] if x != -100]; n0 = t['labels'].index(lab[0])
            self.assertEqual(self.tok.decode(lab), r['completion'][0]['content'] + '<|im_end|>')  # exactly the reply + end of turn
            self.assertEqual(lab[-1], self.tok.convert_tokens_to_ids('<|im_end|>'))
            self.assertTrue(all(x == -100 for x in t['labels'][:n0]) and all(x != -100 for x in t['labels'][n0:]))
            self.assertEqual(t['input_ids'][:n0], common.chat_ids(self.tok, r['prompt']))  # the prompt as generation sees it
            self.assertEqual(len(t['input_ids']), len(t['labels']))
            self.assertIsNone(pc_tokens(self.tok, r['prompt'], r['completion'], max_len=10))
    def test_multi_turn_history_not_trained(self):
        from chat.train.sft_stage import pc_tokens
        r = next(x for x in self.rows if len(x['prompt']) >= 4)  # system + history + user
        t = pc_tokens(self.tok, r['prompt'], r['completion'])
        self.assertNotIn(r['prompt'][2]['content'], self.tok.decode([x for x in t['labels'] if x != -100]))

class TestPairs(unittest.TestCase):
    def test_margin(self):
        from chat.train.dpo_stage import select_pairs
        g = lambda *s: [{'text': f't{i}', 'score': x} for i, x in enumerate(s)]
        groups = [g(0.9, 0.5, 0.7, 0.6), g(0.50, 0.40, 0.45), g(0.5), g(0.5, 0.5), [{'text': 'a', 'score': 0.9}, {'text': 'a', 'score': 0.1}]]
        p = select_pairs(groups, 0.15)
        self.assertEqual([(i, c, r) for i, c, r, _ in p], [(0, 't0', 't1')]); self.assertAlmostEqual(p[0][3], 0.4)
        self.assertEqual([x[0] for x in select_pairs(groups, 0.05)], [0, 1])
        self.assertEqual([x[0] for x in select_pairs(groups, 0.0)], [0, 1])  # ties and identical texts never pair
    def test_history(self):
        from chat.train.dpo_stage import history
        row = {'prompt': [{'role': 'system', 'content': 'S'}, {'role': 'user', 'content': 'u1'}, {'role': 'assistant', 'content': 'a1'},
                          {'role': 'user', 'content': 'u2'}]}
        self.assertEqual([m['content'] for m in history(row, 'c')], ['u1', 'a1', 'u2', 'c'])

class TestRewards(unittest.TestCase):
    def test_wrappers(self):
        from chat.rewards import W, score
        from chat.train.grpo_stage import COMPONENTS, reward_funcs
        fns, w = reward_funcs()
        self.assertEqual([f.__name__ for f in fns], [f'reward_{c}' for c in COMPONENTS]); self.assertAlmostEqual(sum(w), sum(W.values()))
        notes = ['The pilot is a PPO policy that commands body rates at 15 Hz.']
        comps = [[{'role': 'assistant', 'content': 'It is a PPO policy running at 15 Hz. Want to watch it dodge rocks?'}],
                 [{'role': 'assistant', 'content': "That's not in my flight notes, sorry. Want to try the landing instead?"}], 'plain 99 text']
        kw = dict(prompts=[[]] * 3, completions=comps, notes=[notes] * 3, user=['how does it fly'] * 3, must=[['ppo', 'policy'], [], []],
                  abstain=[False, True, False], lead=[True, True, True], trainer_state=None)
        out = {f.__name__: f(**kw) for f in fns}
        self.assertEqual(out['reward_recall'][0], 1.0); self.assertIsNone(out['reward_recall'][1])  # recall does not apply to abstain rows
        self.assertEqual(out['reward_abstain'][1], 1.0); self.assertEqual(out['reward_grounded'][2], 0.5)  # 99 is not in the notes
        s = score(comps[0][0]['content'], {'notes': notes, 'user': 'how does it fly', 'must': ['ppo', 'policy'], 'abstain': False, 'lead': True})
        for c in ('recall', 'grounded', 'lead', 'first', 'length'): self.assertEqual(out[f'reward_{c}'][0], s[c])
        self.assertTrue(all(v is None or 0 <= v <= 1 for vals in out.values() for v in vals))

class TestData(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()); pkg = self.tmp / 'data' / 'capcom-x'; (pkg / 'sft').mkdir(parents=True); (pkg / 'prompts').mkdir()
        w = lambda p, rows: p.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        w(pkg / 'sft/capcom_train.jsonl', [conv(f'q{i}', f'a{i}') | {'meta': {'x': i}} for i in range(30)])
        w(pkg / 'sft/capcom_val.jsonl', [conv('vq', 'va')]); w(pkg / 'sft/general.jsonl', [conv(f'g{i}', f'ga{i}') for i in range(100)])
        w(pkg / 'prompts/train.jsonl', [{'prompt': [], 'user': f'u{i}'} for i in range(20)]); (pkg / 'eval').mkdir()
        (pkg / 'kb.json').write_text(json.dumps({'retrieval': {'min_score': 1.0}, 'facts': [
            {'id': 'A', 'title': 'Landing', 'text': 'The airliner autoland flares at 50 ft.', 'keywords': ['landing', 'flare']},
            {'id': 'B', 'title': 'Pilot', 'text': 'The PPO pilot commands body rates.', 'keywords': ['ppo', 'pilot']}]}))
        files = {str(p.relative_to(pkg)): {'sha256': common.sha256(p), 'rows': None} for p in pkg.rglob('*') if p.is_file()}
        (pkg / 'MANIFEST.json').write_text(json.dumps({'version': 'x', 'files': files})); self.pkg = pkg
    def test_mix(self):
        cap, gen = [{'c': i} for i in range(30)], [{'g': i} for i in range(100)]
        rows = mix_general(cap, gen, 0.25, seed=1)
        self.assertEqual((len(rows), sum('g' in r for r in rows)), (40, 10))  # general = 25 % of the mix
        self.assertEqual(rows, mix_general(cap, gen, 0.25, seed=1)); self.assertNotEqual(rows, mix_general(cap, gen, 0.25, seed=2))
        self.assertEqual(sum('g' in r for r in mix_general(cap, gen[:4], 0.25)), 4)  # capped by what exists
        self.assertEqual(len(mix_general(cap, gen, 0.0)), 30)
    def test_package(self):
        self.assertEqual(find_package(self.tmp / 'data'), self.pkg); self.assertEqual(verify_manifest(self.pkg), [])
        rows = sft_rows(self.pkg, 'train', 0.25, seed=0); self.assertEqual(len(rows), 40); self.assertEqual(set(rows[0]), {'prompt', 'completion'})
        self.assertEqual(len(sft_rows(self.pkg, 'train', 0.25, max_rows=7)), 7); self.assertEqual(len(sft_rows(self.pkg, 'val')), 1)
        self.assertEqual(prompt_rows(self.pkg, 'train', 5, seed=3), prompt_rows(self.pkg, 'train', 5, seed=3))
        self.assertEqual(read_jsonl(self.pkg / 'eval/single.jsonl'), [])  # missing -> []
        (self.pkg / 'eval/x.jsonl').write_text(json.dumps({'t': 'a\u2028b'}, ensure_ascii=False) + '\n\n')
        self.assertEqual(read_jsonl(self.pkg / 'eval/x.jsonl'), [{'t': 'a\u2028b'}])  # U+2028 inside a string is not a line break
        (self.pkg / 'sft/capcom_val.jsonl').write_text('tampered\n')
        self.assertEqual(verify_manifest(self.pkg), [('sft/capcom_val.jsonl', 'sha256')])
    def test_notes_and_simulator(self):
        n = Notes(json.loads((self.pkg / 'kb.json').read_text()))
        self.assertEqual(n('how does the landing flare work'), ['The airliner autoland flares at 50 ft.'])
        self.assertEqual(n('pizza recipes'), [])
        m = sim_messages([{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'Hello! Want a tour?'}], 'kid', 'see rockets.', 'typos', 'ask about the moon')
        self.assertIn('Persona: kid.', m[0]['content']); self.assertIn('What you want next: ask about the moon.', m[0]['content'])
        self.assertIn('CAPCOM: Hello! Want a tour?', m[1]['content'])
        self.assertEqual(clean_visitor('Visitor: "cool, and the moon?"\nCAPCOM: sure'), 'cool, and the moon?')
        self.assertEqual(clean_visitor('  \n'), 'ok, what else can I try?')

KB = {'retrieval': {'min_score': 1.0}, 'facts': [{'id': 'B', 'title': 'Pilot', 'text': 'The PPO pilot commands body rates at 15 Hz.',
                                                  'keywords': ['ppo', 'pilot']}]}

class TestSimulation(unittest.TestCase):
    """simulate_dialogs and the S4 rollouts with generation stubbed (no weights): turn order, intents, futures, scores."""
    def test_dialogs(self):
        import chat.eval as ev
        calls = []
        def fake(model, tok, prompts, n=1, **kw):
            calls.append(model)
            return ['Visitor: "and the moon?"' if model == 'sim' else 'The PPO pilot commands body rates. Want to watch it fly?'] * len(prompts)
        scripts = [{'id': 'E3-1', 'persona': 'kid', 'style': 'typos', 'goal': 'g', 'opening': 'how does the ppo pilot fly',
                    'intents': ['ask about the moon'], 'state': {'scene': 'belt', 'seen': []}},
                   {'id': 'E3-2', 'turns': [{'role': 'user', 'content': 'hi'}]}]
        orig, ev.generate = ev.generate, fake
        try: m, tr = ev.simulate_dialogs('pol', None, 'sim', None, scripts, Notes(KB), turns=3)
        finally: ev.generate = orig
        self.assertEqual((m['n'], len(m['per_turn'])), (6, 3)); self.assertEqual(calls, ['pol', 'sim', 'pol', 'sim', 'pol'])
        self.assertEqual([t['visitor'] for t in tr[0]['turns']], ['how does the ppo pilot fly', 'and the moon?', 'and the moon?'])
        self.assertEqual(tr[1]['turns'][0]['visitor'], 'hi'); self.assertEqual(m['lead_ok'], 1.0)
    def test_rollouts(self):
        import chat.train.dpo_stage as dp
        good = 'It uses PPO at 15 Hz. Want to watch it?'
        def fake(model, tok, prompts, n=1, **kw):
            if n > 1: return [[good, 'No idea lol', 'It uses PPO. Want more? Or not?', 'ok'] for _ in prompts]
            return ['cool, what next?' if model == 'sim' else 'Try the landing next. Want to?'] * len(prompts)
        rows = [{'prompt': [{'role': 'system', 'content': 'S'}, {'role': 'user', 'content': 'how does it fly'}], 'notes': [KB['facts'][0]['text']],
                 'user': 'how does it fly', 'must': ['ppo'], 'abstain': False, 'lead': True, 'meta': {'persona': 'kid', 'goal': 'g'}}]
        orig, dp.generate = dp.generate, fake
        try: groups = dp.rollouts('pol', None, 'sim', None, rows, Notes(KB), {'k': 4, 'gen_bs': 8, 'max_new': 32, 'temperature': 0.8, 'sim_tokens': 16})
        finally: dp.generate = orig
        g = groups[0]; self.assertEqual(len(g), 4); self.assertEqual(max(g, key=lambda c: c['score'])['text'], good)
        for c in g: self.assertAlmostEqual(c['score'], c['now'] + dp.FUTURE_W * c['future']); self.assertEqual(c['visitor'], 'cool, what next?')
        self.assertEqual(dp.select_pairs(groups, 0.15)[0][1:3], (good, 'ok'))

class TestRun(unittest.TestCase):
    def test_markers_mirror_resume(self):
        tmp = Path(tempfile.mkdtemp()); run = Run(tmp / 'w', tmp / 'd', None, CPU)
        self.assertIsNone(run.done('S2')); self.assertEqual(run.latest(), ('base', MODELS['lfm350']['student']))
        m = run.dir('S2') / 'model'; m.mkdir(); (m / 'config.json').write_text('{}'); (run.dir('S2') / 'pairs.jsonl').write_text('{}\n')
        (tmp / 'd' / 'S2' / 'ckpt' / 'checkpoint-4').mkdir(parents=True)
        run.finish('S2', rows=3)
        self.assertEqual(run.done('S2')['rows'], 3); self.assertTrue((tmp / 'd/S2/model/config.json').exists())
        self.assertTrue((tmp / 'd/S2/pairs.jsonl').exists()); self.assertFalse((tmp / 'd/S2/ckpt').exists())
        import shutil; shutil.rmtree(tmp / 'w')  # a Colab disconnect wipes the local disk
        self.assertEqual(run.model('S2'), tmp / 'w/S2/model'); self.assertTrue((tmp / 'w/S2/model/config.json').exists())
        ck = tmp / 'd/S3/ckpt'; (ck / 'checkpoint-2').mkdir(parents=True); (ck / 'checkpoint-9').mkdir()  # 9 incomplete
        (ck / 'checkpoint-2' / '.complete').write_text('ok'); (ck / 'checkpoint-2' / 'x.bin').write_text('w')
        self.assertEqual(Path(run.resume('S3', run.dir('S3'))).name, 'checkpoint-2'); self.assertTrue((tmp / 'w/S3/checkpoint-2/x.bin').exists())
    def test_best(self):
        tmp = Path(tempfile.mkdtemp()); run = Run(tmp / 'w', tmp / 'd', None, CPU)
        for s, sc in (('S2', 50.0), ('S3', 60.0), ('S4', 55.0)):
            (run.dir(s) / 'model').mkdir(); (run.dir(s) / 'model' / 'config.json').write_text('{}'); run.finish(s)
            run.save_eval(s, {'score': sc, 'sets': {}})
        self.assertEqual(run.best()[0], 'S3'); self.assertEqual(run.best('S3')[0], 'S2'); self.assertEqual(run.latest()[0], 'S4')
        self.assertEqual(set(run.evals()), {'S2', 'S3', 'S4'})
        run.finish('S6', final='S3'); self.assertEqual(run.final()[0], 'S3')

class TestWebStage(unittest.TestCase):
    def test_web_with_stub_export(self):
        import chat.export_web as ew
        from chat.train import web_stage
        tmp = Path(tempfile.mkdtemp()); run = Run(tmp / 'w', tmp / 'd', None, CPU, smoke=True); run.kb = {'name': 'kb', 'facts': []}
        (run.dir('S5') / 'model').mkdir(); (run.dir('S5') / 'model' / 'config.json').write_text('{}'); run.finish('S5')
        run.save_eval('S5', {'score': 70.0, 'sets': {}}); run.finish('S6', final='S5')
        calls = []
        def fake_export(src, out, dtypes, ctx, acc=None):
            calls.append((Path(src).parent.name, tuple(dtypes), ctx, acc)); (Path(out) / 'onnx').mkdir(parents=True, exist_ok=True)
            (Path(out) / 'onnx' / 'model_q4.onnx').write_bytes(b'x'); return {'q4': 0.1}
        orig = ew.export, ew.check; ew.export, ew.check = fake_export, lambda src, out: {'greedy_agreement': 1.0, 'tokens': 2}
        try: d = web_stage.web(run, acc=1)
        finally: ew.export, ew.check = orig
        self.assertEqual(calls, [('S5', ('q4',), 4096, 1)]); self.assertEqual((d['final'], d['acc']), ('S5', 1))
        card = json.loads((tmp / 'w/S7/web/capcom.json').read_text())
        self.assertEqual((card['stage'], card['eval_score'], card['int4_accuracy_level']), ('S5', 70.0, 1))
        names = zipfile.ZipFile(tmp / 'd' / 'capcom-web-w.zip').namelist()
        self.assertIn('capcom-web-w/onnx/model_q4.onnx', names); self.assertIn('capcom-web-w/evals/eval_S5.json', names)
        self.assertEqual(web_stage.web(run), run.done('S7'))  # DONE: skipped

class TestEvalMetrics(unittest.TestCase):
    def test_metrics(self):
        from chat.eval import metrics, table
        from chat.rewards import score
        rows = [{'notes': ['The pilot is PPO.'], 'user': 'what is it', 'must': ['ppo'], 'abstain': False, 'lead': True},
                {'notes': [], 'user': 'is there VR?', 'must': [], 'abstain': True, 'lead': True}]
        answers = ['The pilot is a PPO policy. Is it cool? Want more?', 'Yes, VR works with every headset. Try it now.']
        m = metrics([{'answer': a, 'row': r, 's': score(a, r)} for a, r in zip(answers, rows)])
        self.assertEqual((m['n'], m['nagging'], m['false_answer'], m['abstain_acc'], m['recall']), (2, 0.5, 1.0, 0.0, 1.0))
        self.assertEqual(metrics([]), {'n': 0, 'score': None})
        self.assertIn('S2', table({'S2': {'score': 50.0, 'sets': {'val': m}}}))

class TestNotebook(unittest.TestCase):
    def test_build(self):
        from chat.colab import make_notebook as mk
        nb = mk.build(); src = [c['source'] for c in nb.cells]
        for f in mk.LIB:
            text = (mk.REPO / f).read_text()
            if not text: self.assertTrue(any(repr(f) in s and '.touch()' in s for s in src), f); continue  # empty files are touched
            cell = next(s for s in src if s.startswith(f'%%writefile {{CODE}}/{f}\n'))
            self.assertEqual(cell.split('\n', 1)[1], text)
        self.assertFalse(any(s.startswith('%%writefile') and not s.split('\n', 1)[1] for s in src))  # IPython rejects an empty body
        self.assertFalse(any(p.startswith('torch') for p in mk.PINS))
        joined = '\n'.join(src)
        for s in ('S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7'): self.assertIn(f"STAGES['{s}']", joined)
        for k in ('STUDENT', 'TEACHER_SIZE', 'TEACHER_OVERRIDE', 'SMOKE', 'DRIVE_DIR', 'DATA_ZIP', 'RUN_NAME', 'HF_PUSH', 'HF_REPO'): self.assertIn(k + ' =', joined)
        for st in ('S2', 'S3', 'S4', 'S5'): self.assertIn(f"eval_stage(RUN, '{st}')", joined)
    def test_committed_notebook_in_sync(self):
        from chat.colab import make_notebook as mk
        import nbformat
        p = mk.REPO / 'chat/colab/capcom_colab.ipynb'
        if not p.exists(): self.skipTest('notebook not generated yet')
        self.assertEqual(nbformat.writes(mk.build()), p.read_text(), 'regenerate: python3 chat/colab/make_notebook.py')

if __name__ == '__main__':
    unittest.main()
