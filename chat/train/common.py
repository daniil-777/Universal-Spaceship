"""chat/train/common.py — shared plumbing for the CAPCOM stages (spec §6; rulings R1/R2/R5): GPU probe -> tier and precision (T4 has
no bf16: fp16), the model registry, the stage x tier hyper-parameter table (+ SMOKE overrides: a few steps on tiny slices), data loading
from the package (SFT rows mixed with general replay, prompt rows), notes retrieval (the page's BM25 + min_score), the visitor simulator
prompt, batched chat generation, and Run: a local work dir mirrored to Drive with per-stage DONE markers and checkpoint resume."""
import gc, hashlib, json, os, random, shutil, time
from pathlib import Path
from chat.prompt import MAX_NOTES, prev_user
from chat.retrieval import BM25, MIN_SCORE

GENERAL_FRAC = 0.25        # share of general-replay rows in the SFT mix (spec §5)
DEFAULT_STATE = {'scene': 'belt', 'seen': []}
MODELS = {  # R1/R2: the student, its same-vocabulary teacher, the >= 40 GB teacher; s2_lr = the student's full fine-tune lr
    'lfm350': {'student': 'LiquidAI/LFM2.5-350M', 'teacher': 'LiquidAI/LFM2.5-1.2B-Instruct',
               # LFM2.5-2.6B has a 128k vocabulary; DistillationTrainer needs the student's 65,536 — LFM2-2.6B shares it (ids 0-395 and
               # every BPE/chat token are identical; only reserved image slots are named differently)
               'teacher_big': 'LiquidAI/LFM2-2.6B', 's2_lr': 3e-5},
    'smol135': {'student': 'HuggingFaceTB/SmolLM2-135M-Instruct', 'teacher': 'HuggingFaceTB/SmolLM2-1.7B-Instruct',
                'teacher_big': 'HuggingFaceTB/SmolLM2-1.7B-Instruct', 's2_lr': 5e-5},
    'granite350': {'student': 'ibm-granite/granite-4.0-350m', 'teacher': 'ibm-granite/granite-4.0-1b',
                   'teacher_big': 'ibm-granite/granite-4.0-1b', 's2_lr': 3e-5},
}
# hyper-parameters: A100-40 defaults (spec §6 table), per-tier overrides, SMOKE overrides. bs = per-device batch, ga = accumulation.
BASE = {
    's1': dict(lr=1e-4, epochs=2, bs=16, ga=4, max_len=1024, lora_r=64, lora_alpha=128, max_rows=None, max_steps=-1, save=0.25),
    's2': dict(lr=None, epochs=3, bs=32, ga=2, max_len=1024, warmup=0.03, max_rows=None, max_steps=-1, save=0.25),
    's3': dict(lr=1e-5, max_steps=400, bs=16, ga=2, max_completion=128, temperature=1.0, beta=1.0, n_prompts=8000, save=0.25),
    's4': dict(n_ctx=2000, k=4, temperature=0.8, max_new=128, sim_tokens=48, gen_bs=64, rounds=2, margin=0.15,
               lr=1e-6, epochs=2, bs=8, ga=4, beta=0.3, max_len=1024, max_steps=-1, save=0),
    's5': dict(lr=1e-6, max_steps=200, bs=16, ga=2, num_generations=4, beta=0.02, epsilon_high=0.28, max_completion=128,
               n_prompts=4000, save=0.25),
    'eval': dict(n_single=None, n_unans=None, n_val=200, n_scripts=None, turns=4, bs=64, max_new=120),
}
_H100 = {'s1': dict(bs=32, ga=2), 's2': dict(bs=64, ga=1), 's3': dict(max_steps=500, bs=32, ga=1),
         's4': dict(n_ctx=3000, gen_bs=128, bs=16, ga=2), 's5': dict(bs=32, ga=1), 'eval': dict(bs=128)}
TIER = {
    't4': {'s1': dict(epochs=1, bs=4, ga=16, max_rows=20000), 's2': dict(epochs=2, bs=8, ga=8),
           's3': dict(max_steps=200, bs=8, ga=4, n_prompts=6400), 's4': dict(n_ctx=600, gen_bs=16, rounds=1, bs=2, ga=16),
           's5': dict(num_generations=2, bs=4, ga=8, max_steps=150), 'eval': dict(bs=16)},
    'l4': {'s1': dict(epochs=1, bs=8, ga=8), 's2': dict(bs=16, ga=4), 's3': dict(max_steps=300, bs=8, ga=4),
           's4': dict(n_ctx=1200, gen_bs=32, bs=4, ga=8), 's5': dict(bs=8, ga=4), 'eval': dict(bs=32)},
    'a100': {}, 'h100': _H100, 'big': _H100,
    'mps': {'s1': dict(epochs=1, bs=2, ga=32, max_rows=4000), 's2': dict(epochs=1, bs=4, ga=16, max_rows=8000),
            's3': dict(max_steps=50, bs=4, ga=4, n_prompts=800), 's4': dict(n_ctx=100, gen_bs=16, rounds=1, bs=2, ga=16),
            's5': dict(bs=4, ga=4, num_generations=2, max_steps=30), 'eval': dict(bs=16, n_val=100)},
    'cpu': {'s1': dict(epochs=1, bs=1, ga=16, max_rows=500), 's2': dict(epochs=1, bs=2, ga=16, max_rows=1000),
            's3': dict(max_steps=10, bs=2, ga=4, n_prompts=80), 's4': dict(n_ctx=20, gen_bs=8, rounds=1, bs=1, ga=8),
            's5': dict(bs=2, ga=4, num_generations=2, max_steps=10), 'eval': dict(bs=8, n_val=40)},
}
SMOKE = {
    's1': dict(max_steps=3, bs=2, ga=1, max_rows=24, lora_r=8, lora_alpha=16, max_len=768, save=2),
    's2': dict(max_steps=4, bs=2, ga=1, max_rows=32, max_len=768, save=0),
    's3': dict(max_steps=2, bs=2, ga=1, max_completion=32, n_prompts=8, save=0),
    's4': dict(n_ctx=3, k=4, max_new=32, sim_tokens=24, gen_bs=12, rounds=2, margin=0.0, epochs=1, bs=1, ga=1, max_steps=2, max_len=768),
    's5': dict(max_steps=2, bs=2, ga=1, num_generations=2, max_completion=32, n_prompts=8, save=0),
    'eval': dict(n_single=3, n_unans=3, n_val=4, n_scripts=2, turns=2, bs=8, max_new=48),
}

def hp(stage, tier, smoke=False, student='lfm350'):
    """The hyper-parameters of stage ('s1'..'s5', 'eval') on a tier: BASE <- TIER <- SMOKE."""
    h = {**BASE[stage], **TIER.get(tier, {}).get(stage, {})}
    if stage == 's2' and h['lr'] is None: h['lr'] = MODELS[student]['s2_lr']
    if smoke: h.update(SMOKE[stage])
    return h

def probe():
    """{'tier', 'device', 'name', 'mem_gb', 'cc', 'bf16', 'fp16'}: T4 -> fp16 (compute capability 7.5 has no bf16), Ampere+ -> bf16,
    MPS/CPU -> fp32. Tiers: t4, l4, a100 (40 GB), h100, big (other >= 80 GB cards), mps, cpu."""
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0); name, mem, cc = p.name, p.total_memory / 2 ** 30, (p.major, p.minor); n = name.upper()
        tier = ('t4' if 'T4' in n else 'l4' if 'L4' in n else 'h100' if ('H100' in n or 'H200' in n) else 'big' if mem >= 79
                else 'a100' if ('A100' in n or mem >= 38) else 'l4' if mem >= 22 else 't4')
        bf16 = cc >= (8, 0)
        return {'tier': tier, 'device': 'cuda', 'name': name, 'mem_gb': round(mem, 1), 'cc': f'{cc[0]}.{cc[1]}', 'bf16': bf16, 'fp16': not bf16}
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    return {'tier': dev, 'device': dev, 'name': dev, 'mem_gb': None, 'cc': None, 'bf16': False, 'fp16': False}

def default_stages(tier):
    """Stage toggles: everything on, S5 GRPO off on T4 (spec §6)."""
    return {'S1': True, 'S2': True, 'S3': True, 'S4': True, 'S5': tier != 't4', 'S6': True, 'S7': True}

def log(*a):
    print(time.strftime('[%H:%M:%S]'), *a, flush=True)

def seed_all(seed):
    import numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def free():
    import torch
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    elif torch.backends.mps.is_available(): torch.mps.empty_cache()

# ---- data ---------------------------------------------------------------------------------------------------------------------------
def read_jsonl(p):
    """Rows of a JSONL file; [] when it is missing or empty (the dev package has empty eval files)."""
    p = Path(p)  # split on '\n' only: str.splitlines() also breaks on U+2028/U+2029 inside JSON strings
    return [json.loads(l) for l in p.read_text(encoding='utf-8').split('\n') if l.strip()] if p.exists() else []

def find_package(d):
    """The dataset package folder (the one holding MANIFEST.json): d itself or its single child."""
    d = Path(d)
    if (d / 'MANIFEST.json').exists(): return d
    c = [p.parent for p in sorted(d.glob('*/MANIFEST.json')) if not p.parent.name.startswith(('.', '_'))]
    if len(c) != 1: raise FileNotFoundError(f'expected one package with MANIFEST.json under {d}, found {len(c)}')
    return c[0]

def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def verify_manifest(pkg):
    """[(file, problem)] for every MANIFEST file that is missing or whose sha256 differs ([] = the package is intact)."""
    pkg = Path(pkg); man = json.loads((pkg / 'MANIFEST.json').read_text()); bad = []
    for f, m in man['files'].items():
        p = pkg / f
        if not p.exists(): bad.append((f, 'missing'))
        elif sha256(p) != m['sha256']: bad.append((f, 'sha256'))
    return bad

def mix_general(capcom, general, frac=GENERAL_FRAC, seed=0):
    """capcom + a random sample of general rows so that general is `frac` of the result (capped by what exists), shuffled."""
    n = min(len(general), round(frac * len(capcom) / (1 - frac))) if 0 < frac < 1 else 0
    rng = random.Random(seed); rows = capcom + (rng.sample(general, n) if n else [])
    rng.shuffle(rows)
    return rows

def pc(r):
    return {'prompt': r['prompt'], 'completion': r['completion']}

def sft_rows(pkg, split='train', general_frac=GENERAL_FRAC, max_rows=None, seed=0):
    """Prompt-completion rows: sft/capcom_<split> (+ sft/general replay on train), shuffled, cut to max_rows."""
    pkg = Path(pkg); capcom = [pc(r) for r in read_jsonl(pkg / 'sft' / f'capcom_{split}.jsonl')]
    general = [pc(r) for r in read_jsonl(pkg / 'sft' / 'general.jsonl')] if split == 'train' else []
    rows = mix_general(capcom, general, general_frac, seed)
    return rows[:max_rows] if max_rows else rows

def prompt_rows(pkg, split='train', n=None, seed=0):
    """Prompt rows with the scorer fields (prompts/<split>.jsonl), a seeded random n of them."""
    rows = read_jsonl(Path(pkg) / 'prompts' / f'{split}.jsonl')
    return random.Random(seed).sample(rows, n) if n and len(rows) > n else rows

class Notes:
    """The page's retrieval: BM25 top MAX_NOTES over kb.json facts, kb['retrieval']['min_score'] as the no-facts threshold."""
    def __init__(self, kb):
        self.bm = BM25(kb['facts']); self.min_score = kb.get('retrieval', {}).get('min_score', MIN_SCORE)
    def __call__(self, user, history=()):
        return [self.bm.by_id[i]['text'] for i, _ in self.bm.search(user, prev_user(history), k=MAX_NOTES, min_score=self.min_score)]

# ---- the visitor simulator (S4 futures, eval dialogs): the untuned teacher plays the visitor ------------------------------------------
SIM_SYS = ('You are role-playing a visitor to Astro Pilot, a spaceflight demo running in the browser, who is chatting with its guide '
           'CAPCOM. Persona: {persona}.{style} Your hidden goal: {goal}.{intent} Write only the visitor\'s next chat message: one short '
           'message (at most 25 words), in character, reacting to what CAPCOM just said. No quotes, no name labels.')

def sim_messages(turns, persona='curious visitor', goal='find out what this demo is', style='', intent=''):
    """Messages for the simulator; turns = the CAPCOM chat so far [{'role': 'user' (the visitor) | 'assistant' (CAPCOM), 'content'}]."""
    sys = SIM_SYS.format(persona=persona or 'curious visitor', goal=(goal or 'explore the demo').rstrip('.'),
                         style=f' Writing style: {style}.' if style else '', intent=f' What you want next: {intent}.' if intent else '')
    chat = '\n'.join(f"{'Visitor' if m['role'] == 'user' else 'CAPCOM'}: {m['content']}" for m in turns[-6:])
    return [{'role': 'system', 'content': sys}, {'role': 'user', 'content': f"The chat so far:\n{chat}\n\nWrite the visitor's next message."}]

def clean_visitor(text, fallback='ok, what else can I try?'):
    """The first non-empty line without labels/quotes, at most 40 words; fallback when nothing usable is left."""
    for line in (text or '').splitlines():
        line = line.strip().strip('"\'').strip()
        for lab in ('Visitor:', 'visitor:', 'User:', 'user:', 'Me:'):
            if line.startswith(lab): line = line[len(lab):].strip().strip('"\'')
        if line and not line.lower().startswith('capcom'): return ' '.join(line.split()[:40])
    return fallback

# ---- models and generation ----------------------------------------------------------------------------------------------------------
def load_tok(path, side='left'):
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(str(path)); t.padding_side = side
    if t.pad_token is None: t.pad_token = t.eos_token
    return t

def infer_dtype(info, student=False):
    """Inference dtype: bf16 on Ampere+; on T4 fp16 for the teacher/simulator (memory) but fp32 for the small student (no fp16 overflow
    risk where the scores are measured); fp32 on MPS/CPU."""
    import torch
    if info['device'] != 'cuda': return torch.float32
    return torch.bfloat16 if info['bf16'] else torch.float32 if student else torch.float16

def load_model(path, info, dtype=None):
    """A causal LM on the probed device (SDPA attention); dtype default: fp32 (trainable master weights)."""
    import torch
    from transformers import AutoModelForCausalLM
    kw = {'dtype': dtype or torch.float32}
    try: m = AutoModelForCausalLM.from_pretrained(str(path), attn_implementation='sdpa', **kw)
    except (ValueError, ImportError): m = AutoModelForCausalLM.from_pretrained(str(path), **kw)
    return m.to(info['device'])

def eot_ids(tok):
    """End-of-turn token ids: eos plus <|im_end|> when the vocabulary has it."""
    ids = {tok.eos_token_id}
    im = tok.convert_tokens_to_ids('<|im_end|>')
    if isinstance(im, int) and im != tok.unk_token_id: ids.add(im)
    return sorted(i for i in ids if i is not None)

def chat_ids(tok, messages):
    """The prompt's token ids exactly as generation sees them (chat template + generation prompt, no extra specials)."""
    return tok(tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False), add_special_tokens=False)['input_ids']

def generate(model, tok, prompts, n=1, bs=16, max_new_tokens=120, do_sample=False, temperature=1.0, top_p=1.0,
             repetition_penalty=1.1, seed=None, max_prompt=None):
    """Batched chat generation (left padding, longest prompts first). prompts: message lists. Returns [str] (n == 1) or [[str] * n]."""
    import torch
    from transformers import GenerationConfig
    if not prompts: return []
    if seed is not None: torch.manual_seed(seed)
    ids = [chat_ids(tok, p) for p in prompts]
    if max_prompt: ids = [x[-max_prompt:] for x in ids]
    order = sorted(range(len(ids)), key=lambda i: -len(ids[i])); out = [None] * len(ids)
    gc_ = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature if do_sample else 1.0,
                           top_p=top_p if do_sample else 1.0, top_k=0 if do_sample else 50, min_p=None, repetition_penalty=repetition_penalty,
                           num_return_sequences=n, pad_token_id=tok.pad_token_id, eos_token_id=eot_ids(tok))
    was, per = model.training, max(1, bs // n); model.eval()
    with torch.no_grad():
        for s in range(0, len(order), per):
            idx = order[s:s + per]; L = max(len(ids[i]) for i in idx)
            x = torch.full((len(idx), L), tok.pad_token_id, dtype=torch.long); m = torch.zeros_like(x)
            for r, i in enumerate(idx): x[r, L - len(ids[i]):] = torch.tensor(ids[i]); m[r, L - len(ids[i]):] = 1
            g = model.generate(input_ids=x.to(model.device), attention_mask=m.to(model.device), generation_config=gc_)
            texts = tok.batch_decode(g[:, L:], skip_special_tokens=True)
            for r, i in enumerate(idx):
                t = [s_.strip() for s_ in texts[r * n:(r + 1) * n]]; out[i] = t if n > 1 else t[0]
    if was: model.train()
    return out

# ---- the run: local work dir + Drive mirror, DONE markers, resume --------------------------------------------------------------------
def copy_tree(src, dst):
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns('._*', '.DS_Store'))

def keep_model(src, dst, info):
    """Copy a model folder (or save a Hub model) to dst — a stage that keeps its input unchanged."""
    if Path(str(src)).is_dir(): copy_tree(src, dst); return
    import torch
    m, t = load_model(src, info, dtype=torch.bfloat16), load_tok(src)
    m.save_pretrained(str(dst)); t.save_pretrained(str(dst))

def latest_ckpt(d, need_mark=False):
    """The newest checkpoint-<step> folder under d (on Drive only complete copies, marked by .complete)."""
    d = Path(d)
    if not d.exists(): return None
    c = [p for p in d.glob('checkpoint-*') if p.is_dir() and p.name.split('-')[-1].isdigit() and (not need_mark or (p / '.complete').exists())]
    return max(c, key=lambda p: int(p.name.split('-')[-1])) if c else None

STUDENT_STAGES = ('S2', 'S3', 'S4', 'S5', 'S6')

class Run:
    """One training run: work/<stage>/ holds checkpoints and the stage's final `model/`; drive/<stage>/ mirrors the model, the newest
    checkpoint (ckpt/) and DONE.json, so a Colab disconnect resumes where it stopped; evals/eval_<stage>.json on both."""
    def __init__(self, work, drive=None, pkg=None, info=None, student='lfm350', teacher_size='1.2b', teacher_override=None,
                 smoke=False, seed=0, general_frac=GENERAL_FRAC):
        self.work, self.drive = Path(work), (Path(drive) if drive else None)
        self.pkg = Path(pkg) if pkg else None; self.info = info or probe(); self.tier = self.info['tier']
        self.student, self.smoke, self.seed, self.general_frac = student, bool(smoke), seed, general_frac
        m = MODELS[student]; self.student_id = m['student']
        self.teacher_id = teacher_override or m['teacher_big' if teacher_size == '2.6b' else 'teacher']
        self.sim_id = self.teacher_id  # the untuned teacher plays the visitor: the S1 LoRA made it a CAPCOM, not a visitor
        self.kb = json.loads((self.pkg / 'kb.json').read_text()) if self.pkg and (self.pkg / 'kb.json').exists() else None
        self.work.mkdir(parents=True, exist_ok=True)
        if self.drive: self.drive.mkdir(parents=True, exist_ok=True)

    def hp(self, stage): return hp(stage, self.tier, self.smoke, self.student)
    def prec(self): return {'bf16': self.info['bf16'], 'fp16': self.info['fp16']}
    def dir(self, stage):
        p = self.work / stage; p.mkdir(parents=True, exist_ok=True); return p
    def ddir(self, stage): return self.drive / stage if self.drive else None

    def done(self, stage):
        """The DONE.json info of a finished stage (Drive first, then local) or None."""
        for base in (self.drive, self.work):
            if base and (base / stage / 'DONE.json').exists(): return json.loads((base / stage / 'DONE.json').read_text())
        return None

    def finish(self, stage, **info):
        """Mirror work/<stage>/model to Drive, drop the stage's Drive checkpoints, then write DONE.json (last: a partial copy never counts)."""
        src = self.work / stage / 'model'; info = {'stage': stage, 'finished': time.strftime('%Y-%m-%d %H:%M:%S'), 'has_model': src.exists(), **info}
        if self.drive:
            d = self.ddir(stage); d.mkdir(parents=True, exist_ok=True)
            if src.exists(): shutil.rmtree(d / 'model', ignore_errors=True); copy_tree(src, d / 'model')
            for f in self.dir(stage).glob('*.json*'):  # small artefacts (pairs.jsonl, ...)
                if f.name != 'DONE.json' and not f.name.startswith('._'): shutil.copy(f, d / f.name)
            shutil.rmtree(d / 'ckpt', ignore_errors=True)
            (d / 'DONE.json').write_text(json.dumps(info, indent=1))
        self.dir(stage).joinpath('DONE.json').write_text(json.dumps(info, indent=1))
        log(f'{stage} done', json.dumps({k: v for k, v in info.items() if k not in ('stage', 'finished')})[:300])
        return info

    def model(self, stage):
        """Local folder of a finished stage's model (restored from Drive after a disconnect), or None."""
        if not (self.done(stage) or {}).get('has_model'): return None
        loc = self.work / stage / 'model'
        if not (loc / 'config.json').exists() and self.drive and (self.ddir(stage) / 'model' / 'config.json').exists():
            log(f'restoring {stage} model from Drive'); copy_tree(self.ddir(stage) / 'model', loc)
        return loc if (loc / 'config.json').exists() else None

    def resume(self, stage, out):
        """Checkpoint to resume a stage's trainer from: the newest local one, else the Drive mirror copied back."""
        ck = latest_ckpt(out)
        if ck is None and self.drive:
            dck = latest_ckpt(self.ddir(stage) / 'ckpt', need_mark=True)
            if dck: log(f'restoring {stage} checkpoint {dck.name} from Drive'); copy_tree(dck, Path(out) / dck.name); ck = Path(out) / dck.name
        return str(ck) if ck else None

    def mirror_callback(self, stage):
        """A TrainerCallback copying each saved checkpoint to drive/<stage>/ckpt (marked complete, older copies removed)."""
        from transformers import TrainerCallback
        run = self
        class Mirror(TrainerCallback):
            def on_save(self, args, state, control, **kw):
                ck = Path(args.output_dir) / f'checkpoint-{state.global_step}'
                if not run.drive or not ck.exists(): return
                dst = run.ddir(stage) / 'ckpt'; copy_tree(ck, dst / ck.name); (dst / ck.name / '.complete').write_text('ok')
                for old in dst.glob('checkpoint-*'):
                    if old.name != ck.name: shutil.rmtree(old, ignore_errors=True)
        return Mirror()

    def targs(self, cls, stage, out, h, **kw):
        """TRL/Trainer config with the run's precision, batch sizes, checkpointing and logging (kw override)."""
        smoke = self.smoke; save = h.get('save', 0)
        a = dict(output_dir=str(out), per_device_train_batch_size=h['bs'], gradient_accumulation_steps=h['ga'], learning_rate=h['lr'],
                 num_train_epochs=h.get('epochs', 1), max_steps=h.get('max_steps', -1), logging_steps=1 if smoke else 10,
                 save_strategy='steps' if save else 'no', save_steps=save or 500, save_total_limit=1, report_to='none', seed=self.seed,
                 bf16=self.prec()['bf16'], fp16=self.prec()['fp16'], gradient_checkpointing=True, dataloader_num_workers=0,
                 dataloader_pin_memory=self.info['device'] == 'cuda', use_cpu=self.tier == 'cpu', disable_tqdm=smoke,
                 lr_scheduler_type='cosine', warmup_steps=h.get('warmup', 0.03))
        a.update(kw)
        return cls(**a)

    def save_eval(self, stage, report):
        for base in (self.work, self.drive):
            if base: (base / 'evals').mkdir(parents=True, exist_ok=True); (base / 'evals' / f'eval_{stage}.json').write_text(json.dumps(report, indent=1))

    def evals(self):
        """{stage: eval report} from Drive (else local)."""
        out = {}
        for base in (self.work, self.drive):
            if base and (base / 'evals').exists():
                for p in sorted((base / 'evals').glob('eval_*.json')):
                    if not p.name.startswith('._'): out[p.stem[5:]] = json.loads(p.read_text())
        return dict(sorted(out.items(), key=lambda kv: (kv[0] != 'base', kv[0])))

    def latest(self, before=None):
        """(stage, folder) of the newest finished student stage before `before`, else ('base', the Hub id)."""
        stages = STUDENT_STAGES if before is None else STUDENT_STAGES[:STUDENT_STAGES.index(before)]
        for s in reversed(stages):
            m = self.model(s)
            if m: return s, m
        return 'base', self.student_id

    def best(self, before=None):
        """(stage, folder) of the best-scoring evaluated student stage before `before` (R5: a stage is kept only if it improves the eval
        score), falling back to latest()."""
        ev, stages = self.evals(), (STUDENT_STAGES if before is None else STUDENT_STAGES[:STUDENT_STAGES.index(before)])
        cand = [(ev[s]['score'], i, s) for i, s in enumerate(stages) if s in ev and ev[s].get('score') is not None and self.model(s)]
        if not cand: return self.latest(before)
        s = max(cand)[2]
        return s, self.model(s)

    def final(self):
        """(stage, folder) of the model to ship: S6's pick, else best()."""
        d = self.done('S6')
        if d and self.model(d['final']): return d['final'], self.model(d['final'])
        return self.best()

def cli(doc, extra=()):
    """argparse for the stages' local CLIs: returns (Run, args). extra: [(flag, kwargs)]."""
    import argparse
    os.environ.setdefault('HF_HOME', '/Volumes/LaCie/astro-pilot/chat/hf')
    ap = argparse.ArgumentParser(description=doc.split('\n')[0])
    ap.add_argument('--pkg', required=True, help='dataset package folder (or its parent)'); ap.add_argument('--work', required=True)
    ap.add_argument('--drive', help='mirror folder (Drive)'); ap.add_argument('--student', default='lfm350', choices=sorted(MODELS))
    ap.add_argument('--teacher-size', default='1.2b', choices=['1.2b', '2.6b']); ap.add_argument('--teacher', help='teacher id/path override')
    ap.add_argument('--smoke', action='store_true'); ap.add_argument('--seed', type=int, default=0)
    for flag, kw in extra: ap.add_argument(flag, **kw)
    a = ap.parse_args()
    return Run(a.work, a.drive, find_package(a.pkg), None, a.student, a.teacher_size, a.teacher, a.smoke, a.seed), a
