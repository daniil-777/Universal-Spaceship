"""chat/colab/make_notebook.py — writes chat/colab/capcom_colab.ipynb (nbformat v4), CAPCOM's Run-all Colab notebook (spec §6, R5, R9).
Cells: intro + instructions, config (every knob also readable from CAPCOM_* env vars, which chat/tests/nb_e2e.py uses), pinned install
(never torch), the library (%%writefile of every module, read from this repo so the notebook stays in sync), GPU probe -> tier, Drive
mount + data (zip copied to local disk, unzipped, MANIFEST sha256 verified), the stages S1..S7 with evals, the results table and an
optional Hugging Face push. Every stage writes a DONE marker on Drive and is skipped on re-run; checkpoints mirror to Drive.
  python3 chat/colab/make_notebook.py [--out chat/colab/capcom_colab.ipynb]"""
import argparse
from pathlib import Path
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO = Path(__file__).resolve().parents[2]
LIB = ('chat/__init__.py', 'chat/prompt.py', 'chat/retrieval.py', 'chat/rewards.py', 'chat/export_web.py', 'chat/data/__init__.py',
       'chat/train/__init__.py', 'chat/train/common.py', 'chat/train/sft_stage.py', 'chat/train/distill_stage.py', 'chat/train/dpo_stage.py',
       'chat/train/grpo_stage.py', 'chat/train/merge_stage.py', 'chat/train/web_stage.py', 'chat/eval.py')
PINS = ('transformers==5.18.0', 'trl==1.14.1', 'peft==0.21.1', 'accelerate==1.15.0', 'datasets==5.0.1', 'huggingface-hub==1.33.0',
        'onnx==1.23.1', 'onnxscript==0.7.2', 'onnxruntime==1.30.0', 'onnxruntime-genai==0.17.1')

INTRO = """# CAPCOM — train the Astro Pilot guide (Run all)

CAPCOM is the tiny chat guide that runs **inside the visitor's browser** next to the Astro Pilot demo: it answers from ≤ 3 retrieved
fact notes, then offers exactly one next step. This notebook runs the whole recipe on one Colab GPU and leaves a ready-to-serve web
model folder (ONNX int4 for transformers.js) on your Google Drive.

**Recipe** (design spec §6): S1 teacher SFT (LoRA on LFM2.5-1.2B-Instruct, merged) → S2 student SFT (full fine-tune of LFM2.5-350M)
→ S3 on-policy distillation (`trl.DistillationTrainer`, reverse KL) → S4 multi-turn-aware DPO (2-turn simulated futures, verifiable
scorer, `apo_zero`) → S5 optional GRPO (`dapo`, verifiable rewards) → S6 pick the best stage by eval (+ 0.9/0.1 merge with S3 if it
helps) → S7 export (int4: q4f16 for WebGPU + q4 for WebGPU without shader-f16, parity check) → `capcom-web-<RUN_NAME>.zip` on Drive.

## How to run
1. Upload **`capcom-data-v1.zip`** — on the LaCie drive at `/Volumes/LaCie/astro-pilot/chat/data/capcom-data-v1.zip` — to Google Drive
   as **`MyDrive/capcom/capcom-data-v1.zip`** (create the `capcom` folder).
2. *Runtime → Change runtime type →* a GPU: **A100** (best), L4 (fine), or T4 (free; lighter recipe: fp16, smaller batches, one DPO
   round, no GRPO).
3. *Runtime → Run all*, and approve the Google Drive pop-up. Optional: add a Colab secret `HF_TOKEN` (faster Hub downloads; needed only
   for `HF_PUSH`). The GPU cell prints the tier it detected and the hyper-parameters it chose for every stage.

**Resumable.** Every stage writes `DONE.json` to `MyDrive/capcom/runs/<RUN_NAME>/<stage>/` and is skipped on a re-run; a stage in
progress mirrors its newest checkpoint to Drive and resumes from it. After a disconnect: reconnect, *Run all* again.

## Expected time (full dataset; FLOP-based estimates, not measurements)
| GPU | S1 | S2 | S3 | S4 | S5 | evals + S6 + S7 | total | Colab units |
|---|---|---|---|---|---|---|---|---|
| A100-40 | 0.5 h | 0.5 h | 1 h | 1 h | 0.7 h | 0.5 h | ≈ 4–5 h | ≈ 25 |
| L4 | 1 h | 1 h | 1.5 h | 1.5 h | 1 h | 0.7 h | ≈ 6–7 h | ≈ 12 |
| T4 | 2 h | 2 h | 1.5 h | 1 h | off | 1 h | ≈ 7–8 h | ≈ 9 |

Outputs on Drive (`MyDrive/capcom/runs/<RUN_NAME>/`): every stage's model, `evals/eval_<stage>.json`, and
`capcom-web-<RUN_NAME>.zip` (the web model folder: `onnx/model_q4f16.onnx`, `onnx/model_q4.onnx`, tokenizer, `kb.json`, evals,
`capcom.json`). Copy the zip back to `/Volumes/LaCie/astro-pilot/chat/models/` for the browser bench."""

CONFIG = """# ---- configuration: edit, then Runtime > Run all (CAPCOM_* environment variables override, for local tests) ----
import json, os
STUDENT = os.environ.get('CAPCOM_STUDENT', 'lfm350')          # 'lfm350' (primary) | 'smol135' (lite) | 'granite350' (Apache fallback)
TEACHER_SIZE = os.environ.get('CAPCOM_TEACHER_SIZE', '1.2b')  # '1.2b' | '2.6b' (>= 40 GB GPUs; LFM2-2.6B shares the student's vocabulary)
TEACHER_OVERRIDE = os.environ.get('CAPCOM_TEACHER') or None   # a Hub id / path replacing the teacher (smoke tests: the student itself)
STAGES = {'S1': True, 'S2': True, 'S3': True, 'S4': True, 'S5': 'auto', 'S6': True, 'S7': True}  # 'auto': GRPO off on T4
STAGES.update(json.loads(os.environ.get('CAPCOM_STAGES', '{}')))
SMOKE = int(os.environ.get('CAPCOM_SMOKE', '0'))              # 1 = a few steps per stage on tiny slices (a pipeline check, ~30 min)
DRIVE_DIR = os.environ.get('CAPCOM_DRIVE', '/content/drive/MyDrive/capcom')
DATA_ZIP = os.environ.get('CAPCOM_DATA_ZIP', 'capcom-data-v1.zip')
RUN_NAME = os.environ.get('CAPCOM_RUN', 'run1-smoke' if SMOKE else 'run1')
EXPORT_ACC = int(os.environ.get('CAPCOM_EXPORT_ACC', '4'))    # S7 int4 accuracy level: 4 = int8 activations (fast), 1 = float (A/B pending)
HF_PUSH = False                                               # push the web model folder to a private Hub repo (Colab secret HF_TOKEN)
HF_REPO = 'your-hf-name/capcom-' + STUDENT
LOCAL = os.environ.get('CAPCOM_LOCAL') == '1'                 # local run (chat/tests/nb_e2e.py): no Colab, no pip, paths under CAPCOM_ROOT
print(dict(STUDENT=STUDENT, TEACHER_SIZE=TEACHER_SIZE, TEACHER_OVERRIDE=TEACHER_OVERRIDE, STAGES=STAGES, SMOKE=SMOKE, RUN_NAME=RUN_NAME,
           EXPORT_ACC=EXPORT_ACC, LOCAL=LOCAL))"""

INSTALL = """# ---- install the pinned stack (never torch: Colab's preinstalled torch stays; a constraint makes pip fail rather than replace it) ----
import importlib.metadata as md, subprocess, sys
PINS = %s
if LOCAL: print('local run: pip skipped')
else:
    torch_v = md.version('torch'); open('/tmp/capcom-constraints.txt', 'w').write(f'torch=={torch_v}\\n')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-c', '/tmp/capcom-constraints.txt', *PINS], check=True)
    assert md.version('torch') == torch_v, 'torch was changed by pip'
got = {p.split('==')[0]: md.version(p.split('==')[0]) for p in PINS}
print('python', sys.version.split()[0], '| torch', md.version('torch'), '|', ' '.join(f'{k} {v}' for k, v in got.items()))
bad = {p: got[p.split('==')[0]] for p in PINS if got[p.split('==')[0]] != p.split('==')[1]}
assert not bad, f'version mismatch: {bad}'
import torch, transformers, trl, peft, accelerate, datasets, onnx, onnxruntime, onnxruntime_genai  # noqa: import check""" % (repr(list(PINS)),)

PATHS = """# ---- where the library, data and runs live ----
from pathlib import Path
ROOT = Path(os.environ['CAPCOM_ROOT']) if LOCAL else Path('/content/capcom')
CODE = ROOT / 'code'
for d in ('chat/data', 'chat/train'): (CODE / d).mkdir(parents=True, exist_ok=True)
for f in EMPTY: (CODE / f).touch()  # empty package files (%%writefile refuses an empty body)
print('library ->', CODE)"""

IMPORT = """# ---- import the library just written (re-running the library cells picks up changes) ----
import sys
if str(CODE) not in sys.path: sys.path.insert(0, str(CODE))
for m in [m for m in sys.modules if m == 'chat' or m.startswith('chat.')]: del sys.modules[m]
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
from chat.train import common, sft_stage, distill_stage, dpo_stage, grpo_stage, merge_stage, web_stage
from chat import eval as capcom_eval
print('library ok:', common.__file__)"""

PROBE = """# ---- GPU probe -> tier (t4 | l4 | a100 | h100 | big | mps | cpu), precision, stage defaults ----
INFO = common.probe(); print(INFO)
if not LOCAL: assert INFO['device'] == 'cuda', 'Runtime > Change runtime type > pick a GPU'
if STAGES['S5'] == 'auto': STAGES['S5'] = common.default_stages(INFO['tier'])['S5']
if TEACHER_SIZE == '2.6b' and (INFO['mem_gb'] or 0) < 39: print('the 2.6b teacher needs >= 40 GB: using 1.2b'); TEACHER_SIZE = '1.2b'
print('stages:', STAGES)
for s in ('s1', 's2', 's3', 's4', 's5', 'eval'): print(s, common.hp(s, INFO['tier'], bool(SMOKE), STUDENT))"""

DATA = """# ---- Google Drive + data: copy the zip to local disk, unzip, verify every file's sha256 against MANIFEST.json ----
import shutil, zipfile
if LOCAL: DRIVE = Path(os.environ.get('CAPCOM_DRIVE') or ROOT / 'drive')
else:
    from google.colab import drive
    drive.mount('/content/drive'); DRIVE = Path(DRIVE_DIR)
    try:
        from google.colab import userdata
        os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')  # optional: faster, un-throttled Hub downloads
    except Exception: pass
zsrc = DRIVE / DATA_ZIP
assert zsrc.exists(), f'upload {DATA_ZIP} to {DRIVE} first (see the top of the notebook)'
zloc, data = ROOT / DATA_ZIP, ROOT / 'data' / Path(DATA_ZIP).stem
if not zloc.exists() or zloc.stat().st_size != zsrc.stat().st_size: shutil.copy(zsrc, zloc)
if not (data / '.unzipped').exists():
    shutil.rmtree(data, ignore_errors=True); zipfile.ZipFile(zloc).extractall(data); (data / '.unzipped').write_text('ok')
PKG = common.find_package(data); bad = common.verify_manifest(PKG)
assert not bad, f'corrupt package: {bad}'
man = json.loads((PKG / 'MANIFEST.json').read_text())
print(PKG, man['version'], {k: v['rows'] for k, v in man['files'].items() if v['rows'] is not None})
RUN = common.Run(ROOT / 'runs' / RUN_NAME, DRIVE / 'runs' / RUN_NAME, PKG, INFO, STUDENT, TEACHER_SIZE, TEACHER_OVERRIDE, SMOKE)
(RUN.drive / 'requirements.lock').write_text(subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, cwd=Path.home()).stdout)
print('student', RUN.student_id, '| teacher', RUN.teacher_id, '| run', RUN.work, '->', RUN.drive)"""

STAGE_CELLS = [
    ('## Baseline\nThe untuned student with the same prompt contract: the reference row of the results table.',
     "capcom_eval.eval_stage(RUN, 'base')"),
    ('## S1 — teacher SFT\nLoRA r 64 / α 128 on every linear layer of the teacher, lr 1e-4, then merged into a full model: the distillation '
     'teacher of S3. The loss covers only the CAPCOM reply (+ end-of-turn token).', "if STAGES['S1']: sft_stage.teacher_sft(RUN)"),
    ('## S2 — student SFT\nFull fine-tune of the student: lr 3e-5 cosine, warmup 3 %, 3 epochs, effective batch 64, max length 1024; '
     '25 % general-chat replay.', "if STAGES['S2']: sft_stage.student_sft(RUN)"),
    ('### Eval S2\nGreedy replies to the held-out sets, scored by `chat/rewards.py`; multi-turn scripts with the teacher as the visitor.',
     "capcom_eval.eval_stage(RUN, 'S2')"),
    ('## S3 — on-policy distillation\n`trl.DistillationTrainer`: the student samples, the S1 teacher scores every token (reverse KL, '
     'β 1.0), lr 1e-5, 128-token completions.', "if STAGES['S3']: distill_stage.distill(RUN)"),
    ('### Eval S3', "capcom_eval.eval_stage(RUN, 'S3')"),
    ('## S4 — multi-turn-aware DPO\nFor each context: 4 samples; the teacher plays the visitor\'s next message, the student answers it; '
     'reward = score(reply) + 0.5 × score(next reply). Best vs worst pairs (margin ≥ 0.15) → `DPOTrainer` `apo_zero`, β 0.3, lr 1e-6. '
     'Two rounds (one on T4).', "if STAGES['S4']: dpo_stage.dpo(RUN)"),
    ('### Eval S4', "capcom_eval.eval_stage(RUN, 'S4')"),
    ('## S5 — GRPO (optional)\n`dapo` loss, verifiable reward components normalised per group (`normalize_then_sum`), β 0.02, lr 1e-6, '
     '200 steps. Off on T4.', "if STAGES['S5']: grpo_stage.grpo(RUN)"),
    ('### Eval S5', "capcom_eval.eval_stage(RUN, 'S5')"),
    ('## S6 — pick the best stage\nThe best eval score wins; a 0.9 × best + 0.1 × S3 merge is kept only if it scores higher.',
     "if STAGES['S6']: merge_stage.select(RUN)\nprint('final model:', RUN.final())"),
    ('## S7 — web export\nONNX Runtime GenAI builder int4 (block 32): q4f16 for WebGPU with shader-f16 + q4 for WebGPU without it '
     '(the 350M is impractical on WASM: the page falls back to retrieval-only cards), ctx 4096, transformers.js IO, int4 accuracy level '
     '`EXPORT_ACC`; greedy parity vs PyTorch; zipped to Drive.',
     "if STAGES['S7']: print(json.dumps(web_stage.web(RUN, dtypes=('q4',) if SMOKE else ('q4f16', 'q4'), acc=EXPORT_ACC), indent=1))"),
]

RESULTS = """# ---- results ----
print(capcom_eval.table(RUN.evals()))
print('\\nfinal:', RUN.final())
d = RUN.done('S7')
if d: print('web model zip:', d['zip'], f"({d['zip_mb']} MB)", '| sizes MB:', d['sizes_mb'], '| parity:', d['parity'])
print('run folder on Drive:', RUN.drive)"""

PUSH = """# ---- optional: push the web model folder to a private Hugging Face repo (Colab secret HF_TOKEN with write access) ----
if HF_PUSH and not LOCAL and RUN.done('S7'):
    from google.colab import userdata
    print(web_stage.push(RUN, HF_REPO, userdata.get('HF_TOKEN')))
else: print('HF push skipped')"""

def build():
    md, code = new_markdown_cell, new_code_cell
    src = {f: (REPO / f).read_text() for f in LIB}
    cells = [md(INTRO), md('## Configuration'), code(CONFIG), md('## Install'), code(INSTALL),
             md('## Library\nThe CAPCOM modules, written from the repository by `chat/colab/make_notebook.py` (no need to read them).'),
             code(PATHS.replace('EMPTY', repr(tuple(f for f in LIB if not src[f]))))]
    cells += [code(f'%%writefile {{CODE}}/{f}\n' + src[f]) for f in LIB if src[f]]
    cells += [code(IMPORT), md('## GPU'), code(PROBE), md('## Data'), code(DATA)]
    for m, c in STAGE_CELLS: cells += [md(m), code(c)]
    cells += [md('## Results'), code(RESULTS), md('## Optional: Hugging Face push'), code(PUSH)]
    for i, c in enumerate(cells): c['id'] = f'capcom-{i:02d}'  # stable ids: a deterministic file and clean diffs
    nb = new_notebook(cells=cells, metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
                                             'language_info': {'name': 'python'}, 'accelerator': 'GPU',
                                             'colab': {'provenance': [], 'gpuType': 'A100', 'machine_shape': 'hm'}})
    nbformat.validate(nb)
    return nb

def write(out=REPO / 'chat/colab/capcom_colab.ipynb'):
    nb = build(); Path(out).write_text(nbformat.writes(nb))
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='write the CAPCOM Colab notebook'); ap.add_argument('--out', default=str(REPO / 'chat/colab/capcom_colab.ipynb'))
    print(write(ap.parse_args().out))
