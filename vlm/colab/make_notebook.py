"""vlm/colab/make_notebook.py — writes vlm/colab/APV_train_all.ipynb (nbformat 4) from the cells below, so the notebook's code
stays reviewable here. Every code cell is a few calls into vlm/train/colab (which the tests exercise); the ids name the stages
the dry run (vlm/colab/dryrun.py) runs up to.
  python vlm/colab/make_notebook.py"""
import json, os

MD_INTRO = """# Astro Pilot Vision v1: train Pilot Eye and the Narrator, then export the web models

**Run all** (Runtime > Run all) on a **GPU** runtime (Runtime > Change runtime type: A100 is best, then L4; T4 works, slower).
It uses the dataset package you uploaded to **My Drive/apv-open-v1/** and, in order:

1. mounts Drive, checks the package, installs the pinned libraries and picks settings for this GPU;
2. copies the package to the VM's local disk (every file checked against SHA256SUMS) and unpacks the shards;
3. **Pilot Eye**: a full fine-tune (MobileNetV4-Conv-S, mixed precision) for `TIME_BUDGET_EYE_MIN`, then the test metrics
   and gates, and the ONNX export with a parity check;
4. **Narrator**: LoRA on SmolVLM-256M for `TIME_BUDGET_NARRATOR_MIN` (bf16 on A100/L4, fp16 on T4), over a stratified
   epoch of the training rows, with a checkpoint on Drive every `CHECKPOINT_EVERY_MIN` minutes;
5. merges the LoRA and exports the web build (q4f16 decoder, q8 vision encoder, fp16 embeddings: the narrator-v0 layout),
   runs the parity and stop-rate gates and the slot evaluation on the test views;
6. writes **My Drive/apv-open-v1/apv-models-v1.zip** and prints how to install it.

If Colab disconnects, open the notebook again and **Run all**: finished stages are skipped and training resumes from its
last Drive checkpoint (in `apv-open-v1/runs/v1/`). The time budgets count across re-runs."""

SETTINGS = """#@title Settings (the defaults are fine) { display-mode: "form" }
DATA_DIR = '/content/drive/MyDrive/apv-open-v1'   # the uploaded package folder on Google Drive
TIME_BUDGET_EYE_MIN = 30          # Pilot Eye training minutes (counted across re-runs)
TIME_BUDGET_NARRATOR_MIN = 180    # Narrator LoRA training minutes (counted across re-runs)
NARRATOR_EPOCH_ROWS = 400_000     # one stratified epoch (family x task); the time budget usually ends it first
CHECKPOINT_EVERY_MIN = 15         # a Drive checkpoint every N minutes (a disconnect loses at most this much)
INCLUDE_FP32_REFERENCE = False    # also zip the 540 MB fp32 parity-reference decoder (the web app never loads it)

import os
DRY_RUN = os.environ.get('APV_DRYRUN') == '1'     # a CPU smoke test of every cell with tiny steps (vlm/colab/dryrun.py)
if DRY_RUN:
    DATA_DIR = os.environ['APV_DATA_DIR']; TIME_BUDGET_EYE_MIN, TIME_BUDGET_NARRATOR_MIN, NARRATOR_EPOCH_ROWS, CHECKPOINT_EVERY_MIN = 2, 3, 32, 1
OUT_DIR = os.environ.get('APV_OUT_DIR') or f'{DATA_DIR}/runs/v1'     # checkpoints and metrics, on Drive
LOCAL_DIR = os.environ.get('APV_LOCAL_DIR') or '/content/apv'        # the unpacked package and the exports, on the VM's disk
ZIP_PATH = os.environ.get('APV_ZIP_PATH') or f'{DATA_DIR}/apv-models-v1.zip'
MAX_STEPS = {'eye': 3, 'narrator': 2} if DRY_RUN else {'eye': None, 'narrator': None}
print(f'data {DATA_DIR} | runs {OUT_DIR} | local {LOCAL_DIR} | zip {ZIP_PATH}' + (' | DRY RUN' if DRY_RUN else ''))"""

SETUP = """import os, sys, json, time, shutil
try:
    from google.colab import drive
    drive.mount('/content/drive'); ON_COLAB = True
except ImportError:
    ON_COLAB = False; print('Not on Colab: Google Drive is not mounted.')
if not os.path.exists(f'{DATA_DIR}/manifest.json'):
    raise FileNotFoundError(f'{DATA_DIR}/manifest.json not found: upload the package folder to the top of My Drive (see its README.md), '
                            'or set DATA_DIR in the first cell to where it is.')
MANIFEST = json.load(open(f'{DATA_DIR}/manifest.json'))
CODE = f'{LOCAL_DIR}/code'
shutil.rmtree(CODE, ignore_errors=True); shutil.copytree(f'{DATA_DIR}/code', CODE); sys.path.insert(0, CODE)
if ON_COLAB: os.environ['HF_HOME'] = '/content/hf'
os.makedirs(OUT_DIR, exist_ok=True)
print(f"package {MANIFEST['id']} ({MANIFEST['source']}): {MANIFEST['records']} records {MANIFEST['counts']}, {MANIFEST['total_bytes'] / 1e9:.2f} GB")"""

GPU = """from vlm.train.colab import env
if ON_COLAB:
    t = time.time(); env.pip_install(env.PIP_CORE); env.pip_install(env.PIP_EXTRA, required=False); print(f'libraries installed in {time.time() - t:.0f} s')
GPU = env.detect_gpu(); PLAN = env.plan_for(GPU)
print(json.dumps(PLAN, indent=1))
if GPU['device'] != 'cuda' and not DRY_RUN:
    raise RuntimeError('No GPU: Runtime > Change runtime type > pick a GPU (A100, L4 or T4), then Runtime > Run all again.')
os.environ['SMOLVLM_DIR'] = os.environ.get('SMOLVLM_DIR') or env.smolvlm_files()   # the base model's configs and published ONNX parts
print('SmolVLM base files:', os.environ['SMOLVLM_DIR'])"""

DATA = """from vlm.train.colab import pack_io
t = time.time(); MANIFEST = pack_io.sync_package(DATA_DIR, LOCAL_DIR)
os.environ['APV_FRAMES_ROOT'] = pack_io.narrator_frame_root(LOCAL_DIR)   # a narrator row names '<shard>/<key>.jpg'
print(f'package ready on {LOCAL_DIR} in {time.time() - t:.0f} s')"""

EYE_TRAIN = """from vlm.train.colab import eye
EYE_RES = eye.train_eye(LOCAL_DIR, f'{OUT_DIR}/pilot-eye', PLAN, budget_min=TIME_BUDGET_EYE_MIN, ckpt_min=min(5, CHECKPOINT_EVERY_MIN), max_steps=MAX_STEPS['eye'])"""

EYE_EXPORT = """EYE_EXPORT = f'{LOCAL_DIR}/export/pilot-eye-v1'
EYE_SUMMARY = eye.evaluate_and_export(LOCAL_DIR, f'{OUT_DIR}/pilot-eye', EYE_EXPORT, device=GPU['device'])"""

NAR_TRAIN = """from vlm.train.colab import narrator
NAR_RES = narrator.train_narrator(LOCAL_DIR, f'{OUT_DIR}/narrator', f'{CODE}/processor', PLAN, budget_min=TIME_BUDGET_NARRATOR_MIN,
                                  epoch_rows=NARRATOR_EPOCH_ROWS, ckpt_min=CHECKPOINT_EVERY_MIN, max_steps=MAX_STEPS['narrator'])"""

NAR_EXPORT = """EVAL_DIR, WORK, NAR_EXPORT = f'{LOCAL_DIR}/eval', f'{LOCAL_DIR}/work', f'{LOCAL_DIR}/export/narrator-v1'
print('evaluation rows:', narrator.eval_rows(f'{LOCAL_DIR}/records.jsonl', os.environ['APV_FRAMES_ROOT'], EVAL_DIR))
NAR_PARITY = narrator.merge_export(NAR_RES['adapter'], WORK, NAR_EXPORT, f'{CODE}/baselines/g2_parity.json', EVAL_DIR)"""

SLOTS = """try:
    SLOTS = narrator.slot_eval(WORK, NAR_EXPORT, f'{LOCAL_DIR}/records.jsonl', EVAL_DIR, env.ensure_node(), CODE, device=GPU['device'])
except Exception as e:   # the models are packaged either way
    SLOTS = {'error': repr(e)}; print('slot evaluation failed:', e)"""

ZIP = """from vlm.train.colab import package
METRICS = {'plan': PLAN, 'package': {k: MANIFEST[k] for k in ('id', 'source', 'records', 'counts')}, 'pilot_eye_train': EYE_RES, 'pilot_eye_test': EYE_SUMMARY,
           'narrator_train': NAR_RES, 'narrator_parity': NAR_PARITY, 'narrator_slots': SLOTS}
env.save_json(f'{OUT_DIR}/metrics_v1.json', METRICS)
MAN = package.make_zip(ZIP_PATH, EYE_EXPORT, NAR_EXPORT, METRICS, dataset_id=MANIFEST['id'], include_fp32=INCLUDE_FP32_REFERENCE)
print('Pilot Eye gates failed:', EYE_SUMMARY.get('gates_failed')); print('Narrator parity failures:', NAR_PARITY.get('fails'))
print('Narrator slot gates:', SLOTS.get('gates'))
print(package.instructions(ZIP_PATH))"""

MD_END = """## If something goes wrong
- **Disconnected / runtime reset**: open the notebook again and Run all. Pilot Eye and the Narrator resume from `runs/v1/` on Drive.
- **Out of GPU memory**: lower the Narrator batch: after the GPU cell, run `PLAN['narrator']['batch'] //= 2; PLAN['narrator']['accum'] *= 2`.
- **Start over**: delete `My Drive/apv-open-v1/runs/v1/` (and the zip), then Run all.
- **A gate failed**: the zip is written anyway; the metrics are in the zip (`metrics/`) and in `runs/v1/metrics_v1.json`."""

CELLS = [('md', 'intro', MD_INTRO), ('code', 'settings', SETTINGS), ('code', 'setup', SETUP), ('code', 'gpu', GPU), ('code', 'data', DATA),
         ('code', 'eye_train', EYE_TRAIN), ('code', 'eye_export', EYE_EXPORT), ('code', 'narrator_train', NAR_TRAIN), ('code', 'narrator_export', NAR_EXPORT),
         ('code', 'slots', SLOTS), ('code', 'zip', ZIP), ('md', 'help', MD_END)]

def notebook():
    cells = []
    for kind, cid, src in CELLS:
        lines = src.split('\n'); source = [l + '\n' for l in lines[:-1]] + [lines[-1]]
        c = {'cell_type': 'markdown' if kind == 'md' else 'code', 'id': cid, 'metadata': {}, 'source': source}
        if kind == 'code': c.update(execution_count=None, outputs=[])
        cells.append(c)
    return {'cells': cells, 'metadata': {'accelerator': 'GPU', 'colab': {'provenance': [], 'gpuType': 'A100', 'machine_shape': 'hm'},
            'kernelspec': {'display_name': 'Python 3', 'name': 'python3'}, 'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}

if __name__ == '__main__':
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'APV_train_all.ipynb')
    with open(out, 'w') as f: json.dump(notebook(), f, indent=1); f.write('\n')
    print('wrote', out)
