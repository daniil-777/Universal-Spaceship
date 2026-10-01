"""chat/tests/nb_e2e.py — runs chat/colab/capcom_colab.ipynb end to end on this machine (spec §6/§9: "SMOKE=1 runs every stage for a few
steps"): nbclient on the chat venv's kernel 'capcom-venv' (registered into the venv on first use), CAPCOM_LOCAL=1 (no Colab, no pip),
CAPCOM_ROOT = a fresh local root whose drive/ holds the dev package zip — plus 3 single / 3 unanswerable held-out rows built from
data/eval/*_raw.jsonl by chat.data.build.eval_rows and 2 dialog scripts, since the dev build leaves eval/ empty — SMOKE=1, the teacher
overridden to the 350M student (memory), every stage S1..S7 on (S7 exports q4 only), MPS/CPU. Checks: the committed notebook equals
make_notebook's output, every cell ran, DONE.json for S1..S7 on the fake Drive, evals for base/S2..S5 with all four eval sets, the q4
graph + the web zip. Then re-runs the notebook: every stage must be skipped (DONE markers) and the pass must be much faster.
  /Volumes/LaCie/astro-pilot/chat/venv/bin/python -m chat.tests.nb_e2e [--root DIR] [--zip ZIP] [--no-rerun]"""
import argparse, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
import nbformat
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path('/Volumes/LaCie/astro-pilot/chat')  # data/eval/*_raw.jsonl + dialog_scripts.jsonl
NB = REPO / 'chat/colab/capcom_colab.ipynb'
KERNEL = 'capcom-venv'

def ensure_kernel():
    """Register the venv's python as the 'capcom-venv' kernel (from ~, then dot_clean: the venv lives on exFAT)."""
    from jupyter_client.kernelspec import KernelSpecManager
    if KERNEL not in KernelSpecManager().find_kernel_specs():
        subprocess.run([sys.executable, '-m', 'ipykernel', 'install', '--sys-prefix', '--name', KERNEL, '--display-name', 'CAPCOM venv'],
                       check=True, cwd=Path.home())
        subprocess.run(['dot_clean', '-m', str(Path(sys.prefix) / 'share/jupyter/kernels')], check=False)

def smoke_zip(src_zip, out_dir):
    """A copy of the dev package zip with a few real held-out eval rows (chat.data.build.eval_rows over data/eval/*_raw.jsonl) and 2
    dialog scripts, MANIFEST sha256/bytes/rows rewritten — so the notebook's eval exercises every set."""
    from chat.data.build import eval_rows, write_jsonl
    from chat.retrieval import BM25
    from chat.train.common import find_package, read_jsonl, sha256
    tmp = out_dir / '.pkg'; shutil.rmtree(tmp, ignore_errors=True); zipfile.ZipFile(src_zip).extractall(tmp); pkg = find_package(tmp)
    kb = json.loads((pkg / 'kb.json').read_text())
    single, unans = eval_rows(DATA_ROOT, {f['id']: f for f in kb['facts']}, BM25(kb['facts']), kb['retrieval']['min_score'])
    sets = {'single': single[:3], 'unanswerable': unans[:3], 'dialog_scripts': read_jsonl(DATA_ROOT / 'data/eval/dialog_scripts.jsonl')[:2]}
    for k, rows in sets.items():
        assert rows, f'no {k} rows for the e2e'
        write_jsonl(pkg / 'eval' / f'{k}.jsonl', rows)
    man = json.loads((pkg / 'MANIFEST.json').read_text())
    for f, m in man['files'].items():
        q = pkg / f; m.update(sha256=sha256(q), bytes=q.stat().st_size, rows=len(read_jsonl(q)) if q.suffix == '.jsonl' else None)
    (pkg / 'MANIFEST.json').write_text(json.dumps(man, indent=1))
    z = out_dir / Path(src_zip).name
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as zf:
        for q in sorted(pkg.rglob('*')):
            if q.is_file() and not q.name.startswith('._'): zf.write(q, f'{pkg.name}/{q.relative_to(pkg)}')
    shutil.rmtree(tmp, ignore_errors=True)  # exFAT: macOS drops the ._ companions mid-walk
    return z

def execute(nb, root, env, out):
    os.environ.update(env)  # the kernel inherits this process's environment
    t = time.time()
    try: NotebookClient(nb, timeout=5400, kernel_name=KERNEL, resources={'metadata': {'path': str(root)}}).execute()
    finally: out.write_text(nbformat.writes(nb))
    return time.time() - t

def main():
    ap = argparse.ArgumentParser(description='CAPCOM notebook end-to-end (SMOKE)')
    ap.add_argument('--root', default='/Volumes/LaCie/astro-pilot/chat/work/nb-e2e')
    ap.add_argument('--zip', default='/Volumes/LaCie/astro-pilot/chat/data/capcom-data-dev.zip')
    ap.add_argument('--teacher', default='LiquidAI/LFM2.5-350M'); ap.add_argument('--no-rerun', action='store_true')
    a = ap.parse_args()
    sys.path.insert(0, str(REPO))
    from chat.colab import make_notebook
    assert nbformat.writes(make_notebook.build()) == NB.read_text(), 'capcom_colab.ipynb is stale: python3 chat/colab/make_notebook.py'
    ensure_kernel()
    root = Path(a.root); shutil.rmtree(root, ignore_errors=True); (root / 'drive').mkdir(parents=True)
    smoke_zip(a.zip, root / 'drive')
    env = {'CAPCOM_LOCAL': '1', 'CAPCOM_ROOT': str(root), 'CAPCOM_SMOKE': '1', 'CAPCOM_TEACHER': a.teacher, 'CAPCOM_DATA_ZIP': Path(a.zip).name,
           'CAPCOM_RUN': 'e2e', 'CAPCOM_STAGES': json.dumps({s: True for s in ('S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7')}),
           'HF_HOME': os.environ.get('HF_HOME', '/Volumes/LaCie/astro-pilot/chat/hf'), 'HF_HUB_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
           'PYTORCH_ENABLE_MPS_FALLBACK': '1'}
    t1 = execute(nbformat.read(NB, as_version=4), root, env, root / 'executed.ipynb')
    print(f'first pass: {t1:.0f} s', flush=True)
    drive = root / 'drive/runs/e2e'
    for s in ('S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7'): assert (drive / s / 'DONE.json').exists(), f'{s} has no DONE marker'
    for s in ('base', 'S2', 'S3', 'S4', 'S5'):
        rep = json.loads((drive / 'evals' / f'eval_{s}.json').read_text())
        assert set(rep['sets']) == {'single', 'unanswerable', 'val', 'dialogs'} and rep['score'] is not None, (s, rep['sets'].keys())
    s7 = json.loads((drive / 'S7/DONE.json').read_text())
    assert (root / 'runs/e2e/S7/web/onnx/model_q4.onnx').exists() and Path(s7['zip']).exists(), s7
    s6 = json.loads((drive / 'S6/DONE.json').read_text())
    print('S6:', {k: s6.get(k) for k in ('best', 'best_score', 'merge_score', 'final')}, '| S7:', {k: s7.get(k) for k in ('sizes_mb', 'parity', 'zip_mb')})
    done = {s: json.loads((drive / s / 'DONE.json').read_text()) for s in ('S1', 'S2', 'S3', 'S4', 'S5')}
    print('stages:', json.dumps({s: {k: d.get(k) for k in ('steps', 'pairs', 'train_loss') if k in d} for s, d in done.items()}))
    sys.path.insert(0, str(REPO)); from chat.eval import table
    print(table({p.stem[5:]: json.loads(p.read_text()) for p in sorted((drive / 'evals').glob('eval_*.json')) if not p.name.startswith('._')}))
    t2 = None
    if not a.no_rerun:  # resumability: a second Run-all skips every finished stage
        t2 = execute(nbformat.read(NB, as_version=4), root, env, root / 'executed_rerun.ipynb')
        for s, d in done.items(): assert json.loads((drive / s / 'DONE.json').read_text()) == d, f'{s} re-ran'
        print(f'second pass (all DONE): {t2:.0f} s', flush=True)
        assert t2 < t1 / 2, 'the re-run did not skip the finished stages'
    print(json.dumps({'nb_e2e': 'PASS', 'first_pass_s': round(t1), 'rerun_s': round(t2) if t2 else None}))

if __name__ == '__main__':
    main()
