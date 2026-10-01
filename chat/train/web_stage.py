"""chat/train/web_stage.py — S7 (spec §6, R7): export S6's final model with chat.export_web (ONNX Runtime GenAI builder int4 block 32:
q4f16 for WebGPU with shader-f16 + q4 for WebGPU without it — the 350M is impractical on WASM (no GatherBlockQuantized kernel in
ORT-web WASM), so the page falls back to retrieval-only cards there; ctx 4096; transformers.js IO; acc = the int4 accuracy level,
default 4 until the browser A/B decides), the greedy parity check (PyTorch fp32 vs the q4 graph on CPU),
kb.json + the eval reports + capcom.json (what was shipped) beside the model, zipped as capcom-web-<run>.zip on Drive. push() uploads
the web folder to a private Hugging Face repo (the notebook's optional last cell).
  python -m chat.train.web_stage --pkg <package> --work <dir> [--drive <dir>] [--smoke]"""
import json, shutil, zipfile
from pathlib import Path
from chat.train.common import cli, log

def web(run, dtypes=None, ctx=4096, acc=None):
    """S7 -> S7/web + capcom-web-<run>.zip (local and Drive). Skipped when DONE."""
    d = run.done('S7')
    if d: return d
    from chat.export_web import check, export
    stage, src = run.final(); dtypes = tuple(dtypes or (('q4',) if run.smoke else ('q4f16', 'q4')))
    out = run.dir('S7') / 'web'; shutil.rmtree(out, ignore_errors=True)
    log(f'S7: exporting {stage} ({src}) as {dtypes}, ctx {ctx}, int4 accuracy level {acc or 4}')
    sizes = export(src, out, dtypes, ctx, acc)
    parity = check(src, out) if 'q4' in dtypes else None
    log(f'S7: sizes {sizes} MB, parity {parity}')
    if run.kb: (out / 'kb.json').write_text(json.dumps(run.kb, ensure_ascii=False))
    ev = run.evals(); (out / 'evals').mkdir(exist_ok=True)
    for s, rep in ev.items(): (out / 'evals' / f'eval_{s}.json').write_text(json.dumps(rep, indent=1))
    card = {'student': run.student_id, 'stage': stage, 'eval_score': (ev.get(stage) or {}).get('score'), 'teacher': run.teacher_id,
            'dtypes': list(dtypes), 'sizes_mb': sizes, 'parity': parity, 'ctx': ctx, 'int4_accuracy_level': acc or 4,
            'runtime': 'WebGPU q4f16 (shader-f16) -> WebGPU q4 -> retrieval-only cards', 'prompt_contract': 'v1 (chat/prompt.py)',
            'kb': {k: run.kb.get(k) for k in ('name', 'version', 'retrieval')} if run.kb else None,
            'decoding': {'max_new_tokens': 120, 'repetition_penalty': 1.1, 'do_sample': False}, 'smoke': run.smoke}
    (out / 'capcom.json').write_text(json.dumps(card, indent=1))
    z = run.dir('S7') / f'capcom-web-{run.work.name}.zip'
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out.rglob('*')):
            if p.is_file() and not p.name.startswith('._'): zf.write(p, f'capcom-web-{run.work.name}/{p.relative_to(out)}')
    if run.drive: shutil.copy(z, run.drive / z.name)
    return run.finish('S7', final=stage, sizes_mb=sizes, parity=parity, acc=acc or 4, zip=str(run.drive / z.name if run.drive else z),
                      zip_mb=round(z.stat().st_size / 2 ** 20, 1))

def push(run, repo_id, token, private=True):
    """Upload S7/web to the Hugging Face Hub (a private model repo by default); returns the repo URL."""
    from huggingface_hub import HfApi
    api = HfApi(token=token); api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=str(run.work / 'S7' / 'web'), repo_id=repo_id, commit_message=f'CAPCOM {run.work.name}',
                      ignore_patterns=['._*', '.DS_Store'])
    return f'https://huggingface.co/{repo_id}'

if __name__ == '__main__':
    run, _ = cli(__doc__)
    print(json.dumps(web(run), indent=1))
