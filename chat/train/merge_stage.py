"""chat/train/merge_stage.py — S6 (spec §6, R5; SmolLM3's final recipe): pick the best evaluated student stage (S2..S5) by the eval score,
then try a linear interpolation 0.9 x best + 0.1 x S3 (the on-policy-distilled checkpoint) of the two state dicts; the merge is kept
only if its eval score beats the best stage. DONE.json records 'final' — the stage S7 exports.
  python -m chat.train.merge_stage --a <dir> --b <dir> --out <dir> [--w 0.9]     (a bare interpolation)"""
import torch
from chat.train.common import free, load_model, load_tok, log
from chat.train.sft_stage import save_model

CPU = {'device': 'cpu', 'bf16': False, 'fp16': False}

def interpolate(a, b, out, w=0.9):
    """out = w * a + (1 - w) * b over every floating parameter and buffer (same architecture; tied weights counted once), saved bf16 with
    a's config and tokenizer."""
    ma, mb = load_model(a, CPU, dtype=torch.float32), load_model(b, CPU, dtype=torch.float32)
    pb = {**dict(mb.named_parameters()), **dict(mb.named_buffers())}
    with torch.no_grad():
        for n, p in [*ma.named_parameters(), *ma.named_buffers()]:
            if p.is_floating_point() and n in pb: p.mul_(w).add_(pb[n].to(p.dtype), alpha=1 - w)
    save_model(ma, load_tok(a), out)
    del ma, mb; free()
    return out

def select(run, w=0.9):
    """S6 -> DONE {'final': stage, ...}; S6/model exists only when the merge won. Skipped when DONE."""
    d = run.done('S6')
    if d: return d
    from chat.eval import run_eval, table
    ev = run.evals(); best_stage, best_dir = run.best('S6'); s3 = run.model('S3')
    best_score = (ev.get(best_stage) or {}).get('score')
    info = {'best': best_stage, 'best_score': best_score, 'final': best_stage, 'merge_score': None}
    if s3 and best_stage not in ('S3', 'base'):
        out = run.dir('S6') / 'model'
        log(f'S6: merging {w} x {best_stage} + {1 - w:.1f} x S3')
        interpolate(best_dir, s3, out, w)
        rep = run_eval(out, run.pkg, run.hp('eval'), run.info, run.sim_id, run.kb); rep['stage'] = 'S6'; run.save_eval('S6', rep)
        print(table({'S6': rep}), flush=True)
        info['merge_score'] = rep['score']
        if rep['score'] is not None and (best_score is None or rep['score'] > best_score): info['final'] = 'S6'
        else:
            import shutil
            shutil.rmtree(out, ignore_errors=True)
    log(f"S6: final = {info['final']} (best {best_stage} {best_score}, merge {info['merge_score']})")
    return run.finish('S6', **info)

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='linear interpolation of two checkpoints')
    ap.add_argument('--a', required=True); ap.add_argument('--b', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--w', type=float, default=0.9)
    x = ap.parse_args(); print(interpolate(x.a, x.b, x.out, x.w))
