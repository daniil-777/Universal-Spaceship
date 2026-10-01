"""chat/train/distill_stage.py — S3 on-policy distillation (spec §6, R5; research: OPD beats RL ~10x for small students).
trl.DistillationTrainer: the student samples its own replies to prompts/train.jsonl contexts (conversational prompt-only rows: system
notes + state, history, visitor message) and matches the S1 teacher's next-token distribution on them — beta 1.0 (reverse KL),
temperature 1.0, lr 1e-5, max completion 128 tokens, 400 steps x 32 prompts on A100 (tier-scaled). The student starts from the best
evaluated stage so far (S2); the teacher is S1/model (the untuned teacher when S1 is off). Teacher and student share the vocabulary.
  python -m chat.train.distill_stage --pkg <package> --work <dir> [--drive <dir>] [--smoke] [--teacher <id>]"""
import torch
from chat.train.common import cli, free, infer_dtype, load_model, load_tok, log, prompt_rows, seed_all
from chat.train.sft_stage import save_model

def distill(run):
    """S3 -> S3/model. Skipped when DONE."""
    if run.done('S3'): return run.model('S3')
    from datasets import Dataset
    from trl import DistillationConfig, DistillationTrainer
    h, out = run.hp('s3'), run.dir('S3'); seed_all(run.seed)
    src_stage, src = run.best('S3'); teacher = run.model('S1') or run.teacher_id
    rows = prompt_rows(run.pkg, 'train', h['n_prompts'], run.seed)
    if not rows: raise RuntimeError('S3: prompts/train.jsonl is empty')
    log(f'S3: {len(rows)} prompts, student {src_stage} ({src}), teacher {teacher}, hp {h}')
    tok = load_tok(src, 'left')
    model = load_model(src, run.info, dtype=torch.float32)
    tmodel = load_model(teacher, run.info, dtype=infer_dtype(run.info)).eval()
    cfg = run.targs(DistillationConfig, 'S3', out, h, beta=h['beta'], temperature=h['temperature'], max_completion_length=h['max_completion'],
                    num_train_epochs=100, max_steps=h['max_steps'], warmup_steps=0.05, disable_dropout=True, use_vllm=False)
    tr = DistillationTrainer(model=model, teacher_model=tmodel, args=cfg, train_dataset=Dataset.from_list([{'prompt': r['prompt']} for r in rows]),
                             processing_class=tok, callbacks=[run.mirror_callback('S3')])
    res = tr.train(resume_from_checkpoint=run.resume('S3', out))
    save_model(tr.model, tok, out / 'model')
    info = {'student_from': src_stage, 'teacher': str(teacher), 'prompts': len(rows), 'steps': res.global_step,
            'train_loss': round(res.training_loss, 4)}
    del tr, model, tmodel; free()
    run.finish('S3', **info)
    return run.model('S3')

if __name__ == '__main__':
    run, _ = cli(__doc__)
    print(distill(run))
