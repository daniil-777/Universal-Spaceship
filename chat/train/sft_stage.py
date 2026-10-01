"""chat/train/sft_stage.py — S1 teacher SFT and S2 student SFT (spec §6, R5).
S1: LoRA r 64 / alpha 128 on all linear layers of the teacher, lr 1e-4, 2 epochs on A100 (1 on T4/L4), then merge_and_unload -> a full
    HF model (bf16) in S1/model — the on-policy distillation teacher of S3.
S2: full fine-tune of the student, lr 3e-5 cosine (5e-5 for SmolLM2-135M), warmup 3 %, 3 epochs (2 on T4), effective batch 64, max length
    1024, bf16 mixed precision over fp32 master weights (fp16 on T4, fp32 on MPS/CPU).
Rows are TRL prompt-completion rows (chat/data/build.py) + GENERAL_FRAC general replay, tokenized here by pc_tokens so the loss covers
exactly the assistant completion and its end-of-turn token — never the prompt, the history turns (written under other notes) or the
newline the chat template appends after <|im_end|> — and handed to trl.SFTTrainer pre-tokenized (input_ids + labels).
  python -m chat.train.sft_stage --stage s2 --pkg <package> --work <dir> [--drive <dir>] [--smoke] [--teacher <id>]"""
import torch
from chat.train.common import cli, free, load_model, load_tok, log, seed_all, sft_rows, infer_dtype

def pc_tokens(tok, prompt, completion, max_len=None):
    """{'input_ids', 'labels'} of a prompt-completion row; labels are -100 on the prompt and cover the completion text + end-of-turn
    token. The prompt ids equal generation's (template + generation prompt); None when the row exceeds max_len."""
    p = tok.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
    full = tok.apply_chat_template(prompt + completion, tokenize=False)
    if not full.startswith(p): raise ValueError('chat template is not prefix-preserving for this row')
    tail, eos = full[len(p):], tok.eos_token
    i = tail.find(eos)
    tail = tail[:i + len(eos)] if i >= 0 else tail.rstrip() + eos
    ip, ic = tok(p, add_special_tokens=False)['input_ids'], tok(tail, add_special_tokens=False)['input_ids']
    if max_len and len(ip) + len(ic) > max_len: return None
    return {'input_ids': ip + ic, 'labels': [-100] * len(ip) + ic}

def tokenize_rows(tok, rows, max_len):
    """(tokenized rows with a 'length' column, {'too_long': n, 'bad': n})."""
    out, drop = [], {'too_long': 0, 'bad': 0}
    for r in rows:
        try: t = pc_tokens(tok, r['prompt'], r['completion'], max_len)
        except Exception: drop['bad'] += 1; continue
        if t is None: drop['too_long'] += 1; continue
        t['length'] = len(t['input_ids']); out.append(t)
    return out, drop

def save_model(model, tok, dst, dtype=torch.bfloat16):
    """Save a full HF model (bf16 by default: half the Drive space; later stages reload fp32 master weights) + tokenizer."""
    model.to(dtype).save_pretrained(str(dst)); tok.save_pretrained(str(dst))

def _sft(run, stage, base, h, lora):
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    seed_all(run.seed); out = run.dir(stage); tok = load_tok(base, 'right')
    rows = sft_rows(run.pkg, 'train', run.general_frac, h['max_rows'], run.seed)
    data, drop = tokenize_rows(tok, rows, h['max_len'])
    val, _ = tokenize_rows(tok, sft_rows(run.pkg, 'val', 0, 8 if run.smoke else 256, run.seed), h['max_len'])
    if not data: raise RuntimeError(f'{stage}: no trainable rows ({drop})')
    log(f'{stage}: {len(data)} rows ({drop}), {len(val)} val rows, base {base}, hp {h}')
    frozen = lora and run.info['device'] == 'cuda'  # LoRA: frozen base in bf16/fp16, fp32 adapters; full FT: fp32 master weights
    model = load_model(base, run.info, dtype=infer_dtype(run.info) if frozen else torch.float32)
    peft = None
    if lora:
        from peft import LoraConfig
        peft = LoraConfig(r=h['lora_r'], lora_alpha=h['lora_alpha'], lora_dropout=0.05, target_modules='all-linear', task_type='CAUSAL_LM')
    cfg = run.targs(SFTConfig, stage, out, h, max_length=h['max_len'], packing=False, train_sampling_strategy='group_by_length',
                    warmup_steps=h.get('warmup', 0.03), eval_strategy='no', per_device_eval_batch_size=h['bs'])
    tr = SFTTrainer(model=model, args=cfg, train_dataset=Dataset.from_list(data), eval_dataset=Dataset.from_list(val) if val else None,
                    processing_class=tok, peft_config=peft, callbacks=[run.mirror_callback(stage)])
    res = tr.train(resume_from_checkpoint=run.resume(stage, out))
    ev = tr.evaluate() if val else {}
    m = tr.model.merge_and_unload() if lora else tr.model
    save_model(m, tok, out / 'model')
    info = {'base': str(base), 'rows': len(data), 'dropped': drop, 'train_loss': round(res.training_loss, 4),
            'val_loss': round(ev['eval_loss'], 4) if 'eval_loss' in ev else None, 'steps': res.global_step}
    del tr, model, m; free()
    return info

def teacher_sft(run):
    """S1: LoRA SFT of run.teacher_id on the CAPCOM rows, merged -> S1/model. Skipped when DONE."""
    if run.done('S1'): return run.model('S1')
    info = _sft(run, 'S1', run.teacher_id, run.hp('s1'), lora=True)
    run.finish('S1', **info)
    return run.model('S1')

def student_sft(run):
    """S2: full fine-tune of run.student_id -> S2/model. Skipped when DONE."""
    if run.done('S2'): return run.model('S2')
    info = _sft(run, 'S2', run.student_id, run.hp('s2'), lora=False)
    run.finish('S2', **info)
    return run.model('S2')

if __name__ == '__main__':
    run, a = cli(__doc__, [('--stage', dict(choices=['s1', 's2'], required=True))])
    print((teacher_sft if a.stage == 's1' else student_sft)(run))
