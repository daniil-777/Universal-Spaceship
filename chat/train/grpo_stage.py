"""chat/train/grpo_stage.py — S5 optional rubric GRPO (spec §6, R6: verifiable rewards only; off by default on T4).
trl.GRPOTrainer, loss_type 'dapo', scale_rewards 'batch', num_generations 4 (2 on T4), beta 0.02, epsilon_high 0.28, lr 1e-6, 200 steps,
max completion 128, mask_truncated_completions. One reward function per chat.rewards.score component — recall, grounded, lead, first,
length, abstain, clean (min of repetition and leak) — weighted as in rewards.W and aggregated with multi_objective_aggregation
'normalize_then_sum' (GDPO: each reward normalised within its group before the weighted sum; TRL 1.14.1 supports it). A component
that does not apply to a row (recall on an abstain row) returns None -> NaN, which TRL's nan-aware aggregation ignores. Rows come from
prompts/train.jsonl with the scorer fields (notes, user, must, abstain, lead) as dataset columns, passed to the reward functions.
  python -m chat.train.grpo_stage --pkg <package> --work <dir> [--drive <dir>] [--smoke]"""
import torch
from chat.rewards import W, score
from chat.train.common import cli, free, load_model, load_tok, log, prompt_rows, seed_all
from chat.train.sft_stage import save_model

COMPONENTS = ('recall', 'grounded', 'lead', 'first', 'length', 'abstain', 'clean')

def text_of(c):
    """A completion as text (conversational: the last message's content)."""
    return c[-1]['content'] if isinstance(c, list) else c

def component(s, name):
    return min(s['repetition'], s['leak']) if name == 'clean' else s[name]

def reward_funcs():
    """([reward_<component>(prompts, completions, notes, user, must, abstain, lead, **kw) -> [float | None]], weights)."""
    fns = []
    for name in COMPONENTS:
        def f(prompts, completions, notes, user, must, abstain, lead, _name=name, **kw):
            return [component(score(text_of(c), {'notes': n, 'user': u, 'must': m or [], 'abstain': a, 'lead': l}), _name)
                    for c, n, u, m, a, l in zip(completions, notes, user, must, abstain, lead)]
        f.__name__ = f'reward_{name}'; fns.append(f)
    return fns, [W['repetition'] + W['leak'] if n == 'clean' else W[n] for n in COMPONENTS]

def grpo(run):
    """S5 -> S5/model, from the best evaluated stage so far (S4). Skipped when DONE."""
    if run.done('S5'): return run.model('S5')
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    h, out = run.hp('s5'), run.dir('S5'); seed_all(run.seed)
    src_stage, src = run.best('S5')
    rows = prompt_rows(run.pkg, 'train', h['n_prompts'], seed=run.seed + 5)
    ds = Dataset.from_list([{'prompt': r['prompt'], 'notes': r['notes'], 'user': r['user'], 'must': r.get('must') or [],
                             'abstain': bool(r.get('abstain')), 'lead': bool(r.get('lead', True))} for r in rows])
    log(f'S5: {len(rows)} prompts from {src_stage} ({src}), hp {h}')
    fns, weights = reward_funcs()
    tok = load_tok(src, 'left'); model = load_model(src, run.info, dtype=torch.float32)
    cfg = run.targs(GRPOConfig, 'S5', out, h, loss_type='dapo', scale_rewards='batch', num_generations=h['num_generations'], beta=h['beta'],
                    epsilon_high=h['epsilon_high'], max_completion_length=h['max_completion'], mask_truncated_completions=True,
                    multi_objective_aggregation='normalize_then_sum', reward_weights=weights, temperature=1.0, num_train_epochs=100,
                    warmup_steps=0.05, lr_scheduler_type='constant_with_warmup', use_vllm=False,
                    model_init_kwargs={'dtype': 'bfloat16' if run.info['bf16'] else 'float32'})  # the reference model
    tr = GRPOTrainer(model=model, reward_funcs=fns, args=cfg, train_dataset=ds, processing_class=tok, callbacks=[run.mirror_callback('S5')])
    res = tr.train(resume_from_checkpoint=run.resume('S5', out))
    save_model(tr.model, tok, out / 'model')
    info = {'policy_from': src_stage, 'prompts': len(rows), 'steps': res.global_step, 'train_loss': round(res.training_loss, 4)}
    del tr, model; free()
    run.finish('S5', **info)
    return run.model('S5')

if __name__ == '__main__':
    run, _ = cli(__doc__)
    print(grpo(run))
