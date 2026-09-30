"""vlm/train/narrator/lora_train.py — LoRA r 16 / alpha 32 on exactly the 210 text-model projections (spec §10.2 regex), SigLIP,
connector and embed_tokens frozen, batch <= 2 on MPS; the PyTorch SmolVLM gets Task 8's vision_positions_fix; the processor
comes from the exported (G2) folder and frames are square()d to 512² first (amendment A). --smoke = 100 steps on <= 400 rows;
--steps / --max-rows override; --max-minutes stops on the wall clock and still saves the adapter. --q4-features feeds image
features precomputed with the folder's vision_encoder_q4.onnx as image_hidden_states (no pixel_values; not needed per
amendment A, where training runs on fp32 features)."""
import argparse, json, os, re, time
# before transformers/huggingface_hub are imported (they read HF_HOME at import): downloads stay on LaCie
os.environ.setdefault('HF_HOME', '/Volumes/LaCie/astro-pilot/vlm/hf')
import torch
from torch.utils.data import DataLoader
from vlm.train.narrator.data import NarratorData, collate

RX = r'model\.text_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)'
N_TARGETS, SMOKE_STEPS, SMOKE_ROWS = 210, 100, 400

def lora_targets(model):
    names = [n for n, _ in model.named_modules() if re.fullmatch(RX, n)]
    assert len(names) == N_TARGETS and all(n.startswith('model.text_model.') for n in names), f'{len(names)} LoRA targets'
    return names

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', required=True, help='dataset dir (reads <data>/narrator/<split>.jsonl)'); ap.add_argument('--split', default='train')
    ap.add_argument('--out', required=True, help='adapter dir'); ap.add_argument('--processor-dir', required=True, help='the exported (G2) folder')
    ap.add_argument('--smoke', action='store_true'); ap.add_argument('--steps', type=int); ap.add_argument('--max-rows', type=int)
    ap.add_argument('--max-minutes', type=float, help='wall-clock stop; the adapter is saved'); ap.add_argument('--q4-features', action='store_true')
    ap.add_argument('--device', choices=('mps', 'cpu'), default='mps' if torch.backends.mps.is_available() else 'cpu')
    ap.add_argument('--batch', type=int, default=2); ap.add_argument('--lr', type=float, default=2e-4); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--log-every', type=int, default=10)
    a = ap.parse_args(argv)
    if a.batch > 2 or a.batch < 1: ap.error('Narrator training runs at batch <= 2 (spec §12)')
    if a.steps is None: a.steps = SMOKE_STEPS if a.smoke else 3000
    if a.max_rows is None and a.smoke: a.max_rows = SMOKE_ROWS
    return a

def train_loop(model, dl, opt, steps, dev, max_minutes=None, clock=time.time, log=print, log_every=10):
    """Optimizer steps over the loader (re-iterated per epoch) until `steps`, or until max_minutes of wall clock have passed."""
    assert len(dl) > 0, 'no training batches'
    step, t0, losses, stop = 0, clock(), [], 'steps'
    while step < steps and stop == 'steps':
        for b in dl:
            loss = model(**{k: v.to(dev) for k, v in b.items()}).loss
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item()); step += 1
            if step % log_every == 0 or step == steps:
                w = losses[-log_every:]; log(f'step {step} loss {sum(w) / len(w):.4f} {clock() - t0:.0f} s')
            if step >= steps: break
            if max_minutes is not None and clock() - t0 >= 60 * max_minutes:
                stop = 'time'; log(f'wall-clock stop at step {step} after {max_minutes} min'); break
    return {'steps': step, 'stop': stop, 'losses': losses, 'seconds': clock() - t0}

def q4_encoder(processor_dir):
    import numpy as np, onnxruntime as ort
    venc = ort.InferenceSession(f'{processor_dir}/onnx/vision_encoder_q4.onnx')
    return lambda full: venc.run(None, {'pixel_values': full['pixel_values'].numpy().astype(np.float32), 'pixel_attention_mask': full['pixel_attention_mask'].numpy().astype(bool)})[0]

def main(argv=None):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from vlm.train.narrator.export_decoder import BASE, vision_positions_fix
    a = parse_args(argv); torch.manual_seed(a.seed); dev = a.device
    p = AutoProcessor.from_pretrained(a.processor_dir)
    m = vision_positions_fix(AutoModelForVision2Seq.from_pretrained(BASE, dtype=torch.float32)); lora_targets(m)
    for prm in m.parameters(): prm.requires_grad = False
    m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=RX)); m.print_trainable_parameters()
    m.config.use_cache = False; m.to(dev).train()
    ds = NarratorData(f'{a.data}/narrator/{a.split}.jsonl', p, q4_encoder(a.processor_dir) if a.q4_features else None, limit=a.max_rows)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, collate_fn=collate, generator=torch.Generator().manual_seed(a.seed))
    opt = torch.optim.AdamW([x for x in m.parameters() if x.requires_grad], lr=a.lr)
    print(f'{len(ds)} rows, {a.steps} steps, batch {a.batch}, device {dev}', flush=True)
    r = train_loop(m, dl, opt, a.steps, dev, a.max_minutes, log=lambda s: print(s, flush=True), log_every=a.log_every)
    os.makedirs(a.out, exist_ok=True); m.save_pretrained(a.out)
    with open(f'{a.out}/train_log.json', 'w') as f: json.dump({**r, 'rows': len(ds), 'args': vars(a)}, f)
    print('saved adapter', a.out, f"({r['steps']} steps, stop {r['stop']})", flush=True)

if __name__ == '__main__': main()
