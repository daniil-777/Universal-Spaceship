"""vlm/train/pilot_eye/export_onnx.py — encoder.onnx ([1,3,H,W] -> [1,64,H/32,W/32]) and heads.onnx (m0, m1, m2, dt -> 8 heads),
opset 17; parity against PyTorch CPU on 32 val samples (max |d| <= 1e-3, spec §11.4; the first non-empty of val, test, train,
so a pilot with an empty val split still gets a real check) plus 2 random inputs; optional dynamic int8 encoder kept only if
the mean within-family verdict AUROC drops by <= 0.01; labels.json, ATTRIBUTION.txt and MODEL_CARD.md copied in.
  python -m vlm.train.pilot_eye.export_onnx --ckpt <model.pt> --data <dataset> --out <dir> [--int8]
  python -m vlm.train.pilot_eye.export_onnx --untrained --size 160x96 --out <dir>      (latency bench input)"""
import argparse, json, os, shutil, numpy as np, onnxruntime as ort, torch
from vlm.train.pilot_eye.model import PilotEye, HEADS, BACKBONE
from vlm.train.pilot_eye.data import PilotEyeData
TOL = 1e-3
CARD = """# Pilot Eye

Backbone: timm {backbone} (Apache-2.0; ImageNet-1k weights from timm) with a 1x1 reduction to 64 channels, and an MLP on
three frames (the newest map and two differences, avg + max pooled) plus the two frame spacings. Input {size} RGB in [0, 1]
(box-resized, vlm/gen/boxresize.js); the encoder normalises internally. Heads: {heads}.
Dataset: {data} (licence profile {licence}). Checkpoint: {ckpt}.

Intended use: an in-browser safety monitor for Astro Pilot frames; simulator-only (sim-to-real gap), not for real flight,
landing or docking decisions.
"""

def export(m, out, H, W):
    m = m.cpu().eval()
    torch.onnx.export(m.enc, torch.rand(1, 3, H, W), f'{out}/encoder.onnx', input_names=['pixels'], output_names=['map'], opset_version=17, dynamo=False)
    maps = [torch.rand(1, 64, H // 32, W // 32) for _ in range(3)]
    torch.onnx.export(m.heads, (*maps, torch.ones(1, 2)), f'{out}/heads.onnx', input_names=['m0', 'm1', 'm2', 'dt'], output_names=list(HEADS), opset_version=17, dynamo=False)

def _diff(m, enc, hd, f, dt):
    with torch.no_grad(): ref = m(f[None], dt[None])
    maps = [enc.run(None, {'pixels': f[k:k + 1].numpy()})[0] for k in range(3)]
    got = hd.run(None, {'m0': maps[0], 'm1': maps[1], 'm2': maps[2], 'dt': dt[None].numpy()})
    return max(float(np.abs(r.numpy() - g).max()) for r, g in zip(ref, got))

def sessions(out, enc_file='encoder.onnx'):
    return ort.InferenceSession(f'{out}/{enc_file}', providers=['CPUExecutionProvider']), ort.InferenceSession(f'{out}/heads.onnx', providers=['CPUExecutionProvider'])

def parity_split(data, H, W, n=32):
    """(split, dataset) of the first non-empty of val, test, train."""
    for s in ['val', 'test', 'train']:
        ds = PilotEyeData(data, s, limit=n, H=H, W=W)
        if len(ds): return s, ds
    return None, None

def parity(m, out, ds, n=32):
    enc, hd = sessions(out); return max([_diff(m, enc, hd, *ds[i][:2]) for i in range(min(n, len(ds)))], default=0.0)

def random_parity(m, out, H, W, n=2):
    g = torch.Generator().manual_seed(0); enc, hd = sessions(out)
    return max(_diff(m, enc, hd, torch.rand(3, 3, H, W, generator=g), torch.rand(2, generator=g) + 0.5) for _ in range(n))

def licence(data):
    try:
        with open(f'{data}/records.jsonl') as f: return json.loads(f.readline())['render']['licence_profile']
    except (OSError, ValueError, KeyError, TypeError): return 'open'

def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--ckpt'); ap.add_argument('--untrained', action='store_true'); ap.add_argument('--size', default=None, help="default: the checkpoint's size, else 160x96")
    ap.add_argument('--out', required=True); ap.add_argument('--data'); ap.add_argument('--int8', action='store_true')
    ap.add_argument('--no-pretrained', action='store_true', help='with --untrained: a random backbone (tests)')
    a = ap.parse_args(argv)
    if bool(a.ckpt) == a.untrained: ap.error('give exactly one of --ckpt and --untrained')
    return a

def main(argv=None):
    a = parse(argv); ck = torch.load(a.ckpt, map_location='cpu') if a.ckpt else None
    size = a.size or (ck['size'] if ck else '160x96'); W, H = map(int, size.split('x')); os.makedirs(a.out, exist_ok=True)
    m = PilotEye(pretrained=a.untrained and not a.no_pretrained)
    if ck: m.load_state_dict(ck['state'])
    m.eval(); export(m, a.out, H, W); res = {'size': size, 'max_abs_diff_random': random_parity(m, a.out, H, W)}
    if a.data:
        split, ds = parity_split(a.data, H, W)
        res.update(parity_split=split, n=min(32, len(ds)) if ds else 0, max_abs_diff=parity(m, a.out, ds) if ds else 0.0)
        shutil.copy(f'{a.data}/labels.json', f'{a.out}/labels.json')
        if os.path.exists(f'{a.data}/ATTRIBUTION.txt'): shutil.copy(f'{a.data}/ATTRIBUTION.txt', f'{a.out}/ATTRIBUTION.txt')
        if a.int8:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            from vlm.train.pilot_eye.evaluate import ort_family_auroc
            quantize_dynamic(f'{a.out}/encoder.onnx', f'{a.out}/encoder_int8.onnx', weight_type=QuantType.QInt8)
            f32, i8 = ort_family_auroc(a.out, a.data, 'encoder.onnx', H, W), ort_family_auroc(a.out, a.data, 'encoder_int8.onnx', H, W)
            kept = f32 is not None and i8 is not None and f32 - i8 <= 0.01; res['int8'] = {'fp32': f32, 'int8': i8, 'kept': kept}
            if not kept: os.remove(f'{a.out}/encoder_int8.onnx')
    with open(f'{a.out}/MODEL_CARD.md', 'w') as f:
        f.write(CARD.format(backbone=BACKBONE, size=size, heads=', '.join(f'{k} {n}' for k, n in HEADS.items()), data=os.path.basename(a.data.rstrip('/')) if a.data else 'none',
                            licence=licence(a.data) if a.data else 'n/a', ckpt=f"step {ck.get('best_step', ck.get('step'))} of {a.ckpt}" if ck else 'untrained heads'))
    with open(f'{a.out}/parity.json', 'w') as f: json.dump(res, f, indent=1)
    print(json.dumps(res))
    if res.get('max_abs_diff', 0.0) > TOL or res['max_abs_diff_random'] > TOL: raise SystemExit(f'PARITY FAIL: max |d| > {TOL}')

if __name__ == '__main__': main()
