"""Parity of an exported Narrator folder against PyTorch fp32 on CPU (spec §10.2 with the G2 round-2 rulings). Every
decoder is fed the same inputs_embeds, built from PyTorch's fp32 image features (with the training-time patch positions of
vision_positions_fix), so that only the decoder differs.
(a) torch.onnx fp32 reference decoder: from a past-0 first call, greedy tokens equal and max|dlogit| <= 1e-3 per step (3 fixed samples).
(b) each vision_encoder dtype in the folder: mean cosine vs PyTorch's fp32 connector output; the candidate is the smallest file at >= 0.99.
(c) export fidelity: teacher-forced top-1 agreement with fp32 over all samples, per fused deploy dtype, >= the published
    build of the same dtype - TOL (or, for a fine-tuned src, --baseline: the published numbers of a G2 parity.json).
(d) pixel_values [1,1,3,512,512] from square() + the processor, 64 <image> tokens, prompt lengths; the 512² PNG and the
    float32 pixel_values go to <folder>/parity/ so that bench.mjs checks transformers.js within 1e-2 on the same inputs
    (with the fp32 image features, against which bench.mjs checks the vision encoder on the device).
(e) with --stop-set, >= 95 % of greedy fp32 outputs end in 49279 before max_new_tokens."""
import argparse, json, os, sys, time
# before transformers/huggingface_hub are imported (they read HF_HOME at import): downloads stay on LaCie
os.environ.setdefault('HF_HOME', '/Volumes/LaCie/astro-pilot/vlm/hf')
from pathlib import Path
import numpy as np, onnxruntime as ort, torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers.cache_utils import DynamicCache
from vlm.train.narrator.export_decoder import BASE, DTYPES, SUFFIX, published, square, vision_positions_fix

IMG, EOU, TOL, MIN_COS = 49190, 49279, 0.03, 0.99
EZ, OUT, ONE = '/Volumes/LaCie/astro-pilot/test/out/earthzoom/', '/Volumes/LaCie/astro-pilot/test/out/', 'Describe this image in one sentence.'
SAMPLES = [{'image': EZ + 'matterhorn_5km.jpg', 'prompt': ONE}, {'image': EZ + 'manhattan_800m.jpg', 'prompt': ONE}, {'image': EZ + 'pacific_800m.jpg', 'prompt': ONE},
           {'image': OUT + 'corridor5.png', 'prompt': 'What do you see in this image?'},
           {'image': OUT + 'landing/d1_approach_cockpit.jpg', 'prompt': 'Describe this image in detail.'},
           {'image': OUT + 'landing/d1_flare_runway.jpg', 'prompt': 'Is the aircraft about to land? Answer yes or no and explain briefly.'},
           {'image': OUT + 'earthmoon/e_orbit.jpg', 'prompt': 'Is this view taken from space, from an aircraft, or near the ground? Answer in a few words.'},
           {'image': OUT + 'china_orbit.png', 'prompt': 'Caption this image.'},
           {'image': OUT + 'airbus_low.png', 'prompt': 'What is the main object in this image?'},
           {'image': OUT + '4k_dubai.png', 'prompt': 'Describe the scene in detail, including any buildings, roads and water.'},
           {'image': EZ + 'matterhorn_420km.jpg', 'prompt': 'How high above the ground was this picture taken?'},
           {'image': OUT + 'earthmoon/e_chase.jpg', 'prompt': 'Write a short caption for this picture.'},
           {'image': OUT + 'landing/d1_rollout_tower.jpg', 'prompt': 'Describe what is happening in this image.'}]

def prep(processor, s):
    text = processor.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': s['prompt']}]}], add_generation_prompt=True)
    return processor(text=text, images=[square(Image.open(s['image']))], return_tensors='pt', images_kwargs={'do_image_splitting': False})

def embeds(model, inputs, feats):
    e = model.model.text_model.embed_tokens(inputs['input_ids']).clone()
    e[inputs['input_ids'] == IMG] = feats.reshape(-1, e.shape[-1]).to(e.dtype)
    return e

def greedy_torch(model, e, n):
    cache, am, toks, logits_all = DynamicCache(), torch.ones(1, e.shape[1], dtype=torch.int64), [], []
    pos, x = torch.arange(e.shape[1])[None], e
    for _ in range(n):
        out = model.model.text_model(inputs_embeds=x, attention_mask=am, position_ids=pos, past_key_values=cache, use_cache=True)
        lg = model.lm_head(out.last_hidden_state[:, -1]); logits_all.append(lg[0].numpy()); t = int(lg.argmax()); toks.append(t)
        if t == EOU: break
        x = model.model.text_model.embed_tokens(torch.tensor([[t]])); pos = torch.tensor([[am.shape[1]]]); am = torch.ones(1, am.shape[1] + 1, dtype=torch.int64)
    return toks, logits_all

def run_onnx(sess, cfg, e, embed_fn, forced):
    """Step the decoder through the fp32 tokens `forced` (a past-0 prefill, then one token per call); returns its argmax and
    logits per step. The empty past takes the session's KV dtype; position_ids only if the graph declares it (fused builds do not)."""
    ins = {i.name: i for i in sess.get_inputs()}; outs = [o.name for o in sess.get_outputs()]
    kvt = np.float16 if ins['past_key_values.0.key'].type == 'tensor(float16)' else np.float32
    feed = {n: np.zeros((1, cfg.num_key_value_heads, 0, getattr(cfg, 'head_dim', 64)), kvt) for n in ins if n.startswith('past_key_values.')}
    x, am, toks, logits_all = e.numpy().astype(np.float32), np.ones((1, e.shape[1]), np.int64), [], []
    for step, t_ref in enumerate(forced):
        inp = {'inputs_embeds': x, 'attention_mask': am, **feed}
        if 'position_ids' in ins: inp['position_ids'] = np.arange(am.shape[1] - x.shape[1], am.shape[1], dtype=np.int64)[None]
        out = dict(zip(outs, sess.run(None, inp))); lg = out['logits'][0, -1].astype(np.float32); logits_all.append(lg); toks.append(int(lg.argmax()))
        feed = {n: out['present.' + n[len('past_key_values.'):]] for n in feed}
        x = embed_fn(t_ref); am = np.ones((1, am.shape[1] + 1), np.int64)
    return toks, logits_all

def forced_agreement(logits, tokens):
    return sum(int(np.argmax(l)) == t for l, t in zip(logits, tokens)) / max(len(tokens), 1)

def vision_candidate(variants):
    ok = sorted((v['mb'], k) for k, v in variants.items() if v['mean_cos'] >= MIN_COS)
    return ok[0][1] if ok else None

def gate_failures(R):
    f = []
    if not all(R['a']['tokens_equal']): f.append('(a) greedy tokens differ')
    if max(R['a']['max_dlogit']) > 1e-3: f.append(f"(a) max |dlogit| {max(R['a']['max_dlogit']):.2e} > 1e-3")
    if R['b']['candidate'] is None: f.append(f'(b) no vision encoder dtype reaches mean cos {MIN_COS}')
    if not R['c']: f.append('(c) no deploy decoder in the folder')
    for d, c in R['c'].items():
        if c['published'] is None: f.append(f'(c) {d}: no published build or --baseline to compare with')
        elif c['ours'] < c['published'] - TOL: f.append(f"(c) {d}: agreement {c['ours']:.3f} < published {c['published']:.3f} - {TOL}")
    if R['d']['pixel_values_shape'] != [1, 1, 3, 512, 512] or any(n != 64 for n in R['d']['n_image_tokens']): f.append('(d) shapes')
    if R['e'] and R['e']['stop_rate'] < 0.95: f.append('(e) stop rate < 95 %')
    return f

def cosine(ref, f):
    ref, f = ref.reshape(-1, ref.shape[-1]), f.reshape(-1, f.shape[-1])
    return float(np.mean(np.sum(ref * f, 1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(f, axis=1))))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model-dir', required=True); ap.add_argument('--src', required=True); ap.add_argument('--facts', help='sample rows {image, prompt}')
    ap.add_argument('--stop-set'); ap.add_argument('--tokens', type=int, default=32); ap.add_argument('--baseline', help='a G2 parity.json whose published numbers (c) compares with')
    a = ap.parse_args(); md = Path(a.model_dir); torch.set_grad_enabled(False); (md / 'parity').mkdir(exist_ok=True)
    processor = AutoProcessor.from_pretrained(md); model = vision_positions_fix(AutoModelForVision2Seq.from_pretrained(a.src, dtype=torch.float32).eval()); cfg = model.config.text_config
    emb = lambda t: model.model.text_model.embed_tokens(torch.tensor([[t]])).numpy().astype(np.float32)
    samples = [json.loads(l) for l in open(a.facts)] if a.facts else SAMPLES
    R = {'a': {'tokens_equal': [], 'max_dlogit': []}, 'b': {'variants': {}}, 'c': {}, 'd': {'n_image_tokens': [], 'prompt_len': []}, 'e': None, 'samples': [], 'fp32_text': []}
    S = []
    for i, s in enumerate(samples):
        inp = prep(processor, s); pv, pam = inp['pixel_values'], inp['pixel_attention_mask']; R['d']['pixel_values_shape'] = list(pv.shape)
        R['d']['n_image_tokens'].append(int((inp['input_ids'] == IMG).sum())); R['d']['prompt_len'].append(int(inp['input_ids'].shape[1]))
        square(Image.open(s['image'])).save(md / 'parity' / f'{i}.png'); pv.numpy().astype(np.float32).tofile(md / 'parity' / f'{i}.f32')
        R['samples'].append({'image': s['image'], 'prompt': s['prompt'], 'png': f'parity/{i}.png', 'pixels': f'parity/{i}.f32', 'feats': f'parity/{i}.feats.f32'})
        feats = model.get_image_features(pixel_values=pv, pixel_attention_mask=pam); feats.numpy().astype(np.float32).tofile(md / 'parity' / f'{i}.feats.f32')
        e = embeds(model, inp, feats); tt, lt = greedy_torch(model, e, a.tokens)
        R['fp32_text'].append(processor.tokenizer.decode(tt, skip_special_tokens=True)); S.append({'pv': pv.numpy().astype(np.float32), 'pam': pam.numpy().astype(bool), 'feats': feats.numpy(), 'e': e, 'tt': tt, 'lt': lt})
    d32 = ort.InferenceSession(str(md / 'onnx/decoder_model_merged.onnx'))
    for x in S[:3]:
        to, lo = run_onnx(d32, cfg, x['e'], emb, x['tt']); R['a']['tokens_equal'].append(to == x['tt']); R['a']['max_dlogit'].append(float(max(np.abs(p - q).max() for p, q in zip(x['lt'], lo))))
    del d32
    for v in (v for v in SUFFIX if (md / f'onnx/vision_encoder{SUFFIX[v]}.onnx').exists()):
        p = md / f'onnx/vision_encoder{SUFFIX[v]}.onnx'; venc = ort.InferenceSession(str(p)); cos, ms = [], []
        for x in S:
            t0 = time.perf_counter(); f = venc.run(None, {'pixel_values': x['pv'], 'pixel_attention_mask': x['pam']})[0]; ms.append(1e3 * (time.perf_counter() - t0)); cos.append(cosine(x['feats'], f))
        R['b']['variants'][v] = {'mean_cos': float(np.mean(cos)), 'min_cos': float(np.min(cos)), 'cos': cos, 'mb': round(p.stat().st_size / 1e6, 1), 'cpu_ms': float(np.median(ms))}
    R['b']['candidate'] = vision_candidate(R['b']['variants']); R['b']['mean_cos'] = R['b']['variants'][R['b']['candidate']]['mean_cos'] if R['b']['candidate'] else None
    # the candidate is picked on onnxruntime's CPU EP only; the published fp16 encoder is exact there and wrong on WebGPU
    R['b']['candidate_measured_on'], R['b']['needs_device_confirmation'] = 'cpu', 'node vlm/web/bench.mjs narrator --model-dir <folder> --device webgpu --vision all'
    base = json.loads(Path(a.baseline).read_text()) if a.baseline else None
    for d in (d for d in DTYPES if (md / f'onnx/decoder_model_merged{SUFFIX[d]}.onnx').exists()):
        paths = {'ours': md / f'onnx/decoder_model_merged{SUFFIX[d]}.onnx', 'published': published(f'decoder_model_merged{SUFFIX[d]}.onnx') if a.src == BASE else None}
        c = {'ours': None, 'published': base['c'][d]['published'] if base and d in base['c'] else None}
        for k, p in paths.items():
            if p is None: continue
            sess = ort.InferenceSession(str(p)); per = [run_onnx(sess, cfg, x['e'], emb, x['tt'])[1] for x in S]; del sess
            c[k + '_per_sample'] = [forced_agreement(l, x['tt']) for l, x in zip(per, S)]
            c[k] = sum(forced_agreement(l, x['tt']) * len(x['tt']) for l, x in zip(per, S)) / sum(len(x['tt']) for x in S)
        R['c'][d] = c
    if a.stop_set:
        rows = [json.loads(l) for l in open(a.stop_set)]; stops = 0
        for s in rows: inp = prep(processor, s); feats = model.get_image_features(pixel_values=inp['pixel_values'], pixel_attention_mask=inp['pixel_attention_mask']); tt, _ = greedy_torch(model, embeds(model, inp, feats), 90); stops += tt[-1] == EOU
        R['e'] = {'stop_rate': stops / len(rows), 'n': len(rows)}
    fails = gate_failures(R); R['tol'], R['fails'] = TOL, fails
    (md / 'parity.json').write_text(json.dumps(R, indent=1))
    print(json.dumps({k: R[k] for k in 'acde'} | {'b': {k: {'mean_cos': v['mean_cos'], 'mb': v['mb']} for k, v in R['b']['variants'].items()} | {'candidate': R['b']['candidate']}}, indent=1, default=str))
    print('PARITY FAIL: ' + '; '.join(fails) if fails else 'PARITY OK')
    sys.exit(1 if fails else 0)

if __name__ == '__main__':
    main()
