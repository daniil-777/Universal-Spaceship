"""Parity of an exported Narrator folder against PyTorch fp32 on CPU (spec §10.2): (a) decoder-only from identical
inputs_embeds, 32 greedy tokens equal and max|dlogit| <= 1e-3 per step from a past-0 first call; (b) image features from
the published vision encoder, cosine vs PyTorch's connector recorded; (c) q4 token agreement >= 80 % (and, with --facts,
the verifier verdict equal to fp32's); (d) pixel_values [1,1,3,512,512], 64 <image> tokens, prompt lengths; (e) with
--stop-set, >= 95 % of greedy outputs end in 49279 before max_new_tokens."""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np, onnxruntime as ort, torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers.cache_utils import DynamicCache

IMG, EOU = 49190, 49279
SAMPLES = [{'image': '/Volumes/LaCie/astro-pilot/test/out/earthzoom/matterhorn_5km.jpg', 'prompt': 'Describe this image in one sentence.'},
           {'image': '/Volumes/LaCie/astro-pilot/test/out/earthzoom/manhattan_800m.jpg', 'prompt': 'Describe this image in one sentence.'},
           {'image': '/Volumes/LaCie/astro-pilot/test/out/earthzoom/pacific_800m.jpg', 'prompt': 'Describe this image in one sentence.'}]

def prep(processor, s):
    text = processor.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': s['prompt']}]}], add_generation_prompt=True)
    return processor(text=text, images=[Image.open(s['image']).convert('RGB')], return_tensors='pt', images_kwargs={'do_image_splitting': False})

def embeds(model, inputs, feats):
    e = model.model.text_model.embed_tokens(inputs['input_ids']).clone()
    e[inputs['input_ids'] == IMG] = torch.from_numpy(feats).reshape(-1, e.shape[-1]).to(e.dtype)
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

def greedy_onnx(sess, cfg, e, n, embed_fn, forced=None):
    L, kv, hd = cfg.num_hidden_layers, cfg.num_key_value_heads, getattr(cfg, 'head_dim', 64)
    feed = {f'past_key_values.{i}.{k}': np.zeros((1, kv, 0, hd), np.float32) for i in range(L) for k in ('key', 'value')}
    x, am, pos, toks, logits_all = e.numpy().astype(np.float32), np.ones((1, e.shape[1]), np.int64), np.arange(e.shape[1], dtype=np.int64)[None], [], []
    for step in range(n):
        out = sess.run(None, {'inputs_embeds': x, 'attention_mask': am, 'position_ids': pos, **feed})
        lg = out[0][0, -1]; logits_all.append(lg); t = int(lg.argmax()); toks.append(t)
        for i in range(L): feed[f'past_key_values.{i}.key'], feed[f'past_key_values.{i}.value'] = out[1 + 2 * i], out[2 + 2 * i]
        if t == EOU: break
        nxt = forced[step] if forced is not None and step < len(forced) else t
        x = embed_fn(nxt); pos = np.array([[am.shape[1]]], np.int64); am = np.ones((1, am.shape[1] + 1), np.int64)
    return toks, logits_all

def verifier(text, record):
    r = subprocess.run(['node', 'vlm/gen/text/verify_cli.mjs'], input=json.dumps({'text': text, 'record': record, 'names': record.get('names', [])}), capture_output=True, text=True, check=True)
    return json.loads(r.stdout)['verified']

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model-dir', required=True); ap.add_argument('--src', required=True); ap.add_argument('--facts'); ap.add_argument('--stop-set'); ap.add_argument('--tokens', type=int, default=32)
    a = ap.parse_args(); md = Path(a.model_dir); torch.set_grad_enabled(False)
    processor = AutoProcessor.from_pretrained(md); model = AutoModelForVision2Seq.from_pretrained(a.src, torch_dtype=torch.float32).eval(); cfg = model.config.text_config
    venc = ort.InferenceSession(str(md / 'onnx/vision_encoder_q4.onnx')); d32 = ort.InferenceSession(str(md / 'onnx/decoder_model_merged.onnx')); d4 = ort.InferenceSession(str(md / 'onnx/decoder_model_merged_q4.onnx'))
    pub_path = Path('/Volumes/LaCie/astro-pilot/test/out/vlm_explore/hfcache/HuggingFaceTB/SmolVLM-256M-Instruct/onnx/decoder_model_merged_q4.onnx')
    pub = ort.InferenceSession(str(pub_path)) if a.src.endswith('SmolVLM-256M-Instruct') else None
    emb = lambda t: model.model.text_model.embed_tokens(torch.tensor([[t]])).numpy().astype(np.float32)
    samples = [json.loads(l) for l in open(a.facts)] if a.facts else SAMPLES
    R = {'a': {'tokens_equal': [], 'max_dlogit': []}, 'b': {'cos': []}, 'c': {'agree_fp32': [], 'agree_published': [] if pub else None, 'verdict_equal': [] if a.facts else None}, 'd': {'n_image_tokens': [], 'prompt_len': []}, 'e': None, 'samples': [{'image': s['image'], 'prompt': s['prompt']} for s in samples]}
    for s in samples:
        inp = prep(processor, s); pv = inp['pixel_values']; R['d']['pixel_values_shape'] = list(pv.shape)
        R['d']['n_image_tokens'].append(int((inp['input_ids'] == IMG).sum())); R['d']['prompt_len'].append(int(inp['input_ids'].shape[1]))
        feats = venc.run(None, {'pixel_values': pv.numpy().astype(np.float32), 'pixel_attention_mask': inp['pixel_attention_mask'].numpy().astype(bool)})[0]
        ref = model.get_image_features(pixel_values=pv, pixel_attention_mask=inp['pixel_attention_mask']).reshape(-1, feats.shape[-1]).numpy(); f2 = feats.reshape(-1, feats.shape[-1])
        R['b']['cos'].append(float(np.mean(np.sum(ref * f2, 1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(f2, axis=1)))))
        e = embeds(model, inp, feats); tt, lt = greedy_torch(model, e, a.tokens); to, lo = greedy_onnx(d32, cfg, e, a.tokens, emb, forced=tt)
        R['a']['tokens_equal'].append(tt == to); R['a']['max_dlogit'].append(float(max(np.abs(x - y).max() for x, y in zip(lt, lo))))
        t4, _ = greedy_onnx(d4, cfg, e, a.tokens, emb); R['c']['agree_fp32'].append(sum(x == y for x, y in zip(tt, t4)) / max(len(tt), 1))
        if pub: tp, _ = greedy_onnx(pub, cfg, e, a.tokens, emb); R['c']['agree_published'].append(sum(x == y for x, y in zip(t4, tp)) / max(len(t4), 1))
        if a.facts: dec = lambda t: processor.tokenizer.decode(t, skip_special_tokens=True); R['c']['verdict_equal'].append(verifier(dec(tt), s['record']) == verifier(dec(t4), s['record']))
    if a.stop_set:
        rows = [json.loads(l) for l in open(a.stop_set)]; stops = 0
        for s in rows: inp = prep(processor, s); feats = venc.run(None, {'pixel_values': inp['pixel_values'].numpy().astype(np.float32), 'pixel_attention_mask': inp['pixel_attention_mask'].numpy().astype(bool)})[0]; tt, _ = greedy_torch(model, embeds(model, inp, feats), 90); stops += tt[-1] == EOU
        R['e'] = {'stop_rate': stops / len(rows), 'n': len(rows)}
    R['b']['mean_cos'] = float(np.mean(R['b']['cos']))
    fails = []
    if not all(R['a']['tokens_equal']): fails.append('(a) greedy tokens differ')
    if max(R['a']['max_dlogit']) > 1e-3: fails.append(f"(a) max |dlogit| {max(R['a']['max_dlogit']):.2e} > 1e-3")
    if min(R['c']['agree_fp32']) < 0.8: fails.append('(c) q4 token agreement < 80 %')
    if pub and min(R['c']['agree_published']) < 0.8: fails.append('(c) q4 vs published q4 agreement < 80 %')
    if R['c']['verdict_equal'] is not None and not all(R['c']['verdict_equal']): fails.append('(c) verifier verdict differs')
    if R['d']['pixel_values_shape'] != [1, 1, 3, 512, 512] or any(n != 64 for n in R['d']['n_image_tokens']): fails.append('(d) shapes')
    if R['e'] and R['e']['stop_rate'] < 0.95: fails.append('(e) stop rate < 95 %')
    (md / 'parity.json').write_text(json.dumps(R, indent=1)); print(json.dumps({k: R[k] for k in 'abcde'}, indent=1)); print('PARITY FAIL: ' + '; '.join(fails) if fails else 'PARITY OK')
    sys.exit(1 if fails else 0)

if __name__ == '__main__':
    main()
