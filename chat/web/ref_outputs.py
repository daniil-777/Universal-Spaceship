"""chat/web/ref_outputs.py — the Python side of bench.mjs's greedy parity: for each prompt the page rendered (its chat-template text,
its token ids and the prompt-contract messages) tokenize the SAME text with the HF tokenizer (add_special_tokens=False, as the page),
re-render the messages with the Python chat template, and greedy-decode with onnxruntime CPU on the exported graph exactly as
transformers.js does: KV/conv cache loop, num_logits_to_keep 1, repetition penalty over the unique ids of prompt + output, stop at eos.
--torch-src <hf dir> adds the gold reference: PyTorch fp32 greedy (HF generate, same penalty) and teacher-forced agreement — for each
candidate sequence (the page's outputs per config, and the ORT CPU greedy), the share of its first n tokens that fp32 argmax picks
given the same prefix (one early near-tie flip does not zero the rest, unlike a shared-prefix count).
  python chat/web/ref_outputs.py --model-dir <web folder> --prompts <in.json> --out <out.json> [--dtype q4] [--n 32] [--penalty 1.1] [--torch-src <hf dir>]
in: [{text, ids, messages, cands?: {name: [ids]}}] · out: {dtype, eos, rows: [{py_ids, template, gen, ms, torch_gen?, tf?: {name: [agree, n]}}]}"""
import argparse, json, time
from pathlib import Path
import numpy as np, onnxruntime as ort
from transformers import AutoTokenizer

NP = {'tensor(float)': np.float32, 'tensor(float16)': np.float16, 'tensor(int64)': np.int64}

def empty_cache(sess):
    feed = {}
    for i in sess.get_inputs():
        if i.name.startswith(('past_key_values.', 'past_conv.')):
            feed[i.name] = np.zeros([1 if d == 'batch_size' else 0 if isinstance(d, str) else d for d in i.shape], NP[i.type])
    return feed

def greedy(sess, ids, n, eos, penalty):
    names = [o.name for o in sess.get_outputs()]
    past, seq, out = empty_cache(sess), list(ids), []
    step = np.array([ids], np.int64)
    for _ in range(n):
        feed = {'input_ids': step, 'attention_mask': np.ones((1, len(seq)), np.int64), 'num_logits_to_keep': np.array(1, np.int64), **past}
        res = dict(zip(names, sess.run(names, feed)))
        logits = res['logits'][0, -1].astype(np.float32)
        t = int(penalize(logits, set(seq), penalty).argmax()); out.append(t); seq.append(t)
        if t in eos: break
        past = {k.replace('present_conv', 'past_conv').replace('present.', 'past_key_values.'): v for k, v in res.items() if k.startswith('present')}
        step = np.array([[t]], np.int64)
    return out

def penalize(logits, seen, penalty):
    for t in seen: logits[t] = logits[t] * penalty if logits[t] < 0 else logits[t] / penalty
    return logits

class Gold:
    """PyTorch fp32: greedy with repetition penalty, and teacher-forced agreement of candidate continuations."""
    def __init__(self, src, eos, penalty):
        import torch
        from transformers import AutoModelForCausalLM
        self.torch, self.eos, self.penalty = torch, sorted(eos), penalty
        self.model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32).eval()
    def greedy(self, ids, n):
        t = self.torch.tensor([ids])
        with self.torch.no_grad():
            out = self.model.generate(t, attention_mask=self.torch.ones_like(t), max_new_tokens=n, do_sample=False, repetition_penalty=self.penalty,
                                      eos_token_id=self.eos, pad_token_id=0)
        return out[0, len(ids):].tolist()
    def agree(self, ids, cand, n):
        cand = cand[:n]
        if not cand: return [0, 0]
        full = ids + cand
        with self.torch.no_grad(): logits = self.model(self.torch.tensor([full])).logits[0].float().numpy()
        ok = sum(int(penalize(logits[len(ids) + j - 1].copy(), set(full[:len(ids) + j]), self.penalty).argmax() == c) for j, c in enumerate(cand))
        return [ok, len(cand)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True); ap.add_argument('--prompts', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--dtype', default='q4'); ap.add_argument('--n', type=int, default=32); ap.add_argument('--penalty', type=float, default=1.1)
    ap.add_argument('--torch-src', default=None)
    a = ap.parse_args(); d = Path(a.model_dir)
    tok = AutoTokenizer.from_pretrained(str(d))
    g = json.loads((d / 'generation_config.json').read_text()) if (d / 'generation_config.json').exists() else {}
    eos = g.get('eos_token_id', tok.eos_token_id); eos = set(eos if isinstance(eos, list) else [eos])
    suffix = {'q4': '_q4', 'q4f16': '_q4f16', 'fp32': '', 'fp16': '_fp16'}[a.dtype]
    so = ort.SessionOptions(); so.log_severity_level = 3
    sess = ort.InferenceSession(str(d / 'onnx' / f'model{suffix}.onnx'), so, providers=['CPUExecutionProvider'])
    gold = Gold(a.torch_src, eos, a.penalty) if a.torch_src else None
    rows = []
    for p in json.loads(Path(a.prompts).read_text()):
        py_ids = tok(p['text'], add_special_tokens=False)['input_ids']
        template = tok.apply_chat_template(p['messages'], add_generation_prompt=True, tokenize=False) if p.get('messages') else None
        t0 = time.perf_counter(); gen = greedy(sess, p['ids'], a.n, eos, a.penalty)
        row = {'py_ids': py_ids, 'template': template, 'gen': gen, 'ms': round(1000 * (time.perf_counter() - t0))}
        if gold:
            row['torch_gen'] = gold.greedy(p['ids'], a.n)
            row['tf'] = {k: gold.agree(p['ids'], v, a.n) for k, v in {**p.get('cands', {}), f'ort_cpu_{a.dtype}': gen}.items()}
        rows.append(row)
    Path(a.out).write_text(json.dumps({'dtype': a.dtype, 'eos': sorted(eos), 'n': a.n, 'penalty': a.penalty, 'rows': rows}))

if __name__ == '__main__':
    main()
