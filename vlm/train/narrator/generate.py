"""vlm/train/narrator/generate.py — greedy outputs (eos 49279, max 90 new tokens) for evaluation rows {key, image, prompt}
-> {key, text, ended}. Default: PyTorch fp32 with Task 8's vision_positions_fix. --onnx <dtype>: the deploy path of the
exported folder in onnxruntime (vision_encoder q8 -> image features, embed_tokens_fp16, decoder_model_merged_<dtype>), which
G6 uses to pick the decoder dtype (q4f16 vs q4) by slot recall. The processor comes from the exported folder; square() first."""
import argparse, json, os, time
# before transformers/huggingface_hub are imported (they read HF_HOME at import): downloads stay on LaCie
os.environ.setdefault('HF_HOME', '/Volumes/LaCie/astro-pilot/vlm/hf')
from pathlib import Path
import numpy as np, torch
from vlm.train.common import read_jsonl, EOU
from vlm.train.narrator.data import IMG_KW, frame_path
from vlm.train.narrator.export_decoder import SUFFIX, square

IMG = 49190

def encode_prompt(processor, prompt, image):
    text = processor.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}], add_generation_prompt=True)
    return processor(text=text, images=[square(image)], **IMG_KW)

class OnnxNarrator:
    """What narrator.js runs, in onnxruntime on CPU: image features from the vision encoder replace the 64 <image> embeddings,
    then the decoder steps greedily with its KV cache (the empty past takes the decoder's KV dtype; position_ids only if declared)."""
    def __init__(self, folder, decoder='q4f16', vision='q8'):
        import onnxruntime as ort
        o = Path(folder) / 'onnx'; S = lambda n: ort.InferenceSession(str(o / n))
        self.venc, self.emb, self.dec = S(f'vision_encoder{SUFFIX[vision]}.onnx'), S('embed_tokens_fp16.onnx'), S(f'decoder_model_merged{SUFFIX[decoder]}.onnx')
        ins = {i.name: i for i in self.dec.get_inputs()}; k = ins['past_key_values.0.key']
        self.outs, self.past, self.pos = [x.name for x in self.dec.get_outputs()], [n for n in ins if n.startswith('past_key_values.')], 'position_ids' in ins
        self.kv = (np.float16 if k.type == 'tensor(float16)' else np.float32, k.shape[1], k.shape[3])

    def embed(self, ids): return self.emb.run(None, {'input_ids': np.asarray(ids, np.int64)})[0].astype(np.float32)

    def generate(self, inp, max_new=90):
        ids = inp['input_ids'].numpy(); e = self.embed(ids)
        feats = self.venc.run(None, {'pixel_values': inp['pixel_values'].numpy().astype(np.float32), 'pixel_attention_mask': inp['pixel_attention_mask'].numpy().astype(bool)})[0]
        e[ids == IMG] = feats.reshape(-1, e.shape[-1])
        dt, nkv, hd = self.kv; feed = {n: np.zeros((1, nkv, 0, hd), dt) for n in self.past}; am, x, toks = np.ones((1, ids.shape[1]), np.int64), e, []
        for _ in range(max_new):
            f = {'inputs_embeds': x, 'attention_mask': am, **feed}
            if self.pos: f['position_ids'] = np.arange(am.shape[1] - x.shape[1], am.shape[1], dtype=np.int64)[None]
            out = dict(zip(self.outs, self.dec.run(None, f))); t = int(out['logits'][0, -1].argmax()); toks.append(t)
            if t == EOU: break
            feed = {n: out['present.' + n[len('past_key_values.'):]] for n in feed}; x = self.embed([[t]]); am = np.ones((1, am.shape[1] + 1), np.int64)
        return toks

def main(argv=None):
    from PIL import Image
    from transformers import AutoProcessor
    ap = argparse.ArgumentParser(); ap.add_argument('--model', help='PyTorch model id or dir'); ap.add_argument('--onnx', choices=('q4f16', 'q4', 'fp16', 'fp32'), help='deploy decoder dtype of --processor-dir')
    ap.add_argument('--vision', default='q8'); ap.add_argument('--processor-dir', required=True); ap.add_argument('--rows', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--max-new', type=int, default=90); ap.add_argument('--limit', type=int); ap.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu'); a = ap.parse_args(argv)
    if bool(a.model) == bool(a.onnx): ap.error('give exactly one of --model and --onnx')
    p = AutoProcessor.from_pretrained(a.processor_dir)
    if a.onnx: run = OnnxNarrator(a.processor_dir, a.onnx, a.vision).generate
    else:
        from transformers import AutoModelForVision2Seq
        from vlm.train.narrator.export_decoder import vision_positions_fix
        m = vision_positions_fix(AutoModelForVision2Seq.from_pretrained(a.model, dtype=torch.float32)).eval().to(a.device)
        def run(inp, max_new):
            inp = inp.to(a.device)
            with torch.no_grad(): return m.generate(**inp, max_new_tokens=max_new, do_sample=False, eos_token_id=EOU)[0, inp['input_ids'].shape[1]:].tolist()
    rows, t0 = read_jsonl(a.rows)[:a.limit], time.time()
    with open(a.out, 'w') as f:
        for i, r in enumerate(rows):
            new = run(encode_prompt(p, r['prompt'], Image.open(frame_path(r['image']))), a.max_new)
            f.write(json.dumps({'key': r['key'], 'text': p.tokenizer.decode(new, skip_special_tokens=True).strip(), 'ended': bool(new) and new[-1] == EOU, 'runtime': f'onnx-{a.onnx}' if a.onnx else 'torch'}) + '\n'); f.flush()
            if (i + 1) % 10 == 0: print(f'{i + 1}/{len(rows)} {time.time() - t0:.0f} s', flush=True)
    print(f'wrote {len(rows)} outputs -> {a.out}', flush=True)

if __name__ == '__main__': main()
