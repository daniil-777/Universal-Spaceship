"""chat/export_web.py — export a (fine-tuned) causal LM to the folder transformers.js 4.3 loads (spec R7): the ONNX Runtime GenAI model
builder's fused int4 graphs (GroupQueryAttention, SkipSimplifiedLayerNorm, MatMulNBits block 32, 4-bit tied embeddings via
GatherBlockQuantized, LFM2's CausalConvWithState) and fix_io() for the transformers.js contract:
- LFM2 conv states past.N.conv / present.N.conv -> past_conv.N / present_conv.N (transformers.js getCacheNames);
- the symbolic kv_cache_dim -> head_dim (transformers.js sizes the empty cache from the input metadata; unknown symbols become 0);
- a scalar int64 input num_logits_to_keep that slices the hidden states before the LM head, so a prefill computes one row of logits
  instead of prompt_len x vocab (transformers.js feeds it whenever the graph declares it);
- one file, no external data.
max_position_embeddings is cut to --ctx first, so the rotary cos/sin caches hold ctx rows, not 128k.
dtypes: q4f16 = int4 weights, fp16 activations/IO, built for WebGPU; q4 = int4 weights, fp32 IO, built for CPU/WASM (also runs on WebGPU).
  python -m chat.export_web --src <hf dir> --out <folder> [--dtypes q4f16,q4] [--ctx 4096] [--check]"""
import argparse, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
import onnx
from onnx import TensorProto, helper

SUFFIX = {'q4f16': '_q4f16', 'q4': '_q4', 'fp16': '_fp16'}
BUILD = {'q4f16': ('int4', 'webgpu', ['int4_block_size=32', 'int4_accuracy_level=4']),
         'q4': ('int4', 'cpu', ['int4_block_size=32', 'int4_accuracy_level=4']),
         'fp16': ('fp16', 'webgpu', [])}
TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.json', 'merges.txt')

def rename(g, old, new):
    for n in g.node:
        n.input[:] = [new if i == old else i for i in n.input]; n.output[:] = [new if o == old else o for o in n.output]
    for v in [*g.input, *g.output]:
        if v.name == old: v.name = new

def add_num_logits_to_keep(g):
    """Insert Slice(hidden, -k, INT64_MAX, axis 1) in front of the LM head (the MatMul/MatMulNBits that produces the logits)."""
    prod = {o: n for n in g.node for o in n.output}
    head = prod['logits']
    while head.op_type in ('Cast', 'Identity'): head = prod[head.input[0]]
    if head.op_type not in ('MatMul', 'MatMulNBits'): raise ValueError(f'unexpected LM head op {head.op_type}')
    src = head.input[0]
    g.input.append(helper.make_tensor_value_info('num_logits_to_keep', TensorProto.INT64, []))
    g.initializer.extend([helper.make_tensor('nltk/axes', TensorProto.INT64, [1], [1]),
                          helper.make_tensor('nltk/end', TensorProto.INT64, [1], [2 ** 62])])
    nodes = [helper.make_node('Neg', ['num_logits_to_keep'], ['nltk/neg']),
             helper.make_node('Unsqueeze', ['nltk/neg', 'nltk/axes'], ['nltk/start']),
             helper.make_node('Slice', [src, 'nltk/start', 'nltk/end', 'nltk/axes'], ['nltk/hidden'])]
    head.input[0] = 'nltk/hidden'
    i = list(g.node).index(head)
    for k, n in enumerate(nodes): g.node.insert(i + k, n)
    for v in g.output:
        if v.name == 'logits': v.type.tensor_type.shape.dim[1].dim_param = 'num_logits_to_keep'

def fix_io(src, dst, head_dim):
    m = onnx.load(str(src)); g = m.graph
    for v in list(g.input) + list(g.output):
        p = v.name.split('.')
        if len(p) == 3 and p[0] in ('past', 'present') and p[2] == 'conv': rename(g, v.name, f'{p[0]}_conv.{p[1]}')
    for v in [*g.input, *g.output]:
        if v.name.startswith(('past_key_values.', 'present.')):
            d = v.type.tensor_type.shape.dim[3]; d.Clear(); d.dim_value = head_dim
    add_num_logits_to_keep(g)
    onnx.save(m, str(dst), save_as_external_data=False)

def prepare_src(src, work, ctx):
    """A copy of the HF folder whose config caps max_position_embeddings at ctx (smaller rotary caches)."""
    d = Path(work) / 'src'
    shutil.copytree(src, d, ignore=shutil.ignore_patterns('._*', 'optimizer.pt', 'rng_state*', 'scheduler.pt', 'trainer_state.json', 'training_args.bin'))
    cfg = json.loads((d / 'config.json').read_text())
    for k in ('max_position_embeddings', 'seq_length'):
        if k in cfg: cfg[k] = min(int(cfg[k]), ctx)
    (d / 'config.json').write_text(json.dumps(cfg, indent=1))
    return d, cfg

def build(src_dir, cfg, dtype, work):
    precision, ep, extra = BUILD[dtype]; out = Path(work) / dtype
    shared = ['shared_embeddings=true'] if cfg.get('tie_word_embeddings', cfg.get('tie_embedding', False)) else []
    cmd = [sys.executable, '-m', 'onnxruntime_genai.models.builder', '-i', str(src_dir), '-o', str(out), '-p', precision, '-e', ep,
           '-c', str(Path(work) / 'cache'), '--extra_options', *extra, *shared]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return out / 'model.onnx'

def export(src, out, dtypes=('q4f16', 'q4'), ctx=4096):
    out = Path(out); (out / 'onnx').mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='.export_', dir=out.parent))
    try:
        s, cfg = prepare_src(src, work, ctx)
        hd = cfg.get('head_dim') or cfg['hidden_size'] // cfg['num_attention_heads']
        for d in dtypes: fix_io(build(s, cfg, d, work), out / 'onnx' / f'model{SUFFIX[d]}.onnx', hd)
        for f in TOKENIZER_FILES + ('config.json', 'generation_config.json', 'chat_template.jinja'):
            if (s / f).exists(): shutil.copy(s / f, out / f)
        tk = out / 'tokenizer_config.json'
        if (s / 'chat_template.jinja').exists() and tk.exists():
            t = json.loads(tk.read_text()); t['chat_template'] = (s / 'chat_template.jinja').read_text(); tk.write_text(json.dumps(t, indent=1))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {d: round((out / 'onnx' / f'model{SUFFIX[d]}.onnx').stat().st_size / 2 ** 20, 1) for d in dtypes}

def check(src, out, prompts=('Hi! What is Astro Pilot?', 'How does the pilot learn to dodge asteroids?'), n=24):
    """Greedy parity on CPU: PyTorch fp32 vs the q4 graph (ORT CPU) — the share of identical next tokens along the torch path."""
    import numpy as np, onnxruntime as ort, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(src); model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32).eval()
    sess = ort.InferenceSession(str(Path(out) / 'onnx' / 'model_q4.onnx'), providers=['CPUExecutionProvider'])
    meta = {i.name: i for i in sess.get_inputs()}; agree = total = 0
    for p in prompts:
        ids = tok.apply_chat_template([{'role': 'user', 'content': p}], add_generation_prompt=True, return_tensors='pt')
        ids = ids['input_ids'] if hasattr(ids, 'keys') else ids
        with torch.no_grad(): ref = model.generate(ids, max_new_tokens=n, do_sample=False)[0, ids.shape[1]:].tolist()
        seq = ids[0].tolist()
        for t in ref:
            feed = {'input_ids': np.array([seq], np.int64), 'attention_mask': np.ones((1, len(seq)), np.int64), 'num_logits_to_keep': np.array(1, np.int64)}
            for k, v in meta.items():
                if k in feed: continue
                shape = [1 if d == 'batch_size' else 0 if isinstance(d, str) else d for d in v.shape]
                feed[k] = np.zeros(shape, np.float16 if 'float16' in v.type else np.float32)
            logits = sess.run(['logits'], feed)[0]
            agree += int(logits[0, -1].argmax() == t); total += 1; seq.append(t)
    return {'greedy_agreement': agree / total, 'tokens': total}

if __name__ == '__main__':
    os.environ.setdefault('HF_HOME', '/Volumes/LaCie/astro-pilot/chat/hf')
    ap = argparse.ArgumentParser(); ap.add_argument('--src', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--dtypes', default='q4f16,q4'); ap.add_argument('--ctx', type=int, default=4096); ap.add_argument('--check', action='store_true')
    a = ap.parse_args(); sizes = export(a.src, a.out, a.dtypes.split(','), a.ctx); print(json.dumps({'sizes_mb': sizes}))
    if a.check: print(json.dumps(check(a.src, a.out)))
