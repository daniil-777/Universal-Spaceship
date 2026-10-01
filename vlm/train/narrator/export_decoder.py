"""Export the SmolVLM text decoder (Llama: 30 layers, hidden 576, 9 heads, 3 KV heads, head_dim 64) for transformers.js 4.3
(spec §10.2, io_names.py; G2 round-2 rulings) and lay out the folder it loads.
- decoder_model_merged.onnx: the parity reference, traced with torch.onnx from a Decoder written out from the checkpoint's
  own weights (RMSNorm, RoPE, GQA with the past concatenated, SiLU MLP), so tracing never meets transformers' cache or mask helpers.
- decoder_model_merged_{q4,q4f16,fp16}.onnx: the deploy decoders, built by the ONNX Runtime GenAI model builder (fused
  GroupQueryAttention with rotary, SkipSimplifiedLayerNorm, MatMulNBits) from the text model saved as a LlamaForCausalLM,
  then fix_io() gives them the published transformers.js contract.
- The published embed_tokens_fp16 and vision_encoder q4/int8/fp16 are copied unchanged. vision_encoder_quantized (dtype q8)
  is ours: weight-only 8-bit MatMulNBits (block 32) of the published fp32 encoder, whose weights equal the checkpoint:
  cos >= 0.999 on WebGPU and CPU, where the published fp16 encoder breaks on WebGPU (cos 0.14-0.69) and its q8/int8/q4
  reach 0.92-0.95. Block 32 because the WebGPU MatMulNBits kernel runs it 2.7x faster than block 128 (279 vs 752 ms)."""
import argparse, copy, json, os, shutil, subprocess, sys, tempfile
# before transformers/huggingface_hub are imported (they read HF_HOME at import): downloads stay on LaCie
os.environ.setdefault('HF_HOME', '/Volumes/LaCie/astro-pilot/vlm/hf')
from pathlib import Path
import onnx, torch
from PIL import Image
from torch import nn

SMOLVLM_DIR = Path(os.environ.get('SMOLVLM_DIR', '/Volumes/LaCie/astro-pilot/test/out/vlm_explore/hfcache/HuggingFaceTB/SmolVLM-256M-Instruct'))
BASE, EOU, EDGE = 'HuggingFaceTB/SmolVLM-256M-Instruct', 49279, 512
# transformers.js dtype -> file suffix; builder precision and extra options per deploy decoder dtype.
SUFFIX = {'fp32': '', 'fp16': '_fp16', 'q8': '_quantized', 'int8': '_int8', 'q4': '_q4', 'q4f16': '_q4f16'}
FUSED = {'q4': ('int4', ['use_webgpu_fp32=true', 'accuracy_level=1']), 'q4f16': ('int4', ['accuracy_level=1']), 'fp16': ('fp16', [])}
DTYPES, VISION, W8_BLOCK = tuple(FUSED), ('q4', 'int8', 'fp16'), 32

class Decoder(nn.Module):
    def __init__(self, lm, lm_head, cfg):
        super().__init__()
        self.layers, self.norm, self.lm_head = lm.layers, lm.norm, lm_head
        self.nh, self.nkv = cfg.num_attention_heads, cfg.num_key_value_heads
        self.hd = getattr(cfg, 'head_dim', None) or cfg.hidden_size // cfg.num_attention_heads
        self.register_buffer('inv_freq', 1.0 / (cfg.rope_theta ** (torch.arange(0, self.hd, 2, dtype=torch.float32) / self.hd)), persistent=False)

    def rope(self, x, cos, sin):
        h = self.hd // 2
        return x * cos + torch.cat((-x[..., h:], x[..., :h]), dim=-1) * sin

    def forward(self, inputs_embeds, attention_mask, position_ids, *past):
        b, s, _ = inputs_embeds.shape
        freqs = position_ids[:, :, None].float() * self.inv_freq[None, None, :]
        emb = torch.cat((freqs, freqs), dim=-1); cos, sin = emb.cos()[:, None], emb.sin()[:, None]
        past_len = past[0].shape[2]
        q_pos = torch.arange(s, device=inputs_embeds.device)[:, None] + past_len
        k_pos = torch.arange(past_len + s, device=inputs_embeds.device)[None, :]
        allowed = (k_pos <= q_pos)[None, None] & (attention_mask[:, None, None, :] > 0)
        bias = torch.where(allowed, torch.zeros((), dtype=inputs_embeds.dtype), torch.full((), torch.finfo(inputs_embeds.dtype).min, dtype=inputs_embeds.dtype))
        h, presents, rep = inputs_embeds, [], self.nh // self.nkv
        for i, layer in enumerate(self.layers):
            a, x = layer.self_attn, layer.input_layernorm(h)
            q = self.rope(a.q_proj(x).view(b, s, self.nh, self.hd).transpose(1, 2), cos, sin)
            k = self.rope(a.k_proj(x).view(b, s, self.nkv, self.hd).transpose(1, 2), cos, sin)
            v = a.v_proj(x).view(b, s, self.nkv, self.hd).transpose(1, 2)
            k = torch.cat((past[2 * i], k), dim=2); v = torch.cat((past[2 * i + 1], v), dim=2); presents += [k, v]
            att = torch.softmax(q @ k.repeat_interleave(rep, dim=1).transpose(-1, -2) / self.hd ** 0.5 + bias, dim=-1)
            h = h + a.o_proj((att @ v.repeat_interleave(rep, dim=1)).transpose(1, 2).reshape(b, s, self.nh * self.hd))
            x, m = layer.post_attention_layernorm(h), layer.mlp
            h = h + m.down_proj(nn.functional.silu(m.gate_proj(x)) * m.up_proj(x))
        return (self.lm_head(self.norm(h)), *presents)

def io_names(n_layers):
    past = [f'past_key_values.{i}.{kv}' for i in range(n_layers) for kv in ('key', 'value')]
    present = [f'present.{i}.{kv}' for i in range(n_layers) for kv in ('key', 'value')]
    return past, present

def export_onnx(model, out_path):
    cfg = model.config.text_config
    dec = Decoder(model.model.text_model, model.lm_head, cfg).eval()
    L, P, S, hd = cfg.num_hidden_layers, 2, 3, getattr(cfg, 'head_dim', 64)
    past_n, present_n = io_names(L)
    args = (torch.randn(1, S, cfg.hidden_size), torch.ones(1, P + S, dtype=torch.int64), torch.arange(P, P + S)[None], *[torch.zeros(1, cfg.num_key_value_heads, P, hd) for _ in range(2 * L)])
    dyn = {'inputs_embeds': {0: 'batch_size', 1: 'sequence_length'}, 'attention_mask': {0: 'batch_size', 1: 'total_sequence_length'}, 'position_ids': {0: 'batch_size', 1: 'sequence_length'}, 'logits': {0: 'batch_size', 1: 'sequence_length'}}
    dyn.update({n: {0: 'batch_size', 2: 'past_sequence_length'} for n in past_n}); dyn.update({n: {0: 'batch_size', 2: 'total_sequence_length'} for n in present_n})
    with torch.no_grad():
        torch.onnx.export(dec, args, str(out_path), input_names=['inputs_embeds', 'attention_mask', 'position_ids', *past_n], output_names=['logits', *present_n], dynamic_axes=dyn, opset_version=17, dynamo=False)
    onnx.checker.check_model(str(out_path))

def save_text_llama(model, out, tok_src=SMOLVLM_DIR):
    """The base text decoder as a plain HF LlamaForCausalLM (its own lm_head: SmolVLM-256M does not tie it) for the GenAI builder."""
    from transformers import LlamaForCausalLM
    cfg = copy.deepcopy(model.config.text_config); cfg.architectures, cfg.tie_word_embeddings = ['LlamaForCausalLM'], False
    lm = LlamaForCausalLM(cfg).eval(); lm.model.load_state_dict(model.model.text_model.state_dict()); lm.lm_head.load_state_dict(model.lm_head.state_dict())
    lm.save_pretrained(str(out))
    for f in ('tokenizer.json', 'tokenizer_config.json', 'generation_config.json'): shutil.copy(Path(tok_src) / f, Path(out) / f)

def fix_io(src, dst, head_dim):
    """Give a GenAI-builder decoder the published transformers.js contract, as one file without external data:
    past/present get a fixed head_dim (transformers.js 4.3 sizes the empty past from the input metadata and resolves an
    unknown symbol such as kv_cache_dim to 0), and an fp16 graph gets float32 inputs_embeds and logits around it, as the
    published fp16/q4f16 files do (embed_tokens_fp16 emits float32). Names already match: inputs_embeds, attention_mask,
    past_key_values.N.key/value, present.N.key/value, logits; position_ids is absent because the fused rotary takes the
    positions from the mask, and transformers.js feeds position_ids only to graphs that declare it."""
    m = onnx.load(str(src)); g = m.graph; F, H = onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16
    for v in [*g.input, *g.output]:
        if v.name.startswith(('past_key_values.', 'present.')): v.type.tensor_type.shape.dim[3].dim_value = head_dim
    emb, logits = next(v for v in g.input if v.name == 'inputs_embeds'), next(v for v in g.output if v.name == 'logits')
    if emb.type.tensor_type.elem_type == H:
        for n in g.node: n.input[:] = ['inputs_embeds_fp16' if i == 'inputs_embeds' else i for i in n.input]
        g.node.insert(0, onnx.helper.make_node('Cast', ['inputs_embeds'], ['inputs_embeds_fp16'], to=H)); emb.type.tensor_type.elem_type = F
    if logits.type.tensor_type.elem_type == H:
        for n in g.node: n.output[:] = ['logits_fp16' if o == 'logits' else o for o in n.output]
        g.node.append(onnx.helper.make_node('Cast', ['logits_fp16'], ['logits'], to=F)); logits.type.tensor_type.elem_type = F
    # no onnx.checker: the builder's SimplifiedLayerNormalization is an onnxruntime op in the default domain; ORT loads it in check_parity
    onnx.save(m, str(dst), save_as_external_data=False)

def build_fused(llama_dir, dtype, dst, work, head_dim=64):
    precision, extra = FUSED[dtype]; out = Path(work) / dtype
    subprocess.run([sys.executable, '-m', 'onnxruntime_genai.models.builder', '-i', str(llama_dir), '-o', str(out), '-p', precision, '-e', 'webgpu',
                    '-c', str(Path(work) / 'cache'), '--extra_options', 'exclude_embeds=true', *extra], check=True, stdout=subprocess.DEVNULL)
    fix_io(out / 'model.onnx', dst, head_dim)

def quantize_vision_w8(src, dst, block_size=W8_BLOCK):
    """Weight-only 8-bit MatMulNBits (float activations, symmetric, one file). Unlike the published q8/int8 (MatMulInteger,
    activations quantized too: cos 0.92-0.95), this keeps the SigLIP features at cos >= 0.999 on WebGPU and CPU."""
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    q = MatMulNBitsQuantizer(onnx.load(str(src)), bits=8, block_size=block_size, is_symmetric=True); q.process()
    q.model.save_model_to_file(str(dst), use_external_data_format=False)

def preprocessor_config(pre):
    """do_resize off, so callers hand the processor square(image) and both runtimes only rescale and normalize. With the
    shipped do_resize, transformers.js 4.3 resizes to longest_edge 2048 and then squashes to 512² with a bilinear
    Resize without antialiasing, while Python uses PIL LANCZOS twice (max |Δpixel_values| 0.50–0.55 on textured frames)."""
    return {**pre, 'do_resize': False, 'do_image_splitting': False, 'max_image_size': {'longest_edge': EDGE}}

def square(img, edge=EDGE):
    """The one image resize before the processor (training, parity and the Node bench): PIL LANCZOS to edge²."""
    return img.convert('RGB').resize((edge, edge), Image.LANCZOS)

def vision_positions_fix(model):
    """Restore SmolVLM's training-time patch position ids in PyTorch. transformers 4.57's Idefics3VisionEmbeddings buckets
    i/n·(1-1e-6), which puts patch i in bucket i-1 (per axis 0,0,1,…,30: 961 distinct ids of 1024); the checkpoint was
    trained with, and the published vision_encoder ONNX computes, bucketize(arange(0, 1-1e-6, 1/n)) (ids 0…31). Without
    this, PyTorch's image features differ from the deployed encoder's (cos 0.86–0.92); with it, fp32/fp16 ONNX match at
    cos 1.0. Training (Task 15), merge checks and parity must all call it."""
    from transformers.models.idefics3.modeling_idefics3 import Idefics3VisionEmbeddings
    E = model.model.vision_model.embeddings
    if not isinstance(E, Idefics3VisionEmbeddings):
        raise TypeError(f'vision_positions_fix patches Idefics3VisionEmbeddings (SmolVLM, transformers 4.57); model.model.vision_model.embeddings is {type(E).__name__}')

    def forward(pixel_values, patch_attention_mask):
        b, _, h, w = pixel_values.shape; n, dev = E.num_patches_per_side, pixel_values.device
        emb = E.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        bounds, ids = torch.arange(1 / n, 1.0, 1 / n, device=dev), torch.zeros(b, (h // E.patch_size) * (w // E.patch_size), dtype=torch.long, device=dev)
        for i, m in enumerate(patch_attention_mask):
            bh = torch.bucketize(torch.arange(0, 1 - 1e-6, 1 / int(m[:, 0].sum()), device=dev), bounds, right=True)
            bw = torch.bucketize(torch.arange(0, 1 - 1e-6, 1 / int(m[0].sum()), device=dev), bounds, right=True)
            ids[i][m.view(-1)] = (bh[:, None] * n + bw).flatten()
        return emb + E.position_embedding(ids)
    E.forward = forward
    return model

def published(name):
    """A file of the published SmolVLM-256M-Instruct ONNX export: the explorer's cache, else HF_HOME."""
    p = SMOLVLM_DIR / 'onnx' / name
    if p.exists(): return p
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(BASE, f'onnx/{name}'))

ATTRIBUTION = 'Base model: HuggingFaceTB/SmolVLM-256M-Instruct (Apache-2.0). Fine-tuning data: Astro Pilot Vision (see the dataset ATTRIBUTION.txt).\n'
def model_card(src):
    dec = 'the unmodified base text decoder of SmolVLM-256M-Instruct' if str(src) == BASE else f'the LoRA-merged Narrator text decoder ({Path(str(src)).name})'
    return ('# Narrator (SmolVLM-256M, Astro Pilot Vision)\n\n'
            f'Base: SmolVLM-256M-Instruct, Apache-2.0. Decoder: {dec}; decoder_model_merged.onnx is the fp32 parity reference and '
            'decoder_model_merged_{q4,q4f16,fp16}.onnx are the fused ONNX Runtime GenAI builds.\n'
            'Vision encoder and embed_tokens: the base weights. vision_encoder_quantized.onnx (q8) is our weight-only 8-bit re-quantisation '
            'of the published fp32 encoder; vision_encoder_{q4,int8,fp16}.onnx are the published files (fp16 is wrong on WebGPU).\n'
            'Images: preprocessor_config.json sets do_resize false, so every caller must resize the image to 512×512 (LANCZOS) before the processor.\n'
            'Intended use: on-demand captions and safety explanations for Astro Pilot frames in the browser. Trained on simulator frames: '
            'there is a sim-to-real gap; do not use for real flight decisions.\n')

def chat_template(src):
    """The chat template of a processor folder: tokenizer_config.json (the explorer's copy), else a hub snapshot's
    chat_template.json or chat_template.jinja."""
    src = Path(src); tok = json.loads((src / 'tokenizer_config.json').read_text())
    if tok.get('chat_template'): return tok['chat_template']
    if (src / 'chat_template.json').exists(): return json.loads((src / 'chat_template.json').read_text())['chat_template']
    return (src / 'chat_template.jinja').read_text()

def export_folder(src, out, processor_src=SMOLVLM_DIR, dtypes=DTYPES, vision=VISION):
    from transformers import AutoModelForVision2Seq
    out = Path(out); (out / 'onnx').mkdir(parents=True, exist_ok=True)
    model = AutoModelForVision2Seq.from_pretrained(src, dtype=torch.float32).eval(); hd = getattr(model.config.text_config, 'head_dim', 64)
    export_onnx(model, out / 'onnx' / 'decoder_model_merged.onnx')
    work = Path(tempfile.mkdtemp(prefix='.build_', dir=out.parent))
    try:
        save_text_llama(model, work / 'llama', processor_src)
        for d in dtypes: build_fused(work / 'llama', d, out / 'onnx' / f'decoder_model_merged{SUFFIX[d]}.onnx', work, hd)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    for f in ['embed_tokens_fp16.onnx', *[f'vision_encoder{SUFFIX[v]}.onnx' for v in vision]]: shutil.copy(published(f), out / 'onnx' / f)
    quantize_vision_w8(published('vision_encoder.onnx'), out / 'onnx' / f"vision_encoder{SUFFIX['q8']}.onnx")
    for f in ('config.json', 'processor_config.json', 'tokenizer.json', 'tokenizer_config.json'): shutil.copy(Path(processor_src) / f, out / f)
    # transformers 4.57 AutoProcessor reads the template only from chat_template.json/.jinja, not from tokenizer_config.json
    (out / 'chat_template.json').write_text(json.dumps({'chat_template': chat_template(processor_src)}))
    pre = preprocessor_config(json.loads((Path(processor_src) / 'preprocessor_config.json').read_text()))
    (out / 'preprocessor_config.json').write_text(json.dumps(pre, indent=1))
    gen = json.loads((Path(processor_src) / 'generation_config.json').read_text()); gen['eos_token_id'] = EOU
    (out / 'generation_config.json').write_text(json.dumps(gen, indent=1))
    (out / 'ATTRIBUTION.txt').write_text(ATTRIBUTION); (out / 'MODEL_CARD.md').write_text(model_card(src))
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--src', required=True); ap.add_argument('--out', required=True); ap.add_argument('--processor-src', default=str(SMOLVLM_DIR))
    ap.add_argument('--dtypes', default=','.join(DTYPES)); ap.add_argument('--vision', default=','.join(VISION))
    a = ap.parse_args(); print('exported', export_folder(a.src, a.out, a.processor_src, a.dtypes.split(','), a.vision.split(',')))
