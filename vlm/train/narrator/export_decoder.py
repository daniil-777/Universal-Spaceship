"""Export the SmolVLM text decoder (Llama: 30 layers, hidden 576, 9 heads, 3 KV heads, head_dim 64) with the exact
transformers.js contract (spec §10.2, io_names.py), quantize it to 4-bit MatMulNBits, and lay out the folder
transformers.js 4.3 loads. The decoder is written out from the checkpoint's own weights (RMSNorm, RoPE, GQA with the
past concatenated, SiLU MLP) so tracing never meets transformers' cache or mask helpers."""
import argparse, json, os, shutil
from pathlib import Path
import onnx, torch
from torch import nn

SMOLVLM_DIR = Path(os.environ.get('SMOLVLM_DIR', '/Volumes/LaCie/astro-pilot/test/out/vlm_explore/hfcache/HuggingFaceTB/SmolVLM-256M-Instruct'))
EOU = 49279

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

def quantize_q4(fp32_path, q4_path):
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    q = MatMulNBitsQuantizer(onnx.load(str(fp32_path)), bits=4, block_size=32, is_symmetric=True)
    q.process(); q.model.save_model_to_file(str(q4_path), use_external_data_format=False)

ATTRIBUTION = 'Base model: HuggingFaceTB/SmolVLM-256M-Instruct (Apache-2.0). Fine-tuning data: Astro Pilot Vision (see the dataset ATTRIBUTION.txt).\n'
MODEL_CARD = '# Narrator (SmolVLM-256M, Astro Pilot Vision)\n\nBase: SmolVLM-256M-Instruct, Apache-2.0. Decoder LoRA-merged; vision encoder and embeddings unchanged.\nIntended use: on-demand captions and safety explanations for Astro Pilot frames in the browser. Trained on simulator frames: there is a sim-to-real gap; do not use for real flight decisions.\n'

def export_folder(src, out, processor_src=SMOLVLM_DIR):
    from transformers import AutoModelForVision2Seq
    out = Path(out); (out / 'onnx').mkdir(parents=True, exist_ok=True)
    model = AutoModelForVision2Seq.from_pretrained(src, torch_dtype=torch.float32).eval()
    export_onnx(model, out / 'onnx' / 'decoder_model_merged.onnx')
    quantize_q4(out / 'onnx' / 'decoder_model_merged.onnx', out / 'onnx' / 'decoder_model_merged_q4.onnx')
    for f in ('vision_encoder_q4.onnx', 'embed_tokens_fp16.onnx'): shutil.copy(SMOLVLM_DIR / 'onnx' / f, out / 'onnx' / f)
    for f in ('config.json', 'processor_config.json', 'tokenizer.json', 'tokenizer_config.json'): shutil.copy(Path(processor_src) / f, out / f)
    # transformers 4.57 AutoProcessor reads the template only from chat_template.json/.jinja, not from tokenizer_config.json
    tok = json.loads((Path(processor_src) / 'tokenizer_config.json').read_text()); (out / 'chat_template.json').write_text(json.dumps({'chat_template': tok['chat_template']}))
    pre = json.loads((Path(processor_src) / 'preprocessor_config.json').read_text()); pre['do_image_splitting'] = False; pre['max_image_size'] = {'longest_edge': 512}
    (out / 'preprocessor_config.json').write_text(json.dumps(pre, indent=1))
    gen = json.loads((Path(processor_src) / 'generation_config.json').read_text()); gen['eos_token_id'] = EOU
    (out / 'generation_config.json').write_text(json.dumps(gen, indent=1))
    (out / 'ATTRIBUTION.txt').write_text(ATTRIBUTION); (out / 'MODEL_CARD.md').write_text(MODEL_CARD)
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--src', required=True); ap.add_argument('--out', required=True); ap.add_argument('--processor-src', default=str(SMOLVLM_DIR))
    a = ap.parse_args(); print('exported', export_folder(a.src, a.out, a.processor_src))
