import unittest, torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache
from vlm.train.narrator.export_decoder import Decoder

class TestDecoder(unittest.TestCase):
    def test_matches_hf_llama_with_and_without_past(self):
        torch.manual_seed(0)
        cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=97, rope_theta=100000.0, rms_norm_eps=1e-5)
        m = LlamaForCausalLM(cfg).eval(); dec = Decoder(m.model, m.lm_head, cfg).eval()
        e1 = torch.randn(1, 5, 64); am = torch.ones(1, 5, dtype=torch.int64); pos = torch.arange(5)[None]
        empty = [torch.zeros(1, 2, 0, 16) for _ in range(4)]
        with torch.no_grad():
            ref = m(inputs_embeds=e1, attention_mask=am, position_ids=pos, past_key_values=DynamicCache(), use_cache=True)
            out = dec(e1, am, pos, *empty)
            self.assertLess((out[0] - ref.logits).abs().max().item(), 1e-4)
            e2 = torch.randn(1, 1, 64); am2 = torch.ones(1, 6, dtype=torch.int64); pos2 = torch.tensor([[5]])
            ref2 = m(inputs_embeds=e2, attention_mask=am2, position_ids=pos2, past_key_values=ref.past_key_values, use_cache=True)
            out2 = dec(e2, am2, pos2, *out[1:])
            self.assertLess((out2[0] - ref2.logits).abs().max().item(), 1e-4)
            self.assertEqual(tuple(out2[1].shape), (1, 2, 6, 16))
