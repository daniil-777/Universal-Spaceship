import os
os.environ.setdefault('HF_HOME', '/Volumes/LaCie/astro-pilot/vlm/hf')
import tempfile, unittest
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
import numpy as np, onnx, onnxruntime as ort, torch
from onnx import TensorProto, helper
from PIL import Image
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache
from vlm.train.narrator.export_decoder import (BASE, SMOLVLM_DIR, Decoder, build_fused, export_onnx, fix_io, io_names, model_card, preprocessor_config,
                                               quantize_vision_w8, save_text_llama, square, vision_positions_fix)
from vlm.train.narrator.check_parity import TOL, forced_agreement, gate_failures, greedy_torch, run_onnx, vision_candidate

LACIE_TMP = '/Volumes/LaCie/astro-pilot/vlm/tmp'

def tmpdir(case):
    """A temporary directory under $TMPDIR when that is on LaCie (else LaCie's tmp, never the Mac disk), removed after the test."""
    base = os.environ.get('TMPDIR', '')
    d = tempfile.TemporaryDirectory(dir=base if base.startswith('/Volumes/LaCie/') else LACIE_TMP); case.addCleanup(d.cleanup)
    return Path(d.name)

def tiny_vlm():
    """A tiny random Llama, wrapped like Idefics3ForConditionalGeneration (config.text_config, model.text_model, lm_head)."""
    torch.manual_seed(0)
    cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=97, rope_theta=100000.0, rms_norm_eps=1e-5)
    m = LlamaForCausalLM(cfg).eval()
    return m, cfg, SimpleNamespace(config=SimpleNamespace(text_config=cfg), model=SimpleNamespace(text_model=m.model), lm_head=m.lm_head)

def forced_run(case, m, cfg, vlm, sess):
    """torch greedy for 8 tokens, then the ONNX session stepped through the same tokens; returns (tokens, torch logits, onnx logits)."""
    e = torch.randn(1, 5, 64)
    with torch.no_grad():
        tt, lt = greedy_torch(vlm, e, 8); _, lo = run_onnx(sess, cfg, e, lambda t: m.model.embed_tokens(torch.tensor([[t]])).numpy(), tt)
    case.assertEqual(len(tt), 8)
    return tt, lt, lo

class TestDecoder(unittest.TestCase):
    def test_matches_hf_llama_with_and_without_past(self):
        m, cfg, _ = tiny_vlm(); dec = Decoder(m.model, m.lm_head, cfg).eval()
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

class TestExportOnnx(unittest.TestCase):
    def test_traced_decoder_has_the_contract_names_and_follows_torch_greedy(self):
        m, cfg, vlm = tiny_vlm(); d = tmpdir(self); export_onnx(vlm, d / 'dec.onnx'); sess = ort.InferenceSession(str(d / 'dec.onnx'))
        past, present = io_names(cfg.num_hidden_layers)
        self.assertEqual([i.name for i in sess.get_inputs()], ['inputs_embeds', 'attention_mask', 'position_ids', *past])
        self.assertEqual([o.name for o in sess.get_outputs()], ['logits', *present])
        tt, lt, lo = forced_run(self, m, cfg, vlm, sess)
        self.assertEqual(forced_agreement(lo, tt), 1.0); self.assertLess(max(float(np.abs(a - b).max()) for a, b in zip(lt, lo)), 1e-4)

@unittest.skipUnless(find_spec('onnxruntime_genai') and (SMOLVLM_DIR / 'tokenizer.json').exists(), 'needs onnxruntime-genai and the SmolVLM tokenizer files')
class TestBuildFused(unittest.TestCase):
    def test_builder_fp16_decoder_gets_the_contract_without_position_ids_and_follows_torch(self):
        m, cfg, vlm = tiny_vlm(); d = tmpdir(self)
        save_text_llama(vlm, d / 'llama'); build_fused(d / 'llama', 'fp16', d / 'dec.onnx', d / 'work', head_dim=16)
        sess = ort.InferenceSession(str(d / 'dec.onnx')); ins = {i.name: i for i in sess.get_inputs()}; outs = {o.name: o for o in sess.get_outputs()}
        past, present = io_names(cfg.num_hidden_layers)
        self.assertEqual(set(ins), {'inputs_embeds', 'attention_mask', *past}); self.assertEqual(set(outs), {'logits', *present})
        self.assertEqual((ins['inputs_embeds'].type, outs['logits'].type, ins[past[0]].type), ('tensor(float)', 'tensor(float)', 'tensor(float16)'))
        self.assertEqual(ins[past[0]].shape[1:], [2, 'past_sequence_length', 16]); self.assertEqual(outs[present[0]].shape[3], 16)
        tt, lt, lo = forced_run(self, m, cfg, vlm, sess)
        self.assertEqual(forced_agreement(lo, tt), 1.0); self.assertLess(max(float(np.abs(a - b).max()) for a, b in zip(lt, lo)), 2e-2)

def builder_like(dtype):
    """A two-node stand-in for a GenAI-builder decoder: builder I/O names, a symbolic kv_cache_dim, fp16 or fp32 I/O."""
    kv = lambda n, s: helper.make_tensor_value_info(n, dtype, ['batch_size', 2, s, 'kv_cache_dim'])
    ins = [helper.make_tensor_value_info('attention_mask', TensorProto.INT64, ['batch_size', 'total_sequence_length']),
           helper.make_tensor_value_info('inputs_embeds', dtype, ['batch_size', 'sequence_length', 4]), kv('past_key_values.0.key', 'past_sequence_length')]
    outs = [helper.make_tensor_value_info('logits', dtype, ['batch_size', 'sequence_length', 4]), kv('present.0.key', 'total_sequence_length')]
    two = helper.make_tensor('two', dtype, [], [2.0])
    nodes = [helper.make_node('Mul', ['inputs_embeds', 'two'], ['logits']), helper.make_node('Identity', ['past_key_values.0.key'], ['present.0.key'])]
    return helper.make_model(helper.make_graph(nodes, 'g', ins, outs, [two]), opset_imports=[helper.make_opsetid('', 17)], ir_version=10)

class TestFixIo(unittest.TestCase):
    def run_fixed(self, dtype):
        d = tmpdir(self); onnx.save(builder_like(dtype), str(d / 'model.onnx')); fix_io(d / 'model.onnx', d / 'out.onnx', head_dim=8)
        m = onnx.load(str(d / 'out.onnx')); io = {v.name: v.type.tensor_type for v in [*m.graph.input, *m.graph.output]}
        return d, m, io

    def test_fp16_graph_gets_float32_embeds_and_logits_and_a_fixed_head_dim(self):
        d, m, io = self.run_fixed(TensorProto.FLOAT16)
        self.assertEqual(io['inputs_embeds'].elem_type, TensorProto.FLOAT); self.assertEqual(io['logits'].elem_type, TensorProto.FLOAT)
        self.assertEqual(io['past_key_values.0.key'].elem_type, TensorProto.FLOAT16)
        for n in ('past_key_values.0.key', 'present.0.key'): self.assertEqual(io[n].shape.dim[3].dim_value, 8)
        self.assertEqual(sorted(p.name for p in d.iterdir() if not p.name.startswith('._')), ['model.onnx', 'out.onnx'])
        s = ort.InferenceSession(str(d / 'out.onnx')); x = np.random.rand(1, 3, 4).astype(np.float32)
        out = s.run(None, {'attention_mask': np.ones((1, 3), np.int64), 'inputs_embeds': x, 'past_key_values.0.key': np.zeros((1, 2, 0, 8), np.float16)})
        self.assertEqual(out[0].dtype, np.float32); self.assertLess(np.abs(out[0] - 2 * x).max(), 1e-2)

    def test_fp32_graph_keeps_its_nodes_and_only_fixes_the_head_dim(self):
        _, m, io = self.run_fixed(TensorProto.FLOAT)
        self.assertEqual([n.op_type for n in m.graph.node], ['Mul', 'Identity']); self.assertEqual(io['present.0.key'].shape.dim[3].dim_value, 8)

class TestVisionW8(unittest.TestCase):
    def test_weight_only_8bit_keeps_float_activations_and_stays_close(self):
        rng = np.random.default_rng(0); w = rng.standard_normal((256, 128)).astype(np.float32)
        g = helper.make_graph([helper.make_node('MatMul', ['x', 'w'], ['y'])], 'g', [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 256])],
                              [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 128])], [onnx.numpy_helper.from_array(w, 'w')])
        d = tmpdir(self); onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid('', 17)], ir_version=10), str(d / 'fp32.onnx'))
        quantize_vision_w8(d / 'fp32.onnx', d / 'w8.onnx')
        n = onnx.load(str(d / 'w8.onnx')).graph.node; attrs = {a.name: helper.get_attribute_value(a) for a in n[0].attribute}
        self.assertEqual((n[0].op_type, attrs['bits'], attrs['block_size']), ('MatMulNBits', 8, 32))
        x = rng.standard_normal((1, 256)).astype(np.float32); y = ort.InferenceSession(str(d / 'w8.onnx')).run(None, {'x': x})[0]
        self.assertGreater(float(np.dot(y[0], (x @ w)[0]) / (np.linalg.norm(y) * np.linalg.norm(x @ w))), 0.9999)

class TestPreprocessing(unittest.TestCase):
    def test_config_turns_off_the_resize_and_the_splitting(self):
        c = preprocessor_config({'do_resize': True, 'do_image_splitting': True, 'size': {'longest_edge': 2048}, 'max_image_size': {'longest_edge': 364}})
        self.assertEqual((c['do_resize'], c['do_image_splitting'], c['max_image_size']), (False, False, {'longest_edge': 512}))

    def test_square_is_a_512_rgb_image(self):
        self.assertEqual(square(Image.new('RGBA', (1600, 900))).size, (512, 512)); self.assertEqual(square(Image.new('L', (90, 160))).mode, 'RGB')

class TestModelCard(unittest.TestCase):
    def test_card_names_the_decoder_source_the_q8_requantisation_and_the_512_resize(self):
        base, merged = model_card(BASE), model_card('/Volumes/LaCie/astro-pilot/vlm/models/narrator-merged')
        self.assertIn('unmodified base', base); self.assertNotIn('LoRA-merged', base); self.assertIn('LoRA-merged', merged)
        for c in (base, merged): self.assertIn('re-quantisation', c); self.assertIn('512×512', c); self.assertIn('do_resize', c)

class TestVisionPositions(unittest.TestCase):
    def test_patched_embeddings_add_position_rows_0_to_n_minus_1_like_the_published_encoder(self):
        from transformers.models.idefics3.configuration_idefics3 import Idefics3VisionConfig
        from transformers.models.idefics3.modeling_idefics3 import Idefics3VisionEmbeddings
        torch.manual_seed(0); E = Idefics3VisionEmbeddings(Idefics3VisionConfig(image_size=64, patch_size=16, hidden_size=8)).eval()
        vision_positions_fix(SimpleNamespace(model=SimpleNamespace(vision_model=SimpleNamespace(embeddings=E))))
        pv, mask = torch.randn(1, 3, 64, 64), torch.ones(1, 4, 4, dtype=torch.bool)
        with torch.no_grad():
            want = E.patch_embedding(pv).flatten(2).transpose(1, 2) + E.position_embedding.weight[None]
            self.assertLess((E(pv, mask) - want).abs().max().item(), 1e-6)
            half = mask.clone(); half[:, :, 2:] = False; ids = torch.zeros(16, dtype=torch.long); ids[[0, 1, 4, 5, 8, 9, 12, 13]] = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14])
            self.assertLess((E(pv, half) - (E.patch_embedding(pv).flatten(2).transpose(1, 2) + E.position_embedding.weight[ids][None])).abs().max().item(), 1e-6)

    def test_refuses_embeddings_that_are_not_idefics3(self):
        with self.assertRaisesRegex(TypeError, 'Idefics3VisionEmbeddings'):
            vision_positions_fix(SimpleNamespace(model=SimpleNamespace(vision_model=SimpleNamespace(embeddings=torch.nn.Linear(2, 2)))))

class TestGates(unittest.TestCase):
    def test_forced_agreement_is_the_top1_hit_rate(self):
        lg = [np.eye(5)[i] for i in (1, 2, 3, 0)]
        self.assertEqual(forced_agreement(lg, [1, 2, 4, 0]), 0.75)

    def test_vision_candidate_is_the_smallest_file_at_cos_099(self):
        v = {'q4': {'mean_cos': 0.87, 'mb': 63.8}, 'q8': {'mean_cos': 0.995, 'mb': 94.2}, 'fp16': {'mean_cos': 0.9999, 'mb': 187.3}}
        self.assertEqual(vision_candidate(v), 'q8'); self.assertIsNone(vision_candidate({'q4': {'mean_cos': 0.87, 'mb': 63.8}}))

    def ok(self):
        return {'a': {'tokens_equal': [True], 'max_dlogit': [1e-4]}, 'b': {'candidate': 'fp16'}, 'e': None,
                'c': {'q4': {'ours': 0.80, 'published': 0.82}, 'fp16': {'ours': 0.99, 'published': 0.98}},
                'd': {'pixel_values_shape': [1, 1, 3, 512, 512], 'n_image_tokens': [64], 'prompt_len': [82]}}

    def test_gate_passes_within_tol_of_the_published_build(self):
        self.assertEqual(TOL, 0.03); self.assertEqual(gate_failures(self.ok()), [])

    def test_gate_fails_below_published_minus_tol_or_without_a_reference_or_a_vision_candidate(self):
        r = self.ok(); r['c']['q4']['ours'] = 0.78; self.assertEqual(len(gate_failures(r)), 1)
        r = self.ok(); r['c']['q4']['published'] = None; self.assertIn('no published', gate_failures(r)[0])
        r = self.ok(); r['b']['candidate'] = None; self.assertIn('(b)', gate_failures(r)[0])
        r = self.ok(); r['d']['n_image_tokens'] = [63]; self.assertIn('(d)', gate_failures(r)[0])
