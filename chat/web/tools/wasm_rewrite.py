"""chat/web/tools/wasm_rewrite.py — prototype for the WASM rung (proposed for chat/export_web.py; the lead owns it): onnxruntime-web 1.30's
WASM build has no GatherBlockQuantized kernel ("Could not find an implementation for GatherBlockQuantized(1)"), which every int4
tied-embedding LFM2 graph uses (ours and onnx-community's). Replace it with standard ops on the same packed int4 table (no new bytes):
  x = Cast(Gather(data[V, D/2] uint8, ids), int32); hi = Div(x, 16); lo = x - 16*hi; q = interleave(lo, hi) → [.., D/bs, bs]
  emb = Reshape((Cast(q, float) - zp) * Gather(scales[V, D/bs], ids)[..., None], [.., D])  (zp: the zero_points input or 2^(bits-1))
Writes <dst>/onnx/model_<dtype>.onnx and links the tokenizer/config files; --check compares logits with the original on ORT CPU.
--accuracy-level N also sets every MatMulNBits accuracy_level (the builder writes 4 = int8-activation compute; see wasm_speed.mjs).
--dq-matmul: the WASM fast path. ORT-web 1.30 WASM has no SIMD int4 GEMM: MatMulNBits dequantizes its weight on every call (~25x slower
than the fp32 MatMul). Each MatMulNBits (4 bit, no zp) becomes DequantizeLinear(UINT4 [N, K] = the same bytes, scales, zp 8, axis 1,
block) → Transpose → MatMul, and the tied embedding Gathers rows of the lm_head's dequantized table; loaded with the session config
session.disable_quant_qdq=1 ORT constant-folds all of it to fp32 once (download unchanged, ~4 bytes/param of WASM heap).
  python chat/web/tools/wasm_rewrite.py <src web folder> <dst web folder> [--dtypes q4] [--accuracy-level 0] [--dq-matmul] [--check]"""
import argparse, os, sys
from pathlib import Path
import numpy as np, onnx
from onnx import TensorProto, helper, numpy_helper

def rewrite(m):
    g = m.graph; out_type = {v.name: v.type.tensor_type.elem_type for v in g.value_info}
    nodes = [n for n in g.node if n.op_type == 'GatherBlockQuantized']
    for n in nodes:
        a = {x.name: helper.get_attribute_value(x) for x in n.attribute}
        if a.get('bits', 4) != 4 or a.get('gather_axis', 0) != 0 or a.get('quantize_axis', 1) != 1: raise SystemExit(f'unsupported {a}')
        bs, p = a.get('block_size', 128), n.name.strip('/').replace('/', '_') + '_x'
        data, ids, scales = n.input[0], n.input[1], n.input[2]
        zp = n.input[3] if len(n.input) > 3 and n.input[3] else None
        # the embedding dtype = the scales' dtype (fp32 in q4, fp16 in q4f16 graphs)
        st = next(i.data_type for i in g.initializer if i.name == scales)
        c = lambda name, v, t=TensorProto.INT64: g.initializer.append(helper.make_tensor(f'{p}/{name}', t, [len(v)] if isinstance(v, list) else [], v if isinstance(v, list) else [v]))
        c('16', 16, TensorProto.INT32); c('m1', [-1]); c('shape_q', [0, 0, -1, bs]); c('shape_o', [0, 0, -1])
        new = [helper.make_node('Gather', [data, ids], [f'{p}/packed'], axis=0),
               helper.make_node('Cast', [f'{p}/packed'], [f'{p}/i32'], to=TensorProto.INT32),
               helper.make_node('Div', [f'{p}/i32', f'{p}/16'], [f'{p}/hi']),
               helper.make_node('Mul', [f'{p}/hi', f'{p}/16'], [f'{p}/hi16']),
               helper.make_node('Sub', [f'{p}/i32', f'{p}/hi16'], [f'{p}/lo']),
               helper.make_node('Unsqueeze', [f'{p}/lo', f'{p}/m1'], [f'{p}/lo1']),
               helper.make_node('Unsqueeze', [f'{p}/hi', f'{p}/m1'], [f'{p}/hi1']),
               helper.make_node('Concat', [f'{p}/lo1', f'{p}/hi1'], [f'{p}/pairs'], axis=-1),
               helper.make_node('Reshape', [f'{p}/pairs', f'{p}/shape_q'], [f'{p}/q']),
               helper.make_node('Cast', [f'{p}/q'], [f'{p}/qf'], to=st)]
        if zp:  # packed uint8 zero points [V, blocks/2] → [.., blocks, 1]
            new += [helper.make_node('Gather', [zp, ids], [f'{p}/zpk'], axis=0), helper.make_node('Cast', [f'{p}/zpk'], [f'{p}/zi'], to=TensorProto.INT32),
                    helper.make_node('Div', [f'{p}/zi', f'{p}/16'], [f'{p}/zhi']), helper.make_node('Mul', [f'{p}/zhi', f'{p}/16'], [f'{p}/zhi16']),
                    helper.make_node('Sub', [f'{p}/zi', f'{p}/zhi16'], [f'{p}/zlo']), helper.make_node('Unsqueeze', [f'{p}/zlo', f'{p}/m1'], [f'{p}/zlo1']),
                    helper.make_node('Unsqueeze', [f'{p}/zhi', f'{p}/m1'], [f'{p}/zhi1']), helper.make_node('Concat', [f'{p}/zlo1', f'{p}/zhi1'], [f'{p}/zpairs'], axis=-1),
                    helper.make_node('Reshape', [f'{p}/zpairs', f'{p}/shape_o'], [f'{p}/zq']), helper.make_node('Unsqueeze', [f'{p}/zq', f'{p}/m1'], [f'{p}/zq1']),
                    helper.make_node('Cast', [f'{p}/zq1'], [f'{p}/zpf'], to=st)]
        else:
            g.initializer.append(helper.make_tensor(f'{p}/zpf', st, [], [8.0]));
        new += [helper.make_node('Sub', [f'{p}/qf', f'{p}/zpf'], [f'{p}/centered']),
                helper.make_node('Gather', [scales, ids], [f'{p}/s'], axis=0),
                helper.make_node('Unsqueeze', [f'{p}/s', f'{p}/m1'], [f'{p}/s1']),
                helper.make_node('Mul', [f'{p}/centered', f'{p}/s1'], [f'{p}/deq']),
                helper.make_node('Reshape', [f'{p}/deq', f'{p}/shape_o'], [n.output[0]])]
        i = list(g.node).index(n); g.node.remove(n)
        for k, x in enumerate(new): g.node.insert(i + k, x)
    return len(nodes)

def dq_matmul(m):
    """MatMulNBits → DequantizeLinear + Transpose + MatMul (opset 21); GatherBlockQuantized over a converted lm_head → Gather."""
    g = m.graph; init = {i.name: i for i in g.initializer}; prod = {o: n for n in g.node for o in n.output}; head = {}
    for op in m.opset_import:
        if op.domain in ('', 'ai.onnx') and op.version < 21: op.version = 21
    for n in [n for n in g.node if n.op_type == 'MatMulNBits']:
        a = {x.name: helper.get_attribute_value(x) for x in n.attribute}
        if a.get('bits', 4) != 4 or len([i for i in n.input if i]) != 3: raise SystemExit(f'{n.name}: unsupported MatMulNBits {a} {list(n.input)}')
        K, N, bs = a['K'], a['N'], a['block_size']; B, S = init[n.input[1]], init[n.input[2]]; nb = (K + bs - 1) // bs
        if K % bs: raise SystemExit(f'{n.name}: K {K} not a multiple of the block {bs}')
        p = n.name.strip('/').replace('/', '_') + '_dq'
        w4 = onnx.TensorProto(); w4.name = f'{p}/w'; w4.data_type = TensorProto.UINT4; w4.dims[:] = [N, K]; w4.raw_data = numpy_helper.to_array(B).tobytes()
        zp = onnx.TensorProto(); zp.name = f'{p}/zp'; zp.data_type = TensorProto.UINT4; zp.dims[:] = [N, nb]; zp.raw_data = bytes([0x88]) * ((N * nb + 1) // 2)
        sc = numpy_helper.from_array(numpy_helper.to_array(S).reshape(N, nb), f'{p}/s')
        g.initializer.extend([w4, zp, sc]); head[n.input[1]] = (f'{p}/wf', n.input[2])
        new = [helper.make_node('DequantizeLinear', [w4.name, sc.name, zp.name], [f'{p}/wf'], axis=1, block_size=bs),
               helper.make_node('Transpose', [f'{p}/wf'], [f'{p}/wt'], perm=[1, 0]),
               helper.make_node('MatMul', [n.input[0], f'{p}/wt'], [n.output[0]])]
        i = list(g.node).index(n); g.node.remove(n)
        for k, x in enumerate(new): g.node.insert(i + k, x)
    for n in [n for n in g.node if n.op_type == 'GatherBlockQuantized']:
        src = prod.get(n.input[0]); src = src.input[0] if src is not None and src.op_type == 'Reshape' else n.input[0]
        if src in head and head[src][1] == n.input[2] and len([i for i in n.input if i]) == 3:
            i = list(g.node).index(n); g.node.remove(n); g.node.insert(i, helper.make_node('Gather', [head[src][0], n.input[1]], list(n.output), axis=0))
    outs = {o.name for o in g.output}
    while True:  # drop dead nodes (the embedding's Reshape of the packed table), then unused initializers
        used = {i for n in g.node for i in n.input} | outs
        dead = [x for x in g.node if not any(o in used for o in x.output)]
        if not dead: break
        for x in dead: g.node.remove(x)
    for x in [x for x in g.initializer if x.name not in used]: g.initializer.remove(x)
    return len(head)

def check(src, dst, ids=(1, 6, 1098, 13434, 7, 65535, 0, 42)):
    import onnxruntime as ort
    outs = []
    ref = onnx.load(str(src))  # the reference at fp32 compute (accuracy_level 0), as the rewritten graphs compute
    for n in ref.graph.node:
        for x in n.attribute:
            if n.op_type == 'MatMulNBits' and x.name == 'accuracy_level': x.i = 0
    for f in (ref.SerializeToString(), str(dst)):
        so = ort.SessionOptions(); so.log_severity_level = 3; so.add_session_config_entry('session.disable_quant_qdq', '1')
        s = ort.InferenceSession(f, so, providers=['CPUExecutionProvider']); feed = {}
        for i in s.get_inputs():
            if i.name.startswith(('past_key_values.', 'past_conv.')):
                feed[i.name] = np.zeros([1 if d == 'batch_size' else 0 if isinstance(d, str) else d for d in i.shape], np.float16 if 'float16' in i.type else np.float32)
        feed.update(input_ids=np.array([ids], np.int64), attention_mask=np.ones((1, len(ids)), np.int64))
        if 'num_logits_to_keep' in [i.name for i in s.get_inputs()]: feed['num_logits_to_keep'] = np.array(0, np.int64)
        outs.append(s.run(['logits'], feed)[0].astype(np.float32))
    d = np.abs(outs[0] - outs[1]); return {'max_abs_logit_diff': float(d.max()), 'argmax_equal': bool((outs[0].argmax(-1) == outs[1].argmax(-1)).all())}

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('dst'); ap.add_argument('--dtypes', default='q4'); ap.add_argument('--check', action='store_true')
    ap.add_argument('--accuracy-level', type=int, default=None); ap.add_argument('--dq-matmul', action='store_true')
    a = ap.parse_args(); s, d = Path(a.src), Path(a.dst); (d / 'onnx').mkdir(parents=True, exist_ok=True)
    for f in s.iterdir():
        if f.is_file() and not f.name.startswith('._') and not (d / f.name).exists(): os.symlink(f.resolve(), d / f.name)
    for dt in a.dtypes.split(','):
        f = s / 'onnx' / f'model_{dt}.onnx'; m = onnx.load(str(f)); k = dq_matmul(m) if a.dq_matmul else 0; k2 = rewrite(m)
        if a.dq_matmul: print(dt, 'MatMulNBits -> DequantizeLinear+MatMul:', k)
        if a.accuracy_level is not None:
            for n in m.graph.node:
                for x in n.attribute:
                    if n.op_type == 'MatMulNBits' and x.name == 'accuracy_level': x.i = a.accuracy_level
        onnx.save(m, str(d / 'onnx' / f.name), save_as_external_data=False); print(dt, 'decomposed', k2, 'GatherBlockQuantized', '->', d / 'onnx' / f.name)
        if a.check: print(dt, check(f, d / 'onnx' / f.name))
