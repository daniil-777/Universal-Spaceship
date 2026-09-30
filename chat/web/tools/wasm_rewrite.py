"""chat/web/tools/wasm_rewrite.py — prototype for the WASM rung (proposed for chat/export_web.py; the lead owns it): onnxruntime-web 1.30's
WASM build has no GatherBlockQuantized kernel ("Could not find an implementation for GatherBlockQuantized(1)"), which every int4
tied-embedding LFM2 graph uses (ours and onnx-community's). Replace it with standard ops on the same packed int4 table (no new bytes):
  x = Cast(Gather(data[V, D/2] uint8, ids), int32); hi = Div(x, 16); lo = x - 16*hi; q = interleave(lo, hi) → [.., D/bs, bs]
  emb = Reshape((Cast(q, float) - zp) * Gather(scales[V, D/bs], ids)[..., None], [.., D])  (zp: the zero_points input or 2^(bits-1))
Writes <dst>/onnx/model_<dtype>.onnx and links the tokenizer/config files; --check compares logits with the original on ORT CPU.
--accuracy-level N also sets every MatMulNBits accuracy_level (the builder writes 4 = int8-activation compute; see wasm_speed.mjs).
  python chat/web/tools/wasm_rewrite.py <src web folder> <dst web folder> [--dtypes q4] [--accuracy-level 0] [--check]"""
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

def check(src, dst, ids=(1, 6, 1098, 13434, 7, 65535, 0, 42)):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.log_severity_level = 3; outs = []
    for f in (src, dst):
        s = ort.InferenceSession(str(f), so, providers=['CPUExecutionProvider']); feed = {}
        for i in s.get_inputs():
            if i.name.startswith(('past_key_values.', 'past_conv.')):
                feed[i.name] = np.zeros([1 if d == 'batch_size' else 0 if isinstance(d, str) else d for d in i.shape], np.float16 if 'float16' in i.type else np.float32)
        feed.update(input_ids=np.array([ids], np.int64), attention_mask=np.ones((1, len(ids)), np.int64))
        if 'num_logits_to_keep' in [i.name for i in s.get_inputs()]: feed['num_logits_to_keep'] = np.array(0, np.int64)
        outs.append(s.run(['logits'], feed)[0].astype(np.float32))
    d = np.abs(outs[0] - outs[1]); return {'max_abs_logit_diff': float(d.max()), 'argmax_equal': bool((outs[0].argmax(-1) == outs[1].argmax(-1)).all())}

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('dst'); ap.add_argument('--dtypes', default='q4'); ap.add_argument('--check', action='store_true')
    ap.add_argument('--accuracy-level', type=int, default=None)
    a = ap.parse_args(); s, d = Path(a.src), Path(a.dst); (d / 'onnx').mkdir(parents=True, exist_ok=True)
    for f in s.iterdir():
        if f.is_file() and not f.name.startswith('._') and not (d / f.name).exists(): os.symlink(f.resolve(), d / f.name)
    for dt in a.dtypes.split(','):
        f = s / 'onnx' / f'model_{dt}.onnx'; m = onnx.load(str(f)); k = rewrite(m)
        if a.accuracy_level is not None:
            for n in m.graph.node:
                for x in n.attribute:
                    if n.op_type == 'MatMulNBits' and x.name == 'accuracy_level': x.i = a.accuracy_level
        onnx.save(m, str(d / 'onnx' / f.name), save_as_external_data=False); print(dt, 'rewrote', k, 'GatherBlockQuantized', '->', d / 'onnx' / f.name)
        if a.check: print(dt, check(f, d / 'onnx' / f.name))
