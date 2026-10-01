"""vlm/train/colab/env.py — the machine the notebook runs on: the GPU kind and its mixed precision (bf16 on A100/L4/H100,
fp16 on T4/V100, fp32 on a CPU dry run), the batch and worker sizes, the pinned pip set, Node (slot_eval.mjs) and the
SmolVLM base files (processor configs plus the published ONNX pieces the Narrator export reuses)."""
import json, os, shutil, subprocess, sys, tarfile, urllib.request

BASE = 'HuggingFaceTB/SmolVLM-256M-Instruct'
# pinned (the LaCie venv's versions); a spec after '|' is the fallback when the pin has no wheel for this Python
PIP_CORE = ['transformers==4.57.6', 'peft==0.21.1', 'accelerate==1.15.0|accelerate>=1.0', 'onnx==1.23.1|onnx>=1.17', 'onnxruntime==1.30.0|onnxruntime>=1.22',
            'onnxscript==0.7.2|onnxscript', 'onnxruntime-genai==0.15.2|onnxruntime-genai>=0.15,<0.16', 'timm==1.0.30|timm>=1.0.9,<1.1']
PIP_EXTRA = ['webdataset==1.0.2|webdataset']   # optional: the shards are plain tars the notebook extracts itself
NODE_URL = 'https://nodejs.org/dist/v20.18.1/node-v20.18.1-linux-x64.tar.xz'

def pip_install(specs, run=subprocess.run, log=print, required=True):
    """pip install each spec, trying its fallback when the pin fails; returns the specs installed."""
    done = []
    for spec in specs:
        for s in spec.split('|'):
            r = run([sys.executable, '-m', 'pip', 'install', '-q', s], capture_output=True, text=True)
            if r.returncode == 0: done.append(s); break
            log(f'pip install {s} failed: {(r.stderr or "").strip().splitlines()[-1:]}')
        else:
            if required: raise RuntimeError(f'could not install {spec}')
    return done

def detect_gpu():
    import torch
    if not torch.cuda.is_available():
        return {'kind': 'cpu', 'name': 'cpu', 'mem_gb': 0.0, 'bf16': False, 'device': 'cpu'}
    name = torch.cuda.get_device_name(0); mem = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    kind = next((k for k in ('H100', 'A100', 'L4', 'T4', 'V100', 'P100', 'A10G') if k in name), 'other')
    return {'kind': kind, 'name': name, 'mem_gb': round(mem, 1), 'bf16': bool(torch.cuda.is_bf16_supported()), 'device': 'cuda'}

def plan_for(gpu, cpus=None):
    """Batch, accumulation, learning rate and loader workers per GPU (the Narrator's effective batch stays 16-32)."""
    cpus = cpus or os.cpu_count() or 2; k, mem = gpu['kind'], gpu['mem_gb']
    if k in ('H100', 'A100') or mem >= 38: nar, eye = {'batch': 32, 'accum': 1, 'lr': 3e-4}, {'batch': 256}
    elif k in ('L4', 'A10G') or mem >= 20: nar, eye = {'batch': 16, 'accum': 1, 'lr': 2e-4}, {'batch': 192}
    elif gpu['device'] == 'cuda': nar, eye = {'batch': 8, 'accum': 2, 'lr': 2e-4}, {'batch': 128}
    else: nar, eye = {'batch': 1, 'accum': 1, 'lr': 2e-4}, {'batch': 8}
    dtype = 'bf16' if gpu['bf16'] else 'fp16' if gpu['device'] == 'cuda' else 'fp32'
    return {'gpu': gpu, 'dtype': dtype, 'workers': max(0, min(8, cpus - 1)) if gpu['device'] == 'cuda' else 0, 'narrator': nar, 'eye': eye}

def ensure_node(dest='/content/node', min_major=18, log=print):
    """A Node >= 18 binary for slot_eval.mjs: the one on PATH, else the official linux-x64 build unpacked into dest."""
    for cand in (shutil.which('node'), os.path.join(dest, 'bin', 'node')):
        if cand and os.path.exists(cand):
            try:
                v = subprocess.run([cand, '--version'], capture_output=True, text=True).stdout.strip()
                if int(v.lstrip('v').split('.')[0]) >= min_major: return cand
            except (OSError, ValueError): pass
    os.makedirs(dest, exist_ok=True); tmp = os.path.join(dest, 'node.tar.xz')
    log(f'downloading Node from {NODE_URL}'); urllib.request.urlretrieve(NODE_URL, tmp)
    with tarfile.open(tmp) as t:
        for m in t.getmembers():
            parts = m.name.split('/', 1)
            if len(parts) < 2 or '..' in parts[1].split('/') or not (m.isfile() or m.isdir() or m.issym()): continue
            m.name = parts[1]; t.extract(m, dest)
    os.remove(tmp); return os.path.join(dest, 'bin', 'node')

def smolvlm_files(cache_dir=None):
    """The base model's processor/tokenizer configs and the published ONNX files the export reuses (fp32 vision encoder for
    our q8, fp16 embed_tokens): a local snapshot, used as SMOLVLM_DIR."""
    from huggingface_hub import snapshot_download
    return snapshot_download(BASE, cache_dir=cache_dir, allow_patterns=['*.json', '*.jinja', 'onnx/vision_encoder.onnx', 'onnx/embed_tokens_fp16.onnx'])

def save_json(path, obj):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True); tmp = f'{path}.tmp'
    with open(tmp, 'w') as f: json.dump(obj, f, indent=1)
    os.replace(tmp, path)

def load_json(path, default=None):
    try:
        with open(path) as f: return json.load(f)
    except (OSError, ValueError): return default
