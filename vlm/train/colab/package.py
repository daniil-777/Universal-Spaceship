"""vlm/train/colab/package.py — apv-models-v1.zip: pilot-eye-v1/ (the ONNX pair, labels, parity, metrics, card),
narrator-v1/ (the narrator-v0 folder layout: onnx/{decoder_model_merged_q4f16, embed_tokens_fp16, vision_encoder_quantized},
tokenizer and configs, parity), metrics/ (every stage's summary) and MANIFEST.json (SHA-256 and size of every file, the gate
results, the dataset id), which install_models.mjs verifies before it switches the web app to the new models. The fp32
parity-reference decoder (about 540 MB, never loaded by the browser) stays on Drive unless include_fp32 is set."""
import hashlib, json, os, time, zipfile

STORED = ('.onnx', '.png', '.f32', '.safetensors', '.zip')
REQUIRED = {'pilot-eye-v1': ['encoder.onnx', 'heads.onnx', 'labels.json', 'parity.json', 'parity_sample.json'],
            'narrator-v1': ['onnx/decoder_model_merged_q4f16.onnx', 'onnx/embed_tokens_fp16.onnx', 'onnx/vision_encoder_quantized.onnx', 'config.json', 'tokenizer.json',
                            'tokenizer_config.json', 'chat_template.json', 'preprocessor_config.json', 'processor_config.json', 'generation_config.json']}

def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()

def files_of(folder, include_fp32=False):
    out = []
    for d, _, fs in os.walk(folder):
        for f in sorted(fs):
            rel = os.path.relpath(os.path.join(d, f), folder)
            if f.startswith('._') or f.endswith(('.tmp', '.part')) or (rel == os.path.join('onnx', 'decoder_model_merged.onnx') and not include_fp32): continue
            out.append(rel)
    return sorted(out)

def make_zip(zip_path, eye_dir, narrator_dir, metrics, dataset_id=None, include_fp32=False, log=print):
    """-> the MANIFEST dict; the zip is written next to zip_path first and renamed, so a half-written zip never looks done."""
    for name, folder in (('pilot-eye-v1', eye_dir), ('narrator-v1', narrator_dir)):
        miss = [f for f in REQUIRED[name] if not os.path.exists(os.path.join(folder, f))]
        if miss: raise FileNotFoundError(f'{folder} lacks {miss}: an export step did not finish')
    man = {'format': 'apv-models/1', 'created': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'dataset': dataset_id, 'eye': 'pilot-eye-v1', 'narrator': 'narrator-v1',
           'metrics': metrics, 'files': {}}
    tmp = zip_path + '.part'
    with zipfile.ZipFile(tmp, 'w', allowZip64=True) as z:
        for name, folder in (('pilot-eye-v1', eye_dir), ('narrator-v1', narrator_dir)):
            for rel in files_of(folder, include_fp32):
                p = os.path.join(folder, rel); arc = f'{name}/{rel}'.replace(os.sep, '/')
                man['files'][arc] = {'sha256': _sha(p), 'bytes': os.path.getsize(p)}
                z.write(p, arc, compress_type=zipfile.ZIP_STORED if rel.endswith(STORED) else zipfile.ZIP_DEFLATED)
        for k, v in metrics.items(): z.writestr(f'metrics/{k}.json', json.dumps(v, indent=1, default=str))
        z.writestr('MANIFEST.json', json.dumps(man, indent=1, default=str))
    os.replace(tmp, zip_path)
    mb = os.path.getsize(zip_path) / 1e6; log(f'wrote {zip_path} ({mb:.0f} MB, {len(man["files"])} model files)')
    return man

INSTALL = """
Download {zip} from Google Drive (My Drive > {rel}), then on the Mac, in the astro-pilot worktree:

    node vlm/tools/install_models.mjs ~/Downloads/{name}

It unpacks to /Volumes/LaCie/astro-pilot/vlm/models/{{pilot-eye-v1,narrator-v1}}, checks every file's SHA-256, re-runs the
Pilot Eye parity sample in Node, checks the Narrator files and configs, and writes models/current.json. Then

    node vlm/web/serve.mjs --port 8795

and open http://127.0.0.1:8795/ (the site's Narrator card) or http://127.0.0.1:8795/vlm/web/demo.html: both read
/__vlm/models/current.json, so they use the v1 models (and fall back to v0 if it is missing).
"""
def instructions(zip_path, drive_root='/content/drive/MyDrive'):
    rel = os.path.relpath(zip_path, drive_root) if zip_path.startswith(drive_root) else zip_path
    return INSTALL.format(zip=os.path.basename(zip_path), rel=rel, name=os.path.basename(zip_path))
