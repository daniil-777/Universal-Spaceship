"""vlm/train/narrator/merge.py — merge_and_unload, then assert that every vision_model, connector and embed_tokens tensor is
bit-identical to the base (SHA-256), so the published embed_tokens_fp16 and the vision encoders (and our q8 re-quantization
of the published fp32 encoder) are reused unchanged."""
import argparse, hashlib, os
# before transformers/huggingface_hub are imported (they read HF_HOME at import): downloads stay on LaCie
os.environ.setdefault('HF_HOME', '/Volumes/LaCie/astro-pilot/vlm/hf')
import torch

FROZEN = ('model.vision_model.', 'model.connector.', 'model.text_model.embed_tokens')
sha = lambda t: hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def frozen_hashes(state_dict): return {k: sha(v) for k, v in state_dict.items() if k.startswith(FROZEN)}

def frozen_mismatches(base, got):
    """Frozen tensor names missing on either side or whose SHA-256 differs; [] when the merge kept them bit-identical."""
    return sorted(set(base) ^ set(got)) + sorted(k for k in set(base) & set(got) if base[k] != got[k])

def main(argv=None):
    from peft import PeftModel
    from transformers import AutoModelForVision2Seq
    from vlm.train.narrator.export_decoder import BASE
    ap = argparse.ArgumentParser(); ap.add_argument('--adapter', required=True); ap.add_argument('--out', required=True); ap.add_argument('--base', default=BASE)
    a = ap.parse_args(argv)
    base = frozen_hashes(AutoModelForVision2Seq.from_pretrained(a.base, dtype=torch.float32).state_dict())
    m = PeftModel.from_pretrained(AutoModelForVision2Seq.from_pretrained(a.base, dtype=torch.float32), a.adapter).merge_and_unload()
    bad = frozen_mismatches(base, frozen_hashes(m.state_dict()))
    assert not bad and base, f'a frozen tensor changed in the merge: {bad[:5]}'
    m.save_pretrained(a.out, safe_serialization=True); print(f'merged -> {a.out}; {len(base)} frozen tensors identical', flush=True)

if __name__ == '__main__': main()
