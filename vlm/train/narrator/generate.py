"""vlm/train/narrator/generate.py — greedy outputs (eos 49279, max 90 new tokens) for evaluation rows {key, image, prompt}
-> {key, text, ended}. PyTorch fp32 with Task 8's vision_positions_fix; the processor from the exported folder; square() first."""
import argparse, json, os, time
# before transformers/huggingface_hub are imported (they read HF_HOME at import): downloads stay on LaCie
os.environ.setdefault('HF_HOME', '/Volumes/LaCie/astro-pilot/vlm/hf')
import torch
from vlm.train.common import read_jsonl, EOU
from vlm.train.narrator.data import IMG_KW, frame_path
from vlm.train.narrator.export_decoder import square

def encode_prompt(processor, prompt, image):
    text = processor.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}], add_generation_prompt=True)
    return processor(text=text, images=[square(image)], **IMG_KW)

def main(argv=None):
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from vlm.train.narrator.export_decoder import vision_positions_fix
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--processor-dir', required=True); ap.add_argument('--rows', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--max-new', type=int, default=90); ap.add_argument('--limit', type=int)
    ap.add_argument('--device', choices=('cpu', 'mps'), default='cpu'); a = ap.parse_args(argv)
    p = AutoProcessor.from_pretrained(a.processor_dir); m = vision_positions_fix(AutoModelForVision2Seq.from_pretrained(a.model, dtype=torch.float32)).eval().to(a.device)
    rows, t0 = read_jsonl(a.rows)[:a.limit], time.time()
    with open(a.out, 'w') as f:
        for i, r in enumerate(rows):
            inp = encode_prompt(p, r['prompt'], Image.open(frame_path(r['image']))).to(a.device)
            with torch.no_grad(): ids = m.generate(**inp, max_new_tokens=a.max_new, do_sample=False, eos_token_id=EOU)
            new = ids[0, inp['input_ids'].shape[1]:]
            f.write(json.dumps({'key': r['key'], 'text': p.tokenizer.decode(new, skip_special_tokens=True).strip(), 'ended': bool(len(new)) and int(new[-1]) == EOU}) + '\n'); f.flush()
            if (i + 1) % 10 == 0: print(f'{i + 1}/{len(rows)} {time.time() - t0:.0f} s', flush=True)
    print(f'wrote {len(rows)} outputs -> {a.out}', flush=True)

if __name__ == '__main__': main()
