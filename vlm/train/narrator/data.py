"""vlm/train/narrator/data.py — Narrator chat rows -> processor tensors (single 512² image, do_image_splitting False, the
processor loaded from the exported folder so training, parity and the browser share one config) with the prefix loss mask.
Every frame is first resized to 512² with Task 8's square(): the exported preprocessor_config.json has do_resize false."""
import torch
from PIL import Image
from torch.utils.data import Dataset
from vlm.train.common import read_jsonl, build_labels, check_row, IMAGE_IDS, LACIE, PAD
from vlm.train.narrator.export_decoder import square

IMG_KW = {'return_tensors': 'pt', 'images_kwargs': {'do_image_splitting': False}}

def frame_path(p):
    """Dataset frames are stored relative to the LaCie VLM root; absolute paths pass through."""
    return p if p.startswith('/') else f'{LACIE}/{p}'

def load_image(p): return square(Image.open(frame_path(p)))

class NarratorData(Dataset):
    def __init__(self, path, processor, q4_features=None, limit=None):
        self.rows, self.p, self.q4 = read_jsonl(path)[:limit], processor, q4_features
        self.special = set(processor.tokenizer.all_special_ids) | IMAGE_IDS
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r, p = self.rows[i], self.p; img = load_image(r['images'][0])
        full = p(text=p.apply_chat_template(r['messages']), images=[img], **IMG_KW)
        prompt = p(text=p.apply_chat_template(r['messages'][:1], add_generation_prompt=True), images=[img], **IMG_KW)
        ids = full['input_ids'][0].tolist(); labels = build_labels(ids, prompt['input_ids'][0].tolist()); check_row(ids, labels, self.special)
        item = {'input_ids': full['input_ids'][0], 'attention_mask': full['attention_mask'][0], 'labels': torch.tensor(labels)}
        if self.q4 is not None: item['image_hidden_states'] = torch.from_numpy(self.q4(full))[0]   # [64, 576] per single-image row
        else: item['pixel_values'] = full['pixel_values'][0]; item['pixel_attention_mask'] = full['pixel_attention_mask'][0]
        return item

def collate(batch):
    n = max(len(b['input_ids']) for b in batch); pad = lambda t, v: torch.cat([t, torch.full((n - len(t),), v, dtype=t.dtype)])
    out = {'input_ids': torch.stack([pad(b['input_ids'], PAD) for b in batch]), 'attention_mask': torch.stack([pad(b['attention_mask'], 0) for b in batch]),
           'labels': torch.stack([pad(b['labels'], -100) for b in batch])}
    for k in ('pixel_values', 'pixel_attention_mask', 'image_hidden_states'):
        if k in batch[0]: out[k] = torch.stack([b[k] for b in batch])
    return out
