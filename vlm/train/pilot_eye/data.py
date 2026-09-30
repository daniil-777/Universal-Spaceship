"""vlm/train/pilot_eye/data.py — rows of pilot_eye/<split>.jsonl with their 3 frames read sequentially from the uint8 memmap
cache (160x96x3 per frame), spacings / nominal frame_dt, targets and masks as tensors, and the §10.1 augmentations (JPEG
60-95, +-10 % exposure and gamma, a small crop/scale shared by the 3 frames, sensor noise; no geometry change for zoom).
Zoom rows (one frame, no frame_dt_s) fill their 3-row cache slot with the same frame and get unit spacings; D records the
build excluded (axial < 0.5 m, ruling T10-g) are not in the JSONL, so only the cache index is used, never the PNGs."""
import io, json, random
import numpy as np, torch
from PIL import Image
from torch.utils.data import Dataset
from vlm.train.common import read_jsonl, group_half
NOMINAL = {'S': 0.2, 'A': 0.2, 'L': 0.25, 'D': 2.0}
REG = ['log_ttc', 'log_clear', 'agl', 'vs', 'closing', 'loc_dots']

def _nominal(root):
    """labels.json's nominal_frame_dt (build.mjs writes it), else the §10.1 constants."""
    try:
        with open(f'{root}/labels.json') as f: return {**NOMINAL, **(json.load(f).get('nominal_frame_dt') or {})}
    except (OSError, ValueError): return dict(NOMINAL)

class PilotEyeData(Dataset):
    def __init__(self, root, split, augment=False, limit=None, fold=None, H=96, W=160):
        self.rows = read_jsonl(f'{root}/pilot_eye/{split}.jsonl')
        if fold is not None: self.rows = [r for r in self.rows if group_half(r['group']) == fold]
        if limit: self.rows = self.rows[:limit]
        with open(f'{root}/cache/index.json') as f: idx = json.load(f)
        if idx.get('size') and list(idx['size']) != [W, H]:
            raise ValueError(f"cache is {idx['size'][0]}x{idx['size'][1]} but {W}x{H} was asked for: rebuild the dataset with APV_EYE_SIZE={W}x{H}")
        self.index, self.nominal = idx['index'][split], _nominal(root)
        # opened lazily per process: a pickled memmap is a full copy (DataLoader workers spawn on macOS), and an empty split's
        # cache file cannot be mapped at all; a split without rows never opens it
        self.mm_path, self.mm = f'{root}/cache/pilot_eye_{split}.u8', None
        self.augment, self.H, self.W = augment, H, W
    def __getstate__(self): d = dict(self.__dict__); d['mm'] = None; return d
    def _frames(self, s):
        if self.mm is None: self.mm = np.memmap(self.mm_path, dtype=np.uint8, mode='r').reshape(-1, self.H, self.W, 3)
        return np.array(self.mm[s:s + 3])
    def __len__(self): return len(self.rows)
    def _aug(self, x, zoom):
        q, e, g = random.randint(60, 95), random.uniform(0.9, 1.1), random.uniform(0.9, 1.1)
        out = []
        for f in x:
            buf = io.BytesIO(); Image.fromarray(f).save(buf, 'JPEG', quality=q); f = np.asarray(Image.open(buf)).astype(np.float32) / 255
            out.append(np.clip((f * e) ** g + np.random.normal(0, 0.01, f.shape), 0, 1))
        x = np.stack(out)
        if not zoom and random.random() < 0.5:
            s = random.uniform(0.9, 1.0); h, w = int(self.H * s), int(self.W * s); y0, x0 = random.randint(0, self.H - h), random.randint(0, self.W - w)
            x = np.stack([np.asarray(Image.fromarray((f[y0:y0 + h, x0:x0 + w] * 255).astype(np.uint8)).resize((self.W, self.H), Image.BILINEAR)).astype(np.float32) / 255 for f in x])
        return x
    def __getitem__(self, i):
        r = self.rows[i]; s = self.index[r['key']]; x = self._frames(s)
        x = self._aug(x, r['family'] == 'Z') if self.augment else x.astype(np.float32) / 255
        frames = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))).float()
        dt = torch.tensor([d / self.nominal[r['family']] for d in r['frame_dt_s']] if r['frame_dt_s'] else [1.0, 1.0], dtype=torch.float32)
        t, m = r['targets'], r['masks']
        T = {'verdict': torch.tensor(t.get('verdict', 0)), 'severity': torch.tensor(t.get('severity', 0)), 'reasons': torch.tensor(t['reasons'] or [0] * 26, dtype=torch.float32), 'actions': torch.tensor(t['safe_actions'] or [0] * 11, dtype=torch.float32),
             'p_ref': torch.tensor(float(t.get('p_ref', 0))), 'reg': torch.tensor([float(t[k]) for k in REG]), 'tags': torch.tensor(t['zoom_tags'] or [0] * 10, dtype=torch.float32), 'range': torch.tensor(t['range_bin'])}
        M = {'verdict': torch.tensor(float(m['verdict'])), 'severity': torch.tensor(float(m['severity'])), 'reasons': torch.tensor(m['reasons'], dtype=torch.float32), 'actions': torch.tensor(m['actions'], dtype=torch.float32),
             'p_ref': torch.tensor(float(m['p_ref'])), 'reg': torch.tensor(m['reg'], dtype=torch.float32), 'tags': torch.tensor(float(m['zoom_tags'])), 'range': torch.tensor(float(m['range_bin']))}
        return frames, dt, T, M, r['key']
