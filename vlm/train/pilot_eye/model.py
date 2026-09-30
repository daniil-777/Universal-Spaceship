"""vlm/train/pilot_eye/model.py — Pilot Eye (spec §10.1): MobileNetV4-Conv-S to forward_features ([960, 3, 5] at 160x96), a
1x1 conv to 64 channels (the per-frame cache), then heads on [m_t, m_t - m_t-1, m_t-1 - m_t-2] pooled after differencing
(avg + max = 128 each) with the two spacings / nominal: 386 -> 512 -> heads. Loss weights: verdict 2, actions 1, rest 0.5."""
import os
os.environ.setdefault('HF_HOME', '/Volumes/LaCie/astro-pilot/vlm/hf')  # timm's pretrained weights are cached on LaCie, never the Mac disk
import timm, timm.data, torch
from torch import nn
from torch.nn import functional as F
HEADS = {'verdict': 3, 'severity': 5, 'reasons': 26, 'actions': 11, 'p_ref': 1, 'reg': 6, 'tags': 10, 'range': 5}
BACKBONE = 'mobilenetv4_conv_small'

class Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(BACKBONE, pretrained=pretrained, num_classes=0)
        cfg = timm.data.resolve_data_config({}, model=self.backbone)
        self.register_buffer('mean', torch.tensor(cfg['mean']).view(1, 3, 1, 1)); self.register_buffer('std', torch.tensor(cfg['std']).view(1, 3, 1, 1))
        self.reduce = nn.Conv2d(960, 64, 1)
    def forward(self, pixels):
        return self.reduce(self.backbone.forward_features((pixels - self.mean) / self.std))

class Heads(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(386, 512), nn.GELU(), nn.Dropout(0.1))
        self.out = nn.ModuleDict({k: nn.Linear(512, n) for k, n in HEADS.items()})
    def forward(self, m0, m1, m2, dt):
        pool = lambda x: torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1)
        h = self.mlp(torch.cat([pool(m2), pool(m2 - m1), pool(m1 - m0), dt], 1))
        return tuple(self.out[k](h) for k in HEADS)

class PilotEye(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__(); self.enc, self.heads = Encoder(pretrained), Heads()
    def forward(self, frames, dt):
        B = frames.shape[0]; m = self.enc(frames.flatten(0, 1)); m = m.view(B, 3, *m.shape[1:])
        return self.heads(m[:, 0], m[:, 1], m[:, 2], dt)

def _masked(loss, mask):
    return (loss * mask).sum() / mask.sum().clamp(min=1)

def loss_fn(out, T, M):
    v, sev, rea, act, pref, reg, tags, rng = out
    L = 2.0 * _masked(F.cross_entropy(v, T['verdict'], reduction='none'), M['verdict'])
    L = L + 0.5 * _masked(F.cross_entropy(sev, T['severity'], reduction='none'), M['severity'])
    L = L + 0.5 * _masked(F.binary_cross_entropy_with_logits(rea, T['reasons'], reduction='none'), M['reasons'])
    L = L + 1.0 * _masked(F.binary_cross_entropy_with_logits(act, T['actions'], reduction='none'), M['actions'])
    L = L + 0.5 * _masked(F.binary_cross_entropy_with_logits(pref[:, 0], T['p_ref'], reduction='none'), M['p_ref'])
    L = L + 0.5 * _masked(F.huber_loss(reg, T['reg'], reduction='none'), M['reg'])
    L = L + 0.5 * _masked(F.binary_cross_entropy_with_logits(tags, T['tags'], reduction='none'), M['tags'][:, None].expand_as(tags))
    return L + 0.5 * _masked(F.cross_entropy(rng, T['range'], reduction='none'), M['range'])
