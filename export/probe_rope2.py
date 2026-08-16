import sys
sys.path.insert(0, "F:/Seedvr2/ComfyUI-SeedVR2_VideoUpscaler/src")
import torch, numpy as np
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

# mm variant: dim=64//3=21, freqs_for="lang", theta=10000
rope = RotaryEmbedding(dim=21, freqs_for="lang", theta=10000)
T, Hh, Ww = 2, 4, 4
freqs = rope.get_axial_freqs(T, Hh, Ww)   # expected (T,H,W,21)
print("freqs shape", tuple(freqs.shape))

# x as the mm forward does: rearrange "L h d -> h L d", so x shape (h, L, d)
h, d = 20, 128
L = T * Hh * Ww
x = torch.randn(h, L, d)
xr = apply_rotary_emb(freqs, x)   # freqs (T,H,W,21), x (h,L,d)
# which channels of x changed?
diff = (xr - x).abs()
changed = (diff > 1e-5).any(dim=0).any(dim=0)  # over (h,L)
idx = torch.where(changed)[0].tolist()
print("changed channel count:", changed.sum().item())
print("first changed idx:", idx[:6], "last:", idx[-6:])
print("max abs diff on changed:", diff[:, :, idx[0]].max().item())

# Test standard formula: rotate first 2*rot_dim=42 channels, rest unchanged
rot_dim = 21
def rotate_half(y):
    half = 2 * rot_dim
    return torch.cat([-y[..., half:], y[..., :half]], dim=-1)
cosf = freqs.cos(); sinf = freqs.sin()   # (T,H,W,21)
# need freqs reshaped to (L, 21) in (t,h,w) order matching x's L dim
freqs_l = freqs.reshape(L, rot_dim)       # (L,21), row-major (t fastest? C-order: W fastest)
xf = x.clone()
xf[..., :2 * rot_dim] = x[..., :2 * rot_dim] * cosf.unsqueeze(0) + rotate_half(x[..., :2 * rot_dim]) * sinf.unsqueeze(0)
err = (xf - xr).abs().max().item()
print("MY FORMULA max err vs apply_rotary_emb:", err)
