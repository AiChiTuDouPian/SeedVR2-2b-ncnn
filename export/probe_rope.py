import sys
sys.path.insert(0, "F:/Seedvr2/ComfyUI-SeedVR2_VideoUpscaler/src")
import torch, numpy as np

from models.dit_3b.rope import NaMMRotaryEmbedding3d

rope = NaMMRotaryEmbedding3d(dim=64)  # rope_dim=64 (head_dim//2 default)
vid_shape = torch.tensor([[2, 4, 4]])  # t=2,h=4,w=4
txt_shape = torch.tensor([[3]])       # txt len 3
vf, tf = rope.get_freqs(vid_shape, txt_shape)
print("vid_freq shape", tuple(vf.shape), "txt_freq shape", tuple(tf.shape))
print("vf sample[0,:4]", vf[0, :4].tolist())

H = 20; D = 128
Lv = 2 * 4 * 4
vq = torch.randn(Lv, H, D); vk = torch.randn(Lv, H, D)
tq = torch.randn(3, H, D); tk = torch.randn(3, H, D)
vq2, vk2, tq2, tk2 = rope(vq, vk, vid_shape, tq, tk, txt_shape, None)
print("out vid_q shape", tuple(vq2.shape))
rot_dim = vf.shape[-1]
print("rot_dim", rot_dim)
print("real vq2[0,0,0:4]", vq2[0, 0, :4].tolist())
print("real vq2[0,0,42:46]", vq2[0, 0, 42:46].tolist())
print("real vq2[0,0,84:88]", vq2[0, 0, 84:88].tolist())
print("real vq2[0,0,126:128]", vq2[0, 0, 126:128].tolist())

# Try hypothesis: apply_rotary_emb uses freqs of shape (L, rot_dim), applied to x (L,H,D) by
# rotating the FIRST 2*rot_dim channels in pairs (cos/sin broadcast over H).
def rotate_half(x):
    half = 2 * rot_dim
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

cosf = vf.cos(); sinf = vf.sin()           # (L, rot_dim)
# broadcast to (L,H,rot_dim)
q_part = vq[..., :2 * rot_dim]             # (L,H,2*rot_dim)
q_rot = q_part * cosf.unsqueeze(1) + rotate_half(q_part) * sinf.unsqueeze(1)
# compare
err = (q_rot - vq2[..., :2 * rot_dim]).abs().max().item()
print("HYP nth-channel rotate, max err vs real (first 2*rot_dim):", err)
print("real  last chan vq2[0,0,124:128]:", vq2[0, 0, 124:128].tolist())
print("input last chan vq[0,0,124:128]:", vq[0, 0, 124:128].tolist())
