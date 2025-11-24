# test_rope_bw200.py
import torch
import torch.nn.functional as F

B, N, S, H = 1, 32, 128, 128
q = torch.randn(B, N, S, H, device="cuda")
k = torch.randn(B, N, S, H, device="cuda")

# 手动实现简单 RoPE（避免 fused op）
def apply_rope(x):
    seq_len = x.shape[2]
    dim = x.shape[-1]
    theta = 10000.0
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
    t = torch.arange(seq_len, device=x.device).float()
    freqs = torch.einsum("n,d->nd", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(0).unsqueeze(0)
    sin = emb.sin().unsqueeze(0).unsqueeze(0)
    x_rot = torch.cat([-x[..., dim//2:], x[..., :dim//2]], dim=-1)
    return x * cos + x_rot * sin

q_rope = apply_rope(q)
k_rope = apply_rope(k)

# Test attention
attn = (q_rope @ k_rope.transpose(-2, -1)) / (H ** 0.5)
attn = F.softmax(attn, dim=-1)
print("RoPE + Attention success on BW200")
