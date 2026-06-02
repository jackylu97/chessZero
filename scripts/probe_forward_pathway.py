"""Is the FORWARD action pathway strong, or is the inverse head a crutch?

Gives the dynamics the strongest possible discriminative forward signal — train
dynamics(h, a_i) toward MUTUALLY-ORTHOGONAL target directions, one per action,
with a scale-free cosine loss and NO inverse loss — and measures how low the
cross-action cosine can go. cos→0 means the pathway can fully separate actions
(strong); a high plateau means the pathway is bottlenecked (anemic).

Ablates the architectural suspects the critique names:
  - residual skip (x + h re-injects the full state → identity bias)
  - final min-max normalize (per-channel; can wash out a per-channel offset)
  - spatially-uniform action broadcast (vs a distinct spatial pattern per action)
  - embedding width (16 vs 256)

CPU, batched over actions (one fwd/bwd per step) → fast.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F
from src.model.muzero_net import DynamicsNetwork, _min_max_normalize

torch.manual_seed(0)
DEV = "cpu"
C, H, W, A = 96, 8, 8, 4672      # 96ch: action is 16/(96+16)=14% of conv_in (≈ the 128ch real ratio 11%)
N = 12                            # actions to separate
STEPS = 500

h = _min_max_normalize(torch.rand(1, C, H, W)).to(DEV)
h_batch = h.expand(N, C, H, W).contiguous()
acts = torch.arange(N, device=DEV)
# mutually-orthogonal unit target directions — the strongest discriminative signal.
T = torch.linalg.qr(torch.randn(C * H * W, N))[0].T.to(DEV)   # (N, C*H*W) orthonormal rows


def make_dyn(embed_dim=16, spatial=False):
    torch.manual_seed(0)
    d = DynamicsNetwork(hidden_planes=C, num_blocks=8, action_space_size=A,
                        latent_h=H, latent_w=W, fc_hidden=128, action_embed_dim=embed_dim).to(DEV)
    d._spatial = spatial
    d._ed = embed_dim
    if spatial:  # distinct spatial pattern per action instead of a uniform broadcast
        d.action_embedding = nn.Embedding(A, embed_dim * H * W).to(DEV)
    return d


def fwd(d, hh, a, residual=True, norm=True):
    b = hh.shape[0]
    if d._spatial:
        ap = d.action_embedding(a.long()).view(b, d._ed, H, W)
    else:
        ap = d.action_embedding(a.long()).view(b, d._ed, 1, 1).expand(b, d._ed, H, W)
    x = torch.cat([hh, ap], dim=1)
    x = d.conv_in(x); x = d.bn_in(x)
    x = F.relu(x + hh) if residual else F.relu(x)
    x = d.blocks(x)
    if norm:
        x = _min_max_normalize(x)
    return x


def run(label, embed_dim=16, spatial=False, residual=True, norm=True):
    d = make_dyn(embed_dim, spatial)
    opt = torch.optim.Adam(d.parameters(), lr=1e-3)
    for _ in range(STEPS):
        opt.zero_grad()
        out = fwd(d, h_batch, acts, residual, norm).reshape(N, -1)
        loss = (1.0 - F.cosine_similarity(out, T, dim=-1)).mean()   # align each action's output to its target dir
        loss.backward(); opt.step()
    with torch.no_grad():
        out = fwd(d, h_batch, acts, residual, norm).reshape(N, -1)
        align = F.cosine_similarity(out, T, dim=-1).mean().item()    # how well it hit the targets (1=perfect)
        on = F.normalize(out, dim=-1)
        xa = (on @ on.T)[~torch.eye(N, dtype=bool, device=DEV)].mean().item()  # cross-action cos (0=fully separated)
    print(f"  {label:42s} cross-action cos={xa:.3f}  target-align={align:.3f}")


print(f"Targets are ORTHOGONAL (ideal cross-action cos=0). N={N} actions, {C}ch x 8 blocks, {STEPS} steps.\n")
print("If 'current' can't reach low cos but an ablation can, the pathway is the bottleneck (inverse = crutch).")
print("If 'current' already reaches low cos, the pathway is strong (inverse = genuine regularizer).\n")
run("current (16, uniform, +residual, +norm)")
run("  - no residual skip", residual=False)
run("  - no min-max-norm", norm=False)
run("  - embed=256", embed_dim=256)
run("  - SPATIAL action planes (16)", spatial=True)
run("  - SPATIAL + no-residual + no-norm (256)", embed_dim=256, spatial=True, residual=False, norm=False)
