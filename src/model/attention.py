"""Self-attention board encoder for the MuZero representation network.

Adds the three things the conv backbone lacks for long-range endgame geometry:
  - LEARNED POSITIONAL EMBEDDINGS: one trainable vector per board square (attention is
    permutation-equivariant, so position must be re-injected; a conv carries it for free).
  - GLOBAL SELF-ATTENTION over the 64 square-tokens (a1<->h8 in one hop, vs the conv's
    ~7-hop propagation).
  - SMOLGEN (Lc0): a DATA-DEPENDENT additive bias on the attention logits, scaled down for
    a small net (shrunk bottleneck, few heads, shared final projection). Lets the model say
    "this diagonal is hot in this position" — the position-specific geometry endgames need.

In/out contract: (B, C, H, W) -> (B, C, H, W), so it drops into RepresentationNetwork in
place of the residual blocks while preserving the spatial latent the conv policy head needs.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.utils.checkpoint


class Smolgen(nn.Module):
    """Data-dependent attention-logit bias, (B, heads, N, N), added to QK^T pre-softmax.

    compress C->d_comp per token -> flatten -> d_bottle bottleneck -> per-head d_gen ->
    (shared) final projection d_gen -> N*N. The final projection is the expensive part
    (d_gen x N^2); it is shared across layers (passed in) to keep params small, and
    zero-initialised so smolgen starts as a no-op and ramps up during training.
    """

    def __init__(self, dim: int, n_tokens: int, n_heads: int,
                 d_comp: int = 16, d_bottle: int = 96, d_gen: int = 32,
                 final_proj: nn.Linear | None = None):
        super().__init__()
        self.n_tokens = n_tokens
        self.n_heads = n_heads
        self.d_comp = d_comp
        self.d_gen = d_gen
        self.compress = nn.Linear(dim, d_comp, bias=False)
        self.to_bottle = nn.Linear(n_tokens * d_comp, d_bottle)
        self.act = nn.SiLU()
        self.gen = nn.Linear(d_bottle, n_heads * d_gen)
        # Shared (across layers) final projection d_gen -> N*N. Zero-init -> starts as no-op.
        self.final = final_proj if final_proj is not None else nn.Linear(d_gen, n_tokens * n_tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, N, C)
        B, N, _ = x.shape
        c = self.compress(x).reshape(B, N * self.d_comp)          # (B, N*d_comp)
        b = self.act(self.to_bottle(c))                           # (B, d_bottle)
        g = self.gen(b).reshape(B, self.n_heads, self.d_gen)      # (B, heads, d_gen)
        bias = self.final(g)                                      # (B, heads, N*N)
        return bias.reshape(B, self.n_heads, N, N)


class MHSASmolgen(nn.Module):
    """Multi-head self-attention with an optional additive smolgen bias."""

    def __init__(self, dim: int, n_heads: int, smolgen: Smolgen | None = None):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.smolgen = smolgen

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N, C)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                         # (B, heads, N, head_dim)
        att = (q @ k.transpose(-2, -1)) * self.scale            # (B, heads, N, N)
        if self.smolgen is not None:
            att = att + self.smolgen(x)
        att = att.softmax(dim=-1)
        out = (att @ v).transpose(1, 2).reshape(B, N, C)        # (B, N, C)
        return self.proj(out)


class EncoderLayer(nn.Module):
    """Pre-LN transformer encoder layer (stable for small from-scratch nets)."""

    def __init__(self, dim: int, n_heads: int, d_ff: int, smolgen: Smolgen | None = None):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MHSASmolgen(dim, n_heads, smolgen)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, d_ff), nn.SiLU(), nn.Linear(d_ff, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class BoardAttentionEncoder(nn.Module):
    """Self-attention over the H*W board tokens with learned positional embeddings and
    (optional) smolgen. In/out: (B, C, H, W)."""

    def __init__(self, dim: int, h: int, w: int, n_layers: int = 4, n_heads: int = 4,
                 d_ff: int | None = None, use_smolgen: bool = True,
                 smolgen_d_gen: int = 32):
        super().__init__()
        self.h, self.w = h, w
        n = h * w
        d_ff = d_ff if d_ff is not None else 2 * dim
        self.pos = nn.Parameter(torch.zeros(1, n, dim))          # learned positional embeddings
        nn.init.trunc_normal_(self.pos, std=0.02)
        shared_final = None
        if use_smolgen:
            shared_final = nn.Linear(smolgen_d_gen, n * n)       # shared across layers
            # near-zero init: smolgen starts ~no-op (stable) but gradient flows from step 1
            nn.init.normal_(shared_final.weight, std=1e-3); nn.init.zeros_(shared_final.bias)
        self.layers = nn.ModuleList([
            EncoderLayer(
                dim, n_heads, d_ff,
                smolgen=(Smolgen(dim, n, n_heads, d_gen=smolgen_d_gen, final_proj=shared_final)
                         if use_smolgen else None),
            )
            for _ in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B, C, H, W)
        B, C, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)                        # (B, N, C)
        t = t + self.pos
        # Activation checkpointing (2026-07-06, XL scale test): with grad
        # enabled, recompute each layer in backward instead of storing its
        # activations — memory drops ~n_layers-fold on the attention stack for
        # ~25-30% extra training compute. EXACT same math (not an
        # approximation). Enabled per-encoder via .grad_checkpoint; inference
        # (no_grad) paths are unaffected. This is what fits the 24M-param XL's
        # batch-512 training on a 32GB card.
        use_ckpt = getattr(self, "grad_checkpoint", False) and torch.is_grad_enabled()
        for layer in self.layers:
            if use_ckpt:
                t = torch.utils.checkpoint.checkpoint(layer, t, use_reentrant=False)
            else:
                t = layer(t)
        t = self.ln_out(t)
        return t.transpose(1, 2).reshape(B, C, H, W)


class AttnPoolHead(nn.Module):
    """Attention-pooled scalar head (2026-07-07 arch sweep, arm D).

    Replaces the Conv1x1->flatten->MLP squeeze: a single LEARNED QUERY
    cross-attends over the 64 square-tokens, then an MLP maps the pooled
    vector to the output logits. Rationale: scalar judgments like "is this
    mate" are RELATIONAL (king, attackers, escape squares) — one global
    attention read is the natural aggregation, where a 1x1-conv squeeze must
    compress relations into per-square channels first.
    In: (B, C, H, W) latent. Out: (B, out_dim) logits.
    """

    def __init__(self, dim: int, out_dim: int, n_heads: int = 8, mlp_hidden: int | None = None):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ln = nn.LayerNorm(dim)
        h = mlp_hidden if mlp_hidden is not None else dim
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.ReLU(), nn.Linear(h, out_dim))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)                    # (B, 64, C)
        q = self.query.expand(B, -1, -1)
        pooled, _ = self.attn(q, tokens, tokens)                  # (B, 1, C)
        return self.mlp(self.ln(pooled.squeeze(1)))               # (B, out_dim)


def _build_from_to_luts(num_move_types: int = 73):
    """LUTs mapping the flat action space (from*73+mt) onto (from,to) bilinear
    scores + an underpromotion branch. Single source of truth: the codec
    displacement tables (mirrors _action_to_move exactly; same tables the
    symmetry module validates against python-chess).

    Returns (gather_idx [A], is_promo [A], valid [A]):
      - ray/knight action a with on-board destination: gather_idx[a] = from*64+to
      - underpromotion action: gather_idx[a] = from*9 + (mt-64), is_promo[a]=1
      - off-board geometry: valid[a]=0 (logit pinned to -1e4)
    """
    # Single source of truth for move geometry (same tables the symmetry
    # module validates against python-chess). Lazy import: model -> training
    # dependency only at LUT-build time, no cycle.
    from src.training.symmetry import _DIR_MAP, _KNIGHT
    A = 64 * num_move_types
    gather_idx = torch.zeros(A, dtype=torch.long)
    is_promo = torch.zeros(A, dtype=torch.bool)
    valid = torch.zeros(A, dtype=torch.bool)
    for frm in range(A // num_move_types * 0 + 64):
        fr, fc = divmod(frm, 8)
        for mt in range(num_move_types):
            a = frm * num_move_types + mt
            if mt >= 64:
                gather_idx[a] = frm * 9 + (mt - 64)
                is_promo[a] = True
                valid[a] = True
                continue
            if mt < 56:
                dr, dc = _DIR_MAP[mt // 7]
                dist = mt % 7 + 1
                dr, dc = dr * dist, dc * dist
            else:
                dr, dc = _KNIGHT[mt - 56]
            tr, tc = fr + dr, fc + dc
            if 0 <= tr < 8 and 0 <= tc < 8:
                gather_idx[a] = frm * 64 + (tr * 8 + tc)
                valid[a] = True
    return gather_idx, is_promo, valid


class FromToPolicyHead(nn.Module):
    """Relational (from->to) policy head, Lc0-transformer style (arch sweep, arm C).

    Move logits as bilinear scores between from-square and to-square token
    embeddings — a move IS a relation between two squares; the conv head must
    encode 'long-diagonal queen move' as stacked plane patterns instead.
    Underpromotions (mt 64-72) get a per-from-square linear branch.
    In: (B, C, H, W) latent. Out: (B, 4672) logits over the standard action space.
    """

    def __init__(self, dim: int, d_head: int = 64, num_move_types: int = 73):
        super().__init__()
        self.d_head = d_head
        self.q_proj = nn.Linear(dim, d_head)
        self.k_proj = nn.Linear(dim, d_head)
        self.promo = nn.Linear(dim, 9)
        nn.init.zeros_(self.promo.weight); nn.init.zeros_(self.promo.bias)
        gather_idx, is_promo, valid = _build_from_to_luts(num_move_types)
        self.register_buffer("ft_gather", gather_idx, persistent=False)
        self.register_buffer("ft_is_promo", is_promo, persistent=False)
        self.register_buffer("ft_valid", valid, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)                    # (B, 64, C)
        q = self.q_proj(tokens)                                   # (B, 64, dh) from-embeddings
        k = self.k_proj(tokens)                                   # (B, 64, dh) to-embeddings
        S = torch.bmm(q, k.transpose(1, 2)) / (self.d_head ** 0.5)  # (B, 64, 64)
        U = self.promo(tokens)                                    # (B, 64, 9)
        s_flat = S.flatten(1)                                     # (B, 4096)
        u_flat = U.flatten(1)                                     # (B, 576)
        ray = s_flat.gather(1, self.ft_gather.clamp(max=4095).unsqueeze(0).expand(B, -1))
        pro = u_flat.gather(1, self.ft_gather.clamp(max=575).unsqueeze(0).expand(B, -1))
        logits = torch.where(self.ft_is_promo.unsqueeze(0), pro, ray)
        return logits.masked_fill(~self.ft_valid.unsqueeze(0), -1e4)
