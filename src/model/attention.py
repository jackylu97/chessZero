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
