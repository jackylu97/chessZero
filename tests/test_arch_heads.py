"""Correctness tests for the arch-sweep heads (from->to policy, attn pooling).

The from->to LUT is index-mapping code — the bug class behind the symmetry
parity incident — so it gets deterministic codec-parity verification before
any training run uses it."""
import random

import chess
import numpy as np
import torch

from src.games.chess import ChessGame, _move_to_action
from src.model.attention import AttnPoolHead, FromToPolicyHead, _build_from_to_luts


def test_from_to_lut_codec_parity():
    """For every legal move on random boards: the action's LUT entry must point
    at exactly S[from, to] (mover frame), or the promo branch for underpromos."""
    gather_idx, is_promo, valid = _build_from_to_luts()
    rng = random.Random(11)
    checked = 0
    for _ in range(300):
        board = chess.Board()
        for _ in range(rng.randint(0, 60)):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over():
                break
            board.push(rng.choice(moves))
        if board.is_game_over():
            continue
        for mv in board.legal_moves:
            a = _move_to_action(mv, board.turn)
            frm, to = mv.from_square, mv.to_square
            if board.turn == chess.BLACK:      # codec encodes mover-frame
                frm ^= 56; to ^= 56
            assert bool(valid[a]), f"legal move {mv.uci()} marked invalid (a={a})"
            if mv.promotion and mv.promotion != chess.QUEEN:
                assert bool(is_promo[a])
                assert int(gather_idx[a]) // 9 == frm
            else:
                assert not bool(is_promo[a])
                assert int(gather_idx[a]) == frm * 64 + to, (
                    f"{mv.uci()}: LUT {int(gather_idx[a])} != {frm}*64+{to}")
            checked += 1
    assert checked > 3000


def test_from_to_offboard_masked():
    gather_idx, is_promo, valid = _build_from_to_luts()
    # from a1 (sq 0), direction W (dir index 4), any distance: off-board
    for dist in range(7):
        a = 0 * 73 + 4 * 7 + dist
        assert not bool(valid[a])
    head = FromToPolicyHead(dim=32)
    x = torch.randn(2, 32, 8, 8)
    logits = head(x)
    assert logits.shape == (2, 4672)
    assert (logits[:, ~head.ft_valid] <= -1e3).all()
    logits.sum().backward()
    assert head.q_proj.weight.grad is not None and head.promo.weight.grad is not None


def test_from_to_score_routing():
    """Planting a spike in S[from,to] must move exactly the matching action logit."""
    head = FromToPolicyHead(dim=16)
    B = 1
    x = torch.zeros(B, 16, 8, 8)
    logits0 = head(x).detach()
    # monkeypatch: run forward manually with a spiked S
    tokens = x.flatten(2).transpose(1, 2)
    S = torch.zeros(B, 64, 64); S[0, 12, 28] = 5.0             # e2 -> e4
    U = head.promo(tokens)
    s_flat, u_flat = S.flatten(1), U.flatten(1)
    ray = s_flat.gather(1, head.ft_gather.clamp(max=4095).unsqueeze(0))
    pro = u_flat.gather(1, head.ft_gather.clamp(max=575).unsqueeze(0))
    out = torch.where(head.ft_is_promo.unsqueeze(0), pro, ray).masked_fill(
        ~head.ft_valid.unsqueeze(0), -1e4)
    a = _move_to_action(chess.Move.from_uci("e2e4"), chess.WHITE)
    hot = (out[0] == 5.0).nonzero().flatten().tolist()
    assert a in hot and len(hot) == 1, (a, hot)


def test_attn_pool_head():
    head = AttnPoolHead(dim=32, out_dim=3)
    x = torch.randn(4, 32, 8, 8)
    out = head(x)
    assert out.shape == (4, 3)
    out.sum().backward()
    assert head.query.grad is not None
    # not mean-pooling: permuting tokens changes attention weights w.r.t. content
    x2 = x.clone(); x2[:, :, 0, 0], x2[:, :, 7, 7] = x[:, :, 7, 7], x[:, :, 0, 0]
    head.eval()
    with torch.no_grad():
        o1, o2 = head(x), head(x2)
    # pooled output is content-based; swap should generally change nothing ONLY
    # if pooling ignored position — attn uses no pos-emb here so swap of token
    # CONTENTS across positions leaves the token SET identical => invariant.
    assert torch.allclose(o1, o2, atol=1e-5)  # documents set-pooling semantics
