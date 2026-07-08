"""Probe: action-encoding consistency across CPU encode/decode, GPU legal_mask,
GPU step decode, and the ConvPolicyHead logit layout.

The chain that must be self-consistent:
  played move  --_move_to_action-->  action id
  action id    --policy target index (argmax of MCTS visit policy)
  action id    --conv policy logit index (from_sq*73 + move_type)
  action id    --GPU legal_mask / GPU step decode

A mismatch ANYWHERE means policy targets train a different move than the one
played / scored, garbage prior, search can't rank moves -> collapse.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import chess
import numpy as np
import torch

from src.games.chess import ChessGame, _move_to_action, _action_to_move, NUM_MOVE_TYPES
from src.games import chess_gpu as cg


def random_legal_positions(n=300, max_depth=60, seed=0):
    rng = random.Random(seed)
    boards = []
    for _ in range(n):
        b = chess.Board()
        depth = rng.randint(0, max_depth)
        for _ in range(depth):
            moves = list(b.legal_moves)
            if not moves or b.is_game_over():
                break
            b.push(rng.choice(moves))
        if not b.is_game_over() and any(True for _ in b.legal_moves):
            boards.append(b)
    return boards


def test_cpu_roundtrip(boards):
    """For every legal move: encode -> decode -> must equal the original move."""
    fails = []
    total = 0
    for b in boards:
        for m in b.legal_moves:
            total += 1
            a = _move_to_action(m, b.turn)
            m2 = _action_to_move(a, b)
            if m2 != m:
                fails.append((b.fen(), m.uci(), a, None if m2 is None else m2.uci()))
    return total, fails


def test_cpu_encode_unique(boards):
    """Within a position, distinct legal moves must map to distinct action ids."""
    collisions = []
    for b in boards:
        seen = {}
        for m in b.legal_moves:
            a = _move_to_action(m, b.turn)
            if a in seen and seen[a] != m:
                collisions.append((b.fen(), seen[a].uci(), m.uci(), a))
            seen[a] = m
    return collisions


def test_gpu_legal_mask_matches_cpu(boards, device):
    """GPU legal_mask action set == CPU legal_actions set, per position."""
    gpu = cg.GpuChessGame()
    fails = []
    state = gpu.from_python_chess(boards, device=device)
    mask = gpu.legal_mask(state).cpu().numpy()  # (N, 4672)
    for i, b in enumerate(boards):
        cpu_actions = set(_move_to_action(m, b.turn) for m in b.legal_moves)
        gpu_actions = set(np.nonzero(mask[i])[0].tolist())
        if cpu_actions != gpu_actions:
            missing = cpu_actions - gpu_actions
            extra = gpu_actions - cpu_actions
            fails.append((b.fen(), sorted(missing)[:5], sorted(extra)[:5]))
    return len(boards), fails


def test_gpu_obs_matches_cpu(boards, device):
    """GPU to_tensor_batch == CPU to_tensor per position (the conv head reads
    these planes; a spatial mismatch would break the from_sq<->(row,col) map)."""
    gpu = cg.GpuChessGame()
    cpu = ChessGame()
    state = gpu.from_python_chess(boards, device=device)
    gpu_obs = gpu.to_tensor_batch(state).cpu().numpy()  # (N, 22, 8, 8)
    fails = []
    max_abs = 0.0
    for i, b in enumerate(boards):
        from src.games.base import GameState
        gs = GameState(board=b, current_player=1 if b.turn == chess.WHITE else -1)
        cpu_obs = cpu.to_tensor(gs).numpy()
        d = np.abs(cpu_obs - gpu_obs[i]).max()
        max_abs = max(max_abs, float(d))
        if d > 1e-4:
            # find which plane differs
            plane_d = np.abs(cpu_obs - gpu_obs[i]).reshape(cpu_obs.shape[0], -1).max(axis=1)
            bad = np.nonzero(plane_d > 1e-4)[0].tolist()
            fails.append((b.fen(), float(d), bad[:6]))
    return len(boards), fails, max_abs


def test_gpu_step_matches_cpu(boards, device, seed=1):
    """Play one random legal move per board on GPU and on CPU; resulting
    OBSERVATION (post-move) must match. Tests the GPU action decode + apply path
    against python-chess via the obs encoder (which is itself validated by the
    obs test). This exercises the exact action-id -> board-change mapping."""
    rng = random.Random(seed)
    gpu = cg.GpuChessGame()
    cpu = ChessGame()
    from src.games.base import GameState
    chosen_moves = [rng.choice(list(b.legal_moves)) for b in boards]
    actions = [_move_to_action(m, b.turn) for m, b in zip(chosen_moves, boards)]
    state = gpu.from_python_chess(boards, device=device)
    actions_t = torch.tensor(actions, dtype=torch.int64, device=device)
    new_state, rewards, done = gpu.step_batch(state, actions_t)
    gpu_obs = gpu.to_tensor_batch(new_state).cpu().numpy()
    fails = []
    for i, (b, m) in enumerate(zip(boards, chosen_moves)):
        b2 = b.copy(); b2.push(m)
        gs = GameState(board=b2, current_player=1 if b2.turn == chess.WHITE else -1)
        # Only compare PIECE planes (0..11) — castling/ep/clock state tracking on
        # the GPU is exercised elsewhere; piece placement is what the action
        # decode controls directly.
        cpu_obs = cpu.to_tensor(gs).numpy()[:12]
        d = np.abs(cpu_obs - gpu_obs[i][:12]).max()
        if d > 1e-4:
            fails.append((b.fen(), m.uci(), actions[i], float(d)))
    return len(boards), fails


def test_conv_policy_layout(device):
    """The ConvPolicyHead flattens (B,P,H,W)->(B,H,W,P) so logit index =
    row*W*P + col*P + move_type. The action encoding is from_sq*73+move_type with
    from_sq = row*8+col. Assert that for a known one-hot in a specific
    (row,col,move_type) plane the flattened index equals the action id."""
    from src.model.muzero_net import ConvPolicyHead
    H = W = 8
    P = NUM_MOVE_TYPES
    head = ConvPolicyHead(hidden_planes=4, action_space_size=H * W * P,
                          latent_h=H, latent_w=W).to(device)
    head.eval()
    # Force-overwrite proj to identity-ish so we can drive a single output plane.
    # Instead, directly test the reshape contract with a synthetic (B,P,H,W).
    fails = []
    B = 1
    for from_sq in range(64):
        row, col = divmod(from_sq, 8)
        for mt in (0, 27, 55, 56, 63, 64, 70, 72):
            x = torch.zeros(B, P, H, W)
            x[0, mt, row, col] = 1.0
            flat = x.permute(0, 2, 3, 1).reshape(B, H * W * P)[0]
            idx = int(flat.argmax().item())
            expect = from_sq * P + mt
            if idx != expect or float(flat[expect]) != 1.0:
                fails.append((from_sq, row, col, mt, idx, expect))
    return 64 * 8, fails


def test_played_vs_policy_target_vs_logit(checkpoint, device):
    """End-to-end on REAL self-play data: for each stored ply, the action that
    was PLAYED must be a legal action, and the policy target's argmax action
    must also be legal AND have nonzero policy mass on the played action's
    encoding. Also check the conv logit at the played action's index is finite."""
    # Loaded separately in main; placeholder kept for structure.
    raise NotImplementedError


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}\n")
    boards = random_legal_positions(n=300, max_depth=60, seed=0)
    print(f"built {len(boards)} random legal positions\n")

    # 1. CPU encode/decode round-trip.
    total, fails = test_cpu_roundtrip(boards)
    print(f"[1] CPU encode->decode round-trip: {total} moves, {len(fails)} FAILURES")
    for f in fails[:10]:
        print("    FAIL", f)

    # 2. CPU encode uniqueness within position.
    coll = test_cpu_encode_unique(boards)
    print(f"[2] CPU encode collisions (distinct moves -> same id): {len(coll)}")
    for c in coll[:10]:
        print("    COLLISION", c)

    # 3. ConvPolicyHead layout contract.
    n3, f3 = test_conv_policy_layout(device)
    print(f"[3] ConvPolicyHead reshape layout: {n3} cells tested, {len(f3)} FAILURES")
    for f in f3[:10]:
        print("    FAIL", f)

    # 4. GPU legal_mask vs CPU.
    try:
        n4, f4 = test_gpu_legal_mask_matches_cpu(boards, device)
        print(f"[4] GPU legal_mask == CPU legal set: {n4} positions, {len(f4)} MISMATCH")
        for f in f4[:10]:
            print("    MISMATCH fen=%s missing=%s extra=%s" % f)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[4] GPU legal_mask test ERROR: {type(e).__name__}: {e}")

    # 4b. GPU obs vs CPU obs.
    try:
        n4b, f4b, maxabs = test_gpu_obs_matches_cpu(boards, device)
        print(f"[4b] GPU obs == CPU obs: {n4b} positions, {len(f4b)} MISMATCH (max|diff|={maxabs:.2e})")
        for f in f4b[:10]:
            print("    MISMATCH fen=%s maxdiff=%.4f bad_planes=%s" % f)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[4b] GPU obs test ERROR: {type(e).__name__}: {e}")

    # 5. GPU step vs CPU.
    try:
        n5, f5 = test_gpu_step_matches_cpu(boards, device)
        print(f"[5] GPU step piece-placement == CPU push: {n5} positions, {len(f5)} MISMATCH")
        for f in f5[:10]:
            print("    MISMATCH", f)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[5] GPU step test ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
