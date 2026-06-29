"""Verify the 8-frame history stack in _stack_history is correctly aligned:
newest frame == observations[idx], frame t == observations[idx-t], zero-pad early.
Also verify obs[idx] is the board BEFORE actions[idx] by replaying through the engine
and comparing the reconstructed obs to the stored one. CPU only.
"""
import sys
import numpy as np
import torch

sys.path.insert(0, "/workspace/chessZero")
from src.config import get_config
from src.games.chess import ChessGame
from src.training.replay_buffer import ReplayBuffer

BUF = "/workspace/chessZero/checkpoints/chess/2026_06_19_cold2_pc/checkpoint_30000.buf"
cfg = get_config("chess_small"); cfg.device="cpu"
game = ChessGame(); A = game.action_space_size; NP = game.num_planes
buf = ReplayBuffer(max_size=10**9); buf.load(BUF, game=ChessGame())
g = buf.buffer[241] if buf.buffer[241].game_outcome != 0 else buf.buffer[0]
N = len(g.actions)
print(f"game N={N} num_planes={NP} history_frames={cfg.history_frames}")

# 1. _stack_history newest-first correctness at a mid-game idx
idx = min(20, N-1)
stk = g._stack_history(idx, cfg.history_frames)
print(f"\n_stack_history(idx={idx}, T={cfg.history_frames}) shape={tuple(stk.shape)} "
      f"(expect {NP*cfg.history_frames})")
ok = True
for t in range(cfg.history_frames):
    sl = stk[t*NP:(t+1)*NP]
    src_i = idx - t
    if src_i >= 0:
        match = torch.equal(sl, g.observations[src_i])
        print(f"  frame slot {t}: == observations[{src_i}]? {match}")
        ok &= match
    else:
        match = bool((sl == 0).all())
        print(f"  frame slot {t}: src idx {src_i}<0 -> all-zero? {match}")
        ok &= match
print(f"history stacking newest-first alignment: {'PASS' if ok else 'FAIL'}")

# 2. Reconstruct obs[idx] by replaying actions[0..idx-1] and compare (proves
#    observations[idx] is the board BEFORE actions[idx]).
print("\n=== obs-vs-engine reconstruction (obs[idx] is board BEFORE action[idx]) ===")
state = game.reset()
mism = 0
for t in range(min(N, 25)):
    recon = game.to_tensor(state)
    stored = g.observations[t]
    if not torch.allclose(recon.float(), stored.float(), atol=1e-4):
        mism += 1
        if mism <= 3:
            d = (recon.float()-stored.float()).abs().max().item()
            print(f"  ply {t}: MISMATCH maxdiff={d:.4f}")
    state,_,_ = game.step(state, g.actions[t])
print(f"  checked {min(N,25)} plies, mismatches={mism} "
      f"({'PASS' if mism==0 else 'FAIL — obs/action misaligned'})")
