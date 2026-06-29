"""Path B correctness gate: storing policies sparse must not change targets.

1) round-trip: densify(sparsify(dense)) == dense, byte-exact, on real policies.
2) make_target equivalence: same policy targets whether a game stores policies
   sparse (new) or dense (densified copy), over many positions/unroll steps.
3) compact serialization round-trip: to_compact_dict -> from_compact_dict keeps
   densified policies identical.
"""
import sys, copy
import numpy as np
sys.path.insert(0, "/workspace/chessZero")
from src.games.chess import ChessGame
from src.config import get_config
from src.training.replay_buffer import (
    ReplayBuffer, GameHistory, _sparsify_policy, _densify_policy,
)

g = ChessGame(); c = get_config("chess_small"); A = g.action_space_size
BUF = "checkpoints/chess/2026_06_20_scale_gen3/checkpoint_1000.buf"

rb = ReplayBuffer(20000); rb.load(BUF, game=g)
games = [x for x in rb.buffer if x.policies][:40]
print(f"loaded {len(rb.buffer)} games; testing {len(games)}")

# Stored policies should now be sparse tuples (from_compact_dict, Path B).
n_tuple = sum(isinstance(p, tuple) for gm in games for p in gm.policies)
n_total = sum(len(gm.policies) for gm in games)
print(f"sparse-tuple policies: {n_tuple}/{n_total}")
assert n_tuple == n_total, "from_compact_dict did not store sparse"

# (1) round-trip exactness
rt_fail = 0
for gm in games:
    for p in gm.policies:
        dense = _densify_policy(p, A)
        re = _densify_policy(_sparsify_policy(dense), A)
        if not np.array_equal(dense, re):
            rt_fail += 1
print(f"(1) densify/sparsify round-trip mismatches: {rt_fail}")

# (2) make_target equivalence: sparse-stored vs dense-stored game
mt_kwargs = dict(
    value_head_type=getattr(c, "value_head_type", "support"),
    history_frames=getattr(c, "history_frames", 1),
    eval_to_wdl_alpha=c.eval_to_wdl_alpha, eval_to_wdl_beta=c.eval_to_wdl_beta,
    q_ratio=getattr(c, "q_ratio", 0.0), warmstart_q_ratio=getattr(c, "warmstart_q_ratio", 0.0),
    selfplay_q_ratio=c.selfplay_q_ratio, repetition_penalty=c.repetition_penalty,
    repetition_penalty_window=getattr(c, "repetition_penalty_window", 0),
    repetition_penalty_decay=getattr(c, "repetition_penalty_decay", 0.0),
)
mt_fail = 0; checked = 0
for gm in games:
    dense_gm = copy.copy(gm)
    dense_gm.policies = [_densify_policy(p, A) for p in gm.policies]  # dense copy
    for pos in range(0, max(1, len(gm) - 1), 7):
        s = gm.make_target(pos, c.num_unroll_steps, c.td_steps, c.discount, A, **mt_kwargs)
        d = dense_gm.make_target(pos, c.num_unroll_steps, c.td_steps, c.discount, A, **mt_kwargs)
        # index 5 = policies target list
        for ps, pd in zip(s[5], d[5]):
            checked += 1
            if not np.array_equal(np.asarray(ps), np.asarray(pd)):
                mt_fail += 1
print(f"(2) make_target policy-target mismatches: {mt_fail}/{checked}")

# (3) compact serialization round-trip
ser_fail = 0
for gm in games[:10]:
    d = gm.to_compact_dict()
    gm2 = GameHistory.from_compact_dict(d, ChessGame())
    for p1, p2 in zip(gm.policies, gm2.policies):
        if not np.array_equal(_densify_policy(p1, A), _densify_policy(p2, A)):
            ser_fail += 1
print(f"(3) compact round-trip mismatches: {ser_fail}")

ok = (rt_fail == 0 and mt_fail == 0 and ser_fail == 0)
print("RESULT:", "PASS ✓" if ok else "FAIL ✗")
sys.exit(0 if ok else 1)
