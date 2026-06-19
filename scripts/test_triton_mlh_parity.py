"""Thorough parity test: Triton PUCT kernel (with the new moves-left utility)
vs the reference torch ``_select`` path.

The MLH term affects ONLY selection, so we compare the selection walk directly:
populate a TensorMCTS tree with random-but-valid data that exercises the MLH gate
(|Q|>threshold on expanded children with varied moves_left), then run torch
``_select`` and Triton ``_select_triton`` on the SAME tree and assert the outputs
(path_node_idx, path_slot, leaf_parent_node, leaf_slot, leaf_action, depth) are
BIT-EXACT. Three checks per seed:
  1. MLH ON : torch == triton   (the port is correct)
  2. MLH OFF: torch == triton   (regression — the pre-existing kernel still matches)
  3. coverage: MLH ON != MLH OFF in >=1 game (the gate actually fired)

Run: .venv/bin/python scripts/test_triton_mlh_parity.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from src.config import get_config
from src.games.chess import ChessGame
from scripts.bench_selfplay_mcts import build_net
from src.mcts.tensor_mcts import TensorMCTS

FIELDS = ["path_node_idx", "path_slot", "leaf_parent_node", "leaf_slot", "leaf_action", "depth"]
TREE_TENSORS = ["node_visits", "node_moves_left", "child_priors", "child_visits",
                "child_value_sum", "child_rewards", "child_actions", "child_node_idx",
                "mm_min", "mm_max"]


def populate(mcts, N, M, K, asz, gen, dev):
    """Random-but-valid tree exercising the MLH gate."""
    mcts.node_visits = torch.randint(0, 60, (N, M), dtype=torch.int32, device=dev, generator=gen)
    mcts.node_visits[:, 0] = torch.randint(20, 60, (N,), dtype=torch.int32, device=dev, generator=gen)
    mcts.node_moves_left = (torch.rand(N, M, generator=gen, device=dev) * 40.0)  # >=0
    mcts.child_priors = torch.softmax(torch.randn(N, M, K, generator=gen, device=dev), dim=-1)
    mcts.child_visits = torch.randint(0, 20, (N, M, K), dtype=torch.int32, device=dev, generator=gen)
    # value_sum so that value_sum/visits spans ~N(0,1); rewards span so raw_q straddles +-threshold
    mcts.child_value_sum = (torch.randn(N, M, K, generator=gen, device=dev)
                            * mcts.child_visits.clamp(min=1).float())
    mcts.child_rewards = torch.randn(N, M, K, generator=gen, device=dev) * 0.7
    acts = torch.randint(0, asz, (N, M, K), dtype=torch.int32, device=dev, generator=gen)
    acts[torch.rand(N, M, K, generator=gen, device=dev) < 0.2] = -1     # some unmaterialized slots
    mcts.child_actions = acts
    cidx = torch.randint(0, M, (N, M, K), dtype=torch.int32, device=dev, generator=gen)
    cidx[torch.rand(N, M, K, generator=gen, device=dev) < 0.5] = -1     # ~half unexpanded
    mcts.child_node_idx = cidx
    # MinMax: most games with a real range, a few degenerate (no range).
    lo = torch.randn(N, generator=gen, device=dev)
    mcts.mm_min = lo
    mcts.mm_max = lo + torch.rand(N, generator=gen, device=dev) + 0.3
    no_range = torch.rand(N, generator=gen, device=dev) < 0.15
    mcts.mm_max = torch.where(no_range, lo - 1.0, mcts.mm_max)          # mm_max<mm_min => no range
    mcts.mm_min = mcts.mm_min.float(); mcts.mm_max = mcts.mm_max.float()


def snapshot(mcts):
    return {k: getattr(mcts, k).clone() for k in TREE_TENSORS}


def restore(mcts, snap):
    for k, v in snap.items():
        setattr(mcts, k, v.clone())


def run_select(mcts, snap, use_ml):
    restore(mcts, snap)
    mcts._use_ml_utility = use_ml
    torch_out = [t.clone() for t in mcts._select()]
    restore(mcts, snap)
    mcts._use_ml_utility = use_ml
    triton_out = [t.clone() for t in mcts._select_triton()]
    return torch_out, triton_out


def main():
    dev = "cuda"
    cfg = get_config("chess_small"); cfg.device = dev
    game = ChessGame(); asz = game.action_space_size
    torch.manual_seed(0)
    net = build_net(cfg, game, dev)
    # eager + select_backend="eager" so self._select is the raw torch reference
    # (not compiled, not replaced by triton); we call _select / _select_triton directly.
    mcts = TensorMCTS(net, game, cfg, device=dev, compile=False, select_backend="eager")
    assert mcts._use_ml_utility, "moves-left must be enabled for this test"
    print(f"ml_threshold={mcts._ml_threshold} ml_slope={mcts._ml_slope} "
          f"ml_max_effect={mcts._ml_max_effect}  MAX_SELECT_DEPTH={mcts.MAX_SELECT_DEPTH}\n")

    import os as _os
    if _os.environ.get("PARITY_FULL"):
        configs = [(64, 200, "main"), (16, 100, "small"), (256, 200, "wide")]
        seeds = 12
    else:
        configs = [(64, 200, "main")]
        seeds = 8
    n_on_ok = n_off_ok = n_total = n_covered = 0
    worst = []

    for (N, sims, tag) in configs:
        mcts._allocate(N, sims, (cfg.hidden_planes, cfg.latent_h, cfg.latent_w))
        M, K = mcts._allocated_m, mcts.K
        for s in range(seeds):
            gen = torch.Generator(device=dev).manual_seed(1000 * (configs.index((N, sims, tag))) + s)
            populate(mcts, N, M, K, asz, gen, dev)
            snap = snapshot(mcts)

            t_on, tr_on = run_select(mcts, snap, use_ml=True)
            t_off, tr_off = run_select(mcts, snap, use_ml=False)

            on_ok = all(torch.equal(a, b) for a, b in zip(t_on, tr_on))
            off_ok = all(torch.equal(a, b) for a, b in zip(t_off, tr_off))
            # coverage: did MLH change any game's chosen leaf vs MLH-off (torch)?
            covered = not torch.equal(t_on[2], t_off[2]) or not torch.equal(t_on[3], t_off[3]) \
                or not torch.equal(t_on[4], t_off[4])

            n_total += 1
            n_on_ok += on_ok; n_off_ok += off_ok; n_covered += covered
            if not on_ok or not off_ok:
                diffs = {FIELDS[i]: int((t_on[i] != tr_on[i]).sum()) for i in range(6)}
                worst.append(f"  [{tag} N={N} seed={s}] MLH-on={'OK' if on_ok else 'FAIL'} "
                             f"MLH-off={'OK' if off_ok else 'FAIL'}  on-mismatch/field={diffs}")

    print(f"configs={[c[2] for c in configs]}  seeds/config={seeds}  total={n_total}\n")
    print(f"  MLH-ON  parity (torch==triton): {n_on_ok}/{n_total}")
    print(f"  MLH-OFF parity (regression):    {n_off_ok}/{n_total}")
    print(f"  coverage (MLH changed leaf):    {n_covered}/{n_total}")
    if worst:
        print("\n  MISMATCHES:")
        print("\n".join(worst[:20]))
    verdict = "PASS" if (n_on_ok == n_total and n_off_ok == n_total and n_covered > 0) else "FAIL"
    print(f"\n  >>> {verdict}")
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
