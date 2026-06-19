"""Benchmark the GPU-resident MCTS search (run_batch_gpu) — the self-play bottleneck.

Times one full MCTS search (num_simulations) over a batch of N parallel games, for
compile_net OFF vs ON and across num_parallel_games. The search work is position-
independent, so a synthetic observation gives faithful timing. Also reports a net-output
parity check (eager vs compiled initial/recurrent inference) so we know the compile
didn't change the math.

Run (kill/pause other GPU jobs first for clean numbers):
  .venv/bin/python scripts/bench_selfplay_mcts.py --game chess_small --n-list 256,512,1024
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from src.config import get_config
from src.games.chess import ChessGame
from src.model.muzero_net import MuZeroNetwork
from src.mcts.tensor_mcts import TensorMCTS

_DT = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def build_net(cfg, game, device):
    return MuZeroNetwork(
        observation_channels=game.num_planes * cfg.history_frames,
        action_space_size=game.action_space_size, hidden_planes=cfg.hidden_planes,
        num_blocks=cfg.num_residual_blocks, latent_h=cfg.latent_h, latent_w=cfg.latent_w,
        input_h=8, input_w=8, fc_hidden=cfg.fc_hidden, value_support_size=cfg.value_support_size,
        reward_support_size=cfg.reward_support_size, action_embed_dim=cfg.action_embed_dim,
        use_consistency_loss=getattr(cfg, "use_consistency_loss", False),
        proj_hid=cfg.proj_hid, proj_out=cfg.proj_out, pred_hid=cfg.pred_hid, pred_out=cfg.pred_out,
        use_scalar_transform=cfg.use_scalar_transform, value_target_scale=cfg.value_target_scale,
        value_head_type=cfg.value_head_type, draw_score=cfg.draw_score,
        value_head_init_std=getattr(cfg, "value_head_init_std", 0.0),
        use_inverse_dynamics_loss=getattr(cfg, "use_inverse_dynamics_loss", False),
        inverse_dynamics_hidden=getattr(cfg, "inverse_dynamics_hidden", 256),
        policy_head_type=cfg.policy_head_type, use_moves_left=cfg.use_moves_left,
        moves_left_support_size=getattr(cfg, "moves_left_support_size", 10),
    ).to(device).eval()


def make_mcts(net, game, cfg, device, compile_net):
    cfg.tensor_mcts_compile_net = compile_net
    amp = getattr(cfg, "tensor_mcts_amp_dtype", None)
    return TensorMCTS(
        net, game, cfg, device=device,
        hidden_dtype=_DT.get(getattr(cfg, "tensor_mcts_hidden_dtype", "float32"), torch.float32),
        select_backend=getattr(cfg, "tensor_mcts_select_backend", "compile"),
        use_subtree_reuse=getattr(cfg, "tensor_mcts_subtree_reuse", False),
        amp_dtype=_DT.get(amp) if amp else None,
    )


def time_search(mcts, obs, legal, iters):
    for _ in range(3):  # warmup (compile happens here)
        mcts.run_batch_gpu(obs, legal, add_noise=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mcts.run_batch_gpu(obs, legal, add_noise=False)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="chess_small")
    ap.add_argument("--n-list", default="256,512,1024", help="num_parallel_games values to sweep")
    ap.add_argument("--sims", type=int, default=None, help="override num_simulations")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--profile-n", type=int, default=0,
                    help="if >0: build compiled+Triton MCTS at this N and loop searches for "
                         "--profile-seconds (so an external nvidia-smi/top can sample SM%/CPU%).")
    ap.add_argument("--profile-seconds", type=float, default=30.0)
    args = ap.parse_args()

    cfg = get_config(args.game); cfg.device = args.device
    if args.sims:
        cfg.num_simulations = args.sims
    game = ChessGame()
    dev = args.device
    obs_c = game.num_planes * cfg.history_frames
    asz = game.action_space_size

    torch.manual_seed(0)
    net = build_net(cfg, game, dev)
    print(f"game={args.game}  sims={cfg.num_simulations}  select_backend="
          f"{getattr(cfg,'tensor_mcts_select_backend','compile')}  "
          f"moves_left={cfg.use_moves_left}  params={sum(p.numel() for p in net.parameters()):,}\n")

    # Net-output parity: eager vs compiled initial/recurrent inference on a fixed input.
    with torch.no_grad():
        o = torch.randn(64, obs_c, 8, 8, device=dev)
        h, pl, v = net.initial_inference(o)
        cf = torch.compile(net.initial_inference, mode="default", dynamic=False, fullgraph=False)
        h2, pl2, v2 = cf(o)
        print(f"parity initial_inference: max|Δhidden|={(h-h2).abs().max():.2e}  "
              f"max|Δpolicy|={(pl-pl2).abs().max():.2e}  max|Δvalue|={(v-v2).abs().max():.2e}\n")

    if args.profile_n:
        N = args.profile_n
        obs = torch.randn(N, obs_c, 8, 8, device=dev)
        legal = torch.ones(N, asz, dtype=torch.bool, device=dev)
        m = make_mcts(net, game, cfg, dev, compile_net=True)
        for _ in range(3):
            m.run_batch_gpu(obs, legal, add_noise=False)
        torch.cuda.synchronize()
        print(f"\nPROFILE N={N} compiled+Triton (select_backend="
              f"{getattr(cfg,'tensor_mcts_select_backend','?')}); looping {args.profile_seconds}s — "
              f"sample SM%/CPU% now", flush=True)
        t0 = time.perf_counter(); it = 0
        while time.perf_counter() - t0 < args.profile_seconds:
            m.run_batch_gpu(obs, legal, add_noise=False)
            it += 1
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"PROFILE done: {it} searches, {dt/it*1000:.1f} ms/search, {N*it/dt:.0f} games/s",
              flush=True)
        return

    print(f"{'N':>6} {'eager ms':>10} {'compiled ms':>12} {'speedup':>8} "
          f"{'eager g/s':>11} {'compiled g/s':>13}")
    print("  " + "-" * 64)
    for N in [int(x) for x in args.n_list.split(",") if x.strip()]:
        try:
            obs = torch.randn(N, obs_c, 8, 8, device=dev)
            legal = torch.ones(N, asz, dtype=torch.bool, device=dev)
            m_eager = make_mcts(net, game, cfg, dev, compile_net=False)
            te = time_search(m_eager, obs, legal, args.iters)
            del m_eager; torch.cuda.empty_cache()
            m_comp = make_mcts(net, game, cfg, dev, compile_net=True)
            tc = time_search(m_comp, obs, legal, args.iters)
            del m_comp; torch.cuda.empty_cache()
            print(f"{N:>6} {te*1000:>10.1f} {tc*1000:>12.1f} {te/tc:>7.2f}x "
                  f"{N/te:>11.0f} {N/tc:>13.0f}", flush=True)
        except RuntimeError as e:
            print(f"{N:>6}   ERROR: {str(e)[:80]}", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
