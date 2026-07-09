"""Direct endgame-conversion probe: can the model actually MATE forced wins?

Starts from canonical won positions (model = winning side, White) and makes the
model play them out vs a Stockfish DEFENDER (which can only delay mate in these
forced wins, not hold). Measures whether the model converts to checkmate within a
ply cap, and how many plies it takes. Compares across checkpoints to see whether
conversion ability itself improved (vs the self-play draw-rate, which is confounded
by the opponent also being weak).

Usage:
  .venv/bin/python scripts/probe_conversion.py --game chess_small --device cuda \
    --num-simulations 200 --sf-depth 8 --ply-cap 100 \
    --checkpoints checkpoints/chess/<run>/checkpoint_15000.pt \
                  checkpoints/chess/<run>/checkpoint_39000.pt
"""
import argparse, os, sys, time
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); os.chdir(REPO)
import torch, chess, chess.engine
from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame, _move_to_action
from src.model.muzero_net import MuZeroNetwork
from src.mcts.mcts import MCTS, select_action
from src.training.replay_buffer import stack_with_history

# Forced wins (White to move, White winning). SF defense can only delay mate.
POSITIONS = [
    ("KQ_v_K",    "7k/8/8/8/8/8/8/KQ6 w - - 0 1"),
    ("KR_v_K",    "7k/8/8/8/8/8/8/KR6 w - - 0 1"),
    ("KRR_v_K",   "7k/8/8/8/8/8/8/KR1R4 w - - 0 1"),
    ("KP_v_K",    "8/8/8/8/4P3/4K3/8/7k w - - 0 1"),          # white king escorts, black king far
    ("KQ_v_KR",   "7k/7r/8/8/8/8/8/KQ6 w - - 0 1"),           # up a queen for a rook
    ("Q_up_full", "rnb1kbnr/pppp1ppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),  # white up a queen
]


def load_network(ckpt_path, game, cfg, device):
    torch.serialization.add_safe_globals([MuZeroConfig])
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)["model_state_dict"]
    has_conv = any(".policy_head.mix." in k or ".policy_head.proj." in k for k in sd)
    has_ml = any(k.startswith("moves_left_head.") for k in sd)
    has_cons = any(k.startswith("projection.") for k in sd)
    has_inv = any(k.startswith("inverse_dynamics_head.") for k in sd)
    has_mat = any(k.startswith("material_head.") for k in sd)
    mat_support = getattr(cfg, "material_head_support_size", 8)
    if has_mat:
        outw = next((v for k, v in sd.items() if k.startswith("material_head.")
                     and v.ndim == 2 and v.shape[0] % 2 == 1 and v.shape[0] < v.shape[1] * 4), None)
        if outw is not None:
            mat_support = (outw.shape[0] - 1) // 2
    net = MuZeroNetwork(
        observation_channels=game.num_planes * getattr(cfg, "history_frames", 1),
        action_space_size=game.action_space_size, hidden_planes=cfg.hidden_planes,
        num_blocks=cfg.num_residual_blocks, latent_h=cfg.latent_h, latent_w=cfg.latent_w,
        input_h=game.board_size[0], input_w=game.board_size[1], fc_hidden=cfg.fc_hidden,
        value_support_size=cfg.value_support_size, reward_support_size=cfg.reward_support_size,
        action_embed_dim=getattr(cfg, "action_embed_dim", 16),
        use_consistency_loss=has_cons, proj_hid=cfg.proj_hid, proj_out=cfg.proj_out,
        pred_hid=cfg.pred_hid, pred_out=cfg.pred_out, use_scalar_transform=cfg.use_scalar_transform,
        value_target_scale=cfg.value_target_scale, value_head_type=getattr(cfg, "value_head_type", "support"),
        draw_score=getattr(cfg, "draw_score", 0.0),
        use_inverse_dynamics_loss=has_inv, inverse_dynamics_hidden=getattr(cfg, "inverse_dynamics_hidden", 256),
        policy_head_type="conv" if has_conv else "flat",
        use_moves_left=has_ml, moves_left_support_size=getattr(cfg, "moves_left_support_size", 10),
        use_material_head=has_mat, material_head_support_size=mat_support,
    )
    net.load_state_dict(sd); net.to(device); net.eval()
    return net


@torch.no_grad()
def convert_one(game, mcts, engine, fen, sf_limit, ply_cap):
    state = game.reset_from_fen(fen)
    frame_history = []
    hf = mcts.config.history_frames
    plies = 0
    while not state.done and plies < ply_cap:
        if state.board.turn == chess.WHITE:        # model = winning side
            obs = stack_with_history(game.to_tensor(state), frame_history, hf)
            root = mcts.run(obs, game.legal_actions(state), add_noise=False)
            action, _ = select_action(root, temperature=0)
        else:                                       # Stockfish defends
            action = _move_to_action(engine.play(state.board, sf_limit).move, state.board.turn)
        frame_history.append(game.to_tensor(state))
        state, _, _ = game.step(state, action)
        plies += 1
    if state.done and state.winner == 1:
        return "MATE", plies
    if state.done and state.winner == -1:
        return "LOST", plies        # model blundered the win
    return "no_conv", plies          # draw or ply cap -> failed to convert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--game", default="chess_small")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-simulations", type=int, default=200)
    ap.add_argument("--sf-depth", type=int, default=8)
    ap.add_argument("--ply-cap", type=int, default=100)
    ap.add_argument("--stockfish", default="tools/stockfish/stockfish")
    args = ap.parse_args()

    cfg = get_config(args.game); cfg.num_simulations = args.num_simulations; cfg.use_gumbel = False
    game = ChessGame()
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    sf_limit = chess.engine.Limit(depth=args.sf_depth)

    print(f"=== CONVERSION PROBE  sims={args.num_simulations} sf_depth={args.sf_depth} "
          f"ply_cap={args.ply_cap} ===\n", flush=True)
    summary = {}
    for ckpt in args.checkpoints:
        name = os.path.basename(ckpt)
        net = load_network(ckpt, game, cfg, args.device)
        mcts = MCTS(net, game, cfg, args.device)
        conv = 0
        print(f"[{name}]", flush=True)
        for pname, fen in POSITIONS:
            t0 = time.time()
            res, plies = convert_one(game, mcts, engine, fen, sf_limit, args.ply_cap)
            conv += (res == "MATE")
            print(f"    {pname:11s} {res:8s} {plies:3d} plies  ({time.time()-t0:.0f}s)", flush=True)
        summary[name] = (conv, len(POSITIONS))
        print(f"    -> converted {conv}/{len(POSITIONS)}\n", flush=True)

    print("=== SUMMARY (mates / forced-win positions) ===")
    for name, (c, n) in summary.items():
        print(f"  {name:24s} {c}/{n}  ({100*c/n:.0f}%)")
    engine.quit()


if __name__ == "__main__":
    main()
