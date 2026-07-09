"""Head-to-head match between two MuZero checkpoints (A = candidate vs B =
opponent), to measure relative strength / training progress. Both play greedy
serial PUCT MCTS (temperature=0, no noise). Colors alternate each game; a few
random opening plies diversify the games (greedy play is otherwise deterministic
-> identical games). Reports W/D/L + score + implied Elo gap from A's POV.

Usage:
  .venv/bin/python scripts/eval_vs_checkpoint.py \
    --checkpoint-a checkpoints/chess/<run>/checkpoint_30000.pt \
    --checkpoint-b checkpoints/chess/<run>/checkpoint_15000.pt \
    --game chess_small --device cuda --num-simulations 200 --num-games 30 \
    --random-opening-plies 8
"""
import argparse, os, sys, time, math, random
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); os.chdir(REPO)
import torch, chess
from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame
from src.model.muzero_net import MuZeroNetwork
from src.mcts.mcts import MCTS, select_action
from src.training.replay_buffer import stack_with_history


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
def play_game(game, mcts_a, mcts_b, a_is_white, ply_cap, n_random, rng):
    state = game.reset()
    frame_history = []
    hf = mcts_a.config.history_frames
    plies = 0
    while not state.done and plies < ply_cap:
        legal = game.legal_actions(state)
        if plies < n_random:
            action = rng.choice(legal)
        else:
            a_to_move = (state.board.turn == chess.WHITE) == a_is_white
            mcts = mcts_a if a_to_move else mcts_b
            obs = stack_with_history(game.to_tensor(state), frame_history, hf)
            root = mcts.run(obs, legal, add_noise=False)
            action, _ = select_action(root, temperature=0)
        frame_history.append(game.to_tensor(state))
        state, _, _ = game.step(state, action)
        plies += 1
    if not state.done:
        return 0, plies, "plycap"
    w = state.winner
    if w == 0:
        return 0, plies, "draw"
    a_won = (w == 1) == a_is_white
    return (1 if a_won else -1), plies, ("A_mate" if a_won else "B_mate")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-a", required=True, help="candidate (newer)")
    ap.add_argument("--checkpoint-b", required=True, help="opponent (older / previous run)")
    ap.add_argument("--game", default="chess_small")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-simulations", type=int, default=200)
    ap.add_argument("--num-games", type=int, default=30)
    ap.add_argument("--random-opening-plies", type=int, default=8)
    ap.add_argument("--ply-cap", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = get_config(args.game)
    cfg.num_simulations = args.num_simulations
    cfg.use_gumbel = False
    game = ChessGame()
    net_a = load_network(args.checkpoint_a, game, cfg, args.device)
    net_b = load_network(args.checkpoint_b, game, cfg, args.device)
    mcts_a = MCTS(net_a, game, cfg, args.device)
    mcts_b = MCTS(net_b, game, cfg, args.device)
    rng = random.Random(args.seed)

    A, B = os.path.basename(args.checkpoint_a), os.path.basename(args.checkpoint_b)
    print(f"=== A={A}  vs  B={B} ===")
    print(f"sims={args.num_simulations} games={args.num_games} random_open={args.random_opening_plies} "
          f"device={args.device} ply_cap={args.ply_cap}\n", flush=True)

    W = D = L = 0
    terms = {}
    t0 = time.time()
    for g in range(args.num_games):
        a_white = (g % 2 == 0)
        r, plies, term = play_game(game, mcts_a, mcts_b, a_white, args.ply_cap,
                                   args.random_opening_plies, rng)
        terms[term] = terms.get(term, 0) + 1
        if r > 0: W += 1
        elif r < 0: L += 1
        else: D += 1
        res = {1: "A-WIN", 0: "draw ", -1: "B-win"}[r]
        print(f"  game {g+1:2d}/{args.num_games}  A={'W' if a_white else 'B'}  "
              f"{res}  ({plies} plies, {term})   A: +{W}={D}-{L}", flush=True)

    n = args.num_games
    score = (W + 0.5 * D) / n
    print(f"\n=== RESULT  A vs B  ({n} games, {time.time()-t0:.0f}s) ===")
    print(f"  A (newer) W/D/L = {W}/{D}/{L}   A score = {score*100:.1f}%")
    print(f"  terminations: {terms}")
    if 0 < score < 1:
        gap = 400 * math.log10(score / (1 - score))
        print(f"  implied Elo(A) - Elo(B) = {gap:+.0f}")
    elif score == 1.0:
        print("  A won every game (Elo gap > +400, unbounded)")
    elif score == 0.0:
        print("  A lost every game (Elo gap < -400, unbounded)")


if __name__ == "__main__":
    main()
