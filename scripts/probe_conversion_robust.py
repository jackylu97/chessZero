"""Robust endgame-conversion yardstick. For each endgame TYPE, generate many
RANDOM legal positions (winning side to move), verify each is actually won via
Stockfish, then have the model play it out vs full Stockfish defense. Reports a
conversion-% per type + overall, per checkpoint. Averaging over many random
positions removes the SF-defense / single-FEN noise that made the 6-FEN probe
swing 0/6<->2/6.

Usage:
  .venv/bin/python scripts/probe_conversion_robust.py --game chess_small --device cuda \
    --num-simulations 200 --sf-depth 10 --per-type 8 --ply-cap 100 \
    --checkpoints <ckptA.pt> <ckptB.pt>
"""
import argparse, os, sys, time, random
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); os.chdir(REPO)
import torch, chess, chess.engine
from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame, _move_to_action
from src.model.muzero_net import MuZeroNetwork
from src.mcts.mcts import MCTS, select_action
from src.training.replay_buffer import stack_with_history

# endgame type -> (white extra pieces, black extra pieces). Kings always present.
TYPES = {
    # <=5-man baselines (IN the 5-man syzygy tablebase -> directly supervised)
    "5man_KQ_v_K":     ([chess.QUEEN], []),                                                 # 3 total
    "5man_KQ_v_KR":    ([chess.QUEEN], [chess.ROOK]),                                       # 4 total
    # 6-8 man (OUT of the tablebase -> conversion must GENERALIZE past direct supervision)
    "6man_KQRR_v_KR":  ([chess.QUEEN, chess.ROOK, chess.ROOK], [chess.ROOK]),               # 6 total
    "7man_KQRR_v_KRN": ([chess.QUEEN, chess.ROOK, chess.ROOK], [chess.ROOK, chess.KNIGHT]), # 7 total
    "8man_KQQR_v_KRNB":([chess.QUEEN, chess.QUEEN, chess.ROOK],
                        [chess.ROOK, chess.KNIGHT, chess.BISHOP]),                          # 8 total
}


def load_network(ckpt_path, game, cfg, device):
    torch.serialization.add_safe_globals([MuZeroConfig])
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)["model_state_dict"]
    has_conv = any(".policy_head.mix." in k or ".policy_head.proj." in k for k in sd)
    has_from_to = any(".policy_head.q_proj." in k or ".policy_head.k_proj." in k for k in sd)
    policy_head_type = "from_to" if has_from_to else ("conv" if has_conv else "flat")
    has_ml = any(k.startswith("moves_left_head.") for k in sd)
    has_cons = any(k.startswith("projection.") for k in sd)
    has_inv = any(k.startswith("inverse_dynamics_head.") for k in sd)
    has_mat = any(k.startswith("material_head.") for k in sd)
    ms = getattr(cfg, "material_head_support_size", 8)
    if has_mat:
        outw = next((v for k, v in sd.items() if k.startswith("material_head.")
                     and v.ndim == 2 and v.shape[0] % 2 == 1 and v.shape[0] < v.shape[1]*4), None)
        if outw is not None: ms = (outw.shape[0]-1)//2
    rhp = getattr(cfg, "reward_head_planes", 1)
    rw = sd.get("dynamics.reward_head.0.weight")
    if rw is not None: rhp = int(rw.shape[0])
    net = MuZeroNetwork(
        observation_channels=game.num_planes*getattr(cfg,"history_frames",1),
        action_space_size=game.action_space_size, hidden_planes=cfg.hidden_planes,
        num_blocks=cfg.num_residual_blocks, latent_h=cfg.latent_h, latent_w=cfg.latent_w,
        input_h=game.board_size[0], input_w=game.board_size[1], fc_hidden=cfg.fc_hidden,
        value_support_size=cfg.value_support_size, reward_support_size=cfg.reward_support_size,
        reward_head_planes=rhp,
        action_embed_dim=getattr(cfg,"action_embed_dim",16),
        use_consistency_loss=has_cons, proj_hid=cfg.proj_hid, proj_out=cfg.proj_out,
        pred_hid=cfg.pred_hid, pred_out=cfg.pred_out, use_scalar_transform=cfg.use_scalar_transform,
        value_target_scale=cfg.value_target_scale, value_head_type=getattr(cfg,"value_head_type","support"),
        draw_score=getattr(cfg,"draw_score",0.0),
        use_inverse_dynamics_loss=has_inv, inverse_dynamics_hidden=getattr(cfg,"inverse_dynamics_hidden",256),
        policy_head_type=policy_head_type,
        use_moves_left=has_ml, moves_left_support_size=getattr(cfg,"moves_left_support_size",10),
        use_material_head=has_mat, material_head_support_size=ms,
        value_head_planes=getattr(cfg,"value_head_planes",1), value_head_blocks=getattr(cfg,"value_head_blocks",0),
        moves_left_head_planes=getattr(cfg,"moves_left_head_planes",1),
        moves_left_head_blocks=getattr(cfg,"moves_left_head_blocks",0),
        use_repr_attention=getattr(cfg,"use_repr_attention",False),
        use_dyn_attention=getattr(cfg,"use_dyn_attention",False),
        use_pred_attention=getattr(cfg,"use_pred_attention",False), use_smolgen=getattr(cfg,"use_smolgen",True),
        attn_layers=getattr(cfg,"attn_layers",4), attn_heads=getattr(cfg,"attn_heads",4),
        pred_attn_layers=getattr(cfg,"pred_attn_layers",2), hybrid_stem_blocks=getattr(cfg,"hybrid_stem_blocks",0),
    )
    net.load_state_dict(sd); net.to(device); net.eval()
    return net


def random_won_fen(white_extra, black_extra, rng, engine, sf_depth):
    """Random legal position, White (winning side) to move, verified won by SF."""
    for _ in range(200):
        squares = rng.sample(range(64), 2 + len(white_extra) + len(black_extra))
        board = chess.Board(None)
        board.set_piece_at(squares[0], chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(squares[1], chess.Piece(chess.KING, chess.BLACK))
        i = 2
        for pt in white_extra:
            board.set_piece_at(squares[i], chess.Piece(pt, chess.WHITE)); i += 1
        for pt in black_extra:
            board.set_piece_at(squares[i], chess.Piece(pt, chess.BLACK)); i += 1
        board.turn = chess.WHITE
        if not board.is_valid():       # kings adjacent / side-not-to-move in check / etc.
            continue
        if board.is_game_over():
            continue
        info = engine.analyse(board, chess.engine.Limit(depth=sf_depth))
        sc = info["score"].white()
        cp = sc.score(mate_score=100000)
        if cp is not None and cp >= 500:   # White clearly winning
            return board.fen()
    return None


@torch.no_grad()
def convert_one(game, mcts, engine, fen, sf_limit, ply_cap):
    state = game.reset_from_fen(fen); fh = []; hf = mcts.config.history_frames; plies = 0
    while not state.done and plies < ply_cap:
        if state.board.turn == chess.WHITE:
            obs = stack_with_history(game.to_tensor(state), fh, hf)
            root = mcts.run(obs, game.legal_actions(state), add_noise=False)
            action, _ = select_action(root, temperature=0)
        else:
            action = _move_to_action(engine.play(state.board, sf_limit).move, state.board.turn)
        fh.append(game.to_tensor(state)); state, _, _ = game.step(state, action); plies += 1
    return bool(state.done and state.winner == 1), plies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--game", default="chess_small")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-simulations", type=int, default=200)
    ap.add_argument("--sf-depth", type=int, default=10)
    ap.add_argument("--per-type", type=int, default=8)
    ap.add_argument("--ply-cap", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stockfish", default="tools/stockfish/stockfish")
    args = ap.parse_args()

    cfg = get_config(args.game); cfg.num_simulations = args.num_simulations; cfg.use_gumbel = False
    game = ChessGame()
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    sf_limit = chess.engine.Limit(depth=args.sf_depth)

    # Generate a SHARED set of positions once (same for every checkpoint = fair).
    rng = random.Random(args.seed)
    fens = {t: [] for t in TYPES}
    print(f"generating {args.per_type} verified-won positions per type...", flush=True)
    for t, (we, be) in TYPES.items():
        while len(fens[t]) < args.per_type:
            f = random_won_fen(we, be, rng, engine, args.sf_depth)
            if f: fens[t].append(f)
        print(f"  {t}: {len(fens[t])} positions", flush=True)

    print(f"\n=== ROBUST CONVERSION  sims={args.num_simulations} sf_depth={args.sf_depth} "
          f"per_type={args.per_type} ply_cap={args.ply_cap} ===", flush=True)
    for ckpt in args.checkpoints:
        name = os.path.basename(ckpt)
        net = load_network(ckpt, game, cfg, args.device); mcts = MCTS(net, game, cfg, args.device)
        print(f"\n[{name}]", flush=True)
        tot_c = tot_n = 0
        for t in TYPES:
            c = 0
            for fen in fens[t]:
                mated, _ = convert_one(game, mcts, engine, fen, sf_limit, args.ply_cap)
                c += mated
            n = len(fens[t]); tot_c += c; tot_n += n
            print(f"    {t:9s} {c}/{n}  ({100*c/n:.0f}%)", flush=True)
        print(f"    OVERALL  {tot_c}/{tot_n}  ({100*tot_c/tot_n:.0f}%)", flush=True)
    engine.quit()


if __name__ == "__main__":
    main()
