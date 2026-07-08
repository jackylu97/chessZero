#!/usr/bin/env python3
"""Localize the within-position value blindness: is it the DYNAMICS losing the
move signal, or the VALUE HEAD flattening genuinely-different latents?

For each probe position, enumerate all legal moves. Ground truth = -Stockfish eval
of the resulting position (negated: opponent to move). Compare to:
  (a) V_dyn[a]  = value head on dynamics(h_root, a)        [MuZero MCTS path]
  (b) V_repr[a] = value head on representation(actual_next_obs)  [true next state]
  (c) latent_spread = std of pairwise distance of dynamics(h,a) across moves

Report Spearman(V_dyn, SF), Spearman(V_repr, SF), best-move agreement, and the
value spreads. If V_repr ranks siblings but V_dyn does not -> dynamics loses the
signal (mechanistic). If NEITHER ranks but latents are distinct -> value head is
the bottleneck (the V^pi-resolution story). If latents are NOT distinct across
moves -> action-blind dynamics.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F, chess, chess.engine
from scipy.stats import spearmanr

from src.config import MuZeroConfig, get_config
from src.games.chess import ChessGame, _move_to_action
from src.games.base import GameState
from src.model.muzero_net import MuZeroNetwork
from src.model.utils import wdl_to_scalar
from src.training.replay_buffer import stack_with_history

# probe-script local copy of build_network (chess_small)
def build_network(ckpt, game, cfg, device):
    sd = ckpt["model_state_dict"]
    net = MuZeroNetwork(
        observation_channels=game.num_planes * cfg.history_frames,
        action_space_size=game.action_space_size, hidden_planes=cfg.hidden_planes,
        num_blocks=cfg.num_residual_blocks, latent_h=cfg.latent_h, latent_w=cfg.latent_w,
        input_h=8, input_w=8, fc_hidden=cfg.fc_hidden, value_support_size=cfg.value_support_size,
        reward_support_size=cfg.reward_support_size, action_embed_dim=cfg.action_embed_dim,
        use_consistency_loss=any(k.startswith("projection.") for k in sd),
        proj_hid=cfg.proj_hid, proj_out=cfg.proj_out, pred_hid=cfg.pred_hid, pred_out=cfg.pred_out,
        use_scalar_transform=cfg.use_scalar_transform, value_target_scale=cfg.value_target_scale,
        value_head_type=cfg.value_head_type, draw_score=cfg.draw_score,
        use_inverse_dynamics_loss=any(k.startswith("inverse_dynamics_head.") for k in sd),
        inverse_dynamics_hidden=getattr(cfg, "inverse_dynamics_hidden", 96),
        policy_head_type="conv" if any(".policy_head.mix." in k for k in sd) else "flat",
        use_moves_left=any(k.startswith("moves_left_head.") for k in sd),
        moves_left_support_size=getattr(cfg, "moves_left_support_size", 10),
    ).to(device)
    net.load_state_dict(sd); net.eval(); return net

FENS = [
    "r1bq1rk1/ppp2ppp/2n1pn2/2bp4/2BP4/2N1PN2/PPP1QPPP/R1B2RK1 w - - 0 8",
    "r1bq1rk1/pp2nppp/2nppb2/8/2PNP3/2N1B3/PP2BPPP/R2Q1RK1 w - - 4 9",
    "r3k2r/ppp2ppp/2n5/3p1b2/3P4/2N1B3/PPP2PPP/2KR3R w kq - 0 12",
    "r4rk1/pp3ppp/2n1pn2/3p4/3P4/2N1PN2/PP3PPP/R4RK1 w - - 0 14",
    "8/p1p2pkp/1p1p2p1/3P4/2P5/1P3KP1/P4P1P/8 w - - 0 30",
    "r2q1rk1/pp2bppp/2nppn2/8/3NP3/2N1BP2/PPPQB1PP/R4RK1 b - - 1 11",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "2kr3r/ppp2ppp/2n1bn2/2bqp3/8/2NPBN2/PPPQBPPP/2KR3R w - - 0 10",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="chess_small")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sf-depth", type=int, default=12)
    ap.add_argument("--max-moves", type=int, default=24)
    ap.add_argument("--stockfish", default="/usr/games/stockfish")
    args = ap.parse_args()
    dev = args.device; game = ChessGame()
    cfg = get_config(args.config); cfg.device = dev; hf = cfg.history_frames
    torch.serialization.add_safe_globals([MuZeroConfig])
    ck = torch.load(args.checkpoint, map_location=dev, weights_only=True)
    step = ck.get("step", "?"); net = build_network(ck, game, cfg, dev)
    eng = chess.engine.SimpleEngine.popen_uci(args.stockfish)

    print(f"\n=== SIBLING-VIA-DYNAMICS  step {step}  {os.path.basename(args.checkpoint)} ===")
    sp_dyn, sp_repr, sp_dynrepr = [], [], []
    best_dyn, best_repr = [], []
    latent_spreads, vdyn_spreads, vrepr_spreads = [], [], []

    for fen in FENS:
        board = chess.Board(fen)
        stm = chess.WHITE if board.turn == chess.WHITE else chess.BLACK
        state = GameState(board=board, current_player=1 if stm == chess.WHITE else -1)
        cur = game.to_tensor(state)
        obs_root = stack_with_history(cur, [], hf).unsqueeze(0).to(dev)
        with torch.no_grad():
            h_root = net.representation(obs_root)
        legal = list(board.legal_moves)[: args.max_moves]
        if len(legal) < 3:
            continue
        sf_vals, vdyn, vrepr, dyn_lat = [], [], [], []
        with torch.no_grad():
            for mv in legal:
                a = _move_to_action(mv, stm)
                # Stockfish ground truth: eval AFTER the move, negated to STM POV
                nb = board.copy(); nb.push(mv)
                if nb.is_game_over():
                    res = nb.result()
                    sv = 0.0 if res == "1/2-1/2" else (10.0 if (res == "1-0") == (stm == chess.WHITE) else -10.0)
                else:
                    info = eng.analyse(nb, chess.engine.Limit(depth=args.sf_depth))
                    sc = info["score"].pov(stm).score(mate_score=10000)
                    sv = max(-10.0, min(10.0, sc / 100.0))  # centipawns -> pawns, clipped
                sf_vals.append(sv)
                # (a) dynamics path
                hd, _ = net.dynamics(h_root, torch.tensor([a], device=dev))
                _, vl = net.prediction(hd)
                vdyn.append(wdl_to_scalar(vl.float(), draw_score=cfg.draw_score).item())
                dyn_lat.append(hd.flatten())
                # (b) repr-of-actual-next path
                nst = GameState(board=nb, current_player=-state.current_player)
                nobs = stack_with_history(game.to_tensor(nst), [cur], hf).unsqueeze(0).to(dev)
                hr = net.representation(nobs)
                _, vlr = net.prediction(hr)
                # repr_next is OPPONENT to move; negate value to STM POV
                vrepr.append(-wdl_to_scalar(vlr.float(), draw_score=cfg.draw_score).item())

        sf_vals = np.array(sf_vals); vdyn = np.array(vdyn); vrepr = np.array(vrepr)
        if np.std(sf_vals) < 1e-6:
            continue
        sp_dyn.append(spearmanr(vdyn, sf_vals).correlation)
        sp_repr.append(spearmanr(vrepr, sf_vals).correlation)
        sp_dynrepr.append(spearmanr(vdyn, vrepr).correlation)
        best_dyn.append(int(np.argmax(vdyn) == np.argmax(sf_vals)))
        best_repr.append(int(np.argmax(vrepr) == np.argmax(sf_vals)))
        vdyn_spreads.append(float(np.std(vdyn))); vrepr_spreads.append(float(np.std(vrepr)))
        dl = torch.stack(dyn_lat); dln = F.normalize(dl, dim=-1)
        cc = (dln @ dln.T)[~torch.eye(len(dl), dtype=bool)]
        latent_spreads.append(float((1 - cc).std().item()))
    eng.quit()

    def m(x): x = np.array([v for v in x if np.isfinite(v)]); return float(np.mean(x)) if len(x) else float('nan')
    print(f"positions used: {len(sp_dyn)}")
    print(f"  Spearman(V_dyn,  SF)   = {m(sp_dyn):+.3f}   (MCTS path move-ranking)")
    print(f"  Spearman(V_repr, SF)   = {m(sp_repr):+.3f}   (true-next-state move-ranking)")
    print(f"  Spearman(V_dyn, V_repr)= {m(sp_dynrepr):+.3f}   (does dyn agree with repr?)")
    print(f"  best-move agree V_dyn  = {m(best_dyn):.2f}")
    print(f"  best-move agree V_repr = {m(best_repr):.2f}")
    print(f"  V_dyn  std/position    = {m(vdyn_spreads):.4f}")
    print(f"  V_repr std/position    = {m(vrepr_spreads):.4f}")
    print(f"  dyn latent dist-spread = {m(latent_spreads):.4f}  (cross-move latent variety)")
    print(f"SUMMARY step={step} sp_dyn={m(sp_dyn):.3f} sp_repr={m(sp_repr):.3f} "
          f"best_dyn={m(best_dyn):.2f} best_repr={m(best_repr):.2f} "
          f"vdyn_std={m(vdyn_spreads):.4f} vrepr_std={m(vrepr_spreads):.4f} lat_spread={m(latent_spreads):.4f}")

if __name__ == "__main__":
    main()
