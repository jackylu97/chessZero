"""Load a checkpoint and inspect an MCTS tree from a chess position.

Run a configurable MCTS search and print a tree dump showing, per node:

  - the move that led there (SAN),
  - prior π_net(a) and visit-derived posterior N(a)/ΣN(a),
  - visit count + mean Q (mover-POV) under it,
  - network's raw value prediction at the node's hidden state,
  - top-K children sorted by visits.

Useful for diagnosing pathologies like the draw basin: if the value head is
collapsed, all nodes show ~0 raw value regardless of position; if the
policy is too diffuse, priors are flat across many moves and search can't
focus.

Examples:

  # Inspect MCTS from start position with the latest checkpoint of a run.
  python scripts/inspect_mcts.py \\
      --checkpoint checkpoints/chess/2026_05_06_no_warmstart/checkpoint_5000.pt

  # Specific FEN, more sims, more depth.
  python scripts/inspect_mcts.py \\
      --checkpoint <ckpt.pt> \\
      --fen "rnbq1rk1/ppp2ppp/3p1n2/2bPp3/4P3/2N2N2/PPP2PPP/R1BQKB1R w KQ - 0 6" \\
      --num-simulations 400 --max-depth 4 --top-k 6
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
import numpy as np
import torch
import torch.nn.functional as F

from src.config import get_config
from src.games.chess import ChessGame, _action_to_move
from src.games.base import GameState
from src.mcts.mcts import BatchedMCTS, MCTSNode
from src.model.muzero_net import MuZeroNetwork
from src.model.utils import support_to_scalar, wdl_to_scalar
from src.training.replay_buffer import stack_with_history


def load_checkpoint(checkpoint_path: str, config, game, device: str = "cuda"):
    network = MuZeroNetwork(
        observation_channels=game.num_planes * getattr(config, "history_frames", 1),
        action_space_size=game.action_space_size,
        hidden_planes=config.hidden_planes,
        num_blocks=config.num_residual_blocks,
        latent_h=config.latent_h,
        latent_w=config.latent_w,
        input_h=game.board_size[0],
        input_w=game.board_size[1],
        fc_hidden=config.fc_hidden,
        value_support_size=config.value_support_size,
        reward_support_size=config.reward_support_size,
        use_consistency_loss=config.use_consistency_loss,
        proj_hid=config.proj_hid,
        proj_out=config.proj_out,
        pred_hid=config.pred_hid,
        pred_out=config.pred_out,
        use_scalar_transform=config.use_scalar_transform,
        value_target_scale=config.value_target_scale,
        value_head_type=getattr(config, "value_head_type", "support"),
        draw_score=getattr(config, "draw_score", 0.0),
    ).to(device).eval()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict") or ckpt.get("network_state_dict")
    if state_dict is None:
        raise RuntimeError(f"Checkpoint {checkpoint_path!r} missing state_dict (keys: {list(ckpt)})")
    network.load_state_dict(state_dict)
    return network, ckpt.get("step", -1)


def setup_state(game: ChessGame, fen: str | None) -> GameState:
    if fen is None or fen.lower() == "startpos":
        return game.reset()
    board = chess.Board(fen)
    return GameState(
        board=board,
        current_player=1 if board.turn == chess.WHITE else -1,
        done=False,
        winner=0,
    )


@torch.no_grad()
def network_predict(network, hidden: torch.Tensor, value_head_type: str, draw_score: float):
    """Run prediction head on a hidden state. Returns (value_scalar, policy_probs, value_aux)."""
    policy_logits, value_logits = network.prediction(hidden)
    policy_probs = F.softmax(policy_logits, dim=-1)
    if value_head_type == "wdl":
        wdl = F.softmax(value_logits, dim=-1)  # [N, 3] = (W, D, L)
        value = wdl_to_scalar(value_logits, draw_score=draw_score)
        aux = {"P_W": float(wdl[0, 0]), "P_D": float(wdl[0, 1]), "P_L": float(wdl[0, 2])}
    else:
        value = support_to_scalar(value_logits, network.value_support_size)
        aux = {}
    return float(value.squeeze()), policy_probs.squeeze(0), aux


def _action_to_san_safe(board: chess.Board, action: int) -> str:
    move = _action_to_move(int(action), board)
    if move is None:
        return f"<bad {action}>"
    if move not in board.legal_moves:
        return f"<illegal {move.uci()}>"
    try:
        return board.san(move)
    except Exception:
        return move.uci()


def print_node(
    node: MCTSNode,
    board: chess.Board,
    depth: int,
    max_depth: int,
    top_k: int,
    network,
    value_head_type: str,
    draw_score: float,
    show_network: bool,
    indent: str = "",
):
    """Recursively print MCTS tree under `node`."""
    if node.child_actions is None or len(node.child_actions) == 0:
        return

    visits = np.asarray(node.child_visits, dtype=np.float64)
    priors = np.asarray(node.child_priors, dtype=np.float64)
    actions = np.asarray(node.child_actions, dtype=np.int64)
    rewards = np.asarray(node.child_rewards, dtype=np.float64)
    value_sums = np.asarray(node.child_value_sums, dtype=np.float64)

    total_visits = visits.sum()
    posterior = visits / max(total_visits, 1.0)

    # Mover-POV Q for each child slot:
    #   Q(parent_POV) = child.reward − γ · child.value(child_POV)
    discount = 1.0  # chess; matches chess preset
    with np.errstate(invalid="ignore", divide="ignore"):
        child_value_avg = np.where(visits > 0, value_sums / np.maximum(visits, 1.0), 0.0)
    raw_q = rewards - discount * child_value_avg

    # Sort children by visit count, descending.
    order = np.argsort(-visits)
    show = order[:top_k]

    for rank, slot in enumerate(show):
        action = int(actions[slot])
        san = _action_to_san_safe(board, action)
        v = int(visits[slot])
        p = float(priors[slot])
        post = float(posterior[slot])
        q = float(raw_q[slot])
        marker = "★" if rank == 0 else " "
        print(
            f"{indent}{marker} {san:<7} "
            f"prior={p:.3f}  N={v:4d} ({post*100:5.1f}%)  "
            f"Q={q:+.3f}"
            + (f"  reward={float(rewards[slot]):+.2f}" if rewards[slot] != 0 else "")
        )

        # Recurse into materialized children.
        child = node.children[slot] if slot < len(node.children) else None
        if child is None or not child.expanded() or depth + 1 >= max_depth:
            continue

        # Apply move on the python-chess board so we can decode SAN at the
        # next level. If illegal in the real position we just skip recursion
        # (action came from MuZero's full-action sample, not legal-mask).
        move = _action_to_move(action, board)
        if move is None or move not in board.legal_moves:
            continue
        next_board = board.copy()
        next_board.push(move)

        if show_network and child.hidden_state is not None:
            hidden = child.hidden_state.unsqueeze(0)
            net_v, net_pi, aux = network_predict(network, hidden, value_head_type, draw_score)
            entropy = float(-(net_pi * (net_pi.clamp(min=1e-12).log())).sum())
            wdl = ""
            if aux:
                wdl = f"  W/D/L={aux['P_W']:.2f}/{aux['P_D']:.2f}/{aux['P_L']:.2f}"
            print(
                f"{indent}    [net@child] V_raw={net_v:+.3f}  "
                f"H(π)={entropy:.2f}{wdl}"
            )

        print_node(
            child, next_board, depth + 1, max_depth, top_k,
            network, value_head_type, draw_score, show_network,
            indent=indent + "    ",
        )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    p.add_argument("--fen", default=None, help='FEN string or "startpos" (default).')
    p.add_argument("--num-simulations", type=int, default=200)
    p.add_argument("--sample-k", type=int, default=None,
                   help="Override config.sample_k. None = config default.")
    p.add_argument("--max-depth", type=int, default=2,
                   help="Tree print depth (root is 0).")
    p.add_argument("--top-k", type=int, default=10,
                   help="Top-K children (sorted by visits) to show per node.")
    p.add_argument("--show-network", action="store_true",
                   help="Also print network's raw V_pred and policy entropy at each node.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--add-noise", action="store_true",
                   help="Mix Dirichlet exploration noise into root prior. "
                        "Off by default: cleaner snapshot of the network's "
                        "policy at the root.")
    p.add_argument("--pv", type=int, default=0, metavar="DEPTH",
                   help="After dumping the tree, also print the principal "
                        "variation: the most-visited line walked DEPTH plies "
                        "deep. Pass e.g. --pv 8 for an 8-ply PV. 0 disables.")
    args = p.parse_args()

    config = get_config("chess")
    if args.sample_k is not None:
        config.sample_k = args.sample_k
    config.num_simulations = args.num_simulations

    game = ChessGame()
    network, step = load_checkpoint(args.checkpoint, config, game, device=args.device)
    print(f"Loaded checkpoint at step {step} ({args.checkpoint})")
    print(f"  config: sims={config.num_simulations}, sample_k={config.sample_k}, "
          f"hidden={config.hidden_planes}×{config.num_residual_blocks}, "
          f"value_head={getattr(config, 'value_head_type', 'support')}")
    print()

    state = setup_state(game, args.fen)
    print(f"FEN: {state.board.fen()}")
    print(f"To move: {'White' if state.board.turn == chess.WHITE else 'Black'}")
    print(f"Legals: {state.board.legal_moves.count()}")
    print()

    # Build T-frame stacked observation. For a freshly-loaded position we have
    # no prior plies, so missing frames are zero-padded by ``stack_with_history``.
    n_frames = int(getattr(config, "history_frames", 1))
    obs_single = game.to_tensor(state)
    obs_stacked = stack_with_history(obs_single, [], n_frames)
    legal_actions = game.legal_actions(state)

    mcts = BatchedMCTS(network, game, config, args.device)
    roots = mcts.run_batch([obs_stacked], [legal_actions], add_noise=args.add_noise)
    root = roots[0]

    # Network's raw prediction at the root, to compare with MCTS-derived.
    obs_batch = obs_stacked.unsqueeze(0).to(args.device)
    hidden, policy_logits, value_root = network.initial_inference(obs_batch)
    net_v = float(value_root.squeeze())
    net_pi = F.softmax(policy_logits, dim=-1).squeeze(0)
    entropy = float(-(net_pi * (net_pi.clamp(min=1e-12).log())).sum())

    aux = {}
    vh = getattr(config, "value_head_type", "support")
    if vh == "wdl":
        _, value_logits = network.prediction(hidden)
        wdl = F.softmax(value_logits, dim=-1).squeeze(0)
        aux = {"P_W": float(wdl[0]), "P_D": float(wdl[1]), "P_L": float(wdl[2])}

    print(f"=== ROOT ===")
    print(f"  Network raw:    V={net_v:+.3f}, H(π)={entropy:.2f}"
          + (f", W/D/L={aux['P_W']:.2f}/{aux['P_D']:.2f}/{aux['P_L']:.2f}" if aux else ""))
    print(f"  MCTS value:     V={root.value:+.3f}  (visits={root.visit_count})")
    print(f"  Children:       sampled={len(root.child_actions)} of {state.board.legal_moves.count()} legal")
    print()

    print(f"=== Top-{args.top_k} (depth ≤ {args.max_depth}) ===")
    print_node(
        root, state.board.copy(), depth=0, max_depth=args.max_depth, top_k=args.top_k,
        network=network, value_head_type=vh,
        draw_score=getattr(config, "draw_score", 0.0),
        show_network=args.show_network,
    )

    if args.pv > 0:
        print()
        print(f"=== Principal variation (most-visited line, ≤ {args.pv} plies) ===")
        pv_moves: list[str] = []
        node = root
        board = state.board.copy()
        for _ in range(args.pv):
            if node.child_actions is None or len(node.child_actions) == 0:
                break
            visits = np.asarray(node.child_visits)
            if visits.sum() == 0:
                break
            slot = int(np.argmax(visits))
            action = int(node.child_actions[slot])
            move = _action_to_move(action, board)
            if move is None or move not in board.legal_moves:
                pv_moves.append(f"<illegal {action}>")
                break
            try:
                pv_moves.append(board.san(move))
            except Exception:
                pv_moves.append(move.uci())
            board.push(move)
            child = node.children[slot] if slot < len(node.children) else None
            if child is None or not child.expanded():
                break
            node = child
        # Render with move numbers, classical chess notation.
        rendered: list[str] = []
        side_at_root = state.board.turn  # WHITE / BLACK
        ply = 0
        for san in pv_moves:
            if side_at_root == chess.WHITE:
                # White moves at even plies (0, 2, ...).
                if ply % 2 == 0:
                    rendered.append(f"{state.board.fullmove_number + ply // 2}.{san}")
                else:
                    rendered.append(san)
            else:
                # Black moves first; fullmove number unchanged on black-then-white.
                if ply == 0:
                    rendered.append(f"{state.board.fullmove_number}...{san}")
                elif ply % 2 == 1:
                    rendered.append(f"{state.board.fullmove_number + (ply + 1) // 2}.{san}")
                else:
                    rendered.append(san)
            ply += 1
        print("  " + " ".join(rendered))


if __name__ == "__main__":
    main()
