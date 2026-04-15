"""Latent state visualization script."""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.config import get_config
from src.games import GAME_REGISTRY
from src.viz.latent_viz import LatentAnalyzer, collect_game_states


def main():
    parser = argparse.ArgumentParser(description="Visualize MuZero latent states")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--games", type=str, nargs="+", default=["tictactoe"])
    parser.add_argument("--multi-game", action="store_true",
                        help="Use MultiGameMuZeroNetwork")
    parser.add_argument("--num-states", type=int, default=500)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--method", choices=["tsne", "umap"], default="tsne")
    parser.add_argument("--log-dir", type=str, default="runs/latent_viz")
    parser.add_argument("--probe", action="store_true",
                        help="Train probing classifiers")
    args = parser.parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if args.multi_game:
        from src.model.muzero_net import MultiGameMuZeroNetwork
        game_configs = {}
        for name in args.games:
            game = GAME_REGISTRY[name]()
            game_configs[name] = {
                "obs_channels": game.num_planes,
                "action_space": game.action_space_size,
                "input_h": game.board_size[0],
                "input_w": game.board_size[1],
            }
        network = MultiGameMuZeroNetwork(game_configs)
        network.load_state_dict(checkpoint["model_state_dict"])
        network.to(device)
    else:
        from src.model.muzero_net import MuZeroNetwork
        game_name = args.games[0]
        config = get_config(game_name)
        game = GAME_REGISTRY[game_name]()
        network = MuZeroNetwork(
            observation_channels=game.num_planes,
            action_space_size=game.action_space_size,
            hidden_planes=config.hidden_planes,
            num_blocks=config.num_residual_blocks,
            latent_h=config.latent_h,
            latent_w=config.latent_w,
            input_h=game.board_size[0],
            input_w=game.board_size[1],
            fc_hidden=config.fc_hidden,
        )
        network.load_state_dict(checkpoint["model_state_dict"])
        network.to(device)

    analyzer = LatentAnalyzer(network, device)

    # Collect states from all games
    all_hidden = []
    all_labels = []
    all_metadata = []

    for game_name in args.games:
        game = GAME_REGISTRY[game_name]()
        print(f"Collecting {args.num_states} states from {game_name}...")
        state_data = collect_game_states(game, args.num_states)

        observations = [s["observation"] for s in state_data]
        gname = game_name if args.multi_game else None
        hidden = analyzer.extract_hidden_states(observations, gname)
        all_hidden.append(hidden)

        for s in state_data:
            all_labels.append(game_name)
            all_metadata.append(f"{game_name}|{s['game_phase']}|move_{s['move_number']}")

    all_hidden = np.concatenate(all_hidden, axis=0)
    print(f"Total hidden states: {all_hidden.shape}")

    # Dimensionality reduction
    print(f"Computing {args.method}...")
    if args.method == "tsne":
        coords = analyzer.compute_tsne(all_hidden)
    else:
        coords = analyzer.compute_umap(all_hidden)

    # Save to TensorBoard
    analyzer.visualize_to_tensorboard(all_hidden, all_labels, all_metadata, args.log_dir)
    print(f"Embeddings saved to TensorBoard at {args.log_dir}")

    # Probing classifiers
    if args.probe:
        print("\nTraining probing classifiers...")
        # Game type probe
        result = analyzer.train_probe(all_hidden, np.array(all_labels))
        print(f"  Game type probe accuracy: {result['accuracy']:.3f} (+/- {result['std']:.3f})")

        # Game phase probe
        phases = [m.split("|")[1] for m in all_metadata]
        result = analyzer.train_probe(all_hidden, np.array(phases))
        print(f"  Game phase probe accuracy: {result['accuracy']:.3f} (+/- {result['std']:.3f})")

    # Cross-game similarity (if multiple games)
    if len(args.games) > 1:
        print("\nCross-game CKA similarity:")
        game_hidden = {}
        offset = 0
        for name in args.games:
            n = args.num_states
            game_hidden[name] = all_hidden[offset:offset + n]
            offset += n

        for i, name_a in enumerate(args.games):
            for name_b in args.games[i + 1:]:
                n = min(len(game_hidden[name_a]), len(game_hidden[name_b]))
                result = analyzer.cross_game_similarity(
                    game_hidden[name_a][:n], game_hidden[name_b][:n]
                )
                print(f"  {name_a} vs {name_b}: CKA = {result['cka']:.4f}")


if __name__ == "__main__":
    main()
