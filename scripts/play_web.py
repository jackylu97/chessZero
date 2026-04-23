"""Play chess against a trained MuZero checkpoint in the browser.

Usage:
    python scripts/play_web.py --checkpoint checkpoints/chess/checkpoint_2500.pt

Then open http://localhost:5000 in a browser. You play Black; the model plays White.
Promotions default to queen.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from flask import Flask, jsonify, request

from src.config import MuZeroConfig, get_config
from src.games.chess import ChessGame
from src.mcts.mcts import MCTS, BatchedMCTS, select_action, select_action_gumbel
from src.model.muzero_net import MuZeroNetwork
from src.model.utils import support_to_scalar, inverse_scalar_transform


HTML = """<!DOCTYPE html>
<html>
<head>
<title>ChessZero — Play vs Model</title>
<link rel="stylesheet" href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css">
<style>
  body { font-family: -apple-system, system-ui, sans-serif; max-width: 640px; margin: 30px auto; padding: 0 20px; color: #222; }
  h1 { margin-bottom: 4px; }
  .hint { color: #666; font-size: 14px; margin-bottom: 20px; }
  #board { width: 480px; margin: 20px auto; }
  #status { min-height: 1.5em; margin: 10px 0; font-size: 15px; }
  #result { font-weight: bold; }
  #moves { font-family: ui-monospace, monospace; font-size: 13px; white-space: pre-wrap; background: #f5f5f5; padding: 10px; border-radius: 4px; max-height: 220px; overflow-y: auto; margin-top: 8px; }
  #evals { display: flex; gap: 16px; font-family: ui-monospace, monospace; font-size: 13px; margin: 6px 0 4px; padding: 8px 10px; background: #f5f5f5; border-radius: 4px; }
  #evals .label { color: #666; margin-right: 4px; }
  #evals .val { font-weight: 600; }
  .pos { color: #1a7a1a; }
  .neg { color: #a02020; }
  .eq { color: #666; }
  .evalhint { color: #888; font-size: 12px; margin-top: 4px; }
  button { padding: 8px 16px; margin-right: 8px; font-size: 14px; cursor: pointer; border: 1px solid #bbb; background: white; border-radius: 4px; }
  button:hover { background: #f0f0f0; }
</style>
</head>
<body>
<h1>ChessZero</h1>
<p class="hint">You play Black. Drag a piece to move. Pawn promotion defaults to queen.</p>
<div id="board"></div>
<div id="evals">
  <div><span class="label">Raw eval:</span><span class="val" id="raw_eval">—</span></div>
  <div><span class="label">Search eval:</span><span class="val" id="mcts_eval">—</span></div>
</div>
<div class="evalhint">Values from White's perspective. Positive = White advantage, negative = Black advantage. Range [-1, +1].</div>
<div id="status">Loading...</div>
<div>
  <button onclick="newGame()">New game</button>
  <span id="result"></span>
</div>
<h3>Moves</h3>
<div id="moves"></div>

<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
<script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
<script>
let board = null;
let busy = false;

function setStatus(text) { document.getElementById('status').textContent = text; }
function setResult(text) { document.getElementById('result').textContent = text ? '— ' + text : ''; }

function setEval(id, v) {
    const el = document.getElementById(id);
    if (v === null || v === undefined) {
        el.textContent = '—';
        el.className = 'val';
        return;
    }
    const sign = v >= 0 ? '+' : '';
    el.textContent = sign + v.toFixed(3);
    el.className = 'val ' + (v > 0.05 ? 'pos' : v < -0.05 ? 'neg' : 'eq');
}

function setMoves(sanMoves) {
    const lines = [];
    for (let i = 0; i < sanMoves.length; i += 2) {
        const n = Math.floor(i / 2) + 1;
        lines.push(n + '. ' + sanMoves[i] + (sanMoves[i + 1] ? ' ' + sanMoves[i + 1] : ''));
    }
    const el = document.getElementById('moves');
    el.textContent = lines.join('\\n');
    el.scrollTop = el.scrollHeight;
}

function applyState(data) {
    if (data.fen) board.position(data.fen);
    setStatus(data.status);
    setMoves(data.sanMoves || []);
    setResult(data.game_over ? data.result : '');
    setEval('raw_eval', data.value_raw);
    setEval('mcts_eval', data.value_mcts);
}

async function modelMove() {
    busy = true;
    setStatus('Model thinking...');
    const resp = await fetch('/model_move', { method: 'POST' });
    const data = await resp.json();
    applyState(data);
    busy = false;
}

async function newGame() {
    if (busy) return;
    const resp = await fetch('/reset', { method: 'POST' });
    const data = await resp.json();
    applyState(data);
    if (data.model_to_move) await modelMove();
}

async function onDrop(source, target, piece) {
    if (busy) return 'snapback';
    let uci = source + target;
    if ((piece === 'bP' && target[1] === '1') || (piece === 'wP' && target[1] === '8')) {
        uci += 'q';
    }
    busy = true;
    const resp = await fetch('/user_move', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ uci })
    });
    const data = await resp.json();
    if (!data.ok) {
        busy = false;
        return 'snapback';
    }
    applyState(data);
    busy = false;
    if (!data.game_over) await modelMove();
}

window.addEventListener('DOMContentLoaded', () => {
    board = Chessboard('board', {
        draggable: true,
        orientation: 'black',
        pieceTheme: 'https://cdn.jsdelivr.net/gh/oakmac/chessboardjs@master/website/img/chesspieces/wikipedia/{piece}.png',
        onDrop: onDrop,
    });
    newGame();
});
</script>
</body>
</html>
"""


class GameSession:
    def __init__(self, game: ChessGame, network, config, device):
        self.game = game
        self.network = network
        self.config = config
        self.device = device
        # Use Gumbel path (via BatchedMCTS) if configured, matching trainer._agent_action.
        # Serial MCTS.run() is PUCT-only and ignores config.use_gumbel entirely.
        self.use_gumbel = bool(getattr(config, "use_gumbel", False))
        if self.use_gumbel:
            self.mcts = BatchedMCTS(network, game, config, device)
        else:
            self.mcts = MCTS(network, game, config, device)
        self.state = None
        self.san_history: list[str] = []
        # Last MCTS root value from White's POV; updated only when the model moves
        # (user moves don't trigger search). None until the first model move of a game.
        self.last_mcts_value: float | None = None
        self.reset()

    def reset(self):
        self.state = self.game.reset()
        self.san_history = []
        self.last_mcts_value = None

    @torch.no_grad()
    def _raw_value_white_pov(self) -> float | None:
        """Raw value head prediction for the current position, from White's POV, in [-1, +1].

        Returns None when the game is over (value has no meaning at terminal).
        Inverts sign when it's Black to move, since the network emits STM-relative values.
        """
        if self.state.done:
            return None
        obs = self.game.to_tensor(self.state).unsqueeze(0).to(self.device)
        _, _, value_logits = self.network.initial_inference_logits(obs)
        v_trans = support_to_scalar(
            value_logits.float(), self.network.value_support_size
        ).squeeze(-1)
        if self.config.use_scalar_transform:
            v_scalar = inverse_scalar_transform(v_trans)
        else:
            scale = self.config.value_target_scale
            v_scalar = v_trans / scale if scale != 1.0 else v_trans
        v_stm = float(v_scalar.item())
        return v_stm if self.state.current_player == 1 else -v_stm

    def to_json(self):
        return {
            "fen": self.state.board.fen(),
            "sanMoves": self.san_history,
            "status": self._status(),
            "game_over": self.state.done,
            "result": self._result() if self.state.done else None,
            "model_to_move": self.state.current_player == 1 and not self.state.done,
            "value_raw": self._raw_value_white_pov(),
            "value_mcts": self.last_mcts_value,
        }

    def _status(self) -> str:
        if self.state.done:
            return self._result() or "Game over"
        turn = "White (model)" if self.state.current_player == 1 else "Black (you)"
        check = " — check" if self.state.board.is_check() else ""
        return f"{turn} to move{check}"

    def _result(self) -> str:
        if self.state.winner == 1:
            return "Model (White) wins"
        if self.state.winner == -1:
            return "You (Black) win"
        return "Draw"

    def apply_user_move(self, uci: str) -> bool:
        action = self.game.parse_human_move(self.state, uci)
        if action is None:
            return False
        self.san_history.append(self.game.action_to_san(self.state, action))
        self.state, _, _ = self.game.step(self.state, action)
        return True

    def apply_model_move(self):
        if self.state.done or self.state.current_player != 1:
            return
        obs = self.game.to_tensor(self.state)
        legal = self.game.legal_actions(self.state)
        if self.use_gumbel:
            root = self.mcts.run_batch([obs], [legal], add_noise=False)[0]
            action, _ = select_action_gumbel(root, self.config, self.game.action_space_size)
        else:
            root = self.mcts.run(obs, legal, add_noise=False)
            action, _ = select_action(root, temperature=0)
        # root.value is STM POV — STM at root is White (the model). No sign flip needed.
        self.last_mcts_value = float(root.value)
        self.san_history.append(self.game.action_to_san(self.state, action))
        self.state, _, _ = self.game.step(self.state, action)


app = Flask(__name__)
session: GameSession | None = None


@app.route("/")
def index():
    return HTML


@app.route("/reset", methods=["POST"])
def reset():
    session.reset()
    return jsonify(session.to_json())


@app.route("/user_move", methods=["POST"])
def user_move():
    data = request.get_json(force=True) or {}
    uci = data.get("uci", "")
    if not session.apply_user_move(uci):
        return jsonify({"ok": False})
    return jsonify({"ok": True, **session.to_json()})


@app.route("/model_move", methods=["POST"])
def model_move():
    session.apply_model_move()
    return jsonify(session.to_json())


def load_network(checkpoint_path: str, game: ChessGame, config: MuZeroConfig, device: str):
    torch.serialization.add_safe_globals([MuZeroConfig])
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = ckpt["model_state_dict"]

    # Detect whether the checkpoint was trained with the EZ consistency head
    # (projection + prediction_head). Back-compat with pre-EZ chess checkpoints.
    has_consistency = any(k.startswith("projection.") for k in state_dict)

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
        value_support_size=config.value_support_size,
        reward_support_size=config.reward_support_size,
        use_consistency_loss=has_consistency,
        proj_hid=config.proj_hid,
        proj_out=config.proj_out,
        pred_hid=config.pred_hid,
        pred_out=config.pred_out,
        use_scalar_transform=config.use_scalar_transform,
        value_target_scale=config.value_target_scale,
    )
    network.load_state_dict(state_dict)
    network.to(device)
    network.eval()
    return network


def main():
    global session
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to chess checkpoint .pt file")
    parser.add_argument("--device", default=None)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--use-gumbel", dest="use_gumbel", action="store_true", default=None,
                        help="Force Gumbel MuZero root selection at inference (BatchedMCTS path). "
                             "Overrides config.use_gumbel.")
    parser.add_argument("--no-gumbel", dest="use_gumbel", action="store_false",
                        help="Force PUCT root selection at inference (serial MCTS path), "
                             "regardless of training config. Useful for A/B testing inference-time "
                             "Gumbel vs PUCT on the same checkpoint.")
    parser.add_argument("--num-simulations", type=int, default=None,
                        help="Override config.num_simulations for faster/slower play.")
    args = parser.parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    config = get_config("chess")
    if args.use_gumbel is not None:
        config.use_gumbel = args.use_gumbel
    if args.num_simulations is not None:
        config.num_simulations = args.num_simulations
    game = ChessGame()
    network = load_network(args.checkpoint, game, config, device)
    session = GameSession(game, network, config, device)

    print(f"Open http://{args.host}:{args.port} to play "
          f"[use_gumbel={config.use_gumbel}, num_simulations={config.num_simulations}]")
    # threaded=False to serialize MCTS calls (single user, avoid races on session state)
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False, threaded=False)


if __name__ == "__main__":
    main()
