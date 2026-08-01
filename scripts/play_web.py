"""Play chess against a trained MuZero checkpoint in the browser.

Usage:
    # explicit checkpoint
    python scripts/play_web.py --checkpoint checkpoints/chess/checkpoint_2500.pt
    # latest checkpoint of the most-recently-active run (no args needed)
    python scripts/play_web.py
    # latest checkpoint of a specific run
    python scripts/play_web.py --run-id 2026_06_17_convhead

Then open http://localhost:5000 in a browser. You play Black; the model plays White.
Promotions default to queen.
"""

import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import chess.engine
import torch
from flask import Flask, jsonify, request

from src.config import MuZeroConfig, get_config
from src.games.chess import ChessGame, _move_to_action
from src.mcts.mcts import MCTS, BatchedMCTS, select_action, select_action_gumbel
from src.model.muzero_net import MuZeroNetwork
from src.model.utils import support_to_scalar, inverse_scalar_transform
from src.training.replay_buffer import stack_with_history


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
  button:disabled { opacity: 0.4; cursor: default; }
  #nav { margin: 8px 0; display: flex; align-items: center; gap: 6px; }
  #nav button { padding: 4px 10px; margin-right: 0; }
  .plyind { font-family: ui-monospace, monospace; font-size: 13px; min-width: 56px; text-align: center; }
  #histbadge { color: #b06000; font-size: 12px; margin-left: 8px; }
  #sf { font-family: ui-monospace, monospace; font-size: 13px; margin: 6px 0 4px; padding: 8px 10px; background: #eef4ff; border-radius: 4px; }
  #sf .label { color: #666; margin-right: 6px; }
  #board .check-square { box-shadow: inset 0 0 4px 4px #c00; background: #ff9a9a !important; }
  #board .legal-dest { box-shadow: inset 0 0 2px 3px rgba(20, 85, 30, 0.45); }
</style>
</head>
<body>
<h1>ChessZero</h1>
<p class="hint">You play Black. Drag a piece to move. Pawn promotion defaults to queen.</p>
<div id="board"></div>
<div id="nav">
  <button id="nav_first" onclick="navTo(0)" title="start">&#9198;</button>
  <button id="nav_prev" onclick="navTo(viewIdx - 1)" title="back (left arrow)">&#9664;</button>
  <span id="ply" class="plyind">0/0</span>
  <button id="nav_next" onclick="navTo(viewIdx + 1)" title="forward (right arrow)">&#9654;</button>
  <button id="nav_last" onclick="navTo(latest())" title="latest">&#9197;</button>
  <span id="histbadge" style="display:none">viewing history &mdash; board locked</span>
</div>
<div id="sf"><span class="label">SF d8 top moves:</span><span id="sfmoves">&mdash;</span></div>
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
let viewIdx = 0;
let sfToken = 0;
let S = { fens: ['start'], checks: [null], legal: [], modelToMove: false, gameOver: false };

function latest() { return S.fens.length - 1; }

function clearHighlights(cls) {
    document.querySelectorAll('#board .' + cls).forEach(el => el.classList.remove(cls));
}

function navTo(i) {
    viewIdx = Math.max(0, Math.min(latest(), i));
    renderView();
}

function renderView() {
    if (viewIdx > latest()) viewIdx = latest();
    board.position(S.fens[viewIdx], false);
    document.getElementById('ply').textContent = viewIdx + '/' + latest();
    document.getElementById('histbadge').style.display = (viewIdx < latest()) ? 'inline' : 'none';
    document.getElementById('nav_first').disabled = (viewIdx === 0);
    document.getElementById('nav_prev').disabled = (viewIdx === 0);
    document.getElementById('nav_next').disabled = (viewIdx === latest());
    document.getElementById('nav_last').disabled = (viewIdx === latest());
    clearHighlights('check-square');
    clearHighlights('legal-dest');
    const ck = S.checks[viewIdx];
    if (ck) {
        const el = document.querySelector('#board .square-' + ck);
        if (el) el.classList.add('check-square');
    }
    fetchHints();
}

async function fetchHints() {
    const tok = ++sfToken;
    const el = document.getElementById('sfmoves');
    el.textContent = '\\u2026';
    try {
        const resp = await fetch('/sf?ply=' + viewIdx);
        const d = await resp.json();
        if (tok !== sfToken) return;   // stale response for an old view
        if (d.unavailable) { el.textContent = 'engine unavailable'; return; }
        if (!d.moves || !d.moves.length) { el.textContent = '\\u2014'; return; }
        el.textContent = d.moves.map((m, i) => (i + 1) + ') ' + m.san + ' ' + m.eval).join('    ');
    } catch (e) {
        if (tok === sfToken) el.textContent = '\\u2014';
    }
}

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
    S.fens = data.fens || [data.fen];
    S.checks = data.checks || [];
    S.legal = data.legalMoves || [];
    S.modelToMove = !!data.model_to_move;
    S.gameOver = !!data.game_over;
    viewIdx = latest();   // any state change jumps the view back to live
    setStatus(data.status);
    setMoves(data.sanMoves || []);
    setResult(data.game_over ? data.result : '');
    setEval('raw_eval', data.value_raw);
    setEval('mcts_eval', data.value_mcts);
    renderView();
}

function onDragStart(source, piece) {
    // Board is interactive only at the live position, on the user's (Black) turn.
    if (busy || S.gameOver || S.modelToMove) return false;
    if (viewIdx !== latest()) return false;
    if (piece[0] === 'w') return false;
    const from = S.legal.filter(u => u.slice(0, 2) === source);
    if (!from.length) return false;   // piece has no legal move
    from.forEach(u => {
        const el = document.querySelector('#board .square-' + u.slice(2, 4));
        if (el) el.classList.add('legal-dest');
    });
    return true;
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
    clearHighlights('legal-dest');
    if (busy || viewIdx !== latest()) return 'snapback';
    let uci = source + target;
    if ((piece === 'bP' && target[1] === '1') || (piece === 'wP' && target[1] === '8')) {
        uci += 'q';
    }
    // Client-side legality: reject illegal moves without a server round-trip.
    if (S.legal.indexOf(uci) === -1) return 'snapback';
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
        onDragStart: onDragStart,
        onDrop: onDrop,
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') { navTo(viewIdx - 1); e.preventDefault(); }
        else if (e.key === 'ArrowRight') { navTo(viewIdx + 1); e.preventDefault(); }
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
        # Always BatchedMCTS: run_batch is the only search path wired for the
        # root-terminal-draws override (forced_draw_actions). It runs PUCT when
        # config.use_gumbel is False and Gumbel otherwise, matching training/eval.
        # (The serial MCTS.run() can't take forced draws, which let the model walk a
        # won position into stalemate/repetition during interactive play.)
        self.use_gumbel = bool(getattr(config, "use_gumbel", False))
        self.mcts = BatchedMCTS(network, game, config, device)
        self.history_frames = getattr(config, "history_frames", 1)
        self.state = None
        self.san_history: list[str] = []
        # Single-frame observations of PRIOR plies (chronological, newest last), used
        # to build the T-frame history stack the network was trained on.
        self.frame_history: list = []
        # Last MCTS root value from White's POV; updated only when the model moves
        # (user moves don't trigger search). None until the first model move of a game.
        self.last_mcts_value: float | None = None
        self.reset()

    def reset(self):
        self.state = self.game.reset()
        self.san_history = []
        self.frame_history = []
        self.last_mcts_value = None
        # Per-ply position record for the move navigator: FEN after each ply
        # (index 0 = start position) and, when the side to move is in check,
        # the square of their king (for the red check highlight).
        self.fen_history = [self.state.board.fen()]
        self.check_history = [self._check_square()]

    def _check_square(self):
        board = self.state.board
        if not self.state.done and board.is_check():
            return chess.square_name(board.king(board.turn))
        if self.state.done and board.is_checkmate():
            return chess.square_name(board.king(board.turn))
        return None

    def _record_position(self):
        self.fen_history.append(self.state.board.fen())
        self.check_history.append(self._check_square())

    def _stacked_obs(self):
        """T-frame history-stacked observation for the current position (matches
        training-time GameHistory._stack_history: newest frame first)."""
        cur = self.game.to_tensor(self.state)
        return stack_with_history(cur, self.frame_history, self.history_frames)

    @torch.no_grad()
    def _raw_value_white_pov(self) -> float | None:
        """Raw value head prediction for the current position, from White's POV, in [-1, +1].

        Returns None when the game is over (value has no meaning at terminal).
        Inverts sign when it's Black to move, since the network emits STM-relative values.
        """
        if self.state.done:
            return None
        # initial_inference returns the already-decoded scalar value, handling the
        # WDL head (wdl_to_scalar) or the support head transparently — so this works
        # for both current and legacy checkpoints. Feed the history stack.
        obs = self._stacked_obs().unsqueeze(0).to(self.device)
        _, _, value = self.network.initial_inference(obs)
        v_stm = float(value.item())
        return v_stm if self.state.current_player == 1 else -v_stm

    def to_json(self):
        return {
            "fen": self.state.board.fen(),
            "fens": list(self.fen_history),
            "checks": list(self.check_history),
            "legalMoves": ([] if self.state.done
                           else [m.uci() for m in self.state.board.legal_moves]),
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
        # Record this ply's frame before stepping, so it becomes history for later plies.
        self.frame_history.append(self.game.to_tensor(self.state))
        self.state, _, _ = self.game.step(self.state, action)
        self._record_position()
        return True

    def _forced_draw_actions(self) -> set:
        """Root-terminal-draws veto set for the current position (shared helper —
        see src/games/chess.py:forced_draw_root_actions). Empty when disabled."""
        if not getattr(self.config, "root_terminal_draws", False):
            return set()
        from src.games.chess import forced_draw_root_actions
        return forced_draw_root_actions(
            self.state.board,
            int(getattr(self.config, "root_terminal_draws_min_repeats", 2)),
            bool(getattr(self.config, "root_terminal_draws_include_stalemate", True)),
        )

    def apply_model_move(self):
        if self.state.done or self.state.current_player != 1:
            return
        obs = self._stacked_obs()
        legal = self.game.legal_actions(self.state)
        # Root-terminal-draws override: veto moves that hand the opponent a draw
        # (stalemate / repetition / insufficient material) while the model is winning.
        # Works on both PUCT and Gumbel root paths (Gumbel support added 2026-07-18).
        fd = self._forced_draw_actions()
        root = self.mcts.run_batch(
            [obs], [legal], add_noise=False, forced_draw_actions=[fd])[0]
        if self.use_gumbel:
            action, _ = select_action_gumbel(root, self.config, self.game.action_space_size)
        else:
            action, _ = select_action(root, temperature=0)
        # root.value is STM POV — STM at root is White (the model). No sign flip needed.
        self.last_mcts_value = float(root.value)
        self.san_history.append(self.game.action_to_san(self.state, action))
        # Record this ply's frame before stepping.
        self.frame_history.append(self.game.to_tensor(self.state))
        self.state, _, _ = self.game.step(self.state, action)
        self._record_position()


app = Flask(__name__)
session: GameSession | None = None
sf_engine: chess.engine.SimpleEngine | None = None
_SF_CACHE: dict = {}   # fen -> hint list (depth-8 MultiPV-3)


@app.route("/")
def index():
    return HTML


@app.route("/sf")
def sf_hints():
    """Stockfish depth-8 top-3 moves for the VIEWED position (?ply=N indexes
    fen_history). Evals White-POV in pawns; '#N' for mates. Cached per FEN."""
    if sf_engine is None or session is None:
        return jsonify({"moves": [], "unavailable": True})
    try:
        ply = int(request.args.get("ply", -1))
    except ValueError:
        ply = -1
    fens = session.fen_history
    if not fens:
        return jsonify({"moves": []})
    if ply < 0 or ply >= len(fens):
        ply = len(fens) - 1
    fen = fens[ply]
    if fen in _SF_CACHE:
        return jsonify({"moves": _SF_CACHE[fen]})
    board = chess.Board(fen)
    if board.is_game_over(claim_draw=False):
        _SF_CACHE[fen] = []
        return jsonify({"moves": []})
    infos = sf_engine.analyse(board, chess.engine.Limit(depth=8), multipv=3)
    if isinstance(infos, dict):
        infos = [infos]
    moves = []
    for info in infos:
        pv = info.get("pv")
        if not pv:
            continue
        mv = pv[0]
        sc = info["score"].white()
        mate = sc.mate()
        ev = f"#{mate}" if mate is not None else f"{sc.score() / 100.0:+.2f}"
        moves.append({"san": board.san(mv), "uci": mv.uci(), "eval": ev})
    if len(_SF_CACHE) > 2000:
        _SF_CACHE.clear()
    _SF_CACHE[fen] = moves
    return jsonify({"moves": moves})


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


def _checkpoint_step(path: str) -> int:
    m = re.search(r"checkpoint_(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def resolve_checkpoint(checkpoint: str, run_id: str, game_dir: str = "checkpoints/chess") -> str:
    """Return an explicit checkpoint path, or auto-select the latest one.

    Precedence: explicit --checkpoint > latest in --run-id > latest in the
    most-recently-modified run dir under ``game_dir``. "Latest" = highest
    ``checkpoint_<step>.pt`` step in the chosen run directory.
    """
    if checkpoint:
        return checkpoint

    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), game_dir)
    if run_id:
        run_dir = os.path.join(root, run_id)
        if not os.path.isdir(run_dir):
            raise SystemExit(f"run dir not found: {run_dir}")
    else:
        # Most-recently-modified run dir that actually contains a checkpoint.
        candidates = [d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)
                      and glob.glob(os.path.join(d, "checkpoint_*.pt"))]
        if not candidates:
            raise SystemExit(f"no checkpoints found under {root} — pass --checkpoint explicitly")
        run_dir = max(candidates, key=os.path.getmtime)

    pts = glob.glob(os.path.join(run_dir, "checkpoint_*.pt"))
    if not pts:
        raise SystemExit(f"no checkpoint_*.pt files in {run_dir}")
    latest = max(pts, key=_checkpoint_step)
    print(f"Auto-selected latest checkpoint: {latest} (step {_checkpoint_step(latest)})")
    return latest


def load_network(checkpoint_path: str, game: ChessGame, config: MuZeroConfig, device: str):
    torch.serialization.add_safe_globals([MuZeroConfig])
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = ckpt["model_state_dict"]

    # Detect whether the checkpoint was trained with the EZ consistency head
    # (projection + prediction_head). Back-compat with pre-EZ chess checkpoints.
    has_consistency = any(k.startswith("projection.") for k in state_dict)

    has_inverse = any(k.startswith("inverse_dynamics_head.") for k in state_dict)

    # Detect the policy head type from the weights so from_to, conv and flat
    # checkpoints all load: from_to (relational) has policy_head.q_proj/k_proj,
    # the AlphaZero conv head has policy_head.mix/proj, the flat (AlphaGo) head
    # is an nn.Sequential (policy_head.0/.1/.4).
    if any(".policy_head.q_proj." in k or ".policy_head.k_proj." in k for k in state_dict):
        policy_head_type = "from_to"
    elif any(".policy_head.mix." in k or ".policy_head.proj." in k for k in state_dict):
        policy_head_type = "conv"
    else:
        policy_head_type = "flat"

    # Reward-head planes: the 1x1 projection width (historically 1; widened for the
    # mate-beacon head). Recover from the first conv so 8-plane checkpoints load.
    reward_head_planes = getattr(config, "reward_head_planes", 1)
    rw = state_dict.get("dynamics.reward_head.0.weight")
    if rw is not None:
        reward_head_planes = int(rw.shape[0])

    # Moves-left head (Lc0): detect so moves-head checkpoints load.
    has_moves_left = any(k.startswith("moves_left_head.") for k in state_dict)

    # Material head (decisive-signal aux head): detect so material-head
    # checkpoints load. Recover the support size (2K+1) from the output proj.
    has_material = any(k.startswith("material_head.") for k in state_dict)
    material_support = getattr(config, "material_head_support_size", 8)
    if has_material:
        outw = next((v for k, v in state_dict.items()
                     if k.startswith("material_head.") and v.ndim == 2
                     and v.shape[0] % 2 == 1 and v.shape[0] < v.shape[1] * 4), None)
        if outw is not None:
            material_support = (outw.shape[0] - 1) // 2

    network = MuZeroNetwork(
        # Input is the T-frame history stack (num_planes * history_frames), matching training.
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
        reward_head_planes=reward_head_planes,
        action_embed_dim=getattr(config, "action_embed_dim", 16),
        use_consistency_loss=has_consistency,
        proj_hid=config.proj_hid,
        proj_out=config.proj_out,
        pred_hid=config.pred_hid,
        pred_out=config.pred_out,
        use_scalar_transform=config.use_scalar_transform,
        value_target_scale=config.value_target_scale,
        # Detect from the checkpoint so old (support, no-inverse) and current
        # (WDL + inverse head) checkpoints both load.
        value_head_type=getattr(config, "value_head_type", "support"),
        draw_score=getattr(config, "draw_score", 0.0),
        policy_head_type=policy_head_type,
        use_moves_left=has_moves_left,
        moves_left_support_size=getattr(config, "moves_left_support_size", 10),
        use_inverse_dynamics_loss=has_inverse,
        inverse_dynamics_hidden=getattr(config, "inverse_dynamics_hidden", 256),
        use_material_head=has_material,
        material_head_support_size=material_support,
        # Head/body shape params — needed for XL/attention checkpoints (from config preset).
        value_head_planes=getattr(config, "value_head_planes", 1),
        value_head_blocks=getattr(config, "value_head_blocks", 0),
        moves_left_head_planes=getattr(config, "moves_left_head_planes", 1),
        moves_left_head_blocks=getattr(config, "moves_left_head_blocks", 0),
        use_repr_attention=getattr(config, "use_repr_attention", False),
        use_dyn_attention=getattr(config, "use_dyn_attention", False),
        use_pred_attention=getattr(config, "use_pred_attention", False),
        use_smolgen=getattr(config, "use_smolgen", True),
        attn_layers=getattr(config, "attn_layers", 4),
        attn_heads=getattr(config, "attn_heads", 4),
        pred_attn_layers=getattr(config, "pred_attn_layers", 2),
        hybrid_stem_blocks=getattr(config, "hybrid_stem_blocks", 0),
    )
    network.load_state_dict(state_dict)
    network.to(device)
    network.eval()
    return network


def main():
    global session, sf_engine
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="Path to a chess checkpoint .pt file. If omitted, auto-selects the "
                             "latest checkpoint (highest step) of --run-id, or of the most "
                             "recently active run under checkpoints/chess/.")
    parser.add_argument("--run-id", default=None,
                        help="Run id under checkpoints/chess/ to pull the latest checkpoint from "
                             "(ignored if --checkpoint is given).")
    parser.add_argument("--game", default="chess",
                        help="Config preset used to size the network (e.g. chess, chess_small). "
                             "Must match the run that produced the checkpoint.")
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
    parser.add_argument("--stockfish", default="tools/stockfish/stockfish",
                        help="Stockfish binary for the depth-8 hint panel "
                             "(hints disabled if missing).")
    args = parser.parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    config = get_config(args.game)
    if args.use_gumbel is not None:
        config.use_gumbel = args.use_gumbel
    if args.num_simulations is not None:
        config.num_simulations = args.num_simulations
    checkpoint = resolve_checkpoint(args.checkpoint, args.run_id)
    game = ChessGame()
    network = load_network(checkpoint, game, config, device)
    session = GameSession(game, network, config, device)

    try:
        sf_engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
        print(f"SF hint engine: {args.stockfish} (depth 8, MultiPV 3)")
    except Exception as e:
        sf_engine = None
        print(f"SF hint engine unavailable ({e}) — hint panel disabled")

    print(f"Open http://{args.host}:{args.port} to play "
          f"[use_gumbel={config.use_gumbel}, num_simulations={config.num_simulations}]")
    # threaded=False to serialize MCTS calls (single user, avoid races on session state)
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False, threaded=False)


if __name__ == "__main__":
    main()
