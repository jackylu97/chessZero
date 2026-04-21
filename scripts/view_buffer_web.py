"""Browse games from a saved replay buffer in the browser.

Usage:
    python scripts/view_buffer_web.py --buffer checkpoints/chess/<run-id>/checkpoint_11000.buf

Open http://127.0.0.1:5000. Select a game from the dropdown; use the slider or
arrow keys to cycle through moves. Games are sorted by (captures asc, length asc)
so the most pathological shuffle-draws surface first.
"""

import argparse
import gc
import os
import pickle
import sys
import time
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import torch  # required to unpickle GameHistory (observations are torch.Tensor)  # noqa: F401
from flask import Flask, jsonify

from src.games.chess import _action_to_move


@dataclass
class GameRecord:
    idx: int                  # original index in buffer
    outcome: float
    num_plies: int
    captures: int
    checks: int
    fens: list[str]           # length = num_plies + 1 (includes start position)
    sans: list[str]           # length = num_plies


def _outcome_label(o: float) -> str:
    if o > 0.5: return "1-0"
    if o < -0.5: return "0-1"
    return "½-½"


def decode_game(game, idx: int) -> GameRecord:
    board = chess.Board()
    fens = [board.fen()]
    sans: list[str] = []
    captures = 0
    checks = 0
    for act in game.actions:
        move = _action_to_move(act, board)
        if move is None or move not in board.legal_moves:
            sans.append(f"<illegal:{act}>")
            fens.append(board.fen())
            break
        if board.is_capture(move):
            captures += 1
        if board.gives_check(move):
            checks += 1
        sans.append(board.san(move))
        board.push(move)
        fens.append(board.fen())
    return GameRecord(
        idx=idx,
        outcome=float(game.game_outcome),
        num_plies=len(sans),
        captures=captures,
        checks=checks,
        fens=fens,
        sans=sans,
    )


HTML = """<!DOCTYPE html>
<html>
<head>
<title>ChessZero — Buffer Viewer</title>
<link rel="stylesheet" href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css">
<style>
  body { font-family: -apple-system, system-ui, sans-serif; max-width: 900px; margin: 20px auto; padding: 0 20px; color: #222; }
  h1 { margin: 4px 0; }
  .meta { color: #666; font-size: 13px; margin-bottom: 12px; }
  .layout { display: grid; grid-template-columns: 480px 1fr; gap: 24px; }
  #board { width: 480px; }
  .controls { margin: 12px 0; display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
  .controls button { padding: 6px 12px; font-size: 14px; cursor: pointer; border: 1px solid #bbb; background: white; border-radius: 4px; }
  .controls button:hover { background: #f0f0f0; }
  #slider { flex: 1; min-width: 160px; }
  #plyLabel { font-family: ui-monospace, monospace; font-size: 13px; min-width: 90px; }
  select { padding: 6px; font-size: 14px; max-width: 100%; }
  #moves { font-family: ui-monospace, monospace; font-size: 13px; background: #f5f5f5; padding: 10px; border-radius: 4px; max-height: 520px; overflow-y: auto; }
  #moves .ply { display: inline-block; padding: 1px 4px; border-radius: 3px; cursor: pointer; }
  #moves .ply.current { background: #ffd54f; font-weight: bold; }
  #moves .ply:hover { background: #eee; }
  #moves .num { color: #888; }
  .gameinfo { font-size: 13px; margin-bottom: 10px; color: #333; }
  .gameinfo b { font-weight: 600; }
</style>
</head>
<body>
<h1>ChessZero — Buffer Viewer</h1>
<p class="meta" id="bufmeta">Loading...</p>
<div>
  <label for="gameSel"><b>Game:</b></label>
  <select id="gameSel"></select>
</div>
<div class="layout">
  <div>
    <div id="board"></div>
    <div class="controls">
      <button onclick="step(-999999)">⏮</button>
      <button onclick="step(-1)">◀</button>
      <button onclick="step(1)">▶</button>
      <button onclick="step(999999)">⏭</button>
      <input type="range" id="slider" min="0" max="0" value="0">
      <span id="plyLabel">0 / 0</span>
    </div>
    <div class="gameinfo" id="gameinfo"></div>
  </div>
  <div>
    <h3 style="margin-top: 0;">Moves</h3>
    <div id="moves"></div>
  </div>
</div>

<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
<script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
<script>
let board = null;
let currentGame = null;  // {idx, outcome, captures, checks, fens, sans}
let ply = 0;

function fmtOutcome(o) {
    if (o > 0.5) return '1-0 (White wins)';
    if (o < -0.5) return '0-1 (Black wins)';
    return '½-½ (draw)';
}

function renderMoves() {
    const el = document.getElementById('moves');
    const parts = [];
    for (let i = 0; i < currentGame.sans.length; i++) {
        if (i % 2 === 0) parts.push(`<span class="num">${i/2 + 1}.</span>`);
        const cls = (i + 1 === ply) ? 'ply current' : 'ply';
        parts.push(`<span class="${cls}" data-ply="${i+1}">${currentGame.sans[i]}</span>`);
    }
    el.innerHTML = parts.join(' ');
    el.querySelectorAll('.ply').forEach(e => {
        e.addEventListener('click', () => setPly(parseInt(e.dataset.ply, 10)));
    });
    const cur = el.querySelector('.ply.current');
    if (cur) cur.scrollIntoView({ block: 'nearest' });
}

function renderInfo() {
    const g = currentGame;
    document.getElementById('gameinfo').innerHTML =
        `<b>Game ${g.idx}</b> — ${fmtOutcome(g.outcome)} · ${g.num_plies} plies · ${g.captures} captures · ${g.checks} checks`;
}

function setPly(p) {
    if (!currentGame) return;
    ply = Math.max(0, Math.min(currentGame.fens.length - 1, p));
    board.position(currentGame.fens[ply], false);
    document.getElementById('slider').value = ply;
    document.getElementById('plyLabel').textContent = `${ply} / ${currentGame.num_plies}`;
    renderMoves();
}

function step(delta) { setPly(ply + delta); }

async function loadGame(sortedIdx) {
    const resp = await fetch(`/game/${sortedIdx}`);
    currentGame = await resp.json();
    ply = 0;
    document.getElementById('slider').max = currentGame.fens.length - 1;
    renderInfo();
    setPly(0);
}

async function init() {
    board = Chessboard('board', {
        position: 'start',
        pieceTheme: 'https://cdn.jsdelivr.net/gh/oakmac/chessboardjs@master/website/img/chesspieces/wikipedia/{piece}.png',
    });
    const listResp = await fetch('/games');
    const data = await listResp.json();
    document.getElementById('bufmeta').textContent =
        `${data.buffer_path} — ${data.num_games} games, sorted by (captures asc, length asc)`;
    const sel = document.getElementById('gameSel');
    data.games.forEach((g, i) => {
        const opt = document.createElement('option');
        opt.value = i;
        opt.textContent = `#${i} — orig ${g.idx} · ${g.captures} caps · ${g.num_plies} plies · ${fmtOutcome(g.outcome)}`;
        sel.appendChild(opt);
    });
    sel.addEventListener('change', () => loadGame(parseInt(sel.value, 10)));
    document.getElementById('slider').addEventListener('input', (e) => setPly(parseInt(e.target.value, 10)));
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
        if (e.key === 'ArrowLeft') step(-1);
        else if (e.key === 'ArrowRight') step(1);
        else if (e.key === 'Home') step(-999999);
        else if (e.key === 'End') step(999999);
    });
    if (data.games.length) await loadGame(0);
}

window.addEventListener('DOMContentLoaded', init);
</script>
</body>
</html>
"""


app = Flask(__name__)
RECORDS: list[GameRecord] = []
BUFFER_PATH = ""


@app.route("/")
def index():
    return HTML


@app.route("/games")
def games():
    return jsonify({
        "buffer_path": BUFFER_PATH,
        "num_games": len(RECORDS),
        "games": [
            {"idx": r.idx, "outcome": r.outcome, "num_plies": r.num_plies,
             "captures": r.captures, "checks": r.checks}
            for r in RECORDS
        ],
    })


@app.route("/game/<int:i>")
def game(i: int):
    r = RECORDS[i]
    return jsonify({
        "idx": r.idx, "outcome": r.outcome, "num_plies": r.num_plies,
        "captures": r.captures, "checks": r.checks,
        "fens": r.fens, "sans": r.sans,
    })


def load_or_build_records(buffer_path: str, force_rebuild: bool = False) -> list[GameRecord]:
    """Return decoded GameRecords for a buffer, using a sidecar cache when fresh.

    Cache path: `<buffer_path>.viewer.pkl`. Rebuilt if missing, older than the
    buffer, or if --rebuild-cache is passed.
    """
    cache_path = buffer_path + ".viewer.pkl"
    if not force_rebuild and os.path.exists(cache_path):
        if os.path.getmtime(cache_path) >= os.path.getmtime(buffer_path):
            print(f"Loading cache {cache_path} ...")
            t0 = time.time()
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            print(f"  {len(cached)} games loaded from cache in {time.time()-t0:.1f}s")
            return [GameRecord(**d) for d in cached]
        print("  cache is older than buffer, rebuilding")

    print(f"Loading {buffer_path} ({os.path.getsize(buffer_path)/1e9:.1f} GB) ...")
    t0 = time.time()
    with open(buffer_path, "rb") as f:
        data = pickle.load(f)
    buf = data["buffer"]
    print(f"  unpickled in {time.time()-t0:.1f}s, {len(buf)} games. Decoding moves...")
    t1 = time.time()
    recs = [decode_game(g, i) for i, g in enumerate(buf)]
    del buf, data
    gc.collect()
    print(f"  decoded in {time.time()-t1:.1f}s")

    print(f"Writing cache to {cache_path} ...")
    with open(cache_path, "wb") as f:
        pickle.dump([asdict(r) for r in recs], f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  cache size: {os.path.getsize(cache_path)/1e6:.1f} MB")
    return recs


def main():
    global RECORDS, BUFFER_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer", required=True)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Ignore existing viewer cache and rebuild from the buffer.")
    args = parser.parse_args()
    BUFFER_PATH = args.buffer

    recs = load_or_build_records(args.buffer, force_rebuild=args.rebuild_cache)
    recs.sort(key=lambda r: (r.captures, r.num_plies))
    RECORDS = recs
    print(f"  Sorted {len(RECORDS)} games by (captures asc, length asc). "
          f"First: {RECORDS[0].captures} caps, {RECORDS[0].num_plies} plies. "
          f"Last: {RECORDS[-1].captures} caps, {RECORDS[-1].num_plies} plies.")

    print(f"Open http://{args.host}:{args.port} to browse games")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False, threaded=False)


if __name__ == "__main__":
    main()
