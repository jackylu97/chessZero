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
import numpy as np
import torch  # required to unpickle GameHistory (observations are torch.Tensor)  # noqa: F401
from flask import Flask, jsonify

from src.games import GAME_REGISTRY
from src.games.chess import _action_to_move
from src.training.replay_buffer import ReplayBuffer


@dataclass
class GameRecord:
    idx: int                  # original index in buffer
    outcome: float
    num_plies: int
    captures: int
    checks: int
    fens: list[str]           # length = num_plies + 1 (includes start position)
    sans: list[str]           # length = num_plies
    is_selfplay: bool = True   # False => warmstart (external_values populated)
    draw_by_repetition: bool = False
    draw_by_no_progress: bool = False
    reanalyze_count: int = 0   # 0 => policy targets match the net-at-generation-time
    # Per-ply diagnostics (length = num_plies, aligned to fens[i]/the position
    # BEFORE move i). root_values are STM-POV MCTS root values; policy_top is the
    # top-3 (san, prob) of the stored policy target at each position.
    root_values: list[float] = None
    policy_top: list[list] = None


def _outcome_label(o: float) -> str:
    if o > 0.5: return "1-0"
    if o < -0.5: return "0-1"
    return "½-½"


def _policy_top(pol, board, k: int = 3) -> list:
    """Top-k (san, prob) of a policy target at ``board`` (the position to move).

    Policies are stored sparse as an ``(indices, values)`` tuple (Path B); dense
    arrays are still accepted for back-compat with older buffers.
    """
    if pol is None or len(pol) == 0:
        return []
    if isinstance(pol, tuple):
        idx = np.asarray(pol[0], dtype=np.int64)
        val = np.asarray(pol[1], dtype=np.float64)
    else:
        val = np.asarray(pol, dtype=np.float64)
        idx = np.arange(val.shape[0])
    if len(val) == 0:
        return []
    order = np.argsort(val)[::-1][:k]
    out = []
    for j in order:
        a = int(idx[j])
        p = float(val[j])
        if p <= 0.0:
            break
        mv = _action_to_move(a, board)
        if mv is not None and mv in board.legal_moves:
            san, uci = board.san(mv), mv.uci()
        else:
            san, uci = f"?{a}", None
        out.append([san, round(p, 3), uci])  # uci drives the board arrows
    return out


def decode_game(game, idx: int) -> GameRecord:
    board = chess.Board()
    # Endgame-seed / other FEN-start games begin off the standard position.
    start_fen = getattr(game, "start_fen", None)
    if start_fen:
        board = chess.Board(start_fen)
    fens = [board.fen()]
    sans: list[str] = []
    captures = 0
    checks = 0
    root_values: list[float] = []
    policy_top: list[list] = []
    rv = list(getattr(game, "root_values", []) or [])
    pols = list(getattr(game, "policies", []) or [])
    for i, act in enumerate(game.actions):
        root_values.append(float(rv[i]) if i < len(rv) else None)
        policy_top.append(_policy_top(pols[i] if i < len(pols) else None, board))
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
        is_selfplay=len(getattr(game, "external_values", []) or []) == 0,
        draw_by_repetition=bool(getattr(game, "draw_by_repetition", False)),
        draw_by_no_progress=bool(getattr(game, "draw_by_no_progress", False)),
        reanalyze_count=int(getattr(game, "reanalyze_count", 0)),
        root_values=root_values,
        policy_top=policy_top,
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
  .boardwrap { position: relative; width: 480px; }
  #arrows { position: absolute; left: 0; top: 0; z-index: 10; pointer-events: none; }
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
  .plyinfo { font-family: ui-monospace, monospace; font-size: 13px; background: #f5f5f5; padding: 10px; border-radius: 4px; margin-top: 8px; }
  .vbar { height: 14px; background: linear-gradient(90deg,#c62828,#ddd,#2e7d32); border-radius: 3px; position: relative; margin: 4px 0 8px; }
  .vbar .mark { position: absolute; top: -3px; width: 3px; height: 20px; background: #111; }
  .ptarget span { display: inline-block; margin-right: 10px; }
  .tag { font-size: 11px; padding: 1px 6px; border-radius: 8px; color: white; margin-left: 4px; }
  .tag.sp { background: #1565c0; } .tag.ws { background: #6a1b9a; }
  .tag.rep { background: #c62828; } .tag.np { background: #ef6c00; }
  .tag.ra { background: #00838f; } .tag.fresh { background: #558b2f; }
</style>
</head>
<body>
<h1>ChessZero — Buffer Viewer</h1>
<p class="meta" id="bufmeta">Loading...</p>
<div>
  <label for="filterSel"><b>Show:</b></label>
  <select id="filterSel">
    <option value="selfplay">Self-play only</option>
    <option value="all">All games</option>
    <option value="warmstart">Warmstart only</option>
  </select>
  &nbsp;&nbsp;
  <label for="gameSel"><b>Game:</b></label>
  <select id="gameSel"></select>
</div>
<div class="layout">
  <div>
    <div class="boardwrap"><div id="board"></div><svg id="arrows" xmlns="http://www.w3.org/2000/svg"></svg></div>
    <div class="controls">
      <button onclick="step(-999999)">⏮</button>
      <button onclick="step(-1)">◀</button>
      <button onclick="step(1)">▶</button>
      <button onclick="step(999999)">⏭</button>
      <input type="range" id="slider" min="0" max="0" value="0">
      <span id="plyLabel">0 / 0</span>
    </div>
    <div class="gameinfo" id="gameinfo"></div>
    <div class="plyinfo" id="plyinfo"></div>
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
let currentGame = null;  // {idx, outcome, captures, checks, fens, sans, root_values, policy_top}
let ply = 0;
let allGames = [];       // full list from /games (each carries a sortedIdx into RECORDS)

function fmtOutcome(o) {
    if (o > 0.5) return '1-0 (White wins)';
    if (o < -0.5) return '0-1 (Black wins)';
    return '½-½ (draw)';
}

function gameTag(g) {
    let t = g.is_selfplay ? '<span class="tag sp">self-play</span>'
                          : '<span class="tag ws">warmstart</span>';
    if (g.draw_by_repetition) t += '<span class="tag rep">3-fold</span>';
    if (g.draw_by_no_progress) t += '<span class="tag np">75-move</span>';
    if (g.is_selfplay) {
        t += (g.reanalyze_count > 0)
            ? `<span class="tag ra">reanalyzed ${g.reanalyze_count}x</span>`
            : '<span class="tag fresh">not reanalyzed</span>';
    }
    return t;
}

function renderPly() {
    const g = currentGame;
    const el = document.getElementById('plyinfo');
    // root_values/policy_top are aligned to the position BEFORE move `ply` (i.e. fens[ply]).
    if (ply >= g.num_plies) { el.innerHTML = '<i>terminal position</i>'; return; }
    const v = (g.root_values && g.root_values[ply] != null) ? g.root_values[ply] : null;
    const stm = (ply % 2 === 0) ? 'White' : 'Black';
    let html = '';
    if (v != null) {
        const pct = Math.max(0, Math.min(100, (v + 1) * 50));
        html += `<div><b>Root value</b> (${stm} to move, STM-POV): <b>${v >= 0 ? '+' : ''}${v.toFixed(3)}</b></div>`;
        html += `<div class="vbar"><div class="mark" style="left:${pct}%"></div></div>`;
    }
    const pt = (g.policy_top && g.policy_top[ply]) ? g.policy_top[ply] : [];
    if (pt.length) {
        html += '<div class="ptarget"><b>Policy target:</b> ' +
            pt.map(([san, p]) => `<span>${san} ${(p*100).toFixed(0)}%</span>`).join('') + '</div>';
    }
    el.innerHTML = html || '<i>no per-ply data</i>';
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
        `<b>Game ${g.idx}</b> — ${fmtOutcome(g.outcome)} · ${g.num_plies} plies · ` +
        `${g.captures} captures · ${g.checks} checks ${gameTag(g)}`;
}

const SVGNS = 'http://www.w3.org/2000/svg';

function squareCenter(sq, originRect) {
    const el = document.querySelector('#board [data-square="' + sq + '"]');
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return { x: r.left + r.width / 2 - originRect.left, y: r.top + r.height / 2 - originRect.top };
}

function addArrow(svg, x1, y1, x2, y2, width, opacity) {
    const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1;
    const ux = dx / len, uy = dy / len, px = -uy, py = ux;
    const head = Math.max(12, width * 2.0);
    const ex = x2 - ux * head, ey = y2 - uy * head;  // base of the arrowhead
    const hw = head * 0.6, color = '#1565c0';
    const line = document.createElementNS(SVGNS, 'line');
    line.setAttribute('x1', x1); line.setAttribute('y1', y1);
    line.setAttribute('x2', ex); line.setAttribute('y2', ey);
    line.setAttribute('stroke', color); line.setAttribute('stroke-width', width);
    line.setAttribute('stroke-linecap', 'round'); line.setAttribute('opacity', opacity);
    svg.appendChild(line);
    const poly = document.createElementNS(SVGNS, 'polygon');
    poly.setAttribute('points', `${x2},${y2} ${ex + px * hw},${ey + py * hw} ${ex - px * hw},${ey - py * hw}`);
    poly.setAttribute('fill', color); poly.setAttribute('opacity', opacity);
    svg.appendChild(poly);
}

function drawArrows() {
    const svg = document.getElementById('arrows');
    const br = document.getElementById('board').getBoundingClientRect();
    svg.setAttribute('width', br.width); svg.setAttribute('height', br.height);
    svg.style.width = br.width + 'px'; svg.style.height = br.height + 'px';
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    if (!currentGame || ply >= currentGame.num_plies) return;
    const origin = svg.getBoundingClientRect();
    const pt = (currentGame.policy_top && currentGame.policy_top[ply]) || [];
    // weakest first so the highest-probability arrow renders on top
    for (const entry of pt.slice().reverse()) {
        const prob = entry[1], uci = entry[2];
        if (!uci) continue;
        const from = squareCenter(uci.slice(0, 2), origin);
        const to = squareCenter(uci.slice(2, 4), origin);
        if (!from || !to) continue;
        addArrow(svg, from.x, from.y, to.x, to.y, 3 + prob * 14, 0.3 + prob * 0.6);
    }
}

function setPly(p) {
    if (!currentGame) return;
    ply = Math.max(0, Math.min(currentGame.fens.length - 1, p));
    board.position(currentGame.fens[ply], false);
    document.getElementById('slider').value = ply;
    document.getElementById('plyLabel').textContent = `${ply} / ${currentGame.num_plies}`;
    renderMoves();
    renderPly();
    drawArrows();
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
    // Tag each game with its index into the server-side (sorted) RECORDS list.
    allGames = data.games.map((g, i) => ({ ...g, sortedIdx: i }));
    const nSP = allGames.filter(g => g.is_selfplay).length;
    document.getElementById('bufmeta').textContent =
        `${data.buffer_path} — ${data.num_games} games (${nSP} self-play, ${data.num_games - nSP} warmstart), ` +
        `sorted by (captures asc, length asc)`;
    const sel = document.getElementById('gameSel');
    const filterSel = document.getElementById('filterSel');

    function populate() {
        const f = filterSel.value;
        const shown = allGames.filter(g =>
            f === 'all' || (f === 'selfplay' && g.is_selfplay) || (f === 'warmstart' && !g.is_selfplay));
        sel.innerHTML = '';
        shown.forEach(g => {
            const opt = document.createElement('option');
            opt.value = g.sortedIdx;
            const flags = (g.draw_by_repetition ? ' 3fold' : '') + (g.draw_by_no_progress ? ' 75mv' : '');
            opt.textContent = `orig ${g.idx} · ${g.captures} caps · ${g.num_plies} plies · ${fmtOutcome(g.outcome)}${flags}`;
            sel.appendChild(opt);
        });
        if (shown.length) loadGame(parseInt(sel.value, 10));
        else { document.getElementById('gameinfo').innerHTML = '<i>no games match this filter</i>'; }
    }

    filterSel.addEventListener('change', populate);
    sel.addEventListener('change', () => loadGame(parseInt(sel.value, 10)));
    populate();
    document.getElementById('slider').addEventListener('input', (e) => setPly(parseInt(e.target.value, 10)));
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
        if (e.key === 'ArrowLeft') step(-1);
        else if (e.key === 'ArrowRight') step(1);
        else if (e.key === 'Home') step(-999999);
        else if (e.key === 'End') step(999999);
    });
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
             "captures": r.captures, "checks": r.checks,
             "is_selfplay": r.is_selfplay,
             "draw_by_repetition": r.draw_by_repetition,
             "draw_by_no_progress": r.draw_by_no_progress,
             "reanalyze_count": r.reanalyze_count}
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
        "is_selfplay": r.is_selfplay,
        "draw_by_repetition": r.draw_by_repetition,
        "draw_by_no_progress": r.draw_by_no_progress,
        "reanalyze_count": r.reanalyze_count,
        "root_values": r.root_values, "policy_top": r.policy_top,
    })


def load_or_build_records(buffer_path: str, force_rebuild: bool = False,
                          max_games: int | None = None) -> list[GameRecord]:
    """Return decoded GameRecords for a buffer, using a sidecar cache when fresh.

    Cache path: `<buffer_path>.viewer.pkl`. Rebuilt if missing, older than the
    buffer, or if --rebuild-cache is passed. ``max_games`` decodes only the last
    N games (most recent), which skips the expensive per-game replay of the whole
    buffer — the file is still fully unpickled, but decoding dominates. Subsets
    get their own cache so they don't clobber the full one.
    """
    suffix = f".m{max_games}" if max_games else ""
    cache_path = buffer_path + f".viewer4{suffix}.pkl"  # v4: policy_top carries uci (for arrows)
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
    # Route through ReplayBuffer.load so both supported formats work:
    # v2 streaming full GameHistory, v3 streaming compact (requires a game
    # for action replay).
    rb = ReplayBuffer(max_size=10_000_000)
    rb.load(buffer_path, game=GAME_REGISTRY["chess"]())
    buf = rb.buffer
    if max_games and len(buf) > max_games:
        buf = buf[-max_games:]  # most-recent N games
        print(f"  --max-games {max_games}: keeping last {len(buf)} of {len(rb.buffer)} games")
    print(f"  unpickled in {time.time()-t0:.1f}s, {len(buf)} games. Decoding moves...")
    t1 = time.time()
    recs = [decode_game(g, i) for i, g in enumerate(buf)]
    del buf, rb
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
    parser.add_argument("--max-games", type=int, default=None,
                        help="Only decode the last N (most-recent) games — much faster "
                             "first-load on big buffers.")
    args = parser.parse_args()
    BUFFER_PATH = args.buffer

    recs = load_or_build_records(args.buffer, force_rebuild=args.rebuild_cache,
                                 max_games=args.max_games)
    recs.sort(key=lambda r: (r.captures, r.num_plies))
    RECORDS = recs
    print(f"  Sorted {len(RECORDS)} games by (captures asc, length asc). "
          f"First: {RECORDS[0].captures} caps, {RECORDS[0].num_plies} plies. "
          f"Last: {RECORDS[-1].captures} caps, {RECORDS[-1].num_plies} plies.")

    print(f"Open http://{args.host}:{args.port} to browse games")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False, threaded=False)


if __name__ == "__main__":
    main()
