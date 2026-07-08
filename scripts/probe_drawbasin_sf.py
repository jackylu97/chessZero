"""On positions from REPETITION-DRAWN self-play games, compare:
  (a) net predicted V (the value head NOW),
  (b) the training TARGET V (what make_target hands the trainer),
  (c) Stockfish objective eval (ground truth, STM POV).

Classifies the draw collapse:
  - If SF says winning AND net V agrees AND target says draw  => target BUG/limitation
    (the head is RIGHT but is being trained toward draw).
  - If SF says winning AND net V says draw                    => head genuinely can't tell
    (fundamental) OR head has been corrupted by the draw targets.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
import chess, chess.engine

from src.config import MuZeroConfig, get_config
from src.games.chess import ChessGame
from src.training.replay_buffer import ReplayBuffer
from src.model.utils import wdl_to_scalar
from scripts.eval_checkpoint_health import build_network
torch.serialization.add_safe_globals([MuZeroConfig])

CKPT="checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.pt"
BUF ="checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.buf"
SF  ="/usr/games/stockfish"

def sf_eval_stm(eng, board, depth=12):
    info = eng.analyse(board, chess.engine.Limit(depth=depth))
    sc = info["score"].pov(board.turn)
    cp = sc.score(mate_score=10000)
    # squash cp -> [-1,1] like a win prob proxy
    return np.tanh(cp/400.0), cp

def main():
    dev="cpu"; game=ChessGame(); cfg=get_config("chess_small"); HF=cfg.history_frames
    ck=torch.load(CKPT,map_location=dev,weights_only=True); net=build_network(ck,game,cfg,dev)
    rb=ReplayBuffer(max_size=100000); rb.load(BUF,game=game)
    eng=chess.engine.SimpleEngine.popen_uci(SF)

    @torch.no_grad()
    def net_v(g,idx):
        obs=g._stack_history(idx,HF).unsqueeze(0); _,vl=net.prediction(net.representation(obs))
        return wdl_to_scalar(vl.float(),draw_score=cfg.draw_score).item()

    def target_v(g,idx):
        _,_,_,values,_,_=g.make_target(idx,cfg.num_unroll_steps,cfg.td_steps,cfg.discount,
            game.action_space_size,value_head_type=cfg.value_head_type,history_frames=HF,
            eval_to_wdl_alpha=cfg.eval_to_wdl_alpha,eval_to_wdl_beta=cfg.eval_to_wdl_beta,
            q_ratio=cfg.q_ratio,warmstart_q_ratio=cfg.warmstart_q_ratio,selfplay_q_ratio=cfg.selfplay_q_ratio,
            repetition_penalty=cfg.repetition_penalty,repetition_penalty_decay=cfg.repetition_penalty_decay)
        t=np.asarray(values[0]); return t[0]-t[2]+cfg.draw_score*t[1]

    rng=np.random.default_rng(7)
    repgames=[i for i,g in enumerate(rb.buffer) if g.draw_by_repetition and 40<=len(g)<=200 and not g.external_values]
    rng.shuffle(repgames)

    rows=[]
    print(f"{'g':>5} {'ply':>4} {'SFcp':>7} {'SFv':>6} {'netV':>6} {'tgtV':>6}  note")
    for gi in repgames[:40]:
        g=rb.buffer[gi]
        st=game.reset(); boards=[st.board.copy()]
        for a in g.actions:
            st,_,_=game.step(st,a); boards.append(st.board.copy())
        # sample a midgame ply (away from the rep tail) and a pre-rep ply
        for p in [len(g)//3, len(g)//2, max(0,len(g)-12)]:
            b=boards[p]
            if b.is_game_over(): continue
            sfv,cp=sf_eval_stm(eng,b)
            nv=net_v(g,p); tv=target_v(g,p)
            rows.append((gi,p,cp,sfv,nv,tv))
    eng.quit()

    rows=np.array([(r[2],r[3],r[4],r[5]) for r in rows])  # cp,sfv,netv,tgtv
    cp,sfv,netv,tgtv = rows[:,0],rows[:,1],rows[:,2],rows[:,3]
    print(f"\nN={len(rows)} positions from repetition-drawn games")
    print(f"corr(netV, SFv) = {np.corrcoef(netv,sfv)[0,1]:+.3f}")
    print(f"corr(tgtV, SFv) = {np.corrcoef(tgtv,sfv)[0,1]:+.3f}")
    print(f"corr(netV, tgtV)= {np.corrcoef(netv,tgtv)[0,1]:+.3f}")
    # Decisive-by-SF positions (|cp|>=200 ~ winning/losing): what do net and target say?
    win = cp>=200; loss=cp<=-200
    print(f"\nSF-winning positions (cp>=+200): n={win.sum()}")
    if win.sum():
        print(f"   mean netV={netv[win].mean():+.3f}  mean tgtV={tgtv[win].mean():+.3f}  (SF says WIN)")
    print(f"SF-losing positions (cp<=-200): n={loss.sum()}")
    if loss.sum():
        print(f"   mean netV={netv[loss].mean():+.3f}  mean tgtV={tgtv[loss].mean():+.3f}  (SF says LOSS)")
    bal = (np.abs(cp)<100)
    print(f"SF-balanced (|cp|<100): n={bal.sum()}  mean netV={netv[bal].mean():+.3f} tgtV={tgtv[bal].mean():+.3f}")
    # The key bug test: among SF-winning positions, how often does the net AGREE (netV>0.3) but target says draw (tgtV near 0/neg)?
    if win.sum():
        agree_win = (netv[win]>0.3)
        tgt_draw  = (tgtv[win]<0.1)
        both = agree_win & tgt_draw
        print(f"\nBUG-TEST among SF-winning: net agrees(netV>0.3)={agree_win.mean()*100:.0f}%  "
              f"target says draw(tgtV<0.1)={tgt_draw.mean()*100:.0f}%  "
              f"net-right-but-target-draws={both.mean()*100:.0f}%")
if __name__=="__main__":
    main()
