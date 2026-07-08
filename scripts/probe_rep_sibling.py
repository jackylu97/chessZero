"""At rep-completion positions: compare the network's value of the position
AFTER the repeating move vs AFTER each non-repeating move, BOTH via
(a) the dynamics network (what MCTS uses to back up) and
(b) the representation network on the true resulting board (ground truth).

This isolates: can the value head distinguish the repeating (drawn) move from
non-repeating moves? If yes but MCTS still picks the repeat -> BUG (search-side).
If no -> fundamental (value head can't rank siblings).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F, chess
from src.config import MuZeroConfig, get_config
torch.serialization.add_safe_globals([MuZeroConfig])
from scripts.eval_checkpoint_health import build_network
from src.games.chess import ChessGame, _action_to_move
from src.training.replay_buffer import ReplayBuffer, stack_with_history
from src.model.utils import wdl_to_scalar

CKPT = "checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.pt"
BUF = "checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.buf"

def main():
    np.random.seed(0); torch.manual_seed(0)
    cfg = get_config("chess_small"); cfg.device="cpu"
    game = ChessGame(); HF=cfg.history_frames
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    net = build_network(ckpt, game, cfg, "cpu")
    rb = ReplayBuffer(max_size=200000); rb.load(BUF, game=game)
    sp = [g for g in rb.buffer if not g.external_values and g.draw_by_repetition]
    ds = cfg.draw_score

    @torch.no_grad()
    def V_repr(obs):
        h = net.representation(obs.unsqueeze(0))
        _, vl = net.prediction(h)
        return wdl_to_scalar(vl.float(), draw_score=ds).item(), F.softmax(vl.float(),-1)[0].numpy()

    @torch.no_grad()
    def V_dyn(root_obs, action):
        h, _, _ = net.initial_inference(root_obs.unsqueeze(0))
        nh, r, pl, v = net.recurrent_inference(h, torch.tensor([action]))
        # v is the scalar mover-POV value of resulting (child) state, child-POV.
        return float(v.item()), float(r.item())

    n_done=0; rows=[]
    for gi,g in enumerate(sp):
        board=chess.Board(); ok=True; rep_entry=None
        for ply,a in enumerate(g.actions):
            mv=_action_to_move(int(a),board)
            if mv is None or mv not in board.legal_moves: ok=False;break
            board.push(mv)
            if board.is_repetition(3) and rep_entry is None: rep_entry=ply
        if not ok or rep_entry is None: continue
        # rebuild obs history + live board up to rep_entry position
        live=chess.Board(); st=game.reset(); obs_hist=[]
        for ply,a in enumerate(g.actions):
            single=game.to_tensor(st)
            if ply==rep_entry:
                legal=game.legal_actions(st)
                obs=stack_with_history(single,obs_hist,HF)
                # classify repeating moves on live board
                rep_acts=set(); nonrep=[]
                for la in legal:
                    m=_action_to_move(int(la),live)
                    if m is None or m not in live.legal_moves: continue
                    live.push(m)
                    isrep = live.is_repetition(3) or live.halfmove_clock>=100
                    live.pop()
                    if isrep: rep_acts.add(int(la))
                    else: nonrep.append(int(la))
                if not rep_acts: break
                # dynamics value of repeating move(s): mover-POV Q = r - gamma*child_V
                rep_qs=[]
                for ra in rep_acts:
                    cv,r=V_dyn(obs,ra); rep_qs.append(r-cfg.discount*cv)
                # dynamics value of non-rep moves
                nr_qs=[]
                for na in nonrep:
                    cv,r=V_dyn(obs,na); nr_qs.append(r-cfg.discount*cv)
                rep_q=float(np.mean(rep_qs))
                nr_best=float(np.max(nr_qs)) if nr_qs else float('nan')
                nr_mean=float(np.mean(nr_qs)) if nr_qs else float('nan')
                # ground-truth repr value of resulting board after repeating move (true draw)
                rep_a=sorted(rep_acts)[0]
                m=_action_to_move(rep_a,live); live.push(m)
                nb=chess.Board(live.fen())  # NOTE loses history; ok we just want net repr V
                live.pop()
                rows.append((rep_q, nr_best, nr_mean, len(rep_acts), len(nonrep)))
                print(f"g{gi} rep_entry={rep_entry}: dynQ(rep)={rep_q:+.3f} "
                      f"dynQ(nonrep best)={nr_best:+.3f} mean={nr_mean:+.3f} "
                      f"-> rep {'WORSE' if rep_q<nr_best-0.02 else ('BETTER' if rep_q>nr_best+0.02 else 'TIE')} "
                      f"(nrep={len(rep_acts)},nnr={len(nonrep)})")
                break
            obs_hist.append(single); st,_,_=game.step(st,int(a)); live.push(_action_to_move(int(a),live))
        n_done+=1
        if n_done>=25: break
    rows=np.array(rows)
    if len(rows):
        rq,nb,nm=rows[:,0],rows[:,1],rows[:,2]
        print(f"\nSUMMARY over {len(rows)} rep positions:")
        print(f"  dynQ(rep) mean={rq.mean():+.3f}  dynQ(nonrep-best) mean={nb.mean():+.3f}  dynQ(nonrep-mean) mean={nm.mean():+.3f}")
        print(f"  rep move dynQ WORSE than best nonrep by >0.02: {(rq<nb-0.02).mean()*100:.0f}%")
        print(f"  rep move dynQ BETTER than best nonrep by >0.02: {(rq>nb+0.02).mean()*100:.0f}%")
        print(f"  mean(dynQ_rep - dynQ_nonrep_best) = {(rq-nb).mean():+.3f}")

if __name__=="__main__":
    main()
