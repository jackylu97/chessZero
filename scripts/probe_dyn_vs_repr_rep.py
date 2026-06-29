"""The decisive measurement: at the rep-completion move, compare
  (a) DYNAMICS value of the resulting position (what MCTS backs up for the rep move)
  (b) REPRESENTATION value of the TRUE resulting board WITH full history+rep planes
      (what the value head would say next ply, knowing it's a 3fold)
If (a) >> (b), the search is blind to a repetition penalty the value head DOES
encode -> the rep move is over-valued in search by exactly that gap. This is the
mechanism that banks a phantom advantage into the draw.
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

CKPT="checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.pt"
BUF="checkpoints/chess/2026_06_19_cold2_pc/checkpoint_6000.buf"

def main():
    np.random.seed(0); torch.manual_seed(0)
    cfg=get_config("chess_small"); cfg.device="cpu"
    game=ChessGame(); HF=cfg.history_frames; ds=cfg.draw_score
    ckpt=torch.load(CKPT,map_location="cpu",weights_only=True)
    net=build_network(ckpt,game,cfg,"cpu")
    rb=ReplayBuffer(max_size=200000); rb.load(BUF,game=game)
    sp=[g for g in rb.buffer if not g.external_values and g.draw_by_repetition]

    @torch.no_grad()
    def V_repr(obs):
        h=net.representation(obs.unsqueeze(0)); _,vl=net.prediction(h)
        return wdl_to_scalar(vl.float(),draw_score=ds).item()

    @torch.no_grad()
    def Q_dyn(root_obs, action):
        h,_,_=net.initial_inference(root_obs.unsqueeze(0))
        nh,r,pl,v=net.recurrent_inference(h,torch.tensor([action]))
        # mover-POV Q at root for this move = r - gamma*v(child, child-POV)
        return float(r.item()-cfg.discount*v.item())

    rows=[]
    n_done=0
    for g in sp:
        st=game.reset(); singles=[]; ok=True
        for a in g.actions:
            singles.append(game.to_tensor(st))
            try: st,_,_=game.step(st,int(a))
            except Exception: ok=False;break
        if not ok: continue
        n=len(singles)
        if n<HF+2: continue
        # rep completed at last move: action g.actions[n-1] from position n-1
        root_ply=n-1
        root_obs=stack_with_history(singles[root_ply], singles[:root_ply], HF)
        rep_action=int(g.actions[root_ply])
        # (a) dynamics Q for the actual rep move (mover-POV at root)
        q_dyn=Q_dyn(root_obs, rep_action)
        # (b) repr value of the TRUE resulting position (the final terminal obs)
        #     This obs (final) has rep3 plane set. Value is child-POV; negate to root-mover POV.
        # Build final obs with history.
        final_single = g.observations[-1] if len(g.observations)>n else None
        # reconstruct resulting position obs by stepping engine one more time:
        # we already have st = state AFTER all actions (terminal). Use its tensor.
        final_obs = stack_with_history(game.to_tensor(st), singles, HF)
        v_child = V_repr(final_obs)             # child-POV value of the drawn position
        q_repr = -cfg.discount*v_child          # root-mover POV (reward 0 for non-mate)
        rows.append((q_dyn, q_repr, v_child))
        n_done+=1
        if n_done>=120: break
    rows=np.array(rows)
    qd,qr,vc=rows[:,0],rows[:,1],rows[:,2]
    print(f"n={len(rows)} rep-completion moves")
    print(f"  Q_dyn(rep move, mover-POV)   mean={qd.mean():+.3f} std={qd.std():.3f}   "
          f"[what MCTS backs up — dynamics, rep-blind]")
    print(f"  Q_repr(true resulting pos)   mean={qr.mean():+.3f} std={qr.std():.3f}   "
          f"[value head ON the rep position, rep planes SET]")
    print(f"  V_repr(child, rep pos)       mean={vc.mean():+.3f}   "
          f"[raw value head of the drawn position itself]")
    print(f"  GAP  Q_dyn - Q_repr          mean={(qd-qr).mean():+.3f}  "
          f"(positive = search OVER-values the rep move vs what the value head knows)")
    print(f"  fraction Q_dyn > Q_repr+0.05: {(qd>qr+0.05).mean()*100:.0f}%")
    print(f"  fraction Q_dyn > 0.05 (search thinks rep move WINS): {(qd>0.05).mean()*100:.0f}%")
    print(f"  fraction Q_repr near 0 (|.|<0.1, head sees ~draw):   {(np.abs(qr)<0.1).mean()*100:.0f}%")

if __name__=="__main__":
    main()
