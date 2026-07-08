"""Two checks on the rep-draw mechanism, one buffer load:

A) POV consistency of the RAW network value across consecutive plies in a
   quiescent rep endgame. In zero-sum, V(stm at ply t) should ~= -V(stm at t+1)
   for a quiet move. Report the correlation / the (Veven, Vodd) asymmetry.

B) Rep-plane sensitivity: at the rep-completion position, recompute the network
   value with rep planes (19,20) zeroed vs as-is. If V barely changes, the value
   head ignores the repetition signal (can't see the draw it is walking into).
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
    game=ChessGame(); HF=cfg.history_frames; ds=cfg.draw_score; C=game.num_planes
    ckpt=torch.load(CKPT,map_location="cpu",weights_only=True)
    net=build_network(ckpt,game,cfg,"cpu")
    rb=ReplayBuffer(max_size=200000); rb.load(BUF,game=game)
    sp=[g for g in rb.buffer if not g.external_values and g.draw_by_repetition]
    print(f"{len(sp)} rep games, num_planes={C} HF={HF}")

    @torch.no_grad()
    def Vof(obs):
        h=net.representation(obs.unsqueeze(0)); _,vl=net.prediction(h)
        return wdl_to_scalar(vl.float(),draw_score=ds).item()

    # ---- A: consecutive-ply POV consistency over last 16 plies ----
    pair_sum=[]   # V(t)+V(t+1): should be ~0 if POV-consistent quiet
    rep_plane_delta=[]
    n_done=0
    for g in sp:
        # build obs stream
        st=game.reset(); obs_list=[]; singles=[]
        ok=True
        for a in g.actions:
            singles.append(game.to_tensor(st))
            try: st,_,_=game.step(st,int(a))
            except Exception: ok=False;break
        if not ok: continue
        n=len(singles)
        if n<18: continue
        # full obs with history for last 16 plies
        Vs=[]
        for ply in range(n-16,n):
            obs=stack_with_history(singles[ply], singles[:ply], HF)
            Vs.append(Vof(obs))
        Vs=np.array(Vs)
        for i in range(len(Vs)-1):
            pair_sum.append(Vs[i]+Vs[i+1])
        # ---- B: rep-plane sensitivity at the last ply (rep completed just before) ----
        ply=n-1
        obs=stack_with_history(singles[ply], singles[:ply], HF)
        v_as_is=Vof(obs)
        obs_z=obs.clone()
        # zero rep2/rep3 planes (19,20) in the NEWEST frame (slot 0 = planes 0..C-1)
        # stack layout: [frame0(C), frame1(C), ...]; rep planes are indices 19,20 within each frame
        for fr in range(HF):
            base=fr*C
            obs_z[base+19]=0.0
            obs_z[base+20]=0.0
        v_zeroed=Vof(obs_z)
        rep_plane_delta.append(v_as_is - v_zeroed)
        n_done+=1
        if n_done>=120: break
    pair_sum=np.array(pair_sum); rep_plane_delta=np.array(rep_plane_delta)
    print(f"\n[A] POV consistency (consecutive plies, n={len(pair_sum)} pairs):")
    print(f"    V(t)+V(t+1): mean={pair_sum.mean():+.3f} std={pair_sum.std():.3f} "
          f"(0 = consistent zero-sum quiet)")
    print(f"    |V(t)+V(t+1)|>0.3: {(np.abs(pair_sum)>0.3).mean()*100:.0f}% of consecutive pairs")
    print(f"    mean |V(t)+V(t+1)| = {np.abs(pair_sum).mean():.3f}")
    print(f"\n[B] rep-plane sensitivity (n={len(rep_plane_delta)} rep-completion positions):")
    print(f"    V(as-is) - V(rep-planes-zeroed): mean={rep_plane_delta.mean():+.4f} "
          f"std={rep_plane_delta.std():.4f}")
    print(f"    |delta|>0.05: {(np.abs(rep_plane_delta)>0.05).mean()*100:.0f}%  "
          f"|delta|>0.10: {(np.abs(rep_plane_delta)>0.10).mean()*100:.0f}%")

if __name__=="__main__":
    main()
