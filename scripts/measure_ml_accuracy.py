"""Measure the moves-left (DTM) head accuracy vs Gaviota DTM ground truth.

Two kinds of accuracy:
  (1) SCALAR  — predicted plies-to-mate vs |Gaviota DTM| on won positions:
                MAE, within +/-2, within +/-4 plies, Spearman corr.
  (2) RANKING — the search-relevant one: for each won position, does the head's
                argmin-predicted-ML child equal a true DTM-optimal child? (== the
                'mlTopOK' the search rides on.) Also pairwise child-order accuracy.

Run: CKPT=... USE_ATTENTION=1 USE_SMOLGEN=1 USE_DYN_ATTENTION=1 [USE_PRED_ATTENTION=1] ATTN_LAYERS=6 \
     PYTHONPATH=. .venv/bin/python scripts/measure_ml_accuracy.py
"""
import os, pickle, numpy as np, torch, chess, chess.gaviota
from collections import defaultdict
from src.config import get_config
from src.games.chess import ChessGame, _move_to_action
from src.games.chess_gpu import GpuChessGame
from src.model.muzero_net import MuZeroNetwork
from src.model.utils import support_to_scalar, inverse_scalar_transform

DEV="cuda"; cfg=get_config("chess_small"); game=ChessGame(); AS=game.action_space_size; NF=cfg.history_frames
gg=GpuChessGame(); torch.manual_seed(0); np.random.seed(0)
CKPT=os.environ["CKPT"]; ATTN=os.environ.get("USE_ATTENTION","0")=="1"; SMOL=os.environ.get("USE_SMOLGEN","1")=="1"
PREDATTN=os.environ.get("USE_PRED_ATTENTION","0")=="1"; DYNATTN=os.environ.get("USE_DYN_ATTENTION","0")=="1"
ATTNL=int(os.environ.get("ATTN_LAYERS","4")); N_POS=int(os.environ.get("N_POS","400"))
ML_SUPPORT=int(os.environ.get("ML_SUPPORT", str(cfg.moves_left_support_size)))
ML_LINEAR=os.environ.get("ML_LINEAR","0")=="1"
cfg.moves_left_support_size=ML_SUPPORT
net=MuZeroNetwork(observation_channels=game.num_planes*NF, action_space_size=AS, hidden_planes=cfg.hidden_planes,
    num_blocks=cfg.num_residual_blocks, latent_h=cfg.latent_h, latent_w=cfg.latent_w, input_h=game.board_size[0],
    input_w=game.board_size[1], fc_hidden=cfg.fc_hidden, value_support_size=cfg.value_support_size,
    reward_support_size=cfg.reward_support_size, action_embed_dim=cfg.action_embed_dim, use_consistency_loss=False,
    proj_hid=cfg.proj_hid, proj_out=cfg.proj_out, pred_hid=cfg.pred_hid, pred_out=cfg.pred_out,
    use_scalar_transform=cfg.use_scalar_transform, value_target_scale=cfg.value_target_scale, value_head_type="wdl",
    draw_score=0.0, policy_head_type=cfg.policy_head_type, use_material_head=False,
    use_moves_left=True, moves_left_support_size=cfg.moves_left_support_size,
    moves_left_head_planes=16, moves_left_head_blocks=1,
    use_repr_attention=ATTN, attn_layers=ATTNL, attn_heads=4, use_smolgen=SMOL,
    use_pred_attention=PREDATTN, pred_attn_layers=2, use_dyn_attention=DYNATTN).to(DEV)
_res=net.load_state_dict(torch.load(CKPT, map_location=DEV)["model_state_dict"], strict=False)
assert not _res.missing_keys, f"missing {_res.missing_keys[:5]}"
net.eval()

def encode(fens):
    st=gg.from_python_chess([chess.Board(f) for f in fens], device=DEV); obs=gg.to_tensor_batch(st)
    N,C,H,W=obs.shape
    if NF>1: obs=torch.cat([obs, torch.zeros(N,(NF-1)*C,H,W,device=DEV)],1)
    return obs

@torch.no_grad()
def pred_ml(fens):
    out=[]
    for s in range(0,len(fens),1024):
        h=net.representation(encode(fens[s:s+1024]))
        ml=support_to_scalar(net.predict_moves_left(h), cfg.moves_left_support_size)
        if not ML_LINEAR: ml=inverse_scalar_transform(ml)
        ml=ml.clamp(min=0).cpu().numpy().reshape(-1)
        out.append(ml)
    return np.concatenate(out)

def sig(b):
    w=b.turn
    wp=sorted(p.symbol().upper() for _,p in b.piece_map().items() if p.color==w and p.piece_type!=chess.KING)
    lp=sorted(p.symbol().upper() for _,p in b.piece_map().items() if p.color!=w and p.piece_type!=chess.KING)
    return f"K{''.join(wp)}vK{''.join(lp)}"

gav=chess.gaviota.open_tablebase("data/gaviota")
te=pickle.load(open("data/tb5_test.pkl","rb")); te_fen=[x[0] for x in te]; te_v=np.array([x[1] for x in te])
won=[te_fen[i] for i in range(len(te_fen)) if te_v[i]==0]

# --- (1) scalar accuracy on won roots ---
roots=[]; true_dtm=[]
for f in won:
    b=chess.Board(f)
    try:
        d=gav.probe_dtm(b)
    except Exception: d=None
    if d is None or d<=0: continue           # keep only won-for-mover with a real DTM
    roots.append(f); true_dtm.append(abs(d))
    if len(roots)>=N_POS: break
true_dtm=np.array(true_dtm, dtype=float)
pml=pred_ml(roots)
err=pml-true_dtm
def spearman(a,b):
    ra=np.argsort(np.argsort(a)); rb=np.argsort(np.argsort(b))
    return float(np.corrcoef(ra,rb)[0,1])
print(f"loaded {CKPT}", flush=True)
print(f"\n== SCALAR DTM accuracy on {len(roots)} won positions ==", flush=True)
print(f"  MAE {np.mean(np.abs(err)):.2f} plies | median |err| {np.median(np.abs(err)):.2f} | bias(mean err) {np.mean(err):+.2f}", flush=True)
print(f"  bias-corrected MAE (after removing the constant offset) {np.mean(np.abs(err-err.mean())):.2f} plies", flush=True)
print(f"  within +/-2 plies: {np.mean(np.abs(err)<=2):.1%} | within +/-4: {np.mean(np.abs(err)<=4):.1%}", flush=True)
print(f"  Spearman(pred, true DTM): {spearman(pml, true_dtm):.3f}", flush=True)
print(f"  true DTM mean {true_dtm.mean():.1f}  pred mean {pml.mean():.1f}", flush=True)

# --- (2) ranking accuracy: does argmin-pred-ML child == DTM-optimal child? ---
hit=0; tot=0; pair_ok=0; pair_tot=0; byc=defaultdict(lambda:[0,0])
for f in roots:
    b=chess.Board(f); mover=b.turn; childf=[]; childdtm=[]
    for mv in b.legal_moves:
        b.push(mv)
        try: d=gav.probe_dtm(b)            # child: opponent to move; d<0 means opponent is getting mated => good for us
        except Exception: d=None
        b.pop()
        if d is None: childdtm.append(None)
        else: childdtm.append(d)
        childf.append(b.fen() if False else None)  # placeholder
    # recompute child fens cleanly
    childf=[]
    for mv in b.legal_moves:
        b.push(mv); childf.append(b.fen()); b.pop()
    valid=[i for i,d in enumerate(childdtm) if d is not None]
    if len(valid)<2: continue
    # true value to mover = -child_dtm (mate the opponent fast => most negative child_dtm => smallest plies)
    true_to_mover=np.array([ -childdtm[i] for i in valid ], dtype=float)  # want LARGEST (fastest mate = most positive)
    cml=pred_ml([childf[i] for i in valid])                              # model: smaller ML = faster mate
    # DTM-optimal children: those keeping the win with fastest mate
    win_idx=[k for k,i in enumerate(valid) if childdtm[i]<0]             # child lost-for-opponent
    if not win_idx: continue
    best=max(true_to_mover[k] for k in win_idx)
    opt={k for k in win_idx if true_to_mover[k]==best}
    pick=int(np.argmin(cml))                                            # model's pick = smallest predicted ML
    hit+= int(pick in opt); tot+=1; byc[sig(b)][0]+=int(pick in opt); byc[sig(b)][1]+=1
    # pairwise order over win children
    for x in range(len(win_idx)):
        for y in range(x+1,len(win_idx)):
            k1,k2=win_idx[x],win_idx[y]
            if true_to_mover[k1]==true_to_mover[k2]: continue
            faster = k1 if true_to_mover[k1]>true_to_mover[k2] else k2
            pred_faster = k1 if cml[k1]<cml[k2] else k2
            pair_ok+= int(faster==pred_faster); pair_tot+=1
print(f"\n== RANKING (search-relevant) over {tot} positions ==", flush=True)
print(f"  argmin-pred-ML picks a DTM-optimal move: {hit/max(tot,1):.1%}", flush=True)
print(f"  pairwise child order correct           : {pair_ok/max(pair_tot,1):.1%}", flush=True)
for s,(h,n) in sorted(byc.items(), key=lambda kv:-kv[1][1]):
    if n>=8: print(f"    {s:10s} n={n:4d}  optimal-pick {h/n:.1%}", flush=True)
gav.close(); print("\nDONE", flush=True)
