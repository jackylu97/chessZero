"""Clean, decision-relevant accuracy for the ML head + policy, on won positions.

Drops the noisy pairwise metric. Reports what the search actually depends on:
  POLICY:
    pol_opt_top1  : argmax-policy move is a DTM-optimal move
    pol_opt_top5  : a DTM-optimal move is among the policy's top-5 moves (is it AVAILABLE?)
  ML HEAD:
    ml_opt_all    : argmin predicted-ML over ALL legal moves is DTM-optimal (raw head skill)
    ml_opt_in_pol5: among the policy's top-5 moves, the head's lowest-ML pick is DTM-optimal
                    (THE real decision: among the moves search considers, does ML pick the best?)
  Conditioned:
    ml_given_avail: of positions where an optimal move IS in policy-top5, does ML pick it?

Run over several checkpoints to see the ML head's improvement.
Run: CKPT=... USE_ATTENTION=1 USE_SMOLGEN=1 USE_DYN_ATTENTION=1 ATTN_LAYERS=6 [ML_LINEAR=1 ML_SUPPORT=24] \
     PYTHONPATH=. .venv/bin/python scripts/measure_ml_policy_topk.py
"""
import os, pickle, numpy as np, torch, chess, chess.syzygy, chess.gaviota
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
ATTNL=int(os.environ.get("ATTN_LAYERS","4")); N_POS=int(os.environ.get("N_POS","500"))
ML_SUPPORT=int(os.environ.get("ML_SUPPORT", str(cfg.moves_left_support_size)))
ML_LINEAR=os.environ.get("ML_LINEAR","0")=="1"; cfg.moves_left_support_size=ML_SUPPORT
net=MuZeroNetwork(observation_channels=game.num_planes*NF, action_space_size=AS, hidden_planes=cfg.hidden_planes,
    num_blocks=cfg.num_residual_blocks, latent_h=cfg.latent_h, latent_w=cfg.latent_w, input_h=game.board_size[0],
    input_w=game.board_size[1], fc_hidden=cfg.fc_hidden, value_support_size=cfg.value_support_size,
    reward_support_size=cfg.reward_support_size, action_embed_dim=cfg.action_embed_dim, use_consistency_loss=False,
    proj_hid=cfg.proj_hid, proj_out=cfg.proj_out, pred_hid=cfg.pred_hid, pred_out=cfg.pred_out,
    use_scalar_transform=cfg.use_scalar_transform, value_target_scale=cfg.value_target_scale, value_head_type="wdl",
    draw_score=0.0, policy_head_type=cfg.policy_head_type, use_material_head=False,
    use_moves_left=True, moves_left_support_size=ML_SUPPORT, moves_left_head_planes=16, moves_left_head_blocks=1,
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
def policy_softmax(fen):
    h=net.representation(encode([fen])); pl,_=net.prediction(h)
    return torch.softmax(pl,1)[0].cpu().numpy()

@torch.no_grad()
def child_ml(fens):
    h=net.representation(encode(fens)); ml=support_to_scalar(net.predict_moves_left(h), ML_SUPPORT)
    if not ML_LINEAR: ml=inverse_scalar_transform(ml)
    return ml.clamp(min=0).cpu().numpy()

def sig(b):
    w=b.turn
    wp=sorted(p.symbol().upper() for _,p in b.piece_map().items() if p.color==w and p.piece_type!=chess.KING)
    lp=sorted(p.symbol().upper() for _,p in b.piece_map().items() if p.color!=w and p.piece_type!=chess.KING)
    return f"K{''.join(wp)}vK{''.join(lp)}"

gav=chess.gaviota.open_tablebase("data/gaviota"); tb=chess.syzygy.open_tablebase("data/syzygy")
te=pickle.load(open("data/tb5_test.pkl","rb")); te_fen=[x[0] for x in te]; te_v=np.array([x[1] for x in te])
won=[te_fen[i] for i in range(len(te_fen)) if te_v[i]==0][:N_POS]

agg=defaultdict(float); byc=defaultdict(lambda: defaultdict(float))
for fen in won:
    b=chess.Board(fen)
    moves=list(b.legal_moves)
    if len(moves)<2: continue
    acts=[_move_to_action(m,b.turn) for m in moves]
    # true DTM-optimal set (win-preserving, fastest)
    qual=[]
    for m in moves:
        b.push(m)
        try: wdl=-tb.probe_wdl(b); d=gav.probe_dtm(b)
        except Exception: wdl,d=0,0
        cf=b.fen(); b.pop()
        qual.append((100000-abs(d)) if wdl>=2 else ((-100000+abs(d)) if wdl<=-2 else 0))
    qual=np.array(qual); best=qual.max(); opt={acts[i] for i in range(len(acts)) if qual[i]==best}
    # policy
    soft=policy_softmax(fen)
    pol_rank=sorted(range(len(acts)), key=lambda i:-soft[acts[i]])
    pol_top1=acts[pol_rank[0]]; pol_top5=[acts[i] for i in pol_rank[:5]]
    # ML over all children + over the policy-top5 children
    child_fens=[]
    for m in moves: b.push(m); child_fens.append(b.fen()); b.pop()
    cml=child_ml(child_fens)
    ml_pick_all=acts[int(np.argmin(cml))]
    top5_idx=pol_rank[:5]; ml_pick_pol5=acts[min(top5_idx, key=lambda i: cml[i])]
    avail=len(opt & set(pol_top5))>0
    def add(d):
        d['n']+=1
        d['pol_opt_top1']+=int(pol_top1 in opt)
        d['pol_opt_top5']+=int(avail)
        d['ml_opt_all']  +=int(ml_pick_all in opt)
        d['ml_opt_in_pol5']+=int(ml_pick_pol5 in opt)
        if avail: d['ml_given_avail']+=int(ml_pick_pol5 in opt); d['avail_n']+=1
    add(agg); add(byc[sig(b)])
def line(tag,d):
    m=d['n']; ga=d['avail_n'] or 1
    return (f"  {tag:11s} n={int(m):4d} | pol_opt_top1 {d['pol_opt_top1']/m:.0%} | pol_opt_top5 {d['pol_opt_top5']/m:.0%} "
            f"| ml_opt_all {d['ml_opt_all']/m:.0%} | ml_opt_in_pol5 {d['ml_opt_in_pol5']/m:.0%} | ml|avail {d['ml_given_avail']/ga:.0%}")
print(f"loaded {os.path.basename(CKPT)}  (ml_linear={ML_LINEAR} support={ML_SUPPORT})", flush=True)
print(line("ALL", agg), flush=True)
for s,d in sorted(byc.items(), key=lambda kv:-kv[1]['n']):
    if d['n']>=12: print(line(s,d), flush=True)
gav.close(); tb.close(); print("DONE", flush=True)
