"""Policy quality vs the FULL tablebase move ranking (not just the optimal set).

For each won test position, rank every legal move by tablebase quality
(win-preserving + fastest DTM > draw > slow loss), then measure:
  - top1==TB-best     : model's argmax move is THE optimal move (~ the logged policy_acc)
  - top1 in TB-top3/5 : model's argmax is among the 5 best TB moves (is it at least close?)
  - top1 win-preserving: model's argmax keeps the win (doesn't throw it to a draw/loss)
  - optimal in model top5: a TB-optimal move is in the model's 5 highest-policy moves
                           (the MCTS-relevant one: can search even FIND a best move?)
  - mean rank of model's chosen move (out of n legal)

Run: CKPT=... USE_ATTENTION=1 USE_SMOLGEN=1 USE_DYN_ATTENTION=1 ATTN_LAYERS=6 \
     PYTHONPATH=. .venv/bin/python scripts/measure_policy_topk.py
"""
import os, pickle, numpy as np, torch, chess, chess.syzygy, chess.gaviota
from collections import defaultdict
from src.config import get_config
from src.games.chess import ChessGame, _move_to_action
from src.games.chess_gpu import GpuChessGame
from src.model.muzero_net import MuZeroNetwork

DEV="cuda"; cfg=get_config("chess_small"); game=ChessGame(); AS=game.action_space_size; NF=cfg.history_frames
gg=GpuChessGame(); torch.manual_seed(0); np.random.seed(0)
CKPT=os.environ["CKPT"]; ATTN=os.environ.get("USE_ATTENTION","0")=="1"; SMOL=os.environ.get("USE_SMOLGEN","1")=="1"
PREDATTN=os.environ.get("USE_PRED_ATTENTION","0")=="1"; DYNATTN=os.environ.get("USE_DYN_ATTENTION","0")=="1"
ATTNL=int(os.environ.get("ATTN_LAYERS","4")); N_POS=int(os.environ.get("N_POS","600"))
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
net.eval(); print(f"loaded {CKPT}", flush=True)

def encode(fens):
    st=gg.from_python_chess([chess.Board(f) for f in fens], device=DEV); obs=gg.to_tensor_batch(st)
    N,C,H,W=obs.shape
    if NF>1: obs=torch.cat([obs, torch.zeros(N,(NF-1)*C,H,W,device=DEV)],1)
    return obs

@torch.no_grad()
def policy_softmax(fen):
    h=net.representation(encode([fen])); pl,_=net.prediction(h)
    return torch.softmax(pl,1)[0].cpu().numpy()

def sig(b):
    w=b.turn
    wp=sorted(p.symbol().upper() for _,p in b.piece_map().items() if p.color==w and p.piece_type!=chess.KING)
    lp=sorted(p.symbol().upper() for _,p in b.piece_map().items() if p.color!=w and p.piece_type!=chess.KING)
    return f"K{''.join(wp)}vK{''.join(lp)}"

def tb_quality(board, gav, tb):
    """Per legal move: quality to mover. win+fast-mate >> draw >> slow-loss."""
    q={}
    for mv in board.legal_moves:
        board.push(mv)
        try:
            wdl=-tb.probe_wdl(board); d=gav.probe_dtm(board)
        except Exception: wdl,d=0,0
        board.pop()
        if wdl>=2:    quality=100000 - abs(d)        # we win; fastest mate ranks top
        elif wdl<=-2: quality=-100000 + abs(d)       # we lose; survive longest ranks higher
        else:         quality=0                       # draw
        q[_move_to_action(mv, board.turn)]=quality
    return q

gav=chess.gaviota.open_tablebase("data/gaviota"); tb=chess.syzygy.open_tablebase("data/syzygy")
te=pickle.load(open("data/tb5_test.pkl","rb")); te_fen=[x[0] for x in te]; te_v=np.array([x[1] for x in te])
won=[te_fen[i] for i in range(len(te_fen)) if te_v[i]==0][:N_POS]

agg=defaultdict(float); byc=defaultdict(lambda: defaultdict(float))
for fen in won:
    b=chess.Board(fen); s=sig(b)
    q=tb_quality(b, gav, tb)
    if not q: continue
    acts=list(q.keys()); quals=np.array([q[a] for a in acts])
    order=np.argsort(-quals)                       # best first
    ranked=[acts[i] for i in order]
    best_q=quals.max(); optimal={a for a in acts if q[a]==best_q}
    soft=policy_softmax(fen)
    top1=max(acts, key=lambda a: soft[a])
    rank=ranked.index(top1)+1                        # 1 = best
    model_top5=sorted(acts, key=lambda a:-soft[a])[:5]
    def add(d):
        d['n']+=1
        d['top1_best']  += int(top1 in optimal)
        d['top1_in3']   += int(rank<=3)
        d['top1_in5']   += int(rank<=5)
        d['win_preserve']+= int(q[top1]>=100000-1000)   # mover still wins after model's move
        d['opt_in_top5']+= int(len(optimal & set(model_top5))>0)
        d['mean_rank']  += rank
        d['mean_nlegal']+= len(acts)
    add(agg); add(byc[s])
n=agg['n']
def line(tag,d):
    m=d['n']
    return (f"  {tag:12s} n={int(m):4d} | top1=best {d['top1_best']/m:.1%} | in-top3 {d['top1_in3']/m:.1%} | "
            f"in-top5 {d['top1_in5']/m:.1%} | win-preserve {d['win_preserve']/m:.1%} | "
            f"opt-in-model-top5 {d['opt_in_top5']/m:.1%} | mean-rank {d['mean_rank']/m:.1f}/{d['mean_nlegal']/m:.0f}")
print(f"\n== POLICY top-k vs tablebase ranking on {int(n)} won positions ==", flush=True)
print(line("ALL", agg), flush=True)
print("\n  per config:", flush=True)
for s,d in sorted(byc.items(), key=lambda kv:-kv[1]['n']):
    if d['n']>=10: print(line(s,d), flush=True)
gav.close(); tb.close(); print("\nDONE", flush=True)
