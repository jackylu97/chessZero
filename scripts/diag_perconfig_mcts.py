"""Per-material-config conversion through MCTS, terminal-aware search OFF vs ON, on the
moves-left-trained checkpoint (tb5_seq_ml.pt). Tests #1 (trained MLH utility, baked into the
ckpt) + #2 (stalemate/insufficient terminal mask pinned to draw in the REAL TensorMCTS pin
path). Classifies every game mate/cap/draw/lost so we can watch the 79%-stalemate rate move.

Run: PYTHONPATH=. .venv/bin/python scripts/diag_perconfig_mcts.py
"""
import pickle, time, numpy as np, torch, chess, chess.syzygy
from collections import defaultdict, Counter
from src.config import get_config
from src.games.chess import ChessGame, _action_to_move, _move_to_action
from src.games.chess_gpu import GpuChessGame
from src.model.muzero_net import MuZeroNetwork
from src.mcts.tensor_mcts import TensorMCTS

import os
DEV="cuda"; cfg=get_config(os.environ.get("CFG","chess_small")); game=ChessGame(); AS=game.action_space_size; NF=cfg.history_frames
gg=GpuChessGame(); torch.manual_seed(0); np.random.seed(0)
CKPT=os.environ.get("CKPT","checkpoints/tb5_seq_ml.pt"); ATTN=os.environ.get("USE_ATTENTION","0")=="1"
SMOL=os.environ.get("USE_SMOLGEN","1")=="1"; PREDATTN=os.environ.get("USE_PRED_ATTENTION","0")=="1"
DYNATTN=os.environ.get("USE_DYN_ATTENTION","0")=="1"
ATTNL=int(os.environ.get("ATTN_LAYERS","4"))
# CFG=chess_hybrid picks up hidden/fc/head widths from the preset; these envs then
# only need to mirror the attention body switches.
HYBRID_STEM=int(os.environ.get("HYBRID_STEM", str(getattr(cfg,"hybrid_stem_blocks",0))))
PREDATTN_LAYERS=int(os.environ.get("PRED_ATTN_LAYERS", str(getattr(cfg,"pred_attn_layers",2))))
VHP=int(os.environ.get("VALUE_HEAD_PLANES", str(getattr(cfg,"value_head_planes",1))))
ML_SUPPORT=int(os.environ.get("ML_SUPPORT", str(cfg.moves_left_support_size)))
ML_LINEAR=os.environ.get("ML_LINEAR","0")=="1"
cfg.moves_left_support_size=ML_SUPPORT; cfg.moves_left_use_transform=not ML_LINEAR
N_WON=400; MAX_PLIES=80; SIMS=200
net=MuZeroNetwork(observation_channels=game.num_planes*NF, action_space_size=AS, hidden_planes=cfg.hidden_planes,
    num_blocks=cfg.num_residual_blocks, latent_h=cfg.latent_h, latent_w=cfg.latent_w, input_h=game.board_size[0],
    input_w=game.board_size[1], fc_hidden=cfg.fc_hidden, value_support_size=cfg.value_support_size,
    reward_support_size=cfg.reward_support_size, action_embed_dim=cfg.action_embed_dim, use_consistency_loss=False,
    proj_hid=cfg.proj_hid, proj_out=cfg.proj_out, pred_hid=cfg.pred_hid, pred_out=cfg.pred_out,
    use_scalar_transform=cfg.use_scalar_transform, value_target_scale=cfg.value_target_scale, value_head_type="wdl",
    draw_score=0.0, policy_head_type=cfg.policy_head_type, use_material_head=False,
    use_moves_left=True, moves_left_support_size=cfg.moves_left_support_size,
    moves_left_head_planes=int(os.environ.get("MLH_PLANES", getattr(cfg,"moves_left_head_planes",16))),
    moves_left_head_blocks=int(os.environ.get("MLH_BLOCKS", getattr(cfg,"moves_left_head_blocks",1))),
    use_repr_attention=ATTN, attn_layers=ATTNL, attn_heads=4, use_smolgen=SMOL,
    use_pred_attention=PREDATTN, pred_attn_layers=PREDATTN_LAYERS, use_dyn_attention=DYNATTN,
    hybrid_stem_blocks=HYBRID_STEM, value_head_planes=VHP).to(DEV)
# strict=False: the checkpoint may carry training-only heads (projection/predictor/inverse)
# that the inference backbone lacks; those are ignored. Missing keys => arch mismatch (fatal).
_res=net.load_state_dict(torch.load(CKPT, map_location=DEV)["model_state_dict"], strict=False)
assert not _res.missing_keys, f"ARCH MISMATCH, missing keys: {_res.missing_keys[:6]}"
net.eval()
print(f"loaded {CKPT}  (rep={'attn' if ATTN else 'conv'} smol={SMOL} predattn={PREDATTN}); "
      f"ignored {len(_res.unexpected_keys)} training-only keys", flush=True)
te=pickle.load(open("data/tb5_test.pkl","rb")); te_fen=[x[0] for x in te]; te_v=np.array([x[1] for x in te])

def encode(fens):
    st=gg.from_python_chess([chess.Board(f) for f in fens], device=DEV); obs=gg.to_tensor_batch(st)
    N,C,H,W=obs.shape
    if NF>1: obs=torch.cat([obs, torch.zeros(N,(NF-1)*C,H,W,device=DEV)],1)
    return obs

def sig(b, winner):
    wp=sorted(p.symbol().upper() for _,p in b.piece_map().items() if p.color==winner and p.piece_type!=chess.KING)
    lp=sorted(p.symbol().upper() for _,p in b.piece_map().items() if p.color!=winner and p.piece_type!=chess.KING)
    return f"K{''.join(wp)}vK{''.join(lp)}"

@torch.no_grad()
def term_draw_mask(boards):
    """[N, AS] bool: legal moves leading to stalemate or insufficient material (mover POV action)."""
    fdm=torch.zeros(len(boards), AS, dtype=torch.bool, device=DEV)
    for j,b in enumerate(boards):
        mover=b.turn
        for mv in b.legal_moves:
            b.push(mv); td=b.is_stalemate() or b.is_insufficient_material(); b.pop()
            if td: fdm[j, _move_to_action(mv, mover)]=True
    return fdm

@torch.no_grad()
def mcts_moves(mcts, boards, use_term):
    obs=encode([b.fen() for b in boards])
    lm=torch.zeros(len(boards), AS, dtype=torch.bool, device=DEV)
    for j,b in enumerate(boards):
        for mv in b.legal_moves: lm[j, _move_to_action(mv, b.turn)]=True
    fdm=term_draw_mask(boards) if use_term else None
    rd=mcts.run_batch_gpu(obs, lm, add_noise=False, forced_draw_mask=fdm)
    ca=rd["child_actions"].cpu().numpy(); cv=rd["child_visits"].cpu().numpy(); res=[]
    for j,b in enumerate(boards):
        slot=int(np.argmax(np.where(ca[j]!=-1, cv[j], -1)))
        mv=_action_to_move(int(ca[j][slot]), b)
        res.append(mv if (mv is not None and mv in b.legal_moves) else list(b.legal_moves)[0])
    return res

def play_classify(move_fn, tb):
    won=[te_fen[i] for i in range(len(te_fen)) if te_v[i]==0][:N_WON]
    boards=[chess.Board(f) for f in won]; winner=[b.turn for b in boards]
    sigs=[sig(boards[i],winner[i]) for i in range(len(boards))]
    outcome=[None]*len(boards); pmate=[None]*len(boards); done=[False]*len(boards)
    for ply in range(MAX_PLIES):
        active=[]
        for i,b in enumerate(boards):
            if done[i]: continue
            if b.is_game_over():
                done[i]=True
                if b.is_checkmate(): outcome[i]='mate' if b.turn!=winner[i] else 'lost'; pmate[i]=(ply if b.turn!=winner[i] else None)
                else: outcome[i]='draw'
            else: active.append(i)
        if not active: break
        mvs=move_fn([boards[i] for i in active])
        for j,i in enumerate(active):
            boards[i].push(mvs[j])
            if boards[i].is_checkmate():
                done[i]=True
                if boards[i].turn!=winner[i]: outcome[i]='mate'; pmate[i]=ply+1
                else: outcome[i]='lost'
    for i in range(len(boards)):
        if not done[i]: outcome[i]='cap'
    cap_won=cap_lost=0
    for i in range(len(boards)):
        if outcome[i]=='cap':
            b=boards[i]
            try:
                wdl=tb.probe_wdl(b); mover=wdl if b.turn==winner[i] else -wdl
                cap_won+=int(mover>=2); cap_lost+=int(mover<2)
            except Exception: cap_lost+=1
    return outcome, pmate, sigs, cap_won, cap_lost

def report(tag, outcome, pmate, sigs, cap_won, cap_lost):
    N=len(outcome); oc=Counter(outcome)
    print(f"\n===== {tag} =====")
    print(f"  CONVERTED(mate)={oc['mate']/N:.3f}  cap={oc['cap']/N:.3f}  DRAW(stalemate/rule)={oc['draw']/N:.3f}  lost={oc['lost']/N:.3f}")
    print(f"  of {oc['cap']} cap: still-won(shuffle)={cap_won} thrown={cap_lost}")
    mp=[p for p in pmate if p is not None]
    if mp: print(f"  mean plies-to-mate(converted): {np.mean(mp):.1f}")
    byc=defaultdict(lambda:[0,0,0,0,0])
    for i in range(N):
        byc[sigs[i]][0]+=1; byc[sigs[i]][{'mate':1,'cap':2,'draw':3,'lost':4}[outcome[i]]]+=1
    rows=[(m/n,s,n,m,c,d,l) for s,(n,m,c,d,l) in byc.items() if n>=10]
    for conv,s,n,m,c,d,l in sorted(rows, reverse=True):
        print(f"    {s:12s} n={n:4d} CONV={conv:.2f} cap={c/n:.2f} draw={d/n:.2f} lost={l/n:.2f}")

tb=chess.syzygy.open_tablebase("data/syzygy")
for use_term in (False, True):
    cfg.num_simulations=SIMS; cfg.moves_left_mcts=True; cfg.tb_root_probe=False; cfg.root_terminal_draws=use_term
    mcts=TensorMCTS(net, game, cfg, device=DEV, hidden_dtype=torch.float32, select_backend="eager")
    t0=time.time()
    o,p,s,cw,cl=play_classify(lambda bs: mcts_moves(mcts, bs, use_term), tb)
    report(f"MCTS  terminal_mask={'ON' if use_term else 'OFF'}  MLH=ON  ({time.time()-t0:.0f}s)", o,p,s,cw,cl)
tb.close()
print("\nDONE")
