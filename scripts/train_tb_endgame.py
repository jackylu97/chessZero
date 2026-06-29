"""From-scratch endgame training on the 5-man tablebase, BOTH fixes ON, watching the
learning curve over time:
  #1 moves-left/DTM head (supervised on |DTZ| / draw-sentinel), and
  #2 terminal-aware MCTS (stalemate + insufficient pinned to draw in the real pin path).
Every EVAL_MCTS steps it plays won positions out through MCTS (terminal mask ON) and
reports CONVERTED(mate) / DRAW(stalemate) / cap(shuffle) so we can see whether the
combination learns to convert as it trains. Checkpoints periodically.

Run: PYTHONPATH=. .venv/bin/python scripts/train_tb_endgame.py
"""
import pickle, time, os, numpy as np, torch, torch.nn.functional as F, chess
from collections import Counter
from src.config import get_config
from src.games.chess import ChessGame, _action_to_move, _move_to_action
from src.games.chess_gpu import GpuChessGame
from src.model.muzero_net import MuZeroNetwork
from src.model.utils import scalar_transform, scalar_to_support, inverse_scalar_transform, support_to_scalar

DEV="cuda"; cfg=get_config("chess_small"); game=ChessGame(); AS=game.action_space_size; NF=cfg.history_frames
K=5; gg=GpuChessGame(); torch.manual_seed(0); np.random.seed(0)
STEPS=int(os.environ.get("STEPS","30000")); B=384; ML_W=float(cfg.moves_left_loss_weight)
EVAL_CHEAP=1000; EVAL_MCTS=3000; CKPT_INT=6000; N_MCTS=300; MAX_PLIES=80; SIMS=200
ATTN=os.environ.get("USE_ATTENTION","0")=="1"        # smolgen attention representation
SMOL=os.environ.get("USE_SMOLGEN","1")=="1"          # smolgen on/off (only matters with ATTN)
PREDATTN=os.environ.get("USE_PRED_ATTENTION","0")=="1"  # shared attention body in the policy/value model
DYNATTN=os.environ.get("USE_DYN_ATTENTION","0")=="1"    # attention body in the DYNAMICS (matches the rep so consistency is reachable)
CONS=os.environ.get("USE_CONSISTENCY","0")=="1"      # EfficientZero SimSiam consistency loss
INV=os.environ.get("USE_INVERSE","0")=="1"           # ICM inverse-dynamics loss
CONS_W=float(cfg.consistency_loss_weight); INV_W=float(cfg.inverse_dynamics_loss_weight)
_parts=["attn" if ATTN else "conv"]
if ATTN and not SMOL: _parts.append("nosmol")
if DYNATTN: _parts.append("dynattn")
if PREDATTN: _parts.append("predattn")
if CONS: _parts.append("ssl")
if INV: _parts.append("inv")
TAG="_".join(_parts)
def _neg_cos(p, z):   # SimSiam negative cosine, per-sample
    p=F.normalize(p, dim=-1); z=F.normalize(z, dim=-1); return -(p*z).sum(-1)
print("loading sequences...", flush=True)
SEQ=pickle.load(open("data/tb5_seq.pkl","rb")); assert len(SEQ[0])==5, "need 5-tuple (moves-left) data"
te=pickle.load(open("data/tb5_test.pkl","rb")); te_fen=[x[0] for x in te]; te_v=np.array([x[1] for x in te]); te_p=[np.array(x[2]) for x in te]
print(f"sequences {len(SEQ)}  test {len(te_fen)}  STEPS {STEPS}", flush=True)

net=MuZeroNetwork(observation_channels=game.num_planes*NF, action_space_size=AS, hidden_planes=cfg.hidden_planes,
    num_blocks=cfg.num_residual_blocks, latent_h=cfg.latent_h, latent_w=cfg.latent_w, input_h=game.board_size[0],
    input_w=game.board_size[1], fc_hidden=cfg.fc_hidden, value_support_size=cfg.value_support_size,
    reward_support_size=cfg.reward_support_size, action_embed_dim=cfg.action_embed_dim, use_consistency_loss=CONS,
    proj_hid=cfg.proj_hid, proj_out=cfg.proj_out, pred_hid=cfg.pred_hid, pred_out=cfg.pred_out,
    use_scalar_transform=cfg.use_scalar_transform, value_target_scale=cfg.value_target_scale, value_head_type="wdl",
    draw_score=0.0, policy_head_type=cfg.policy_head_type, use_material_head=False,
    use_moves_left=True, moves_left_support_size=cfg.moves_left_support_size,
    moves_left_head_planes=16, moves_left_head_blocks=1,
    use_inverse_dynamics_loss=INV, inverse_dynamics_hidden=cfg.inverse_dynamics_hidden,
    use_repr_attention=ATTN, attn_layers=4, attn_heads=4, use_smolgen=SMOL,
    use_pred_attention=PREDATTN, pred_attn_layers=2, use_dyn_attention=DYNATTN).to(DEV)
opt=torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
print(f"params {sum(p.numel() for p in net.parameters())/1e6:.2f}M (FROM SCRATCH, TAG={TAG}; "
      f"attn={ATTN} smolgen={SMOL} dynattn={DYNATTN} predattn={PREDATTN} consistency={CONS} inverse={INV} STEPS={STEPS})", flush=True)

def encode(fens):
    st=gg.from_python_chess([chess.Board(f) for f in fens], device=DEV); obs=gg.to_tensor_batch(st)
    N,C,H,W=obs.shape
    if NF>1: obs=torch.cat([obs, torch.zeros(N,(NF-1)*C,H,W,device=DEV)],1)
    return obs

def pol_dense(pols):
    t=torch.zeros(len(pols), AS, device=DEV)
    for i,p in enumerate(pols):
        p=[a for a in p if a<AS]
        if p: t[i, torch.tensor(p, device=DEV)]=1.0/len(p)
    return t

@torch.no_grad()
def term_draw_mask(boards):
    fdm=torch.zeros(len(boards), AS, dtype=torch.bool, device=DEV)
    for j,b in enumerate(boards):
        mover=b.turn
        for mv in b.legal_moves:
            b.push(mv); td=b.is_stalemate() or b.is_insufficient_material(); b.pop()
            if td: fdm[j,_move_to_action(mv,mover)]=True
    return fdm

@torch.no_grad()
def cheap_eval():
    net.eval(); vcorr=pcorr=n=0; ml_won=[]; ml_draw=[]
    for s in range(0, min(len(te_fen), 8000), 1024):
        fb=te_fen[s:s+1024]; h=net.representation(encode(fb)); pl,vl=net.prediction(h)
        vcorr+=(vl.argmax(1).cpu().numpy()==te_v[s:s+1024]).sum()
        mlp=inverse_scalar_transform(support_to_scalar(net.predict_moves_left(h), cfg.moves_left_support_size)).clamp(min=0).cpu().numpy().reshape(-1)
        vb=te_v[s:s+1024]; ml_won.extend(mlp[vb==0].tolist()); ml_draw.extend(mlp[vb==1].tolist())
        soft=pl.cpu().numpy()
        for i,f in enumerate(fb):
            b=chess.Board(f); legal=[_move_to_action(m,b.turn) for m in b.legal_moves]
            if max(legal, key=lambda a: soft[i,a]) in set(te_p[s+i].tolist()): pcorr+=1
        n+=len(fb)
    net.train()
    mlw=float(np.mean(ml_won)) if ml_won else 0.0; mld=float(np.mean(ml_draw)) if ml_draw else 0.0
    return vcorr/n, pcorr/n, mlw, mld

@torch.no_grad()
def mcts_eval():
    net.eval()
    cfg.num_simulations=SIMS; cfg.moves_left_mcts=True; cfg.tb_root_probe=False; cfg.root_terminal_draws=True
    from src.mcts.tensor_mcts import TensorMCTS
    m=TensorMCTS(net, game, cfg, device=DEV, hidden_dtype=torch.float32, select_backend="eager")
    won=[te_fen[i] for i in range(len(te_fen)) if te_v[i]==0][:N_MCTS]
    boards=[chess.Board(f) for f in won]; winner=[b.turn for b in boards]; done=[False]*len(boards); outcome=[None]*len(boards)
    for ply in range(MAX_PLIES):
        active=[]
        for i,b in enumerate(boards):
            if done[i]: continue
            if b.is_game_over():
                done[i]=True; outcome[i]=('mate' if (b.is_checkmate() and b.turn!=winner[i]) else ('lost' if b.is_checkmate() else 'draw'))
            else: active.append(i)
        if not active: break
        ab=[boards[i] for i in active]; obs=encode([b.fen() for b in ab])
        lm=torch.zeros(len(ab), AS, dtype=torch.bool, device=DEV)
        for j,b in enumerate(ab):
            for mv in b.legal_moves: lm[j,_move_to_action(mv,b.turn)]=True
        rd=m.run_batch_gpu(obs, lm, add_noise=False, forced_draw_mask=term_draw_mask(ab))
        ca=rd["child_actions"].cpu().numpy(); cv=rd["child_visits"].cpu().numpy()
        for j,i in enumerate(active):
            b=boards[i]; slot=int(np.argmax(np.where(ca[j]!=-1, cv[j], -1)))
            mv=_action_to_move(int(ca[j][slot]), b)
            if mv is None or mv not in b.legal_moves: mv=list(b.legal_moves)[0]
            b.push(mv)
            if b.is_checkmate(): done[i]=True; outcome[i]='mate' if b.turn!=winner[i] else 'lost'
    for i in range(len(boards)):
        if not done[i]: outcome[i]='cap'
    net.train(); nn=len(boards); oc=Counter(outcome)
    return oc['mate']/nn, oc['draw']/nn, oc['cap']/nn

os.makedirs("checkpoints", exist_ok=True)
t0=time.time()
for step in range(1, STEPS+1):
    bi=np.random.randint(0, len(SEQ), B); batch=[SEQ[i] for i in bi]
    obs_k=[]; act_k=[]; vc_k=[]; pol_k=[]; mask_k=[]; ml_k=[]
    for k in range(K+1):
        fens=[s[0][min(k, len(s[0])-1)] for s in batch]; obs_k.append(encode(fens))
        vc_k.append(torch.tensor([s[2][min(k, len(s[2])-1)] for s in batch], device=DEV))
        pol_k.append(pol_dense([s[3][min(k, len(s[3])-1)] for s in batch]))
        ml_k.append(torch.tensor([float(s[4][min(k, len(s[4])-1)]) for s in batch], device=DEV))
        mask_k.append(torch.tensor([1.0 if k<len(s[0]) else 0.0 for s in batch], device=DEV))
        if k<K: act_k.append(torch.tensor([s[1][k] if k<len(s[1]) else 0 for s in batch], dtype=torch.long, device=DEV))
    h=net.representation(obs_k[0]); loss=0.0; mloss=1.0/(K+1)
    for k in range(K+1):
        if k==0:
            pl,vl=net.prediction(h)
        else:
            h_in=h
            h,rl,pl,vl=net.recurrent_inference_logits(h, act_k[k-1]); h.register_hook(lambda g: g*0.5)
            if CONS:   # EfficientZero SimSiam: online dyn proj vs stop-grad repr(next) proj
                dyn_proj=net.project(h, with_grad=True)
                with torch.no_grad():
                    ht=net.representation(obs_k[k]); tgt_proj=net.project(ht, with_grad=False)
                cons=_neg_cos(dyn_proj, tgt_proj) * mask_k[k]
                loss=loss + mloss*CONS_W*cons.mean()
            if INV:    # ICM inverse-dynamics: recover act_k[k-1] from (h_{k-1}, h_k)
                inv_logits=net.predict_inverse_action(h_in, h)
                invce=F.cross_entropy(inv_logits, act_k[k-1], reduction='none') * mask_k[k]
                loss=loss + mloss*INV_W*invce.mean()
        vce=F.cross_entropy(vl, vc_k[k], reduction='none')*mask_k[k]
        pce=-(pol_k[k]*F.log_softmax(pl,1)).sum(1)*mask_k[k]
        ml_logits=net.predict_moves_left(h)
        ml_tgt=scalar_to_support(scalar_transform(ml_k[k]), cfg.moves_left_support_size).to(ml_logits.device)
        mlce=-(ml_tgt*F.log_softmax(ml_logits,1)).sum(1)*mask_k[k]
        loss=loss + mloss*(vce.mean()+pce.mean()+ML_W*mlce.mean())
    opt.zero_grad(); loss.backward(); opt.step()
    if step % EVAL_CHEAP == 0:
        vacc,pacc,mlw,mld=cheap_eval(); extra=""
        if step % EVAL_MCTS == 0:
            conv,draw,cap=mcts_eval()
            extra=f"  || MCTS(term-ON): CONV {conv:.3f}  stalemate-DRAW {draw:.3f}  cap {cap:.3f}"
        print(f"step {step:5d} ({time.time()-t0:.0f}s) loss {loss.item():.3f} | value_acc {vacc:.3f} "
              f"policy_acc {pacc:.3f} | ml_won {mlw:.0f} ml_draw {mld:.0f}{extra}", flush=True)
    if step % CKPT_INT == 0:
        torch.save({"model_state_dict": net.state_dict(), "step": step}, f"checkpoints/tb5_endgame_{TAG}_{step}.pt")
        print(f"  saved checkpoints/tb5_endgame_{TAG}_{step}.pt", flush=True)
torch.save({"model_state_dict": net.state_dict(), "step": STEPS}, f"checkpoints/tb5_endgame_{TAG}_final.pt")
print(f"saved checkpoints/tb5_endgame_{TAG}_final.pt", flush=True); print("DONE", flush=True)
