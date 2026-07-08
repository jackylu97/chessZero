"""Trace WHY won endgames fail to convert. Plays failing positions out through MCTS
(term-ON) and, at every ply, measures whether the bottleneck is the POLICY PRIOR, the
VALUE resolution, or the MOVES-LEFT ranking — against syzygy ground truth.

Per ply we log:
  - legal_prior_mass : softmax prob the raw policy head puts on legal moves (vs the
                       4672 illegal actions — the deferred leaf-mask leakage)
  - prior@opt        : raw-policy rank/prob of a DTZ-optimal move
  - prior_top_ok     : is the policy head's argmax-legal move WDL-win-preserving + DTZ-optimal?
  - visit_top_ok     : is MCTS's argmax-visit move WDL-win-preserving + DTZ-optimal?
  - val_spread       : std of the model's value over the legal children (can value RANK moves?)
  - ml_spread / ml_ok: std of moves-left over children, and is argmin-ML child DTZ-optimal?
  - dtz              : syzygy DTZ from mover POV (progress toward mate)

Run: CKPT=... USE_ATTENTION=1 USE_SMOLGEN=1 USE_DYN_ATTENTION=1 ATTN_LAYERS=6 \
     PYTHONPATH=. .venv/bin/python scripts/trace_conversion_bottleneck.py
"""
import os, pickle, numpy as np, torch, chess, chess.syzygy
from collections import Counter, defaultdict
from src.config import get_config
from src.games.chess import ChessGame, _action_to_move, _move_to_action
from src.games.chess_gpu import GpuChessGame
from src.model.muzero_net import MuZeroNetwork
from src.mcts.tensor_mcts import TensorMCTS

DEV="cuda"; cfg=get_config("chess_small"); game=ChessGame(); AS=game.action_space_size; NF=cfg.history_frames
gg=GpuChessGame(); torch.manual_seed(0); np.random.seed(0)
CKPT=os.environ["CKPT"]; ATTN=os.environ.get("USE_ATTENTION","0")=="1"; SMOL=os.environ.get("USE_SMOLGEN","1")=="1"
PREDATTN=os.environ.get("USE_PRED_ATTENTION","0")=="1"; DYNATTN=os.environ.get("USE_DYN_ATTENTION","0")=="1"
ATTNL=int(os.environ.get("ATTN_LAYERS","4")); SIMS=200; MAX_PLIES=80
WANT=os.environ.get("CONFIGS","KRvK,KPvK,KQvK").split(","); N_GAMES=int(os.environ.get("N_GAMES","8"))

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
print(f"loaded {CKPT}\n", flush=True)
from src.model.utils import support_to_scalar, inverse_scalar_transform

def encode(fens):
    st=gg.from_python_chess([chess.Board(f) for f in fens], device=DEV); obs=gg.to_tensor_batch(st)
    N,C,H,W=obs.shape
    if NF>1: obs=torch.cat([obs, torch.zeros(N,(NF-1)*C,H,W,device=DEV)],1)
    return obs

@torch.no_grad()
def model_eval_positions(fens):
    """Return per-position (value_scalar, moves_left_scalar) from the model."""
    h=net.representation(encode(fens)); _,vl=net.prediction(h)
    # wdl head: V = P(W)-P(L)
    p=torch.softmax(vl,1); v=(p[:,0]-p[:,2]).cpu().numpy()
    ml=inverse_scalar_transform(support_to_scalar(net.predict_moves_left(h), cfg.moves_left_support_size)).clamp(min=0).cpu().numpy().reshape(-1)
    return v, ml

@torch.no_grad()
def policy_prior(fen):
    h=net.representation(encode([fen])); pl,_=net.prediction(h)
    return torch.softmax(pl,1)[0].cpu().numpy()   # [AS]

def sig(b, winner):
    wp=sorted(p.symbol().upper() for _,p in b.piece_map().items() if p.color==winner and p.piece_type!=chess.KING)
    lp=sorted(p.symbol().upper() for _,p in b.piece_map().items() if p.color!=winner and p.piece_type!=chess.KING)
    return f"K{''.join(wp)}vK{''.join(lp)}"

def optimal_actions(board, tb, winner):
    """DTZ-optimal, WDL-win-preserving moves (mover POV), as action indices."""
    rows=[]
    for mv in board.legal_moves:
        board.push(mv)
        try: wdl=-tb.probe_wdl(board); dtz=-tb.probe_dtz(board)
        except Exception: wdl,dtz=-99,99
        board.pop()
        rows.append((mv,wdl,dtz))
    wins=[(mv,dtz) for mv,wdl,dtz in rows if wdl>=2]      # still winning for us
    if not wins: return set(), rows
    best=min(d for _,d in wins)                            # smallest DTZ = fastest mate
    opt={_move_to_action(mv, board.turn) for mv,d in wins if d==best}
    return opt, rows

@torch.no_grad()
def term_draw_mask(boards):
    fdm=torch.zeros(len(boards), AS, dtype=torch.bool, device=DEV)
    for j,b in enumerate(boards):
        mover=b.turn
        for mv in b.legal_moves:
            b.push(mv); td=b.is_stalemate() or b.is_insufficient_material(); b.pop()
            if td: fdm[j,_move_to_action(mv,mover)]=True
    return fdm

cfg.num_simulations=SIMS; cfg.moves_left_mcts=True; cfg.tb_root_probe=False; cfg.root_terminal_draws=True
mcts=TensorMCTS(net, game, cfg, device=DEV, hidden_dtype=torch.float32, select_backend="eager")
tb=chess.syzygy.open_tablebase("data/syzygy")
te=pickle.load(open("data/tb5_test.pkl","rb")); te_fen=[x[0] for x in te]; te_v=np.array([x[1] for x in te])

# collect a few won positions per requested config
picks=defaultdict(list)
for i in range(len(te_fen)):
    if te_v[i]!=0: continue
    b=chess.Board(te_fen[i]); s=sig(b,b.turn)
    if s in WANT and len(picks[s])<N_GAMES: picks[s].append(te_fen[i])

agg=Counter()
for s in WANT:
    for gi,fen in enumerate(picks[s]):
        board=chess.Board(fen); winner=board.turn
        print(f"\n===== {s} game {gi}  {fen}", flush=True)
        print(f"  {'ply':>3} {'legalP':>6} {'pTopOK':>6} {'vTopOK':>6} {'mlTopOK':>7} {'valSpr':>6} {'mlSpr':>6} {'rootV':>6} {'rootML':>6} {'dtz':>5}", flush=True)
        outcome="cap"
        for ply in range(MAX_PLIES):
            if board.is_game_over():
                outcome=('mate' if (board.is_checkmate() and board.turn!=winner) else
                         ('lost' if board.is_checkmate() else 'draw')); break
            mover=board.turn
            opt, rows = optimal_actions(board, tb, winner)
            legal_acts=[_move_to_action(mv,mover) for mv in board.legal_moves]
            prior=policy_prior(board.fen())
            legalP=float(prior[legal_acts].sum())
            ptop=max(legal_acts, key=lambda a: prior[a]); pTopOK=int(ptop in opt)
            # child value/ML spread over legal moves
            child_fens=[]; child_acts=[]
            for mv in board.legal_moves:
                board.push(mv); child_fens.append(board.fen()); board.pop(); child_acts.append(_move_to_action(mv,mover))
            cv, cml = model_eval_positions(child_fens)
            cv=-cv  # child value is from opponent POV; negate to mover POV
            valSpr=float(np.std(cv)); mlSpr=float(np.std(cml))
            vtop_act=child_acts[int(np.argmax(cv))]; vTopOK=int(vtop_act in opt)      # value picks fastest?
            mltop_act=child_acts[int(np.argmin(cml))]; mlTopOK=int(mltop_act in opt)  # moves-left picks fastest?
            rv, rml = model_eval_positions([board.fen()]); rv=float(rv[0]); rml=float(rml[0])
            try: dtz=tb.probe_dtz(board)
            except Exception: dtz=0
            # actual MCTS move
            lm=torch.zeros(1, AS, dtype=torch.bool, device=DEV)
            for a in legal_acts: lm[0,a]=True
            rd=mcts.run_batch_gpu(encode([board.fen()]), lm, add_noise=False, forced_draw_mask=term_draw_mask([board]))
            ca=rd["child_actions"][0].cpu().numpy(); cvis=rd["child_visits"][0].cpu().numpy()
            slot=int(np.argmax(np.where(ca!=-1, cvis, -1))); mv_act=int(ca[slot])
            mv=_action_to_move(mv_act, board);
            if mv is None or mv not in board.legal_moves: mv=list(board.legal_moves)[0]
            visit_top_ok=int(_move_to_action(mv,mover) in opt)
            if ply<24:
                print(f"  {ply:3d} {legalP:6.3f} {pTopOK:6d} {vTopOK:6d} {mlTopOK:7d} {valSpr:6.3f} {mlSpr:6.2f} {rv:6.3f} {rml:6.1f} {dtz:5d}  vis_ok={visit_top_ok}", flush=True)
            agg['plies']+=1; agg['pTopOK']+=pTopOK; agg['vTopOK']+=vTopOK; agg['mlTopOK']+=mlTopOK; agg['visTopOK']+=visit_top_ok
            agg['legalP']+=legalP; agg['valSpr']+=valSpr; agg['mlSpr']+=mlSpr
            board.push(mv)
        print(f"  -> {outcome}", flush=True); agg[outcome]+=1

n=max(agg['plies'],1)
print("\n===== AGGREGATE over", agg['plies'], "plies =====", flush=True)
print(f"  policy-prior argmax is DTZ-optimal : {agg['pTopOK']/n:.2%}", flush=True)
print(f"  value argmax-child is DTZ-optimal  : {agg['vTopOK']/n:.2%}", flush=True)
print(f"  moves-left argmin-child DTZ-optimal: {agg['mlTopOK']/n:.2%}", flush=True)
print(f"  MCTS argmax-visit is DTZ-optimal   : {agg['visTopOK']/n:.2%}", flush=True)
print(f"  mean legal-prior mass (vs illegal) : {agg['legalP']/n:.3f}", flush=True)
print(f"  mean value spread over children    : {agg['valSpr']/n:.4f}", flush=True)
print(f"  mean moves-left spread over children: {agg['mlSpr']/n:.3f}", flush=True)
print(f"  outcomes: {dict((k,agg[k]) for k in ('mate','draw','cap','lost') if agg[k])}", flush=True)
tb.close(); print("DONE", flush=True)
