"""Reward-precision probe: mate-in-1 recall vs false-fire on non-mating winning
moves, through the dynamics (as search consumes it). THE standing instrument for
the conversion channel (strategy doc §11-12). Usage:
  CFG=chess_hybrid_xl REWARD_PLANES=8 POLICY_HEAD=from_to PYTHONPATH=. \
    .venv/bin/python scripts/probe_reward_precision.py <checkpoint.pt>
"""
import os
import sys, pickle, numpy as np, torch, chess
from src.config import get_config
from src.games.chess import ChessGame, _move_to_action
from src.games.chess_gpu import GpuChessGame
from src.model.muzero_net import MuZeroNetwork
from src.model.utils import support_to_scalar
CKPT=sys.argv[1]
DEV="cuda"; cfg=get_config(os.environ.get("CFG","chess_hybrid")); game=ChessGame(); gg=GpuChessGame(); NF=cfg.history_frames
net=MuZeroNetwork(observation_channels=game.num_planes*NF, action_space_size=game.action_space_size,
    hidden_planes=cfg.hidden_planes, num_blocks=cfg.num_residual_blocks, latent_h=8, latent_w=8,
    input_h=8, input_w=8, fc_hidden=cfg.fc_hidden, value_support_size=cfg.value_support_size,
    reward_support_size=cfg.reward_support_size, reward_head_planes=int(os.environ.get("REWARD_PLANES","8")),
    action_embed_dim=cfg.action_embed_dim, value_head_type="wdl",
    policy_head_type=os.environ.get("POLICY_HEAD", cfg.policy_head_type), use_moves_left=True,
    moves_left_support_size=cfg.moves_left_support_size, moves_left_head_planes=8,
    use_material_head=False, value_head_planes=cfg.value_head_planes,
    use_repr_attention=True, use_dyn_attention=True, use_pred_attention=True, use_smolgen=True,
    attn_layers=cfg.attn_layers, attn_heads=cfg.attn_heads, pred_attn_layers=cfg.pred_attn_layers,
    hybrid_stem_blocks=cfg.hybrid_stem_blocks).to(DEV)
r=net.load_state_dict(torch.load(CKPT, map_location=DEV)["model_state_dict"], strict=False)
assert not r.missing_keys; net.eval()
def enc(fens):
    st=gg.from_python_chess([chess.Board(f) for f in fens], device=DEV); obs=gg.to_tensor_batch(st)
    N,C,H,W=obs.shape
    if NF>1: obs=torch.cat([obs, torch.zeros(N,(NF-1)*C,H,W,device=DEV)],1)
    return obs
te=pickle.load(open("data/tb5_test.pkl","rb"))
mates=[]; nonmates=[]
for rec in te:
    if rec[1]!=0: continue
    b=chess.Board(rec[0])
    for mv in list(b.legal_moves):
        b.push(mv); is_m=b.is_checkmate(); b.pop()
        if is_m:
            mates.append((rec[0], _move_to_action(mv, b.turn)))
            for mv2 in list(b.legal_moves):
                b.push(mv2); m2=b.is_checkmate(); over=b.is_game_over(); b.pop()
                if not m2 and not over:
                    nonmates.append((rec[0], _move_to_action(mv2, b.turn))); break
            break
    if len(mates)>=200: break
with torch.no_grad():
    for tag, pairs in [("mate", mates), ("non-mate-win", nonmates)]:
        h=net.representation(enc([f for f,_ in pairs]))
        a=torch.tensor([x for _,x in pairs], device=DEV)
        _, rl = net.dynamics(h, a)
        rr=support_to_scalar(rl, cfg.reward_support_size).cpu().numpy().reshape(-1)
        print(f"{tag}: mean {rr.mean():.3f} fire>0.5: {(rr>0.5).mean():.2f}", flush=True)
