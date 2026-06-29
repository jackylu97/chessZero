"""Does the value head still RESPOND to its input, or has it collapsed to a
constant (input-independent) output? Run on 8 wildly different positions and
report the per-position std of the 3 WDL logits at each checkpoint. If the std
collapses toward 0 by step 30000, the value head is emitting a fixed
distribution regardless of position (the terminal signature of value blindness).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np, chess
from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from eval_checkpoint_health import build_network

torch.serialization.add_safe_globals([MuZeroConfig])
game = ChessGame(); cfg = get_config("chess_small"); cfg.device = "cpu"
hf = cfg.history_frames
fens = [chess.STARTING_FEN,
        "4k3/8/8/8/8/8/8/3QK3 w - - 0 1", "3qk3/8/8/8/8/8/8/4K3 w - - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/ppp2ppp/2n1bn2/3pp3/3PP3/2N1BN2/PPP2PPP/R3K2R w KQkq - 0 1",
        "8/8/8/3k4/8/3K4/8/7R w - - 0 1", "8/8/8/3k4/8/3K4/8/7r w - - 0 1",
        "6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1"]
obs = []
for f in fens:
    st = game.reset_from_fen(f); cur = game.to_tensor(st)
    obs.append(torch.cat([cur] + [torch.zeros_like(cur)] * (hf - 1), 0))
X = torch.stack(obs)
import torch.nn.functional as F
for step in [1000, 6000, 30000]:
    ck = torch.load(f"checkpoints/chess/2026_06_19_cold2_pc/checkpoint_{step}.pt",
                    map_location="cpu", weights_only=True)
    net = build_network(ck, game, cfg, "cpu")
    with torch.no_grad():
        vl = net.prediction(net.representation(X))[1].float()
        wdl = F.softmax(vl, -1).numpy()
    vln = vl.numpy()
    print(f"step {step}: logit per-pos std (W,D,L)={np.round(vln.std(0),3)} "
          f"mean={np.round(vln.mean(0),3)} | P(D) range=[{wdl[:,1].min():.3f},{wdl[:,1].max():.3f}]")
