"""Probe the 0.5 hidden-state grad hook in trainer._train_step.

The code registers, inside the unroll loop, on the dynamics OUTPUT each step:
    hidden, ... = recurrent_inference_logits(hidden, action_k)
    hidden.register_hook(lambda grad: grad * 0.5)

MuZero's spec (Appendix G) is to scale the gradient at the *start* of the
dynamics function by 1/2 — i.e. halve the gradient flowing from step k+1 back
into the (h_k, a_k) that produced it, applied ONCE per unroll transition.

Questions:
  (1) Is the hook applied exactly once per hidden tensor, or does the closure /
      re-registration accidentally double-apply or skip?
  (2) Does the deepest hidden state get 0.5^K (compounded) as intended by the
      recurrent half-scaling, and is that what MuZero specifies?
  (3) Late-binding closure bug? All lambdas capture the SAME `grad` arg (no), but
      do they reference a loop variable incorrectly? (lambda grad: grad*0.5 has
      no free loop var, so safe — verify.)

We build a tiny synthetic unroll mirroring the trainer and measure the actual
gradient multiplier landing on each step's hidden via a reference (no-hook) run.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from src.config import get_config, MuZeroConfig
from src.games.chess import ChessGame
from scripts.eval_checkpoint_health import build_network


def run_unroll(net, obs, actions, K, with_hook):
    """Mirror trainer's unroll; return (value_loss_root_sum, per-step hidden refs+grads)."""
    hooks_seen = {"count": 0}
    hidden, policy_logits, value_logits = net.initial_inference_logits(obs)
    hiddens = [hidden]
    hidden.retain_grad()
    for k in range(K):
        hidden, reward_logits, policy_logits, value_logits = \
            net.recurrent_inference_logits(hidden, actions[:, k])
        if with_hook:
            def hk(grad, _c=hooks_seen):
                _c["count"] += 1
                return grad * 0.5
            hidden.register_hook(hk)
        hidden.retain_grad()
        hiddens.append(hidden)
    # Loss only on the DEEPEST value logits so we can read the multiplier chain.
    loss = value_logits.float().sum()
    loss.backward()
    return loss, hiddens, hooks_seen["count"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.serialization.add_safe_globals([MuZeroConfig])
    dev = args.device
    game = ChessGame()
    cfg = get_config("chess_small"); cfg.device = dev
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=True)
    net = build_network(ckpt, game, cfg, dev)
    net.train()

    K = cfg.num_unroll_steps
    B = 4
    C = game.num_planes * cfg.history_frames
    obs = torch.randn(B, C, 8, 8, device=dev)
    actions = torch.randint(0, game.action_space_size, (B, K), device=dev)

    print(f"=== 0.5 hidden-hook probe (K={K}) ===")

    # Reference: no hook. Grad on each hidden from deepest-value loss.
    net.zero_grad(set_to_none=True)
    _, hiddens_ref, _ = run_unroll(net, obs.clone(), actions, K, with_hook=False)
    ref_norms = [h.grad.float().norm().item() if h.grad is not None else 0.0 for h in hiddens_ref]

    # With hook.
    net.zero_grad(set_to_none=True)
    _, hiddens_hk, nhooks = run_unroll(net, obs.clone(), actions, K, with_hook=True)
    hk_norms = [h.grad.float().norm().item() if h.grad is not None else 0.0 for h in hiddens_hk]

    print(f"hooks fired during backward: {nhooks} (expected exactly K={K}, one per recurrent step)")
    print(f"\n{'step':>4} {'ref_grad_norm':>16} {'hook_grad_norm':>16} {'ratio(hook/ref)':>16} {'expected 0.5^?':>16}")
    for i in range(K + 1):
        r = ref_norms[i]; h = hk_norms[i]
        ratio = (h / r) if r > 1e-12 else float("nan")
        # hidden[i] is produced before i hooks downstream of it apply... actually
        # grad into hidden[i] passes through (K - i) hook applications? No: the
        # hook is on the OUTPUT hidden of step k = hiddens[k+1]. Grad from the
        # deepest value (on hiddens[K]) flows back; each register_hook on
        # hiddens[j] (j>=1) multiplies grad as it passes THROUGH that node.
        # Grad arriving at hiddens[i] has passed through hooks on hiddens[K..i+1]
        # = (K - i) hooks for i<K, and the hook on hiddens[K] itself applies to
        # grad LEAVING hiddens[K]. So multiplier at hiddens[i] ~ 0.5^(K-i) for the
        # body grad, with the node's own hook included.
        exp_pow = (K - i)
        print(f"{i:>4} {r:>16.6e} {h:>16.6e} {ratio:>16.4f} {'0.5^'+str(exp_pow)+'='+format(0.5**exp_pow,'.4f'):>16}")

    # Double-check: does the value HEAD weight grad get halved by the hook chain?
    # (It shouldn't be halved beyond the per-step rule; value head reads hiddens[K]
    #  directly so its grad is unaffected by hooks on hiddens — confirm.)
    print("\nNote: the hook multiplies grad PASSING THROUGH each output-hidden node "
          "once. A node grad that has traversed m hook-nodes is scaled 0.5^m. "
          "The value/policy/reward HEAD weights at step k read hiddens[k] and are "
          "NOT halved by hooks downstream of them — only the recurrent body grad is.")


if __name__ == "__main__":
    main()
