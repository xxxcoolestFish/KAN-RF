"""Phase 2: Decision via gradient descent in action space through frozen KAN."""
import torch
from kanrf import KAN
from env import PointMass


def decide(model: KAN, s: torch.Tensor, s_target: torch.Tensor,
           n_iter: int = 200, lr: float = 0.1,
           a_min: float = -0.5, a_max: float = 0.5) -> torch.Tensor:
    """Find action a* to drive s to s_target via gradient descent.

    Optimizes L(a) = ||f_KAN(s, a) - s_target||^2 using Adam in action space,
    with KAN parameters frozen.

    Args:
        model: frozen KAN world model f(s, a) → s_next
        s: (batch, state_dim) current state
        s_target: (batch, state_dim) desired next state
        n_iter: inner-loop optimization steps
        lr: Adam learning rate for action optimization
        a_min, a_max: action bounds
    Returns:
        a: (batch, action_dim) optimized action
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    a = torch.zeros_like(s, requires_grad=True)
    opt = torch.optim.Adam([a], lr=lr)

    for _ in range(n_iter):
        opt.zero_grad()
        s_pred = model(torch.cat([s, a], dim=-1))
        loss = ((s_pred - s_target) ** 2).sum(-1).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            a.clamp_(a_min, a_max)

    for p in model.parameters():
        p.requires_grad = True
    return a.detach()


def main():
    torch.manual_seed(42)

    # Load trained world model
    model = KAN([4, 5, 2], grid_size=5, spline_order=3)
    model.load_state_dict(torch.load("kan_world_model.pt", weights_only=True))
    model.eval()

    env = PointMass(nonlinear=False)

    # --- Test: sample (s, a_true), compute s_target = s + a_true, recover a ---
    n_tests = 10
    print(f"{'s_t':^20s}  {'a_true':^20s}  {'s*':^20s}  {'a* (pred)':^20s}  {'|a_pred-a_true|':>16s}")
    print("-" * 105)

    for _ in range(n_tests):
        s = (torch.rand(1, 2) * 2 - 1) * 0.5           # state in [-0.5, 0.5]
        a_true = (torch.rand(1, 2) * 2 - 1) * 0.5       # action in [-0.5, 0.5]
        s_target = s + a_true                            # reachable target in [-1, 1]

        a_pred = decide(model, s, s_target, n_iter=200, lr=0.05)

        s_str = f"[{s[0,0]:.2f},{s[0,1]:.2f}]"
        g_str = f"[{a_true[0,0]:.3f},{a_true[0,1]:.3f}]"
        t_str = f"[{s_target[0,0]:.2f},{s_target[0,1]:.2f}]"
        p_str = f"[{a_pred[0,0]:.3f},{a_pred[0,1]:.3f}]"
        diff = (a_pred - a_true).abs().max().item()
        print(f"{s_str:>20s}  {g_str:>20s}  {t_str:>20s}  {p_str:>20s}  {diff:16.6f}")

    # --- Batch evaluation ---
    s_batch = (torch.rand(100, 2) * 2 - 1) * 0.5
    a_true_batch = (torch.rand(100, 2) * 2 - 1) * 0.5
    targets = s_batch + a_true_batch

    a_pred_batch = decide(model, s_batch, targets, n_iter=200, lr=0.05)
    action_error = (a_pred_batch - a_true_batch).abs()

    with torch.no_grad():
        s_next_batch = env.step(s_batch, a_pred_batch)
    state_error = (s_next_batch - targets).abs()

    print(f"\nBatch eval (n=100):")
    print(f"  |a_pred - a_true| mean: {action_error.mean():.6f}  max: {action_error.max():.6f}")
    print(f"  |s_next - s*|     mean: {state_error.mean():.6f}  max: {state_error.max():.6f}")


if __name__ == "__main__":
    main()
