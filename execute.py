"""Execution Layer: single-step gradient descent through frozen KAN.

Given (s, s_mid), finds a* by optimizing:
    min_a ||f_KAN(s, a) - s_mid||² + lambda * a²

Only ONE forward pass through KAN per optimization iteration.
No rollout — model exploitation window is eliminated.
"""
import torch
from kanrf import KAN


def execute(model: KAN, s: torch.Tensor, s_mid: torch.Tensor,
            n_iter: int = 150, lr: float = 0.1,
            lambda_ctrl: float = 0.01,
            a_min: float = -2.0, a_max: float = 2.0) -> float:
    """Find action a* to move from s toward s_mid in one step.

    Args:
        model: frozen KAN world model f(s_norm, a_norm) → s_next_norm
        s: (1, 3) current state [cosθ, sinθ, θ̇] — RAW (unnormalized)
        s_mid: (1, 3) target intermediate state — RAW
        n_iter: Adam inner-loop iterations
        lr: Adam learning rate
        lambda_ctrl: control cost weight
        a_min, a_max: action bounds

    Returns:
        a_raw: scalar action (torque in [-2, 2])
        final_loss: final optimization loss (diagnostic — high loss = KAN couldn't
                    find a good action for this s→s_mid)
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Normalize
    s_norm = s.clone(); s_norm[:, 2] /= 8.0
    s_mid_norm = s_mid.clone(); s_mid_norm[:, 2] /= 8.0

    # Initialize action (warm start from zero)
    a_norm = torch.zeros(1, 1, requires_grad=True)
    opt = torch.optim.Adam([a_norm], lr=lr)

    for _ in range(n_iter):
        opt.zero_grad()
        x = torch.cat([s_norm, a_norm], dim=-1)
        s_pred = model(x)
        loss = ((s_pred - s_mid_norm) ** 2).sum() + lambda_ctrl * (a_norm ** 2).sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            a_norm.clamp_(a_min / 2.0, a_max / 2.0)

    for p in model.parameters():
        p.requires_grad = True

    final_loss = loss.item()
    return (a_norm.detach().item() * 2.0, final_loss)
