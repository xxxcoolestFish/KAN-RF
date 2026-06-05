"""Multi-step shooting planner: optimize action sequence through frozen KAN world model."""
import torch
from kanrf import KAN


def shoot(model: KAN, s0: torch.Tensor, s_target: torch.Tensor,
          horizon: int = 40, n_iters: int = 500, lr: float = 0.1,
          lambda_ctrl: float = 0.001, n_restarts: int = 3,
          verbose: bool = True, log_fn=None
          ) -> tuple[torch.Tensor, torch.Tensor]:
    _log = log_fn if log_fn is not None else (lambda *a, **kw: print(*a, **kw) or None)
    """Optimize action sequence via gradient descent through frozen KAN.

    Works in normalized space (theta_dot/8, torque/2).
    Returns denormalized actions for execution.

    Args:
        model: frozen KAN world model f(s_norm, a_norm) → s_next_norm
        s0: (1, 3) initial state [cosθ, sinθ, θ̇]  — raw (unnormalized)
        s_target: (1, 3) target state [cosθ, sinθ, θ̇]  — raw
        horizon: number of steps to plan
        n_iters: inner-loop Adam iterations per restart
        lr: Adam learning rate
        lambda_ctrl: control cost weight
        n_restarts: number of random initializations, take best

    Returns:
        actions_raw: (horizon, 1) denormalized torques in [-2, 2]
        final_state_raw: (1, 3) predicted final state (denormalized)
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Normalize state and target
    s0_norm = s0.clone()
    s0_norm[:, 2] /= 8.0
    s_target_norm = s_target.clone()
    s_target_norm[:, 2] /= 8.0

    best_loss = float('inf')
    best_actions_norm = None

    for restart in range(n_restarts):
        a_norm = torch.zeros(horizon, 1, requires_grad=True)
        torch.nn.init.uniform_(a_norm, a=-0.3, b=0.3)  # small random init
        a_norm.requires_grad_(True)

        opt = torch.optim.Adam([a_norm], lr=lr)

        for step in range(n_iters):
            opt.zero_grad()
            s = s0_norm.clone()
            for h in range(horizon):
                x = torch.cat([s, a_norm[h:h + 1]], dim=-1)  # (1, 4)
                s = model(x)
                norm = (s[:, :2] ** 2).sum(dim=-1, keepdim=True).sqrt().clamp(min=1e-6)
                s = torch.cat([s[:, :2] / norm, s[:, 2:]], dim=-1)

            loss_terminal = ((s - s_target_norm) ** 2).sum()
            loss_ctrl = (a_norm ** 2).sum()
            loss = loss_terminal + lambda_ctrl * loss_ctrl
            loss.backward()
            opt.step()
            with torch.no_grad():
                a_norm.clamp_(-1.0, 1.0)

            if verbose and (step % 100 == 0 or step == n_iters - 1):
                with torch.no_grad():
                    s_final = s0_norm.clone()
                    for h in range(horizon):
                        s_final = model(torch.cat([s_final, a_norm[h:h+1]], dim=-1))
                        nrm = s_final[:, :2].norm(dim=-1, keepdim=True).clamp(min=1e-6)
                        s_final = torch.cat([s_final[:, :2] / nrm, s_final[:, 2:]], dim=-1)
                    angle_err = torch.acos(
                        (s_final[:, :2] * s_target_norm[:, :2]).sum(-1).clamp(-1, 1)
                    ).item()
                _log(f"    restart {restart+1}/{n_restarts}  "
                     f"iter {step:4d}  "
                     f"loss={loss_terminal.item():.4f}+{lambda_ctrl*loss_ctrl.item():.4f}  "
                     f"|Δθ|={angle_err:.3f}rad")

        with torch.no_grad():
            total_loss = loss_terminal.item() + lambda_ctrl * loss_ctrl.item()

        if total_loss < best_loss:
            best_loss = total_loss
            best_actions_norm = a_norm.detach().clone()

    for p in model.parameters():
        p.requires_grad = True

    # Denormalize actions
    actions_raw = best_actions_norm * 2.0

    # Predict final state with denormalized actions
    with torch.no_grad():
        s = s0_norm.clone()
        for h in range(horizon):
            x = torch.cat([s, best_actions_norm[h:h + 1]], dim=-1)
            s = model(x)
            norm = (s[:, :2] ** 2).sum(dim=-1, keepdim=True).sqrt().clamp(min=1e-6)
            s = torch.cat([s[:, :2] / norm, s[:, 2:]], dim=-1)
        s_final_norm = s.clone()
        s_final_norm[:, 2] *= 8.0  # denormalize theta_dot
        # De-normalize cos/sin (already ~normalized, just keep as-is)

    return actions_raw, s_final_norm
