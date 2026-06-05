"""Multi-step shooting with B-spline uncertainty penalty.

Extension of shoot.py: adds KAN-specific epistemic uncertainty penalty
to prevent model exploitation. The B-spline activation density is a free
signal — no extra model training needed.
"""
import torch
from kanrf import KAN
from kanrf import compute_uncertainty


def shoot_uncertainty(model: KAN, s0: torch.Tensor, s_target: torch.Tensor,
                      horizon: int = 10, n_iters: int = 200, lr: float = 0.1,
                      lambda_ctrl: float = 0.01, beta_unc: float = 0.01,
                      n_restarts: int = 1, sigma2: float = 0.01,
                      verbose: bool = True, log_fn=None):
    """Multi-step shooting with B-spline uncertainty penalty.

    Optimizes action sequence A = [a_0, ..., a_{H-1}] through frozen KAN:
      min_A  ||s_H - s*||² + λ Σ||a_h||² + β Σ L_unc(s_h, a_h)

    The uncertainty penalty discourages the optimizer from venturing
    into regions where B-spline activations are sparse (model is uncertain).

    Args:
        model: frozen KAN world model f(s_norm, a_norm) → s_next_norm
        s0: (1, 3) initial state [cosθ, sinθ, θ̇] — raw
        s_target: (1, 3) target state [cosθ, sinθ, θ̇] — raw
        horizon: planning steps
        n_iters: Adam iterations per restart
        lr: learning rate
        lambda_ctrl: control cost weight
        beta_unc: uncertainty penalty weight
        n_restarts: random restarts
        sigma2: B-spline uncertainty scale parameter
        verbose, log_fn: logging

    Returns:
        actions_raw: (horizon, 1) denormalized torques
        final_state_raw: (1, 3) predicted final state
        diag: diagnostics dict
    """
    _log = log_fn if log_fn is not None else (lambda *a, **kw: None)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Normalize
    s0_norm = s0.clone(); s0_norm[:, 2] /= 8.0
    s_target_norm = s_target.clone(); s_target_norm[:, 2] /= 8.0

    best_loss = float('inf')
    best_actions_norm = None
    best_diag = {}

    for restart in range(n_restarts):
        a_norm = torch.zeros(horizon, 1, requires_grad=False)
        torch.nn.init.uniform_(a_norm, a=-0.3, b=0.3)
        a_norm.requires_grad_(True)

        opt = torch.optim.Adam([a_norm], lr=lr)

        for step in range(n_iters):
            opt.zero_grad()
            s = s0_norm.clone()
            loss_unc_total = 0.0

            for h in range(horizon):
                x = torch.cat([s, a_norm[h:h + 1]], dim=-1)
                s, B_list, E_list = model(x, return_activations=True)
                loss_unc_total += compute_uncertainty(B_list, E_list, sigma2)

                # cos/sin normalization
                norm = (s[:, :2] ** 2).sum(dim=-1, keepdim=True).sqrt().clamp(min=1e-6)
                s = torch.cat([s[:, :2] / norm, s[:, 2:]], dim=-1)

            loss_terminal = ((s - s_target_norm) ** 2).sum()
            loss_ctrl = (a_norm ** 2).sum()
            loss = loss_terminal + lambda_ctrl * loss_ctrl + beta_unc * loss_unc_total

            loss.backward()
            opt.step()
            with torch.no_grad():
                a_norm.clamp_(-1.0, 1.0)

            if verbose and step % 50 == 0:
                with torch.no_grad():
                    s_check = s0_norm.clone()
                    for h in range(horizon):
                        s_check = model(torch.cat([s_check, a_norm[h:h+1]], dim=-1))
                        nrm = s_check[:, :2].norm(dim=-1, keepdim=True).clamp(min=1e-6)
                        s_check = torch.cat([s_check[:, :2] / nrm, s_check[:, 2:]], dim=-1)
                    angle_err = torch.acos(
                        (s_check[:, :2] * s_target_norm[:, :2]).sum(-1).clamp(-1, 1)
                    ).item()
                _log(f"    restart {restart+1}/{n_restarts}  "
                     f"iter {step:4d}  "
                     f"term={loss_terminal.item():.4f}  "
                     f"ctrl={lambda_ctrl*loss_ctrl.item():.4f}  "
                     f"unc={beta_unc*loss_unc_total.item():.4f}  "
                     f"|Δθ|={angle_err:.3f}rad")

        with torch.no_grad():
            total_loss = loss_terminal.item() + lambda_ctrl * loss_ctrl.item() \
                         + beta_unc * loss_unc_total.item()

        if total_loss < best_loss:
            best_loss = total_loss
            best_actions_norm = a_norm.detach().clone()
            best_diag = {
                'loss_terminal': loss_terminal.item(),
                'loss_ctrl': lambda_ctrl * loss_ctrl.item(),
                'loss_unc': beta_unc * loss_unc_total.item(),
            }

    for p in model.parameters():
        p.requires_grad = True

    actions_raw = best_actions_norm * 2.0

    # Predict final state
    with torch.no_grad():
        s = s0_norm.clone()
        for h in range(horizon):
            x = torch.cat([s, best_actions_norm[h:h + 1]], dim=-1)
            s = model(x)
            nrm = s[:, :2].norm(dim=-1, keepdim=True).clamp(min=1e-6)
            s = torch.cat([s[:, :2] / nrm, s[:, 2:]], dim=-1)
        s_final_norm = s.clone()
        s_final_norm[:, 2] *= 8.0

    return actions_raw, s_final_norm, best_diag
