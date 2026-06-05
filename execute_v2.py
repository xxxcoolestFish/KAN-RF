"""Execution Layer v2: Gauss-Newton warm start + controllability-weighted loss.

Given (s, v_des), finds a* by:
  1. Gauss-Newton step from a=0 using exact KAN Jacobian J_a = df/da
  2. Controllability-weighted loss: theta_dot (controllable) weighted high,
     cos/sin (uncontrollable) weighted low
  3. Adam refinement from the Gauss-Newton warm start

Key improvement over v1: no intermediate state s_mid needed.
The desired velocity field v_des directly specifies direction and rate.
"""
import torch
import torch.autograd.functional as AF
from kanrf import KAN


def gauss_newton_init(model, s_norm, v_des_norm):
    """Single Gauss-Newton step at a=0 using exact KAN Jacobian.

    Linearizes: f(s, a) ≈ f(s, 0) + J_a * a  where J_a = df/da|_{a=0}
    Solves:    min_a ||f(s,0) + J_a*a - (s + v_des)||^2

    Returns (a_init_norm, f_zero).
    """
    a_zero = torch.zeros(1, 1, requires_grad=True)
    x = torch.cat([s_norm, a_zero], dim=-1)

    # f(s, 0): no grad needed
    with torch.no_grad():
        f_zero = model(x.clone())

    # Jacobian J_a = df/da at a=0  (3,) vector
    f_a = lambda a_: model(torch.cat([s_norm, a_], dim=-1))
    J = AF.jacobian(f_a, a_zero)  # (1, 3, 1, 1)
    J_a = J.squeeze()             # (3,)

    # Residual: r = s_target - f_zero
    s_target = s_norm + v_des_norm
    residual = (s_target - f_zero).squeeze(0)  # (3,)

    # Gauss-Newton: a = (J^T J)^(-1) J^T r
    JtJ = (J_a ** 2).sum()
    JtR = (J_a * residual).sum()

    if JtJ > 1e-8:
        a_init = JtR / JtJ
    else:
        a_init = torch.tensor(0.0)

    a_init = torch.clamp(a_init, -1.0, 1.0)
    return a_init.item(), f_zero, J_a


def execute_v2(model: KAN, s: torch.Tensor, v_des: torch.Tensor,
               n_iter: int = 15, lr: float = 0.05,
               lambda_ctrl: float = 0.01,
               w_controllable: float = 3.0,
               a_min: float = -2.0, a_max: float = 2.0):
    """Find a* to move in direction v_des.

    Args:
        model: frozen KAN
        s: (1, 3) current state [cos, sin, thd] — raw
        v_des: (1, 3) desired velocity [dcos, dsin, dthd] — per-step delta, raw
        n_iter: Adam refinement steps (default 15)
        lr: Adam learning rate
        lambda_ctrl: control penalty
        w_controllable: extra weight on theta_dot (the directly controllable dim)
    Returns:
        a_raw: scalar torque, final_loss, a_init (diagnostic)
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Normalize state
    s_norm = s.clone(); s_norm[:, 2] /= 8.0
    v_des_norm = v_des.clone(); v_des_norm[:, 2] /= 8.0

    # --- Gauss-Newton warm start ---
    a_norm_init, f_zero, J_a = gauss_newton_init(model, s_norm, v_des_norm)
    a_norm = torch.tensor([[a_norm_init]], dtype=torch.float32, requires_grad=True)

    # --- Adam refinement ---
    s_target = s_norm + v_des_norm
    opt = torch.optim.Adam([a_norm], lr=lr)

    for _ in range(n_iter):
        opt.zero_grad()
        x = torch.cat([s_norm, a_norm], dim=-1)
        s_pred = model(x)

        err = s_pred - s_target  # (1, 3)
        # Only optimize theta_dot (controllable). cos/sin are not
        # directly controllable — torque changes theta_ddot, which
        # integrates to theta_dot. Position follows via natural dynamics.
        loss = w_controllable * err[:, 2]**2 + lambda_ctrl * (a_norm ** 2).sum()

        loss.backward()
        opt.step()
        with torch.no_grad():
            a_norm.clamp_(a_min / 2.0, a_max / 2.0)

    for p in model.parameters():
        p.requires_grad = True

    final_loss = loss.item()
    a_raw = a_norm.detach().item() * 2.0
    a_init_raw = a_norm_init * 2.0

    # Diagnostics dict for debugging wrong-direction decisions
    # Denormalize theta_dot for readability
    f_zero_raw = f_zero.squeeze(0).clone(); f_zero_raw[2] *= 8.0
    s_pred_raw = s_pred.detach().squeeze(0).clone(); s_pred_raw[2] *= 8.0
    s_target_raw = s_target.squeeze(0).clone(); s_target_raw[2] *= 8.0
    # J_a[2] needs scaling: d(thd_norm)/da_norm * (8/2) = d(thd_norm)/da_norm * 4
    J_a_raw = J_a.clone(); J_a_raw[2] *= 4.0  # df_thd_raw / da_raw
    diag = {
        'a_init': a_init_raw,
        'a_final': a_raw,
        'final_loss': final_loss,
        'f_zero': f_zero_raw.numpy(),
        's_pred': s_pred_raw.numpy(),
        's_target': s_target_raw.numpy(),
        'J_a': J_a_raw.numpy(),
    }
    return (a_raw, final_loss, a_init_raw, diag)
