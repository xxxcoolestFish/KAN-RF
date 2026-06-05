"""Multi-step shooting with learnable horizon via sigmoid mask.

The action at step h is:  a_h = σ(α·(H_learn - h)) · ã_h
  - H_learn: learnable scalar — the "soft horizon"
  - α: steepness (higher = sharper cutoff)
  - ã_h: raw action parameters

Both H_learn and ã_h are optimized jointly.
The mask naturally decays actions beyond the effective horizon.
"""
import torch
from kanrf import KAN


def shoot_variable_horizon(model, s0, s_target, h_max=30, n_iters=300, lr=0.1,
                           lambda_ctrl=0.01, n_restarts=3, alpha=2.0,
                           init_horizon=None, verbose=True):
    """Shooting with learnable horizon via sigmoid action mask.

    Args:
        model: frozen KAN world model
        s0: (1, 3) initial state [cos, sin, thd] — raw
        s_target: (1, 3) target state — raw
        h_max: maximum planning horizon
        n_iters, lr, lambda_ctrl, n_restarts: standard shooting params
        alpha: sigmoid steepness (higher = sharper transition)
        init_horizon: initial H_learn value (default: h_max/2)
        verbose: print progress

    Returns:
        actions_raw: (h_effective, 1) optimized torques (trimmed to effective horizon)
        final_state_raw: (1, 3) predicted final state
        diag: dict with H_learn, full_mask, etc.
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    s0_norm = s0.clone(); s0_norm[:, 2] /= 8.0
    s_target_norm = s_target.clone(); s_target_norm[:, 2] /= 8.0

    if init_horizon is None:
        init_horizon = h_max / 2

    best_loss = float('inf')
    best_result = None

    for restart in range(n_restarts):
        # Raw actions + learnable horizon
        a_raw = torch.zeros(h_max, 1, requires_grad=False)
        torch.nn.init.uniform_(a_raw, a=-0.3, b=0.3)
        a_raw.requires_grad_(True)

        H_learn = torch.tensor([init_horizon], requires_grad=True).float()

        params = [a_raw, H_learn]
        opt = torch.optim.Adam(params, lr=lr)

        for step in range(n_iters):
            opt.zero_grad()

            # Sigmoid mask: m_h ≈ 1 for h ≪ H_learn, ≈ 0 for h ≫ H_learn
            h_idx = torch.arange(h_max, dtype=torch.float32).unsqueeze(1)  # (h_max, 1)
            mask = torch.sigmoid(alpha * (H_learn - h_idx))                 # (h_max, 1)
            a_masked = mask * a_raw

            # Rollout
            s = s0_norm.clone()
            for h in range(h_max):
                x = torch.cat([s, a_masked[h:h+1]], dim=-1)
                s = model(x)
                # Cos/sin normalization
                nrm = s[:, :2].norm(dim=-1, keepdim=True).clamp(min=1e-6)
                s = torch.cat([s[:, :2] / nrm, s[:, 2:]], dim=-1)

            loss_terminal = ((s - s_target_norm) ** 2).sum()
            loss_ctrl = (a_masked ** 2).sum()
            loss_h = 0.001 * H_learn  # encourage shorter horizon
            loss = loss_terminal + lambda_ctrl * loss_ctrl + loss_h

            loss.backward()
            opt.step()

            if verbose and step % 100 == 0:
                with torch.no_grad():
                    h_eff = H_learn.item()
                    # Angle error
                    cos_err = (s[:, :2] * s_target_norm[:, :2]).sum(-1).clamp(-1, 1)
                    angle_err = torch.acos(cos_err).item()
                print(f"  restart {restart+1}/{n_restarts}  iter {step:4d}  "
                      f"H={h_eff:.1f}  term={loss_terminal.item():.4f}  "
                      f"ctrl={lambda_ctrl*loss_ctrl.item():.4f}  |Δθ|={angle_err:.3f}rad")

        with torch.no_grad():
            total_loss = loss_terminal.item() + lambda_ctrl * loss_ctrl.item()

        if total_loss < best_loss:
            best_loss = total_loss
            h_eff = H_learn.detach().item()
            mask_final = torch.sigmoid(alpha * (H_learn.detach() - h_idx))
            a_final = (mask_final * a_raw.detach()).clone()
            best_result = {
                'actions_norm': a_final,
                'H_learn': h_eff,
                'mask': mask_final,
                'loss': total_loss,
            }

    # Trim to effective horizon
    with torch.no_grad():
        mask = best_result['mask'].squeeze()
        # Find last step where mask > 0.01 (effective action)
        effective_steps = (mask > 0.01).sum().item()
        h_eff = max(int(effective_steps), 1)
        actions_norm = best_result['actions_norm'][:h_eff]

    # Predict final state
    with torch.no_grad():
        s = s0_norm.clone()
        for h in range(len(actions_norm)):
            x = torch.cat([s, actions_norm[h:h+1]], dim=-1)
            s = model(x)
            nrm = s[:, :2].norm(dim=-1, keepdim=True).clamp(min=1e-6)
            s = torch.cat([s[:, :2] / nrm, s[:, 2:]], dim=-1)
        s_final = s.clone(); s_final[:, 2] *= 8.0

    for p in model.parameters():
        p.requires_grad = True

    diag = {'H_learn': best_result['H_learn'], 'mask': best_result['mask']}
    return actions_norm * 2.0, s_final, diag
