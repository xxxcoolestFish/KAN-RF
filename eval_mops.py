"""Evaluate MOPS-trained models: forward MSE, inverse accuracy,
control-point roughness, and Jacobian cosine similarity.

Usage:
  python eval_mops.py
"""
import torch, numpy as np
from kanrf import KAN
from kanrf import p_spline_penalty


def true_jacobian(s_next_norm):
    """Analytic ds'_norm / da_norm for Pendulum-v1."""
    cos_p, sin_p = s_next_norm[..., 0], s_next_norm[..., 1]
    J_cos = -0.015 * sin_p
    J_sin =  0.015 * cos_p
    J_thd =  0.0375 * torch.ones_like(cos_p)
    return torch.stack([J_cos, J_sin, J_thd], dim=-1)


def load_model(path):
    ckpt = torch.load(path, weights_only=True)
    dims = [4]
    for k in sorted(ckpt.keys()):
        if 'base_weight' in k:
            dims.append(ckpt[k].shape[0])
    model = KAN(dims, grid_size=5, spline_order=3)
    model.load_state_dict(ckpt)
    return model.eval()


def inverse_error(model, s, s_target, n_restarts=5, n_iters=200):
    """Find a* that minimizes ||f(s,a) - s_target||², compare to a_true."""
    best_loss, best_a = float('inf'), None
    for _ in range(n_restarts):
        a = torch.empty(1, 1)
        torch.nn.init.uniform_(a, -1, 1)
        a.requires_grad_(True)
        opt = torch.optim.Adam([a], lr=0.05)
        for __ in range(n_iters):
            opt.zero_grad()
            loss = ((model(torch.cat([s.unsqueeze(0), a], dim=-1)) - s_target.unsqueeze(0)) ** 2).sum()
            loss.backward(); opt.step()
            with torch.no_grad(): a.clamp_(-1.0, 1.0)
        with torch.no_grad():
            fl = ((model(torch.cat([s.unsqueeze(0), a], dim=-1)) - s_target.unsqueeze(0)) ** 2).sum().item()
        if fl < best_loss: best_loss = fl; best_a = a.detach().clone()
    return best_a.item(), best_loss


def jacobian_cos_sim(model, s, a_norm):
    """Cosine similarity between model Jacobian and true Jacobian."""
    a = a_norm.clone().detach().reshape(1, 1).requires_grad_(True)
    s_pred = model(torch.cat([s.unsqueeze(0), a], dim=-1))
    J_model = torch.zeros(3)
    for dim in range(3):
        g = torch.autograd.grad(s_pred[0, dim], a, retain_graph=True)[0]
        J_model[dim] = g.detach()
    J_true = true_jacobian(s_pred.detach()).squeeze()
    cos = (J_model @ J_true) / (J_model.norm() * J_true.norm() + 1e-10)
    return cos.item()


def main():
    device = torch.device('cpu')
    lam_values = [0.0, 0.01, 0.1, 1.0]

    # Load test data
    x_all, y_all = torch.load("pendulum_data_v4.pt", weights_only=True)
    n_train = int(len(x_all) * 0.85)
    x_test, y_test = x_all[n_train:], y_all[n_train:]

    # Inverse evaluation subset
    n_inv = 200
    idx = torch.randperm(len(x_test))[:n_inv]
    x_inv, y_inv = x_test[idx], y_test[idx]

    print(f"{'λ':>6s}  {'val_MSE':>9s}  {'|a_err|_mean':>11s}  "
          f"{'|a_err|_med':>10s}  {'|a_err|_P90':>10s}  "
          f"{'cos_sim_mean':>12s}  {'||Δ²c||_mean':>11s}  {'model_file':>25s}")
    print("-" * 110)

    for lam in lam_values:
        fname = f"kan_mops_lam{lam}.pt"
        model = load_model(fname).to(device)

        # Forward MSE
        with torch.no_grad():
            pred = model(x_test)
            val_mse = ((pred - y_test) ** 2).mean().item()

        # Control point roughness
        roughness = p_spline_penalty(model).item()

        # Inverse accuracy
        a_errs = []
        for i in range(n_inv):
            s_i = x_inv[i, :3]
            a_true = x_inv[i, 3].item()
            s_tgt = y_inv[i]
            a_opt, _ = inverse_error(model, s_i, s_tgt)
            a_errs.append(abs(a_opt - a_true))
        a_errs = np.array(a_errs)

        # Jacobian cosine similarity (random subset for speed)
        n_jac = 50
        jac_idx = torch.randperm(len(x_inv))[:n_jac]
        cos_vals = []
        for i in range(n_jac):
            s_i = x_inv[jac_idx[i], :3]
            a_i = x_inv[jac_idx[i], 3]
            cos_vals.append(jacobian_cos_sim(model, s_i, a_i))
        cos_mean = np.mean(cos_vals)

        print(f"{lam:6.2f}  {val_mse:9.6f}  {a_errs.mean():11.4f}  "
              f"{np.median(a_errs):10.4f}  {np.percentile(a_errs, 90):10.4f}  "
              f"{cos_mean:12.4f}  {roughness:11.4f}  {fname:>25s}")


if __name__ == "__main__":
    main()
