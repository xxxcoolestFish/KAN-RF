"""Evaluate all trained models side by side:
forward MSE, inverse |a_err|, control-point roughness, Jacobian cos_sim.

Usage:
  python eval_all.py
"""
import torch, numpy as np, glob
from kanrf import KAN
from kanrf import p_spline_penalty


def true_jacobian(s_next_norm):
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
    a = a_norm.clone().detach().reshape(1, 1).requires_grad_(True)
    s_pred = model(torch.cat([s.unsqueeze(0), a], dim=-1))
    J_model = torch.zeros(3)
    for dim in range(3):
        g = torch.autograd.grad(s_pred[0, dim], a, retain_graph=True)[0]
        J_model[dim] = g.detach()
    J_true = true_jacobian(s_pred.detach()).squeeze()
    cos = (J_model @ J_true) / (J_model.norm() * J_true.norm() + 1e-10)
    return cos.item()


def eval_model(model_path, x_test, y_test, x_inv, y_inv, n_inv=200, n_jac=50):
    model = load_model(model_path)
    device = torch.device('cpu')
    model.to(device)

    # Forward MSE
    with torch.no_grad():
        pred = model(x_test)
        val_mse = ((pred - y_test) ** 2).mean().item()

    # Roughness
    roughness = p_spline_penalty(model).item()

    # Inverse accuracy
    a_errs = []
    for i in range(n_inv):
        a_opt, _ = inverse_error(model, x_inv[i, :3], y_inv[i], n_restarts=3, n_iters=150)
        a_errs.append(abs(a_opt - x_inv[i, 3].item()))
    a_errs = np.array(a_errs)

    # Jacobian cos_sim
    jac_idx = torch.randperm(len(x_inv))[:n_jac]
    cos_vals = [jacobian_cos_sim(model, x_inv[j, :3], x_inv[j, 3]) for j in jac_idx]
    cos_mean = np.mean(cos_vals)

    return val_mse, roughness, a_errs, cos_mean


def main():
    # Data
    x_all, y_all = torch.load("pendulum_data_v4.pt", weights_only=True)
    n_train = int(len(x_all) * 0.85)
    x_test, y_test = x_all[n_train:], y_all[n_train:]
    n_inv = 200
    idx = torch.randperm(len(x_test))[:n_inv]
    x_inv, y_inv = x_test[idx], y_test[idx]

    models = sorted(glob.glob("kan_mops_lam*.pt") + glob.glob("kan_cws_nu*.pt") + glob.glob("kan_hybrid_*.pt"))

    if not models:
        print("No model files found.")
        return

    print(f"{'method':>18s}  {'val_MSE':>9s}  {'|a_err|_mean':>11s}  "
          f"{'|a_err|_med':>10s}  {'|a_err|_P90':>10s}  "
          f"{'cos_sim':>8s}  {'||Δ²c||':>8s}")
    print("-" * 100)

    for path in models:
        val_mse, roughness, a_errs, cos_mean = eval_model(path, x_test, y_test, x_inv, y_inv, n_inv, n_jac=40)
        # Short label
        label = path.replace("kan_", "").replace(".pt", "")
        print(f"{label:>18s}  {val_mse:9.6f}  {a_errs.mean():11.4f}  "
              f"{np.median(a_errs):10.4f}  {np.percentile(a_errs, 90):10.4f}  "
              f"{cos_mean:8.4f}  {roughness:8.4f}")


if __name__ == "__main__":
    main()
