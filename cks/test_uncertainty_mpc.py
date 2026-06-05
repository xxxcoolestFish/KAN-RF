"""Uncertainty-weighted MPC: score(a,k) = task_loss + λ·U(s)/||J||².

U(s) from B-spline activation density + control point roughness.
No hard k selection — k is scored alongside actions.
"""
import torch, numpy as np, gymnasium as gym, sys, os, time
from kanrf import KAN
from cks_core import (state_activation_density, control_point_roughness,
                       compute_activation_density, _forward_k)

PI_2 = np.pi / 2
K_SET = (1, 2, 4, 8, 16)
MAX_K_EXEC = 1  # always execute 1 step (50Hz replanning)


def compute_uncertainty(s_norm, model, density, w1=0.5, w2=0.05):
    """U(s) = w1·(1-ρ̄(s)) + w2·‖Δ²c‖_max"""
    rho = state_activation_density(s_norm, model, density)
    _, d2 = control_point_roughness(model)
    return w1 * (1.0 - rho) + w2 * d2


def jacobian_norm_fd(s_norm, a_norm, model, state_dim, action_dim, eps=0.01):
    """||∂f/∂a|| via finite difference."""
    s_a = _forward_k(s_norm, a_norm, model, state_dim, action_dim, 1)
    s_ap = _forward_k(s_norm, a_norm + eps, model, state_dim, action_dim, 1)
    return (s_ap - s_a).norm().item() / eps


def uncertainty_weighted_mpc(s_norm, model, density, state_dim, action_dim,
                              score_fn, action_set, lam=0.1):
    """Score all (a,k) pairs.  Returns best (a, k) and diagnostics."""
    U = compute_uncertainty(s_norm, model, density)
    best_score = float('inf')
    best_a, best_k = None, 1
    diag = {'U': U}

    for a in action_set:
        if isinstance(a, (int, float)):
            a_t = torch.tensor([[a]], dtype=torch.float32)
        else:
            a_t = a

        J_norm = jacobian_norm_fd(s_norm, a_t, model, state_dim, action_dim)
        if J_norm < 1e-6:
            J_norm = 1e-6

        for k in K_SET:
            pred = _forward_k(s_norm, a_t, model, state_dim, action_dim, k)
            task_score = score_fn(pred)
            total_score = task_score + lam * U / (J_norm * J_norm)

            if total_score < best_score:
                best_score = total_score
                best_a, best_k = a, k

            if k == K_SET[0]:
                diag[f'task_a{a}'] = task_score

    diag['best_score'] = best_score
    diag['best_k'] = best_k
    return best_a, best_k, diag


# ─── Pendulum Test ──────────────────────────────────────────────────────

def pendulum_score(pred):
    """Distance to upright [0,1,0] in normalized state space."""
    target = torch.tensor([[0., 1., 0.]])
    return (pred - target).pow(2).sum().item()


def test_pendulum(wm_path, lam=0.1, n_trials=10):
    wm = KAN([5, 16, 3], grid_size=5, spline_order=3)
    wm.load_state_dict(torch.load(wm_path, weights_only=True))
    wm.eval()
    density = compute_activation_density(wm)

    env = gym.make('Pendulum-v1')
    k_used = []; successes = 0

    for t in range(n_trials):
        trial_seed = 42 + t * 100
        obs, _ = env.reset(seed=trial_seed)
        for step in range(200):
            s_norm = torch.tensor([[obs[0], obs[1], obs[2]/8.0]], dtype=torch.float32)

            # Uncertainty-weighted MPC over continuous action space
            # For Pendulum, actions are continuous. Use inverse optimization weighted by k.
            best_a, best_k_val, best_total = None, 1, float('inf')
            for k in K_SET:
                # Inverse optimization: minimize f(s,a,k) → upright
                a = torch.zeros(1, 1)
                a.requires_grad_(True)
                opt = torch.optim.Adam([a], lr=0.05)
                for _ in range(50):
                    opt.zero_grad()
                    pred = _forward_k(s_norm, a, wm, 3, 1, k)
                    loss = (pred - torch.tensor([[0., 1., 0.]])).pow(2).sum()
                    loss.backward()
                    opt.step()
                    with torch.no_grad():
                        a.clamp_(-1.0, 1.0)

                with torch.no_grad():
                    pred_final = _forward_k(s_norm, a, wm, 3, 1, k)
                    task_score = (pred_final - torch.tensor([[0., 1., 0.]])).pow(2).sum().item()

                U = compute_uncertainty(s_norm, wm, density)
                J_norm = jacobian_norm_fd(s_norm, a.item(), wm, 3, 1)
                if J_norm < 1e-6: J_norm = 1e-6
                total = task_score + lam * U / (J_norm * J_norm)

                if total < best_total:
                    best_total = total
                    best_a = a.item()
                    best_k_val = k

            a_raw = best_a * 2.0
            k_used.append(best_k_val)
            obs, _, term, _, _ = env.step([a_raw])

            angle = np.arctan2(obs[1], obs[0])
            if abs(angle - PI_2) < 0.2:
                successes += 1
                break

        init_err = abs(np.arctan2(
            np.arctan2(env.unwrapped.state[1], env.unwrapped.state[0])
            if hasattr(env.unwrapped, 'state') else 0, 0) - PI_2)
        print(f'  Trial {t+1}: {"OK" if successes > (0 if t==0 else successes-sum(1 for _ in range(t) if False)) else "FAIL"}  k_avg={np.mean(k_used[-50:]):.1f}')

    env.close()
    print(f'  Success: {successes}/{n_trials}  avg_k={np.mean(k_used):.1f}')
    return successes


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lam', type=float, default=0.1)
    args = parser.parse_args()

    print('Uncertainty-Weighted MPC on Pendulum')
    test_pendulum('/Users/zhuangxinyu/KAN/KAN-RF/kan_ms.pt', lam=args.lam)
