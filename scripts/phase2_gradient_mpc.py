"""Phase 2: Gradient-guided action optimization vs random shooting.

Compare gradient-based search (KAN + true-dynamics) with random shooting,
all starting from Transport nominal with action trust region.

Key question: does gradient guidance provide meaningful improvement
over random sampling, and does predicted improvement correlate with
real improvement?
"""

from __future__ import annotations

import argparse, sys, os, json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.prescreen_hopper_physics_shifts import SHIFTS, make_shifted_env
from scripts.validate_hopper_joint_online_adaptation import FrozenSourcePolicy

# Hopper-v5 exact params
FORWARD_WEIGHT, HEALTHY_REWARD, CTRL_COST_WEIGHT = 1.0, 1.0, 0.001
Z_LO, Z_HI = 0.7, float('inf')
ANG_LO, ANG_HI = -0.2, 0.2
DT = 0.008 * 4


def hopper_reward_smooth_torch(obs, action, prev_obs=None):
    """Differentiable Hopper reward surrogate."""
    fwd = obs[..., 5] if prev_obs is None else (obs[..., 0] - prev_obs[..., 0]) / DT
    z, a = obs[..., 1], obs[..., 2]
    h_z = torch.sigmoid((z - Z_LO) / 0.02) * torch.sigmoid((Z_HI - z) / 0.02) if Z_HI != float('inf') else torch.sigmoid((z - Z_LO) / 0.02)
    h_a = torch.sigmoid((a - ANG_LO) / 0.05) * torch.sigmoid((ANG_HI - a) / 0.05)
    ctrl = CTRL_COST_WEIGHT * (action ** 2).sum(dim=-1)
    return FORWARD_WEIGHT * fwd + HEALTHY_REWARD * h_z * h_a - ctrl


def hopper_reward_np(obs, action, prev_obs=None):
    """Exact Hopper reward (numpy)."""
    fwd = obs[5] if prev_obs is None else (obs[0] - prev_obs[0]) / DT
    z_ok = Z_LO < obs[1] < Z_HI
    a_ok = ANG_LO < obs[2] < ANG_HI
    healthy = 1.0 if (z_ok and a_ok) else 0.0
    ctrl = CTRL_COST_WEIGHT * np.sum(action ** 2)
    return FORWARD_WEIGHT * fwd + HEALTHY_REWARD * healthy - ctrl


@torch.no_grad()
def kan_rollout(s0, da_seq, a0_seq, basis, ctx, H):
    """KAN dynamics rollout. Returns (states, actions, cumulative reward)."""
    s = s0.clone()
    total_r = 0.0
    for t in range(H):
        a = (a0_seq[t:t+1] + da_seq[t:t+1]).clamp(-1, 1)
        effect = ctx.acceleration(basis, s, a)
        s_next = s + effect
        r = hopper_reward_smooth_torch(s_next.squeeze(0), a.squeeze(0))
        total_r += r
        s = s_next
    return total_r


def optimize_gradient(s0, a0_seq, basis, ctx, H, eps_a, lr, n_steps, l2_reg, smooth_reg,
                      safety_weight=5.0):
    """Gradient-based optimization of action residuals da_seq.

    Includes safety barrier to prevent termination.
    """
    da = torch.zeros(H, a0_seq.shape[1], device=s0.device, requires_grad=True)
    opt = torch.optim.Adam([da], lr=lr)

    for _ in range(n_steps):
        opt.zero_grad()
        s = s0.clone()
        total_r = 0.0
        reg = 0.0
        safety = 0.0
        for t in range(H):
            a = (a0_seq[t:t+1] + da[t:t+1]).clamp(-1, 1)
            effect = ctx.acceleration(basis, s, a)
            s_next = s + effect
            total_r += hopper_reward_smooth_torch(s_next.squeeze(0), a.squeeze(0))

            # Safety barrier: penalize going below healthy threshold
            z = s_next[0, 1]
            angle = s_next[0, 2]
            safety += F.softplus(0.75 - z) * safety_weight  # z < 0.75 → penalty
            safety += F.softplus(angle.abs() - 0.18) * safety_weight  # |angle| > 0.18 → penalty

            reg += l2_reg * (da[t] ** 2).sum()
            if t > 0:
                reg += smooth_reg * ((da[t] - da[t-1]) ** 2).sum()
            s = s_next

        loss = -total_r + reg + safety
        loss.backward()
        opt.step()

        with torch.no_grad():
            da.clamp_(-eps_a, eps_a)

    return da.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--H", type=int, default=4)
    parser.add_argument("--eps-a", type=float, default=0.10)
    parser.add_argument("--n-states", type=int, default=30)
    parser.add_argument("--json-out", default="results/phase2_gradient_mpc.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    shift = SHIFTS["friction_070"]
    H = args.H

    print(f"Phase 2: Gradient MPC (H={H}, eps_a={args.eps_a})")

    # Load
    source_policy = FrozenSourcePolicy(
        "results/hopper_source_sb3_ppo_continued_seed1811.zip",
        "results/hopper_source_sb3_vecnorm_continued_seed1811.pkl",
        device, args.seed, env="hopper",
    )
    import argparse as argparse_mod
    from scripts.diagnose_hopper_pullback_effect import fit_distilled_source_counterfactual_context, load_source_twin
    from scripts.validate_hopper_joint_online_adaptation import load_cognition

    source_twin = load_source_twin("results/hopper_source_affine_twin_cloud_seed1811.pt", device)
    basis, source_context, _, _ = load_cognition(
        argparse_mod.Namespace(cognition_checkpoint="results/hopper_source_control_sobolev_calibrated_seed1811.pt", device="cuda"),
        device,
    )
    fit_args = argparse_mod.Namespace(
        target="friction_070", seed=args.seed, env="hopper", device="cuda",
        cognition_warmup=1024, warmup_noise=0.3, transform_ridge=10.0, drift_ridge=100.0,
        drift_spectral_eta=0.0, drift_spectral_beta=1.0, drift_spectral_mode="max",
        drift_smooth_lambda=0.0, diagonal_transform=False,
    )
    target_ctx, _ = fit_distilled_source_counterfactual_context(
        source_policy, basis, source_context, fit_args, device, source_twin,
    )

    # Collect states with Transport
    fd_env = make_shifted_env(shift, args.seed, "hopper")()
    obs, _ = fd_env.reset(seed=args.seed)
    records = []
    for _ in range(500):
        s_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        nominal = source_policy.action(s_t)
        s_eff = source_context.acceleration(basis, s_t, nominal)
        a_tr = target_ctx.transport_action(basis, s_t, desired_effect=s_eff, nominal_action=nominal, regularization=1e-2).clamp(-1,1).squeeze(0).cpu().numpy()
        qp = fd_env.unwrapped.data.qpos.copy(); qv = fd_env.unwrapped.data.qvel.copy()
        records.append({"qp": qp, "qv": qv, "obs": obs.copy(), "a_tr": a_tr})
        obs, _, t, tr, _ = fd_env.step(a_tr)
        if t or tr: break

    n_test = min(args.n_states, len(records))
    idx = np.linspace(0, len(records)-1, n_test, dtype=int)

    results = {"transport": [], "random_shooting": [], "kan_gradient": [],
               "kan_pred_improve": [], "random_pred_improve": [],
               "kan_real_improve": [], "random_real_improve": []}

    for i in idx:
        rec = records[i]
        s0 = torch.as_tensor(rec["obs"], device=device, dtype=torch.float32).unsqueeze(0)
        a0 = torch.as_tensor(rec["a_tr"], device=device, dtype=torch.float32).unsqueeze(0)
        a0_seq = a0.repeat(H, 1)

        # --- Transport baseline ---
        fd_env.unwrapped.set_state(rec["qp"], rec["qv"])
        tr_total = 0.0; prev = None
        for _ in range(H):
            obs_r, _, t, tr, _ = fd_env.step(rec["a_tr"])
            tr_total += hopper_reward_np(obs_r, rec["a_tr"], prev)
            prev = obs_r.copy()
            if t or tr: break
        results["transport"].append(tr_total)

        # --- Random shooting (baseline) ---
        best_rand_r = tr_total  # fallback to transport if all die
        best_rand_da = np.zeros((H, 3))
        best_rand_pred = 0.0
        for _ in range(256):
            da_rand = np.random.randn(H, 3) * args.eps_a * 0.5
            da_rand = np.clip(da_rand, -args.eps_a, args.eps_a)
            # Predicted (KAN)
            pred_r = float(kan_rollout(s0, torch.as_tensor(da_rand, device=device, dtype=torch.float32), a0_seq, basis, target_ctx, H))
            # Real
            fd_env.unwrapped.set_state(rec["qp"], rec["qv"])
            real_r = 0.0; prev_r = None
            alive = True
            for t in range(H):
                a_t = np.clip(rec["a_tr"] + da_rand[t], -1, 1)
                obs_r, _, terminated, truncated, _ = fd_env.step(a_t)
                real_r += hopper_reward_np(obs_r, a_t, prev_r)
                prev_r = obs_r.copy()
                if terminated or truncated: alive = False; break
            if real_r > best_rand_r and alive:
                best_rand_r = real_r; best_rand_da = da_rand; best_rand_pred = pred_r

        results["random_shooting"].append(best_rand_r)
        results["random_pred_improve"].append(best_rand_pred)
        results["random_real_improve"].append(best_rand_r - tr_total)

        # --- KAN Gradient optimization ---
        da_grad = optimize_gradient(
            s0, a0_seq, basis, target_ctx, H,
            eps_a=args.eps_a, lr=0.01, n_steps=50,
            l2_reg=0.01, smooth_reg=0.05,
        )
        da_np = da_grad.cpu().numpy()

        # Predicted improvement
        pred_after = float(kan_rollout(s0, da_grad, a0_seq, basis, target_ctx, H))

        # Real execution
        fd_env.unwrapped.set_state(rec["qp"], rec["qv"])
        grad_real = 0.0; prev_g = None; alive = True
        for t in range(H):
            a_t = np.clip(rec["a_tr"] + da_np[t], -1, 1)
            obs_g, _, terminated, truncated, _ = fd_env.step(a_t)
            grad_real += hopper_reward_np(obs_g, a_t, prev_g)
            prev_g = obs_g.copy()
            if terminated or truncated: alive = False; break

        if alive:
            results["kan_gradient"].append(grad_real)
        else:
            results["kan_gradient"].append(tr_total)  # fell back to transport
        results["kan_pred_improve"].append(pred_after)
        results["kan_real_improve"].append(grad_real - tr_total if alive else 0.0)

    fd_env.close()

    # Report
    tr_m = np.mean(results["transport"])
    rs_m = np.mean(results["random_shooting"])
    kg_m = np.mean(results["kan_gradient"]) if results["kan_gradient"] else 0

    print(f"\n=== Phase 2 Results ===")
    print(f"  Transport:           {tr_m:.4f}")
    print(f"  Random shooting:     {rs_m:.4f}  ({rs_m-tr_m:+.4f})")
    print(f"  KAN Gradient:        {kg_m:.4f}  ({kg_m-tr_m:+.4f})")

    if results["kan_real_improve"]:
        imp = np.array(results["kan_real_improve"])
        print(f"  Gradient improved:   {np.mean(imp > 0):.1%}")
        print(f"  Mean real improv:    {np.mean(imp):+.4f}")

    # Predicted vs real correlation
    if len(results["kan_pred_improve"]) > 5 and len(results["kan_real_improve"]) > 5:
        corr = np.corrcoef(results["kan_pred_improve"], results["kan_real_improve"])[0, 1]
        print(f"  Pred vs real corr:   {corr:.4f}")

    kg_vs_tr = kg_m - tr_m
    if kg_vs_tr > 0.02:
        print(f"\n  Verdict: PASS — gradient optimization beats Transport")
    elif kg_vs_tr > 0:
        print(f"\n  Verdict: MARGINAL — minimal improvement")
    else:
        print(f"\n  Verdict: FAIL — gradient doesn't help in current setup")


if __name__ == "__main__":
    main()
