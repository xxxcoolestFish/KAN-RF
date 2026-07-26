"""Experiment 3: KAN reachability calibration.

Can KAN dynamics model predict local-state reachability well enough
for the task router to select good goals?

Key metrics:
  1. Spearman rank correlation: do KAN and real env agree on goal ordering?
  2. Top-k hit rate: do KAN's top picks match real top picks?
  3. False positive rate: does KAN say "reachable" when it's not?
  4. Horizon sensitivity: what H gives best prediction?
"""

from __future__ import annotations

import argparse, sys, os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cpbn.cognitive_pca import CognitivePCA, PCARanges
from cpbn.generic_affine_kan import AffineKANContext
from scripts.diagnose_hopper_pullback_effect import (
    fit_distilled_source_counterfactual_context,
    load_source_twin,
)
from scripts.prescreen_hopper_physics_shifts import SHIFTS, make_shifted_env
from scripts.validate_hopper_joint_online_adaptation import (
    FrozenSourcePolicy,
    load_cognition,
)


def collect_source_states(n_states=50, seed=1811):
    """Collect diverse states from source env."""
    source_policy = FrozenSourcePolicy(
        "results/hopper_source_sb3_ppo_continued_seed1811.zip",
        "results/hopper_source_sb3_vecnorm_continued_seed1811.pkl",
        torch.device("cuda"), seed, env="hopper",
    )
    env = make_shifted_env(SHIFTS["source"], seed, "hopper")()
    obs, _ = env.reset(seed=seed)
    states = []
    for _ in range(n_states * 3):
        a = source_policy.action(torch.tensor(obs, dtype=torch.float32)).cpu().numpy()
        # Add some exploration
        a = np.clip(a + np.random.randn(*a.shape) * 0.1, -1, 1)
        obs, _, t, tr, _ = env.step(a)
        states.append(obs.copy())
        if t or tr:
            obs, _ = env.reset()
    env.close()
    # Sample uniformly to cover diverse states
    idx = np.linspace(0, len(states) - 1, n_states, dtype=int)
    return np.stack([states[i] for i in idx])


def generate_diverse_goals(n_goals=20, s_dim=11):
    """Generate diverse candidate goals covering reachable and unreachable."""
    goals = []
    # Various types of goals
    for _ in range(n_goals // 5):
        # Forward + stable (likely reachable)
        g = np.zeros(s_dim)
        g[0] = np.random.uniform(0.5, 2.0)  # forward
        g[1] = np.random.uniform(0.8, 1.3)  # height
        g[5] = np.random.uniform(1.0, 4.0)  # velocity
        goals.append(g)

    for _ in range(n_goals // 5):
        # High/risky (maybe reachable)
        g = np.zeros(s_dim)
        g[0] = np.random.uniform(0, 1.0)
        g[1] = np.random.uniform(0.9, 1.5)
        g[2] = np.random.uniform(-0.3, 0.3)  # tilted
        goals.append(g)

    for _ in range(n_goals // 5):
        # Clearly unreachable (far, impossible pose)
        g = np.zeros(s_dim)
        g[0] = np.random.uniform(5, 10)  # very far
        g[1] = np.random.uniform(0.2, 0.5)  # too low
        goals.append(g)

    for _ in range(n_goals - len(goals)):
        # Random
        g = np.random.randn(s_dim) * 2
        g[0] += 1.0
        g[1] += 1.0
        goals.append(g)

    return np.stack(goals).astype(np.float32)


def kan_reachability_cost(states, goals, basis, drifted_ctx, device,
                          horizon=5, n_candidates=32):
    """Estimate reachability cost using KAN dynamics + random shooting.

    For each (s, g) pair, minimize ||s_H - g|| via random action sequences.

    Returns: costs array (n_states, n_goals)
    """
    N, s_dim = states.shape
    K = len(goals)
    a_dim = basis.action_dim

    costs = np.zeros((N, K))
    s_t = torch.tensor(states, device=device, dtype=torch.float32)

    for i in range(N):
        s_i = s_t[i:i+1]
        for j in range(K):
            g_j = torch.tensor(goals[j], device=device, dtype=torch.float32)

            best_dist = float('inf')
            # Random shooting
            for _ in range(n_candidates):
                s_curr = s_i.clone()
                for _ in range(horizon):
                    a = torch.randn(1, a_dim, device=device) * 0.3
                    a = a.clamp(-1, 1)
                    effect = drifted_ctx.acceleration(basis, s_curr, a)
                    s_curr = s_curr + effect
                dist = float(torch.norm(s_curr.squeeze(0) - g_j))
                if dist < best_dist:
                    best_dist = dist
            costs[i, j] = best_dist

    return costs


def real_reachability_cost(states, goals, executor, source_policy, shift,
                           horizon=5, n_episodes_per=3, seed=1911):
    """Measure real reachability using executor in target environment.

    For each (s, g), run executor in target env and measure final distance.

    Returns: costs array (n_states, n_goals)
    """
    N = len(states)
    K = len(goals)
    device = next(executor.parameters()).device

    costs = np.zeros((N, K))
    for i in range(N):
        s_i = states[i]
        for j in range(K):
            g_j = goals[j]
            best_dist = float('inf')

            for ep in range(n_episodes_per):
                env = make_shifted_env(shift, seed + ep * 100, "hopper")()
                # Cannot directly set state, so just start from reset and use first H steps
                # Alternative: use goal directly from current state
                obs, _ = env.reset(seed=seed + ep * 100)
                # Execute from env start for H steps towards goal
                s_curr = obs.copy()
                for _ in range(horizon):
                    s_t = torch.tensor(s_curr, device=device, dtype=torch.float32).unsqueeze(0)
                    g_t = torch.tensor(g_j, device=device, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        a = executor(s_t, g_t).clamp(-1, 1).squeeze(0).cpu().numpy()
                    obs, _, t, tr, _ = env.step(a)
                    s_curr = obs.copy()
                    if t or tr:
                        break
                env.close()
                dist = float(np.linalg.norm(s_curr - g_j))
                if dist < best_dist:
                    best_dist = dist
            costs[i, j] = best_dist

    return costs


def compute_metrics(kan_costs, real_costs):
    """Compute calibration metrics between KAN and real reachability."""
    kan_flat = kan_costs.flatten()
    real_flat = real_costs.flatten()

    # Spearman rank correlation
    rho, pval = spearmanr(kan_flat, real_flat)

    # Top-k hit rate
    k = 3
    n_states = len(kan_costs)
    hits = 0
    for i in range(n_states):
        kan_top = np.argsort(kan_costs[i])[:k]
        real_top = np.argsort(real_costs[i])[:k]
        hits += len(set(kan_top) & set(real_top))
    topk_hit = hits / (n_states * k)

    # False positive rate: KAN says reachable, real says not
    reachable_thresh = np.percentile(real_flat, 30)  # top 30% are "reachable"
    kan_reachable = kan_flat < reachable_thresh
    real_not_reachable = real_flat >= reachable_thresh
    fp_rate = float((kan_reachable & real_not_reachable).mean())

    # Mean absolute error
    mae = float(np.abs(kan_flat - real_flat).mean())

    return {
        "spearman_rho": float(rho),
        "spearman_p": float(pval),
        "top3_hit_rate": float(topk_hit),
        "false_positive_rate": float(fp_rate),
        "mae": mae,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--n-states", type=int, default=30)
    parser.add_argument("--n-goals", type=int, default=20)
    parser.add_argument("--json-out", default="results/kan_reachability_calibration.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    shift = SHIFTS["friction_070"]

    # ── Load components ──────────────────────────────────────────────────
    print("Loading components ...", flush=True)
    source_policy = FrozenSourcePolicy(
        "results/hopper_source_sb3_ppo_continued_seed1811.zip",
        "results/hopper_source_sb3_vecnorm_continued_seed1811.pkl",
        device, args.seed, env="hopper",
    )
    source_twin = load_source_twin(
        "results/hopper_source_affine_twin_cloud_seed1811.pt", device,
    )
    basis, source_context, _, _ = load_cognition(
        argparse.Namespace(
            cognition_checkpoint="results/hopper_source_control_sobolev_calibrated_seed1811.pt",
            device=args.device,
        ),
        device,
    )

    # Fit target KAN context
    print("Fitting target KAN ...", flush=True)
    fit_args = argparse.Namespace(
        target="friction_070", seed=args.seed, env="hopper", device=args.device,
        cognition_warmup=1024, warmup_noise=0.3,
        transform_ridge=10.0, drift_ridge=100.0,
        drift_spectral_eta=0.0, drift_spectral_beta=1.0,
        drift_spectral_mode="max", drift_smooth_lambda=0.0, diagonal_transform=False,
    )
    target_ctx, _ = fit_distilled_source_counterfactual_context(
        source_policy, basis, source_context, fit_args, device, source_twin,
    )

    # ── Train executor (Exp 1 approach) ──────────────────────────────────
    from scripts.diagnose_oracle_goal_executor import GoalExecutor, collect_oracle_trajectories
    from torch.utils.data import DataLoader, TensorDataset

    print("Training executor ...", flush=True)
    oracle_traj = collect_oracle_trajectories(shift, n_episodes=10, seed=args.seed)
    all_s, all_g, all_a = [], [], []
    H = 10
    for st, ac in oracle_traj:
        for t in range(len(st) - H):
            all_s.append(st[t]); all_g.append(st[H + t]); all_a.append(ac[t])
    X_s = torch.tensor(np.stack(all_s), dtype=torch.float32)
    X_g = torch.tensor(np.stack(all_g), dtype=torch.float32)
    Y_a = torch.tensor(np.stack(all_a), dtype=torch.float32)
    executor = GoalExecutor(X_s.shape[1], X_s.shape[1], Y_a.shape[1]).to(device)
    opt = torch.optim.Adam(executor.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(X_s, X_g, Y_a), batch_size=256, shuffle=True)
    for _ in range(150):
        for sb, gb, ab in loader:
            sb, gb, ab = sb.to(device), gb.to(device), ab.to(device)
            loss = F.mse_loss(executor(sb, gb), ab)
            opt.zero_grad(); loss.backward(); opt.step()

    # ── Collect states and generate goals ────────────────────────────────
    states = collect_source_states(args.n_states, args.seed)
    goals = generate_diverse_goals(args.n_goals)
    print(f"States: {states.shape}, Goals: {goals.shape}")

    # ── Test different horizons ──────────────────────────────────────────
    results = {}
    for H_test in [2, 4, 8]:
        print(f"\n=== Horizon H={H_test} ===", flush=True)

        # KAN reachability
        print("  KAN prediction ...", flush=True, end=" ")
        kan_costs = kan_reachability_cost(
            states, goals, basis, target_ctx, device,
            horizon=H_test, n_candidates=64,
        )
        print("done", flush=True)

        # Real reachability
        print("  Real measurement ...", flush=True, end=" ")
        real_costs = real_reachability_cost(
            states, goals, executor, source_policy, shift,
            horizon=H_test, n_episodes_per=2, seed=args.seed,
        )
        print("done", flush=True)

        metrics = compute_metrics(kan_costs, real_costs)
        results[f"H={H_test}"] = metrics

        print(f"  Spearman ρ: {metrics['spearman_rho']:.4f} (p={metrics['spearman_p']:.4f})")
        print(f"  Top-3 hit:  {metrics['top3_hit_rate']:.1%}")
        print(f"  FP rate:    {metrics['false_positive_rate']:.1%}")
        print(f"  MAE:        {metrics['mae']:.4f}")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n=== Summary ===")
    for H_test in [2, 4, 8]:
        m = results[f"H={H_test}"]
        verdict = "GOOD" if m['spearman_rho'] > 0.3 and m['top3_hit_rate'] > 0.3 else "WEAK" if m['spearman_rho'] > 0 else "FAIL"
        print(f"  H={H_test}: ρ={m['spearman_rho']:.4f} top3={m['top3_hit_rate']:.1%} "
              f"FP={m['false_positive_rate']:.1%} [{verdict}]")

    json.dump(results, open(args.json_out, "w"), indent=2)
    print(f"\nSaved: {args.json_out}")


if __name__ == "__main__":
    main()
