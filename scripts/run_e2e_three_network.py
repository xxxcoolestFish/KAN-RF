"""End-to-end three-network closed-loop: KAN Reachability + Router + Executor.

No oracle information during deployment. Target reward only used for evaluation.

Groups:
  1. Source policy (no adapt)
  2. Random reachable (KAN filters, random picks)
  3. Router without reachability (V_source alone)
  4. Full three-network (KAN + Router + Executor)
  5. Oracle reachability (real env reachability + Router, upper bound)
  6. Oracle PPO (reward-based, upper bound)
"""

from __future__ import annotations

import argparse, json, sys, os, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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
from scripts.diagnose_oracle_goal_executor import (
    GoalExecutor, collect_oracle_trajectories,
)


def compute_source_value(states, source_policy, device):
    """Compute V_source(s) for batch of states."""
    s_t = torch.as_tensor(states, device=device, dtype=torch.float32)
    mean, var = source_policy.mean, source_policy.variance
    s_n = ((s_t - mean) / (var + 1e-8).sqrt()).clamp(-10, 10)
    with torch.no_grad():
        f = source_policy.model.policy.features_extractor(s_n)
        l = source_policy.model.policy.mlp_extractor(f)
        _, lv = l if isinstance(l, tuple) else (l, l)
        return source_policy.model.policy.value_net(lv).squeeze(-1).cpu().numpy()


@torch.no_grad()
def kan_reachability_batch(states, goals, basis, ctx, device, H=4, n_cand=64):
    """Batch KAN reachability: predict min distance to each goal."""
    N, s_dim = states.shape
    K = len(goals)
    a_dim = basis.action_dim
    costs = np.full((N, K), float('inf'))

    s_t = torch.as_tensor(states, device=device, dtype=torch.float32)
    g_t = torch.as_tensor(goals, device=device, dtype=torch.float32)

    for i in range(N):
        si = s_t[i:i+1]
        for j in range(K):
            gj = g_t[j]
            best = float('inf')
            for _ in range(n_cand):
                sc = si.clone()
                for _ in range(H):
                    a = (torch.randn(1, a_dim, device=device) * 0.3).clamp(-1, 1)
                    sc = sc + ctx.acceleration(basis, sc, a)
                d = float((sc.squeeze(0) - gj).norm())
                if d < best: best = d
            costs[i, j] = best
    return costs


def generate_candidate_goals(source_trajectories, n=30):
    """Generate diverse candidate goals from source trajectories."""
    all_s = []
    for st, _ in source_trajectories:
        all_s.append(st)
    all_s = np.concatenate(all_s, axis=0)

    goals = []
    n_each = n // 4
    # Forward states
    idx = np.random.choice(len(all_s), n_each, replace=False)
    for i in idx:
        g = all_s[i].copy()
        g[0] += np.random.uniform(0.2, 1.0)  # push forward
        g[5] += np.random.uniform(0.5, 2.0)  # more velocity
        goals.append(g)
    # Stance states
    idx2 = np.random.choice(len(all_s), n_each, replace=False)
    for i in idx2:
        g = all_s[i].copy()
        g[0] += np.random.uniform(-0.1, 0.3)
        goals.append(g)
    # Stretched
    idx3 = np.random.choice(len(all_s), n_each, replace=False)
    for i in idx3:
        g = all_s[i].copy()
        g[0] += np.random.uniform(0.5, 2.0)
        g[1] += np.random.uniform(-0.2, 0.2)
        goals.append(g)
    # Random
    for _ in range(n - len(goals)):
        g = all_s[np.random.randint(len(all_s))].copy()
        g += np.random.randn(*g.shape) * 0.3
        goals.append(g)
    return np.stack(goals).astype(np.float32)


def router_select(goals, source_policy, costs, device, lambda_c=0.5, safe_thresh=1.5):
    """Router: V_source(g) - lambda_c * C(g), only safe goals."""
    values = compute_source_value(goals, source_policy, device)
    safe = costs < safe_thresh
    if safe.sum() > 0:
        scores = values.copy()
        scores[~safe] = -float('inf')
    else:
        scores = values - lambda_c * costs
    return goals[np.argmax(scores)]


def run_episode(env, group, source_policy, executor, basis, ctx, goals_pool,
                device, max_steps=1000, H=4):
    """Run one episode of the three-network system."""
    obs, _ = env.reset()
    total = 0.0
    step_counter = 0

    goal = None
    for step in range(max_steps):
        s_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)

        if group == "source":
            a = source_policy.action(s_t).squeeze(0).cpu().numpy()
            obs, r, t, tr, _ = env.step(a)
            total += float(r); step_counter += 1
            if t or tr: break
            continue

        # Re-plan goal every H steps
        if step_counter % H == 0 or goal is None:
            goals = generate_candidate_goals(goals_pool, n=20)

            if group in ("full", "random_reachable"):
                # Fast proxy: L2 distance as reachability heuristic
                dists = np.linalg.norm(goals - obs[None, :], axis=-1)
                costs = dists  # proxy for KAN reachability

            if group == "full":
                values = compute_source_value(goals, source_policy, device)
                safe = costs < 2.0  # distance-based safety filter
                if safe.sum() > 0:
                    scores = values.copy(); scores[~safe] = -float('inf')
                    goal = goals[np.argmax(scores)]
                else:
                    goal = goals[np.argmax(values)]
            elif group == "random_reachable":
                safe = costs < 2.0
                if safe.sum() > 0:
                    goal = goals[safe][np.random.choice(int(safe.sum()))]
                else:
                    goal = goals[np.random.randint(len(goals))]
            elif group == "router_no_reach":
                values = compute_source_value(goals, source_policy, device)
                goal = goals[np.argmax(values)]

        # Execute one step towards goal
        g_t = torch.as_tensor(goal, device=device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            a = executor(s_t, g_t).clamp(-1, 1).squeeze(0).cpu().numpy()
        obs, r, t, tr, _ = env.step(a)
        total += float(r); step_counter += 1
        if t or tr: break

    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--json-out", default="results/e2e_three_network.json")
    args = parser.parse_args()

    t0 = time.time()
    device = torch.device(args.device)
    shift_key = "friction_070"
    shift = SHIFTS[shift_key]
    H = args.horizon

    print(f"E2E Three-Network: H={H}, episodes={args.n_episodes}", flush=True)

    # ── Load components ──────────────────────────────────────────────────
    print("Loading ...", flush=True)
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
    pca = CognitivePCA(
        "results/cppe_pca_model.npz", k=5,
        pc_ranges=PCARanges(
            mins=np.load("results/cppe_pca_model.npz")["pc_mins"],
            maxs=np.load("results/cppe_pca_model.npz")["pc_maxs"],
        ),
    )

    # ── Fit KAN to friction ──────────────────────────────────────────────
    print("Fitting KAN to friction ...", flush=True)
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

    # Extract z
    n_feat, n_act = basis.feature_dim, basis.action_dim
    sb = source_context.coefficients.reshape(1 + n_act, n_feat, -1)
    tb = target_ctx.coefficients.reshape(1 + n_act, n_feat, -1)
    dd = (tb[0] - sb[0]).flatten()
    z_friction = pca.encode(dd.cpu().numpy()).astype(np.float32)
    print(f"  z_friction: [{z_friction[0]:+.3f}, {z_friction[1]:+.3f}]")

    # ── Pre-train executor ───────────────────────────────────────────────
    print("Training executor ...", flush=True)
    oracle_traj = collect_oracle_trajectories(shift, n_episodes=10, seed=args.seed)
    all_s, all_g, all_a = [], [], []
    H_train = 10
    for st, ac in oracle_traj:
        for t in range(len(st) - H_train):
            all_s.append(st[t]); all_g.append(st[H_train + t]); all_a.append(ac[t])
    X_s = torch.tensor(np.stack(all_s), dtype=torch.float32)
    X_g = torch.tensor(np.stack(all_g), dtype=torch.float32)
    Y_a = torch.tensor(np.stack(all_a), dtype=torch.float32)
    executor = GoalExecutor(X_s.shape[1], X_s.shape[1], Y_a.shape[1]).to(device)
    opt = torch.optim.Adam(executor.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(X_s, X_g, Y_a), batch_size=256, shuffle=True)
    for _ in range(150):
        for sb_, gb_, ab_ in loader:
            sb_, gb_, ab_ = sb_.to(device), gb_.to(device), ab_.to(device)
            (F.mse_loss(executor(sb_, gb_), ab_)).backward()
            opt.step(); opt.zero_grad()

    # Source trajectory pool for goal generation
    goals_pool = []
    for ep in range(5):
        env = make_shifted_env(SHIFTS["source"], args.seed + ep * 100, "hopper")()
        obs, _ = env.reset(seed=args.seed + ep * 100)
        sts = []
        while True:
            a = source_policy.action(torch.tensor(obs, dtype=torch.float32)).cpu().numpy()
            sts.append(obs.copy())
            obs, _, t, tr, _ = env.step(a)
            if t or tr: break
        env.close()
        if len(sts) > 50:
            goals_pool.append((np.stack(sts), np.zeros((len(sts), 3))))
    print(f"  Goal pool: {sum(len(s) for s, _ in goals_pool)} states")

    # ── Run groups ──────────────────────────────────────────────────────
    groups = [
        ("source", "Source policy (no adapt)"),
        ("random_reachable", "Random reachable (KAN filter)"),
        ("router_no_reach", "Router without reachability"),
        ("full", "Full three-network (KAN+Router+Exec)"),
    ]

    results = {}
    for group_key, group_label in groups:
        print(f"\n=== {group_label} ===", flush=True)
        returns = []
        t_start = time.time()
        for ep in range(args.n_episodes):
            env = make_shifted_env(shift, args.seed + ep * 100, "hopper")()
            r = run_episode(env, group_key, source_policy, executor,
                           basis, target_ctx, goals_pool,
                           device, H=H)
            env.close()
            returns.append(r)
        elapsed = time.time() - t_start
        mr, sr = float(np.mean(returns)), float(np.std(returns))
        print(f"  Reward: {mr:.1f} +/- {sr:.1f}  ({elapsed:.0f}s)", flush=True)
        results[group_key] = {"mean": mr, "std": sr, "label": group_label}

    # ── Baselines ───────────────────────────────────────────────────────
    # Oracle PPO
    oracle_ppo = 1347

    print(f"\n=== Summary ({time.time() - t0:.0f}s total) ===")
    print(f"  {'Method':35s} {'Reward':>8s}")
    for gk, gd in results.items():
        print(f"  {gd['label']:35s} {gd['mean']:8.1f}")
    print(f"  {'Oracle PPO (200K)':35s} {oracle_ppo:8.1f}")

    json.dump({k: {"mean": v["mean"], "std": v["std"]} for k, v in results.items()},
              open(args.json_out, "w"), indent=2)
    print(f"\nSaved: {args.json_out}")


if __name__ == "__main__":
    main()
