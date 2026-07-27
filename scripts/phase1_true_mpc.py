"""Phase 1: True-dynamics MPC upper bound.

Can optimizing actions with the TRUE simulator dynamics beat Transport?
If NO -> objective function is wrong (not KAN's fault).
If YES -> objective is viable, proceed to KAN-based MPC.
"""

from __future__ import annotations

import argparse, sys, os, json
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.prescreen_hopper_physics_shifts import SHIFTS, make_shifted_env
from scripts.validate_hopper_joint_online_adaptation import FrozenSourcePolicy


def compute_source_value(states, source_policy, device):
    s_t = torch.as_tensor(states, device=device, dtype=torch.float32)
    mean, var = source_policy.mean, source_policy.variance
    s_n = ((s_t - mean) / (var + 1e-8).sqrt()).clamp(-10, 10)
    with torch.no_grad():
        f = source_policy.model.policy.features_extractor(s_n)
        l = source_policy.model.policy.mlp_extractor(f)
        _, lv = l if isinstance(l, tuple) else (l, l)
        return source_policy.model.policy.value_net(lv).squeeze(-1).cpu().numpy()


def rollout_true(env, qp, qv, action_seq):
    """Roll out action_seq from (qp, qv) using true env. Returns (final_obs, total_reward)."""
    env.unwrapped.set_state(qp, qv)
    total_r = 0.0
    obs = None
    for a in action_seq:
        obs, r, t, tr, _ = env.step(np.clip(a, -1, 1))
        total_r += float(r)
        if t or tr:
            break
    return obs, total_r


def true_mpc_step(fd_env, qp, qv, a_transport, goal, source_policy, device,
                  H=4, n_cand=256, beta=0.1):
    """True-dynamics MPC: find best first action via random shooting.

    Objective: minimize ||s_H - goal||^2 (task-weighted) + beta * sum(|da|^2)
    + maximize V_source(s_H) for ranking.
    """
    a_dim = len(a_transport)
    task_mask = np.array([1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 0.2, 0.2, 0.2, 0.1, 0.1])

    candidates = []
    for _ in range(n_cand):
        da_seq = np.random.randn(H, a_dim) * 0.15  # small residuals
        s_obs = None
        dist_cost = 0.0
        action_cost = 0.0

        fd_env.unwrapped.set_state(qp, qv)
        alive = True
        for t in range(H):
            a_t = np.clip(a_transport + da_seq[t], -1, 1)
            s_obs, _, terminated, truncated, _ = fd_env.step(a_t)
            action_cost += float((da_seq[t] ** 2).sum())
            if terminated or truncated:
                alive = False
                break

        if alive and s_obs is not None:
            diff = (s_obs - goal) * task_mask
            dist_cost = float((diff ** 2).sum())

        total_cost = dist_cost + beta * action_cost
        v_terminal = float(compute_source_value(s_obs[None, :], source_policy, device)[0]) if alive and s_obs is not None else -float('inf')

        candidates.append({
            "first_a": np.clip(a_transport + da_seq[0], -1, 1),
            "cost": total_cost,
            "v": v_terminal,
            "alive": alive,
        })

    # Rank by V_source among alive, low-cost candidates
    alive_cands = [c for c in candidates if c["alive"]]
    if alive_cands:
        alive_cands.sort(key=lambda c: c["cost"])
        top_k = alive_cands[:10]  # top-10 by trajectory cost
        top_k.sort(key=lambda c: c["v"], reverse=True)  # then by V_source
        return top_k[0]["first_a"]
    else:
        return a_transport  # all died, return transport


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--H", type=int, default=4)
    parser.add_argument("--n-cand", type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device)
    shift = SHIFTS["friction_070"]
    H = args.H

    print(f"Phase 1: True-dynamics MPC (H={H}, n_cand={args.n_cand})")

    # Load source policy
    source_policy = FrozenSourcePolicy(
        "results/hopper_source_sb3_ppo_continued_seed1811.zip",
        "results/hopper_source_sb3_vecnorm_continued_seed1811.pkl",
        device, args.seed, env="hopper",
    )

    # Goal pool from source
    pool = []
    for ep in range(5):
        env = make_shifted_env(SHIFTS["source"], args.seed + ep * 100, "hopper")()
        obs, _ = env.reset(seed=args.seed + ep * 100); sts = []
        while True:
            a = source_policy.action(torch.tensor(obs, dtype=torch.float32)).cpu().numpy()
            sts.append(obs.copy()); obs, _, t, tr, _ = env.step(a)
            if t or tr: break
        env.close()
        if len(sts) > 50: pool.append(np.stack(sts))
    all_src = np.concatenate(pool, axis=0)

    def gen_goals(n=20):
        gs = []
        for _ in range(n // 3):
            i = np.random.randint(len(all_src)); g = all_src[i].copy()
            g[0] += np.random.uniform(0.3, 1.5); gs.append(g)
        for _ in range(n // 3):
            i = np.random.randint(len(all_src)); g = all_src[i].copy()
            g[0] += np.random.uniform(-0.1, 0.3); gs.append(g)
        for _ in range(n - len(gs)):
            i = np.random.randint(len(all_src)); g = all_src[i].copy()
            g += np.random.randn(*g.shape) * 0.3; gs.append(g)
        return np.stack(gs).astype(np.float32)

    def compute_v(states):
        return compute_source_value(states, source_policy, device)

    # Test Transport vs True-MPC
    # First collect Transport actions + qpos/qvel from friction env
    print("Collecting baseline Transport trajectory...", flush=True)
    fd_env = make_shifted_env(shift, args.seed, "hopper")()
    obs, _ = fd_env.reset(seed=args.seed)
    records = []
    for _ in range(200):
        s_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        nominal = source_policy.action(s_t)
        from cpbn.generic_affine_kan import AffineKANContext
        from scripts.diagnose_hopper_pullback_effect import fit_distilled_source_counterfactual_context, load_source_twin
        from scripts.validate_hopper_joint_online_adaptation import load_cognition
        # Need KAN for transport - load it
        # For now skip - use simple action perturbation test instead
        break  # placeholder

    # Actually, let me do the transport loading once
    import argparse as argparse_mod
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

    # Re-collect with transport
    obs, _ = fd_env.reset(seed=args.seed)
    records = []
    for step_idx in range(500):
        s_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        nominal = source_policy.action(s_t)
        s_eff = source_context.acceleration(basis, s_t, nominal)
        a_tr = target_ctx.transport_action(basis, s_t, desired_effect=s_eff, nominal_action=nominal, regularization=1e-2).clamp(-1,1).squeeze(0).cpu().numpy()
        qp = fd_env.unwrapped.data.qpos.copy()
        qv = fd_env.unwrapped.data.qvel.copy()
        records.append({"qp": qp, "qv": qv, "obs": obs.copy(), "a_tr": a_tr})
        obs, _, t, tr, _ = fd_env.step(a_tr)
        if t or tr: break

    print(f"  Collected {len(records)} states")

    # Compare: Transport vs True-MPC on a subset
    n_test = min(30, len(records))
    idx_test = np.linspace(0, len(records) - 1, n_test, dtype=int)

    transport_rewards = []
    mpc_rewards = []
    mpc_improvements = []

    for i in idx_test:
        rec = records[i]
        goals = gen_goals(20)
        values = compute_v(goals)
        dists = np.linalg.norm(goals - rec["obs"][None, :], axis=-1)
        safe = dists < 2.0
        if safe.sum() > 0:
            scores = values.copy(); scores[~safe] = -float('inf')
            goal = goals[np.argmax(scores)]
        else:
            goal = goals[np.argmax(values)]

        # Transport rollout
        fd_env.unwrapped.set_state(rec["qp"], rec["qv"])
        tr_obs, tr_r = rollout_true(fd_env, rec["qp"], rec["qv"],
                                     np.tile(rec["a_tr"], (H, 1)))
        transport_rewards.append(tr_r)

        # True-MPC
        mpc_a = true_mpc_step(fd_env, rec["qp"], rec["qv"], rec["a_tr"],
                              goal, source_policy, device, H=H, n_cand=args.n_cand)
        fd_env.unwrapped.set_state(rec["qp"], rec["qv"])
        mpc_obs, mpc_r = rollout_true(fd_env, rec["qp"], rec["qv"],
                                       np.tile(mpc_a, (H, 1)))
        mpc_rewards.append(mpc_r)
        mpc_improvements.append(mpc_r - tr_r)

    fd_env.close()

    tr_mean = np.mean(transport_rewards)
    mpc_mean = np.mean(mpc_rewards)
    imp_mean = np.mean(mpc_improvements)

    print(f"\n=== Phase 1 Results (H={H}, n_cand={args.n_cand}) ===")
    print(f"  Transport H-step reward:  {tr_mean:.3f}")
    print(f"  True-MPC H-step reward:   {mpc_mean:.3f}")
    print(f"  Mean improvement:         {imp_mean:+.3f}")
    print(f"  Fraction improved:        {np.mean(np.array(mpc_improvements) > 0):.1%}")

    if mpc_mean > tr_mean * 1.05:
        print(f"  Verdict: PASS — True-MPC beats Transport, objective is viable")
    elif mpc_mean > tr_mean:
        print(f"  Verdict: MARGINAL — slight improvement, objective may need tuning")
    else:
        print(f"  Verdict: FAIL — objective function does not align with task reward")


if __name__ == "__main__":
    main()
