"""Diagnostic: does transport_action work in closed-loop target deployment?

This is the critical fork-point test.
  - If transport_action rollout is GOOD → policy imitation is the problem
  - If transport_action rollout is BAD → counterfactual controller needs fixing

Evaluates transport_action as a controller directly in target env:
  a_cf(s) = transport_action(s, desired=source_effect, nominal=pi_source(s))
"""

from __future__ import annotations

import argparse, json, sys, os
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cpbn.cognitive_pca import CognitivePCA, PCARanges
from cpbn.generic_affine_kan import AffineKANContext
from scripts.diagnose_hopper_pullback_effect import (
    fit_distilled_source_counterfactual_context,
    load_source_twin,
)
from scripts.prescreen_hopper_physics_shifts import ENVS, SHIFTS, make_shifted_env
from scripts.validate_hopper_joint_online_adaptation import (
    FrozenSourcePolicy,
    load_cognition,
)


def rollout_transport_action(
    source_policy,
    basis,
    source_context,
    context,  # fitted target KAN context
    shift,
    device,
    n_episodes=10,
    seed=1911,
    regularization=1e-2,
    action_clip=True,
):
    """Run transport_action in closed loop on target environment.

    At each step:
      1. Compute source effect: e_source = source_accel(s, pi_source(s))
      2. Compute a_cf = transport_action(s, desired=e_source, nominal=pi_source(s))
      3. Step target env with a_cf

    Returns list of episode returns.
    """
    returns = []
    for ep in range(n_episodes):
        env = make_shifted_env(shift, seed + ep * 100, "hopper")()
        obs, _ = env.reset(seed=seed + ep * 100)
        total = 0.0
        step = 0
        while True:
            s_t = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            nominal = source_policy.action(s_t)
            source_effect = source_context.acceleration(basis, s_t, nominal)

            a_cf = context.transport_action(
                basis, s_t,
                desired_effect=source_effect,
                nominal_action=nominal,
                regularization=regularization,
            )

            if action_clip:
                a_cf = a_cf.clamp(-1.0, 1.0)

            obs, reward, terminated, truncated, info = env.step(
                a_cf.squeeze(0).cpu().numpy()
            )
            total += float(reward)
            step += 1

            if terminated or truncated:
                break

        env.close()
        returns.append(total)
    return returns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--json-out", default="results/transport_action_closed_loop.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

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
    basis, source_context, _, delta_scale = load_cognition(
        argparse.Namespace(
            cognition_checkpoint="results/hopper_source_control_sobolev_calibrated_seed1811.pt",
            device=args.device,
        ),
        device,
    )
    pca_data = np.load("results/cppe_pca_model.npz")

    # ── Source baseline ──────────────────────────────────────────────────
    print("\n=== Source baseline (pi_source in source env) ===")
    source_returns = []
    for ep in range(args.n_episodes):
        env = make_shifted_env(SHIFTS["source"], args.seed + ep * 100, "hopper")()
        obs, _ = env.reset(seed=args.seed + ep * 100)
        total = 0.0
        while True:
            a = source_policy.action(torch.tensor(obs, device=device, dtype=torch.float32))
            obs, r, t, tr, _ = env.step(a.cpu().numpy())
            total += float(r)
            if t or tr: break
        env.close()
        source_returns.append(total)
    r_source = float(np.mean(source_returns))
    print(f"  Source: {r_source:.1f} +/- {float(np.std(source_returns)):.1f}")

    # ── Target shifts ────────────────────────────────────────────────────
    target_shifts = {
        "payload_125": SHIFTS["payload_125"],
        "friction_070": SHIFTS["friction_070"],
        "combo_medium": SHIFTS["combo_medium"],
    }

    fit_args = argparse.Namespace(
        target="source", seed=args.seed, env="hopper",
        device=args.device,
        cognition_warmup=args.budget,
        warmup_noise=0.3,
        transform_ridge=10.0,
        drift_ridge=100.0,
        drift_spectral_eta=0.0,
        drift_spectral_beta=1.0,
        drift_spectral_mode="max",
        drift_smooth_lambda=0.0,
        diagonal_transform=False,
    )

    results = {"source": r_source, "shifts": {}}

    for shift_name, shift in target_shifts.items():
        print(f"\n=== {shift_name} ===", flush=True)

        # --- No-adaptation: pi_source directly in target env ---
        drop_returns = []
        for ep in range(args.n_episodes):
            env = make_shifted_env(shift, args.seed + ep * 100, "hopper")()
            obs, _ = env.reset(seed=args.seed + ep * 100)
            total = 0.0
            while True:
                a = source_policy.action(torch.tensor(obs, device=device, dtype=torch.float32))
                obs, r, t, tr, _ = env.step(a.cpu().numpy())
                total += float(r)
                if t or tr: break
            env.close()
            drop_returns.append(total)
        r_drop = float(np.mean(drop_returns))
        print(f"  No-adapt: {r_drop:.1f} +/- {float(np.std(drop_returns)):.1f}")

        # --- Fit target KAN (same pipeline used for z extraction) ---
        print(f"  Fitting target KAN (budget={args.budget}) ...", flush=True, end=" ")
        fit_args.target = shift_name
        target_context, _ = fit_distilled_source_counterfactual_context(
            source_policy, basis, source_context, fit_args, device, source_twin,
        )

        # --- Transport_action closed-loop ---
        print("rollout ...", flush=True, end=" ")
        cf_returns = rollout_transport_action(
            source_policy, basis, source_context, target_context,
            shift, device,
            n_episodes=args.n_episodes, seed=args.seed,
        )
        r_cf = float(np.mean(cf_returns))
        s_cf = float(np.std(cf_returns))
        print(f"done", flush=True)

        # --- Compare with CPPE policy ---
        # Load CPPE model and evaluate with oracle z
        pca = CognitivePCA(
            "results/cppe_pca_model.npz", k=5,
            pc_ranges=PCARanges(
                mins=pca_data["pc_mins"], maxs=pca_data["pc_maxs"],
            ),
        )
        n_feature = basis.feature_dim
        n_action = basis.action_dim
        source_blocks = source_context.coefficients.reshape(1 + n_action, n_feature, -1)
        target_blocks = target_context.coefficients.reshape(1 + n_action, n_feature, -1)
        drift_delta = (target_blocks[0] - source_blocks[0]).flatten()
        z_oracle = pca.encode(drift_delta.cpu().numpy()).astype(np.float32)

        # Evaluate CPPE policy with this z
        from cpbn.cppe_env import PhysicsConditionedEnv
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        model_tag = "cppe_v2_full_z2d_cf0.1_a1.0_blend0.2"
        model_path = f"results/cppe_training/{model_tag}_seed{args.seed}.zip"
        norm_path = f"results/cppe_training/{model_tag}_norm_seed{args.seed}.pkl"

        def make():
            base = make_shifted_env(shift, 1911, "hopper")()
            cond = PhysicsConditionedEnv(base, z_dim=pca.k)
            cond.set_z(z_oracle)
            return cond

        venv = DummyVecEnv([make])
        venv = VecNormalize.load(norm_path, venv)
        venv.training = False; venv.norm_reward = False
        model = PPO.load(model_path, env=venv, device=device)
        cppe_returns = []
        for _ in range(args.n_episodes):
            obs = venv.reset(); total = 0.0
            while True:
                a, _ = model.predict(obs, deterministic=True)
                obs, r, d, i = venv.step(a)
                total += float(r[0])
                if d[0]: break
            cppe_returns.append(total)
        venv.close()
        r_cppe = float(np.mean(cppe_returns))

        print(f"  Transport_action (teacher): {r_cf:.1f} +/- {s_cf:.1f}")
        print(f"  CPPE policy (student):      {r_cppe:.1f}")
        print(f"  Gap (teacher - student):    {r_cf - r_cppe:.1f}")

        if r_cf > r_drop and r_cf > r_cppe:
            print(f"  Verdict: Teacher WORKS → policy imitation is the bottleneck")
        elif r_cf <= r_drop:
            print(f"  Verdict: Teacher FAILS → counterfactual controller needs fixing")
        else:
            print(f"  Verdict: Mixed — teacher better than no-adapt but CPPE student does better")

        results["shifts"][shift_name] = {
            "r_source": r_source,
            "r_drop": r_drop,
            "r_transport_action": r_cf,
            "r_cppe": r_cppe,
        }

    # ── Save ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    json.dump(results, open(args.json_out, "w"), indent=2)
    print(f"\nSaved: {args.json_out}")

    # Summary
    print("\n=== Fork-point Summary ===")
    print(f"  {'Shift':16s} {'No-adapt':>8s} {'Transport':>10s} {'CPPE':>8s} {'Verdict'}")
    for name, data in results["shifts"].items():
        if data["r_transport_action"] > data["r_drop"] and \
           data["r_transport_action"] > data["r_cppe"]:
            verdict = "TEACHER OK → fix imitation"
        elif data["r_transport_action"] <= data["r_drop"]:
            verdict = "TEACHER FAILS → fix controller"
        else:
            verdict = "MIXED"
        print(f"  {name:16s} {data['r_drop']:8.1f} {data['r_transport_action']:10.1f} "
              f"{data['r_cppe']:8.1f} {verdict}")


if __name__ == "__main__":
    main()
