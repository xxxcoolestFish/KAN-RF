"""CPPE recovery curve experiment v2.

Reuses the existing cognitive fitting pipeline to estimate z at each budget,
then evaluates the CPPE policy pi(s,z) using the fitted z.

Protocol:
  1. For each budget in {0, 256, 512, 1024, 2048}:
     a. Fit W_t from target transitions (existing pipeline)
     b. Extract z = PCA(W_t - W_source)
     c. Evaluate CPPE policy pi(s, z) on target environment
  2. Report recovery: (R_after - R_drop) / (R_source - R_drop)

This separates z estimation (proven) from z-conditioned policy (new).
"""

from __future__ import annotations

import argparse, json, sys, os
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cpbn.cognitive_pca import CognitivePCA, PCARanges
from cpbn.cppe_env import PhysicsConditionedEnv
from scripts.diagnose_hopper_pullback_effect import (
    fit_distilled_source_counterfactual_context,
    load_source_twin,
)
from scripts.prescreen_hopper_physics_shifts import ENVS, SHIFTS, make_shifted_env
from scripts.validate_hopper_joint_online_adaptation import (
    FrozenSourcePolicy,
    load_cognition,
)


def extract_z_from_context(context, basis, source_context, pca):
    """Extract drift delta from fitted context and encode to z."""
    n_action = basis.action_dim
    n_feature = basis.feature_dim
    source_blocks = source_context.coefficients.reshape(1 + n_action, n_feature, -1)
    target_blocks = context.coefficients.reshape(1 + n_action, n_feature, -1)
    drift_delta = (target_blocks[0] - source_blocks[0]).flatten()
    return pca.encode(drift_delta.cpu().numpy()).astype(np.float32)


def evaluate_cppe(model_path, norm_path, shift, z_value, device, n_episodes=10):
    """Evaluate CPPE policy on a physics shift with given z."""
    def make():
        base = make_shifted_env(shift, 1911, "hopper")()
        cond = PhysicsConditionedEnv(base, z_dim=len(z_value))
        cond.set_z(z_value)
        return cond
    venv = DummyVecEnv([make])
    venv = VecNormalize.load(norm_path, venv)
    venv.training = False
    venv.norm_reward = False
    model = PPO.load(model_path, env=venv, device=device)
    returns = []
    for _ in range(n_episodes):
        obs = venv.reset()
        total = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = venv.step(action)
            total += float(reward[0])
            if dones[0]:
                break
        returns.append(total)
    venv.close()
    return float(np.mean(returns)), float(np.std(returns))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tag", default="cppe_v2_full_z2d_cf0.1_a1.0_blend0.2")
    parser.add_argument("--budgets", default="0,256,512,1024,2048")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--pca-model", default="results/cppe_pca_model.npz")
    parser.add_argument("--json-out", default="results/cppe_recovery_curve.json")
    parser.add_argument("--eval-episodes", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    model_path = f"results/cppe_training/{args.model_tag}_seed{args.seed}.zip"
    norm_path = f"results/cppe_training/{args.model_tag}_norm_seed{args.seed}.pkl"
    print(f"Model: {model_path}")

    budgets = [int(x) for x in args.budgets.split(",")]

    # ── Load PCA ────────────────────────────────────────────────────────
    pca_data = np.load(args.pca_model)
    pc_ranges = PCARanges(mins=pca_data["pc_mins"], maxs=pca_data["pc_maxs"])
    pca = CognitivePCA(args.pca_model, k=5, pc_ranges=pc_ranges)
    z_source = pca.z_source.astype(np.float32)
    print(f"PCA: k={pca.k}")

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

    # ── Source reference ─────────────────────────────────────────────────
    print("\n=== Source reference ===")
    r_source, s_source = evaluate_cppe(
        model_path, norm_path, SHIFTS["source"], z_source, device,
        n_episodes=args.eval_episodes,
    )
    print(f"  Source CPPE: {r_source:.1f} +/- {s_source:.1f}")

    # ── Target shifts ────────────────────────────────────────────────────
    target_shifts = {
        "payload_125": SHIFTS["payload_125"],
        "friction_070": SHIFTS["friction_070"],
        "combo_medium": SHIFTS["combo_medium"],
    }

    results = {"source": r_source, "shifts": {}}

    for shift_name, shift in target_shifts.items():
        print(f"\n=== {shift_name} ===", flush=True)

        # --- R_drop: no adaptation (z = z_source) ---
        r_drop, s_drop = evaluate_cppe(
            model_path, norm_path, shift, z_source, device,
            n_episodes=args.eval_episodes,
        )
        print(f"  R_drop (z=z_source): {r_drop:.1f} +/- {s_drop:.1f}")

        # --- Recovery at each budget ---
        curve = []
        prev_context = None

        for budget in budgets:
            if budget == 0:
                z_val = z_source
                r_val, s_val = r_drop, s_drop
            else:
                # Fit W_t using existing pipeline
                print(f"  Fitting budget {budget} ...", flush=True, end=" ")
                fit_args = argparse.Namespace(
                    target=shift_name, seed=args.seed, env="hopper",
                    device=args.device,
                    cognition_warmup=budget,
                    warmup_noise=0.3,
                    transform_ridge=10.0,
                    drift_ridge=100.0,
                    drift_spectral_eta=0.0,
                    drift_spectral_beta=1.0,
                    drift_spectral_mode="max",
                    drift_smooth_lambda=0.0,
                    diagonal_transform=False,
                )
                context, _ = fit_distilled_source_counterfactual_context(
                    source_policy, basis, source_context, fit_args, device,
                    source_twin,
                )
                z_val = extract_z_from_context(context, basis, source_context, pca)

                # Evaluate CPPE with fitted z
                r_val, s_val = evaluate_cppe(
                    model_path, norm_path, shift, z_val, device,
                    n_episodes=args.eval_episodes,
                )
                print(f"return={r_val:.1f} z=[{z_val[0]:+.3f}, {z_val[1]:+.3f}]", flush=True)

            curve.append({
                "budget": budget, "mean": r_val, "std": s_val,
                "z": z_val.tolist(),
            })

        # Recovery metric
        r_final = curve[-1]["mean"]
        recovery = (r_final - r_drop) / max(r_source - r_drop, 1.0)

        print(f"  Recovery: {recovery:.1%}  (R_source={r_source:.0f}, R_drop={r_drop:.0f}, R_final={r_final:.0f})")
        results["shifts"][shift_name] = {
            "r_drop": r_drop, "r_final": r_final, "recovery": recovery, "curve": curve,
        }

    # ── Save ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    json.dump(results, open(args.json_out, "w"), indent=2,
              default=lambda x: float(x) if isinstance(x, (np.floating,)) else str(x))
    print(f"\nSaved: {args.json_out}")

    # Summary
    print("\n=== Recovery Summary ===")
    print(f"  R_source = {r_source:.0f}")
    print(f"  {'Shift':16s} {'R_drop':>8s} {'R_final':>8s} {'Recovery':>8s}")
    for name, data in results["shifts"].items():
        print(f"  {name:16s} {data['r_drop']:8.1f} {data['r_final']:8.1f} "
              f"{data['recovery']:7.1%}")


if __name__ == "__main__":
    main()
