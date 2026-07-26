"""Phase A: Clean friction_070 recovery curve.

x-axis: interaction budget (0 to 4096 target transitions)
y-axis: reward on target physics

Methods:
  - No adapt: source policy, no adaptation (lower bound)
  - CPPE: KAN + PCA z + pi(s,z), reward-free
  - Transport teacher: transport_action directly, reward-free
  - Planner teacher: KAN reward planner directly, reward-free
  - Oracle PPO: train PPO from scratch on target (upper bound)
"""

from __future__ import annotations

import argparse, json, sys, os
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cpbn.cognitive_pca import CognitivePCA, PCARanges
from cpbn.cppe_env import PhysicsConditionedEnv
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
from scripts.train_cppe_policy import kan_reward_planner


def evaluate_cppe(model_path, norm_path, shift, z_value, device, n_episodes=10):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    def make():
        base = make_shifted_env(shift, 1911, "hopper")()
        cond = PhysicsConditionedEnv(base, z_dim=len(z_value))
        cond.set_z(z_value)
        return cond
    venv = DummyVecEnv([make])
    venv = VecNormalize.load(norm_path, venv)
    venv.training = False; venv.norm_reward = False
    model = PPO.load(model_path, env=venv, device=device)
    returns = []
    for _ in range(n_episodes):
        obs = venv.reset(); total = 0.0
        while True:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, d, i = venv.step(a)
            total += float(r[0])
            if d[0]: break
        returns.append(total)
    venv.close()
    return float(np.mean(returns)), float(np.std(returns))


def evaluate_teacher(source_policy, basis, source_context, context, shift,
                     device, teacher_mode, pca, z_val, n_episodes=10):
    """Evaluate teacher controller on target env without policy."""
    returns = []
    for ep in range(n_episodes):
        env = make_shifted_env(shift, 1911 + ep * 100, "hopper")()
        obs, _ = env.reset(seed=1911 + ep * 100)
        total = 0.0
        while True:
            s_t = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            nominal = source_policy.action(s_t)
            source_effect = source_context.acceleration(basis, s_t, nominal)

            if teacher_mode == "planner":
                a = kan_reward_planner(
                    s_t, z_val[None, :], source_policy, source_context,
                    basis, pca, device, horizon=5, n_candidates=32,
                ).squeeze(0).cpu().numpy()
            else:
                a = context.transport_action(
                    basis, s_t, desired_effect=source_effect,
                    nominal_action=nominal, regularization=1e-2,
                ).clamp(-1, 1).squeeze(0).cpu().numpy()

            obs, r, t, tr, _ = env.step(a)
            total += float(r)
            if t or tr: break
        env.close()
        returns.append(total)
    return float(np.mean(returns)), float(np.std(returns))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budgets", default="0,256,512,1024,2048")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--model-tag", default="cppe_v2_full_z2d_cf0.5_a1.0_blend0.3")
    parser.add_argument("--json-out", default="results/friction_recovery_curve.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    budgets = [int(x) for x in args.budgets.split(",")]

    model_path = f"results/cppe_training/{args.model_tag}_seed{args.seed}.zip"
    norm_path = f"results/cppe_training/{args.model_tag}_norm_seed{args.seed}.pkl"

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
    basis, source_context, _, delta_scale = load_cognition(
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
    z_source = pca.z_source.astype(np.float32)
    shift = SHIFTS["friction_070"]

    # ── Oracle PPO ───────────────────────────────────────────────────────
    oracle = 1347  # from train_oracle_target_ppo.py at 200K steps

    # ── Recovery curve ───────────────────────────────────────────────────
    curve = {"no_adapt": None, "oracle": oracle, "budgets": []}
    fit_args = argparse.Namespace(
        target="friction_070", seed=args.seed, env="hopper",
        device=args.device,
        cognition_warmup=0, warmup_noise=0.3,
        transform_ridge=10.0, drift_ridge=100.0,
        drift_spectral_eta=0.0, drift_spectral_beta=1.0,
        drift_spectral_mode="max", drift_smooth_lambda=0.0,
        diagonal_transform=False,
    )

    for budget in budgets:
        print(f"\n=== Budget {budget} ===", flush=True)

        if budget == 0:
            z_val = z_source
            context = source_context
        else:
            fit_args.cognition_warmup = budget
            context, _ = fit_distilled_source_counterfactual_context(
                source_policy, basis, source_context, fit_args, device, source_twin,
            )
            n_feature = basis.feature_dim; n_action = basis.action_dim
            src_b = source_context.coefficients.reshape(1+n_action, n_feature, -1)
            tgt_b = context.coefficients.reshape(1+n_action, n_feature, -1)
            drift_delta = (tgt_b[0] - src_b[0]).flatten()
            z_val = pca.encode(drift_delta.cpu().numpy()).astype(np.float32)

        # CPPE policy
        r_cppe, s_cppe = evaluate_cppe(model_path, norm_path, shift, z_val, device, args.n_episodes)

        # Transport teacher
        r_transport, s_transport = evaluate_teacher(
            source_policy, basis, source_context, context,
            shift, device, "transport", pca, z_val, args.n_episodes,
        )

        # Planner teacher (only at budget>0, expensive)
        if budget > 0:
            r_planner, s_planner = evaluate_teacher(
                source_policy, basis, source_context, context,
                shift, device, "planner", pca, z_val, args.n_episodes,
            )
        else:
            r_planner, s_planner = r_transport, s_transport

        # No-adapt (once)
        if curve["no_adapt"] is None:
            r_no, s_no = evaluate_cppe(model_path, norm_path, SHIFTS["source"],
                                       z_source, device, args.n_episodes)
            # Actually evaluate no-adapt by running source policy on friction
            r_no = 0; n_ep = 10
            for ep in range(n_ep):
                env = make_shifted_env(shift, 1911+ep*100, "hopper")()
                obs, _ = env.reset(seed=1911+ep*100); total=0.0
                while True:
                    a = source_policy.action(torch.tensor(obs, device=device, dtype=torch.float32)).cpu().numpy()
                    obs, r, t, tr, _ = env.step(a)
                    total += float(r)
                    if t or tr: break
                env.close(); r_no += total
            curve["no_adapt"] = r_no / n_ep

        curve["budgets"].append({
            "budget": budget,
            "cppe": {"mean": r_cppe, "std": s_cppe},
            "transport_teacher": {"mean": r_transport, "std": s_transport},
            "planner_teacher": {"mean": r_planner, "std": s_planner},
        })

        rec = (r_cppe - curve["no_adapt"]) / max(oracle - curve["no_adapt"], 1)
        print(f"  CPPE: {r_cppe:.1f} +/- {s_cppe:.1f}  (recovery: {rec:.1%})")
        print(f"  Transport teacher: {r_transport:.1f}")
        print(f"  Planner teacher:   {r_planner:.1f}")

    # ── Summary ──────────────────────────────────────────────────────────
    r_no = curve["no_adapt"]
    print(f"\n=== Friction Recovery Curve ===")
    print(f"  No adapt: {r_no:.0f}  |  Oracle PPO: {oracle}")
    print(f"  {'Budget':>6s}  {'CPPE':>8s}  {'Transport':>10s}  {'Planner':>10s}  {'Recovery':>8s}")
    for b in curve["budgets"]:
        rec = (b["cppe"]["mean"] - r_no) / max(oracle - r_no, 1)
        print(f"  {b['budget']:6d}  {b['cppe']['mean']:8.1f}  "
              f"{b['transport_teacher']['mean']:10.1f}  "
              f"{b['planner_teacher']['mean']:10.1f}  "
              f"{rec:7.1%}")

    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    json.dump(curve, open(args.json_out, "w"), indent=2)
    print(f"\nSaved: {args.json_out}")


if __name__ == "__main__":
    main()
