"""Phase 0: KAN action Jacobian audit.

Can the KAN's local action gradient G(s) = dF/da guide differentiable MPC?

For each state, compare finite-difference true gradient vs KAN analytical
gradient along random action directions. Report:
  - Cosine similarity (direction match, >0.7 = good)
  - Sign agreement (correct sign of effect, >80% = good)
  - Relative magnitude error
  - Per-mode breakdown (stance vs flight)
"""

from __future__ import annotations

import argparse, sys, os, json
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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


def is_flight(state):
    """Heuristic: flight if foot not in contact (state[1] high, state[-2:] near 0)."""
    z = state[1]  # body height
    foot_contact = np.abs(state[-2:]).sum()  # foot contact forces ~0 in flight
    return z > 0.85 and foot_contact < 0.5


def audit_jacobian(source_policy, basis, source_context, target_ctx, device,
                   n_states=200, n_directions=20, eps=0.05):
    """Audit KAN action Jacobian against true finite-difference dynamics."""

    # Collect (qpos, qvel, action, mode) from friction env
    shift = SHIFTS["friction_070"]
    env = make_shifted_env(shift, 1811, "hopper")()
    obs, _ = env.reset(seed=1811)

    records = []
    for _ in range(n_states * 3):
        s_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        nominal = source_policy.action(s_t)
        s_eff = source_context.acceleration(basis, s_t, nominal)
        a = target_ctx.transport_action(
            basis, s_t, desired_effect=s_eff,
            nominal_action=nominal, regularization=1e-2,
        ).clamp(-1, 1).squeeze(0).cpu().numpy()
        try:
            qp = env.unwrapped.data.qpos.copy()
            qv = env.unwrapped.data.qvel.copy()
            records.append({"qpos": qp, "qvel": qv, "action": a, "obs": obs.copy(),
                           "flight": is_flight(obs)})
        except AttributeError:
            qp = env.unwrapped.sim.data.qpos.copy()
            qv = env.unwrapped.sim.data.qvel.copy()
            records.append({"qpos": qp, "qvel": qv, "action": a, "obs": obs.copy(),
                           "flight": is_flight(obs)})
        obs, _, t, tr, _ = env.step(a)
        if t or tr:
            obs, _ = env.reset()
    env.close()

    # Subsample
    idx = np.linspace(0, len(records) - 1, n_states, dtype=int)
    records = [records[i] for i in idx]

    # Temp env for FD
    fd_env = make_shifted_env(shift, 1811, "hopper")()

    a_dim = basis.action_dim
    s_dim = basis.state_dim
    results = {"cosines": [], "sign_agree": [], "mag_ratio": [],
               "stance_cosines": [], "flight_cosines": [],
               "stance_sign": [], "flight_sign": []}

    for i, rec in enumerate(records):
        s_np = rec["obs"]
        a_np = rec["action"]
        qp, qv = rec["qpos"], rec["qvel"]
        s_t = torch.as_tensor(s_np, device=device, dtype=torch.float32).unsqueeze(0)

        # KAN gain matrix G(s) at this state
        _, gain = target_ctx.drift_and_gain(basis, s_t)
        G = gain.squeeze(0)  # (s_dim, a_dim)

        for _ in range(n_directions):
            v = np.random.randn(a_dim)
            v = v / (np.linalg.norm(v) + 1e-10)

            a_plus = np.clip(a_np + eps * v, -1, 1)
            a_minus = np.clip(a_np - eps * v, -1, 1)

            # FD
            fd_env.unwrapped.set_state(qp, qv)
            obs_plus, _, _, _, _ = fd_env.step(a_plus)
            fd_env.unwrapped.set_state(qp, qv)
            obs_minus, _, _, _, _ = fd_env.step(a_minus)

            true_grad = (obs_plus - obs_minus) / (2.0 * eps)

            G_np = G.cpu().numpy()
            kan_grad = G_np @ v

            cos = np.dot(true_grad, kan_grad) / (
                np.linalg.norm(true_grad) * np.linalg.norm(kan_grad) + 1e-10
            )
            results["cosines"].append(float(cos))

            top_dims = np.argsort(np.abs(true_grad))[-5:]
            sign_match = np.mean(np.sign(true_grad[top_dims]) == np.sign(kan_grad[top_dims]))
            results["sign_agree"].append(float(sign_match))

            mag_r = np.linalg.norm(kan_grad) / (np.linalg.norm(true_grad) + 1e-10)
            results["mag_ratio"].append(float(mag_r))

            if rec["flight"]:
                results["flight_cosines"].append(float(cos))
                results["flight_sign"].append(float(sign_match))
            else:
                results["stance_cosines"].append(float(cos))
                results["stance_sign"].append(float(sign_match))

    fd_env.close()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--n-states", type=int, default=100)
    parser.add_argument("--n-directions", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Loading components...", flush=True)
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

    print("Fitting KAN to friction...", flush=True)
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

    print(f"Auditing Jacobian ({args.n_states} states x {args.n_directions} dirs)...", flush=True)
    results = audit_jacobian(
        source_policy, basis, source_context, target_ctx, device,
        n_states=args.n_states, n_directions=args.n_directions,
    )

    # Report
    n_stance = len(results["stance_cosines"])
    n_flight = len(results["flight_cosines"])
    print(f"\n=== KAN Action Jacobian Audit ===")
    print(f"  Samples: {args.n_states} states, {args.n_directions} directions each")
    print(f"  Stance: {n_stance}, Flight: {n_flight}")
    print()
    print(f"  {'Metric':25s} {'Overall':>10s} {'Stance':>10s} {'Flight':>10s}")
    for label, key_all, key_stance, key_flight in [
        ("Cosine similarity", "cosines", "stance_cosines", "flight_cosines"),
        ("Sign agreement", "sign_agree", "stance_sign", "flight_sign"),
        ("Magnitude ratio", "mag_ratio", None, None),
    ]:
        v_all = np.mean(results[key_all])
        v_s = np.mean(results[key_stance]) if key_stance in results else 0
        v_f = np.mean(results[key_flight]) if key_flight in results else 0
        print(f"  {label:25s} {v_all:10.4f} {v_s:10.4f} {v_f:10.4f}")

    # Verdict
    cos_mean = np.mean(results["cosines"])
    sign_mean = np.mean(results["sign_agree"])
    print(f"\n  Verdict: ", end="")
    if cos_mean > 0.5 and sign_mean > 0.7:
        print(f"COS={cos_mean:.3f} SIGN={sign_mean:.1%} -> PASS: KAN Jacobian direction usable")
    elif cos_mean > 0:
        print(f"COS={cos_mean:.3f} SIGN={sign_mean:.1%} -> WEAK: Jacobian has some signal but noisy")
    else:
        print(f"COS={cos_mean:.3f} SIGN={sign_mean:.1%} -> FAIL: KAN Jacobian unreliable")

    json.dump({k: float(np.mean(v)) for k, v in results.items()},
              open("results/phase0_jacobian_audit.json", "w"), indent=2)


if __name__ == "__main__":
    main()
