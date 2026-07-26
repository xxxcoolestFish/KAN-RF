"""Diagnose KAN imagination bias: compare imagined vs real target states.

For payload (imagination hurt) and friction (imagination helped),
compare state distributions from:
  - Real target rollout (ground truth)
  - KAN imagination rollout (model-based)
  - Source rollout (reference)

Hypothesis: payload has larger model bias → imagination hurts;
friction has smaller model bias → imagination helps.
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


def collect_real_rollout(shift, source_policy, n_steps=1000, seed=1911):
    """Collect states from real target env using source policy + exploration."""
    env = make_shifted_env(shift, seed, "hopper")()
    obs, _ = env.reset(seed=seed)
    states = []
    rng = np.random.default_rng(seed)
    for _ in range(n_steps):
        s_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        a = source_policy.action(s_t).cpu().numpy()[0]
        # Small exploration
        a = np.clip(a + rng.uniform(-0.1, 0.1, size=a.shape), -1.0, 1.0)
        obs, _, terminated, truncated, _ = env.step(a)
        states.append(obs.copy())
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    return np.stack(states)


def collect_source_rollout(source_policy, n_steps=1000, seed=1911):
    """Collect states from source env."""
    return collect_real_rollout(SHIFTS["source"], source_policy, n_steps, seed)


def collect_imagination_rollout(
    init_states, z_value, source_policy, source_context,
    basis, pca, device, horizon=10,
):
    """KAN imagination rollout from init states with given z."""
    N = len(init_states)
    n_feature = basis.feature_dim
    n_action = basis.action_dim

    delta_W = pca.decode(z_value)
    delta_W_t = torch.tensor(delta_W, device=device, dtype=torch.float32)
    source_blocks = source_context.coefficients.clone().reshape(
        1 + n_action, n_feature, -1,
    )

    all_states = []
    current_s = init_states.clone()

    for t in range(horizon):
        nominal = source_policy.action(current_s)
        source_effect = source_context.acceleration(basis, current_s, nominal)

        next_states = []
        for i in range(N):
            dw = delta_W_t.reshape(n_feature, -1)
            drifted_blocks = source_blocks.clone()
            drifted_blocks[0] = source_blocks[0] + dw
            drifted_ctx = AffineKANContext(
                drifted_blocks.reshape_as(source_context.coefficients)
            )
            a_cf = drifted_ctx.transport_action(
                basis, current_s[i:i+1],
                desired_effect=source_effect[i:i+1],
                nominal_action=nominal[i:i+1],
                regularization=1e-2,
            ).clamp(-1.0, 1.0)

            effect = drifted_ctx.acceleration(basis, current_s[i:i+1], a_cf)
            next_states.append(current_s[i:i+1] + effect)

        current_s = torch.cat(next_states, dim=0)
        all_states.append(current_s.cpu().numpy())

    return np.concatenate(all_states, axis=0)


def compute_mmd(X, Y, sigma=1.0):
    """Compute Maximum Mean Discrepancy (Gaussian kernel) between two datasets."""
    # Use a subset for efficiency
    n = min(500, len(X), len(Y))
    idx_x = np.random.choice(len(X), n, replace=False)
    idx_y = np.random.choice(len(Y), n, replace=False)
    X_s = X[idx_x]
    Y_s = Y[idx_y]

    def kernel(A, B):
        dist2 = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
        return np.exp(-dist2 / (2.0 * sigma ** 2))

    K_xx = kernel(X_s, X_s).mean()
    K_yy = kernel(Y_s, Y_s).mean()
    K_xy = kernel(X_s, Y_s).mean()
    return float(K_xx + K_yy - 2 * K_xy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--json-out", default="results/imagination_bias.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

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
    pca_data = np.load("results/cppe_pca_model.npz")
    pca = CognitivePCA(
        "results/cppe_pca_model.npz", k=5,
        pc_ranges=PCARanges(mins=pca_data["pc_mins"], maxs=pca_data["pc_maxs"]),
    )

    # Oracle z for each shift
    z_oracle = {}
    for i, name in enumerate(pca_data["shift_names"]):
        z_oracle[str(name)] = pca_data["z_values"][i, :5].astype(np.float32)

    # ── Collect states ───────────────────────────────────────────────────
    print("Collecting source states ...", flush=True)
    source_states = collect_source_rollout(source_policy, args.n_steps, args.seed)

    results = {}
    for shift_name in ["payload_125", "friction_070", "combo_medium"]:
        print(f"\n=== {shift_name} ===", flush=True)

        # Real target states
        print("  Real target rollout ...", flush=True, end=" ")
        real_states = collect_real_rollout(
            SHIFTS[shift_name], source_policy, args.n_steps, args.seed,
        )
        print(f"{len(real_states)} states", flush=True)

        # KAN imagination states
        print("  Imagination rollout ...", flush=True, end=" ")
        n_init = 100
        idx = np.random.choice(len(source_states), n_init, replace=False)
        init_s = torch.tensor(source_states[idx], device=device, dtype=torch.float32)
        imag_states = collect_imagination_rollout(
            init_s, z_oracle[shift_name],
            source_policy, source_context, basis, pca, device,
            horizon=10,
        )
        print(f"{len(imag_states)} states", flush=True)

        # ── Compare distributions ─────────────────────────────────────
        # MMD: source vs real target
        mmd_src_real = compute_mmd(source_states, real_states)

        # MMD: imagination vs real target
        mmd_imag_real = compute_mmd(imag_states, real_states)

        # MMD: imagination vs source
        mmd_imag_src = compute_mmd(imag_states, source_states)

        # State statistics
        real_mean = real_states.mean(axis=0)
        imag_mean = imag_states.mean(axis=0)
        src_mean = source_states.mean(axis=0)

        mean_shift_real = float(np.linalg.norm(real_mean - src_mean))
        mean_shift_imag = float(np.linalg.norm(imag_mean - src_mean))
        mean_error = float(np.linalg.norm(imag_mean - real_mean))

        results[shift_name] = {
            "mmd_source_vs_real": mmd_src_real,
            "mmd_imagination_vs_real": mmd_imag_real,
            "mmd_imagination_vs_source": mmd_imag_src,
            "mean_shift_real": mean_shift_real,
            "mean_shift_imag": mean_shift_imag,
            "mean_error_imag_vs_real": mean_error,
        }

        print(f"  MMD(source, real):       {mmd_src_real:.4f}")
        print(f"  MMD(imagination, real):  {mmd_imag_real:.4f}")
        print(f"  MMD(imagination, source):{mmd_imag_src:.4f}")
        print(f"  Mean shift (real vs src):  {mean_shift_real:.4f}")
        print(f"  Mean shift (imag vs src):  {mean_shift_imag:.4f}")
        print(f"  Mean error (imag vs real): {mean_error:.4f}")

        # Interpretation
        if mmd_imag_real < mmd_src_real:
            print(f"  => Imagination CLOSER to real than source → should help")
        else:
            print(f"  => Imagination FARTHER from real than source → model bias issue")

    # ── Save ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    json.dump(results, open(args.json_out, "w"), indent=2)
    print(f"\nSaved: {args.json_out}")

    # Summary
    print("\n=== Summary ===")
    print(f"  {'Shift':16s} {'MMD(src,real)':>13s} {'MMD(imag,real)':>14s} {'Verdict'}")
    for name, data in results.items():
        better = data["mmd_imagination_vs_real"] < data["mmd_source_vs_real"]
        v = "IMAGINATION HELPS" if better else "MODEL BIAS ISSUE"
        print(f"  {name:16s} {data['mmd_source_vs_real']:13.4f} "
              f"{data['mmd_imagination_vs_real']:14.4f} {v}")


if __name__ == "__main__":
    main()
