"""Train a residual policy adapter: pi(s,z) = pi_source(s) + delta_pi(s,z).

The source policy is FROZEN. Only the residual adapter is trained.
This preserves source performance and forces the adapter to learn
meaningful z-dependence for target physics compensation.

Training uses supervised learning on physics buffer data:
  target = a_teacher(s, z) - a_source(s)
  loss = MSE(delta_pi(s, z), target)
"""

from __future__ import annotations

import argparse, json, sys, os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cpbn.cognitive_pca import CognitivePCA, PCARanges
from cpbn.cppe_env import PhysicsConditionedEnv
from cpbn.generic_affine_kan import AffineKANContext
from scripts.prescreen_hopper_physics_shifts import ENVS, SHIFTS, make_shifted_env
from scripts.validate_hopper_joint_online_adaptation import (
    FrozenSourcePolicy,
    load_cognition,
)
from scripts.train_cppe_policy import (
    compute_counterfactual_action,
    kan_reward_planner,
    kan_imagination_rollout,
)


class ResidualAdapter(nn.Module):
    """Small MLP that predicts action residuals from (s, z).

    Architecture: [s|z] → [128] → [128] → Δa
    Initialized to predict near-zero to preserve source behavior.
    """

    def __init__(self, s_dim: int, z_dim: int, a_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, a_dim),
        )
        # Initialize last layer to near-zero
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, z], dim=-1)
        return self.net(x)


def generate_physics_buffer(
    source_policy, basis, source_context, pca, z_source,
    teacher_mode, blend_ratio, device, n_states, n_z_per_state=4,
    use_imagination=True,
):
    """Generate physics buffer: (s, z) → a_teacher - a_source.

    Collects states from source env via source policy, then for each state
    samples multiple z values and computes teacher residuals.
    Optionally uses KAN imagination for target-like states.
    """
    all_s, all_z, all_residual = [], [], []

    # Collect source states
    env = make_shifted_env(SHIFTS["source"], 1811, "hopper")()
    obs, _ = env.reset(seed=1811)
    source_states = []
    for _ in range(n_states + 50):
        a = source_policy.action(torch.tensor(obs, device=device, dtype=torch.float32)).cpu().numpy()
        obs, _, terminated, truncated, _ = env.step(a)
        source_states.append(obs.copy())
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    source_states = source_states[-n_states:]
    s_tensor = torch.tensor(np.stack(source_states), device=device, dtype=torch.float32)

    # Generate teacher residuals for each state + sampled z
    for i in range(n_states):
        si = s_tensor[i:i+1]
        a_src = source_policy.action(si)

        # Sample z values
        z_samples = pca.sample_z(n_z_per_state).astype(np.float32)

        # Repeat state to match z batch size
        si_batch = si.repeat(n_z_per_state, 1)  # (n_z, s_dim)

        # Teacher actions
        if teacher_mode == "planner":
            a_teacher = kan_reward_planner(
                si_batch, z_samples, source_policy, source_context,
                basis, pca, device, horizon=5, n_candidates=32,
            )
        else:
            a_teacher = compute_counterfactual_action(
                si_batch, z_samples, source_policy, source_context,
                basis, pca, device,
            )
            # Soft blend
            a_teacher = (1.0 - blend_ratio) * a_src + blend_ratio * a_teacher

        residual = a_teacher - a_src  # (n_z, a_dim)

        for j in range(n_z_per_state):
            all_s.append(si.squeeze(0).cpu().numpy())
            all_z.append(z_samples[j])
            all_residual.append(residual[j].cpu().numpy())

    # Optional: KAN imagination states
    if use_imagination:
        n_im = min(16, n_states)
        idx = np.random.choice(n_states, n_im, replace=False)
        init_s = s_tensor[idx]
        z_im = pca.sample_z(n_im).astype(np.float32)
        im_s, im_z, _ = kan_imagination_rollout(
            init_s, z_im, source_policy, source_context,
            basis, pca, device, horizon=5,
        )
        for j in range(len(im_s)):
            si = torch.tensor(im_s[j], device=device, dtype=torch.float32).unsqueeze(0)
            a_src = source_policy.action(si)
            zi = im_z[j:j+1]
            si_b = si.repeat(len(zi), 1)
            a_teacher = compute_counterfactual_action(
                si_b, zi, source_policy, source_context, basis, pca, device,
            )
            a_teacher = (1.0 - blend_ratio) * a_src + blend_ratio * a_teacher
            residual = a_teacher - a_src
            all_s.append(im_s[j])
            all_z.append(im_z[j])
            all_residual.append(residual.squeeze(0).cpu().numpy())

    # Source anchor: compute baseline residual at z_source, center all targets
    # This makes z_source → ~0 correction, z_target → relative correction
    z_src_arr = np.tile(z_source[None, :], (min(100, n_states), 1)).astype(np.float32)
    s_sample = s_tensor[:min(100, n_states)]
    a_src_sample = source_policy.action(s_sample)
    a_teacher_src = compute_counterfactual_action(
        s_sample, z_src_arr, source_policy, source_context,
        basis, pca, device,
    )
    baseline_residual = (a_teacher_src - a_src_sample).mean(dim=0)  # (a_dim,)
    baseline_np = baseline_residual.cpu().numpy()
    print(f"  Baseline residual (at z_source): {baseline_np}")

    # Center all residuals
    all_residual_centered = [r - baseline_np for r in all_residual]

    # Add source anchor samples
    for i in range(min(200, n_states)):
        all_s.append(source_states[i])
        all_z.append(z_src_arr[i % len(z_src_arr)])
        all_residual_centered.append(np.zeros(source_policy.action(s_tensor[[0]]).shape[1], dtype=np.float32))

    return (
        np.stack(all_s).astype(np.float32),
        np.stack(all_z).astype(np.float32),
        np.stack(all_residual_centered).astype(np.float32),
        baseline_np,
    )


def evaluate_adapter(adapter, source_policy, shift, z_value, baseline, device, n_episodes=10):
    """Evaluate residual policy: a = a_source + adapter(s, z)."""
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from cpbn.cppe_env import PhysicsConditionedEnv
    from scripts.prescreen_hopper_physics_shifts import make_shifted_env, SHIFTS

    def make():
        base = make_shifted_env(shift, 1911, "hopper")()
        return base  # Raw env for direct control

    returns = []
    for ep in range(n_episodes):
        env = make_shifted_env(shift, 1911 + ep * 100, "hopper")()
        obs, _ = env.reset(seed=1911 + ep * 100)
        total = 0.0
        while True:
            s_t = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            z_t = torch.tensor(z_value, device=device, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                a_src = source_policy.action(s_t)
                base = torch.tensor(baseline, device=device, dtype=torch.float32)
                delta = adapter(s_t, z_t)
                a = (a_src + base + delta).clamp(-1.0, 1.0)
            obs, r, t, tr, _ = env.step(a.squeeze(0).cpu().numpy())
            total += float(r)
            if t or tr: break
        env.close()
        returns.append(total)
    return float(np.mean(returns)), float(np.std(returns))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", choices=("transport", "planner"), default="transport")
    parser.add_argument("--n-states", type=int, default=500)
    parser.add_argument("--n-z-per-state", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--blend", type=float, default=0.3)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--model-out", default="results/residual_adapter.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    print(f"Residual Adapter Training: teacher={args.teacher}, blend={args.blend}")

    # ── Load components ──────────────────────────────────────────────────
    print("Loading ...", flush=True)
    source_policy = FrozenSourcePolicy(
        "results/hopper_source_sb3_ppo_continued_seed1811.zip",
        "results/hopper_source_sb3_vecnorm_continued_seed1811.pkl",
        device, args.seed, env="hopper",
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
    z_source = pca.z_source.astype(np.float32)
    s_dim = basis.state_dim
    a_dim = basis.action_dim
    z_dim = pca.k

    # ── Generate training data ──────────────────────────────────────────
    print(f"Generating buffer ({args.n_states} states x {args.n_z_per_state} z) ...", flush=True)
    X_s, X_z, Y_residual, baseline = generate_physics_buffer(
        source_policy, basis, source_context, pca, z_source,
        args.teacher, args.blend, device,
        n_states=args.n_states, n_z_per_state=args.n_z_per_state,
        use_imagination=True,
    )
    print(f"  Buffer: {len(X_s)} samples ({X_s.shape}, {X_z.shape}, {Y_residual.shape})")

    # ── Train adapter ───────────────────────────────────────────────────
    adapter = ResidualAdapter(s_dim, z_dim, a_dim, hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=args.lr)

    s_t = torch.tensor(X_s, device=device)
    z_t = torch.tensor(X_z, device=device)
    r_t = torch.tensor(Y_residual, device=device)
    dataset = TensorDataset(s_t, z_t, r_t)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print(f"Training {args.epochs} epochs ...", flush=True)
    for epoch in range(args.epochs):
        total_loss = 0.0
        for s_b, z_b, r_b in loader:
            pred = adapter(s_b, z_b)
            loss = F.mse_loss(pred, r_b)
            # Extra: L2 reg on output (prevent large residuals)
            l2 = 0.001 * (pred ** 2).mean()
            total = loss + l2
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            total_loss += float(loss)
        if epoch % 50 == 0:
            print(f"  epoch {epoch:3d}: loss={total_loss/len(loader):.6f}", flush=True)

    # ── Evaluate ─────────────────────────────────────────────────────────
    print("\n=== Evaluation ===", flush=True)

    # Source performance
    r_src, s_src = evaluate_adapter(adapter, source_policy, SHIFTS["source"],
                                    z_source, baseline, device, n_episodes=10)
    print(f"  Source: {r_src:.1f} +/- {s_src:.1f}")

    # Friction_070
    z_friction = np.load("results/cppe_pca_model.npz")["z_values"][2, :5].astype(np.float32)
    for label, zv in [("z_source", z_source), ("z_friction_oracle", z_friction)]:
        r, s = evaluate_adapter(adapter, source_policy, SHIFTS["friction_070"],
                                zv, baseline, device, n_episodes=10)
        print(f"  Friction ({label}): {r:.1f} +/- {s:.1f}")

    # baseline already computed and returned from generate_physics_buffer
    print(f"  Baseline residual: {baseline}")

    # ── Save ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    torch.save(adapter.state_dict(), args.model_out)
    print(f"\nSaved: {args.model_out}")


if __name__ == "__main__":
    main()
