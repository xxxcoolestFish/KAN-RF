"""Train CPPE policy v2: PPO + physics manifold training with source anchor.

Key improvements over v1:
  - Source anchor loss: pi(s, z_source) should match pi_source(s)
  - Physics buffer includes z_source samples (no degrade on source)
  - 2D PC combination sampling (not just single-axis perturbation)
  - Loss scale monitoring for debugging

Training loss:
  L = L_PPO + lambda_cf * L_cf + lambda_anchor * L_anchor
"""

from __future__ import annotations

import argparse, json, sys, time, os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cpbn.cognitive_pca import CognitivePCA, PCARanges
from cpbn.cppe_env import PhysicsConditionedEnv
from cpbn.generic_affine_kan import AffineKANContext
from scripts.prescreen_hopper_physics_shifts import ENVS, SHIFTS, make_shifted_env
from scripts.validate_hopper_joint_online_adaptation import (
    FrozenSourcePolicy,
    load_cognition,
)


def compute_counterfactual_action(
    states: torch.Tensor,
    z_values: np.ndarray,
    source_policy: FrozenSourcePolicy,
    source_context: AffineKANContext,
    basis,
    pca: CognitivePCA,
    device: torch.device,
    regularization: float = 1e-2,
) -> torch.Tensor:
    """Generate counterfactual compensatory actions for (s, z) pairs."""
    N = len(states)
    a_dim = basis.action_dim
    width = basis.feature_dim

    nominal = source_policy.action(states)
    source_effect = source_context.acceleration(basis, states, nominal)

    delta_W = torch.tensor(
        np.stack([pca.decode(z_values[i]) for i in range(N)]),
        device=device, dtype=torch.float32,
    )

    source_blocks = source_context.coefficients.clone().reshape(
        1 + a_dim, width, -1,
    )

    actions_cf = []
    for i in range(N):
        dw = delta_W[i].reshape(width, -1)
        drifted_blocks = source_blocks.clone()
        drifted_blocks[0] = source_blocks[0] + dw
        drifted_ctx = AffineKANContext(
            drifted_blocks.reshape_as(source_context.coefficients)
        )
        a_cf = drifted_ctx.transport_action(
            basis, states[i:i+1],
            desired_effect=source_effect[i:i+1],
            nominal_action=nominal[i:i+1],
            regularization=regularization,
        )
        actions_cf.append(a_cf)
    return torch.cat(actions_cf, dim=0)


def compute_soft_cf_target(
    states: torch.Tensor,
    z_values: np.ndarray,
    source_policy: FrozenSourcePolicy,
    source_context: AffineKANContext,
    basis,
    pca: CognitivePCA,
    device: torch.device,
    blend_ratio: float = 0.2,
    regularization: float = 1e-2,
) -> torch.Tensor:
    """Soft counterfactual target: blend source action with CF compensation.

    a_target = (1 - eta) * pi_source(s) + eta * a_cf

    This prevents the policy from jumping too far from source behavior,
    reducing closed-loop instability.
    """
    a_cf = compute_counterfactual_action(
        states, z_values, source_policy, source_context,
        basis, pca, device, regularization,
    )
    a_src = source_policy.action(states)
    return (1.0 - blend_ratio) * a_src + blend_ratio * a_cf


def compute_cf_distance_weights(
    z_values: np.ndarray,
    z_source: np.ndarray,
    gamma: float = 10.0,
) -> torch.Tensor:
    """Distance-based weights for CF supervision.

    w(z) = exp(-gamma * ||z - z_source||)

    Near-source z gets higher weight (KAN more reliable).
    Far-from-source z gets lower weight (KAN may be inaccurate).
    """
    z_src = np.asarray(z_source, dtype=np.float32)
    dist = np.linalg.norm(z_values - z_src[None, :], axis=-1)
    weights = np.exp(-gamma * dist)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_policy(model, env, n_episodes=10):
    """Evaluate a PPO model on a VecEnv."""
    returns = []
    for _ in range(n_episodes):
        obs = env.reset()
        total = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            total += float(reward[0])
            if dones[0]:
                break
        returns.append(total)
    return float(np.mean(returns)), float(np.std(returns))


def _forward_policy_mean(model, obs_normalized: torch.Tensor) -> torch.Tensor:
    """Get action mean from SB3 policy for a normalized observation."""
    features = model.policy.mlp_extractor(
        model.policy.features_extractor(obs_normalized)
    )
    if isinstance(features, tuple):
        features = features[0]
    return model.policy.action_net(features)


def main():
    parser = argparse.ArgumentParser()
    # Training
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-envs", type=int, default=4)
    # CPPE
    parser.add_argument("--lambda-cf", type=float, default=1.0,
                        help="Weight for physics counterfactual loss")
    parser.add_argument("--lambda-anchor", type=float, default=1.0,
                        help="Weight for source anchor loss")
    parser.add_argument("--cf-blend", type=float, default=0.3,
                        help="Blend ratio for soft CF target (0=source only, 1=full CF)")
    parser.add_argument("--cf-distance-gamma", type=float, default=5.0,
                        help="Distance decay for CF loss weight (higher = more local)")
    parser.add_argument("--cf-updates-per-iter", type=int, default=5)
    parser.add_argument("--k-pcs", type=int, default=5)
    # Z sampling
    parser.add_argument("--z-sampling", choices=("1d", "2d"), default="2d")
    # Evaluation
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    # I/O
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--pca-model", default="results/cppe_pca_model.npz")
    parser.add_argument("--out-dir", default="results/cppe_training")
    # Ablation
    parser.add_argument("--ablation",
                        choices=("full", "ppo_only", "random_z", "pca_only"),
                        default="full")
    # Debug
    parser.add_argument("--debug-loss", action="store_true",
                        help="Print loss scales each iteration")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"CPPE v2: lambda_cf={args.lambda_cf}, lambda_anchor={args.lambda_anchor}, "
          f"cf_blend={args.cf_blend}, cf_gamma={args.cf_distance_gamma}, "
          f"ablation={args.ablation}, z_sampling={args.z_sampling}, device={device}")

    # ── Load PCA ────────────────────────────────────────────────────────
    pca_data = np.load(args.pca_model)
    pc_ranges = PCARanges(mins=pca_data["pc_mins"], maxs=pca_data["pc_maxs"])
    pca = CognitivePCA(args.pca_model, k=args.k_pcs, pc_ranges=pc_ranges)
    z_source = pca.z_source.astype(np.float32)
    z_dim = pca.k
    print(f"PCA: k={z_dim}, cumulated={pca.explained_cumulative:.1%}, z_source={z_source}")

    # ── Load source components ──────────────────────────────────────────
    print("Loading source policy + cognition ...", flush=True)
    source_policy = FrozenSourcePolicy(
        "results/hopper_source_sb3_ppo_continued_seed1811.zip",
        "results/hopper_source_sb3_vecnorm_continued_seed1811.pkl",
        device, args.seed, env="hopper",
    )
    basis, source_context, _, delta_scale = load_cognition(
        argparse.Namespace(
            cognition_checkpoint="results/hopper_source_control_sobolev_calibrated_seed1811.pt",
            device=args.device,
        ),
        device,
    )

    # ── Environments ────────────────────────────────────────────────────
    def make_env():
        base = make_shifted_env(SHIFTS["source"], args.seed, "hopper")()
        cond = PhysicsConditionedEnv(base, z_dim=z_dim)
        cond.set_z(z_source)
        return cond

    vec_env = DummyVecEnv([make_env for _ in range(args.n_envs)])
    vec_env = VecNormalize(vec_env, training=True, norm_obs=True, norm_reward=True)

    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)
    eval_env.obs_rms = vec_env.obs_rms

    # ── Phase 1: PPO ────────────────────────────────────────────────────
    phase1_steps = args.total_steps // 2
    phase2_steps = args.total_steps - phase1_steps
    print(f"\n=== Phase 1: PPO ({phase1_steps} steps) ===", flush=True)
    model = PPO("MlpPolicy", vec_env,
                n_steps=args.n_steps // args.n_envs,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                learning_rate=args.lr,
                device=device, verbose=0)
    model.learn(total_timesteps=phase1_steps, progress_bar=False)
    r, s = evaluate_policy(model, eval_env, args.eval_episodes)
    print(f"  Phase 1 done: return={r:.1f} +/- {s:.1f}")

    # ── Phase 2: CPPE ───────────────────────────────────────────────────
    print(f"\n=== Phase 2: CPPE ({phase2_steps} steps) ===", flush=True)

    # Mixed physics buffer: (state, z, target_action)
    buffer_s, buffer_z, buffer_a = [], [], []

    # Source anchor: pi(s, z_source) should match pi_source(s)
    source_anchor_s, source_anchor_z, source_anchor_a = [], [], []

    step = phase1_steps
    n_steps_per_env = args.n_steps // args.n_envs

    while step < args.total_steps:
        # --- Collect rollout with z = z_source ---
        for env_idx in range(args.n_envs):
            vec_env.envs[env_idx].set_z(z_source)

        remaining = args.total_steps - step
        chunk = min(args.eval_freq, remaining)
        model.learn(total_timesteps=step + chunk,
                    reset_num_timesteps=False, progress_bar=False)
        step += chunk

        # --- Generate physics buffer data ---
        if hasattr(model, 'rollout_buffer') and model.rollout_buffer is not None:
            rb = model.rollout_buffer
            obs_data = rb.observations[:rb.pos].copy() if rb.pos > 0 else rb.observations.copy()
            n_states = min(128, len(obs_data))
            if n_states > 0:
                indices = np.random.choice(len(obs_data), n_states, replace=False)
                raw_obs = obs_data[indices]
                s_dim_actual = raw_obs.shape[1] - z_dim
                states_np = raw_obs[:, :s_dim_actual]
                states = torch.tensor(states_np, device=device, dtype=torch.float32)

                # --- Source anchor: (s, z_source, pi_source(s)) ---
                if args.ablation != "ppo_only":
                    source_actions = source_policy.action(states).cpu().numpy()
                    for i in range(n_states):
                        source_anchor_s.append(states_np[i])
                        source_anchor_z.append(z_source.copy())
                        source_anchor_a.append(source_actions[i])

                # --- Physics samples ---
                if args.ablation == "random_z":
                    z_vals = np.random.randn(n_states, z_dim).astype(np.float32) * 0.5
                elif args.ablation in ("full", "pca_only"):
                    z_vals = pca.sample_z(n_states).astype(np.float32)

                if args.ablation in ("full", "random_z"):
                    # Soft CF target: blend source action + CF compensation
                    a_cf = compute_soft_cf_target(
                        states, z_vals, source_policy, source_context,
                        basis, pca, device,
                        blend_ratio=args.cf_blend,
                    )
                    for i in range(n_states):
                        buffer_s.append(states_np[i])
                        buffer_z.append(z_vals[i])
                        buffer_a.append(a_cf[i].cpu().numpy())
                elif args.ablation == "pca_only":
                    for i in range(n_states):
                        buffer_s.append(states_np[i])
                        buffer_z.append(z_vals[i])
                        buffer_a.append(np.zeros(3, dtype=np.float32))

        # --- Physics supervised updates ---
        if args.ablation != "ppo_only":
            self_optimizer = model.policy.optimizer
            has_physics = len(buffer_s) >= args.batch_size

            for u in range(args.cf_updates_per_iter):
                cf_loss_val = 0.0
                anchor_loss_val = 0.0

                # -- Physics CF loss --
                if has_physics and args.ablation in ("full", "random_z"):
                    b_idx = np.random.choice(len(buffer_s), args.batch_size, replace=False)
                    s_b = torch.tensor(np.stack([buffer_s[i] for i in b_idx]),
                                       device=device, dtype=torch.float32)
                    z_b = torch.tensor(np.stack([buffer_z[i] for i in b_idx]),
                                       device=device, dtype=torch.float32)
                    a_tgt = torch.tensor(np.stack([buffer_a[i] for i in b_idx]),
                                         device=device, dtype=torch.float32)

                    # Normalize state (pad with z_source, normalize, strip)
                    z_pad = np.tile(z_source[None, :], (len(s_b), 1))
                    s_padded = np.concatenate([s_b.cpu().numpy(), z_pad], axis=-1)
                    s_norm = torch.tensor(
                        vec_env.normalize_obs(s_padded)[:, :s_dim_actual],
                        device=device, dtype=torch.float32,
                    )
                    policy_input = torch.cat([s_norm, z_b], dim=-1)
                    pred = _forward_policy_mean(model, policy_input)
                    # Distance-weighted CF loss
                    w_dist = compute_cf_distance_weights(
                        z_b.cpu().numpy(), z_source,
                        gamma=args.cf_distance_gamma,
                    ).to(device)
                    cf_loss = (w_dist[:, None] * (pred - a_tgt) ** 2).mean()
                    cf_loss_val = float(cf_loss)

                    self_optimizer.zero_grad()
                    (args.lambda_cf * cf_loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
                    self_optimizer.step()

                # -- Source anchor loss --
                if len(source_anchor_s) >= args.batch_size:
                    a_idx = np.random.choice(len(source_anchor_s), args.batch_size, replace=False)
                    s_a = torch.tensor(np.stack([source_anchor_s[i] for i in a_idx]),
                                       device=device, dtype=torch.float32)
                    z_a = torch.tensor(np.stack([source_anchor_z[i] for i in a_idx]),
                                       device=device, dtype=torch.float32)
                    a_src = torch.tensor(np.stack([source_anchor_a[i] for i in a_idx]),
                                         device=device, dtype=torch.float32)

                    z_pad = np.tile(z_source[None, :], (len(s_a), 1))
                    s_padded = np.concatenate([s_a.cpu().numpy(), z_pad], axis=-1)
                    s_norm = torch.tensor(
                        vec_env.normalize_obs(s_padded)[:, :s_dim_actual],
                        device=device, dtype=torch.float32,
                    )
                    policy_input_a = torch.cat([s_norm, z_a], dim=-1)
                    pred_a = _forward_policy_mean(model, policy_input_a)
                    anchor_loss = F.mse_loss(pred_a, a_src)
                    anchor_loss_val = float(anchor_loss)

                    self_optimizer.zero_grad()
                    (args.lambda_anchor * anchor_loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
                    self_optimizer.step()

                if args.debug_loss and (cf_loss_val > 0 or anchor_loss_val > 0):
                    print(f"  [debug] step={step} cf_loss={cf_loss_val:.6f} "
                          f"anchor_loss={anchor_loss_val:.6f}", flush=True)

        # --- Buffer management ---
        max_buffer = 10000
        if len(buffer_s) > max_buffer:
            keep = np.random.choice(len(buffer_s), max_buffer // 2, replace=False)
            buffer_s = [buffer_s[i] for i in keep]
            buffer_z = [buffer_z[i] for i in keep]
            buffer_a = [buffer_a[i] for i in keep]
        if len(source_anchor_s) > max_buffer:
            keep = np.random.choice(len(source_anchor_s), max_buffer // 2, replace=False)
            source_anchor_s = [source_anchor_s[i] for i in keep]
            source_anchor_z = [source_anchor_z[i] for i in keep]
            source_anchor_a = [source_anchor_a[i] for i in keep]

        # --- Evaluate ---
        r, s = evaluate_policy(model, eval_env, args.eval_episodes)
        print(f"  step={step:6d}  return={r:.1f} +/- {s:.1f}  "
              f"buf={len(buffer_s)}  anchor={len(source_anchor_s)}", flush=True)

    # ── Save ────────────────────────────────────────────────────────────
    tag = f"cppe_v2_{args.ablation}_z{args.z_sampling}_cf{args.lambda_cf}_a{args.lambda_anchor}_blend{args.cf_blend}"
    model_path = os.path.join(args.out_dir, f"{tag}_seed{args.seed}.zip")
    norm_path = os.path.join(args.out_dir, f"{tag}_norm_seed{args.seed}.pkl")
    model.save(model_path)
    vec_env.save(norm_path)
    r, s = evaluate_policy(model, eval_env, args.eval_episodes)
    print(f"\nFinal source return: {r:.1f} +/- {s:.1f}")
    print(f"Saved: {model_path}")


if __name__ == "__main__":
    main()
