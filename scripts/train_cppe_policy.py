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


@torch.no_grad()
def kan_reward_planner(
    states: torch.Tensor,
    z_values: np.ndarray,
    source_policy: FrozenSourcePolicy,
    source_context: AffineKANContext,
    basis,
    pca: CognitivePCA,
    device: torch.device,
    horizon: int = 5,
    n_candidates: int = 64,
    exploration_scale: float = 0.15,
):
    """KAN-based reward-aware counterfactual planner.

    Uses random shooting in the KAN-imagined target world to find
    reward-maximizing actions. The reward is computed from state/action
    (forward velocity + health - control cost), not from target env.

    Returns:
        a_planned: (N, action_dim) first action of best sequence
    """
    N = len(states)
    n_action = basis.action_dim
    n_feature = basis.feature_dim

    delta_W = torch.tensor(
        np.stack([pca.decode(z_values[i]) for i in range(N)]),
        device=device, dtype=torch.float32,
    )
    source_blocks = source_context.coefficients.clone().reshape(
        1 + n_action, n_feature, -1,
    )

    nominal = source_policy.action(states)
    best_actions = nominal.clone()

    for i in range(N):
        s0 = states[i:i+1]
        dw = delta_W[i].reshape(n_feature, -1)
        db = source_blocks.clone()
        db[0] = source_blocks[0] + dw
        drifted_ctx = AffineKANContext(
            db.reshape_as(source_context.coefficients)
        )
        nom = nominal[i:i+1]

        best_reward = -float('inf')
        best_first_a = nom.clone()

        # Batch candidates for efficiency
        noises = torch.randn(n_candidates, horizon, n_action, device=device) * exploration_scale
        noises[:, 0, :] = 0.0  # first action: start from nominal

        for c in range(n_candidates):
            s_t = s0.clone()
            total_r = 0.0
            alive = True
            first_a = None

            for t in range(horizon):
                a_t = (nom + noises[c, t:t+1]).clamp(-1.0, 1.0)
                if first_a is None:
                    first_a = a_t.clone()

                effect = drifted_ctx.acceleration(basis, s_t, a_t)
                s_next = s_t + effect

                # Reward components
                fwd_vel = s_next[0, 0] - s_t[0, 0]
                height_ok = (s_next[0, 1] > 0.7).float()
                angle_ok = (s_next[0, 2:5].abs() < 1.0).all().float()
                healthy = height_ok * angle_ok
                ctrl_cost = 0.001 * (a_t ** 2).sum()
                r_t = fwd_vel + 1.0 * healthy - ctrl_cost
                total_r += float(r_t)

                if s_next[0, 1] < 0.3 or s_next[0, 2:5].abs().max() > 2.0:
                    total_r -= 5.0
                    alive = False

                s_t = s_next
                if not alive:
                    break

            if total_r > best_reward:
                best_reward = total_r
                best_first_a = first_a.clone()

        best_actions[i:i+1] = best_first_a

    return best_actions


def generate_teacher_action(
    states: torch.Tensor, z_values: np.ndarray, teacher_mode: str,
    source_policy, source_context, basis, pca, device,
    blend_ratio: float = 0.3,
):
    """Unified interface: generate teacher actions for physics buffer."""
    if teacher_mode == "planner":
        a_raw = kan_reward_planner(
            states, z_values, source_policy, source_context,
            basis, pca, device,
        )
        # Soft blend with source action for stability
        a_src = source_policy.action(states)
        return (1.0 - blend_ratio) * a_src + blend_ratio * a_raw
    else:
        return compute_soft_cf_target(
            states, z_values, source_policy, source_context,
            basis, pca, device, blend_ratio=blend_ratio,
        )


@torch.no_grad()
def kan_imagination_rollout(
    init_states: torch.Tensor,      # (N, s_dim)
    z_values: np.ndarray,           # (N, k)
    source_policy: FrozenSourcePolicy,
    source_context: AffineKANContext,
    basis,
    pca: CognitivePCA,
    device: torch.device,
    horizon: int = 5,
    regularization: float = 1e-2,
) -> tuple[list, list, list]:
    """Generate counterfactual trajectories via KAN world model.

    For each (s_0, z'), rolls out horizon steps in the IMAGINED target physics:
      s_{t+1} = s_t + f_KAN(s_t, a_cf; W_target)
    where a_cf is computed via transport_action at each step.

    This produces states s' that are ON-POLICY for target physics,
    solving the state distribution mismatch.

    Returns: (all_states, all_z, all_actions) — flattened across batch and horizon.
    """
    N = len(init_states)
    n_action = basis.action_dim
    n_feature = basis.feature_dim

    delta_W = torch.tensor(
        np.stack([pca.decode(z_values[i]) for i in range(N)]),
        device=device, dtype=torch.float32,
    )
    source_blocks = source_context.coefficients.clone().reshape(
        1 + n_action, n_feature, -1,
    )

    all_s, all_z, all_a = [], [], []
    current_s = init_states.clone()

    for t in range(horizon):
        nominal = source_policy.action(current_s)
        source_effect = source_context.acceleration(basis, current_s, nominal)

        actions_cf = []
        next_states = []
        for i in range(N):
            dw = delta_W[i].reshape(n_feature, -1)
            drifted_blocks = source_blocks.clone()
            drifted_blocks[0] = source_blocks[0] + dw
            drifted_ctx = AffineKANContext(
                drifted_blocks.reshape_as(source_context.coefficients)
            )
            a_cf = drifted_ctx.transport_action(
                basis, current_s[i:i+1],
                desired_effect=source_effect[i:i+1],
                nominal_action=nominal[i:i+1],
                regularization=regularization,
            )
            actions_cf.append(a_cf)

            # KAN predicts next state: s' = s + f_KAN(s, a; W_target)
            effect = drifted_ctx.acceleration(basis, current_s[i:i+1], a_cf)
            next_s = current_s[i:i+1] + effect
            next_states.append(next_s)

        a_t = torch.cat(actions_cf, dim=0).clamp(-1.0, 1.0)
        ns_t = torch.cat(next_states, dim=0)

        all_s.append(current_s.cpu().numpy())
        all_z.append(z_values)
        all_a.append(a_t.cpu().numpy())

        current_s = ns_t

    return (
        np.concatenate(all_s, axis=0),
        np.tile(z_values, (horizon, 1)),
        np.concatenate(all_a, axis=0),
    )


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
    # DAgger
    parser.add_argument("--dagger", action="store_true",
                        help="Use DAgger-style student rollout for physics buffer states")
    parser.add_argument("--dagger-interval", type=int, default=5,
                        help="Do DAgger collection every N iterations")
    parser.add_argument("--cf-blend-schedule", choices=("none", "curriculum"), default="none",
                        help="Curriculum: blend ratio increases 0.2->0.5->1.0 over training")
    parser.add_argument("--kan-imagination", action="store_true",
                        help="Use KAN world model to generate counterfactual trajectories")
    parser.add_argument("--imagination-horizon", type=int, default=5,
                        help="Horizon for KAN imagination rollout")
    parser.add_argument("--imagination-threshold", type=float, default=0.0,
                        help="||ΔW|| threshold for adaptive imagination (0=always on)")
    parser.add_argument("--teacher", choices=("transport", "planner"), default="transport",
                        help="Action teacher: transport_action or KAN reward planner")
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

        # --- Curriculum blend ratio ---
        if args.cf_blend_schedule == "curriculum":
            progress = min(1.0, step / args.total_steps)
            if progress < 0.33:
                current_blend = 0.2
            elif progress < 0.66:
                current_blend = 0.5
            else:
                current_blend = 1.0
        else:
            current_blend = args.cf_blend

        s_dim_actual = 11  # Hopper state dimension (without z)

        # --- DAgger: student rollout using a single conditioned env ---
        if args.dagger and args.ablation == "full" and step % (args.eval_freq * args.dagger_interval) == 0:
            dagger_states = []
            for ep in range(4):
                # Create fresh env, copy normalization stats
                de = DummyVecEnv([make_env])
                de = VecNormalize(de, training=False, norm_obs=True, norm_reward=False)
                de.obs_rms = vec_env.obs_rms
                obs = de.reset()
                for _ in range(200):
                    dagger_states.append(obs[0, :s_dim_actual].copy())
                    action, _ = model.predict(obs, deterministic=False)
                    obs, _, dones, _ = de.step(action)
                    if dones[0]:
                        break
                de.close()
            if len(dagger_states) > 0:
                d_states = torch.tensor(np.stack(dagger_states[:128]),
                                        device=device, dtype=torch.float32)
                d_z = pca.sample_z(len(d_states)).astype(np.float32)
                d_cf = compute_counterfactual_action(
                    d_states, d_z, source_policy, source_context,
                    basis, pca, device,
                )
                for i in range(len(d_states)):
                    buffer_s.append(dagger_states[i])
                    buffer_z.append(d_z[i])
                    buffer_a.append(d_cf[i].cpu().numpy())
                if args.debug_loss:
                    print(f"  [dagger] collected {len(d_states)} states from student rollout",
                          flush=True)

        # --- KAN imagination: adaptive gate by ||ΔW|| ---
        if args.kan_imagination and args.ablation == "full":
            if hasattr(model, 'rollout_buffer') and model.rollout_buffer is not None:
                rb = model.rollout_buffer
                obs_data = rb.observations[:rb.pos].copy() if rb.pos > 0 else rb.observations.copy()
                n_init = min(16, len(obs_data))
                if n_init > 0:
                    idx = np.random.choice(len(obs_data), n_init, replace=False)
                    init_s = torch.tensor(
                        obs_data[idx][:, :s_dim_actual],
                        device=device, dtype=torch.float32,
                    )
                    z_im = pca.sample_z(n_init).astype(np.float32)

                    # Gate: decode ΔW, compute magnitude, split by threshold
                    dw_norms = np.array([
                        np.linalg.norm(pca.decode(z_im[i]))
                        for i in range(len(z_im))
                    ])
                    threshold = args.imagination_threshold
                    high_mask = dw_norms > threshold
                    low_mask = ~high_mask

                    n_high = int(high_mask.sum())
                    n_low = int(low_mask.sum())

                    # High-shift: use KAN imagination
                    if n_high > 0:
                        im_s, im_z, im_a = kan_imagination_rollout(
                            init_s[high_mask], z_im[high_mask],
                            source_policy, source_context, basis, pca, device,
                            horizon=args.imagination_horizon,
                        )
                        for i in range(len(im_s)):
                            buffer_s.append(im_s[i])
                            buffer_z.append(im_z[i])
                            buffer_a.append(im_a[i])

                    # Low-shift: use direct CF on source states (skip imagination)
                    if n_low > 0:
                        a_cf_low = compute_soft_cf_target(
                            init_s[low_mask], z_im[low_mask],
                            source_policy, source_context, basis, pca, device,
                            blend_ratio=current_blend,
                        )
                        for i in range(n_low):
                            buffer_s.append(init_s[low_mask][i].cpu().numpy())
                            buffer_z.append(z_im[low_mask][i])
                            buffer_a.append(a_cf_low[i].cpu().numpy())

                    if args.debug_loss:
                        print(f"  [imagination] high={n_high} (rollout) low={n_low} "
                              f"(direct) threshold={threshold:.3f}", flush=True)

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
                    a_cf = generate_teacher_action(
                        states, z_vals, args.teacher,
                        source_policy, source_context, basis, pca, device,
                        blend_ratio=current_blend,
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
