"""Phase 0B: Oracle target critic vs source critic gradient audit.

Key question:
  Does a target critic (trained on real friction transitions with known
  Hopper reward) provide better action-gradient direction than V_source?

Architecture:
  g = B_KAN^T * grad(V)(s'_KAN)    [normalized to unit direction]
  Compare V_source vs V_target on identical states.

If target >> source: architecture holds, problem = "how to learn target value reward-free"
If target ≈ source: source critic already sufficient, proceed to closed-loop
If target << source: target critic gradient geometry is wrong (don't kill architecture)
"""

from __future__ import annotations

import argparse, sys, os, json
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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

# Hopper-v5 exact params
FORWARD_WEIGHT = 1.0
HEALTHY_REWARD = 1.0
CTRL_COST_WEIGHT = 0.001
Z_LO, Z_HI = 0.7, float('inf')
ANG_LO, ANG_HI = -0.2, 0.2
DT = 0.008 * 4


# ═══════════════════════════════════════════════════════════════════════
# Reward helpers
# ═══════════════════════════════════════════════════════════════════════

def hopper_reward_np(obs, action, prev_obs=None):
    """Exact Hopper-v5 reward."""
    fwd = obs[5] if prev_obs is None else (obs[0] - prev_obs[0]) / DT
    z_ok = Z_LO < obs[1] < Z_HI
    a_ok = ANG_LO < obs[2] < ANG_HI
    healthy = 1.0 if (z_ok and a_ok) else 0.0
    ctrl = CTRL_COST_WEIGHT * float(np.sum(action ** 2))
    return FORWARD_WEIGHT * fwd + HEALTHY_REWARD * healthy - ctrl


def is_flight(state):
    z = state[1]
    foot_contact = np.abs(state[-2:]).sum()
    return z > 0.85 and foot_contact < 0.5


# ═══════════════════════════════════════════════════════════════════════
# Oracle Target Critic
# ═══════════════════════════════════════════════════════════════════════

class TargetCritic(nn.Module):
    """Simple MLP value function V(s)."""
    def __init__(self, s_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # Initialize last layer small for stable gradients
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, s):
        return self.net(s).squeeze(-1)


def collect_target_trajectories(source_policy, basis, source_context, target_ctx,
                                shift, device, n_episodes=20, seed=1811):
    """Collect trajectories from friction env using Transport policy.

    Returns:
        trajectories: list of lists of (obs, action, reward, next_obs, done)
    """
    trajectories = []
    for ep in range(n_episodes):
        env = make_shifted_env(shift, seed + ep * 100, "hopper")()
        obs, _ = env.reset(seed=seed + ep * 100)
        traj = []
        prev_obs = None
        while True:
            s_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            nominal = source_policy.action(s_t)
            s_eff = source_context.acceleration(basis, s_t, nominal)
            a_tr = target_ctx.transport_action(
                basis, s_t, desired_effect=s_eff,
                nominal_action=nominal, regularization=1e-2,
            ).clamp(-1, 1).squeeze(0).cpu().numpy()

            next_obs, _, terminated, truncated, _ = env.step(a_tr)
            r = hopper_reward_np(next_obs, a_tr, prev_obs)
            done = terminated or truncated

            traj.append({
                "obs": obs.copy(), "action": a_tr.copy(),
                "reward": r, "next_obs": next_obs.copy(), "done": done,
            })
            prev_obs = obs.copy()
            obs = next_obs
            if done:
                break
        env.close()
        trajectories.append(traj)
        if (ep + 1) % 5 == 0:
            print(f"    Collected {ep + 1}/{n_episodes} episodes...", flush=True)
    return trajectories


def compute_mc_returns(trajectories, gamma=0.99):
    """Compute Monte Carlo discounted returns for each state."""
    all_states = []
    all_returns = []
    for traj in trajectories:
        returns = []
        G = 0.0
        for t in reversed(traj):
            G = t["reward"] + gamma * G * (1 - t["done"])
            returns.append(G)
        returns.reverse()
        for t, G in zip(traj, returns):
            all_states.append(t["obs"])
            all_returns.append(G)
    return np.stack(all_states).astype(np.float32), np.array(all_returns, dtype=np.float32)


def train_oracle_critic(states, returns, s_dim, device, epochs=200, batch_size=512):
    """Train V(s) to predict MC returns."""
    X = torch.tensor(states, dtype=torch.float32)
    y = torch.tensor(returns, dtype=torch.float32)

    # Normalize targets
    y_mean, y_std = y.mean().item(), y.std().item()
    y_norm = (y - y_mean) / (y_std + 1e-8)

    dataset = TensorDataset(X, y_norm)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    critic = TargetCritic(s_dim).to(device)
    opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    best_loss = float('inf')
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred = critic(batch_x)
            loss = F.mse_loss(pred, batch_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(batch_x)
        scheduler.step()
        avg_loss = total_loss / len(dataset)
        if avg_loss < best_loss:
            best_loss = avg_loss

        if (epoch + 1) % 50 == 0:
            print(f"    epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}", flush=True)

    print(f"  Oracle critic trained: best MSE={best_loss:.6f}, "
          f"y_mean={y_mean:.2f}, y_std={y_std:.2f}", flush=True)
    return critic, y_mean, y_std


# ═══════════════════════════════════════════════════════════════════════
# Gradient computation for both critics
# ═══════════════════════════════════════════════════════════════════════

def compute_source_value_gradient(states, source_policy):
    """grad_s V_source(s) via autograd through source PPO critic."""
    mean = source_policy.mean
    var = source_policy.variance
    s_rg = states.detach().clone().requires_grad_(True)
    s_norm = ((s_rg - mean) / (var + 1e-8).sqrt()).clamp(-10, 10)
    features = source_policy.model.policy.features_extractor(s_norm)
    latent = source_policy.model.policy.mlp_extractor(features)
    if isinstance(latent, tuple):
        _, latent_vf = latent
    else:
        latent_vf = latent
    values = source_policy.model.policy.value_net(latent_vf).squeeze(-1)
    grad_v = torch.autograd.grad(values.sum(), s_rg)[0].detach()
    return grad_v, values.detach()


def compute_target_value_gradient(states, target_critic):
    """grad_s V_target(s) via autograd through trained oracle critic."""
    s_rg = states.detach().clone().requires_grad_(True)
    values = target_critic(s_rg)
    grad_v = torch.autograd.grad(values.sum(), s_rg)[0].detach()
    return grad_v, values.detach()


def compute_normalized_action_gradient(s, a, basis, target_ctx, critic, critic_type):
    """Compute normalized one-step action gradient.

    g = normalize(B_KAN^T * grad(V_critic)(s'_KAN))

    Note: dr/da (ctrl_cost) is negligible (0.0016 vs 0.0084 for B^T*gradV)
    and has cos=-0.04 with true gradient. We omit it for cleaner comparison
    between critics. The line search uses real env which includes ctrl_cost.

    Returns:
        g_hat: (a_dim,) unit-norm gradient direction
        g_raw: (a_dim,) unnormalized gradient
        v_next: scalar value at KAN-predicted next state
    """
    effect = target_ctx.acceleration(basis, s, a)
    s_next = s + effect

    if critic_type == "source":
        grad_v, v_next = compute_source_value_gradient(s_next, critic)
    else:
        grad_v, v_next = compute_target_value_gradient(s_next, critic)

    _, gain = target_ctx.drift_and_gain(basis, s)
    B = gain.squeeze(0)  # (s_dim, a_dim)
    g_raw = B.T @ grad_v.squeeze(0)  # (a_dim,)

    norm = g_raw.norm()
    if norm > 1e-8:
        g_hat = g_raw / norm
    else:
        g_hat = torch.zeros_like(g_raw)

    return g_hat, g_raw, v_next.item() if v_next.numel() == 1 else v_next.mean().item()


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--n-states", type=int, default=80,
                       help="Target number of alive states to audit")
    parser.add_argument("--fd-eps", type=float, default=0.05,
                       help="FD step size for true gradient")
    parser.add_argument("--critic-epochs", type=int, default=200)
    parser.add_argument("--critic-episodes", type=int, default=20,
                       help="Episodes to collect for training oracle critic")
    parser.add_argument("--json-out", default="results/phase0b_oracle_target_critic.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    shift = SHIFTS["friction_070"]
    # Normalized alphas for unit-direction line search
    # (these are step sizes along the unit gradient direction)
    alphas = [-0.1, -0.05, -0.02, 0.02, 0.05, 0.1, 0.15, 0.2]

    print("=" * 72)
    print("Phase 0B: Oracle Target Critic vs Source Critic Gradient Audit")
    print(f"  n_states_target={args.n_states}, fd_eps={args.fd_eps}")
    print("=" * 72)

    # ── 1. Load components ──────────────────────────────────────────────
    print("\n[1/5] Loading source components...", flush=True)

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

    print("  Fitting KAN to friction_070...", flush=True)
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

    # ── 2. Train oracle target critic ───────────────────────────────────
    print(f"\n[2/5] Training oracle target critic ({args.critic_episodes} episodes)...",
          flush=True)

    train_trajs = collect_target_trajectories(
        source_policy, basis, source_context, target_ctx,
        shift, device, n_episodes=args.critic_episodes, seed=args.seed,
    )

    # Count total transitions
    n_transitions = sum(len(t) for t in train_trajs)
    avg_ep_len = n_transitions / len(train_trajs)
    print(f"  Collected {n_transitions} transitions, avg episode length={avg_ep_len:.0f}",
          flush=True)

    states_mc, returns_mc = compute_mc_returns(train_trajs, gamma=0.99)
    print(f"  MC returns: mean={returns_mc.mean():.2f}, std={returns_mc.std():.2f}, "
          f"min={returns_mc.min():.2f}, max={returns_mc.max():.2f}", flush=True)

    target_critic, v_mean, v_std = train_oracle_critic(
        states_mc, returns_mc, basis.state_dim, device,
        epochs=args.critic_epochs,
    )
    target_critic.eval()

    # ── 3. Collect candidate states (from env A) ─────────────────────────
    print(f"\n[3/5] Collecting candidate states...", flush=True)

    collect_env = make_shifted_env(shift, args.seed, "hopper")()
    candidates = []
    for ep in range(20):
        obs, _ = collect_env.reset(seed=args.seed + 1000 + ep * 100)
        step = 0
        while step < 300:
            s_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            nominal = source_policy.action(s_t)
            s_eff = source_context.acceleration(basis, s_t, nominal)
            a_tr = target_ctx.transport_action(
                basis, s_t, desired_effect=s_eff,
                nominal_action=nominal, regularization=1e-2,
            ).clamp(-1, 1).squeeze(0).cpu().numpy()

            qp = collect_env.unwrapped.data.qpos.copy()
            qv = collect_env.unwrapped.data.qvel.copy()
            candidates.append({
                "qp": qp, "qv": qv, "obs": obs.copy(), "a_tr": a_tr,
                "flight": is_flight(obs),
            })
            obs, _, t, tr, _ = collect_env.step(a_tr)
            step += 1
            if t or tr:
                break
        if len(candidates) >= args.n_states * 5:
            break
    collect_env.close()
    print(f"  Collected {len(candidates)} candidates", flush=True)

    # ── 4. Audit: verify + gradient comparison in SINGLE env ────────────
    print(f"\n[4/5] Auditing states (target: {args.n_states} alive in audit env)...",
          flush=True)

    fd_env = make_shifted_env(shift, args.seed + 999, "hopper")()
    a_dim = basis.action_dim
    verified = []

    # First pass: verify which candidates survive baseline in THIS env
    print("  Verifying baseline survival...", flush=True)
    for rec in candidates:
        if len(verified) >= args.n_states:
            break
        fd_env.reset(seed=args.seed + 999)
        fd_env.unwrapped.set_state(rec["qp"], rec["qv"])
        alive = True
        for _ in range(4):
            _, _, t_bs, tr_bs, _ = fd_env.step(rec["a_tr"])
            if t_bs or tr_bs:
                alive = False
                break
        if alive:
            verified.append(rec)

    print(f"  Verified alive: {len(verified)}/{min(len(candidates), args.n_states)} "
          f"({100*len(verified)/max(1,min(len(candidates), args.n_states)):.1f}%)",
          flush=True)

    if len(verified) < 20:
        print(f"  WARNING: Few verified states. Continuing with {len(verified)}.", flush=True)
    if len(verified) < 5:
        print(f"  ABORT: Not enough alive states.", flush=True)
        fd_env.close()
        return

    # Second pass: gradient audit on verified states (SAME env instance)
    print(f"  Running gradient audit on {len(verified)} states...", flush=True)

    metrics = {
        "source": {"cosines": [], "improved": [], "best_dr": [], "terminated": []},
        "target": {"cosines": [], "improved": [], "best_dr": [], "terminated": []},
    }
    per_state = []

    for i, rec in enumerate(verified):
        s_t = torch.as_tensor(rec["obs"], device=device, dtype=torch.float32).unsqueeze(0)
        a_t = torch.as_tensor(rec["a_tr"], device=device, dtype=torch.float32).unsqueeze(0)
        a_np = rec["a_tr"]
        qp, qv = rec["qp"], rec["qv"]

        # --- True gradient via FD (SAME env) ---
        fd_env.reset(seed=args.seed + 999)  # full reset to clear internal state
        fd_env.unwrapped.set_state(qp, qv)
        baseline_r = 0.0
        alive_bl = True
        prev_bl = None
        for _ in range(4):
            o_bl, r_bl, t_bl, tr_bl, _ = fd_env.step(a_np)
            baseline_r += float(r_bl)
            if t_bl or tr_bl:
                alive_bl = False
                break
            prev_bl = o_bl.copy()

        if not alive_bl:
            continue

        g_true = np.zeros(a_dim)
        for dim in range(a_dim):
            for sign in [+1, -1]:
                da = np.zeros(a_dim)
                da[dim] = sign * args.fd_eps
                a_pert = np.clip(a_np + da, -1, 1)
                fd_env.unwrapped.set_state(qp, qv)
                total_r = 0.0
                alive_pert = True
                prev_p = None
                for _ in range(4):
                    o_p, r_p, t_p, tr_p, _ = fd_env.step(a_pert)
                    total_r += float(r_p)
                    if t_p or tr_p:
                        alive_pert = False
                        break
                    prev_p = o_p.copy()
                if alive_pert:
                    g_true[dim] += sign * total_r / (2.0 * args.fd_eps)
                else:
                    g_true[dim] += sign * (-10.0) / (2.0 * args.fd_eps)

        g_true_norm = np.linalg.norm(g_true)
        if g_true_norm < 1e-8:
            continue

        state_result = {"baseline_r": float(baseline_r), "flight": rec["flight"]}

        for critic_type, critic_obj in [("source", source_policy), ("target", target_critic)]:
            g_hat, g_raw, v_next = compute_normalized_action_gradient(
                s_t, a_t, basis, target_ctx, critic_obj, critic_type,
            )
            g_hat_np = g_hat.cpu().numpy()

            cos = float(np.dot(g_hat_np, g_true) / (g_true_norm + 1e-10))
            metrics[critic_type]["cosines"].append(cos)

            best_dr = -float('inf')
            best_alpha = None
            any_terminated = False
            dr_values = {}
            for alpha in alphas:
                da = alpha * g_hat_np
                a_test = np.clip(a_np + da, -1, 1)
                fd_env.unwrapped.set_state(qp, qv)
                total_r = 0.0
                alive_ls = True
                prev_ls = None
                for _ in range(4):
                    o_ls, r_ls, t_ls, tr_ls, _ = fd_env.step(a_test)
                    total_r += float(r_ls)
                    if t_ls or tr_ls:
                        alive_ls = False
                        break
                    prev_ls = o_ls.copy()
                dr = total_r - baseline_r if alive_ls else float('nan')
                dr_values[str(alpha)] = float(dr) if alive_ls else None
                if alive_ls and dr > best_dr:
                    best_dr = dr
                    best_alpha = alpha
                if not alive_ls:
                    any_terminated = True

            improved = best_dr > 0 if best_alpha is not None else False
            metrics[critic_type]["improved"].append(float(improved))
            metrics[critic_type]["best_dr"].append(float(best_dr) if best_alpha is not None else 0.0)
            metrics[critic_type]["terminated"].append(float(any_terminated))

            state_result[f"{critic_type}_cos"] = cos
            state_result[f"{critic_type}_best_alpha"] = best_alpha
            state_result[f"{critic_type}_best_dr"] = float(best_dr) if best_alpha is not None else None
            state_result[f"{critic_type}_v_next"] = float(v_next)
            state_result[f"{critic_type}_terminated"] = any_terminated
            state_result[f"{critic_type}_dr_values"] = dr_values

        per_state.append(state_result)

        if (i + 1) % 20 == 0:
            src_mean = np.mean(metrics["source"]["cosines"]) if metrics["source"]["cosines"] else 0
            tgt_mean = np.mean(metrics["target"]["cosines"]) if metrics["target"]["cosines"] else 0
            print(f"    {len(per_state)} valid so far, "
                  f"source_cos={src_mean:.4f}, target_cos={tgt_mean:.4f}", flush=True)

    fd_env.close()
    n_valid = len(per_state)

    # ── 5. Report ───────────────────────────────────────────────────────
    print(f"\n[5/5] Results ({n_valid} valid states)", flush=True)
    print("=" * 72)

    def safe_mean(arr):
        return float(np.mean(arr)) if arr else float('nan')

    def safe_frac(arr):
        return float(np.mean(arr)) if arr else float('nan')

    src_cos = metrics["source"]["cosines"]
    tgt_cos = metrics["target"]["cosines"]
    src_imp = metrics["source"]["improved"]
    tgt_imp = metrics["target"]["improved"]
    src_dr = metrics["source"]["best_dr"]
    tgt_dr = metrics["target"]["best_dr"]
    src_term = metrics["source"]["terminated"]
    tgt_term = metrics["target"]["terminated"]

    # Also compute stance/flight breakdown
    stance_idx = [j for j, ps in enumerate(per_state) if not ps["flight"]]
    flight_idx = [j for j, ps in enumerate(per_state) if ps["flight"]]

    print(f"\n  {'Metric':35s} {'Source':>12s} {'Target':>12s} {'Delta':>12s}")
    print(f"  {'-'*72}")

    rows = [
        ("Mean cosine w/ g_true", safe_mean(src_cos), safe_mean(tgt_cos)),
        ("Positive cosine fraction", safe_frac([c > 0 for c in src_cos]),
         safe_frac([c > 0 for c in tgt_cos])),
        ("Cosine > 0.3 fraction", safe_frac([c > 0.3 for c in src_cos]),
         safe_frac([c > 0.3 for c in tgt_cos])),
        ("Line-search improved fraction", safe_mean(src_imp), safe_mean(tgt_imp)),
        ("Mean best ΔR", safe_mean(src_dr), safe_mean(tgt_dr)),
        ("Median best ΔR", float(np.median(src_dr)) if src_dr else float('nan'),
         float(np.median(tgt_dr)) if tgt_dr else float('nan')),
        ("Any termination fraction", safe_mean(src_term), safe_mean(tgt_term)),
    ]
    if stance_idx:
        rows.append(("Stance cosine", safe_mean([src_cos[j] for j in stance_idx]),
                     safe_mean([tgt_cos[j] for j in stance_idx])))
    if flight_idx:
        rows.append(("Flight cosine", safe_mean([src_cos[j] for j in flight_idx]),
                     safe_mean([tgt_cos[j] for j in flight_idx])))

    for label, sv, tv in rows:
        delta = tv - sv if not (np.isnan(sv) or np.isnan(tv)) else float('nan')
        delta_str = f"{delta:+.4f}" if not np.isnan(delta) else "N/A"
        sv_str = f"{sv:.4f}" if not np.isnan(sv) else "N/A"
        tv_str = f"{tv:.4f}" if not np.isnan(tv) else "N/A"
        print(f"  {label:35s} {sv_str:>12s} {tv_str:>12s} {delta_str:>12s}")

    # Head-to-head: per-state winner
    src_wins = sum(1 for ps in per_state
                   if (ps["source_best_dr"] or 0) > (ps["target_best_dr"] or 0))
    tgt_wins = sum(1 for ps in per_state
                   if (ps["target_best_dr"] or 0) > (ps["source_best_dr"] or 0))
    ties = n_valid - src_wins - tgt_wins
    print(f"\n  Per-state best-ΔR comparison:")
    print(f"    Source wins: {src_wins}/{n_valid} ({src_wins/n_valid:.1%})")
    print(f"    Target wins: {tgt_wins}/{n_valid} ({tgt_wins/n_valid:.1%})")
    print(f"    Ties:        {ties}/{n_valid} ({ties/n_valid:.1%})")

    # ── Verdict ─────────────────────────────────────────────────────────
    src_mean_cos = safe_mean(src_cos)
    tgt_mean_cos = safe_mean(tgt_cos)
    cos_delta = tgt_mean_cos - src_mean_cos
    src_imp_frac = safe_mean(src_imp)
    tgt_imp_frac = safe_mean(tgt_imp)

    print(f"\n  {'='*60}")
    if cos_delta > 0.1 and tgt_imp_frac >= 0.85:
        print(f"  Verdict: TARGET >> SOURCE")
        print(f"  Target critic substantially improves gradient direction.")
        print(f"  Architecture confirmed. Problem = reward-free target value acquisition.")
    elif abs(cos_delta) < 0.05:
        print(f"  Verdict: SOURCE ≈ TARGET")
        print(f"  Source critic already captures task-relevant direction well.")
        print(f"  Proceed directly to closed-loop one-step value-gradient controller.")
    elif cos_delta < -0.05:
        print(f"  Verdict: TARGET < SOURCE")
        print(f"  Target critic gradient geometry is worse than source.")
        print(f"  Possible cause: critic overfit, poor gradient geometry despite good values.")
    else:
        print(f"  Verdict: MARGINAL — target shows some improvement but below clear threshold.")

    # ── Save ─────────────────────────────────────────────────────────────
    def sanitize(v):
        if isinstance(v, float) and np.isnan(v):
            return None
        if isinstance(v, bool):
            return bool(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        return v

    def sanitize_per_state(ps):
        out = {}
        for k, v in ps.items():
            if isinstance(v, dict):
                out[k] = {str(ak): sanitize(av) for ak, av in v.items()}
            elif isinstance(v, float) and np.isnan(v):
                out[k] = None
            elif isinstance(v, bool):
                out[k] = bool(v)
            elif isinstance(v, (np.integer,)):
                out[k] = int(v)
            elif isinstance(v, (np.floating,)):
                out[k] = float(v)
            elif isinstance(v, (str, int, type(None))):
                out[k] = v
            else:
                out[k] = str(v)
        return out

    summary = {
        "n_states": n_valid,
        "n_stance": len(stance_idx),
        "n_flight": len(flight_idx),
        "source": {
            "mean_cosine": sanitize(safe_mean(src_cos)),
            "positive_frac": sanitize(safe_frac([c > 0 for c in src_cos])),
            "improved_frac": sanitize(src_imp_frac),
            "mean_best_dr": sanitize(safe_mean(src_dr)),
            "median_best_dr": sanitize(float(np.median(src_dr)) if src_dr else float('nan')),
            "termination_frac": sanitize(safe_mean(src_term)),
        },
        "target": {
            "mean_cosine": sanitize(safe_mean(tgt_cos)),
            "positive_frac": sanitize(safe_frac([c > 0 for c in tgt_cos])),
            "improved_frac": sanitize(tgt_imp_frac),
            "mean_best_dr": sanitize(safe_mean(tgt_dr)),
            "median_best_dr": sanitize(float(np.median(tgt_dr)) if tgt_dr else float('nan')),
            "termination_frac": sanitize(safe_mean(tgt_term)),
        },
        "cosine_delta": sanitize(float(cos_delta)),
        "source_wins": src_wins,
        "target_wins": tgt_wins,
        "ties": ties,
        "verdict": ("TARGET_GT_SOURCE" if cos_delta > 0.1 and tgt_imp_frac >= 0.85
                    else "SOURCE_APPROX_TARGET" if abs(cos_delta) < 0.05
                    else "TARGET_LT_SOURCE" if cos_delta < -0.05
                    else "MARGINAL"),
        "critic_info": {
            "n_train_transitions": n_transitions,
            "avg_ep_len": float(avg_ep_len),
            "mc_return_mean": float(returns_mc.mean()),
            "mc_return_std": float(returns_mc.std()),
            "v_mean": float(v_mean),
            "v_std": float(v_std),
        },
        "per_state": [sanitize_per_state(ps) for ps in per_state],
    }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.json_out, "w"), indent=2)
    print(f"\n  Results saved to {args.json_out}")


if __name__ == "__main__":
    main()
