"""Phase 0A: One-step combined gradient audit.

Audit question:
  Can B_KAN^T * grad(V_source) provide a useful action improvement direction?

If YES (>0.3 cos, >65% positive, >55% real improvement):
  -> directly use source critic for one-step value-gradient control.
If NO:
  -> proceed to Phase 0B (oracle target critic diagnostic).

Key insight from Phase 0/1/2:
  - B_t = dF/da is reliable (cos=0.95 with true dynamics)
  - A_t chain collapses (multi-step return gradient cos=-0.81)
  - This script tests whether B_t alone, combined with external value gradient,
    can provide a useful one-step action improvement direction.
"""

from __future__ import annotations

import argparse, sys, os, json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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
# Helper: compute reward and its action gradient
# ═══════════════════════════════════════════════════════════════════════

def hopper_reward_np(obs, action, prev_obs=None):
    """Exact Hopper-v5 reward (numpy, for true env evaluation)."""
    fwd = obs[5] if prev_obs is None else (obs[0] - prev_obs[0]) / DT
    z_ok = Z_LO < obs[1] < Z_HI
    a_ok = ANG_LO < obs[2] < ANG_HI
    healthy = 1.0 if (z_ok and a_ok) else 0.0
    ctrl = CTRL_COST_WEIGHT * float(np.sum(action ** 2))
    return FORWARD_WEIGHT * fwd + HEALTHY_REWARD * healthy - ctrl


def hopper_reward_smooth_torch(obs, action, prev_obs=None):
    """Differentiable Hopper reward surrogate (for gradient computation)."""
    if prev_obs is None:
        fwd = obs[..., 5]
    else:
        fwd = (obs[..., 0] - prev_obs[..., 0]) / DT
    z, a = obs[..., 1], obs[..., 2]
    if Z_HI == float('inf'):
        h_z = torch.sigmoid((z - Z_LO) / 0.02)
    else:
        h_z = torch.sigmoid((z - Z_LO) / 0.02) * torch.sigmoid((Z_HI - z) / 0.02)
    h_a = torch.sigmoid((a - ANG_LO) / 0.05) * torch.sigmoid((ANG_HI - a) / 0.05)
    ctrl = CTRL_COST_WEIGHT * (action ** 2).sum(dim=-1)
    return FORWARD_WEIGHT * fwd + HEALTHY_REWARD * h_z * h_a - ctrl


def grad_r_wrt_a(action_torch):
    """Analytic gradient of Hopper reward w.r.t action: only ctrl_cost depends on a."""
    return -2.0 * CTRL_COST_WEIGHT * action_torch


def is_flight(state):
    """Heuristic: flight if body high and foot contact forces near zero."""
    z = state[1]
    foot_contact = np.abs(state[-2:]).sum()
    return z > 0.85 and foot_contact < 0.5


# ═══════════════════════════════════════════════════════════════════════
# Gradient computation
# ═══════════════════════════════════════════════════════════════════════

def compute_source_value_gradient(states, source_policy):
    """Compute grad_s V_source(s) via autograd through source PPO critic.

    Args:
        states: (N, s_dim) torch tensor
        source_policy: FrozenSourcePolicy instance

    Returns:
        grad_v: (N, s_dim) gradient of V_source w.r.t state
        values: (N,) value estimates
    """
    mean = source_policy.mean
    var = source_policy.variance
    s_requires_grad = states.detach().clone().requires_grad_(True)
    s_norm = ((s_requires_grad - mean) / (var + 1e-8).sqrt()).clamp(-10, 10)

    features = source_policy.model.policy.features_extractor(s_norm)
    latent = source_policy.model.policy.mlp_extractor(features)
    if isinstance(latent, tuple):
        _, latent_vf = latent
    else:
        latent_vf = latent
    values = source_policy.model.policy.value_net(latent_vf).squeeze(-1)

    grad_v = torch.autograd.grad(
        values.sum(), s_requires_grad, create_graph=False, retain_graph=False,
    )[0].detach()

    return grad_v, values.detach()


def compute_combined_gradient(s, a, basis, target_ctx, source_policy):
    """Compute one-step combined gradient g = dr/da + B^T * grad(V_source)(s').

    Uses KAN-predicted next state s'_KAN for the value gradient input.
    This matches what a real controller using KAN would compute.

    Args:
        s: (1, s_dim) current state
        a: (1, a_dim) current action
        basis: KAN basis dictionary
        target_ctx: fitted target KAN context
        source_policy: FrozenSourcePolicy (for critic)

    Returns:
        g_combined: (a_dim,) combined gradient
        dr_da: (a_dim,) reward gradient w.r.t action
        bt_gradv: (a_dim,) B^T * grad(V) component
        s_next_kan: (1, s_dim) KAN-predicted next state
        grad_v: (1, s_dim) value gradient at s_next_kan
        v_next: scalar value at s_next_kan
    """
    # KAN prediction of next state
    effect = target_ctx.acceleration(basis, s, a)
    s_next = s + effect

    # Value gradient at KAN-predicted next state
    grad_v, v_next = compute_source_value_gradient(s_next, source_policy)

    # B_t = dF/da = G(s) from KAN
    _, gain = target_ctx.drift_and_gain(basis, s)
    B = gain.squeeze(0)  # (s_dim, a_dim)

    # dr/da (only ctrl_cost depends directly on a)
    dr_da = grad_r_wrt_a(a.squeeze(0))  # (a_dim,)

    # B^T * grad(V) component
    bt_gradv = B.T @ grad_v.squeeze(0)  # (a_dim,)

    # Combined: g = dr/da + B^T * grad(V)
    g_combined = dr_da + bt_gradv  # (a_dim,)

    return g_combined, dr_da, bt_gradv, s_next, grad_v, v_next.item()


def compute_true_gradient_fd(fd_env, qp, qv, a_nominal, eps=0.05):
    """Compute true one-step return gradient via finite difference.

    Gradient of r(s,a,s') + gamma*V_true(s') w.r.t a, where V_true is estimated
    via a long rollout from the environment (not from a learned critic).

    We use the REAL env for both the immediate reward and the next state,
    then estimate terminal value via extended true rollout.

    Args:
        fd_env: friction env with set_state capability
        qp, qv: MuJoCo state to reset to
        a_nominal: (a_dim,) nominal action (transport)
        eps: finite difference step size

    Returns:
        g_true: (a_dim,) estimated true gradient
        baseline_return: scalar H-step return from a_nominal
    """
    a_dim = len(a_nominal)
    H = 4  # short horizon for terminal value estimation

    # Baseline: execute a_nominal for H steps, measure true return
    fd_env.unwrapped.set_state(qp, qv)
    baseline_return = 0.0
    prev = None
    alive = True
    for _ in range(H):
        obs, r, t, tr, _ = fd_env.step(a_nominal)
        baseline_return += float(r)
        prev = obs.copy()
        if t or tr:
            alive = False
            break

    if not alive:
        # Can't compute meaningful gradient if baseline dies
        return np.zeros(a_dim), baseline_return, False

    # FD per action dimension
    g_true = np.zeros(a_dim)
    for dim in range(a_dim):
        for sign in [+1, -1]:
            da = np.zeros(a_dim)
            da[dim] = sign * eps
            a_perturbed = np.clip(a_nominal + da, -1, 1)

            fd_env.unwrapped.set_state(qp, qv)
            total_r = 0.0
            alive_pert = True
            prev_p = None
            for _ in range(H):
                obs_p, r_p, t_p, tr_p, _ = fd_env.step(a_perturbed)
                total_r += float(r_p)
                prev_p = obs_p.copy()
                if t_p or tr_p:
                    alive_pert = False
                    break

            if alive_pert:
                g_true[dim] += sign * total_r / (2.0 * eps)
            else:
                # Perturbed action caused death → gradient points away from death
                # Use a large negative return to indicate bad direction
                g_true[dim] += sign * (-10.0) / (2.0 * eps)

    return g_true, baseline_return, True


# ═══════════════════════════════════════════════════════════════════════
# Main audit
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--n-states", type=int, default=80,
                       help="Number of states to audit")
    parser.add_argument("--fd-eps", type=float, default=0.05,
                       help="FD step size for true gradient")
    parser.add_argument("--line-search-alphas", type=str, default="-0.1,-0.05,-0.02,0.02,0.05,0.1,0.2",
                       help="Comma-separated alpha values for line search")
    parser.add_argument("--json-out", default="results/phase0a_combined_gradient.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    shift = SHIFTS["friction_070"]
    alphas = [float(x) for x in args.line_search_alphas.split(",")]

    print("=" * 72)
    print("Phase 0A: One-Step Combined Gradient Audit")
    print(f"  n_states={args.n_states}, fd_eps={args.fd_eps}")
    print(f"  line_search_alphas={alphas}")
    print("=" * 72)

    # ── Load components ─────────────────────────────────────────────────
    print("\n[1/4] Loading components...", flush=True)

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

    # ── Collect states from friction env using Transport ─────────────────
    print("\n[2/4] Collecting states with Transport on friction_070...", flush=True)

    collect_env = make_shifted_env(shift, args.seed, "hopper")()
    all_records = []
    episodes = 0
    while len(all_records) < args.n_states * 8:  # oversample 8x, filter for alive
        obs, _ = collect_env.reset(seed=args.seed + episodes * 100)
        step_in_ep = 0
        while True:
            s_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            nominal = source_policy.action(s_t)
            s_eff = source_context.acceleration(basis, s_t, nominal)
            a_tr = target_ctx.transport_action(
                basis, s_t, desired_effect=s_eff,
                nominal_action=nominal, regularization=1e-2,
            ).clamp(-1, 1).squeeze(0).cpu().numpy()

            qp = collect_env.unwrapped.data.qpos.copy()
            qv = collect_env.unwrapped.data.qvel.copy()
            all_records.append({
                "qp": qp, "qv": qv, "obs": obs.copy(), "a_tr": a_tr,
                "flight": is_flight(obs), "step": step_in_ep,
            })
            obs, _, t, tr, _ = collect_env.step(a_tr)
            step_in_ep += 1
            if t or tr or step_in_ep > 200:
                break
        episodes += 1

    collect_env.close()
    print(f"  Collected {len(all_records)} states from {episodes} episodes")

    # Pre-filter: test baseline H-step survival on each candidate state
    print("  Pre-filtering for baseline survival (H=4)...", flush=True)
    fd_filter = make_shifted_env(shift, args.seed + 999, "hopper")()
    alive_records = []
    for rec in all_records[:args.n_states * 4]:  # limit pre-filtering
        fd_filter.unwrapped.set_state(rec["qp"], rec["qv"])
        alive = True
        prev = None
        for _ in range(4):
            obs_f, _, t_f, tr_f, _ = fd_filter.step(rec["a_tr"])
            if t_f or tr_f:
                alive = False
                break
            prev = obs_f.copy()
        if alive:
            alive_records.append(rec)
    fd_filter.close()
    print(f"  Alive after baseline H=4: {len(alive_records)}/{min(len(all_records), args.n_states * 4)} "
          f"({100*len(alive_records)/max(1,min(len(all_records), args.n_states*4)):.1f}%)")

    # Sample from alive records
    if len(alive_records) < 20:
        print(f"  WARNING: Very few alive states ({len(alive_records)}). "
              f"Transport may be near termination boundary in friction env.")
        # Fall back to using all records but note the issue
        records = all_records[:args.n_states]
        print(f"  Falling back to {len(records)} unfiltered states.")
    else:
        n_sample = min(args.n_states, len(alive_records))
        idx = np.linspace(0, len(alive_records) - 1, n_sample, dtype=int)
        records = [alive_records[i] for i in idx]
        print(f"  Selected {len(records)} alive states for audit.")

    # ── Run audit ────────────────────────────────────────────────────────
    print(f"\n[3/4] Auditing {args.n_states} states...", flush=True)

    fd_env = make_shifted_env(shift, args.seed + 999, "hopper")()
    a_dim = basis.action_dim

    results = {
        "cosines": [],
        "sign_agree_top": [],
        "combined_norm": [],
        "true_norm": [],
        "drda_norm": [],           # |∂r/∂a| component
        "bt_gradv_norm": [],       # |Bᵀ∇V| component
        "drda_cosine": [],         # cos(∂r/∂a, g_true)
        "bt_gradv_cosine": [],     # cos(Bᵀ∇V, g_true)
        "grad_v_norm": [],         # |∇V_source|
        "v_next": [],              # V_source(s'_KAN)
        "line_search": {str(alpha): [] for alpha in alphas},
        "baseline_return": [],
        "stance_cosines": [],
        "flight_cosines": [],
        "stance_sign": [],
        "flight_sign": [],
        "alive": [],
    }

    for i, rec in enumerate(records):
        s_t = torch.as_tensor(rec["obs"], device=device, dtype=torch.float32).unsqueeze(0)
        a_t = torch.as_tensor(rec["a_tr"], device=device, dtype=torch.float32).unsqueeze(0)
        a_np = rec["a_tr"]
        qp, qv = rec["qp"], rec["qv"]

        # --- Compute combined (KAN) gradient ---
        g_combined, dr_da, bt_gradv, s_next_kan, grad_v, v_next = compute_combined_gradient(
            s_t, a_t, basis, target_ctx, source_policy,
        )
        g_c = g_combined.cpu().numpy()
        dr_da_np = dr_da.cpu().numpy()
        bt_gradv_np = bt_gradv.cpu().numpy()
        grad_v_np = grad_v.squeeze(0).cpu().numpy()

        # --- Compute true gradient via FD ---
        g_true, baseline_r, alive = compute_true_gradient_fd(
            fd_env, qp, qv, a_np, eps=args.fd_eps,
        )
        results["baseline_return"].append(float(baseline_r))
        results["alive"].append(alive)

        if not alive or np.linalg.norm(g_true) < 1e-8:
            # Skip cosine for dead/already-optimal states
            results["cosines"].append(float('nan'))
            results["sign_agree_top"].append(float('nan'))
            results["stance_cosines"].append(float('nan'))
            results["flight_cosines"].append(float('nan'))
            results["stance_sign"].append(float('nan'))
            results["flight_sign"].append(float('nan'))
            results["combined_norm"].append(float(np.linalg.norm(g_c)))
            results["true_norm"].append(float(np.linalg.norm(g_true)))
            for alpha in alphas:
                results["line_search"][str(alpha)].append(float('nan'))
            continue

        # --- Cosine similarity ---
        cos = np.dot(g_c, g_true) / (
            np.linalg.norm(g_c) * np.linalg.norm(g_true) + 1e-10
        )
        results["cosines"].append(float(cos))

        # --- Sign agreement on top-2 dominant action dimensions ---
        top_dims = np.argsort(np.abs(g_true))[-2:]
        sign_match = np.mean(np.sign(g_c[top_dims]) == np.sign(g_true[top_dims]))
        results["sign_agree_top"].append(float(sign_match))

        results["combined_norm"].append(float(np.linalg.norm(g_c)))
        results["true_norm"].append(float(np.linalg.norm(g_true)))

        # Decomposition: dr/da component vs B^T*grad(V) component
        results["drda_norm"].append(float(np.linalg.norm(dr_da_np)))
        results["bt_gradv_norm"].append(float(np.linalg.norm(bt_gradv_np)))
        results["grad_v_norm"].append(float(np.linalg.norm(grad_v_np)))
        results["v_next"].append(float(v_next))

        if np.linalg.norm(dr_da_np) > 1e-10:
            drda_cos = float(np.dot(dr_da_np, g_true) / (np.linalg.norm(dr_da_np) * np.linalg.norm(g_true) + 1e-10))
        else:
            drda_cos = float('nan')
        results["drda_cosine"].append(drda_cos)

        if np.linalg.norm(bt_gradv_np) > 1e-10:
            bt_cos = float(np.dot(bt_gradv_np, g_true) / (np.linalg.norm(bt_gradv_np) * np.linalg.norm(g_true) + 1e-10))
        else:
            bt_cos = float('nan')
        results["bt_gradv_cosine"].append(bt_cos)

        if rec["flight"]:
            results["flight_cosines"].append(float(cos))
            results["flight_sign"].append(float(sign_match))
            results["stance_cosines"].append(float('nan'))
            results["stance_sign"].append(float('nan'))
        else:
            results["stance_cosines"].append(float(cos))
            results["stance_sign"].append(float(sign_match))
            results["flight_cosines"].append(float('nan'))
            results["flight_sign"].append(float('nan'))

        # --- Line search: test real improvement along g_combined ---
        norm = np.linalg.norm(g_c)
        if norm > 1e-8:
            direction = g_c / norm
        else:
            direction = g_c

        for alpha in alphas:
            da = alpha * direction
            a_test = np.clip(a_np + da, -1, 1)

            fd_env.unwrapped.set_state(qp, qv)
            total_r = 0.0
            alive_ls = True
            prev_ls = None
            for _ in range(4):  # H=4 as in Phase 1/2
                obs_ls, r_ls, t_ls, tr_ls, _ = fd_env.step(a_test)
                total_r += float(r_ls)
                prev_ls = obs_ls.copy()
                if t_ls or tr_ls:
                    alive_ls = False
                    break

            if alive_ls:
                results["line_search"][str(alpha)].append(float(total_r - baseline_r))
            else:
                results["line_search"][str(alpha)].append(float('nan'))

        if (i + 1) % 20 == 0:
            valid = [c for c in results["cosines"] if not np.isnan(c)]
            mean_cos = np.mean(valid) if valid else float('nan')
            print(f"    {i+1}/{args.n_states} done, running mean cosine={mean_cos:.4f}",
                  flush=True)

    fd_env.close()

    # ── Report ───────────────────────────────────────────────────────────
    print(f"\n[4/4] Results", flush=True)
    print("=" * 72)

    valid_cos = [c for c in results["cosines"] if not np.isnan(c)]
    valid_sign = [s for s in results["sign_agree_top"] if not np.isnan(s)]
    stance_cos = [c for c in results["stance_cosines"] if not np.isnan(c)]
    flight_cos = [c for c in results["flight_cosines"] if not np.isnan(c)]
    stance_sign = [s for s in results["stance_sign"] if not np.isnan(s)]
    flight_sign = [s for s in results["flight_sign"] if not np.isnan(s)]

    n_valid = len(valid_cos)
    n_alive = sum(results["alive"])
    n_stance = len(stance_cos)
    n_flight = len(flight_cos)

    print(f"\n  States: {args.n_states} total, {n_alive} alive at baseline, "
          f"{n_valid} with valid gradient")
    print(f"  Stance: {n_stance}, Flight: {n_flight}")

    mean_cos = np.mean(valid_cos) if valid_cos else float('nan')
    mean_sign = np.mean(valid_sign) if valid_sign else float('nan')
    pos_fraction = np.mean(np.array(valid_cos) > 0) if valid_cos else float('nan')
    pos_weak = np.mean(np.array(valid_cos) > 0.1) if valid_cos else float('nan')

    print(f"\n  {'Metric':30s} {'Overall':>10s} {'Stance':>10s} {'Flight':>10s}")
    print(f"  {'-'*60}")
    print(f"  {'Cosine similarity':30s} {mean_cos:10.4f} "
          f"{np.mean(stance_cos) if stance_cos else float('nan'):10.4f} "
          f"{np.mean(flight_cos) if flight_cos else float('nan'):10.4f}")
    print(f"  {'Sign agreement (top-2)':30s} {mean_sign:10.4f} "
          f"{np.mean(stance_sign) if stance_sign else float('nan'):10.4f} "
          f"{np.mean(flight_sign) if flight_sign else float('nan'):10.4f}")
    print(f"  {'Positive cosine frac':30s} {pos_fraction:10.3f}")
    print(f"  {'Cosine > 0.1 frac':30s} {pos_weak:10.3f}")
    print(f"  {'Mean |g_combined|':30s} {np.mean(results['combined_norm']):10.4f}")
    print(f"  {'Mean |g_true|':30s} {np.mean([x for x in results['true_norm'] if x > 1e-8]):10.4f}")

    # Gradient decomposition
    print(f"\n  Gradient Decomposition (for valid states):")
    drda_valid = [x for x in results["drda_norm"] if x > 1e-10]
    bt_valid = [x for x in results["bt_gradv_norm"] if x > 1e-10]
    gradv_valid = [x for x in results["grad_v_norm"] if x > 1e-10]
    drda_cos_valid = [x for x in results["drda_cosine"] if not np.isnan(x)]
    bt_cos_valid = [x for x in results["bt_gradv_cosine"] if not np.isnan(x)]
    v_next_valid = [x for x in results["v_next"] if not np.isnan(x)]

    print(f"  {'Component':30s} {'|grad|':>10s} {'cos w/ g_true':>15s}")
    print(f"  {'-'*57}")
    print(f"  {'dr/da (ctrl_cost only)':30s} "
          f"{np.mean(drda_valid) if drda_valid else 0:10.6f} "
          f"{np.mean(drda_cos_valid) if drda_cos_valid else float('nan'):15.4f}")
    print(f"  {'B^T * grad(V_source)':30s} "
          f"{np.mean(bt_valid) if bt_valid else 0:10.6f} "
          f"{np.mean(bt_cos_valid) if bt_cos_valid else float('nan'):15.4f}")
    print(f"  {'|grad(V_source)| (state)':30s} "
          f"{np.mean(gradv_valid) if gradv_valid else 0:10.6f}")
    print(f"  {'V_source(s_next_KAN)':30s} "
          f"{np.mean(v_next_valid) if v_next_valid else float('nan'):10.4f}")
    print(f"  {'|g_combined| / |g_true|':30s} "
          f"{np.mean(results['combined_norm']) / max(np.mean([x for x in results['true_norm'] if x > 1e-8]), 1e-10):10.4f}")
    print(f"  {'|B^T*gradV| / |dr/da|':30s} "
          f"{np.mean(bt_valid) / max(np.mean(drda_valid), 1e-10) if bt_valid and drda_valid else 0:10.4f}")

    # Line search results
    print(f"\n  Line Search (H=4 real return improvement vs baseline):")
    print(f"  {'Alpha':>8s}  {'Mean dR':>10s}  {'Median dR':>10s}  {'Improved%':>10s}  {'Alive%':>10s}")
    print(f"  {'-'*60}")
    for alpha in alphas:
        drs = [x for x in results["line_search"][str(alpha)] if not np.isnan(x)]
        if drs:
            improved = np.mean(np.array(drs) > 0)
            alive_frac = len(drs) / args.n_states
            print(f"  {alpha:>+8.3f}  {np.mean(drs):10.4f}  {np.median(drs):10.4f}  "
                  f"{improved:10.3f}  {alive_frac:10.3f}")
        else:
            print(f"  {alpha:>+8.3f}  {'(all nan)':>10s}")

    # Best alpha
    best_alpha = None
    best_improve = -float('inf')
    for alpha in alphas:
        drs = [x for x in results["line_search"][str(alpha)] if not np.isnan(x)]
        if drs and np.mean(drs) > best_improve:
            best_improve = np.mean(drs)
            best_alpha = alpha

    print(f"\n  Best line-search alpha: {best_alpha} (mean improvement={best_improve:+.4f})")

    # ── Verdict ──────────────────────────────────────────────────────────
    print(f"\n  {'='*60}")
    print(f"  Verdict thresholds:")
    print(f"    Mean cosine       > 0.30  : {'PASS' if mean_cos > 0.3 else 'FAIL'} ({mean_cos:.3f})")
    print(f"    Positive fraction > 0.65  : {'PASS' if pos_fraction > 0.65 else 'FAIL'} ({pos_fraction:.3f})")

    best_improve_frac = np.mean(np.array([x for x in results["line_search"][str(best_alpha)]
                                 if not np.isnan(x)]) > 0) if best_alpha else 0
    print(f"    Best LS improved  > 0.55  : {'PASS' if best_improve_frac > 0.55 else 'FAIL'} "
          f"({best_improve_frac:.3f} at alpha={best_alpha})")

    all_pass = (mean_cos > 0.3 and pos_fraction > 0.65 and best_improve_frac > 0.55)

    if all_pass:
        print(f"\n  => PASS: source critic gradient can guide one-step improvement.")
        print(f"     Proceed to Phase 1a: one-step value-gradient controller.")
    elif mean_cos > 0.1 and pos_fraction > 0.5:
        print(f"\n  => MARGINAL: source critic has some signal but below threshold.")
        print(f"     Proceed to Phase 0B: oracle target critic diagnostic.")
    else:
        print(f"\n  => FAIL: source critic gradient is unreliable for action guidance.")
        print(f"     Proceed to Phase 0B: test whether target critic fixes the issue.")

    # ── Save ─────────────────────────────────────────────────────────────
    # Convert nan to None for JSON serialization
    def sanitize(v):
        if isinstance(v, float) and np.isnan(v):
            return None
        return v

    summary = {
        "n_states": args.n_states,
        "n_valid": n_valid,
        "n_alive": n_alive,
        "n_stance": n_stance,
        "n_flight": n_flight,
        "mean_cosine": sanitize(float(mean_cos)) if not np.isnan(mean_cos) else None,
        "mean_sign_agree": sanitize(float(mean_sign)) if not np.isnan(mean_sign) else None,
        "positive_fraction": sanitize(float(pos_fraction)) if not np.isnan(pos_fraction) else None,
        "stance_cosine": sanitize(float(np.mean(stance_cos))) if stance_cos else None,
        "flight_cosine": sanitize(float(np.mean(flight_cos))) if flight_cos else None,
        "best_alpha": best_alpha,
        "best_improvement": sanitize(float(best_improve)),
        "best_improved_fraction": sanitize(float(best_improve_frac)),
        "line_search": {str(k): [sanitize(float(x)) for x in v]
                       for k, v in results["line_search"].items()},
        "verdict": "PASS" if all_pass else ("MARGINAL" if mean_cos > 0.1 else "FAIL"),
        "raw_cosines": [sanitize(float(x)) for x in results["cosines"]],
    }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.json_out, "w"), indent=2)
    print(f"\n  Results saved to {args.json_out}")


if __name__ == "__main__":
    main()
