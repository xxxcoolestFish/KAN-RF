"""Experiment D: Closed-loop adaptive control with KAN world model.

Phases:
  0. Warm-up: collect data with strategy-v2-guided actions (not shooting),
     update model via batched three-factor BP, repeat until model_err drops.
  1. Plan:  shoot through frozen KAN (batch-parallel multi-restart).
            Truncate at convergence → dynamic effective horizon.
  2. Execute+Monitor: step-by-step, check ||s_real - s_pred||.
  3. Online Update: three-factor BP every step.
     Replan when deviation > hard threshold.

Usage:
  python exp_D_closed_loop.py --trials 5 --device mps --warmup-episodes 3
"""
import sys, time, argparse, os
import torch
import gymnasium as gym
import numpy as np
from kanrf import KAN
from online_learning_v2 import ThreeFactorUpdater, compute_training_stats
from strategy_v2 import compute_gap, desired_velocity, strategy_mode

G = 10.0; PI_2 = np.pi / 2


# ─── Helpers ────────────────────────────────────────────────────────────────

def _normalize_state(s, device=None):
    """Raw numpy s → normalized torch tensor (1,3)."""
    t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
    t[0, 2] /= 8.0
    return t.to(device) if device is not None else t


def _normalize_action(a):
    """Raw float a → normalized torch tensor (1,1)."""
    return torch.tensor([[a / 2.0]], dtype=torch.float32)


def deviation(s_real, s_pred_norm, device=None):
    """||real - predicted|| in normalized state space."""
    s_n = _normalize_state(s_real, device=device)
    return (s_n - s_pred_norm.to(s_n.device)).norm().item()


# ─── Phase 0: Warm-up ───────────────────────────────────────────────────────

def warmup_model(model, env, updater, device,
                 n_episodes=3, steps_per_ep=60):
    """Collect data with strategy-v2 guidance, update model.

    Returns (model_err_history, updated_updater).
    Strategy v2 provides physically-informed exploration (energy-guided),
    much better than random actions for covering swing-up-relevant regions.
    """
    err_history = []
    all_transitions = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_errs = []

        for step in range(steps_per_ep):
            # Strategy v2 action (energy-guided, no shooting needed)
            gap = compute_gap(obs)
            mode = strategy_mode(gap)
            v_des = desired_velocity(gap, mode)

            # Simple heuristic: action proportional to desired theta_dot change
            # v_des[2] is desired theta_dot change → a ≈ v_des[2] * (2*dt/3)
            # (from pendulum dynamics: thd_dot = 15*sin(theta) + 3*torque)
            a_des = v_des[2] / (3.0 * 0.05)  # rough inverse dynamics
            a = np.clip(a_des, -2.0, 2.0)

            obs_next, _, _, _, _ = env.step([a])

            # Record transition
            s_norm = _normalize_state(obs, device=device)
            a_norm = _normalize_action(a).to(device)
            s_next_norm = _normalize_state(obs_next, device=device)

            # Model error before update
            with torch.no_grad():
                x = torch.cat([s_norm, a_norm], dim=-1)
                s_pred = model(x)
            err = (s_pred - s_next_norm).norm().item()
            ep_errs.append(err)

            # Online update
            updater.update(s_norm, a_norm, s_next_norm)

            obs = obs_next

        ep_mean_err = np.mean(ep_errs)
        err_history.append(ep_mean_err)
        print(f"  Warm-up ep {ep+1}/{n_episodes}: "
              f"mean_model_err={ep_mean_err:.4f}  steps={len(ep_errs)}")

    return err_history, updater


# ─── Exploration Burst (on model failure) ───────────────────────────────────

def exploration_burst(model, env, updater, obs_start, device,
                      n_steps=12, burst_eta_mult=10, n_extra_passes=3):
    """When deviation is large, collect data locally and learn intensively.

    Uses random actions (uniform[-2,2]) to explore the local action space
    from the current OOD state. Strategy-v2 would stay near well-trained
    regions; random actions maximize coverage of the model's blind spot.

    Returns (final_obs, n_updates_done).
    """
    original_eta0 = updater.eta0
    updater.eta0 = original_eta0 * burst_eta_mult

    transitions = []
    obs = obs_start
    n_updates = 0

    for step in range(n_steps):
        # Random action sweep: cover the full torque range
        # Mix: half uniform random, half structured sweep
        if step % 2 == 0:
            a = np.random.uniform(-2.0, 2.0)
        else:
            # Sweep through magnitudes: small → medium → large
            frac = step / n_steps
            mag = 0.3 + 1.7 * frac  # 0.3 to 2.0
            sign = 1.0 if step % 4 == 1 else -1.0
            a = sign * mag

        obs_next, _, _, _, _ = env.step([a])
        transitions.append((obs.copy(), a, obs_next.copy()))

        s_norm = _normalize_state(obs, device=device)
        a_norm = _normalize_action(a).to(device)
        s_next_norm = _normalize_state(obs_next, device=device)
        updater.update(s_norm, a_norm, s_next_norm)
        n_updates += 1

        obs = obs_next

    # Extra passes: re-learn collected transitions with boosted eta
    for _ in range(n_extra_passes):
        for (s, a, s_next) in transitions:
            s_norm = _normalize_state(s, device=device)
            a_norm = _normalize_action(a).to(device)
            s_next_norm = _normalize_state(s_next, device=device)
            updater.update(s_norm, a_norm, s_next_norm)
            n_updates += 1

    updater.eta0 = original_eta0
    return obs, n_updates


# ─── Phase 1-3: Shoot ──────────────────────────────────────────────────────

def shoot_kan_batch(model, s0, s_target, horizon=30, n_iters=300, lr=0.1,
                    lambda_ctrl=0.001, tol=1e-4, n_restarts=1, device=None):
    """Batch-parallel multi-restart shooting through frozen KAN.

    Multiple restarts are optimized simultaneously in a batch dimension
    for better GPU utilization. Returns the best plan across restarts.

    Args:
        n_restarts: number of parallel random initializations
    Returns:
        actions:            (H,1) raw torques in [-2,2] (CPU) — best restart
        predicted_states:   list of (1,3) normalized tensors (CPU) — best restart
        diag:               dict
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    B = n_restarts
    s0_n = s0.clone().to(device); s0_n[:, 2] /= 8.0
    st_n = s_target.clone().to(device); st_n[:, 2] /= 8.0

    # Batch-parallel actions: (horizon, B)
    a_n = torch.zeros(horizon, B, device=device)
    torch.nn.init.uniform_(a_n, a=-0.3, b=0.3)
    a_n.requires_grad_(True)

    opt = torch.optim.Adam([a_n], lr=lr)
    iters_used, converged = n_iters, False

    for step in range(n_iters):
        opt.zero_grad()

        # Batch forward: s expands to (B, 3)
        s = s0_n.expand(B, -1).clone()  # (B, 3)

        for h in range(horizon):
            x = torch.cat([s, a_n[h:h + 1].t()], dim=-1)  # (B, 4)
            s = model(x)
            nrm = s[:, :2].norm(dim=-1, keepdim=True).clamp(min=1e-6)
            s = torch.cat([s[:, :2] / nrm, s[:, 2:]], dim=-1)

        # Per-restart terminal loss
        loss_term = ((s - st_n) ** 2).sum(dim=-1)  # (B,)
        loss_ctrl = (a_n ** 2).sum(dim=0)           # (B,)
        loss = (loss_term + lambda_ctrl * loss_ctrl).sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            a_n.clamp_(-1.0, 1.0)

        min_loss = loss_term.min().item()
        if min_loss < tol:
            converged = True
            iters_used = step + 1
            break

    # Select best restart
    with torch.no_grad():
        s_check = s0_n.expand(B, -1).clone()
        for h in range(horizon):
            x = torch.cat([s_check, a_n[h:h + 1].t()], dim=-1)
            s_check = model(x)
            nrm = s_check[:, :2].norm(dim=-1, keepdim=True).clamp(min=1e-6)
            s_check = torch.cat([s_check[:, :2] / nrm, s_check[:, 2:]], dim=-1)
        final_losses = ((s_check - st_n) ** 2).sum(dim=-1)  # (B,)
        best_idx = final_losses.argmin().item()

    best_a_n = a_n[:, best_idx:best_idx + 1].detach()  # (H, 1)

    # Best restart: forward rollout for predicted states
    with torch.no_grad():
        predicted_states = [s0_n.clone()]
        s = s0_n.clone()
        for h in range(horizon):
            x = torch.cat([s, best_a_n[h:h + 1]], dim=-1)
            s = model(x)
            nrm = s[:, :2].norm(dim=-1, keepdim=True).clamp(min=1e-6)
            s = torch.cat([s[:, :2] / nrm, s[:, 2:]], dim=-1)
            predicted_states.append(s.clone())

    for p in model.parameters():
        p.requires_grad = True

    actions_cpu = (best_a_n * 2.0).cpu()
    states_cpu = [ps.cpu() for ps in predicted_states]

    return (actions_cpu, states_cpu,
            {'iters': iters_used, 'conv': converged,
             'term_loss': final_losses[best_idx].item()})


def truncate_at_convergence(actions, predicted_states, s_target, tol=0.005):
    """Find earliest step h where predicted s_h ≈ s_target."""
    st_n = s_target.clone(); st_n[:, 2] /= 8.0
    for h in range(1, len(predicted_states)):
        loss = ((predicted_states[h] - st_n) ** 2).sum().item()
        if loss < tol:
            return actions[:h], predicted_states[:h + 1], h
    return actions, predicted_states, len(actions)


# ─── Main trial loop ────────────────────────────────────────────────────────

def run_trial(model, env, s_goal, updater, stats, trial_seed,
              H_max=20, total_steps=60, n_iters=200, n_restarts=3,
              dev_threshold_hard=None, device=None):
    """Closed-loop control: plan → execute+monitor → update → replan."""
    if dev_threshold_hard is None:
        dev_threshold_hard = 3.0 * stats['sigma_train']

    obs, _ = env.reset(seed=trial_seed)
    step_count = 0
    plan_cycles = 0
    update_count = 0
    deviations = []
    model_errors = []
    all_eff_H = []

    while step_count < total_steps:
        s_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        # ── Plan ──
        actions, predicted_states, diag = shoot_kan_batch(
            model, s_t, s_goal, horizon=H_max, n_iters=n_iters,
            n_restarts=n_restarts, device=device)

        # ── Truncate ──
        actions_trunc, pred_states_trunc, eff_H = truncate_at_convergence(
            actions, predicted_states, s_goal)
        all_eff_H.append(eff_H)
        plan_cycles += 1

        # ── Execute + Monitor + Update ──
        replan = False
        for i in range(len(actions_trunc)):
            if step_count >= total_steps:
                break

            a = actions_trunc[i].item()
            s_before = obs.copy()
            obs_next, _, term, trunc, _ = env.step([a])
            step_count += 1

            # Deviation
            s_pred_i1 = pred_states_trunc[i + 1]
            dev = deviation(obs_next, s_pred_i1, device=device)
            deviations.append(dev)

            # Model error (diagnostic)
            s_norm = _normalize_state(s_before, device=device)
            a_norm = _normalize_action(a).to(device)
            with torch.no_grad():
                s_pred = model(torch.cat([s_norm, a_norm], dim=-1))
            pred_raw = s_pred.cpu().clone(); pred_raw[:, 2] *= 8.0
            model_err = np.linalg.norm(pred_raw.squeeze(0).numpy() - obs_next)
            model_errors.append(model_err)

            # Online update (every step)
            s_true_n = _normalize_state(obs_next, device=device)
            updater.update(s_norm, a_norm, s_true_n)
            update_count += 1

            obs = obs_next

            # Goal check
            if abs(np.arctan2(obs[1], obs[0]) - PI_2) < 0.2:
                term = True
                break

            # Hard threshold: exploration burst → learn, then replan
            if dev > dev_threshold_hard:
                obs, n_burst = exploration_burst(
                    model, env, updater, obs, device,
                    n_steps=12, burst_eta_mult=10, n_extra_passes=3)
                update_count += n_burst
                replan = True
                break

            if term or trunc:
                break

        if term or trunc:
            break

    angle_final = np.arctan2(obs[1], obs[0])
    angle_err = abs(angle_final - PI_2)

    return {
        'success': angle_err < 0.2,
        'angle_err': angle_err,
        'steps_taken': step_count,
        'plan_cycles': plan_cycles,
        'update_count': update_count,
        'mean_eff_H': float(np.mean(all_eff_H)) if all_eff_H else 0,
        'max_deviation': float(max(deviations)) if deviations else 0,
        'mean_deviation': float(np.mean(deviations)) if deviations else 0,
        'mean_model_err': float(np.mean(model_errors)) if model_errors else 0,
        'n_hard_replans': sum(1 for d in deviations if d > dev_threshold_hard),
    }


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='kan_pendulum_model_v4.pt')
    parser.add_argument('--data', type=str, default='pendulum_data_v4.pt')
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--h-max', type=int, default=20)
    parser.add_argument('--n-iters', type=int, default=200)
    parser.add_argument('--n-restarts', type=int, default=3)
    parser.add_argument('--eta0', type=float, default=1e-3)
    parser.add_argument('--warmup-episodes', type=int, default=3)
    parser.add_argument('--device', type=str, default='mps',
                       choices=['cpu', 'mps', 'cuda'])
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # ── Device ──
    if args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # ── Load model ──
    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}"); sys.exit(1)
    model = KAN([4, 12, 3], grid_size=5, spline_order=3)
    model.load_state_dict(torch.load(args.model, weights_only=True))
    model = model.to(device)

    # ── Load data & stats ──
    if not os.path.exists(args.data):
        print(f"Data not found: {args.data}"); sys.exit(1)
    data = torch.load(args.data, weights_only=True)
    if isinstance(data, tuple) and len(data) == 2:
        x_train, y_train = data
    else:
        print(f"Unknown data format"); sys.exit(1)

    n_stats = min(len(x_train), 5000)
    idx = torch.randperm(len(x_train))[:n_stats]
    stats = compute_training_stats(model, x_train[idx].to(device),
                                   y_train[idx].to(device))

    updater = ThreeFactorUpdater(model, stats, eta0=args.eta0)

    # ── Phase 0: Warm-up ──
    env = gym.make("Pendulum-v1")
    print("=" * 80)
    print("Phase 0: Model Warm-up (strategy-v2 guided exploration)")
    print(f"  Episodes: {args.warmup_episodes} × 60 steps")
    print(f"  Initial sigma_train = {stats['sigma_train']:.4f}")
    print("=" * 80)

    warmup_start = time.time()
    err_history, updater = warmup_model(
        model, env, updater, device,
        n_episodes=args.warmup_episodes, steps_per_ep=60)
    print(f"  Warm-up time: {time.time() - warmup_start:.0f}s")
    print(f"  Model err: {err_history[0]:.4f} → {err_history[-1]:.4f}")

    # ── Phases 1-3: Closed-loop control ──
    s_goal = torch.tensor([[0.0, 1.0, 0.0]])
    dev_threshold_hard = 3.0 * stats['sigma_train']

    print(f"\n{'=' * 80}")
    print(f"Experiment D: Closed-Loop Adaptive KAN Control")
    print(f"Model: {args.model}  |  Device: {device}")
    print(f"Trials: {args.trials}  |  seed={args.seed}")
    print(f"H_max={args.h_max}  |  n_iters={args.n_iters}  |  restarts={args.n_restarts}")
    print(f"sigma_train={stats['sigma_train']:.4f}  |  hard_thresh={dev_threshold_hard:.4f}")
    print(f"eta0={args.eta0}  |  warmup_eps={args.warmup_episodes}")
    print("=" * 80)

    header = (f"\n{'Trial':>4s}  {'|dth0|':>7s}  {'|dth_f|':>9s}  {'R':>5s}  "
              f"{'eff_H':>6s}  {'mod_err':>8s}  {'max_dev':>8s}  "
              f"{'cycl':>5s}  {'repl':>5s}  {'upd':>5s}")
    sep = (f"{'─'*4}  {'─'*7}  {'─'*9}  {'─'*5}  {'─'*6}  {'─'*8}  "
           f"{'─'*8}  {'─'*5}  {'─'*5}  {'─'*5}")
    print(header)
    print(sep)

    t_start = time.time()
    ok = 0
    all_results = []

    for t in range(args.trials):
        trial_seed = args.seed + t * 100
        obs0, _ = env.reset(seed=trial_seed)
        init_err = abs(np.arctan2(obs0[1], obs0[0]) - PI_2)

        r = run_trial(model, env, s_goal, updater, stats, trial_seed,
                      H_max=args.h_max, total_steps=60, n_iters=args.n_iters,
                      n_restarts=args.n_restarts,
                      dev_threshold_hard=dev_threshold_hard, device=device)

        if r['success']: ok += 1
        all_results.append(r)

        elapsed = time.time() - t_start
        n_rem = args.trials - t - 1
        eta = elapsed / (t + 1) * n_rem if t > 0 and n_rem > 0 else 0
        print(f"  {t+1:4d}  {init_err:7.3f}  {r['angle_err']:9.4f}  "
              f"{'Y' if r['success'] else 'N':>5s}  "
              f"{r['mean_eff_H']:5.1f}  {r['mean_model_err']:8.4f}  "
              f"{r['max_deviation']:8.4f}  "
              f"{r['plan_cycles']:5d}  {r['n_hard_replans']:5d}  "
              f"{r['update_count']:5d}  "
              f"[{elapsed:.0f}s ETA {eta:.0f}s]")

    print(f"\n{'=' * 80}")
    print(f"SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Successes:        {ok}/{args.trials}")
    errs = [r['angle_err'] for r in all_results]
    print(f"  Mean |dth_final|: {np.mean(errs):.4f} rad")
    print(f"  Mean model_err:   {np.mean([r['mean_model_err'] for r in all_results]):.4f}")
    print(f"  Mean updates:     {np.mean([r['update_count'] for r in all_results]):.0f}")
    print(f"  Total time:       {time.time() - t_start:.0f}s")

    env.close()


if __name__ == "__main__":
    main()
