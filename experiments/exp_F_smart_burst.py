"""Experiment F: Smart burst — active model verification + targeted correction.

Key improvement over exp_E:
  When deviation detected, instead of random exploration, burst uses the
  model's OWN energy-guided decisions to test itself:
    1. Model predicts: "action a will increase energy by ΔE_pred"
    2. Execute a, observe real ΔE_real
    3. Discrepancy → model was wrong at (s,a) → targeted correction
    4. Repeat at the new state (now in a region the model misunderstands)

  This probes the model at its DECISION BOUNDARY — the most informative
  region — rather than random points.

Usage:
  python exp_F_smart_burst.py --trials 10 --device mps --model kan_pendulum_model_v6.pt
"""
import sys, time, argparse, os
import torch
import gymnasium as gym
import numpy as np
from kanrf import KAN
from control.online_learning_v2 import ThreeFactorUpdater, compute_training_stats

G = 10.0; PI_2 = np.pi / 2; E_DES = G


# ─── Helpers ────────────────────────────────────────────────────────────────

def _normalize_state(s, device=None):
    t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
    t[0, 2] /= 8.0
    return t.to(device) if device is not None else t


def _normalize_action(a, device=None):
    t = torch.tensor([[a / 2.0]], dtype=torch.float32)
    return t.to(device) if device is not None else t


def deviation(s_real, s_pred_norm, device=None):
    s_n = _normalize_state(s_real, device=device)
    return (s_n - s_pred_norm.to(s_n.device)).norm().item()


# ─── Action optimization ────────────────────────────────────────────────────

def find_action_energy(model, s_norm, s_target_norm, n_iters=50, lr=0.05,
                       lambda_ctrl=0.001, device=None):
    """Single-step energy-guided action optimization."""
    s_raw = s_norm.clone(); s_raw[0, 2] *= 8.0
    sin_th = s_raw[0, 1].item(); thd = s_raw[0, 2].item()
    near_upright = abs(s_raw[0, 0].item()) < 0.5 and sin_th > 0 and abs(thd) < 3.0
    w_E, w_pos = 1.0, (3.0 if near_upright else 0.0)

    a_n = torch.zeros(1, 1, device=device)
    torch.nn.init.uniform_(a_n, a=-0.3, b=0.3)
    a_n.requires_grad_(True)
    opt = torch.optim.Adam([a_n], lr=lr)

    for _ in range(n_iters):
        opt.zero_grad()
        x = torch.cat([s_norm, a_n], dim=-1)
        s_pred = model(x)
        E_pred = 0.5 * (s_pred[0, 2] * 8.0) ** 2 + G * s_pred[0, 1]
        loss_E = w_E * (E_pred - E_DES) ** 2
        loss_pos = torch.tensor(0.0, device=device)
        if w_pos > 0:
            loss_pos = w_pos * ((s_pred[0, :2] - s_target_norm[0, :2]) ** 2).sum()
        loss = loss_E + loss_pos + lambda_ctrl * (a_n ** 2).sum()
        loss.backward()
        opt.step()
        with torch.no_grad(): a_n.clamp_(-1.0, 1.0)

    with torch.no_grad():
        s_pred = model(torch.cat([s_norm, a_n], dim=-1))
    return a_n.detach().item() * 2.0, s_pred


# ─── Warm-up ────────────────────────────────────────────────────────────────

def warmup_model(model, env, updater, device, n_episodes=2, steps_per_ep=60):
    """Random-action warm-up."""
    err_history = []
    for ep in range(n_episodes):
        obs, _ = env.reset(); ep_errs = []
        for step in range(steps_per_ep):
            s_norm = _normalize_state(obs, device=device)
            a = np.random.uniform(-2.0, 2.0)
            obs_next, _, _, _, _ = env.step([a])
            a_norm = _normalize_action(a, device=device)
            s_next_norm = _normalize_state(obs_next, device=device)
            with torch.no_grad():
                s_pred = model(torch.cat([s_norm, a_norm], dim=-1))
            pred_raw = s_pred.clone().cpu(); pred_raw[0, 2] *= 8.0
            ep_errs.append(np.linalg.norm(pred_raw.squeeze(0).numpy() - obs_next))
            updater.update(s_norm, a_norm, s_next_norm)
            obs = obs_next
        err_history.append(np.mean(ep_errs))
        print(f"  Warm-up ep {ep+1}/{n_episodes}: mean_model_err={err_history[-1]:.4f}")
    return err_history, updater


# ─── Smart Burst: active verification + targeted correction ─────────────────

def smart_burst(model, env, updater, obs_start, s_target_norm, device,
                n_steps=5, burst_eta_mult=20, n_extra_passes=5):
    """When model fails, test it at its own decision boundary.

    Uses energy-guided actions (what the model WOULD decide) during burst.
    Each step reveals where the model's prediction was wrong → immediate
    correction at the model's most critical region: the decision boundary.

    Higher eta and more passes than exploration burst because:
    - Each data point is at a failure region (highly informative)
    - We want to fix this specific region aggressively
    """
    original_eta0 = updater.eta0
    updater.eta0 = original_eta0 * burst_eta_mult

    transitions = []
    obs = obs_start
    n_updates = 0

    for step in range(n_steps):
        s_norm = _normalize_state(obs, device=device)

        # Model's own decision: what action does it think is best?
        a, s_pred = find_action_energy(
            model, s_norm, s_target_norm, n_iters=50, lr=0.05, device=device)

        obs_next, _, _, _, _ = env.step([a])

        # Record the discrepancy: model predicted s_pred, reality is obs_next
        a_norm = _normalize_action(a, device=device)
        s_next_norm = _normalize_state(obs_next, device=device)
        transitions.append((s_norm, a_norm, s_next_norm, s_pred))

        # Immediate correction (boosted eta)
        updater.update(s_norm, a_norm, s_next_norm)
        n_updates += 1

        obs = obs_next

    # Extra passes: solidify learning from this failure region
    for _ in range(n_extra_passes):
        for (s_norm, a_norm, s_next_norm, s_pred) in transitions:
            updater.update(s_norm, a_norm, s_next_norm)
            n_updates += 1

    updater.eta0 = original_eta0
    return obs, n_updates


# ─── Main trial ─────────────────────────────────────────────────────────────

def run_trial(model, env, s_goal, updater, stats, trial_seed,
              total_steps=60, n_iters=50, dev_threshold_hard=None, device=None):
    if dev_threshold_hard is None:
        dev_threshold_hard = 2.5 * stats['sigma_train']

    obs, _ = env.reset(seed=trial_seed)
    s_target_norm = torch.tensor([[0.0, 1.0, 0.0]], device=device)
    step_count = 0; replan_count = 0; update_count = 0
    deviations = []; model_errors = []

    for step in range(total_steps):
        s_norm = _normalize_state(obs, device=device)

        # 1. Energy-guided action
        a, s_pred = find_action_energy(
            model, s_norm, s_target_norm, n_iters=n_iters, lr=0.05, device=device)

        s_before = obs.copy()
        obs_next, _, term, trunc, _ = env.step([a])
        step_count += 1

        # 2. Deviation check
        dev = deviation(obs_next, s_pred, device=device)
        deviations.append(dev)

        # 3. Model error
        pred_raw = s_pred.clone().cpu(); pred_raw[0, 2] *= 8.0
        model_err = np.linalg.norm(pred_raw.squeeze(0).numpy() - obs_next)
        model_errors.append(model_err)

        # 4. Online update (every step)
        a_norm = _normalize_action(a, device=device)
        s_next_norm = _normalize_state(obs_next, device=device)
        updater.update(s_norm, a_norm, s_next_norm)
        update_count += 1

        # 5. Smart burst on large deviation
        if dev > dev_threshold_hard:
            obs, n_burst = smart_burst(
                model, env, updater, obs_next, s_target_norm, device,
                n_steps=5, burst_eta_mult=20, n_extra_passes=5)
            update_count += n_burst
            replan_count += 1
        else:
            obs = obs_next

        if abs(np.arctan2(obs[1], obs[0]) - PI_2) < 0.2:
            break
        if term or trunc:
            break

    angle_final = np.arctan2(obs[1], obs[0])
    angle_err = abs(angle_final - PI_2)

    return {
        'success': angle_err < 0.2,
        'angle_err': angle_err,
        'steps_taken': step_count,
        'update_count': update_count,
        'replan_count': replan_count,
        'max_deviation': float(max(deviations)) if deviations else 0,
        'mean_deviation': float(np.mean(deviations)) if deviations else 0,
        'mean_model_err': float(np.mean(model_errors)) if model_errors else 0,
    }


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='kan_pendulum_model_v6.pt')
    parser.add_argument('--data', type=str, default='pendulum_data_v4.pt')
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-iters', type=int, default=50)
    parser.add_argument('--eta0', type=float, default=1e-3)
    parser.add_argument('--warmup-episodes', type=int, default=1)
    parser.add_argument('--threshold-mult', type=float, default=2.5)
    parser.add_argument('--device', type=str, default='mps',
                       choices=['cpu', 'mps', 'cuda'])
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    if args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # Load model
    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}"); sys.exit(1)
    ckpt = torch.load(args.model, weights_only=True)
    layer_dims = [4]  # input dim
    for key in sorted(ckpt.keys()):
        if 'base_weight' in key:
            layer_dims.append(ckpt[key].shape[0])
    model = KAN(layer_dims, grid_size=5, spline_order=3)
    model.load_state_dict(ckpt)
    model = model.to(device)

    # Load data & stats
    data = torch.load(args.data, weights_only=True)
    if isinstance(data, tuple) and len(data) == 2:
        x_train, y_train = data
    else:
        print("Unknown data format"); sys.exit(1)
    n_stats = min(len(x_train), 5000)
    idx = torch.randperm(len(x_train))[:n_stats]
    stats = compute_training_stats(model, x_train[idx].to(device),
                                   y_train[idx].to(device))
    updater = ThreeFactorUpdater(model, stats, eta0=args.eta0)
    dev_threshold_hard = args.threshold_mult * stats['sigma_train']

    # Warm-up
    env = gym.make("Pendulum-v1")
    print("=" * 80)
    print(f"Phase 0: Warm-up ({args.warmup_episodes} episodes, random actions)")
    print(f"  sigma_train={stats['sigma_train']:.4f}  |  hard_thresh={dev_threshold_hard:.4f}")
    print("=" * 80)
    t0 = time.time()
    err_history, updater = warmup_model(
        model, env, updater, device,
        n_episodes=args.warmup_episodes, steps_per_ep=60)
    print(f"  Warm-up: {time.time()-t0:.0f}s")

    # Main
    s_goal = torch.tensor([[0.0, 1.0, 0.0]])
    print(f"\n{'=' * 80}")
    print(f"Experiment F: Smart Burst (active verification + targeted correction)")
    print(f"Model: {args.model}  |  Device: {device}  |  n_iters={args.n_iters}")
    print(f"burst: energy-guided decisions, eta×20, 5 steps + 5 extra passes")
    print("=" * 80)

    print(f"\n{'Trial':>4s}  {'|dth0|':>7s}  {'|dth_f|':>9s}  {'R':>5s}  "
          f"{'mod_err':>8s}  {'max_dev':>8s}  {'repl':>5s}  {'upd':>5s}")
    print(f"{'─'*4}  {'─'*7}  {'─'*9}  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*5}  {'─'*5}")

    t_start = time.time(); ok = 0; all_results = []

    for t in range(args.trials):
        trial_seed = args.seed + t * 100
        obs0, _ = env.reset(seed=trial_seed)
        init_err = abs(np.arctan2(obs0[1], obs0[0]) - PI_2)

        r = run_trial(model, env, s_goal, updater, stats, trial_seed,
                      total_steps=60, n_iters=args.n_iters,
                      dev_threshold_hard=dev_threshold_hard, device=device)

        if r['success']: ok += 1
        all_results.append(r)

        elapsed = time.time() - t_start
        n_rem = args.trials - t - 1
        eta_str = f"{elapsed/(t+1)*n_rem:.0f}s" if t > 0 and n_rem > 0 else "0s"
        print(f"  {t+1:4d}  {init_err:7.3f}  {r['angle_err']:9.4f}  "
              f"{'Y' if r['success'] else 'N':>5s}  "
              f"{r['mean_model_err']:8.4f}  {r['max_deviation']:8.4f}  "
              f"{r['replan_count']:5d}  {r['update_count']:5d}  "
              f"[{elapsed:.0f}s ETA {eta_str}]")

    print(f"\n{'=' * 80}")
    print(f"SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Successes:        {ok}/{args.trials}")
    errs = [r['angle_err'] for r in all_results]
    print(f"  Mean |dth_final|: {np.mean(errs):.4f} rad")
    print(f"  Mean model_err:   {np.mean([r['mean_model_err'] for r in all_results]):.4f}")
    print(f"  Total time:       {time.time() - t_start:.0f}s")

    env.close()


if __name__ == "__main__":
    main()
