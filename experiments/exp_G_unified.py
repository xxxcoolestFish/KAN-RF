"""Experiment G: Unified shooting + energy-guided steps + smart burst + online learning.

Key unification:
  Shooting loss = sum_h w_E * (E(s_h) - E_des)^2  (per-step energy guide)
                + ||s_H - s*||^2                      (terminal position)
                + lambda * sum a_h^2                  (control cost)

  Per-step energy loss prevents optimization from wandering into model blind
  spots during the rollout. Terminal loss provides the global objective.

  Execution: step-by-step with monitoring, smart burst on large deviation,
  three-factor online update every step.

Usage:
  python exp_G_unified.py --trials 10 --device mps --model kan_pendulum_model_v6.pt
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


# ─── Shooting with per-step energy guidance ─────────────────────────────────

def shoot_energy_guided(model, s0, s_target, horizon=20, n_iters=300, lr=0.1,
                        lambda_ctrl=0.001, w_E=0.001, tol=1e-4,
                        n_restarts=4, device=None):
    """Batch-parallel shooting with per-step energy loss.

    Loss = w_E * sum_h (E(s_h) - E_des)^2  +  ||s_H - s*||^2  +  lambda * sum a^2
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    B = n_restarts
    s0_n = s0.clone().to(device); s0_n[:, 2] /= 8.0
    st_n = s_target.clone().to(device); st_n[:, 2] /= 8.0

    a_n = torch.zeros(horizon, B, device=device)
    torch.nn.init.uniform_(a_n, a=-0.3, b=0.3)
    a_n.requires_grad_(True)

    opt = torch.optim.Adam([a_n], lr=lr)
    iters_used, converged = n_iters, False

    for step in range(n_iters):
        opt.zero_grad()
        s = s0_n.expand(B, -1).clone()  # (B, 3)
        total_E_loss = torch.zeros(B, device=device)

        for h in range(horizon):
            x = torch.cat([s, a_n[h:h + 1].t()], dim=-1)  # (B, 4)
            s = model(x)
            nrm = s[:, :2].norm(dim=-1, keepdim=True).clamp(min=1e-6)
            s = torch.cat([s[:, :2] / nrm, s[:, 2:]], dim=-1)

            # Per-step energy (raw units)
            E_h = 0.5 * (s[:, 2] * 8.0) ** 2 + G * s[:, 1]
            total_E_loss += (E_h - E_DES) ** 2

        loss_term = ((s - st_n) ** 2).sum(dim=-1)  # (B,)
        loss_ctrl = (a_n ** 2).sum(dim=0)           # (B,)
        loss = (loss_term + w_E * total_E_loss + lambda_ctrl * loss_ctrl).sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            a_n.clamp_(-1.0, 1.0)

        if loss_term.min().item() < tol:
            converged = True; iters_used = step + 1; break

    # Select best restart
    with torch.no_grad():
        s_check = s0_n.expand(B, -1).clone()
        for h in range(horizon):
            x = torch.cat([s_check, a_n[h:h + 1].t()], dim=-1)
            s_check = model(x)
            nrm = s_check[:, :2].norm(dim=-1, keepdim=True).clamp(min=1e-6)
            s_check = torch.cat([s_check[:, :2] / nrm, s_check[:, 2:]], dim=-1)
        best_idx = ((s_check - st_n) ** 2).sum(dim=-1).argmin().item()

    best_a_n = a_n[:, best_idx:best_idx + 1].detach()

    # Final rollout
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
            {'iters': iters_used, 'conv': converged})


# ─── Truncation ─────────────────────────────────────────────────────────────

def truncate_at_convergence(actions, predicted_states, s_target, tol=0.005):
    st_n = s_target.clone(); st_n[:, 2] /= 8.0
    for h in range(1, len(predicted_states)):
        loss = ((predicted_states[h] - st_n) ** 2).sum().item()
        if loss < tol:
            return actions[:h], predicted_states[:h + 1], h
    return actions, predicted_states, len(actions)


# ─── Smart Burst ────────────────────────────────────────────────────────────

def smart_burst(model, env, updater, s_target_norm, obs_start, device,
                n_steps=5, burst_eta_mult=20, n_extra_passes=5):
    """Active model verification at the current (failure) state."""
    original_eta0 = updater.eta0
    updater.eta0 = original_eta0 * burst_eta_mult

    transitions = []; obs = obs_start; n_updates = 0

    for step in range(n_steps):
        s_norm = _normalize_state(obs, device=device)

        # Model's own best-guess action via energy-guided optimization
        s_raw = s_norm.clone(); s_raw[0, 2] *= 8.0
        sin_th = s_raw[0, 1].item(); thd = s_raw[0, 2].item()
        near_upright = abs(s_raw[0, 0].item()) < 0.5 and sin_th > 0 and abs(thd) < 3.0
        w_pos = 3.0 if near_upright else 0.0

        a_opt = torch.zeros(1, 1, device=device)
        torch.nn.init.uniform_(a_opt, a=-0.3, b=0.3)
        a_opt.requires_grad_(True)
        opt_a = torch.optim.Adam([a_opt], lr=0.05)

        for _ in range(50):
            opt_a.zero_grad()
            x = torch.cat([s_norm, a_opt], dim=-1)
            s_pred = model(x)
            E_pred = 0.5*(s_pred[0, 2]*8.0)**2 + G*s_pred[0, 1]
            loss_E = (E_pred - E_DES)**2
            loss_pos = torch.tensor(0.0, device=device)
            if w_pos > 0:
                loss_pos = w_pos*((s_pred[0,:2]-s_target_norm[0,:2])**2).sum()
            loss = loss_E + loss_pos + 0.001*(a_opt**2).sum()
            loss.backward(); opt_a.step()
            with torch.no_grad(): a_opt.clamp_(-1, 1)

        a = a_opt.detach().item() * 2.0
        obs_next, _, _, _, _ = env.step([a])

        a_norm = _normalize_action(a, device=device)
        s_next_norm = _normalize_state(obs_next, device=device)
        transitions.append((s_norm, a_norm, s_next_norm))

        updater.update(s_norm, a_norm, s_next_norm)
        n_updates += 1
        obs = obs_next

    for _ in range(n_extra_passes):
        for (s_n, a_n, sn_n) in transitions:
            updater.update(s_n, a_n, sn_n)
            n_updates += 1

    updater.eta0 = original_eta0
    return obs, n_updates


# ─── Warm-up ────────────────────────────────────────────────────────────────

def warmup_model(model, env, updater, device, n_episodes=1, steps_per_ep=60):
    err_history = []
    for ep in range(n_episodes):
        obs, _ = env.reset(); ep_errs = []
        for _ in range(steps_per_ep):
            s_norm = _normalize_state(obs, device=device)
            a = np.random.uniform(-2, 2)
            obs_next, _, _, _, _ = env.step([a])
            a_norm = _normalize_action(a, device=device)
            s_next_norm = _normalize_state(obs_next, device=device)
            with torch.no_grad():
                s_pred = model(torch.cat([s_norm, a_norm], dim=-1))
            pred_raw = s_pred.clone().cpu(); pred_raw[0, 2] *= 8.0
            ep_errs.append(np.linalg.norm(pred_raw.squeeze(0).numpy()-obs_next))
            updater.update(s_norm, a_norm, s_next_norm)
            obs = obs_next
        err_history.append(np.mean(ep_errs))
        print(f"  Warm-up ep {ep+1}/{n_episodes}: mean_err={err_history[-1]:.4f}")
    return err_history, updater


# ─── Main trial ─────────────────────────────────────────────────────────────

def run_trial(model, env, s_goal, updater, stats, trial_seed,
              H_max=20, total_steps=60, n_iters=300, n_restarts=4,
              w_E=0.001, dev_threshold_hard=None, device=None):
    if dev_threshold_hard is None:
        dev_threshold_hard = 2.5 * stats['sigma_train']

    obs, _ = env.reset(seed=trial_seed)
    s_target_norm = torch.tensor([[0.0, 1.0, 0.0]], device=device)
    step_count = 0; plan_cycles = 0; burst_count = 0; update_count = 0
    deviations = []; model_errors = []; all_eff_H = []

    while step_count < total_steps:
        s_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        # ── 1. Shoot with energy-guided loss ──
        actions, predicted_states, diag = shoot_energy_guided(
            model, s_t, s_goal, horizon=H_max, n_iters=n_iters,
            n_restarts=n_restarts, w_E=w_E, device=device)

        actions_trunc, pred_states_trunc, eff_H = truncate_at_convergence(
            actions, predicted_states, s_goal)
        all_eff_H.append(eff_H); plan_cycles += 1

        # ── 2. Execute + Monitor + Learn ──
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
            a_norm = _normalize_action(a, device=device)
            with torch.no_grad():
                s_pred = model(torch.cat([s_norm, a_norm], dim=-1))
            pred_raw = s_pred.cpu().clone(); pred_raw[0, 2] *= 8.0
            model_err = np.linalg.norm(pred_raw.squeeze(0).numpy() - obs_next)
            model_errors.append(model_err)

            # Online update (every step)
            s_next_norm = _normalize_state(obs_next, device=device)
            updater.update(s_norm, a_norm, s_next_norm)
            update_count += 1

            obs = obs_next

            # Goal check
            if abs(np.arctan2(obs[1], obs[0]) - PI_2) < 0.2:
                term = True; break

            # Smart burst + replan
            if dev > dev_threshold_hard:
                obs, n_burst = smart_burst(
                    model, env, updater, s_target_norm, obs, device,
                    n_steps=5, burst_eta_mult=20, n_extra_passes=5)
                update_count += n_burst; burst_count += 1
                replan = True; break

            if term or trunc:
                break

        if term or trunc:
            break

    angle_final = np.arctan2(obs[1], obs[0])
    angle_err = abs(angle_final - PI_2)

    return {
        'success': angle_err < 0.2,
        'angle_err': angle_err, 'steps_taken': step_count,
        'plan_cycles': plan_cycles, 'burst_count': burst_count,
        'update_count': update_count,
        'mean_eff_H': float(np.mean(all_eff_H)) if all_eff_H else 0,
        'max_dev': float(max(deviations)) if deviations else 0,
        'mean_model_err': float(np.mean(model_errors)) if model_errors else 0,
    }


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='kan_pendulum_model_v6.pt')
    parser.add_argument('--data', type=str, default='pendulum_data_v4.pt')
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--h-max', type=int, default=20)
    parser.add_argument('--n-iters', type=int, default=300)
    parser.add_argument('--n-restarts', type=int, default=4)
    parser.add_argument('--w-E', type=float, default=0.001,
                       help='weight of per-step energy loss in shooting')
    parser.add_argument('--eta0', type=float, default=1e-3)
    parser.add_argument('--warmup-episodes', type=int, default=1)
    parser.add_argument('--threshold-mult', type=float, default=2.5)
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'mps', 'cuda'])
    parser.add_argument('--compile', action='store_true', default=True,
                       help='Use torch.compile on KAN forward (CPU only)')
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    if args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
        args.compile = False  # compile not supported on MPS
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}  |  compile: {args.compile}")

    # Load model
    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}"); sys.exit(1)
    ckpt = torch.load(args.model, weights_only=True)
    layer_dims = [4]
    for key in sorted(ckpt.keys()):
        if 'base_weight' in key:
            layer_dims.append(ckpt[key].shape[0])
    model = KAN(layer_dims, grid_size=5, spline_order=3)
    model.load_state_dict(ckpt)
    model = model.to(device)

    if args.compile and device.type == 'cpu':
        import torch._dynamo as _dynamo
        _dynamo.config.suppress_errors = True
        _dynamo.config.cache_size_limit = 64
        model = torch.compile(model)
        # warmup
        for _ in range(5): _ = model(torch.randn(4, 4, device=device))
        print(f"  torch.compile enabled (cache_size=64)")

    print(f"Model: {layer_dims}  |  {sum(p.numel() for p in model.parameters())} params")

    # Stats
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
    print(f"Phase 0: Warm-up  |  sigma={stats['sigma_train']:.4f}  thresh={dev_threshold_hard:.4f}")
    print("=" * 80)
    t0 = time.time()
    err_history, updater = warmup_model(model, env, updater, device,
                                        n_episodes=args.warmup_episodes, steps_per_ep=60)
    print(f"  Warm-up: {time.time()-t0:.0f}s")

    # Main
    s_goal = torch.tensor([[0.0, 1.0, 0.0]])
    print(f"\n{'=' * 80}")
    print(f"Experiment G: Unified Shooting + Energy Guidance + Smart Burst")
    print(f"Model: {args.model}  |  w_E={args.w_E}  |  H={args.h_max}")
    print(f"n_iters={args.n_iters}  |  restarts={args.n_restarts}")
    print("=" * 80)

    print(f"\n{'Trial':>4s}  {'|d0|':>6s}  {'|df|':>8s}  {'R':>4s}  "
          f"{'effH':>5s}  {'mErr':>7s}  {'maxDv':>7s}  "
          f"{'plan':>5s}  {'brst':>5s}  {'upd':>5s}")
    print(f"{'─'*4}  {'─'*6}  {'─'*8}  {'─'*4}  {'─'*5}  {'─'*7}  "
          f"{'─'*7}  {'─'*5}  {'─'*5}  {'─'*5}")

    t_start = time.time(); ok = 0; all_results = []

    for t in range(args.trials):
        trial_seed = args.seed + t * 100
        obs0, _ = env.reset(seed=trial_seed)
        init_err = abs(np.arctan2(obs0[1], obs0[0]) - PI_2)

        r = run_trial(model, env, s_goal, updater, stats, trial_seed,
                      H_max=args.h_max, total_steps=60,
                      n_iters=args.n_iters, n_restarts=args.n_restarts,
                      w_E=args.w_E, dev_threshold_hard=dev_threshold_hard,
                      device=device)

        if r['success']: ok += 1
        all_results.append(r)

        elapsed = time.time() - t_start
        n_rem = args.trials - t - 1
        eta_str = f"{elapsed/(t+1)*n_rem:.0f}s" if t > 0 and n_rem > 0 else "0s"
        print(f"  {t+1:4d}  {init_err:6.3f}  {r['angle_err']:8.4f}  "
              f"{'Y' if r['success'] else 'N':>4s}  "
              f"{r['mean_eff_H']:4.1f}  {r['mean_model_err']:7.4f}  "
              f"{r['max_dev']:7.4f}  "
              f"{r['plan_cycles']:5d}  {r['burst_count']:5d}  "
              f"{r['update_count']:5d}  [{elapsed:.0f}s ETA {eta_str}]")

    errs = [r['angle_err'] for r in all_results]
    print(f"\n{'=' * 80}")
    print(f"  Success: {ok}/{args.trials}  |  Mean |df|: {np.mean(errs):.4f} rad")
    print(f"  Mean model_err: {np.mean([r['mean_model_err'] for r in all_results]):.4f}")
    print(f"  Total time: {time.time()-t_start:.0f}s")

    env.close()


if __name__ == "__main__":
    main()
