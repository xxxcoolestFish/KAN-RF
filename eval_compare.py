"""Compare MPC vs OpenLoop on Pendulum-v1 with detailed per-step diagnostics.

Usage:
  python eval_compare.py --mode mpc       --model kan_pendulum_model_v3.pt
  python eval_compare.py --mode openloop  --model kan_pendulum_model_v2.pt
  python eval_compare.py --mode both      --model kan_pendulum_model.pt --trials 2
"""
import sys, time, builtins, argparse, os
import torch
import gymnasium as gym
import numpy as np
from kanrf import KAN
from shoot import shoot

_LOG = None

def log(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    builtins.print(msg, **kwargs)
    if _LOG is not None:
        _LOG.write(msg + "\n")
        _LOG.flush()


def norm_state(s):
    """s: (3,) numpy -> normalized tensor (1,3)"""
    t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
    t[:, 2] /= 8.0
    return t


def denorm_state(t):
    """Normalized tensor (1,3) -> numpy (3,)"""
    t = t.clone()
    t[:, 2] *= 8.0
    return t.squeeze(0).numpy()


def model_predict(model, s_norm, a_norm):
    """Single-step model prediction. Returns (pred_norm, pred_raw_numpy)."""
    with torch.no_grad():
        x = torch.cat([s_norm, a_norm], dim=-1)
        pred_norm = model(x)
    return pred_norm, denorm_state(pred_norm)


def per_step_error(pred_raw, actual_raw):
    """Per-dim absolute error: [cos_err, sin_err, thetadot_err]."""
    return np.abs(pred_raw - actual_raw)


def compute_energy(obs):
    """Compute pendulum energy: E = 0.5*thd^2 + g*sin(th)."""
    cos_th, sin_th, thd = obs
    return 0.5 * thd * thd + 10.0 * sin_th


def run_mpc_trial(model, env, s_goal, total_steps=30, h_mpc=10,
                  n_iters=200, n_restarts=1, lambda_ctrl=0.01, verbose=1):
    """MPC with per-step model-vs-reality tracking."""
    obs, _ = env.reset()

    traj_real = [obs.copy()]
    traj_model = []
    traj_actions = []
    traj_model_errs = []
    traj_model_err_components = []
    step_times = []
    traj_energies = [compute_energy(obs)]

    if verbose >= 1:
        log(f"  {'Step':>4s}  {'action':>8s}  {'|Δθ|':>7s}  "
            f"{'err_cos':>8s}  {'err_sin':>8s}  {'err_thd':>8s}  "
            f"{'err_L2':>8s}  {'E':>7s}")
        log(f"  {'─'*4}  {'─'*8}  {'─'*7}  "
            f"{'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*7}")

    t_trial_start = time.time()
    for step in range(total_steps):
        t0 = time.time()
        s_now = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        s_now_norm = s_now.clone()
        s_now_norm[:, 2] /= 8.0

        actions, _ = shoot(model, s_now, s_goal, horizon=h_mpc,
                           n_iters=n_iters, lr=0.1, lambda_ctrl=lambda_ctrl,
                           n_restarts=n_restarts, verbose=False)

        a = actions[0].item()
        a_norm = torch.tensor([[a / 2.0]], dtype=torch.float32)
        _, pred_raw = model_predict(model, s_now_norm, a_norm)

        obs_next, _, terminated, truncated, _ = env.step([a])

        err = per_step_error(pred_raw, obs_next)
        err_l2 = np.linalg.norm(err)
        traj_model_errs.append(err_l2)
        traj_model_err_components.append(err)
        traj_real.append(obs_next.copy())
        traj_model.append(pred_raw)
        traj_actions.append(a)
        traj_energies.append(compute_energy(obs_next))
        step_times.append(time.time() - t0)

        # Per-step log
        angle = np.arctan2(obs_next[1], obs_next[0])
        cur_err = abs(angle - np.pi / 2)
        E_now = traj_energies[-1]

        if verbose >= 1:
            log(f"  {step:4d}  {a:+8.3f}  {cur_err:7.3f}  "
                f"{err[0]:8.4f}  {err[1]:8.4f}  {err[2]:8.4f}  "
                f"{err_l2:8.4f}  {E_now:+7.2f}")

        obs = obs_next
        if terminated or truncated:
            break

    t_total = time.time() - t_trial_start
    n_actual = len(traj_actions)

    return (obs, traj_real, traj_model, traj_model_errs, traj_model_err_components,
            traj_actions, traj_energies, step_times, n_actual, t_total)


def run_openloop_trial(model, env, s_goal, horizon=30, n_iters=300,
                       n_restarts=2, lambda_ctrl=0.01, verbose=1):
    """Open-loop with full trajectory model-vs-reality tracking."""
    obs, _ = env.reset()
    s0 = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

    # Plan all actions
    t0 = time.time()
    actions, s_final_model = shoot(model, s0, s_goal, horizon=horizon,
                                   n_iters=n_iters, lr=0.1, lambda_ctrl=lambda_ctrl,
                                   n_restarts=n_restarts, verbose=(verbose >= 2), log_fn=log)
    plan_time = time.time() - t0

    # Model-predicted trajectory
    s_norm = s0.clone(); s_norm[:, 2] /= 8.0
    traj_model = [denorm_state(s_norm)]
    for a in actions.numpy().flatten():
        a_norm = torch.tensor([[a / 2.0]], dtype=torch.float32)
        s_norm, pred_raw = model_predict(model, s_norm, a_norm)
        traj_model.append(pred_raw)

    if verbose >= 1:
        log(f"  Plan: {plan_time:.0f}s  Model |Δθ_final|={np.arctan2(s_final_model[0,1].item(), s_final_model[0,0].item()):.3f}rad")
        log(f"  {'Step':>4s}  {'action':>8s}  {'|Δθ|':>7s}  "
            f"{'err_cos':>8s}  {'err_sin':>8s}  {'err_thd':>8s}  "
            f"{'err_L2':>8s}  {'E':>7s}")
        log(f"  {'─'*4}  {'─'*8}  {'─'*7}  "
            f"{'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*7}")

    # Execute open-loop
    obs, _ = env.reset()
    traj_real = [obs.copy()]
    traj_model_errs = []
    traj_model_err_components = []
    traj_actions = []
    traj_energies = [compute_energy(obs)]

    for i, a in enumerate(actions.numpy().flatten()):
        obs_next, _, _, _, _ = env.step([a])
        err = per_step_error(traj_model[i + 1], obs_next)
        err_l2 = np.linalg.norm(err)
        traj_model_errs.append(err_l2)
        traj_model_err_components.append(err)
        traj_real.append(obs_next.copy())
        traj_actions.append(a)
        traj_energies.append(compute_energy(obs_next))

        angle = np.arctan2(obs_next[1], obs_next[0])
        cur_err = abs(angle - np.pi / 2)
        if verbose >= 1:
            log(f"  {i:4d}  {a:+8.3f}  {cur_err:7.3f}  "
                f"{err[0]:8.4f}  {err[1]:8.4f}  {err[2]:8.4f}  "
                f"{err_l2:8.4f}  {traj_energies[-1]:+7.2f}")

    n_actual = len(traj_actions)
    return (obs, traj_real, traj_model, traj_model_errs, traj_model_err_components,
            traj_actions, traj_energies, plan_time, n_actual)


def print_trial_diagnostics(mode, trial, obs_final, traj_real, traj_model,
                            traj_model_errs, traj_model_err_components,
                            traj_actions, traj_energies, plan_time, n_actual):
    """Print summary diagnostics for one trial."""
    angle_final = np.arctan2(obs_final[1], obs_final[0])
    angle_err = abs(angle_final - np.pi / 2)
    success = angle_err < 0.2

    errs = np.array(traj_model_err_components)
    log(f"\n  {'─'*70}")
    log(f"  Trial diagnostics ({n_actual} steps):")

    # Actions
    actions = np.array(traj_actions)
    log(f"  Actions:      mean={actions.mean():+.4f}  std={actions.std():.4f}  "
        f"min={actions.min():+.4f}  max={actions.max():+.4f}")

    # Energy
    energies = np.array(traj_energies)
    log(f"  Energy E:     min={energies.min():+.3f}  max={energies.max():+.3f}  "
        f"mean={energies.mean():+.3f}  initial={energies[0]:+.3f}  "
        f"final={energies[-1]:+.3f}")

    # Model error
    log(f"  Model error L2:      mean={np.mean(traj_model_errs):.4f}  "
        f"max={np.max(traj_model_errs):.4f}  "
        f"steps>0.5: {sum(1 for e in traj_model_errs if e>0.5)}/{n_actual}")
    log(f"  Model error per component (mean):")
    log(f"    cosθ: {errs[:,0].mean():.4f}  (max={errs[:,0].max():.4f}, "
        f"steps>0.2: {(errs[:,0]>0.2).sum()})")
    log(f"    sinθ: {errs[:,1].mean():.4f}  (max={errs[:,1].max():.4f}, "
        f"steps>0.2: {(errs[:,1]>0.2).sum()})")
    log(f"    θ̇:    {errs[:,2].mean():.4f}  (max={errs[:,2].max():.4f}, "
        f"steps>1.0: {(errs[:,2]>1.0).sum()})")

    # Per-step table for key indices
    key_indices = set([0, 1, 2] + list(range(4, n_actual, 5)) + [n_actual - 1])
    log(f"\n  {'Step':>5s}  {'action':>8s}  "
        f"{'cos(real)':>10s} {'sin(real)':>10s} {'thd(real)':>10s}  "
        f"{'cos(err)':>9s} {'sin(err)':>9s} {'thd(err)':>9s}  {'||err||':>8s}")
    for i in sorted(key_indices):
        if i >= n_actual:
            continue
        err = errs[i]
        actual = traj_real[i + 1]
        a = traj_actions[i]
        log(f"  {i:5d}  {a:+8.3f}  "
            f"{actual[0]:10.4f} {actual[1]:10.4f} {actual[2]:10.4f}  "
            f"{err[0]:9.4f} {err[1]:9.4f} {err[2]:9.4f}  "
            f"{np.linalg.norm(err):8.4f}")

    # Angle trajectory
    angle_traj = np.array([np.arctan2(s[1], s[0]) for s in traj_real])
    err_traj = np.abs(angle_traj - np.pi / 2)
    min_err_step = np.argmin(err_traj)
    log(f"  Best |Δθ|:           {err_traj[min_err_step]:.4f} rad at step {min_err_step}")

    return angle_err, success


def run_experiment(mode, n_trials, model, env, s_goal, model_name='', verbose=1):
    """Run n_trials with the given mode, logging diagnostics."""
    log(f"\n{'='*70}")
    log(f"Model: {model_name}")
    if mode == 'mpc':
        log(f"KAN MPC  |  {n_trials} trials  |  total_steps=30  H_mpc=10  "
            f"iters=150  restarts=1")
    else:
        log(f"KAN OpenLoop  |  {n_trials} trials  |  H=30  iters=250  restarts=1")
    log(f"{'='*70}")

    t_start = time.time()
    ok = 0
    angle_errs = []
    all_trials_data = []

    for trial in range(n_trials):
        obs0, _ = env.reset()
        s0 = torch.tensor(obs0, dtype=torch.float32).unsqueeze(0)
        init_angle = np.arctan2(s0[0, 1].item(), s0[0, 0].item())
        init_err = abs(init_angle - np.pi / 2)

        log(f"\n{'─'*70}")
        log(f"[Trial {trial+1}/{n_trials}]  "
            f"s0=[{s0[0,0]:+.2f},{s0[0,1]:+.2f},{s0[0,2]:+.2f}]  "
            f"|Δθ₀|={init_err:.3f}rad  angle₀={init_angle:.3f}rad")
        log(f"{'─'*70}")

        if mode == 'mpc':
            result = run_mpc_trial(
                model, env, s_goal, total_steps=30, h_mpc=10,
                n_iters=150, n_restarts=1, lambda_ctrl=0.01, verbose=verbose)
            (obs_final, traj_real, traj_model, traj_model_errs,
             traj_model_err_components, traj_actions, traj_energies,
             step_times, n_actual, t_trial) = result
            plan_time = sum(step_times)

            angle_err, success = print_trial_diagnostics(
                mode, trial, obs_final, traj_real, traj_model,
                traj_model_errs, traj_model_err_components,
                traj_actions, traj_energies, plan_time, n_actual)

        else:
            result = run_openloop_trial(
                model, env, s_goal, horizon=30, n_iters=250,
                n_restarts=1, lambda_ctrl=0.01, verbose=verbose)
            (obs_final, traj_real, traj_model, traj_model_errs,
             traj_model_err_components, traj_actions, traj_energies,
             plan_time, n_actual) = result

            angle_err, success = print_trial_diagnostics(
                mode, trial, obs_final, traj_real, traj_model,
                traj_model_errs, traj_model_err_components,
                traj_actions, traj_energies, plan_time, n_actual)

        angle_errs.append(angle_err)
        if success:
            ok += 1

        elapsed = time.time() - t_start
        eta = elapsed / (trial + 1) * (n_trials - trial - 1)

        log(f"\n  >>> FINAL  "
            f"s=[{obs_final[0]:+.3f},{obs_final[1]:+.3f},{obs_final[2]:+.3f}]  "
            f"|Δθ_final|={angle_err:.3f}rad  "
            f"{'✓ SUCCESS' if success else '✗ FAIL'}  "
            f"[{elapsed:.0f}s elapsed, ETA {eta:.0f}s]")

        # Save trial data
        all_trials_data.append({
            'traj_real': np.array(traj_real),
            'traj_model': np.array(traj_model),
            'actions': np.array(traj_actions),
            'energies': np.array(traj_energies),
            'model_err_l2': np.array(traj_model_errs),
            'model_err_components': np.array(traj_model_err_components),
            'angle_final_err': angle_err,
            'success': success,
        })

    log(f"\n--- {mode.upper()} Summary ---")
    log(f"  Successes:       {ok}/{n_trials}")
    log(f"  Mean |Δθ|:       {np.mean(angle_errs):.3f} rad")
    log(f"  Min  |Δθ|:       {np.min(angle_errs):.3f} rad")
    log(f"  Max  |Δθ|:       {np.max(angle_errs):.3f} rad")
    log(f"  Total time:      {time.time() - t_start:.0f}s")

    return ok, angle_errs, all_trials_data


def main():
    global _LOG
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['mpc', 'openloop', 'both'], default='both')
    parser.add_argument('--trials', type=int, default=2)
    parser.add_argument('--model', type=str, default='kan_pendulum_model_v2.pt')
    parser.add_argument('--verbose', type=int, default=1,
                       help='0=minimal 1=per-step 2=full shooting log')
    parser.add_argument('--save-traj', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = KAN([4, 12, 3], grid_size=5, spline_order=3)
    model.load_state_dict(torch.load(args.model, weights_only=True))
    s_goal = torch.tensor([[0.0, 1.0, 0.0]])

    model_tag = os.path.splitext(args.model)[0]

    if args.mode in ('mpc', 'both'):
        _LOG = open(f"eval_mpc_{model_tag}_log.txt", "w")
        env = gym.make("Pendulum-v1")
        np.random.seed(args.seed)
        ok, angle_errs, all_data = run_experiment(
            'mpc', args.trials, model, env, s_goal, args.model, args.verbose)
        if args.save_traj and all_data:
            save_dict = {}
            for i, td in enumerate(all_data):
                for k, v in td.items():
                    save_dict[f"trial{i}_{k}"] = v
            np.savez(f"traj_mpc_{model_tag}.npz", **save_dict)
            log(f"\n  Trajectories saved to: traj_mpc_{model_tag}.npz")
        _LOG.close()
        env.close()

    if args.mode in ('openloop', 'both'):
        _LOG = open(f"eval_openloop_{model_tag}_log.txt", "w")
        env = gym.make("Pendulum-v1")
        np.random.seed(args.seed)
        ok, angle_errs, all_data = run_experiment(
            'openloop', args.trials, model, env, s_goal, args.model, args.verbose)
        if args.save_traj and all_data:
            save_dict = {}
            for i, td in enumerate(all_data):
                for k, v in td.items():
                    save_dict[f"trial{i}_{k}"] = v
            np.savez(f"traj_openloop_{model_tag}.npz", **save_dict)
            log(f"\n  Trajectories saved to: traj_openloop_{model_tag}.npz")
        _LOG.close()
        env.close()


if __name__ == "__main__":
    main()
