"""Strategy+Execution integration: main loop with per-step diagnostics.

Usage:
  python run_strategy.py --model kan_pendulum_model_v3.pt --trials 3
  python run_strategy.py --model kan_pendulum_model.pt --trials 1 --verbose 2
"""
import sys, time, argparse, os
import torch
import gymnasium as gym
import numpy as np
from kanrf import KAN
from strategy import deviation, strategy_mode, intermediate_target
from execute import execute

_LOG = None


def log(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    print(msg, **kwargs)
    if _LOG is not None:
        _LOG.write(msg + "\n")
        _LOG.flush()


def run_trial(model, env, s_goal, obs0, total_steps=60, verbose=1):
    """Run one trial with Strategy+Execution.

    verbose: 0=minimal, 1=per-step summary, 2=full state dump each step
    """
    obs = obs0.copy()
    traj_real = [obs.copy()]
    traj_actions = []
    traj_modes = []
    traj_devs = []
    traj_model_errs = []       # per-step L2 norms
    traj_model_err_components = []  # per-step [cos_err, sin_err, thd_err]
    traj_exec_losses = []      # per-step execution optimizer final loss
    traj_s_pred = []           # model predictions
    traj_s_mid = []            # intermediate targets
    mode_transitions = []      # (step, from_mode, to_mode, reason)
    t_strategy = []            # strategy layer timing
    t_execution = []           # execution layer timing

    s_goal_np = s_goal.squeeze(0).numpy()
    prev_mode = None

    log(f"  {'Step':>4s}  {'mode':>10s}  {'a':>7s}  "
        f"{'|Δθ|':>7s}  {'E':>7s}  {'ΔE':>8s}  "
        f"{'err_cos':>8s}  {'err_sin':>8s}  {'err_thd':>8s}  {'err_L2':>8s}  "
        f"{'execloss':>9s}")
    log(f"  {'─'*4}  {'─'*10}  {'─'*7}  "
        f"{'─'*7}  {'─'*7}  {'─'*8}  "
        f"{'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  "
        f"{'─'*9}")

    t_trial_start = time.time()
    for step in range(total_steps):
        s_now = obs
        s_tensor = torch.tensor(s_now, dtype=torch.float32).unsqueeze(0)

        # --- Strategy Layer ---
        t0 = time.time()
        dev = deviation(s_now, s_goal_np)
        mode = strategy_mode(dev)
        s_mid = intermediate_target(s_now, mode)
        t_strategy.append(time.time() - t0)

        # Track mode transitions
        if prev_mode is not None and mode != prev_mode:
            reason = (f"near={dev['near_upright']} d_pos={dev['d_pos']:.3f} "
                      f"E={dev['E']:.2f} ΔE={dev['delta_E']:+.2f}")
            mode_transitions.append((step, prev_mode, mode, reason))
            if verbose >= 1:
                log(f"  ── Mode switch: {prev_mode} → {mode}  ({reason})")
        prev_mode = mode

        s_mid_tensor = torch.tensor(s_mid, dtype=torch.float32).unsqueeze(0)

        # --- Execution Layer ---
        t0 = time.time()
        a, exec_loss = execute(model, s_tensor, s_mid_tensor, n_iter=150, lr=0.1)
        t_execution.append(time.time() - t0)

        # --- Model prediction for diagnostics ---
        s_norm = s_tensor.clone(); s_norm[:, 2] /= 8.0
        a_norm = torch.tensor([[a / 2.0]], dtype=torch.float32)
        with torch.no_grad():
            x = torch.cat([s_norm, a_norm], dim=-1)
            pred_norm = model(x)
        pred_raw = pred_norm.clone(); pred_raw[:, 2] *= 8.0
        pred_raw_np = pred_raw.squeeze(0).numpy()

        # --- Step environment ---
        obs_next, _, terminated, truncated, _ = env.step([a])

        # --- Per-component diagnostics ---
        err_components = np.abs(pred_raw_np - obs_next)  # [cos_err, sin_err, thd_err]
        model_err = np.linalg.norm(err_components)
        traj_model_errs.append(model_err)
        traj_model_err_components.append(err_components)
        traj_exec_losses.append(exec_loss)
        traj_real.append(obs_next.copy())
        traj_actions.append(a)
        traj_modes.append(mode)
        traj_devs.append(dev)
        traj_s_pred.append(pred_raw_np)
        traj_s_mid.append(s_mid)

        # --- Per-step logging ---
        angle_now = np.arctan2(obs_next[1], obs_next[0])
        angle_err = abs(angle_now - np.pi / 2)

        # Warn if execution loss is high (optimizer couldn't converge)
        loss_warn = " ⚠" if exec_loss > 0.1 else ""
        log(f"  {step:4d}  {mode:>10s}  {a:+7.3f}  "
            f"{angle_err:7.3f}  {dev['E']:+7.2f}  {dev['delta_E']:+8.2f}  "
            f"{err_components[0]:8.4f}  {err_components[1]:8.4f}  "
            f"{err_components[2]:8.4f}  {model_err:8.4f}  "
            f"{exec_loss:8.5f}{loss_warn}")

        if verbose >= 2:
            log(f"         s_now= [{s_now[0]:+.4f},{s_now[1]:+.4f},{s_now[2]:+.4f}]")
            log(f"         s_mid= [{s_mid[0]:+.4f},{s_mid[1]:+.4f},{s_mid[2]:+.4f}]")
            log(f"         s_pred=[{pred_raw_np[0]:+.4f},{pred_raw_np[1]:+.4f},{pred_raw_np[2]:+.4f}]")
            log(f"         s_next=[{obs_next[0]:+.4f},{obs_next[1]:+.4f},{obs_next[2]:+.4f}]")
            log(f"         d_pos={dev['d_pos']:.4f}  e_kin={dev['e_kin']:.3f}  "
                f"near_upright={dev['near_upright']}")

        obs = obs_next
        if terminated or truncated:
            log(f"  ── Episode terminated at step {step}")
            break

    t_total = time.time() - t_trial_start
    n_actual = len(traj_actions)

    # --- Per-trial summary ---
    log(f"\n  {'─'*70}")
    log(f"  Trial diagnostics ({n_actual} steps, {t_total:.0f}s total):")

    # Mode distribution
    mode_counts = {}
    for m in traj_modes:
        mode_counts[m] = mode_counts.get(m, 0) + 1
    mode_str = "  ".join(f"{m}:{c}" for m, c in mode_counts.items())
    log(f"  Modes:        {mode_str}")
    if mode_transitions:
        log(f"  Transitions:  {len(mode_transitions)}")
        for step, fm, to, reason in mode_transitions:
            log(f"    step {step}: {fm} → {to}  ({reason})")

    # Action statistics
    actions = np.array(traj_actions)
    log(f"  Actions:      mean={actions.mean():+.4f}  std={actions.std():.4f}  "
        f"min={actions.min():+.4f}  max={actions.max():+.4f}")

    # Energy trajectory
    energies = np.array([d['E'] for d in traj_devs])
    log(f"  Energy E:     min={energies.min():+.3f}  max={energies.max():+.3f}  "
        f"mean={energies.mean():+.3f}  final={energies[-1]:+.3f}")

    # Model error breakdown
    errs = np.array(traj_model_err_components)  # (n, 3)
    log(f"  Model error per component (mean):")
    log(f"    cosθ: {errs[:,0].mean():.4f}  (max={errs[:,0].max():.4f}, "
        f"steps>0.1: {(errs[:,0]>0.1).sum()})")
    log(f"    sinθ: {errs[:,1].mean():.4f}  (max={errs[:,1].max():.4f}, "
        f"steps>0.1: {(errs[:,1]>0.1).sum()})")
    log(f"    θ̇:    {errs[:,2].mean():.4f}  (max={errs[:,2].max():.4f}, "
        f"steps>0.5: {(errs[:,2]>0.5).sum()})")
    log(f"    L2:   mean={np.mean(traj_model_errs):.4f}  "
        f"max={np.max(traj_model_errs):.4f}")
    log(f"    (θ̇ is denormalized — max error in real rad/s)")

    # Execution loss (high = KAN couldn't find a good action)
    exec_losses_arr = np.array(traj_exec_losses)
    high_loss_steps = (exec_losses_arr > 0.1).sum()
    log(f"  Exec loss:    mean={exec_losses_arr.mean():.6f}  "
        f"max={exec_losses_arr.max():.6f}  "
        f"steps>0.1: {high_loss_steps}/{n_actual}"
        f"{' ⚠ HIGH' if high_loss_steps > n_actual * 0.3 else ''}")

    # Timing
    log(f"  Timing:       strategy={np.sum(t_strategy):.3f}s  "
        f"execution={np.sum(t_execution):.1f}s  "
        f"per_step={t_total/n_actual:.1f}s")

    # Early stopping indicator
    angle_traj = np.array([np.arctan2(s[1], s[0]) for s in traj_real])
    err_traj = np.abs(angle_traj - np.pi / 2)
    min_err_step = np.argmin(err_traj)
    log(f"  Best |Δθ|:    {err_traj[min_err_step]:.4f} rad at step {min_err_step}")

    return (obs, traj_real, traj_actions, traj_modes, traj_devs,
            traj_model_errs, traj_model_err_components, traj_exec_losses,
            traj_s_pred, traj_s_mid,
            mode_transitions, t_total)


def main():
    global _LOG
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='kan_pendulum_model_v3.pt')
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--verbose', type=int, default=1,
                       help='0=minimal 1=per-step(rec) 2=full state dump')
    parser.add_argument('--save-traj', action='store_true', default=True,
                       help='Save trajectory data to .npz')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    _LOG = open("eval_strategy_log.txt", "w")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = KAN([4, 12, 3], grid_size=5, spline_order=3)
    model.load_state_dict(torch.load(args.model, weights_only=True))

    env = gym.make("Pendulum-v1")
    s_goal = torch.tensor([[0.0, 1.0, 0.0]])

    model_tag = os.path.splitext(args.model)[0]
    log(f"{'='*70}")
    log(f"Strategy+Execution  |  {args.trials} trials × {args.steps} steps  "
        f"|  model={args.model}  |  verbose={args.verbose}")
    log(f"{'='*70}")

    t_start = time.time()
    ok = 0
    all_angle_errs = []
    all_trials_data = []

    for trial in range(args.trials):
        obs0, _ = env.reset()
        s0 = torch.tensor(obs0, dtype=torch.float32).unsqueeze(0)
        init_angle = np.arctan2(s0[0, 1].item(), s0[0, 0].item())
        init_err = abs(init_angle - np.pi / 2)

        log(f"\n{'─'*70}")
        log(f"[Trial {trial+1}/{args.trials}]  "
            f"s0=[{s0[0,0]:+.2f},{s0[0,1]:+.2f},{s0[0,2]:+.2f}]  "
            f"|Δθ₀|={init_err:.3f}rad  "
            f"angle₀={init_angle:.3f}rad")
        log(f"{'─'*70}")

        result = run_trial(model, env, s_goal, obs0, total_steps=args.steps,
                          verbose=args.verbose)
        (final_obs, traj, actions, modes, devs,
         model_errs, model_err_components, exec_losses,
         s_pred_list, s_mid_list,
         mode_transitions, t_trial) = result

        angle_final = np.arctan2(final_obs[1], final_obs[0])
        angle_err = abs(angle_final - np.pi / 2)
        all_angle_errs.append(angle_err)
        success = angle_err < 0.2
        if success:
            ok += 1

        mean_model_err = np.mean(model_errs) if model_errs else 0
        elapsed = time.time() - t_start
        eta = elapsed / (trial + 1) * (args.trials - trial - 1)

        log(f"\n  >>> FINAL  "
            f"s=[{final_obs[0]:+.3f},{final_obs[1]:+.3f},{final_obs[2]:+.3f}]  "
            f"|Δθ_final|={angle_err:.3f}rad  "
            f"{'✓ SUCCESS' if success else '✗ FAIL'}  "
            f"mean_model_err={mean_model_err:.4f}  "
            f"[{elapsed:.0f}s elapsed, ETA {eta:.0f}s]")

        # Save trial data
        all_trials_data.append({
            'traj_real': np.array(traj),
            'actions': np.array(actions),
            'modes': np.array(modes),
            'model_err_l2': np.array(model_errs),
            'model_err_components': np.array(model_err_components),
            'exec_losses': np.array(exec_losses),
            's_pred': np.array(s_pred_list) if s_pred_list else np.array([]),
            's_mid': np.array(s_mid_list) if s_mid_list else np.array([]),
            'devs_E': np.array([d['E'] for d in devs]),
            'devs_deltaE': np.array([d['delta_E'] for d in devs]),
            'devs_d_pos': np.array([d['d_pos'] for d in devs]),
            'angle_final_err': angle_err,
            'success': success,
        })

    # --- Global summary ---
    log(f"\n{'='*70}")
    log(f"Summary")
    log(f"{'='*70}")
    log(f"  Successes:       {ok}/{args.trials}")
    log(f"  Mean |Δθ_final|: {np.mean(all_angle_errs):.3f} rad")
    log(f"  Min  |Δθ_final|: {np.min(all_angle_errs):.3f} rad")
    log(f"  Max  |Δθ_final|: {np.max(all_angle_errs):.3f} rad")
    log(f"  Total time:      {time.time() - t_start:.0f}s")

    # Per-trial summary table
    log(f"\n  {'Trial':>5s}  {'|Δθ₀|':>7s}  {'|Δθ_final|':>10s}  "
        f"{'Result':>6s}  {'Steps':>5s}  {'mean_err':>8s}  {'modes'}")
    log(f"  {'─'*5}  {'─'*7}  {'─'*10}  {'─'*6}  {'─'*5}  {'─'*8}  {'─'*30}")
    for i, td in enumerate(all_trials_data):
        init_e = abs(np.arctan2(td['traj_real'][0, 1], td['traj_real'][0, 0]) - np.pi/2)
        n_steps = len(td['actions'])
        mean_e = np.mean(td['model_err_l2'])
        mode_str = " ".join(f"{m}:{list(td['modes']).count(m)}"
                          for m in sorted(set(td['modes'])))
        log(f"  {i+1:5d}  {init_e:7.3f}  {td['angle_final_err']:10.3f}  "
            f"{'✓' if td['success'] else '✗':>6s}  {n_steps:5d}  "
            f"{mean_e:8.4f}  {mode_str}")

    # Save all trajectories
    if args.save_traj and all_trials_data:
        save_path = f"traj_strategy_{model_tag}.npz"
        save_dict = {}
        for i, td in enumerate(all_trials_data):
            for k, v in td.items():
                save_dict[f"trial{i}_{k}"] = v
        np.savez(save_path, **save_dict)
        log(f"\n  Trajectories saved to: {save_path}")

    _LOG.close()
    env.close()


if __name__ == "__main__":
    main()
