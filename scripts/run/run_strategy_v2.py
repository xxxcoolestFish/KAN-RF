"""Strategy+Execution v2: velocity-field guidance + Gauss-Newton + controllability loss.

Usage:
  python run_strategy_v2.py --model kan_pendulum_model.pt --trials 3
"""
import sys, time, argparse, os
import torch
import gymnasium as gym
import numpy as np
from kanrf import KAN
from control.strategy_v2 import compute_gap, desired_velocity, strategy_mode
from control.execute_v2 import execute_v2

_LOG = None


def log(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    print(msg, **kwargs)
    if _LOG is not None:
        _LOG.write(msg + "\n")
        _LOG.flush()


def run_trial(model, env, s_goal, obs0, total_steps=60, verbose=1):
    obs = obs0.copy()
    traj_real = [obs.copy()]
    traj_actions = []
    traj_modes = []
    traj_gaps = []
    traj_a_inits = []
    traj_model_errs = []
    traj_model_err_components = []
    traj_exec_losses = []
    traj_s_pred = []

    log(f"  {'Step':>4s}  {'mode':>10s}  {'a_init':>7s}  {'a*':>7s}  "
        f"{'|Δθ|':>7s}  {'E':>7s}  {'ΔE':>8s}  "
        f"{'err_cos':>8s}  {'err_sin':>8s}  {'err_thd':>8s}  {'err_L2':>8s}  {'execloss':>9s}")
    log(f"  {'─'*4}  {'─'*10}  {'─'*7}  {'─'*7}  "
        f"{'─'*7}  {'─'*7}  {'─'*8}  "
        f"{'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*9}")

    t_trial_start = time.time()
    for step in range(total_steps):
        s_now = obs

        # --- Strategy Layer v2 ---
        gap = compute_gap(s_now)
        mode = strategy_mode(gap)
        v_des = desired_velocity(gap, mode)  # (3,) numpy, per-step delta in raw units

        s_tensor = torch.tensor(s_now, dtype=torch.float32).unsqueeze(0)
        v_des_tensor = torch.tensor(v_des, dtype=torch.float32).unsqueeze(0)

        # --- Execution Layer v2 ---
        a, exec_loss, a_init, diag = execute_v2(model, s_tensor, v_des_tensor,
                                                n_iter=15, lr=0.05)

        # KAN predictions from execution diagnostics (already computed)
        pred_raw_np = diag['s_pred']  # f_KAN(s, a*), raw units
        f_zero_np = diag['f_zero']    # f_KAN(s, a=0), raw units

        # --- Step environment ---
        obs_next, _, terminated, truncated, _ = env.step([a])

        # --- Diagnostics ---
        err_components = np.abs(pred_raw_np - obs_next)
        model_err = np.linalg.norm(err_components)

        traj_model_errs.append(model_err)
        traj_model_err_components.append(err_components)
        traj_exec_losses.append(exec_loss)
        traj_real.append(obs_next.copy())
        traj_actions.append(a)
        traj_modes.append(mode)
        traj_gaps.append(gap)
        traj_a_inits.append(a_init)
        traj_s_pred.append(pred_raw_np)

        angle_now = np.arctan2(obs_next[1], obs_next[0])
        angle_err = abs(angle_now - np.pi / 2)

        loss_warn = " ⚠" if exec_loss > 0.3 else ""
        log(f"  {step:4d}  {mode:>10s}  {a_init:+7.3f}  {a:+7.3f}  "
            f"{angle_err:7.3f}  {gap['E']:+7.2f}  {gap['delta_E']:+8.2f}  "
            f"{err_components[0]:8.4f}  {err_components[1]:8.4f}  "
            f"{err_components[2]:8.4f}  {model_err:8.4f}  "
            f"{exec_loss:8.5f}{loss_warn}")

        if verbose >= 2:
            log(f"         ┌── Strategy")
            log(f"         │ v_des =[{v_des[0]:+8.4f} {v_des[1]:+8.4f} {v_des[2]:+8.4f}] (desired state delta)")
            log(f"         ├── Gauss-Newton")
            J = diag['J_a']
            log(f"         │ J_a   =[{J[0]:+8.4f} {J[1]:+8.4f} {J[2]:+8.4f}] (df/da at a=0, raw units)")
            log(f"         │ a_init= {diag['a_init']:+8.4f}  (Gauss-Newton step)")
            log(f"         ├── Model predictions")
            log(f"         │ f(s,0)    =[{f_zero_np[0]:+8.4f} {f_zero_np[1]:+8.4f} {f_zero_np[2]:+8.4f}] (natural evolution)")
            log(f"         │ s+v_des   =[{diag['s_target'][0]:+8.4f} {diag['s_target'][1]:+8.4f} {diag['s_target'][2]:+8.4f}] (strategy target)")
            log(f"         │ f(s,a*)   =[{pred_raw_np[0]:+8.4f} {pred_raw_np[1]:+8.4f} {pred_raw_np[2]:+8.4f}] (predicted next)")
            log(f"         ├── Result: a*={a:+8.4f}  a_init={a_init:+8.4f}  loss={exec_loss:.6f}")
            log(f"         └── Reality: s'=[{obs_next[0]:+8.4f} {obs_next[1]:+8.4f} {obs_next[2]:+8.4f}] (actual next)")

        obs = obs_next
        if terminated or truncated:
            break

    t_total = time.time() - t_trial_start
    n_actual = len(traj_actions)

    # Diagnostics
    log(f"\n  {'─'*70}")
    log(f"  Trial diagnostics ({n_actual} steps, {t_total:.0f}s):")

    mode_counts = {}
    for m in traj_modes:
        mode_counts[m] = mode_counts.get(m, 0) + 1
    mode_str = "  ".join(f"{m}:{c}" for m, c in mode_counts.items())
    log(f"  Modes:        {mode_str}")

    actions = np.array(traj_actions)
    a_inits = np.array(traj_a_inits)
    log(f"  Actions:      mean={actions.mean():+.4f}  std={actions.std():.4f}  "
        f"min={actions.min():+.4f}  max={actions.max():+.4f}")
    log(f"  a_init:       mean={a_inits.mean():+.4f}  std={a_inits.std():.4f}")

    energies = np.array([g['E'] for g in traj_gaps])
    log(f"  Energy E:     min={energies.min():+.3f}  max={energies.max():+.3f}  "
        f"mean={energies.mean():+.3f}  final={energies[-1]:+.3f}")

    errs = np.array(traj_model_err_components)
    log(f"  Model error components (mean):")
    log(f"    cosθ: {errs[:,0].mean():.4f}  sinθ: {errs[:,1].mean():.4f}  θ̇: {errs[:,2].mean():.4f}")
    log(f"    L2 mean={np.mean(traj_model_errs):.4f}  max={np.max(traj_model_errs):.4f}")

    execs = np.array(traj_exec_losses)
    log(f"  Exec loss:    mean={execs.mean():.6f}  max={execs.max():.6f}  "
        f"steps>0.3: {(execs>0.3).sum()}/{n_actual}")

    angle_traj = np.array([np.arctan2(s[1], s[0]) for s in traj_real])
    err_traj = np.abs(angle_traj - np.pi / 2)
    best_step = np.argmin(err_traj)
    log(f"  Best |Δθ|:    {err_traj[best_step]:.4f} rad at step {best_step}")

    return (obs, traj_real, traj_actions, traj_modes, traj_gaps,
            traj_model_errs, traj_model_err_components, traj_exec_losses,
            traj_s_pred, traj_a_inits, t_total)


def main():
    global _LOG
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='kan_pendulum_model.pt')
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--save-traj', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    _LOG = open("eval_strategy_v2_log.txt", "w")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = KAN([4, 12, 3], grid_size=5, spline_order=3)
    model.load_state_dict(torch.load(args.model, weights_only=True))
    env = gym.make("Pendulum-v1")
    s_goal = torch.tensor([[0.0, 1.0, 0.0]])

    model_tag = os.path.splitext(args.model)[0]
    log(f"{'='*70}")
    log(f"Strategy+Execution v2 | {args.trials} trials x {args.steps} steps | model={args.model}")
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
            f"|Δθ₀|={init_err:.3f}rad  angle₀={init_angle:.3f}rad")
        log(f"{'─'*70}")

        result = run_trial(model, env, s_goal, obs0, total_steps=args.steps,
                          verbose=args.verbose)
        (final_obs, traj, actions, modes, gaps,
         model_errs, model_err_components, exec_losses,
         s_pred_list, a_inits, t_trial) = result

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

        all_trials_data.append({
            'traj_real': np.array(traj),
            'actions': np.array(actions),
            'a_inits': np.array(a_inits),
            'modes': np.array(modes),
            'model_err_l2': np.array(model_errs),
            'model_err_components': np.array(model_err_components),
            'exec_losses': np.array(exec_losses),
            'energies': np.array([g['E'] for g in gaps]),
            'delta_Es': np.array([g['delta_E'] for g in gaps]),
            'angle_final_err': angle_err,
            'success': success,
        })

    log(f"\n{'='*70}")
    log(f"Summary")
    log(f"{'='*70}")
    log(f"  Successes:       {ok}/{args.trials}")
    log(f"  Mean |Δθ_final|: {np.mean(all_angle_errs):.3f} rad")
    log(f"  Min  |Δθ_final|: {np.min(all_angle_errs):.3f} rad")
    log(f"  Max  |Δθ_final|: {np.max(all_angle_errs):.3f} rad")
    log(f"  Total time:      {time.time() - t_start:.0f}s")

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

    if args.save_traj and all_trials_data:
        save_path = f"traj_strategy_v2_{model_tag}.npz"
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
