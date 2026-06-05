"""Strategy+Execution v2 with KAN online learning.

After each control step, the observed (s, a, s') is used to update
the KAN world model via online SGD.  This breaks the vicious cycle of
"model inaccurate at top → bad decisions → never reach top → model
never gets data from top."

Update modes:
  --update fa          Feedback Alignment (Approach C)
  --update sgd         Full SGD, small lr (Approach B)
  --update local       B-spline-local SGD (Approach B with masking)
  --update threefactor Three-factor dynamic lr (error + density + count)

Usage:
  python run_online.py --model kan_pendulum_model_v4.pt --update threefactor --trials 3
"""
import sys, time, argparse, os
import torch
import gymnasium as gym
import numpy as np
from kanrf import KAN
from control.strategy_v2 import compute_gap, desired_velocity, strategy_mode
from control.execute_v2 import execute_v2
from control.online_learning import (
    FeedbackAlignmentUpdater,
    online_update_sgd,
    online_update_sgd_local,
)
from control.online_learning_v2 import compute_training_stats, ThreeFactorUpdater

_LOG = None


def log(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    print(msg, **kwargs)
    if _LOG is not None:
        _LOG.write(msg + "\n")
        _LOG.flush()


def run_trial_online(model, updater, env, s_goal, obs0,
                     total_steps=60, update_mode='fa', lr=1e-4, verbose=1):
    """Run one trial with online KAN updates."""
    obs = obs0.copy()
    traj_real = [obs.copy()]
    traj_actions = []
    traj_model_errs = []
    traj_update_losses = []
    traj_modes = []

    log(f"  {'Step':>4s}  {'mode':>10s}  {'a':>7s}  "
        f"{'|Δθ|':>7s}  {'E':>7s}  "
        f"{'err_L2':>8s}  {'upd_loss':>9s}")
    log(f"  {'─'*4}  {'─'*10}  {'─'*7}  "
        f"{'─'*7}  {'─'*7}  {'─'*8}  {'─'*9}")

    t_trial_start = time.time()
    for step in range(total_steps):
        s_now = obs

        # Strategy
        gap = compute_gap(s_now)
        mode = strategy_mode(gap)
        v_des = desired_velocity(gap, mode)

        s_tensor = torch.tensor(s_now, dtype=torch.float32).unsqueeze(0)
        v_des_tensor = torch.tensor(v_des, dtype=torch.float32).unsqueeze(0)

        # Execution
        a, exec_loss, a_init, diag = execute_v2(
            model, s_tensor, v_des_tensor, n_iter=15, lr=0.05)

        # Model prediction (for diagnostics)
        pred_raw_np = diag['s_pred']

        # Execute
        obs_next, _, terminated, truncated, _ = env.step([a])

        # Model error
        model_err = np.linalg.norm(pred_raw_np - obs_next)

        # ─── Online KAN Update ───
        s_norm = s_tensor.clone(); s_norm[:, 2] /= 8.0
        a_norm = torch.tensor([[a / 2.0]], dtype=torch.float32)
        s_true_norm = torch.tensor(obs_next, dtype=torch.float32).unsqueeze(0)
        s_true_norm[:, 2] /= 8.0

        if update_mode == 'fa':
            updater.update(s_norm, a_norm, s_true_norm)
            update_loss = None
        elif update_mode == 'threefactor':
            update_loss, max_eta = updater.update(s_norm, a_norm, s_true_norm)
        elif update_mode == 'local':
            update_loss = online_update_sgd_local(
                model, s_norm, a_norm, s_true_norm, lr=lr)
        else:  # 'sgd'
            update_loss = online_update_sgd(
                model, s_norm, a_norm, s_true_norm, lr=lr)

        # ─── Diagnostics ───
        angle_now = np.arctan2(obs_next[1], obs_next[0])
        angle_err = abs(angle_now - np.pi / 2)
        E = 0.5 * obs_next[2]**2 + 10 * obs_next[1]

        traj_model_errs.append(model_err)
        traj_actions.append(a)
        traj_modes.append(mode)
        traj_real.append(obs_next.copy())
        if update_loss is not None:
            traj_update_losses.append(update_loss)

        upd_str = f"{update_loss:.6f}" if update_loss is not None else "    N/A"
        log(f"  {step:4d}  {mode:>10s}  {a:+7.3f}  "
            f"{angle_err:7.3f}  {E:+7.2f}  "
            f"{model_err:8.4f}  {upd_str:>9s}")

        obs = obs_next
        if terminated or truncated:
            break

    t_total = time.time() - t_trial_start
    n_actual = len(traj_actions)

    log(f"\n  Trial: {n_actual} steps, {t_total:.0f}s")
    log(f"  Mean model err:  {np.mean(traj_model_errs):.4f}")
    if traj_update_losses:
        log(f"  Mean upd loss:   {np.mean(traj_update_losses):.6f}")
        log(f"  Upd loss trend:  {traj_update_losses[0]:.6f} → {traj_update_losses[-1]:.6f}")

    return obs, traj_real, traj_actions, traj_modes, traj_model_errs, t_total


def main():
    global _LOG
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='kan_pendulum_model.pt')
    parser.add_argument('--update', choices=['fa', 'sgd', 'local', 'threefactor'], default='local')
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    model_tag = os.path.splitext(args.model)[0]
    _LOG = open(f"eval_online_{args.update}_{model_tag}_log.txt", "w")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = KAN([4, 12, 3], grid_size=5, spline_order=3)
    model.load_state_dict(torch.load(args.model, weights_only=True))
    env = gym.make("Pendulum-v1")
    s_goal = torch.tensor([[0.0, 1.0, 0.0]])

    # Setup updater
    if args.update == 'threefactor':
        x_train, y_train = torch.load('pendulum_data_v4.pt', weights_only=True)
        stats = compute_training_stats(model, x_train[:5000], y_train[:5000])
        updater = ThreeFactorUpdater(model, stats, eta0=args.lr)
        log(f"σ_train={stats['sigma_train']:.6f}")
    elif args.update == 'fa':
        updater = FeedbackAlignmentUpdater(model, lr=args.lr)
    else:
        updater = None

    log(f"{'='*70}")
    log(f"KAN Online Learning  |  mode={args.update}  lr={args.lr}")
    log(f"Model: {args.model}  |  {args.trials} trials × {args.steps} steps")
    log(f"{'='*70}")

    t_start = time.time()
    ok = 0
    all_angle_errs = []

    for trial in range(args.trials):
        obs0, _ = env.reset()
        s0 = torch.tensor(obs0, dtype=torch.float32).unsqueeze(0)
        init_angle = np.arctan2(s0[0, 1].item(), s0[0, 0].item())
        init_err = abs(init_angle - np.pi / 2)

        log(f"\n{'─'*70}")
        log(f"[Trial {trial+1}/{args.trials}]  "
            f"s0=[{s0[0,0]:+.2f},{s0[0,1]:+.2f},{s0[0,2]:+.2f}]  "
            f"|Δθ₀|={init_err:.3f}rad")
        log(f"{'─'*70}")

        final_obs, traj, actions, modes, model_errs, t_trial = \
            run_trial_online(model, updater, env, s_goal, obs0,
                           total_steps=args.steps, update_mode=args.update,
                           lr=args.lr, verbose=args.verbose)

        angle_final = np.arctan2(final_obs[1], final_obs[0])
        angle_err = abs(angle_final - np.pi / 2)
        all_angle_errs.append(angle_err)
        success = angle_err < 0.2
        if success:
            ok += 1

        elapsed = time.time() - t_start
        eta = elapsed / (trial + 1) * (args.trials - trial - 1)
        log(f"\n  >>> FINAL  "
            f"s=[{final_obs[0]:+.3f},{final_obs[1]:+.3f},{final_obs[2]:+.3f}]  "
            f"|Δθ_final|={angle_err:.3f}rad  "
            f"{'✓ SUCCESS' if success else '✗ FAIL'}  "
            f"mean_err={np.mean(model_errs):.4f}  "
            f"[{elapsed:.0f}s elapsed, ETA {eta:.0f}s]")

    log(f"\n{'='*70}")
    log(f"Summary")
    log(f"{'='*70}")
    log(f"  Successes:       {ok}/{args.trials}")
    log(f"  Mean |Δθ_final|: {np.mean(all_angle_errs):.3f} rad")
    log(f"  Total time:      {time.time() - t_start:.0f}s")

    _LOG.close()
    env.close()


if __name__ == "__main__":
    main()
