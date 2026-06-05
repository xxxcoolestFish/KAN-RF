"""KAN-MPC: Multi-step shooting with B-spline uncertainty + MPC replanning.

Key features:
  1. Plans H=10 steps through frozen KAN world model
  2. B-spline activation density penalty prevents model exploitation
  3. MPC: executes first action only, replans from new observation
  4. General: works for any KAN-based world model, not just pendulum

Usage:
  python run_kan_mpc.py --model kan_pendulum_model_v4.pt --trials 3
"""
import sys, time, argparse, os
import torch
import gymnasium as gym
import numpy as np
from kanrf import KAN
from control.shoot_v2 import shoot_uncertainty

_LOG = None


def log(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    print(msg, **kwargs)
    if _LOG is not None:
        _LOG.write(msg + "\n")
        _LOG.flush()


def run_mpc_trial(model, env, s_goal, total_steps=60, h_mpc=10,
                  n_iters=150, lr=0.1, lambda_ctrl=0.01,
                  beta_unc=0.01, sigma2=0.01, n_restarts=1, verbose=1):
    """MPC with B-spline uncertainty-penalized shooting.

    Each step: plan H_mpc actions, execute first, replan from observation.
    """
    obs, _ = env.reset()
    traj_real = [obs.copy()]
    traj_actions = []
    traj_model_errs = []
    traj_uncertainties = []
    plan_times = []

    if verbose >= 1:
        log(f"  {'Step':>4s}  {'action':>8s}  {'|Δθ|':>7s}  "
            f"{'E':>7s}  {'err_L2':>8s}  {'L_unc':>8s}")
        log(f"  {'─'*4}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*8}")

    t_trial_start = time.time()
    for step in range(total_steps):
        s_now = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        t0 = time.time()
        actions, s_final_model, diag = shoot_uncertainty(
            model, s_now, s_goal, horizon=h_mpc,
            n_iters=n_iters, lr=lr, lambda_ctrl=lambda_ctrl,
            beta_unc=beta_unc, sigma2=sigma2,
            n_restarts=n_restarts, verbose=(verbose >= 2), log_fn=log)
        plan_times.append(time.time() - t0)

        a = actions[0].item()

        # Model prediction for diagnostics
        s_norm = s_now.clone(); s_norm[:, 2] /= 8.0
        a_norm = torch.tensor([[a / 2.0]], dtype=torch.float32)
        with torch.no_grad():
            x = torch.cat([s_norm, a_norm], dim=-1)
            s_pred, B_list, E_list = model(x, return_activations=True)
            from kanrf import compute_uncertainty
            unc_val = compute_uncertainty(B_list, E_list, sigma2).item()
        pred_raw = s_pred.clone(); pred_raw[:, 2] *= 8.0
        pred_raw_np = pred_raw.squeeze(0).numpy()

        # Execute
        obs_next, _, terminated, truncated, _ = env.step([a])

        # Diagnostics
        model_err = np.linalg.norm(pred_raw_np - obs_next)
        angle_now = np.arctan2(obs_next[1], obs_next[0])
        angle_err = abs(angle_now - np.pi / 2)
        E = 0.5 * obs_next[2]**2 + 10 * obs_next[1]

        traj_model_errs.append(model_err)
        traj_uncertainties.append(unc_val)
        traj_real.append(obs_next.copy())
        traj_actions.append(a)

        if verbose >= 1:
            log(f"  {step:4d}  {a:+8.3f}  {angle_err:7.3f}  "
                f"{E:+7.2f}  {model_err:8.4f}  {unc_val:8.4f}")

        obs = obs_next
        if terminated or truncated:
            break

    t_total = time.time() - t_trial_start
    n_actual = len(traj_actions)

    # Summary
    log(f"\n  Trial: {n_actual} steps, {t_total:.0f}s")
    log(f"  Mean plan time: {np.mean(plan_times):.1f}s/step")
    log(f"  Mean model err: {np.mean(traj_model_errs):.4f}")
    log(f"  Mean L_unc:     {np.mean(traj_uncertainties):.4f}")
    log(f"  Total plan:     {sum(plan_times):.0f}s")

    return obs, traj_real, traj_actions, traj_model_errs, traj_uncertainties, t_total


def main():
    global _LOG
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='kan_pendulum_model_v4.pt')
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--n-iters', type=int, default=150)
    parser.add_argument('--beta-unc', type=float, default=0.01,
                       help='B-spline uncertainty penalty weight')
    parser.add_argument('--sigma2', type=float, default=0.01,
                       help='B-spline uncertainty scale')
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    model_tag = os.path.splitext(args.model)[0]
    _LOG = open(f"eval_kan_mpc_{model_tag}_log.txt", "w")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = KAN([4, 12, 3], grid_size=5, spline_order=3)
    model.load_state_dict(torch.load(args.model, weights_only=True))
    env = gym.make("Pendulum-v1")
    s_goal = torch.tensor([[0.0, 1.0, 0.0]])

    log(f"{'='*70}")
    log(f"KAN-MPC + B-spline Uncertainty")
    log(f"Model: {args.model}  |  {args.trials} trials × {args.steps} steps")
    log(f"H={args.horizon}  iters={args.n_iters}  β_unc={args.beta_unc}  σ²={args.sigma2}")
    log(f"{'='*70}")

    t_start = time.time()
    ok = 0
    all_angle_errs = []
    all_model_errs = []
    all_uncs = []

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

        final_obs, traj, actions, model_errs, uncertainties, t_trial = \
            run_mpc_trial(model, env, s_goal, total_steps=args.steps,
                         h_mpc=args.horizon, n_iters=args.n_iters,
                         beta_unc=args.beta_unc, sigma2=args.sigma2,
                         verbose=args.verbose)

        angle_final = np.arctan2(final_obs[1], final_obs[0])
        angle_err = abs(angle_final - np.pi / 2)
        all_angle_errs.append(angle_err)
        all_model_errs.append(np.mean(model_errs))
        all_uncs.append(np.mean(uncertainties))
        success = angle_err < 0.2
        if success:
            ok += 1

        elapsed = time.time() - t_start
        eta = elapsed / (trial + 1) * (args.trials - trial - 1)
        log(f"\n  >>> FINAL  "
            f"s=[{final_obs[0]:+.3f},{final_obs[1]:+.3f},{final_obs[2]:+.3f}]  "
            f"|Δθ_final|={angle_err:.3f}rad  "
            f"{'✓ SUCCESS' if success else '✗ FAIL'}  "
            f"mean_err={np.mean(model_errs):.4f}  mean_unc={np.mean(uncertainties):.4f}  "
            f"[{elapsed:.0f}s elapsed, ETA {eta:.0f}s]")

    log(f"\n{'='*70}")
    log(f"Summary")
    log(f"{'='*70}")
    log(f"  Successes:       {ok}/{args.trials}")
    log(f"  Mean |Δθ_final|: {np.mean(all_angle_errs):.3f} rad")
    log(f"  Min  |Δθ_final|: {np.min(all_angle_errs):.3f} rad")
    log(f"  Max  |Δθ_final|: {np.max(all_angle_errs):.3f} rad")
    log(f"  Mean model err:  {np.mean(all_model_errs):.4f}")
    log(f"  Mean L_unc:      {np.mean(all_uncs):.4f}")
    log(f"  Total time:      {time.time() - t_start:.0f}s")

    _LOG.close()
    env.close()


if __name__ == "__main__":
    main()
