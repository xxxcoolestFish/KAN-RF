"""Phase 2 evaluation: shoot planner on Pendulum-v1 swing-up task.

Two modes:
  --openloop: plan H steps once, execute all (prone to model exploitation)
  --mpc:      plan H_mpc steps, execute first action, re-observe, re-plan
              (standard MBRL defense against model exploitation)
"""
import sys, time, builtins
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


def run_mpc_trial(model, env, s_goal, total_steps=30, h_mpc=10,
                  n_iters=200, n_restarts=1, lambda_ctrl=0.01):
    """MPC: plan h_mpc steps, execute first action, re-plan from new observation.

    Returns: final_obs, trajectory (list of obs), plan_times
    """
    obs, _ = env.reset()
    traj = [obs.copy()]
    plan_times = []

    for step in range(total_steps):
        s_now = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        t0 = time.time()
        actions, _ = shoot(model, s_now, s_goal, horizon=h_mpc,
                           n_iters=n_iters, lr=0.1, lambda_ctrl=lambda_ctrl,
                           n_restarts=n_restarts, verbose=False)
        plan_times.append(time.time() - t0)

        obs, _, terminated, truncated, _ = env.step([actions[0].item()])
        traj.append(obs.copy())

        if terminated or truncated:
            break

    return obs, traj, plan_times


def run_openloop_trial(model, env, s_goal, horizon=30,
                       n_iters=300, n_restarts=2, lambda_ctrl=0.01):
    """Plan H steps once, execute all open-loop."""
    obs, _ = env.reset()
    s0 = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

    t0 = time.time()
    actions, _ = shoot(model, s0, s_goal, horizon=horizon,
                       n_iters=n_iters, lr=0.1, lambda_ctrl=lambda_ctrl,
                       n_restarts=n_restarts, verbose=True, log_fn=log)
    plan_time = time.time() - t0

    obs, _ = env.reset()
    for a in actions.numpy().flatten():
        obs, _, _, _, _ = env.step([a])
    return obs, plan_time


def evaluate(use_mpc: bool = True):
    global _LOG
    mode = "MPC" if use_mpc else "OpenLoop"
    _LOG = open(f"eval_{mode.lower()}_log.txt", "w")

    torch.manual_seed(42)
    np.random.seed(42)

    model = KAN([4, 12, 3], grid_size=5, spline_order=3)
    model.load_state_dict(torch.load("kan_pendulum_model.pt", weights_only=True))
    env = gym.make("Pendulum-v1")
    s_goal = torch.tensor([[0.0, 1.0, 0.0]])

    n_trials = 5

    # --- KAN shooting ---
    if use_mpc:
        log(f"{'='*70}")
        log(f"KAN MPC  |  {n_trials} trials  |  total_steps=30  H_mpc=10  "
            f"iters=200  restarts=1")
        log(f"{'='*70}")
    else:
        log(f"{'='*70}")
        log(f"KAN OpenLoop  |  {n_trials} trials  |  H=30  iters=300  restarts=2")
        log(f"{'='*70}")

    t_start = time.time()
    ok = 0
    angle_errs = []

    for trial in range(n_trials):
        obs0, _ = env.reset()
        s0 = torch.tensor(obs0, dtype=torch.float32).unsqueeze(0)
        init_angle = np.arctan2(s0[0, 1].item(), s0[0, 0].item())

        if use_mpc:
            log(f"\n[Trial {trial+1}/{n_trials}]  s0=[{s0[0,0]:+.2f},{s0[0,1]:+.2f},{s0[0,2]:+.2f}]  "
                f"θ0={init_angle:.2f}rad")
            log(f"  (planning each step, this will take a while...)")

            final_obs, traj, plan_times = run_mpc_trial(
                model, env, s_goal, total_steps=30, h_mpc=10,
                n_iters=200, n_restarts=1, lambda_ctrl=0.01)
            total_plan = sum(plan_times)
        else:
            log(f"\n[Trial {trial+1}/{n_trials}]  s0=[{s0[0,0]:+.2f},{s0[0,1]:+.2f},{s0[0,2]:+.2f}]  "
                f"θ0={init_angle:.2f}rad")

            final_obs, plan_time = run_openloop_trial(
                model, env, s_goal, horizon=30, n_iters=300, n_restarts=2, lambda_ctrl=0.01)
            total_plan = plan_time

        final_angle = np.arctan2(final_obs[1], final_obs[0])
        angle_err = abs(final_angle - np.pi / 2)
        angle_errs.append(angle_err)
        success = angle_err < 0.2
        if success:
            ok += 1

        elapsed = time.time() - t_start
        eta = elapsed / (trial + 1) * (n_trials - trial - 1)
        log(f"  => final=[{final_obs[0]:+.3f},{final_obs[1]:+.3f},{final_obs[2]:+.3f}]  "
            f"|Δθ|={angle_err:.3f}rad  {'✓' if success else '✗'}  "
            f"plan={total_plan:.0f}s  [{elapsed:.0f}s elapsed, ETA {eta:.0f}s]")

    # --- Random baseline ---
    log(f"\n{'='*70}")
    log(f"Random Baseline  |  {n_trials} trials  |  30 steps")
    log(f"{'='*70}")
    rand_ok = 0
    rand_errs = []
    for trial in range(n_trials):
        actions = np.random.uniform(-2, 2, (30,))
        obs, _ = env.reset()
        for a in actions:
            obs, _, _, _, _ = env.step([a])
        final_angle = np.arctan2(obs[1], obs[0])
        angle_err = abs(final_angle - np.pi / 2)
        rand_errs.append(angle_err)
        if angle_err < 0.2:
            rand_ok += 1
        log(f"  Trial {trial+1}: final=[{obs[0]:+.3f},{obs[1]:+.3f},{obs[2]:+.3f}]  "
            f"|Δθ|={angle_err:.3f}rad  {'✓' if angle_err < 0.2 else '✗'}")

    # --- Summary ---
    log(f"\n{'='*70}")
    log(f"Summary ({mode})")
    log(f"{'='*70}")
    log(f"  KAN:    {ok}/{n_trials} successes  "
        f"mean|Δθ|={np.mean(angle_errs):.3f}rad  "
        f"total_time={time.time()-t_start:.0f}s")
    log(f"  Random: {rand_ok}/{n_trials} successes  "
        f"mean|Δθ|={np.mean(rand_errs):.3f}rad")

    _LOG.close()
    env.close()


if __name__ == "__main__":
    import sys
    use_mpc = "--openloop" not in sys.argv  # default to MPC
    evaluate(use_mpc=use_mpc)


