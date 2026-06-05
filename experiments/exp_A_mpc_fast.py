"""Experiment A (fast): MPC with perfect physics model — warm-start + early-stop.

Key optimizations over naive MPC:
  1. Warm-start: first step uses random init; subsequent steps inherit shifted
     actions from previous plan (a_{0:H-2} → a_{1:H-1}, append 0).
  2. Early stopping: exit optimizer when terminal_loss < tol (converged).
  3. Single restart: warm-start eliminates need for multiple random restarts.

Tests: open-loop baseline + MPC across horizon={5,10,15,20,30} × budget={low,med,high}.
Estimates ~10min total (physics model is cheap).
"""
import sys, time, argparse
import torch
import gymnasium as gym
import numpy as np

G = 10.0
PI_2 = np.pi / 2
MAX_SPEED = 8.0


class PerfectPhysicsModel:
    def __init__(self):
        self.dt = 0.05
        self.max_speed = MAX_SPEED

    def __call__(self, x):
        cos_norm, sin_norm, thd_norm, torque_norm = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]
        thd = thd_norm * 8.0
        theta = torch.atan2(sin_norm, cos_norm)
        thd_new = (thd + (15.0 * torch.sin(theta) + 3.0 * torque_norm * 2.0) * self.dt).clamp(-self.max_speed, self.max_speed)
        theta_new = theta + thd_new * self.dt
        return torch.cat([torch.cos(theta_new), torch.sin(theta_new), thd_new / 8.0], dim=-1)


def shoot_perfect(model, s0, s_target, horizon, n_iters, lr, lambda_ctrl,
                  tol=1e-4, a_warm=None):
    """Shooting with optional warm-start and early stopping.

    Args:
        a_warm: (horizon, 1) warm-start actions (normalized), or None for random init.
        tol: early-stop when terminal_loss < tol.

    Returns: (actions_raw, s_final_raw, best_loss, n_iters_used, converged)
    """
    s0_norm = s0.clone(); s0_norm[:, 2] /= 8.0
    s_target_norm = s_target.clone(); s_target_norm[:, 2] /= 8.0

    # Init: warm-start or random
    a_norm = torch.zeros(horizon, 1)
    if a_warm is not None:
        with torch.no_grad():
            a_norm.copy_(a_warm)
    else:
        torch.nn.init.uniform_(a_norm, a=-0.3, b=0.3)
    a_norm.requires_grad_(True)

    opt = torch.optim.Adam([a_norm], lr=lr)
    converged = False
    iters_used = n_iters

    for step in range(n_iters):
        opt.zero_grad()
        s = s0_norm.clone()
        for h in range(horizon):
            x = torch.cat([s, a_norm[h:h + 1]], dim=-1)
            s = model(x)
            norm = (s[:, :2] ** 2).sum(dim=-1, keepdim=True).sqrt().clamp(min=1e-6)
            s = torch.cat([s[:, :2] / norm, s[:, 2:]], dim=-1)

        loss_terminal = ((s - s_target_norm) ** 2).sum()
        loss_ctrl = (a_norm ** 2).sum()
        loss = loss_terminal + lambda_ctrl * loss_ctrl
        loss.backward()
        opt.step()
        with torch.no_grad():
            a_norm.clamp_(-1.0, 1.0)

        if loss_terminal.item() < tol:
            converged = True
            iters_used = step + 1
            break

    with torch.no_grad():
        # Predicted final state
        s = s0_norm.clone()
        for h in range(horizon):
            x = torch.cat([s, a_norm[h:h + 1]], dim=-1)
            s = model(x)
            norm = s[:, :2].norm(dim=-1, keepdim=True).clamp(min=1e-6)
            s = torch.cat([s[:, :2] / norm, s[:, 2:]], dim=-1)
        s_final_raw = s.clone(); s_final_raw[:, 2] *= 8.0

    return a_norm.detach() * 2.0, s_final_raw, loss_terminal.item(), iters_used, converged


def warm_shift(actions_prev, horizon):
    """Shift previous actions by 1, append zero. Returns normalized actions."""
    a_shifted = torch.zeros(horizon, 1)
    if actions_prev.shape[0] >= horizon + 1:
        a_shifted[:-1, 0] = actions_prev[1:horizon, 0]
    elif actions_prev.shape[0] == horizon:
        a_shifted[:-1, 0] = actions_prev[1:, 0]
    else:
        n_copy = min(actions_prev.shape[0] - 1, horizon - 1)
        a_shifted[:n_copy, 0] = actions_prev[1:n_copy + 1, 0]
    return a_shifted


def run_openloop(model, env, s_goal, trial_seed, n_restarts=1,
                 n_iters=500, lr=0.1, lambda_ctrl=0.001, tol=1e-4):
    obs0, _ = env.reset(seed=trial_seed)
    s0 = torch.tensor(obs0, dtype=torch.float32).unsqueeze(0)

    t0 = time.time()
    best_total = float('inf'); best_actions = None; best_final = None
    best_iters = 0; best_converged = False

    for r in range(n_restarts):
        actions, s_final, term_loss, iters, conv = shoot_perfect(
            model, s0, s_goal, horizon=30, n_iters=n_iters, lr=lr,
            lambda_ctrl=lambda_ctrl, tol=tol, a_warm=None)
        if term_loss < best_total:
            best_total = term_loss; best_actions = actions
            best_final = s_final; best_iters = iters; best_converged = conv

    plan_t = time.time() - t0
    model_err = abs(np.arctan2(best_final[0, 1].item(), best_final[0, 0].item()) - PI_2)

    # Execute
    obs = obs0
    for a in best_actions.numpy().flatten():
        obs, _, term, trunc, _ = env.step([a])
        if term or trunc: break

    angle_final = np.arctan2(obs[1], obs[0])
    angle_err = abs(angle_final - PI_2)
    return {
        'success': angle_err < 0.2, 'angle_err': angle_err,
        'model_err': model_err, 'plan_time': plan_t,
        'init_err': abs(np.arctan2(obs0[1], obs0[0]) - PI_2),
        'iters_used': best_iters, 'converged': best_converged,
        'final_state': obs,
    }


def run_mpc_fast(model, env, s_goal, trial_seed, horizon=10,
                 n_iters=200, lr=0.1, lambda_ctrl=0.001, tol=1e-4,
                 total_steps=60, warmstart=True):
    obs, _ = env.reset(seed=trial_seed)
    obs0 = obs.copy()
    plan_times = []
    iters_used_list = []
    converged_list = []
    a_warm = None  # first step: random init

    for step in range(total_steps):
        s_now = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        t0 = time.time()
        actions, s_final, _, iters_used, converged = shoot_perfect(
            model, s_now, s_goal, horizon=horizon, n_iters=n_iters, lr=lr,
            lambda_ctrl=lambda_ctrl, tol=tol, a_warm=a_warm)
        plan_times.append(time.time() - t0)
        iters_used_list.append(iters_used)
        converged_list.append(converged)

        # Warm-start for next step: shift + re-normalize
        if warmstart:
            a_warm = warm_shift(actions / 2.0, horizon)  # store normalized

        obs, _, term, trunc, _ = env.step([actions[0].item()])
        if term or trunc: break

    angle_final = np.arctan2(obs[1], obs[0])
    angle_err = abs(angle_final - PI_2)
    return {
        'success': angle_err < 0.2, 'angle_err': angle_err,
        'steps_taken': len(plan_times),
        'mean_plan_time': np.mean(plan_times),
        'total_plan_time': np.sum(plan_times),
        'mean_iters': np.mean(iters_used_list),
        'conv_rate': np.mean(converged_list),
        'init_err': abs(np.arctan2(obs0[1], obs0[0]) - PI_2),
        'final_state': obs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--warmstart', type=int, default=1,
                       help='1=use warm-start (default), 0=no warm-start')
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = PerfectPhysicsModel()
    env = gym.make("Pendulum-v1")
    s_goal = torch.tensor([[0.0, 1.0, 0.0]])
    log = open("exp_A_mpc_fast_log.txt", "w")

    def p(*a, **kw):
        msg = " ".join(str(x) for x in a)
        print(msg, **kw); log.write(msg + "\n"); log.flush()

    WS = bool(args.warmstart)
    p(f"{'='*70}")
    p(f"Experiment A (Fast): MPC + Warm-start={WS} + Early-stop")
    p(f"Perfect Physics Model  |  {args.trials} trials  |  seed={args.seed}")
    p(f"{'='*70}")

    # ── Baseline: Open-loop H=30 ──
    p(f"\n{'─'*70}")
    p(f"BASELINE: Open-loop H=30 (500 iters, 1 restart, tol=1e-4)")
    p(f"{'─'*70}")
    p(f"  {'Tr':>3s}  {'|Δθ₀|':>7s}  {'model|Δθ|':>9s}  {'real|Δθ|':>9s}  "
      f"{'R':>5s}  {'iters':>5s}  {'t(s)':>6s}")
    p(f"  {'─'*3}  {'─'*7}  {'─'*9}  {'─'*9}  {'─'*5}  {'─'*5}  {'─'*6}")

    bl_ok, bl_details = 0, []
    for t in range(args.trials):
        r = run_openloop(model, env, s_goal, args.seed + t * 100,
                         n_restarts=1, n_iters=500, tol=1e-4)
        bl_details.append(r)
        if r['success']: bl_ok += 1
        p(f"  {t+1:3d}  {r['init_err']:7.3f}  {r['model_err']:9.4f}  "
          f"{r['angle_err']:9.4f}  {'✓' if r['success'] else '✗':>5s}  "
          f"{r['iters_used']:5d}  {r['plan_time']:5.1f}s")
    p(f"  => {bl_ok}/{args.trials}")
    if bl_ok < args.trials:
        p(f"  WARNING: open-loop should be 5/5 with perfect model!")

    # ── MPC: horizon × budget sweep ──
    budgets = {
        'low':    (100, 1e-1),
        'medium': (300, 1e-1),
        'high':   (500, 5e-2),
    }
    horizons = [5, 10, 15, 20, 30]

    p(f"\n{'='*70}")
    p(f"MPC SWEEP (warmstart={'ON' if WS else 'OFF'})")
    p(f"{'='*70}")
    p(f"  {'Config':>28s}  {'ok':>3s}  {'err':>7s}  {'plan':>7s}  "
      f"{'iters':>6s}  {'conv%':>5s}  {'total':>7s}")
    p(f"  {'─'*28}  {'─'*3}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*5}  {'─'*7}")

    results = []
    for horizon in horizons:
        for bname, (n_iters, lr) in budgets.items():
            t_start = time.time()
            ok, errs, details = 0, [], []
            for t in range(args.trials):
                r = run_mpc_fast(
                    model, env, s_goal, args.seed + t * 100,
                    horizon=horizon, n_iters=n_iters, lr=lr,
                    warmstart=WS, total_steps=60)
                details.append(r); errs.append(r['angle_err'])
                if r['success']: ok += 1

            mean_err = np.mean(errs)
            mean_plan = np.mean([d['mean_plan_time'] for d in details])
            mean_iters = np.mean([d['mean_iters'] for d in details])
            conv_rate = np.mean([d['conv_rate'] for d in details])
            total_t = time.time() - t_start

            label = f"H={horizon:2d}  {bname:6s}  I={n_iters}"
            status = "✓✓✓" if ok == args.trials else (">0" if ok > 0 else "✗")
            p(f"  {label:28s}  {ok:>2d}/{args.trials}  {mean_err:7.4f}  "
              f"{mean_plan:6.2f}s  {mean_iters:5.0f}  {conv_rate:.0%}  "
              f"{total_t:5.0f}s")

            results.append({
                'horizon': horizon, 'budget': bname, 'n_iters': n_iters,
                'successes': ok, 'n_trials': args.trials,
                'mean_err': mean_err, 'mean_plan_time': mean_plan,
                'mean_iters': mean_iters, 'conv_rate': conv_rate,
            })

    # ── Summary ──
    p(f"\n{'='*70}")
    p(f"SUMMARY (sorted by success rate)")
    p(f"{'='*70}")
    results.sort(key=lambda r: (-r['successes'], r['mean_err']))
    p(f"  {'Rank':>4s}  {'Config':>28s}  {'Ok':>3s}  {'Mean|Δθ|':>9s}  "
      f"{'Plan/st':>7s}  {'Iters':>6s}  {'Conv%':>5s}")
    p(f"  {'─'*4}  {'─'*28}  {'─'*3}  {'─'*9}  {'─'*7}  {'─'*6}  {'─'*5}")
    for i, r in enumerate(results):
        h_label = f"H={r['horizon']:2d} {r['budget']}"
        cr = r['conv_rate']
        p(f"  {i+1:4d}  {h_label:>28s}  "
          f"{r['successes']:>2d}/{r['n_trials']}  {r['mean_err']:9.4f}  "
          f"{r['mean_plan_time']:5.2f}s  {r['mean_iters']:5.0f}  "
          f"{cr:.0%}")

    # Per-trial detail: best + worst config
    p(f"\n{'='*70}")
    p(f"PER-TRIAL for open-loop baseline")
    p(f"{'='*70}")
    for t, r in enumerate(bl_details):
        p(f"  Trial {t+1}: init|Δθ|={r['init_err']:.3f}  "
          f"model|Δθ|={r['model_err']:.4f}  real|Δθ|={r['angle_err']:.4f}  "
          f"{'✓' if r['success'] else '✗'}  "
          f"iters={r['iters_used']}  conv={r['converged']}")

    log.close(); env.close()


if __name__ == "__main__":
    main()
