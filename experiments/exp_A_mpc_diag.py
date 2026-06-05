"""Experiment A: MPC with perfect physics model — systematic parameter sweep.

Tests whether MPC (receding-horizon control) can succeed with a perfect model.
If MPC fails even with a perfect model, the bottleneck is the MPC framework,
not KAN accuracy.

Key variables:
  - horizon: planning horizon per step
  - n_restarts: random restarts per planning
  - n_iters: Adam iterations per restart
"""
import sys, time, argparse
import torch
import gymnasium as gym
import numpy as np

G = 10.0
PI_2 = np.pi / 2
MAX_SPEED = 8.0


class PerfectPhysicsModel:
    """Exact Pendulum-v1 dynamics including max_speed clipping, fully differentiable."""

    def __init__(self):
        self.dt = 0.05
        self.max_speed = MAX_SPEED

    def __call__(self, x):
        """x: (batch, 4) = [cos, sin, thd/8, torque/2] (normalized)"""
        cos_norm = x[:, 0:1]
        sin_norm = x[:, 1:2]
        thd_norm = x[:, 2:3]
        torque_norm = x[:, 3:4]

        thd = thd_norm * 8.0
        torque = torque_norm * 2.0
        theta = torch.atan2(sin_norm, cos_norm)

        # Pendulum-v1 dynamics: exact match including max_speed clipping
        thd_new = thd + (15.0 * torch.sin(theta) + 3.0 * torque) * self.dt
        thd_new = thd_new.clamp(-self.max_speed, self.max_speed)
        theta_new = theta + thd_new * self.dt

        cos_new = torch.cos(theta_new)
        sin_new = torch.sin(theta_new)
        thd_new_norm = thd_new / 8.0

        return torch.cat([cos_new, sin_new, thd_new_norm], dim=-1)


def shoot_perfect(model, s0, s_target, horizon, n_iters, lr, lambda_ctrl, n_restarts):
    """Multi-step shooting through perfect physics model.

    Returns: (actions_raw, s_final_raw, best_loss, diag)
      actions_raw: (horizon, 1) raw torque in [-2, 2]
      s_final_raw: (1, 3) predicted final state [cos, sin, thd] raw
    """
    s0_norm = s0.clone(); s0_norm[:, 2] /= 8.0
    s_target_norm = s_target.clone(); s_target_norm[:, 2] /= 8.0

    best_loss = float('inf')
    best_actions_norm = None
    best_s_final = None
    best_restart = -1
    best_terminal = float('inf')
    best_ctrl = 0

    for restart in range(n_restarts):
        a_norm = torch.zeros(horizon, 1, requires_grad=False)
        torch.nn.init.uniform_(a_norm, a=-0.3, b=0.3)
        a_norm.requires_grad_(True)
        opt = torch.optim.Adam([a_norm], lr=lr)

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

        with torch.no_grad():
            total = loss_terminal.item() + lambda_ctrl * loss_ctrl.item()
            if total < best_loss:
                best_loss = total
                best_actions_norm = a_norm.detach().clone()
                best_restart = restart
                best_terminal = loss_terminal.item()
                best_ctrl = lambda_ctrl * loss_ctrl.item()

    # Compute predicted final state
    with torch.no_grad():
        s = s0_norm.clone()
        for h in range(horizon):
            x = torch.cat([s, best_actions_norm[h:h + 1]], dim=-1)
            s = model(x)
            norm = s[:, :2].norm(dim=-1, keepdim=True).clamp(min=1e-6)
            s = torch.cat([s[:, :2] / norm, s[:, 2:]], dim=-1)
        s_final_raw = s.clone()
        s_final_raw[:, 2] *= 8.0

    return best_actions_norm * 2.0, s_final_raw, best_loss, {
        'best_restart': best_restart,
        'terminal_loss': best_terminal,
        'ctrl_cost': best_ctrl,
    }


def run_openloop_trial(model, env, s_goal, trial_seed, horizon=30,
                       n_iters=500, n_restarts=3, lr=0.1, lambda_ctrl=0.001):
    """Run a single open-loop trial: reset env, plan, execute all actions.

    CRITICAL: env.reset() with trial_seed ensures correct initial state.
    """
    obs0, _ = env.reset(seed=trial_seed)
    s0 = torch.tensor(obs0, dtype=torch.float32).unsqueeze(0)

    t0 = time.time()
    actions, s_final_model, _, diag = shoot_perfect(
        model, s0, s_goal,
        horizon=horizon, n_iters=n_iters, lr=lr,
        lambda_ctrl=lambda_ctrl, n_restarts=n_restarts)
    plan_t = time.time() - t0

    model_err = abs(np.arctan2(s_final_model[0, 1].item(), s_final_model[0, 0].item()) - PI_2)

    # Execute open-loop from current env state (which IS obs0 because we just reset)
    obs = obs0
    for i, a in enumerate(actions.numpy().flatten()):
        obs, _, term, trunc, _ = env.step([a])
        if term or trunc:
            break

    angle_final = np.arctan2(obs[1], obs[0])
    angle_err = abs(angle_final - PI_2)
    success = angle_err < 0.2

    return {
        'success': success,
        'angle_err': angle_err,
        'model_err': model_err,
        'plan_time': plan_t,
        'init_angle': np.arctan2(obs0[1], obs0[0]),
        'init_err': abs(np.arctan2(obs0[1], obs0[0]) - PI_2),
        'final_state': obs,
        'n_actions': len(actions),
        'mean_torque': abs(actions.numpy()).mean(),
    }


def run_mpc_trial(model, env, s_goal, trial_seed, horizon=10,
                  n_iters=200, n_restarts=1, lr=0.1, lambda_ctrl=0.001,
                  total_steps=60):
    """Run a single MPC trial with env reset for correct initial state."""
    obs, _ = env.reset(seed=trial_seed)
    obs0 = obs.copy()
    plan_times = []
    best_restarts_used = []

    for step in range(total_steps):
        s_now = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        t0 = time.time()
        actions, s_final_model, _, diag = shoot_perfect(
            model, s_now, s_goal,
            horizon=horizon, n_iters=n_iters, lr=lr,
            lambda_ctrl=lambda_ctrl, n_restarts=n_restarts)
        plan_times.append(time.time() - t0)
        best_restarts_used.append(diag['best_restart'])

        obs, _, term, trunc, _ = env.step([actions[0].item()])
        if term or trunc:
            break

    angle_final = np.arctan2(obs[1], obs[0])
    angle_err = abs(angle_final - PI_2)
    success = angle_err < 0.2

    return {
        'success': success,
        'angle_err': angle_err,
        'steps_taken': len(plan_times),
        'mean_plan_time': np.mean(plan_times),
        'total_plan_time': np.sum(plan_times),
        'init_angle': np.arctan2(obs0[1], obs0[0]),
        'init_err': abs(np.arctan2(obs0[1], obs0[0]) - PI_2),
        'final_state': obs,
        'restart_usage': {r: best_restarts_used.count(r) for r in range(n_restarts)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: only test most promising configs')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = PerfectPhysicsModel()
    env = gym.make("Pendulum-v1")
    s_goal = torch.tensor([[0.0, 1.0, 0.0]])
    log = open("exp_A_mpc_log.txt", "w")

    def p(*a, **kw):
        msg = " ".join(str(x) for x in a)
        print(msg, **kw)
        log.write(msg + "\n")
        log.flush()

    # ── Config definitions ──
    if args.quick:
        configs = [
            (10, 1, 200,   "H=10 R=1 I=200  (original baseline)"),
            (10, 3, 500,   "H=10 R=3 I=500  (medium)"),
            (10, 5, 1000,  "H=10 R=5 I=1000 (high-budget)"),
            (15, 3, 500,   "H=15 R=3 I=500  (longer horizon)"),
            (20, 3, 500,   "H=20 R=3 I=500  (longest horizon)"),
            (15, 5, 1000,  "H=15 R=5 I=1000 (max budget)"),
            (30, 3, 200,   "H=30 R=3 I=200  (openloop-length, light opt)"),
            (30, 3, 500,   "H=30 R=3 I=500  (openloop-length, full opt)"),
        ]
    else:
        configs = [
            (10, 1, 200,   "H=10 R=1 I=200"),
            (10, 3, 200,   "H=10 R=3 I=200"),
            (10, 5, 200,   "H=10 R=5 I=200"),
            (10, 1, 500,   "H=10 R=1 I=500"),
            (10, 3, 500,   "H=10 R=3 I=500"),
            (10, 5, 500,   "H=10 R=5 I=500"),
            (10, 3, 1000,  "H=10 R=3 I=1000"),
            (15, 3, 500,   "H=15 R=3 I=500"),
            (20, 3, 500,   "H=20 R=3 I=500"),
            (30, 3, 200,   "H=30 R=3 I=200"),
            (30, 3, 500,   "H=30 R=3 I=500"),
        ]

    p(f"{'='*70}")
    p(f"Experiment A: MPC with Perfect Physics Model (incl. max_speed=8)")
    p(f"{args.trials} trials per config | {len(configs)} configs | seed={args.seed}")
    p(f"{'='*70}")

    # ── Baseline: Open-loop H=30 ──
    p(f"\n{'='*70}")
    p(f"BASELINE: Open-loop H=30, 500 iters, 3 restarts")
    p(f"{'='*70}")
    p(f"  {'Trial':>5s}  {'|Δθ₀|':>7s}  {'model|Δθ|':>9s}  {'real|Δθ|':>9s}  {'Result':>6s}  {'plan_time':>9s}")
    p(f"  {'─'*5}  {'─'*7}  {'─'*9}  {'─'*9}  {'─'*6}  {'─'*9}")

    baseline_ok = 0
    baseline_results = []
    for t in range(args.trials):
        trial_seed = args.seed + t * 100
        r = run_openloop_trial(model, env, s_goal, trial_seed)
        baseline_results.append(r)
        if r['success']:
            baseline_ok += 1
        p(f"  {t+1:5d}  {r['init_err']:7.3f}  {r['model_err']:9.4f}  "
          f"{r['angle_err']:9.4f}  {'✓' if r['success'] else '✗':>6s}  "
          f"{r['plan_time']:7.1f}s")
    p(f"  BASELINE OPEN-LOOP: {baseline_ok}/{args.trials}")

    # ── MPC Sweep ──
    results = []
    p(f"\n{'='*70}")
    p(f"MPC CONFIG SWEEP")
    p(f"{'='*70}")

    for cfg_idx, (horizon, n_restarts, n_iters, label) in enumerate(configs):
        t_start = time.time()
        ok = 0
        errs = []
        details = []

        for t in range(args.trials):
            trial_seed = args.seed + t * 100
            r = run_mpc_trial(
                model, env, s_goal, trial_seed,
                horizon=horizon, n_restarts=n_restarts, n_iters=n_iters,
                total_steps=60)
            details.append(r)
            errs.append(r['angle_err'])
            if r['success']:
                ok += 1

        mean_err = np.mean(errs)
        mean_plan = np.mean([d['mean_plan_time'] for d in details])
        total_t = time.time() - t_start

        status = "✓✓✓" if ok == args.trials else ("✓" if ok > 0 else "✗")
        p(f"\n  [{cfg_idx+1:2d}/{len(configs)}] {label:40s}  "
          f"{ok}/{args.trials} {status:>3s}  err={mean_err:.4f}rad  "
          f"{mean_plan:.2f}s/step  total={total_t:.0f}s")

        if ok < args.trials:
            for t, d in enumerate(details):
                if not d['success']:
                    p(f"      Trial {t+1} FAIL: |Δθ₀|={d['init_err']:.3f}  "
                      f"|Δθ_f|={d['angle_err']:.3f}rad  steps={d['steps_taken']}  "
                      f"restarts={d['restart_usage']}")

        results.append({
            'horizon': horizon, 'n_restarts': n_restarts, 'n_iters': n_iters,
            'label': label, 'successes': ok, 'n_trials': args.trials,
            'mean_err': mean_err, 'mean_plan_time': mean_plan,
        })

    # ── Summary ──
    p(f"\n{'='*70}")
    p(f"SUMMARY (sorted by success rate)")
    p(f"{'='*70}")
    results.sort(key=lambda r: (-r['successes'], r['mean_err']))

    p(f"  {'Rank':>4s}  {'Config':>45s}  {'Success':>8s}  "
      f"{'Mean|Δθ|':>9s}  {'Plan/step':>9s}")
    p(f"  {'─'*4}  {'─'*45}  {'─'*8}  {'─'*9}  {'─'*9}")
    for i, r in enumerate(results):
        p(f"  {i+1:4d}  {r['label']:>45s}  {r['successes']:>4d}/{r['n_trials']:<4d}  "
          f"{r['mean_err']:9.4f}  {r['mean_plan_time']:7.2f}s")

    # Per trial breakdown for top configs
    p(f"\n{'='*70}")
    p(f"PER-TRIAL DETAIL for best configs")
    p(f"{'='*70}")

    # Show baseline details
    p(f"\n  Open-loop baseline:")
    for t, r in enumerate(baseline_results):
        p(f"    Trial {t+1}: init|Δθ|={r['init_err']:.3f}  model|Δθ|={r['model_err']:.4f}  "
          f"real|Δθ|={r['angle_err']:.4f}  {'✓' if r['success'] else '✗'}")

    log.close()
    env.close()


if __name__ == "__main__":
    main()
