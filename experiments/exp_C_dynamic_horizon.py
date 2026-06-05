"""Experiment C: Dynamic-horizon shooting with execution monitoring.

Key idea:
  1. Optimize H_max actions, but dynamically determine effective H:
     truncate at the first step where model-predicted state reaches goal.
  2. Record predicted states for each action in the plan.
  3. Execute one-by-one, comparing real vs predicted state.
     If deviation > threshold, replan from current state.

For perfect model: deviation always 0, so replan never triggers.
The "dynamic horizon" is the key — for near-upright starts, effective H≪H_max.

Contrast with exp_B:
  - exp_B: fixed H per plan, execute K>1, replan with fresh H budget
  - exp_C: variable H per plan (truncated at convergence), execute all, replan on deviation
"""
import sys, time, argparse
import torch
import gymnasium as gym
import numpy as np

G = 10.0; PI_2 = np.pi / 2; MAX_SPEED = 8.0


class PerfectPhysicsModel:
    def __init__(self):
        self.dt = 0.05; self.max_speed = MAX_SPEED
    def __call__(self, x):
        cn, sn, tn, un = x[:,0:1], x[:,1:2], x[:,2:3], x[:,3:4]
        thd = tn*8.0; theta = torch.atan2(sn, cn)
        thd_new = (thd + (15.0*torch.sin(theta) + 3.0*un*2.0)*self.dt).clamp(-self.max_speed, self.max_speed)
        return torch.cat([torch.cos(theta + thd_new*self.dt), torch.sin(theta + thd_new*self.dt), thd_new/8.0], dim=-1)


def shoot_with_states(model, s0, s_target, horizon=30, n_iters=500, lr=0.1,
                      lambda_ctrl=0.001, tol=1e-4, a_warm=None):
    """Shooting that returns actions, predicted states, and diagnostics.

    Returns:
        actions:  (H, 1) denormalized torques in [-2, 2]
        states:   list of (1,3) tensors [s0_norm, s1_norm, ..., sH_norm] — NORMALIZED
        diag:     dict with convergence info
    """
    s0_n = s0.clone(); s0_n[:,2] /= 8.0
    st_n = s_target.clone(); st_n[:,2] /= 8.0

    a_n = torch.zeros(horizon, 1)
    if a_warm is not None:
        with torch.no_grad():
            n_copy = min(a_n.shape[0], a_warm.shape[0])
            a_n[:n_copy] = a_warm[:n_copy]
    else:
        torch.nn.init.uniform_(a_n, a=-0.3, b=0.3)
    a_n.requires_grad_(True)
    opt = torch.optim.Adam([a_n], lr=lr)
    iters_used, converged = n_iters, False

    for step in range(n_iters):
        opt.zero_grad()
        s = s0_n.clone()
        for h in range(horizon):
            x = torch.cat([s, a_n[h:h+1]], dim=-1); s = model(x)
            nrm = s[:,:2].norm(dim=-1,keepdim=True).clamp(min=1e-6)
            s = torch.cat([s[:,:2]/nrm, s[:,2:]], dim=-1)
        loss = ((s-st_n)**2).sum() + lambda_ctrl*(a_n**2).sum()
        loss.backward(); opt.step()
        with torch.no_grad(): a_n.clamp_(-1.0, 1.0)
        if ((s-st_n)**2).sum().item() < tol:
            converged = True; iters_used = step+1; break

    # Forward rollout to collect predicted states
    with torch.no_grad():
        predicted_states = [s0_n.clone()]  # s0_n
        s = s0_n.clone()
        for h in range(horizon):
            x = torch.cat([s, a_n[h:h+1]], dim=-1); s = model(x)
            nrm = s[:,:2].norm(dim=-1,keepdim=True).clamp(min=1e-6)
            s = torch.cat([s[:,:2]/nrm, s[:,2:]], dim=-1)
            predicted_states.append(s.clone())

        # Denormalize final state for reporting
        sf = s.clone(); sf[:,2] *= 8.0

    return (a_n.detach()*2.0, predicted_states,
            {'iters': iters_used, 'conv': converged,
             'term_loss': ((s-st_n)**2).sum().item()})


def truncate_at_convergence(actions, predicted_states, s_target, tol=0.005):
    """Find the earliest step where model predicts goal reached.

    Truncates actions and states at that point.
    Returns (truncated_actions, truncated_states, effective_H).
    """
    st_n = s_target.clone(); st_n[:,2] /= 8.0
    for h in range(1, len(predicted_states)):
        s_h = predicted_states[h]
        loss = ((s_h - st_n) ** 2).sum().item()
        if loss < tol:
            return actions[:h], predicted_states[:h+1], h
    return actions, predicted_states, len(actions)


def deviation(s_real, s_pred_norm):
    """Compute ||real - predicted|| in normalized space."""
    s_real_norm = torch.tensor(s_real, dtype=torch.float32).unsqueeze(0)
    s_real_norm[0, 2] /= 8.0
    return (s_real_norm - s_pred_norm).norm().item()


def run_dynamic(model, env, s_goal, trial_seed, H_max=30,
                total_steps=60, n_iters=500, dev_threshold=0.15):
    """Dynamic-horizon + execution monitoring.

    1. Plan H_max actions, truncate at effective horizon.
    2. Execute one-by-one, check ||s_real - s_pred|| each step.
    3. Replan if deviation > threshold (not triggered for perfect model).
    """
    obs, _ = env.reset(seed=trial_seed)
    step_count = 0
    cycles = 0
    actions_taken = []
    all_eff_H = []
    deviations = []

    while step_count < total_steps:
        s_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        actions, predicted_states, diag = shoot_with_states(
            model, s_t, s_goal, horizon=H_max, n_iters=n_iters)

        # Truncate: use only actions up to first goal-reaching step
        actions_trunc, pred_states_trunc, eff_H = truncate_at_convergence(
            actions, predicted_states, s_goal)
        all_eff_H.append(eff_H)
        cycles += 1

        # Execute one-by-one with monitoring
        replan = False
        for i, a in enumerate(actions_trunc):
            if step_count >= total_steps:
                break
            obs_next, _, term, trunc, _ = env.step([a.item()])
            actions_taken.append(a.item())
            step_count += 1

            # Check deviation from predicted state
            s_pred_i1 = pred_states_trunc[i + 1]
            dev = deviation(obs_next, s_pred_i1)
            deviations.append(dev)

            if dev > dev_threshold:
                replan = True
                obs = obs_next
                break

            obs = obs_next
            # Exit if goal reached (Pendulum-v1 doesn't auto-terminate)
            if abs(np.arctan2(obs[1], obs[0]) - PI_2) < 0.2:
                term = True
                break
            if term or trunc:
                break

        if term:
            break
        if replan:
            continue
        if term or trunc:
            break

    angle_final = np.arctan2(obs[1], obs[0])
    angle_err = abs(angle_final - PI_2)
    return {
        'success': angle_err < 0.2, 'angle_err': angle_err,
        'steps_taken': step_count, 'cycles': cycles,
        'mean_eff_H': np.mean(all_eff_H), 'eff_H_list': all_eff_H,
        'max_deviation': max(deviations) if deviations else 0,
        'mean_deviation': np.mean(deviations) if deviations else 0,
    }


def run_openloop(model, env, s_goal, trial_seed, horizon=30, n_iters=500):
    """Baseline: fixed open-loop H=30."""
    obs0, _ = env.reset(seed=trial_seed)
    s0 = torch.tensor(obs0, dtype=torch.float32).unsqueeze(0)
    t0 = time.time()
    actions, predicted_states, diag = shoot_with_states(
        model, s0, s_goal, horizon=horizon, n_iters=n_iters)
    plan_t = time.time() - t0

    obs = obs0
    for a in actions.numpy().flatten():
        obs, _, t, tr, _ = env.step([a])
        if t or tr: break
    err = abs(np.arctan2(obs[1], obs[0]) - PI_2)
    return {
        'success': err < 0.2, 'angle_err': err, 'plan_t': plan_t,
        'n_steps': len(actions), 'iters': diag['iters'], 'conv': diag['conv'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--h-max', type=int, default=30)
    parser.add_argument('--dev-threshold', type=float, default=0.15)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = PerfectPhysicsModel()
    env = gym.make("Pendulum-v1")
    s_goal = torch.tensor([[0.0, 1.0, 0.0]])

    def p(*a):
        print(" ".join(str(x) for x in a))

    p("=" * 80)
    p("Experiment C: Dynamic-Horizon Shooting + Execution Monitoring")
    p(f"Perfect Physics Model  |  {args.trials} trials  |  seed={args.seed}")
    p(f"H_max={args.h_max}  |  dev_threshold={args.dev_threshold}")
    p("=" * 80)

    methods = [
        ("Open-loop H=30", lambda seed: run_openloop(
            model, env, s_goal, seed, horizon=30, n_iters=500)),
        ("Dynamic (ours)", lambda seed: run_dynamic(
            model, env, s_goal, seed, H_max=args.h_max,
            n_iters=500, dev_threshold=args.dev_threshold)),
    ]

    all_results = []
    for name, method in methods:
        p(f"\n{'─' * 80}")
        p(f"  {name}")
        if "Dynamic" in name:
            p(f"  {'Trial':>4s}  {'|Δθ₀|':>7s}  {'|Δθ_f|':>9s}  {'R':>5s}  "
              f"{'eff_H':>6s}  {'max_dev':>8s}  {'cycles':>6s}")
            p(f"  {'─'*4}  {'─'*7}  {'─'*9}  {'─'*5}  {'─'*6}  {'─'*8}  {'─'*6}")
        else:
            p(f"  {'Trial':>4s}  {'|Δθ₀|':>7s}  {'|Δθ_f|':>9s}  {'R':>5s}  "
              f"{'iters':>5s}  {'t':>6s}")
            p(f"  {'─'*4}  {'─'*7}  {'─'*9}  {'─'*5}  {'─'*5}  {'─'*6}")

        ok = 0; errs = []
        for t in range(args.trials):
            trial_seed = args.seed + t * 100
            r = method(trial_seed)
            errs.append(r['angle_err'])
            if r['success']: ok += 1

            if "Dynamic" in name:
                p(f"  {t+1:4d}  {0.0:7.3f}  {r['angle_err']:9.4f}  "
                  f"{'✓' if r['success'] else '✗':>5s}  "
                  f"{r['mean_eff_H']:5.1f}  {r['max_deviation']:8.4f}  "
                  f"{r['cycles']:6d}")
            else:
                p(f"  {t+1:4d}  {0.0:7.3f}  {r['angle_err']:9.4f}  "
                  f"{'✓' if r['success'] else '✗':>5s}  "
                  f"{r['iters']:5d}  {r['plan_t']:4.1f}s")

        p(f"  => {ok}/{args.trials}  mean_err={np.mean(errs):.4f}rad")
        all_results.append({'name': name, 'ok': ok, 'mean_err': np.mean(errs),
                           'n_trials': args.trials})

    p(f"\n{'=' * 80}")
    p(f"SUMMARY")
    p(f"{'=' * 80}")
    p(f"  {'Method':>25s}  {'Success':>8s}  {'Mean|Δθ|':>9s}")
    p(f"  {'─'*25}  {'─'*8}  {'─'*9}")
    for r in all_results:
        p(f"  {r['name']:>25s}  {r['ok']:>4d}/{r['n_trials']:<4d}  {r['mean_err']:9.4f}")

    env.close()


if __name__ == "__main__":
    main()
