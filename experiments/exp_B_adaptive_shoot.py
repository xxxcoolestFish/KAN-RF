"""Experiment B: Adaptive-Horizon Staggered Open-Loop Shooting.

Key idea:
  - H adapts to energy deficit: far from goal → long horizon (full resonance)
                             near goal → short horizon (fine-tuning)
  - Execute K steps from the SAME plan before replanning (preserves coherence)
  - K = max(2, H // 3): keep plan coherence while allowing feedback correction

Compares:
  1. Open-loop H=30 (baseline — works with perfect model)
  2. Pure MPC H=10 (baseline — known to fail)
  3. Adaptive staggered (our method)
  4. Fixed-horizon staggered (H=30, K=5 as ablation)

All with perfect physics model.
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


def shoot(model, s0, s_target, horizon, n_iters=500, lr=0.1, lambda_ctrl=0.001,
          tol=1e-4, a_warm=None):
    """Shooting with warm-start + early-stop. Returns (actions_raw, final_state, diag)."""
    s0_n = s0.clone(); s0_n[:,2]/=8.0
    st_n = s_target.clone(); st_n[:,2]/=8.0

    a_n = torch.zeros(horizon, 1)
    if a_warm is not None:
        with torch.no_grad(): n_copy = min(a_n.shape[0], a_warm.shape[0]); a_n[:n_copy] = a_warm[:n_copy]
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

    with torch.no_grad():
        s = s0_n.clone()
        for h in range(horizon):
            x = torch.cat([s, a_n[h:h+1]], dim=-1); s = model(x)
            nrm = s[:,:2].norm(dim=-1,keepdim=True).clamp(min=1e-6)
            s = torch.cat([s[:,:2]/nrm, s[:,2:]], dim=-1)
        sf = s.clone(); sf[:,2]*=8.0

    return a_n.detach()*2.0, sf, {'iters': iters_used, 'conv': converged,
                                   'term_loss': ((s-st_n)**2).sum().item()}


def warm_shift(actions_prev, horizon, shift=1):
    """Shift previous actions by `shift` steps, append zeros."""
    a_new = torch.zeros(horizon, 1)
    n = min(actions_prev.shape[0] - shift, horizon)
    if n > 0:
        a_new[:n, 0] = actions_prev[shift:shift + n, 0]
    return a_new


# ─── Adaptive Horizon ───────────────────────────────────────────────────

def adaptive_horizon(gap, H_min=3, H_max=30):
    """Horizon scales with energy deficit. Big gap → long plan, small gap → short plan."""
    delta_E = abs(gap['delta_E'])
    fraction = min(delta_E / 15.0, 1.0)
    return max(H_min, int(H_min + fraction * (H_max - H_min)))


def compute_gap(s):
    cos_th, sin_th, thd = s
    E = 0.5*thd*thd + G*sin_th
    delta_E = G - E
    angle = np.arctan2(sin_th, cos_th)
    d_pos = angle - PI_2
    d_pos = (d_pos + np.pi) % (2*np.pi) - np.pi
    near = abs(cos_th) < 0.5 and sin_th > 0 and abs(thd) < 3.0
    return {'delta_E': delta_E, 'E': E, 'd_pos': d_pos, 'near_upright': near}


# ─── Controllers ────────────────────────────────────────────────────────

def run_openloop(model, env, s_goal, trial_seed, horizon=30):
    obs0, _ = env.reset(seed=trial_seed)
    s0 = torch.tensor(obs0, dtype=torch.float32).unsqueeze(0)
    t0 = time.time()
    actions, sf, diag = shoot(model, s0, s_goal, horizon=horizon)
    plan_t = time.time()-t0
    obs = obs0
    for a in actions.numpy().flatten():
        obs, _, t, tr, _ = env.step([a])
        if t or tr: break
    err = abs(np.arctan2(obs[1], obs[0])-PI_2)
    return {'success': err<0.2, 'err': err, 'plan_t': plan_t, 'n_steps': len(actions),
            'iters': diag['iters'], 'conv': diag['conv']}


def run_mpc(model, env, s_goal, trial_seed, horizon=10, total_steps=60, n_iters=200):
    obs, _ = env.reset(seed=trial_seed)
    t0 = time.time()
    a_warm = None
    iters_used = []
    for step in range(total_steps):
        s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        actions, _, diag = shoot(model, s, s_goal, horizon=horizon,
                                 n_iters=n_iters, a_warm=a_warm)
        a_warm = warm_shift(actions/2.0, horizon)
        iters_used.append(diag['iters'])
        obs, _, t, tr, _ = env.step([actions[0].item()])
        if t or tr: break
    err = abs(np.arctan2(obs[1], obs[0])-PI_2)
    return {'success': err<0.2, 'err': err, 'steps_taken': len(iters_used),
            'plan_t': (time.time()-t0)/max(1,len(iters_used)),
            'mean_iters': np.mean(iters_used)}


def run_adaptive_staggered(model, env, s_goal, trial_seed, total_steps=60,
                           H_min=3, H_max=30, K_frac=4, n_iters=500):
    """Adaptive horizon + staggered execution.

    Each cycle: compute gap → adaptive H → plan H steps → execute K = max(2, H//K_frac) steps → replan
    """
    obs, _ = env.reset(seed=trial_seed)
    t0 = time.time()
    step_count = 0
    cycles = 0
    actions_taken = []
    a_warm = None
    all_H = []

    while step_count < total_steps:
        s = obs
        gap = compute_gap(s)
        H = adaptive_horizon(gap, H_min=H_min, H_max=H_max)
        K = max(2, H // K_frac)
        all_H.append(H)

        s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        actions, _, diag = shoot(model, s_t, s_goal, horizon=H,
                                 n_iters=n_iters, a_warm=a_warm)
        # Warm-start: remaining actions shifted by K (executed count)
        if K < H:
            a_warm = warm_shift(actions/2.0, H, shift=K)
        else:
            a_warm = None

        # Execute K steps from SAME plan
        for k in range(K):
            if step_count >= total_steps:
                break
            a = actions[k].item() if k < len(actions) else 0.0
            obs, _, t, tr, _ = env.step([a])
            actions_taken.append(a)
            step_count += 1
            if t or tr:
                break
        cycles += 1
        if t or tr:
            break

    err = abs(np.arctan2(obs[1], obs[0])-PI_2)
    return {'success': err<0.2, 'err': err, 'steps_taken': step_count,
            'cycles': cycles, 'mean_H': np.mean(all_H),
            'plan_t': (time.time()-t0)/max(1,step_count),
            'mean_plan_len': np.mean(all_H)}


def run_fixed_staggered(model, env, s_goal, trial_seed, H=30, K=5,
                        total_steps=60, n_iters=500):
    """Fixed-horizon staggered open-loop (ablation to isolate adaptive benefit)."""
    obs, _ = env.reset(seed=trial_seed)
    t0 = time.time()
    step_count = 0
    a_warm = None

    while step_count < total_steps:
        s_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        actions, _, diag = shoot(model, s_t, s_goal, horizon=H,
                                 n_iters=n_iters, a_warm=a_warm)
        a_warm = warm_shift(actions/2.0, H, shift=K)
        for k in range(K):
            if step_count >= total_steps: break
            obs, _, t, tr, _ = env.step([actions[k].item()])
            step_count += 1
            if t or tr: break

    err = abs(np.arctan2(obs[1], obs[0])-PI_2)
    return {'success': err<0.2, 'err': err, 'steps_taken': step_count,
            'plan_t': (time.time()-t0)/max(1,step_count)}


# ─── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = PerfectPhysicsModel()
    env = gym.make("Pendulum-v1")
    s_goal = torch.tensor([[0.0, 1.0, 0.0]])

    def p(*a):
        msg = " ".join(str(x) for x in a); print(msg)

    p("="*80)
    p("Experiment B: Adaptive-Horizon Staggered Open-Loop Shooting")
    p(f"Perfect Physics Model  |  {args.trials} trials  |  seed={args.seed}")
    p("="*80)

    methods = [
        ("Open-loop H=30", lambda seed: run_openloop(model, env, s_goal, seed, horizon=30)),
        ("MPC H=10",       lambda seed: run_mpc(model, env, s_goal, seed, horizon=10, n_iters=200)),
        ("Adaptive-K/4",   lambda seed: run_adaptive_staggered(model, env, s_goal, seed,
                                                               K_frac=4, n_iters=500)),
        ("Adaptive-K/5",   lambda seed: run_adaptive_staggered(model, env, s_goal, seed,
                                                               K_frac=5, n_iters=500)),
        ("Fixed H=30 K=5", lambda seed: run_fixed_staggered(model, env, s_goal, seed,
                                                            H=30, K=5, n_iters=500)),
        ("Fixed H=30 K=8", lambda seed: run_fixed_staggered(model, env, s_goal, seed,
                                                            H=30, K=8, n_iters=500)),
    ]

    all_results = []
    for name, method in methods:
        p(f"\n{'─'*80}")
        p(f"  {name}")
        p(f"  {'Trial':>4s}  {'|Δθ₀|':>7s}  {'|Δθ_f|':>9s}  {'R':>5s}  {'t/step':>7s}  {'detail'}")
        p(f"  {'─'*4}  {'─'*7}  {'─'*9}  {'─'*5}  {'─'*7}  {'─'*40}")

        ok = 0; errs = []
        for t in range(args.trials):
            trial_seed = args.seed + t*100
            r = method(trial_seed)
            errs.append(r['err'])
            if r['success']: ok += 1

            detail = ""
            if 'cycles' in r:
                detail = f"H_mean={r['mean_H']:.1f}  cycles={r['cycles']}"
            if 'mean_iters' in r:
                detail = f"iters={r['mean_iters']:.0f}"
            p(f"  {t+1:4d}  {0.0:7.3f}  {r['err']:9.4f}  "
              f"{'✓' if r['success'] else '✗':>5s}  {r['plan_t']:5.2f}s  {detail}")

        p(f"  => {ok}/{args.trials}  mean_err={np.mean(errs):.4f}rad")

        all_results.append({'name': name, 'ok': ok, 'mean_err': np.mean(errs),
                           'n_trials': args.trials})

    p(f"\n{'='*80}")
    p(f"SUMMARY")
    p(f"{'='*80}")
    p(f"  {'Method':>25s}  {'Success':>8s}  {'Mean|Δθ|':>9s}")
    p(f"  {'─'*25}  {'─'*8}  {'─'*9}")
    for r in all_results:
        p(f"  {r['name']:>25s}  {r['ok']:>4d}/{r['n_trials']:<4d}  {r['mean_err']:9.4f}")

    env.close()


if __name__ == "__main__":
    main()
