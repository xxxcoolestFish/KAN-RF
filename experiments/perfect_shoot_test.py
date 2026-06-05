"""Test multi-step shooting with PERFECT physics world model.

This isolates: if model is 100% accurate, can multi-step shooting succeed?
"""
import sys, time, argparse
import torch
import gymnasium as gym
import numpy as np

G = 10.0
PI_2 = np.pi / 2


class PerfectPhysicsModel:
    """Exact Pendulum-v1 dynamics, differentiable.

    theta_ddot = 15*sin(theta) + 3*torque
    dt = 0.05
    """

    def __init__(self):
        self.dt = 0.05

    def __call__(self, x):
        """x: (batch, 4) = [cos, sin, thd/8, torque/2]"""
        cos_norm = x[:, 0:1]
        sin_norm = x[:, 1:2]
        thd_norm = x[:, 2:3]
        torque_norm = x[:, 3:4]

        thd = thd_norm * 8.0
        torque = torque_norm * 2.0
        theta = torch.atan2(sin_norm, cos_norm)

        thd_new = thd + (15.0 * torch.sin(theta) + 3.0 * torque) * self.dt
        theta_new = theta + thd_new * self.dt

        cos_new = torch.cos(theta_new)
        sin_new = torch.sin(theta_new)
        thd_new_norm = thd_new / 8.0

        return torch.cat([cos_new, sin_new, thd_new_norm], dim=-1)


def shoot_perfect(model, s0, s_target, horizon=30, n_iters=500, lr=0.1,
                  lambda_ctrl=0.001, n_restarts=3, verbose=True):
    """Multi-step shooting through perfect physics model."""
    s0_norm = s0.clone(); s0_norm[:, 2] /= 8.0
    s_target_norm = s_target.clone(); s_target_norm[:, 2] /= 8.0

    best_loss = float('inf')
    best_actions_norm = None

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

            if verbose and step % 100 == 0:
                with torch.no_grad():
                    sc = s0_norm.clone()
                    for h in range(horizon):
                        sc = model(torch.cat([sc, a_norm[h:h+1]], dim=-1))
                        nrm = sc[:, :2].norm(dim=-1, keepdim=True).clamp(min=1e-6)
                        sc = torch.cat([sc[:, :2] / nrm, sc[:, 2:]], dim=-1)
                    angle_err = torch.acos(
                        (sc[:, :2] * s_target_norm[:, :2]).sum(-1).clamp(-1, 1)).item()
                print(f"    restart {restart+1}/{n_restarts}  "
                      f"iter {step:4d}  loss={loss_terminal.item():.4f}+{lambda_ctrl*loss_ctrl.item():.4f}  "
                      f"model|Δθ|={angle_err:.3f}rad  (terminal prediction)")

        with torch.no_grad():
            total = loss_terminal.item() + lambda_ctrl * loss_ctrl.item()
        if total < best_loss:
            best_loss = total
            best_actions_norm = a_norm.detach().clone()

    actions_raw = best_actions_norm * 2.0

    # Predict final state with best actions
    with torch.no_grad():
        s = s0_norm.clone()
        for h in range(horizon):
            x = torch.cat([s, best_actions_norm[h:h + 1]], dim=-1)
            s = model(x)
            norm = s[:, :2].norm(dim=-1, keepdim=True).clamp(min=1e-6)
            s = torch.cat([s[:, :2] / norm, s[:, 2:]], dim=-1)
        s_final_norm = s.clone()
        s_final_norm[:, 2] *= 8.0

    return actions_raw, s_final_norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--horizon', type=int, default=30)
    parser.add_argument('--n-iters', type=int, default=500)
    parser.add_argument('--restarts', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', choices=['openloop', 'mpc'], default='openloop')
    parser.add_argument('--mpc-horizon', type=int, default=10)
    parser.add_argument('--mpc-iters', type=int, default=200)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = PerfectPhysicsModel()
    env = gym.make("Pendulum-v1")
    s_goal = torch.tensor([[0.0, 1.0, 0.0]])

    print(f"{'='*70}")
    print(f"Multi-step Shooting + PERFECT Physics Model")
    print(f"Mode: {args.mode}  |  {args.trials} trials  |  H={args.horizon}")
    print(f"{'='*70}")

    t_start = time.time()
    ok = 0
    angle_errs = []

    for trial in range(args.trials):
        obs0, _ = env.reset()
        s0 = torch.tensor(obs0, dtype=torch.float32).unsqueeze(0)
        init_angle = np.arctan2(s0[0, 1].item(), s0[0, 0].item())
        init_err = abs(init_angle - PI_2)

        print(f"\n[Trial {trial+1}/{args.trials}]  "
              f"s0=[{s0[0,0]:+.2f},{s0[0,1]:+.2f},{s0[0,2]:+.2f}]  "
              f"|Δθ₀|={init_err:.3f}rad")

        if args.mode == 'openloop':
            # Plan all H steps, execute open-loop
            t0 = time.time()
            actions, s_final_model = shoot_perfect(
                model, s0, s_goal, horizon=args.horizon,
                n_iters=args.n_iters, n_restarts=args.restarts)
            plan_time = time.time() - t0

            model_err = abs(np.arctan2(
                s_final_model[0, 1].item(), s_final_model[0, 0].item()) - PI_2)
            print(f"  Model predicts: |Δθ_final|={model_err:.4f}rad  plan_time={plan_time:.0f}s")

            # Execute from the SAME initial state used for planning.
            # env is already at obs0 (from the reset() above); just step.
            obs = obs0
            for a in actions.numpy().flatten():
                obs, _, _, _, _ = env.step([a])

        else:
            # MPC: plan H_mpc, execute first, replan
            obs = obs0  # use same initial state
            total_plan = 0
            for step in range(args.horizon):
                s_now = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                t0 = time.time()
                actions, _ = shoot_perfect(
                    model, s_now, s_goal, horizon=args.mpc_horizon,
                    n_iters=args.mpc_iters, n_restarts=1, verbose=False)
                total_plan += time.time() - t0
                obs, _, terminated, truncated, _ = env.step([actions[0].item()])
                if terminated or truncated:
                    break
            plan_time = total_plan

        angle_final = np.arctan2(obs[1], obs[0])
        angle_err = abs(angle_final - PI_2)
        angle_errs.append(angle_err)
        success = angle_err < 0.2
        if success:
            ok += 1

        print(f"  REAL final: s=[{obs[0]:+.3f},{obs[1]:+.3f},{obs[2]:+.3f}]  "
              f"|Δθ_final|={angle_err:.3f}rad  {'✓' if success else '✗'}")

    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"  Successes:       {ok}/{args.trials}")
    print(f"  Mean |Δθ_final|: {np.mean(angle_errs):.3f} rad")
    print(f"  Total time:      {time.time() - t_start:.0f}s")

    env.close()


if __name__ == "__main__":
    main()
