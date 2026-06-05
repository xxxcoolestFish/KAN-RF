"""Test Strategy v2 with a PERFECT physics world model.

If the strategy works with a perfect model, the bottleneck is model accuracy.
If it still fails, the bottleneck is the strategy design itself.

Physics: Pendulum-v1 exact dynamics, fully differentiable.
  theta_ddot = (3g/(2l)) * sin(theta) + 3/(m*l^2) * torque
  With m=1, l=1, g=10: theta_ddot = 15*sin(theta) + 3*torque
  dt = 0.05
"""
import sys, time, argparse, os
import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from control.strategy_v2 import compute_gap, desired_velocity, strategy_mode

_LOG = None
G = 10.0
PI_2 = np.pi / 2


def log(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    print(msg, **kwargs)
    if _LOG is not None:
        _LOG.write(msg + "\n")
        _LOG.flush()


class PerfectPhysicsModel(nn.Module):
    """Exact Pendulum-v1 dynamics as a differentiable PyTorch module.

    State: [cos(theta), sin(theta), theta_dot]
    Action: torque in [-2, 2]

    This replaces KAN entirely — 100% accurate, zero model error.
    """

    def __init__(self):
        super().__init__()
        self.g = 10.0
        self.dt = 0.05
        self.max_torque = 2.0

    def forward(self, x):
        """x: (batch, 4) = [cos, sin, thd/8, torque/2] (normalized)"""
        cos_norm = x[:, 0:1]
        sin_norm = x[:, 1:2]
        thd_norm = x[:, 2:3]
        torque_norm = x[:, 3:4]

        # Denormalize
        thd = thd_norm * 8.0
        torque = torque_norm * 2.0

        # Physics step
        theta = torch.atan2(sin_norm, cos_norm)  # current angle
        thd_new = thd + (15.0 * torch.sin(theta) + 3.0 * torque) * self.dt
        theta_new = theta + thd_new * self.dt

        # Renormalize
        cos_new = torch.cos(theta_new)
        sin_new = torch.sin(theta_new)
        thd_new_norm = thd_new / 8.0

        return torch.cat([cos_new, sin_new, thd_new_norm], dim=-1)


def gauss_newton_init_physics(model, s_norm, v_des_norm):
    """Gauss-Newton step using PerfectPhysicsModel Jacobian.

    For the physics model, we can compute the Jacobian analytically:
      dthd_new/dtorque = 3 * dt = 0.15 (raw) = 0.01875 (normalized)
      dtheta_new/dtorque ≈ 3 * dt^2 = 0.0075 rad (raw, through theta_ddot effect)

    But we use autograd for exact values.
    """
    a_zero = torch.zeros(1, 1, requires_grad=True)
    x = torch.cat([s_norm, a_zero], dim=-1)
    with torch.no_grad():
        f_zero = model(x)

    # Jacobian via autograd
    import torch.autograd.functional as AF
    f_a = lambda a_: model(torch.cat([s_norm, a_], dim=-1))
    J = AF.jacobian(f_a, a_zero)
    J_a = J.squeeze()  # (3,)

    s_target = s_norm + v_des_norm
    residual = (s_target - f_zero).squeeze(0)

    JtJ = (J_a ** 2).sum()
    JtR = (J_a * residual).sum()
    if JtJ > 1e-8:
        a_init = JtR / JtJ
    else:
        a_init = torch.tensor(0.0)

    a_init = torch.clamp(a_init, -1.0, 1.0)
    return a_init.item(), f_zero, J_a


def execute_physics(model, s, v_des, n_iter=15, lr=0.05,
                    lambda_ctrl=0.01, w_controllable=3.0):
    """Execute with perfect physics model."""
    model.eval()

    s_norm = s.clone(); s_norm[:, 2] /= 8.0
    v_des_norm = v_des.clone(); v_des_norm[:, 2] /= 8.0

    a_norm_init, f_zero, J_a = gauss_newton_init_physics(model, s_norm, v_des_norm)
    a_norm = torch.tensor([[a_norm_init]], dtype=torch.float32, requires_grad=True)

    s_target = s_norm + v_des_norm
    opt = torch.optim.Adam([a_norm], lr=lr)

    for _ in range(n_iter):
        opt.zero_grad()
        x = torch.cat([s_norm, a_norm], dim=-1)
        s_pred = model(x)
        err = s_pred - s_target
        loss = w_controllable * err[:, 2]**2 + lambda_ctrl * (a_norm ** 2).sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            a_norm.clamp_(-1.0, 1.0)

    final_loss = loss.item()
    return (a_norm.detach().item() * 2.0, final_loss, a_norm_init * 2.0)


def run_trial(model, env, s_goal, obs0, total_steps=60, verbose=1):
    """Run one trial with perfect physics model."""
    obs = obs0.copy()
    traj_real = [obs.copy()]
    traj_actions = []

    log(f"  {'Step':>4s}  {'mode':>10s}  {'a':>7s}  "
        f"{'|Δθ|':>7s}  {'E':>7s}  {'exec_loss':>9s}")
    log(f"  {'─'*4}  {'─'*10}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*9}")

    for step in range(total_steps):
        s_now = obs

        # Strategy
        gap = compute_gap(s_now)
        mode = strategy_mode(gap)
        v_des = desired_velocity(gap, mode)

        s_tensor = torch.tensor(s_now, dtype=torch.float32).unsqueeze(0)
        v_des_tensor = torch.tensor(v_des, dtype=torch.float32).unsqueeze(0)

        # Execution (with perfect model)
        a, exec_loss, a_init = execute_physics(model, s_tensor, v_des_tensor)

        # Execute in REAL environment (gym)
        obs_next, _, terminated, truncated, _ = env.step([a])

        angle_now = np.arctan2(obs_next[1], obs_next[0])
        angle_err = abs(angle_now - PI_2)
        E = 0.5 * obs_next[2]**2 + G * obs_next[1]

        traj_actions.append(a)
        traj_real.append(obs_next.copy())

        log(f"  {step:4d}  {mode:>10s}  {a:+7.3f}  "
            f"{angle_err:7.3f}  {E:+7.2f}  {exec_loss:9.6f}")

        obs = obs_next
        if terminated or truncated:
            break

    return obs, traj_real, traj_actions


def main():
    global _LOG
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    _LOG = open("eval_perfect_model_log.txt", "w")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    physics_model = PerfectPhysicsModel()
    env = gym.make("Pendulum-v1")
    s_goal = torch.tensor([[0.0, 1.0, 0.0]])

    log(f"{'='*70}")
    log(f"Strategy v2 + PERFECT Physics Model")
    log(f"{args.trials} trials × {args.steps} steps")
    log(f"{'='*70}")

    ok = 0
    all_angle_errs = []

    for trial in range(args.trials):
        obs0, _ = env.reset()
        init_angle = np.arctan2(obs0[1], obs0[0])
        init_err = abs(init_angle - PI_2)

        log(f"\n{'─'*70}")
        log(f"[Trial {trial+1}/{args.trials}]  "
            f"s0=[{obs0[0]:+.2f},{obs0[1]:+.2f},{obs0[2]:+.2f}]  "
            f"|Δθ₀|={init_err:.3f}rad")
        log(f"{'─'*70}")

        final_obs, traj, actions = run_trial(
            physics_model, env, s_goal, obs0, total_steps=args.steps)

        angle_final = np.arctan2(final_obs[1], final_obs[0])
        angle_err = abs(angle_final - PI_2)
        all_angle_errs.append(angle_err)
        success = angle_err < 0.2
        if success:
            ok += 1

        log(f"\n  >>> FINAL  "
            f"s=[{final_obs[0]:+.3f},{final_obs[1]:+.3f},{final_obs[2]:+.3f}]  "
            f"|Δθ_final|={angle_err:.3f}rad  "
            f"{'✓ SUCCESS' if success else '✗ FAIL'}")

    log(f"\n{'='*70}")
    log(f"Summary")
    log(f"{'='*70}")
    log(f"  Successes:       {ok}/{args.trials}")
    log(f"  Mean |Δθ_final|: {np.mean(all_angle_errs):.3f} rad")
    log(f"  Min  |Δθ_final|: {np.min(all_angle_errs):.3f} rad")
    log(f"  Max  |Δθ_final|: {np.max(all_angle_errs):.3f} rad")

    _LOG.close()
    env.close()


if __name__ == "__main__":
    main()
