"""Test KAN-iLQR on Pendulum: local linear model from KAN + iLQR + env line search."""
import torch, numpy as np, sys, os, time
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import KAN
from control.kan_knowledge import KANKnowledge
from control.kan_ilqr import kan_ilqr
from experiments.test_solution1 import pendulum_rollout, pendulum_step

PI_2 = np.pi / 2
G_VAL = 10.0


class KANiLQRController:
    """Controller using KAN's local linear model + iLQR planning."""

    def __init__(self, kan, horizon=8, device='cpu'):
        self.kk = KANKnowledge(kan, device=device)
        self.horizon = horizon
        self.s_target = np.array([0.0, 1.0, 0.0])  # upright
        self.g = G_VAL
        self.prev_action = 0.0

    def set_gravity(self, g):
        self.g = g

    def get_action(self, obs):
        s_norm = np.array([obs[0], obs[1], obs[2]/8.0], dtype=np.float32)
        sin_theta = s_norm[1]

        # Bottom: simple energy heuristic
        if sin_theta < -0.3:
            thd = obs[2]
            E = 0.5*thd**2 + self.g*sin_theta
            a = np.clip(1.5*(E-self.g)*thd/10.0, -1.0, 1.0)
            self.prev_action = a
            return a, {'method': 'energy'}

        # KAN-Jacobian pseudo-inverse + env rollout for verification
        # Use KAN's B (accurate) + analytical A (computed via finite diff on env)
        # Simplified: just use KAN B pseudoinverse as action, verify with env
        s_t = torch.tensor(s_norm, dtype=torch.float32, device=self.kk.device).unsqueeze(0)
        J = self.kk.jacobian(s_t).cpu().squeeze().numpy()
        state_err = self.s_target - s_norm
        J_norm = float(np.dot(J, J)) + 1e-4
        a = float(np.dot(J, state_err)) / J_norm

        # Verify with env: try a, a+eps, a-eps, pick best
        eps = 0.05
        candidates = [a, a+eps, a-eps]
        best_cost = float('inf'); best_a = a
        for cand in candidates:
            cand = np.clip(cand, -1.0, 1.0)
            s1 = pendulum_step(np.array([obs[0], obs[1], obs[2]]), cand, g=self.g)
            s1_norm = s1.copy(); s1_norm[2] /= 8.0
            cost = np.sum((s1_norm - self.s_target)**2) + 0.01*cand**2
            if cost < best_cost:
                best_cost = cost; best_a = cand

        self.prev_action = best_a
        return best_a, {'method': 'kan_ilqr'}


def test_standard(kan_path, n_trials=10, horizon=8):
    """Standard Pendulum test with KAN-iLQR."""
    device = torch.device('cpu')
    kan = KAN([4, 12, 3], grid_size=5, spline_order=3)
    kan.load_state_dict(torch.load(kan_path, weights_only=True, map_location=device))
    kan.to(device); kan.eval()

    ctrl = KANiLQRController(kan, horizon=horizon, device=device)
    env = gym.make('Pendulum-v1')

    successes = 0; all_steps = []; all_errors = []
    for trial in range(n_trials):
        seed = 42 + trial*100; obs, _ = env.reset(seed=seed)
        for step in range(300):
            a_norm, info = ctrl.get_action(obs)
            a_raw = a_norm * 2.0
            obs, _, term, trunc, _ = env.step([a_raw])
            err = min(abs(np.arctan2(obs[1], obs[0]) - PI_2),
                     2*np.pi - abs(np.arctan2(obs[1], obs[0]) - PI_2))
            if err < 0.2:
                successes += 1; all_steps.append(step+1); all_errors.append(err); break
        else:
            all_steps.append(300)
            err = min(abs(np.arctan2(obs[1], obs[0]) - PI_2),
                     2*np.pi - abs(np.arctan2(obs[1], obs[0]) - PI_2))
            all_errors.append(err)
        print(f"  Trial {trial+1:2d}  steps={all_steps[-1]:3d}  "
              f"err={all_errors[-1]:.3f}  {'✓' if all_errors[-1]<0.2 else '✗'}  "
              f"method={info.get('method','?')}")

    env.close()
    sr = successes/n_trials
    print(f"\n  Success: {successes}/{n_trials} ({sr*100:.0f}%)  "
          f"mean|Δθ|={np.mean(all_errors):.3f}±{np.std(all_errors):.3f}")
    return sr


def test_stabilization(kan_path, horizon=8, g=10.0, n_steps=500):
    """Test if KAN-iLQR can maintain upright indefinitely."""
    device = torch.device('cpu')
    kan = KAN([4, 12, 3], grid_size=5, spline_order=3)
    kan.load_state_dict(torch.load(kan_path, weights_only=True, map_location=device))
    kan.to(device); kan.eval()

    ctrl = KANiLQRController(kan, horizon=horizon, device=device)
    ctrl.set_gravity(g)

    # Start near upright
    s_raw = np.array([0.0, 1.0, 0.2])  # slight perturbation
    errors_deg = []

    for step in range(n_steps):
        s_norm = s_raw.copy(); s_norm[2] /= 8.0
        a_norm, info = ctrl.get_action(s_raw)
        s_raw = pendulum_step(s_raw, a_norm, g=g)

        err = min(abs(np.arctan2(s_raw[1], s_raw[0]) - PI_2),
                 2*np.pi - abs(np.arctan2(s_raw[1], s_raw[0]) - PI_2))
        errors_deg.append(np.rad2deg(err))

        if step < 5 or step % 100 == 0:
            print(f"  Step {step:3d}  |Δθ|={np.rad2deg(err):.2f}deg  "
                  f"a={a_norm:+.3f}  method={info.get('method','?')}")

        if err > 1.5:  # fell over
            print(f"  FAILED at step {step}: |Δθ|={np.rad2deg(err):.1f}deg")
            break

    print(f"\n  Survived {step+1}/{n_steps} steps")
    print(f"  Mean |Δθ|: {np.mean(errors_deg):.2f} ± {np.std(errors_deg):.2f} deg")
    return step+1 >= n_steps


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--kan', type=str, default='/tmp/kanrf_cl_exp/kan_cws_cl.pt')
    parser.add_argument('--mode', type=str, default='stabilize', choices=['test', 'stabilize'])
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--horizon', type=int, default=8)
    parser.add_argument('--g', type=float, default=10.0)
    parser.add_argument('--steps', type=int, default=500)
    args = parser.parse_args()

    if args.mode == 'stabilize':
        test_stabilization(args.kan, args.horizon, args.g, args.steps)
    else:
        test_standard(args.kan, args.trials, args.horizon)
