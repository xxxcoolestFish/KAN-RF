"""Test KAN-adapted decision network (v3) on Pendulum-v1.

Deployment: s → π_θ → a  (pure forward pass, KAN not involved).

Compares against baselines:
  1. v3 policy (KAN gradient trained)
  2. Behavioral cloning baseline (same architecture, BC from action explorer data)
  3. Inverse optimization (KAN model-based, for reference)
  4. Energy controller (oracle, 10/10 upper bound)

Usage:
  python test_pendulum.py --policy kan_policy_v3.pt --trials 10
  python test_pendulum.py --policy kan_policy_v3.pt --compare-all
"""
import sys, os, argparse, time
import torch
import numpy as np
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decision_v3.core import KANPolicy, ResidualPhysicsPolicy

PI_2 = np.pi / 2
MAX_STEPS = 300
SUCCESS_THRESH = 0.2  # radians from upright


# ═══════════════════════════════════════════════════════════════════════════════
# Controllers
# ═══════════════════════════════════════════════════════════════════════════════

class V3PolicyController:
    """decision_v3 trained policy: s → π_θ → a (or a, k for multi-scale)."""

    def __init__(self, policy, device='cpu', multi_scale=False):
        self.policy = policy.to(device)
        self.device = device
        self.multi_scale = multi_scale
        self.policy.eval()

    def get_action(self, obs):
        s = torch.tensor([[obs[0], obs[1], obs[2] / 8.0]],
                         dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.policy(s)
            if self.multi_scale:
                a_norm = out[0, 0].item()
                k = max(1, min(16, round(out[0, 1].item() * 16)))
                return np.clip(a_norm * 2.0, -2.0, 2.0), k
            else:
                a_norm = out.item() if out.numel() == 1 else out[0, 0].item()
                return np.clip(a_norm * 2.0, -2.0, 2.0)


class EnergyController:
    """Oracle energy-based controller for Pendulum swing-up + stabilize.

    E_des = m*g*l = 10.0 (upright at rest, in Gym default params)
    u = k_swing * (E - E_des) * θ̇  (energy shaping)
    When near upright: switch to LQR stabilization.

    This is the theoretical optimal strategy and should achieve 10/10.
    """

    def __init__(self, k_swing=1.5, k_stable=5.0, k_damp=1.0):
        self.k_swing = k_swing
        self.k_stable = k_stable
        self.k_damp = k_damp
        self.G = 10.0

    def get_action(self, obs):
        cos_th, sin_th, thd = obs
        E = 0.5 * thd * thd + self.G * sin_th
        E_des = self.G

        near_upright = abs(cos_th) < 0.5 and sin_th > 0 and abs(thd) < 3.0

        if near_upright:
            # LQR stabilization
            angle = np.arctan2(sin_th, cos_th)
            angle_err = angle - PI_2
            angle_err = (angle_err + np.pi) % (2 * np.pi) - np.pi
            u = -self.k_stable * angle_err - self.k_damp * thd
        else:
            # Energy shaping swing-up
            u = self.k_swing * (E - E_des) * thd

        return np.clip(u, -2.0, 2.0)


class RandomController:
    """Random actions (lower bound)."""
    def get_action(self, obs):
        return np.random.uniform(-2.0, 2.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def run_trial(controller, env, trial_seed, max_steps=MAX_STEPS, verbose=False):
    """Run one trial. Returns (success, n_steps, final_error, trajectory).

    Supports both single-action controllers (return a) and multi-scale
    controllers (return (a, k) where k is execution duration in steps).
    """
    obs, _ = env.reset(seed=trial_seed)
    traj = [obs.copy()]
    total_steps = 0

    for _ in range(max_steps):
        result = controller.get_action(obs)
        if isinstance(result, tuple):
            a, k = result
            k = min(k, 12)  # cap execution duration
        else:
            a, k = result, 1

        for _ in range(k):
            obs, _, term, trunc, _ = env.step([a])
            traj.append(obs.copy())
            total_steps += 1

            if term or trunc:
                break

            # Check if upright
            angle = np.arctan2(obs[1], obs[0])
            err = abs(angle - PI_2)
            err = min(err, 2 * np.pi - err)
            if err < SUCCESS_THRESH:
                break

        if term or trunc:
            break
        angle = np.arctan2(obs[1], obs[0])
        err = abs(angle - PI_2)
        err = min(err, 2 * np.pi - err)
        if err < SUCCESS_THRESH:
            break

    final_angle = np.arctan2(obs[1], obs[0])
    final_err = min(abs(final_angle - PI_2), 2 * np.pi - abs(final_angle - PI_2))
    success = final_err < SUCCESS_THRESH

    if verbose:
        status = '✓' if success else '✗'
        print(f"    {status} steps={total_steps:3d}  final_err={final_err:.3f}rad  "
              f"final_state=[{obs[0]:+.2f},{obs[1]:+.2f},{obs[2]:+.2f}]")

    return success, total_steps, final_err, np.array(traj)


def evaluate(controller, env, n_trials=10, seed_base=42, label=''):
    """Evaluate controller over n_trials with different seeds."""
    successes = 0
    total_steps = 0
    final_errors = []
    trial_results = []

    t_start = time.time()
    for t in range(n_trials):
        trial_seed = seed_base + t * 100
        ok, steps, ferr, traj = run_trial(controller, env, trial_seed,
                                           verbose=(n_trials <= 10))
        if ok:
            successes += 1
        total_steps += steps
        final_errors.append(ferr)
        trial_results.append({'ok': ok, 'steps': steps, 'ferr': ferr, 'seed': trial_seed})

    t_elapsed = time.time() - t_start
    mean_err = np.mean(final_errors)
    std_err = np.std(final_errors)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Success:  {successes}/{n_trials} ({successes/n_trials*100:.0f}%)")
    print(f"  Mean |Δθ|: {mean_err:.3f} ± {std_err:.3f} rad")
    print(f"  Mean steps: {total_steps/n_trials:.1f}")
    print(f"  Time: {t_elapsed:.1f}s ({t_elapsed/n_trials:.1f}s/trial)")

    # Per-trial summary
    if n_trials <= 10:
        print(f"  {'Trial':>5s}  {'Seed':>5s}  {'Steps':>5s}  {'|Δθ|':>7s}  {'Result':>6s}")
        for t, r in enumerate(trial_results):
            status = '✓' if r['ok'] else '✗'
            print(f"  {t+1:5d}  {r['seed']:5d}  {r['steps']:5d}  "
                  f"{r['ferr']:7.3f}  {status:>6s}")

    return {
        'success_rate': successes / n_trials,
        'mean_error': mean_err,
        'std_error': std_err,
        'mean_steps': total_steps / n_trials,
        'trial_results': trial_results,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Test KAN-adapted decision network (v3)')
    parser.add_argument('--policy', type=str, default='kan_policy_v3.pt',
                        help='Path to trained policy checkpoint')
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--seed-base', type=int, default=42)
    parser.add_argument('--compare-all', action='store_true',
                        help='Compare v3 against baselines (energy, random)')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    env = gym.make("Pendulum-v1")

    # ── Load v3 policy ──
    print(f"Loading policy: {args.policy}")
    ckpt = torch.load(args.policy, map_location=args.device, weights_only=True)
    multi_scale = ckpt.get('multi_scale', 'off') in ('fixed', 'policy')
    output_k = ckpt.get('output_k', False) or (ckpt.get('multi_scale', 'off') == 'policy')
    policy_type = ckpt.get('policy_type', 'mlp')

    if policy_type == 'residual':
        policy = ResidualPhysicsPolicy(
            state_dim=ckpt['state_dim'], action_dim=1,
            hidden=ckpt['hidden'], n_layers=ckpt['n_layers'],
            init_k_energy=ckpt.get('k_energy_init', 0.15),
            init_k_stable=ckpt.get('k_stable_init', -2.0),
            init_k_damp=ckpt.get('k_damp_init', -0.3))
        print(f"  Architecture: ResidualPhysics (energy shaping + MLP residual)")
    else:
        policy = KANPolicy(state_dim=ckpt['state_dim'], action_dim=1,
                           hidden=ckpt['hidden'], n_layers=ckpt['n_layers'],
                           output_k=output_k)
        print(f"  Architecture: MLP([{ckpt['state_dim']}, {ckpt['hidden']}×{ckpt['n_layers']}], "
              f"out={'a,k' if output_k else 'a'})")

    policy.load_state_dict(ckpt['policy_state_dict'])
    policy.to(args.device)
    policy.eval()
    print(f"  Multi-step training: H={ckpt.get('multi_step', 1)}")
    print(f"  Multi-scale: {ckpt.get('multi_scale', 'off')}")

    v3_ctrl = V3PolicyController(policy, args.device, multi_scale=multi_scale)

    # ── Evaluate ──
    result_v3 = evaluate(v3_ctrl, env, args.trials, args.seed_base,
                         label='decision_v3 (KAN gradient trained)')

    if args.compare_all:
        # Baselines
        energy_ctrl = EnergyController()
        evaluate(energy_ctrl, env, min(args.trials, 5), args.seed_base,
                 label='Energy controller (oracle)')

        random_ctrl = RandomController()
        evaluate(random_ctrl, env, min(args.trials, 5), args.seed_base,
                 label='Random actions (lower bound)')

    env.close()
    return result_v3


if __name__ == '__main__':
    main()
