"""Non-stationary continual learning: KAN vs MLP world model.

Experiment:
  1. Both world models + policies trained on Pendulum (l=1.0)
  2. Deploy in modified environment (l=1.5)
  3. Apply online learning to world models, fine-tune policies
  4. Measure: success rate recovery, forgetting of original dynamics

The key hypothesis: KAN's B-spline local support enables fast adaptation
without catastrophic forgetting. MLP's global activation causes forgetting
of previously learned regions when adapting to new dynamics.

Usage:
  python scripts/eval/eval_continual.py --wm-kan kan_hybrid_lam0.1_nu0.1.pt --policy-kan kan_policy_mlp_cws.pt
"""
import sys, os, argparse, time, copy
import torch
import numpy as np
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from kanrf import KAN
from decision_v3.core import KANPolicy, KANEnergyTrainer
from control.online_learning_v2 import ThreeFactorUpdater, compute_training_stats

PI_2 = np.pi / 2


# ═══════════════════════════════════════════════════════════════
# Modified Pendulum Environment
# ═══════════════════════════════════════════════════════════════

def make_modified_pendulum(length=1.5, mass=1.0):
    """Create Pendulum-v1 with modified physical parameters.

    Standard Pendulum-v1: g=10.0, m=1.0, l=1.0, max_torque=2.0
    We modify l (affects moment of inertia: I = m*l^2, thus thdd = torque/I - g/l*sin).
    """
    env = gym.make("Pendulum-v1")
    env.unwrapped.length = length
    env.unwrapped.m = mass
    # max_speed scales with sqrt(g/l), update accordingly
    env.unwrapped.max_speed = 8.0 * np.sqrt(1.0 / length)
    return env


# ═══════════════════════════════════════════════════════════════
# MLP World Model (same architecture as KAN in capacity)
# ═══════════════════════════════════════════════════════════════

class MLPWorldModel(torch.nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, 3)
        )

    def forward(self, x, return_activations=False):
        if return_activations:
            return self.net(x), [], []
        return self.net(x)


# ═══════════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════════

def evaluate_episode(policy, env, n_trials=5, max_steps=300):
    """Evaluate policy over multiple trials."""
    successes = 0
    for t in range(n_trials):
        obs, _ = env.reset(seed=42 + t * 100)
        for step in range(max_steps):
            s = torch.tensor([[obs[0], obs[1], obs[2] / 8.0]], dtype=torch.float32)
            with torch.no_grad():
                a_norm = policy(s).item()
            a_raw = np.clip(a_norm * 2.0, -2.0, 2.0)
            obs, _, term, trunc, _ = env.step([a_raw])
            if term or trunc:
                break
            err = abs(np.arctan2(obs[1], obs[0]) - PI_2)
            if min(err, 2 * np.pi - err) < 0.2:
                break
        fe = min(abs(np.arctan2(obs[1], obs[0]) - PI_2),
                 2 * np.pi - abs(np.arctan2(obs[1], obs[0]) - PI_2))
        if fe < 0.2:
            successes += 1
    return successes / n_trials


def online_adapt_kan(kan, policy, env, n_episodes=20, eta0=1e-3, device='cpu'):
    """Online adaptation loop for KAN world model + policy.

    After each episode, compute training stats and update KAN.
    Periodically fine-tune policy using updated KAN.
    """
    x, y = torch.load('c:/Users/32510/Desktop/RF/KAN-RF/pendulum_data.pt', weights_only=True)
    stats = compute_training_stats(kan, x[:2000], y[:2000])
    updater = ThreeFactorUpdater(kan, stats, eta0=eta0)

    success_history = []
    wm_error_history = []

    s_target = torch.tensor([[0., 1., 0.]], device=device)

    for ep in range(1, n_episodes + 1):
        # Collect one episode of data
        transitions = []
        obs, _ = env.reset(seed=42 + ep * 17)
        for _ in range(200):
            s_norm = torch.tensor([[obs[0], obs[1], obs[2] / 8.0]], dtype=torch.float32, device=device)
            with torch.no_grad():
                a_norm = policy(torch.tensor([[obs[0], obs[1], obs[2] / 8.0]], dtype=torch.float32)).item()
            a_raw = np.clip(a_norm * 2.0, -2.0, 2.0)
            next_obs, _, term, trunc, _ = env.step([a_raw])
            s_next_norm = torch.tensor([[next_obs[0], next_obs[1], next_obs[2] / 8.0]],
                                       dtype=torch.float32, device=device)
            a_norm_t = torch.tensor([[a_norm]], dtype=torch.float32, device=device)
            transitions.append((s_norm, a_norm_t, s_next_norm))
            obs = next_obs
            if term or trunc:
                break

        # Online update KAN with collected data
        wm_errors = []
        for s_n, a_n, s_nn in transitions:
            err, _ = updater.update(s_n, a_n, s_nn)
            wm_errors.append(err)
        wm_error_history.append(np.mean(wm_errors))

        # Fine-tune policy with updated KAN (few steps)
        if ep % 3 == 0 and len(transitions) > 10:
            # Sample states from collected transitions
            s_batch = torch.cat([t[0] for t in transitions[:32]], dim=0)
            trainer = KANEnergyTrainer(kan, policy, s_target, G=10.0, lr=3e-4, lambda_ctrl=0.01, device=device)
            for _ in range(5):
                trainer.train_step(s_batch)

        # Evaluate
        if ep % 2 == 0 or ep == 1:
            sr = evaluate_episode(policy, env, n_trials=5)
            success_history.append((ep, sr))
            print(f"  Ep {ep:3d}: sr={sr:.0%}  wm_err={wm_error_history[-1]:.4f}")

    return success_history, wm_error_history


def online_adapt_mlp(mlp, policy, env, n_episodes=20, lr=1e-4, device='cpu'):
    """Online adaptation for MLP world model (standard SGD).

    MLP has no B-spline locality, so we use standard minibatch SGD
    with small learning rate to avoid catastrophic forgetting.
    """
    s_target = torch.tensor([[0., 1., 0.]], device=device)
    opt = torch.optim.SGD(mlp.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    success_history = []
    wm_error_history = []

    for ep in range(1, n_episodes + 1):
        # Collect transitions
        transitions = []
        obs, _ = env.reset(seed=42 + ep * 17)
        for _ in range(200):
            a_norm = policy(torch.tensor([[obs[0], obs[1], obs[2] / 8.0]], dtype=torch.float32)).item()
            a_raw = np.clip(a_norm * 2.0, -2.0, 2.0)
            next_obs, _, term, trunc, _ = env.step([a_raw])
            transitions.append((
                torch.tensor([[obs[0], obs[1], obs[2] / 8.0, a_norm]], dtype=torch.float32, device=device),
                torch.tensor([[next_obs[0], next_obs[1], next_obs[2] / 8.0]], dtype=torch.float32, device=device),
            ))
            obs = next_obs
            if term or trunc:
                break

        # Batch SGD update
        if len(transitions) > 4:
            batch_x = torch.cat([t[0] for t in transitions], dim=0)
            batch_y = torch.cat([t[1] for t in transitions], dim=0)
            mlp.train()
            opt.zero_grad()
            pred = mlp(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            opt.step()
            mlp.eval()
            wm_error_history.append(loss.item())
        else:
            wm_error_history.append(0)

        # Fine-tune policy
        if ep % 3 == 0 and len(transitions) > 10:
            s_batch = torch.cat([t[0][:, :3] for t in transitions[:32]], dim=0)
            trainer = KANEnergyTrainer(mlp, policy, s_target, G=10.0, lr=3e-4, lambda_ctrl=0.01, device=device)
            for _ in range(5):
                trainer.train_step(s_batch)

        if ep % 2 == 0 or ep == 1:
            sr = evaluate_episode(policy, env, n_trials=5)
            success_history.append((ep, sr))
            print(f"  Ep {ep:3d}: sr={sr:.0%}  wm_err={wm_error_history[-1]:.6f}")

    return success_history, wm_error_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wm-kan', type=str, default='kan_hybrid_lam0.1_nu0.1.pt')
    parser.add_argument('--policy-kan', type=str, default='kan_policy_mlp_cws.pt')
    parser.add_argument('--policy-mlp', type=str, default='kan_policy_mlp_wm.pt')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(42)
    np.random.seed(42)

    # ── Modified environment (l=1.5 instead of 1.0) ──
    env_mod = make_modified_pendulum(length=1.5)

    # ── KAN Setup ──
    print("=" * 60)
    print("KAN WORLD MODEL + Online Adaptation")
    print("=" * 60)
    kan = KAN([4, 12, 3], grid_size=5, spline_order=3)
    kan.load_state_dict(torch.load(args.wm_kan, map_location=device, weights_only=True))
    kan.to(device)

    policy_kan = KANPolicy(state_dim=3, action_dim=1, hidden=64, n_layers=2)
    policy_kan.load_state_dict(torch.load(args.policy_kan, map_location=device, weights_only=True)['policy_state_dict'])
    policy_kan.to(device)

    # Evaluate before adaptation
    sr_before = evaluate_episode(policy_kan, env_mod, n_trials=10)
    print(f"Before adaptation: {sr_before:.0%}")

    # Adapt
    kan_hist, kan_errs = online_adapt_kan(kan, policy_kan, env_mod, n_episodes=args.episodes, device=device)

    sr_after = evaluate_episode(policy_kan, env_mod, n_trials=10)
    print(f"After {args.episodes} episodes: {sr_after:.0%}")

    # ── MLP Setup ──
    print("\n" + "=" * 60)
    print("MLP WORLD MODEL + Online Adaptation")
    print("=" * 60)
    mlp = MLPWorldModel(hidden=64).to(device)
    mlp_ckpt = torch.load(args.policy_mlp, map_location=device, weights_only=True)
    if 'wm_state_dict' in mlp_ckpt:
        mlp.load_state_dict(mlp_ckpt['wm_state_dict'])
    mlp.eval()

    policy_mlp = KANPolicy(state_dim=3, action_dim=1, hidden=64, n_layers=2)
    policy_mlp.load_state_dict(mlp_ckpt['policy_state_dict'])
    policy_mlp.to(device)

    sr_before_mlp = evaluate_episode(policy_mlp, env_mod, n_trials=10)
    print(f"Before adaptation: {sr_before_mlp:.0%}")

    mlp_hist, mlp_errs = online_adapt_mlp(mlp, policy_mlp, env_mod, n_episodes=args.episodes, device=device)

    sr_after_mlp = evaluate_episode(policy_mlp, env_mod, n_trials=10)
    print(f"After {args.episodes} episodes: {sr_after_mlp:.0%}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY: Continual Learning (Pendulum l: 1.0 → 1.5)")
    print("=" * 60)
    print(f"{'':20s} {'Before':>8s} {'After':>8s} {'Recovery':>10s}")
    print(f"{'KAN WM + Policy':20s} {sr_before:8.0%} {sr_after:8.0%} {sr_after-sr_before:10.0%}")
    print(f"{'MLP WM + Policy':20s} {sr_before_mlp:8.0%} {sr_after_mlp:8.0%} {sr_after_mlp-sr_before_mlp:10.0%}")

    env_mod.close()


if __name__ == '__main__':
    main()
