"""Evaluate decision network on pendulum swing-up task."""
import torch, numpy as np, time, argparse
import gymnasium as gym
from control.decision_network import DecisionKAN
from kanrf import KAN

G = 10.0; PI_2 = np.pi / 2


def load_world_model(path='kan_hybrid_lam0.1_nu0.1.pt'):
    ckpt = torch.load(path, weights_only=True)
    dims = [4]
    for k in sorted(ckpt.keys()):
        if 'base_weight' in k: dims.append(ckpt[k].shape[0])
    model = KAN(dims, grid_size=5, spline_order=3)
    model.load_state_dict(ckpt)
    return model.eval()


def run_trial(decision_net, world_model, env, trial_seed, max_steps=60, device='cpu'):
    obs, _ = env.reset(seed=trial_seed)
    s_target_norm = torch.tensor([[0., 1., 0.]], device=device)

    for step in range(max_steps):
        s_norm = torch.tensor([[obs[0], obs[1], obs[2] / 8.0]], dtype=torch.float32).to(device)

        with torch.no_grad():
            a_norm, h_logits = decision_net(s_norm, s_target_norm)
            a = a_norm.item() * 2.0   # denormalize to [-2, 2]

        obs, _, term, trunc, _ = env.step([a])

        angle = np.arctan2(obs[1], obs[0])
        if abs(angle - PI_2) < 0.2:
            return True, step + 1
        if term or trunc:
            return False, step + 1
    return False, max_steps


def main(model_path='kan_decision.pt', world_path='kan_hybrid_lam0.1_nu0.1.pt',
         n_trials=10, device='cpu'):
    torch.manual_seed(42); np.random.seed(42)

    data = torch.load('decision_data.pt', weights_only=True)
    n_classes = data['H_class'].max().item() + 1

    decision_net = DecisionKAN(hidden_dim=10, n_horizon_classes=n_classes).to(device)
    decision_net.load_state_dict(torch.load(model_path, weights_only=True))
    decision_net.eval()

    env = gym.make("Pendulum-v1")
    successes = 0
    total_steps = 0

    print(f"{'Trial':>5s}  {'|dth0|':>7s}  {'Result':>8s}  {'Steps':>5s}")
    print("-" * 30)

    for t in range(n_trials):
        trial_seed = 42 + t * 100
        obs0, _ = env.reset(seed=trial_seed)
        init_err = abs(np.arctan2(obs0[1], obs0[0]) - PI_2)

        ok, steps = run_trial(decision_net, None, env, trial_seed, device=device)
        if ok: successes += 1
        total_steps += steps
        print(f"  {t+1:5d}  {init_err:7.3f}  {'OK' if ok else 'FAIL':>8s}  {steps:5d}")

    print(f"\nSuccess: {successes}/{n_trials}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='kan_decision.pt')
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    main(model_path=args.model, n_trials=args.trials, device=args.device)
