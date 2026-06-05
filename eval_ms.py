"""Evaluate multi-scale decision network on Pendulum-v1.

Plan A: decision net outputs (a, k_cont).
Execute constant torque for k = round(k_cont * 16) steps, then replan.
"""
import torch, numpy as np, time, argparse
import gymnasium as gym
from kanrf import KAN

PI_2 = np.pi / 2


def load_model(path, layer_dims):
    model = KAN(layer_dims, grid_size=5, spline_order=3)
    model.load_state_dict(torch.load(path, weights_only=True))
    return model.eval()


def run_trial(decision_net, env, trial_seed, max_steps=60, device='cpu'):
    obs, _ = env.reset(seed=trial_seed)
    s_target_norm = torch.tensor([[0., 1., 0.]], device=device)
    k_values = []

    for step_idx in range(max_steps):
        s_norm = torch.tensor([[obs[0], obs[1], obs[2] / 8.0]],
                              dtype=torch.float32).to(device)
        x = torch.cat([s_norm, s_target_norm], dim=-1)

        with torch.no_grad():
            out = decision_net(x)
            a = out[0, 0].item() * 2.0           # denormalize: a_norm ∈ [-1,1] → torque ∈ [-2,2]
            k_cont = out[0, 1].item()

        k = max(1, min(16, round(k_cont * 16)))
        k_values.append(k)

        # Execute constant torque for k steps
        for _ in range(k):
            obs, _, term, trunc, _ = env.step([a])
            if term or trunc:
                break
            angle = np.arctan2(obs[1], obs[0])
            if abs(angle - PI_2) < 0.2:
                return True, step_idx + 1, k_values, obs

        if term or trunc:
            break

    angle_final = np.arctan2(obs[1], obs[0])
    return abs(angle_final - PI_2) < 0.2, step_idx + 1, k_values, obs


def main(model_path='kan_decision_ms.pt', n_trials=10, device='cpu'):
    torch.manual_seed(42)
    np.random.seed(42)

    decision_net = load_model(model_path, [6, 12, 2]).to(device)
    env = gym.make("Pendulum-v1")

    print(f"Multi-Scale Decision Net: [6, 12, 2]")
    print(f"{'Trial':>5s}  {'|dth0|':>7s}  {'|dth_f|':>9s}  {'Result':>8s}  {'Steps':>5s}  {'k_avg':>6s}")
    print("-" * 55)

    successes = 0
    t0 = time.time()

    for t in range(n_trials):
        trial_seed = 42 + t * 100
        obs0, _ = env.reset(seed=trial_seed)
        init_err = abs(np.arctan2(obs0[1], obs0[0]) - PI_2)

        ok, steps, k_vals, final_obs = run_trial(decision_net, env, trial_seed, device=device)
        if ok:
            successes += 1

        final_err = abs(np.arctan2(final_obs[1], final_obs[0]) - PI_2)
        k_avg = np.mean(k_vals) if k_vals else 0

        print(f"  {t+1:5d}  {init_err:7.3f}  {final_err:9.4f}  "
              f"{'OK' if ok else 'FAIL':>8s}  {steps:5d}  {k_avg:5.1f}")

    elapsed = time.time() - t0
    print(f"\nSuccess: {successes}/{n_trials}  |  time: {elapsed:.0f}s")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='kan_decision_ms.pt')
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    main(model_path=args.model, n_trials=args.trials, device=args.device)
