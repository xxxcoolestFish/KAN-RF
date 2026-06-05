"""Autoregressive controller: multi-scale decision net + world model rollout.

Plan B: decision net is called repeatedly in a closed-loop imagination loop.
Executes MPC-style: only the first macro-action runs on the real environment,
then replan from the observed state.
"""
import torch, numpy as np, time, argparse
import gymnasium as gym
from kanrf import KAN

PI_2 = np.pi / 2
MAX_K = 16


def load_model(path, layer_dims):
    model = KAN(layer_dims, grid_size=5, spline_order=3)
    model.load_state_dict(torch.load(path, weights_only=True))
    return model.eval()


def cos_sin_normalize(s):
    nrm = s[:, :2].norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return torch.cat([s[:, :2] / nrm, s[:, 2:]], dim=-1)


def angle_error(s, s_target):
    cos_err = (s[:, :2] * s_target[:, :2]).sum(-1).clamp(-1, 1)
    return torch.acos(cos_err).item()


def autoregressive_plan(decision_net, world_model, s_norm, s_target_norm,
                        max_depth=6, angle_thresh=0.15):
    """Expand decision net through world model to generate a macro-action plan."""
    s = s_norm.clone()
    plan = []

    for _ in range(max_depth):
        x = torch.cat([s, s_target_norm], dim=-1)
        with torch.no_grad():
            out = decision_net(x)
            a_norm = out[:, 0:1]
            k_cont = out[:, 1].item()

        k = max(1, min(MAX_K, round(k_cont * MAX_K)))
        plan.append((a_norm.item(), k))

        # One world-model forward: f_ms(s, a, k_norm) simulates k*dt
        k_norm = torch.tensor([[k_cont]])
        wm_input = torch.cat([s, a_norm, k_norm], dim=-1)
        s = world_model(wm_input)
        s = cos_sin_normalize(s)

        if angle_error(s, s_target_norm) < angle_thresh:
            return plan, True

    return plan, False


def main(model_dn='kan_decision_ms.pt', model_wm='kan_ms.pt',
         n_trials=10, device='cpu'):
    torch.manual_seed(42)
    np.random.seed(42)

    dn = load_model(model_dn, [6, 12, 2]).to(device)
    wm = load_model(model_wm, [5, 16, 3]).to(device)

    env = gym.make("Pendulum-v1")
    s_target_norm = torch.tensor([[0., 1., 0.]], device=device)

    print(f"Plan B: Autoregressive Controller (no online learning)")
    print(f"  Decision net: [6,12,2]  |  World model: [5,16,3]")
    print(f"{'Trial':>5s}  {'|dth0|':>7s}  {'|dth_f|':>9s}  {'Result':>8s}  "
          f"{'Steps':>5s}  {'depth':>6s}")
    print("-" * 55)

    successes = 0
    t0 = time.time()

    for t in range(n_trials):
        trial_seed = 42 + t * 100
        obs, _ = env.reset(seed=trial_seed)
        init_err = abs(np.arctan2(obs[1], obs[0]) - PI_2)

        step_count = 0
        max_depth_used = 0

        for macro_step in range(60):
            s_norm = torch.tensor([[obs[0], obs[1], obs[2] / 8.0]],
                                  dtype=torch.float32).to(device)

            # --- Autoregressive imagination ---
            plan, _ = autoregressive_plan(dn, wm, s_norm, s_target_norm, max_depth=5)
            max_depth_used = max(max_depth_used, len(plan))

            # --- Execute first macro-action (MPC) ---
            a_norm, k = plan[0]
            a_raw = a_norm * 2.0

            for _ in range(k):
                obs, _, term, trunc, _ = env.step([a_raw])
                step_count += 1

                angle = np.arctan2(obs[1], obs[0])
                if abs(angle - PI_2) < 0.2:
                    term = True
                    break
                if term or trunc:
                    break
            if term or trunc:
                break

        final_err = abs(np.arctan2(obs[1], obs[0]) - PI_2)
        ok = final_err < 0.2
        if ok:
            successes += 1

        print(f"  {t+1:5d}  {init_err:7.3f}  {final_err:9.4f}  "
              f"{'OK' if ok else 'FAIL':>8s}  {step_count:5d}  {max_depth_used:5d}")

    elapsed = time.time() - t0
    print(f"\nSuccess: {successes}/{n_trials}  |  time: {elapsed:.0f}s")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dn', type=str, default='kan_decision_ms.pt')
    parser.add_argument('--wm', type=str, default='kan_ms.pt')
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    main(model_dn=args.dn, model_wm=args.wm, n_trials=args.trials, device=args.device)
