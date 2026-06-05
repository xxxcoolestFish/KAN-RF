"""General continuous learning: NO system-specific heuristics.

Mechanism:
  1. Model-based inverse optimization for control
  2. When prediction error is high → add action noise to explore diverse states
  3. Collect all (s, a, k, s') transitions
  4. After each episode, batch fine-tune world model on collected data
  5. Model improves → prediction error drops → exploration quiets → control converges

Only assumption: we can compute ||f_model(s,a,k) - s'_real||.
"""
import torch, numpy as np, time, argparse, copy
import gymnasium as gym
from kanrf import KAN
from control.continuous_learner import ContinuousLearner

PI_2 = np.pi / 2
K_VALS = [1, 2, 4, 8, 16]
MAX_K = 12


def load_model(path, layer_dims):
    model = KAN(layer_dims, grid_size=5, spline_order=3)
    model.load_state_dict(torch.load(path, weights_only=True))
    return model


def inverse_optimize(wm, s_norm, k):
    s_target = torch.tensor([[0., 1., 0.]])
    k_norm = torch.tensor([[k / 16.0]])
    best_loss, best_a = float('inf'), None
    for _ in range(3):
        a = torch.empty(1, 1)
        torch.nn.init.uniform_(a, -1, 1)
        a.requires_grad_(True)
        opt = torch.optim.Adam([a], lr=0.05)
        for __ in range(200):
            opt.zero_grad()
            loss = ((wm(torch.cat([s_norm, a, k_norm], dim=-1)) - s_target) ** 2).sum()
            loss.backward()
            opt.step()
            with torch.no_grad():
                a.clamp_(-1.0, 1.0)
        with torch.no_grad():
            final = ((wm(torch.cat([s_norm, a, k_norm], dim=-1))
                     - s_target) ** 2).sum().item()
        if final < best_loss:
            best_loss, best_a = final, a.detach().clone()
    return best_a.item(), best_loss


def pick_best_action(wm, s_norm):
    best_k, best_a, best_loss = None, None, float('inf')
    for k in K_VALS:
        a, loss = inverse_optimize(wm, s_norm, k)
        if loss < best_loss:
            best_loss = loss
            best_a = a
            best_k = k
    return best_a, best_k


def run_episode(wm, learner, env, trial_seed, explore_noise=0.0):
    """One control episode.

    When explore_noise > 0: adds Gaussian noise to the chosen action.
    Larger noise when model error is high → visits diverse states naturally.
    """
    obs, _ = env.reset(seed=trial_seed)
    step_count, macro_count = 0, 0

    for macro_step in range(60):
        s_norm = torch.tensor([[obs[0], obs[1], obs[2] / 8.0]], dtype=torch.float32)

        wm.eval()
        for p in wm.parameters():
            p.requires_grad = False

        a_norm, k = pick_best_action(wm, s_norm)

        # Adaptive noise: larger when model is surprised (high recent error)
        noise_scale = explore_noise
        if explore_noise > 0 and learner.is_surprised():
            noise_scale = explore_noise * 3.0  # triple the noise in uncertain regions

        a_norm = a_norm + np.random.normal(0, noise_scale)
        a_norm = max(-1.0, min(1.0, a_norm))
        k = min(k, MAX_K)

        s_before = obs.copy()
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

        macro_count += 1

        # Record transition
        s_n = torch.tensor([[s_before[0], s_before[1], s_before[2] / 8.0]],
                           dtype=torch.float32)
        s_next_n = torch.tensor([[obs[0], obs[1], obs[2] / 8.0]],
                                dtype=torch.float32)
        learner.observe(s_n, a_norm, k, s_next_n)

        if term or trunc:
            break

    final_angle = np.arctan2(obs[1], obs[0])
    final_err = min(abs(final_angle - PI_2), 2 * np.pi - abs(final_angle - PI_2))
    return final_err < 0.2, step_count, macro_count


def main(model_path='kan_ms.pt', n_train_episodes=8, device='cpu',
         lr=1e-4, ft_epochs=30, explore_noise=0.2):
    torch.manual_seed(42)
    np.random.seed(42)

    wm = load_model(model_path, [5, 16, 3]).to(device)
    wm_orig = copy.deepcopy(wm)

    learner = ContinuousLearner(wm, lr=lr, error_threshold=0.20)

    env = gym.make("Pendulum-v1")
    trial_seed = 142  # Trial 2

    print(f"General Continuous Learning — no heuristics")
    print(f"  Mechanism: adaptive action noise + prediction-error detection + batch fine-tune")
    print(f"  Episodes={n_train_episodes}  lr={lr}  ft_epochs={ft_epochs}  "
          f"noise={explore_noise}")
    print()

    for ep in range(1, n_train_episodes + 1):
        ok, steps, macros = run_episode(wm, learner, env, trial_seed,
                                        explore_noise=explore_noise)
        loss = learner.fine_tune(epochs=ft_epochs)
        s = learner.summary()
        print(f"  Ep {ep:2d}: {'OK' if ok else 'FAIL'}  "
              f"steps={steps:3d}  macros={macros:2d}  "
              f"err={s['mean_error']:.4f}  max_err={s['max_error']:.4f}  "
              f"loss={loss:.5f}")

    # Final test: pure model-based, zero noise
    print(f"\n{'='*50}")
    print("FINAL TEST: zero noise (pure inverse optimization)")
    print(f"{'='*50}")

    learner.buffer_x, learner.buffer_y, learner.error_log = [], [], []
    ok_cl, steps_cl, macros_cl = run_episode(wm, learner, env, trial_seed,
                                             explore_noise=0.0)
    print(f"  After learning: {'OK' if ok_cl else 'FAIL'}  "
          f"steps={steps_cl}  macros={macros_cl}")

    ok_orig, steps_orig, macros_orig = run_episode(
        wm_orig, learner, env, trial_seed, explore_noise=0.0)
    print(f"  Original model: {'OK' if ok_orig else 'FAIL'}  "
          f"steps={steps_orig}  macros={macros_orig}")

    if ok_cl and not ok_orig:
        print(f"\n  *** Continuous learning fixed Trial 2 — no heuristics needed! ***")
    elif ok_cl:
        print(f"\n  Trial 2 succeeds in both cases.")
    else:
        print(f"\n  Trial 2 still fails.  Try more episodes or adjust lr/ft_epochs.")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='kan_ms.pt')
    parser.add_argument('--episodes', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--ft-epochs', type=int, default=30)
    parser.add_argument('--noise', type=float, default=0.2,
                       help='Action noise std during exploration episodes')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    main(model_path=args.model, n_train_episodes=args.episodes, device=args.device,
         lr=args.lr, ft_epochs=args.ft_epochs, explore_noise=args.noise)
