"""Approach 1: action exploration + label correction + decision network fine-tuning.

When the model gets stuck (angle error not improving), the ActionExplorer:
  1. Saves the environment state
  2. Tries multiple candidate actions from the SAME state
  3. Picks the one that actually works best
  4. Records the corrected (s, a*, k*) as a training example

After each episode, use corrected examples to fine-tune the DECISION NETWORK.
The decision network learns: "when you see this state, output this action".
"""
import torch, numpy as np, time, argparse, copy
import gymnasium as gym
from kanrf import KAN
from action_explorer import ActionExplorer, angle_error

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
            loss = ((wm(torch.cat([s_norm, a, k_norm], dim=-1)) - s_target)**2).sum()
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
            best_loss, best_a, best_k = loss, a, k
    return best_a, best_k


def train_decision_network(model, x, a_label, k_label, epochs=500, lr=1e-3):
    """Fine-tune decision network on corrected labels."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()

    for _ in range(epochs):
        opt.zero_grad()
        out = model(x)
        loss = mse(out[:, 0:1], a_label) + 0.5 * mse(out[:, 1:2], k_label)
        loss.backward()
        opt.step()

    model.eval()
    return loss.item()


def run_episode(wm, dn, explorer, env, trial_seed, use_explorer=False):
    """One episode.  If use_explorer, tries alternative actions when stuck."""
    obs, _ = env.reset(seed=trial_seed)
    step_count, macro_count, explore_count = 0, 0, 0
    prev_err = angle_error(obs)
    stuck_counter = 0
    s_target_norm = torch.tensor([[0., 1., 0.]])

    for macro_step in range(60):
        s_norm = torch.tensor([[obs[0], obs[1], obs[2] / 8.0]],
                              dtype=torch.float32)

        wm.eval()
        for p in wm.parameters():
            p.requires_grad = False

        model_a, model_k = pick_best_action(wm, s_norm)

        # --- Decide whether to explore ---
        if use_explorer and stuck_counter >= 3:
            candidates = explorer.generate_candidates(model_a, model_k)
            a_norm, k, progress = explorer.try_candidates(env, s_norm, candidates)

            if progress > 0.01:
                explorer.record_correction(
                    s_norm, model_a, model_k, a_norm, k, progress)

            explore_count += 1
            stuck_counter = 0
        else:
            a_norm, k = model_a, model_k

        k = min(k, MAX_K)
        a_raw = a_norm * 2.0
        s_before = obs.copy()

        # Execute
        for _ in range(k):
            obs, _, term, trunc, _ = env.step([a_raw])
            step_count += 1
            a = np.arctan2(obs[1], obs[0])
            if abs(a - PI_2) < 0.2:
                term = True
                break
            if term or trunc:
                break

        macro_count += 1
        if term or trunc:
            break

        # Track if stuck
        curr_err = angle_error(obs)
        if curr_err >= prev_err - 0.02:
            stuck_counter += 1
        else:
            stuck_counter = max(0, stuck_counter - 1)
        prev_err = curr_err

    final_angle = np.arctan2(obs[1], obs[0])
    final_err = min(abs(final_angle - PI_2), 2 * np.pi - abs(final_angle - PI_2))
    return final_err < 0.2, step_count, macro_count, explore_count


def test_all_trials(wm, label, n=10):
    """Evaluate pure inverse optimization on all 10 trials."""
    env = gym.make("Pendulum-v1")
    ok_cnt = 0
    for t in range(n):
        trial_seed = 42 + t * 100
        obs, _ = env.reset(seed=trial_seed)
        for macro in range(60):
            sn = torch.tensor([[obs[0], obs[1], obs[2] / 8.0]], dtype=torch.float32)
            wm.eval()
            for p in wm.parameters():
                p.requires_grad = False
            an, k = pick_best_action(wm, sn)
            k = min(k, MAX_K)
            ar = an * 2.0
            for _ in range(k):
                obs, _, term, trunc, _ = env.step([ar])
                if abs(np.arctan2(obs[1], obs[0]) - PI_2) < 0.2:
                    break
            if term or trunc:
                break
            if abs(np.arctan2(obs[1], obs[0]) - PI_2) < 0.2:
                break
        fe = np.arctan2(obs[1], obs[0])
        fe = min(abs(fe - PI_2), 2 * np.pi - abs(fe - PI_2))
        if fe < 0.2:
            ok_cnt += 1
    env.close()
    print(f"  {label}: {ok_cnt}/{n}")
    return ok_cnt


def main(model_wm='kan_ms.pt', model_dn='kan_decision_ms.pt',
         n_train_episodes=10, device='cpu'):
    torch.manual_seed(42)
    np.random.seed(42)

    wm = load_model(model_wm, [5, 16, 3]).to(device)
    wm_orig = copy.deepcopy(wm)

    # Decision network: will be fine-tuned with corrected labels
    dn = load_model(model_dn, [6, 12, 2]).to(device)

    explorer = ActionExplorer(n_candidates=5, n_random=3)
    env = gym.make("Pendulum-v1")
    trial_seed = 142

    print("Approach 1: Action Exploration + Decision Network Fine-Tuning")
    print(f"  n_train_episodes={n_train_episodes}")
    print(f"  Explorer: 5 candidates (model, opposite, 3 random)")
    print()

    all_corrections = []

    for ep in range(1, n_train_episodes + 1):
        ok, steps, macros, explores = run_episode(
            wm, dn, explorer, env, trial_seed, use_explorer=True)

        # Train decision network on ALL accumulated corrections
        s_data, a_data, k_data = explorer.get_training_data()
        n_corrections = len(explorer.corrections)
        if s_data is not None and len(s_data) > 4:
            s_target = torch.tensor([[0., 1., 0.]], dtype=torch.float32).expand(len(s_data), -1)
            x_in = torch.cat([s_data, s_target], dim=-1)
            loss = train_decision_network(dn, x_in, a_data, k_data)
        else:
            loss = 0.0

        print(f"  Ep {ep:2d}: {'OK' if ok else 'FAIL'}  "
              f"steps={steps:3d}  macros={macros:2d}  "
              f"explores={explores}  corrections={n_corrections}  dn_loss={loss:.4f}")

    # Final test
    print(f"\n{'='*50}")
    print("FINAL: test world model + explore-enhanced decision network")
    print(f"{'='*50}")

    # Test decision network (uses dn which was fine-tuned)
    print("\n  Decision network (after fine-tuning):")
    env_test = gym.make("Pendulum-v1")
    ok_cnt = 0
    for t in range(10):
        trial_seed = 42 + t * 100
        obs, _ = env_test.reset(seed=trial_seed)
        s_target_norm = torch.tensor([[0., 1., 0.]])
        for _ in range(60):
            sn = torch.tensor([[obs[0], obs[1], obs[2] / 8.0]], dtype=torch.float32)
            with torch.no_grad():
                out = dn(torch.cat([sn, s_target_norm], dim=-1))
                an = out[0, 0].item()
                k = max(1, min(MAX_K, round(out[0, 1].item() * 16)))
            ar = an * 2.0
            for __ in range(k):
                obs, _, term, trunc, _ = env_test.step([ar])
                if abs(np.arctan2(obs[1], obs[0]) - PI_2) < 0.2:
                    break
            if term or trunc:
                break
            if abs(np.arctan2(obs[1], obs[0]) - PI_2) < 0.2:
                break
        fe = min(abs(np.arctan2(obs[1], obs[0]) - PI_2),
                 2 * np.pi - abs(np.arctan2(obs[1], obs[0]) - PI_2))
        if fe < 0.2:
            ok_cnt += 1
    env_test.close()
    print(f"  {ok_cnt}/10")

    torch.save(dn.state_dict(), 'kan_decision_explored.pt')
    print(f"  Saved: kan_decision_explored.pt")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wm', type=str, default='kan_ms.pt')
    parser.add_argument('--dn', type=str, default='kan_decision_ms.pt')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    main(model_wm=args.wm, model_dn=args.dn,
         n_train_episodes=args.episodes, device=args.device)
