"""Online training: run episodes, explore when stuck, fine-tune decision network.

Combines ActionExplorer (try candidates in real env) with KAN-informed features.
Only generates correction labels for states that actually cause failures.

Usage:
  python train_online.py --episodes 15
"""
import torch, numpy as np, gymnasium as gym, time, sys, os, argparse
from kanrf import KAN
from decision_v2.core import FeatureComputer, TinyDecisionNet
from action_explorer import ActionExplorer, angle_error

PI_2 = np.pi / 2
MAX_K = 12


def load_models(device):
    root = os.path.dirname(os.path.dirname(__file__))

    # Single-scale for features
    wm = KAN([4, 12, 3], grid_size=5, spline_order=3)
    wm.load_state_dict(torch.load(os.path.join(root, 'kan_pendulum_model_v4.pt'),
                                   weights_only=True))
    wm.eval().to(device)
    for p in wm.parameters():
        p.requires_grad = False

    # Multi-scale for inverse optimization (to get action suggestions)
    wm_ms = KAN([5, 16, 3], grid_size=5, spline_order=3)
    wm_ms.load_state_dict(torch.load(os.path.join(root, 'kan_ms.pt'),
                                      weights_only=True))
    wm_ms.eval().to(device)
    for p in wm_ms.parameters():
        p.requires_grad = False

    return wm, wm_ms


def inverse_opt_action(wm_ms, s_norm, k_val, device):
    """Get action suggestion for a given k via inverse optimization."""
    s_target = torch.tensor([[0., 1., 0.]], device=device)
    kn = torch.tensor([[k_val / 16.0]], device=device)
    best_loss, best_a = float('inf'), None
    for _ in range(2):
        a = torch.empty(1, 1, device=device)
        torch.nn.init.uniform_(a, -1, 1)
        a.requires_grad_(True)
        opt = torch.optim.Adam([a], lr=0.05)
        for __ in range(100):
            opt.zero_grad()
            pred = wm_ms(torch.cat([s_norm, a, kn.expand(1, -1)], dim=-1))[:, :3]
            loss = ((pred - s_target) ** 2).sum()
            loss.backward()
            opt.step()
            with torch.no_grad():
                a.clamp_(-1.0, 1.0)
        with torch.no_grad():
            final = ((wm_ms(torch.cat([s_norm, a, kn.expand(1, -1)], dim=-1))[:, :3]
                     - s_target) ** 2).sum().item()
        if final < best_loss:
            best_loss, best_a = final, a.detach().clone()
    return best_a.item(), best_loss


def generate_candidates(wm_ms, s_norm, explorer, device):
    """Generate candidate (a, k) pairs using inverse opt + random."""
    candidates = []
    # Model suggestions for each k
    for k_val in [1, 2, 4, 8, 16]:
        a_val, _ = inverse_opt_action(wm_ms, s_norm, k_val, device)
        candidates.append((a_val, k_val))
    # Random candidates
    for _ in range(3):
        candidates.append((np.random.uniform(-1.0, 1.0),
                           np.random.choice([1, 2, 4, 8, 16])))
    return candidates


def fine_tune_dn(dn, X_list, Y_list, epochs=200, lr=1e-3, device='mps'):
    """Fine-tune decision network on accumulated corrections."""
    if len(X_list) < 8:
        return 0.0
    X = torch.stack(X_list).to(device)
    Y = torch.stack(Y_list).to(device)

    dn.train()
    opt = torch.optim.Adam(dn.parameters(), lr=lr)
    mse = torch.nn.MSELoss()

    for _ in range(epochs):
        opt.zero_grad()
        pred = dn({
            'a_init': X[:, 0:1], 'gap': X[:, 1:4],
            'align': X[:, 4:5], 'ctrl': X[:, 5:6], 'trust': X[:, 6:7],
        }, X[:, 7:10])
        loss = mse(pred[:, 0:1], Y[:, 0:1])
        if Y.shape[1] > 1:
            loss = loss + 0.5 * mse(pred[:, 1:2], Y[:, 1:2])
        loss.backward()
        opt.step()

    dn.eval()
    return loss.item()


def run_episode(fc, dn, wm_ms, explorer, env, trial_seed, device, output_k):
    """One episode.  Explores when stuck, records corrections."""
    obs, _ = env.reset(seed=trial_seed)
    s_target = torch.tensor([[0., 1., 0.]], device=device)
    prev_err = angle_error(obs)
    stuck_counter = 0
    corrections = []

    for macro_step in range(60):
        s_norm = torch.tensor([[obs[0], obs[1], obs[2] / 8.0]],
                              dtype=torch.float32).to(device)

        # Decision network prediction
        with torch.no_grad():
            features = fc.compute_features(s_norm, s_target)
            out = dn(features, s_norm)
            model_a = out[0, 0].item()
            k = 1
            if output_k:
                k_cont = out[0, 1].item()
                k = max(1, min(MAX_K, round(k_cont * 16)))

        # Explore when stuck
        explore_k = k
        if stuck_counter >= 3:
            candidates = generate_candidates(wm_ms, s_norm, explorer, device)
            a_chosen, k_chosen, progress = explorer.try_candidates(
                env, s_norm, candidates)

            if progress > 0.005:
                # Record correction: features → better action
                f_cpu = {key: val.cpu().squeeze(0)
                         for key, val in features.items()}
                feat_vec = torch.cat([
                    f_cpu['a_init'].reshape(1),
                    f_cpu['gap'].reshape(-1),
                    f_cpu['align'].reshape(1),
                    f_cpu['ctrl'].reshape(1),
                    f_cpu['trust'].reshape(1),
                    s_norm.cpu().squeeze(0).reshape(-1),
                ])
                corrections.append((
                    feat_vec,
                    torch.tensor([a_chosen, k_chosen / 16.0])
                    if output_k else torch.tensor([a_chosen])
                ))

            model_a, k = a_chosen, k_chosen
            stuck_counter = 0
        else:
            model_a, k = out[0, 0].item(), k
            # Use decision network's suggested k
            pass

        k = min(k, MAX_K)
        a_raw = model_a * 2.0
        s_before = obs.copy()

        for _ in range(k):
            obs, _, term, trunc, _ = env.step([a_raw])
            if abs(np.arctan2(obs[1], obs[0]) - PI_2) < 0.2:
                return True, corrections
            if term or trunc:
                break

        curr_err = angle_error(obs)
        if curr_err >= prev_err - 0.02:
            stuck_counter += 1
        else:
            stuck_counter = max(0, stuck_counter - 1)
        prev_err = curr_err

        if term or trunc:
            break

    final_err = min(abs(np.arctan2(obs[1], obs[0]) - PI_2),
                    2 * np.pi - abs(np.arctan2(obs[1], obs[0]) - PI_2))
    return final_err < 0.2, corrections


def main(n_episodes=15, device='mps', output_k=True):
    if device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'
    torch.manual_seed(42)
    np.random.seed(42)
    t0 = time.time()
    tag = 'with_k' if output_k else 'no_k'
    print(f'Online training ({tag}), {n_episodes} episodes')

    wm, wm_ms = load_models(device)
    fc = FeatureComputer(wm, device=device)
    dn = TinyDecisionNet(hidden=32, output_k=output_k).to(device)
    explorer = ActionExplorer(n_candidates=3, n_random=3)
    env = gym.make('Pendulum-v1')

    trial_seed = 142
    all_corrections_X, all_corrections_Y = [], []

    for ep in range(1, n_episodes + 1):
        ok, corrections = run_episode(
            fc, dn, wm_ms, explorer, env, trial_seed, device, output_k)

        # Accumulate corrections
        for feat_vec, label in corrections:
            all_corrections_X.append(feat_vec)
            all_corrections_Y.append(label)

        # Fine-tune on all corrections
        loss = fine_tune_dn(dn, all_corrections_X, all_corrections_Y,
                           epochs=200, device=device)
        n_corr = len(all_corrections_X)

        ok_str = 'OK' if ok else 'FAIL'
        print(f'  Ep {ep:2d}: {ok_str}  corrections={len(corrections)}  '
              f'total={n_corr}  dn_loss={loss:.4f}')

    # Final test
    print(f'\n=== Final test (10 trials) ===')
    env_test = gym.make('Pendulum-v1')
    ok = 0
    for t in range(10):
        trial_s = 42 + t * 100
        success, _ = run_episode(
            fc, dn, wm_ms, explorer, env_test, trial_s, device, output_k)
        if success:
            ok += 1
        obs, _ = env_test.reset(seed=trial_s)
        err = min(abs(np.arctan2(obs[1], obs[0]) - PI_2),
                  2 * np.pi - abs(np.arctan2(obs[1], obs[0]) - PI_2))
        print(f'  T{t+1}: {"OK" if success else "FAIL"}  '
              f'initial |dθ|={np.rad2deg(err):.0f}deg')
    print(f'\n{ok}/10  |  {time.time()-t0:.0f}s')
    env.close()
    env_test.close()

    out_path = os.path.join(os.path.dirname(__file__), f'tiny_dn_online_{tag}.pt')
    torch.save(dn.state_dict(), out_path)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=15)
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--no-k', action='store_true', default=False)
    args = parser.parse_args()
    main(n_episodes=args.episodes, device=args.device, output_k=not args.no_k)
