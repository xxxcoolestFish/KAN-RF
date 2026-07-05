"""Train KAN policy via frozen CWS-KAN world model gradient on Pendulum."""
import torch, numpy as np, sys, os, argparse, time
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import KAN
from control.kan_policy_net import KANPolicy, KANPolicyTrainer, SafeKANPolicy
from control.kan_knowledge import KANKnowledge, CombinedUncertainty

PI_2 = np.pi / 2
S_TARGET = torch.tensor([[0.0, 1.0, 0.0]])


def generate_states(n=20000, method='mixed'):
    """Generate training states covering pendulum state space."""
    if method == 'uniform':
        angles = np.random.uniform(-np.pi, np.pi, n)
        s_raw = np.stack([
            np.cos(angles), np.sin(angles),
            np.random.uniform(-8.0, 8.0, n)
        ], axis=1)
    elif method == 'rollout':
        env = gym.make('Pendulum-v1')
        states = []
        obs, _ = env.reset()
        while len(states) < n:
            a = env.action_space.sample()
            obs, _, term, trunc, _ = env.step(a)
            states.append(obs.copy())
            if term or trunc:
                obs, _ = env.reset()
        env.close()
        s_raw = np.array(states[:n])
    else:  # mixed
        n_half = n // 2
        angles = np.random.uniform(-np.pi, np.pi, n_half)
        s1 = np.stack([np.cos(angles), np.sin(angles),
                       np.random.uniform(-8.0, 8.0, n_half)], axis=1)
        env = gym.make('Pendulum-v1')
        states = []; obs, _ = env.reset()
        while len(states) < n_half:
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            states.append(obs.copy())
            if term or trunc: obs, _ = env.reset()
        env.close()
        s2 = np.array(states[:n_half])
        s_raw = np.vstack([s1, s2]); np.random.shuffle(s_raw)

    s_norm = s_raw.copy(); s_norm[:, 2] /= 8.0
    return torch.tensor(s_norm, dtype=torch.float32)


def evaluate(policy_fn, n_trials=10, max_steps=300, label='', verbose=True):
    """Evaluate a policy function on Pendulum-v1."""
    env = gym.make('Pendulum-v1')
    successes = 0; all_steps = []; all_errors = []
    for trial in range(n_trials):
        seed = 42 + trial * 100; obs, _ = env.reset(seed=seed)
        for step in range(max_steps):
            s_norm = np.array([obs[0], obs[1], obs[2]/8.0], dtype=np.float32)
            result = policy_fn(s_norm)
            a_norm = result[0] if isinstance(result, tuple) else result
            a_raw = a_norm * 2.0
            obs, _, term, trunc, _ = env.step([a_raw])
            err = min(abs(np.arctan2(obs[1], obs[0]) - PI_2),
                     2*np.pi - abs(np.arctan2(obs[1], obs[0]) - PI_2))
            if err < 0.2:
                successes += 1; all_steps.append(step+1); all_errors.append(err); break
        else:
            all_steps.append(max_steps)
            err = min(abs(np.arctan2(obs[1], obs[0]) - PI_2),
                     2*np.pi - abs(np.arctan2(obs[1], obs[0]) - PI_2))
            all_errors.append(err)
        if verbose:
            print(f"  [{label}] Trial {trial+1:2d}  {'✓' if all_errors[-1]<0.2 else '✗'}  "
                  f"steps={all_steps[-1]:3d}  err={all_errors[-1]:.3f}")

    env.close(); sr = successes/n_trials
    print(f"  [{label}] {successes}/{n_trials} ({sr*100:.0f}%)  "
          f"mean|Δθ|={np.mean(all_errors):.3f}±{np.std(all_errors):.3f}")
    return sr


def analyze_policy(policy):
    """Print edge function analysis of trained KAN policy."""
    edges = policy.get_edge_functions()
    print(f"\n  KAN Policy Structure:")
    for i, e in enumerate(edges):
        sw = e['spline_weight']  # (out, in, n_basis)
        mean_mag = np.abs(sw).mean(axis=-1)  # (out, in)
        n_active = (mean_mag > 0.01).sum()
        print(f"    Layer {i}: {e['in_dim']}→{e['out_dim']}, "
              f"active_edges={n_active}/{e['in_dim']*e['out_dim']}")

    # Attribute a few test states
    print(f"\n  Attribution examples:")
    for label, s in [('bottom', [0, -1, 0]), ('mid', [0, 0, 0.5]), ('upright', [0, 1, 0])]:
        s_t = torch.tensor([s], dtype=torch.float32)
        attr = policy.attribute(s_t)
        print(f"    {label:8s}: {attr[0]:.3f}(cos) {attr[1]:.3f}(sin) {attr[2]:.3f}(thd)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wm', type=str, default='/tmp/kanrf_cl_exp/kan_cws_cl.pt')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--hidden', type=int, default=8)
    parser.add_argument('--n-layers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output', type=str, default='/tmp/kan_policy_trained.pt')
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── 1. Load frozen world model ──
    wm = KAN([4, 12, 3], grid_size=5, spline_order=3)
    wm.load_state_dict(torch.load(args.wm, weights_only=True, map_location=device))
    wm.to(device); wm.eval()
    print(f"World model: {sum(p.numel() for p in wm.parameters())} params")

    # ── 2. Create KAN policy ──
    policy = KANPolicy(state_dim=3, action_dim=1, hidden_dim=args.hidden,
                       n_layers=args.n_layers, grid_size=5, spline_order=3)
    n_params = sum(p.numel() for p in policy.parameters())
    dims = [3] + [args.hidden]*args.n_layers + [1]
    print(f"KAN Policy: {dims}, {n_params} params")

    # ── 3. Generate training states ──
    s_dataset = generate_states(20000, 'mixed').to(device)
    print(f"Training states: {s_dataset.shape}")

    # ── 4. Train ──
    s_target = S_TARGET.to(device)
    trainer = KANPolicyTrainer(wm, policy, s_target, lr=args.lr, device=device)

    print(f"\nTraining ({args.epochs} epochs)...")
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        ld = trainer.train_epoch(s_dataset)
        if ep % 20 == 0:
            print(f"  Epoch {ep:3d}  loss={ld['total']:.4f}  "
                  f"energy={ld['energy']:.4f}  dist={ld['dist']:.4f}")
    print(f"  Done in {time.time()-t0:.0f}s")

    # ── 5. Save ──
    torch.save(policy.state_dict(), args.output)
    print(f"Saved: {args.output}")

    # ── 6. Evaluate ──
    print(f"\n{'='*60}\nEvaluation\n{'='*60}")
    print("\n  [KAN Policy (trained via WM gradient)]")
    evaluate(lambda s: trainer.get_action(s), n_trials=10, label='kan')

    # ── 7. Analyze ──
    print(f"\n{'='*60}\nPolicy Analysis\n{'='*60}")
    analyze_policy(policy)


if __name__ == '__main__':
    main()
