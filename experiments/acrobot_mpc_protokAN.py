"""ProtoKAN WM + Model Predictive Search for Acrobot.

Instead of training a policy via WM gradient (which fails for discrete actions),
directly evaluate all 3 actions through the WM at each step and pick the best.

This is the purest test of WM decision quality.
"""
import torch, numpy as np, time, sys, os, gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import ProtoKAN

MAX_V1 = 6.0; MAX_V2 = 8.0
S_TARGET = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # both upright

def tip_height(s):
    """Height of Acrobot tip. Maximize this to encourage swing-up.
    h = cosθ1 + cos(θ1+θ2) = s[0] + s[0]*s[2] - s[1]*s[3]
    Maximum (2.0) when both links upright. Minimum (-2.0) when both hang down."""
    cos_th1, sin_th1 = s[..., 0], s[..., 1]
    cos_th2, sin_th2 = s[..., 2], s[..., 3]
    cos_th12 = cos_th1 * cos_th2 - sin_th1 * sin_th2
    return cos_th1 + cos_th12  # range [-2, 2]


def cost_to_go(s, method='height'):
    """Cost of a state. Lower is better."""
    if method == 'height':
        # Maximize tip height
        return -tip_height(s)
    elif method == 'target':
        # Distance to upright target
        w = torch.tensor([5., 5., 5., 5., 1., 1.])
        return ((s - S_TARGET) ** 2 * w).sum(dim=-1)
    elif method == 'energy':
        # Negative total energy (potential + kinetic)
        h = tip_height(s)
        v_sq = s[..., 4] ** 2 + s[..., 5] ** 2
        return -(h + 0.1 * v_sq)
    else:
        raise ValueError(f"Unknown method: {method}")


def try_all_actions(wm, s, device='cpu', cost_method='height'):
    """Evaluate all 3 actions through WM, return best action."""
    best_action = 0
    best_cost = float('inf')
    best_s_next = None

    for a in range(3):
        a_oh = torch.zeros(1, 3, device=device); a_oh[0, a] = 1.0
        wm_in = torch.cat([s.unsqueeze(0), a_oh], dim=-1)
        with torch.no_grad():
            s_pred = wm(wm_in).squeeze(0)
        c = cost_to_go(s_pred, method=cost_method).item()
        if c < best_cost:
            best_cost = c
            best_action = a
            best_s_next = s_pred

    return best_action, best_cost, best_s_next


def mpc_lookahead(wm, s, depth=2, device='cpu', cost_method='height'):
    """Multi-step lookahead: evaluate all action sequences of length `depth`."""
    n_actions = 3
    n_seqs = n_actions ** depth
    best_seq = None
    best_final_cost = float('inf')

    for seq_idx in range(n_seqs):
        seq = []
        tmp = seq_idx
        for _ in range(depth):
            seq.append(tmp % n_actions)
            tmp //= n_actions

        s_cur = s.clone()
        for a in seq:
            a_oh = torch.zeros(1, 3, device=device); a_oh[0, a] = 1.0
            wm_in = torch.cat([s_cur.unsqueeze(0), a_oh], dim=-1)
            with torch.no_grad():
                s_cur = wm(wm_in).squeeze(0)

        c = cost_to_go(s_cur, method=cost_method).item()
        if c < best_final_cost:
            best_final_cost = c
            best_seq = seq

    return best_seq[0] if best_seq else 0


def random_shooting(wm, s, horizon=6, n_samples=200, device='cpu', cost_method='height'):
    """Random shooting MPC: sample random action sequences, pick best first action."""
    n_actions = 3
    best_cost = float('inf')
    best_first_action = 0

    # Sample random sequences
    for _ in range(n_samples):
        seq = torch.randint(0, n_actions, (horizon,), device=device)

        s_cur = s.clone()
        for a in seq:
            a_oh = torch.zeros(1, 3, device=device); a_oh[0, a] = 1.0
            wm_in = torch.cat([s_cur.unsqueeze(0), a_oh], dim=-1)
            with torch.no_grad():
                s_cur = wm(wm_in).squeeze(0)

        c = cost_to_go(s_cur, method=cost_method).item()
        if c < best_cost:
            best_cost = c
            best_first_action = seq[0].item()

    return best_first_action


def evaluate(wm, n_trials=10, method='1-step', depth=2, horizon=6,
             n_samples=200, cost_method='height', device='cpu'):
    """Test MPC on Acrobot."""
    env = gym.make('Acrobot-v1')
    successes = 0
    all_steps = []
    t0 = time.time()

    for trial in range(n_trials):
        seed = 42 + trial * 100
        obs, _ = env.reset(seed=seed)
        ok = False

        for step in range(500):
            s_n = torch.tensor([obs[0], obs[1], obs[2], obs[3],
                               obs[4]/MAX_V1, obs[5]/MAX_V2],
                               dtype=torch.float32, device=device)

            if method == '1-step':
                action, _, _ = try_all_actions(wm, s_n, device=device, cost_method=cost_method)
            elif method == 'lookahead':
                action = mpc_lookahead(wm, s_n, depth=depth, device=device,
                                       cost_method=cost_method)
            elif method == 'shooting':
                action = random_shooting(wm, s_n, horizon=horizon,
                                         n_samples=n_samples, device=device,
                                         cost_method=cost_method)
            else:
                raise ValueError(f"Unknown method: {method}")

            obs, _, term, trunc, _ = env.step(action)
            if term:
                successes += 1
                all_steps.append(step + 1)
                ok = True
                break

        if not ok:
            all_steps.append(500)
        elapsed = time.time() - t0
        status = '✓' if ok else '✗'
        print(f"  [{trial+1:2d}] {status} steps={all_steps[-1]}  ({elapsed:.0f}s)")

    env.close()
    return successes, all_steps


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wm', type=str, default='/tmp/acrobot_protokAN_wm.pt')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--depth', type=int, default=1, help='MPC lookahead depth')
    parser.add_argument('--method', type=str, default='1-step',
                        choices=['1-step', 'lookahead', 'shooting'])
    parser.add_argument('--cost', type=str, default='height',
                        choices=['height', 'target', 'energy'])
    parser.add_argument('--horizon', type=int, default=6)
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--trials', type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(42); np.random.seed(42)

    print("=" * 70)
    print(f"ProtoKAN WM MPC: {args.method} (cost={args.cost}) on Acrobot")
    print("=" * 70)

    # Load WM
    wm = ProtoKAN([9, 32, 6], n_prototypes=16).to(device)
    wm.load_state_dict(torch.load(args.wm, weights_only=True, map_location=device))
    wm.eval()
    print(f"Loaded WM: {sum(p.numel() for p in wm.parameters())} params")

    # Quick WM accuracy check
    print("\nWM sanity check — predict next state for 3 actions from random state:")
    s_test = torch.tensor([0.5, -0.8, 0.3, 0.9, 0.1, -0.2], device=device)
    for a in range(3):
        a_oh = torch.zeros(1, 3, device=device); a_oh[0, a] = 1.0
        wm_in = torch.cat([s_test.unsqueeze(0), a_oh], dim=-1)
        with torch.no_grad():
            sp = wm(wm_in).squeeze(0)
        print(f"  Action {a} ({['-1',' 0','+1'][a]}): s' = [{sp[0]:.3f}, {sp[1]:.3f}, {sp[2]:.3f}, {sp[3]:.3f}, {sp[4]:.3f}, {sp[5]:.3f}]")

    # Run trials
    print(f"\nRunning {args.trials} trials...")
    successes, all_steps = evaluate(wm, n_trials=args.trials, method=args.method,
                                    depth=args.depth, horizon=args.horizon,
                                    n_samples=args.n_samples,
                                    cost_method=args.cost, device=device)

    print(f"\n{'='*70}")
    print(f"RESULT: {successes}/{args.trials} ({successes*100/args.trials:.0f}%)")
    print(f"  Mean steps: {np.mean(all_steps):.0f}")
    print(f"  Previous (KAN WM gradient policy): 0/10")
    print(f"  Previous (ProtoKAN WM gradient policy): 0/10")
    print("=" * 70)


if __name__ == '__main__':
    main()
