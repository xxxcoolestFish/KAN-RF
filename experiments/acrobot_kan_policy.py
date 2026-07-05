"""Acrobot: KAN Policy via frozen multi-scale WM gradient.

State: 6D [cosθ1, sinθ1, cosθ2, sinθ2, dθ1/6, dθ2/8]
Action: discrete {0,1,2} → policy outputs 3 logits → softmax → action_probs
WM: KAN([10,24,6]) multi-scale (k=1), one-hot action input
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, time, sys, os, gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import KAN

MAX_V1 = 6.0; MAX_V2 = 8.0
S_TARGET = torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0, 0.0]])  # both links upright


class AcrobotPolicy(nn.Module):
    """KAN Policy for discrete 3-action output."""
    def __init__(self, state_dim=6, hidden=12, n_layers=2):
        super().__init__()
        from kanrf import KANLayer
        self.layers = nn.ModuleList()
        d = state_dim
        for _ in range(n_layers):
            self.layers.append(KANLayer(d, hidden, grid_size=5, spline_order=3))
            d = hidden
        self.out = nn.Linear(d, 3)  # 3 logits

    def forward(self, s, return_activations=False):
        x = s
        Bs, Es = [], []
        for layer in self.layers:
            if return_activations:
                x, B, E = layer(x, return_activations=True)
                Bs.append(B); Es.append(E)
            else:
                x = layer(x)
        logits = self.out(x)
        probs = F.softmax(logits, dim=-1)
        if return_activations:
            return probs, Bs, Es
        return probs


def make_wm_input(s, action_probs):
    """Build WM input: [state(6), action_probs(3)] (single-scale, no k_norm)."""
    return torch.cat([s, action_probs], dim=-1)


def train():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wm', type=str, default='acrobot/acrobot_wm.pt')
    parser.add_argument('--data', type=str, default='acrobot/acrobot_data_ms.pt')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--hidden', type=int, default=12)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output', type=str, default='/tmp/acrobot_kan_policy.pt')
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load WM
    print("Loading Acrobot WM...")
    wm = KAN([9, 24, 6], grid_size=5, spline_order=3)
    wm.load_state_dict(torch.load(args.wm, weights_only=True, map_location=device))
    wm.to(device); wm.eval()
    print(f"  WM: {sum(p.numel() for p in wm.parameters())} params")

    # Load k=1 training data
    x_data, y_data = torch.load(args.data, weights_only=True, map_location=device)
    x_data, y_data = x_data.float(), y_data.float()
    # Use all data states (WM is single-scale, no k_norm)
    s_states = x_data[:, :6]
    print(f"  states: {s_states.shape[0]}")

    # Policy
    policy = AcrobotPolicy(state_dim=6, hidden=args.hidden, n_layers=2).to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  Policy: [6,{args.hidden},{args.hidden},3] = {n_params} params")

    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    s_dataset = s_states.to(device)
    N = len(s_dataset)
    batch_size = 256; n_batches = 60

    print(f"\nTraining ({args.epochs} epochs)...")
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        total_loss = 0
        for _ in range(n_batches):
            idx = torch.randint(0, N, (batch_size,), device=device)
            s_b = s_dataset[idx]

            policy.train(); opt.zero_grad()
            ap = policy(s_b)
            wm_in = make_wm_input(s_b, ap)
            s_pred = wm(wm_in)

            # Loss: push toward upright
            w = torch.tensor([5.,5.,5.,5.,1.,1.], device=device)
            loss = ((s_pred - S_TARGET.to(device)).pow(2) * w).mean()
            # Entropy bonus to prevent premature convergence
            loss = loss - 0.01 * (ap * (ap + 1e-8).log()).sum(-1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            opt.step()
            policy.eval()
            total_loss += loss.item()

        if ep % 60 == 0:
            print(f"  Epoch {ep:3d}  loss={total_loss/n_batches:.4f}")

    print(f"  Done in {time.time()-t0:.0f}s")
    torch.save(policy.state_dict(), args.output)
    print(f"  Saved: {args.output}")

    # Evaluate
    print(f"\n{'='*60}")
    print("Acrobot Evaluation (10 trials)")
    print("=" * 60)
    env = gym.make('Acrobot-v1')
    successes = 0; all_steps = []
    for trial in range(10):
        seed = 42 + trial * 100
        obs, _ = env.reset(seed=seed); ok = False
        for step in range(500):
            s_n = torch.tensor([[obs[0], obs[1], obs[2], obs[3],
                                obs[4]/MAX_V1, obs[5]/MAX_V2]],
                               dtype=torch.float32, device=device)
            with torch.no_grad():
                ap = policy(s_n).squeeze().cpu().numpy()
            action = int(np.argmax(ap))
            obs, _, term, trunc, _ = env.step(action)
            if term:
                successes += 1; all_steps.append(step + 1); ok = True; break
        if not ok: all_steps.append(500)
        print(f"  Trial {trial+1:2d}  {'✓' if ok else '✗'}  steps={all_steps[-1]}")
    env.close()
    sr = successes / 10
    print(f"\n  Result: {successes}/10 ({sr*100:.0f}%)  mean_steps={np.mean(all_steps):.0f}")


if __name__ == '__main__':
    train()
