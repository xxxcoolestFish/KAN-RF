"""Acrobot decision_v3: compare single-step vs sequence policy.

Acrobot-v1: chaotic underactuated 2-link robot.
State: [cosθ1, sinθ1, cosθ2, sinθ2, dθ1/6, dθ2/8]  (6D)
Action: discrete {0, 1, 2} → policy outputs 3 logits → softmax
Target: tip above the line [cosθ1≈1, sinθ1≈0, cosθ2≈1, sinθ2≈0, 0, 0]

Key test: Acrobot NEEDS multi-step coordination (pumping). Single-step
gradient may be insufficient — this is where sequence policy should win.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, argparse, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import KAN

# Constants from Acrobot-v1
DT = 0.2   # timestep
MAX_VEL1 = 6.0; MAX_VEL2 = 8.0  # normalization bounds
TARGET = torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0, 0.0]])  # both links upright

# Existing model architectures from train_acrobot.py
# Input: state(6) + action_onehot(3) + k_norm(1) = 10 dims
# For k=1, k_norm = 1/8 = 0.125


class AcrobotPolicy(nn.Module):
    """MLP policy: state(6) → 3 logits → softmax → action_probs."""
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, s):
        return F.softmax(self.net(s), dim=-1)  # (B, 3) action probabilities


def make_kan_input(s, action_probs, k_norm=0.125):
    """Build KAN input: [state(6), action_probs(3), k_norm(1)]."""
    B = s.shape[0]
    k = torch.full((B, 1), k_norm, device=s.device, dtype=s.dtype)
    return torch.cat([s, action_probs, k], dim=-1)


class SingleStepTrainer:
    def __init__(self, kan, policy, lr=1e-3, clip_grad=10.0, device='cpu'):
        self.kan = kan; self.policy = policy.to(device); self.device = device
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.clip_grad = clip_grad
        self.loss_history = []
        self.kan.eval()
        for p in self.kan.parameters(): p.requires_grad = False

    def train_step(self, s_batch):
        B = s_batch.shape[0]; self.policy.train(); self.optimizer.zero_grad()
        ap = self.policy(s_batch)  # (B, 3) softmax probs
        kan_in = make_kan_input(s_batch, ap)
        s_pred = self.kan(kan_in)

        # Loss: push state toward upright target
        # Weight: angle dimensions matter most
        w = torch.tensor([5.0, 5.0, 5.0, 5.0, 1.0, 1.0], device=self.device)
        loss = ((s_pred - TARGET.to(self.device).expand(B, -1)).pow(2) * w).mean()

        # Entropy bonus to prevent premature convergence to one action
        entropy = -(ap * (ap + 1e-8).log()).sum(dim=-1).mean()
        total = loss - 0.01 * entropy
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_grad)
        self.optimizer.step()
        self.loss_history.append({'loss': loss.item(), 'entropy': entropy.item()})
        return self.loss_history[-1]

    def train_epoch(self, s_dataset, batch_size=256):
        N = s_dataset.shape[0]
        losses = []
        for _ in range(max(1, N // batch_size)):
            idx = torch.randint(0, N, (batch_size,), device=self.device)
            losses.append(self.train_step(s_dataset[idx]))
        return {k: np.mean([l[k] for l in losses]) for k in losses[0]}


class SequenceTrainer(SingleStepTrainer):
    def __init__(self, *args, horizon=4, gamma=0.85, **kwargs):
        super().__init__(*args, **kwargs)
        self.horizon = horizon; self.gamma = gamma

    def train_step(self, s_batch):
        B = s_batch.shape[0]; self.policy.train(); self.optimizer.zero_grad()
        w = torch.tensor([5.0, 5.0, 5.0, 5.0, 1.0, 1.0], device=self.device)

        s = s_batch; total_loss = torch.tensor(0.0, device=self.device)
        for t in range(self.horizon):
            ap = self.policy(s)
            s = self.kan(make_kan_input(s, ap))
            step_loss = ((s - TARGET.to(self.device).expand(B, -1)).pow(2) * w).mean()
            total_loss = total_loss + (self.gamma ** t) * step_loss

        total = total_loss / self.horizon
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_grad)
        self.optimizer.step()
        self.loss_history.append({'loss': total.item(), 'entropy': 0.0})
        return self.loss_history[-1]


def evaluate(policy, device, n_trials=10, max_steps=500, label=''):
    """Evaluate on Acrobot-v1 using policy that outputs action probabilities."""
    import gymnasium as gym
    env = gym.make('Acrobot-v1')
    successes = []; all_steps = []
    for trial in range(n_trials):
        seed = 42 + trial * 100
        obs, _ = env.reset(seed=seed)
        for step in range(max_steps):
            s_norm = torch.tensor([
                obs[0], obs[1], obs[2], obs[3],
                obs[4]/MAX_VEL1, obs[5]/MAX_VEL2
            ], dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                ap = policy(s_norm).squeeze().cpu().numpy()
            action = int(np.argmax(ap))

            obs, _, term, trunc, _ = env.step(action)
            if term:
                successes.append(True); all_steps.append(step + 1)
                break
        else:
            successes.append(False); all_steps.append(max_steps)
        if term:
            break  # already succeeded, continue outer loop

    env.close()
    sr = sum(successes) / n_trials
    print(f"  {label}: {sum(successes)}/{n_trials} ({sr*100:.0f}%)  "
          f"mean_steps={np.mean(all_steps):.0f}")
    return sr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--horizon', type=int, default=4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--cache-dir', type=str, default='/tmp/kanrf_cl_ac')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.cache_dir, exist_ok=True)

    # ── Load existing Acrobot KAN ──
    wm_path = 'acrobot/acrobot_wm.pt'
    if not os.path.exists(wm_path):
        wm_path = os.path.join(os.path.dirname(__file__), '..', 'acrobot/acrobot_wm.pt')
    print(f"Loading KAN: {wm_path}")
    kan = KAN([10, 24, 6], grid_size=5, spline_order=3)
    kan.load_state_dict(torch.load(wm_path, weights_only=True, map_location=device))
    kan.to(device); kan.eval()
    print(f"  params: {sum(p.numel() for p in kan.parameters())}")

    # ── Load k=1 training data ──
    data_path = 'acrobot/acrobot_data_ms.pt'
    x, y = torch.load(data_path, weights_only=True, map_location=device)
    x, y = x.float(), y.float()
    mask_k1 = (x[:, -1] == 0.125)  # k=1 samples only
    x_k1 = x[mask_k1]
    y_k1 = y[mask_k1]
    s_states = x_k1[:, :6]  # just the state part
    print(f"  k=1 samples: {len(s_states)}")

    # ── Train policies ──
    n_train = min(4000, len(s_states))
    s_dataset = s_states[:n_train].to(device)
    print(f"\nTraining on {n_train} states")

    # Single-step
    print("\n[Single-step]")
    p1 = AcrobotPolicy().to(device)
    t1 = SingleStepTrainer(kan, p1, device=device)
    for ep in range(1, args.epochs + 1):
        ld = t1.train_epoch(s_dataset)
        if ep % 40 == 0: print(f"  Epoch {ep:3d}  loss={ld['loss']:.4f}  ent={ld['entropy']:.4f}")

    # Sequence
    print(f"\n[Sequence H={args.horizon}]")
    p2 = AcrobotPolicy().to(device)
    t2 = SequenceTrainer(kan, p2, horizon=args.horizon, device=device)
    for ep in range(1, args.epochs + 1):
        ld = t2.train_epoch(s_dataset)
        if ep % 40 == 0: print(f"  Epoch {ep:3d}  loss={ld['loss']:.4f}")

    # ── Evaluate ──
    print(f"\n{'='*60}\nEvaluation (Acrobot-v1, {10} trials)\n{'='*60}")
    p1.eval(); p2.eval()
    evaluate(p1, device, label='Single-step')
    evaluate(p2, device, label=f'Sequence H={args.horizon}')

    print("\nDone!")


if __name__ == '__main__':
    main()
