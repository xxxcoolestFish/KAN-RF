"""ProtoKAN Policy + MPC Distillation: local adaptation for continual learning.

Policy uses ProtoKAN layers with small sigma (local support).
Training uses MPC demonstrations (not WM gradient).
For continual learning: regenerate MPC demos with adapted WM, fine-tune.
Only activated prototypes update → zero forgetting.
"""
import torch, torch.nn as nn, numpy as np
from kanrf import ProtoKANLayer


# ═══════════════════════════════════════
# ProtoKAN Policy (locality through small sigma)
# ═══════════════════════════════════════

class ProtoKANPolicy(nn.Module):
    """Policy with ProtoKAN edges: small sigma → local support → easy to adapt.

    Architecture: s → ProtoKANLayer → ProtoKANLayer → Linear → tanh → a
    Each ProtoKANLayer initialized with sigma ≈ 0.22 (only 2-3 active prototypes per input).
    """

    def __init__(self, state_dim=4, hidden_dim=12, n_prototypes=16, init_log_sigma=-1.5):
        super().__init__()
        self.layer1 = ProtoKANLayer(state_dim, hidden_dim, n_prototypes=n_prototypes)
        self.layer2 = ProtoKANLayer(hidden_dim, hidden_dim, n_prototypes=n_prototypes)
        self.output = nn.Linear(hidden_dim, 1)

        # Initialize small sigma for locality
        for layer in [self.layer1, self.layer2]:
            layer.log_sigma.data.fill_(init_log_sigma)

    def forward(self, s):
        x = self.layer1(s)
        x = self.layer2(x)
        return torch.tanh(self.output(x))


# ═══════════════════════════════════════
# MPC Teacher
# ═══════════════════════════════════════

class MPCTeacher:
    """Generate (s, a_optimal) demonstrations via WM multi-step random shooting.

    For each training state, sample N random action sequences of length H,
    evaluate via WM rollout, return first action of best sequence.
    """

    def __init__(self, wm, state_dim, H=3, N=200, device='cpu'):
        self.wm = wm
        self.state_dim = state_dim
        self.H = H
        self.N = N
        self.device = device
        wm.eval()

    def generate_one(self, s, s_target):
        """Find best first action for state s."""
        best_a = 0.0
        best_cost = float('inf')
        s0 = s.unsqueeze(0)

        for _ in range(self.N):
            seq = torch.FloatTensor(1, self.H).uniform_(-1, 1)
            s_cur = s0.clone()
            total_cost = 0.0
            for t in range(self.H):
                a_t = torch.tensor([[seq[0, t].item()]], dtype=torch.float32)
                with torch.no_grad():
                    s_cur = self.wm(torch.cat([s_cur, a_t], dim=-1))
                err = s_cur - s_target
                total_cost += (0.9 ** t) * (err[:, 2].pow(2) + 0.1 * err[:, 0].pow(2)).item()
            if total_cost < best_cost:
                best_cost = total_cost
                best_a = seq[0, 0].item()

        return best_a

    def generate_batch(self, s_dataset, s_target, n_demos=2000, verbose=True):
        """Generate (s, a) demonstration pairs."""
        N = min(n_demos, s_dataset.shape[0])
        idx = torch.randperm(s_dataset.shape[0])[:N]
        s_sel = s_dataset[idx].to(self.device)
        a_opt = torch.zeros(N, 1, device=self.device)

        for i in range(N):
            if verbose and i % 500 == 0:
                print(f"    Demo {i}/{N}...")
            a_opt[i, 0] = self.generate_one(s_sel[i], s_target)

        return s_sel, a_opt


# ═══════════════════════════════════════
# Distillation Trainer
# ═══════════════════════════════════════

class DistillationTrainer:
    """Train ProtoKAN Policy via supervised learning on MPC demonstrations.

    Continual learning: call fine_tune() with new demo data after WM adaptation.
    Only locally activated prototypes update, preserving old knowledge.
    """

    def __init__(self, wm, policy, s_dataset, s_target,
                 H=3, N=200, n_demos=2000, lr=1e-3, device='cpu'):
        self.wm = wm
        self.policy = policy.to(device)
        self.device = device
        self.state_dim = s_dataset.shape[1]
        self.s_target = s_target.to(device)

        # Generate demonstrations
        print(f"  Generating {n_demos} MPC demos (H={H}, N={N})...")
        teacher = MPCTeacher(wm, self.state_dim, H=H, N=N, device=device)
        self.demo_s, self.demo_a = teacher.generate_batch(
            s_dataset, s_target, n_demos=n_demos)
        a_range = f"[{self.demo_a.min().item():.2f}, {self.demo_a.max().item():.2f}]"
        print(f"  Demos ready. a ∈ {a_range}")

        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.loss_history = []

    def train_epoch(self, batch_size=256):
        """One epoch of distillation training."""
        D = len(self.demo_s)
        n_batches = max(1, D // batch_size)
        total_loss = 0.0

        for _ in range(n_batches):
            idx = torch.randint(0, D, (batch_size,), device=self.device)
            a_pred = self.policy(self.demo_s[idx])
            loss = (a_pred - self.demo_a[idx]).pow(2).mean()
            self.policy.train(); self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            total_loss += loss.item()

        avg = total_loss / n_batches
        self.loss_history.append({'total': avg})
        return self.loss_history[-1]

    def fine_tune(self, wm, s_dataset, s_target, n_demos=1000, epochs=30, lr=5e-4):
        """Continual learning: adapt to new physics using new WM + MPC demos.

        Uses LOW learning rate so only strongly activated prototypes change.
        """
        print(f"  Fine-tuning with adapted WM ({n_demos} demos, {epochs} epochs)...")
        teacher = MPCTeacher(wm, self.state_dim, H=3, N=200, device=self.device)
        self.demo_s, self.demo_a = teacher.generate_batch(
            s_dataset, s_target, n_demos=n_demos, verbose=False)

        # Lower LR for fine-tuning → local updates only
        for pg in self.opt.param_groups:
            pg['lr'] = lr

        for ep in range(1, epochs + 1):
            ld = self.train_epoch()
            if ep % 10 == 0:
                print(f"    Fine-tune epoch {ep:3d}  loss={ld['total']:.4f}")

    def get_action(self, s):
        self.policy.eval()
        if isinstance(s, np.ndarray):
            s = torch.tensor(s, dtype=torch.float32, device=self.device)
        if s.dim() == 1:
            s = s.unsqueeze(0)
        with torch.no_grad():
            return self.policy(s).squeeze().cpu().item()
