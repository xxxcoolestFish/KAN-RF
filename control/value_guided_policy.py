"""Value-Guided Policy: use LQR cost-to-go P as loss function.

Instead of imitating LQR actions (distillation), Policy minimizes
V(s') = (s' - target)^T P (s' - target), where P is the LQR cost-to-go
matrix encoding multi-step dynamics and hierarchy structure.

Key advantage: Policy retains freedom to find smooth actions, unlike
distillation which forces imitation of bang-bang LQR solutions.
"""
import torch, numpy as np
from control.hierarchical_lqr import HierarchicalLQR
from control.thtp import TemporalHierarchy


class ValueGuidedTrainer:
    """Train Policy using LQR value function as loss.

    Pre-computes averaged P matrix from LQR solutions on training states.
    Training loss = WM_gradient_loss + λ * (s_pred - target)^T P (s_pred - target)
    """

    def __init__(self, wm, policy, hierarchy, s_dataset, s_target,
                 lr=1e-3, horizon=4, q_base=10.0, lambda_value=0.5,
                 n_value_samples=300, device='cpu'):
        self.wm = wm
        self.policy = policy.to(device)
        self.s_target = s_target.to(device)
        self.lambda_value = lambda_value
        self.device = device
        state_dim = s_dataset.shape[1]

        wm.eval()
        for p in wm.parameters():
            p.requires_grad = False

        # Setup H-LQR
        self.hlqr = HierarchicalLQR(wm, hierarchy, horizon=horizon,
                                     q_base=q_base, device=device)

        # Pre-compute averaged P matrix from LQR solutions
        print(f"  Computing value function P from {n_value_samples} LQR solutions...")
        N = min(n_value_samples, s_dataset.shape[0])
        idx = torch.randperm(s_dataset.shape[0])[:N]
        P_sum = torch.zeros(state_dim, state_dim, device=device)
        for i in range(N):
            _, _, _, P = self._solve_and_get_P(s_dataset[idx[i]], s_target.squeeze(0))
            P_sum += P
        self.P_avg = P_sum / N
        print(f"  P_avg diagonal: {self.P_avg.diag().tolist()}")
        print(f"  P_avg trace: {self.P_avg.trace().item():.2f}")

        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.loss_history = []

    def _solve_and_get_P(self, s, s_target):
        """Solve LQR and return P matrix (terminal cost-to-go)."""
        u_0, _, _, _ = self.hlqr.solve(s, s_target)

        # Re-run the Riccati to extract P (terminal)
        n = s.shape[0]
        A, B, f0 = self.hlqr.linearize(s)
        c = f0 - A @ s
        d = (A - torch.eye(n, device=self.device)) @ s_target + c
        P = self.hlqr.Q.clone()
        for _ in range(self.hlqr.horizon):
            BtPB = B.T @ P @ B
            R_eff = self.hlqr.r_weight + BtPB
            BtPA = B.T @ P @ A
            AtPA = A.T @ P @ A
            AtPB = A.T @ P @ B
            P = self.hlqr.Q + AtPA - (AtPB @ BtPA) / R_eff
        return u_0, A, B, P

    def train_epoch(self, s_dataset, batch_size=256):
        N = s_dataset.shape[0]
        n_batches = max(1, N // batch_size)
        total_loss = 0.0

        for _ in range(n_batches):
            idx = torch.randint(0, N, (batch_size,), device=self.device)
            s_b = s_dataset[idx]
            self.policy.train()
            self.opt.zero_grad()

            a = self.policy(s_b)
            s_pred = self.wm(torch.cat([s_b, a], dim=-1))

            # WM gradient loss (energy-guided or stabilization)
            if s_b.shape[1] == 3:  # Pendulum
                thd = s_b[:, 2] * 8.0
                Ec = 0.5 * thd.pow(2) + 10.0 * s_b[:, 1]
                thdp = s_pred[:, 2] * 8.0
                Ep = 0.5 * thdp.pow(2) + 10.0 * s_pred[:, 1]
                deficit = (10.0 - Ec).detach()
                egain = (Ep - Ec) * torch.sign(deficit)
                sin = s_b[:, 1]
                ws = ((1.0 + sin) / 2.0).clamp(0, 1)
                wm_loss = (-egain.mean() +
                           (ws * (s_pred - self.s_target.expand(batch_size, -1))
                            .pow(2).sum(-1)).mean() +
                           0.01 * a.pow(2).mean())
            else:  # CartPole
                wm_loss = (s_pred[:, 2].pow(2).mean() +
                           0.1 * s_pred[:, 0].pow(2).mean() +
                           0.5 * s_pred[:, 3].pow(2).mean() +
                           0.01 * a.pow(2).mean())

            # Value function loss: (s' - target)^T P (s' - target)
            err = s_pred - self.s_target.expand(batch_size, -1)
            value_loss = (err @ self.P_avg @ err.T).diag().mean()

            loss = wm_loss + self.lambda_value * value_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
            self.opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / n_batches
        self.loss_history.append({'total': avg_loss})
        return self.loss_history[-1]

    def get_action(self, s):
        self.policy.eval()
        if isinstance(s, np.ndarray):
            s = torch.tensor(s, dtype=torch.float32, device=self.device)
        if s.dim() == 1:
            s = s.unsqueeze(0)
        with torch.no_grad():
            return self.policy(s).squeeze().cpu().item()
