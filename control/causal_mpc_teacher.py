"""Causal MPC Teacher: use hierarchy-informed cost + WM multi-step rollout
to generate optimal action demonstrations for Policy training.

Key innovation: the causal hierarchy determines WHAT to optimize (cost function),
the WM determines HOW (multi-step simulation). Policy learns from the result.
"""
import torch, numpy as np
from control.thtp import TemporalHierarchy


class CausalCostFunction:
    """Cost function: cost(s) = Σ_i w_i * (s[i] - target[i])²

    w_i = γ^(max_tier - tier[i]) — deeper tiers (goals) get exponentially higher weight
    This is derived from the causal hierarchy: Tier 0 = means, Tier k = goals.
    """

    def __init__(self, hierarchy: TemporalHierarchy, gamma=2.0):
        max_tier = max(hierarchy.tier_of)
        self.weights = torch.tensor(
            [gamma ** (max_tier - hierarchy.tier_of[i])
             for i in range(hierarchy.state_dim)],
            dtype=torch.float32
        )

    def __call__(self, s, s_target):
        """Compute scalar cost."""
        err = s - s_target
        return (err ** 2 * self.weights.unsqueeze(0)).sum(dim=-1)


class CausalMPCTeacher:
    """Generate optimal action demonstrations via WM multi-step random shooting.

    For each state, samples N random action sequences of length H,
    evaluates total hierarchy-weighted cost via WM rollout,
    returns the first action of the best sequence.
    """

    def __init__(self, wm, hierarchy: TemporalHierarchy,
                 horizon=6, n_samples=500, gamma=2.0, device='cpu'):
        self.wm = wm
        self.horizon = horizon
        self.n_samples = n_samples
        self.device = device
        self.cost_fn = CausalCostFunction(hierarchy, gamma=gamma)
        wm.eval()

    def generate_one(self, s, s_target):
        """Generate optimal first action for a single state.

        Args:
            s: (state_dim,) current state
            s_target: (state_dim,) target state

        Returns:
            best_a: optimal first action
            best_cost: achieved cost over horizon
        """
        best_a = 0.0
        best_cost = float('inf')
        state_dim = len(s)

        for _ in range(self.n_samples):
            seq = torch.FloatTensor(1, self.horizon).uniform_(-1, 1)
            total_cost = 0.0
            s_cur = s.clone().unsqueeze(0)  # (1, n)

            for t in range(self.horizon):
                a_t = seq[0, t].unsqueeze(0).unsqueeze(0)  # (1, 1)
                with torch.no_grad():
                    s_cur = self.wm(torch.cat([s_cur, a_t], dim=-1))
                total_cost += self.cost_fn(s_cur, s_target.unsqueeze(0)).item()

            if total_cost < best_cost:
                best_cost = total_cost
                best_a = seq[0, 0].item()

        return best_a, best_cost

    def generate_batch(self, s_dataset, s_target, n_demos=2000):
        """Pre-compute demonstrations: (s, a_optimal) pairs.

        Returns:
            demo_s: (n_demos, state_dim)
            demo_a: (n_demos, 1)
        """
        N = min(n_demos, s_dataset.shape[0])
        idx = torch.randperm(s_dataset.shape[0])[:N]
        s_sel = s_dataset[idx]
        demo_a = torch.zeros(N, 1, device=self.device)

        for i in range(N):
            if i % 500 == 0:
                print(f"    Generating demo {i}/{N}...")
            a_opt, _ = self.generate_one(s_sel[i], s_target.squeeze(0))
            demo_a[i, 0] = a_opt

        return s_sel, demo_a


class CausalMPCPolicyTrainer:
    """Train Policy using Causal MPC demonstrations + WM gradient.

    Two complementary signals:
    1. Behavioral cloning: π(s) should match MPC's optimal action
    2. WM gradient: π(s) should minimize predicted cost in one step

    The MPC signal provides multi-step foresight.
    The WM gradient provides local fine-tuning.
    """

    def __init__(self, wm, policy, hierarchy, s_dataset, s_target,
                 lr=1e-3, horizon=6, n_mpc_samples=500, n_demos=2000,
                 lambda_demo=0.5, device='cpu'):
        self.wm = wm
        self.policy = policy.to(device)
        self.s_target = s_target.to(device)
        self.lambda_demo = lambda_demo
        self.device = device
        self.cost_fn = CausalCostFunction(hierarchy, gamma=2.0)

        wm.eval()
        for p in wm.parameters():
            p.requires_grad = False

        # Generate MPC demonstrations
        print(f"  Generating {n_demos} MPC demos (H={horizon}, N={n_mpc_samples})...")
        teacher = CausalMPCTeacher(wm, hierarchy, horizon=horizon,
                                    n_samples=n_mpc_samples, device=device)
        self.demo_s, self.demo_a = teacher.generate_batch(
            s_dataset, s_target, n_demos=n_demos)
        print(f"  Demos ready. a ∈ [{self.demo_a.min().item():.2f}, "
              f"{self.demo_a.max().item():.2f}]")

        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)

    def train_epoch(self, s_dataset, batch_size=256):
        N = s_dataset.shape[0]
        n_batches = max(1, N // batch_size)
        total_loss = 0.0

        for _ in range(n_batches):
            self.policy.train()
            self.opt.zero_grad()

            # 1) WM gradient path: local single-step cost minimization
            idx_wm = torch.randint(0, N, (batch_size // 2,), device=self.device)
            s_wm = s_dataset[idx_wm]
            a_wm = self.policy(s_wm)
            s_pred = self.wm(torch.cat([s_wm, a_wm], dim=-1))
            wm_loss = self.cost_fn(s_pred,
                                    self.s_target.expand(batch_size // 2, -1)).mean()
            wm_loss = wm_loss + 0.01 * a_wm.pow(2).mean()

            # 2) Behavioral cloning: match MPC demonstrations
            idx_d = torch.randint(0, len(self.demo_s), (batch_size // 2,),
                                  device=self.device)
            a_pol = self.policy(self.demo_s[idx_d])
            demo_loss = (a_pol - self.demo_a[idx_d]).pow(2).mean()

            loss = wm_loss + self.lambda_demo * demo_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
            self.opt.step()
            total_loss += loss.item()

        return {'total': total_loss / n_batches}

    def get_action(self, s):
        self.policy.eval()
        if isinstance(s, np.ndarray):
            s = torch.tensor(s, dtype=torch.float32, device=self.device)
        if s.dim() == 1:
            s = s.unsqueeze(0)
        with torch.no_grad():
            return self.policy(s).squeeze().cpu().item()
