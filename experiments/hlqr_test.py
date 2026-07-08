"""Test Hierarchical LQR on Pendulum and CartPole."""
import torch, numpy as np, time, sys, os, gymnasium as gym
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import ProtoKAN
from control.kan_policy_net import KANPolicy
from control.thtp import TemporalHierarchy
from control.hierarchical_lqr import HierarchicalLQR

PI_2 = np.pi / 2
S_TARGET = torch.tensor([[0.0, 1.0, 0.0]])
S_TARGET_CP = torch.zeros(1, 4)


class HLQRPolicyTrainer:
    """Train Policy using Hierarchical LQR as teacher."""

    def __init__(self, wm, policy, hlqr, s_dataset, s_target,
                 lr=1e-3, n_distill=500, device='cpu'):
        self.wm = wm; self.policy = policy.to(device)
        self.hlqr = hlqr; self.s_target = s_target.to(device)
        self.device = device
        wm.eval()
        for p in wm.parameters(): p.requires_grad = False

        print(f"  Pre-computing H-LQR targets ({n_distill} states, H={hlqr.horizon})...")
        self.distill_s, self.distill_a = hlqr.compute_batch(
            s_dataset, s_target, n_samples=n_distill)
        print(f"  Done. Targets: a ∈ [{self.distill_a.min().item():.3f}, "
              f"{self.distill_a.max().item():.3f}]")

        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)

    def train_epoch(self, s_dataset, batch_size=256):
        N = s_dataset.shape[0]; n_batches = max(1, N // batch_size)
        total_loss = 0.0
        for _ in range(n_batches):
            self.policy.train(); self.opt.zero_grad()

            # WM gradient loss (half batch)
            idx_wm = torch.randint(0, N, (batch_size // 2,), device=self.device)
            s_wm = s_dataset[idx_wm]; a_wm = self.policy(s_wm)
            s_pred = self.wm(torch.cat([s_wm, a_wm], dim=-1))

            if s_wm.shape[1] == 3:  # Pendulum
                thd = s_wm[:, 2] * 8.0; Ec = 0.5*thd.pow(2) + 10.0*s_wm[:, 1]
                thdp = s_pred[:, 2]*8.0; Ep = 0.5*thdp.pow(2) + 10.0*s_pred[:, 1]
                deficit = (10.0 - Ec).detach()
                egain = (Ep - Ec) * torch.sign(deficit)
                sin = s_wm[:, 1]; ws = ((1.0+sin)/2.0).clamp(0,1)
                wm_loss = (-egain.mean() +
                           (ws*(s_pred-self.s_target.expand(batch_size//2,-1))
                            .pow(2).sum(-1)).mean() + 0.01*a_wm.pow(2).mean())
            else:
                s_t = torch.zeros(1, s_wm.shape[1], device=self.device)
                wm_loss = (s_pred[:,2].pow(2).mean() + 0.1*s_pred[:,0].pow(2).mean() +
                           0.5*s_pred[:,3].pow(2).mean() + 0.01*a_wm.pow(2).mean())

            # Distillation loss (half batch)
            idx_d = torch.randint(0, len(self.distill_s), (batch_size//2,),
                                  device=self.device)
            a_pol = self.policy(self.distill_s[idx_d])
            distill_loss = (a_pol - self.distill_a[idx_d]).pow(2).mean()

            loss = wm_loss + 0.3 * distill_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
            self.opt.step()
            total_loss += loss.item()
        return {'total': total_loss/n_batches}

    def get_action(self, s):
        self.policy.eval()
        if isinstance(s, np.ndarray):
            s = torch.tensor(s, dtype=torch.float32, device=self.device)
        if s.dim() == 1: s = s.unsqueeze(0)
        with torch.no_grad():
            return self.policy(s).squeeze().cpu().item()


def test_pendulum():
    from experiments.baseline_sweep import (
        generate_pendulum_data, train_wm, generate_policy_states, evaluate_policy
    )
    device='cpu'; torch.manual_seed(42); np.random.seed(42)
    print("=" * 70)
    print("H-LQR TEST: Pendulum")
    print("=" * 70)
    X, Y = generate_pendulum_data(5000, seed=42)
    wm, wm_val = train_wm(X.to(device), Y.to(device))
    print(f"WM val_mse={wm_val:.6f}")

    names = ['cosθ', 'sinθ', 'θ̇']
    hier = TemporalHierarchy(wm, 3, n_samples=200, device=device)
    print(hier.summary(names))

    # Test H-LQR on a few states
    hlqr = HierarchicalLQR(wm, hier, horizon=4, q_base=10.0, device=device)
    print(f"\nH-LQR Q matrix: {hlqr.Q.diag().tolist()}")
    print("H-LQR examples:")
    for label, s_vec in [('bottom', [0.0, -1.0, 0.0]),
                           ('mid-right', [0.0, 1.0, 0.0]),
                           ('swing', [1.0, 0.0, 0.5])]:
        s_t = torch.tensor(s_vec, dtype=torch.float32, device=device)
        u_0, K_list, k_list, cost = hlqr.solve(s_t, S_TARGET.squeeze(0))
        print(f"  {label:10s}: u*={u_0.item():+.3f}  cost-to-go={cost:.4f}")

    # Train Policy
    s_pol = generate_policy_states(10000, seed=42).to(device)
    policy = KANPolicy(state_dim=3, action_dim=1, hidden_dim=12, n_layers=2).to(device)
    trainer = HLQRPolicyTrainer(wm, policy, hlqr, s_pol, S_TARGET,
                                n_distill=500, device=device)
    for ep in range(1, 201):
        ld = trainer.train_epoch(s_pol)
        if ep % 40 == 0: print(f"  Epoch {ep:3d}  total={ld['total']:.4f}")
    s, st, er = evaluate_policy(trainer)
    print(f"  Pendulum H-LQR: {s}/10  steps={np.mean(st):.0f}  err={np.mean(er):.3f}")


def test_cartpole():
    from experiments.cartpole_continual import (
        generate_wm_data, train_wm, generate_policy_states,
        X_S, XD_S, TH_S, THD_S, step_cartpole
    )
    device='cpu'; torch.manual_seed(42); np.random.seed(42)
    print("\n" + "=" * 70)
    print("H-LQR TEST: CartPole")
    print("=" * 70)
    X, Y = generate_wm_data(g=9.8, n=5000, device=device)
    wm, wm_val = train_wm(X, Y, 'protokan', 80, device)
    print(f"WM val_mse={wm_val:.6f}")

    names = ['x', 'ẋ', 'θ', 'θ̇']
    hier = TemporalHierarchy(wm, 4, n_samples=200, device=device)
    print(hier.summary(names))

    # Larger Q for theta dimension (Tier 1 in CartPole)
    hlqr = HierarchicalLQR(wm, hier, horizon=4, q_base=10.0, device=device)
    print(f"\nH-LQR Q matrix: {hlqr.Q.diag().tolist()}")

    s_pol = generate_policy_states(15000, device)
    policy = KANPolicy(state_dim=4, action_dim=1, hidden_dim=12, n_layers=2).to(device)
    trainer = HLQRPolicyTrainer(wm, policy, hlqr, s_pol, S_TARGET_CP,
                                n_distill=500, device=device)
    for ep in range(1, 201):
        ld = trainer.train_epoch(s_pol)
        if ep % 40 == 0: print(f"  Epoch {ep:3d}  total={ld['total']:.4f}")

    succ=0; steps=[]
    for trial in range(20):
        seed=42+trial*100; np.random.seed(seed)
        th=np.random.uniform(-0.05,0.05)
        s_raw=torch.tensor([[0.,0.,th,0.]], dtype=torch.float32)
        for step in range(500):
            sn=s_raw.clone(); sn[:,0]/=X_S; sn[:,1]/=XD_S; sn[:,2]/=TH_S; sn[:,3]/=THD_S
            a=trainer.get_action(sn[0].numpy())
            s_raw=step_cartpole(s_raw,torch.tensor([a]))
            if abs(s_raw[0,2].item())>0.21 or abs(s_raw[0,0].item())>2.4: break
        steps.append(step+1)
        if step+1>=500: succ+=1
    print(f"  CartPole H-LQR: {succ}/20 ({succ*5}%)  mean_steps={np.mean(steps):.0f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='both')
    args = parser.parse_args()
    if args.env in ['pendulum', 'both']: test_pendulum()
    if args.env in ['cartpole', 'both']: test_cartpole()
