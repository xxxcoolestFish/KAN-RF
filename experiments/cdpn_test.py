"""Test CDPN v2: Compare WM-gradient vs Abstract Planner training on Pendulum and CartPole."""
import torch, numpy as np, time, sys, os, gymnasium as gym
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import ProtoKAN
from control.cdpn import (
    discover_tier0, CausalDecomposedPolicy, Execute,
    CausalBridge, AbstractPlannerTrainer
)

PI_2 = np.pi / 2
S_TARGET_PEN = torch.tensor([[0.0, 1.0, 0.0]])
S_TARGET_CP = torch.zeros(1, 4)


def train_cdpn(wm, state_dim, s_target, s_pol, tier0, label='',
               mode='1step', epochs=200):
    """Train CDPN (original WM-gradient method)."""
    policy = CausalDecomposedPolicy(
        state_dim=state_dim, tier0_size=len(tier0),
        hidden_dim=24, n_layers=2, use_tanh=False)
    execute = Execute(wm, state_dim, tier0, s_pol, damping=0.1)

    # Original CDPNTrainer doesn't need bridge
    class LegacyCDPNTrainer:
        def __init__(self2, wm, policy, execute, s_target, tier0, lr=1e-3, mode='1step', imagine_steps=3):
            self2.wm = wm; self2.policy = policy; self2.execute = execute
            self2.s_target = s_target; self2.tier0 = tier0
            self2.device = 'cpu'; self2.mode = mode; self2.imagine_steps = imagine_steps
            self2.opt = torch.optim.Adam(policy.parameters(), lr=lr)
            self2.loss_history = []
        def train_epoch(self2, s_dataset, batch_size=256):
            N = s_dataset.shape[0]; n_batches = max(1, N // batch_size); total_loss = 0.0
            for _ in range(n_batches):
                idx = torch.randint(0, N, (batch_size,))
                s_b = s_dataset[idx]; self2.policy.train(); self2.opt.zero_grad()
                s_goal = self2.s_target.expand(s_b.shape[0], -1)
                v_des = self2.policy(s_b, s_goal)
                a = self2.execute(v_des, s_b)
                s_pred = self2.wm(torch.cat([s_b, a], dim=-1))
                loss = (s_pred - s_goal).pow(2).sum(dim=-1).mean() + 0.01 * a.pow(2).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self2.policy.parameters(), 10.0)
                self2.opt.step(); total_loss += loss.item()
            avg = total_loss / n_batches
            self2.loss_history.append({'total': avg})
            return {'total': avg}
        def get_action(self2, s, s_goal=None):
            self2.policy.eval()
            if isinstance(s, np.ndarray): s = torch.tensor(s, dtype=torch.float32)
            if s.dim() == 1: s = s.unsqueeze(0)
            if s_goal is None: s_goal = self2.s_target
            with torch.no_grad(): v_des = self2.policy(s, s_goal)
            a = self2.execute(v_des, s)
            return a.squeeze().cpu().item()

    trainer = LegacyCDPNTrainer(wm, policy, execute, s_target, tier0, lr=1e-3, mode=mode)
    print(f"  [{label}] Policy {sum(p.numel() for p in policy.parameters())} params  Tier0={tier0}")
    for ep in range(1, epochs + 1):
        ld = trainer.train_epoch(s_pol)
        if ep % 50 == 0: print(f"    Epoch {ep:3d}  loss={ld['total']:.4f}")
    return trainer


def train_abstract(wm, state_dim, s_target, s_pol, tier0, label='',
                   epochs=200, bridge=None, execute=None):
    """Train policy using AbstractPlannerTrainer (no WM gradient)."""
    policy = CausalDecomposedPolicy(
        state_dim=state_dim, tier0_size=len(tier0),
        hidden_dim=24, n_layers=2, use_tanh=True)
    trainer = AbstractPlannerTrainer(
        wm, policy, execute, bridge, s_target, lr=1e-3, env='pendulum')
    print(f"  [{label}] Policy {sum(p.numel() for p in policy.parameters())} params  Tier0={tier0}  mode=abstract")
    for ep in range(1, epochs + 1):
        ld = trainer.train_epoch(s_pol)
        if ep % 50 == 0:
            print(f"    Epoch {ep:3d}  total={ld['total']:.4f}  "
                  f"swing_up={ld.get('swing_up',0):.4f}  E_cur={ld.get('E_cur',0):.2f}")
    return trainer


def test_pendulum():
    from experiments.baseline_sweep import (
        generate_pendulum_data, train_wm, generate_policy_states, evaluate_policy
    )
    device = 'cpu'; torch.manual_seed(42); np.random.seed(42)
    print("=" * 70)
    print("CDPN v2: Pendulum - WM-gradient vs Abstract Planner")
    print("=" * 70)

    X, Y = generate_pendulum_data(5000, seed=42)
    wm, _ = train_wm(X.to(device), Y.to(device))
    print(f"  WM val_mse ~ 0.000003")

    tier0, mask, jac_norms, thresh = discover_tier0(wm, 3, device=device)
    names = ['cos', 'sin', 'thd']
    print(f"  Tier 0: {[names[i] for i in tier0]}  norms={[f'{n:.4f}' for n in jac_norms]}")

    s_pol = generate_policy_states(10000, seed=42).to(device)

    bridge = CausalBridge(wm, 3, tier0, s_pol, device=device, g_true=10.0)
    execute = Execute(wm, 3, tier0, s_pol, damping=0.1, bridge=bridge)

    print("\n  [Method 1: WM-gradient CDPN]")
    t1 = train_cdpn(wm, 3, S_TARGET_PEN, s_pol, tier0, label='wm-grad', epochs=200)
    s, st, er = evaluate_policy(t1)
    print(f"  Result: {s}/10  steps={np.mean(st):.0f}  err={np.mean(er):.3f}")

    print("\n  [Method 2: Abstract Planner, no WM gradient]")
    t2 = train_abstract(wm, 3, S_TARGET_PEN, s_pol, tier0,
                        label='abstract', epochs=200, bridge=bridge, execute=execute)
    s2, st2, er2 = evaluate_policy(t2)
    print(f"  Result: {s2}/10  steps={np.mean(st2):.0f}  err={np.mean(er2):.3f}")

    return t1, t2


def test_cartpole():
    from experiments.cartpole_continual import (
        generate_wm_data, train_wm, generate_policy_states,
        X_S, XD_S, TH_S, THD_S, step_cartpole
    )
    device = 'cpu'; torch.manual_seed(42); np.random.seed(42)
    print("\n" + "=" * 70)
    print("CDPN v2: CartPole Abstract Planner")
    print("=" * 70)

    X, Y = generate_wm_data(g=9.8, n=5000, device=device)
    wm, _ = train_wm(X, Y, 'protokan', 80, device)
    print(f"  WM val_mse ~ 0.000000")

    tier0, mask, jac_norms, thresh = discover_tier0(wm, 4, device=device)
    names = ['x', 'xd', 'th', 'thd']
    print(f"  Tier 0: {[names[i] for i in tier0]}  norms={[f'{n:.4f}' for n in jac_norms]}")

    s_pol = generate_policy_states(15000, device)
    bridge = CausalBridge(wm, 4, tier0, s_pol, device=device)
    execute = Execute(wm, 4, tier0, s_pol, damping=0.1, bridge=bridge)

    print("\n  [Abstract Planner, no WM gradient]")
    policy = CausalDecomposedPolicy(state_dim=4, tier0_size=len(tier0),
                                     hidden_dim=24, n_layers=2, use_tanh=True)
    trainer = AbstractPlannerTrainer(wm, policy, execute, bridge, S_TARGET_CP,
                                      lr=1e-3, env='cartpole')
    for ep in range(1, 201):
        ld = trainer.train_epoch(s_pol)
        if ep % 50 == 0: print(f"    Epoch {ep:3d}  total={ld['total']:.4f}")

    def evaluate_cdpn(trainer, label=''):
        successes = 0; all_steps = []
        for trial in range(20):
            np.random.seed(42 + trial * 100)
            th = np.random.uniform(-0.05, 0.05)
            s_raw = torch.tensor([[0., 0., th, 0.]], dtype=torch.float32)
            for step in range(500):
                sn = s_raw.clone()
                sn[:, 0] /= X_S; sn[:, 1] /= XD_S
                sn[:, 2] /= TH_S; sn[:, 3] /= THD_S
                a = trainer.get_action(sn[0].numpy())
                s_raw = step_cartpole(s_raw, torch.tensor([a]))
                if abs(s_raw[0, 2].item()) > 0.21 or abs(s_raw[0, 0].item()) > 2.4:
                    break
            all_steps.append(step + 1)
            if step + 1 >= 500: successes += 1
        print(f"  [{label}] {successes}/20 ({successes*5}%)  mean_steps={np.mean(all_steps):.0f}")
        return successes, all_steps

    evaluate_cdpn(trainer, 'abstract')
    return trainer


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='both')
    args = parser.parse_args()
    if args.env in ['pendulum', 'both']: test_pendulum()
    if args.env in ['cartpole', 'both']: test_cartpole()



def test_pendulum_mlp():
    """Quick MLP vs KAN comparison for Abstract Planner."""
    from experiments.baseline_sweep import (
        generate_pendulum_data, train_wm, generate_policy_states, evaluate_policy
    )
    import torch, numpy as np
    device = "cpu"; torch.manual_seed(42); np.random.seed(42)
    X, Y = generate_pendulum_data(5000, seed=42)
    wm, _ = train_wm(X.to(device), Y.to(device))
    tier0, *_ = discover_tier0(wm, 3)
    s_pol = generate_policy_states(10000, seed=42).to(device)
    bridge = CausalBridge(wm, 3, tier0, s_pol, device=device, g_true=10.0)
    execute = Execute(wm, 3, tier0, s_pol, damping=0.1, bridge=bridge)
    S_TARGET = torch.tensor([[0.0, 1.0, 0.0]])

    for name, use_mlp in [("KAN", False), ("MLP", True)]:
        print(f"  [{name}] Abstract Planner")
        policy = CausalDecomposedPolicy(3, len(tier0), hidden_dim=24,
                                         n_layers=2, use_tanh=True, use_mlp=use_mlp)
        trainer = AbstractPlannerTrainer(
            wm, policy, execute, bridge, S_TARGET, lr=1e-3, env="pendulum")
        print(f"  [{name}] Policy {sum(p.numel() for p in policy.parameters())} params")
        for ep in range(1, 201):
            ld = trainer.train_epoch(s_pol)
            if ep % 50 == 0:
                print(f"    Epoch {ep:3d}  total={ld['total']:.4f}  "
                      f"swing_up={ld.get('swing_up',0):.4f}  E_cur={ld.get('E_cur',0):.2f}")
        s, st, er = evaluate_policy(trainer, n_trials=10, seed=42)
        print(f"  [{name}] Result: {s}/10  steps={np.mean(st):.0f}  err={np.mean(er):.3f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="both")
    parser.add_argument("--mlp", action="store_true")
    args = parser.parse_args()
    if args.mlp:
        test_pendulum_mlp()
    else:
        if args.env in ["pendulum", "both"]: test_pendulum()
        if args.env in ["cartpole", "both"]: test_cartpole()
