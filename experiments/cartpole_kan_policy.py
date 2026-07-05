"""CartPole: KAN Policy trained via frozen CWS-KAN world model gradient.

Tests cross-task generalizability of the KAN Policy framework.
"""
import torch, torch.nn as nn
import numpy as np, time, sys, os
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import KAN
from control.kan_policy_net import KANPolicy, KANPolicyTrainer
from control.kan_interpretability import KANInterpreter

S_TARGET = torch.tensor([[0.0, 0.0, 0.0, 0.0]])  # [x=0, xd=0, theta=0, thd=0]


def evaluate_cartpole(policy_fn, n_trials=10, max_steps=500, continuous=True):
    """Evaluate on CartPole. If continuous=True, use analytical dynamics.

    CartPole returns (x, xd, theta, thd) normalized.
    Success: survive all steps without pole falling (|theta| < 0.21 rad, |x| < 2.4).
    """
    from experiments.cartpole_decision_v3 import (step_cartpole_cont, X_SCALE,
                                                    XD_SCALE, TH_SCALE, THD_SCALE, FORCE_MAX)
    successes = 0; all_steps = []; all_errors = []
    for trial in range(n_trials):
        seed = 42 + trial * 100
        # Random start near upright
        np.random.seed(seed)
        th = np.random.uniform(-0.05, 0.05)
        s_raw = torch.tensor([[0.0, 0.0, th, 0.0]], dtype=torch.float32)
        for step in range(max_steps):
            s_norm = s_raw.clone()
            s_norm[:, 0] /= X_SCALE; s_norm[:, 1] /= XD_SCALE
            s_norm[:, 2] /= TH_SCALE; s_norm[:, 3] /= THD_SCALE
            a_norm = float(policy_fn(s_norm[0].numpy()))
            s_raw = step_cartpole_cont(s_raw, torch.tensor([a_norm]))
            theta = s_raw[0, 2].item(); x = s_raw[0, 0].item()
            if abs(theta) > 0.21 or abs(x) > 2.4:
                break
        all_steps.append(step + 1)
        ok = step + 1 >= max_steps
        if ok: successes += 1
        if n_trials <= 10:
            print(f"  Trial {trial+1:2d}  {'✓' if ok else '✗'}  steps={step+1}")
    sr = successes / n_trials
    print(f"  Result: {successes}/{n_trials} ({sr*100:.0f}%)  mean_steps={np.mean(all_steps):.0f}")
    return sr


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wm', type=str, default='/tmp/kanrf_cl_cp/cartpole_kan_cws.pt')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--hidden', type=int, default=12)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output', type=str, default='/tmp/cartpole_kan_policy.pt')
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── 1. Load frozen CartPole CWS-KAN world model ──
    print("Loading CartPole CWS-KAN World Model...")
    wm = KAN([5, 12, 4], grid_size=5, spline_order=3)
    wm.load_state_dict(torch.load(args.wm, weights_only=True, map_location=device))
    wm.to(device); wm.eval()
    print(f"  WM: {sum(p.numel() for p in wm.parameters())} params")

    # ── 2. Create KAN Policy ──
    policy = KANPolicy(state_dim=4, action_dim=1, hidden_dim=args.hidden, n_layers=2)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  Policy: [4,{args.hidden},{args.hidden},1] = {n_params} params")

    # ── 3. Training states: CartPole distribution (near-upright + random) ──
    n_samples = 20000
    # Mix of near-upright and wider distributions
    n_near = n_samples // 2; n_wide = n_samples - n_near
    # Near-upright
    s_near = np.stack([
        np.random.uniform(-1.0, 1.0, n_near) / 2.5,       # x
        np.random.uniform(-1.5, 1.5, n_near) / 3.0,       # xd
        np.random.uniform(-0.15, 0.15, n_near) / 0.3,     # theta (small)
        np.random.uniform(-1.5, 1.5, n_near) / 3.0,       # thd
    ], axis=1)
    # Wider (includes failed states)
    s_wide = np.stack([
        np.random.uniform(-2.4, 2.4, n_wide) / 2.5,
        np.random.uniform(-3.0, 3.0, n_wide) / 3.0,
        np.random.uniform(-0.3, 0.3, n_wide) / 0.3,       # theta (wider but still near)
        np.random.uniform(-3.0, 3.0, n_wide) / 3.0,
    ], axis=1)
    s_all = np.vstack([s_near, s_wide]); np.random.shuffle(s_all)
    s_dataset = torch.tensor(s_all[:n_samples], dtype=torch.float32).to(device)
    print(f"  Training states: {s_dataset.shape}")

    # ── 4. Custom trainer for CartPole (no energy loss) ──
    class CartPoleTrainer:
        def __init__(self, wm_model, pol, lr=1e-3, clip=10.0):
            self.wm = wm_model; self.pol = pol.to(device); self.clip = clip
            self.opt = torch.optim.Adam(pol.parameters(), lr=lr)
            self.loss_history = []
            self.wm.eval()
            for p in self.wm.parameters(): p.requires_grad = False

        def train_epoch(self, s_dataset, batch_size=256, n_batches=40):
            N = s_dataset.shape[0]; losses = []
            for _ in range(n_batches):
                idx = torch.randint(0, N, (batch_size,), device=device)
                s_b = s_dataset[idx]; B = s_b.shape[0]
                self.pol.train(); self.opt.zero_grad()
                a = self.pol(s_b)
                s_pred = self.wm(torch.cat([s_b, a], dim=-1))
                # Pole angle + cart position loss
                th_err = s_pred[:, 2].pow(2)       # minimize |theta|
                x_err = 0.1 * s_pred[:, 0].pow(2)  # minimize |x| (secondary)
                ctrl = 0.01 * a.pow(2).mean()
                loss = th_err.mean() + x_err.mean() + ctrl
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.pol.parameters(), self.clip)
                self.opt.step()
                losses.append(loss.item())
            self.pol.eval()
            self.loss_history.append(np.mean(losses))
            return {'total': np.mean(losses)}

        @torch.no_grad()
        def get_action(self, s):
            self.pol.eval()
            if isinstance(s, np.ndarray):
                s = torch.tensor(s, dtype=torch.float32, device=device)
            if s.dim() == 1: s = s.unsqueeze(0)
            return self.pol(s).squeeze().cpu().item()

    trainer = CartPoleTrainer(wm, policy, lr=1e-3)

    # ── 5. Train ──
    print(f"\nTraining ({args.epochs} epochs)...")
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        ld = trainer.train_epoch(s_dataset)
        if ep % 60 == 0:
            print(f"  Epoch {ep:3d}  loss={ld['total']:.4f}")
    print(f"  Done in {time.time()-t0:.0f}s")

    # ── 6. Save ──
    torch.save(policy.state_dict(), args.output)
    print(f"  Saved: {args.output}")

    # ── 7. Evaluate ──
    print(f"\n{'='*60}")
    print("CartPole Evaluation (continuous force, 20 trials)")
    print("=" * 60)
    evaluate_cartpole(trainer.get_action, n_trials=20)

    # ── 8. Interpretability ──
    print(f"\n{'='*60}")
    print("Interpretability Analysis")
    print("=" * 60)
    interp = KANInterpreter(policy, input_names=['x', 'x_dot', 'theta', 'thd'])
    interp.prune_and_analyze(threshold=0.1)
    g = interp.causal_graph
    for layer in g['layers']:
        print(f"  Layer {layer['layer_idx']}: sparsity={layer['sparsity']*100:.0f}%")
    interp.attribute_batch([
        np.array([0.0, 0.0, 0.3/0.3, 0.0]),    # leaning right
        np.array([0.0, 0.0, -0.3/0.3, 0.0]),   # leaning left
        np.array([1.0, 0.0, 0.0, 0.0]),         # cart off-center
        np.array([0.0, 0.0, 0.0, 0.0]),          # perfect
    ], ['lean-right', 'lean-left', 'cart-off', 'perfect'])
    for label, data in interp.attributions.items():
        c = data['contributions']
        print(f"  {label:12s}: a={data['action']:+.3f}  "
              f"x={c[0]*100:.0f}% xd={c[1]*100:.0f}% th={c[2]*100:.0f}% thd={c[3]*100:.0f}%")


if __name__ == '__main__':
    main()
