"""KAN Policy Adaptation: gravity switch g=10→18 with continual learning.

Protocol:
  Phase 1: Train KAN policy on g=10 → 10/10 verified
  Phase 2: g=18 — measure performance drop + prediction error rise
  Phase 3: Fine-tune KAN world model on collected g=18 data
  Phase 4: Fine-tune KAN policy using updated WM gradient
  Phase 5: Re-test — measure recovery

Key advantage over old framework:
  - KAN policy itself supports local B-spline updates (no forgetting)
  - World model fine-tuning immediately improves gradient quality for policy
  - Deployment only needs policy forward pass (fast + interpretable)
"""
import torch, torch.nn as nn
import numpy as np, sys, os, time, argparse
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import KAN
from control.kan_policy_net import KANPolicy, KANPolicyTrainer
from experiments.continual_learning import ConfigurablePendulum

PI_2 = np.pi / 2
S_TARGET = torch.tensor([[0.0, 1.0, 0.0]])
MAX_STEPS = 300


def train_kan_policy(wm, policy, s_dataset, epochs=100, lr=1e-3, device='cpu'):
    """Train KAN policy using frozen world model gradient. Returns trainer."""
    trainer = KANPolicyTrainer(wm, policy, S_TARGET.to(device),
                               lr=lr, device=device)
    for ep in range(1, epochs + 1):
        ld = trainer.train_epoch(s_dataset.to(device))
        if ep % 20 == 0:
            print(f"    Epoch {ep:3d}  loss={ld['total']:.4f}")
    return trainer


def evaluate_policy(policy_fn, env, n_episodes=10, label='', collect_data=True):
    """Evaluate policy on environment. Returns (sr, steps, errs, transitions)."""
    successes = 0; all_steps = []; all_errors = []; all_trans = []
    for ep in range(n_episodes):
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        for step in range(MAX_STEPS):
            s_norm = np.array([obs[0], obs[1], obs[2]/8.0], dtype=np.float32)
            res = policy_fn(s_norm)
            a_norm = res[0] if isinstance(res, tuple) else res
            a_raw = float(a_norm) * 2.0
            step_res = env.step([a_raw])
            obs_next = step_res[0]
            term = step_res[2] if len(step_res) > 2 else False

            if collect_data:
                s_true = np.array([obs_next[0], obs_next[1], obs_next[2]/8.0], dtype=np.float32)
                all_trans.append((s_norm, float(a_norm), s_true))

            err = min(abs(np.arctan2(obs_next[1], obs_next[0]) - PI_2),
                     2*np.pi - abs(np.arctan2(obs_next[1], obs_next[0]) - PI_2))
            if err < 0.2:
                successes += 1; all_steps.append(step+1); all_errors.append(err); break
            obs = obs_next
        else:
            all_steps.append(MAX_STEPS)
            err = min(abs(np.arctan2(obs[1], obs[0]) - PI_2),
                     2*np.pi - abs(np.arctan2(obs[1], obs[0]) - PI_2))
            all_errors.append(err)
        print(f"    [{label}] Ep {ep+1:2d}  {'✓' if all_errors[-1]<0.2 else '✗'}  "
              f"steps={all_steps[-1]:3d}  err={all_errors[-1]:.3f}")

    sr = successes / n_episodes
    print(f"    [{label}] {successes}/{n_episodes} ({sr*100:.0f}%)  "
          f"mean_steps={np.mean(all_steps):.0f}  mean_err={np.mean(all_errors):.3f}")
    return sr, all_steps, all_errors, all_trans


def finetune_world_model(wm, transitions, n_epochs=50, lr=1e-3, device='cpu'):
    """Fine-tune KAN world model on new transitions."""
    if len(transitions) < 32:
        return
    xs = []; ys = []
    for s_norm, a_norm, s_true in transitions:
        x = torch.cat([torch.tensor(s_norm, dtype=torch.float32).unsqueeze(0),
                       torch.tensor([[a_norm]], dtype=torch.float32)], dim=-1)
        y = torch.tensor(s_true, dtype=torch.float32).unsqueeze(0)
        xs.append(x); ys.append(y)
    X = torch.cat(xs, dim=0).to(device); Y = torch.cat(ys, dim=0).to(device)
    N = len(X)

    wm.train()
    for p in wm.parameters(): p.requires_grad = True
    opt = torch.optim.Adam(wm.parameters(), lr=lr)

    losses = []
    for epoch in range(n_epochs):
        idx = torch.randint(0, N, (min(32, N),))
        pred = wm(X[idx]); loss = nn.functional.mse_loss(pred, Y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())

    wm.eval()
    for p in wm.parameters(): p.requires_grad = False
    print(f"    WM fine-tuned: loss {losses[0]:.6f} → {losses[-1]:.6f}")
    return losses


@torch.no_grad()
def compute_pred_error(wm, transitions, device='cpu'):
    if len(transitions) == 0: return 0
    errors = []
    for s_norm, a_norm, s_true in transitions:
        x = torch.cat([torch.tensor(s_norm, dtype=torch.float32, device=device).unsqueeze(0),
                       torch.tensor([[a_norm]], dtype=torch.float32, device=device)], dim=-1)
        y = torch.tensor(s_true, dtype=torch.float32, device=device).unsqueeze(0)
        pred = wm(x); errors.append((pred - y).norm().item())
    return np.mean(errors)


def compare_edge_functions(policy_before, policy_after, save_path='kan_policy_adaptation.png'):
    """Compare edge functions before and after adaptation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    edges_before = policy_before.get_edge_functions()
    edges_after = policy_after.get_edge_functions()

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    input_names = ['cosθ', 'sinθ', 'thd/8']

    x_grid = np.linspace(-1, 1, 100)
    from kanrf import bspline_basis

    for in_idx in range(3):
        ax = axes[0, in_idx]
        e = edges_before[0]; grid = e['grid']
        x_t = torch.tensor(x_grid, dtype=torch.float32)
        B = bspline_basis(x_t, torch.tensor(grid), 3).numpy()
        # Average edge function for this input (across all hidden units)
        sw = e['spline_weight']  # (12, 3, n_basis)
        for out_idx in range(min(4, sw.shape[0])):
            c = sw[out_idx, in_idx, :]
            ax.plot(x_grid, B @ c, alpha=0.4, linewidth=1, color='blue')
        ax.set_title(f'Before: {input_names[in_idx]} → hidden')
        ax.set_xlim(-1, 1); ax.axhline(y=0, color='gray', ls=':', alpha=0.5)

        ax2 = axes[1, in_idx]
        e2 = edges_after[0]
        sw2 = e2['spline_weight']
        for out_idx in range(min(4, sw2.shape[0])):
            c = sw2[out_idx, in_idx, :]
            ax2.plot(x_grid, B @ c, alpha=0.4, linewidth=1, color='red')
        ax2.set_title(f'After: {input_names[in_idx]} → hidden')
        ax2.set_xlim(-1, 1); ax2.axhline(y=0, color='gray', ls=':', alpha=0.5)

    fig.suptitle('KAN Policy Edge Functions: Before vs After g=10→18 Adaptation',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"    Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wm', type=str, default='/tmp/kanrf_cl_exp/kan_cws_cl.pt')
    parser.add_argument('--policy', type=str, default='/tmp/kan_policy_trained.pt')
    parser.add_argument('--epochs-wm', type=int, default=50)
    parser.add_argument('--epochs-policy', type=int, default=100)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── Load world model ──
    wm = KAN([4, 12, 3], grid_size=5, spline_order=3)
    wm.load_state_dict(torch.load(args.wm, weights_only=True, map_location=device))
    wm.to(device); wm.eval()
    print(f"World Model: {sum(p.numel() for p in wm.parameters())} params\n")

    # ── Load pre-trained KAN policy ──
    policy = KANPolicy(state_dim=3, action_dim=1, hidden_dim=12, n_layers=2)
    policy.load_state_dict(torch.load(args.policy, weights_only=True, map_location=device))
    policy.to(device); policy.eval()
    print(f"KAN Policy: {sum(p.numel() for p in policy.parameters())} params\n")

    # Save a copy for before/after comparison
    policy_before = KANPolicy(state_dim=3, action_dim=1, hidden_dim=12, n_layers=2)
    policy_before.load_state_dict(policy.state_dict())
    policy_before.eval()

    # Dummy trainer for evaluation (get_action doesn't need dataset)
    trainer = KANPolicyTrainer(wm, policy, S_TARGET.to(device),
                               lr=1e-3, device=device)

    # ═══ Phase 1: g=10 baseline ═══
    print("=" * 60)
    print("Phase 1: g=10 baseline")
    print("=" * 60)
    env1 = gym.make('Pendulum-v1')
    sr1, st1, er1, tr1 = evaluate_policy(
        lambda s: trainer.get_action(s), env1, args.episodes, 'g=10')
    pe1 = compute_pred_error(wm, tr1, device)
    print(f"  Pred error: {pe1:.4f}\n")
    env1.close()

    # ══════════════════════════════════════════════════════════════
    # Phase 2: g=18 (perturbed)
    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("Phase 2: g=18 (perturbed, policy unchanged)")
    print("=" * 60)
    env2 = ConfigurablePendulum(g=18.0, seed=42)
    env2.set_g(18.0)
    sr2, st2, er2, tr2 = evaluate_policy(
        lambda s: trainer.get_action(s), env2, args.episodes, 'g=18-pre')
    pe2 = compute_pred_error(wm, tr2, device)
    print(f"  Pred error: {pe1:.4f} → {pe2:.4f}  ({pe2/pe1:.1f}x)")
    print(f"  Success:   {sr1*100:.0f}% → {sr2*100:.0f}%\n")
    env2.env.close()

    # ══════════════════════════════════════════════════════════════
    # Phase 3: Fine-tune world model
    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print(f"Phase 3: Fine-tune world model ({args.epochs_wm} epochs)")
    print("=" * 60)
    finetune_world_model(wm, tr2, n_epochs=args.epochs_wm, device=device)
    pe3 = compute_pred_error(wm, tr2, device)
    print(f"  Pred error: {pe2:.4f} → {pe3:.4f}  "
          f"({(1-pe3/pe2)*100:.0f}% improvement)\n")

    # ═══ Phase 3.5: Generate successful trajectories via MPC in fine-tuned WM ═══
    print("=" * 60)
    print("Phase 3.5: MPC in fine-tuned WM generates successful trajectories")
    print("=" * 60)

    def mental_mpc(wm_model, s_norm, horizon=3, n_candidates=50):
        s_target = np.array([0.0, 1.0, 0.0])
        s_t = torch.tensor(s_norm, dtype=torch.float32, device=device).unsqueeze(0)
        # Jacobian via finite difference (avoids autograd issues)
        eps = 0.01
        x0 = torch.cat([s_t, torch.zeros(1, 1, device=device)], dim=-1)
        x1 = torch.cat([s_t, torch.ones(1, 1, device=device)*eps], dim=-1)
        with torch.no_grad():
            J = (wm_model(x1) - wm_model(x0)).squeeze().cpu().numpy() / eps
        state_err = s_target - s_norm
        base_a = np.clip(float(np.dot(J, state_err))/(float(np.dot(J, J))+1e-4), -1, 1)

        best_cost = float('inf'); best_seq = np.full(horizon, base_a)
        for _ in range(n_candidates):
            seq = best_seq + np.random.randn(horizon) * 0.3
            seq = np.clip(seq, -1, 1)
            s = s_norm.copy(); cost = 0
            for h in range(horizon):
                x = torch.cat([torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0),
                              torch.tensor([[seq[h]]], dtype=torch.float32, device=device)], dim=-1)
                with torch.no_grad():
                    s = wm_model(x).cpu().squeeze().numpy()
                cost += np.sum((s[:2]-s_target[:2])**2) + 0.01*seq[h]**2
            if cost < best_cost:
                best_cost = cost; best_seq = seq.copy()
        return best_seq[0]

    np.random.seed(42)
    n_success = 0; mental_transitions = []
    n_total = 200; report_every = 50
    for i in range(n_total):
        angle = np.random.uniform(-np.pi, np.pi)
        s_raw = np.array([np.cos(angle), np.sin(angle), np.random.uniform(-8, 8)])
        s_norm = s_raw.copy(); s_norm[2] /= 8.0
        for _ in range(150):
            a = mental_mpc(wm, s_norm)
            x = torch.cat([torch.tensor(s_norm, dtype=torch.float32, device=device).unsqueeze(0),
                          torch.tensor([[a]], dtype=torch.float32, device=device)], dim=-1)
            with torch.no_grad():
                s_next = wm(x).cpu().squeeze().numpy()
            mental_transitions.append((s_norm.copy(), a, s_next.copy()))
            s_next_raw = s_next.copy(); s_next_raw[2] *= 8.0
            err = min(abs(np.arctan2(s_next_raw[1], s_next_raw[0])-PI_2),
                     2*np.pi-abs(np.arctan2(s_next_raw[1], s_next_raw[0])-PI_2))
            if err < 0.3: n_success += 1; break
            s_norm = s_next
        if i % report_every == 0:
            print(f"    {i}/{n_total} trajectories, {n_success} reached upright")
    print(f"    {n_success}/{n_total} ({n_success/n_total*100:.0f}%) reached upright, "
          f"{len(mental_transitions)} transitions\n")

    # ═══ Phase 4: Fine-tune KAN policy using mental trajectories ═══
    print("=" * 60)
    print(f"Phase 4: Fine-tune KAN policy on mental trajectories")
    print("=" * 60)
    # Use mental trajectory data — these are "successful" trajectories from MPC
    s_list = []; a_list = []
    for s_norm, a, s_next in mental_transitions:
        s_list.append(s_norm); a_list.append([a])
    s_dataset = torch.tensor(np.array(s_list), dtype=torch.float32)
    a_targets = torch.tensor(np.array(a_list), dtype=torch.float32)
    print(f"  Data: {s_dataset.shape[0]} (s,a) pairs from mental MPC")
    trainer = KANPolicyTrainer(wm, policy, S_TARGET.to(device),
                               lr=5e-4, device=device)
    for ep in range(1, args.epochs_policy + 1):
        ld = trainer.train_epoch(s_dataset.to(device))
        if ep % 20 == 0:
            print(f"  Epoch {ep:3d}  loss={ld['total']:.4f}  "
                  f"energy={ld['energy']:.4f}  dist={ld['dist']:.4f}")
    print()

    # ══════════════════════════════════════════════════════════════
    # Phase 5: g=18 after adaptation
    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("Phase 5: g=18 (after world model + policy fine-tuning)")
    print("=" * 60)
    env3 = ConfigurablePendulum(g=18.0, seed=42)
    env3.set_g(18.0)
    sr5, st5, er5, _ = evaluate_policy(
        lambda s: trainer.get_action(s), env3, args.episodes, 'g=18-post')
    env3.env.close()

    # ══════════════════════════════════════════════════════════════
    # Compare edge functions before vs after
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Edge Function Comparison")
    print("=" * 60)
    compare_edge_functions(policy_before, policy)

    # ══════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Phase 1 (g=10, baseline):    {sr1*100:.0f}%  pred_err={pe1:.4f}")
    print(f"  Phase 2 (g=18, degraded):    {sr2*100:.0f}%  pred_err={pe2:.4f}")
    print(f"  Phase 3 (WM fine-tuned):     pred_err={pe3:.4f}")
    print(f"  Phase 5 (g=18, recovered):   {sr5*100:.0f}%")
    if sr5 > sr2:
        print(f"\n  ✓ Policy adaptation successful: {sr2*100:.0f}% → {sr5*100:.0f}%")


if __name__ == '__main__':
    main()
