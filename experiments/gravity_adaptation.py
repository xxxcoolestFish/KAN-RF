"""Gravity Adaptation: Enhanced MPC swing-up under gravity change g=10→3.

Tests whether KAN + online learning can adapt its world model and restore
swing-up performance after a physics parameter change.

Protocol:
  Phase 1 (g=10): Run N episodes, measure success rate & prediction error
  Phase 2 (g=3):  Switch gravity, run N episodes WITHOUT fine-tuning
  Phase 3:        Fine-tune KAN on collected transitions from Phase 2
  Phase 4 (g=3):  Run N episodes with fine-tuned KAN, measure recovery

Key distinction from earlier failed experiments:
  - Uses REAL gymnasium Pendulum dynamics (no analytical step errors)
  - Swing-up task (not impossible stabilization)
  - Enhanced MPC controller (verified 10/10)
"""
import torch, torch.nn as nn
import numpy as np, time, sys, os
import gymnasium as gym
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import KAN
from control.kan_enhanced_mpc import KANEnhancedMPC

PI_2 = np.pi / 2
MAX_STEPS = 300


def run_episode(env, mpc, max_steps=MAX_STEPS, collect_data=True):
    """Run one episode. Returns (success, n_steps, final_err, transitions)."""
    result = env.reset()
    obs = result[0] if isinstance(result, tuple) else result
    transitions = []
    for step in range(max_steps):
        s_norm = np.array([obs[0], obs[1], obs[2]/8.0], dtype=np.float32)
        a_norm, info = mpc.get_action(s_norm)
        a_raw = a_norm * 2.0
        result = env.step([a_raw])
        obs_next = result[0]
        term = result[2] if len(result) > 2 else False
        trunc = result[3] if len(result) > 3 else False

        if collect_data:
            s_true = np.array([obs_next[0], obs_next[1], obs_next[2]/8.0], dtype=np.float32)
            transitions.append((s_norm, a_norm, s_true))

        err = min(abs(np.arctan2(obs_next[1], obs_next[0]) - PI_2),
                  2*np.pi - abs(np.arctan2(obs_next[1], obs_next[0]) - PI_2))
        if err < 0.2:
            return True, step + 1, err, transitions
        obs = obs_next

    err = min(abs(np.arctan2(obs[1], obs[0]) - PI_2),
              2*np.pi - abs(np.arctan2(obs[1], obs[0]) - PI_2))
    return False, max_steps, err, transitions


def evaluate(env, mpc, n_episodes=10, label='', collect_data=True):
    """Run N episodes, return stats + all transitions."""
    successes = 0; all_steps = []; all_errors = []; all_transitions = []
    for ep in range(n_episodes):
        ok, steps, err, trans = run_episode(env, mpc, collect_data=collect_data)
        if ok: successes += 1
        all_steps.append(steps); all_errors.append(err)
        all_transitions.extend(trans)
        print(f"  [{label}] Ep {ep+1:2d}  {'✓' if ok else '✗'}  "
              f"steps={steps:3d}  err={err:.3f}rad")
    sr = successes / n_episodes
    print(f"  [{label}] Success: {successes}/{n_episodes} ({sr*100:.0f}%)  "
          f"mean_steps={np.mean(all_steps):.0f}  mean_err={np.mean(all_errors):.3f}")
    return sr, all_steps, all_errors, all_transitions


def finetune_kan(kan, transitions, n_epochs=50, batch_size=32, lr=1e-3, device='cpu'):
    """Fine-tune KAN on collected transitions."""
    if len(transitions) < batch_size:
        return

    xs = []; ys = []
    for s_norm, a_norm, s_true in transitions:
        x = torch.cat([
            torch.tensor(s_norm, dtype=torch.float32).unsqueeze(0),
            torch.tensor([[a_norm]], dtype=torch.float32)
        ], dim=-1)
        y = torch.tensor(s_true, dtype=torch.float32).unsqueeze(0)
        xs.append(x); ys.append(y)

    X = torch.cat(xs, dim=0).to(device)
    Y = torch.cat(ys, dim=0).to(device)
    N = len(X)

    kan.train()
    for p in kan.parameters(): p.requires_grad = True
    opt = torch.optim.Adam(kan.parameters(), lr=lr)

    losses = []
    for epoch in range(n_epochs):
        idx = torch.randint(0, N, (min(batch_size, N),))
        pred = kan(X[idx])
        loss = nn.functional.mse_loss(pred, Y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())

    kan.eval()
    for p in kan.parameters(): p.requires_grad = False

    print(f"  Fine-tuned KAN: {n_epochs} epochs, loss {losses[0]:.6f} → {losses[-1]:.6f}")
    return losses


@torch.no_grad()
def compute_pred_error(kan, transitions, device='cpu'):
    """Compute mean prediction error on transitions."""
    if len(transitions) == 0: return 0
    errors = []
    for s_norm, a_norm, s_true in transitions:
        x = torch.cat([
            torch.tensor(s_norm, dtype=torch.float32, device=device).unsqueeze(0),
            torch.tensor([[a_norm]], dtype=torch.float32, device=device)
        ], dim=-1)
        y = torch.tensor(s_true, dtype=torch.float32, device=device).unsqueeze(0)
        pred = kan(x)
        errors.append((pred - y).norm().item())
    return np.mean(errors)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--kan', type=str, default='/tmp/kanrf_cl_exp/kan_cws_cl.pt')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--finetune-epochs', type=int, default=50)
    parser.add_argument('--finetune-lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load KAN
    kan = KAN([4, 12, 3], grid_size=5, spline_order=3)
    kan.load_state_dict(torch.load(args.kan, weights_only=True, map_location=device))
    kan.to(device); kan.eval()
    print(f"KAN: {sum(p.numel() for p in kan.parameters())} params\n")

    # Create MPC
    mpc = KANEnhancedMPC(kan, state_dim=3, action_dim=1,
                         horizon=5, n_refine_steps=20,
                         eta_low=0.99, device=device)

    # ══════════════════════════════════════════════════════════════
    # Phase 1: g=10 (baseline)
    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("Phase 1: g=10.0 (baseline)")
    print("=" * 60)
    env1 = gym.make('Pendulum-v1')
    sr1, steps1, errs1, trans1 = evaluate(env1, mpc, args.episodes, 'g=10', collect_data=True)
    pred_err1 = compute_pred_error(kan, trans1, device)
    print(f"  Pred error: {pred_err1:.4f}\n")
    env1.close()

    # ══════════════════════════════════════════════════════════════
    # Phase 2: g=3 (perturbed, NO fine-tuning)
    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("Phase 2: g=3.0 (perturbed, no fine-tuning)")
    print("=" * 60)
    # Use ConfigurablePendulum for gravity change
    from experiments.continual_learning import ConfigurablePendulum
    env2 = ConfigurablePendulum(g=3.0, seed=42)
    env2.set_g(3.0)
    mpc.kan.eval()  # ensure no accidental training during eval

    sr2, steps2, errs2, trans2 = evaluate(env2, mpc, args.episodes, 'g=3-pre',
                                           collect_data=True)
    pred_err2 = compute_pred_error(kan, trans2, device)
    print(f"  Pred error: {pred_err2:.4f}")
    print(f"  Degradation: success {sr1*100:.0f}% → {sr2*100:.0f}%")
    print(f"  Pred error change: {pred_err1:.4f} → {pred_err2:.4f}\n")
    env2.env.close()

    # ══════════════════════════════════════════════════════════════
    # Phase 3: Fine-tune KAN on g=3 data
    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print(f"Phase 3: Fine-tuning KAN ({args.finetune_epochs} epochs)")
    print("=" * 60)
    finetune_kan(kan, trans2, n_epochs=args.finetune_epochs, lr=args.finetune_lr, device=device)

    # Re-verify prediction error after fine-tuning
    pred_err3 = compute_pred_error(kan, trans2, device)
    print(f"  Pred error: {pred_err2:.4f} → {pred_err3:.4f} "
          f"({(1-pred_err3/pred_err2)*100:.0f}% improvement)\n")

    # ══════════════════════════════════════════════════════════════
    # Phase 4: g=3 (after fine-tuning)
    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("Phase 4: g=3.0 (after fine-tuning)")
    print("=" * 60)
    env3 = ConfigurablePendulum(g=3.0, seed=42)
    env3.set_g(3.0)

    sr4, steps4, errs4, _ = evaluate(env3, mpc, args.episodes, 'g=3-post',
                                      collect_data=False)
    print(f"  Recovery: {sr2*100:.0f}% → {sr4*100:.0f}%\n")
    env3.env.close()

    # ══════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  g=10 baseline:     success={sr1*100:.0f}%  pred_err={pred_err1:.4f}  steps={np.mean(steps1):.0f}")
    print(f"  g=3  (no adapt):   success={sr2*100:.0f}%  pred_err={pred_err2:.4f}  steps={np.mean(steps2):.0f}")
    print(f"  g=3  (after fine): success={sr4*100:.0f}%  pred_err={pred_err3:.4f}  steps={np.mean(steps4):.0f}")

    if sr4 > sr2:
        print(f"\n  ✓ KAN fine-tuning improved success from {sr2*100:.0f}% → {sr4*100:.0f}%")
    if pred_err3 < pred_err2:
        print(f"  ✓ Prediction error reduced from {pred_err2:.4f} → {pred_err3:.4f}")


if __name__ == '__main__':
    main()
