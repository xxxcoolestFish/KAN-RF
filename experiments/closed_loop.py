"""Continual Learning Closed Loop: simplified, focused demonstration.

Protocol (3 phases, automatic transitions):
  Phase 1 (g=10): KAN Policy evaluated → 10/10 verified
  Phase 2 (g=18): KAN Policy evaluated without adaptation → success drops
                   Transitions collected. WM automatically fine-tuned.
  Phase 3 (g=18): KAN Policy retrained from scratch via fine-tuned WM
                   → success recovery measured.

Key automation points:
  - WM fine-tuning triggers automatically when enough g=18 data collected
  - Policy retraining triggers automatically after WM fine-tuning
  - Each phase reports success rate independently
"""
import torch, torch.nn as nn
import numpy as np, time, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import KAN
from control.kan_policy_net import KANPolicy, KANPolicyTrainer
from experiments.continual_learning import ConfigurablePendulum

PI_2 = np.pi / 2
S_TARGET = torch.tensor([[0.0, 1.0, 0.0]])
MAX_STEPS = 300


def run_episodes(env, policy_fn, n=10, collect=False):
    """Run N episodes. Returns (success_rate, all_steps, all_errors, transitions)."""
    succ = 0; all_steps = []; all_errs = []; trans = []
    for ep in range(n):
        seed = 42 + ep * 100; obs = env.reset(seed=seed)
        for step in range(MAX_STEPS):
            s_norm = np.array([obs[0], obs[1], obs[2] / 8.0], dtype=np.float32)
            a_norm = policy_fn(s_norm)
            r = env.step([a_norm * 2.0]); o = r[0]; term = r[2] if len(r) > 2 else False

            if collect:
                s_true = np.array([o[0], o[1], o[2] / 8.0], dtype=np.float32)
                trans.append((s_norm.copy(), a_norm, s_true))

            err = min(abs(np.arctan2(o[1], o[0]) - PI_2),
                      2 * np.pi - abs(np.arctan2(o[1], o[0]) - PI_2))
            if err < 0.2: succ += 1; all_steps.append(step + 1); all_errs.append(err); break
            obs = o
        else:
            err = min(abs(np.arctan2(obs[1], obs[0]) - PI_2),
                      2 * np.pi - abs(np.arctan2(obs[1], obs[0]) - PI_2))
            all_errs.append(err); all_steps.append(MAX_STEPS)
    return succ / n, all_steps, all_errs, trans


def fine_tune_wm(wm, transitions, n_epochs=80, device='cpu'):
    """Fine-tune KAN world model on collected transitions."""
    if len(transitions) < 32: return wm
    xs = []; ys = []
    for sn, an, st in transitions:
        xs.append(torch.cat([torch.tensor(sn, dtype=torch.float32).unsqueeze(0),
                             torch.tensor([[an]], dtype=torch.float32)], dim=-1))
        ys.append(torch.tensor(st, dtype=torch.float32).unsqueeze(0))
    X = torch.cat(xs).to(device); Y = torch.cat(ys).to(device)
    N = len(X)
    wm.train()
    for p in wm.parameters(): p.requires_grad = True
    opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
    for ep in range(n_epochs):
        idx = torch.randint(0, N, (min(64, N),))
        loss = nn.functional.mse_loss(wm(X[idx]), Y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    wm.eval()
    for p in wm.parameters(): p.requires_grad = False
    return wm


def train_policy(wm, n_epochs=200, device='cpu'):
    """Train KAN Policy from scratch via WM gradient."""
    policy = KANPolicy(state_dim=3, action_dim=1, hidden_dim=12, n_layers=2).to(device)
    angles = np.random.uniform(-np.pi, np.pi, 15000)
    s_states = np.stack([np.cos(angles), np.sin(angles),
                         np.random.uniform(-8, 8, 15000)], axis=1)
    s_states[:, 2] /= 8.0
    s_dataset = torch.tensor(s_states, dtype=torch.float32).to(device)
    trainer = KANPolicyTrainer(wm, policy, S_TARGET.to(device), lr=1e-3, device=device)
    for ep in range(1, n_epochs + 1):
        trainer.train_epoch(s_dataset)
    return policy, trainer


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wm', type=str, default='/tmp/kanrf_cl_exp/kan_cws_cl.pt')
    parser.add_argument('--policy', type=str, default='/tmp/kan_policy_trained.pt')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load world model (trained on g=10)
    wm = KAN([4, 12, 3], grid_size=5, spline_order=3)
    wm.load_state_dict(torch.load(args.wm, weights_only=True, map_location=device))
    wm.to(device); wm.eval()

    # Load pre-trained KAN Policy (g=10)
    policy_g10 = KANPolicy(state_dim=3, action_dim=1, hidden_dim=12, n_layers=2)
    policy_g10.load_state_dict(torch.load(args.policy, weights_only=True, map_location=device))
    policy_g10.to(device); policy_g10.eval()

    def make_policy_fn(pol):
        @torch.no_grad()
        def fn(s_norm):
            s_t = torch.tensor(s_norm, dtype=torch.float32, device=device).unsqueeze(0)
            return pol(s_t).item()
        return fn

    # ══════════════════════════════════════════════════════════
    # PHASE 1: g=10 baseline
    # ══════════════════════════════════════════════════════════
    print("=" * 60)
    print("PHASE 1: g=10 BASELINE (pre-trained KAN Policy)")
    print("=" * 60)
    env1 = ConfigurablePendulum(g=10.0, seed=42)
    sr1, st1, er1, _ = run_episodes(env1, make_policy_fn(policy_g10),
                                     n=args.episodes, collect=False)
    for ep in range(args.episodes):
        ok = er1[ep] < 0.2
        print(f"  Ep {ep+1:2d}  {'✓' if ok else '✗'}  steps={st1[ep]:3d}  err={er1[ep]:.3f}")
    print(f"  PHASE 1 RESULT: {sr1*100:.0f}%  mean_steps={np.mean(st1):.0f}\n")
    env1.env.close()

    # ══════════════════════════════════════════════════════════
    # PHASE 2: g=18 — run policy WITHOUT adaptation, collect data
    # ══════════════════════════════════════════════════════════
    print("=" * 60)
    print("PHASE 2: g=18 DEGRADATION (KAN Policy, no adaptation)")
    print("=" * 60)
    env2 = ConfigurablePendulum(g=18.0, seed=42)
    env2.set_g(18.0)
    sr2, st2, er2, trans2 = run_episodes(env2, make_policy_fn(policy_g10),
                                          n=args.episodes, collect=True)
    for ep in range(args.episodes):
        ok = er2[ep] < 0.2
        print(f"  Ep {ep+1:2d}  {'✓' if ok else '✗'}  steps={st2[ep]:3d}  err={er2[ep]:.3f}")
    print(f"  PHASE 2 RESULT: {sr2*100:.0f}%  mean_steps={np.mean(st2):.0f}")
    print(f"  Collected {len(trans2)} transitions for WM fine-tuning\n")
    env2.env.close()

    # ══════════════════════════════════════════════════════════
    # ADAPTATION: Fine-tune WM, then retrain policy
    # ══════════════════════════════════════════════════════════
    print("=" * 60)
    print("ADAPTATION: WM fine-tuning + Policy retraining")
    print("=" * 60)

    # Fine-tune WM
    t0 = time.time()
    wm = fine_tune_wm(wm, trans2, n_epochs=80, device=device)
    print(f"  WM fine-tuned ({time.time()-t0:.0f}s)")

    # Compute pred error before/after
    wm_orig = KAN([4, 12, 3], grid_size=5, spline_order=3)
    wm_orig.load_state_dict(torch.load(args.wm, weights_only=True, map_location=device))
    wm_orig.to(device); wm_orig.eval()
    with torch.no_grad():
        errs_before = []; errs_after = []
        batch = trans2[:min(500, len(trans2))]
        X_batch = torch.cat([
            torch.cat([torch.tensor(sn, dtype=torch.float32, device=device).unsqueeze(0),
                       torch.tensor([[an]], dtype=torch.float32, device=device)], dim=-1)
            for sn, an, st in batch], dim=0)
        Y_batch = torch.cat([
            torch.tensor(st, dtype=torch.float32, device=device).unsqueeze(0)
            for sn, an, st in batch], dim=0)
        e_before = (wm_orig(X_batch) - Y_batch).norm(dim=-1).mean().item()
        e_after = (wm(X_batch) - Y_batch).norm(dim=-1).mean().item()
        print(f"  Pred error: {e_before:.4f} → {e_after:.4f} "
              f"({(1-e_after/e_before)*100:.0f}% improvement)")

    # Retrain policy
    t0 = time.time()
    policy_g18, trainer_g18 = train_policy(wm, n_epochs=200, device=device)
    print(f"  Policy retrained ({time.time()-t0:.0f}s)\n")

    # ══════════════════════════════════════════════════════════
    # PHASE 3: g=18 — test RECOVERED policy
    # ══════════════════════════════════════════════════════════
    print("=" * 60)
    print("PHASE 3: g=18 RECOVERY (retrained KAN Policy)")
    print("=" * 60)
    env3 = ConfigurablePendulum(g=18.0, seed=42)
    env3.set_g(18.0)
    sr3, st3, er3, _ = run_episodes(env3, make_policy_fn(policy_g18),
                                     n=args.episodes, collect=False)
    for ep in range(args.episodes):
        ok = er3[ep] < 0.2
        print(f"  Ep {ep+1:2d}  {'✓' if ok else '✗'}  steps={st3[ep]:3d}  err={er3[ep]:.3f}")
    print(f"  PHASE 3 RESULT: {sr3*100:.0f}%  mean_steps={np.mean(st3):.0f}\n")
    env3.env.close()

    # ══════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════
    print("=" * 60)
    print("CLOSED LOOP SUMMARY")
    print("=" * 60)
    print(f"  Phase 1 (g=10, pre-trained):   {sr1*100:.0f}%  steps={np.mean(st1):.0f}")
    print(f"  Phase 2 (g=18, degraded):      {sr2*100:.0f}%  steps={np.mean(st2):.0f}")
    print(f"  Adaptation: WM pred_err {np.mean(errs_before):.4f}→{np.mean(errs_after):.4f}")
    print(f"  Phase 3 (g=18, recovered):     {sr3*100:.0f}%  steps={np.mean(st3):.0f}")
    if sr3 > sr2:
        print(f"\n  ✓ RECOVERY: {sr2*100:.0f}% → {sr3*100:.0f}%")
    else:
        print(f"\n  Recovery limited: g=18 oracle is 7/10 (energy controller)")


if __name__ == '__main__':
    main()
