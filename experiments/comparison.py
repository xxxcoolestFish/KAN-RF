"""KAN Policy vs MLP Policy comparison on Pendulum-v1.

Metrics: success rate, parameters, inference speed, interpretability dimensions.
"""
import torch, numpy as np, time, gymnasium, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import KAN
from control.kan_policy_net import KANPolicy, KANPolicyTrainer
from control.kan_interpretability import KANInterpreter
from decision_v3.core import KANPolicy as MLPPolicy, KANEnergyTrainer

PI_2 = np.pi / 2
S_TARGET = torch.tensor([[0.0, 1.0, 0.0]])


def main():
    device = torch.device('cpu')

    # Load frozen CWS-KAN world model
    wm = KAN([4, 12, 3], grid_size=5, spline_order=3)
    wm.load_state_dict(torch.load('/tmp/kanrf_cl_exp/kan_cws_cl.pt', weights_only=True))
    wm.to(device); wm.eval()

    # Training states
    angles = np.random.uniform(-np.pi, np.pi, 20000)
    s_states = np.stack([np.cos(angles), np.sin(angles), np.random.uniform(-8, 8, 20000)], axis=1)
    s_states[:, 2] /= 8.0
    s_dataset = torch.tensor(s_states, dtype=torch.float32).to(device)

    # ── 1. KAN Policy ──
    print("=" * 60)
    print("1. KAN Policy [3,12,12,1]")
    print("=" * 60)
    t0 = time.time()
    kan_pol = KANPolicy(state_dim=3, action_dim=1, hidden_dim=12, n_layers=2).to(device)
    kan_trainer = KANPolicyTrainer(wm, kan_pol, S_TARGET.to(device), lr=1e-3, device=device)
    for ep in range(1, 301):
        ld = kan_trainer.train_epoch(s_dataset)
        if ep % 50 == 0: print(f"  Ep {ep:3d}  loss={ld['total']:.4f}")
    kan_time = time.time() - t0
    kan_params = sum(p.numel() for p in kan_pol.parameters())

    # ── 2. MLP Policy ──
    print("\n" + "=" * 60)
    print("2. MLP Policy [3,64,64,1] (decision_v3)")
    print("=" * 60)
    t0 = time.time()
    mlp_pol = MLPPolicy(state_dim=3, action_dim=1, hidden=64, n_layers=2).to(device)
    mlp_trainer = KANEnergyTrainer(wm, mlp_pol, S_TARGET.to(device), lr=1e-3, device=device)
    for ep in range(1, 301):
        ld = mlp_trainer.train_epoch(s_dataset)
        if ep % 50 == 0: print(f"  Ep {ep:3d}  loss={ld['total']:.4f}")
    mlp_time = time.time() - t0
    mlp_params = sum(p.numel() for p in mlp_pol.parameters())

    # ── 3. Evaluate ──
    print("\n" + "=" * 60)
    print("3. Pendulum-v1 Evaluation (10 trials)")
    print("=" * 60)

    def evaluate(policy_fn, n=10):
        env = gymnasium.make('Pendulum-v1')
        succ = 0; steps_list = []; errs = []; times = []
        for trial in range(n):
            seed = 42 + trial * 100; obs, _ = env.reset(seed=seed)
            for step in range(300):
                t1 = time.time()
                s_n = np.array([obs[0], obs[1], obs[2] / 8.0], dtype=np.float32)
                a = policy_fn(s_n); times.append(time.time() - t1)
                a_raw = float(a) * 2.0
                obs, _, term, trunc, _ = env.step([a_raw])
                err = min(abs(np.arctan2(obs[1], obs[0]) - PI_2),
                          2 * np.pi - abs(np.arctan2(obs[1], obs[0]) - PI_2))
                if err < 0.2: succ += 1; steps_list.append(step + 1); errs.append(err); break
            else:
                err = min(abs(np.arctan2(obs[1], obs[0]) - PI_2),
                          2 * np.pi - abs(np.arctan2(obs[1], obs[0]) - PI_2))
                errs.append(err); steps_list.append(300)
        env.close()
        return succ / n, np.mean(steps_list), np.mean(errs), np.mean(times) * 1000

    kan_sr, kan_steps, kan_err, kan_infer = evaluate(lambda s: kan_trainer.get_action(s))
    mlp_sr, mlp_steps, mlp_err, mlp_infer = evaluate(lambda s: mlp_trainer.get_action(s))
    print(f"  KAN Policy: {kan_sr*100:.0f}%  steps={kan_steps:.0f}  infer={kan_infer:.3f}ms")
    print(f"  MLP Policy: {mlp_sr*100:.0f}%  steps={mlp_steps:.0f}  infer={mlp_infer:.3f}ms")

    # ── 4. Interpretability ──
    print("\n" + "=" * 60)
    print("4. Interpretability (KAN Policy only)")
    print("=" * 60)
    interp = KANInterpreter(kan_pol, input_names=['cosθ', 'sinθ', 'thd/8'])
    g = interp.prune_and_analyze(threshold=0.15)
    sp_l0 = g['layers'][0]['sparsity']
    sp_l1 = g['layers'][1]['sparsity']
    print(f"  Sparsity (th=0.15): L0={sp_l0*100:.0f}%  L1={sp_l1*100:.0f}%")

    interp.attribute_batch([
        np.array([1., 0., 0.]), np.array([-1., 0., 0.]),
        np.array([0., 1., 0.]), np.array([0., -1., 0.])
    ], ['far-right', 'far-left', 'upright', 'bottom'])
    for label, data in interp.attributions.items():
        c = data['contributions']
        print(f"  {label:12s}: a={data['action']:+.3f}  "
              f"cos={c[0]*100:.0f}% sin={c[1]*100:.0f}% thd={c[2]*100:.0f}%")

    sym = interp.fit_symbolic(min_r2=0.90)
    forms = {}
    for s in sym: forms[s['form']] = forms.get(s['form'], 0) + 1
    form_str = ' + '.join(f'{n}{f}' for f, n in sorted(forms.items()))
    print(f"  Symbolic (R²>0.9): {len(sym)} edges ({form_str})")

    # ── 5. Summary Table ──
    print("\n" + "=" * 60)
    print("5. SUMMARY")
    print("=" * 60)
    rows = [
        ("Architecture", "[3,12,12,1]", "[3,64,64,1]"),
        ("Parameters", str(kan_params), str(mlp_params)),
        ("Training time", f"{kan_time:.0f}s", f"{mlp_time:.0f}s"),
        ("Success rate", f"{kan_sr*100:.0f}%", f"{mlp_sr*100:.0f}%"),
        ("Mean steps", f"{kan_steps:.0f}", f"{mlp_steps:.0f}"),
        ("Inference", f"{kan_infer:.3f}ms", f"{mlp_infer:.3f}ms"),
        ("Edge functions", "✓ 72 B-splines", "✗ Black-box"),
        ("Causal graph", f"✓ {sp_l0*100:.0f}%/{sp_l1*100:.0f}% sparse", "✗ N/A"),
        ("Attribution", "✓ Per-input", "✗ N/A"),
        ("Symbolic", f"✓ {len(sym)} edges", "✗ N/A"),
        ("Continual learning", "✓ B-spline local", "✗ Global weights"),
    ]
    for name, kan_val, mlp_val in rows:
        print(f"  {name:20s}  {kan_val:>18s}  {mlp_val:>18s}")


if __name__ == '__main__':
    main()
