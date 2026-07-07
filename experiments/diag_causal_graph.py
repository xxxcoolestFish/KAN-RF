"""Extract actual causal graph structure from ProtoKAN WM.

Computes both ∂s'/∂a (action→state) and ∂s'/∂s (state→state) Jacobians
from trained ProtoKAN world models for Pendulum and CartPole.
"""
import torch, numpy as np, sys
sys.path.insert(0, '.')
from kanrf import ProtoKAN

device = 'cpu'
torch.manual_seed(42)

# State dimension names
PENDULUM_NAMES = ['cosθ', 'sinθ', 'θ̇']
CART_POLE_NAMES = ['x', 'ẋ', 'θ', 'θ̇']


def extract_causal_structure(wm, state_dim, state_names, n_samples=500):
    """Extract ∂s'/∂a and ∂s'/∂s from ProtoKAN WM over random states."""
    wm.eval()

    Ja_mag = np.zeros(state_dim)      # |∂s'[i]/∂a|
    Ja_std = np.zeros(state_dim)
    Js_mag = np.zeros((state_dim, state_dim))  # |∂s'[i]/∂s[j]|
    Js_std = np.zeros((state_dim, state_dim))

    all_Ja = [[] for _ in range(state_dim)]
    all_Js = [[[] for _ in range(state_dim)] for _ in range(state_dim)]

    for _ in range(n_samples):
        s = torch.randn(1, state_dim) * 0.5
        s = s.clamp(-1, 1)

        # ∂s'/∂a — per-dim action Jacobian
        a = torch.zeros(1, 1, requires_grad=True)
        s_pred = wm(torch.cat([s, a], dim=-1))
        for i in range(state_dim):
            g = torch.autograd.grad(s_pred[0, i], a, retain_graph=True)[0]
            all_Ja[i].append(abs(g[0, 0].item()))

        # ∂s'/∂s — per-input state Jacobian
        for j in range(state_dim):
            s_j = s.clone().detach().requires_grad_(True)
            a_fixed = torch.zeros(1, 1)
            s_pred = wm(torch.cat([s_j, a_fixed], dim=-1))
            for i in range(state_dim):
                grad_out = torch.autograd.grad(s_pred[0, i], s_j, retain_graph=True)[0]
                g = grad_out[0, j].item()
                all_Js[i][j].append(abs(g))

    for i in range(state_dim):
        Ja_mag[i] = np.mean(all_Ja[i])
        Ja_std[i] = np.std(all_Ja[i])
        for j in range(state_dim):
            Js_mag[i, j] = np.mean(all_Js[i][j])
            Js_std[i, j] = np.std(all_Js[i][j])

    return Ja_mag, Ja_std, Js_mag, Js_std


def print_causal_graph(Ja_mag, Ja_std, Js_mag, Js_std, state_names):
    """Print causal graph as adjacency matrices."""
    n = len(state_names)
    all_names = state_names + ['a']

    print(f"\n  {'='*60}")
    print(f"  Action → State (∂s'/∂a):")
    print(f"  {'='*60}")
    for i, name in enumerate(state_names):
        rel = Ja_mag[i] / (Ja_mag.max() + 1e-8)
        bar = '█' * int(rel * 20)
        print(f"    a → {name:4s}: {Ja_mag[i]:.6f} ± {Ja_std[i]:.6f}  {bar}")

    print(f"\n  {'='*60}")
    print(f"  State → State (∂s'/∂s):")
    print(f"  {'='*60}")
    # Header
    header = "        " + "".join(f"{name:>10s}" for name in state_names)
    print(header)
    max_val = Js_mag.max()
    for i, name_i in enumerate(state_names):
        row = f"    {name_i:>3s} →"
        for j in range(n):
            val = Js_mag[i, j]
            rel = val / (max_val + 1e-8)
            bar = '█' if rel > 0.3 else ('▓' if rel > 0.1 else ('░' if rel > 0.01 else ' '))
            row += f"  {val:.4f}{bar}"
        print(row)

    # Find causal paths from a to each state dimension
    print(f"\n  {'='*60}")
    print(f"  Causal influence paths (a → ... → s[i]):")
    print(f"  {'='*60}")
    threshold = 0.01
    for target in range(n):
        # Direct: a → s[target]
        direct = Ja_mag[target]
        # Indirect paths of length 2: a → s[k] → s[target]
        paths = []
        for k in range(n):
            indirect = Ja_mag[k] * Js_mag[target, k]
            if indirect > threshold:
                paths.append((k, indirect))

        print(f"    Target {state_names[target]}:")
        print(f"      Direct (a→target): {direct:.6f}")
        for k, val in sorted(paths, key=lambda x: -x[1]):
            print(f"      2-step (a→{state_names[k]}→target): {val:.6f}")


def main():
    # ── Pendulum ──
    print("=" * 70)
    print("PENDULUM ProtoKAN WM [4,12,3] — Causal Graph")
    print("=" * 70)

    wm_pen = ProtoKAN([4, 12, 3], n_prototypes=16)
    try:
        wm_pen.load_state_dict(torch.load('/tmp/protokAN_wm_pendulum.pt', weights_only=True))
        print("  Loaded trained WM")
    except:
        print("  Training fresh...")
        from experiments.baseline_sweep import generate_pendulum_data, train_wm
        X, Y = generate_pendulum_data(5000, seed=42)
        wm_pen, _ = train_wm(X, Y)

    Ja, Jas, Js, Jss = extract_causal_structure(wm_pen, 3, PENDULUM_NAMES)
    print_causal_graph(Ja, Jas, Js, Jss, PENDULUM_NAMES)

    # ── CartPole ──
    print("\n" + "=" * 70)
    print("CART POLE ProtoKAN WM [5,16,4] — Causal Graph")
    print("=" * 70)

    wm_cp = ProtoKAN([5, 16, 4], n_prototypes=16)
    try:
        wm_cp.load_state_dict(torch.load('/tmp/cartpole_proto_wm.pt', weights_only=True))
        print("  Loaded trained WM")
    except:
        print("  Training fresh...")
        from experiments.cartpole_continual import generate_wm_data, train_wm
        X, Y = generate_wm_data(g=9.8, n=5000)
        wm_cp, _ = train_wm(X, Y, 'protokan', 80, device)

    Ja, Jas, Js, Jss = extract_causal_structure(wm_cp, 4, CART_POLE_NAMES)
    print_causal_graph(Ja, Jas, Js, Jss, CART_POLE_NAMES)


if __name__ == '__main__':
    main()
