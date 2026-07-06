"""CartPole: ProtoKAN vs KAN — full comparison where WM accuracy matters.

CartPole is the right difficulty: dynamics are more complex than Pendulum
(4D state, nonlinear, cart-pole coupling) but MPC-solvable unlike Acrobot.

Tests whether ProtoKAN's superior WM accuracy translates to better decisions.
"""
import torch, torch.nn as nn, numpy as np, time, sys, os, gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import ProtoKAN, ProtoKANLayer, KAN, KANLayer

# CartPole dynamics constants
G = 9.8; MC = 1.0; MP = 0.1; L = 0.5; DT = 0.02
TOTAL_MASS = MC + MP; PML = MP * L
X_S = 2.5; XD_S = 3.0; TH_S = 0.3; THD_S = 3.0; FM = 10.0


def step_cartpole(state, a_norm):
    """CartPole step with continuous force. state: (B,4) raw [x,xd,th,thd]."""
    x, xd, th, thd = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    force = a_norm * FM
    costh, sinth = torch.cos(th), torch.sin(th)
    temp = (force + PML * thd ** 2 * sinth) / TOTAL_MASS
    denom = 0.5 * (4.0 / 3.0 - MP * costh ** 2 / TOTAL_MASS)
    th_acc = (G * sinth - costh * temp) / (denom + 1e-8)
    x_acc = temp - PML * th_acc * costh / TOTAL_MASS
    xd_n = xd + x_acc * DT
    thd_n = thd + th_acc * DT
    x_n = x + xd_n * DT
    th_n = th + thd_n * DT
    return torch.stack([x_n, xd_n, th_n, thd_n], dim=-1)


def generate_data(n_states=8000, device='cpu'):
    """Generate (s, a, s') triplets from CartPole simulator."""
    xs, ys = [], []
    for _ in range(n_states):
        x = np.random.uniform(-2.4, 2.4)
        xd = np.random.uniform(-3.0, 3.0)
        th = np.random.uniform(-0.3, 0.3)
        thd = np.random.uniform(-3.0, 3.0)
        a = np.random.uniform(-1.0, 1.0)
        s_raw = torch.tensor([[x, xd, th, thd]], dtype=torch.float32)
        s_next = step_cartpole(s_raw, torch.tensor([a]))
        s_norm = s_raw.clone(); s_norm[:, 0] /= X_S; s_norm[:, 1] /= XD_S
        s_norm[:, 2] /= TH_S; s_norm[:, 3] /= THD_S
        sn_norm = s_next.clone(); sn_norm[:, 0] /= X_S; sn_norm[:, 1] /= XD_S
        sn_norm[:, 2] /= TH_S; sn_norm[:, 3] /= THD_S
        xs.append(torch.cat([s_norm, torch.tensor([[a]])], dim=-1))
        ys.append(sn_norm)
    return (torch.cat(xs, dim=0).float().to(device),
            torch.cat(ys, dim=0).float().to(device))


def train_wm(X, Y, wm_type='protokan', n_proto=16, n_lbfgs=100, device='cpu'):
    """Train WM [5, 16, 4] with L-BFGS."""
    n_tr = int(len(X) * 0.85)
    X_tr, Y_tr = X[:n_tr], Y[:n_tr]
    X_val, Y_val = X[n_tr:], Y[n_tr:]

    if wm_type == 'protokan':
        wm = ProtoKAN([5, 16, 4], n_prototypes=n_proto).to(device)
    else:
        wm = KAN([5, 16, 4], grid_size=5, spline_order=3).to(device)

    n_p = sum(p.numel() for p in wm.parameters())
    print(f"  {wm_type} WM [5,16,4]: {n_p} params")

    mse_fn = nn.MSELoss()
    best_val = float('inf')
    best_state = None

    def closure():
        opt.zero_grad()
        loss = mse_fn(wm(X_tr), Y_tr)
        loss.backward()
        return loss

    opt = torch.optim.LBFGS(wm.parameters(), lr=1.0, max_iter=20,
                            history_size=50, line_search_fn='strong_wolfe')

    for step in range(1, n_lbfgs + 1):
        opt.step(closure)
        with torch.no_grad():
            val = mse_fn(wm(X_val), Y_val).item()
        if val < best_val:
            best_val = val
            best_state = {k: v.clone() for k, v in wm.state_dict().items()}
        if step % 25 == 0 or step == 1:
            print(f"    L-BFGS {step:3d}/{n_lbfgs}  val_mse={val:.6f}  best={best_val:.6f}")

    wm.load_state_dict(best_state)
    wm.eval()
    return wm, best_val


def train_policy(wm, s_dataset, policy_type='kan', n_proto=16, epochs=300,
                 lr=1e-3, device='cpu'):
    """Train a policy via frozen WM gradient."""
    if policy_type == 'protokan':
        policy = nn.Sequential(
            ProtoKANLayer(4, 12, n_prototypes=n_proto),
            ProtoKANLayer(12, 12, n_prototypes=n_proto),
            nn.Linear(12, 1),
        ).to(device)
        # Wrap with tanh output
        class PolicyWrapper(nn.Module):
            def __init__(self, net):
                super().__init__(); self.net = net
            def forward(self, s):
                return torch.tanh(self.net(s))
        policy = PolicyWrapper(policy)
    else:
        from control.kan_policy_net import KANPolicy
        policy = KANPolicy(state_dim=4, action_dim=1, hidden_dim=12, n_layers=2).to(device)

    n_p = sum(p.numel() for p in policy.parameters())
    print(f"  {policy_type} Policy [4,12,12,1]: {n_p} params")

    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False

    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    N = len(s_dataset)
    s_target = torch.zeros(1, 4, device=device)

    for ep in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = max(1, N // 256)
        for _ in range(n_batches):
            idx = torch.randint(0, N, (256,), device=device)
            s_b = s_dataset[idx]
            policy.train(); opt.zero_grad()
            a = policy(s_b)
            s_pred = wm(torch.cat([s_b, a], dim=-1))
            # Stabilization loss: keep pole upright, cart centered
            loss = (s_pred[:, 2].pow(2).mean() + 0.1 * s_pred[:, 0].pow(2).mean() +
                    0.5 * s_pred[:, 3].pow(2).mean() + 0.01 * a.pow(2).mean())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            opt.step()
            total_loss += loss.item()
        if ep % 60 == 0 or ep == 1:
            print(f"    Epoch {ep:3d}  loss={total_loss / n_batches:.4f}")
    return policy


def evaluate_policy(policy, n_trials=20, max_steps=500, device='cpu'):
    """Evaluate on CartPole with analytical dynamics."""
    successes = 0
    all_steps = []
    for trial in range(n_trials):
        seed = 42 + trial * 100
        np.random.seed(seed)
        th = np.random.uniform(-0.05, 0.05)
        s_raw = torch.tensor([[0.0, 0.0, th, 0.0]], dtype=torch.float32, device=device)
        for step in range(max_steps):
            s_norm = s_raw.clone()
            s_norm[:, 0] /= X_S; s_norm[:, 1] /= XD_S
            s_norm[:, 2] /= TH_S; s_norm[:, 3] /= THD_S
            with torch.no_grad():
                a_norm = policy(s_norm).item()
            s_raw = step_cartpole(s_raw, torch.tensor([a_norm], device=device))
            theta = s_raw[0, 2].item()
            x = s_raw[0, 0].item()
            if abs(theta) > 0.21 or abs(x) > 2.4:
                break
        all_steps.append(step + 1)
        if step + 1 >= max_steps:
            successes += 1
        status = '✓' if step + 1 >= max_steps else '✗'
        if trial < 10 or status == '✗':
            print(f"    [{trial+1:2d}] {status} steps={step+1}")
    return successes, all_steps


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_states', type=int, default=8000)
    parser.add_argument('--n_policy', type=int, default=15000)
    parser.add_argument('--n_proto', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(42); np.random.seed(42)

    print("=" * 70)
    print("CartPole: ProtoKAN vs KAN — WM Accuracy → Decision Quality")
    print("=" * 70)

    # ── 1. Data ──
    print(f"\n[1] Generating {args.n_states} (s,a,s') triplets...")
    t0 = time.time()
    X, Y = generate_data(args.n_states, device)
    print(f"    {X.shape[0]} samples in {time.time()-t0:.0f}s")

    # Policy training states
    s_pol = torch.cat([
        X[:, :4],  # use existing states
        torch.randn(args.n_policy - X.shape[0], 4, device=device) * 0.5  # + random
    ], dim=0)[:args.n_policy]
    print(f"    Policy states: {s_pol.shape[0]}")

    # ── 2. Train ProtoKAN WM ──
    print(f"\n[2] Training ProtoKAN WM...")
    t0 = time.time()
    proto_wm, proto_mse = train_wm(X, Y, 'protokan', args.n_proto, 100, device)
    torch.save(proto_wm.state_dict(), '/tmp/cartpole_proto_wm.pt')
    print(f"    val_mse={proto_mse:.6f}  time={time.time()-t0:.0f}s  saved")

    # ── 3. Train KAN WM ──
    print(f"\n[3] Training KAN WM...")
    t0 = time.time()
    kan_wm, kan_mse = train_wm(X, Y, 'kan', args.n_proto, 100, device)
    torch.save(kan_wm.state_dict(), '/tmp/cartpole_kan_wm.pt')
    print(f"    val_mse={kan_mse:.6f}  time={time.time()-t0:.0f}s  saved")
    print(f"    ProtoKAN / KAN: {kan_mse/proto_mse:.1f}x better" if proto_mse < kan_mse
          else f"    KAN / ProtoKAN: {proto_mse/kan_mse:.1f}x")

    # ── 4. Train ProtoKAN Policy ──
    print(f"\n[4] Training ProtoKAN Policy via ProtoKAN WM...")
    proto_pol = train_policy(proto_wm, s_pol, 'protokan', args.n_proto,
                             args.epochs, device=device)

    # ── 5. Train KAN Policy via KAN WM ──
    print(f"\n[5] Training KAN Policy via KAN WM...")
    kan_pol = train_policy(kan_wm, s_pol, 'kan', args.n_proto,
                           args.epochs, device=device)

    # ── 6. Evaluate ──
    print(f"\n[6] Evaluating ProtoKAN Stack...")
    p_s, p_steps = evaluate_policy(proto_pol, n_trials=20, device=device)
    print(f"    Result: {p_s}/20 ({p_s*5}%)  mean_steps={np.mean(p_steps):.0f}")

    print(f"\n[7] Evaluating KAN Stack...")
    k_s, k_steps = evaluate_policy(kan_pol, n_trials=20, device=device)
    print(f"    Result: {k_s}/20 ({k_s*5}%)  mean_steps={np.mean(k_steps):.0f}")

    # ── 7. Report ──
    print(f"\n{'='*70}")
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"  WM Accuracy:")
    print(f"    ProtoKAN:  val_mse={proto_mse:.6f}")
    print(f"    KAN:       val_mse={kan_mse:.6f}")
    print(f"    Ratio:     {kan_mse/proto_mse:.1f}x ProtoKAN better")
    print(f"  Decision Quality:")
    print(f"    ProtoKAN Stack:  {p_s}/20 ({p_s*5}%)  mean_steps={np.mean(p_steps):.0f}")
    print(f"    KAN Stack:       {k_s}/20 ({k_s*5}%)  mean_steps={np.mean(k_steps):.0f}")


if __name__ == '__main__':
    main()
