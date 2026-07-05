"""Train Acrobot CWS-KAN World Model from scratch.

CWS (Controllability-Weighted Sobolev): matches KAN Jacobian ∂f/∂a to
true Jacobian via finite-difference on the Acrobot simulator.
"""
import torch, numpy as np, time, sys, os
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import KAN


def generate_data(n_samples=10000, device='cpu'):
    """Generate single-scale (k=1) Acrobot training data with Jacobian labels.

    Returns:
        X: (N, 9) = [state(6) + action_onehot(3)]  (no k_norm, single-scale)
        Y: (N, 6) next state
        J: (N, 6, 3) true Jacobian ∂s_next/∂a (finite difference)
    """
    env = gym.make('Acrobot-v1')
    env.reset()

    max_v1, max_v2 = 6.0, 8.0
    xs, ys, js = [], [], []

    for _ in range(n_samples):
        # Random state
        theta1 = np.random.uniform(-np.pi, np.pi)
        theta2 = np.random.uniform(-np.pi, np.pi)
        dtheta1 = np.random.uniform(-max_v1, max_v1)
        dtheta2 = np.random.uniform(-max_v2, max_v2)
        env.unwrapped.state = (theta1, theta2, dtheta1, dtheta2)
        s0_obs = env.unwrapped._get_ob()

        # Compute s' for each action + Jacobian via finite difference
        s_next_list = []
        for a in range(3):
            env.unwrapped.state = (theta1, theta2, dtheta1, dtheta2)
            obs, _, term, _, _ = env.step(a)
            s_next_list.append(obs)

        # Normalize
        s_norm = np.array([s0_obs[0], s0_obs[1], s0_obs[2], s0_obs[3],
                          s0_obs[4]/max_v1, s0_obs[5]/max_v2], dtype=np.float32)

        s_next_norms = []
        for obs in s_next_list:
            sn = np.array([obs[0], obs[1], obs[2], obs[3],
                          obs[4]/max_v1, obs[5]/max_v2], dtype=np.float32)
            s_next_norms.append(sn)
        s_next_norms = np.array(s_next_norms)  # (3, 6)

        # Jacobian: 6×3 matrix via finite difference between discrete actions
        # We treat the action as a continuous 3-dim one-hot vector.
        # J = [∂s'/∂a0, ∂s'/∂a1, ∂s'/∂a2]
        # Approx: J[:,i] ≈ s'(a=i+ε*e_i) - s'(a=i) / ε ... but actions are discrete.
        # Instead: J[:,i] = s'(a=i) - s'(reference)  ...use a=1 as reference
        J = np.zeros((6, 3), dtype=np.float32)
        # Use centered difference around the "average" state
        ref = s_next_norms.mean(axis=0)  # average next state
        for i in range(3):
            J[:, i] = s_next_norms[i] - ref

        # Generate 3 data points (one per action)
        for a in range(3):
            a_oh = np.zeros(3, dtype=np.float32); a_oh[a] = 1.0
            x = np.concatenate([s_norm, a_oh])  # (9,)
            y = s_next_norms[a]  # (6,)
            xs.append(x); ys.append(y); js.append(J)

    env.close()
    X = torch.tensor(np.array(xs), dtype=torch.float32).to(device)
    Y = torch.tensor(np.array(ys), dtype=torch.float32).to(device)
    J = torch.tensor(np.array(js), dtype=torch.float32).to(device)
    return X, Y, J


def cws_jacobian_loss_acrobot(model, s_batch, a_batch, J_true):
    """Compute CWS Jacobian matching loss for Acrobot.

    Args:
        model: KAN world model
        s_batch: (B, 6) normalized states
        a_batch: (B, 3) action one-hot (requires grad)
        J_true: (B, 6, 3) true Jacobian from finite difference

    Returns:
        scalar loss: mean ||J_model - J_true||²
    """
    a = a_batch.clone().detach().requires_grad_(True)
    x = torch.cat([s_batch, a], dim=-1)
    s_pred = model(x)

    J_model = []
    for dim in range(6):
        grads = []
        for act_dim in range(3):
            g = torch.autograd.grad(
                s_pred[:, dim].sum(), a,
                retain_graph=True, create_graph=True
            )[0][:, act_dim:act_dim+1]
            grads.append(g)
        J_model.append(torch.cat(grads, dim=-1))  # (B, 3)
    J_model = torch.stack(J_model, dim=1)  # (B, 6, 3)

    err = (J_model - J_true).pow(2).mean()
    return err


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=20000)
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--nu', type=float, default=0.1, help='CWS weight')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(42); np.random.seed(42)

    # Generate data
    print(f"Generating {args.samples} Acrobot samples with Jacobian labels...")
    t0 = time.time()
    X, Y, J = generate_data(args.samples, device)
    # Note: X is 3*n_samples because we have one per action
    print(f"  Generated {len(X)} samples in {time.time()-t0:.0f}s")
    print(f"  X: {X.shape}, Y: {Y.shape}, J: {J.shape}")

    # Train/val split
    n_train = int(len(X) * 0.85)
    X_tr, Y_tr, J_tr = X[:n_train], Y[:n_train], J[:n_train]
    X_val, Y_val = X[n_train:], Y[n_train:]

    # Model
    model = KAN([9, 24, 6], grid_size=5, spline_order=3).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: [9,24,6], {n_params} params")
    print(f"CWS ν={args.nu}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=150, gamma=0.5)
    mse_fn = torch.nn.MSELoss()

    batch_size = 1024  # smaller because Jacobian computation is expensive
    best_val = float('inf'); best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        idx = torch.randperm(n_train)[:batch_size]
        xb, yb, jb = X_tr[idx], Y_tr[idx], J_tr[idx]
        sb, ab = xb[:, :6], xb[:, 6:9]

        opt.zero_grad()
        pred = model(xb)
        loss_mse = mse_fn(pred, yb)
        loss_cws = args.nu * cws_jacobian_loss_acrobot(model, sb, ab, jb)
        loss = loss_mse + loss_cws
        loss.backward()
        opt.step()
        scheduler.step()

        if epoch % 100 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_mse = mse_fn(model(X_val), Y_val).item()
            if val_mse < best_val:
                best_val = val_mse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  Epoch {epoch:3d}  lr={scheduler.get_last_lr()[0]:.4f}  "
                  f"val_mse={val_mse:.6f}  best={best_val:.6f}  "
                  f"mse={loss_mse.item():.6f}  cws={loss_cws.item():.6f}")

    model.load_state_dict(best_state)
    model.eval()
    print(f"\n  Final val_mse: {best_val:.6f}")
    torch.save(model.state_dict(), 'acrobot_wm_cws.pt')
    print(f"  Saved: acrobot_wm_cws.pt")
    print(f"  Total time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
