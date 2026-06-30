"""Train KAN-adapted decision network (v3).

The policy π_θ is trained using a frozen KAN world model as a differentiable
loss function.  KAN provides gradient signals through its accurate Jacobian,
telling π_θ "which direction should the action move to get closer to the goal?"

Usage:
  python train.py --wm ../path/to/kan_model.pt --epochs 200
  python train.py --wm ../path/to/kan_model.pt --multi-step 3 --epochs 100
"""
import sys, os, argparse, time
import torch
import numpy as np
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import KAN
from decision_v3.core import (KANPolicy, ResidualPhysicsPolicy,
                               KANGradientTrainer, KANEnergyTrainer,
                               KANDensityWeight, KANMultiStepTrainer)

PI_2 = np.pi / 2
S_TARGET = torch.tensor([[0.0, 1.0, 0.0]])  # upright, at rest


def generate_training_states(n_samples=20000, method='uniform', device='cpu'):
    """Generate training states that cover the pendulum state space.

    Args:
        n_samples: number of states to generate
        method:
          'uniform' — uniformly sample cos, sin, thd ranges
          'random_rollouts' — collect states from random action trajectories
          'mixed' — half uniform, half from rollouts

    Returns:
        s_dataset: (N, 3) normalized states [cos, sin, thd/8]
    """
    if method == 'uniform':
        # Uniform sampling in the valid state space
        # cos ∈ [-1, 1], sin ∈ [-1, 1], but cos²+sin²=1 constraint
        angles = np.random.uniform(-np.pi, np.pi, n_samples)
        cos = np.cos(angles)
        sin = np.sin(angles)
        thd = np.random.uniform(-8.0, 8.0, n_samples)  # raw angular velocity
        s_raw = np.stack([cos, sin, thd], axis=1)

    elif method == 'random_rollouts':
        env = gym.make("Pendulum-v1")
        states = []
        while len(states) < n_samples:
            obs, _ = env.reset(seed=np.random.randint(0, 100000))
            for _ in range(200):
                a = env.action_space.sample()
                obs, _, term, trunc, _ = env.step(a)
                states.append(obs.copy())
                if term or trunc:
                    break
                if len(states) >= n_samples:
                    break
        env.close()
        s_raw = np.array(states[:n_samples])

    elif method == 'mixed':
        n_each = n_samples // 2
        uniform = generate_training_states(n_each, 'uniform', device)
        rollout = generate_training_states(n_each, 'random_rollouts', device)
        s_raw = np.concatenate([uniform, rollout], axis=0)
        np.random.shuffle(s_raw)
        s_raw = s_raw[:n_samples]

    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize: [cos, sin, thd/8]
    s_norm = s_raw.copy()
    s_norm[:, 2] /= 8.0

    return torch.tensor(s_norm, dtype=torch.float32, device=device)


def load_kan(model_path, layer_dims, device='cpu'):
    """Load a trained KAN world model."""
    kan = KAN(layer_dims, grid_size=5, spline_order=3)
    kan.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    kan.to(device)
    kan.eval()
    for p in kan.parameters():
        p.requires_grad = False
    return kan


def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── 1. Load frozen KAN world model ──
    print(f"Loading KAN world model: {args.wm}")
    kan = load_kan(args.wm, args.kan_dims, device)
    n_params_kan = sum(p.numel() for p in kan.parameters())
    print(f"  KAN architecture: {args.kan_dims}, params: {n_params_kan}")

    # ── 2. Generate / load training states ──
    print(f"Generating training states ({args.n_samples}, method={args.data_method})")
    s_dataset = generate_training_states(args.n_samples, args.data_method, device)
    print(f"  Dataset: {s_dataset.shape}")

    # ── 3. Create policy network ──
    # Detect multi-scale: KAN input dim > state_dim + action_dim
    state_dim = 3  # pendulum: cos, sin, thd/8
    action_dim = 1
    kan_in_dim = args.kan_dims[0]
    ms_mode = args.multi_scale  # 'off', 'fixed', 'policy'

    if kan_in_dim == state_dim + action_dim + 1:
        # Multi-scale: [s(3), a(1), k(1)] → auto-enable
        if ms_mode == 'off':
            ms_mode = 'fixed'  # default to fixed k
            print(f"  Auto-detected multi-scale KAN (in_dim={kan_in_dim})")

    output_k = (ms_mode == 'policy')
    policy_type = args.policy_type

    if policy_type == 'residual':
        policy = ResidualPhysicsPolicy(
            state_dim=state_dim, action_dim=action_dim,
            hidden=args.hidden, n_layers=args.n_layers,
            init_k_energy=args.k_energy_init,
            init_k_stable=args.k_stable_init,
            init_k_damp=args.k_damp_init)
        print(f"  Policy: ResidualPhysics (energy shaping + MLP residual)")
        print(f"    Physics params: k_energy={args.k_energy_init}, "
              f"k_stable={args.k_stable_init}, k_damp={args.k_damp_init}")
        print(f"    Residual: [{state_dim}, {args.hidden}×{args.n_layers}, {action_dim}]")
    else:
        policy = KANPolicy(state_dim=state_dim, action_dim=action_dim,
                           hidden=args.hidden, n_layers=args.n_layers,
                           output_k=output_k)
        out_desc = '(a, k)' if output_k else 'a'
        print(f"  Policy: MLP [{state_dim}, {args.hidden}×{args.n_layers}, {out_desc}]")

    n_params_policy = sum(p.numel() for p in policy.parameters())
    n_physics = 3 if policy_type == 'residual' else 0
    n_residual = n_params_policy - n_physics
    if policy_type == 'residual':
        print(f"    Total params: {n_params_policy} (physics: {n_physics}, residual: {n_residual})")
    else:
        print(f"    Total params: {n_params_policy}")

    # ── 4. Set up trainer ──
    s_target = S_TARGET.to(device)
    k_norm = None
    if ms_mode == 'fixed':
        k_val = args.k_fixed
        k_norm = torch.tensor([[k_val / 16.0]], device=device)
        print(f"  Multi-scale: fixed k={k_val} (k_norm={k_val/16:.4f})")

    loss_type = args.loss_type
    trainer_kwargs = dict(
        kan=kan, policy=policy, s_target=s_target,
        lr=args.lr, lambda_ctrl=args.lambda_ctrl,
        clip_grad=args.clip_grad, device=device,
    )
    if ms_mode in ('fixed', 'policy'):
        trainer_kwargs['multi_scale'] = ms_mode
        if ms_mode == 'fixed':
            trainer_kwargs['k_norm'] = k_norm

    if loss_type == 'energy':
        trainer = KANEnergyTrainer(**trainer_kwargs)
        print(f"  Trainer: Energy-based (swing: maximize energy gain, stabilize: minimize distance)")
    elif args.multi_step > 1:
        trainer_kwargs['horizon'] = args.multi_step
        trainer = KANMultiStepTrainer(**trainer_kwargs)
        print(f"  Trainer: MultiStep (H={args.multi_step})")
    else:
        trainer = KANGradientTrainer(**trainer_kwargs)
        print(f"  Trainer: SingleStep MSE(s_pred, s*)")

    # ── 5. Optional: activation density weighting ──
    density_weight = None
    if args.use_density_weight:
        density_weight = KANDensityWeight(kan, device)
        print(f"  Sample weighting: B-spline activation density")

    # ── 6. Training loop ──
    print(f"\nTraining ({args.epochs} epochs, batch_size={args.batch_size})")
    print(f"{'='*60}")

    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        weight_fn = density_weight.as_weights if density_weight else None
        losses = trainer.train_epoch(s_dataset,
                                     batch_size=args.batch_size,
                                     n_batches=args.batches_per_epoch,
                                     weight_fn=weight_fn)

        if epoch % args.report_every == 0 or epoch == 1:
            elapsed = time.time() - t_start
            print(f"  Epoch {epoch:4d}/{args.epochs}  "
                  f"loss={losses['total']:.6f}  "
                  f"pred={losses['pred']:.6f}  "
                  f"ctrl={losses['ctrl']:.6f}  "
                  f"[{elapsed:.0f}s]")

    t_total = time.time() - t_start
    print(f"\nTraining complete: {t_total:.0f}s ({t_total/args.epochs:.1f}s/epoch)")

    # ── 7. Save ──
    save_path = args.output
    ckpt = {
        'policy_state_dict': policy.state_dict(),
        'policy_type': policy_type,
        'kan_dims': args.kan_dims,
        'state_dim': state_dim,
        'hidden': args.hidden,
        'n_layers': args.n_layers,
        'multi_step': args.multi_step,
        'multi_scale': ms_mode,
        'output_k': output_k,
        'k_fixed': args.k_fixed if ms_mode == 'fixed' else None,
        'loss_history': trainer.loss_history,
    }
    if policy_type == 'residual':
        ckpt['k_energy_init'] = args.k_energy_init
        ckpt['k_stable_init'] = args.k_stable_init
        ckpt['k_damp_init'] = args.k_damp_init
    torch.save(ckpt, save_path)
    print(f"Saved: {save_path}")

    return trainer


def main():
    parser = argparse.ArgumentParser(description='Train KAN-adapted decision network (v3)')
    parser.add_argument('--wm', type=str, required=True,
                        help='Path to trained KAN world model')
    parser.add_argument('--kan-dims', type=int, nargs='+', default=[4, 12, 3],
                        help='KAN layer dimensions (input includes action)')
    parser.add_argument('--output', type=str, default='kan_policy_v3.pt')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--batches-per-epoch', type=int, default=40)
    parser.add_argument('--n-samples', type=int, default=20000)
    parser.add_argument('--data-method', type=str, default='mixed',
                        choices=['uniform', 'random_rollouts', 'mixed'])
    parser.add_argument('--hidden', type=int, default=32,
                        help='Hidden size for residual MLP (smaller for residual policy)')
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--policy-type', type=str, default='residual',
                        choices=['mlp', 'residual'],
                        help="'residual': physics prior + learned residual. 'mlp': pure MLP.")
    parser.add_argument('--k-energy-init', type=float, default=0.15,
                        help='Initial energy shaping gain (residual policy)')
    parser.add_argument('--k-stable-init', type=float, default=-2.0,
                        help='Initial stabilization gain (residual policy)')
    parser.add_argument('--k-damp-init', type=float, default=-0.3,
                        help='Initial damping gain (residual policy)')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lambda-ctrl', type=float, default=0.01)
    parser.add_argument('--clip-grad', type=float, default=10.0)
    parser.add_argument('--multi-step', type=int, default=1,
                        help='Multi-step rollout horizon (1 = single-step)')
    parser.add_argument('--loss-type', type=str, default='energy',
                        choices=['mse', 'energy'],
                        help="'energy': physics-informed energy gain + distance blend. 'mse': pure MSE(s_pred, s*)")
    parser.add_argument('--multi-scale', type=str, default='off',
                        choices=['off', 'fixed', 'policy'],
                        help="'fixed': fixed k for all states. 'policy': policy outputs (a,k)")
    parser.add_argument('--k-fixed', type=int, default=4,
                        help='Fixed k value when --multi-scale=fixed')
    parser.add_argument('--use-density-weight', action='store_true',
                        help='Weight samples by KAN activation density')
    parser.add_argument('--report-every', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
