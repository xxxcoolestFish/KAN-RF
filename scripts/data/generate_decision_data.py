"""Generate (s, a*, H*) training data for decision network via shooting.

Shoots through the trained KAN world model to find optimal action sequences,
then distills them into single-step (first_action, horizon) labels.
"""
import torch, numpy as np, time, argparse
from kanrf import KAN
from control.shoot import shoot


def load_model(path):
    ckpt = torch.load(path, weights_only=True)
    dims = [4]
    for k in sorted(ckpt.keys()):
        if 'base_weight' in k:
            dims.append(ckpt[k].shape[0])
    model = KAN(dims, grid_size=5, spline_order=3)
    model.load_state_dict(ckpt)
    return model.eval()


def sample_states(n, bias_bottom=True):
    """Sample pendulum states. If bias_bottom, oversample bottom region."""
    states = []
    for _ in range(n):
        if bias_bottom and np.random.random() < 0.6:
            # Bottom region: angle near -pi/2 (hanging down)
            angle = np.random.uniform(-np.pi, -np.pi/4)
            thd = np.random.uniform(-6, 6)
        else:
            angle = np.random.uniform(-np.pi, np.pi)
            thd = np.random.uniform(-8, 8)
        cos, sin = np.cos(angle), np.sin(angle)
        states.append([cos, sin, thd])
    return np.array(states, dtype=np.float32)


def find_horizon(model, s0_raw, actions_raw, max_h, angle_thresh=0.2):
    """Find first step t where predicted state is within angle_thresh of target."""
    s0_norm = s0_raw.clone()
    s0_norm[:, 2] /= 8.0
    s_target_norm = torch.tensor([[0., 1., 0.]])

    with torch.no_grad():
        s = s0_norm
        for t in range(max_h):
            a_norm = actions_raw[t:t+1] / 2.0  # (1, 1)
            x = torch.cat([s, a_norm], dim=-1)
            s = model(x)
            nrm = s[:, :2].norm(dim=-1, keepdim=True).clamp(min=1e-6)
            s = torch.cat([s[:, :2] / nrm, s[:, 2:]], dim=-1)

            # Angle error from upright
            cos_err = (s[:, :2] * s_target_norm[:, :2]).sum(-1).clamp(-1, 1)
            angle_err = torch.acos(cos_err).item()
            if angle_err < angle_thresh:
                return max(t + 1, 3)  # minimum 3 steps
    return max_h


def main(model_path='kan_hybrid_lam0.1_nu0.1.pt', n_samples=300, horizon=25, device='cpu'):
    torch.manual_seed(42); np.random.seed(42)

    print(f"Loading model: {model_path}")
    model = load_model(model_path).to(device)
    for p in model.parameters():
        p.requires_grad = False

    states = sample_states(n_samples, bias_bottom=True)
    samples = []
    t0 = time.time()

    for i, s_raw in enumerate(states):
        s0 = torch.tensor(s_raw, dtype=torch.float32).unsqueeze(0)
        s_target = torch.tensor([[0., 1., 0.]])  # upright

        try:
            actions_raw, final_state = shoot(
                model, s0, s_target, horizon=horizon,
                n_iters=200, lr=0.1, lambda_ctrl=0.01,
                n_restarts=1, verbose=False)
        except Exception:
            continue

        # Determine H: first step reaching near-upright
        H = find_horizon(model, s0, actions_raw, horizon)

        a_first = actions_raw[0].item() / 2.0  # normalize to [-1, 1]

        # Normalized state
        s_norm = s_raw.copy()
        s_norm[2] /= 8.0

        samples.append({
            's_norm': s_norm,          # (3,) normalized
            's_target_norm': np.array([0., 1., 0.], dtype=np.float32),  # (3,)
            'a_norm': a_first,         # scalar, normalized
            'H_class': H - 1,          # 0-indexed for CrossEntropy
        })

        if (i + 1) % 30 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_samples - i - 1)
            print(f"  [{i+1:4d}/{n_samples}]  H_dist: {np.mean([s['H_class']+1 for s in samples]):.1f}  "
                  f"[{elapsed:.0f}s ETA {eta:.0f}s]")

    # Save
    out = {
        's_norm': torch.tensor([s['s_norm'] for s in samples]),
        's_target_norm': torch.tensor([s['s_target_norm'] for s in samples]),
        'a_norm': torch.tensor([s['a_norm'] for s in samples]).unsqueeze(-1),
        'H_class': torch.tensor([s['H_class'] for s in samples], dtype=torch.long),
    }
    torch.save(out, 'decision_data.pt')
    print(f"\nSaved {len(samples)} samples to decision_data.pt")
    print(f"H distribution: mean={out['H_class'].float().mean().item()+1:.1f}, "
          f"min={out['H_class'].min().item()+1}, max={out['H_class'].max().item()+1}")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='kan_hybrid_lam0.1_nu0.1.pt')
    parser.add_argument('--n-samples', type=int, default=300)
    parser.add_argument('--horizon', type=int, default=25)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    main(model_path=args.model, n_samples=args.n_samples, horizon=args.horizon, device=args.device)
