"""Generate decision network training data using multi-scale world model.

For each sampled pendulum state, find the best single (a, k) pair to reach upright
via inverse optimization through the frozen multi-scale world model f_ms(s, a, k).

Output: decision_data_ms.pt with (s_norm, s_target_norm, a_norm, k_norm_cont),
where k_norm_cont = k/16 is a continuous value.
"""
import torch, numpy as np, time, argparse
from kanrf import KAN


K_VALUES = [1, 2, 4, 8, 16]


def load_model(path):
    ckpt = torch.load(path, weights_only=True)
    dims = [5]
    for k in sorted(ckpt.keys()):
        if 'base_weight' in k:
            dims.append(ckpt[k].shape[0])
    model = KAN(dims, grid_size=5, spline_order=3)
    model.load_state_dict(ckpt)
    return model.eval()


def sample_states(n, bias_bottom=True):
    """Sample pendulum states. Oversample bottom region (hard cases)."""
    states = []
    for _ in range(n):
        if bias_bottom and np.random.random() < 0.6:
            angle = np.random.uniform(-np.pi, -np.pi / 4)
            thd = np.random.uniform(-6, 6)
        else:
            angle = np.random.uniform(-np.pi, np.pi)
            thd = np.random.uniform(-8, 8)
        states.append([np.cos(angle), np.sin(angle), thd])
    return np.array(states, dtype=np.float32)


def inverse_optimize(model, s_norm, k, n_restarts=2, n_iters=200):
    """Find a* = argmin_a ‖f_ms(s, a, k) - s_target‖².

    k is the number of dt steps. k/16 is fed as the 5th input dimension.
    """
    s_target = torch.tensor([[0., 1., 0.]])  # upright
    k_norm = torch.tensor([[k / 16.0]])

    best_loss, best_a = float('inf'), None

    for _ in range(n_restarts):
        a = torch.empty(1, 1)
        torch.nn.init.uniform_(a, -1, 1)
        a.requires_grad_(True)
        opt = torch.optim.Adam([a], lr=0.05)

        for __ in range(n_iters):
            opt.zero_grad()
            x = torch.cat([s_norm.unsqueeze(0), a, k_norm], dim=-1)  # (1, 5)
            pred = model(x)
            loss = ((pred - s_target) ** 2).sum()
            loss.backward()
            opt.step()
            with torch.no_grad():
                a.clamp_(-1.0, 1.0)

        with torch.no_grad():
            x = torch.cat([s_norm.unsqueeze(0), a, k_norm], dim=-1)
            final_loss = ((model(x) - s_target) ** 2).sum().item()
        if final_loss < best_loss:
            best_loss = final_loss
            best_a = a.detach().clone()

    return best_a.item(), best_loss


def main(model_path='kan_ms.pt', n_samples=500, device='cpu'):
    torch.manual_seed(42)
    np.random.seed(42)

    print(f"Loading world model: {model_path}")
    model = load_model(model_path).to(device)
    for p in model.parameters():
        p.requires_grad = False
    print(f"  params: {sum(p.numel() for p in model.parameters())}")

    states = sample_states(n_samples, bias_bottom=True)
    samples = []
    k_counts = {k: 0 for k in K_VALUES}
    t0 = time.time()

    for i, s_raw in enumerate(states):
        s_norm = torch.tensor(s_raw, dtype=torch.float32)
        s_norm[2] /= 8.0  # normalize theta_dot

        # Try each k, pick best
        best_k, best_a, best_loss = None, None, float('inf')
        for k in K_VALUES:
            a, loss = inverse_optimize(model, s_norm, k)
            if loss < best_loss:
                best_loss = loss
                best_a = a
                best_k = k

        k_counts[best_k] += 1
        s_target_norm = np.array([0., 1., 0.], dtype=np.float32)  # upright

        samples.append({
            's_norm': s_raw.copy(),
            's_target_norm': s_target_norm.copy(),
            'a_norm': best_a,
            'k_norm_cont': best_k / 16.0,
            'k_actual': best_k,
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_samples - i - 1)
            print(f"  [{i+1:4d}/{n_samples}]  k_dist: {k_counts}  "
                  f"[{elapsed:.0f}s ETA {eta:.0f}s]")

    # Save
    out = {
        's_norm': torch.tensor([s['s_norm'] for s in samples]),
        's_target_norm': torch.tensor([s['s_target_norm'] for s in samples]),
        'a_norm': torch.tensor([[s['a_norm']] for s in samples]),
        'k_norm_cont': torch.tensor([[s['k_norm_cont']] for s in samples]),
    }
    torch.save(out, 'decision_data_ms.pt')
    print(f"\nSaved {len(samples)} samples to decision_data_ms.pt")
    print(f"k distribution: {k_counts}")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='kan_ms.pt')
    parser.add_argument('--n-samples', type=int, default=500)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    main(model_path=args.model, n_samples=args.n_samples, device=args.device)
