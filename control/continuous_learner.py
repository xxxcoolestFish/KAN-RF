"""General continuous learner: no system-specific heuristics.

Mechanism:
  1. Model-based inverse optimization for control
  2. When prediction error exceeds threshold, explore by trying alternative actions
  3. After each episode, fine-tune world model on collected experience
  4. Over episodes, the model improves → control improves → success

The only assumption: we can compute prediction error = ||f_model(s,a,k) - s'_real||
This holds for ANY world model on ANY system.
"""
import torch, numpy as np
import torch.nn.functional as F


class ContinuousLearner:
    def __init__(self, model, lr=1e-4, error_threshold=0.15, noise_scale=0.3):
        self.model = model
        self.lr = lr
        self.error_threshold = error_threshold  # when to trigger exploration
        self.noise_scale = noise_scale            # action noise during exploration
        self.buffer_x = []
        self.buffer_y = []
        self.error_log = []
        self.total_updates = 0

    def observe(self, s_norm, a_norm, k, s_next_norm):
        """Record a transition. Returns prediction error."""
        k_norm = k / 16.0
        x = torch.cat([
            s_norm.detach(),
            torch.tensor([[a_norm]], dtype=torch.float32),
            torch.tensor([[k_norm]], dtype=torch.float32),
        ], dim=-1)

        with torch.no_grad():
            pred = self.model(x)
            error = (pred - s_next_norm).norm().item()

        self.error_log.append(error)
        self.buffer_x.append(x)
        self.buffer_y.append(s_next_norm.detach())
        return error

    def is_surprised(self):
        """Check if recent prediction errors exceed threshold."""
        if len(self.error_log) < 3:
            return False
        recent = self.error_log[-3:]
        return np.mean(recent) > self.error_threshold

    def explore_action(self, a_norm):
        """Generate alternative actions via noise injection (general mechanism)."""
        # Try: opposite sign, random magnitude, noise around current
        candidates = [
            -a_norm,                                    # opposite direction
            a_norm + np.random.uniform(-0.5, 0.5),      # noisy current
            np.random.uniform(-1.0, 1.0),               # random
            a_norm * np.random.uniform(0.3, 1.7),       # scaled current
        ]
        return [max(-1.0, min(1.0, c)) for c in candidates]

    def fine_tune(self, epochs=20, verbose=False, replay_ratio=0.3):
        """Batch fine-tune.  If replay_buffer is set, mixes old data to prevent forgetting."""
        if len(self.buffer_x) < 4:
            return 0.0

        xs = torch.cat(self.buffer_x, dim=0)
        ys = torch.cat(self.buffer_y, dim=0)

        # Mix in replay data to prevent catastrophic forgetting
        if hasattr(self, 'replay_x') and len(self.replay_x) > 0:
            n_replay = min(int(len(xs) * replay_ratio), len(self.replay_x))
            idx = torch.randperm(len(self.replay_x))[:n_replay]
            xs = torch.cat([xs, self.replay_x[idx]], dim=0)
            ys = torch.cat([ys, self.replay_y[idx]], dim=0)

        self.model.train()
        for layer in self.model.layers:
            layer.base_weight.requires_grad = False
            layer.spline_weight.requires_grad = True

        opt = torch.optim.Adam(
            [layer.spline_weight for layer in self.model.layers], lr=self.lr)

        final_loss = 0.0
        for _ in range(epochs):
            opt.zero_grad()
            preds = self.model(xs)
            loss = F.mse_loss(preds, ys)
            if torch.isnan(loss):
                break
            loss.backward()
            opt.step()
            final_loss = loss.item()

        self.model.eval()
        for layer in self.model.layers:
            layer.base_weight.requires_grad = True

        self.total_updates += epochs

        if verbose:
            print(f"  [fine_tune] {len(self.buffer_x)} samples × {epochs} epochs  "
                  f"loss={final_loss:.5f}")

        self.buffer_x, self.buffer_y = [], []
        return final_loss

    def summary(self):
        return {
            'buffer_size': len(self.buffer_x),
            'total_updates': self.total_updates,
            'mean_error': (sum(self.error_log) / len(self.error_log)
                          if self.error_log else 0),
            'max_error': max(self.error_log) if self.error_log else 0,
        }
