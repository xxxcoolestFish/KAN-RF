"""Action explorer: try multiple candidate actions from the same state.

Uses env.unwrapped.state save/restore to fairly compare actions.
Returns the best action based on actual environment outcome.
No system-specific knowledge required.
"""
import numpy as np
import torch

PI_2 = np.pi / 2


def angle_error(obs):
    """Angular distance to upright, in radians."""
    err = abs(np.arctan2(obs[1], obs[0]) - PI_2)
    return min(err, 2 * np.pi - err)


class ActionExplorer:
    def __init__(self, n_candidates=5, n_random=3, fixed_k=None):
        self.n_candidates = n_candidates
        self.n_random = n_random
        self.fixed_k = fixed_k  # if set, use this k for all candidates
        self.corrections = []   # (s_norm, a_correct, k_correct, improvement)

    def generate_candidates(self, model_a, model_k):
        """Generate candidate (a, k) pairs to try.  Always includes:
        - The model's suggestion
        - Opposite torque direction
        - N random actions
        All with the same k for fair comparison.
        """
        k = self.fixed_k if self.fixed_k else model_k
        candidates = [(model_a, k), (-model_a, k)]
        for _ in range(self.n_random):
            candidates.append((np.random.uniform(-1.0, 1.0), k))
        return candidates

    def try_candidates(self, env, s_norm, candidates):
        """Try each candidate from the same starting state.  Returns best (a,k).

        Saves and restores env state so all candidates are compared fairly.
        """
        theta, thetadot = env.unwrapped.state
        best_progress = -float('inf')
        best_a, best_k = candidates[0]

        for a_norm, k in candidates:
            # Restore state
            env.unwrapped.state = (theta, thetadot)

            a_raw = a_norm * 2.0
            err_before = angle_error(
                np.array([np.cos(theta), np.sin(theta), thetadot]))
            final_obs = None
            steps_used = 0

            for _ in range(min(k, 12)):
                obs, _, term, trunc, _ = env.step([a_raw])
                final_obs = obs
                steps_used += 1
                if term or trunc:
                    break

            # Progress = reduction in angle error
            err_after = angle_error(final_obs) if final_obs is not None else err_before
            progress = err_before - err_after

            if progress > best_progress:
                best_progress = progress
                best_a = a_norm
                best_k = k

        # Restore and execute best action
        env.unwrapped.state = (theta, thetadot)
        return best_a, best_k, best_progress

    def record_correction(self, s_norm, model_a, model_k, best_a, best_k, progress):
        """Record a corrected (state, action, k) pair if better than model's."""
        arr = s_norm.detach().cpu().numpy().copy().squeeze()  # (3,) not (1,3)
        self.corrections.append({
            's_norm': arr,
            'a_old': model_a,
            'a_old': model_a,
            'k_old': model_k,
            'a_correct': best_a,
            'k_correct': best_k,
            'improvement': progress,
        })

    def get_training_data(self):
        """Return corrected (s, a, k) as tensors for decision network training."""
        if not self.corrections:
            return None
        s_batch = torch.tensor([c['s_norm'] for c in self.corrections],
                               dtype=torch.float32)
        a_batch = torch.tensor([[c['a_correct']] for c in self.corrections],
                               dtype=torch.float32)
        k_batch = torch.tensor([[c['k_correct'] / 16.0] for c in self.corrections],
                               dtype=torch.float32)
        return s_batch, a_batch, k_batch
