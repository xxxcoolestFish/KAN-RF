"""Temporal Hierarchy Target Propagation (THTP).

Auto-discovers temporal hierarchy from ProtoKAN WM Jacobian,
then back-propagates control targets through the hierarchy layer by layer.

Core insight: all physical control systems share a universal structure:
  Tier 0 (instant):  velocities — directly controllable by action
  Tier 1 (1-step):   positions — reachable from Tier 0 via integration
  Tier 2 (2-step):   observations — reachable from Tier 1

This structure is discovered purely from WM Jacobian data, requiring
no hand-crafted physics knowledge.
"""
import torch, torch.nn as nn, numpy as np


# ═══════════════════════════════════════════════════════════
# 1. Hierarchy Discovery
# ═══════════════════════════════════════════════════════════

class TemporalHierarchy:
    """Auto-discovered temporal hierarchy from WM Jacobian.

    Attributes:
        tiers: list of lists, tiers[k] = indices of state dims in tier k
        controllability: (state_dim,) mean |∂s'[i]/∂a|
        transfer: (state_dim, state_dim) mean |∂s'[i]/∂s[j]|
        tier_of: (state_dim,) which tier each dimension belongs to
    """

    def __init__(self, wm, state_dim, n_samples=300, device='cpu'):
        self.state_dim = state_dim
        self.device = device

        # Compute Jacobians
        Ja_mag, Js_mag = self._compute_jacobians(wm, state_dim, n_samples, device)

        self.controllability = Ja_mag      # (state_dim,)
        self.transfer = Js_mag              # (state_dim, state_dim)

        # Auto-discover tiers
        self.tiers = self._discover_tiers(Ja_mag, Js_mag)
        self.tier_of = self._assign_tiers()

    def _compute_jacobians(self, wm, state_dim, n_samples, device):
        """Compute average Jacobian magnitudes over random states."""
        wm.eval()
        Ja_acc = torch.zeros(state_dim, device=device)
        Js_acc = torch.zeros(state_dim, state_dim, device=device)

        for _ in range(n_samples):
            s = torch.randn(1, state_dim, device=device) * 0.5
            s = s.clamp(-1, 1)

            # ∂s'/∂a
            a = torch.zeros(1, 1, device=device, requires_grad=True)
            s_pred = wm(torch.cat([s, a], dim=-1))
            for i in range(state_dim):
                g = torch.autograd.grad(s_pred[0, i], a, retain_graph=True)[0]
                Ja_acc[i] += g[0, 0].abs()

            # ∂s'/∂s — state-to-state transfer
            s_j = s.clone().detach().requires_grad_(True)
            a_fixed = torch.zeros(1, 1, device=device)
            s_pred = wm(torch.cat([s_j, a_fixed], dim=-1))
            for j in range(state_dim):
                for i in range(state_dim):
                    g = torch.autograd.grad(s_pred[0, i], s_j, retain_graph=True)[0]
                    Js_acc[i, j] += g[0, j].abs()

        Ja_mag = (Ja_acc / n_samples).cpu().numpy()
        Js_mag = (Js_acc / n_samples).cpu().numpy()
        return Ja_mag, Js_mag

    def _discover_tiers(self, Ja_mag, Js_mag):
        """Auto-discover temporal tiers from Jacobian data.

        Tier 0: directly controllable (high |∂s'/∂a|)
        Tier k+1: reachable from Tier k via strong state transition
        """
        n = self.state_dim
        assigned = set()
        tiers = []

        # Tier 0: top 40% most controllable dimensions
        threshold = np.percentile(Ja_mag, 60)
        tier0 = [i for i in range(n) if Ja_mag[i] >= threshold]
        tiers.append(tier0)
        assigned.update(tier0)

        # Tier 1+: dimensions reachable from previous tier
        max_tiers = 4  # safety limit
        for _ in range(max_tiers):
            if len(assigned) >= n:
                break
            prev_tier = tiers[-1]
            candidates = []
            for i in range(n):
                if i in assigned:
                    continue
                # Can this dimension be reached from any prev-tier dimension?
                max_transfer = max(Js_mag[i, j] for j in prev_tier)
                candidates.append((i, max_transfer))

            # Take dimensions with strong transfer from previous tier
            if candidates:
                transfers = [c[1] for c in candidates]
                t = np.percentile(transfers, 50)  # top 50% of remaining
                new_tier = [c[0] for c in candidates if c[1] >= t]
                if new_tier:
                    tiers.append(new_tier)
                    assigned.update(new_tier)
                else:
                    break
            else:
                break

        # Any remaining dimensions → last tier
        remaining = [i for i in range(n) if i not in assigned]
        if remaining:
            tiers.append(remaining)

        return tiers

    def _assign_tiers(self):
        """Build tier_of mapping."""
        tier_of = np.zeros(self.state_dim, dtype=int)
        for k, tier in enumerate(self.tiers):
            for i in tier:
                tier_of[i] = k
        return tier_of

    def summary(self, state_names=None):
        """Print hierarchy summary."""
        if state_names is None:
            state_names = [f'dim{i}' for i in range(self.state_dim)]
        lines = []
        for k, tier in enumerate(self.tiers):
            dims = [f"{state_names[i]}(ctrl={self.controllability[i]:.3f})"
                    for i in tier]
            lines.append(f"  Tier {k} (→{len(self.tiers)-k-1} steps to target): {', '.join(dims)}")
        lines.append(f"  Transfer matrix diagonal: "
                     f"{[f'{self.transfer[i,i]:.3f}' for i in range(self.state_dim)]}")
        return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════
# 2. Target Propagation
# ═══════════════════════════════════════════════════════════

class TargetPropagation:
    """Back-propagate control targets through the temporal hierarchy.

    Given (s_current, s_target), computes sub-goals for each tier
    and the final action a_des by solving layer-by-layer.

    Uses damped pseudo-inverse for numerical stability.
    """

    def __init__(self, wm, hierarchy: TemporalHierarchy, alpha=0.3, damping=0.1,
                 device='cpu'):
        self.wm = wm
        self.h = hierarchy
        self.alpha = alpha        # step size for sub-goal updates
        self.damping = damping    # damping for pseudo-inverse
        self.device = device

    def propagate(self, s_current, s_target):
        """Compute a_des and intermediate subgoals.

        Args:
            s_current: (state_dim,) current normalized state
            s_target: (state_dim,) target normalized state

        Returns:
            a_des: scalar action
            subgoals: list of (state_dim,) tensors, one per tier
            diagnostics: dict with per-tier errors and Jacobians
        """
        wm = self.wm
        wm.eval()

        # Temporarily enable gradients for Jacobian computation
        was_frozen = not next(wm.parameters()).requires_grad
        if was_frozen:
            for p in wm.parameters():
                p.requires_grad = True

        state_dim = self.h.state_dim
        s_cur = s_current.clone().detach()
        s_tgt = s_target.clone().detach()

        diagnostics = {}

        # Pre-compute full Jacobians at current state
        # ∂s'/∂a
        a = torch.zeros(1, 1, device=self.device, requires_grad=True)
        s_pred = wm(torch.cat([s_cur.unsqueeze(0), a], dim=-1))
        Ja = torch.zeros(state_dim, device=self.device)
        for i in range(state_dim):
            g = torch.autograd.grad(s_pred[0, i], a, retain_graph=True)[0]
            Ja[i] = g[0, 0]

        # ∂s'/∂s
        s_j = s_cur.unsqueeze(0).clone().detach().requires_grad_(True)
        a_fixed = torch.zeros(1, 1, device=self.device)
        s_pred = wm(torch.cat([s_j, a_fixed], dim=-1))
        Js = torch.zeros(state_dim, state_dim, device=self.device)
        for j in range(state_dim):
            for i in range(state_dim):
                g = torch.autograd.grad(s_pred[0, i], s_j, retain_graph=True)[0]
                Js[i, j] = g[0, j]

        if was_frozen:
            for p in wm.parameters():
                p.requires_grad = False

        diagnostics['Ja'] = Ja.cpu().numpy()
        diagnostics['Js'] = Js.cpu().numpy()

        # Work backward through tiers
        # Start: compute error at deepest tier
        s_des = s_tgt.clone()  # desired state — will be modified tier by tier
        final_subgoals = [s_des.clone()]

        for tier_idx in range(len(self.h.tiers) - 1, 0, -1):
            # Tier k (deeper) and Tier k-1 (shallower)
            tier_deep = self.h.tiers[tier_idx]
            tier_shallow = self.h.tiers[tier_idx - 1]

            # Error at this tier
            error_deep = s_des[tier_deep] - s_cur[tier_deep]

            # Jacobian from shallow tier to deep tier
            # J_shallow→deep: (|tier_deep|, |tier_shallow|)
            J_sd = Js[tier_deep][:, tier_shallow]

            # Damped pseudo-inverse: (J^T J + λI)^{-1} J^T
            JJt = J_sd @ J_sd.T
            n_deep = len(tier_deep)
            damped = JJt + self.damping * torch.eye(n_deep, device=self.device)

            # Solve for Δs_shallow
            try:
                delta_shallow = torch.linalg.solve(damped, error_deep)
                delta_shallow = J_sd.T @ delta_shallow  # (|tier_shallow|,)
            except:
                # Fallback: scaled gradient step
                delta_shallow = J_sd.T @ error_deep * 0.1

            # Update desired state for shallow tier
            s_des[tier_shallow] = s_cur[tier_shallow] + self.alpha * delta_shallow
            s_des[tier_shallow] = s_des[tier_shallow].clamp(-1, 1)

            final_subgoals.append(s_des.clone())
            diagnostics[f'delta_tier{tier_idx-1}'] = delta_shallow.cpu().numpy()

        # Final tier (Tier 0): map to action
        tier0 = self.h.tiers[0]
        error_0 = s_des[tier0] - s_cur[tier0]
        J_a0 = Ja[tier0]  # (|tier0|,)

        # Damped 1D pseudo-inverse per dimension
        a_updates = []
        for i, idx in enumerate(tier0):
            j_sq = J_a0[i] ** 2 + self.damping
            a_updates.append((J_a0[i] / j_sq) * error_0[i])
        a_des = torch.stack(a_updates).sum().clamp(-2, 2)

        diagnostics['a_des'] = a_des.item()
        diagnostics['error_tier0'] = error_0.cpu().numpy()

        return a_des, [s_des.clone()] + final_subgoals, diagnostics


# ═══════════════════════════════════════════════════════════
# 3. Hierarchical Policy Training
# ═══════════════════════════════════════════════════════════

class THTPTrainer:
    """Train Policy via THTP: pre-compute routing targets, then distill.

    Two-phase training:
    Phase 1: Pre-compute (s, a_des) pairs using propagate() — slow, done once
    Phase 2: Policy learns to match a_des via MSE — fast, standard training
    """

    def __init__(self, wm, policy, hierarchy, thtp, s_dataset, s_target,
                 lr=1e-3, n_distill=500, device='cpu'):
        self.wm = wm
        self.policy = policy.to(device)
        self.h = hierarchy
        self.thtp = thtp
        self.s_target = s_target.to(device)
        self.device = device

        wm.eval()
        for p in wm.parameters():
            p.requires_grad = False

        # Phase 1: Pre-compute distillation targets
        print(f"  Pre-computing THTP routing targets ({n_distill} states)...")
        N = s_dataset.shape[0]
        idx = torch.randperm(N)[:n_distill]
        self.distill_s = s_dataset[idx].to(device)
        self.distill_a = torch.zeros(n_distill, 1, device=device)
        for i in range(n_distill):
            a_des, _, _ = self.thtp.propagate(
                self.distill_s[i], self.s_target.squeeze(0))
            self.distill_a[i, 0] = a_des
        print(f"  Done. Distillation targets ready.")

        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.loss_history = []

    def train_epoch(self, s_dataset, batch_size=256):
        N = s_dataset.shape[0]
        n_batches = max(1, N // batch_size)
        total_loss = 0.0

        for _ in range(n_batches):
            self.policy.train()
            self.opt.zero_grad()

            # Mix: real WM gradient loss + distillation from THTP
            # 1) WM gradient path (energy-guided or stabilization)
            idx_wm = torch.randint(0, N, (batch_size // 2,), device=self.device)
            s_wm = s_dataset[idx_wm]
            a_wm = self.policy(s_wm)
            s_pred = self.wm(torch.cat([s_wm, a_wm], dim=-1))

            if s_wm.shape[1] == 3:  # Pendulum
                thd = s_wm[:, 2] * 8.0
                Ec = 0.5 * thd.pow(2) + 10.0 * s_wm[:, 1]
                thdp = s_pred[:, 2] * 8.0
                Ep = 0.5 * thdp.pow(2) + 10.0 * s_pred[:, 1]
                deficit = (10.0 - Ec).detach()
                egain = (Ep - Ec) * torch.sign(deficit)
                sin = s_wm[:, 1]
                ws = ((1.0 + sin) / 2.0).clamp(0, 1)
                wm_loss = (-egain.mean() +
                           (ws * (s_pred - self.s_target.expand(batch_size//2, -1))
                            .pow(2).sum(-1)).mean() +
                           0.01 * a_wm.pow(2).mean())
            else:  # CartPole
                s_target = torch.zeros(1, s_wm.shape[1], device=self.device)
                wm_loss = (s_pred[:, 2].pow(2).mean() +
                           0.1 * s_pred[:, 0].pow(2).mean() +
                           0.5 * s_pred[:, 3].pow(2).mean() +
                           0.01 * a_wm.pow(2).mean())

            # 2) Distillation: match pre-computed THTP actions
            idx_d = torch.randint(0, len(self.distill_s), (batch_size // 2,),
                                  device=self.device)
            s_d = self.distill_s[idx_d]
            a_d = self.distill_a[idx_d]
            a_pol = self.policy(s_d)
            distill_loss = (a_pol - a_d).pow(2).mean()

            loss = wm_loss + 0.3 * distill_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
            self.opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / n_batches
        self.loss_history.append({'total': avg_loss})
        return self.loss_history[-1]

    def get_action(self, s):
        self.policy.eval()
        if isinstance(s, np.ndarray):
            s = torch.tensor(s, dtype=torch.float32, device=self.device)
        if s.dim() == 1: s = s.unsqueeze(0)
        with torch.no_grad():
            return self.policy(s).squeeze().cpu().item()
