"""KAN Online Learning: approaches C (Feedback Alignment) and B (full BP).

Usage:
  from online_learning import online_update_feedback_alignment, online_update_sgd

  # After each env step:
  online_update_feedback_alignment(model, s, a, s_true, lr=1e-4)
  # or:
  online_update_sgd(model, s, a, s_true, lr=1e-5)
"""
import torch
import torch.nn.functional as F
from kanrf import KAN
from kanrf import KANLayer


# ─── Approach C: Feedback Alignment ───────────────────────────────────

class FeedbackAlignmentUpdater:
    """Online learning via Feedback Alignment (Lillicrap et al. 2016).

    Replaces backprop through hidden layers with fixed random feedback
    matrices B.  Each layer's pseudo-gradient is B · e (not ∂L/∂h).
    This is local, biologically motivated, and exploits B-spline locality.

    For our 2-layer KAN [4→12→3]:
      Forward:  h = Layer1(x)   →   y = Layer2(h)
      Error:    e = y - y_true  (3D)
      Layer2:   standard SGD gradient ∂L/∂θ₂ = ∂L/∂y · ∂y/∂θ₂
      Layer1:   pseudo-gradient ∂L/∂h ≈ B^T · e, where B is fixed random (3×12)
                then ∂L/∂θ₁ = ∂L/∂h · ∂h/∂θ₁ (using pseudo-gradient)
    """

    def __init__(self, model: KAN, lr: float = 1e-4):
        self.model = model
        self.lr = lr
        # Fixed random feedback matrix: maps output error → hidden pseudo-gradient
        out_dim = model.layers[-1].out_dim       # 3
        hidden_dim = model.layers[-1].in_dim     # 12
        self.B = torch.randn(out_dim, hidden_dim) * 0.1
        # Normalize B rows for stability
        self.B = self.B / (self.B.norm(dim=1, keepdim=True) + 1e-8)

    def update(self, s_norm, a_norm, s_true_norm):
        """One online update step.

        Args:
            s_norm: (1, 3) normalized state [cos, sin, thd/8]
            a_norm: (1, 1) normalized action [torque/2]
            s_true_norm: (1, 3) normalized true next state
        """
        x = torch.cat([s_norm, a_norm], dim=-1)  # (1, 4)

        # ── Forward pass (with activations for B-spline locality) ──
        # Layer 1: hidden
        h, B1, E1 = self.model.layers[0](x, return_activations=True)  # (1, 12)
        # Layer 2: output
        y, B2, E2 = self.model.layers[1](h, return_activations=True)  # (1, 3)

        # ── Error ──
        e = y - s_true_norm  # (1, 3)

        # ── Layer 2 update: standard SGD ──
        grad_w2, grad_c2 = self._layer_gradients(
            self.model.layers[1], h, e, B2)
        for p, g in [(self.model.layers[1].base_weight, grad_w2),
                      (self.model.layers[1].spline_weight, grad_c2)]:
            p.data -= self.lr * g

        # ── Layer 1 update: Feedback Alignment pseudo-gradient ──
        # Pseudo-gradient for hidden: δ_h = B^T · e  (not ∂L/∂h)
        delta_h = (e @ self.B)  # (1, 12) = (1, 3) @ (3, 12)
        grad_w1, grad_c1 = self._layer_gradients(
            self.model.layers[0], x, delta_h, B1)
        for p, g in [(self.model.layers[0].base_weight, grad_w1),
                      (self.model.layers[0].spline_weight, grad_c1)]:
            p.data -= self.lr * g

    def _layer_gradients(self, layer, inputs, output_grad, B_vals):
        """Compute gradients for one KANLayer.

        Forward: y_i = Σ_j (w_{i,j}·silu(x_j) + Σ_k c_{i,j,k}·B_k(x_j))
        Gradients:
          ∂L/∂w_{i,j} = δ_i · silu(x_j)
          ∂L/∂c_{i,j,k} = δ_i · B_k(x_j)

        Args:
            layer: KANLayer
            inputs: (1, in_dim) input to this layer
            output_grad: (1, out_dim) gradient of loss w.r.t. layer output
            B_vals: (1, in_dim, n_basis) B-spline basis values
        Returns:
            grad_w: (out_dim, in_dim)
            grad_c: (out_dim, in_dim, n_basis)
        """
        delta = output_grad  # (1, out_dim)
        x_in = inputs        # (1, in_dim)

        # grad_w_{i,j} = δ_i · silu(x_j)
        silu_x = F.silu(x_in)  # (1, in_dim)
        grad_w = delta.T @ silu_x  # (out_dim, 1) @ (1, in_dim) = (out_dim, in_dim)

        # grad_c_{i,j,k} = δ_i · B_k(x_j)
        # delta: (1, out_dim), B_vals: (1, in_dim, n_basis)
        grad_c = torch.einsum('bo,bjk->ojk', delta, B_vals)

        return grad_w, grad_c


# ─── Approach B: Full Backpropagation ──────────────────────────────────

def online_update_sgd(model: KAN, s_norm, a_norm, s_true_norm, lr: float = 1e-5):
    """One online SGD step through the full network (standard backprop).

    This is the simplest approach: compute loss, backprop through all
    layers, update all parameters.  Very small lr prevents catastrophic
    forgetting of pretrained weights.
    """
    x = torch.cat([s_norm, a_norm], dim=-1)
    y = model(x)
    loss = ((y - s_true_norm) ** 2).sum()
    loss.backward()

    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.data -= lr * p.grad
                p.grad.zero_()

    return loss.item()


# ─── B-spline-aware version of Approach B ──────────────────────────────

def online_update_sgd_local(model: KAN, s_norm, a_norm, s_true_norm,
                            lr: float = 1e-4):
    """Full backprop but with B-spline-local learning rate scaling.

    Scale the learning rate for each control point c_{i,j,k} by B_k(x_j).
    Control points for inactive basis functions get zero update — this
    recovers the B-spline locality advantage within standard backprop.

    This is mathematically equivalent to the Hebbian update:
        Δc_{i,j,k} = -η · (∂L/∂y_i) · B_k(x_j)
    """
    x = torch.cat([s_norm, a_norm], dim=-1)
    x.requires_grad_(True)  # need this for autograd through input

    y, B_list, E_list = model(x, return_activations=True)
    loss = ((y - s_true_norm) ** 2).sum()

    # Standard gradients for base_weight and all spline_weight entries
    loss.backward()

    with torch.no_grad():
        for layer_idx, layer in enumerate(model.layers):
            # base_weight: standard update
            layer.base_weight.data -= lr * layer.base_weight.grad
            layer.base_weight.grad.zero_()

            # spline_weight: mask by B-spline activation
            # B_list[layer_idx]: (1, in_dim, n_basis)
            # spline_weight: (out_dim, in_dim, n_basis)
            # spline_weight.grad: (out_dim, in_dim, n_basis)
            B = B_list[layer_idx].squeeze(0)  # (in_dim, n_basis)
            for j in range(layer.in_dim):
                for k in range(layer.spline_weight.shape[2]):
                    if B[j, k] > 1e-8:
                        layer.spline_weight.data[:, j, k] -= \
                            lr * layer.spline_weight.grad[:, j, k]
            layer.spline_weight.grad.zero_()

    return loss.item()
