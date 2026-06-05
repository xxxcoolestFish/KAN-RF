# KAN-RF: KAN-Based Differentiable World Model for Model-Based Planning

## 1. Core Idea

Build a **KAN-based differentiable world model** that serves dual purpose:
- **Forward**: predict $s_{t+1}$ from $(s_t, a_t)$ — as a world model
- **Backward (via frozen parameters)**: gradient descent in action space to find $a^*$ that drives $s_t \to s^*$ — as a planner

**One network, two uses.** No separate policy network needed.

### Key Insight

KAN's edges are explicit 1D functions $\phi_{i,j}(x)$ (B-spline parameterized). After training, these functions are **fully determined** and **everywhere differentiable**. The gradient $\partial f_{\text{KAN}} / \partial a$ can be computed exactly through the frozen KAN parameters, enabling gradient-based optimization in action space.

---

## 2. Architecture

```
Training Phase (Phase 1)              Decision Phase (Phase 2)
─────────────────────────             ─────────────────────────

  (s, a, s') data                     s_t (observed)
       │                                  │
       ▼                                  ▼
  BP train KAN                      s* (target state)
       │                                  │
       ▼                                  ▼
  Frozen KAN                         Frozen KAN
  φ_{i,j} determined                 φ_{i,j} unchanged
                                     │
                                     ▼
                              min_a || f_KAN(s_t, a) - s* ||²
                                     │
                                     ▼
                                   a* → execute
```

### Input / Output of KAN World Model

$$\hat{s}_{t+1} = f_{\text{KAN}}(s_t, a_t)$$

| Variable | Meaning | Dim |
|----------|---------|-----|
| $s_t$ | Current system state | $d_s$ |
| $a_t$ | Action | $d_a$ |
| $\hat{s}_{t+1}$ | Predicted next state | $d_s$ |

### Why Multi-Layer KAN Is Necessary

A single-layer KAN is an additive model: $f(x) = \sum_j \phi_j(x_j)$. It **cannot** capture variable interactions like $x_1 \cdot x_2$ or $\sin(x_1 \cdot x_2)$.

Two-layer KAN enables interaction terms via nesting:
$$f(x) = \sum_q \Phi_q\left(\sum_p \phi_{q,p}(x_p)\right)$$

Physical dynamics almost always involve state-action coupling, so **at least 2 layers are required**.

---

## 3. Phase 1: Pre-training the KAN World Model

**Goal**: Determine the 1D functions $\phi_{i,j}$ by fitting $(s, a) \to s'$.

### Data Collection
- Random policy or simple heuristic
- Collect tuples $(s_t, a_t, s_{t+1})$ from environment interaction

### Training
- Standard supervised learning with BP + Adam
- Loss: $\mathcal{L} = \|f_{\text{KAN}}(s, a) - s'\|_2^2$
- Train until prediction error is sufficiently low

### Result
- All B-spline control points $c_k$ and scale factors $w$ are determined
- Every edge function $\phi_{i,j}(x) = w \cdot (\text{SiLU}(x) + \sum_k c_k B_k(x))$ is fixed
- The KAN is frozen and ready for Phase 2

---

## 4. Phase 2: Decision via Gradient Descent in Action Space

### 4.1 Single-Step Planning

Given $(s_t, s^*)$ and frozen KAN world model $f_{\text{KAN}}$:

Initialize $a^{(0)}$ randomly. For $k = 0, 1, ..., K-1$:

1. Forward: $\hat{s} = f_{\text{KAN}}(s_t, a^{(k)})$
2. Loss: $\mathcal{L} = \| \hat{s} - s^* \|_2^2$
3. Gradient: $\nabla_a \mathcal{L}$ (autograd through frozen KAN)
4. Update: $a^{(k+1)} = a^{(k)} - \alpha \cdot \nabla_a \mathcal{L}$
5. Project: $a^{(k+1)} = \text{clamp}(a^{(k+1)}, a_{\min}, a_{\max})$

Return $a^* = a^{(K)}$.

### 4.2 Multi-Step Shooting (Generalization)

For tasks requiring $H$-step planning, optimize the full action sequence $A = [a_0, ..., a_{H-1}]$:

$$\mathcal{L}(A) = \|s_H - s^*\|^2 + \lambda \sum_{h=0}^{H-1} \|a_h\|^2$$

where $s_{h+1} = f_{\text{KAN}}(s_h, a_h)$ for $h = 0, ..., H-1$.

Gradient flows through $H$ consecutive KAN forward passes back to each $a_h$. Cos/sin normalization is applied at each step to maintain the unit-circle constraint.

**Key implementation details**:
- Adam optimizer on the full action sequence (not SGD — ~20× faster convergence)
- Multiple random restarts (2-3) to escape local minima
- Control penalty $\lambda$ balances terminal accuracy vs. energy efficiency

### Why This Works

- KAN is **fully differentiable**: $\phi_{i,j}(x)$ uses B-splines with closed-form derivatives
- During decision, model parameters are frozen — only actions are optimized
- PyTorch autograd handles the chain rule through frozen KAN parameters
- B-spline basis functions $B_k(x)$ and their derivatives $B_k'(x)$ are analytical

### Generalized Objective

For tasks beyond exact state matching:
$$\mathcal{L}(A) = \sum_{h=0}^{H-1} C_h(s_h, a_h) + C_{\text{terminal}}(s_H, s^*)$$

where $C_h$ are differentiable stage costs.

---

## 5. Advantages Over Standard MBRL

| Aspect | Standard MBRL | KAN-RF |
|--------|--------------|--------|
| World model | Black-box NN | KAN — interpretable 1D functions |
| Policy | Separate policy network | Gradient descent in action space (no extra net) |
| Explainability | None | Can trace which $\phi_{i,j}$ contribute to decision |
| Sample efficiency (Phase 1) | Needs lots of data | KAN's parameter efficiency may help |
| Task switching | Retrain policy | Only change $s^*$ in objective |

---

## 6. Central Challenge: Model Exploitation (Model-Reality Gap)

### 6.1 The Problem

When the shooting planner optimizes actions through the learned world model, it can discover trajectories that **the model predicts well but the real environment executes poorly**. This is *model exploitation* — the optimizer finds the model's blind spots and exploits them.

This is **not unique to KAN-RF**. It is the central challenge of all model-based RL methods (MBPO, Dreamer, PETS, MOPO — all dedicate significant portions to addressing it).

### 6.2 Pendulum Case Study

**Trial with small initial offset (θ₀=1.21 rad, only 0.36 rad from upright)**:
- Model prediction: |Δθ| = 0.13 rad ✓
- Real execution: |Δθ| = 0.13 rad ✓
- Success: the trajectory stayed within the training distribution

**Trial requiring full swing-up (θ₀=-1.91 rad, 3.48 rad from upright)**:
- Model prediction: |Δθ| = **0.12 rad** (model thinks it found a near-perfect solution)
- Real execution: |Δθ| = **1.04 rad** (pendulum flew past upright at 4.68 rad/s)
- Failure: the trajectory required 30-step coordinated actions far outside the training distribution

### 6.3 Root Cause — Three Layers

**Layer 1: Training data from random actions covers only "aimless wobbling"**
Random actions produce pendulum trajectories near the bottom with moderate velocities. The shooting optimizer discovers coordinated swing-up trajectories involving energy pumping and braking — state-action pairs never seen during training.

**Layer 2: The optimizer actively seeks model blind spots**
It cannot distinguish "the model is correct" from "the model is hallucinating." It only sees loss decreasing — if a trajectory exploits B-spline extrapolation errors to produce a low model-loss, the optimizer will prefer it.

**Layer 3: 30-step open-loop error accumulation**
Even with per-step RMSE ≈ 0.03, accumulated error over 30 steps can exceed 0.16 rad. With OOD extrapolation errors compounding, total error exceeds 0.9 rad.

### 6.4 Standard MBRL Solutions

| Solution | Core Idea |
|----------|-----------|
| Short-horizon MPC | Plan 5-10 steps, execute first action, replan |
| Ensemble + uncertainty penalty | Train multiple models, penalize high-variance regions during planning |
| Iterative bootstrapping (Dyna) | Collect new data with planned actions, retrain model, repeat |
| Action noise regularization | Add noise during planning to discourage exploitation of model errors |

### 6.5 KAN-Specific Advantage: B-Spline Activation as Free Uncertainty Signal

Unlike MLPs which extrapolate confidently (and wrongly) outside their training distribution, KAN's B-spline basis functions have **strictly local support**:

$$B_k(x) > 0 \text{ only when } x \in [t_k, t_{k+d+1}]$$

When an input falls in a region rarely visited during training, the corresponding B-spline activations are near zero, and the model degrades gracefully to the SiLU baseline. This means:

> **The B-spline activation pattern itself is a built-in proxy for epistemic uncertainty — without requiring additional ensemble training or uncertainty networks.**

**Proposed KAN-specific solution**: During shooting optimization, add a penalty term proportional to the *inverse* of B-spline activation density along the planned trajectory:

$$\mathcal{L}_{\text{uncertainty}} = -\beta \sum_{h} \sum_{i,j,k} \log\left(B_k(\text{input}_{h,j}) \cdot \mathbb{1}[B_k > \epsilon]\right)$$

This discourages the optimizer from steering into regions where B-spline activations are sparse — i.e., regions where the model "knows it doesn't know." **No extra ensemble training needed.**

---

## 7. Future: Hebbian Fast Adaptation

Once the full framework is validated, Hebbian learning can be added for real-time adaptation:

When $f_{\text{KAN}}(s_t, a_t) \neq s_{t+1}^{\text{actual}}$:
$$\Delta c_{i,j,k} = \eta \cdot B_k(\text{input}_j) \cdot e_i$$

This enables **forward-only, O(1)-memory** parameter updates without backpropagation, leveraging:
- B-spline spatial locality (only activated grids update)
- Activation path traceability (we know exactly which control points participated)

---

## 8. Experimental Plan

### Stage 1: Minimal Prototype (2D Point-Mass) — ✅ VERIFIED

**Linear**: $s_{t+1} = s_t + a_t$
**Nonlinear**: $s_{t+1} = s_t + a_t + 0.1 \cdot \sin(s_t) \cdot \cos(a_t)$

| Metric | Linear | Nonlinear |
|--------|--------|-----------|
| KAN arch | [4, 5, 2] | [4, 8, 2] |
| Params | 270 | 432 |
| World model val MSE | 0.00038 | 0.00051 |
| Decision |a_pred - a_true| mean | 0.012 | 0.014 |
| Decision |s_next - s*| mean | 0.012 | 0.014 |
| Decision |s_next - s*| max | 0.046 | 0.062 |

**Key findings**:
- Adam inner-loop (lr=0.1, 200 iter) dramatically outperforms SGD (0.23 → 0.012 error)
- Remaining error is bounded by world model accuracy — improving prediction directly improves decisions
- Multi-layer KAN successfully learns nonlinear state-action coupling
- Gradient descent through frozen KAN reliably finds actions even when no analytical inverse exists
- [x] Train 2-layer KAN world model on random actions
- [x] Evaluate prediction accuracy
- [x] Test decision: given $s_t$ and $s^*$, does gradient descent in $a$-space find the correct action?
- [x] Test with nonlinear dynamics variant

### Stage 2: Pendulum Swing-Up — 🔄 FUNCTIONAL, Model Exploitation Exposed

**Environment**: Pendulum-v1 (gymnasium). State: [cosθ, sinθ, θ̇] (3D), Action: torque ∈ [-2, 2] (1D).

**Setup**:
- KAN world model: [4, 12, 3], grid=5, order=3, 756 params
- Training data: 15000 random-action transitions
- Val MSE: 0.0010 (RMSE ~0.03 per dim)
- Multi-step shooting: H=30, Adam lr=0.1, 2 restarts, control penalty λ=0.01

| Metric | Result |
|--------|--------|
| Single-trial feasibility (small offset, θ₀=1.21 rad) | ✓ Success — |Δθ|=0.13 rad |
| Full swing-up (θ₀=-1.91 rad) | ✗ Model |Δθ|=0.12 rad → Real |Δθ|=1.04 rad |
| Optimization convergence | ✓ Loss decreases reliably (e.g., 4.15→0.06) |
| Speed | ~660s/trial (B-spline for-loop bottleneck) |

**Key findings**:
- Multi-step shooting through frozen KAN **works** when the trajectory stays within the training distribution
- When the optimizer discovers trajectories in unseen regions, it suffers from **model exploitation** — the classic MBRL failure mode
- The model-reality gap is amplified by 30-step open-loop error accumulation
- B-spline's local support provides a **free, built-in uncertainty signal** that can be leveraged to penalize OOD trajectories during planning (see §6.5)
- Speed bottleneck is the non-vectorized B-spline implementation — vectorization would reduce from ~660s to ~10s per trial

### Stage 3: Hebbian Fast Adaptation (Future)

- Add prediction-error-driven Hebbian updates
- Measure adaptation speed vs. BP retraining
- Test under distribution shift
- Multi-layer credit assignment for Hebbian updates

---

## 9. Current Status & Next Steps

### What Works
1. ✅ Single-step planning on 2D point-mass (linear + nonlinear)
2. ✅ Multi-step shooting optimization converges reliably
3. ✅ B-spline forward+backward through frozen KAN supports gradient-based planning
4. ✅ KAN world model learns nonlinear dynamics from random data

### What Needs Work
1. 🔄 **Model exploitation** — the central MBRL challenge, now with a KAN-specific solution path
2. 🐛 **Speed** — B-spline vectorization (engineering, not algorithmic)
3. 🔜 **KAN-specific uncertainty penalty** — implement B-spline activation sparsity penalty
4. 🔜 **MPC-style replanning** — short horizon + replan after each step
5. 🔜 **Comparison with standard MBRL methods** on classic control benchmarks

### Open Questions
1. **B-spline activation as uncertainty proxy** — does the penalty method (§6.5) effectively prevent model exploitation?
2. **Scalability to high-dimensional action spaces** — gradient descent in action space for high-dim actions
3. **Comparison with model-free baselines** — how does KAN world model + planning compare to SAC/PPO on standard benchmarks?
4. **Naming** — KAN-RF or alternative? (RF originally undefined; could be "Representation-Free", "Reactive Feedback", etc.)
