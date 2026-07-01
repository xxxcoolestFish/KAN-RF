# Decision Network v3: KAN-Adapted Architecture

## 1. Core Principle

KAN provides **gradients**, not features.

| | decision_v2 (dead end) | decision_v3 (this) |
|---|---|---|
| KAN role | Feature computer | Differentiable judge |
| KAN → DN | 5 scalars (hard compression) | Full gradient through KAN |
| DN input | KAN features + s | s only |
| Training signal | BC from inverse opt labels | KAN-predicted outcome |
| Deployment | DN forward pass | DN forward pass (same) |
| Root cause 3 | Still deadly (a_init explodes) | Avoided (no inverse needed) |

## 2. Architecture

```
π_θ: MLP([3, 64, 64, 1])  or  KAN([3, 8, 1])
Input:  s ∈ R³  (cosθ, sinθ, θ̇/8)
Output: a ∈ [-1, 1]  (normalized torque)

Training (offline, using frozen KAN as differentiable simulator):

  for each training step:
    1. Sample batch of states s from dataset
    2. a = π_θ(s)                     # forward through policy
    3. s'_pred = f_KAN(s, a)          # forward through frozen KAN
    4. L = MSE(s'_pred, s_target)     # KAN evaluates "how good is this action?"
       + λ_ctrl * ||a||²             # control penalty
       + λ_smooth * ||π_θ(s)||²      # smoothness prior
    5. L.backward()                   # gradient flows: L → s'_pred → f_KAN → a → π_θ
    6. optimizer.step()               # only π_θ parameters updated
```

Key: the gradient ∂L/∂θ = ∂L/∂s'_pred · ∂s'_pred/∂a · ∂a/∂θ

∂s'_pred/∂a is KAN's Jacobian — accurate after CWS training (cos_sim=0.92).
This means KAN reliably tells π_θ: "if you increase a, the next state will change
in THIS direction."

## 3. Why This Avoids the Forward-Inverse Gap

Root cause 3 (underactuated amplification) kills inverse optimization because:
  a* = argmin_a ||f(s,a) - s*||²
  → error in a ≈ (1/||J||) · error in f  (amplification factor ~25×)

But training π_θ via gradient through KAN only needs:
  ∂L/∂a = 2 · J^T · (f(s,a) - s*)

This is a gradient descent step in a-space, NOT an inverse. As long as J points
in approximately the right direction (cos_sim > 0), the gradient will push a
toward improvement. The magnitude of J doesn't matter for the direction.

**The gradient through KAN is a training signal for π_θ, not a deployment-time
optimization target.** This is the crucial distinction.

## 4. Training Data Strategy

Option A: Uniform sampling from state space
  - Generate states uniformly across the pendulum state space
  - For each s, target state is upright [0, 1, 0]
  - Simple, but may not cover challenging regions (bottom of swing)

Option B: Trajectory replay from action explorer
  - Run action explorer (10/10) to collect successful trajectories
  - Train π_θ to mimic the trajectory distribution + KAN gradient refinement
  - Advantage: covers the regions that actually matter

Option C: Adversarial / curriculum sampling
  - Start with states near upright (easy)
  - Gradually add states further from upright
  - KAN's activation density ρ(s) tells us which regions are poorly covered

## 5. Multi-Step Extension (v3.1)

For tasks requiring coordinated multi-step behavior:

  L_multi = Σ_{t=0}^{H-1} ||s_{t+1} - s_target||²

  where s_{t+1} = f_KAN(s_t, π_θ(s_t))

Gradient flows through H KAN passes → back to each π_θ(s_t) call.
KAN's B-spline derivatives have bounded norm → gradient doesn't explode.

But: H-step rollout accumulates prediction errors. Mitigations:
  - Use short H (3-5 steps)
  - Weight earlier steps more heavily
  - Apply B-spline activation penalty for OOD states

## 6. Online Adaptation (v3.2)

After deployment, KAN can be updated online (three-factor learning rate) and
π_θ can be fine-tuned:

  for each real env step:
    1. a = π_θ(s)
    2. Execute a, observe s'_true
    3. KAN online update: f_KAN ← f_KAN - η · ∇L(s, a, s'_true)
    4. π_θ fine-tune: one gradient step using updated KAN

KAN's local support ensures online updates don't destroy old knowledge.
π_θ stays aligned with the latest dynamics.

## 7. Interpretability Benefits

After training, we can analyze:
  1. Which KAN edges contribute most to π_θ's gradient → which physics matter
  2. Where ρ(s) is low → where the policy may be unreliable
  3. How π_θ's decisions correlate with KAN's internal representations

## 8. Expected Advantages Over Previous Approaches

| Approach | Success | Uses KAN for decision? | Avoids root cause 3? |
|---|---|---|---|
| Inverse opt (v1) | 7/10 | Yes (but wrong way) | No |
| Multi-scale DN (Plan A) | 9/10 | Partially | Partially (k>1 helps) |
| Action explorer | 10/10 | No (bypasses KAN) | Yes (real env, not model) |
| decision_v2 | 7/10 | Yes (as features) | No |
| **decision_v3** | **?** | **Yes (as gradient judge)** | **Yes** |

## 9. Architecture Variant: Residual Physics Policy (PINN-Inspired)

### Motivation

PINN philosophy: keep the network simple, inject physics knowledge through
the architecture and loss function.

Standard decision_v3 policy:
  a(s) = MLP(s)     ← 纯黑箱，KAN 梯度训练

Residual physics policy:
  a(s) = a_physics(s) + δ_θ(s)
       = energy_shaping(s) + small_MLP(s)

The physics prior encodes ~80% of the correct behavior:
  - Swing-up: pump energy when E < E_des  (k_energy · (E-E_des) · θ̇)
  - Stabilize: LQR-like near upright       (k_stable · dθ)
  - Damping: oppose velocity               (k_damp · θ̇)

The residual δ_θ only needs to learn what physics misses (friction, fine
stabilization, model mismatch).  This means:
  - δ_θ is small → KAN gradient errors have proportionally less impact
  - a_physics provides safe fallback when KAN is uncertain
  - Each term is interpretable

### Learnable Parameters

| Parameter | Initial | Meaning |
|-----------|---------|---------|
| k_energy | 0.15 | Energy shaping gain (swing-up) |
| k_stable | -2.0 | Stabilization gain (upright) |
| k_damp | -0.3 | Damping gain |
| residual_net | ~1k params | MLP for δ_θ(s) |
| residual_scale | 0.1 | Scales δ_θ down (physics dominates early) |

### Smooth Transition

swing_weight ∈ [0, 1] transitions smoothly from swing-up (sin→-1, bottom)
to stabilization (sin→1, top):
  a = swing_weight · a_swing + (1-swing_weight) · (a_stable + a_damp)

### Expected Advantage Over Pure MLP

1. Sample efficiency: physics prior bootstraps behavior, KAN gradient only
   needs to refine (not discover from scratch)
2. Robustness to model error: even if KAN gradient is poor in OOD regions,
   a_physics provides reasonable action
3. Interpretability: can analyze k_energy, k_stable, k_damp evolution during
   training, and visualize where δ_θ is large (model mismatch regions)

## 9. Experimental Results (2026-07-01)

### Setup
- KAN world model: [4, 12, 3], basic MSE training (val MSE=0.0064, Jacobian cos_sim ≈ 0.1)
- Training states: 20k uniform samples from state space
- Epochs: 200, batch_size: 256, lr: 1e-3

### Key Finding: Energy Loss > Distance Loss

MSE(s_pred, s*) fails because no single action can reach upright from the
bottom — the gradient through KAN kills the energy-shaping term.

Energy-guided loss succeeds:
  L = -w_swing * energy_gain + w_stable * MSE(s_pred, s*)

### Results

**Phase 1: Basic MSE-trained KAN (Jacobian cos_sim ≈ 0.10)**

| Policy | Loss | Success | Notes |
|--------|------|:---:|------|
| Pure MLP (4.5k params) | Energy | **10/10** | First 10/10 using KAN for decision! |
| Pure MLP (4.5k params) | MSE | N/A | Loss stuck at 2.0 (impossible to minimize) |
| ResidualPhysics (k_e=0.15 frozen) | Energy | 7/10 | k_energy too small for swing-up |
| ResidualPhysics (k_e=1.5 frozen) | Energy | 7/10 | k_damp pushed to +1.31 (anti-damping) by wrong gradient |
| ResidualPhysics (all frozen) | Energy | 7/10 | Residual interferes with correct prior |
| v1 Inverse opt (baseline) | — | 7/10 | Root cause 3 |
| Action explorer (oracle) | — | 10/10 | Bypasses KAN entirely |

**Phase 2: CWS-trained KAN (Jacobian cos_sim ≈ 0.70)**

KAN training: Hybrid MOPS(λ=0.1) + CWS(ν=0.1), 1200 epochs, Val MSE=0.000749.
Jacobian cos_sim improved from 0.10 → 0.70 (7x).

| Policy | Loss | Success | Notes |
|--------|------|:---:|------|
| Pure MLP (4.5k params) | Energy | **10/10** | Consistent with Phase 1 — robust result |
| ResidualPhysics (trainable) | Energy | ? | Pending |

### Analysis

1. **Pure MLP 10/10 proves the paradigm**: KAN can provide useful training
   signal through gradients, without appearing in the deployed policy.
   Result is robust across KAN quality levels (both basic and CWS KAN).

2. **Energy loss is essential**: For underactuated systems, single-step
   distance to target is the wrong objective. Energy gain is the right
   intermediate objective — exactly what PINN philosophy prescribes.

3. **Residual physics with basic KAN fails**: k_energy=0.15 is 10x smaller
   than optimal (1.5). With poor Jacobian, trainable physics params are
   pushed in wrong directions (k_damp → anti-damping).

4. **CWS KAN dramatically improves model quality**: Val MSE from 0.0064 →
   0.00075 (8.5x), Jacobian cos_sim from 0.10 → 0.70 (7x). Pending test:
   does this enable successful residual physics training?

## 10. Implementation Plan

### Step 1: Minimal working version (v3.0)
- [ ] Simple MLP policy π_θ: [3, 64, 64, 1]
- [ ] Frozen KAN f_KAN from CWS training
- [ ] Training: uniform state sampling, single-step KAN evaluation
- [ ] Baseline comparison: BC-trained MLP (no KAN gradient)

### Step 2: Evaluate and diagnose
- [ ] Pendulum 10-trial test
- [ ] Measure gradient quality: cos_sim(∂L/∂a, true improvement direction)
- [ ] Visualize policy vs inverse opt decisions

### Step 3: Multi-step (v3.1)
- [ ] H=3 rollout training
- [ ] Compare single-step vs multi-step training

### Step 4: Online (v3.2)
- [ ] Non-stationary pendulum experiment
- [ ] Compare KAN+π_θ adaptation vs MLP+π_θ
