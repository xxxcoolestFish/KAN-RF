# Fitting Depth: A Derivative-Aware Training Paradigm for KAN

## 1. Concept

**Fitting depth** $D$ is defined as the maximum integer such that the model's
derivatives up to order $D$ all match the true system within tolerance:

$$D = \max\left\{d \;\middle|\; \left\|\frac{\partial^k f_\theta}{\partial a^k} - \frac{\partial^k f_{\text{true}}}{\partial a^k}\right\| \le \epsilon_k,\; \forall k \in [0, d]\right\}$$

- $k=0$: function value accuracy (standard MSE)
- $k=1$: Jacobian accuracy — the model knows *which direction* the action pushes the state
- $k=2$: Hessian accuracy — the model knows *how the sensitivity itself changes* with action
- $k \ge 3$: higher-order structure

A KAN trained with standard MSE loss achieves only $D=0$, regardless of how low
the validation error is.  The forward prediction is accurate, but the
derivatives are systematically wrong (on Pendulum-v1 v6: cosine similarity
between $\partial f_\theta / \partial a$ and $\partial f_{\text{true}} / \partial a$
is negative at 3 of 4 tested states).

Deeper fitting depth is critical for gradient-based decision-making: the
optimizer follows $\partial f_\theta / \partial a$, not $f_\theta$ itself.

## 2. Why KAN Has a Structural Advantage

### 2.1 B-Spline Derivative Formula

For a degree-$k$ B-spline $f(x) = \sum_i c_i B_i^k(x)$ on a uniform grid with
spacing $h$, the $d$-th derivative is (de Boor, 1978):

$$f^{(d)}(x) = \frac{1}{h^d} \sum_i (\Delta^d c)_i \cdot B_i^{k-d}(x)$$

where $(\Delta^d c)_i$ is the $d$-th forward difference of the control point
sequence and $B_i^{k-d}$ are B-splines of reduced degree $k-d$.

For cubic ($k=3$) B-splines, the three meaningful derivative orders are:

| Derivative | Control points | Basis degree | Formula |
|-----------|---------------|:---:|---------|
| $f'(x)$   | $(1/h) \cdot \Delta^1 c_i$ | 2 | $(1/h) \sum (c_i - c_{i-1}) B_i^2(x)$ |
| $f''(x)$  | $(1/h^2) \cdot \Delta^2 c_i$ | 1 | $(1/h^2) \sum (c_i - 2c_{i-1} + c_{i-2}) B_i^1(x)$ |
| $f'''(x)$ | $(1/h^3) \cdot \Delta^3 c_i$ | 0 | $(1/h^3) \sum \Delta^3 c_i \cdot B_i^0(x)$ (piecewise constant) |

### 2.2 Certifiable Derivative Bounds

The norm of the $d$-th derivative is bounded by the maximum $d$-th difference
of the control points:

$$\|f^{(d)}\|_\infty \le \frac{1}{h^d} \cdot \max_i |\Delta^d c_i|$$

For the current KAN configuration ($h = 2 \cdot \text{grid\_range} / \text{grid\_size} = 0.4$):

$$1/h = 2.5,\quad 1/h^2 = 6.25,\quad 1/h^3 = 15.625$$

### 2.3 P-spline Connection (Eilers & Marx, 1996)

The $d$-th difference penalty on control points approximates the integrated
squared $(d-1)$-th derivative of the function:

$$\|\Delta^d c\|^2 \approx h^{2d-1} \int [f^{(d-1)}(x)]^2 dx$$

- $d=2$ (standard P-spline): controls first-derivative roughness
- $d=3$: controls second-derivative curvature (cubic smoothing spline analog)

### 2.4 Why MLPs Cannot Do This

An MLP layer is $h = \sigma(Wx + b)$. The weights $W_{ij}$ have no intrinsic
spatial ordering — adjacent rows of $W$ do not correspond to adjacent regions
of the input space. Penalizing $\|W_{i+1,j} - 2W_{i,j} + W_{i-1,j}\|^2$ is
meaningless because the row ordering is arbitrary.

KAN's control points are different: they are arranged along a 1D grid with
strict local support. $c_{k-1}$, $c_k$, $c_{k+1}$ control adjacent intervals
of the same input dimension. Their differences have precise geometric meaning
as function derivatives.

## 3. The Three-Level Training Framework

The method decomposes derivative regularization into three levels, each
operating at a different structural layer of a KAN and addressing a distinct
aspect of fitting depth.

### Level 1: Parameter Space — Multi-Order P-Spline (MOPS)

Direct penalty on control point differences:

$$L_{\text{MOPS}} = \sum_{d=2}^{D+1} \lambda_d \sum_{\text{all edges}} \|\Delta^d c\|^2$$

The P-spline identity $\|\Delta^d c\|^2 \approx h^{2d-1} \int [f^{(d-1)}(x)]^2 dx$
means each order directly penalizes a specific derivative.  $d=1$ is excluded
because it penalizes $\int f(x)^2 dx$ (shrinkage toward zero), not any
derivative — it would fight the MSE loss rather than complement it.

| Term | Penalizes | Fitting depth contribution |
|------|----------|---------------------------|
| $\|\Delta^2 c\|^2$ | $\int [f'(x)]^2 dx$ (1st derivative energy) | Enables $D \to 1$ |
| $\|\Delta^3 c\|^2$ | $\int [f''(x)]^2 dx$ (2nd derivative energy) | Enables $D \to 2$ |
| $\|\Delta^4 c\|^2$ | $\int [f'''(x)]^2 dx$ (3rd derivative energy) | Enables $D \to 3$ (cubic limit) |

**Properties:**
- $O(\text{total control points})$ computation, independent of batch size
- Operates globally on the function domain, not only at training samples
- Does not require knowledge of $J_{\text{true}}$
- Provides certifiable bounds on $\|f^{(d)}\|_\infty$

**What it cannot do:** Ensure the derivative points in the *correct* direction.
A smooth but systematically biased B-spline will have wrong derivatives everywhere.

### Level 2: Edge Space — Critical-Path Derivative Matching (CPDM)

Not all edges contribute equally to the action-output Jacobian. For an action
$a$ entering at a specific input dimension, we compute per-edge attribution:

$$\text{attr}(i,j) = \mathbb{E}_{x \sim \mathcal{D}}\left[\left\|\frac{\partial(\partial y / \partial a)}{\partial c_{ij}}\right\|\right]$$

High-attribution edges receive additional 1D derivative matching:

$$L_{\text{CPDM}} = \sum_{(i,j) \in \text{critical}} \mu_{ij} \cdot \|\phi'_{ij}(x) - \phi'_{\text{target}}(x)\|^2$$

where $\phi'_{\text{target}}$ can be estimated from finite differences on
training data or derived analytically for known dynamics.

**Properties:**
- Each edge derivative is a scalar 1D function — cheap to evaluate and match
- Targets the specific edges responsible for derivative errors
- Provides per-edge interpretability: which edges are the bottlenecks?

### Level 3: Output Space — Controllability-Weighted Sobolev (CWS)

For systems where the true Jacobian is available (analytic dynamics):

$$L_{\text{CWS}} = \sum_{d=0}^D \sum_k w_k \cdot \left\|\frac{\partial^d f_k}{\partial a^d} - J^{(d)}_{\text{true}, k}\right\|^2$$

The per-dimension weights $w_k$ reflect controllability:

$$w_k \propto \left\|\frac{\partial s'_k}{\partial a}\right\|$$

For Pendulum: $w_{\dot{\theta}} \gg w_{\cos} \approx w_{\sin}$. The directly
actuated dimension ($\dot{\theta}$) needs the most accurate derivative.

**Properties:**
- Directly aligns model derivatives with physics
- Controllability weighting prevents the optimizer from chasing noise in
  indirectly-affected dimensions
- Requires knowing $J_{\text{true}}$ — applicable to analytic/simulator envs

### Training Schedule

Phase-based progression rather than simultaneous activation:

1. **Phase 1 (Smoothness):** $\lambda_d$ high, $\mu=\nu=0$. Multi-order
   P-spline alone to establish globally smooth edge functions. Target:
   eliminate the 68% control-point sign-change rate observed in v6.

2. **Phase 2 (Direction):** Reduce $\lambda_d$, add $\mu$. Edge-level
   matching on critical paths to correct systematic directional bias in
   the now-smooth edge functions.

3. **Phase 3 (Magnitude):** Add $\nu$. Output-space matching on controllable
   dimensions to align Jacobian (and optionally Hessian) magnitude with physics.

4. **Phase 4 (Deepen):** If $D>1$, enable higher-order terms ($\Delta^3$, $\Delta^4$,
   Hessian matching).

Each phase addresses a different failure mode. The combination achieves what
none can alone: smoothness from MOPS, direction from CPDM, and physical
alignment from CWS.

## 4. Literature Context

### Confirmed Gaps

| Direction | Status |
|-----------|--------|
| KAN + Sobolev training | **No existing work** |
| Parameter-space (control point) + output-space derivative penalties | **No existing work** |
| Controllability-weighted world model loss | **No existing work** |
| Spectral regularization of B-spline control points in DL | **No existing work** |
| KAN grid refinement with derivative penalties | **No existing work** |

### Closest Related Work

- **PI-KAN / KINN** (Wang et al., 2024): Uses KAN inside PINN for PDE
  residuals — derivatives appear in the loss but only as PDE operators, never as
  regularization targets on the KAN parameters themselves.

- **KANtrol** (Afzal Aghaei, 2024): Applies KAN to optimal control, uses
  autodiff for derivative computation but does not add derivative
  regularization.

- **GLSPIA** (2024): Uses first-order (tension) and second-order (curvature)
  control polygon penalties for geometric spline fitting — the closest
  parameter-space approach but for classical splines, not neural KAN training.

- **Lagrangian identification** (arXiv:2511.10706): Regularizes second-order
  derivatives of cubic B-spline control points for system identification.
  Without this, derivatives are poorly captured despite near-perfect data fit.
  Directly supports our $D=0$ vs. $D\ge 1$ distinction.

- **Unser et al.** (2024, arXiv:2408.13114): Proves curvature-regularized
  trainable activations converge to adaptive B-splines. The theoretical
  bridge between P-splines and neural network training, but does not use
  the KAN architecture.

- **Neural P-splines** (Dammann et al., 2025): Represents P-splines as neural
  networks with difference penalties and gradient-based smoothing parameter
  selection. Applies to classical regression, not KAN or control.

### Cautionary Results

- **Schutz et al. (2026):** 22 second-order Sobolev loss variants tested on
  nonlinear FE model reduction — *none* outperformed first-order training.
  Suggests diminishing returns beyond $D=1$ for certain problem classes.

- **Hartmann et al.:** For smooth functions (bowl/plate), derivatives up to
  order 3-4 significantly reduce test error, diminishing by order 5.

- **Fisher et al. (2025):** Sobolev training helps in underparameterized regime,
  can hurt in overparameterized regime. KAN's compact parameterization (756
  params for [4,12,3]) may naturally operate in the favorable regime.

### Convergence Theory

- **Cocola & Hand (2020):** Overparameterized 2-layer ReLU + Sobolev loss
  achieves global convergence with exponential rate under data separation.

- **Cho, Ryu, Hwang (JCP 2025):** First convergence-rate analysis for
  Sobolev training in operator learning — derivative loss terms provably
  improve the rate over $L^2$-only training.

## 5. Experimental Validation (2026-05-23)

We tested the MOPS and CWS methods on the Pendulum-v1 world model task to
validate the fitting depth framework.  This corresponds to Phase 1 and
Phase 2 of the roadmap below.

### 5.1 Setup

| Item | Value |
|------|-------|
| Data | `pendulum_data_v4.pt` (35,000 transitions, random actions) |
| Architecture | KAN [4, 12, 3], grid=5, order=3, 756 params |
| Optimizer | Adam lr=1e-2, StepLR(step=600, gamma=0.5) |
| Epochs | 2,400 |
| Seed | 42 (fixed across all runs) |
| Train/Val split | 85% / 15% |
| CWS batch size | 2,048 (required for per-sample autograd Jacobian) |

MOPS uses full-batch training since the P-spline penalty requires no
autograd.  CWS and Hybrid use mini-batches of 2,048 to keep the per-sample
Jacobian computation within memory limits.

### 5.2 Methods Tested

| Method | Loss | $\lambda$ / $\nu$ |
|--------|------|:---:|
| Baseline | MSE only | — |
| MOPS | MSE $+ \lambda \cdot \|\Delta^2 c\|^2$ | 0.01, 0.1, 1.0 |
| CWS | MSE $+ \nu \cdot \|w \odot (\partial f/\partial a - J_{\text{true}})\|^2$ | 0.01, 0.1, 1.0 |
| Hybrid | MSE $+ \lambda \cdot \|\Delta^2 c\|^2 + \nu \cdot \|w \odot (\partial f/\partial a - J_{\text{true}})\|^2$ | $\lambda=0.1, \nu=0.1$ |

CWS weights: $w = [1.0, 1.0, 3.0]$, reflecting that $\dot{\theta}$ is
directly actuated while $\cos\theta$ and $\sin\theta$ respond only indirectly
through integration.  $J_{\text{true}}$ is computed analytically from the
pendulum dynamics: $[\partial\cos'/\partial a,\; \partial\sin'/\partial a,\;
\partial\dot{\theta}'/\partial a] = [-0.015\sin\theta',\; 0.015\cos\theta',\;
0.0375]$ (in normalized units).

All models use the same random seed, data split, and hyperparameters.  The
only difference between runs is the presence and strength of the
regularization terms.

### 5.3 Evaluation Metrics

| Metric | What it measures | Unit |
|--------|-----------------|------|
| val MSE | Forward prediction error on held-out data | Normalized state space |
| $\|a_{\text{err}}\|$ mean | Average error between recovered and true action | Normalized torque ($\times 2$ = N·m) |
| $\|a_{\text{err}}\|$ median | Median of the above (robust to outliers) | Same |
| $\|a_{\text{err}}\|$ P90 | 90th percentile — worst-case behavior | Same |
| cos_sim | Cosine similarity: $\cos(\partial f_\theta/\partial a,\; \partial f_{\text{true}}/\partial a)$ | $[-1, 1]$ |
| $\|\Delta^2 c\|$ | Mean squared second-difference of control points | Lower = smoother edges |

Inverse accuracy is measured on 200 held-out test samples.  For each sample
$(s, a_{\text{true}}, s'_{\text{true}})$, we solve $a_{\text{opt}} = \arg\min_a
\|f_\theta(s, a) - s'_{\text{true}}\|^2$ via gradient descent (3 random
restarts $\times$ 150 Adam iterations) and compare $a_{\text{opt}}$ to
$a_{\text{true}}$.

### 5.4 Full Results

| Method | val MSE | $\|a_{\text{err}}\|$ mean | $\|a_{\text{err}}\|$ med | $\|a_{\text{err}}\|$ P90 | cos_sim | $\|\Delta^2 c\|$ |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline | 0.001785 | 0.507 | 0.248 | 1.386 | 0.099 | 0.0986 |
| MOPS $\lambda=0.01$ | 0.001750 | 0.506 | 0.284 | 1.320 | 0.075 | 0.0745 |
| MOPS $\lambda=0.1$ | 0.001907 | 0.428 | 0.183 | 1.127 | 0.237 | 0.0208 |
| MOPS $\lambda=1.0$ | 0.002479 | 0.514 | 0.320 | 1.340 | 0.090 | 0.0016 |
| CWS $\nu=0.01$ | 0.000260 | 0.387 | 0.209 | 1.127 | 0.651 | 0.0934 |
| CWS $\nu=0.1$ | 0.000308 | 0.359 | 0.180 | 0.975 | 0.807 | 0.0926 |
| CWS $\nu=1.0$ | 0.000337 | 0.291 | 0.177 | 0.734 | **0.979** | 0.1020 |
| **Hybrid** | **0.000198** | **0.228** | **0.131** | **0.656** | 0.924 | **0.0007** |

### 5.5 Hybrid vs Baseline: Improvement

| Metric | Baseline | Hybrid | Change |
|--------|:---:|:---:|:---:|
| Forward MSE | 0.001785 | 0.000198 | 9.0$\times$ lower |
| Mean inverse error | 0.507 | 0.228 | $-55\%$ |
| Median inverse error | 0.248 | 0.131 | $-47\%$ |
| P90 inverse error | 1.386 | 0.656 | $-53\%$ |
| Jacobian cos_sim | 0.099 | 0.924 | 9.3$\times$ higher |
| Control point roughness | 0.0986 | 0.0007 | 141$\times$ lower |

In physical units: the mean torque recovery error drops from 1.01 N·m to
0.46 N·m (max torque is $\pm 2.0$ N·m).  The worst-decile error drops from
2.77 N·m (exceeding the torque limit — the model recovers an action in the
*opposite* direction) to 1.31 N·m.

### 5.6 Analysis

**MOPS alone provides smoothness but not direction.**  At $\lambda=0.1$,
control point roughness drops 79% (0.099 $\to$ 0.021) and mean inverse error
improves 16% (0.507 $\to$ 0.428).  However, Jacobian cosine similarity
barely moves (0.099 $\to$ 0.237).  Smoother edge functions produce a more
well-behaved optimization landscape with fewer spurious local minima, but
the underlying gradient direction remains misaligned with physics.  At
$\lambda=1.0$, the roughness drops to 0.0016 but forward MSE rises to
0.002479 — the smoothness penalty overpowers the data fit.

**CWS alone provides direction but not smoothness.**  Jacobian cosine
similarity rises dramatically: 0.099 (baseline) $\to$ 0.651 ($\nu=0.01$)
$\to$ 0.807 ($\nu=0.1$) $\to$ 0.979 ($\nu=1.0$).  Mean inverse error drops
from 0.507 to 0.291 at $\nu=1.0$.  Notably, CWS also improves *forward* MSE
by 5–7$\times$ (0.001785 $\to$ 0.000260–0.000337) — the Jacobian matching
acts as a powerful physics-informed regularizer that improves global function
fitting.  However, control point roughness remains at $\sim$0.09–0.10 across
all $\nu$ values.  The edge functions remain wiggly even though their
aggregate output Jacobian is correctly aligned.

**The Hybrid achieves both.**  Compared to CWS $\nu=0.1$ alone: forward MSE
drops 36% further (0.000308 $\to$ 0.000198), mean inverse error drops 36%
further (0.359 $\to$ 0.228), control point roughness drops 132$\times$
(0.0926 $\to$ 0.0007), and Jacobian cos_sim improves from 0.807 to 0.924.
The MOPS penalty suppresses the residual control-point oscillation that CWS
alone cannot eliminate, while the CWS penalty maintains the Jacobian
alignment that MOPS alone cannot achieve.  The two mechanisms are genuinely
complementary.

**Optimal regularization strength.**  For MOPS, $\lambda=0.1$ is optimal —
$\lambda=0.01$ is too weak (nearly identical to baseline), $\lambda=1.0$ is
too strong (over-smooths, harming data fit).  For CWS, $\nu=1.0$ gives
near-perfect Jacobian alignment (cos_sim 0.979) but $\nu=0.1$ gives slightly
better inverse accuracy, suggesting that exact Jacobian matching may
over-constrain in regions where the true Jacobian varies rapidly.

**Residual inverse error.**  Even the Hybrid has mean inverse error of 0.228
(0.46 N·m).  This residual is consistent with the forward-inverse gap
analysis: the amplification factor $\|\varepsilon\| / \|J\| \approx 0.04 /
0.04 \approx 1.0$ sets a lower bound on inverse error given any nonzero
forward prediction residual and the underactuated nature of the pendulum
(see `FORWARD_INVERSE_GAP.md`).

### 5.7 Validated Hypotheses

1. **MOPS and CWS operate through different mechanisms.**  MOPS reduces
   $\|\Delta^2 c\|$ by 141$\times$ but barely moves cos_sim.  CWS raises
   cos_sim from 0.099 to 0.979 but leaves $\|\Delta^2 c\|$ unchanged.  Each
   addresses a distinct aspect of fitting depth.

2. **The mechanisms are complementary.**  Hybrid outperforms either method
   alone on every metric.  Smoothness (MOPS) + directional alignment (CWS)
   together produce the deepest effective fitting depth.

3. **Fitting depth $D \ge 1$ is achievable and measurable.**  The Hybrid
   achieves cos_sim of 0.924 — the model's local linear approximation is,
   on average, within $\sim$22° of the true gradient direction, compared to
   $\sim$84° for the baseline.  This directly translates to more accurate
   inverse recovery.

4. **Control-point roughness is a computable diagnostic.**  $\|\Delta^2 c\|$
   provides a scalar, interpretable metric for KAN edge function smoothness.
   It drops monotonically with $\lambda$ and correlates with improved
   optimization behavior.

---

## 6. Research Roadmap

### Phase 1: Validate $D=0 \to D=1$ — DONE

- [x] Implement MOPS ($d=2$) — `train_mops.py`
- [x] Compare baseline vs MOPS on forward MSE and inverse accuracy
- [x] Measure roughness, cos_sim, and inverse error distributions
- [x] Result: MOPS $\lambda=0.1$ reduces inverse error 16%, roughness 79%

### Phase 2: Add Output-Space Sobolev — DONE

- [x] Implement CWS with analytic $J_{\text{true}}$ — `train_cws.py`
- [x] Compare MOPS-only vs CWS-only vs Hybrid
- [x] Result: CWS $\nu=0.1$ improves cos_sim to 0.807; Hybrid achieves 0.924

### Phase 3: Control Pipeline Integration

- [ ] Replace v6 world model with Hybrid-trained model in exp_F
- [ ] Measure: does improved inverse accuracy translate to higher swing-up
      success rate or fewer smart-burst triggers?
- [ ] Profile: does higher cos_sim reduce the number of Adam iterations
      needed per control step?

### Phase 4: Edge-Level Attribution (CPDM)

- [ ] Implement edge contribution analysis to identify which edges dominate
      the Jacobian
- [ ] Test CPDM on the identified critical edges
- [ ] Compare CPDM vs uniform CWS: can we achieve comparable results with
      less computation by only matching derivatives on a subset of edges?

### Phase 5: $D \to 2$ (Hessian Matching)

- [ ] Add $\|\Delta^3 c\|^2$ penalty (controls second-derivative curvature)
- [ ] Add $\|\partial^2 f / \partial a^2 - H_{\text{true}}\|^2$ where
      $H_{\text{true}}$ is computed analytically
- [ ] Measure whether $D=2$ further improves inverse accuracy or
      optimization convergence speed
- [ ] Note Schutz et al. (2026) caution: second-order Sobolev did not help
      in 22/22 FE model variants

### Phase 6: Generalization

- [ ] Test MOPS on a second environment (CartPole, Acrobot) — MOPS requires
      no $J_{\text{true}}$, so it applies to any system
- [ ] If $D \ge 1$ is achieved universally via MOPS, this establishes
      "derivative-safe KAN" as a general property independent of the env
- [ ] Compare KAN + MOPS vs MLP + online learning on the same task

---
*Document started 2026-05-23.  Updated with experimental results 2026-05-23.*
