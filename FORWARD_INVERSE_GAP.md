# The Forward-Inverse Gap: Why Accurate Forward Prediction Does Not Imply Accurate Inverse Recovery

## 1. Problem Statement

We train a KAN world model $f_\theta$ to predict the next state of a dynamical
system given the current state and action:

$$\hat{s}' = f_\theta(s, a)$$

The training minimizes mean squared error on collected transition data:

$$\theta^* = \arg\min_\theta \;\mathbb{E}_{(s,a,s') \sim \mathcal{D}}\left[ \|f_\theta(s, a) - s'\|^2 \right]$$

After training, the forward prediction error is small: $f_{\theta^*} \approx f_{\text{true}}$
pointwise across the training distribution.  On Pendulum-v1 with $35{,}000$
transitions and a $[4, 12, 3]$ KAN (756 parameters), the held-out root mean
squared error is $0.037$ in normalized state space, corresponding to
approximately $0.14$ radians of angular error.

During deployment, the model is used in reverse.  Given a current state $s$
and a desired next state $s'_{\text{target}}$, we find the action by
gradient-based optimization through the frozen model:

$$a^* = \arg\min_a \;\|f_\theta(s, a) - s'_{\text{target}}\|^2$$

**The empirical finding is that this inverse recovery is poor**: when ground
truth $(s, a_{\text{true}}, s'_{\text{true}})$ is held out and we solve for
$a^*$ given $(s, s'_{\text{true}})$, the mean absolute error in the recovered
action is $|a^* - a_{\text{true}}| \approx 0.87$ N·m — nearly half the
maximum torque range of $\pm 2.0$ N·m.  Only $36.5\%$ of recoveries fall
within $0.2$ N·m of the true action.  For the worst decile, the error exceeds
$2.33$ N·m, meaning the model recovers an action in the opposite direction
from the true action.

This document provides a rigorous explanation of why this gap exists and why
it is, to a significant degree, fundamental to the problem structure rather
than a mere artifact of insufficient training.

---

## 2. Mathematical Formalization

### 2.1 Notation

- $s \in \mathcal{S} \subset \mathbb{R}^3$: normalized pendulum state $[\cos\theta,\; \sin\theta,\; \dot{\theta}/8]$
- $a \in \mathcal{A} = [-1, 1]$: normalized torque (raw torque divided by 2)
- $f_{\text{true}}: \mathcal{S} \times \mathcal{A} \to \mathcal{S}$: the true (unknown) dynamics
- $f_\theta: \mathcal{S} \times \mathcal{A} \to \mathcal{S}$: the learned KAN world model
- $J_f(s, a) = \frac{\partial f}{\partial a}(s, a) \in \mathbb{R}^3$: the Jacobian of $f$ with respect to the action
- $\varepsilon(s, a) = f_\theta(s, a) - f_{\text{true}}(s, a) \in \mathbb{R}^3$: the forward prediction error

### 2.2 Two Fundamentally Different Optimization Problems

**Training (forward):**

$$\theta^* = \arg\min_\theta \sum_i \|f_\theta(s_i, a_i) - s'_i\|^2$$

This minimizes the discrepancy between predicted and observed next states
*at the data points where actions are given*.  The decision variable is
$\theta$ (model parameters).  The loss is evaluated at known $(s, a)$ pairs.

**Deployment (inverse):**

$$a^* = \arg\min_a \|f_{\theta^*}(s, a) - s'_{\text{target}}\|^2$$

This minimizes the discrepancy between the predicted next state (using a
*to-be-determined* action) and a target state.  The decision variable is
$a$ (the action).  The loss must be evaluated at points where $a$ is not
known a priori — the optimizer explores the action space guided by the
model's geometry.

**The critical distinction**: training evaluates $f_\theta$ at *given*
$(s,a)$ pairs drawn from the data distribution.  Deployment evaluates
$f_{\theta^*}$ at *arbitrary* $a$ values chosen by an optimizer that trusts
the model's local geometry.  These two modes place fundamentally different
demands on the model.

---

## 3. Root Cause 1: Derivative Mismatch (The Trainable Gap)

### 3.1 The Optimizer Uses Derivatives, Not Just Function Values

Gradient descent on the inverse objective updates the action via:

$$a^{(k+1)} = a^{(k)} - \eta \cdot \underbrace{J_f(s, a^{(k)})^T}_{\text{Jacobian}} \cdot \underbrace{(f_\theta(s, a^{(k)}) - s'_{\text{target}})}_{\text{prediction error}}$$

The update direction is determined by two factors multiplied together: the
local Jacobian $J_f$ and the current prediction error.  Even if the
prediction error at the true action is small, the optimizer may never reach
$a_{\text{true}}$ if the Jacobian points elsewhere.

### 3.2 Why Standard Training Fails to Constrain the Jacobian

The MSE loss used in training is:

$$\mathcal{L}_{\text{MSE}}(\theta) = \|f_\theta(s, a) - s'\|^2$$

Its gradient with respect to model parameters is:

$$\frac{\partial \mathcal{L}_{\text{MSE}}}{\partial \theta} = 2(f_\theta(s, a) - s')^T \cdot \frac{\partial f_\theta}{\partial \theta}$$

This gradient contains no term involving $\frac{\partial f_\theta}{\partial a}$.
The training signal never directly penalizes an incorrect Jacobian.  A model
can achieve arbitrarily low MSE while having an arbitrarily wrong Jacobian,
so long as the function values match at the training points.

### 3.3 Empirical Evidence on Pendulum-v1

For a trained KAN (v6), we computed the cosine similarity between the model's
Jacobian $\partial f_\theta / \partial a$ and the true Jacobian
$\partial f_{\text{true}} / \partial a$ at several states:

| State description | $\cos\_\text{sim}(J_{\text{model}}, J_{\text{true}})$ |
|------------------|:------------------------------------------------------:|
| Bottom, stationary | $-0.58$ |
| Bottom, swinging | $+0.66$ |
| Upright, stationary | $-0.74$ |
| Mid-angle | $-0.26$ |

A cosine similarity of $-0.58$ means the model's gradient points in a
direction more than $125^\circ$ away from the true gradient.  The optimizer
receives a descent direction that moves it *further* from $a_{\text{true}}$.

### 3.4 Mechanism: Why the Jacobian Can Be Wrong While the Function Value Is Right

Consider a simplified 1D analogy.  Let the true function be $g_{\text{true}}(a) = 0.5a$
(a line through the origin).  Let the learned function be:

$$g_\theta(a) = 0.5a + 0.05 \cdot \sin(15a)$$

At any point $a$, the function value error is at most $0.05$ — negligible
by most standards.  But the derivative is:

$$g'_\theta(a) = 0.5 + 0.75 \cdot \cos(15a)$$

This oscillates between $-0.25$ and $1.25$, crossing zero and changing sign
repeatedly.  The forward error is small, but the derivative is wrong by up
to $150\%$ of the true value and frequently points in the opposite direction.

The mathematical reason is that a highly oscillatory perturbation of small
amplitude can have a derivative of large amplitude.  The $L^\infty$ norm
bounds the function value but not the derivative — this is a fundamental
property of function spaces.  Formally, for any $\epsilon > 0$ and any
$M > 0$, there exists a function $h$ such that $\|h\|_\infty \le \epsilon$
but $\|h'\|_\infty \ge M$.  The standard example is
$h(a) = \epsilon \cdot \sin(M a / \epsilon)$.

In the KAN specifically, this oscillation is realized through the B-spline
control points.  Each edge function $\phi(x) = w \cdot (\text{silu}(x) + \sum_k c_k B_k(x))$
has control points $c_k$ that can oscillate independently within their local
support intervals while the overall function remains close to the target.
Empirical measurement on the v6 model confirms this: across all 84 edges,
$68\%$ of adjacent control point differences have alternating signs, and
$85$–$94\%$ of edges exhibit high curvature (mean $|\Delta^2 c| > 0.1$).

### 3.5 This Gap Is Trainable

Root Cause 1 can be addressed by augmenting the training loss with a term
that explicitly penalizes Jacobian mismatch:

$$\mathcal{L} = \mathcal{L}_{\text{MSE}} + \lambda \cdot \left\|\frac{\partial f_\theta}{\partial a} - \frac{\partial f_{\text{true}}}{\partial a}\right\|^2$$

This is the Sobolev training direction discussed in `FITTING_DEPTH.md`.
It directly constrains the model's local geometry to match the true system's
local geometry.

---

## 4. Root Cause 2: The Loss Landscape Shift (The Residual Gap)

### 4.1 Even with Perfect Derivatives, the Minimum Shifts

Assume, hypothetically, that training has achieved $D \ge 1$ fitting depth:
the model's Jacobian matches the true Jacobian everywhere.  Then by the
fundamental theorem of calculus, for any fixed $s$:

$$f_\theta(s, a) - f_\theta(s, a_{\text{true}}) = \int_{a_{\text{true}}}^a \frac{\partial f_\theta}{\partial a}(s, t)\,dt = \int_{a_{\text{true}}}^a \frac{\partial f_{\text{true}}}{\partial a}(s, t)\,dt = f_{\text{true}}(s, a) - f_{\text{true}}(s, a_{\text{true}})$$

Rearranging:

$$f_\theta(s, a) - f_{\text{true}}(s, a) = f_\theta(s, a_{\text{true}}) - f_{\text{true}}(s, a_{\text{true}}) \equiv \varepsilon_0(s)$$

The forward prediction error is a function of $s$ only — it does not depend
on $a$.  Denote this constant offset as $\varepsilon_0$.  (Here "constant"
means constant with respect to $a$, not constant across all states.)

Now examine the inverse objective at the true action:

$$\frac{\partial}{\partial a}\left[ \|f_\theta(s, a) - s'_{\text{true}}\|^2 \right]_{a = a_{\text{true}}} = 2 \cdot J_f(s, a_{\text{true}})^T \cdot \underbrace{(f_\theta(s, a_{\text{true}}) - s'_{\text{true}})}_{= \varepsilon_0}$$

Since $s'_{\text{true}} = f_{\text{true}}(s, a_{\text{true}})$, the residual
at the true action is exactly $\varepsilon_0$.  If $\varepsilon_0 \neq 0$
and $J_f^T \varepsilon_0 \neq 0$, then **$a_{\text{true}}$ is not a stationary
point of the inverse objective**.  The optimizer will move away from
$a_{\text{true}}$ in search of a lower loss.

### 4.2 Geometric Interpretation

The inverse objective $\mathcal{L}_{\text{inv}}(a) = \|f_\theta(s, a) - s'_{\text{true}}\|^2$
measures the squared Euclidean distance between the model's predicted next
state and the target.  The set $\{f_\theta(s, a) : a \in [-1, 1]\}$ is a
1-dimensional curve (a manifold) embedded in the 3-dimensional state space,
parameterized by the scalar action $a$.

The true action $a_{\text{true}}$ maps to the point $f_\theta(s, a_{\text{true}})$
on this curve.  The target $s'_{\text{true}}$ is a point in the ambient
3D space.  The optimizer seeks the point on the model's curve that is closest
(in Euclidean distance) to the target.

If the model's curve is offset from the true curve by $\varepsilon_0$, then
the orthogonal projection of $s'_{\text{true}}$ onto the model's curve will
generally correspond to an action $a^* \neq a_{\text{true}}$.

### 4.3 Concrete Illustration

Consider a state $s$ where the true dynamics are:

$$f_{\text{true}}(s, a) = \begin{bmatrix} 0 \\ 0 \\ 0.0375 \end{bmatrix} \cdot a + \begin{bmatrix} c_1 \\ c_2 \\ c_3 \end{bmatrix}$$

(the system is approximately linear in $a$ over a small range).  Suppose the
trained model is:

$$f_\theta(s, a) = f_{\text{true}}(s, a) + \begin{bmatrix} 0.01 \\ -0.02 \\ 0.005 \end{bmatrix}$$

The forward error is $\|\varepsilon_0\| \approx 0.023$ — small.  But the
inverse objective at $a$ is:

$$\mathcal{L}_{\text{inv}}(a) = \left\|\begin{bmatrix} 0 \\ 0 \\ 0.0375 \end{bmatrix} (a - a_{\text{true}}) + \varepsilon_0\right\|^2$$

Taking the derivative and setting to zero yields the optimal action:

$$a^* = a_{\text{true}} - \frac{J_{\text{true}}^T \varepsilon_0}{\|J_{\text{true}}\|^2}$$

With $J_{\text{true}} = [0, 0, 0.0375]^T$ and $\varepsilon_0 = [0.01, -0.02, 0.005]^T$:

$$J_{\text{true}}^T \varepsilon_0 = 0.0375 \times 0.005 = 0.0001875$$
$$\|J_{\text{true}}\|^2 = 0.0375^2 \approx 0.001406$$
$$a^* - a_{\text{true}} \approx -0.133 \quad \text{(normalized)}$$

This corresponds to a torque error of $0.27$ N·m.  The forward error
component parallel to the Jacobian — just $0.005$ out of a total forward
error norm of $0.023$ — is what shifts the minimum.  The orthogonal
components contribute to the loss value at the minimum but not to its
location.

### 4.4 This Gap Is Not Fully Trainable

Root Cause 2 can be reduced by further decreasing $\varepsilon_0$ — i.e.,
improving forward prediction accuracy.  But $\varepsilon_0$ can never be
exactly zero for a learned model on finite data.  There will always be some
residual, and that residual will always shift the inverse optimum away from
$a_{\text{true}}$ to some degree.

---

## 5. Root Cause 3: Underactuation and Ill-Conditioning (The Structural Gap)

### 5.1 The Magnification Factor

The shift in the optimal action derived above is:

$$|a^* - a_{\text{true}}| \approx \frac{|J^T \varepsilon_0|}{\|J\|^2}$$

The denominator $\|J\|^2$ is the squared norm of the Jacobian — how much
the predicted next state changes per unit change in action.  When this
quantity is small, even a tiny forward error $\varepsilon_0$ can cause a
large shift in the optimal action.

### 5.2 The Pendulum Is Underactuated

The pendulum state space is 3-dimensional ($\cos\theta$, $\sin\theta$, $\dot{\theta}$),
but control is exercised through a single scalar torque.  The mapping
$a \mapsto f(s, a)$ has a 1-dimensional image — a curve in 3D space.

The true Jacobian for the pendulum (derived analytically from the dynamics
and verified against finite differences) is:

$$\frac{\partial}{\partial a_{\text{norm}}} \begin{bmatrix} \cos\theta' \\ \sin\theta' \\ \dot{\theta}'/8 \end{bmatrix} = \begin{bmatrix} -0.015 \cdot \sin\theta' \\ 0.015 \cdot \cos\theta' \\ 0.0375 \end{bmatrix}$$

The norm of this Jacobian is bounded:

$$\|J_{\text{true}}\|^2 = (0.015)^2 \cdot (\sin^2\theta' + \cos^2\theta') + (0.0375)^2 = 0.000225 + 0.001406 = 0.001631$$

So $\|J_{\text{true}}\| \approx 0.0404$.  This is small because:

1. The position components $(\cos\theta, \sin\theta)$ are only *indirectly*
   affected by torque — torque changes $\dot{\theta}$, which then integrates
   to change $\theta$, which then changes $\cos\theta$ and $\sin\theta$.
   Over a single 0.05 s timestep, this indirect effect is tiny ($0.015$).

2. The velocity component $\dot{\theta}$ is directly affected, but the
   coefficient ($0.0375$ in normalized units) reflects the physical parameters
   (moment of inertia, timestep).

### 5.3 The Amplification Ratio

Given $\|J\| \approx 0.0404$ and a forward RMSE of $\varepsilon \approx 0.037$,
the expected inverse error due to Root Cause 2 alone is:

$$\mathbb{E}[|a^* - a_{\text{true}}|] \approx \frac{\|\varepsilon\|}{\|J\|} \approx \frac{0.037}{0.0404} \approx 0.92$$

in normalized action space, corresponding to $1.84$ N·m.  This is consistent
with the empirically observed mean inverse error of $0.87$ N·m.

The key ratio $\|\varepsilon\| / \|J\|$ is the **amplification factor** that
maps forward prediction error to inverse recovery error.  When the system is
strongly underactuated (small $\|J\|$), this factor is large.

### 5.5 Comparison with a Well-Actuated System

Consider a hypothetical 2D point-mass system where $s' = s + a$ (the "linear"
case from the KAN-RF prototype, Stage 1).  Here $J = I_{2 \times 2}$ and
$\|J\| = 1$.  The amplification factor is $\|\varepsilon\| / 1 \approx \|\varepsilon\|$.
Forward error and inverse error are of the same order.

In the IDEAR.md Stage 1 results for this system: forward MSE was $0.00038$
and inverse error $|a_{\text{pred}} - a_{\text{true}}|$ was $0.012$.  Both
are small and consistent — because the Jacobian is large relative to the
error.

The pendulum is fundamentally different: the action has only an indirect,
attenuated effect on most of the state, making the inverse problem
ill-conditioned.

---

## 6. Synthesis: Why the Gap Persists

The forward-inverse gap arises from three compounding factors, ordered from
most to least remediable:

| Root Cause | Nature | Remediability |
|-----------|--------|:---:|
| 1. Jacobian mismatch | The model's local geometry (derivative) is wrong even though function values are right. | **Partially remediable** via Sobolev training / P-spline regularization (see `FITTING_DEPTH.md`) |
| 2. Residual forward error | $\varepsilon_0 \neq 0$ shifts the inverse objective's minimum away from $a_{\text{true}}$. | **Reducible but ineliminable** — finite data and model capacity guarantee $\varepsilon_0 > 0$ |
| 3. Underactuation amplification | Small $\|J\|$ means small $\varepsilon_0$ produces large $|a^* - a_{\text{true}}|$. | **Structural and irremediable** — it is a property of the physical system, not the model |

The third factor is the most fundamental.  For the pendulum, $\|J\| \approx 0.04$
sets a hard lower bound on inverse error given any nonzero forward error.  No
amount of training can make $\varepsilon_0 = 0$ exactly.  Therefore, some
degree of inverse inaccuracy is unavoidable for this system when using
single-step gradient-based inversion through a learned forward model.

### 6.1 What CAN Be Improved

- **Derivative accuracy** (Root Cause 1): Sobolev training and P-spline
  regularization can make the optimizer's search direction correct, reducing
  the number of iterations needed and preventing convergence to spurious
  local minima caused by sign errors in the Jacobian.

- **Forward accuracy** (Root Cause 2): More data, better coverage, and
  iterative bootstrapping (DAgger) can reduce $\varepsilon_0$, narrowing
  the gap.

### 6.2 What CANNOT Be Eliminated

- **The amplification factor** $\|\varepsilon\| / \|J\|$: As long as
  $\varepsilon > 0$ and $\|J\|$ is small, the inverse error has a positive
  lower bound.  This is a consequence of the physics (underactuation) and
  the finite capacity of any learned model.

---

## 7. Practical Implications for the KAN-RF Project

### 7.1 Why Multi-Step Control Works Despite the Gap

The current best-performing controller (exp_F, 8/10 success) does not rely
on accurate single-step inverse recovery.  It uses:

1. **Energy-guided single-step optimization**: provides a reasonable (not
   perfect) action direction at each step.

2. **Online learning with three-factor learning rate**: corrects model
   errors using real observed transitions, gradually reducing $\varepsilon_0$
   in the regions the controller actually visits.

3. **Smart burst**: when deviation exceeds a threshold, actively probes the
   model at its decision boundary with boosted learning rate, providing
   targeted correction at the most critical regions.

These mechanisms work *around* the forward-inverse gap rather than closing it.
Each control step receives real environmental feedback before the next
decision, so the single-step inverse error does not accumulate into a
multi-step trajectory divergence — it gets corrected online.

### 7.2 Why the Project's Core Narrative Still Holds

The narrative is: KAN's B-spline local support provides natural
anti-forgetting during online adaptation, unlike MLPs which require replay
buffers or ensembles.  This narrative does not depend on perfect inverse
recovery — it depends on the model being *correctable* through local updates
without destroying previously learned knowledge.

The forward-inverse gap means that a KAN (or any learned forward model) will
never be a perfect inverse on its first attempt for an underactuated system.
But the gap's severity — how far off the first guess is, and how many
correction steps are needed — is where KAN's structural properties (and the
fitting depth improvements discussed in `FITTING_DEPTH.md`) make a difference.

---

## Appendix: Key Concepts Explained

### A.1 What Is a Jacobian?

The Jacobian of a vector-valued function $f: \mathbb{R}^n \to \mathbb{R}^m$
is the $m \times n$ matrix of all first-order partial derivatives.  For our
case, $f$ maps a scalar action $a$ to a 3D state change, so the Jacobian is
a $3 \times 1$ column vector:

$$J_f(a) = \frac{\partial f}{\partial a} = \begin{bmatrix} \partial f_1 / \partial a \\ \partial f_2 / \partial a \\ \partial f_3 / \partial a \end{bmatrix}$$

It answers the question: "if I increase the action by a tiny amount, in which
direction (in the 3D state space) does the predicted next state move, and how
fast?"

### A.2 What Is a Manifold in This Context?

A manifold is a smooth surface (or curve) embedded in a higher-dimensional
space.  The set of all possible next-state predictions $\{f_\theta(s, a) : a \in [-1, 1]\}$
is a 1-dimensional manifold (a curve) inside the 3-dimensional state space,
because it is parameterized by a single scalar $a$.  The optimizer searching
over $a$ is essentially walking along this curve, trying to get as close as
possible to the target point $s'_{\text{true}}$ which lies somewhere in the
ambient 3D space.

### A.3 What Does "Underactuated" Mean?

An underactuated system has fewer independent control inputs than degrees of
freedom.  The pendulum has 2 degrees of freedom (angle $\theta$ and angular
velocity $\dot{\theta}$) but only 1 control input (torque).  This means the
controller cannot independently specify both the position and velocity at
the next timestep — it can only influence them through the coupled dynamics.

In the forward-inverse context, underactuation means the Jacobian has a
small norm: the action can only move the state in a limited set of
directions, and the magnitude of this movement per unit action is constrained
by the physics.

### A.4 What Does "Ill-Conditioned" Mean?

A problem is ill-conditioned when small changes in the input (here, the
forward prediction error $\varepsilon$) produce large changes in the output
(here, the recovered action $a^*$).  The condition number is the ratio of
output sensitivity to input sensitivity.  For the inverse problem, the
effective condition number is $1 / \|J\|$: when $\|J\|$ is small, the
inverse is ill-conditioned.

### A.5 What Is a Stationary Point?

A stationary point of a function is a point where the gradient is zero.
For the inverse objective $\mathcal{L}_{\text{inv}}(a)$, a stationary point
$a^*$ satisfies $\nabla_a \mathcal{L}_{\text{inv}}(a^*) = 0$.  Gradient
descent converges to a stationary point (typically a local minimum).  If
$a_{\text{true}}$ is not a stationary point — i.e., the gradient is nonzero
there — the optimizer will not stay at $a_{\text{true}}$.

---

*Document started 2026-05-23.*
