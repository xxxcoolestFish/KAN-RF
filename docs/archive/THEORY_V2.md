# KAN-RF Decision Framework v2

## Problem diagnosed

v1's `intermediate_target` generates a 3D intermediate state `s_mid` and optimizes
`min_a ||f_KAN(s,a) - s_mid||^2`.  Three flaws:

1. **Equal weighting of controllable and uncontrollable dimensions.**  Torque
   directly affects theta_dot but position (cos,sin) only indirectly through
   integration.  The 3D L2 loss forces the optimizer to compromise.

2. **Geometric targets contradict physics.**  At the bottom of the swing
   (cos≈-1, sin≈0), d(sin)/dt = cos*theta_dot = -theta_dot while the geometric
   push toward (0,1) asks sin to increase.  The target direction fights the
   natural dynamics.

3. **Fixed intermediate state loses phase information.**  Swing-up requires
   resonance pumping—torque must align with velocity.  A static s_mid cannot
   encode this phase-dependent requirement.

## Four mathematical approaches surveyed

### 1. Control Lyapunov Functions (CLF)
- **Formulation:** V(s) >= 0, V(s*) = 0.  Control satisfies dV/dt < 0.
  Sontag formula gives universal controller from any CLF.
- **Rigor:** Artstein's theorem proves necessity + sufficiency.
- **Generality:** Universal in theory.  Construction is hard in practice
  (SOS programming for polynomials, neural CLF learning otherwise).
- **For pendulum:** Energy V = (E - E_des)^2 is a natural CLF for swing-up.
- **Key refs:** Sontag (1989, TAC); Chang, Roohi, Gao (NeurIPS 2019).

### 2. Contraction Theory / Riemannian Metrics
- **Formulation:** Riemannian metric M(x) such that geodesic distance between
  any two trajectories decreases exponentially.  LMI: F = M_dot + M J + J^T M
  is negative definite.
- **Rigor:** Exponential convergence guarantee (Lohmiller & Slotine 1998).
- **Generality:** All smooth nonlinear systems.  Finding M requires per-point
  LMI (O(n^3)), tractable via Cholesky parameterization M = L L^T.
- **For our case:** Could learn M_theta as second KAN, but too heavy for
  real-time per-step decision.

### 3. Differential Flatness
- **Formulation:** Flat outputs z such that x = phi(z,dz,...,z^(r)) and
  u = psi(z,dz,...,z^(r+1)).  Gap problem becomes trivial in flat space.
- **Rigor:** Complete when applicable.
- **Generality:** Narrow.  Pendulum is flat (z = theta).  Traffic networks
  are generally NOT flat.  Checking flatness for arbitrary systems is open.
- **Verdict:** Not general enough for our framework.

### 4. Vector Field Guidance (RECOMMENDED)
- **Formulation:** Define desired velocity field v_d(s) on state space.
  Instead of "reach state X", ask "are you moving in direction v_d?"
  Loss = alignment error: -cos_sim(f(s,a)-s, v_d(s)) or ||(f-a)-v_d*dt||.
- **Rigor:** No global convergence guarantee without Lyapunov structure,
  but can be combined with CLF: set v_d = -kappa * grad V.
- **Generality:** Extremely broad—only requires defining v_d.  Underlies
  Riemannian Motion Policies (Ratliff et al. 2018), CBF (Ames et al. 2019),
  and learning from demonstration (Khansari-Zadeh & Billard 2011).
- **Natural fit:** Our Strategy layer already defines a mode-dependent
  direction; converting to a velocity field eliminates s_mid entirely.

## Loss function design options

| Approach | Key idea |
|----------|----------|
| Directional loss | -cos_sim(delta_s_pred, v_des) encourages alignment |
| Controllability decomposition | Weight controllable (theta_dot) >> uncontrollable (cos,sin) |
| Gauss-Newton optimization | Second-order step: (J^T J + lambda I)^(-1) J^T (s_pred - s_target) |
| Task-space control | Reduce 3D matching to 1D energy regulation |
| MPPI-style | Sample actions, weight by softmin of cost |
| Lyapunov CLF-QP | min_a ||a||^2 s.t. grad_V * f(s,a) <= -alpha V |

## Initial action computation: Gauss-Newton warm start

Linearize f at a=0:  f(s,a) ≈ f(s,0) + J_a * a, where J_a = df/da.

Closed-form solution to the linearized problem:
    a_init = J_a^+ * (s_target - f(s,0))

J_a is exact (B-spline analytical derivatives).  Cost: 1 forward + 1 backward
pass.  If f were linear, this is the exact optimum.

## Recommended implementation

Combine **vector field guidance + task-space reduction + controllability
decomposition + Gauss-Newton warm start**:

1. Gap = energy deficit Delta_E = E_des - E  (scalar)
   + direction = grad_E = [0, g, theta_dot]  (3D vector)

2. Desired velocity:  v_des = kappa * Delta_E * grad_E

3. Initial action:  a_init = J_a^+ * v_des * dt  (Gauss-Newton)

4. Loss:
   L = w1 * ||theta_dot_pred - (theta_dot + v_des_theta_dot*dt)||^2   (controllable)
     + w2 * [-cos_sim(delta_s_pred, v_des)]                             (direction alignment)
     + lambda * a^2                                                     (regularization)

5. Refine with Adam (~30 steps from a_init, not 150 from a=0)

## Generality

For a system with state s and known energy-like scalar function E(s)
such that E(s*) = E_des:
  - v_des(s) = kappa * (E_des - E(s)) * grad_s E(s)
  - J_a computed from KAN world model
  - Controllability projector P_C from linearized dynamics at current s
  - Same loss structure applies

For systems where E(s) is not obvious, learn it: train a scalar network
E_theta(s) jointly with the KAN world model to satisfy the Lyapunov
decrease condition: E_theta(f(s,a)) < E_theta(s) when a moves toward goal.
