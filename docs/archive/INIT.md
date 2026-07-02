# KAN-RF Prototype — Build Plan

## Rules (User Mandated)
1. No unwarranted assumptions or simplifications
2. Minimal, clean code
3. Think before coding each step
4. Step by step: implement → verify → next step

## Steps

### Step 1: B-Spline Basis
- Cox-de Boor recursion
- Uniform grid on [a, b]
- Verify: partition of unity, local support, non-negativity

### Step 2: KAN Layer
- φ(x) = w · (SiLU(x) + Σ c_k · B_k(x))
- One set of B-spline params per (input_dim, output_dim) edge
- Verify: forward pass shape, finite values

### Step 3: KAN Network (2-layer)
- Layer 1: in_dim → hidden_dim
- Layer 2: hidden_dim → out_dim
- Verify: end-to-end forward pass

### Step 4: Environment & Training Data
- 2D point-mass: s_{t+1} = s_t + a_t
- Generate (s, a, s') with random actions
- Optional: nonlinear variant for later testing

### Step 5: Phase 1 — Train KAN World Model
- Supervised: (s, a) → s' (or Δs)
- BP + Adam
- Verify: prediction error on held-out data

### Step 6: Phase 2 — Decision via Gradient Descent
- Freeze KAN, given (s_t, s*)
- Gradient descent on a
- Verify: can it find correct action?
