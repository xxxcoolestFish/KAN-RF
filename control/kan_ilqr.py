"""KAN-iLQR: Local linear model from KAN + iLQR trajectory optimization.

KAN provides: drift f(s,0), state Jacobian A=∂f/∂s, control Jacobian B=∂f/∂a
iLQR uses this to compute a stable feedback policy.
Forward line search on real env (or high-precision model).

This solves the drift problem of Jacobian pseudo-inverse by properly
modeling the state's natural evolution (gravity, damping, etc.)
"""
import torch
import numpy as np


def kan_ilqr(knowledge, s_current, s_target, horizon=8,
             max_iter=10, lambda_init=1.0, lambda_factor=1.5,
             lambda_max=1000.0, tol=1e-3,
             rollout_fn=None, g=10.0):
    """iLQR using KAN's local linear model, line search on real env.

    Args:
        knowledge: KANKnowledge instance
        s_current: (state_dim,) current normalized state
        s_target: (state_dim,) target normalized state
        horizon: planning horizon
        max_iter: max iLQR iterations
        lambda_init: initial regularization
        rollout_fn: fn(s_norm, actions) → states array for line search.
                    If None, uses KAN's own nonlinear rollout.
        g: gravity (for env rollout)

    Returns:
        best_action: scalar action ∈ [-1, 1]
        info: dict with convergence details
    """
    state_dim = len(s_current)
    action_dim = 1  # Pendulum

    device = knowledge.device
    s_target_t = torch.tensor(s_target, dtype=torch.float32, device=device)

    # ── 1. Get KAN linear model at current state ──
    s_t = torch.tensor(s_current, dtype=torch.float32, device=device).unsqueeze(0)
    s_drift, A, B = knowledge.get_linear_model(s_t)
    # s_drift: (1, state_dim), A: (state_dim, state_dim), B: (state_dim, action_dim)

    s_drift_np = s_drift.squeeze(0).cpu().numpy()
    A_np = A.cpu().numpy()  # (state_dim, state_dim)
    B_np = B.cpu().numpy()  # (state_dim, action_dim)

    # ── 2. Initialize action sequence ──
    # Use Jacobian pseudo-inverse as warm start
    B_vec = B_np.squeeze()  # (state_dim,)
    B_norm_sq = float(np.dot(B_vec, B_vec)) + 1e-4
    init_a = float(np.dot(B_vec, s_target - s_current)) / B_norm_sq
    init_a = np.clip(init_a, -1.0, 1.0)

    actions = np.full((horizon, action_dim), init_a * 0.5)

    # ── 3. Cost matrices ──
    Q = np.diag([10.0, 10.0, 1.0])   # state cost: angle matters more
    R = np.array([[0.1]])              # control cost
    Qf = np.diag([50.0, 50.0, 5.0])  # terminal cost

    # ── 4. iLQR iterations ──
    lamb = lambda_init
    best_cost = float('inf')
    best_actions = actions.copy()

    for iteration in range(max_iter):
        # ── Forward pass (on linear model for gradient computation) ──
        states_lin = [s_current.copy()]
        for h in range(horizon):
            s_h = states_lin[-1]
            ds = s_h - s_current
            s_next = s_drift_np + A_np @ ds + B_np @ actions[h]
            states_lin.append(s_next)
        states_lin = np.array(states_lin)

        # ── Compute cost + gradients ──
        total_cost = 0.0
        s_cost = np.zeros((horizon+1, state_dim))   # ∂cost/∂s
        s_hess = np.zeros((horizon+1, state_dim, state_dim))  # ∂²cost/∂s²

        # Terminal cost
        ds_T = states_lin[-1] - s_target
        total_cost += 0.5 * ds_T @ Qf @ ds_T
        s_cost[-1] = Qf @ ds_T
        s_hess[-1] = Qf

        # Running cost
        for h in range(horizon):
            ds_h = states_lin[h+1] - s_target
            total_cost += 0.5 * ds_h @ Q @ ds_h + 0.5 * actions[h] @ R @ actions[h]
            s_cost[h+1] += Q @ ds_h

        # ── Backward pass (Riccati recursion) ──
        V_s = s_cost[-1].copy()    # value function gradient
        V_ss = s_hess[-1].copy()   # value function Hessian

        k_feedforward = np.zeros((horizon, action_dim))     # open-loop correction
        K_feedback = np.zeros((horizon, action_dim, state_dim))  # feedback gain

        backward_failed = False

        for h in range(horizon-1, -1, -1):
            # Q-function derivatives
            Q_s = s_cost[h+1] + A_np.T @ V_s
            Q_a = R @ actions[h] + B_np.T @ V_s

            Q_ss = s_hess[h+1] if h+1 < horizon else Qf
            Q_aa = R + B_np.T @ V_ss @ B_np
            Q_sa = A_np.T @ V_ss @ B_np

            # Regularize Q_aa to ensure positive definiteness
            Q_aa_reg = Q_aa + lamb * np.eye(action_dim)

            try:
                Q_aa_inv = np.linalg.inv(Q_aa_reg)
            except np.linalg.LinAlgError:
                backward_failed = True
                break

            # Optimal control law
            k_feedforward[h] = -Q_aa_inv @ Q_a
            K_feedback[h] = -Q_aa_inv @ Q_sa.T

            # Update value function
            V_s = Q_s - K_feedback[h].T @ Q_aa @ k_feedforward[h]
            V_ss = Q_ss - K_feedback[h].T @ Q_aa @ K_feedback[h]

        if backward_failed:
            lamb *= lambda_factor
            continue

        # ── Forward line search on REAL ENV ──
        new_actions = np.zeros_like(actions)
        for h in range(horizon):
            ds = states_lin[h] - s_current
            new_actions[h] = actions[h] + k_feedforward[h] + K_feedback[h] @ ds
            new_actions[h] = np.clip(new_actions[h], -1.0, 1.0)

        # Evaluate on real env
        if rollout_fn is not None:
            states_real = rollout_fn(s_current, new_actions.squeeze(), g=g)
        else:
            # Fallback: use KAN rollout
            s_k = torch.tensor(s_current, dtype=torch.float32, device=device).unsqueeze(0)
            states_real = [s_current.copy()]
            for h in range(horizon):
                a_t = torch.tensor(new_actions[h:h+1], dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    s_k = knowledge.kan(torch.cat([s_k, a_t], dim=-1))
                states_real.append(s_k.squeeze(0).cpu().numpy())
            states_real = np.array(states_real)

        # Real cost
        real_cost = 0.0
        ds_T_real = states_real[-1] - s_target
        real_cost += 0.5 * ds_T_real @ Qf @ ds_T_real
        for h in range(horizon):
            ds_h = states_real[h+1] - s_target
            real_cost += 0.5 * ds_h @ Q @ ds_h + 0.5 * new_actions[h] @ R @ new_actions[h]

        # ── Accept or reject ──
        if real_cost < best_cost:
            best_cost = real_cost
            best_actions = new_actions.copy()
            actions = new_actions.copy()
            lamb = max(lamb / lambda_factor, 1e-6)

            # Check convergence
            if iteration > 0 and abs(real_cost - total_cost) / (abs(total_cost) + 1e-8) < tol:
                break
        else:
            lamb = min(lamb * lambda_factor, lambda_max)

            # If line search keeps failing, accept anyway with small step
            if lamb >= lambda_max:
                actions = actions + 0.1 * (new_actions - actions)
                lamb = lambda_init

    # ── Return first action ──
    return float(np.clip(best_actions[0, 0], -1.0, 1.0)), {
        'method': 'ilqr',
        'iterations': iteration + 1,
        'cost': float(best_cost),
        'horizon': horizon,
    }
