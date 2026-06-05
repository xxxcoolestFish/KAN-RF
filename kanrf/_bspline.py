import torch


def extend_grid(grid: torch.Tensor, k: int) -> torch.Tensor:
    """Extend uniform grid with k points on each side for B-spline evaluation."""
    h = grid[1] - grid[0]
    left = torch.arange(k, 0, -1, device=grid.device, dtype=grid.dtype) * (-h) + grid[0]
    right = torch.arange(1, k + 1, device=grid.device, dtype=grid.dtype) * h + grid[-1]
    return torch.cat([left, grid, right])


def bspline_basis(x: torch.Tensor, grid: torch.Tensor, k: int) -> torch.Tensor:
    """Evaluate B-spline basis of order k at points x  (VECTORIZED Cox-de Boor).

    Args:
        x:      (*,) input values — arbitrary batch shape
        grid:   (G+1,) uniform grid points on [a, b]
        k:      spline order (degree = k)

    Returns:
        basis:  (*, G+k) B-spline basis functions B_i(x)
    """
    *batch_dims, = x.shape
    x_flat = x.reshape(-1)  # (N,)
    device, dtype = x.device, x.dtype

    t = extend_grid(grid, k)  # (G+1+2k,)
    G = len(grid) - 1
    n_basis = G + k

    # ── Degree-0 basis ──
    n0 = len(t) - 1  # G + 2k
    left = t[:n0]
    right = t[1:n0 + 1]

    x2d = x_flat.unsqueeze(-1)  # (N, 1)
    basis = ((x2d >= left) & (x2d < right)).to(dtype)  # (N, n0)

    # Handle x == t[-1]
    at_right = (x2d == t[-1])
    if at_right.any():
        basis[:, n0 - 1] = basis[:, n0 - 1] + at_right.squeeze(-1).to(dtype)
        basis = basis.clamp(0, 1)

    # ── Vectorized Cox-de Boor recursion for degrees 1..k ──
    for d in range(1, k + 1):
        current_n = basis.shape[1]  # starts at G+2k, decreases by 1 each step
        new_n = current_n - 1

        t_i0 = t[:new_n]               # t[i]
        t_id = t[d:d + new_n]          # t[i+d]
        t_i1 = t[1:new_n + 1]          # t[i+1]
        t_id1 = t[d + 1:d + new_n + 1]  # t[i+d+1]

        denom1 = t_id - t_i0           # (new_n,)
        denom2 = t_id1 - t_i1          # (new_n,)

        safe1 = denom1.abs() > 1e-12
        safe2 = denom2.abs() > 1e-12

        x2d = x_flat.unsqueeze(1)  # (N, 1)

        term1 = torch.where(
            safe1.unsqueeze(0),
            (x2d - t_i0.unsqueeze(0)) / denom1.clamp(1e-12).unsqueeze(0) * basis[:, :new_n],
            torch.zeros_like(basis[:, :new_n])
        )
        term2 = torch.where(
            safe2.unsqueeze(0),
            (t_id1.unsqueeze(0) - x2d) / denom2.clamp(1e-12).unsqueeze(0) * basis[:, 1:new_n + 1],
            torch.zeros_like(basis[:, 1:new_n + 1])
        )

        basis = term1 + term2

    # basis: (N, G+k) → (*batch_dims, G+k)
    return basis.reshape(*batch_dims, n_basis)


def bspline_derivative(x: torch.Tensor, grid: torch.Tensor, k: int,
                       dc: torch.Tensor) -> torch.Tensor:
    """Evaluate first derivative of B-spline sum at points x.

    f(x) = Σ c_i B_i^k(x)  →  f'(x) = (1/h) Σ (Δc_i) B_i^{k-1}(x)

    Args:
        x:   (*,) input values
        grid: (G+1,) uniform grid
        k:   spline order
        dc:  (*, n_basis) first differences of control points Δc_i = c_i - c_{i-1}
             For n_basis = G+k, dc should have n_basis entries (dc[0] is ignored).

    Returns:
        f_prime: (*,) derivative values
    """
    h = grid[1] - grid[0]
    # Evaluate order-(k-1) basis at x
    reduced_basis = bspline_basis(x, grid, k - 1)  # (*, G+k-1)

    # dc has G+k entries; reduced_basis has G+k-1 entries.
    # The formula uses dc[1:] (differences starting from i=1).
    # dc shape: (..., G+k), reduced_basis shape: (..., G+k-1)
    dc_shifted = dc[..., 1:]  # (..., G+k-1)

    f_prime = (1.0 / h) * (dc_shifted * reduced_basis).sum(dim=-1)
    return f_prime
