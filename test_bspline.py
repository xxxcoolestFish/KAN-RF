"""Verify B-spline basis properties."""
import torch
from kanrf import bspline_basis, extend_grid

# Setup
G, k = 5, 3
grid = torch.linspace(-1, 1, G + 1)  # G+1 points
print(f"Grid: {grid}")
print(f"Expected basis count: G+k = {G}+{k} = {G+k}")

# Test points across the domain
x = torch.linspace(-1, 1, 50)
basis = bspline_basis(x, grid, k)
print(f"\nBasis shape: {basis.shape}")  # should be (50, G+k) = (50, 8)

# 1. Non-negativity
assert (basis >= -1e-8).all(), "FAIL: negative values found"
print("✓ All basis values >= 0")

# 2. Partition of unity: each row should sum to ~1
row_sums = basis.sum(dim=-1)
assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), \
    f"FAIL: partition of unity violated, max dev: {(row_sums - 1).abs().max()}"
print("✓ Partition of unity holds")

# 3. Local support: for any x, at most k+1 basis functions are active
active_per_x = (basis > 1e-8).sum(dim=-1)
max_active = active_per_x.max().item()
print(f"✓ Max active bases per point: {max_active} (expected <= {k+1})")
assert max_active <= k + 1, f"FAIL: too many active bases"

# 4. Check specific value at x=0
x0 = torch.tensor([0.0])
b0 = bspline_basis(x0, grid, k)
print(f"\nBasis at x=0: {b0.squeeze().tolist()}")
print(f"Sum at x=0: {b0.sum():.6f}")

# 5. Test gradient
x.requires_grad_(True)
basis2 = bspline_basis(x, grid, k)
loss = basis2.sum()
loss.backward()
print(f"\n✓ Gradients computed, max |grad|: {x.grad.abs().max():.4f}")

print("\n=== All checks passed ===")
