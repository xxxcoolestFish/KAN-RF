"""Verify KAN layer forward pass and gradient flow."""
import torch
from kanrf import KANLayer

# Test 1: Basic forward pass
layer = KANLayer(in_dim=4, out_dim=2, grid_size=5, spline_order=3)
x = torch.randn(8, 4)
y = layer(x)
print(f"Input:  {x.shape}")
print(f"Output: {y.shape}")
assert y.shape == (8, 2), f"Expected (8, 2), got {y.shape}"
print("✓ Forward pass shape correct")

# Test 2: All finite
assert y.isfinite().all(), "NaN/Inf in output"
print("✓ All values finite")

# Test 3: Gradient flows back through all parameters
loss = y.sum()
loss.backward()
for name, p in layer.named_parameters():
    assert p.grad is not None, f"FAIL: no grad for {name}"
    assert p.grad.isfinite().all(), f"FAIL: NaN/Inf grad for {name}"
print(f"✓ Gradients flow through base_weight ({layer.base_weight.grad.abs().max():.4f})")
print(f"✓ Gradients flow through spline_weight ({layer.spline_weight.grad.abs().max():.4f})")

# Test 4: Output changes when spline_weight changes
with torch.no_grad():
    y1 = layer(x)
    layer.spline_weight.add_(0.5)
    y2 = layer(x)
    layer.spline_weight.sub_(0.5)  # restore
    diff = (y2 - y1).abs().max()
    assert diff > 0, "FAIL: spline parameters don't affect output"
    print(f"✓ Spline params affect output (max diff: {diff:.4f})")

# Test 5: Multiple layers chain (preview of network)
layer1 = KANLayer(4, 4)
layer2 = KANLayer(4, 2)
y = layer2(layer1(x))
assert y.shape == (8, 2)
print("✓ Two-layer chain works")

# Test 6: Parameter count
n_base = 2 * 4  # out_dim * in_dim
n_spline = 2 * 4 * (5 + 3)  # out_dim * in_dim * n_basis
total = n_base + n_spline
actual = sum(p.numel() for p in layer.parameters())
assert actual == total, f"Expected {total} params, got {actual}"
print(f"✓ Parameter count: {total}")

print("\n=== All checks passed ===")
