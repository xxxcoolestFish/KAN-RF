# Decision Network v3: KAN-Adapted Policy

## Core Idea

**KAN provides gradients, not features.**

Previous approaches (v1, v2) tried to use KAN for inverse optimization or
feature extraction — both hit the Forward-Inverse Gap (root cause 3).

v3 changes the paradigm: KAN serves as a **differentiable judge** during
policy training. The gradient flows through KAN's accurate Jacobian to tell
the policy network "which direction should the action move?"

```
 Training:  s → [π_θ] → a → [frozen KAN] → s'_pred → L(s'_pred, s*)
            gradient flows: L → s'_pred → KAN → a → π_θ

 Deployment: s → [π_θ] → a   (pure forward, KAN not involved)
```

## Why This Avoids the Forward-Inverse Gap

Root cause 3 (underactuated amplification) is deadly for inverse optimization
because it requires **exact** inverse: a* = argmin ||f(s,a) - s*||²

v3 only needs **approximate gradient direction**: ∂L/∂a = 2J^T(f(s,a) - s*)
As long as J points roughly in the right direction (cos_sim > 0, which CWS
training guarantees at 0.92), the gradient pushes a toward improvement.

## Quick Start

```bash
# 1. Train policy using frozen KAN world model
python train.py --wm ../path/to/hybrid_kan.pt --epochs 200

# 2. Evaluate on Pendulum
python test_pendulum.py --policy kan_policy_v3.pt --trials 10

# 3. Compare against baselines
python test_pendulum.py --policy kan_policy_v3.pt --compare-all
```

## Architecture

- **Policy π_θ**: MLP([3, 64, 64, 1]) with tanh output (~4.5k params)
- **KAN f**: Frozen [4, 12, 3] world model (756 params, CWS-trained)
- **Training**: Adam on π_θ params only; KAN frozen, gradient passes through

## Files

- `design.md` — Full design rationale and comparison with v1/v2
- `core.py` — KANPolicy, KANGradientTrainer, KANMultiStepTrainer, KANDensityWeight
- `train.py` — Training script
- `test_pendulum.py` — Pendulum evaluation + baseline comparison
