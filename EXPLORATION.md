# KAN Native Curiosity-Driven Exploration

## 1. Core Idea

**KAN 的 B-样条局部支撑天然携带"知不知道"的信号，无需额外探索模块。**

MLP 做好奇心驱动探索需要额外的 RND / ICM / Random Network 来量化 state novelty。KAN 不需要——训练后，每个 B-样条基函数的激活频率 ρ[j,k] 直接告诉你模型的训练覆盖情况。

核心逻辑：

```
离线训练 KAN → 统计 ρ（每个 B-样条基函数在训练数据中的激活频率）
    ↓
识别欠学习区域：低 ρ 区间 = 模型盲区
    ↓
主动探索：用 shooting 规划轨迹，目标 = 到达低 ρ 区域
    ↓
交互收集 (s, a, s')，加入训练集
    ↓
重训练 → 重新统计 ρ → 循环，直到覆盖充分
```

### 为什么这打破了之前的死循环

之前：模型在顶部不准 → 决策到不了顶部 → 顶部永远没数据 → 模型永远不准。

探索驱动：低 ρ 直接告诉你"顶部数据不足"，然后专门往那走，不依赖模型精度。

## 2. Theoretical Basis

### B-样条局部支撑

B-spline basis $B_k(x)$ has strictly local support: $B_k(x) > 0$ only when $x \in [t_k, t_{k+d+1}]$. For grid=5, order=3: exactly 4 out of 8 basis functions are non-zero for any input.

After training on data $D = \{(s_i, a_i, s'_i)\}$, for each input dimension $j$ and each basis function $k$, compute:

$$\rho_{j,k} = \frac{1}{|D|} \sum_{(s,a) \in D} \mathbb{1}[B_k(\text{input}_j) > 0]$$

$\rho_{j,k}$ is the empirical activation frequency. Low $\rho_{j,k}$ means: few training samples fell in the support interval $[t_k, t_{k+d+1}]$ of that basis function.

### State-Level Uncertainty

For a given state-action pair $(s, a)$, the uncertainty is:

$$U(s, a) = \frac{1}{N_{\text{edges}}} \sum_{l,i,j} \frac{\sigma^2}{\sigma^2 + \|\phi_{l,i,j}(x)\|^2}$$

where $\phi_{l,i,j}$ is the B-spline output on edge $(i,j)$ of layer $l$.

For state-only uncertainty (for target selection):

$$U_{\text{state}}(s) = \mathbb{E}_{a \sim [-\epsilon, \epsilon]}[U(s, a)]$$

### Exploration Target Selection

Grid-search over state space $\mathcal{S}$, pick states with highest $U_{\text{state}}(s)$ as exploration targets.

## 3. Algorithm

### Phase 1: Initial Training + Uncertainty Map

```
1. Train KAN on random data D_0
2. Compute ρ statistics from D_0
3. For each state on a grid over S, compute U_state(s)
4. Visualize uncertainty map → confirm it identifies blind spots
```

### Phase 2: Iterative Exploration

```
For iteration t = 1, 2, ..., T:
    1. Select K exploration targets: states with highest U_state(s)
    2. For each target:
       a. Reset env → get s_0
       b. Use shooting through KAN to plan trajectory toward target
          (even imperfect — the trajectory data is valuable)
       c. Execute with noise, collect (s, a, s') transitions
    3. Add collected data to training set
    4. Retrain KAN (or fine-tune from previous weights)
    5. Recompute ρ and U_state(s)
    6. If coverage sufficient → stop
```

### Shooting with Uncertainty-Guided Exploration

During exploration, the shooting objective is modified:

$$\min_{a_0,...,a_{H-1}} \|s_H - s_{\text{target}}\|^2 + \lambda \sum\|a_h\|^2 \quad \textcolor{red}{+ \beta \cdot U(s_H)}$$

Wait — this doesn't make sense if we WANT to reach uncertain regions. Instead:

**Option A: Pure targeting.** Just shoot toward the low-ρ target. The optimizer naturally finds trajectories; we collect data along the way.

**Option B: Uncertainty bonus.** Add $- \beta \cdot \sum_h U(s_h, a_h)$ to encourage the trajectory to pass through uncertain regions.

Start with Option A (simpler, clearer signal).

## 4. Experiment Design

### Experiment 1: Uncertainty Map Visualization

**Goal**: Verify that ρ correctly identifies the upright region (sin≈1) as under-trained.

**Setup**:
- KAN [4, 12, 3], trained on 5k random transitions
- Compute ρ for all basis functions
- Grid [cos, sin, thd] → compute U_state(s) for each grid point
- Visualize: which regions have high/low uncertainty?

**Expected**: sin≈1, |thd| small → high uncertainty (sparse in random data).
sin≈-1 (bottom) → low uncertainty (dense in random data).

### Experiment 2: Single-Round Exploration

**Goal**: Show that targeting a high-uncertainty region produces useful training data.

**Setup**:
1. Train KAN on 5k random transitions
2. Select top-3 highest-uncertainty states as targets
3. For each target, run 5 exploration episodes:
   - Use shooting (H=15) through KAN to plan toward target
   - Execute with small noise
   - Collect (s, a, s') along the way
4. Add collected data to training set, retrain
5. Compare: model error before vs after on held-out data from the target region

**Metric**: Model error on test data with sin>0.8 (the under-trained region).

### Experiment 3: Iterative Exploration (Full Cycle)

**Goal**: Demonstrate the complete explore-retrain loop.

**Setup**:
- Start: KAN trained on 5k random data
- 5 iterations of: target selection → explore → collect → retrain
- Each iteration: collect ~500 new transitions
- Track: uncertainty map evolution, model error distribution, data coverage

### Experiment 4: Ablation — Exploration vs Random Collection

**Goal**: Show that targeted exploration is more efficient than random data collection.

**Setup**:
- A: Targeted exploration (our method): collect N transitions
- B: Random actions: collect N transitions
- Compare model error improvement per transition collected

**Expected**: A should improve faster, especially in the sparse (sin≈1, thd≈0) region.

## 5. Results

*(To be filled after running experiments)*

### 5.1 Uncertainty Map

### 5.2 Single-Round Exploration

### 5.3 Iterative Exploration

### 5.4 Ablation

## 6. Discussion

### Advantages over MLP-based Exploration

| | RND / ICM | KAN Native |
|---|---|---|
| Extra network | Yes (predictor/target/dynamics) | No |
| Training overhead | Train extra networks | Zero |
| Signal meaning | Implicit (prediction error) | Explicit (training coverage) |
| Architecture | Separate from policy | Same model |

### Limitations

1. **Exploration signal is input-space, not dynamics-space.** ρ tells you which inputs are novel, not which transitions are surprising. A state might be well-covered but the dynamics might still be wrong (systematic error within distribution).

2. **Dimension-dependent.** Grid-based uncertainty scales poorly to high-dimensional inputs. For systems with dim(s) > 10, density estimation may need alternative approaches.

3. **Shooting-dependent exploration.** Using shooting to reach targets assumes the model is accurate enough to plan, which may not hold in very under-trained regions.

### Relationship to Existing Work

- **RND** (Burda et al. 2019): Trains a random network to predict features; prediction error = novelty. Our signal is deterministic and training-free.
- **ICM** (Pathak et al. 2017): Learns a forward dynamics model; error = curiosity. Our signal comes from the world model itself.
- **Count-based exploration** (Bellemare et al. 2016): Discretizes state space and counts visits. Our ρ is a soft, continuous analogue via B-spline activations.

## 7. Code

- `explore_kan.py`: Full exploration experiment (uncertainty map + single-round + iterative)
- `bspline_uncertainty.py`: Uncertainty computation (existing)
- `online_learning_v2.py`: Training statistics computation (existing)

## References

- Burda, Y., Edwards, H., Storkey, A., & Klimov, O. (2019). Exploration by Random Network Distillation. ICLR.
- Pathak, D., Agrawal, P., Efros, A.A., & Darrell, T. (2017). Curiosity-driven Exploration by Self-supervised Prediction. CVPR.
- Bellemare, M.G., Srinivasan, S., Ostrovski, G., Schaul, T., Saxton, D., & Munos, R. (2016). Unifying Count-Based Exploration and Intrinsic Motivation. NeurIPS.
