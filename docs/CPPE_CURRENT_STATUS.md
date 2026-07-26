# CPPE 当前状态诊断

> 2026-07-26 | 分支 feature/cognitive-pca-decision | 控制瓶颈诊断节点

---

## 1. Physics identification 已基本成立 ✅

KAN 可以通过目标环境中的 `(s, a, s')` 在线拟合物理变化，PCA latent `z` 的方向和范围也符合真实物理变化。目前没有证据表明主要瓶颈来自物理辨识。

**证据**：
- PCA 前 5 PC 累积方差 96.8%
- PC 方向对应物理类型（质量/摩擦/执行器）且种子一致
- MMD 实验验证 imagination 质量随物理类型变化

---

## 2. Reward-free adaptation 原则上可行 ⚠️

| 方法 | friction_070 reward | 使用的信号 |
|:---|:---|:---|
| No adaptation | 7 | 无 |
| CPPE reward-free | 252 | 仅 (s,a,s') |
| Oracle PPO | 1347 | reward |

CPPE 仅用 (s,a,s') 实现 36× 提升。Physics-only signal 存在，但仅恢复 18% oracle gap。

---

## 3. 当前瓶颈：physics-conditioned control ❌

```
KAN identification → z  ✅
         ↓
z → target-conditioned action  ❌ ← 当前瓶颈
```

### 3.1 Transport teacher

```
Source policy:  671
Transport:     570–710
Teacher residual: ||a_teacher - a_source|| = 0.15 (z_src) → 0.24 (z_fric)
Δ = 0.09  (动作空间 [-1, 1])
```

Transport 以最小动作改动抵消瞬时动力学漂移，天然锚定在 source action 附近。z-dependent control signal 太弱。

### 3.2 Residual adapter

```
z = z_source:     694
z = z_friction:   718  (+24 only)
```

正确 z 仅带来 24 reward 提升。证明学生结构非瓶颈——teacher 本身没有提供足够强的 z→action 信号。

### 3.3 Reward-aware planner

```
CEM planner:      241  (worse than no-adapt 671)
KAN planner:      671  (equal to source)
Oracle PPO:      1347
```

失败原因：
1. 反事实 KAN 对目标物理有模型偏差，多步 rollout 放大误差
2. 手写 reward 与 Hopper 真实 reward 不一致

---

## 4. 已排除的方向 ❌

- 增大 imitation loss（v5）
- DAgger on source states（v5）
- curriculum hard-CF（v5）
- residual adapter
- `||ΔW||` imagination gate（v7）
- 在当前 KAN 上增加 CEM horizon

全部不解决 `z → control` 的信息不足问题。

---

## 5. Friction recovery curve

| Budget | CPPE | Transport | CEM Planner |
|:---|:---|:---|:---|
| 0 | **791** | 671 | — |
| 256 | 37 | 710 | — |
| 512 | 37 | 710 | — |
| 1024 | 37 | 571 | — |
| 2048 | 37 | 576 | — |

Budget 0 (z=z_source)：CPPE > source。Budget > 0 (z=fitted)：CPPE 崩溃。

---

## 6. Oracle PPO 对照

| Shift | Oracle PPO 200K | CPPE best | % Oracle |
|:---|:---|:---|:---|
| payload_125 | 1204 | 997 (v3) | 83% |
| friction_070 | 1347 | 252 (v6) | 19% |
| combo_medium | 371 | 222 (v6) | 60% |

---

## 7. 核心研究问题

> 在不使用目标环境 reward 的条件下，怎样从已辨识的动力学变化中构造足够强、闭环稳定、任务相关的控制信号？

当前 CPPE 三个层次：
1. **Physics identification** ✅ — KAN + PCA 已验证
2. **Physics-conditioned control** ❌ — 当前瓶颈
3. **Reward-free planning** ❌ — 受限于模型精度 + reward 定义

下一步方向：
- 是否能从 source policy/value 中提取比瞬时动力学匹配更强的任务结构
- 是否需要学习 trajectory-level compensation 而非 action-level
- KAN 是否适合承担多步规划模型，还是只应负责局部物理辨识
- 是否需要将控制目标从"恢复 source 下一步效果"改为"保持 source 的闭环不变量"
