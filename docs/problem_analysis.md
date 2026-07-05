# KAN-RF 核心问题分析

## 1. 尝试过的方法与结果

### 让 KAN 参与决策的尝试

| 方法 | KAN 在部署时参与 | 成功率 | 核心问题 |
|------|:---:|:---:|------|
| 单点逆优化 | ✓ 前向值 | 7/10 | 根因三：单步 Jacobian 太小，放大因子 25× |
| 多尺度 + 决策网络 (Plan A) | ✓ 作为特征源 | 9/10 | 引入 k-选择问题，选错 k 导致灾难 |
| 动作探索器 | ✗ 绕过 | **10/10** | 不依赖 KAN 做决策 |
| decision_v2 (特征压缩) | ✗ 5 标量 | 7/10 | KAN 知识被压缩丢失 |
| decision_v3 (梯度训练) | ✗ 仅训练时 | **10/10** | KAN 在部署时不存在 |
| WM+V (世界模型+价值网络) | ✓ 前向预测 | 77% | 贪婪 MPC 天花板 |
| **KAN-MPC** | **✓ 预测+Jacobian+不确定性** | **70%** | **KAN 预测精度不足** |

### 提高 KAN 精度的尝试

| 方法 | val MSE | Jacobian cos_sim | 效果 |
|------|:---:|:---:|------|
| 基础 MSE 训练 | 0.0018 | 0.099 | 基线 |
| MOPS (P-spline) | 0.0019 | 0.237 | 光滑但不改善方向 |
| CWS (Jacobian 匹配) | 0.00034 | 0.979 | Jacobian 方向大幅改善 |
| Hybrid (MOPS+CWS) | **0.00020** | 0.924 | 精度和方向同时最优 |
| CWS + 去 MOPS | 0.0011 | ~0.98 | 持续学习最优 |

### 持续学习验证

| 模型 | g=10→3 恢复程度 |
|------|:---:|
| KAN-CWS | **0.9x** 基线 ✓ |
| MLP | 16.3x 基线 ✗ |

---

## 2. 核心问题：KAN 作为世界模型的精度上限

### 2.1 现象

KAN-MPC 是唯一让 KAN 在**部署时**用全部知识（预测 + Jacobian + 不确定性）参与决策的方法。但它只在 Pendulum 上达到 70%，CartPole 上 0%。

**同一个 KAN** 用来训练一个简单的 MLP 策略（decision_v3），Pendulum 100%，CartPole 100%。

为什么同样的 KAN，间接使用（训练策略）比直接使用（MPC 展开）效果更好？

### 2.2 机制

decision_v3 使用 KAN 的方式：训练时，$s \to \pi_\theta(s) \to a \to \text{KAN} \to s'$，梯度 $\partial \mathcal{L} / \partial s' \cdot \partial s' / \partial a \cdot \partial a / \partial \theta$ 告诉策略"动作应该往哪调"。策略学到的是动作的**方向**——即使 Jacobian 不精确，方向对就够。

KAN-MPC 使用 KAN 的方式：展开 $H$ 步，每一步都用 KAN 预测 $s_{t+1}$。这要求 KAN 的**函数值**和**Jacobian 方向**都精确——每一步的预测误差在 $H$ 步中累积。

| 需求 | decision_v3 | KAN-MPC |
|------|:---:|:---:|
| Jacobian 方向正确 | ✓ 需要 | ✓ 需要 |
| Jacobian 大小精确 | ✗ 不需要 | ✓ 需要（影响评分权重） |
| 函数值精确 | ✗ 不需要 | ✓ **关键**（H 步累积） |

decision_v3 只需要 CWS 给对方向。KAN-MPC 需要 CWS 给对方向 **+** KAN 函数值足够精确以支撑 H 步 rollout。

### 2.3 定量分析

KAN 当前精度（Pendulum, val MSE = 0.0011）：

$$\text{RMSE per dim} \approx \sqrt{0.0011/3} \approx 0.019 \text{ (normalized)}$$

H=5 步 rollout 后，理想情况下：

$$\text{累积误差} \approx 5 \times 0.019 = 0.095 \text{ (normalized)}$$

这约等于 0.09 × 8 = 0.72 rad 的角度误差，已经大到无法区分好动作和坏动作。

MLP 同样的计算：

$$\text{RMSE per dim} \approx \sqrt{0.000026/3} \approx 0.0029$$

$$5 \times 0.0029 = 0.015 \text{ (normalized)} \approx 0.12 \text{ rad}$$

MLP 的 5 步累积角度误差约 0.12 rad——仍在可控范围内。

### 2.4 根因

KAN 和 MLP 之间的 43× 精度差距来自**表达能力的系统性差异**：

- MLP：全局矩阵乘法 + 非线性激活，对光滑函数的逼近效率已被广泛证明
- KAN：B-样条边函数，表达能力受 grid 精度和 spline 阶数限制。当前 grid=5（每个维度 6 个控制点），所有维度组合的函数值由这少量控制点的组合决定

这不是训练不够的问题——我们试过 2400 epochs、Hybrid (MOPS+CWS)、去 MOPS、更大网络，KAN 的 val MSE 始终在 0.0002–0.001 量级，MLP 在 0.00003 量级。

## 3. 矛盾

**决策方面**：KAN-MPC 是架构上最正确的方法——KAN 的所有独特能力（精确 Jacobian、激活密度不确定性、有界导数）在 MPC 框架中都有直接的用途。如果 KAN 足够精确，KAN-MPC 就是最优方案。

**精度方面**：KAN 的函数值精度系统性低于 MLP，阻碍了 MPC rollout 中区分好动作和坏动作的能力。提高精度的尝试（更大网络、更久训练、Hybrid 正则化）将 val MSE 从 0.0018 降到了 0.0002（9×），但 MLP 仍然是 0.00003，差距 7×。

**替代方案方面**：MLP 预测更准，MLP-MPC 理论上可能比 KAN-MPC 好——但这意味着用两个黑箱（MLP 世界模型 + MLP 策略），完全放弃 KAN 的物理可解释性。这就回到了传统 MBRL，KAN 的所有独特贡献（B-样条可解释性、激活密度不确定性、有界导数）都不存在了。

---

*记录于 2026-07-04。*
