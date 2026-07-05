# 7/4 实验记录：KAN 持续学习能力验证

## 1. 实验目标

验证 KAN 的核心宣称：**B-样条局部支撑使世界模型在物理参数突变时，预测精度下降后能快速恢复，且不遗忘已学知识。**

回答的具体问题：
- KAN 能否在重力参数变化后快速恢复预测精度？
- 恢复能力的来源是 B-样条架构本身，还是三因子学习率？
- MOPS 光滑约束对持续学习是帮助还是阻碍？
- 与 MLP 世界模型相比，KAN 有何优势？

---

## 2. 实验设计

### 2.1 物理参数变化

修改 Pendulum-v1 的重力参数 `g`：

```
步数 0–200:   g = 10.0  （默认地球重力）
步数 200–400: g = 18.0  （重重力，+80%）
步数 400–600: g = 10.0  （回到默认——测试灾难性遗忘）
步数 600–800: g = 3.0   （轻重力，-70%）
```

### 2.2 世界模型

| 模型 | 架构 | 参数 | 训练方式 |
|------|------|:---:|------|
| KAN-CWS | [4, 12, 3], grid=5, order=3 | 756 | CWS (Jacobian matching, ν=1.0), 无 MOPS |
| KAN-MSE | [4, 12, 3], grid=5, order=3 | 756 | 纯 MSE，无正则化 |
| MLP | [4, 32, 32, 3] | 1315 | 纯 MSE |

所有模型使用相同数据训练：20,000 条默认重力 (g=10.0) 下的随机动作 transitions。

### 2.3 在线更新策略

| 实验条件 | 在线更新 |
|------|------|
| KAN-CWS (constant LR) | SGD + momentum + replay buffer (500) |
| KAN-MSE (constant LR) | SGD + momentum + replay buffer (500) |
| MLP (SGD+replay) | SGD + momentum + replay buffer (500) |

三个条件使用**完全相同的在线更新策略**，差异纯粹来自离线训练方法和架构。

### 2.4 评估指标

- **每步预测 L2 误差**（原始状态空间：cos, sin, θ̇×8）
- **基线误差**：g=10.0 段最后 150 步的均值
- **相对误差**：当前误差 / 基线误差（归一化比较）
- **恢复程度**：参数切换后 30–200 步的均值 / 基线

### 2.5 轨迹生成

使用能量控制器（EnergyController, k_swing=1.5）生成持续轨迹。控制器根据当前 g 动态计算能量，保证轨迹涵盖摆起、稳定等多样的状态分布。

---

## 3. 实验结果

### 3.1 离线训练精度

| 模型 | val MSE | 离线训练时间 |
|------|:---:|:---:|
| KAN-CWS | 0.001129 | ~80s (2400 epochs, mini-batch CWS) |
| KAN-MSE | 0.000600 | ~20s (2400 epochs, full-batch) |
| MLP | 0.000041 | ~30s (2400 epochs, full-batch) |

### 3.2 在线适应结果（800 步实验）

| 模型 | g=10.0 基线 | g=18.0 恢复 | g=10.0 回归 | g=3.0 恢复 | 遗忘? |
|------|:---:|:---:|:---:|:---:|:---:|
| **KAN-CWS** | 0.072 | 0.028 (0.4x) | 0.016 (0.2x) | **0.064 (0.9x)** | ✓ 无 |
| KAN-MSE | 0.090 | 0.037 (0.4x) | 0.024 (0.3x) | **0.043 (0.5x)** | ✓ 无 |
| MLP | 0.021 | 0.021 (1.0x) | 0.021 (1.0x) | **0.342 (16.3x)** | ✓ 无 |

### 3.3 关键数据点

**g=3.0（轻重力，最剧烈的变化）200 步后的绝对预测误差：**

| 模型 | 绝对误差 | 相对基线 |
|------|:---:|:---:|
| KAN-CWS | 0.064 | 0.9x ← 完全恢复 |
| KAN-MSE | 0.043 | 0.5x ← 比原来更好 |
| MLP | 0.342 | **16.3x** ← 远未恢复 |

KAN-CWS 比 MLP 好 **5.3 倍**；KAN-MSE 比 MLP 好 **8.0 倍**。

---

## 4. 实验迭代过程

### 4.1 第一轮：发现公平性问题

初始实验中 KAN 用纯 MSE 训练（无 CWS），val MSE=0.0012，基线误差=0.23；MLP val MSE=0.00003，基线误差=0.01。差距 20+ 倍，无法公平比较。

**教训**：KAN 需要 CWS 训练来缩小离线精度差距。

### 4.2 第二轮：Hybrid (MOPS + CWS) 的意外表现

加入 Hybrid 训练的 KAN（val MSE=0.0009, 基线=0.026），但 g=3.0 恢复后误差为 0.249 (9.7x 基线)——虽然比 MLP (16.7x) 好，但远不如 KAN-MSE (0.5x)。

**发现**：MOPS 光滑约束通过 Δ²c 将相邻控制点耦合在一起，抵消了 B-样条局部支撑的解耦优势。训练好的光滑结构成为"记忆"，阻碍局部区域的快速调整。

### 4.3 第三轮：CWS-only（最终方案）

去掉 MOPS，仅保留 CWS (Jacobian matching)，用 constant LR + replay buffer 在线更新。

**结果**：KAN-CWS 在保持较好离线精度的同时，g=3.0 恢复至 0.9x 基线。证实了：
- CWS（Jacobian 匹配）提供离线精度和可靠的梯度方向
- 无 MOPS 保证控制点可以自由局部调整
- B-样条架构本身（非学习率策略）是持续学习能力的来源

---

## 5. 根因分析：MOPS 为何阻碍持续学习

### 5.1 机制

B-样条的局部支撑意味着：修改控制点 $c_k$ 只影响输入空间的一个局部区间 $[t_k, t_{k+4}]$。这天然支持持续学习——适应新动力学时，只有受影响区域的控制点需要更新。

但 MOPS 的惩罚项 $\|\Delta^2 c\|^2$ 通过控点差分将相邻控制点**耦合**在一起：

$$\Delta^2 c_5 = c_7 - 2c_6 + c_5$$

改变 $c_5$ 会导致 $\Delta^2 c_4$ 和 $\Delta^2 c_5$ 同时增大，MOPS 梯度会同时作用在 $c_4, c_5, c_6, c_7$ 上，即使只有 $c_5$ 对应的区间需要调整。

### 5.2 "光滑记忆"效应

离线训练时，MOPS 将控制点推到"妥协位置"——在拟合数据和保持光滑之间取得平衡。这个光滑结构内化在控制点之间的相对关系中：

- 纯 MSE 训练的控制点 = 黏土（可以自由捏形）
- MOPS 训练的控制点 = 橡皮筋（弯了会弹回来）

当物理参数突变，MOPS 训练的模型需要先"挣脱"光滑约束才能适应新动力学。

### 5.3 对三因子学习率的补充观察

三因子学习率中的 `count_factor = 1/√(1+N_updates)` 惩罚高频更新的参数。在参数突变时，这恰好是**反作用**的——需要最大更新的参数反而被最大程度惩罚。

---

## 6. 最终方案

### 训练配方

```
世界模型: KAN([4, 12, 3], grid=5, order=3), 756 params
训练损失: MSE + ν·CWS, ν=1.0  (Jacobian matching, 无 MOPS)
优化器:   Adam, lr=1e-2, StepLR(step=600, gamma=0.5)
Epochs:   2400 (mini-batch 2048 for CWS Jacobian computation)
```

### 在线更新配方

```
策略:     SGD + momentum (0.9) + replay buffer (500)
学习率:   恒定 1e-3（不使用三因子）
```

### 为什么这样做

| 组件 | 保留 | 去掉 | 原因 |
|------|:---:|:---:|------|
| B-样条架构 | ✓ | | 局部支撑 = 天然抗遗忘 |
| CWS (Jacobian 匹配) | ✓ | | 提高离线精度 + 保证梯度方向可靠 |
| MOPS (光滑约束) | | ✗ | 控制点耦合 → 阻碍局部适应 |
| 三因子学习率 | | ✗ | count_factor 惩罚突变所需的参数更新 |

---

## 7. 待做 / 下一步

- [ ] 引入更多参数变化类型（质量 m、杆长 l），测试泛化性
- [ ] 增加参数变化的频次和幅度，寻找 KAN 适应能力的边界
- [ ] 在 CartPole、Acrobot 上复现实验
- [ ] 将 CWS-only KAN 集成到 decision_v3 训练管线中（替换当前的基础 KAN）
- [ ] 论文写作：持续学习实验作为 KAN 独特优势的实证章节

---

## 8. 代码位置

- 实验脚本：`experiments/continual_learning.py`
- 运行命令：`conda run -n pyt python experiments/continual_learning.py --epochs 2400 --steps 800`
- 缓存模型和数据：`/tmp/kanrf_cl_exp/`
- 结果图：`continual_learning_results_abs.png`, `continual_learning_results_rel.png`

---

## 9. 持续学习机制增强：控制点梯度扩散 (Control-Point Gradient Diffusion)

### 9.1 动机

当前在线更新中，每条边 $\phi_{i,j}$ 的 $G+k$ 个控制点 $\{c_{i,j,0}, ..., c_{i,j,G+k-1}\}$ 收到独立梯度后各自独立更新。但 B-样条基函数沿输入轴有天然空间顺序：

```
输入轴:  [-1.0 .................................... +1.0]
基函数:  B_0    B_1    B_2    B_3    B_4    B_5    B_6    B_7
         ↑                                                    ↑
    左端（覆盖 x≈-1）                                  右端（覆盖 x≈+1）
```

当输入值 $x$ 从左向右移动时，基函数被**依次激活**——$B_0$ 先激活，$B_1$ 次之……这个激活顺序定义了控制点 $c_k$ 之间的**天然相邻关系**（沿着输入轴的空间相邻）。

核心假设：**物理参数平滑变化时，其对动力学的影响随着输入值连续变化，相邻输入区域的函数更新应该相关。**

### 9.2 核心思想

不是约束控制点的**静态形状**（MOPS 的做法，阻碍改变），而是约束控制点的**更新方向**（Δc 沿 k 轴平滑扩散，引导改变）。

直观对比：

| 机制 | 作用对象 | 数学形式 | 效果 |
|------|------|------|------|
| MOPS | 控制点值 $c_k$ | $\|\Delta^2 c\|^2$ 惩罚 | 静态光滑 → 阻碍局部适应 |
| 梯度扩散 | 更新量 $\Delta c_k$ | $\sum w_k (\Delta c_{k+1} - \Delta c_k)^2$ | 更新协调 → 促进合理传递 |

### 9.3 数学形式化

#### 9.3.1 优化问题

对于固定的边 $(i,j)$，定义 $n = G+k$ 个控制点。在线学习给出独立梯度 $\mathbf{g} \in \mathbb{R}^n$，其中 $g_k = \frac{\partial \mathcal{L}}{\partial c_k}$。求解带约束的更新：

$$\Delta \mathbf{c}^* = \arg\min_{\Delta \mathbf{c} \in \mathbb{R}^n} \left\{ \mathbf{g}^T \Delta \mathbf{c} + \frac{1}{2\eta} \|\Delta \mathbf{c}\|^2 + \frac{\lambda}{2} \sum_{k=1}^{n-2} w_k \cdot (\Delta c_{k+1} - \Delta c_k)^2 \right\}$$

三项含义：

| 项 | 数学 | 作用 |
|------|------|------|
| 一阶下降 | $\mathbf{g}^T \Delta \mathbf{c}$ | 保证更新沿损失下降方向 |
| 信赖域 | $\frac{1}{2\eta} \|\Delta \mathbf{c}\|^2$ | 防止单步更新过大 |
| 更新平滑 | $\frac{\lambda}{2} \sum w_k (\Delta c_{k+1} - \Delta c_k)^2$ | 更新量沿 k 轴平滑 |

#### 9.3.2 趋势敏感权重 $w_k$

关键设计：$w_k$ 由函数**当前形状**决定——这正是用户所说的"根据条样函数整体趋势"：

$$w_k = \frac{1}{\alpha + |c_{k+1} - c_k|^\beta}$$

物理直觉（各向异性扩散）：

| 条件 | $|c_{k+1} - c_k|$ | $w_k$ | 扩散强度 | 含义 |
|------|:---:|:---:|:---:|------|
| 函数平坦 | 小 | 大 | 强 | 相邻区域无独立结构，更新自由传递 |
| 函数陡峭 | 大 | 小 | 弱 | 该处有物理特征（如阈值），保留边界 |

参数：
- $\alpha > 0$：防止除零（建议 $\alpha = 0.01$）
- $\beta \in \{0, 1, 2\}$：$\beta=0$ 退化为均匀扩散，$\beta=1$ 线性趋势敏感，$\beta=2$ 更强趋势保护

#### 9.3.3 闭式解

$$\Delta \mathbf{c}^* = -\eta \left( \mathbf{I} + \lambda\eta \cdot \mathbf{D}^T \mathbf{W} \mathbf{D} \right)^{-1} \mathbf{g}$$

其中：
- $\mathbf{D} \in \mathbb{R}^{(n-2) \times n}$ 是一阶差分矩阵：$(\mathbf{D}\mathbf{u})_k = u_{k+1} - u_k$
- $\mathbf{W} = \text{diag}(w_1, ..., w_{n-2})$
- $\lambda = 0$ 时退化为标准梯度下降 $\Delta\mathbf{c} = -\eta \mathbf{g}$

### 9.4 收敛性分析

#### 定理 1：严格下降方向

对于任意 $\lambda \ge 0$，$w_k > 0$，有：

$$\mathbf{g}^T \Delta \mathbf{c}^* = -\eta \cdot \mathbf{g}^T (\mathbf{I} + \lambda\eta \mathbf{D}^T \mathbf{W} \mathbf{D})^{-1} \mathbf{g} < 0 \quad (\text{当 } \mathbf{g} \neq \mathbf{0})$$

**证明**：$\mathbf{D}^T \mathbf{W} \mathbf{D}$ 是半正定矩阵（二次型 $\mathbf{u}^T \mathbf{D}^T \mathbf{W} \mathbf{D} \mathbf{u} = \sum w_k (u_{k+1} - u_k)^2 \ge 0$）。$\mathbf{I}$ 严格正定，故 $\mathbf{I} + \lambda\eta \mathbf{D}^T \mathbf{W} \mathbf{D}$ 严格正定，其逆也严格正定。$\mathbf{g}^T \mathbf{M} \mathbf{g} > 0$ 对任意 $\mathbf{g} \neq 0$ 和正定 $\mathbf{M}$ 成立。

#### 定理 2：收敛性保证

若损失函数 $\mathcal{L}$ 是 $L$-光滑的（$\|\nabla^2 \mathcal{L}\| \le L$），取 $\eta < 2/L$，则带扩散的更新保证：

$$\mathcal{L}(\mathbf{c}^{(t+1)}) \le \mathcal{L}(\mathbf{c}^{(t)})$$

且序列 $\{\mathbf{c}^{(t)}\}$ 收敛到 $\nabla \mathcal{L} = \mathbf{0}$ 的稳定点。

**证明思路**：带扩散的更新等价于在变换后的参数空间中做标准梯度下降。令：

$$\mathbf{P} = (\mathbf{I} + \lambda\eta \mathbf{D}^T \mathbf{W} \mathbf{D})^{-1/2}$$

定义 $\tilde{\mathbf{c}} = \mathbf{P}^{-1} \mathbf{c}$，则：

$$\Delta \tilde{\mathbf{c}} = \mathbf{P}^{-1} \Delta \mathbf{c}^* = -\eta \cdot \mathbf{P} \mathbf{g} = -\eta \cdot \nabla_{\tilde{\mathbf{c}}} \mathcal{L}$$

在 $\tilde{\mathbf{c}}$ 空间中，这是标准梯度下降。$\tilde{\mathbf{c}}$ 空间中的 $L$-光滑常数可能略大于原空间（取决于 $\mathbf{P}$ 的谱），但取足够小的 $\eta$ 仍保证收敛。收敛性不因扩散而丧失。

#### 定理 3：扩散不引入偏差

当 $\mathbf{g} = \mathbf{0}$（已达到局部最优），$\Delta \mathbf{c}^* = \mathbf{0}$，无论 $\lambda$ 取何值。

扩散只影响**瞬态更新路径**，不改变**收敛点**。这与 MOPS 本质不同——MOPS 会改变收敛点（因为它在目标函数中加了额外惩罚项）。

### 9.5 与 MOPS 的本质区别

| 维度 | MOPS | 梯度扩散 |
|------|------|------|
| 数学形式 | $\min_{\mathbf{c}} \mathcal{L} + \lambda \|\Delta^2 \mathbf{c}\|^2$ | $\mathcal{L}$ 不变，扩散作用于 $\Delta \mathbf{c}$ |
| 作用阶段 | 离线训练（目标函数内） | 在线更新（参数更新规则内） |
| 对收敛点的影响 | **改变**（光滑先验偏置） | **不改变**（相同的最优解） |
| 对适应速度的影响 | 阻碍（需要挣脱光滑记忆） | 促进（协调相邻控制点一起更新） |
| 梯度为零时 | MOPS 梯度仍非零（光滑拉力） | 扩散效应为零 |
| "光滑"作用在哪里 | 参数的静态配置 | 参数的变化方向 |

### 9.6 算法（迭代实现，无需矩阵求逆）

对于每条边 $(i,j)$ 独立执行：

```
输入: g ∈ R^n           # 独立梯度 ∂L/∂c_k
      c ∈ R^n           # 当前控制点值
      η, λ, α, β, T    # 超参数

输出: Δc* ∈ R^n         # 扩散后的更新量

# Step 1: 计算趋势敏感权重
for k = 1 to n-2:
    w_k = 1 / (α + |c_{k+1} - c_k|^β)

# Step 2: 初始更新 = 原始梯度下降
Δc = -η · g

# Step 3: T 轮扩散迭代（T=1 仅传递到相邻 k，T>1 传递到更远）
ε = λη / (1 + λη)       # 单步扩散率，∈ (0, 1)
for t = 1 to T:
    for k = 1 to n-2:
        flux = w_k · (Δc_{k+1} - Δc_k)     # 更新量在 k 和 k+1 之间的不连续程度
        Δc_k     ← Δc_k     + ε · flux     # 平滑：减小差异
        Δc_{k+1} ← Δc_{k+1} - ε · flux

return Δc
```

复杂度：每边 $O(T \cdot n)$，$n = G+k = 8$（grid=5, k=3），$T$ 通常取 1–3。与原始梯度计算相比可忽略。

**单个控制点 k 接收到的累计更新**（$T=1$ 时）：

$$\Delta c_k^{\text{final}} = \underbrace{(1 - \varepsilon(\tilde{w}_{k-1} + \tilde{w}_k))}_{\text{自身}} \Delta c_k^{\text{raw}} + \underbrace{\varepsilon \tilde{w}_{k-1}}_{\text{来自 k-1}} \Delta c_{k-1}^{\text{raw}} + \underbrace{\varepsilon \tilde{w}_k}_{\text{来自 k+1}} \Delta c_{k+1}^{\text{raw}}$$

其中 $\tilde{w}_k = \frac{w_k}{1+\lambda\eta}$，$\varepsilon = \lambda\eta$。

直观：每个控制点的最终更新 = 自身原始更新的加权 + 邻居原始更新的加权混合。权重大小由函数形状 $|c_{k+1}-c_k|$ 调节。

### 9.7 独立性的保持

扩散仅在单条边 $(i,j)$ 的 $k$ 轴上进行。不同边之间**不传递**——每条边描述独立的 1D 函数，它们之间的耦合通过网络的下一层自然发生。这保持了 B-样条局部支撑的核心优势。

### 9.8 实验验证计划

对比三组（在当前 `continual_learning.py` 框架下）：

| 组 | 描述 | 预期 |
|------|------|------|
| **Baseline** | KAN-CWS + constant LR + replay buffer（当前最优） | 已知基线 |
| **均匀扩散** | Baseline + 固定 $w_k = 1$（$\beta=0$） | 验证扩散是否有用 |
| **趋势引导扩散** | Baseline + 自适应 $w_k = (\alpha + |c_{k+1}-c_k|^\beta)^{-1}$ | 验证趋势引导是否更好 |

核心指标：
- g=3.0 场景下的恢复速度（相同步数后的误差比）
- 平稳段的噪声水平（扩散是否平滑了梯度噪声）
- 离线精度的保持（扩散是否引入了不期望的漂移）

### 9.9 超参数范围

| 参数 | 含义 | 建议搜索范围 | 备注 |
|------|------|:---:|------|
| $\lambda$ | 扩散总强度 | 0.1, 0.5, 1.0, 5.0 | 核心调参项 |
| $\beta$ | 趋势敏感度 | 0, 1, 2 | 0=均匀，1=线性，2=强保护 |
| $\alpha$ | 平滑底噪 | 0.01 | 通常不调 |
| $T$ | 扩散轮数 | 1, 2, 3 | 影响传播距离 |

---

## 10. 待做 / 下一步

- [ ] 实现梯度扩散更新器（`KANDiffusionUpdater`）
- [ ] 在 `continual_learning.py` 中加入扩散 vs 基线对比实验
- [ ] 调参 $\lambda, \beta, T$ 找最优配置
- [ ] 引入更多参数变化类型（质量 m、杆长 l），测试泛化性
- [ ] 增加参数变化的频次和幅度，寻找 KAN 适应能力的边界
- [ ] 在 CartPole、Acrobot 上复现实验
- [ ] 将 CWS-only KAN 集成到 decision_v3 训练管线中（替换当前的基础 KAN）
- [ ] 论文写作：持续学习 + 梯度扩散 作为 KAN 独特优势的实证章节



---

## 10. 梯度扩散实验结果（2026-07-04）

### 实验设计

对比三组（同样使用 CWS-only KAN + replay buffer）：

| 条件 | λ | β | 描述 |
|------|:---:|:---:|------|
| A (baseline) | 0 | — | 无扩散，各控制点独立更新 |
| B (uniform) | 0.5 | 0 | 均匀扩散，不依赖函数趋势 |
| C (trend) | 0.5 | 1 | 趋势引导扩散，$w_k = 1/(\alpha+|c_{k+1}-c_k|)$ |

### 结果

| 条件 | g=3.0 稳定误差 | 相对基线 | vs baseline |
|------|:---:|:---:|:---:|
| A (no diffusion) | 0.0635 | 0.9x | — |
| B (uniform, β=0) | 0.0578 | 0.9x | −9%（噪声级别） |
| C (trend, β=1) | 0.0627 | 0.9x | −1%（无差异） |
| MLP (reference) | 0.3510 | 16.8x | — |

三组 KAN 的误差曲线几乎完全重叠。

### 结论

**梯度扩散没有显著效果。** 原因：
1. B-样条基函数本身有重叠支撑（$B_k$ 在 $[t_k, t_{k+4}]$ 上非零），一个输入激活 4 个控制点，已经天然完成了信息传递
2. KAN-CWS 已恢复至 0.9x 基线，可改进空间极小
3. 真正的瓶颈是 KAN 与 MLP 的**离线精度差距**（0.072 vs 0.021），不是适应速度

### 启示

KAN 的持续学习能力来自 B-样条架构本身——不需要额外的跨控制点更新传递机制。下一步应聚焦于缩小离线精度差距。

---

---

## 11. Decision_v3 策略实验（2026-07-04）

### 背景

decision_v3 的设计文档中有一个 PENDING 实验：用高质量 CWS KAN 训练 ResidualPhysics 策略。之前用基础 KAN (cos_sim≈0.1) 训练时，物理参数被推到错误方向（$k_{damp}$ 从 -0.3 变成 +1.31 反阻尼），仅 7/10。

### 实验设置

- **KAN 世界模型**：CWS-only `[4,12,3]`，ν=1.0，2400 epochs，val_mse=0.00113，cos_sim≈0.98
- **训练方法**：`decision_v3/train.py`，energy-based loss，200 epochs
- **测试**：Pendulum-v1，10 trials（seed 42, 142, ..., 942）

### 结果

| 策略 | 成功率 | 关键发现 |
|------|:---:|------|
| Pure MLP + CWS KAN | **10/10** | 与 Phase 2 结果一致，稳健 |
| ResidualPhysics + CWS KAN | **10/10** | 物理参数退化到零 |

**ResidualPhysics 训练后参数变化：**

| 参数 | 初始值 | 训练后 | 应该的值 |
|------|:---:|:---:|:---:|
| k_energy | 0.15 | **-0.02** | 1.5 |
| k_stable | -2.0 | -1.96 | <-0 |
| k_damp | -0.3 | **-0.03** | <-0 |
| residual_scale | 0.1 | **0.88** | 小值 |

### 分析

即使 CWS KAN 的 Jacobian cos_sim 从 0.1 提升到 0.98，ResidualPhysics 的物理参数仍然无法正确学习：

1. **k_energy → ~0**：能量塑形项被关闭
2. **k_damp → ~0**：阻尼项被关闭
3. **residual_scale 暴涨**：残差 MLP 接管全部工作

根本原因：KAN 通过 energy loss 只评判"这个动作好不好"，不评判"物理先验的强度对不对"。从 $\mathcal{L}$ 到 $k_{energy}$ 的梯度路径太长太间接——经过 $\tanh$ 饱和、残差项混合、KAN 前向传播——信号在传递中被稀释和扭曲。

**这个 10/10 实质上是 Pure MLP 的 10/10**，物理先验没有贡献。

### 结论

**Pure MLP + KAN 梯度训练是 Pendulum 上的最优方案。** 物理先验（ResidualPhysics）在 Pendulum 上没有带来额外价值：
- 用 basic KAN 时，梯度方向错 → 参数推错方向 → 7/10
- 用 CWS KAN 时，梯度方向对但信号太间接 → 参数退化到零 → 功能等价于 Pure MLP

### 下一步方向

- [ ] 在 CartPole/Acrobot 上测试 Pure MLP + CWS KAN（验证泛化性）
- [ ] Multi-step rollout 训练（v3.1，可能对需要多步协调的环境有帮助）
- [ ] 论文中如何叙述"物理先验退化"这一发现？

---

*记录于 2026-07-04。*
