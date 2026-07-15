# 损失函数设计的数学表述

## 一、已知信息（我们手上有什么）

### 1.1 环境动力学

真实动力学 $f: \mathcal{S} \times \mathcal{A} \times \mathcal{G} \to \mathcal{S}$，其中：
- $\mathcal{S} \subset \mathbb{R}^6$：状态空间（归一化）
- $\mathcal{A} = [-1, 1]$：动作空间（连续力矩）
- $\mathcal{G} = [9.8, 19.6]$：重力参数范围

当前状态 $s = [\cos\theta_1, \sin\theta_1, \cos\theta_2, \sin\theta_2, \dot\theta_1/6, \dot\theta_2/8]$

### 1.2 可用的网络

| 网络 | 意义 | 可微分 | 冻结 | 精度 |
|-----|------|:----:|:----:|:----:|
| $\text{WM}_0(s,a) = s'$ | $f(s,a; g=9.8)$ 的近似 | ✓ | ✓ | val_mse ≈ 1.7×10^{-4} |
| $\text{WM}_1(s,a) = s'$ | $f(s,a; g=14.7)$ 的近似 | ✓ | ✓ | val_mse ≈ 2.3×10^{-4} |
| $\pi_0(s) = a$ | $g=9.8$ 下的最优策略 | ✓ | ✓ | g=9.8 成功率 20/20 |
| $\pi(s) = a$ | 待训练的适应策略 | ✓ | — | — |

### 1.3 可以从 WM 中提取的信息

**信息源 1：一阶动力学**
$$\text{WM}(s,a) = s', \quad \nabla_a \text{WM}(s,a) = \frac{\partial s'}{\partial a} \in \mathbb{R}^{6 \times 1}$$

直接可用：预测任何 $(s,a)$ 下的一步结果及其对动作的敏感性。
在底部附近，$\partial s'/\partial a$ 中速度分量占主导：
$$\frac{\partial \dot\theta_1}{\partial a} \approx \text{常数}, \quad \frac{\partial \dot\theta_2}{\partial a} \approx \text{常数}$$
角度分量为二阶小量：$\frac{\partial \theta_1}{\partial a} \approx \mathcal{O}(\Delta t)$

**信息源 2：动力学变化场**
$$\Delta(s,a) = \text{WM}_1(s,a) - \text{WM}_0(s,a) \in \mathbb{R}^6$$

编码"物理参数变化对动力学的影响"。在 $g$ 变化影响大的状态区域，$\|\Delta\|$ 大；在不变的区域，$\|\Delta\| \approx 0$。

**信息源 3：WM 的一步能量信息**
$$E(\text{WM}(s,a)) \quad \text{其中 } E(s) = \text{KE}(s) + \text{PE}(s)$$
其梯度：
$$\nabla_a E(s') = \frac{\partial E}{\partial s'} \cdot \frac{\partial s'}{\partial a} \in \mathbb{R}^1$$

我们已经验证：此项在底部非零（$\partial \text{PE}/\partial \theta_1 \propto \cos\theta_1 \neq 0$ 在 $\theta_1=0$ 处）。

**信息源 4：WM 的多步 rollout 信息**
展开 $H$ 步可计算轨迹级属性（能量、可达性）。但伴随 BPTT 梯度衰减。

**信息源 5：ProtoKAN 内部结构**
- 原型位置 $x_n$：状态空间中被关注的特征点
- 边缘值 $y_n$ 和斜率 $d_n$：局部线性动力学近似
- 核宽度 $\sigma$：模型的局部化程度

---

## 二、损失函数的需求（我们想要什么）

我们要找一个损失函数 $\mathcal{L}(s, \pi(s))$，满足以下数学条件：

### R1：方向正确性（必要条件）

$$\langle \nabla_a \mathcal{L}(s, a), \nabla_a J(s, a; g) \rangle > 0 \quad \forall g \in [9.8, 19.6], \forall s \in \mathcal{S}$$

其中 $J(s, a; g)$ 是"从状态 $s$ 执行动作 $a$ 后在真实动力学 $g$ 下最终成功"的概率。$\nabla_a \mathcal{L}$ 必须与真实梯度方向同向。

> 实际上我们不知道 $J$ 的梯度，所以这个条件是用来检验 $\mathcal{L}$ 的设计的。

### R2：梯度非退化

$$\|\nabla_a \mathcal{L}(s, a)\| \geq \varepsilon > 0 \quad \text{对于 } s \in \mathcal{S}_{\text{critical}}$$

其中 $\mathcal{S}_{\text{critical}}$ 是策略做错误决策的状态集合（如底部附近）。梯度不能在这些区域消失，必须提供有意义的优化信号。

### R3：对称性破缺

$$\nabla_a^2 \mathcal{L}(s, a) \quad \text{不能是全局正定}$$

在底部附近，$\mathcal{L}$ 在动作 $a=0$ 处的 Hessian 不能是正定的，否则梯度总是指向 $a$ 的当前符号方向（局部凸→锁定错误方向）。要求：

$$\text{sign}\left( \frac{\partial L}{\partial a} \bigg|_{a=+1} \right) = \text{sign}\left( \frac{\partial L}{\partial a} \bigg|_{a=-1} \right)$$

在底部附近成立。即：**无论策略当前输出什么符号的动作，梯度都指向全局正确的方向。** 这是打破局部凸性的关键条件。

### R4：参数不变性

$$\mathcal{L} \text{ 不依赖任何特定 } g \text{ 值下的参考策略}$$

$\mathcal{L}$ 的定义不能包含 $\pi_0$ 或任何"在某个 $g$ 下训练好的行为参考"。否则当物理参数变化超出训练分布时，$\mathcal{L}$ 的方向会失效。

### R5：任务对齐

$$\arg\min_\pi \mathbb{E}_{s \sim \mathcal{S}}[\mathcal{L}(s, \pi(s))] \approx \arg\max_\pi J(\pi; g)$$

最小化 $\mathcal{L}$ 应该近似等价于最大化任务成功率。$\mathcal{L}$ 的优化不动点应该是能完成任务的好策略。

---

## 三、矛盾分析

### 3.1 已有方案为什么不满足这些条件

| 损失函数 | R1 方向正确 | R2 非退化 | R3 对称破缺 | R4 不依赖参考 | R5 任务对齐 |
|---------|:--------:|:--------:|:--------:|:----------:|:--------:|
| $\|\text{WM}_1(s,\pi) - \text{WM}_0(s,\pi_0)\|^2$ | ✗ (受 $\pi_0$ 偏差) | ✓ | ✗ (底部局部凸) | ✗ (依赖 $\pi_0$) | ✗ |
| $\sum \gamma^t\|s_t - s_{\text{goal}}\|^2$ (BPTT) | ✓ | ✗ (H步衰减) | ✓ | ✓ | ✗ (短视) |
| $\|\pi(s) - a^*_{\text{grid}}\|^2$ | ✗ (标签噪声) | ✓ | ? | ? | ✗ |
| $-E(\text{WM}_1(s,\pi))$ | ✓ | ✓ | ✓ | ✓ | ✗ (贪婪) |

**没有一个同时满足所有条件。**

### 3.2 核心矛盾

核心问题是 **R3（对称性破缺）** 和 **R5（任务对齐）** 之间的张力：

- 要实现 R3，损失函数必须包含**非局部信息**——它在某一点的梯度必须知道"全局的最优方向"，而不仅仅是"当前动作的局部效果"。
- 要包含非局部信息，通常需要**多步展开**或**先验知识**。
- 但多步展开（BPTT）导致 R2 退化（梯度衰减）。
- 先验知识（如能量函数）不完全满足 R5（只懂能量不懂任务）。

### 3.3 可能的数学突破口

**突破口：两步信息提取**

如果我们不要求 $\mathcal{L}$ 在单一点提供全局信息，而是：
1. 先用 $\text{WM}_1$ 提取状态空间的"局部动力学结构"（Jacobian、能量梯度、可控性）
2. 然后用这些结构信息构造一个 **状态相关权重函数**  $\lambda(s)$
3. 最后定义：
$$\mathcal{L}(s, a) = \lambda(s) \cdot \underbrace{\mathcal{L}_{\text{local}}(s,a)}_{\text{如能量}} + (1-\lambda(s)) \cdot \underbrace{\mathcal{L}_{\text{goal}}(s,a)}_{\text{如状态距离}}$$

其中 $\lambda(s)$ 的确定不依赖 $\pi_0$，而是从 $\text{WM}$ 的内部结构中计算出来（如可控性格拉姆矩阵、能量差异函数等）。

这可能是满足所有条件的路径。

---

## 四、形式化的问题陈述

**给定**：
1. $\text{WM}_0, \text{WM}_1 : \mathcal{S} \times \mathcal{A} \to \mathcal{S}$（可微）
2. $s_{\text{goal}} \in \mathcal{S}$（目标状态）
3. $\pi_0 : \mathcal{S} \to \mathcal{A}$（参考策略，可选）
4. 任务成功判据 $\text{success}(s) = [\text{tip\_height}(s) > 1.0]$

**求**：
$$\mathcal{L} : \mathcal{S} \times \mathcal{A} \to \mathbb{R}$$

满足 R1-R5，且：
$$\nabla_\theta \mathcal{L}(s, \pi_\theta(s)) = \underbrace{\nabla_a \mathcal{L}}_{\text{1步通过WM}} \cdot \underbrace{\nabla_\theta \pi_\theta(s)}_{\text{策略梯度}}$$

其中 $\nabla_a \mathcal{L}$ 只经过 1 步 WM 计算。
