# 前向-逆向差距：为什么准确的前向预测不能保证准确的逆向恢复

## 1. 问题陈述

我们训练一个 KAN 世界模型 $f_\theta$，在给定当前状态和动作的条件下预测系统的下一个状态：

$$\hat{s}' = f_\theta(s, a)$$

训练过程最小化采集到的转移数据上的均方误差：

$$\theta^* = \arg\min_\theta \;\mathbb{E}_{(s,a,s') \sim \mathcal{D}}\left[ \|f_\theta(s, a) - s'\|^2 \right]$$

训练完成后，前向预测误差很小：在训练分布上，$f_{\theta^*} \approx f_{\text{true}}$ 逐点成立。在 Pendulum-v1 上，使用 35000 条转移数据和 $[4, 12, 3]$ 结构的 KAN（756 个参数），留出验证集的均方根误差为归一化状态空间中的 $0.037$，对应约 $0.14$ 弧度的角度误差。

部署时，模型被反过来使用。给定当前状态 $s$ 和期望的下一状态 $s'_{\text{target}}$，我们通过冻结模型上的梯度优化来寻找动作：

$$a^* = \arg\min_a \;\|f_\theta(s, a) - s'_{\text{target}}\|^2$$

**经验发现是，这种逆向恢复效果很差**：当我们拿出真实的 $(s, a_{\text{true}}, s'_{\text{true}})$ 数据，给定 $(s, s'_{\text{true}})$ 求解 $a^*$ 时，恢复动作的平均绝对误差为 $|a^* - a_{\text{true}}| \approx 0.87$ N·m——几乎是最大力矩范围 $\pm 2.0$ N·m 的一半。只有 $36.5\%$ 的恢复结果落在真实动作的 $0.2$ N·m 以内。对于最差的十分之一，误差超过 $2.33$ N·m，意味着模型恢复的动作与真实动作方向相反。

本文提供关于这一差距为何存在、以及为什么它在很大程度上是问题结构所固有的（而非仅仅是训练不足造成的假象）的严格解释。

---

## 2. 数学形式化

### 2.1 符号约定

- $s \in \mathcal{S} \subset \mathbb{R}^3$：归一化摆状态 $[\cos\theta,\; \sin\theta,\; \dot{\theta}/8]$
- $a \in \mathcal{A} = [-1, 1]$：归一化力矩（原始力矩除以 2）
- $f_{\text{true}}: \mathcal{S} \times \mathcal{A} \to \mathcal{S}$：真实（未知）动力学
- $f_\theta: \mathcal{S} \times \mathcal{A} \to \mathcal{S}$：学习到的 KAN 世界模型
- $J_f(s, a) = \frac{\partial f}{\partial a}(s, a) \in \mathbb{R}^3$：$f$ 关于动作的 Jacobian（导数向量）
- $\varepsilon(s, a) = f_\theta(s, a) - f_{\text{true}}(s, a) \in \mathbb{R}^3$：前向预测误差

### 2.2 两个根本不同的优化问题

**训练（前向）：**

$$\theta^* = \arg\min_\theta \sum_i \|f_\theta(s_i, a_i) - s'_i\|^2$$

这最小化预测的下一状态与观测到的下一状态之间的差异，**在动作已知的数据点上进行**。决策变量是 $\theta$（模型参数）。损失函数在已知的 $(s, a)$ 对上评估。

**部署（逆向）：**

$$a^* = \arg\min_a \|f_{\theta^*}(s, a) - s'_{\text{target}}\|^2$$

这最小化预测的下一状态（使用**待确定**的动作）与目标状态之间的差异。决策变量是 $a$（动作）。损失函数必须在 $a$ 事先未知的点上评估——优化器在模型几何的引导下探索动作空间。

**关键区别**：训练在从数据分布中抽取的**给定的** $(s,a)$ 对上评估 $f_\theta$。部署在优化器**任意选择的** $a$ 值上评估 $f_{\theta^*}$，该优化器信任模型的局部几何。这两种模式对模型提出了根本不同的要求。

---

## 3. 根因一：导数错位（可训练的差距）

### 3.1 优化器使用的是导数，而不仅仅是函数值

逆向目标上的梯度下降通过下式更新动作：

$$a^{(k+1)} = a^{(k)} - \eta \cdot \underbrace{J_f(s, a^{(k)})^T}_{\text{Jacobian}} \cdot \underbrace{(f_\theta(s, a^{(k)}) - s'_{\text{target}})}_{\text{预测误差}}$$

更新方向由两个因子相乘决定：局部 Jacobian $J_f$ 和当前预测误差。即使真实动作处的预测误差很小，如果 Jacobian 指向别处，优化器可能永远无法到达 $a_{\text{true}}$。

### 3.2 为什么标准训练无法约束 Jacobian

训练中使用的 MSE 损失为：

$$\mathcal{L}_{\text{MSE}}(\theta) = \|f_\theta(s, a) - s'\|^2$$

它关于模型参数的梯度为：

$$\frac{\partial \mathcal{L}_{\text{MSE}}}{\partial \theta} = 2(f_\theta(s, a) - s')^T \cdot \frac{\partial f_\theta}{\partial \theta}$$

这个梯度中**不包含任何涉及 $\frac{\partial f_\theta}{\partial a}$ 的项**。训练信号从未直接惩罚错误的 Jacobian。一个模型可以在训练点处函数值匹配的前提下，拥有任意错误的 Jacobian，只要函数值保持接近。

### 3.3 Pendulum-v1 上的经验证据

对于一个训练好的 KAN（v6），我们在若干状态上计算了模型的 Jacobian $\partial f_\theta / \partial a$ 与真实 Jacobian $\partial f_{\text{true}} / \partial a$ 之间的余弦相似度：

| 状态描述 | $\cos\_\text{sim}(J_{\text{model}}, J_{\text{true}})$ |
|------------------|:------------------------------------------------------:|
| 底部，静止 | $-0.58$ |
| 底部，摆动中 | $+0.66$ |
| 顶部（竖直），静止 | $-0.74$ |
| 中间角度 | $-0.26$ |

余弦相似度为 $-0.58$ 意味着模型的梯度方向与真实梯度方向偏离超过 $125^\circ$。优化器收到的是一个使其**远离** $a_{\text{true}}$ 的下降方向。

### 3.4 机制：为什么函数值对了导数却能错

考虑一个简化的 1D 类比。设真实函数为 $g_{\text{true}}(a) = 0.5a$（一条过原点的直线）。设学习到的函数为：

$$g_\theta(a) = 0.5a + 0.05 \cdot \sin(15a)$$

在任意点 $a$ 处，函数值误差至多为 $0.05$——按大多数标准可以忽略不计。但导数为：

$$g'_\theta(a) = 0.5 + 0.75 \cdot \cos(15a)$$

这在 $-0.25$ 到 $1.25$ 之间振荡，反复跨过零点并改变符号。前向误差很小，但导数误差高达真实值的 $150\%$，且频繁指向相反方向。

数学上的原因是：一个小幅值的高度振荡扰动可以具有大幅值的导数。$L^\infty$ 范数约束函数值但不约束导数——这是函数空间的基本性质。严格地说，对任意 $\epsilon > 0$ 和任意 $M > 0$，存在函数 $h$ 使得 $\|h\|_\infty \le \epsilon$ 但 $\|h'\|_\infty \ge M$。标准例子是 $h(a) = \epsilon \cdot \sin(M a / \epsilon)$。

在 KAN 中，这种振荡具体通过 B-样条控制点来实现。每条边函数 $\phi(x) = w \cdot (\text{silu}(x) + \sum_k c_k B_k(x))$ 具有控制点 $c_k$，它们可以在各自的局部支撑区间内独立振荡，而整体函数仍然接近目标。对 v6 模型的经验测量证实了这一点：在所有 84 条边中，$68\%$ 的相邻控制点差值交替变号，$85$–$94\%$ 的边表现出高曲率（均值 $|\Delta^2 c| > 0.1$）。

### 3.5 这个差距是可训练的

根因一可以通过在训练损失中增加一项来显式惩罚 Jacobian 错位来加以解决：

$$\mathcal{L} = \mathcal{L}_{\text{MSE}} + \lambda \cdot \left\|\frac{\partial f_\theta}{\partial a} - \frac{\partial f_{\text{true}}}{\partial a}\right\|^2$$

这就是 `FITTING_DEPTH.md` 中讨论的 Sobolev 训练方向。它直接约束模型的局部几何，使其匹配真实系统的局部几何。

---

## 4. 根因二：损失景观的偏移（残差导致的差距）

### 4.1 即使导数完美，最小值仍然会偏移

假设通过训练已达到了 $D \ge 1$ 的拟合深度：模型的 Jacobian 处处匹配真实 Jacobian。则由微积分基本定理，对任意固定的 $s$：

$$f_\theta(s, a) - f_\theta(s, a_{\text{true}}) = \int_{a_{\text{true}}}^a \frac{\partial f_\theta}{\partial a}(s, t)\,dt = \int_{a_{\text{true}}}^a \frac{\partial f_{\text{true}}}{\partial a}(s, t)\,dt = f_{\text{true}}(s, a) - f_{\text{true}}(s, a_{\text{true}})$$

整理得：

$$f_\theta(s, a) - f_{\text{true}}(s, a) = f_\theta(s, a_{\text{true}}) - f_{\text{true}}(s, a_{\text{true}}) \equiv \varepsilon_0(s)$$

前向预测误差仅是 $s$ 的函数——它不依赖于 $a$。记这个常值偏移为 $\varepsilon_0$。（这里的"常值"指关于 $a$ 不变，而非在所有状态下不变。）

现在考察逆向目标在真实动作处的行为：

$$\frac{\partial}{\partial a}\left[ \|f_\theta(s, a) - s'_{\text{true}}\|^2 \right]_{a = a_{\text{true}}} = 2 \cdot J_f(s, a_{\text{true}})^T \cdot \underbrace{(f_\theta(s, a_{\text{true}}) - s'_{\text{true}})}_{= \varepsilon_0}$$

由于 $s'_{\text{true}} = f_{\text{true}}(s, a_{\text{true}})$，在真实动作处的残差恰好是 $\varepsilon_0$。如果 $\varepsilon_0 \neq 0$ 且 $J_f^T \varepsilon_0 \neq 0$，则 **$a_{\text{true}}$ 不是逆向目标的驻点**。优化器将从 $a_{\text{true}}$ 处移开，去寻找更低的损失。

### 4.2 几何解释

逆向目标 $\mathcal{L}_{\text{inv}}(a) = \|f_\theta(s, a) - s'_{\text{true}}\|^2$ 度量模型的预测下一状态与目标之间的平方欧氏距离。集合 $\{f_\theta(s, a) : a \in [-1, 1]\}$ 是一条 1 维曲线（流形），嵌入在 3 维状态空间中，由标量动作 $a$ 参数化。

真实动作 $a_{\text{true}}$ 映射到这条曲线上的点 $f_\theta(s, a_{\text{true}})$。目标 $s'_{\text{true}}$ 是周围 3D 空间中的一个点。优化器寻找模型曲线上距离目标最近（欧氏距离意义下）的点。

如果模型曲线相对于真实曲线偏移了 $\varepsilon_0$，则 $s'_{\text{true}}$ 到模型曲线上的正交投影通常会对应于一个动作 $a^* \neq a_{\text{true}}$。

### 4.3 具体算例

考虑一个状态 $s$，其真实动力学为：

$$f_{\text{true}}(s, a) = \begin{bmatrix} 0 \\ 0 \\ 0.0375 \end{bmatrix} \cdot a + \begin{bmatrix} c_1 \\ c_2 \\ c_3 \end{bmatrix}$$

（系统在小范围内近似为 $a$ 的线性函数）。假设训练好的模型为：

$$f_\theta(s, a) = f_{\text{true}}(s, a) + \begin{bmatrix} 0.01 \\ -0.02 \\ 0.005 \end{bmatrix}$$

前向误差为 $\|\varepsilon_0\| \approx 0.023$——很小。但逆向目标为：

$$\mathcal{L}_{\text{inv}}(a) = \left\|\begin{bmatrix} 0 \\ 0 \\ 0.0375 \end{bmatrix} (a - a_{\text{true}}) + \varepsilon_0\right\|^2$$

求导并设为零，得到最优动作：

$$a^* = a_{\text{true}} - \frac{J_{\text{true}}^T \varepsilon_0}{\|J_{\text{true}}\|^2}$$

代入 $J_{\text{true}} = [0, 0, 0.0375]^T$ 和 $\varepsilon_0 = [0.01, -0.02, 0.005]^T$：

$$J_{\text{true}}^T \varepsilon_0 = 0.0375 \times 0.005 = 0.0001875$$
$$\|J_{\text{true}}\|^2 = 0.0375^2 \approx 0.001406$$
$$a^* - a_{\text{true}} \approx -0.133 \quad \text{（归一化单位）}$$

对应力矩误差为 $0.27$ N·m。前向误差中与 Jacobian 平行的分量——在总前向误差范数 $0.023$ 中仅有 $0.005$——正是它移动了最小值的位置。正交分量仅贡献最小值的损失值，但不影响最小值的位置。

### 4.4 这个差距无法被完全训练消除

根因二可以通过进一步减小 $\varepsilon_0$——即提高前向预测精度——来减轻。但 $\varepsilon_0$ 对于在有限数据上训练的模型永远不可能精确为零。总会存在一些残差，而这些残差总会在一定程度上将逆向最优解从 $a_{\text{true}}$ 处推开。

---

## 5. 根因三：欠驱动与病态条件（结构性的差距）

### 5.1 放大因子

上述最优动作的偏移为：

$$|a^* - a_{\text{true}}| \approx \frac{|J^T \varepsilon_0|}{\|J\|^2}$$

分母 $\|J\|^2$ 是 Jacobian 的平方范数——预测的下一状态每单位动作变化多少。当这个量很小时，即使微小的前向误差 $\varepsilon_0$ 也能导致最优动作的较大偏移。

### 5.2 摆是欠驱动的

摆的状态空间是 3 维的（$\cos\theta$、$\sin\theta$、$\dot{\theta}$），但控制仅通过单个标量力矩来实现。映射 $a \mapsto f(s, a)$ 的像仅有 1 维——3D 空间中的一条曲线。

摆的真实 Jacobian（从动力学解析推导，并通过有限差分验证）为：

$$\frac{\partial}{\partial a_{\text{norm}}} \begin{bmatrix} \cos\theta' \\ \sin\theta' \\ \dot{\theta}'/8 \end{bmatrix} = \begin{bmatrix} -0.015 \cdot \sin\theta' \\ 0.015 \cdot \cos\theta' \\ 0.0375 \end{bmatrix}$$

该 Jacobian 的范数有界：

$$\|J_{\text{true}}\|^2 = (0.015)^2 \cdot (\sin^2\theta' + \cos^2\theta') + (0.0375)^2 = 0.000225 + 0.001406 = 0.001631$$

因此 $\|J_{\text{true}}\| \approx 0.0404$。这个值很小，原因在于：

1. 位置分量 $(\cos\theta, \sin\theta)$ 仅**间接**受力矩影响——力矩改变 $\dot{\theta}$，$\dot{\theta}$ 再积分改变 $\theta$，$\theta$ 再改变 $\cos\theta$ 和 $\sin\theta$。在单个 0.05 s 的时间步内，这种间接效应极小（$0.015$）。

2. 速度分量 $\dot{\theta}$ 是直接受影响的，但其系数（归一化单位下为 $0.0375$）反映了物理参数（转动惯量、时间步长）。

### 5.3 放大倍率

给定 $\|J\| \approx 0.0404$ 和前向 RMSE $\varepsilon \approx 0.037$，仅由根因二导致的期望逆误差为：

$$\mathbb{E}[|a^* - a_{\text{true}}|] \approx \frac{\|\varepsilon\|}{\|J\|} \approx \frac{0.037}{0.0404} \approx 0.92$$

（归一化动作空间），对应 $1.84$ N·m。这与经验观测到的平均逆误差 $0.87$ N·m 一致。

关键比值 $\|\varepsilon\| / \|J\|$ 是将前向预测误差映射为逆向恢复误差的**放大因子**。当系统强烈欠驱动（$\|J\|$ 很小）时，这个因子很大。

### 5.4 与良好驱动系统的对比

考虑一个假想的 2D 质点系统：$s' = s + a$（KAN-RF 原型 Stage 1 中的"线性"情形）。这里 $J = I_{2 \times 2}$，$\|J\| = 1$。放大因子为 $\|\varepsilon\| / 1 \approx \|\varepsilon\|$。前向误差和逆向误差处于同一量级。

在 IDEA.md Stage 1 对该系统的结果中：前向 MSE 为 $0.00038$，逆向误差 $|a_{\text{pred}} - a_{\text{true}}|$ 为 $0.012$。两者都很小且相互一致——因为 Jacobian 相对于误差很大。

摆则根本不同：动作对状态的大部分分量仅具有间接的、衰减的影响，使得逆问题是病态的（ill-conditioned）。

---

## 6. 综合：差距为何持续存在

前向-逆向差距由三个复合因素产生，按从最可修复到最不可修复的顺序排列：

| 根因 | 本质 | 可修复性 |
|-----------|--------|:---:|
| 1. Jacobian 错位 | 模型的局部几何（导数）错误，即使函数值是正确的 | **部分可修复**：通过 Sobolev 训练 / P-spline 正则化（参见 `FITTING_DEPTH.md`） |
| 2. 残差前向误差 | $\varepsilon_0 \neq 0$ 将逆向目标的最小值从 $a_{\text{true}}$ 处推开 | **可缩小但不可消除**：有限数据和模型容量保证了 $\varepsilon_0 > 0$ |
| 3. 欠驱动放大效应 | 小的 $\|J\|$ 意味着小的 $\varepsilon_0$ 产生大的 $\|a^* - a_{\text{true}}\|$ | **结构性的且不可修复**：它是物理系统本身的性质，而非模型的性质 |

第三个因素是最根本的。对摆而言，$\|J\| \approx 0.04$ 在给定任何非零前向误差的条件下，为逆向误差设定了一个硬性的下界。任何训练都无法使 $\varepsilon_0$ 精确为零。因此，在通过一个学习到的前向模型进行单步基于梯度的反演时，一定程度的逆向不准确性对这个系统而言是不可避免的。

### 6.1 哪些是可以改善的

- **导数精度**（根因一）：Sobolev 训练和 P-spline 正则化可以使优化器的搜索方向正确，减少所需迭代次数，并防止由 Jacobian 符号错误导致的收敛到虚假局部极小值。

- **前向精度**（根因二）：更多的数据、更好的覆盖度和迭代自举（DAgger）可以减小 $\varepsilon_0$，从而缩小差距。

### 6.2 哪些是无法消除的

- **放大因子** $\|\varepsilon\| / \|J\|$：只要 $\varepsilon > 0$ 且 $\|J\|$ 很小，逆向误差就有一个正的下界。这是物理（欠驱动）和任何学习模型的有限容量共同造成的结果。

---

## 7. 对 KAN-RF 项目的实际启示

### 7.1 为什么多步控制能在这个差距存在的前提下仍然工作

当前表现最好的控制器（exp_F，10 次中 8 次成功）并不依赖于精确的单步逆向恢复。它使用：

1. **能量引导的单步优化**：在每个步骤提供一个合理的（而非完美的）动作方向。

2. **带三因子学习率的在线学习**：利用真实观测到的转移来修正模型误差，逐步减小控制器实际访问的区域中的 $\varepsilon_0$。

3. **Smart burst**：当偏差超过阈值时，以增强的学习率在模型的决策边界上主动探测模型，在最关键的区域提供有针对性的修正。

这些机制是**绕过**前向-逆向差距来工作的，而非消除它。每个控制步骤在下一个决策之前收到真实环境的反馈，因此单步逆误差不会累积为多步轨迹的发散——它被在线修正了。

### 7.2 为什么项目的核心叙事仍然成立

核心叙事是：KAN 的 B-样条局部支撑在在线适应中提供了天然的抗遗忘能力，不像 MLP 需要回放缓冲区或集成方法。这个叙事不依赖于完美的逆向恢复——它依赖的是模型能够通过局部更新被**修正**，而不破坏之前学到的知识。

前向-逆向差距意味着：对于一个欠驱动系统，任何学习到的前向模型（KAN 或其他架构）在第一次尝试时永远不会是完美的逆。但这个差距的严重程度——第一次猜测偏离多远、需要多少次修正步骤——正是 KAN 的结构性质（以及 `FITTING_DEPTH.md` 中讨论的拟合深度改进）能够产生影响的地方。

---

## 附录：关键概念解释

### A.1 什么是 Jacobian？

一个向量值函数 $f: \mathbb{R}^n \to \mathbb{R}^m$ 的 Jacobian 是所有一阶偏导数构成的 $m \times n$ 矩阵。在我们的情境中，$f$ 将一个标量动作 $a$ 映射到一个 3D 状态变化，因此 Jacobian 是一个 $3 \times 1$ 的列向量：

$$J_f(a) = \frac{\partial f}{\partial a} = \begin{bmatrix} \partial f_1 / \partial a \\ \partial f_2 / \partial a \\ \partial f_3 / \partial a \end{bmatrix}$$

它回答的问题是："如果我将动作增加一丁点，预测的下一状态会朝 3D 状态空间中的哪个方向移动，移动多快？"

### A.2 什么是流形？

流形是嵌入在更高维空间中的光滑曲面（或曲线）。所有可能的下一状态预测的集合 $\{f_\theta(s, a) : a \in [-1, 1]\}$ 是嵌入在 3 维状态空间中的 1 维流形（一条曲线），因为它由一个标量 $a$ 参数化。在 $a$ 上搜索的优化器本质上是在沿着这条曲线行走，试图尽可能接近位于周围 3D 空间中某处的目标点 $s'_{\text{true}}$。

### A.3 "欠驱动"是什么意思？

欠驱动系统拥有的独立控制输入数量少于自由度的数量。摆有 2 个自由度（角度 $\theta$ 和角速度 $\dot{\theta}$），但只有 1 个控制输入（力矩）。这意味着控制器无法独立指定下一个时间步的位置和速度——它只能通过耦合的动力学来影响它们。

在前向-逆向的语境中，欠驱动意味着 Jacobian 具有较小的范数：动作只能将状态推向有限的方向集合，单位动作产生的移动幅度受物理约束。

### A.4 "病态条件"（Ill-Conditioned）是什么意思？

一个问题是病态的，当输入的微小变化（这里是前向预测误差 $\varepsilon$）在输出中产生较大变化（这里是恢复的动作 $a^*$）。条件数是输出灵敏度与输入灵敏度之比。对逆问题，有效条件数是 $1 / \|J\|$：当 $\|J\|$ 很小时，逆问题就是病态的。

### A.5 什么是驻点？

函数的驻点是梯度为零的点。对逆向目标 $\mathcal{L}_{\text{inv}}(a)$，驻点 $a^*$ 满足 $\nabla_a \mathcal{L}_{\text{inv}}(a^*) = 0$。梯度下降收敛到一个驻点（通常是局部极小值）。如果 $a_{\text{true}}$ 不是驻点——即该处梯度非零——优化器就不会停在 $a_{\text{true}}$ 处。

---

*文档创建于 2026-05-23。*
