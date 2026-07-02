# KAN-RF: 基于 KAN 可微世界模型的模型预测控制

用 **Kolmogorov-Arnold Network (KAN)** 构建可微分世界模型，实现"一个网络，两种用途"——训练时预测系统动态，部署时通过冻结模型梯度反推最优控制动作。在 Pendulum-v1、CartPole-v1、MountainCar-v0、Acrobot-v1 四个经典控制环境中验证。

## 1. 核心思路

### 为什么用 KAN 而非 MLP

KAN 的每条边是显式的 1D B-样条函数：

$$\phi(x) = w \cdot \text{SiLU}(x) + \sum_k c_k \cdot B_k(x)$$

这带来了三个 MLP 不具备的能力：

| 性质 | 机制 | 用途 |
|------|------|------|
| **局部支撑** | $B_k(x) > 0$ 仅在 $[t_k, t_{k+d+1}]$ 区间 | 在线学习天然抗灾难性遗忘 |
| **分析导数** | $f'(x) = \frac{1}{h}\sum(\Delta c_i) \cdot B_i^{k-1}(x)$ | 精确梯度用于动作优化 |
| **可认证界限** | $\|f^{(d)}\|_\infty \le h^{-d} \cdot \max\|\Delta^d c\|$ | 无需 ensemble 即可量化不确定性 |

### 两步框架

```
训练阶段:                          部署阶段:
(s, a, s') 数据                    当前状态 s_t
     │                                  │
     ▼                                  ▼
BP 训练 KAN 世界模型              冻结的 KAN f_θ
f_θ(s, a) → s'                         │
     │                                  ▼
     ▼                           min_a ||f_θ(s_t, a) - s*||²
冻结的 KAN                              │
     │                                  ▼
     ▼                              a* → 执行
```

## 2. 核心挑战：前向-逆向差距 (Forward-Inverse Gap)

**这是整个项目最重要的发现。**

训练时世界模型的前向预测非常准确（RMSE ≈ 0.037，约 0.14 rad 角度误差）。但在部署时，用梯度下降从状态反推动作的误差高达 0.87 N·m（力矩范围 ±2.0 N·m 的 44%）。仅有 36.5% 的恢复落在 0.2 N·m 以内。

**前向 MSE 好 ≠ 反推动作准。**

### 三个根因

**根因一：Jacobian 不匹配（可训练）**

MSE 损失 $\mathcal{L} = \|f_\theta(s,a) - s'\|^2$ 的梯度中**不包含** $\partial f_\theta / \partial a$ 的项。一个模型可以在任意低的 MSE 下拥有完全错误的导数。

实测：训练后模型的 Jacobian 与真实 Jacobian 的余弦相似度仅为 **0.099**（约 84° 偏差）。

直观类比：真实函数 $g(a) = 0.5a$，模型学了 $g_\theta(a) = 0.5a + 0.05\sin(15a)$。函数值误差 ≤ 0.05，但导数为 $0.5 + 0.75\cos(15a)$，在 -0.25 到 1.25 之间振荡——**经常指向相反方向**。

**根因二：残差偏移（可减少但不可消除）**

即使 Jacobian 完美匹配，前向残差 $\varepsilon_0 = f_\theta(s, a_{\text{true}}) - f_{\text{true}}(s, a_{\text{true}}) \neq 0$ 也会将逆目标的最小值从 $a_{\text{true}}$ 移开：

$$a^* = a_{\text{true}} - \frac{J^T \varepsilon_0}{\|J\|^2}$$

对于有限数据和模型容量，$\varepsilon_0$ 永远不可能精确为零。

**根因三：欠驱动放大效应（结构性、不可消除）**

Pendulum 有 2 个自由度（角度、角速度）但只有 1 个控制输入（力矩）。Jacobian 的范数 $\|J\| \approx 0.04$ 非常小。前向误差被放大：

$$\mathbb{E}[|a^* - a_{\text{true}}|] \approx \frac{\|\varepsilon\|}{\|J\|} \approx \frac{0.037}{0.0404} \approx 0.92$$

放大因子 ≈ 25。即使前向误差为 0.037，逆向动作误差仍约 0.92（归一化单位）。**这是物理系统的结构性约束，不是模型能解决的。**

| 根因 | 性质 | 可修复性 |
|------|------|:---:|
| 1. Jacobian 不匹配 | 模型训练不约束导数 | **部分可修**（Sobolev 训练） |
| 2. 残差偏移 | 有限数据和容量 | **可减少不可消除** |
| 3. 欠驱动放大 | 物理系统的结构性约束 | **不可消除** |

## 3. 解决方案一：拟合深度训练 (Fitting Depth)

### 核心概念

定义**拟合深度 D** 为模型导数匹配真实导数的最高阶数：

$$D = \max\left\{d \;\middle|\; \left\|\frac{\partial^k f_\theta}{\partial a^k} - \frac{\partial^k f_{\text{true}}}{\partial a^k}\right\| \le \epsilon_k,\; \forall k \in [0, d]\right\}$$

标准 MSE 训练只能达到 $D=0$。梯度下降优化 $a$ 时跟随的是 $\partial f_\theta/\partial a$，因此 $D \ge 1$ 是关键目标。

### 三级训练框架

**Level 1: MOPS (Multi-Order P-Spline)** — 参数空间正则化

直接惩罚 B-样条控制点的二阶差分：
$$\mathcal{L}_{\text{MOPS}} = \lambda \cdot \sum_{\text{所有边}} \|\Delta^2 c\|^2$$

利用 P-spline 恒等式 $\|\Delta^d c\|^2 \approx h^{2d-1} \int [f^{(d-1)}(x)]^2 dx$，控制二阶差分等价于惩罚一阶导数能量。仅作用于 KAN 的 B-样条控制点，**不需要真实 Jacobian**。

**Level 3: CWS (Controllability-Weighted Sobolev)** — 输出空间 Jacobian 匹配

当真实 Jacobian 可用时（解析动力学），直接惩罚模型 Jacobian 与真实 Jacobian 的偏差：
$$\mathcal{L}_{\text{CWS}} = \nu \cdot \|w \odot (\partial f_\theta/\partial a - J_{\text{true}})\|^2$$

可操控性加权：直接受控维度（如 $\dot{\theta}$）获得更高的权重。

### 实验结果

| 方法 | Forward MSE | Jacobian cos_sim | ‖Δ²c‖ | 逆误差均值 |
|------|:---:|:---:|:---:|:---:|
| Baseline (MSE only) | 0.001785 | 0.099 | 0.0986 | 0.507 |
| MOPS λ=0.1 | 0.001907 | 0.237 | 0.0208 | 0.428 |
| CWS ν=1.0 | 0.000337 | **0.979** | 0.1020 | 0.291 |
| **Hybrid** | **0.000198** | 0.924 | **0.0007** | **0.228** |

**关键发现**：MOPS 和 CWS 通过不同机制起作用——MOPS 降低控制点粗糙度（141×）但不改善方向，CWS 改善 Jacobian 方向（0.099→0.979）但不改善光滑度。两者互补，Hybrid 在所有指标上最优。逆误差从 1.01 N·m 降至 0.46 N·m（降 55%），但残余误差验证了根因三的存在。

## 4. 解决方案二：多时间尺度世界模型

### 动机

根因三指出单步 Jacobian 太小（$\|J\| \approx 0.04$）。**但如果预测的是一段较长时间后的状态，动作对状态的影响会更大。**

### 方法

将世界模型从 $f(s, a) \rightarrow s_{t+dt}$ 扩展为 $f(s, a, k) \rightarrow s_{t+k\cdot dt}$，$k \in \{1, 2, 4, 8, 16\}$。输入多一维 $k/16$ 表示时间尺度。

### 架构

- 世界模型：KAN([5, 16, 3])，1152 参数
- 训练数据：对每个 (s, a)，用解析动力学生成 5 个时间尺度的 $s_{t+k\cdot dt}$
- 训练方法：MOPS（λ=0.1）

### 决策网络（Plan A）

将逆优化的结果蒸馏为一个快速的前馈决策网络：

$$(s, s^*) \xrightarrow{\text{KAN([6,12,2])}} (a, k)$$

决策网络同时输出动作和时间尺度。训练标签通过逆优化生成（对每个状态，尝试所有 k，选最佳 (a, k)）。

**结果：9/10 Pendulum swing-up 成功。** k=16 被选中 75% 的时间，验证了大时间尺度的必要性。

## 5. 核心难题：k-选择 (k-Selection)

### k 的困境

多尺度世界模型引入了新问题：在每个状态下，应该选哪个 k？

- **k 太小**：Jacobian 太小，前向误差被放大（根因三）
- **k 太大**：预测误差累积。混沌系统（如 Acrobot）中误差指数增长
- **需要状态相关的 k**：Pendulum 底部用 k=16（需要大 Jacobian），顶部用 k=1（精细控制）

### 尝试过的方法

**方法 1：世界模型自评 → 失败**

思路：让世界模型评估"哪个 k 的预测结果最好"。

失败原因：**循环推理**。世界模型在大 k 时预测更大的状态变化（更乐观），但预测更不准，导致偏向大 k。Acrobot 因此选了 k=8（0% 成功率），而正确的 k=1 是 92%。

**方法 2：CKS — 可认证 k-选择 → 理论正确，工程失败**

思路：利用 B-样条导数上界 $\|f^{(d)}\|_\infty \le h^{-d} \cdot \max|\Delta^d c|$，分析性地计算每个 k 的预测误差上界：

$$k_{\text{cert}}(s) = \max\{k : E_k(s)/G(s,k) \le \varepsilon\}$$

建立了完整的 6-定理理论框架，包括 PAC 式保证和四个系统类型的自动适配。

失败原因：多尺度世界模型在 k=1 的基础预测误差已经太高（L2≈0.38），导致对所有 k 的认证上界都太松，无法区分。

**方法 3：不确定性加权 MPC → 区分力不够**

思路：用 B-样条激活密度 $\rho(s)$ 作为不确定性代理，加权 MPC 评分。

问题：训练数据密度均匀，$\rho(s)$ 跨状态变化太小，加上混沌系统的 Jacobian 补偿使惩罚项归零。

## 6. 最终方案：动作探索器 (10/10)

### 思路

不再试图完美解决 k-选择。让系统在卡住时直接在环境中尝试不同动作，将成功经验蒸馏回决策网络。

### 三个全通用机制

**1. 检测卡住**：角度误差连续 N 步不下降 → 当前动作方向可能错误。

**2. 原地尝试候选动作**：
- 保存环境状态 `env.unwrapped.state`
- 在同一状态下尝试多个候选动作（模型建议的、反方向的、随机的）
- 用真实环境反馈衡量改善程度（不依赖世界模型预测）
- 恢复状态，选最优候选

**3. 记录纠正标签**：$(s, a_{\text{wrong}}) \rightarrow (s, a_{\text{correct}})$。用积累的纠正标签微调决策网络。

### 结果

**Pendulum-v1: 10/10 全通过。** 之前失败的所有 Trial（2, 3, 6）全部修复。这是 Pendulum 上的最优方案。

## 7. 其他尝试

### WM+V：世界模型 + 价值网络

核心思路：不再选 k。世界模型永远只用 k=1（最准），用一个微型 MLP 价值网络 V(s) 估计"从 s 出发的累积未来奖励"。

| 组件 | 职责 | 训练方式 |
|------|------|------|
| 世界模型 f(s,a) | 单步预测 s' | 离线 MOPS |
| 价值网络 V(s) | 估计累积未来奖励 | 在线 TD(0) |

与 AlphaGo 的 MCTS + value network 同构。世界模型替代 MCTS rollout，V(s) 替代 value network。

**结果：22/30 (73%)。** 瓶颈是冷启动鸡-蛋死锁——V(s) 初始化为接近零的随机值，MPC 退化为贪婪策略。

### decision_v2：KAN 特征 → MLP 决策

冻结的 KAN 提取物理特征（drift, Jacobian, ctrl, align, trust），喂给微型 MLP 输出动作。

**结果：7/10 天花板。** 特征质量从 0.03 提到 0.80，但性能不变——瓶颈在"把 48 条边函数压缩成 5 个标量再让 MLP 重新推导"这个范式本身。

### 最新方向（待实现）：KAN 作为可微裁判

KAN 的知识不应该作为输入特征，而应该作为**训练时的可微裁判**：

```
训练时:                          部署时:
s → [π_θ] → a                   s → [π_θ] → a
      │                               (纯前向，KAN 不参与)
      ▼
f_KAN(s, a) → s'_pred
      │
      ▼
loss = ||s' - s*||²
      │
      ▼
θ ← θ - α · ∂loss/∂θ            ← 梯度经过 KAN 的可信导数
```

- KAN 不出现在决策网络的输入中
- KAN 出现在训练损失的计算中——它评价"这个动作好不好"
- 梯度通过 KAN 反向传播到 π_θ——KAN 用可信的导数告诉网络"动作应该往哪调"
- 部署时只有 π_θ 前向传播，KAN 不需要

## 8. 四环境完整结果

| 环境 | 世界模型 | 控制方法 | 成功率 | 关键洞察 |
|------|------|------|:---:|------|
| Pendulum-v1 | [5,16,3] 多尺度 | 决策网络 + 动作探索器 | **100%** | 原地尝试真实环境反馈是关键 |
| Pendulum-v1 | [4,12,3] 单尺度 | WM+V (k=4 预训练) | 77% | 预训练改善早期，天花板受限于贪婪 MPC |
| CartPole-v1 | [7,20,4] 多尺度 | MPC k=2 + 能量评分 | **99%** | 失败是小车漂移，非杆子倒下 |
| MountainCar-v0 | [6,16,2] 多尺度 | MPC k=4 + 能量评分 | **100%** | 评分函数从"最大化位置"改为"最大化能量"是关键 |
| Acrobot-v1 | [10,24,6] 多尺度 | MPC k=1 | **92%** | 混沌系统，k=8 时误差爆炸 |

## 9. 已确认的死路

| 方法 | 死因 |
|------|------|
| 世界模型自评选 k | 循环推理，模型高估长 horizon 预测能力 |
| CKS 认证 k-选择 | 理论正确，基础误差太高导致上界太松 |
| 单点逆优化 | 根因三是结构性约束，提高模型精度无法解决 |
| KAN 知识硬压缩成标量特征 | 信息丢失，性能天花板 7/10 |
| 在线逐样本 SGD | 高方差单样本梯度破坏 B-样条连续性，模型爆炸 |

## 10. 关键洞察总结

1. **前向 MSE 好 ≠ 反推动作准。** 根因三（欠驱动放大）是结构性的物理约束，即使模型完美也存在。
2. **多尺度世界模型部分解决根因三。** k=16 的 Jacobian 比 k=1 大 16 倍，但代价是引入了 k-选择问题。
3. **动作探索器是最强的方案。** 直接在真实环境中尝试候选动作，不依赖模型预测未来——没有模型误差放大问题。
4. **B-样条局部支撑是 KAN 的核心优势。** 这是本项目与其他 MBRL 工作的本质区别。
5. **Hybrid 模型的"反直觉"表现。** 越精确的模型 → $\|J\|$ 越准确越小 → 放大因子越大 → 求逆越不稳定。
6. **动作探索器 (10/10) 是绕过问题，不是解决问题。** 它证明了一个控制器可以达到 10/10，但没有证明 KAN 在决策中的价值。本项目的学术贡献在于展示 KAN 如何帮助决策，而不只是帮助预测。

## 11. 项目结构

```
KAN-RF/
├── kanrf/                    # 核心库 (pip install -e .)
│   ├── _bspline.py           # B-样条 Cox-de Boor 递归
│   ├── _layer.py             # KAN 层 (φ = w·SiLU + Σc·B)
│   ├── _network.py           # 多层 KAN
│   ├── _uncertainty.py       # B-样条激活密度 → 认知不确定性
│   └── _regularization.py    # MOPS P-spline + CWS Jacobian 对齐
│
├── control/                  # 控制算法库
│   ├── shoot.py              # 多步 shooting 规划器
│   ├── strategy_v2.py        # 策略层 (能量场引导)
│   ├── execute_v2.py         # 执行层 (Gauss-Newton + 可控性加权)
│   ├── action_explorer.py    # 动作探索器 (10/10 关键)
│   ├── online_learning_v2.py # 三因子动态学习率在线更新
│   └── ...
│
├── scripts/
│   ├── train/                # 训练脚本 (15个)
│   ├── eval/                 # 评估脚本 (10个)
│   ├── run/                  # 运行入口 (4个)
│   └── data/                 # 数据生成 (7个)
│
├── experiments/              # 历史实验 exp_A → exp_G
├── decision_v2/              # KAN 特征 → MLP 决策
├── wm_v/                     # 世界模型 + 价值网络
├── cks/                      # 可认证 k-选择理论
├── acrobot/ mountaincar/     # 子环境
├── docs/                     # 理论文档 + 历史归档
│   ├── theory/
│   │   ├── FORWARD_INVERSE_GAP.md       # 前向-逆向差距三根因
│   │   ├── FITTING_DEPTH.md             # 拟合深度 + MOPS/CWS
│   │   ├── CONTINUOUS_LEARNING.md       # B-样条持续学习理论
│   │   └── CERTIFIED_K_SELECTION.md    # CKS 6-定理框架
│   ├── archive/               # 历史/过程文档
│   └── handovers/             # 交接文档
└── pyproject.toml
```

## 12. 快速开始

```bash
conda activate pyt
cd KAN-RF
pip install -e .

# 训练 Pendulum 世界模型 (MOPS)
python scripts/train/train_mops.py --lam 0.1

# 评估决策网络
python scripts/eval/eval_ms.py

# 动作探索器 (10/10)
python scripts/eval/eval_action_explore.py --episodes 10
```

---

*项目基于 [KAN (Kolmogorov-Arnold Networks)](https://github.com/KindXiaoming/pykan) 架构。*
