# 认知—决策框架根因分析与理论重置

更新时间：2026-07-24

## 1. 核心结论

当前困难不是某一个损失函数、正则系数或在线更新门槛没有调好，而是我们
把三个本质不同的问题压缩成了一个“预测网络参数传给决策网络”的问题：

1. **辨识问题**：从闭环转移中识别环境发生了什么变化；
2. **控制相关性问题**：判断哪些动力学变化真正会改变最优动作；
3. **策略运输问题**：把控制相关的动力学变化变成决策网络的正确变化。

单步预测损失只部分解决第一个问题。直接搬运参数、局部 Riccati、IV、
bootstrap 和稳定门槛，都在不同程度上跳过了第二或第三个问题。因此才会
反复出现：

- 单步预测更准，但控制更差；
- \(B\) 估计明显改善，但策略仍崩溃；
- Oracle 动力学可达约 98%，在线点估计只能达到约 70%–85%；
- 单个种子达到 90% 以上，换一条闭环数据轨迹便降到 0%；
- 不确定性门槛减少灾难更新，却稳定地保护了有偏模型。

这个问题有解，但必须收缩研究命题：**不可能从一个源环境推断任意未知
物理变化并保证立即适应**。可研究且有价值的目标应是：

> 对一类共享状态、动作和局部动力学结构的系统，在只用源环境训练任务
> 策略和基础世界模型后，通过少量目标闭环转移，识别控制相关的动力学
> 变化，并沿由最优性条件导出的策略切方向快速适应。

## 2. 当前代码实际实现了什么

当前主实验并不是原始设想中的完整认知—决策网络。

### 2.1 认知部分

`TwoLinkProtoKANBasis` 使用每个状态维度上的高斯局部原型，然后接一个
64 维 MLP 交互网络：

\[
\phi(s)=
\left[
1,\ \phi_{\rm local}(s),\
\operatorname{MLP}(\phi_{\rm local}(s))
\right].
\]

它并不是纯 B-spline KAN。在线实验中该主干被
`requires_grad_(False)` 完全冻结。

### 2.2 在线学习部分

最新的 `randomized_moment`、`direct` 和 IV 路线在线估计的是目标点附近
的局部线性算子

\[
x_{t+1}\approx A_t x_t+B_tu_t,
\]

而不是持续更新完整 ProtoKAN 世界模型。ProtoKAN 主要提供源先验或源
Jacobian。

### 2.3 决策部分

决策不是一个包含认知参数的 Actor 前向网络，而是

\[
K_t=\operatorname{Riccati}(A_t,B_t,Q,R),\qquad
u_t=-K_tx_t.
\]

所以最近的实验准确名称应是：

> 带冻结 ProtoKAN 源先验的在线局部自适应 LQR。

它可以作为强诊断基线，但不能直接代表最初论文架构。

## 3. 数学根因

### 3.1 闭环辨识的不可辨识性

在线数据来自当前策略：

\[
u_t=\pi_t(x_t)+e_t.
\]

在线性局部情形中，

\[
x_{t+1}
=Ax_t+Bu_t
=(A+BK_t)x_t+Be_t.
\]

如果没有充分且独立的激励，数据首先识别的是闭环组合 \(A+BK_t\)，而
不是分别识别 \(A\) 与 \(B\)。随机激励可以帮助识别 \(B\)，但仍存在：

- 动作裁剪使实际激励与原始激励不一致；
- 更新策略会改变后续状态分布；
- 非线性漂移会进入局部 \(A\)；
- 少样本下 \(A\)、\(B\) 的联合方向误差仍很大。

这解释了为什么 IV 曾把 \(B\) 误差降到约 18%，控制仍然可能接近 0%。

### 3.2 预测范数与控制范数不一致

普通认知损失是

\[
\mathcal L_{\rm pred}
=
\mathbb E\left[
\|f_\theta(x,u)-x'\|^2
\right].
\]

但策略真正敏感的是模型误差经过价值梯度和闭环占用分布后的投影，例如

\[
\mathcal L_{\rm ctrl}
\approx
\mathbb E_{d^\pi}
\left[
\left(
\nabla_xV(x')^\top
(f_\theta(x,u)-f^\star(x,u))
\right)^2
\right].
\]

两个模型可以有几乎相同的 MSE，却在动作通道、目标附近 Jacobian 或
长期可达方向上完全不同。Value-Aware Model Learning、Value-Targeted
Regression 和 control-oriented identification 都是在解决这一目标错配。

### 3.3 从模型到控制器的映射可能病态

当前决策映射为

\[
(A,B)\longmapsto K(A,B).
\]

其一阶扰动满足

\[
\Delta K
\approx
D_AK[\Delta A]+D_BK[\Delta B].
\]

当执行器变弱、动作代价 \(R\) 很小、系统接近可稳定边界或动作发生饱和
时，\(\|D K\|\) 会显著增大。于是“一步预测误差不大”并不能推出
“策略误差不大”。weak-actuator 环境正好同时具备这些条件，所以它不是
普通难度增加，而是暴露了决策映射的条件数问题。

### 3.4 单源环境下的未知变化方向不可学习

源环境只提供一个函数 \(f_0\)。它能够告诉网络“当前世界如何运行”，但
不能告诉网络“世界将来可能沿哪些方向变化”。

形式上，存在两个目标环境 \(f_1,f_2\)，它们在算法前 \(n\) 步访问过的
所有状态动作上完全一致，却在未访问区域需要相反动作。任何算法在前
\(n\) 步观察到相同历史，只能输出相同策略，因此至少会在一个环境失败。

所以“一个源环境训练后，对任意物理变化零代价快速适应”在信息论上
不可能。快速适应必须依赖至少一种额外结构：

- 已知变化属于低维或稀疏函数子空间；
- 动力学满足控制仿射、守恒、对称性等结构；
- 可以进行有信息量的在线交互；
- 或离线训练阶段见过多环境变化。

我们的论文必须明确选择哪一种假设。

### 3.5 原始参数运输在表示上不适定

神经网络参数不是唯一的物理坐标。神经元置换、缩放以及过参数化平坦
方向可以让不同参数实现同一个函数。因此

\[
\theta_1\ne\theta_2,\qquad
f_{\theta_1}=f_{\theta_2}
\]

完全可能成立。直接把“全部认知参数”映射为决策参数，要求决策网络理解
认知网络任意参数化选择，而不是理解物理函数本身。

KAN 的局部基函数可以减轻部分表示漂移，但不能自动消除层间组合、
基函数重排、缩放以及同输入域内规律变化造成的非唯一性。因此更合理的
接口必须位于**函数空间、导数空间或规范化算子空间**，不能依赖未经规范
化的原始权重。

## 4. 深度学习与 KAN 视角的根因

### 4.1 当前表示没有把“稳定规律”和“环境变化”分开

当前 ProtoKAN 把所有规律编码到同一套特征与系数中。在线全量更新会
产生遗忘；冻结主干只更新线性头又可能没有足够表达力。这不是优化器
问题，而是稳定性—可塑性没有结构化分解。

### 4.2 单环境训练无法自动产生有意义的环境 latent

如果训练数据只来自一个环境，那么任意常量 latent 都能达到同样预测
损失。网络没有统计信号把某些维度解释成质量、重力或控制增益。CaDM、
CoDA 和 conditional neural processes 通常依赖多环境或任务分布来学习
这种上下文几何；这正是它们与我们的单源设定之间的重要差异。

### 4.3 KAN 的局部可塑性有条件

B-spline 的紧支撑只在新旧任务激活不同输入区域时自然隔离梯度。我们的
物理变化前后访问的是相同状态动作区域，只是映射关系改变，因此会更新
同一批 spline 系数。KAN 不能单靠局部支撑解决同域异规律的遗忘问题。

### 4.4 当前认知损失没有学习“如何影响动作”

预测训练只优化 \(f_\theta\)。决策网络真正需要的是

\[
\frac{\partial \pi^\star}{\partial f},
\]

即最优策略对动力学变化的敏感性。我们过去尝试的参数映射、探针、可控
算子和 Riccati 都是在近似这个对象，但没有把它明确建模为核心学习对象。

## 5. 建议的新核心：控制切空间认知运输

建议将研究问题重写为 **Control-Tangent Cognitive Transport（CTCT）**，
中文可称“控制切空间认知运输”。

### 5.1 认知模型：基础规律加规范化变化字典

\[
x_{t+1}
=
f_{\omega,0}(x_t,u_t)
+
D_\omega(x_t,u_t)c_t.
\]

其中：

- \(f_{\omega,0}\) 是源环境基础世界模型；
- \(D_\omega\) 是固定、规范化、具有局部支撑和跨状态交互的 KAN 变化
  字典；
- \(c_t\) 是在线推断的低维或稀疏变化系数；
- 在线阶段优先更新 \(c_t\)，不直接覆盖基础规律。

这里不声称字典能从单源数据“发现所有未来物理参数”。它是我们明确加入
的函数类假设：目标变化在该字典张成空间内近似稀疏。

KAN 的真正作用应是：

- 构造局部、可微、可规范化的函数变化基；
- 允许按 knot 或基函数维护后验重要性；
- 在必要时扩展新基，而不覆盖已有基；
- 提供解析 Jacobian 和可能的符号化解释。

### 5.2 在线认知：推断后验而不是重写整个网络

\[
p(c_t\mid H_t)
\propto
p(s_{t+1}\mid s_t,a_t,c_t)\,
p(c_t\mid H_{t-1}).
\]

可使用递归最小二乘、稀疏贝叶斯或小型 amortized inference encoder。
变化检测只决定是否新建或召回 context，不使用环境标签。

抗遗忘不再依靠“KAN 天生不会忘”，而依靠：

- 基础模型冻结或缓慢更新；
- context 后验与环境记忆分离；
- 函数空间锚定旧 context；
- 新变化无法由旧字典解释时才进行结构扩展。

### 5.3 决策运输：由最优性条件推导，不学习任意参数映射

设 Actor 参数为 \(\phi\)，认知变化系数为 \(c\)，源策略满足

\[
F(\phi,c)
=
\nabla_\phi J(\phi;c)
=0.
\]

由隐函数定理，

\[
\boxed{
\frac{\partial\phi^\star}{\partial c}
=
-
\left(
\frac{\partial F}{\partial\phi}
\right)^{-1}
\frac{\partial F}{\partial c}
}
\]

即

\[
S_{\phi c}
=
-
H_{\phi\phi}^{-1}H_{\phi c}.
\]

这给出动力学变化到策略变化的数学运输算子。在线初始化为

\[
\phi_t
=
\phi_0+S_{\phi c}\hat c_t,
\]

较大变化时使用连续同伦或重新在线估计局部切映射。决策网络仍可使用真实
奖励进行 Actor-Critic 持续学习：

\[
\phi_t
=
\phi_0+S_{\phi c}\hat c_t+\Delta\phi_t^{\rm RL}.
\]

认知负责提供**有方向的快速初始化与更新**，Actor-Critic 负责修正高阶
误差。这符合我们已经确认的原则：不要求映射一步达到 100%，但必须让
恢复显著快于无认知持续学习。

### 5.4 强制深度融合

运输结果不应作为可被 Actor 忽略的附加输入，而应直接调制决策网络参数
或中间层：

\[
h_{\ell+1}
=
\sigma\left(
\left[
W_\ell+
\sum_j \hat c_{t,j}\Delta W_{\ell,j}
\right]h_\ell
\right).
\]

\(\Delta W_{\ell,j}\) 不是任意 hypernetwork 输出，而是由策略敏感性
\(S_{\phi c}\) 分解得到或受其约束。这样所有前向决策都经过认知变化所
调制的权重，同时保留独立 Actor 参数用于任务学习。

### 5.5 控制相关辨识

认知系数不应按普通预测 MSE 等权估计。由策略运输矩阵定义控制度量

\[
G_c
=
S_{\phi c}^\top H_{\phi\phi}S_{\phi c}.
\]

在线辨识优先降低

\[
\operatorname{tr}(G_c\Sigma_c),
\]

即最可能改变策略的认知不确定性。这连接了：

- 世界模型预测；
- task-optimal exploration；
- 决策网络敏感性；
- 快速成功率恢复。

如果需要主动动作，也只探索会降低
\(\operatorname{tr}(G_c\Sigma_c)\) 的方向，而不是无目的地增加高斯噪声。

## 6. 与已有工作的边界

以下思想本身已有研究，不能单独作为创新点：

- 认知模型与决策模型分离；
- latent context 条件动力学；
- value-aware / policy-aware 模型损失；
- 通过随机激励做自适应 LQR；
- 对最优控制或 Bellman 方程做隐式微分；
- KAN 的局部 spline 与 continual learning。

可能形成论文创新的组合必须更具体：

1. **单源策略训练**下的规范化 KAN 动力学变化字典；
2. 从 Bellman/Actor 最优性条件得到的
   **认知变化—策略参数切运输算子**；
3. 由该运输算子定义的
   **控制相关在线认知后验与探索度量**；
4. context 隔离、函数锚定和基函数扩展组成的
   **持续适应与抗遗忘机制**；
5. 证明或经验验证：控制误差由
   \(\|c-\hat c\|_{G_c}\) 而非普通预测 MSE 更好地解释。

但必须注意：
“control-oriented model learning with implicit differentiation”已经存在，
因此仅把 KAN 换进已有框架不够。论文的理论核心应是**策略切运输、KAN
变化字典和在线后验三者之间的结构化对应关系**。

## 7. 决定性实验顺序

这次不能再直接把所有模块组合后看成功率。必须逐层证伪。

### Gate A：Oracle 策略运输

暂时提供目标动力学变化系数 \(c^\star\)，但不提供目标最优策略。验证

\[
\phi_0+S_{\phi c}c^\star
\]

是否稳定优于源 Actor。

若失败，说明策略切映射不够，不应研究在线认知。

### Gate B：Oracle 函数字典可表达性

用目标转移离线拟合 \(c^\star\)，比较

\[
f_0+D c^\star
\]

与完整目标模型的控制相关误差，而不仅是预测 MSE。

若失败，说明变化字典没有覆盖目标物理变化。

### Gate C：少样本认知可辨识性

固定 Actor 和数据分布，比较 16、32、64、128 条数据下
\(\hat c\) 的普通误差与 \(G_c\)-加权误差。

若普通误差下降而控制误差不下降，说明在线采样不具任务信息量。

### Gate D：在线闭环恢复

最后才组合：

1. 源 Actor 与世界模型；
2. 目标环境切换；
3. context 后验更新；
4. 切运输调制 Actor；
5. Actor-Critic 真实反馈持续学习。

必须比较：

- 冻结源 Actor；
- 仅 Actor-Critic 持续学习；
- 仅认知运输；
- 认知运输 + Actor-Critic；
- Oracle context；
- MLP 字典替代 KAN 字典。

## 8. 现在应停止做什么

- 停止继续扫描 IV 正则、bootstrap 分位数和谱半径门槛；
- 停止把一次偶然的单种子提升当成结构成立；
- 停止把局部 \(A,B\) 拟合称为完整 ProtoKAN 持续认知；
- 停止假设网络原始参数天然具有稳定物理语义；
- 停止在 Gate A 未通过前进行完整在线组合实验。

## 9. 参考研究

- [Value-Aware Loss Function for Model-based Reinforcement Learning,
  AISTATS 2017](https://proceedings.mlr.press/v54/farahmand17a.html)
- [Model-Based Reinforcement Learning with Value-Targeted Regression,
  ICML 2020](https://proceedings.mlr.press/v119/ayoub20a.html)
- [Naive Exploration is Optimal for Online LQR,
  ICML 2020](https://proceedings.mlr.press/v119/simchowitz20a.html)
- [Towards a Dimension-Free Understanding of Adaptive Linear Control,
  COLT 2021](https://proceedings.mlr.press/v134/perdomo21a.html)
- [Optimal Exploration for Model-Based RL in Nonlinear Systems,
  NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/31e018f43ab9c7065c058cc2c5848128-Abstract-Conference.html)
- [Context-aware Dynamics Model,
  ICML 2020](https://proceedings.mlr.press/v119/lee20g.html)
- [Generalizing to New Physical Systems via Context-Informed Dynamics Model,
  ICML 2022](https://proceedings.mlr.press/v162/kirchmeyer22a.html)
- [Neural Processes with Event Triggers for Fast Adaptation to Changes,
  L4DC 2024](https://proceedings.mlr.press/v242/brunzema24a.html)
- [Revisiting Implicit Differentiation for Learning Problems in Optimal
  Control, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/bcfcf7232cb74e1ef82d751880ff835b-Abstract-Conference.html)
- [Calibrated Value-Aware Model Learning with Probabilistic Environment
  Models, ICML 2025](https://proceedings.mlr.press/v267/voelcker25a.html)
- [Catastrophic Forgetting in Kolmogorov-Arnold Networks,
  AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39697)

## 10. Gate A：Oracle 控制切运输结果

我们首先在 LQR 特例中验证策略切运输，而没有直接实现神经 Actor 的完整
Bellman 隐式微分。目标环境的真实 \((\Delta A,\Delta B)\) 仅用于提供
Oracle 认知变化方向；运输过程中不使用目标最优增益。

在源点通过中心差分计算

\[
DK_0[\Delta A,\Delta B]
\approx
\frac{
K(A_0+\epsilon\Delta A,B_0+\epsilon\Delta B)
-
K(A_0-\epsilon\Delta A,B_0-\epsilon\Delta B)
}{2\epsilon}.
\]

一次性一阶运输

\[
K_{\rm one}=K_0+DK_0[\Delta A,\Delta B]
\]

的结果具有明显局部性：

- weak-actuator 从源增益约 65.1% 提高到约 80.6%；
- long-link 从 100% 降到约 77.1%；
- light-low-gravity 和 feasible-composite 仍为 0%；
- 在 25% 物理变化处，四个环境的切线增益相对误差仅约
  0.7%–13.1%，到完整变化时增至约 27%–103%。

这说明切方向本身含有正确控制信息，但一次映射超出了局部有效半径。

随后沿线性认知路径

\[
(A(\alpha),B(\alpha))
=
(A_0,B_0)+\alpha(\Delta A,\Delta B)
\]

进行 2、4、8 段 Euler 切运输，每段只计算当前位置的局部策略导数。三种
随机初始状态种子的平均成功率如下：

| 环境 | 源增益 | 一次运输 | 2 段 | 4 段 | 8 段 | Oracle |
|---|---:|---:|---:|---:|---:|---:|
| long-link | 100% | 77.1% | 100% | 100% | 100% | 100% |
| light-low-gravity | 0% | 0% | 0% | 0% | 100% | 100% |
| feasible-composite | 0% | 0% | 6.6% | 100% | 100% | 100% |
| weak-actuator | 65.1% | 80.6% | 89.7% | 95.8% | 98.3% | 98.9% |

因此 Gate A 在解析 LQR 特例中通过。实验支持的不是“一次参数映射”，而是：

\[
\boxed{
\text{在线认知路径}
\longrightarrow
\text{局部策略敏感性}
\longrightarrow
\text{连续策略运输}
}.
\]

该结果只证明策略运输机制存在，并未证明在线认知能够从少量闭环转移中
恢复正确变化路径。下一步应进入 Gate B：构造不读取目标参数的规范化 KAN
变化字典，并首先验证目标动力学变化是否能由少量字典系数表达。

## 11. Gate B：当前 ProtoKAN 变化字典结果

首先确认 `two_link_protokan_basis_seed107.pt` 的训练脚本只使用源环境，
不存在目标动力学泄漏。

### 11.1 完整表达能力成立，但坐标严重病态

使用 8192 条目标转移拟合完整 97 组 ProtoKAN context 后：

- long-link、light-low-gravity、feasible-composite 均达到 100%；
- weak-actuator 达到约 97.95%；
- 局部 \(A,B\) 与控制增益误差均显著下降。

所以源训练函数基能够表达目标动力学。

但是按归一化参数组幅值保留 4、8、16、32 或 64 组时，四个环境成功率
几乎全部接近 0%。即使保留约 94%–96% 的参数变化能量，预测 RMSE 仍从
个位数上升到约 1400–1700。

原因是当前表示包含：

- softmax 原型之间的线性相关性；
- 常数项与原型和之间的冗余；
- 64 维 MLP 交互特征；
- 动作乘积特征与状态特征之间的强相关。

完整解依靠大量巨大正负系数抵消。参数能量不是函数能量，更不是控制
能量。

### 11.2 \(L_2\) 白化仍不能给出控制坐标

在源状态动作分布上对白化后的 KAN 函数特征做谱截断：

- 291 个原始参数方向中约 117 个属于高方差独立模式；
- 仅保留这些模式时，light/composite 可以恢复，long-link 为 0%，
  weak-actuator 仅约 30.6%；
- 降低谱阈值后需要保留约 273 个模式才能恢复完整表示。

这说明部分对控制极重要的方向在普通数据分布中的 \(L_2\) 能量非常低。
普通 PCA、白化或预测方差排序会系统性删除这些方向。

### 11.3 目标相关控制切排序有效，但不能部署

如果先用目标转移拟合完整变化，再按每个变化模式经过
\(DK\) 后的目标相关增益影响排序：

- 128 模式时，weak-actuator 从参数幅值排序的约 30.6% 提高到约 94.2%；
- 192 模式时，从约 87.1% 提高到约 97.3%；
- 其他三个环境在 128 模式时达到 100%。

这证明控制切度量比参数幅值更相关，但该排序使用了目标变化系数，只能
作为诊断，不能作为源阶段固定字典。

### 11.4 源任务固定控制模式失败

只在源环境计算每个模式的单位策略敏感性并固定排序后，性能高度非单调；
对每个固定子空间重新做受限最小二乘也不能稳定恢复目标控制。这说明单独
模式的敏感性不足以描述多个认知参数的联合变化。

进一步构造完整策略 Jacobian

\[
J
=
\frac{\partial\operatorname{vec}K}{\partial\theta}
\in\mathbb R^{8\times546}.
\]

其奇异值为约

\[
(215.4,82.98,48.00,30.80,5.56,3.36,1.31,0.57),
\]

因此一阶策略变化确实只有 8 维。右奇异向量给出了严格的源策略切字典。
但是使用全局目标预测残差拟合这 1、2、4、8 个系数后，四个环境仍未
恢复控制。全局 MSE 被远离任务流形的动力学误差支配，没有恢复目标附近
正确的 Jacobian。

### 11.5 Gate B 结论

Gate B 对**当前 ProtoKAN 表示**判定失败：

\[
\boxed{
\text{完整函数可表达}
\;\not\Rightarrow\;
\text{存在紧凑稳定的控制变化坐标}
}.
\]

当前 basis 是为源环境全局加速度 MSE 训练的，事后白化、稀疏化或策略
排序不能把它变成控制一致字典。

下一版认知网络必须在训练阶段就学习控制 Sobolev/jet 表示，例如联合约束

\[
\mathcal L_{\rm cognition}
=
\mathcal L_{\rm value}
+
\lambda_x\mathcal L_{\partial f/\partial x}
+
\lambda_u\mathcal L_{\partial f/\partial u}
+
\lambda_{\rm orth}\mathcal L_{\rm canonical}.
\]

字典正交性也应定义在任务占用分布与策略敏感性诱导的度量中，而不是普通
输入分布的 \(L_2\) 度量。Gate B 重新通过前，不应进入少样本在线组合。

## 12. 控制 jet 接口与 Gate B/C/D 结果

前述实验说明不应强迫目标物理变化在 KAN 权重空间中低维。更规范、且
不依赖未知物理参数数量的接口是动力学 jet：

\[
\mathcal J f(x,u)
=
\left(
f(x,u),
\frac{\partial f}{\partial x},
\frac{\partial f}{\partial u}
\right).
\]

它的维度仅由状态和动作维度决定。认知网络负责学习函数并提供可微 jet，
决策网络通过策略切算子运输 jet，而不是解释原始 KAN 权重。

### 12.1 目标局部转移估计 jet

仅使用目标附近的 \((s,a,s')\) 转移，通过局部回归估计 \(A,B\)，然后沿
估计 jet 路径做 2、4、8 段策略切运输。8192 条无噪声局部转移下：

| 环境 | 源增益 | 一次运输 | 2 段 | 4 段 | 8 段 | Oracle |
|---|---:|---:|---:|---:|---:|---:|
| long-link | 100% | 77.3% | 100% | 100% | 100% | 100% |
| light-low-gravity | 0% | 0% | 0% | 0% | 100% | 100% |
| feasible-composite | 0% | 0% | 6.9% | 100% | 100% | 100% |
| weak-actuator | 65.1% | 80.6% | 89.7% | 95.8% | 98.3% | 98.9% |

学习器没有读取目标质量、重力或执行器参数。估计的 \(A,B\) 相对误差约
为 0.01%–0.09%，几乎复现 Oracle Gate A。因此控制 jet 是可行的
认知—决策函数接口。

### 12.2 固定局部分布少样本门槛

在同一无噪声、可局部复位的数据分布下，将样本数降到 16、32、64、128，
8 段运输结果几乎不变。16 条转移时 \(A,B\) 相对误差仍约
0.02%–0.38%，四个环境已经达到与大样本相同的控制水平。

这说明局部 jet 本身样本维度很低且可辨识。但该实验属于理想 Gate C，
因为允许在目标平衡点附近主动采样。

### 12.3 真实闭环数据门槛

将 8 段策略运输接入现有策略生成的在线数据：

- direct 闭环算子在 long/light/composite 上保持原有恢复；
- weak-actuator 在 128 条反馈后仅约 72.0%，低于直接 Riccati 约 87%；
- randomized-moment 加连续运输仍在 16–32 条时归零，128 条约 80.6%。

因此连续策略运输不能修复有偏认知路径：

\[
\text{平滑地沿错误 jet 运输}
\neq
\text{到达正确策略}.
\]

当前剩余瓶颈被精确定位为：

> 如何让认知网络从当前轨迹上的闭环转移，可靠更新当前状态及任务流形上
> 的控制 jet，而不是要求系统先到达目标点附近才能辨识目标 jet。

这要求下一版从平衡点 LQR 升级为状态相关策略切运输。对每个状态，最优
动作满足

\[
\nabla_a Q_f(s,a^\star)=0.
\]

由隐函数定理，动力学变化引起的局部动作修正为

\[
\delta a^\star(s)
=
-
\left[\nabla_{aa}^2Q_f(s,a^\star)\right]^{-1}
D_f\!\left[\nabla_a Q_f(s,a^\star)\right][\delta f].
\]

在线认知在哪些状态收到转移，就更新这些状态附近的 KAN 局部 jet；该 jet
通过上式直接调制 Actor 中间层或动作输出。这样不再要求先用失败的源策略
到达目标平衡点，也不再把全局动力学压成单个原点 \(A,B\)。

## 13. 状态相关 Bellman 策略切门槛

为从平衡点 LQR 推进到非线性 Actor，我们使用源 PPO checkpoint 中的
Actor 与状态价值网络，构造

\[
Q^{(1)}_f(s,a)
=
r(f(s,a))+\gamma V_0(f(s,a)),
\]

并在源 Actor 动作处通过动作 Hessian 做 Newton 修正。比较：

1. 原始源 Actor；
2. 使用源动力学计算的 Bellman 修正；
3. 使用 Oracle 目标动力学计算的 Bellman 修正。

一步结果失败：

- long-link 的源 Actor 约 99.2%，源认知修正降到约 14.8%，目标认知
  修正约 9.4%；
- weak-actuator 的源 Actor 约 43.8%，源认知修正约 53.9%，目标认知
  修正约 2.3%；
- light/composite 基本没有恢复。

这首先说明 PPO 的 \(V_0(s)\) 不是可靠的动作价值曲率。它只在源策略状态
占用分布上拟合回报；通过目标动力学把动作映射到新状态后，
\(\nabla_aQ\) 和 \(\nabla_{aa}^2Q\) 会放大 critic 外推误差。

随后使用 5 步可微闭环回报，首动作可优化，后续由源 Actor 执行并在终点
连接 \(V_0\)。weak-actuator 小规模门槛中：

- 原始 Actor 约 37.5%；
- 源动力学多步修正约 68.8%；
- 目标动力学多步修正仍为 0%。

多步曲率能够改善源闭环，但固定源价值函数不能解释目标动力学下的未来。
同时逐状态、逐时间步求完整动作 Hessian 计算开销过高，不适合作为最终
部署机制。

### 13.1 理论修正

价值函数本身也是动力学 \(f\) 的隐函数。正确运输不能冻结 \(V_0\) 只更新
\(\pi\)，而必须联合求解 Bellman—policy 最优性系统。令

\[
\mathcal F(V,\pi,f)
=
\begin{bmatrix}
V-\mathcal T_f^\pi V\\
\nabla_aQ_f^V(s,\pi(s))
\end{bmatrix}
=0.
\]

由隐函数定理，

\[
\boxed{
\begin{bmatrix}
\delta V\\
\delta\pi
\end{bmatrix}
=
-
\left[
\frac{\partial\mathcal F}{\partial(V,\pi)}
\right]^{-1}
\frac{\partial\mathcal F}{\partial f}[\delta f]
}.
\]

LQR Gate A 之所以成功，是因为 Riccati 方程实际上已经联合更新了 value
曲率 \(P\) 和策略增益 \(K\)。状态相关实验失败，是因为我们只保留了
\(\delta\pi\) 的局部动作 Newton 形式，却错误地令 \(\delta V=0\)。

因此下一版神经结构的正确对象不是单独的 Actor 调制器，而是：

\[
\text{KAN 认知 jet}
\longrightarrow
\text{联合 Value--Policy 切运输}
\longrightarrow
\text{Actor 与 Critic 同步调制}.
\]

实现上应使用 Jacobian-vector product 和线性求解近似联合隐式系统，再把
求得的切更新离线蒸馏为轻量调制网络；不应在每个环境步实时构造完整
Hessian。

## 14. 目标 Critic 适应与信赖域诊断

为了检验第 13 节的判断，我们固定源 Actor，在 weak-actuator 目标动力学
中采集 64,000 个真实闭环状态—回报样本，先适应 Critic，再使用目标
动力学计算动作 Newton 修正。

第一次直接拟合原始回报时，MSE 只从 23,068 降到 23,013，适应后的
Critic 几乎没有改变动作。这说明大尺度、长时域回报使普通均方误差优化
在有限更新内失效。将回报标准化拟合、并在前向时严格恢复原始尺度后，
标准化 MSE 从 1.008 降到 0.030，函数值拟合已经充分收敛；但动作修正的
平均二范数从约 0.005 激增至 27.0，成功率反而从源 Actor 的 40.6% 降到
1.6%。

这给出一个比“Critic 没训练好”更精确的根因：

\[
\boxed{
\|V_\phi-V^\pi\|_{L_2(\rho^\pi)}\ \text{很小}
\;\not\Rightarrow\;
\|\nabla_sV_\phi-\nabla_sV^\pi\|_{L_2(\rho^\pi)}\ \text{很小}
}.
\]

策略修正使用
\(\nabla_a V(f(s,a))=\nabla_sV(f(s,a))\,\partial f/\partial a\)，
因此它依赖 Critic 的状态导数，而普通 value regression 只约束采样点上的
函数值。神经网络可以具有很小的点值误差，同时在点间产生错误且过大的
斜率。

进一步将动作修正限制在半径 0.25、1.0 和 4.0 的信赖域内：

| 最大修正半径 | 源 Actor | 目标认知 + 源 Critic | 目标认知 + 适应 Critic |
|---:|---:|---:|---:|
| 0.25 | 40.6% | 42.2% | 43.8% |
| 1.0 | 40.6% | 23.4% | 20.3% |
| 4.0 | 40.6% | 3.1% | 6.3% |

极小信赖域只能避免灾难性动作，不能产生显著恢复；步长稍大后错误导数
立即主导策略。因此问题不是选择一个更好的 Newton 步长，而是当前
Critic 没有辨识出控制所需的导数几何。

下一门槛应从点值 Critic 改为**控制一致的 Sobolev Critic**：使用真实
动作扰动或可微认知模型生成局部反事实，直接监督 advantage 差值和
动作导数；同时用信赖域 Actor--Critic 反复交替，而不是一次性贪心
Newton 修正。必须先验证学得的动作排序/导数与真实有限时域回报一致，
再接入完整在线恢复实验。

## 15. 效果空间联合 Actor--Critic

进一步的四路归因表明，认知差异仅作为 Actor 输入特征时没有可测收益；
恢复几乎全部来自认知模型对基础动作的一步控制等价 pullback。为消除
动作空间残差绕过认知的路径，我们把在线 Actor 的输出改为标准效果残差
\(\Delta v\)，并令最终动作必须满足

\[
a_t
=
\arg\min_a
\|\hat d_t(s)+\hat G_t(s)a-(v_0(s)+\Delta v_\omega(s))\|^2
+\lambda\|a-\pi_0(s)\|^2.
\]

三种子固定协议中，weak-actuator 的效果空间最终成功率为
\(99.5\%\pm0.7\%\)，动作空间为 \(93.2\%\pm2.7\%\)；feasible-composite
的效果空间最终成功率为 \(100.0\%\pm0.0\%\)，动作空间仅
\(2.6\%\pm3.7\%\)。冻结认知的单种子严格归因始终只有 50%。

这说明有效方向不是进一步修补单步 Newton Critic，而是保留 Actor--Critic
的长时域信用分配，同时把策略更新约束在认知模型定义的控制等价效果
坐标中。完整定义、无泄漏边界与后续验证要求见
`docs/CONTROL_EQUIVALENT_EFFECT_SPACE_ACTOR_CN.md`。

## 16. 高维 Hopper 暴露出的源认知基底问题

效果空间结构迁移到 Hopper 后，两时间尺度共享认知训练器能够运行，但
没有恢复健康步态。真实反事实诊断发现当前 ProtoKAN pullback 只在
20.3% 的状态改善效果，平均把误差放大约 1.98 倍。预测的环境变化与真实
变化平均余弦仅 0.135。

增加目标样本、阻尼、Critic 度量、策略中心化、Stein 矩估计和低秩目标
变换均未通过门槛。这说明目标适应器不是首要瓶颈；普通源预测预训练没有
产生可靠的控制 Jacobian 基底。

因此源认知必须升级为黑箱控制 Sobolev 训练：在唯一源环境中用同状态
成对动作扰动构造有限差分 \(G_0(s)\) 标签，显式监督 KAN 的动作导数。
目标阶段只在该可靠基底上估计低维控制变换。该路线不使用目标物理参数，
也不依赖已知公式，但承认“预测准确自动蕴含可控制知识”在高维系统中不
成立。

源 Sobolev 实验已将验证 Jacobian 余弦提高到 0.9946、相对误差降到
9.0%，证明控制基底可以从单一源环境黑箱学习。但源基线效果 RMSE 仍为
0.360，导致目标在线阶段无法区分真实物理变化与源模型逼近误差。
source→source 恒等性门槛因此仍失败。

下一根因是模型偏差可辨识性，而非 Jacobian：必须为源基线模型学习并
校准不确定性，只允许超过源置信域的残差写入目标上下文。恒等环境的假
更新率必须先接近零。
