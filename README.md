# KAN-RF：认知 ProtoKAN 与决策网络的物理泛化控制

KAN-RF 研究一个明确的问题：能否只在一种物理条件下训练一个小型控制系统，在质量、阻尼、驱动力或惯量发生变化后，通过持续更新认知网络，把新学到的物理规律传入决策网络并快速恢复控制能力？

当前分支 `feature/cognitive-embedded-decision` 是这一思想的实验验证版本。它不是已经完成的强泛化系统：当前最好结构在源 Acrobot 环境达到 **55/64（85.94%）**，在中等参数变化下可以恢复到 **30/32（93.75%）**，但在严重参数变化下在线学习后只有 **0/32–1/32**。仓库保留这些失败结果，因为它们定位了认知—决策接口和长程策略学习中的核心问题。

## 1. 研究假设

系统由两个职责分离、前向耦合的网络组成：

1. 认知网络只学习环境状态转移；
2. 决策网络只根据真实任务回报学习动作；
3. 两套损失不混合；
4. 认知网络必须参与决策前向传播；
5. 物理条件改变后，认知网络利用新转移持续学习，决策网络也继续学习；
6. 理想情况下，认知参数的变化应当成为决策适应的主要物理信息来源。

认知模型为

$$
\hat s_{t+1}=F_\theta(s_t,a_t),
$$

决策策略为

$$
a_t=\pi_\phi(s_t,g;F_\theta),
$$

其中 $\theta$ 负责动力学预测，$\phi$ 负责完成目标 $g$。

## 2. 当前最好结构：PSD 时序因果策略

当前主体实现位于 `scripts/stage73_psd_causal_preconditioner_actor.py`。它由五部分组成。

### 2.1 认知 ProtoKAN

`SimpleCognitiveKAN` 使用完整 ProtoKAN 学习

$$
(s_t,a_t)\in\mathbb R^7\longmapsto\hat s_{t+1}\in\mathbb R^6.
$$

当前 Acrobot 状态包含两组角度的正余弦和两维角速度，动作是一维连续力矩。认知网络的默认隐藏宽度为 32，每条 ProtoKAN 边使用 8 个原型。

源环境预训练只使用状态转移预测，不使用动作教师或任务奖励。在线更新使用不跨 episode reset 的连续真实序列，并以自由滚动的八步预测误差训练。

### 2.2 动作序列提议器

决策 proposal MLP 接收当前状态和目标状态，输出长度 $H=8$ 的动作序列：

$$
\mathbf u=P_\phi(s_t,g)=[u_0,\ldots,u_{H-1}].
$$

这条序列是待修正的动作计划，并非最终动作。

### 2.3 ProtoKAN 原生时序因果路由

候选动作经过认知网络自由滚动：

$$
\hat s_{t+h+1}=F_\theta(\hat s_{t+h},\tanh u_h).
$$

代码显式追踪 ProtoKAN 函数边的值、一阶导数、有限正负响应和曲率，形成每一步的状态传播与动作传播关系。随后从长程可达性评分反向传播消息：

$$
\lambda_h=A_h^\top\lambda_{h+1},\qquad
r_h=B_h^\top\lambda_{h+1}.
$$

$r_h$ 表示第 $h$ 个动作对未来可达性目标的有符号影响。Stage62–64 验证了显式边路由与直接 autograd 多步梯度的一致性。

### 2.4 正半定时序预条件器

策略不让自由 MLP 任意解释因果方向，而是构造正半定矩阵

$$
K_\phi=L_\phi L_\phi^\top+\operatorname{diag}(d_\phi)+\epsilon I\succeq0.
$$

$K_\phi$ 根据路由绝对值、预测状态、时间评分、时间权重和目标生成，并对八个时间步之间的作用进行低秩耦合。动作修正为

$$
\Delta\mathbf u=K_\phi\mathbf r,
\qquad
\tilde{\mathbf u}=\mathbf u+\eta\Delta\mathbf u.
$$

它满足

$$
\mathbf r^\top K_\phi\mathbf r\ge 0,
$$

因此学习型算子不能把完整因果路由解释成整体相反的方向。系统执行

$$
a_t=\tanh(\tilde u_0),
$$

然后在下一个真实时间步重新规划。

### 2.5 PPO 决策学习

策略是 tanh-squashed Gaussian actor，critic 是读取 $(s_t,g)$ 的 MLP。决策部分通过真实环境奖励和 PPO 更新；认知网络与因果路由器在 PPO 更新期间冻结。

当前任务奖励由平滑高度项、终止成功奖励和小幅动作惩罚组成。成功条件是 Acrobot 末端高度达到阈值 1.0，评估最多运行 500 步。

## 3. 训练和在线适应流程

### 3.1 源环境训练

当前源物理因子为

$$
(g,\text{damping},\text{actuation},\text{inertia})
=(7.35,0.0,0.8,0.8).
$$

训练顺序为：

1. 用源环境转移预训练认知 ProtoKAN；
2. 训练 ProtoKAN 非线性边路由器；
3. 冻结认知网络和路由器；
4. 用 PPO 训练 proposal、PSD 预条件器、策略标准差和 critic；
5. 保存完整 actor 与 critic checkpoint。

认知损失和决策损失始终分开：

$$
L_{\text{cog}}
=\frac1H\sum_{h=1}^{H}
\ell(\hat s_{t+h},s_{t+h}),
$$

$$
L_{\text{decision}}=L_{\text{PPO}}(r_t,V_\omega).
$$

### 3.2 物理参数变化后的在线阶段

每轮执行：

1. 当前整体策略在目标环境收集真实转移；
2. 冻结认知网络，用 PPO 更新决策参数和 critic；
3. 冻结决策参数，用连续真实转移的八步预测损失更新认知网络；
4. 下一轮决策立即使用更新后的认知参数。

这是当前对“认知持续学习驱动决策适应”的直接实现。Stage76 的动作信任域和 Stage77 的真实回报门控是故障诊断，不属于最终核心方法。

## 4. 当前实验结果

### 4.1 源环境控制

相同低预算配置为 32 个并行环境、每轮 128 步、60 次策略更新。

| 方法 | 独立终测成功率 |
|---|---:|
| 同预算普通 MLP PPO | 48/64 = 75.00% |
| Stage66 自由时序解码 | 46/64 = 71.88% |
| Stage70 固定标量因果修正 | 15/64 = 23.44% |
| **Stage73 PSD 时序因果策略** | **55/64 = 85.94%** |

Stage73 是目前源环境表现最好的结构，但仍未达到简单任务期望的 100%。ProtoKAN 和普通 MLP 的差距也尚未经过足够随机种子验证。

### 4.2 中等参数变化

中等目标为

$$
(9.8,0.04,1.1,0.9).
$$

在固定 32 个评估初态上：

| 累计目标环境策略转移 | 成功率 |
|---:|---:|
| 0 | 30/32 = 93.75% |
| 4,096 | 24/32 = 75.00% |
| 12,288 | 27/32 = 84.38% |
| 36,864 | 29/32 = 90.63% |
| 61,440 | 30/32 = 93.75% |

这一结果只说明系统在中等目标上经历扰动后可以恢复。该目标零样本本身已经较容易，不能单独证明认知持续学习带来了强泛化。

### 4.3 严重参数变化

严重目标为

$$
(13.475,0.06,0.90,1.10).
$$

Stage73 在 64 个初态上的零样本成功率为 3/64（4.69%）。在 32 个固定在线评估初态上，完整联合更新从 3/32 降到 0/32。随机在线轨迹中曾累计出现约 70 次成功，说明任务可达且探索并未完全失败，但确定性均值策略没有吸收这些成功经验。

后续诊断如下：

| 诊断 | 目标 | 结果 |
|---|---|---:|
| Stage75：仅更新决策、重置 critic，6 轮 | 排除源 critic 和认知漂移 | 4/32 = 12.50% |
| Stage75：联合更新、重置 critic，6 轮 | 检查直接认知更新 | 0/32 |
| Stage76：认知延迟到第 7 轮，并限制策略动作 RMSE 约 0.05 | 检查小步认知更新 | 0/32 |
| Stage76：仅决策更新，15 轮 | 检查决策自身恢复 | 1/32 = 3.13% |
| Stage77：真实长程回报门控认知更新，15 轮 | 诊断是否存在可安全选择的认知更新 | 0/32 |

Stage77 的门控还额外消耗 144,000 条诊断环境转移，因此它不能作为样本高效的最终算法。

## 5. 已确认的问题

### 5.1 预测梯度与决策收益不对齐

降低 $L_{\text{cog}}$ 并不保证提高任务回报：

$$
-\nabla_\theta L_{\text{cog}}
\not\Rightarrow
\nabla_\theta J_{\text{task}}.
$$

认知预测可以改善，但其参数变化仍会改变预测轨迹、函数边导数和因果路由，使原决策参数失去接口兼容性。

### 5.2 “参与计算”不等于“被决策依赖”

当前结构为

$$
\tilde{\mathbf u}=P_\phi(s,g)+\eta K_\phi\mathbf r_\theta.
$$

ProtoKAN 每次都参与前向计算，但 proposal 存在直接动作路径。如果因果修正相对较小，策略仍可能主要依赖 proposal。这是一条软绕过路径，也是当前架构与理想深度融合之间的差距。

### 5.3 单步状态准确不保证多步控制导数准确

控制需要的不只是 $F_\theta$ 的输出，还需要多步复合中的动作导数。八步滚动包含模型误差累积和 Jacobian 连乘；较小的单步状态误差仍可能对应错误或不稳定的长程动作方向。

### 5.4 在线决策学习没有固化成功经验

严重目标中，随机策略轨迹能够多次成功，但策略标准差长期约为 0.86，最终确定性策略仍失败。当前 PPO 存在长程信用分配、critic 稳定性以及从探索到策略均值固化不足的问题。因此失败不只来自认知网络。

### 5.5 当前结构是研究原型，不是最终论文方法

因果路由、PSD 算子、信任域和回报门控分别回答了不同诊断问题，但继续叠加模块会偏离最初的两网络目标。下一版应围绕一个更明确的参数运输或函数保持机制重新收敛架构，而不是继续增加补丁。

## 6. 当前结论与论文状态

已经得到支持的结论：

1. ProtoKAN 函数边可以构造数值准确的多步因果图；
2. 长程决策需要保留因果序列的时间顺序；
3. 正半定时序算子比自由解码和固定标量修正更好地平衡容量与方向约束；
4. 认知网络的预测更新会造成决策接口漂移；
5. 严重参数变化下，当前在线认知—决策系统尚未实现稳定恢复。

因此，PSD 时序因果算子具有方法创新潜力，但当前证据不足以宣称强物理泛化，也还不足以构成完整的 AAAI 实验结果。至少还需要：

- 严重参数变化下的可靠恢复；
- 多随机种子；
- 多种控制环境；
- 认知冻结、直接更新、运输更新和普通 MLP 的等预算消融；
- 在线样本效率与计算开销报告。

## 7. 关键文件

| 文件 | 作用 |
|---|---|
| `scripts/stage73_psd_causal_preconditioner_actor.py` | 当前最好源策略的完整训练入口 |
| `scripts/stage74_joint_online_psd_causal_transfer.py` | 认知与决策损失分离的在线联合适应 |
| `scripts/stage75_severe_transfer_attribution.py` | critic、认知更新和决策更新的失败归因 |
| `scripts/stage76_delayed_trust_cognition.py` | 延迟与策略动作信任域诊断 |
| `scripts/stage77_return_gated_cognition.py` | 真实长程回报门控诊断 |
| `kanrf/protokan_causal_router.py` | ProtoKAN 函数边追踪和非线性路由 |
| `kanrf/protokan_temporal_route.py` | 八步滚动与时间反向因果传播 |
| `docs/stage65_74_temporal_causal_policy_report.md` | Stage65–74 的完整推导和结果 |
| `results/stage73_psd_causal_preconditioner_actor_seed0_60iter.json` | 当前最好源环境结果 |
| `results/stage74_joint_online_psd_causal_transfer_severe_seed0.json` | 严重迁移联合更新结果 |
| `results/stage75_severe_transfer_attribution_seed0.json` | 失败归因结果 |
| `results/stage76_delayed_trust_cognition_seed0.json` | 信任域诊断结果 |
| `results/stage77_return_gated_cognition_seed0.json` | 回报门控诊断结果 |

历史实验保留在 `scripts/stage27_*.py` 到 `scripts/stage72_*.py`、相应 `docs/` 报告和 JSON 结果中。它们记录了参数直接迁移、强制耦合、因果图、公式算子、DeepONet、Oracle 动力学和时序路由等被验证或否定的路径。

## 8. 环境与复现

项目要求 Python 3.10 或更高版本，依赖 PyTorch、NumPy 和 Gymnasium。本地实验统一在 conda 环境 `dl_env` 中运行：

```powershell
conda activate dl_env
cd C:\Users\32510\Desktop\RF\KAN-RF
pip install -e .
```

训练 Stage73 源策略：

```powershell
python -m scripts.stage73_psd_causal_preconditioner_actor `
  --checkpoint-out results/stage73_source_seed0_checkpoint.pt `
  --json-out results/stage73_psd_causal_preconditioner_actor_seed0_60iter.json
```

模型权重由 `.gitignore` 排除，需要先运行上面的源训练。随后执行严重目标联合在线适应：

```powershell
python -m scripts.stage74_joint_online_psd_causal_transfer `
  --checkpoint results/stage73_source_seed0_checkpoint.pt `
  --target-factor 13.475 0.06 0.90 1.10 `
  --json-out results/stage74_joint_online_psd_causal_transfer_severe_seed0.json
```

运行最近的失败归因与安全更新诊断：

```powershell
python -m scripts.stage75_severe_transfer_attribution `
  --json-out results/stage75_severe_transfer_attribution_seed0.json

python -m scripts.stage76_delayed_trust_cognition `
  --json-out results/stage76_delayed_trust_cognition_seed0.json

python -m scripts.stage77_return_gated_cognition `
  --json-out results/stage77_return_gated_cognition_seed0.json
```

这些正式实验在 CPU 上运行时间较长。JSON 中保存了完整超参数、训练历史、交互预算和评估结果。

## 9. 下一研究问题

当前最需要解决的不是继续增加门控，而是设计一个认知到决策的兼容运输机制。设认知参数发生预测更新 $\Delta\theta$，下一版候选方向是同步求解决策参数补偿：

$$
\Delta\phi^*
=\arg\min_{\Delta\phi}
\left\|J_\theta\Delta\theta+J_\phi\Delta\phi\right\|^2
+\lambda\|\Delta\phi\|^2.
$$

目标是在认知网络学习新物理规律时保持策略函数连续，再让决策网络利用新认知继续学习。它比简单缩小、接受或拒绝认知更新更接近项目最初的“认知参数被决策网络自然承接”目标。

---

本仓库当前是研究代码与实验记录，不是稳定控制库。引用结果时请同时报告随机种子、物理因子、评估初态、交互预算和是否使用额外诊断轨迹。