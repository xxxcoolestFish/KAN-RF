# KAN-RF 项目协作交接文档

更新日期：2026-07-25（当日第二次更新，含协作首日全部进展）

交接分支：`collab/handoff-latest-20260725`

基线提交：`47620a4`；当日新增提交 `48d6a6f`（管线泛化到 Walker2d/HalfCheetah）

目标投稿：AAAI 2027，当前仍处于方法与主实验收敛阶段

---

## 0. 一页摘要

这个项目研究的不是普通的“换环境后继续训练一个强化学习策略”，而是一个更具体的问题：

> 能否只在一种源物理条件下训练任务策略与世界模型，在部署后的物理规律发生未知变化时，仅依靠新的状态转移快速更新认知，并把更新后的物理知识转化为策略真正可用的动作修正，从而以较少真实交互恢复闭环任务性能？

我们把世界模型称为“认知网络”，把 Actor/Critic 称为“决策网络”。两者的损失分开：

- 认知网络只学习状态转移；
- 决策网络学习任务；
- 认知网络更新后的物理规律必须进入动作生成路径，不能只是一个可能被 Actor 忽略的上下文向量。

当前最有效的高维 Hopper 方案是：

\[
\text{冻结源 Actor}
\rightarrow
\text{源控制仿射认知}
\rightarrow
\text{在线目标 ProtoKAN}
\rightarrow
\text{控制等价动作反解}.
\]

在隐藏的 `combo_medium` 变化下，源 Actor 的回报约为 \(341.2\)，认知模型用 2048 条目标转移更新后达到 \(407.0\pm1.9\)，三个种子的归一化恢复 AUC 为 \(0.767\pm0.043\)。该实验不使用目标物理参数、不使用变化类别，也不使用目标奖励更新 Actor。

但是，项目还没有达到可直接投稿的程度：

1. 多物理变化中存在负迁移和后期漂移（其机制已确诊，见 §0.5.3 与 §0.5.4）；
2. Hopper 上尚缺 KAN 与参数匹配 MLP 的严格消融；
3. 决策网络持续学习尚未与当前认知运输正结果完整结合；
4. 当前 LaTeX 草稿仍讲较早的 CartPole/Two-Link PCET 故事，与最新 Hopper 主线不同步；
5. 第二高维环境首次尝试（Walker2d）系统性失败，家族泛化性仍未证明（§0.5.3）；
6. 缺少完整计算成本统计。

接手后最重要的不是继续堆新结构，而是先完成公平消融、稳定性修复和论文叙事同步。

---

## 0.5 协作首日进展（2026-07-25 晚）

本节记录交接后第一天发生的四件大事：论文贡献层级正式对齐、管线泛化工程、Walker2d 第二家族尝试的完整历程与最终裁决、以及由此产生的修复转向。**本节信息优先于后文任何冲突表述。**

### 0.5.1 论文贡献层级（写作与实验设计的最高依据）

三层结构已正式确定：

```text
问题：  actionability gap —— 世界模型学会了新动力学，但物理知识未必能被策略使用
方法：  控制等价认知接口 —— 持续更新的世界模型位于动作生成必经路径（first-order claim）
结果：  冻结源 Actor、无目标奖励，数百条转移内出现显著恢复并随预算提升（结果卖点）
```

- first-order claim（标题、方法图中央、审稿人记忆点）是**接口本身**：把在线更新的预测动力学显式转化为决策动作修正；
- actionability gap 只作切入点（"预测准≠决策有效"单独可能被视为已知直觉）；
- 样本效率只作结果卖点，**严禁**写"数百条转移即可恢复"。严谨表述：
  > 在数百条无奖励目标转移内出现显著恢复，并随认知反馈预算继续提升。
  （实际数字：256 条 +26.6，512 条 +47.4，2048 条达 407.0）
- 贡献列表写法：Formalize(gap) / Method(接口) / Evidence(恢复)；
- 推论：结构消融的核心任务是守住"接口"主张（vs 无运输、vs 可绕过上下文接口、vs 逆动力学迁移），KAN-vs-MLP 消融服务于"算子结构是否重要"，不是证明 KAN 优越。

**协议澄清（重要）**：所谓"新环境"一律指**同一任务家族内物理参数变化**（质量/摩擦/执行器缩放），绝不是跨任务迁移（如 Hopper 学成后直接部署到 Walker2d——状态/动作空间与任务语义不同，无意义）。第二家族验证的定义是：在新家族的源物理中从零重建整条源管线，再施加该家族自己的参数变化，重复同一协议。

### 0.5.2 工程：管线泛化到多任务家族（提交 `48d6a6f`）

- `scripts/prescreen_hopper_physics_shifts.py` 新增 `ENVS` 注册表：`hopper`/`walker2d`/`halfcheetah`（gym id + 质量作用刚体名），`make_shifted_env(shift, seed, env="hopper")` 向后兼容；
- 7 个实验脚本接入 `--env` 并贯穿；3 处 `action_dim` 硬编码改为从检查点/策略模型推导，旧检查点经 `payload.get("action_dim", 3)` 回退；
- `CognitiveResidualHopper` 观测/动作空间维度动态化（Walker2d: 40 维特征、6 维动作）；
- 新增 `tests/test_env_registry.py`（注册表、Walker2d 字典维度 222/1554、shift 缩放生效、默认 hopper），pytest 26 项全过；
- 后续又修复孪生构造遗漏的 `action_dim` 透传（`train_hopper_source_affine_twin.py` 从模板检查点读取）并补回归测试，pytest 27/27。

运行注意事项（本机实测）：

- 所有脚本必须以 `python -m scripts.xxx` 形式从仓库根运行（`scripts/` 无 `__init__.py`，依赖 cwd 在 sys.path）；
- `conda run -n dl_env` 在本机会触发 conda 内部错误，**直接调用** `C:\Users\32510\miniconda3\envs\dl_env\python.exe`；
- 后台/新 shell 命令不继承工作目录，一律显式 `cd` 到仓库根。

### 0.5.3 Walker2d 第二家族尝试：全过程与最终裁决

按批准的实验设计（判据预注册：健康完成率→100% + 恢复比 + Oracle 参照；两阶段先认知后决策）执行，结果 **Walker2d 家族系统性失败**，章节已关闭。

**源管线（顺利部分）**：

- 源 PPO 需 3M 步才收敛（1M: 767 → 2M: 2027 → 3M: **5376.8，健康完成率 100%**），Walker2d 明显比 Hopper 慢；
- 认知组件经两轮修复：pair_modes 1→2（特征 222→630）、探针 1024→4096、孪生 3000→12000 步且 Jacobian 权重 0.2→1.0，最终 Sobolev 余弦 0.914、孪生 0.832（仍低于 Hopper 的 0.99/0.99）。

**闭环结果（失败部分）**：四个变化快速筛查全灭：

| 变化 | 0 条 | 512 条 | 2048 条 |
|---|---:|---:|---:|
| combo_medium | 397.4 | 354.8 | **3.3** |
| actuator_080 | 958.3 | 187.2 | 314.4 |
| payload_125 | 4729.3 | 955.7 | 1693.5 |
| friction_070 | 2216.3 | **2894.1** | 1732.0 |

对照 Hopper 同期 7 变化 6 正增益。**根因诊断（三轮纪律收敛）**：

1. **运输目标结构性不可达**：配对真源模拟器 Oracle 对照下，修正仍放大一步误差 1.6×、仅 46.9% 状态改善（Hopper 当年 69.5%/−11.7%）。Walker2d 双脚接触 + 6 执行器强耦合，源效果 \(e_0(s)\) 在变化后物理中复现不出来——接口的成立条件（目标物理中源效果可达）在此不满足；
2. **漂移幅值失控**：friction_070 逐预算诊断显示 \(\|W\|\) 从 0.89 增长到 2.17，而留出预测 RMSE 全程平稳（0.709±0.004）——不是递归估计漂移，而是**预测目标的固定 ridge 与运输目标失配**：有用信号早期饱和，有害幅值持续累积，闭式反解将其直接转为有害修正（p95 修正幅值达 1.38）。这同时解释了 Hopper friction_070 的后期漂移，是两家族共有的核心问题；
3. **饱和下激励消失**：Walker2d 动作饱和度为 Hopper 两倍（30%+ vs 15% 在 \(|a|>0.99\)），warmup 激励幅度 \(\min(0.3, 1-|a|)\) 在饱和方向≈0，T_t 在执行器方向辨识薄弱（余弦 0.71）。

**被证据排除的假设**：探针标签噪声（eps 0.02/0.05/0.1 一致性 0.996–0.999）、数据量（4 倍探针仅 +0.04 余弦）、模型容量/优化（单步余弦 0.74→0.91 反而闭环更糟——再次印证 §5.2）。

**抢救尝试与复验纪律（重要教训）**：drift_ridge=1000 曾在 2 回合评估下给出 512 条 5148.8 的"近完全恢复"，但 5 回合复验拆穿——零样本基线随回合数剧变（2216↔3752），实际仅 512 处 +24.6% 且全非单调；且 ridge=1000 与 Hopper 不兼容（340→340→362，杀死恢复），统一超参不可行。**一切少于 5 回合的对比都不可信，此后正式实验一律 ≥5 回合并报告 std。**

**裁决**：D 路线（聚焦修复 Walker2d）失败。Walker2d 全部材料转为论文 Limitations/边界章节素材（运输目标可达性条件、漂移信赖域失控机制、评估噪声纪律——本身就是硬贡献）。

### 0.5.4 修复转向：先解决问题，再谈第二环境

经对齐，当前优先级从"铺第二环境"转为"修复已确诊的三个问题"，顺序与方案已批准：

1. **问题 1（核心）：漂移信赖域失控**。方案 (a)：信赖域约束更新 \(\min_W\|\Phi W-Y\|^2\ \mathrm{s.t.}\ \|W\|\le r\)，对偶为每预算自适应 ridge；r 由第一个非零预算（256）的拟合范数设定（无奖励）。（注：首次求解器实现已应要求撤回，具体实现方式以重新讨论为准。）
2. **问题 2：运输目标可达性（payload_125 类负迁移）**。方案：可达性残差比 \(\rho(s)=\|G_t\delta a^\*-(e_0-b_t)\|/\|e_0-b_t\|\)——闭式解的天然副产品、无奖励、与运输目标直接对齐（不同于 §5.9 失败的支撑距离置信度）。验收：payload_125 应被自动判"不运输"而保住零样本 1854.7。
3. **问题 3：饱和下激励消失**。方案：饱和方向强制向内最小幅度激励（向内永远可行，协议干净）。

**验证协议（复用现有检查点，零新源训练）**：Hopper 的 friction_070、payload_125 必须改善，**combo_medium 主结果不得回归**（保 341→407）；Walker2d 的 friction_070、combo_medium 崩溃案例复测；一律 ≥5 回合 + 多种子 + 全预算曲线。

### 0.5.5 对原计划的影响

- **HalfCheetah 第二家族：暂停**（注册表已就绪，源管线未开始），待三个修复验证后再评估是否重启；
- **原 P0（KAN vs 参数匹配 MLP 结构消融）：顺延**——`LearnedMLPDictionary` 已备好但 `load_cognition` 仍硬编码 KAN 字典，打通加载路径是届时第一步；
- **决策持续学习（Gate D）与论文重写：排在修复之后**；
- 原 §10 任务清单已按本节更新，冲突时以本节为准。

---

## 1. 论文真正想回答的问题

### 1.1 场景

考虑一族任务相同、但动力学不同的连续控制 MDP：

\[
\mathcal M_\xi=(\mathcal S,\mathcal A,P_\xi,r,\gamma).
\]

状态、动作和任务语义不变，质量、重力、摩擦、阻尼、惯量或执行器能力等物理因素改变。训练时只允许访问一个源动力学 \(\xi_0\)。部署后 \(\xi_t\) 未知、可能突然改变，算法只观察：

\[
s_t,\quad a_t,\quad r_t,\quad s_{t+1}.
\]

我们关心的主要指标不是零样本立即成功，而是：

- 变化后初始性能；
- 随目标转移数量变化的恢复曲线；
- 恢复 AUC；
- 达到某个回报阈值所需的真实反馈量；
- 最终稳定回报；
- 环境再次切回时的遗忘和召回速度。

### 1.2 为什么只训练世界模型不够

准确的世界模型回答：

> 在当前物理条件下，执行动作 \(a\) 会发生什么？

决策需要回答：

> 为了延续源策略的任务意图，现在应该执行什么动作？

两者之间缺少一个从“预测知识”到“动作改变”的接口。我们把它称为认知信息的可行动性问题。项目真正的研究核心是设计这个接口，而不是简单证明 KAN 能预测动力学。

### 1.3 当前候选论文主张

目前最合理、也最诚实的主张是：

> 持续学习的控制仿射 ProtoKAN 可以作为一种结构化的目标动力学算子；通过控制等价动作运输，它能够仅依靠目标状态转移，把更新后的物理认知转化为闭环动作修正，并在部分未见物理变化上提高有限反馈预算下的恢复速度。

当前不能声称：

- “认知网络和决策网络分离”本身是新贡献；
- KAN 天然不会遗忘；
- 任意未知物理变化都能零样本泛化；
- 当前方法已经在所有 Hopper 变化上稳定有效；
- 当前性能主要来自 KAN，而不是 Dense 源孪生；
- 当前系统已经完成完整的认知与决策联合持续学习。

---

## 2. 与已有研究的创新边界

以下思想已经有先例，不能单独作为贡献：

- 世界模型与策略分离；
- 从近期转移推断动力学上下文；
- 根据动力学上下文条件化策略；
- 逆动力学迁移和动作变换；
- 在线世界模型持续学习并用于规划；
- KAN 用于函数逼近。

与我们最接近的方向包括 CaDM、Augmented World Models、inverse-dynamics transfer、Grounded Action Transformation，以及在线世界模型规划。当前可能形成区别的组合是：

1. 只用一个源物理环境训练；
2. 目标变化类型和参数不可见；
3. 认知模型只用自监督转移更新；
4. 源任务策略可以冻结；
5. 目标 ProtoKAN 暴露显式的漂移与动作效果算子；
6. 动作由控制等价反解产生，认知算子位于动作生成必经路径；
7. 评价重点是恢复速度与有限反馈预算，而不只是最终性能；
8. 后续计划允许决策网络持续学习，但需要消融证明认知贡献。

这些点必须通过与上下文模型、逆动力学迁移和同预算在线 Actor-Critic 的直接对照来守住。

---

## 3. 当前网络与数学结构

### 3.1 源 Actor

源 Actor 在标准 Hopper 源环境中用 PPO 训练：

\[
a_0=\pi_0(s).
\]

当前正式认知恢复实验冻结 \(\pi_0\)，这样可以明确判断提升是否来自认知更新。项目长期目标并不禁止决策网络持续学习；冻结 Actor 只是现阶段的归因协议。

### 3.2 策略中心化的源认知

令动作扰动为：

\[
\delta a=a-\pi_0(s).
\]

源控制仿射 ProtoKAN 表示：

\[
\hat f_0^{\mathrm{KAN}}(s,\delta a)
=b_0^{\mathrm{KAN}}(s)
+G_0^{\mathrm{KAN}}(s)\delta a.
\]

这里：

- \(b_0(s)\) 是源策略动作附近的名义状态增量；
- \(G_0(s)\) 是动作扰动对状态增量的局部影响；
- 固定 KAN 字典 \(\Phi(s)\) 表达状态相关结构；
- 动作 Jacobian 通过对称动作探针和 Sobolev 监督学习。

### 3.3 源反事实数字孪生

当前最强源反事实模块不是纯 KAN，而是 Dense 控制仿射网络：

\[
\hat f_0^{\mathrm{twin}}(s,\delta a)
=b_0^{\mathrm{twin}}(s)
+G_0^{\mathrm{twin}}(s)\delta a.
\]

它只使用源环境数据，包括源策略轨迹、动作扰动、状态云增强和动作 Jacobian 监督。它的任务是在目标状态上估计“如果仍处于源物理环境，这个动作会产生什么效果”。

验证集指标：

- 状态增量归一化 RMSE：约 0.196；
- 动作 Jacobian 相对误差：约 0.131；
- 动作 Jacobian 余弦：约 0.989。

必须注意：Dense 源孪生是当前性能的重要组成部分，因此论文不能把全部性能归因给 KAN。

### 3.4 在线目标 ProtoKAN

部署后只观察目标转移：

\[
(s_t,a_t,s_{t+1}).
\]

源孪生先给出源反事实增量：

\[
\Delta\hat s_{0,t}
=\hat f_0^{\mathrm{twin}}
\bigl(s_t,a_t-\pi_0(s_t)\bigr).
\]

目标相对残差为：

\[
y_t=(s_{t+1}-s_t)-\Delta\hat s_{0,t}.
\]

目标控制变化用低维矩阵表示：

\[
G_t(s)\approx G_0^{\mathrm{KAN}}(s)T_t,
\]

目标漂移变化在固定 KAN 字典上递归拟合：

\[
\Delta b_t(s)=\Phi(s)W_t.
\]

于是目标认知为：

\[
\hat f_t^{\mathrm{KAN}}(s,\delta a)
=b_0^{\mathrm{KAN}}(s)
+\Phi(s)W_t
+G_0^{\mathrm{KAN}}(s)T_t\delta a.
\]

目标阶段不读取质量、摩擦、执行器缩放或变化类别。

### 3.5 控制等价动作运输

源策略意图对应的标准效果为：

\[
e_0(s)=\hat f_0^{\mathrm{KAN}}(s,0).
\]

在目标认知下反解动作扰动：

\[
\delta a_t^\star
=
\arg\min_{\delta a}
\left\|
\hat f_t^{\mathrm{KAN}}(s,\delta a)-e_0(s)
\right\|_2^2
+\lambda\|\delta a\|_2^2.
\]

最终动作：

\[
a_t=\operatorname{clip}\bigl(\pi_0(s)+\delta a_t^\star\bigr).
\]

当前最强结果使用无门控运输。认知上下文直接改变反解算子，因此不能被 Actor 绕过。

---

## 4. 当前实验协议

### 4.1 公平性约束

正式实验必须遵守：

- 只在源环境训练 Actor 和源认知；
- 目标物理参数不输入模型；
- 不人工告诉算法发生了哪种变化；
- 不使用人工设计的能量教师作为方法成立条件；
- 目标模拟器只能用于离线评价或明确标注的 Oracle；
- warmup 与正式评估使用不同环境实例和随机种子；
- 当前认知归因实验不使用目标奖励更新 Actor；
- 后续允许 Actor 持续学习，但必须有同预算消融；
- 报告恢复曲线，而不只选最终最好回报。

### 4.2 隐藏物理变化

定义位于 `scripts/prescreen_hopper_physics_shifts.py`。名称只用于实验配置和日志，不输入学习器：

- `payload_125`
- `payload_150`
- `friction_070`
- `actuator_080`
- `actuator_065`
- `combo_mild`
- `combo_medium`

### 4.3 当前正式主结果

`combo_medium` 三种子结果：

| 目标反馈数 | 平均回报 |
|---:|---:|
| 0 | \(341.2\pm0.8\) |
| 256 | \(367.8\pm7.4\) |
| 512 | \(388.6\pm6.2\) |
| 1024 | \(399.3\pm3.9\) |
| 2048 | \(407.0\pm1.9\) |

三个种子的恢复曲线均单调上升：

\[
\mathrm{AUC}_{\mathrm{norm}}=0.767\pm0.043.
\]

主要文件：

- `results/hopper_cognitive_recovery_three_seed_summary.json`
- `results/hopper_distilled_policy_recovery_n*_combo_medium_seed*.json`

### 4.4 多物理变化快速筛查

固定超参数、种子 1811：

| 变化 | 0 条 | 512 条 | 2048 条 | 当前判断 |
|---|---:|---:|---:|---|
| `payload_125` | 1854.7 | 1554.8 | 1691.1 | 明显负迁移 |
| `payload_150` | 834.7 | 864.1 | 877.9 | 小幅单调改善 |
| `friction_070` | 588.0 | 680.8 | 585.7 | 早期有效，后期漂移 |
| `actuator_080` | 595.5 | 728.5 | 801.1 | 强单调恢复 |
| `actuator_065` | 420.5 | 475.6 | 503.0 | 稳定恢复 |
| `combo_mild` | 539.3 | 557.0 | 564.0 | 小幅改善 |
| `combo_medium` | 340.0 | 379.9 | 405.5 | 明显恢复 |

除 `combo_medium` 外，大多数仍是快速筛查，不能作为论文最终统计。

### 4.5 归因对照

在 `combo_medium`、2048 条反馈下，已有记录显示：

| 方法 | 使用目标奖励训练策略 | 回报 |
|---|---:|---:|
| 继续训练 PPO | 是 | 339.9 |
| 动作残差 Actor-Critic | 是 | 340.1 |
| 只更新控制增益的效果残差 | 是 | 340.0 |
| 当前认知运输 | 否 | 约 405.8 |

这说明当前正增益不能简单归因于目标奖励上的策略继续训练。但这些对照需要在最终统一代码与多随机种子协议中重跑。

---

## 5. 已经踩过的主要坑

### 5.1 把人工物理知识悄悄写进方法

早期版本使用人工能量函数、人工物理语义分割或手工选择变化类型。即使性能好，也无法证明模型自己学到了可迁移物理规律。

以后新增任何特征都要问：

1. 它是否使用目标真实物理参数？
2. 它是否暗示了变化类别？
3. 它是否只适用于 Acrobot/Hopper 的人工公式？
4. 换一个环境后能否从相同的数据协议自动得到？

如果答案不理想，只能标 Oracle 或诊断，不能作为主方法。

### 5.2 单步预测准确不等于长时域控制正确

我们曾认为单步因果图或一步预测准确后，多步只需连续拼接。实际中闭环状态分布会被模型误差和策略动作共同改变，误差不是独立相加，而是经过动力学 Jacobian 放大。

因此：

- 一步 RMSE 只能衡量局部模型；
- 多步滚动误差反映开环模型；
- 最终必须看闭环回报和恢复曲线。

不能用单步误差单独否决或确认一个控制方法。

### 5.3 用认知模型做长程 MPC 并不自动优雅

模型 MPC 有数学基础，但多步滚动误差、计算成本和动作序列优化会掩盖我们真正的贡献。项目目标最终仍是让决策网络直接输出动作，或通过闭式控制算子产生动作，而不是每步长时间搜索。

MPC 可以保留为 Oracle 或性能上界，不应成为主方法的默认依赖。

### 5.4 直接搬运全部参数没有解决“物理语义可用性”

我们尝试过：

- 将认知网络全部参数映射到 Actor；
- 超网络生成决策参数；
- 参数敏感性损失；
- 强制决策对认知参数敏感；
- 防止 Actor 绕过认知分支。

主要问题是：预测网络参数包含物理信息，不代表它们在重新排列后仍具有稳定、可识别、对控制有意义的坐标。强制敏感只保证“用了”，不保证“用对了”。

因此当前转向结构化函数接口：漂移 \(b(s)\)、控制效果 \(G(s)\) 和控制等价反解。

### 5.5 可逆重参数化不会创造新的决策信息

若目标控制 Jacobian 与源 Jacobian只相差可逆变换，那么把动作残差从一个坐标系换到另一个坐标系，并不会增加可辨识信息。早期“只更新 \(G_t\)”方法在高维 Hopper 上没有显著帮助，就是这个原因之一。

需要传递的是控制任务相关的闭环规律，而不是仅做形式上的坐标变换。

### 5.6 KAN 不是天然抗所有遗忘

局部 B-spline 支撑能减少输入区域分离任务的干扰，但如果环境变化前后的输入区域重叠，而相同 \((s,a)\) 对应不同动力学关系，普通 KAN 仍会覆盖旧规律。

因此：

- KAN 的局部性不等于完整持续学习；
- 同域不同规律需要上下文、记忆、参数隔离、回放或后验机制；
- 不能在论文中声称“KAN 天然解决灾难性遗忘”。

### 5.7 低维成功不能直接外推到 Hopper

Two-Link 上控制等价效果空间表现很好，曾达到接近 100% 成功率。但 Hopper 中存在接触、多刚体耦合和高维联合状态分布外问题。每个状态维度分别落在源范围内，不代表联合状态在源数据支持内。

低维结果适合作为机制验证，不能代替高维证据。

### 5.8 高容量源孪生仍可能无法识别目标状态上的源反事实

有限源数据只约束：

\[
f_0(s,a),\qquad s\sim\rho_0.
\]

目标闭环访问：

\[
s\sim\rho_t.
\]

在 \(\rho_t\setminus\rho_0\) 上，真实源反事实不能由有限源数据唯一确定。Dense MLP、状态云扩增和第一版可组合 KAN 都不能凭空补充缺失信息。

Oracle 源模拟器曾显著改善效果误差和闭环回报，说明动作运输公式有上界，但也暴露源反事实是关键瓶颈。

### 5.9 预测不确定性不能直接作为动作门

联合支持置信度能预测源模型误差，其与真实误差的秩相关在多个变化上达到约 0.44–0.69。但把置信度直接乘在动作修正上，会把 `combo_medium` 的回报从约 405.8 压回 341.6。

原因是：

> 世界模型的一步误差风险，不等于策略修正的长期价值。

置信度更适合：

- 调节认知学习率；
- 决定是否写入长期记忆；
- 触发额外探索；
- 选择模型 ensemble；
- 作为决策网络的附加输入；
- 触发安全回退。

暂时不要直接缩放最终动作。

### 5.10 只看最终性能会掩盖恢复速度与漂移

摩擦变化在 512 条反馈时明显改善，但到 2048 条时收益消失。若只报告某个最好检查点，会错误声称稳定适应。

所有正式实验都应保存统一预算点，例如：

\[
N\in\{0,256,512,1024,2048\},
\]

并报告整条曲线。

### 5.11 当前论文草稿已经落后于实验

`paper/main.tex` 仍以 PCET、CartPole 和 Two-Link 为主，摘要中还声称冻结策略、后验风险拉回和非覆盖记忆。最新 Hopper 正结果使用的是 Dense 源孪生、在线目标 ProtoKAN 和无门控动作运输，且尚未验证完整记忆机制。

因此：

- 当前 LaTeX 只能作为结构和相关工作素材；
- 不能直接提交；
- 不要继续在旧实验表上补数字；
- 应先重新确定论文主故事，再重写摘要、方法和实验。

### 5.12 评估回合数能彻底改变结论

Walker2d friction_070 的零样本回报在 2 回合评估下为 2216.3，在 5 回合下为 3752.2——同一策略、同一变化，仅评估回合数不同。任何基于 2 回合的对比都曾给出严重误导（包括一次虚假的"近完全恢复"）。

因此：

- 正式实验一律 ≥5 评估回合，并报告 std 与逐回合回报；
- 快速筛查的结论必须标注回合数，且不得直接进入论文；
- 零样本基线必须与处理组使用完全相同的评估种子集合。

### 5.13 预测目标与运输目标的正则强度不是一回事

漂移拟合的 ridge 是按预测目标设定的，但运输目标对漂移幅值有独立的信赖要求。Walker2d 证据：预测留出 RMSE 平稳时，漂移范数仍可增长 2.4 倍，并被闭式反解直接转成有害修正；而把 ridge 调强到能压住漂移后，又同时压死了 Hopper 需要的有效修正。

因此：

- 不要用预测指标（RMSE、留出误差）单独标定运输相关超参；
- 漂移/修正在运输意义上存在"有用幅值区间"，需要显式的信赖机制（见 §0.5.4 问题 1）；
- 家族间统一的固定超参可能不存在，超参选择必须有与目标回报无关的判据。

---

## 6. 代码地图

### 6.1 核心模型

- `kanrf/_layer.py`：KAN 基础层。
- `cpbn/generic_affine_kan.py`：控制仿射 ProtoKAN 字典、上下文和递归估计。
- `cpbn/hopper_source_twin.py`：Dense/稀疏源数字孪生及支持域校准。
- `cpbn/adaptive_local_kan_atlas.py`：局部 KAN 字典基础。
- `cpbn/chart_scaled_kan_atlas.py`：尺度化局部图册。
- `cpbn/bayesian_recursive_kan_pullback.py`：Bayesian/Risk pullback 基础；当前部分接口仍由通用 KAN 模块依赖。

### 6.2 源阶段

- `scripts/train_hopper_sb3_ppo.py`
- `scripts/validate_hopper_centered_protokan_cognition.py`
- `scripts/train_hopper_control_sobolev_cognition.py`
- `scripts/train_hopper_source_affine_twin.py`

### 6.3 在线恢复

- `scripts/validate_hopper_joint_online_adaptation.py`
- `scripts/validate_hopper_cognitive_recovery_grid.py`
- `scripts/prescreen_hopper_physics_shifts.py`

### 6.4 诊断

- `scripts/diagnose_hopper_pullback_effect.py`
- `scripts/diagnose_hopper_source_support_confidence.py`
- `scripts/diagnose_hopper_support_gated_pullback.py`
- `scripts/validate_hopper_support_gated_policy.py`

### 6.5 汇总

- `scripts/summarize_hopper_cognitive_recovery.py`
- `scripts/summarize_hopper_physics_grid.py`

---

## 7. 环境与运行约定

### 7.1 本机环境

所有 Python 命令必须通过本地 Conda 环境 `dl_env`：

```powershell
$conda = "C:\Users\32510\miniconda3\Scripts\conda.exe"
& $conda run -n dl_env python -m pytest -q
```

当前验证版本：

- Python 3.10.19
- PyTorch 2.5.1
- CUDA 12.1
- NVIDIA RTX 3070 Ti Laptop GPU
- NumPy 2.0.1
- Gymnasium 1.3.0
- MuJoCo 3.10.0
- Stable-Baselines3 2.4.1
- pytest 9.1.1

### 7.2 GPU

认知网络训练和批量实验优先使用 `--device cuda`。MuJoCo 环境步进主要仍受 CPU 影响，因此并非所有实验都会获得与神经网络训练相同的 GPU 加速比。

### 7.3 不进入 Git 的文件

下列文件默认忽略：

- PPO 模型：`.zip`
- 归一化状态：`.pkl`
- 认知检查点：`.pt`
- smoke 结果和临时日志
- LaTeX 中间文件与生成 PDF

这些文件在原开发机器的 `results/` 中仍可能存在，但新同事克隆仓库后不会自动获得。首次协作时应选择：

1. 从头按下面的流程重建；或
2. 通过团队内部文件传输获取同一批检查点，并核对文件哈希。

不要把二进制模型直接提交普通 Git。

---

## 8. 复现流程

以下命令展示依赖顺序。正式重跑前先执行 smoke 配置，确认 MuJoCo、CUDA 和路径正常。

### 8.1 安装

```powershell
python -m pip install -e ".[research,test]"
```

在本机应改用：

```powershell
C:\Users\32510\miniconda3\Scripts\conda.exe run -n dl_env python -m pip install -e ".[research,test]"
```

### 8.2 训练源 PPO

第一阶段：

```powershell
$conda = "C:\Users\32510\miniconda3\Scripts\conda.exe"
& $conda run -n dl_env python scripts/train_hopper_sb3_ppo.py `
  --physics-shift source `
  --seed 1811 `
  --model-out results/hopper_source_sb3_ppo_seed1811 `
  --norm-out results/hopper_source_sb3_vecnorm_seed1811.pkl `
  --json-out results/hopper_source_sb3_ppo_seed1811.json
```

如需复现当前使用的 continued Actor，再从上一检查点继续：

```powershell
& $conda run -n dl_env python scripts/train_hopper_sb3_ppo.py `
  --physics-shift source `
  --seed 1811 `
  --initial-model results/hopper_source_sb3_ppo_seed1811.zip `
  --initial-norm results/hopper_source_sb3_vecnorm_seed1811.pkl `
  --model-out results/hopper_source_sb3_ppo_continued_seed1811 `
  --norm-out results/hopper_source_sb3_vecnorm_continued_seed1811.pkl `
  --json-out results/hopper_source_sb3_ppo_continued_seed1811.json
```

### 8.3 生成源策略中心化 ProtoKAN

```powershell
& $conda run -n dl_env python scripts/validate_hopper_centered_protokan_cognition.py `
  --seed 1811 `
  --checkpoint-out results/hopper_source_centered_protokan_seed1811.pt `
  --json-out results/hopper_centered_protokan_combo_mild_seed1811.json
```

检查点在目标学习之前保存，因此源检查点本身不含目标物理信息；脚本后半段的目标评估只是诊断。

### 8.4 控制 Sobolev 校准

```powershell
& $conda run -n dl_env python scripts/train_hopper_control_sobolev_cognition.py `
  --device cuda `
  --template-checkpoint results/hopper_source_centered_protokan_seed1811.pt `
  --checkpoint-out results/hopper_source_control_sobolev_calibrated_seed1811.pt `
  --json-out results/hopper_source_control_sobolev_seed1811.json
```

### 8.5 训练源控制仿射数字孪生

```powershell
& $conda run -n dl_env python scripts/train_hopper_source_affine_twin.py `
  --device cuda `
  --model-type dense_mlp `
  --template-checkpoint results/hopper_source_control_sobolev_calibrated_seed1811.pt `
  --checkpoint-out results/hopper_source_affine_twin_cloud_seed1811.pt `
  --json-out results/hopper_source_affine_twin_cloud_seed1811.json
```

### 8.6 运行单个恢复曲线

```powershell
& $conda run -n dl_env python scripts/validate_hopper_cognitive_recovery_grid.py `
  --device cuda `
  --target combo_medium `
  --seed 1811 `
  --budgets 0,256,512,1024,2048 `
  --evaluation-episodes 3 `
  --json-out results/hopper_cognitive_recovery_grid_combo_medium_seed1811.json
```

### 8.7 测试

```powershell
& $conda run -n dl_env python -m pytest -q
```

---

## 9. 论文状态

### 9.1 当前可以沿用的内容

`paper/main.tex` 中可以保留和重构：

- actionability gap 的问题动机；
- CaDM、AugWM、inverse dynamics、GAT 等相关工作；
- 源任务知识与物理实现分离的讨论；
- 恢复曲线而非仅最终性能的评价观点；
- “KAN 不天然抗遗忘”的谨慎边界。

### 9.2 必须重写的内容

- 标题和摘要；
- 方法主体：改为 Dense 源反事实孪生 + 在线目标 ProtoKAN + 无门控控制等价运输；
- 实验主体：以 Hopper 正式恢复曲线为核心；
- KAN 贡献表述；
- 决策网络是否冻结及是否允许持续学习的协议；
- 记忆和 recurring dynamics 主张；
- 所有尚未重跑的 CartPole/Two-Link 数字。

### 9.3 推荐论文故事

推荐的故事线是：

1. 在线世界模型可以重新学会预测，但预测知识不自动转化为有效动作；
2. 直接参数搬运、上下文拼接和一步模型优化不能保证控制语义；
3. 我们构造控制仿射认知接口，把源策略意图表示为局部可控效果；
4. 目标 ProtoKAN 从少量转移中递归学习目标控制算子；
5. 控制等价反解将新认知变为动作，且认知位于必经路径；
6. Hopper 中闭环回报随认知反馈预算稳定恢复；
7. 多物理变化揭示无需适应检测和长期认知漂移仍是边界；
8. 通过公平消融证明增益究竟来自结构化 KAN、源孪生还是普通在线估计。

不要把故事写成“我们首次把 KAN 用于 RL”，也不要把“两网络分离”当作核心创新。

---

## 10. 下一步任务，按优先级

> 2026-07-25 晚更新：以下 P0 已被替换。原 P0（结构消融、稳定性修复）顺延为 P1/P2，见 §0.5.4 与 §0.5.5。

### P0：修复三个已确诊问题（顺序 1→2→3，各自独立验证）

1. 漂移信赖域失控（信赖域约束更新，详见 §0.5.4）；
2. 运输目标可达性判定（可达性残差比，payload_125 验收）；
3. 饱和下激励消失（强制向内最小幅度激励）。

验证协议：Hopper friction_070 / payload_125 改善且 **combo_medium 主结果不回归（保 341→407）**；Walker2d friction_070 / combo_medium 复测；≥5 回合、多种子、全预算曲线。

### P1：结构消融（原 P0，顺延）

同一源 Actor、相同目标转移、相同预算、相同评估种子比较：

1. Dense 源孪生 + ProtoKAN 目标算子；
2. Dense 源孪生 + 参数匹配 MLP 目标算子；
3. KAN 源孪生 + ProtoKAN 目标算子；
4. 只更新全局控制矩阵；
5. 无认知运输；
6. 同预算目标 PPO；
7. 同预算动作残差 Actor-Critic。

新增第 8 项（源自贡献层级对齐）：**可绕过的认知上下文接口**（concat 到可学习 Actor）vs 必经路径接口——这是直接捍卫 first-order claim 的对照。`LearnedMLPDictionary` 已备好，需先打通 `load_cognition` 的字典加载路径。

### P2：决策持续学习接入正结果（Gate D，原 P1）

项目允许决策网络持续学习。应从当前认知运输动作初始化残差 Actor-Critic，并比较：

- 认知冻结、只学决策；
- 决策冻结、只学认知；
- 认知与决策联合学习；
- 普通目标 Actor-Critic。

必须记录早期 AUC，防止评审认为最终提升全部来自策略继续训练。

### P2：第二个高维环境（原 P1，暂停后重启评估）

Walker2d 尝试已关闭（§0.5.3）。HalfCheetah 注册表就绪但源管线未启动。是否重启、选哪个家族，待 P0 修复验证后重新评估——修复后的接口若在 Walker2d 崩溃案例上转好，Walker2d 本身即可作为第二家族完成验证。

### P2：论文同步

完成 P0 修复验证后立即重写论文方法和实验部分，不要等所有实验结束后再改。每个主张都要绑定：

- 对应代码；
- 对应结果 JSON；
- 对应随机种子；
- 对应表格或图；
- 对应消融。

---

## 11. 实验纪律

每次实验前记录：

- Git commit；
- 环境名称；
- 源/目标物理配置；
- 随机种子；
- 目标转移预算；
- 是否更新认知；
- 是否更新决策；
- 是否使用目标奖励；
- 是否使用 Oracle；
- 模型检查点；
- 结果 JSON 路径。

每次实验后至少检查：

- 0 反馈基线是否一致；
- warmup 与评估种子是否隔离；
- 恢复是否单调；
- 动作是否饱和；
- 认知预测是否数值发散；
- 是否存在目标参数泄漏；
- 是否只是挑选了最佳预算点；
- 是否需要多随机种子复核。

新想法快速实验最多迭代三轮。三轮后仍失败，应先写清楚：

1. 失败现象；
2. 哪个假设被否定；
3. 是优化问题、表达问题、可辨识问题还是评价协议问题；
4. 是否值得继续。

---

## 12. Git 与历史

### 当前分支

```text
collab/handoff-latest-20260725
```

只保留当前 Hopper 主线、必要测试、正式结果和论文工作区。

当日提交：

```text
48d6a6f  特性：通过环境注册表将实验管线泛化到 Walker2d
```

另有未提交工作区改动（待确认后提交）：孪生 `action_dim` 透传修复与回归测试、halfcheetah 注册、两个 Walker2d 诊断脚本（`diagnose_walker2d_probe_jacobian.py`、`diagnose_walker2d_friction_drift.py`）、本日全部 Walker2d 实验 JSON。

### 整理前论文候选

```text
research/paper-candidate-v1
```

提交：

```text
47620a4
```

### 完整历史

```text
archive/research-history-20260725
```

提交：

```text
5f20465
```

所有旧 Acrobot、Two-Link、CartPole、失败 Hopper 方法、诊断代码和旧结果都在完整历史分支。需要追溯某个负结果时从归档读取，不要把整批历史重新合并回协作分支。

---

## 13. 接手后的推荐第一周

> 2026-07-25 晚注：本节为原始交接时的计划。当日进展已覆盖其中第一、四天（诊断部分），后续以 §0.5.4 与 §10 的新 P0 为准。

第一天：

- 跑完全部单元测试；
- 用已有本地检查点重跑 `combo_medium` 的 0/512 小规模结果；
- 对照 JSON 确认协议一致。

第二至三天：

- 实现参数匹配的 MLP 目标算子；
- 固定目标转移数据，做 ProtoKAN/MLP 离线拟合对照；
- 检查两者的动作修正与闭环回报，而不只比较 RMSE。

第四天：

- 分析 `friction_070` 从 512 到 2048 的漂移；
- 保存 \(T_t\)、\(W_t\)、动作修正和接触相位的中间统计。

第五天：

- 统一结构消融脚本；
- 至少完成一个目标、三个种子的快速筛查；
- 更新论文主张—证据表。

第一周不要做：

- 新造一个完全不同的网络；
- 增加人工物理标签；
- 只追求单个最好回报；
- 在旧 LaTeX 表格上继续补不一致结果；
- 未做 MLP 对照就宣称 KAN 是性能来源。

---

## 14. 最终交接结论

项目已经越过”只有概念、没有高维正结果”的阶段：Hopper `combo_medium` 上已经出现稳定、三随机种子、随认知反馈预算单调恢复的闭环结果。

但项目尚未越过”论文贡献已经被严格隔离”的阶段。2026-07-25 的 Walker2d 尝试进一步表明，当前接口形式存在明确的适用边界：运输目标可达性、漂移信赖域、饱和激励三个条件不满足时，方法会系统性失效。这三个条件既是问题清单，也精确定义了接口的成立前提，是论文 Limitations 与后续方法迭代的核心素材。

当前按优先级要回答的问题是（顺序已按 2026-07-25 晚对齐更新）：

1. 漂移信赖域约束能否在不回归 Hopper 主结果的前提下修复两家族共有的后期崩溃？
2. 可达性残差比能否自动识别”不该运输”的变化（payload_125 类）？
3. 修复后的接口能否在 Walker2d 崩溃案例上转好，从而完成第二家族验证？
4. （顺延）ProtoKAN 目标算子是否比参数匹配的普通在线模型更好？
5. （顺延）当决策网络也允许持续学习时，认知运输能否仍显著提高早期恢复速度？

如果前三个问题得到扎实答案，项目不仅能修复稳定性边界，还会把”接口何时成立、何时失效”变成论文区别于已有工作的深层贡献。
