# 实验与代码索引

更新日期：2026-07-25  
工作分支：`research/paper-candidate-v1`

## 1. 分支定位

本分支只服务于当前 Hopper 论文候选路线：在单一源物理环境中训练源策略和认知模型；物理条件未知变化后，ProtoKAN 从真实在线转移中持续更新目标动力学，并通过控制等价动作运输影响决策，使闭环回报随交互预算恢复。

完整的历史探索、失败实验、旧环境结果和对应测试保存在：

- 分支：`archive/research-history-20260725`
- 完整归档提交：`5f20465`
- 归档前的 Acrobot/ProtoKAN 批次：`2a4bc88`

除非是在追溯负结果或设计消融，不应从归档分支中的旧实验直接形成论文主张。

## 2. 当前方法

当前候选链路为：

\[
\text{源 Actor/Critic}
\rightarrow
\text{源控制仿射动力学 } f_0(s,a)
\rightarrow
\text{在线目标 ProtoKAN } f_t(s,a)
\rightarrow
\text{控制等价动作运输}
\rightarrow
\text{目标环境闭环动作}.
\]

其核心原则是：

1. 源 Actor 只在一个源物理环境中预训练；
2. 目标物理参数及变化类型不提供给算法；
3. 目标 ProtoKAN 只使用在线获得的 \((s_t,a_t,s_{t+1})\) 更新；
4. 决策端显式使用源、目标认知算子的可控效果差异；
5. 目标环境奖励可以用于决策网络持续学习，但认知模型仍由预测损失独立训练；
6. 论文报告完整恢复曲线和反馈样本预算，而不是只报告最终最好值。

详细架构和理论边界见：

- `docs/CURRENT_RESEARCH_STATUS_REPORT_CN.md`
- `docs/HOPPER_CLOSED_LOOP_COGNITIVE_TRANSPORT_STAGE_CN.md`
- `docs/CONTROL_EQUIVALENT_EFFECT_SPACE_ACTOR_CN.md`
- `docs/HOPPER_IDENTIFIABILITY_STAGE_CN.md`
- `docs/ROOT_CAUSE_AND_THEORETICAL_RESET_CN.md`

## 3. 主实现

### 3.1 模型与基础组件

- `cpbn/generic_affine_kan.py`：控制仿射 KAN/ProtoKAN 认知算子。
- `cpbn/hopper_source_twin.py`：Hopper 源动力学数字孪生及相关数据处理。
- `kanrf/_layer.py`：KAN 核心层实现。

### 3.2 源阶段

- `scripts/train_hopper_sb3_ppo.py`：训练源 PPO Actor。
- `scripts/train_hopper_source_affine_twin.py`：训练源控制仿射动力学模型。
- `scripts/train_hopper_control_sobolev_cognition.py`：加入动作导数约束的认知训练。
- `scripts/prescreen_hopper_physics_shifts.py`：统一定义评测用物理变化；这些标签只用于实验配置，不作为模型输入。

### 3.3 在线恢复与汇总

- `scripts/validate_hopper_joint_online_adaptation.py`：认知与决策联合在线适应主入口。
- `scripts/validate_hopper_cognitive_recovery_grid.py`：跨物理变化恢复曲线。
- `scripts/summarize_hopper_cognitive_recovery.py`：多随机种子恢复速度汇总。
- `scripts/summarize_hopper_physics_grid.py`：多物理变化汇总。

## 4. 当前正式证据

### 4.1 复合物理变化的三随机种子恢复曲线

主文件：`results/hopper_cognitive_recovery_three_seed_summary.json`

在 `combo_medium` 下，同一协议的平均回报为：

| 在线转移数 | 平均回报 |
|---:|---:|
| 0 | \(341.2\pm0.8\) |
| 256 | \(367.8\pm7.4\) |
| 512 | \(388.6\pm6.2\) |
| 1024 | \(399.3\pm3.9\) |
| 2048 | \(407.0\pm1.9\) |

归一化恢复 AUC 为 \(0.767\pm0.043\)。对应的逐预算、逐随机种子 JSON 文件使用
`results/hopper_distilled_policy_recovery_n*_combo_medium_seed*.json` 命名。

### 4.2 多种物理变化

主文件：`results/hopper_cognitive_recovery_physics_grid_summary.json`

逐环境原始文件使用 `results/hopper_cognitive_recovery_grid_*.json` 命名。当前证据表明方法在执行器减弱及若干复合变化上具有稳定恢复趋势，但在载荷变化和摩擦变化上仍存在负迁移或后期漂移。因此，这部分目前是边界证据，不应写成“对所有物理变化均有效”。

## 5. 诊断实验

下列实验用于解释机制，不应替代正式端到端对照：

- `scripts/diagnose_hopper_pullback_effect.py`：动作运输及反事实效果误差。
- `scripts/diagnose_hopper_source_support_confidence.py`：源模型支持域置信度。
- `scripts/diagnose_hopper_support_gated_pullback.py`：支持域门控诊断。
- `scripts/validate_hopper_support_gated_policy.py`：闭环门控实验。

目前结论是：支持域置信度可以预测模型误差，但直接按置信度缩小动作运输会破坏闭环收益，因此它更适合作为安全回退或学习率调度信号，而不是简单的动作门。

## 6. 结果与模型文件管理

- 正式的小型 JSON 指标可以提交 Git。
- `.pt`、`.zip`、`.pkl` 模型与归一化状态不提交 Git；它们保留在本地并由训练脚本重建。
- `_` 开头的 smoke、临时日志、LaTeX 中间文件和 PDF 均被忽略。
- 新结果进入论文证据前，应至少记录：环境变化、随机种子、在线预算、基线、是否使用目标奖励、是否使用 Oracle 信息。

## 7. 当前论文边界

当前最可信的贡献不是“认知网络与决策网络分离”，也不是“所有动力学变化上零样本泛化”。候选贡献是：

> 将持续学习的控制仿射 ProtoKAN 认知算子转化为决策可使用的控制等价接口，并以有限目标交互预算下的闭环恢复速度作为主要评价对象。

在形成最终论文结论前仍需补齐：

1. 目标认知算子的 KAN 与参数匹配 MLP 对照；
2. 同预算在线 Actor-Critic、仅认知更新、仅决策更新及联合更新消融；
3. 载荷和摩擦变化上的负迁移修复或清晰适用域；
4. 至少一个额外高维连续控制环境；
5. 完整多随机种子统计和计算成本报告。
