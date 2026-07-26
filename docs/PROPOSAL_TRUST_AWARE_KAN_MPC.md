# Transport-Anchored Robust Differentiable KAN MPC

> 分支: feature/trust-aware-kan-mpc | 2026-07-26

## 一句话

不再让执行器提前学习 z→a，而是在部署时以 Transport 为保守控制先验，利用在线辨识的模式感知 KAN ensemble，在数据支持区域内对动作残差进行受约束的鲁棒可微优化；只有当多模型梯度方向、预测改进和真实一步校验均支持该修正时才接受，否则回退到 Transport。

## 核心转变

旧方法试图学"物理变化→动作"（单源数据学不到）。
新方案改为"冻结当前 KAN → 优化动作序列 → 执行"（不需要提前学）。

唯一仍然学习的部分：在线拟合当前目标动力学 F_t(s,a)。不更新策略网络参数。

## 关键组件

1. **Transport nominal** — 安全动作先验（friction 572），所有优化在其附近进行
2. **Mode-aware KAN ensemble** — 多模型 + 接触/腾空分模式
3. **短参考轨迹损失** — Router 输出轨迹而非单点，任务状态空间
4. **V_source 用于选择而非梯度** — 对候选终点排序，不参与裸梯度优化
5. **三层信任域** — 动作约束 / 数据支持 / 模型一致性
6. **复合可信度** — c = c_dir × c_imp × c_data，三者同时满足才接受
7. **硬回退机制** — 不满足 acceptance rule → ΔU = 0 → 纯 Transport
8. **阻尼 Gauss-Newton/LM** — 残差向量形式，可信时 λ 小、大步
9. **真实一步校准** — e_pred 在线调整 ε_a、λ、ensemble 风险权重
10. **接触动力学分模式** — stance/flight 分别处理

## 完整流程

```
目标环境 transitions
    → 在线拟合 mode-aware KAN ensemble
    → Router 产生短参考轨迹
    → Universal Executor + Transport 生成 nominal 动作序列
    → 在 nominal 附近优化动作残差
    → 目标: 轨迹跟踪 + 安全 + 平滑 + 动作先验
    → 约束: 动作信任域 + 数据支持 + 模型一致性
    → 梯度: ensemble 方向一致 + 预测改进一致
    → 阻尼 Gauss–Newton / LM
    → 满足 acceptance rule 才接受，否则 ΔU = 0
    → 只执行第一步
    → 真实预测误差在线校准信任域
```

## 实验计划

- **Phase 0**: 局部梯度审计 — KAN action Jacobian 方向是否可信
- **Phase 1**: True-dynamics MPC 上界 — 目标函数是否与任务对齐
- **Phase 2**: KAN MPC + trust region 消融
- **Phase 3**: 损失函数消融（轨迹 vs 终点 vs +V_source）
- **Phase 4**: 可信度机制消融（ensemble / acceptance rule / 一步校准）
- **Phase 5**: 完整 reward-free 闭环 vs baselines

## 六个关键修正（vs 初版理解）

1. **旧方法失败原因不同**：策略学习类因单源不可辨识；planner 类因模型利用和目标错位
2. **不是完全不学习**：仍需在线拟合 KAN，只是不学跨物理映射
3. **V_source 用于选择而非梯度**：防止 value model exploitation
4. **LM 需要残差向量形式**：构造 e(U) = [e_traj; e_transport; e_smooth; e_safe; e_risk]
5. **回退是硬规则**：acceptance rule 不满足 → 直接 Transport，不是靠阻尼期望
6. **接触动力学分模式**：stance/flight 分开，跨模式时缩小信任域

## 判定逻辑

1. Phase 0 失败 → KAN Jacobian 不可信，先修正模型
2. Phase 1 失败 → 目标函数与任务不对齐
3. Phase 2 失败 → KAN 局部精度不足
4. Phase 3 失败 → 需要更丰富的目标表示
5. Phase 5 > 672 → reward-free 闭环成立
