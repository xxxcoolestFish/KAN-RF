# Reward-Free 跨物理控制：Trust-Aware Differentiable KAN MPC

> 分支: feature/trust-aware-kan-mpc | 2026-07-26

完整方案文档见上文对话。此处记录关键设计决策和实验计划。

## 核心思想

不再训练神经网络提前学习 z → a，而是在部署阶段冻结在线辨识的 KAN 动力学模型，以 Transport 为安全动作先验，将未来动作序列作为优化变量，结合短期目标、长期 source value、安全约束和多层可信度机制，实时求解目标物理下的动作。

## 关键组件

1. **Transport nominal** — 安全动作先验（当前 friction 572）
2. **Differentiable KAN MPC** — 通过冻结 KAN 计算图优化动作残差
3. **三层 trust region** — 动作约束 / 数据支持 / 模型一致性
4. **KAN ensemble** — 梯度方向一致性校验
5. **真实一步校验** — 在线校准置信度
6. **阻尼 Gauss-Newton 优化** — 可信时大步、不可信时回退 Transport

## 实验计划

- Phase 1: True-dynamics MPC 上界
- Phase 2: KAN MPC + trust region 消融
- Phase 3: 损失函数消融
- Phase 4: 可信度机制消融
- Phase 5: 完整 reward-free 闭环

## 判定逻辑

1. True-dynamics MPC 无法超过 Transport → 目标函数问题
2. True-dynamics 有效、KAN 无效 → 模型精度问题
3. 模型预测改善但真实恶化 → model exploitation
4. Trust-aware KAN MPC > 572 且 > 672 → 方案成立
