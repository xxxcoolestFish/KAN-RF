# KAN 结构驱动层次化决策框架

## 项目目标

构建**基于 KAN 结构知识的可解释、可自适应的智能决策框架**，在非平稳环境中：
- 达到与黑箱 MLP 同等或更优的任务成功率
- 提供 MLP 无法提供的决策可解释性（因果图、归因分析、符号控制律）
- 在环境变化时通过分层恢复机制（WM 局部适应 + 策略重构）实现安全自适应
- 形成完整的"检测→降级→学习→恢复"持续学习闭环

---

## 一、架构演化历史

### 1.1 早期探索

| 阶段 | 方法 | 结果 | 发现 |
|------|------|------|------|
| 原始 KAN-MPC | KAN 做 Rollout 规划 | 70% | KAN 函数值精度不足，多步推演累积误差致命 |
| 增强 KAN-MPC | Jacobian 初始化 + 信任域 | 100% | KAN 的精确 Jacobian 方向可极大改善优化效率 |
| decision_v3 | MLP 策略，WM 梯度训练 | 100% | 世界模型梯度方向可训练出高性能策略 |

### 1.2 原框架（feature/residual-physics-policy）：三层架构

```
Strategy 层（无模型物理）      Execute 层（KAN 精确反解）       DecisionKAN（学习型决策）
───────────────────────      ────────────────────────      ─────────────────────────
s → energy_gap(s)            s, v_des → Gauss-Newton       (s, s_goal) → (a, k)
  → delta_E, grad_E            (J_a = ∂f/∂a)                a: 连续动作
  → v_des (desired velocity)   + Adam 精修                  k: 宏动作持续步数
  → 模式(摆起/刹车/稳定)       + 可控性加权(仅优化 θ̇)
```

**核心设计思想**：
- **Strategy 层**：用物理公式算出"状态往哪变"，无模型，代码一行
- **Execute 层**：用 KAN 的 Jacobian ∂s/∂a 做 Gauss-Newton 精确反解——"什么动作能达到 v_des？"这是 KAN 真正不可替代的价值
- **DecisionKAN**：学习 (s, s_goal) → (a, k) 的通用映射，包含目标泛化和时间抽象
- **在线学习**：Three-Factor Updater——只更新激活的基函数，η ∝ 误差 × (1-训练密度) / √(1+更新次数)，保证局部性和抗遗忘

### 1.3 简化版 V1.0（当前 feature/novel-framework）：单层 Policy 梯度训练

```
状态 s → KAN Policy → 动作 a → 冻结 WM(s, a) → s_pred → loss(s_pred, target)
                                                          ↓
                                                   梯度回传更新 Policy
```

**与三层架构的对比**：

| 组件 | 原三层架构 | V1.0 简化版 |
|------|-----------|------------|
| Strategy 层 | 能量公式 → v_des | **无** |
| Execute 层 | Gauss-Newton + Jacobian | **无** |
| Policy 输入 | (s, s_goal) | 仅 s |
| Policy 输出 | (a, k) 动作+持续步数 | 仅 a |
| 物理先验 | Strategy 显式提供 | Policy 需自己从梯度中悟 |
| 可控性加权 | 仅优化 θ̇ | 全维度均匀 |
| 时间抽象 | 宏动作 k 步 | 每步重决策 |
| 在线学习 | Three-Factor (per-basis) | Adam (全参数) |

**成效与局限**：
- 验证了"冻结 WM 梯度可训练 KAN Policy"这一核心假设
- 但 Policy 训练在标准复现条件下仅达 7/10（10/10 为幸运 seed，不可复现）
- 1-step 梯度训练有固有的近视天花板——Oracle（完美物理模拟器）也被卡在同一处

---

## 二、ProtoKAN：打破 WM 精度天花板

### 2.1 动机

B-样条 KAN 的四个结构性问题：
1. **Grid 限制表达力**：每条边仅 G+K 个控制点
2. **Cox-De Boor 递归不可并行**：O(k²) 计算
3. **优势来自 B-样条本身**：移植到 MLP 同样有效
4. **对噪声敏感**：纯局部支撑

### 2.2 ProtoKAN 设计

用**可学习原型点 + 高斯核插值**替代 B-样条：

```
φ(x) = Σ_n softmax(-(x-x_n)²/2σ²) · [y_n + d_n·(x-x_n)]
```

每个原型点有**位置 x_n、值 y_n、导数 d_n** 三个可学习参数。σ 控制局部性。

### 2.3 精度验证

| 任务 | ProtoKAN WM | KAN WM (B-spline) | CWS-KAN WM | MLP | vs KAN |
|------|------------|-------------------|------------|-----|--------|
| Acrobot 1-step | 0.000175 | 0.013153 | — | 0.000208 | **75x** |
| Pendulum 1-step | 0.000007 | 0.009242 | 0.009242 | — | **1300x** |
| CartPole 1-step | 0.000000 | 0.000744 | — | — | **1582x** |
| CartPole 20-step | 0.002557 | 2.387178 | — | — | **934x** |

### 2.4 关键验证

- **Jacobian 精度**（无 CWS 训练）：cos-sim = 0.96，超过 CWS-KAN 的 0.85
- **能量预测误差**：0.061 vs CWS-KAN 的 2.207（36x）
- **持续学习局部性**：σ=0.22 初始化 → 遗忘比 = 1.03（零遗忘），KAN B-样条遗忘比 ≈ 1.0
- **多步 Rollout**：KAN 在第 10 步已崩溃（MSE 1.32），ProtoKAN 仍精准（0.0001）
- **ProtoKAN 首次让 KAN 类架构在回归精度上超越 MLP**

### 2.5 在架构中的位置

| 位置 | 角色 | 状态 |
|------|------|------|
| **WM 层** | ProtoKAN 替代 B-spline KAN | ✅ 精度+局部性双验证 |
| **Policy 层** | ProtoKAN 作为策略基架构 | ⚠️ 训练不稳定，待研究 |

---

## 三、当前性能基线

### 3.1 Pendulum 决策成功率

| 方法 | 可复现成功率 | 说明 |
|------|:-----------:|------|
| ProtoKAN WM + Shooting MPC (H=8, N=500) | **9/10** | 当前最强，待多种子验证 |
| CWS-KAN WM + KAN Policy（重训） | 7/10 | 标准复现 |
| ProtoKAN WM + KAN Policy（重训） | 待测试 | 替换 WM 重训 |
| CWS-KAN WM + KAN Policy（存档 checkpoint） | 10/10 | 不可复现（幸运 seed） |
| Oracle（完美物理模拟器）+ 1-step MPC | 11/20 (g=15) | 1-step 天花板 |

### 3.2 1-step 近视天花板

不管 WM 多完美（包括 Oracle），1-step 贪心策略在某些任务上必然失败：

| 任务 | Oracle 1-step 成功率 | 原因 |
|------|:-------------------:|------|
| Pendulum g=10 | ~100% | 标准重力，一步贪心够用 |
| Pendulum g=15 | 55% | 重力加大，需摆起蓄力 |
| CartPole 大角度 | 0% | 必须多步协调 |
| Acrobot | 0% | 需要几十步摆动蓄能 |

ProtoKAN WM 在这些任务上的表现**已经达到 Oracle 水平**。瓶颈不在 WM，在决策方式。

---

## 四、五层可解释知识模块

从训练好的 WM 和 Policy 中提取，服务决策全流程：

| 层次 | 内容 | 在决策中的角色 |
|------|------|---------------|
| **层1：因果图** | 剪枝后的稀疏连接结构 | 降维规划、动作屏蔽、验证策略结构 |
| **层2：局部线性模型** | Jacobian ∂f/∂a, 激活函数形状 | 策略梯度源、动作饱和/死区约束 |
| **层3：不确定性 & Lipschitz** | 激活密度+预测误差组合 U | 分布外检测、安全回退触发、动作变化率约束 |
| **层4：可加归因** | 输入维度对输出的贡献分解 | 实时解释每一步动作的构成 |
| **层5：符号公式** | 边函数的符号表达式 | 发现物理控制律、接入经典最优控制 |

---

## 五、持续学习闭环

```
正常运作（Mode B）：
  状态 → KAN Policy / Shooting MPC → 安全动作 → 环境

检测（层3）：
  组合不确定性 U 飙升 → 触发降级

安全降级（Mode C）：
  切换至保守策略（能量启发式 / decision_v3）

WM 快速适应：
  滑动窗口数据 → 局部微调 ProtoKAN WM → 预测误差恢复
  （ProtoKAN 遗忘比 1.03，旧知识零损失）

策略重构：
  恢复后的 WM 梯度 → 重训 KAN Policy
  或直接用 Shooting MPC（精度已够支撑多步 rollout）

恢复上线：
  U 回落 → 切回 Mode B → 高性能运行
```

---

## 六、当前文件结构

```
kanrf/
  _protokan.py                     ← ProtoKAN 核心实现
  _layer.py, _network.py           ← B-spline KAN
  _bspline.py, _uncertainty.py     ← 基础设施
  _regularization.py

control/
  kan_policy_net.py                ← KAN Policy + Trainer
  kan_interpretability.py          ← 五层可解释分析
  kan_knowledge.py                 ← 层2+3 知识提取
  kan_enhanced_mpc.py              ← KAN 增强 MPC
  strategy_v2.py (原分支)          ← Strategy 层
  execute_v2.py (原分支)           ← Execute 层 (Gauss-Newton)
  decision_network.py (原分支)     ← DecisionKAN
  continuous_learner.py (原分支)   ← 持续学习器
  online_learning_v2.py (原分支)   ← Three-Factor 在线学习

experiments/
  protokan_benchmark.py            ← Adam 基准
  protokan_lbfgs.py                ← L-BFGS 精度对比
  pendulum_protokAN_stack.py       ← Pendulum 全栈
  cartpole_protokAN_compare.py     ← CartPole 对比
  closed_loop_protokAN.py          ← 闭环持续学习
  continual_protokAN.py            ← 持续学习验证
  diag_rollout.py                  ← 多步 rollout 诊断
  train_kan_policy.py              ← KAN Policy 训练
```

---

## 七、下一步计划

### 🔴 优先级 1：稳定复现最强基线（本周）

- [ ] ProtoKAN WM + Shooting MPC：10 个不同 seed，统计成功率分布
- [ ] ProtoKAN WM + KAN Policy：10 个 seed 重训，统计成功率分布
- [ ] 确定论文核心性能基线

### 🟡 优先级 2：可解释性分析（基线确定后）

- [ ] 策略剪枝 + 因果图可视化
- [ ] 归因分析 + 曲线图
- [ ] 符号回归尝试
- [ ] 与 MLP 量化对比（参数量、持续学习能力）

### 🟢 优先级 3：串联全自动持续学习闭环

- [ ] 检测→降级→WM 微调→策略重训→恢复，全流程自动化

### ⚪ 优先级 4：架构增强（后续论文/Journal 扩展）

- [ ] Policy 输入加入 s_goal（目标泛化）
- [ ] 物理残差学习（策略输出 = 先验 + KAN 残差）
- [ ] n-step WM rollout 辅助 loss
- [ ] 从 WM Jacobian 自动提取可控性权重
- [ ] Gauss-Newton 作为 Shooting MPC 的初始猜测
- [ ] ProtoKAN Policy 训练稳定性研究
