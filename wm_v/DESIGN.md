# World Model + Value Network: 去掉 k 的模型预测控制

## 1. 问题回顾

之前的管线需要选择预测 horizon k：

```
s → 世界模型 f(s, a, k) 预测 k 步后状态 → 根据预测选动作
```

k 的困境：太小 → 动作对状态的影响微乎其微（Jacobian 太小，FORWARD_INVERSE_GAP_CN.md 根因三）；太大 → 预测误差累积。所有 k-selection 机制的根因是：**让世界模型承担了它不擅长的"看远"任务。**

## 2. 新方案：分工

| 组件 | 职责 | 怎么训练 |
|------|------|------|
| 世界模型 f(s,a) | 单步预测 s' = f(s, a, dt) | 离线，MOPS（最准） |
| 价值网络 V(s) | 估计从 s 出发的累积奖励 | 在线，TD(0) |

世界模型只做单步——它最擅长的、预测最准的。价值网络负责看远——通过 TD 学习从数据中隐式编码"从 s 出发多久能到目标"。

k 不再需要。世界模型永远用 k=1。

## 3. 控制循环

```
每步:
  1. 对每个候选动作 a:
     a. 世界模型预测下一步: s'_pred = f(s, a)
     b. 计算即时奖励: r = R(s, a, s'_pred)
     c. 估计未来价值: v = V(s'_pred)
     d. score(a) = r + γ · v

  2. 执行 score 最高的动作

  3. 观察真实下一步 s'_real 和真实奖励 r_real

  4. TD 更新 V(s):
     V(s) ← V(s) + α [r_real + γ·V(s'_real) - V(s)]
```

## 4. 为什么 V(s) 能捕捉长 horizon 效果

Pendulum 底部为例。单步预测下，推一下摆锤几乎不动：

```
s = [cos≈-1, sin≈0, thd≈0]
动作 a = +1.0（大力矩）: f(s,a) → s' ≈ [cos≈-1, sin≈0, thd≈+0.05]

只看即时 s'：几乎没变化。但 V(s') 通过 TD 学习链学会了：
  "thd=+0.05 比 thd=0 离目标更近，
   因为 thd=+0.05 意味着正在向上摆，
   继续推就能到顶部"
```

V(s) 的值通过 TD 从顶部逐步传播到底部——和传统 RL 的价值传播完全一样。

## 5. 两个网络

### 世界模型：KAN

- 架构：KAN([state_dim + action_dim, hidden, state_dim])
- 训练：MOPS（P-spline, λ=0.1），离线，全批量
- 冻结后不再更新

为什么用 KAN：B-样条导数结构让 MOPS 可以施加光滑性约束。FITTING_DEPTH.md 的完整框架。

### 价值网络：MLP

- 架构：MLP([state_dim, 32, 1])，~200 params
- 训练：TD(0)，在线，带 replay buffer（500 transitions）
- 每步更新一次

为什么用 MLP：价值函数是简单的标量映射，不需要 B-样条结构。MLP 计算快。小的 replay buffer 打破数据相关性。

## 6. 四环境预测

| 环境 | V(s) 学会什么 | 预期效果 |
|------|------|------|
| Pendulum | 离直立越近→值越高，底部需要 swing-up | ≥ 之前多尺度管线 |
| CartPole | 杆子垂直+小车居中→值高 | ≥ 99% |
| MountainCar | 速度大→值高，即使位置还在左边 | 捕获蓄力策略 |
| Acrobot | 末端高度→值高 | ≥ 之前 |

## 7. 和传统 RL 的关系

DQN/PPO：直接学策略 π(s)→a 或 Q(s,a)→标量。

本方案：世界模型 + 价值函数，用计划（MPC）融合两者。这和 AlphaGo 的 MCTS + value network 是同构的——世界模型替代了 MCTS 的 rollout，V(s) 替代了 value network。

好处：世界模型可以复用（换目标只需重新定义 R(s,a,s')），价值网络很小（只学标量）。且继承了 MPC 的安全性——即使 V(s) 不完全准确，世界模型单步预测仍然提供合理的动作方向。

## 8. 实现计划

1. `wm_v_core.py` — 通用控制器：MPC + V(s)，不依赖环境
2. 在 Pendulum 上首测（四环境中 k 选择最关键的）
3. 至少达到之前 10/10 的成功率
4. 扩展到其他三个环境
