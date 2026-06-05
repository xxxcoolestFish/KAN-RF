# decision_v2: KAN-Informed Decision Network

## 核心思路

决策网络不需要自己学物理——物理规律以**计算好的特征**的形式从冻结的 KAN 世界模型注入。

```
KAN 世界模型 (冻结, D≥1)
  │
  ├── drift  = f(s, a=0)           自然漂移（系统不加力时的演化）
  ├── J      = ∂f/∂a|_s            动作 Jacobian（B-样条分析导数）
  ├── ρ      = activation_density   训练分布覆盖度
  │
  ▼
  computed features:
  ├── gap    = s* - drift           "还需要改变多少"
  ├── align  = cos_sim(J, gap)      "动作方向对得上吗"
  ├── ctrl   = ||J||                "动作对状态有多大影响"
  └── trust  = ρ                    "模型对这段区域熟悉吗"
  │
  ▼
  决策网络 (微型 MLP, ~200 params)
  ├── 输入: [a_init, gap, align, ctrl, trust, s]
  └── 输出: a
```

## 与旧方案的区别

| | 旧决策网络 (Plan A) | WM+V | 本方案 |
|------|:---:|:---:|:---:|
| 输入 | (s, s*) raw | s (to V) | KAN 计算的特征 |
| 物理知识 | 从逆优化标签中隐式学习 | 无 | KAN 显式注入 |
| 网络大小 | KAN [6,12,2] ~500 params | MLP [3,32,1] ~200p | MLP ~200p |
| KAN 的角色 | 生成训练标签 | 单步预测 | 特征计算器 |

## 实验计划

1. `core.py` — FeatureComputer + TinyDecisionNet
2. `train.py` — 用逆优化标签训练（和 Plan A 同一批数据，对比输入特征的差异）
3. `test_pendulum.py` — Pendulum 实测
