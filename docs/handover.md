# 项目交接文档

## 项目目标

构建基于 KAN 结构知识的**可解释、可自适应**的智能决策框架。具体目标：

1. 决策系统在标称环境下达到与黑箱 MLP 同等或更优的成功率
2. 提供 MLP 无法提供的可解释性（因果图、归因分析、符号控制律）
3. 物理参数变化时，通过 WM 局部适应 + Policy 跟随更新，实现**不重训**的持续学习
4. 最终交付五层可解释知识模块 + 安全持续学习闭环

---

## 当前最新架构（feature/cdpn 分支）

### 两个核心网络

```
┌─────────────────────────────────────────────────┐
│              ProtoKAN WM（预测网络）               │
│  唯一职责：学物理规律                               │
│  输入: (s, a) → 输出: s'                         │
│  训练: L-BFGS(离线) / Adam(在线单步)              │
│  特性: 小σ初始化(0.22), 局部支持, 遗忘比1.03      │
│                                                  │
│  提供三个关键产物:                                 │
│    1. BPTT 多步想象空间                           │
│    2. Jacobian → Tier 层级 → Riccati P 矩阵       │
│    3. 不确定性 U(s) 用于安全门控                   │
└─────────────────────────────────────────────────┘
                      ↓ P矩阵, Tier结构
┌─────────────────────────────────────────────────┐
│           KAN Policy / B-spline（决策网络）        │
│  输入: s → 输出: a                               │
│  训练: BPTT through WM (H=3) 或 Lyapunov-BPTT   │
│  架构: KANLayer(3/4→12) → KANLayer(12→12) → Linear → tanh │
└─────────────────────────────────────────────────┘
```

### 训练管线

**离线预训练**：
1. 用仿真器/物理公式生成 (s, a, s') 数据
2. ProtoKAN WM L-BFGS 训练（批量，~100轮，秒级完成）
3. Policy 通过 BPTT through WM 训练（200轮）

**在线持续学习**：
1. 部署中每步收集 (s, a, s'_real)
2. WM 用单步 Adam 在线更新（局部适应）
3. Policy 用更新后的 WM 继续 BPTT 微调

### 关键设计决策

- **ProtoKAN** 替代原始 B-spline KAN 作为 WM：精度提升 1500x
- **BPTT** 替代单步 WM 梯度训练 Policy：CartPole 从 0% 到 100%
- **Lyapunov P 矩阵** 自动合成：不手写 loss 函数
- **小 σ ProtoKAN** 保证局部性：遗忘比 1.03 vs MLP 16.3x

---

## 尝试过的架构全记录

### 一、WM 架构

| 架构 | 精度 (CartPole) | 优缺点 | 状态 |
|------|:---:|------|:---:|
| B-spline KAN [4,12,3] | val_mse≈0.009 | 精度受限于B-样条天花板 | ❌ 废弃 |
| **ProtoKAN [4,12,3] σ=0.22** | val_mse≈0.000000, 1582x KAN | 精度+局部性双优 | ✅ 当前 |
| CWS-KAN (Jacobian训练) | cos-sim=0.85 | Jacobian不如ProtoKAN | ❌ 废弃 |

### 二、Policy 架构

| 架构 | Pendulum | CartPole | 优缺点 | 状态 |
|------|:---:|:---:|------|:---:|
| KAN B-spline [n,12,12,1] | 10/10 | 0% (单步) | 稳定、可解释 | ✅ 当前 |
| ProtoKAN Policy [n,12,12,1] σ=0.22 | ~7/10 | 0% | 训练不稳定 | ⚠️ 探索中 |
| Goal-Conditioned (s,s_goal→a) | 10/10 | 0% | 扩大输入未改善 | ❌ 废弃 |
| Causal-Decomposed (v_des→Execute→a) | 6/10 | 0% | 数学等价于线性变换 | ❌ 废弃 |
| Structure-Aware (s,P_diag→a) | 55% | 未测试 | 喂P_diag未带来泛化 | ⚠️ 探索中 |

### 三、Policy 训练方法

| 方法 | Pendulum | CartPole | 原理 |
|------|:---:|:---:|------|
| 单步 WM 梯度 (能量引导) | 10/10 | 0% | `s→a→WM→s'→loss` |
| **BPTT H=3 (多步想象)** | 7/10 | **20/20** | H步rollout，累积梯度 |
| Lyapunov-BPTT (因果 Q) | 8/10 | 20/20 | Riccati P+多步 |
| MPC 蒸馏 | 10/10 | 训练不稳定 | MPC教师→Policy学生 |
| Lyapunov-BPTT+ProtoKAN | 55% | 未测试 | 局部适应+Lyapunov |

### 四、代价函数

| 代价函数 | Pendulum 成功率 | 来源 |
|------|:--:|------|
| 手工 energy `0.5θ̇²+10sinθ` | 10/10 | 物理公式 |
| Lyapunov P (Identity Q) | 6/10 | Riccati 自动 |
| Lyapunov P (causal Q) | 8/10 | Riccati+因果层级 |
| 层级动态代价 (THTP) | 6/10 | WM Jacobian |
| V_θ 引导 (learned nonlinear) | 7/10 | FD梯度MPC学习 |
| 双模式 Lyapunov | 55% | 因果层级+阈值 |

### 五、决策方法（无 Policy）

| 方法 | CartPole | Pendulum | 速度 |
|------|:---:|:---:|:---:|
| Batch shooting MPC (N=500,H=3) | 20/20 | 55% | ~12s/5trials |
| FD 梯度 MPC | 未测试 | 55% | 极慢 |

---

## 当前问题和挑战

### 问题一：Pendulum 摆起任务的代价函数瓶颈（核心未解决问题）

**现象**：所有从 WM 自动导出的代价函数在 Pendulum 摆起任务上 ≤ 55%（接近 Oracle 1-step 上限 62%）。手工能量公式 `E=0.5θ̇²+10sinθ` 可达 10/10。

**根因**：能量 E 的梯度是全局一致的指南针（"增加能量"永远是对的）。Lyapunov V(s) = (s-s*)^T P (s-s*) 的梯度指向"几何距离的直线下降"，与真实动力学不对齐。摆起恰好需要暂时增加能量（远离目标的几何距离）。这是代价函数的**拓扑问题**——不是 Policy 容量问题。

**可探索方向**：
- 从 WM 多步 rollout 推导保守量（能量类比）
- 将手工能量公式作为"提示"，学习其结构
- 对摆起和稳定化用不同模式的代价函数

### 问题二：BPTT 训练稳定性（CartPole 有时失败）

**现象**：BPTT H=3 在 CartPole 上有时 20/20，有时 1/20。训练结果依赖初始化 seed。

**可能原因**：
- B-spline Policy 的输出层初始化随机
- BPTT 计算图深度=H，梯度方差大
- 训练超参未针对 BPTT 调优

### 问题三：ProtoKAN backward 速度慢

**原因**：ProtoKAN 使用 softmax + exp，backward 比 B-spline KAN 慢约 5x。导致 BPTT 等需要多步 backward 的方法训练慢。

**缓解**：减小 σ 减少活跃原型点数；批量并行化；FD 梯度替代 autograd。

---

## 文件导航

```
kanrf/
  _protokan.py              ← ProtoKAN 核心实现
  _layer.py, _network.py    ← B-spline KAN

control/                     ← 控制与决策模块
  kan_policy_net.py          ← KANPolicy + KANPolicyTrainer (当前Policy)
  bptt_trainer.py            ← BPTT 训练器
  lyapunov_bptt.py           ← Lyapunov合成 + Lyapunov-BPTT
  gradient_mpc.py            ← 批量MPC (shooting + FD gradient)
  learned_lyapunov.py        ← 学习非线性V_θ
  hierarchical_cost.py       ← 层级动态代价
  lyapunov_policy.py         ← ProtoKAN Policy + Lyapunov-BPTT
  structure_aware_policy.py  ← 结构感知Policy (P_diag输入)
  cdpn.py                    ← Causal-Decomposed Policy (探索)
  protokAN_distill.py        ← ProtoKAN Policy + MPC蒸馏

experiments/                 ← 实验脚本
  baseline_sweep.py           ← Pendulum基线验证
  cartpole_continual.py       ← CartPole持续学习
  cartpole_protokAN_compare.py ← CartPole WM对比
  bptt_test.py                ← BPTT测试
  gradient_mpc_sweep.py       ← MPC sweep
  cdpn_test.py                ← CDPN测试

docs/
  architecture.md             ← 完整架构文档
  causal_routing.md           ← 因果路由设计
  handover.md                 ← 本文档
```

---

## 快速上手

### 训练一个 CartPole Policy

```python
from experiments.cartpole_continual import generate_wm_data, train_wm, generate_policy_states
from control.kan_policy_net import KANPolicy
from control.bptt_trainer import BPTTTrainer

# 1. 训练WM
X, Y = generate_wm_data(g=9.8, n=5000)
wm, _ = train_wm(X, Y, 'protokan', 80)

# 2. 训练Policy (BPTT H=3)
s_pol = generate_policy_states(15000)
policy = KANPolicy(state_dim=4, action_dim=1, hidden_dim=12, n_layers=2)
trainer = BPTTTrainer(wm, policy, S_TARGET, lr=1e-3, horizon=3)
for ep in range(1, 101):
    trainer.train_epoch(s_pol)

# 3. 部署
a = trainer.get_action(s_numpy)
```

### 训练一个 Pendulum Policy

```python
from experiments.baseline_sweep import generate_pendulum_data, train_wm, generate_policy_states
from control.kan_policy_net import KANPolicy, KANPolicyTrainer

X, Y = generate_pendulum_data(5000, seed=42)
wm, _ = train_wm(X.to(device), Y.to(device))
s_pol = generate_policy_states(10000, seed=42).to(device)

policy = KANPolicy(state_dim=3, action_dim=1, hidden_dim=12, n_layers=2)
trainer = KANPolicyTrainer(wm, policy, S_TARGET, lr=1e-3)
for ep in range(1, 201):
    trainer.train_epoch(s_pol)
```
