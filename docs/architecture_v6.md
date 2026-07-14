# KAN-RF Architecture v6 — Cognitive-Decision Separation with BSRM

## 1. 核心哲学

构建一个类似人类认知的决策框架，包含三个层次：

```
认知层（ProtoKAN WM）
  → 学习环境动力学 f(s, a) → s" 
  → 持续学习，检测环境变化
       ?
通信层（BSRM — B-spline Relevance Shift Mapping）
  → 用 WM 的预测差异定位"动力学变化区域"
  → 将变化区域映射到决策网络的 B-spline 系数
       ?
决策层（KAN / ProtoKAN Policy）
  → 直接通过可微 WM 训练（不依赖 reward）
  → 只更新被 BSRM 选中的系数
```

---

## 2. 各组件定义

### 2.1 认知层：ProtoKAN WM

**输入**： (s, a)  ∈ ?^(state_dim + 1)  
**输出**： ?"     ∈ ?^(state_dim)  
**结构**： ProtoKAN([state_dim + 1, hidden_dim, state_dim])  
**核函数**： Gaussian 原型核

每层的计算：
```
diff_n = x - x_n                          # 到原型 n 的距离
w_n    = softmax(-diff_n2 / 2σ2)          # 原型权重
φ(x)   = Σ_n w_n · (y_n + d_n · diff_n)  # 边函数输出
base_out = silu(x) · W_base               # 跳跃连接
output = base_out + Σ_edges φ(x)          # 最终输出
```

**关键性质**：
- Gaussian 核光滑，全空间非零梯度 — 可微训练
- σ ≈ 0.22（初始化 log_sigma = -1.5），实际影响范围约 ±3σ
- L-BFGS 预训练，val_mse 可达 ~3e-6

### 2.2 决策层：KAN / ProtoKAN Policy

**输入**： s ∈ ?^(state_dim)  
**输出**： a ∈ [-1, 1]^(action_dim)  

**两个候选架构**：

| 架构 | 数学公式 | 局部性 | 训练特性 |
|------|---------|--------|---------|
| KAN (B-spline) | φ(x) = w·silu(x) + Σ c_k·B_k(x) | 严格局部（B_k 在区间外为零） | 梯度在区间边界不连续 |
| ProtoKAN | φ(x) = Σ w_n(x)·(y_n + d_n·(x-x_n)) | 近似局部（softmax 集中在最近原型） | 梯度全空间光滑 |

**策略训练方式**（核心创新）：直接通过可微 WM 训练

```
┌──────────────────────────────────────────┐
│  训练循环（θ? 下）：                       │
│                                          │
│  s? ~ ρ (初始状态分布)                    │
│  for t = 0 to H-1:                      │
│    a_t = Policy(s_t)                     │
│    s_{t+1} = WM(s_t, a_t)  [可微！]     │
│    loss += γ? · ||s_{t+1} - s_goal||2   │
│  loss.backward() → 梯度通过 WM → Policy  │
└──────────────────────────────────────────┘

不需要 reward！不需要 SAC！
WM 本身提供了环境的"仿真"，Policy 通过学习 WM 来理解动作的后果
```

### 2.3 通信层：BSRM

**功能**：在环境变化时（θ? → θ"），用 WM 的预测差异定位动力学变化区域，然后映射到决策网络的 B-spline 系数。

**算法**：

```
输入：WM?（θ? 训练）、WM"（θ" 微调）、少量新环境数据 D"

步骤 1：检测动力学变化
  for each (s, a) ∈ D":
    pred? = WM?(s, a)
    pred" = WM"(s, a)
    δ(s) = ||pred" - pred?||2
  → δ(s) 在动力学变化大的区域高，不变的区域低

步骤 2：映射到 B-spline 系数
  for each 系数 c_{ijk} (输入 i → 隐藏 j → 第 k 个基函数):
    support = [t_k, t_{k+order+1}]  （B-spline 的局部支撑区间）
    或 proto_range = [x_n - 3σ, x_n + 3σ]  （ProtoKAN 原型有效范围）
    
    relevance_{ijk} = Σ_{s ∈ D"} δ(s) · I(s_i ∈ support)
    
步骤 3：选取需要更新的系数
  threshold = mean(relevance) + 2·std(relevance)  （启发式）
  或使用 95% 累积贡献曲线
  R = {c_{ijk} | relevance_{ijk} > threshold}

步骤 4：在 R 子空间上优化
  冻结不在 R 中的所有系数
  用 WM" 做仿真器，在 R 上跑几步梯度上升
  → 适应完成
```

---

## 3. 训练与适应流程

### 3.1 训练阶段（θ?）

```
Phase A：预训练 WM
  数据：随机 (s, a, s") 覆盖全状态空间
  优化器：L-BFGS
  目标：min ||WM(s,a) - s"||2
  结果：WM? 学会了 θ? 下的环境动力学

Phase B：训练 Policy（通过可微 WM）
  优化器：Adam
  目标：min Σ? γ? · ||WM(s_t, Policy(s_t)) - s_goal||2
  梯度：loss → Policy（通过可微 WM 反传）
  冻结 WM? 参数（只反传梯度给 Policy）
  结果：Policy? 学会了在 θ? 下达到目标状态的动作序列
```

### 3.2 适应阶段（θ? → θ"）

```
Phase C：WM 微调
  数据：少量 θ" 下的 (s, a, s") 样本
  优化器：Adam（~500 步）
  目标：min ||WM"(s,a) - s"||2
  结果：WM" 学会了 θ" 下的新动力学

Phase D：BSRM 定位变化
  用 BSRM 计算 relevance，选取需要更新的系数集 R

Phase E：选择性策略更新
  冻结 ? R 的系数，在 R 子空间上优化
  用 WM" 做仿真器
  目标：min Σ? γ? · ||WM"(s_t, Policy(s_t)) - s_goal||2
  结果：Policy" 适应了 θ" 下的新动力学，且只改了必要的参数
```

---

## 4. 与之前版本的关键区别

| 方面 | CDPN v1-v4 | CDPN v5 | CDPN v6 (本架构) |
|------|-----------|---------|-----------------|
| 训练方式 | WM 梯度 / 抽象动力学 / ES | SAC（独立、无 WM） | 通过可微 WM 训练 Policy |
| 决策网络 | MLP / KAN | MLP / KAN | KAN / ProtoKAN |
| 通信方式 | Bridge / 无 | 无 | **BSRM** |
| 适应后选择 | 全量微调 | SAC 自然适应 | **只改 BSRM 选中的系数** |
| 依赖 reward | 否 | 是（SAC） | 否 |

---

## 5. 待验证的实验问题

1. **通过可微 WM 训练 KAN/ProtoKAN Policy 是否收敛？**  ProtoKAN 的 Gaussian 核应提供光滑梯度，KAN 的 B-spline 在单步反传下梯度不消失（不同于 BPTT 多步）

2. **WM 在 KAN/ProtoKAN 动作数据上的预测误差是多少？** 决定了 BSRM 的 δ(s) 信号质量

3. **BSRM 选出的系数集 R 是否稳定？** 是否在多次 bootstrap 下一致

4. **适应后的遗忘测试**：回到 θ? 环境下，性能是否保持不变

---

## 6. ProtoKAN BSRM 注意点

ProtoKAN 使用 Gaussian 核，原型在全局都有响应：
```
w_n(x) = softmax(-(x - x_n)2 / 2σ2)
```

当 σ 很小时（≈0.22），离原型超过 3σ 的输入贡献指数级小。因此 prototye n 的"有效支撑区间"可定义为：
```
proto_n 影响范围 = [x_n - 3σ, x_n + 3σ]
```

此范围覆盖了该原型 95% 以上的权重贡献。超出此范围的输入，该原型的权重指数级小（~e^(-4.5) ≈ 0.01），可以忽略。

**ProtoKAN 上 BSRM 的 relevance 计算**：
```
relevance_n = Σ_{s ∈ D"} δ(s) · I(s_i ∈ [x_n - 3σ, x_n + 3σ])
```

选择需要更新的原型参数 {x_n, y_n, d_n}，而非 B-spline 系数。
