# 时间层级目标传播 (Temporal Hierarchy Target Propagation, THTP)

## 1. 动机

现有框架的 Policy 训练有两个根本缺陷：
- **从误差直接映射到动作**：Policy 被要求学 `(s_target - s) → a` 的映射，但因果图上不存在这条边
- **对所有状态维度一视同仁**：`||s_pred - s_target||²` 要求 Policy 同时最小化所有维度的误差，但这些维度处于不同的时间层级

Pendulum 碰巧能工作是因为能量 loss（`增加 sinθ`）隐式地通过 θ̇ 路由了目标。CartPole 没有这种巧合。

## 2. 核心发现

从 ProtoKAN WM 的 Jacobian 自动提取的实际因果结构表明，所有物理控制系统遵循统一的时间层级：

```
Tier 0 (即时可控):   θ̇, ẋ         ← |∂s'/∂a| 大
Tier 1 (一步积分):   θ, x         ← Tier0→Tier1 转移强
Tier 2 (两步积分):   cosθ, sinθ   ← 通过 Tier1 间接到达
```

这是从数据中自动发现的，不依赖任何手工物理公式。

## 3. 方法

### Step 1: 自动发现层级

```python
# 从 WM Jacobian 计算层级
controllability[i] = E_s[|∂s'[i]/∂a|]       # 动作对每个维度的直接影响力
transfer[i→j] = E_s[|∂s'[j]/∂s[i]|]          # 状态内部转移强度

# Tier 0 = controllability 高的维度 (> 阈值)
# Tier k = 可以通过 Tier k-1 以高 transfer 到达的维度
```

### Step 2: 逐层目标传播

从最深 Tier 向 Tier 0 逐层反传：

```
For each tier from deepest to Tier 0:
    For dimension i in current tier:
        error[i] = s_des[i] - s_current[i]
        
        For dimension j in (tier-1) that can affect i:
            Jacobian J_{j→i} = ∂s'[i]/∂s[j]   # 相邻层转移
            
        # 求解: 需要 s[j] 变化多少来弥补 error[i]?
        Δs[tier-1] = pinv(J_{tier-1 → tier}) · error[tier]
        
        # 向下层传播子目标
        s_des[j] = s_current[j] + α · Δs[j]
```

### Step 3: 最终层反解

```
# Tier 0: 直接可控维度
error_0 = s_des[0] - s_current[0]
Δa = pinv(∂s'[Tier0]/∂a) · error_0
a_des = a_current + Δa
```

### Step 4: Policy 训练

Policy 不仅学习最终动作 a_des，而且学习对齐中间子目标：

```python
# 训练时
subgoals = [s_des_tier2, s_des_tier1, s_des_tier0, a_des]
# Policy 内部也可分层，每层对应一个时间层级
loss = Σ_tier ||π_hidden[tier](s) - subgoals[tier]||² + ||π(s) - a_des||²
```

## 4. 与现有框架的融合

```
原三层框架:       Strategy(物理公式) → Execute(Jacobian) → Policy(KAN)
THTP 替代:        Hierarchy Discovery → Target Propagation → Policy w/ subgoal alignment
                   (自动替代Strategy)   (替代Execute)
```

不确定性门控: 当 U(s) 高时，使用保守的子目标（更小的 α 步长）。

## 5. 预期效果

- Pendulum: 保持 100%，且 Policy 内部状态可解释
- CartPole: Policy 训练从不可收敛变为可收敛
- 任何新系统: 自动从 WM 发现层级，无需手工编码物理
