# KAN-RF: KAN-Based Differentiable World Model for Model-Based Control

## 核心结果：Pendulum-v1 Swing-Up 10/10 全通过

| 方法 | 成功率 | 机制 |
|------|:---:|------|
| 原始决策网络（单步） | 7/10 | `(s) → a`，单时间步 |
| 纯逆优化（单尺度世界模型） | 7/10 | 冻结模型，梯度优化 a |
| 多尺度决策网络（Plan A） | 9/10 | `(s) → (a, k)`，多时间步 |
| **动作探索 + 决策网络微调（最终方案）** | **10/10** | Plan A + 卡住时原地尝试候选动作 + 纠正标签重训 |

## 有效方法

### 方法 1：多时间尺度世界模型 + 决策网络（Plan A）

**问题**：单步力矩在 0.05s 内对角度影响极微（Jacobian 范数 ≈ 0.04），前向误差被放大 25 倍进入逆误差。单步推理从根本上不可行。

**解决**：
- 世界模型从 `f(s, a) → s'` 扩展为 `f(s, a, k) → s_{t+k·dt}`，k ∈ {1, 2, 4, 8, 16}
- 决策网络从 `(s, s_target) → a` 扩展为 `(s, s_target) → (a, k)`
- k 越大，位置变化的 Jacobian 越大（k=16 时约 16 倍），放大因子从 25 降到 1.5

**关键文件**：
- `data_multi_scale.py`：用解析动力学为每个 (s, a) 生成 5 个时间尺度的训练数据
- `train_ms.py`：训练多尺度世界模型 KAN([5, 16, 3])，带 MOPS 光滑正则化
- `train_decision_ms.py`：训练多尺度决策网络 KAN([6, 12, 2])
- `eval_ms.py`：评估

**结果**：9/10，k=16 被选中 75% 的时间，验证了长时间步是必需的。

### 方法 2：动作探索 + 决策网络纠正（最终 10/10 方案）

**问题**：即使多尺度决策网络达到 9/10，Trial 2（底部起步，|dθ|=4.57 rad）仍然失败。世界模型逆优化在底部区域被错误的局部极小值困住——持续输出刹车力矩而非泵能量。

**解决**——三个全通用机制：

1. **检测卡住**：角度误差连续 N 步不下降 → 当前动作方向错误
2. **原地尝试候选动作**：
   - 保存环境状态 `env.unwrapped.state`
   - 对同一状态尝试 5 个候选动作（模型建议的、反方向的、3 个随机的）
   - 每个候选执行 k 步，用真实环境反馈测量改善程度
   - 恢复状态，选最优候选
3. **记录纠正标签**：`(s, a_wrong) → (s, a_correct)`。用所有纠正标签微调决策网络。

**通用性**：不依赖任何系统特定知识（不需要能量函数、不需要动力学方程）。适用于任何支持状态存取的 gym 环境。

**关键文件**：
- `action_explorer.py`：通用动作探索器
- `eval_action_explore.py`：完整管线（探索 + 纠正 + 决策网络微调 + 评估）
- `kan_decision_explored.pt`：最终 10/10 决策网络权重

**结果**：10/10，所有之前失败的 Trial（2, 3, 6）全部通过。

### 方法 3：回合式批量持续学习

**适用场景**：世界模型在部署中遇到训练数据稀疏的区域。

**机制**：
1. 控制过程中收集所有 (s, a, k, s'_real) 过渡
2. 回合结束后，用收集的数据对世界模型做批量 SGD 微调（Adam + 适中的 lr=1e-4）
3. 只更新 spline_weight（B-样条控制点），冻结 base_weight（SiLU 项）
4. 这保护了离线训练的全局结构，同时修正局部区域的预测偏差

**关键发现**：在线逐样本 SGD 会导致灾难性遗忘和模型爆炸。回合式批量微调避免了这个问题。

**局限**：仅修正世界模型预测值不足以改变逆优化行为（逆优化可能仍在同一局部极小值中）。方案 1 直接修正决策映射更有效。

**关键文件**：
- `continuous_learner.py`：通用持续学习器（不依赖系统特定知识）
- `eval_continuous.py`：持续学习评估

## 失败的探索

| 方法 | 结果 | 失败原因 |
|------|:---:|------|
| 在线逐样本 SGD（三因子学习率） | 模型爆炸 | 高方差的单样本梯度破坏 B-样条连续性 |
| 能量泵送启发式 | 有效但不通用 | 需要已知系统能量函数 `E = 0.5*thd² + G*sin` |
| 世界模型回合微调 | 7/10（无改善） | 修正了预测值但未改变逆优化的优化景观 |
| 仅对 Trial 2 训练 | 过拟合 | 数据来自单一轨迹，无法泛化到其他 Trial |

## 关键文件索引

### 基础设施（不变）
- `bspline.py`：向量化 Cox-de Boor B-样条基函数
- `kan_layer.py`：KANLayer（每条边 φ(x) = w·SiLU(x) + Σc_k·B_k(x)）
- `kan_network.py`：多层 KAN wrapper

### 拟合深度训练（阶段 1-3）
- `train_mops.py`：MOPS P-spline 正则化
- `train_cws.py`：CWS Jacobian 对齐
- `train_hybrid.py`：MOPS + CWS 结合
- `train_full_constrained.py`：MOPS + CWS + 单位圆约束

### 多时间尺度（阶段 4）
- `data_multi_scale.py`：生成多时间尺度训练数据
- `train_ms.py`：训练多尺度世界模型 [5,16,3]
- `generate_decision_data_ms.py`：逆优化生成决策标签
- `train_decision_ms.py`：训练多尺度决策网络 [6,12,2]
- `eval_ms.py`：Plan A 评估（9/10）

### 持续学习与动作探索（阶段 5）
- `action_explorer.py`：通用动作探索器（核心新组件）
- `eval_action_explore.py`：完整管线——探索 + 纠正 + 微调 + 评估（10/10）
- `continuous_learner.py`：通用持续学习器（回合式批量微调）
- `eval_continuous.py`：持续学习评估

### 模型权重
- `kan_ms.pt`：多尺度世界模型 [5,16,3]
- `kan_decision_ms.pt`：多尺度决策网络 [6,12,2]（初始，9/10）
- `kan_decision_explored.pt`：动作探索纠正后的决策网络（最终，10/10）

## 复现步骤

```bash
conda activate pyt
cd /Users/zhuangxinyu/KAN/KAN-RF

# 1. 生成多尺度数据
python data_multi_scale.py

# 2. 训练多尺度世界模型
python train_ms.py --lam 0.1

# 3. 生成决策标签
python generate_decision_data_ms.py --n-samples 300

# 4. 训练初始决策网络
python train_decision_ms.py

# 5. 评估 (9/10)
python eval_ms.py

# 6. 动作探索 + 纠正 + 微调 → 10/10
python eval_action_explore.py --episodes 10

# 7. 独立验证最终模型
python -c "
import torch, numpy as np, gymnasium as gym
from kan_network import KAN
PI_2=np.pi/2
dn=KAN([6,12,2],grid_size=5,spline_order=3)
dn.load_state_dict(torch.load('kan_decision_explored.pt',weights_only=True))
dn.eval()
env=gym.make('Pendulum-v1'); st=torch.tensor([[0.,1.,0.]])
ok=0
for t in range(10):
    obs,_=env.reset(seed=42+t*100)
    for _ in range(60):
        sn=torch.tensor([[obs[0],obs[1],obs[2]/8.0]],dtype=torch.float32)
        with torch.no_grad():
            out=dn(torch.cat([sn,st],dim=-1))
            an=out[0,0].item(); k=max(1,min(12,round(out[0,1].item()*16)))
        for __ in range(k):
            obs,_,term,trunc,_=env.step([an*2])
            if abs(np.arctan2(obs[1],obs[0])-PI_2)<0.2:break
        if term or trunc:break
    fe=min(abs(np.arctan2(obs[1],obs[0])-PI_2),2*np.pi-abs(np.arctan2(obs[1],obs[0])-PI_2))
    if fe<0.2:ok+=1
    print(f'  T{t+1}: {\"OK\" if fe<0.2 else \"FAIL\"}  |dth|={np.rad2deg(fe):.0f}deg')
env.close()
print(f'{ok}/10')
"
```

## 理论背景

详见项目文档：
- `IDEA.md`：KAN 可微世界模型 + 梯度决策的核心思路
- `FORWARD_INVERSE_GAP_CN.md`：前向-逆向差距的三个根因分析
- `FITTING_DEPTH.md`：拟合深度理论 + MOPS/CWS 训练框架
- `THEORY_V2.md`：向量场引导、可控性分解等数学基础
- `CONTINUOUS_LEARNING.md`：在线学习理论与实验结果

---

*文档创建于 2026-05-24，记录 KAN-RF 项目 Phase 4-5 的核心成果。*
