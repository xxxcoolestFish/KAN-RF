
# KAN-RF 项目交接文档 V2

> 日期：2026-07-12
> 当前分支：feature/cdpn
> 本文档面向接手该项目的同事，说明项目的目标、现状、实验进展和下一步计划。


## 一、我们的目的是什么？

构建一个基于 KAN（Kolmogorov-Arnold Network）的、可解释的自适应决策框架，发表在 AAAI 2026（7月底截稿）。

**核心思路**：
- 预测网络（ProtoKAN WM）学习环境的物理动力学 f(s,a) → s'
- 决策网络（KAN Policy）通过预测网络的梯度信号来训练，部署时 KAN 不参与推理
- 环境物理参数变化时（如重力变化），WM 通过在线学习适应（不重训），决策网络跟随更新
- KAN 的 B-样条局部支撑提供了抗遗忘能力，这是 MLP 做不到的


## 二、我们所搭建的框架最理想的样子是什么？

```
预训练：
  仿真器 → 生成 (s,a,s') → ProtoKAN WM (L-BFGS)
                             ↓ (冻结)
  WM a=0 rollout → 测 G → 能量代价函数 → KAN Policy (梯度训练)

部署 + 适应：
  环境参数变化 → WM 预测误差飙升 → 检测
    ↓
  WM 在线适应 → 预测误差恢复
    ↓
  策略重训（通过适应后的 WM 梯度）→ 任务成功率恢复
    ↓
  正常运作
```

**统一方法论**（同一套框架处理不同任务）：
| 组件 | 作用 |
|------|------|
| 预测网络 | ProtoKAN WM（可微、局部支撑抗遗忘） |
| 决策网络 | KAN Policy / KANBasePolicy（从预测网络梯度训练） |
| 代价函数 | 从 WM 结构自动推导（测 G 或加速度匹配） |
| 适应机制 | WM 微调 + G 重新发现 + 策略重训 |


## 三、我们目前的情况是什么？

### 进展

| 项目 | 状态 |
|------|:----:|
| Pendulum 自动发现能量 → 10/10 | ✅ |
| 梯度诊断（能量 94% vs MSE 55%） | ✅ |
| CartPole KANBasePolicy + BPTT → 91% | ✅ |
| 能量方法的重力泛化（g=3~5 保持 100%） | ✅ |
| ProtoKAN WM 精度（val MSE ≈ 0.000000）| ✅ |
| Pendulum 基线稳定性（54% vs 10/10）| ❌ 不一致 |
| CartPole seed 敏感（16~20/20）| ⚠️ 部分解决 |
| 适应学习 recovery | ❌ 效果有限 |

### 当前分支

`feature/cdpn`，最新 commit: `788a78a`

### 环境

- `conda activate pyt`（Python 3.11, PyTorch 2.12.1）
- Gymnasium（Pendulum-v1, CartPole-v1）
- 注意：`baseline_sweep.py` 和 `cartpole_continual.py` 有同名函数（`train_wm`、`evaluate_policy`），导入易冲突


## 四、做了哪些实验？

### 实验 1：代价函数梯度诊断

对比能量、MSE、Lyapunov 的梯度结构：

| 代价函数 | 方向正确 | 梯度幅度 | 成功率 |
|---------|:-------:|:-------:|:-----:|
| **能量（发现/手工）** | **94%** | **1.08** | **100%** |
| Lyapunov causal Q | 55% | 1.52 | 80% |
| MSE | 55% | 0.04 | 60% |

结论：好的代价函数需要方向正确 + 幅度够大二者兼备。

### 实验 2：能量代价函数自动发现

尝试了 5 种方法：

| 方法 | 需要手工公式？ | 成功率 |
|------|:------------:|:-----:|
| 从 WM 测 G → 构造能量 | 否 | 10/10 ✅ |
| E_net（从公式数据学）| 是 | 10/10 |
| 守恒量学习（MLP）| 否 | 7/10 |
| 慢模式线性投影 | 否 | 5/10 |
| 随机 shooting V_net | 否 | 7/10 |

核心方法：ProtoKAN WM 以 a=0 rollout → 利用能量守恒反推 G ≈ 11（真值 10）→ 构造能量亏缺。

### 实验 3：重力泛化

| g | 成功率 |
|:-:|:-----:|
| 3 | 100% |
| 5 | 100% |
| 10（训练）| 96% |
| 15 | 85% |
| 20 | 60% |

### 实验 4：CartPole 策略训练

| 方法 | 平均 | 说明 |
|------|:---:|------|
| **KANBasePolicy + BPTT H=3** | **91%** | 主结果 |
| MLP Policy + BPTT | 100% | 对照上界 |
| KAN Policy + BPTT（原始） | 0% | KANLayer 梯度消失 |

KANLayer 在 BPTT 中梯度消失，所以 CartPole 改用 KANBasePolicy（SiLU + Linear，去掉 B-样条路径），梯度流通正常。

### 实验 5：适应学习

重力 g=10→20：预训练策略从 100% 掉到 60%。WM fine-tune + G 重新发现 + 策略重训后 recovery 不明显，待进一步优化。


## 五、我们需要做的是什么？

### 🔴 高（截稿前必须）

- **修复基线不一致**：独立进程中验证 `train_wm` 的来源，确保用 `baseline_sweep.py` 的 Pendulum 版本
- **补全论文基线对比**：Dreamer、SAC、PPO 在 Pendulum 上的结果（可引用文献）
- **写论文**：梯度诊断 → 自动发现能量 → 统一框架 → 实验验证

### 🟡 中（有最好）

- **CartPole seed 敏感优化**：更多 epoch、调 PD 增益、学习率调度
- **适应学习完整曲线**：g=10→20 的 drop→recover→no-forget 展示
- **ProtoKAN vs MLP 遗忘对比**：展示 ProtoKAN 的局部支撑优势

### 🟢 低（后续）

- KANLayer einsum BPTT 兼容性修复
- ThreeFactorUpdater 适配 ProtoKAN
- 更多环境扩展（Acrobot, MountainCar）


## 六、文件说明

### 本次新增

| 文件 | 内容 |
|------|------|
| `experiments/exp_cost_discovery.py` | 梯度诊断 + 三种自发现方法 |
| `experiments/exp_mvgpo.py` | MVGPO 实验（随机 rollout V 函数 + BPTT）|
| `experiments/exp_adaptation.py` | 重力泛化 + 适应学习实验 |
| `docs/handover_v2.md` | 本文档 |

### 关键已有

| 文件 | 说明 |
|------|------|
| `kanrf/_protokan.py` | ProtoKAN 核心 |
| `control/kan_policy_net.py` | KANPolicy + Trainer |
| `control/bptt_trainer.py` | BPTT 训练器 |
| `experiments/baseline_sweep.py` | Pendulum 基线 |
| `experiments/cartpole_continual.py` | CartPole 数据 + WM 训练 |


## 七、快速开始

```bash
conda activate pyt

# 梯度诊断 + 代价发现
python experiments/exp_cost_discovery.py

# 重力泛化
PYTHONUNBUFFERED=1 python experiments/exp_adaptation.py

# 检查 train_wm 来源
python -c "
from experiments.baseline_sweep import train_wm
import inspect; print(inspect.getfile(train_wm))
"
```

---
*覆盖截至 2026-07-12 的所有进展。*
