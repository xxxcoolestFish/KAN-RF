# KAN-RF 项目交接文档 (2026-05-19)

## 重启步骤（按顺序读）

1. **先读这个文档**（你在看了）
2. **读核心 idea 文档**: `IDEA.md`（同目录下）——完整架构、数学推导、实验结果、开放问题
3. **读构建计划**: `INIT.md` ——代码分步构建记录
4. **如果 Desktop 权限丢失**: 代码已备份到 `~/kan-rf-safe/`，可直接从那里继续

### 恢复实验环境
```bash
conda activate pyt
cd ~/kan-rf-safe   # 或 cd ~/Desktop/KAN/KAN-RF/
```

---

## 1. 项目概要

**KAN-RF**: 用 KAN 作为可微世界模型，一网两用：
- **前向**：$s_{t+1} = f_{\text{KAN}}(s_t, a_t)$ — 预测下一个状态
- **反向（穿冻结参数）**：在动作空间做梯度下降 / 多步 shooting，找动作序列 $a^*$

**核心创新**: 无独立 policy network。决策 = 在已被训练的 KAN world model 上做梯度优化。B-样条的处处可微性使这成为可能。

---

## 2. 当前进度（截至 2026-05-19）

### Stage 1: 2D Point-Mass — ✅ 完成

- 线性 + 非线性两种环境
- 单步规划：给定 $s_t$ 和 $s^*$，梯度下降找 $a^*$
- 决策误差 ~0.012 (线性), ~0.014 (非线性)

### Stage 2: Pendulum-v1 — 🔄 功能验证完成，核心问题暴露

**已实现**:
- World model: KAN [4, 12, 3], grid=5, order=3, 756 params
- 训练数据: 15000 条随机动作 transitions
- Val MSE: 0.001 (RMSE ~0.03 per dim)
- 多步 shooting: H=30, Adam lr=0.1, 2 restarts, λ=0.01

**实验结果**:

| Trial | 初始状态 | Model 预测 \|Δθ\| | 真实执行 \|Δθ\| | 结果 |
|-------|---------|------------------|----------------|------|
| 0 (debug) | θ₀=1.21 rad (0.36 rad from upright) | 0.13 | 0.13 | ✓ |
| 1 (正式) | θ₀=-1.91 rad (3.48 rad from upright) | 0.12 | **1.04** | ✗ |
| 2 (未完成) | θ₀=2.37 rad (0.80 rad from upright) | 0.05 (优化中) | — | 🔄 |

**核心发现: Model Exploitation**
- Trial 0 成功是因为初始状态本来就接近直立，trajectory 在训练分布内
- Trial 1 需要完整摆起（从悬挂到底 → 直立），shooting 优化器找到了模型的盲点
- 模型预测 0.12 rad → 真实 1.04 rad，差距 9 倍
- 根因：随机数据只覆盖"底部乱晃"，shooting 发现的高能摆起轨迹在训练分布外
- **这是所有 MBRL 的通病，不是 KAN-RF 独有的**

### 速度瓶颈

B-样条 Python for-loop 导致 ~660s/trial。向量化后可降至 ~10s/trial。尚未优化。

---

## 3. 下一步计划

### 立即要做的

1. **实现 MPC 短地平线重规划** ——解决 model exploitation 的工程方案
   - 当前：30 步开环 shooting
   - 改为：10 步 horizon，只执行第 1 步，观测新状态，重新规划
   - 改动量：~15 行，改 `eval_pendulum.py`
   - 不声称 novelty——这是标准 MBRL 做法

2. **实现 B-样条激活不确定性惩罚** ——KAN 专属的差异化贡献
   - 在 shooting loss 中加一项：惩罚 B-样条激活稀疏的 trajectory
   - 无需额外 ensemble / uncertainty network
   - 这是 KAN vs MLP 的独特优势

3. **Ablation 实验**: MPC only vs MPC + B-spline penalty ——证明 KAN 专属贡献

### 稍后做的

4. 速度优化：B-样条向量化
5. Hebbian 快速适应（框架已预留，尚未验证）
6. 更多 benchmark 对比（CartPole, MountainCar）

---

## 4. 代码文件索引

| 文件 | 内容 | 依赖 |
|------|------|------|
| `bspline.py` | B-样条基函数 (Cox-de Boor) | torch |
| `kan_layer.py` | KAN 层: φ(x)=w·silu(x)+ΣcₖBₖ(x) | bspline |
| `kan_network.py` | 多层 KAN 堆叠 | kan_layer |
| `env.py` | 2D PointMass 环境（线性+非线性）| torch |
| `train_world_model.py` | Phase 1: BP 训练 world model (PointMass) | kan_network, env |
| `decide.py` | Phase 2: 单步梯度决策 | kan_network, env |
| `data_pendulum.py` | 从 Pendulum-v1 采集数据 | gymnasium |
| `train_pendulum.py` | Phase 1: 训练 Pendulum world model | kan_network |
| `shoot.py` | 多步 shooting planner | kan_network |
| `eval_pendulum.py` | Pendulum 评估脚本 | shoot, gymnasium |
| `test_bspline.py` | B-样条性质验证 | bspline |
| `test_kan_layer.py` | KAN 层验证 | kan_layer |

### 模型文件 (.pt)

| 文件 | 内容 |
|------|------|
| `kan_world_model.pt` | PointMass 线性 world model |
| `kan_world_model_nonlinear.pt` | PointMass 非线性 world model |
| `kan_pendulum_model.pt` | Pendulum world model |
| `pendulum_data.pt` | 15000 条 Pendulum transitions |

---

## 5. 容易理解偏差的内容

1. **"KAN-RF" 中的 RF 尚未定义**——不是 Random Forest。可以后续定义为 Representation-Free、Reactive Feedback 等。

2. **Model exploitation 不是我们的问题，是 MBRL 的通用问题**
   - MBPO、Dreamer、PETS、MOPO 都用大量篇幅解决它
   - 我们的优势：B-样条局部性天然携带 "知不知道" 的信号，比 MLP 更适合解决

3. **Trial 0 的 "成功" 是假象**
   - Pendulum-v1 的随机 reset 有时把初始状态放在上半平面
   - 小偏移（0.36 rad）不需要摆起，只需要微调——这对任何模型都 trivial
   - 真正的挑战是 Trial 1：从悬挂摆起到直立（3.48 rad 偏移）

4. **MPC 是我们用的工程方法，不是我们的学术贡献**
   - 地位类似于 MBPO 用 ensemble、Dreamer 用 KL 约束
   - 学术贡献是 "KAN as differentiable world model + B-spline uncertainty proxy"

5. **B-样条激活作为 uncertainty proxy 是 KAN 独有的**
   - MLP 的激活函数（ReLU、tanh）是全局的，无法提供不确定性信号
   - KAN 的 B-样条有严格局部支撑：Bₖ(x)>0 只在特定区间
   - 轨迹上的 B-样条激活稀疏性 = 模型不知道该区域的概率
   - 这个信号是**免费的**——不需要额外训练 ensemble 或 uncertainty net

6. **单层 vs 多层 KAN**
   - 单层是加性模型，无法表达 x₁·x₂ 等交叉项
   - 物理系统的动力学几乎一定有状态-动作交互，必须至少 2 层 KAN
   - 2 层 KAN 通过嵌套实现交叉项：f(x)=Σ_q Φ_q(Σ_p φ_{q,p}(x_p))

7. **环境说明**
   - Python + PyTorch 2.5.1, conda 环境名: `pyt`
   - M2 芯片, MPS 可用但未使用（B-样条在 CPU 上运行）
   - gymnasium 1.3.0 已安装
   - 代码备份路径: `~/kan-rf-safe/`
   - 原始项目路径: `~/Desktop/KAN/KAN-RF/`（可能有权限问题）

8. **计划中的 MPC 不会破坏原有方法**
   - `shoot.py` 不变——多步 shooting 仍然作为底层 planner
   - `eval_pendulum.py` 改——从 30 步开环执行 → MPC 循环：计划 10 步，执行第 1 步，重新观测，重新规划
   - 本质上只是把 shooting 的重规划循环从 "per trial" 变成 "per step"
