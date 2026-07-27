# KAN-RF：任务可控效果接口

当前分支 `research/task-controllable-effect-interface` 是一次干净的方法重构。
旧 Hopper、CPPE、ILC 和 Planner-Router 实验仍保存在原分支和 Git 历史中，
不再混入本分支的代码与结论。

## 研究问题

我们只在一个固定物理环境中学习任务。认知网络仅通过
`(state, action, next_state)` 学习动力学，任务网络通过奖励学习任务意图。
两者之间不再使用人工挑选的速度、高度或关节维度，而是学习一个同时满足
“任务充分”和“动作可控”的效果空间：

\[
z=\psi(s),\qquad
\delta z=R_\phi(s,g),\qquad
a^\star=\arg\min_a
\|\psi(F_\theta(s,a))-(z+\delta z)\|^2.
\]

当前阶段只验证固定物理环境中的任务完成能力，不测试物理泛化。

## 第一阶段实验

环境：`Pusher-v5`。

1. 标准 SAC：确认任务、奖励和可达到的性能上界。
2. 真实 MuJoCo Oracle-CEM：排除认知模型误差，验证规划闭环。
3. 自动效果空间：依次比较完整状态、PCA、随机投影和学习表示。
4. ProtoKAN 替换 Oracle：只有前三步通过后才进行。

详细方法和判决标准见
[`docs/METHOD_AND_EXPERIMENT_PLAN_CN.md`](docs/METHOD_AND_EXPERIMENT_PLAN_CN.md)。

当前已完成 Gate A，并通过容量对照、配对策略后悔诊断和一次规划器在环数据
聚合定位 Gate B 的闭环可行动性缺口。加入候选优势和困难间隔训练后，当前
4 维接口保留约 93.5% 的精确 Critic 推进能力，但尚未达到严格成功阈值。
完整实验结果、失败对照和未完成边界见
[`docs/CURRENT_STATUS_CN.md`](docs/CURRENT_STATUS_CN.md)。

最新最小对照还表明：沿经验可达方向匹配局部 Critic 梯度会把平均最终距离从
0.0678 轻微恶化到 0.0701，并将配对策略后悔从 0.1673 放大到 0.5088。因此该方向
已被否决；当前瓶颈仍是有限时域候选序列的控制等价，而不是静态价值或局部梯度拟合。

新的可达图路由分支首先使用精确动力学验证带动作边的直接执行。第一版能够让机械臂
接近物体，但无法发现建立接触和推动所需的协调动作。实验、调试证据和当前边界见
[`docs/REACHABILITY_ROUTING_STATUS_CN.md`](docs/REACHABILITY_ROUTING_STATUS_CN.md)。
局部控制敏感性产生的多关节动作同样未能跨过接触模式边界，说明单点 Jacobian
不足以构造连续接触任务的有效可达边。

## 环境

所有 Python 命令必须使用本机 Conda 环境 `dl_env`：

```powershell
C:\Users\32510\miniconda3\envs\dl_env\python.exe -m pytest -q
```

已确认 CUDA 可用，训练脚本默认使用 GPU；MuJoCo 模拟与 Oracle-CEM 主要运行在 CPU。

## 快速开始

环境与数值检查：

```powershell
C:\Users\32510\miniconda3\envs\dl_env\python.exe -m scripts.inspect_pusher_env
```

快速 Oracle-CEM smoke test：

```powershell
C:\Users\32510\miniconda3\envs\dl_env\python.exe -m scripts.evaluate_pusher_baselines `
  --controllers zero random oracle --episodes 2 --max-steps 25 `
  --horizon 3 --action-repeat 2 --population 32 --iterations 2 --debug-every 5
```

训练 SAC：

```powershell
C:\Users\32510\miniconda3\envs\dl_env\python.exe -m scripts.train_pusher_sac `
  --total-steps 500000 --device cuda --log-every 5000 --eval-every 25000
```

## 目录

- `kanrf/`：KAN/ProtoKAN 核心、效果接口和 Oracle 规划器。
- `scripts/`：本阶段唯一实验入口。
- `tests/`：数值、状态恢复、梯度与形状测试。
- `docs/`：当前方法与实验记录。
- `results/`：正式 JSON 指标；模型和日志由 `.gitignore` 排除。
