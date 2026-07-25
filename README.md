# KAN-RF：持续认知驱动的物理泛化

这是项目的协作交接分支，面向当前 Hopper 论文候选路线。

研究目标是：只在一种源物理环境中训练任务策略和认知模型；部署后物理参数发生未知变化时，认知模型仅利用新获得的状态转移持续学习，并把更新后的动力学规律转化为决策可直接使用的控制等价动作，使闭环任务性能在有限交互预算内快速恢复。

## 从这里开始

第一次接手项目，请按顺序阅读：

1. [`docs/COLLABORATOR_HANDOFF_CN.md`](docs/COLLABORATOR_HANDOFF_CN.md)：完整交接，包括研究目的、数学方法、实验结论、失败路线、复现方式和下一步任务。
2. [`docs/HOPPER_CLOSED_LOOP_COGNITIVE_TRANSPORT_STAGE_CN.md`](docs/HOPPER_CLOSED_LOOP_COGNITIVE_TRANSPORT_STAGE_CN.md)：当前最新有效 Hopper 方法。
3. [`docs/EXPERIMENT_INDEX.md`](docs/EXPERIMENT_INDEX.md)：代码和结果文件索引。
4. [`paper/README.md`](paper/README.md)：论文草稿状态。注意：当前 LaTeX 仍是较早的 PCET/CartPole/Two-Link 草稿，尚未与最新 Hopper 结果完全同步。

## 当前最可信结果

在隐藏的 `combo_medium` 物理变化下，冻结源 Actor，只用目标环境转移更新认知模型：

| 目标转移数 | Hopper 平均回报 |
|---:|---:|
| 0 | \(341.2\pm0.8\) |
| 256 | \(367.8\pm7.4\) |
| 512 | \(388.6\pm6.2\) |
| 1024 | \(399.3\pm3.9\) |
| 2048 | \(407.0\pm1.9\) |

三个随机种子的归一化恢复 AUC 为 \(0.767\pm0.043\)。

这证明了“在线认知更新能够通过控制等价接口改善闭环决策”的可行性，但还不能证明对任意物理变化普遍有效。载荷变化存在负迁移，摩擦变化存在后期漂移，KAN/MLP 公平消融也尚未补齐。

## 核心入口

- `scripts/validate_hopper_cognitive_recovery_grid.py`：固定协议的单环境恢复曲线。
- `scripts/validate_hopper_joint_online_adaptation.py`：认知与决策联合在线适应框架。
- `scripts/train_hopper_sb3_ppo.py`：源 PPO Actor。
- `scripts/validate_hopper_centered_protokan_cognition.py`：源策略中心化 ProtoKAN 检查点。
- `scripts/train_hopper_control_sobolev_cognition.py`：源控制 Sobolev 认知。
- `scripts/train_hopper_source_affine_twin.py`：源控制仿射数字孪生。

## 环境

本机实验统一使用 Conda 环境 `dl_env`：

```powershell
C:\Users\32510\miniconda3\Scripts\conda.exe run -n dl_env python -m pytest -q
```

已验证的主要版本：

- Python 3.10.19
- PyTorch 2.5.1，CUDA 12.1
- NumPy 2.0.1
- Gymnasium 1.3.0
- MuJoCo 3.10.0
- Stable-Baselines3 2.4.1

也可以安装研究依赖：

```powershell
python -m pip install -e ".[research,test]"
```

## 分支与历史

- 当前协作分支：`collab/handoff-latest-20260725`
- 整理前论文候选：`research/paper-candidate-v1`
- 完整研究历史：`archive/research-history-20260725`
- 完整历史归档提交：`5f20465`

协作分支移除了旧 Acrobot、早期 Hopper 和大量失败实验文件，但它们没有被永久删除，均可从归档分支恢复。

## 文件管理

模型权重和归一化状态不进入 Git：

- `*.pt`
- `*.zip`
- `*.pkl`

正式的小型 JSON 指标可以提交。Smoke 结果、日志、LaTeX 中间文件和生成的 PDF 均被忽略。
