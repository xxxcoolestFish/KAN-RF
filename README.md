# KAN-RF：少环境动力学适应

当前分支：`research/few-environment-adaptation`

## 研究目标

只使用极少数源物理环境学习动力学的共享规律和变化方向。在未知目标环境中，
不读取真实物理参数、不使用环境 ID，只利用少量无奖励转移
`(state, action, next_state)` 识别当前动力学，并快速恢复控制性能。

我们分别限制：

- 源环境数量 \(K\)；
- 目标环境转移数量 \(N\)；
- 目标环境带奖励交互数量。

目标不是用大量 domain randomization 训练一个保守策略，而是研究：

\[
\text{少环境动力学变化学习}
\rightarrow
\text{少转移在线识别}
\rightarrow
\text{闭环控制恢复}.
\]

完整数学假设、实验门槛和创新边界见
[研究路线](docs/FEW_ENVIRONMENT_RESEARCH_PLAN_CN.md)。

上一阶段 Oracle 控制等价实验及其局限见
[Oracle 上限报告](docs/ORACLE_CONTROL_EQUIVALENCE_ADAPTER_GATE_CN.md)。

## 当前保留代码

```text
kanrf/
  _bspline.py                   B-spline 基函数
  _layer.py                     KAN 层
  _network.py                   KAN 网络
  _protokan.py                  ProtoKAN
  _regularization.py            导数与样条正则
  _uncertainty.py               预测不确定性
  control_equivalence_adapter.py  局部控制等价 Oracle 接口
  pusher_oracle.py              Pusher 状态克隆与 Oracle 工具

scripts/
  inspect_pusher_env.py
  train_pusher_sac.py
  quick_validate_oracle_control_equivalence_adapter.py

tests/
  test_kan_core.py
  test_pusher_oracle.py
  test_control_equivalence_adapter.py
```

旧 Router、可达图、效果空间和残差策略实验已从本分支移除，完整保存在：

```text
archive/pre-few-env-pivot-20260727
```

## 环境

所有 Python 命令必须使用本地 conda 环境 `dl_env`：

```powershell
C:\Users\32510\miniconda3\envs\dl_env\python.exe -m pytest -q
```

环境检查：

```powershell
C:\Users\32510\miniconda3\envs\dl_env\python.exe -m scripts.inspect_pusher_env
```

源 SAC 训练：

```powershell
C:\Users\32510\miniconda3\envs\dl_env\python.exe -m scripts.train_pusher_sac `
  --total-steps 500000 --device cuda
```

Oracle 控制等价快速门：

```powershell
C:\Users\32510\miniconda3\envs\dl_env\python.exe `
  -m scripts.quick_validate_oracle_control_equivalence_adapter `
  --device cuda
```

## 下一步

先实现 rank-1 的 `FewEnvironmentDynamics`：

1. 在两个匿名源环境中学习共享 ProtoKAN 和一个参数变化方向；
2. 进入目标环境后仅更新低维 latent；
3. 对比单源 ProtoKAN、同容量 MLP 和全参数微调；
4. 分别测试插值和外推；
5. Gate A 通过后再连接决策网络。
