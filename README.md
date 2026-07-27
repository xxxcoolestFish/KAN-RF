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

样条梯度扩散的已验证边界见
[样条耦合外推快速门](docs/SPLINE_COUPLING_EXTRAPOLATION_GATE_CN.md)。

共享函数坐标与低维动作调制的首次正结果见
[函数对齐动作调制 ProtoKAN 快速门](docs/FUNCTION_MODULATED_PROTOKAN_GATE_CN.md)。

认知函数到 Actor 低秩参数调制的闭环结果见
[认知必经低秩 Actor 闭环快速门](docs/COGNITIVE_LOWRANK_CONTROL_GATE_CN.md)。

## 当前保留代码

```text
kanrf/
  _bspline.py                   B-spline 基函数
  _layer.py                     KAN 层
  _network.py                   KAN 网络
  _protokan.py                  ProtoKAN
  _regularization.py            导数与样条正则
  _uncertainty.py               预测不确定性
  cognition_modulated_actor.py  认知必经的低秩 Actor 参数调制
  function_modulated_dynamics.py  共享函数坐标认知模型
  control_equivalence_adapter.py  局部控制等价 Oracle 接口
  pusher_oracle.py              Pusher 状态克隆与 Oracle 工具

scripts/
  inspect_pusher_env.py
  train_pusher_sac.py
  quick_validate_function_modulated_protokan.py
  quick_validate_cognitive_lowrank_lqr.py
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

当前认知函数适应门和低秩 Actor 闭环机制门均出现正信号。下一步不再调整
低维 LQR，而是把固定接口接入多环境 Actor-Critic：

1. 先保证所有源环境的 Actor 达到预设性能门；
2. 目标环境保持 64 条连续无奖励转移预算；
3. 冻结目标 Actor，只改变认知 latent，隔离认知贡献；
4. 对比 pooled、concat、低秩调制和认知置零；
5. 通过后再加入历史 context 编码器和目标 Actor 持续学习。
