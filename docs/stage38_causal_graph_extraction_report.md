# Stage 38：ProtoKAN 时间因果图提取与真实性验证

## 实验目的

从训练好的认知 ProtoKAN 中提取局部和多步动作因果影响，并与已知的精确源环境动力学进行对照。此阶段不修改决策网络。

## 提取方法

对认知网络

\[
\hat s_{t+1}=F_{\theta_c}(s_t,a_t)
\]

使用中心有限差分干预：

\[
\widehat{\partial y\over\partial x_j}
=
{F(x+\epsilon e_j)-F(x-\epsilon e_j)\over 2\epsilon}.
\]

提取两类图：

1. 单步输入到下一状态的局部影响；
2. 固定动作序列下，`action_t` 对 `state_{t+k}` 的时间影响。

每条边记录中位绝对影响、平均符号和符号稳定性。认知网络和精确源动力学都使用相同的状态、动作探针。

## 结果

三随机种子、128 个状态探针下：

| 因果影响 | 与真实动力学的平均方向相似度 |
|---|---:|
| 单步全部输入 Jacobian | 约 **0.77** |
| `action_t → state_{t+1:t+1}` | 约 **0.05** |
| `action_t → state_{t+1:t+2}` | 约 **0.05** |
| `action_t → state_{t+1:t+4}` | 约 **0.05** |
| `action_t → state_{t+1:t+8}` | 约 **0.04** |

将有限差分步长从 `1e-3` 改为 `1e-2` 后，结论基本不变。

## 解释

单步整体 Jacobian 的 0.77 主要由状态到状态的近似恒等结构贡献，不能说明动作因果方向已经准确。

当前动作列的影响很弱，而认知网络的单步预测误差仍可能大于真实动作引起的微小状态变化。因此：

> 单步 MSE 较小，不代表认知网络的动作敏感性或多步因果结构正确。

这也是此前决策网络难以从认知模型中获得有效长程策略信号的一个直接证据。

## 当前结论

1. ProtoKAN 的逐边结构可以用于提取局部因果候选边；
2. 但当前认知网络还不能直接作为可靠的多步因果路由器；
3. 必须先解决动作敏感性学习问题，或使用干预数据/多步因果一致性损失增强认知网络；
4. 下一阶段路由实验应先以精确源动力学作为 oracle，对比认知图路由与真实图路由，不能直接假设认知图是正确的。

实验模块：`physics_transfer/causal_graph.py`。

实验脚本：`scripts/stage38_causal_graph_extraction.py`。

结果文件：`results/stage38_causal_graph_extraction.json`、`results/stage38_causal_graph_extraction_eps01.json`。
