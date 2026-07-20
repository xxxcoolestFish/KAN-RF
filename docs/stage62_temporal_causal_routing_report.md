# Stage 62：时序因果图拼接与路由验证

## 目的

验证以下核心设想是否在数学和代码上成立：

1. 从认知动力学的每一步提取状态影响矩阵和动作影响矩阵；
2. 将一步影响图沿时间前后拼接；
3. 从未来目标状态向前面的动作反向路由影响信号；
4. 得到每个动作对长时域可达目标的调整方向。

## 方法

对一步认知动力学

\[
s_{k+1}=F_\phi(s_k,a_k)
\]

提取

\[
A_k=\frac{\partial F_\phi}{\partial s_k},
\qquad
B_k=\frac{\partial F_\phi}{\partial a_k}.
\]

使用平滑最大高度作为可达性分数。目标状态信号通过伴随方程反向传播：

\[
\lambda_k=A_k^\top\lambda_{k+1}+\nabla_{s_k}c_k,
\]

\[
r_k=B_k^\top\lambda_{k+1}.
\]

其中 `r_k` 表示第 `k` 个动作对未来可达性分数的局部影响方向。

本阶段使用精确目标动力学验证数学实现，动作序列来自 Stage 61 的粗粒度 CEM 结果：32 个动作块，每个动作保持 16 个物理步。

## 因果图拼接正确性

将显式 `A/B` 图路由得到的动作影响，与完整动力学展开后直接自动微分得到的动作梯度比较：

| 指标 | 结果 |
|---|---:|
| 显式因果路由分数 | 0.6853129 |
| 完整自动微分分数 | 0.6853129 |
| 梯度余弦相似度 | **1.000000** |
| 最大绝对梯度误差 | **2.62e-6** |

这证明一步动力学影响图可以通过链式法则准确拼接成多步动作—状态影响图。显式路由与完整模型梯度在数值上等价。

## 路由修正实验

使用显式因果路由反复修正动作序列：

- 较大固定步长 `0.12` 导致明显奇偶振荡；
- 较小步长 `0.03` 将块端点最高高度从 `0.682` 提高到约 `0.710`；
- 20 次局部路由仍未达到高度阈值 `1.0`。

因此，线性化因果路由提供了正确的局部方向，但在 Acrobot 强非线性、长时域任务中不能单独完成全局策略搜索。

## 结论

用户提出的核心设想成立：

\[
\boxed{
\text{一步 ProtoKAN 动力学图}
\rightarrow
\text{时序拼接}
\rightarrow
\text{目标反向路由}
\rightarrow
\text{动作影响方向}
}
\]

但最终架构不能只使用固定 Jacobian 乘积。正式的因果路由头需要：

1. 保留 ProtoKAN 边函数的非线性形式，而不仅是当前点的一阶导数；
2. 同时建立短程、中程和长程路径，避免长链乘积消失或爆炸；
3. 使用可学习门控决定不同因果路径和不同路由步长的权重；
4. 由动作提议头给出初始序列，再通过固定层数的因果消息传播完成修正；
5. 将整个过程做成决策网络的一次前向传播，而不是部署时进行梯度下降。

这一结果支持继续实现“ProtoKAN 非线性时序因果路由头”。

## 文件

- `kanrf/temporal_causal_routing.py`
- `kanrf/temporal_causal_routing_fixed.py`
- `scripts/stage62_temporal_causal_route_validation.py`
- `scripts/stage62b_temporal_causal_route_validation.py`
- `results/stage62_temporal_causal_route_validation.json`
- `results/stage62_temporal_causal_route_smallstep.json`
