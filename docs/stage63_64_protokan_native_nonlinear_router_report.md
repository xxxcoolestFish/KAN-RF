# Stage 63–64：ProtoKAN 原生非线性因果路由

## 目标

将 Stage 62 的通用 Jacobian 路由升级为 ProtoKAN 原生函数边路由，并验证：

1. ProtoKAN 每条函数边能否被精确提取；
2. 全部认知参数是否参与边函数和路由；
3. 层内与时间维度的图拼接是否准确；
4. 非线性边响应是否能改进有限幅度动作效果估计；
5. 路由器只在源认知网络上训练后，能否直接适配更新后的目标认知参数。

## ProtoKAN 原生函数边

对每条 ProtoKAN 边提取：

- 当前函数值；
- 精确一阶导数；
- 正向有限扰动响应；
- 负向有限扰动响应；
- 局部曲率。

这些量均直接由原型位置、原型值、原型导数、核宽度和 SiLU 基路径计算，而不是读取一个压缩后的参数向量。

## Stage 63：精确性验证

认知网络在源环境进行 300 步单步预测训练，最终预测损失为 `0.00439`。

| 验证项 | 结果 |
|---|---:|
| 原生边求和与 ProtoKAN 前向最大误差 | 2.38e-7 |
| 一步原生边路由最大误差 | 3.58e-7 |
| 一步路由余弦相似度 | 1.000000 |
| 8 步时序最终状态最大误差 | 1.67e-6 |
| 8 步动作路由最大误差 | 2.09e-7 |
| 8 步动作路由平均余弦相似度 | 1.000000 |

以下全部认知参数均获得了非零梯度：

- `proto_pos`
- `proto_val`
- `proto_der`
- `log_sigma`
- `base_weight`

因此，原生因果路由不是局部探针，也不是部分参数摘要，而是完整 ProtoKAN 函数结构的另一种决策侧表达。

## 非线性边信息

两层 ProtoKAN 的平均绝对曲率分别约为：

- 第一层：`0.0757`
- 第二层：`0.1026`

有限扰动响应与线性导数近似存在稳定非零差异，证明边函数中确实包含 Jacobian 没有保留的非线性信息。

## Stage 64：稳定非线性残差路由

初版非线性路由将所有边残差直接求和，导致一次更新就放大并破坏原本正确的线性方向。正式版本改为：

\[
\text{route}
=
\text{exact linear route}
+
\alpha\,\operatorname{mean}_{edges}(\text{nonlinear correction}),
\]

其中非线性残差零初始化、按层宽归一化并限制幅度，保证训练从精确线性路由开始渐进修正。

## 结果

路由目标是认知模型内部、动作变化幅度为 `0.25` 时的真实 8 步有限动作效果。

| 认知状态 | 线性路由 MSE | 非线性路由 MSE | 相对降低 |
|---|---:|---:|---:|
| 源认知网络 | 6.56e-7 | 3.50e-7 | 46.7% |
| 目标更新前的独立批次 | 5.77e-7 | 2.58e-7 | 55.3% |
| 认知参数更新到目标环境后 | 2.44e-6 | 1.47e-6 | 39.6% |

目标环境中只更新了认知网络，路由器没有继续训练。更新后非线性路由平均余弦相似度仍为 `0.99856`。

## 结论

这一阶段验证了一个重要机制：

\[
\boxed{
\text{ProtoKAN 参数变化}
\rightarrow
\text{函数边变化}
\rightarrow
\text{时序因果路径变化}
\rightarrow
\text{动作影响变化}
}
\]

决策侧不需要理解原始参数坐标，也不需要在源训练中见过目标物理参数。它读取的是具有稳定数学含义的边函数值、导数、正负响应和曲率。

当前结果验证的是认知模型内部的路由准确性与参数更新后的接口稳定性，尚未直接证明环境控制成功率。下一阶段需要把它接入：

1. 动作序列提议头；
2. 短程、中程、长程路由分支；
3. 可学习动作解码器；
4. PPO 或其他任务损失训练；
5. 最终只执行路由后动作序列的第一个动作。

## 文件

- `kanrf/protokan_causal_router.py`
- `kanrf/protokan_causal_router_stable.py`
- `scripts/stage63_protokan_native_causal_validation.py`
- `scripts/stage64_nonlinear_causal_router_transfer.py`
- `scripts/stage64b_nonlinear_causal_router_transfer.py`
- `scripts/stage64c_stable_nonlinear_causal_router_transfer.py`
- `results/stage63_protokan_native_causal_validation.json`
- `results/stage64_stable_nonlinear_causal_router_transfer.json`
