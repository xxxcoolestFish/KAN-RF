# Stage 41：ProtoKAN 嵌合 Actor-Critic 初步验证

## 实验目的

验证以下新决策框架是否能够替代原来的“对动作做内层梯度优化”机制：

\[
\text{state, goal}
\rightarrow q_t
\rightarrow f_\phi(s_t,q_t)
\rightarrow \text{decision head}
\rightarrow a_t.
\]

其中 \(f_\phi\) 是完整的 ProtoKAN 认知网络，\(q_t\) 是 actor 内部产生的候选动作，认知网络参数在每个动作的前向计算中都会被使用。PPO 只更新 actor 的新增决策参数和 critic，认知网络仍然只使用下一状态预测损失更新。

## 实现

* `DirectGaussianActor`：普通目标条件 actor 基线；
* `CognitiveEmbeddedGaussianActor`：候选动作先经过完整认知 ProtoKAN，再由决策头输出最终动作；
* `ValueCritic`：目标条件状态价值函数；
* PPO：从真实物理转移收集回报，使用 clipped policy loss 和 GAE；
* 认知参数在 PPO 更新中冻结，确保认知损失和决策损失分离。

## 第一组结果：随机初始状态

配置：64 个并行状态、64 步 rollout、60 次 PPO 更新、64 个测试状态。

| actor | 源环境 | held-out 参数 |
|---|---:|---:|
| Direct PPO | 54.7% | 59.4% |
| ProtoKAN Embedded PPO | 59.4% | 54.7% |

嵌合 actor 的源环境结果略高于 direct actor，但差异未经过多随机种子验证，不能称为性能优势。随机初始状态本身会让部分状态接近目标，因此这一测试不能证明完整 swing-up 能力。

## 第二组结果：悬垂初始状态

为了避免随机初始状态造成的成功率虚高，新增了从悬垂状态附近开始的测试。初始状态约为：

\[
s_0=(1,0,1,0,0,0).
\]

在 30 次 PPO 更新、128 步 rollout 下：

| actor | 源环境成功率 | 平均最大高度 |
|---|---:|---:|
| Direct PPO | 0% | 约 -1.95 |
| ProtoKAN Embedded PPO | 0% | 约 -1.92 |

这说明当前 PPO 探索和奖励设置尚未发现 Acrobot 的长时域摆起动作序列。它不能说明嵌合结构错误，但说明“换成标准 actor-critic”并不会自动解决长时域探索问题。

## 当前结论

1. 决策框架已经成功切换到真正的策略—价值学习形式；
2. ProtoKAN 参数确实参与了 actor 的前向传播，而不是只用于初始化；
3. 认知损失和 PPO 决策损失已经分离；
4. 源环境随机状态下暂时没有稳定性能优势；
5. 固定悬垂状态下两种 actor 都没有学会摆起，当前主要瓶颈是长时域探索、奖励设计和训练时域，而不是 ProtoKAN 是否接入。

因此，下一步不应立即比较物理参数泛化，而应先让普通 PPO actor 在正确的初始任务定义下完成源环境 swing-up，再将 ProtoKAN 嵌合 actor 接入同一套训练协议。否则无法判断失败来自强化学习训练、奖励、任务时域，还是认知参数耦合。

## 文件

* 实现：`scripts/stage41_ppo_cognitive_actor.py`；
* 固定悬垂初始状态验证：`scripts/stage41_ppo_cognitive_actor_v2.py`；
* 绝对高度奖励对照：`scripts/stage41_ppo_cognitive_actor_v3.py`；
* 随机初始状态结果：`results/stage41_ppo_source.json`；
* 固定初始状态结果：`results/stage41_ppo_fixed_quick.json`；
* 高度奖励结果：`results/stage41_ppo_height_quick.json`。
