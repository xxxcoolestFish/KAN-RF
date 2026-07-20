# Stage 50：动作敏感性表示验证

## 假设

决策网络可能缺少“动作会怎样改变系统”的信息，因此在 FiLM 接口中加入认知网络对内部 query 的有限差分敏感性：

\[
J_q \approx \frac{f(s,q+\epsilon)-f(s,q-\epsilon)}{2\epsilon}.
\]

决策头不直接接收 query，只接收 FiLM 调制后的预测状态、状态变化量、`J_q` 和目标。

## 结果

| 接口 | 源环境均值 | 难参数变化 B 均值 |
|---|---:|---:|
| FiLM（无 Jacobian） | 82.8% | 13.5% |
| FiLM + query Jacobian | 64.6% | 3.1% |

Jacobian 三个种子的源环境成功率为 70.3%、71.9%、51.6%；难参数变化成功率为 6.3%、3.1%、0%。

## 解释

这次实验没有支持“直接加入动作敏感性就能提升泛化”的假设，原因可能是：

1. query 是决策网络的内部动作查询，并不等同于真实环境动作；
2. ProtoKAN 的 query 导数没有经过尺度归一化，容易给 PPO 带来噪声；
3. 单步局部导数不能代表长时域动作效果；
4. 当前认知网络还没有显式的动态上下文，Jacobian 仍然是源环境局部函数的导数。

因此，动作效果表示不能简单通过拼接一个 Jacobian 解决。下一步应从真实历史转移中推断隐式动态上下文，再让认知预测和 FiLM 调制共同依赖该上下文。

脚本：`scripts/stage50_ppo_cognitive_jacobian_actor_fixed.py`；结果：`results/stage50_jacobian_secondheldout_seed*_30iter.json`。
