# Stage 47：认知输出—决策头接口消融

## 动机

原始嵌合 actor 只把 `predicted_next` 和 goal 送入决策头，导致 PPO 优化较慢、参数变化后的零样本表现弱。我们比较了三种接口，均使用相同的 30 轮 PPO、3 个随机种子和难参数变化 B `(13.475, 0.06, 0.90, 1.10)`。

## 三种接口

1. **原始接口**：`[predicted_next, goal]`。
2. **直接 delta 接口**：`[predicted_next, predicted_next-state, query, goal]`。它提升明显，但 query 可以绕过认知预测直接影响决策，因此只能作为诊断上限。
3. **FiLM 门控接口**：query 经过线性调制，只产生 `scale,bias` 作用于 `predicted_next`；决策头只接收 `[modulated_prediction, modulated_prediction-state, goal]`。没有 query 到动作的独立直连分支，认知预测仍是必经路径。

## 30 轮结果

| 接口 | 源环境均值 | 难参数变化 B 均值 |
|---|---:|---:|
| 普通 PPO（无认知） | 77.6% | 5.2% |
| 原始 ProtoKAN 嵌合 | 68.2% | 6.3% |
| 直接 delta + query | 78.6% | 13.5% |
| **FiLM 门控（无 query 直连）** | **82.8%** | **13.5%** |

FiLM 三种子源环境为 84.4%、82.8%、81.3%，变化环境为 9.4%、12.5%、18.8%。标准差分别约 1.6 和 4.8 个百分点。

## 解释

这说明瓶颈主要在“如何把认知预测表示变成可控的决策特征”，而不只是 ProtoKAN 单步预测精度。FiLM 让决策 query 只能调制认知表示，保留了认知必经约束，同时提供了任务所需的动作条件适配。它比直接把 query 拼到决策头更符合最初的结构要求。

## 限制与下一步

- 目前仍是单个 Acrobot 任务、30 轮训练；需要更长训练和更多环境。
- 需要把 FiLM actor 接入 Stage 46 的在线认知更新，确认物理参数变化后的恢复是否也提升。
- 还应加入认知冻结/随机认知/无 FiLM 等消融，检验收益是否确实来自物理表示而非参数量增加。

脚本：`scripts/stage47c_ppo_cognitive_film_actor_fixed.py`；结果：`results/stage47c_film_secondheldout_seed*_30iter.json`。
