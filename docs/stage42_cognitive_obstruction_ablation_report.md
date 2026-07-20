# Stage 42：认知参数是否阻碍决策网络

## 实验目的

保持 PPO、奖励、采样、rollout horizon、actor/critic 规模和训练轮数一致，只替换嵌合 actor 中的认知模块：

* `direct`：普通状态—目标 actor；
* `identity_receiver`：与嵌合 actor 结构相同，但把认知输出替换为当前状态；
* `random_proto`：随机初始化、未训练的 ProtoKAN；
* `trained_proto`：在源物理环境上预训练的 ProtoKAN。

三个随机种子在所有条件之间严格匹配。

## 结果

| 条件 | 平均源环境成功率 | 标准差 | 平均最大高度 |
|---|---:|---:|---:|
| direct | 59.4% | 5.6% | 1.129 |
| identity_receiver | 61.5% | 3.7% | 1.132 |
| random_proto | 60.4% | 5.2% | 1.133 |
| trained_proto | 59.9% | 5.2% | 1.124 |

把训练完成的 `trained_proto` actor 中的 ProtoKAN 参数替换成随机 ProtoKAN 后，平均成功率仍约为 59.9%，与原值几乎一致。

## 前向敏感性检查

为了确认“结果相近”不是因为认知模块完全没有参与，进一步对同一个训练好的 actor 做认知参数替换：

* 认知预测状态平均变化：0.609；
* 最终动作平均绝对变化：0.075；
* 最终动作最大变化：0.218；
* 原认知和随机认知下动作的状态方差分别为 0.0745 和 0.0285。

因此，认知参数确实影响了每次前向动作，但在当前随机初始状态评估中，这种影响还没有转化为成功率差异。

## 当前判断

本实验不能支持“认知参数是主要阻碍”这一结论。更准确的结论是：

1. 预训练 ProtoKAN 没有显著优于 direct 或 identity receiver；
2. 预训练 ProtoKAN 也没有显著低于这些对照，差异在当前方差范围内；
3. 认知参数确实参与了前向，但当前决策训练没有充分利用其信息；
4. 随机初始状态下的成功率约 60%，不足以检测长时域控制能力。

固定悬垂状态的 Stage 41 实验中所有 actor 成功率均为 0%，说明当前更大的瓶颈仍然是长时域探索、训练时域和任务奖励，而不是单纯的认知参数干扰。

## 重要限制

当前消融使用的是随机初始状态。由于部分状态天然接近目标，成功率差异会被初始状态分布掩盖。下一步必须在固定悬垂状态或受控初始分布上，先让 direct PPO 学会源环境摆起，再重复完全相同的认知消融。

实验脚本：`scripts/stage42_cognitive_obstruction_ablation_v2.py`；前向敏感性脚本：`scripts/stage42_cognitive_sensitivity_probe.py`；结果：`results/stage42_cognitive_obstruction_ablation_seedmatched.json`。
