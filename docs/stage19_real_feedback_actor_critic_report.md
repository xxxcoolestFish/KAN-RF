# Stage 19：无教师实时 Actor–Critic 学习

## 1. 实验目的

本实验不在实时决策阶段使用 MPC 教师或动作标签。认知网络只负责预训练和一次性生成固定算子 `q_const`；部署后决策网络仅从真实环境转移

```text
(state, operator, action, reward, next_state, next_operator, done)
```

中学习。

## 2. 方法

- Actor：当前决策网络，先测试全参数可训练版本。
- Critic：两个独立的状态—算子—动作价值网络，使用较小值构造 TD3 风格目标。
- Critic 损失：

  \[
  y_t=r_t+\gamma(1-d_t)\min_j Q_{\bar\psi_j}(s_{t+1},q_{t+1},a_{t+1}),
  \]

  \[
  L_Q=\operatorname{Huber}(Q_1-y_t)+\operatorname{Huber}(Q_2-y_t).
  \]

- Actor 损失：

  \[
  L_\pi=-Q_1(s_t,q_t,\pi_\theta(s_t,q_t)).
  \]

- 成功判据与训练目标统一：尖端高度首次达到 `1.0` 时记录终端成功奖励并结束该 episode。
- 训练阶段使用动作噪声探索，评估阶段完全关闭探索。
- 固定 20 个初始状态用于学习前后比较。

## 3. 结果

环境为 `(7.35, 0, 0.8, 0.8)`，固定测试集为 `test_seed=20260718`。

| 训练量 | 学习前 | 在线训练成功率 | 学习后固定集 |
|---:|---:|---:|---:|
| 100 episodes | 14/20 = 70% | 59% | 15/20 = 75% |
| 250 episodes | 14/20 = 70% | 60.8% | 14/20 = 70% |

因此，真实反馈 Actor–Critic 确实在短训练中带来过小幅提升，但扩大训练量后没有继续提升到 20/20，训练过程也出现了价值估计不稳定：后期 Critic 损失和 Actor 损失明显上升，策略性能回落。

## 4. 结论

1. **不需要教师这一点是正确的。** 实时阶段可以完全使用真实转移和 TD 学习。
2. **当前失败不是因为没有教师，而是 Critic 还没有学出稳定的长时域价值函数。** 早期成功奖励很稀少，Actor 得到的价值梯度噪声较大。
3. **直接放开全参数也没有达到 20/20。** 这说明仅替换损失形式仍不够，必须进一步处理信用分配、成功样本覆盖和价值过估计。
4. 当前 `q_const` 仍是一次性固定算子；这符合本次决策单独实验，但不是完整的在线认知—决策系统。

## 5. 下一步

下一步不再引入教师，而是改进真实反馈学习本身：

1. 对成功轨迹进行优先回放，保证 Critic 经常看到终端成功转移；
2. 使用 n-step return 或 Retrace，缩短成功奖励传回早期动作的路径；
3. 对 Critic 目标和奖励做归一化，并加入保守的 Actor trust-region，避免后期价值过估计导致策略漂移；
4. 先在固定初始状态集合上验证收敛，再恢复随机初始状态和认知算子实时更新。

## 6. 复现

```powershell
& 'C:\Users\32510\miniconda3\Scripts\conda.exe' run --no-capture-output -n dl_env python -m scripts.stage19_actor_critic_v2 --actor-scope full --episodes 100 --seed 42
& 'C:\Users\32510\miniconda3\Scripts\conda.exe' run --no-capture-output -n dl_env python -m scripts.stage19_actor_critic_v2 --actor-scope full --episodes 250 --seed 42
```

