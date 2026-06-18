# Policy / Baselines

Policy / Baselines 指 Habitat-Baselines 中的决策模型、训练器、评估循环和实验基础设施。它接收 observation，输出 action，并在 trainer 中完成 rollout、训练、评估、保存 checkpoint 和记录指标。

一句话：Policy 决定“下一步做什么”，Baselines 提供“训练和评估这个 policy 的框架”。

## Policy 输入输出数据形态

| 数据 | 常见形态 | 非代码用例 | 效果 |
| --- | --- | --- | --- |
| observations | dict / TensorDict，key 到 tensor | `rgb`、`depth`、`pointgoal` 同时输入 policy | policy 获得当前环境信息 |
| RGB observation | batch 后常见为 `(B, H, W, 3)` 或模型内部转为 `(B, 3, H, W)` | 视觉导航 policy 看图 | 提供外观信息 |
| Depth observation | `(B, H, W, 1)` 或 `(B, 1, H, W)` | PointNav policy 用深度避障 | 提供几何信息 |
| goal vector | `(B, 2)` 或 `(B, 3)` | PointNav 输入目标距离和角度 | 告诉 policy 目标方向 |
| recurrent hidden state | `(num_layers, B, hidden)` 或等价结构 | LSTM policy 记住之前看过的信息 | 支持部分可观测环境 |
| previous action | `(B, 1)` 或 `(B,)` | policy 知道上一动作 | 帮助时序决策 |
| masks | `(B, 1)` bool / float | episode reset 时清空 hidden state | 防止跨 episode 记忆污染 |
| output action | 离散 scalar id 或连续 vector | 输出 TURN_LEFT 或速度向量 | 驱动环境 step |
| policy info | dict | 记录 value、log prob、可视化信息 | 训练和评估时辅助记录 |

实际 shape 会随 batch、环境数量、policy 实现和 observation transform 改变。

## 常见 Policy 类型

| Policy 类型 | 输入数据形态 | 输出数据形态 | 非代码用例 | 效果 |
| --- | --- | --- | --- | --- |
| Random policy | observation 可忽略 | action id / vector | 检查环境能否跑通 | 作为最低 baseline |
| Rule-based policy | observation dict | action id / vector | 根据简单规则转向或前进 | 可解释，但能力有限 |
| PointNav policy | depth + pointgoal | 离散动作或连续动作 | 走到给定目标点 | 测几何导航 |
| CNN policy | image tensor | action distribution | 从 RGB / Depth 学动作 | 端到端视觉导航 |
| Recurrent policy | observation + hidden state | action + new hidden state | 记住过去观察 | 适合部分可观测任务 |
| Actor-Critic policy | observation | action、value、log prob | PPO / DDPPO 训练 | 同时学习策略和价值函数 |
| Continuous-control policy | observation | float action vector | 速度控制或机械臂控制 | 更细粒度动作 |

Habitat-Baselines 的具体 policy 名称和可用实现取决于安装版本与注册项。

## Trainer 选项

| Trainer / 机制 | 数据形态 | 非代码用例 | 效果 |
| --- | --- | --- | --- |
| PPO trainer | rollout batch + advantage + returns | 单机强化学习训练导航 policy | 更新 actor-critic 参数 |
| DDPPO trainer | 多进程 / 多 GPU rollout | 大规模训练 PointNav policy | 提高采样吞吐 |
| evaluation loop | observations、actions、infos | 加载 checkpoint 在 val split 上跑 | 得到 Success、SPL、视频 |
| rollout storage | time × env × observation tensors | 存一段交互轨迹 | 用于 PPO 更新 |
| vectorized environments | batch of envs，B 个并行环境 | 同时跑多个 episode | 提高训练 / 评估速度 |
| checkpoint manager | model weights + optimizer / config state | 保存和恢复训练 | 支持复现和继续训练 |
| TensorBoard logging | scalar time series | 记录 reward、success、loss | 观察训练趋势 |
| video generation | frame list / images | 保存评估 episode 视频 | 调试行为 |

## PPO / DDPPO 相关数据

| 数据 | 常见形态 | 用例 | 效果 |
| --- | --- | --- | --- |
| reward | `(T, B, 1)` scalar 序列 | 每步环境奖励 | 训练目标来源 |
| value prediction | `(T, B, 1)` | critic 估计状态价值 | 计算 advantage |
| action log prob | `(T, B, 1)` | PPO ratio 计算 | 控制策略更新幅度 |
| advantage | `(T, B, 1)` | 策略梯度权重 | 判断哪些动作比预期好 |
| return | `(T, B, 1)` | 累计折扣回报 | 训练 value function |
| hidden states | time/env/layer 维度张量 | recurrent policy 训练 | 保留时序记忆 |
| done masks | `(T, B, 1)` | episode 结束清 hidden | 防止跨 episode 混合 |

这些是训练框架内部数据形态。普通评估只需要 observation、action、info 和 metrics。

## Checkpoint 和评估

| 能力 | 数据形态 | 非代码用例 | 效果 |
| --- | --- | --- | --- |
| checkpoint path | string 路径 | 加载训练好的 PointNav policy | 复现实验 |
| model weights | tensor state dict | 保存神经网络参数 | policy 行为可恢复 |
| config snapshot | structured config | 记录训练时 sensor/action/policy 设置 | 避免 checkpoint 和当前配置不匹配 |
| eval split | string | 在 val split 上评估 | 得到可比较指标 |
| eval episode count | scalar int | 只跑 100 集快速看结果 | 控制评估成本 |
| deterministic action | bool | 评估时取最高概率动作 | 提高复现稳定性 |
| video option | list / flags | 保存视频到磁盘 | 可视化失败 |

## 非代码用例库

| 需求 | Habitat-Baselines 选择 | 数据形态重点 | 效果 |
| --- | --- | --- | --- |
| 训练 PointNav | PointNav policy + PPO / DDPPO | depth tensor + pointgoal vector | 学到走点能力 |
| 评估训练结果 | evaluation loop + checkpoint | checkpoint path + eval episodes | 得到 Success / SPL |
| 提高训练吞吐 | vectorized envs + DDPPO | batch size 增大 | 更快收集 rollout |
| 做视觉输入消融 | 换 observation space 或 obs transforms | RGB / Depth tensor 变化 | 看输入对 policy 的影响 |
| 调试 episode 行为 | 开 video generation | frame list / TopDownMap | 看 agent 每步怎么走 |
| 训练 recurrent policy | recurrent hidden state + masks | hidden tensor + done masks | policy 能利用历史 |
| 训练连续动作 policy | continuous action space | action vector | 支持速度或机械臂控制 |
| 复现实验 | 保存 checkpoint + config | weights + config | 能重新加载同一策略 |

## Policy 与其他边界的关系

| 边界 | Policy 依赖它什么 |
| --- | --- |
| Simulator | 环境 step 后产生下一状态和 sim sensor |
| Sensor | observation key 和数据 shape |
| Action | policy 输出必须匹配 action space |
| Task | 决定目标、奖励、终止和成功语义 |
| Dataset | 决定训练 / 评估 episode 分布 |
| Measurement | 决定评估指标和 info |
| ObsTransform | 决定 policy 实际看到的输入 shape |
| Config | 决定使用哪个 policy、trainer、checkpoint 和参数 |

## 边界

Policy / Baselines 负责：

- 决策模型。
- 训练循环。
- 评估循环。
- rollout 数据。
- checkpoint。
- 日志和视频基础设施。

它不负责：

- 定义任务规则，那是 Task。
- 生成 episode，那是 Dataset。
- 渲染图像，那是 Simulator。
- 新增 observation，那是 Sensor。
- 新增动作语义，那是 Action。

## 什么时候改 Policy / Baselines

优先改 Policy / Baselines，当你要改变：

- 使用哪个 policy 架构。
- 是否使用 recurrent memory。
- 使用 PPO、DDPPO 或其他 trainer。
- checkpoint 加载和保存。
- 训练 rollout 参数。
- 评估循环、视频输出和日志。
- policy 输入和 action space 的适配逻辑。

