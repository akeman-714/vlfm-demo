# Habitat 能力地图

这组文档只讨论 Habitat 体系内部的能力边界：Habitat-Sim、Habitat-Lab、Habitat-Baselines 和它们通过 config / registry 暴露出来的扩展口。

不讨论具体项目里的 VLM、真机、第三方探索包、持久化记忆、外部检测器或业务 demo。那些可以使用 Habitat 的接口，但不是 Habitat 自己的能力边界。

每个能力点都回答四件事：

- Habitat 里面这个边界负责什么。
- 这个边界下有哪些常见选项。
- 这些选项的数据形态大概是什么。
- 非代码用例是什么，使用后会产生什么效果。

## 能力总览

| 文档 | Habitat 边界 | 负责什么 |
| --- | --- | --- |
| [01_simulator.md](01_simulator.md) | Simulator | 场景加载、agent 身体、相机渲染、深度/语义输出、碰撞、navmesh、路径查询 |
| [02_task.md](02_task.md) | Task | 定义任务类型、目标语义、成功条件、episode 进行逻辑 |
| [03_dataset_episode.md](03_dataset_episode.md) | Dataset / Episode | 定义场景、起点、目标、split、episode 元数据和难度分布 |
| [04_sensor.md](04_sensor.md) | Sensor | 定义 observation 里有哪些信息，例如 RGB、Depth、GPS、Compass、ObjectGoal |
| [05_action.md](05_action.md) | Action | 定义 agent 可以执行的动作，例如 STOP、前进、转向、抬头、抓取 |
| [06_measurement.md](06_measurement.md) | Measurement | 定义评估和诊断指标，例如 Success、SPL、DistanceToGoal、Collisions |
| [07_policy_baselines.md](07_policy_baselines.md) | Policy / Baselines | 定义 Habitat-Baselines 中训练、评估和策略组件的边界 |
| [08_obs_transform.md](08_obs_transform.md) | ObsTransform | 定义 observation 进入 policy 前如何 resize、crop、normalize、变换 shape |
| [09_config_hydra.md](09_config_hydra.md) | Config / Hydra | 定义如何把 simulator、task、dataset、sensor、action、measurement、policy 组合成实验 |

## 快速判断

| 你要改什么 | 优先看 | 原因 |
| --- | --- | --- |
| 相机分辨率、FOV、高度、深度范围 | Simulator | 这是仿真器如何生成原始观测 |
| RGB、Depth、GPS、Compass、ObjectGoal 是否进入 observation | Sensor | 这是 agent 能读到什么 |
| STOP、MOVE_FORWARD、TURN_LEFT 等动作是否存在 | Action | 这是 agent 能输出什么 |
| 从 PointNav 改成 ObjectNav 或 ImageNav | Task | 任务规则和目标类型变了 |
| 换场景、起点、目标类别、split | Dataset / Episode | 样本分布变了 |
| 成功率、SPL、碰撞、到目标距离怎么记录 | Measurement | 评估和诊断方式变了 |
| 用 PPO、DDPPO、PointNav policy 训练或评估 | Policy / Baselines | 决策模型和训练循环变了 |
| 把 `(H, W, 3)` 图像变成模型需要的尺寸 | ObsTransform | 信息没变，输入形态变了 |
| 用配置组合这些组件做 ablation | Config / Hydra | 组件存在，只是接线方式变了 |

## 数据形态约定

文档里的 shape 都写成通用形式：

| 表达 | 含义 |
| --- | --- |
| `(H, W, 3)` | 高为 H、宽为 W、3 通道图像，常见于 RGB |
| `(H, W, 1)` 或 `(H, W)` | 单通道矩阵，常见于 depth 或 semantic |
| `(2,)` | 二维向量，例如平面位置、距离和角度 |
| `(3,)` | 三维向量，例如 3D 位置 |
| scalar | 单个数，例如类别 id、角度、距离、成功标志 |
| dict / structured object | 结构化字段，例如 episode、measurement info |

实际 `H`、`W`、坐标维度、dtype 会由 Habitat 版本、任务配置、sensor 配置和 wrapper 决定。这里强调能力边界和常见形态，不替代具体运行时配置。

## 重要边界

Simulator 是世界和原始仿真信号。

Sensor 是 observation 入口。

Action 是可执行动作集合。

Task 是任务规则。

Dataset / Episode 是题目分布。

Measurement 是评估侧信息。

Policy / Baselines 是决策和训练评估框架。

ObsTransform 是输入预处理。

Config / Hydra 是组合方式。

如果一个想法跨多个边界，应该拆开看。例如“找椅子”是 Task / Dataset；“看 RGB 图像”是 Sensor / Simulator；“成功率怎么算”是 Measurement；“怎么决定下一步走哪里”是 Policy。

