# Task

Task 是 Habitat-Lab 里定义“agent 被要求完成什么”的层。它规定目标是什么、哪些 sensor 属于这个任务、成功条件是什么、episode 如何推进、奖励如何计算。

一句话：Task 决定“题目规则”，不是“场景资产”也不是“policy 解法”。

## 常见任务类型

| Task | 目标数据形态 | 常见 observation | 非代码用例 | 效果 |
| --- | --- | --- | --- | --- |
| PointNav | 目标点，常见为 `(2,)` 距离+角度或 `(3,)` 位置 | Depth、PointGoal、GPS、Compass | 让 agent 从起点走到指定坐标 | 测试几何导航和局部避障，不测试语义识别 |
| ObjectNav | 目标类别 id，通常是 scalar int | RGB、Depth、ObjectGoal、GPS、Compass | 让 agent 找到 chair 或 bed | 测试语义目标搜索和停止判断 |
| ImageNav | 目标图像，常见为 `(H, W, 3)` 或图像特征 | RGB、Depth、ImageGoal | 让 agent 找到目标图片对应的位置 | 测试视觉匹配和导航 |
| VLN / Instruction Navigation | 指令字符串或 token 序列 | RGB、Depth、Instruction | 按“穿过走廊，在厨房左转”这类指令走 | 测试语言理解和路线执行 |
| EQA | 问题文本 + 环境观测 | RGB、Depth、Question / Instruction | 先导航再回答“桌上有什么” | 测试主动感知和问答 |
| Rearrange | 物体状态、目标状态、机器人关节状态 | RGB、Depth、joint、gripper、object state sensors | 把物体从 A 放到 B | 测试导航、操作和物理交互 |

不同 Habitat 版本和任务包会包含不同任务。能力边界上，它们都属于 Task：定义目标、过程和成功条件。

## Task 目标形式

| 目标形式 | 数据形态 | 非代码用例 | 效果 |
| --- | --- | --- | --- |
| 点目标 | `(2,)` 或 `(3,)` 浮点向量，或距离+角度 | PointNav 去 5 米外的目标点 | agent 不需要识别物体，只需要走到点 |
| 类别目标 | scalar 类别 id，或类别名映射 | ObjectNav 找 sofa | agent 知道目标类别，但不直接知道目标位置 |
| 图像目标 | `(H, W, 3)` 图像或 embedding | ImageNav 找到和照片相同的位置 | agent 依赖视觉相似性 |
| 语言目标 | string 或 token 序列 | VLN 按自然语言路线走 | agent 需要解析语言指令 |
| 物体状态目标 | structured object state | Rearrange 中杯子要在桌上 | 成功不只是到达，还要改变世界状态 |
| 问答目标 | question string + answer space | EQA 中回答房间里物体 | 任务结果可能是文本或类别答案 |

## 成功条件选项

| 成功条件 | 数据形态 | 非代码用例 | 效果 |
| --- | --- | --- | --- |
| 距离阈值 | scalar 距离，例如 success distance | PointNav 到目标点 0.2 米内算成功 | 简单稳定，适合几何导航 |
| 主动 STOP | 离散 action id | ObjectNav 中 agent 必须自己停 | 测试“知道何时结束”的能力 |
| 可见性条件 | bool 或可见像素比例 | 只有看见目标才算找到 | 避免只靠近但没观察到目标 |
| 交互状态条件 | bool / structured state | 物体已经被放到目标区域 | Rearrange 中判断操作是否完成 |
| 答案正确 | bool 或分类结果 | EQA 回答正确才成功 | 成功不只取决于位置 |
| 步数限制 | scalar step count | 500 步内完成任务 | 控制 episode 长度和难度 |
| 复合条件 | dict / 多个 bool | 到达、看见、STOP 同时满足 | 更贴近严格任务，但更难 |

## 奖励和 episode 进行逻辑

| 选项 | 数据形态 | 非代码用例 | 效果 |
| --- | --- | --- | --- |
| success reward | scalar | 成功时给正奖励 | 强化学习更重视完成任务 |
| slack reward | scalar，通常每步一个小负值 | 每走一步扣一点 | 鼓励更短路径 |
| distance reward | scalar，基于距离变化 | 靠近目标给奖励 | 训练更容易，但依赖距离真值 |
| collision penalty | scalar | 撞墙扣分 | 鼓励安全导航 |
| max episode steps | scalar int | 超过步数自动结束 | 防止 episode 无限运行 |
| task stage | enum / string / int | Rearrange 中先导航再抓取再放置 | 支持多阶段任务 |
| episode reset | structured state | 新 episode 重置 agent 和任务状态 | 保证每集独立开始 |

奖励主要服务训练。评估时，成功条件和 measurement 更关键。

## Task Sensor 绑定

| Task | 常见 task sensor | 数据形态 | 效果 |
| --- | --- | --- | --- |
| PointNav | PointGoal / PointGoalWithGPSCompass | `(2,)` 或 `(3,)` | 告诉 agent 目标相对方向和距离 |
| ObjectNav | ObjectGoal | scalar int | 告诉 agent 当前要找哪类物体 |
| ImageNav | ImageGoal | `(H, W, 3)` 或特征向量 | 给出目标视图 |
| VLN | Instruction | string 或 token ids | 给出语言路线 |
| EQA | Question | string 或 token ids | 给出要回答的问题 |
| Rearrange | object state / robot state sensors | vectors / dict | 给出物体、关节、抓取状态 |

边界：Task 决定哪些 task sensor 有意义；Sensor 决定它们如何进入 observation。

## 非代码用例库

| 需求 | Task 选择 | 数据形态重点 | 效果 |
| --- | --- | --- | --- |
| 只想测能不能走到坐标 | PointNav | PointGoal `(2,)`，Depth `(H, W, 1)` | 屏蔽物体识别，只看导航 |
| 想测找目标类别 | ObjectNav | ObjectGoal scalar，RGB `(H, W, 3)` | agent 需要搜索和识别类别 |
| 想测按图找位置 | ImageNav | ImageGoal 图像 | agent 依赖图像匹配 |
| 想测语言路线跟随 | VLN | Instruction token 序列 | agent 需要理解“先后左右”等语言 |
| 想测问答 | EQA | Question + RGB-D | agent 要移动收集证据再回答 |
| 想测抓放物体 | Rearrange | 物体状态、关节状态、gripper 状态 | agent 要完成物理交互 |
| 想让 ObjectNav 更严格 | ObjectNav + 可见性 + STOP | STOP action + visible bool | 减少“靠近但没看到”的成功 |
| 想让训练更快收敛 | 加距离奖励或 shaping reward | distance scalar | policy 更容易学习，但评估要仍看标准指标 |

## Task 的边界

Task 可以定义：

- 目标类型。
- 任务 sensor。
- 成功条件。
- 奖励。
- episode reset / step 逻辑。
- 多阶段任务状态。

Task 不应该直接定义：

- 3D mesh 如何渲染，那是 Simulator。
- episode 起点和目标分布，那是 Dataset / Episode。
- policy 如何决策，那是 Policy。
- 评估表里额外统计什么，那是 Measurement。

## 什么时候改 Task

优先改 Task，当你要改变：

- PointNav / ObjectNav / ImageNav / Rearrange 这类任务类型。
- 目标从坐标变成类别、图像、语言或物体状态。
- 成功条件从“靠近”变成“靠近且 STOP”或“看见且 STOP”。
- 奖励 shaping。
- episode 内的阶段逻辑。

