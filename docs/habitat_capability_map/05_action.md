# Action

Action 是 Habitat 中定义 agent 能对环境做什么的边界。它决定 action space 的形态，以及 simulator / task 收到动作后如何推进状态。

一句话：Action 是“输出接口”。

## Action Space 数据形态

| 类型 | 数据形态 | 非代码用例 | 效果 |
| --- | --- | --- | --- |
| 离散动作 | scalar int，表示动作 id | 0 是 STOP，1 是前进，2 是左转 | policy 每步从有限动作里选一个 |
| 连续动作 | `(N,)` float vector | 输出线速度和角速度 | policy 可以更细粒度控制 |
| 结构化动作 | dict / structured object | 同时控制底盘、手臂、夹爪 | 适合移动操作和 Rearrange |
| 多智能体动作 | dict，agent id 到 action | 多 agent 环境中每个 agent 一个动作 | 多个 agent 同时推进 |

具体 action space 由 Habitat config、task 和 simulator action configuration 决定。

## 常见导航离散动作

| Action | 数据形态 | 非代码用例 | 效果 |
| --- | --- | --- | --- |
| STOP | scalar int | ObjectNav 中 agent 认为已经找到目标，主动结束 | 如果满足成功条件，episode 成功；否则可能误停 |
| MOVE_FORWARD | scalar int | agent 向前走一步 | 位姿沿当前朝向移动，距离由 simulator 参数决定 |
| TURN_LEFT | scalar int | agent 左转 30 度或配置角度 | 改变朝向，不改变位置或只微小变化 |
| TURN_RIGHT | scalar int | agent 右转 30 度或配置角度 | 改变朝向 |
| LOOK_UP | scalar int | agent 抬头看高处 | 改变相机俯仰，适合找高处目标 |
| LOOK_DOWN | scalar int | agent 低头看地面或桌面 | 改变相机俯仰，适合看低处或近处 |

这些动作通常出现在导航任务中，但具体是否启用取决于 action space 配置。

## 连续导航动作

| Action 类型 | 数据形态 | 非代码用例 | 效果 |
| --- | --- | --- | --- |
| 线速度 / 角速度 | `(2,)` float，例如 linear、angular | agent 像差速底盘一样连续移动 | 比离散动作更平滑，更接近机器人控制 |
| 速度控制向量 | `(N,)` float | 同时控制前进、横移、旋转 | 支持更复杂运动模型 |
| 位姿增量 | `(N,)` float | 每步给一个相对运动增量 | 可模拟连续控制或简化运动学 |
| 相机控制量 | scalar 或 vector | 连续调整相机俯仰 / 云台角度 | 支持主动视觉 |

边界：连续动作让 action space 更细，但训练和评估更难。离散动作更容易 benchmark 和复现。

## Rearrange / 交互动作

| Action 类型 | 数据形态 | 非代码用例 | 效果 |
| --- | --- | --- | --- |
| Arm joint control | `(N,)` float | 控制机械臂各关节移动 | 改变手臂姿态 |
| End-effector control | `(3,)` 或更高维 pose / delta | 让夹爪靠近杯子 | 直接控制末端位置或姿态 |
| Grip / release | bool、scalar 或离散 id | 抓住或松开物体 | 改变是否持有物体 |
| Pick / place primitive | 离散 id 或 structured action | 用高层动作抓取 / 放置 | 简化低层控制 |
| Base + arm combined action | dict 或拼接 vector | 一边移动底盘一边调整手臂 | 支持移动操作 |
| Articulated object action | scalar / vector | 开门、开抽屉、推动可动部件 | 改变环境物体状态 |

这些动作通常属于 Rearrange 或交互任务，不是标准 PointNav / ObjectNav 必然启用。

## 动作物理参数

| 参数 | 数据形态 | 非代码用例 | 效果 |
| --- | --- | --- | --- |
| forward_step_size | scalar，单位通常是米 | MOVE_FORWARD 每次走 0.25 米 | 改变路径粒度和碰撞风险 |
| turn_angle | scalar，单位通常是度 | TURN_LEFT 每次转 30 度 | 改变朝向控制精度 |
| tilt_angle | scalar，单位通常是度 | LOOK_DOWN 每次低头 30 度 | 改变相机俯仰步长 |
| allow_sliding | bool | 撞墙后是否沿墙滑动 | 影响局部导航难度和真实感 |
| action repeat / control frequency | scalar | 一个动作执行多久 | 影响连续控制平滑度 |
| velocity limits | vector / scalar | 限制最大线速度和角速度 | 控制安全性和运动范围 |

这些参数通常在 Simulator / action configuration 中体现。Action 定义可选动作，Simulator 决定动作物理效果。

## 非代码用例库

| 需求 | Action 选择 | 数据形态 | 效果 |
| --- | --- | --- | --- |
| 标准 PointNav | MOVE_FORWARD、TURN_LEFT、TURN_RIGHT、STOP | scalar int | agent 学会走到点并停止 |
| 标准 ObjectNav | MOVE_FORWARD、TURN_LEFT、TURN_RIGHT、STOP | scalar int | agent 搜索目标并主动 STOP |
| 搜索高处目标 | 加 LOOK_UP | scalar int | agent 能主动改变视角看高处 |
| 搜索低处或地面 | 加 LOOK_DOWN | scalar int | agent 能看脚下、低矮目标或近处障碍 |
| 更接近连续机器人 | 使用 velocity action | `(2,)` 或更高维 float | 运动更平滑，但控制更难 |
| 做抓取放置 | base action + arm action + gripper action | structured action | agent 能移动并操作物体 |
| 做开门任务 | 导航动作 + articulated object action | discrete 或 vector | agent 能改变门 / 抽屉状态 |
| 做动作空间消融 | 比较 4 动作、6 动作、连续动作 | 不同 action space | 看动作能力对成功率和 SPL 的影响 |

## Action 与 Task 的关系

| Task | 常见 Action | 为什么 |
| --- | --- | --- |
| PointNav | 前进、转向、STOP | 走到目标点即可 |
| ObjectNav | 前进、转向、STOP，可选 LOOK_UP/DOWN | 需要搜索目标并停下 |
| ImageNav | 前进、转向、STOP | 找到目标图像对应位置 |
| VLN | 前进、转向、STOP，可选视角动作 | 按指令路线移动 |
| Rearrange | 底盘、手臂、夹爪、交互动作 | 需要改变物体状态 |

## 边界

Action 只定义 agent 能做什么。

它不定义：

- 什么时候做某个动作，那是 Policy。
- 动作是否算成功，那是 Task / Measurement。
- 动作的物理碰撞和位姿更新，那是 Simulator。
- 这集是否要求抓取或导航，那是 Task。

## 什么时候改 Action

优先改 Action，当你要改变：

- 离散动作集合。
- 是否允许 STOP。
- 是否允许 LOOK_UP / LOOK_DOWN。
- 前进步长、转角、俯仰角。
- 离散控制和连续控制的选择。
- Rearrange 中机械臂、夹爪、交互 primitive。

