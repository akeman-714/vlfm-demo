# Measurement

Measurement 是 Habitat-Lab 里定义评估和诊断信息的边界。它通常出现在 episode 的 info / metrics 中，用来回答“这一步或这一集表现如何”。

一句话：Measurement 是评估侧输出，不是 agent 的普通 observation。

## 常见导航 Measurement

| Measurement | 数据形态 | 非代码用例 | 效果 |
| --- | --- | --- | --- |
| Success | bool 或 0/1 scalar | ObjectNav 中是否成功找到并 STOP | 给出最核心的成功率 |
| SPL | scalar float，通常 0 到 1 | 成功且路径越短得分越高 | 衡量成功和路径效率 |
| SoftSPL | scalar float | 即使失败，也按接近目标程度给部分分 | 区分“差一点”和“完全没接近” |
| DistanceToGoal | scalar float | 当前离目标还有几米 | 分析失败是否接近目标 |
| NumSteps / EpisodeSteps | scalar int | 一集走了多少步 | 分析效率和是否超时 |
| EpisodeLength / PathLength | scalar float | agent 实际走过的距离 | SPL 和路径效率分析 |
| Collisions | scalar / dict | 记录碰撞次数或是否碰撞 | 分析安全性 |
| TopDownMap | 2D 矩阵 / 图像 / dict | 可视化俯视地图和轨迹 | 直观看 agent 走了哪里 |

具体名称和字段会随 Habitat 版本、task 和 config 略有差异。

## PointNav 常见指标用例

| 指标 | 数据形态 | 用例 | 效果 |
| --- | --- | --- | --- |
| DistanceToGoal | scalar float | 每步看离点目标是否变近 | 判断 policy 是否朝正确方向走 |
| Success | bool / 0-1 | 终点是否进入成功半径 | 得到 PointNav 成功率 |
| SPL | scalar float | 比较两种 policy 谁更短路径成功 | 避免只看成功、不看绕路 |
| Collisions | count / bool | 检查 agent 是否频繁撞墙 | 判断局部避障质量 |
| NumSteps | scalar int | 是否用了太多转向或重复动作 | 分析动作效率 |

## ObjectNav 常见指标用例

| 指标 | 数据形态 | 用例 | 效果 |
| --- | --- | --- | --- |
| Success | bool / 0-1 | 是否找到目标类别并 STOP | 得到 ObjectNav 成功率 |
| SPL | scalar float | 成功路径是否接近最短路径 | 衡量目标搜索效率 |
| DistanceToGoal | scalar float | 失败时离最近目标实例多远 | 判断是没找对区域还是停远了 |
| SoftSPL | scalar float | 失败但接近目标也有部分分 | 更细地比较失败质量 |
| TopDownMap | 2D map / image | 看搜索轨迹是否覆盖了正确房间 | 调试探索路线 |
| Collisions | count / bool | 看找目标时是否撞墙 | 分析安全性 |

## Rearrange / 交互 Measurement

| Measurement 类型 | 数据形态 | 非代码用例 | 效果 |
| --- | --- | --- | --- |
| object state success | bool | 物体是否被放到目标位置 | 判断操作任务是否完成 |
| distance to object | scalar float | gripper 离目标物体多远 | 分析抓取前是否接近 |
| distance to goal receptacle | scalar float | 物体离目标容器多远 | 分析放置质量 |
| force / collision metrics | scalar / vector | 操作中是否产生过大碰撞 | 衡量交互安全性 |
| pick success | bool | 是否成功抓住物体 | 分析抓取阶段 |
| place success | bool | 是否成功放置物体 | 分析放置阶段 |
| constraint violation | bool / count | 机械臂是否违反限制 | 调试控制策略 |

这些指标取决于具体 Rearrange task 和 measurement 配置。

## 可视化类 Measurement

| Measurement | 数据形态 | 非代码用例 | 效果 |
| --- | --- | --- | --- |
| TopDownMap | `(H, W)` 或 RGB 图像 / dict | 生成 episode 俯视轨迹图 | 可检查路线、转圈、绕路 |
| fog-of-war map | 2D 矩阵 | 显示 agent 已观察区域 | 分析探索覆盖 |
| collision overlay | 图像 / structured info | 在视频中标出碰撞时刻 | 快速定位失败动作 |
| goal marker | 坐标 / map overlay | 在俯视图标出目标 | 对比实际轨迹和目标位置 |

注意：可视化 Measurement 常用于评估和视频，不一定应该作为 agent 输入。如果作为输入，就变成 Sensor 问题。

## 自定义 Measure 扩展口

Habitat 允许用户自定义 Measure。这里是 Habitat 的扩展边界，不代表所有例子都是默认内置。

| 自定义 Measure 例子 | 数据形态 | 非代码用例 | 效果 |
| --- | --- | --- | --- |
| false stop count | scalar int | STOP 但不满足成功条件的次数 | 区分“没停”和“停错” |
| stuck steps | scalar int | 多步位置变化很小 | 诊断局部导航卡住 |
| repeated area ratio | scalar float | agent 重复经过同一区域比例 | 分析探索是否绕圈 |
| time to success | scalar int / float | 成功前用了多少步或秒 | 比 NumSteps 更直接用于成功样本 |
| visible target at stop | bool | STOP 时目标是否在视野中 | 分析目标确认质量 |
| per-room success | dict | 不同房间类型的成功率 | 做细粒度错误分析 |

边界：自定义 Measure 可以使用 simulator 或 task 的真值来评估，但如果这个真值进入 policy，就不是 Measurement，而是 Sensor。

## Measurement 数据形态汇总

| 类型 | 常见形态 | 例子 |
| --- | --- | --- |
| 成功标志 | bool / 0-1 scalar | Success |
| 连续分数 | scalar float | SPL、SoftSPL |
| 距离 | scalar float | DistanceToGoal |
| 计数 | scalar int | NumSteps、collision count |
| 轨迹 / 路径 | list of positions | PathLength 相关分析 |
| 地图 | `(H, W)`、`(H, W, 3)` 或 dict | TopDownMap |
| 多项统计 | dict | per-category success、collision details |

## 非代码用例库

| 你想知道什么 | Measurement 选择 | 效果 |
| --- | --- | --- |
| 是否完成任务 | Success | 得到成功率 |
| 是否走得高效 | SPL + PathLength | 区分短路径成功和绕路成功 |
| 失败离目标有多远 | DistanceToGoal / SoftSPL | 区分接近失败和完全失败 |
| 是否安全 | Collisions | 看碰撞次数和碰撞率 |
| 是否探索过目标附近 | TopDownMap + DistanceToGoal | 视频上看路径是否接近目标 |
| 是否经常超时 | NumSteps / max step termination | 看是否卡在探索或局部导航 |
| Rearrange 哪个阶段失败 | pick success / place success / distance to object | 拆解操作任务失败原因 |
| 自定义任务需要新诊断 | 自定义 Measure | 记录标准指标看不到的信息 |

## Measurement 与 Sensor 的边界

| 问题 | 更像 Sensor | 更像 Measurement |
| --- | --- | --- |
| policy 是否能读到它 | 是 | 否 |
| 是否影响动作选择 | 是 | 否 |
| 是否只在 info / log / video 里出现 | 否 | 是 |
| 是否可以使用评估侧真值 | 谨慎 | 可以，但要说明 |

例如 DistanceToGoal 如果只用于评估，是 Measurement；如果作为 observation 给 policy，就是 Sensor 或奖励 shaping 的一部分。

## 什么时候改 Measurement

优先改 Measurement，当你要改变：

- 成功率、SPL、距离、碰撞等指标。
- episode info 中输出什么。
- 视频和 top-down map 可视化。
- 自定义失败诊断。
- Rearrange 的操作成功统计。

如果这个量要进入 observation，改 Sensor。

如果这个量定义成功条件，改 Task。

