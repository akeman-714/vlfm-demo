# VLFM 代码走读笔记（中文）

> 目的：把你"从论文里读到的直觉"对照本仓库的真实实现，逐条核对、补全、修正，并给出每一步中间数据的具体形态（矩阵大小 / 图像 / JSON 等），方便系统学习。
>
> 阅读顺序建议：00 → 01 → 02 → 03 → 04 → 05。

| 文件 | 主题 |
| --- | --- |
| [00_总览与直觉核对.md](./00_总览与直觉核对.md) | 你给出的直觉 vs 代码事实；逐条勾正 |
| [01_BLIP2_ValueMap_推理与融合.md](./01_BLIP2_ValueMap_推理与融合.md) | BLIP2-ITM 余弦 → 锥形投影 → 置信度通道 → v_new/c_new 融合公式 |
| [02_YOLO_Grounding_SAM_对象定位.md](./02_YOLO_Grounding_SAM_对象定位.md) | 三个检测/分割模型如何协同得到目标点云与"最近点" |
| [03_ObstacleMap_与_Frontier.md](./03_ObstacleMap_与_Frontier.md) | 障碍地图、已探索区、可航行区、frontier 的生成 |
| [04_决策流程_act_explore_navigate.md](./04_决策流程_act_explore_navigate.md) | 一次 `act()` 里 BLIP2 / YOLO / Grounding 的真实先后与覆盖关系 |
| [05_数据形态速查表.md](./05_数据形态速查表.md) | 每个张量 / 数组 / JSON 的形状、单位与典型数值 |
| [06_仿真环境_Habitat与接口.md](./06_仿真环境_Habitat与接口.md) | Habitat 仿真本体、配置/传感器/动作接口、RGBD 是怎么从 sim 流到策略再到 VLM RPC 的 |

如果你只读一篇，先看 **00_总览与直觉核对.md**。
如果你想搞清楚"环境到底怎么塞 RGBD 给代码"，直接看 **06_仿真环境_Habitat与接口.md** 的 6.4 节。
