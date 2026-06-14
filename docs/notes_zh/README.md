# VLFM 代码走读笔记（中文）

> 目的：把你"从论文里读到的直觉"对照本仓库的真实实现，逐条核对、补全、修正，并给出每一步中间数据的具体形态（矩阵大小 / 图像 / JSON 等），方便系统学习。
>
> 阅读顺序建议：
>
> - **想深入读代码**：00 → 01 → 02 → 03 → 04 → 05 → 06。
> - **只想做汇报/评审**：直接读 **07_VLFM_全景介绍.md**（多对比表、少代码，含问题定义 / 视频生成 / Habitat→ROS / 与 SLAM 等对比 / 应用展望 5 大块）。

| 文件 | 主题 | 适合 |
| --- | --- | --- |
| [00_总览与直觉核对.md](./00_总览与直觉核对.md) | 你给出的直觉 vs 代码事实；逐条勾正 | 入门 |
| [01_BLIP2_ValueMap_推理与融合.md](./01_BLIP2_ValueMap_推理与融合.md) | BLIP2-ITM 余弦 → 锥形投影 → 置信度通道 → v_new/c_new 融合公式 | 代码细节 |
| [02_YOLO_Grounding_SAM_对象定位.md](./02_YOLO_Grounding_SAM_对象定位.md) | 三个检测/分割模型如何协同得到目标点云与"最近点" | 代码细节 |
| [03_ObstacleMap_与_Frontier.md](./03_ObstacleMap_与_Frontier.md) | 障碍地图、已探索区、可航行区、frontier 的生成 | 代码细节 |
| [04_决策流程_act_explore_navigate.md](./04_决策流程_act_explore_navigate.md) | 一次 `act()` 里 BLIP2 / YOLO / Grounding 的真实先后与覆盖关系 | 代码细节 |
| [05_数据形态速查表.md](./05_数据形态速查表.md) | 每个张量 / 数组 / JSON 的形状、单位与典型数值 | 查表 |
| [06_仿真环境_Habitat与接口.md](./06_仿真环境_Habitat与接口.md) | Habitat 仿真本体、配置/传感器/动作接口、RGBD 是怎么从 sim 流到策略再到 VLM RPC 的 | 代码细节 |
| [07_VLFM_全景介绍.md](./07_VLFM_全景介绍.md) | 问题定义 / 视频生成 / Habitat→ROS 移植 / 与 SLAM 等导航对比 / 应用端展望 | **汇报评审** |
| [08_Reality真机数据格式端到端变化链.md](./08_Reality真机数据格式端到端变化链.md) | 真机(Reality)RGBD / 位姿 / 动作端到端数据格式变化链 | 代码细节 |
| [09_YOLO26_TensorRT替换方案.md](./09_YOLO26_TensorRT替换方案.md) | 用 YOLO26+TensorRT 替换 YOLOv7-e6e:思路 / 坑 / 验收 | 工程方案 |
| [10_YOLO26n升级到l方案.md](./10_YOLO26n升级到l方案.md) | yolo26n → yolo26l 升级:改动面 / n→l 专属坑 / 验收闸门 | 工程方案 |
| [11_SigLIP2_ITM_可回滚替换方案.md](./11_SigLIP2_ITM_可回滚替换方案.md) | SigLIP2-ITM 可回滚替换 BLIP2-ITM:规划 / 坑 / 解决方案 / 验收 | 工程方案 |

如果你只读一篇：

- 做技术汇报、对外讲故事 → **07_VLFM_全景介绍.md**。
- 想从直觉入手核对代码事实 → **00_总览与直觉核对.md**。
- 想搞清楚"环境到底怎么塞 RGBD 给代码"→ **06_仿真环境_Habitat与接口.md** 的 6.4 节。
