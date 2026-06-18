# Config / Hydra

Config / Hydra 是 Habitat 里把各个能力边界组合成一次实验的层。它不实现 simulator、task、sensor、policy，但决定启用哪些组件、参数是多少、组件之间如何接起来。

一句话：Config 是“实验接线图”。

## 配置数据形态

| 配置内容 | 数据形态 | 非代码用例 | 效果 |
| --- | --- | --- | --- |
| defaults list | list of config group entries | 选择 PointNav task、HM3D dataset、某个 policy | 组合出完整实验 |
| scalar 参数 | int / float / bool / string | 设置 max episode steps、turn angle | 改变单个行为参数 |
| vector 参数 | list / tuple | 设置 sensor position `[x, y, z]` | 改变相机或 agent 几何 |
| nested config | dict / structured config | simulator 下包含 agents、sensors、physics | 表达复杂组件 |
| registry name | string | policy name、trainer name、sensor type | 通过注册表找到实现 |
| override | key-value 修改 | 临时把 split 从 train 改 val | 不改基础配置也能切实验 |
| output config snapshot | structured config 文件 | 运行后保存完整配置 | 复现实验 |

## 主要配置边界

| 配置区域 | 控制什么 | 非代码用例 | 效果 |
| --- | --- | --- | --- |
| habitat.simulator | 场景、agent、sim sensors、物理参数 | 改 RGB 分辨率或 agent 半径 | 改原始仿真环境 |
| habitat.task | 任务类型、奖励、成功条件、task sensors、measurements | PointNav 改 ObjectNav | 改任务规则 |
| habitat.dataset | dataset type、data path、split、scene path | 从 train split 切到 val split | 改 episode 来源 |
| habitat.environment | max episode steps、iterator options | 限制每集最多 500 步 | 改 episode 运行方式 |
| habitat_baselines | trainer、eval、checkpoint、日志、视频、并行环境 | 评估 checkpoint 并保存视频 | 改训练 / 评估框架 |
| habitat_baselines.rl.policy | policy 名称和参数 | 使用 PointNavResNetPolicy | 改决策模型 |
| obs_transforms | 输入预处理组件 | 启用 resize | 改 policy 输入 shape |

## Simulator 配置选项

| 选项 | 数据形态 | 非代码用例 | 效果 |
| --- | --- | --- | --- |
| scene | string | 指定当前加载哪个场景 | 改 agent 所处环境 |
| agent height / radius | scalar float | 模拟不同大小机器人 | 改可通行性和碰撞 |
| sim_sensors | nested dict | 开启 rgb_sensor、depth_sensor | observation 里出现对应 sim sensor |
| sensor width / height | scalar int | RGB 从 256x256 改 640x480 | 改图像矩阵 shape |
| sensor position | `(3,)` list | 相机装在更高位置 | 改视角和深度投影 |
| sensor hfov | scalar | 改相机视野 | 改可见范围 |
| min / max depth | scalar | 限制 depth sensor 范围 | 改 depth 有效值 |
| allow_sliding | bool | 是否允许撞墙滑动 | 改碰撞后的运动效果 |
| forward_step_size / turn_angle | scalar | 改离散动作步长和转角 | 改动作物理尺度 |

## Task / Sensor / Measurement 配置选项

| 选项 | 数据形态 | 非代码用例 | 效果 |
| --- | --- | --- | --- |
| task type | registry name string | 选择 PointNav 或 ObjectNav | 改任务规则 |
| lab_sensors list | list of sensor configs | 加 GPS、Compass、ObjectGoal | observation 多对应 key |
| measurements list | list of measure configs | 加 Success、SPL、TopDownMap | info 多对应指标 |
| success distance | scalar float | 成功半径从 0.2 改 0.1 | 成功更严格 |
| reward values | scalar float | success reward、slack reward | 改训练信号 |
| max episode steps | scalar int | 每集最多 500 步 | 控制任务时长 |
| goal sensor config | nested config | PointGoal 维度、ObjectGoal 类别 | 改目标输入形式 |

## Dataset 配置选项

| 选项 | 数据形态 | 非代码用例 | 效果 |
| --- | --- | --- | --- |
| dataset type | registry name string | 选择 PointNav dataset 或 ObjectNav dataset | 决定 episode 解析方式 |
| data_path | string | 指向 train / val episode 文件 | 改题目来源 |
| split | string | train、val、test | 改评估或训练 split |
| scenes_dir | string | 指向场景资产目录 | scene_id 能被加载 |
| content scenes | list of scene ids | 只跑某些场景 | 做场景切片或快速调试 |
| episode count limit | scalar int | 只评估前 100 集 | 降低调试成本 |

## Baselines 配置选项

| 选项 | 数据形态 | 非代码用例 | 效果 |
| --- | --- | --- | --- |
| trainer_name | string | 选择 PPO 或评估 trainer | 决定训练 / 评估主循环 |
| evaluate | bool | true 时只评估 checkpoint | 不进行训练更新 |
| eval_ckpt_path | string | 加载已有模型 | 复现实验结果 |
| num_environments | scalar int | 同时跑 16 个环境 | 提升 rollout 吞吐 |
| torch_gpu_id | scalar int | 指定训练 GPU | 控制计算设备 |
| tensorboard_dir | string | 输出 scalar 日志 | 观察训练曲线 |
| video_dir | string | 保存评估视频 | 调试行为 |
| eval split | string | 在 val 上评估 | 控制评估数据 |
| PPO hyperparameters | scalars | learning rate、clip、gamma、num steps | 影响训练过程 |
| policy config | nested config | hidden size、backbone、rnn type | 改模型结构 |

## ObsTransform 配置选项

| 选项 | 数据形态 | 非代码用例 | 效果 |
| --- | --- | --- | --- |
| transform list | list | 启用 resize，再 normalize | 顺序处理 observation |
| resize size | `(H, W)` list / tuple | 图像统一到 224x224 | 输入 shape 固定 |
| target keys | list of strings | 只处理 rgb 和 depth | 控制哪些 observation 被改 |
| channels_last | bool | 指明输入是 HWC 还是 CHW | 防止维度解释错误 |
| interpolation mode | string / enum | semantic 用 nearest | 保持语义 id |

## 非代码实验组合用例

| 实验目的 | Config 选择 | 数据形态变化 | 效果 |
| --- | --- | --- | --- |
| 快速 PointNav 训练 | PointNav task + depth sensor + PointNav policy | depth `(H,W,1)` + pointgoal `(2,)` | 训练几何导航 |
| ObjectNav 评估 | ObjectNav task + RGB-D + ObjectGoal + Success/SPL | RGB、Depth、ObjectGoal scalar | 测语义目标导航 |
| RGB vs Depth 消融 | 分别启用 RGB-only、Depth-only、RGB-D | observation keys 改变 | 看输入模态贡献 |
| 分辨率消融 | sensor H/W 或 resize size 改变 | 图像矩阵大小改变 | 看速度和性能权衡 |
| 动作空间消融 | 4 动作 vs 加 LOOK_UP/DOWN | action id 集合变化 | 看主动视角是否有帮助 |
| 指标扩展 | 加 Collisions、TopDownMap | info 多 count / map | 更好解释失败 |
| dataset 切片 | 改 split 或 content scenes | episode list 改变 | 做快速调试或正式评估 |
| trainer 对比 | PPO vs DDPPO | rollout 组织不同 | 看训练吞吐和结果 |
| recurrent policy | policy config 开 recurrent | hidden state tensor 出现 | policy 能利用历史 |
| 视频调试 | eval video option + TopDownMap | 输出 frame / map | 直观看行为 |

## 运行后配置产物

| 产物 | 数据形态 | 用途 |
| --- | --- | --- |
| composed config | structured config | 查看最终启用了哪些组件 |
| hydra overrides | list of key-value | 知道本次临时改了什么 |
| logs | text / scalar series | 调试运行过程 |
| TensorBoard events | scalar time series | 看训练曲线 |
| videos | image frame sequence / mp4 | 看 agent 行为 |
| checkpoints | tensor weights + metadata | 恢复 policy |

## 常见配置错误

| 现象 | 可能边界 | 原因 |
| --- | --- | --- |
| observation 缺少 rgb | Config / Sensor / Simulator | sim sensor 没启用或 key 不匹配 |
| policy 输入 shape 不对 | Config / ObsTransform / Policy | resize、channels 或 observation key 不匹配 |
| action id 无效 | Config / Action / Policy | policy 输出和 action space 不一致 |
| measurement 没出现在 info | Config / Measurement | measure 没加入配置或 uuid 不匹配 |
| dataset 加载失败 | Config / Dataset | data_path、scene path 或 split 错 |
| checkpoint 加载失败 | Config / Baselines | policy 结构和 checkpoint 不匹配 |
| 评估结果不可复现 | Config | split、seed、checkpoint、sensor 参数不一致 |

## 边界

Config 可以：

- 选择组件。
- 设置参数。
- 组合实验。
- 做 override。
- 保存最终运行配置。

Config 不能：

- 凭空提供未注册的 Sensor / Measure / Policy。
- 修复 shape 不兼容的 policy。
- 替代 Task 定义成功条件。
- 替代 Simulator 生成不存在的场景资产。

## 什么时候改 Config / Hydra

优先改 Config / Hydra，当你要改变：

- 使用哪个 task、dataset、policy、trainer。
- 开启哪些 sensors、actions、measurements。
- 相机和动作参数。
- 训练 / 评估 split。
- checkpoint、日志、视频输出。
- obs transforms。
- ablation 实验组合。

