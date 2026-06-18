# Sensor

Sensor 是 Habitat 中把信息放进 observation 的能力边界。它决定 agent 每一步能读到什么。

一句话：Sensor 是“输入接口”，不是 policy，也不是评估指标。

## 来源分类

| 来源 | 含义 | 常见例子 |
| --- | --- | --- |
| Habitat-Sim sim sensor | 仿真器根据 3D 场景和相机参数渲染出来 | RGB、Depth、Semantic |
| Habitat-Lab task / lab sensor | 任务层根据 episode、agent state 或 task state 生成 | GPS、Compass、Heading、ObjectGoal、PointGoal、ImageGoal、Instruction、Proximity |
| Rearrange / embodied task sensor | 交互任务里描述机器人和物体状态 | joint state、end-effector state、is holding、object goal state |
| 自定义 Sensor 扩展口 | 用户按 Habitat Sensor 接口新增 observation | 自定义风险分数、任务阶段 id、额外几何量 |

下面的表以 Habitat 常见能力为主。不同 Habitat 版本、不同 task package 会略有差异，具体以当前安装版本的 registry 和 config 为准。

## Habitat-Sim 视觉 Sensor

| Sensor | 数据形态 | 非代码用例 | 效果 |
| --- | --- | --- | --- |
| RGB | `(H, W, 3)`，通常是 `uint8` | ObjectNav 中 agent 看图找沙发 | 获得外观、颜色、纹理、物体形状 |
| Depth | `(H, W, 1)` 或 `(H, W)`，通常是 `float32` | PointNav 中根据前方深度避障 | 获得每个像素到相机的距离 |
| Semantic | `(H, W)`，每个像素是语义类别 id | 语义导航 oracle 或调试语义标注 | 直接知道像素类别，普通公平评测中要谨慎 |
| 多 RGB 相机 | 多个 `(H, W, 3)` key | 前视加侧视相机 | 增加视野，但 observation 更大 |
| 多 Depth 相机 | 多个 depth 矩阵 | 前向深度加下视深度 | 获得更多几何覆盖 |
| 不同分辨率视觉 sensor | H、W 由 config 决定 | 训练用低分辨率，评估可用高分辨率 | 分辨率越高细节越多，计算越重 |
| 不同相机位置 / 朝向 | position `(3,)`，orientation | 头部相机、低位相机、俯视相机 | 改变可见区域和投影几何 |

边界：RGB / Depth / Semantic 是原始仿真信号。它们不会自动告诉 policy 哪个动作最好，只是提供像素矩阵。

## 导航位姿 Sensor

| Sensor | 数据形态 | 非代码用例 | 效果 |
| --- | --- | --- | --- |
| GPS | 常见为 `(2,)` 或 `(3,)` 浮点位置 | agent 知道自己相对 episode 起点的位置 | 可计算当前位置、轨迹和相对目标 |
| Compass | scalar 或 `(1,)` 角度 | agent 知道当前朝向 | 可把目标方向转成左转 / 右转依据 |
| Heading | scalar 或 `(1,)` 角度 | 记录或输入 agent heading | 与 Compass 类似，但具体定义随任务实现 |
| PointGoal | `(2,)` 或 `(3,)`，常见是距离+角度或相对坐标 | PointNav 中告诉 agent 目标在哪里 | agent 不需要语义搜索，只需走到点 |
| PointGoalWithGPSCompass | `(2,)`，常见是 rho/theta | PointNav policy 用 GPS+Compass 得到目标相对极坐标 | 输入紧凑，适合经典 PointNav |
| Proximity | scalar 距离 | agent 知道最近障碍有多近 | 可用于减少贴墙或碰撞 |

注意：GPS / Compass 在仿真中通常很准。真实系统如果要对齐，需要外部定位或里程计，但那已经不属于 Habitat 原生 sensor 范围。

## 任务目标 Sensor

| Sensor | 数据形态 | 非代码用例 | 效果 |
| --- | --- | --- | --- |
| ObjectGoal | scalar int，表示类别 id | ObjectNav 中目标类别是 chair | agent 知道要找哪类物体 |
| ImageGoal | `(H, W, 3)` 图像、图像路径或特征 | ImageNav 中给一张目标照片 | agent 需要寻找与目标图像相似的位置 |
| Instruction | string 或 token ids | VLN 中给“沿走廊到厨房” | agent 能接收自然语言路线 |
| Question | string 或 token ids | EQA 中问“房间里有什么” | agent 要导航并回答问题 |
| Goal object state | vector / structured state | Rearrange 中目标是杯子在桌上 | agent 知道交互任务的目标状态 |

边界：ObjectGoal 只告诉 agent “找什么类别”，不会告诉目标在哪里。目标实例位置属于 episode / oracle 信息，普通 policy 是否能看到取决于 sensor 设计。

## Rearrange / 交互任务 Sensor

| Sensor 类型 | 数据形态 | 非代码用例 | 效果 |
| --- | --- | --- | --- |
| joint state | `(N,)` 浮点向量 | 机械臂任务中知道各关节角 | policy 能控制和理解手臂姿态 |
| end-effector state | `(3,)` 位置或更完整姿态 | 知道夹爪在哪里 | 方便抓取和放置 |
| is holding | bool 或 scalar | 判断是否已经抓住物体 | 支持 pick/place 阶段切换 |
| object position sensor | `(3,)` 或相对位置向量 | 知道目标物体相对机器人位置 | 用于交互任务训练或 oracle 设置 |
| target receptacle state | structured state | 放置任务中知道目标容器状态 | 判断该把物体放到哪里 |
| robot base state | position / rotation vector | 移动操作中知道底盘状态 | 导航和操作结合 |

这些 sensor 是否存在取决于具体 Rearrange task 和配置。它们属于 Habitat 交互任务的输入边界，不是所有导航任务默认都有。

## 自定义 Sensor 扩展口

Habitat 允许用户新增 Sensor。这里的重点是“扩展口存在”，不是说下面例子是 Habitat 默认内置。

| 自定义 Sensor 例子 | 数据形态 | 非代码用例 | 效果 |
| --- | --- | --- | --- |
| 自定义阶段 id | scalar int | 多阶段任务中告诉 agent 当前是导航还是交互 | policy 输入中显式包含阶段信息 |
| 自定义风险分数 | scalar float | 某些区域靠近危险物时风险更高 | agent 可学习避开高风险区域 |
| 自定义局部地图 | `(H, W)` 矩阵 | 把局部可通行区域作为 observation | policy 不必完全从 raw depth 学几何 |
| 自定义目标属性 | vector 或 token | 目标是红色、金属、可抓取 | agent 能处理更细目标描述 |
| 自定义历史摘要 | vector | 输入最近若干步的摘要 | policy 获得短期记忆提示 |

边界：只要这个量会进入 agent 的 observation 并影响决策，它就是 Sensor 候选。如果只是评估后记录，不应放 Sensor，应放 Measurement。

## Sensor 数据形态汇总

| Sensor | 常见 shape / 类型 |
| --- | --- |
| RGB | `(H, W, 3)` uint8 |
| Depth | `(H, W, 1)` 或 `(H, W)` float32 |
| Semantic | `(H, W)` int |
| GPS | `(2,)` 或 `(3,)` float |
| Compass / Heading | scalar 或 `(1,)` float |
| PointGoal | `(2,)` 或 `(3,)` float |
| ObjectGoal | scalar int |
| ImageGoal | `(H, W, 3)` 或图像特征 |
| Instruction / Question | string 或 token id 序列 |
| Proximity | scalar float |
| Joint state | `(N,)` float |
| Is holding | bool / scalar |
| 自定义矩阵 sensor | `(H, W)`、`(H, W, C)` 或其他约定 shape |

## 非代码用例库

| 需求 | Sensor 选择 | 效果 |
| --- | --- | --- |
| 只训练几何 PointNav | Depth + PointGoal | agent 主要依靠距离图和目标向量导航 |
| 训练带视觉的 ObjectNav | RGB + Depth + ObjectGoal | agent 同时知道目标类别、外观和几何 |
| 做语义 oracle 对比 | Semantic + ObjectGoal | 可以测试“语义感知完美时”任务还剩多少难度 |
| 做 ImageNav | RGB + Depth + ImageGoal | agent 对比当前视图和目标图像 |
| 做 VLN | RGB + Depth + Instruction | agent 按语言指令移动 |
| 做简单避障增强 | Depth + Proximity | agent 除了图像深度，还知道最近障碍距离 |
| 做机械臂抓取 | RGB-D + joint state + end-effector state + is holding | agent 能观察场景、手臂和抓取状态 |
| 做多相机实验 | 多个 RGB / Depth sim sensors | agent 视野更大，输入更复杂 |

## 什么时候改 Sensor

优先改 Sensor，当你要改变：

- observation 里是否有 RGB、Depth、Semantic。
- observation 里是否有 GPS、Compass、PointGoal、ObjectGoal。
- ImageNav / VLN / Rearrange 等任务需要的输入。
- 视觉 sensor 的数量、key、shape。
- 自定义 observation。

如果要改 sensor 的物理相机位置和渲染参数，也要同时看 Simulator。

如果要改输入进 policy 前的 resize / crop / normalize，看 ObsTransform。

如果一个量只用于统计结果，不进 observation，看 Measurement。

