# Simulator

Simulator 指 Habitat-Sim 这一层。它负责加载 3D 场景、维护 agent 状态、渲染视觉传感器、处理碰撞、查询 navmesh，并提供最短路径和可通行性等几何能力。

一句话：Simulator 决定“agent 处在哪个物理世界里，以及原始观测如何被仿真出来”。

## 场景加载能力

| 选项 | 数据形态 | 非代码用例 | 效果 |
| --- | --- | --- | --- |
| 3D 场景网格 | 场景文件路径或 scene id | 在一个室内房屋中评测导航 | agent 能在具体空间里移动和观察 |
| 场景数据集配置 | structured config / 文件路径 | 一次注册多个场景资产目录 | Habitat 能找到 mesh、semantic asset、navmesh |
| 单场景 | 一个 scene id | 调试某个失败 episode | 复现稳定，便于看视频 |
| 多场景 | scene id 列表或 split 间接指定 | 正式 benchmark | 结果更能代表泛化能力 |
| 语义场景资产 | semantic mesh / semantic annotation | 需要 semantic sensor 或语义评估 | simulator 能渲染像素级语义 id |

边界：Simulator 只负责加载世界，不定义任务目标。某个场景里有没有椅子属于资产事实；这一集要求找不找椅子属于 Task / Dataset。

## Agent 身体能力

| 选项 | 数据形态 | 非代码用例 | 效果 |
| --- | --- | --- | --- |
| agent position | `(3,)` 浮点向量 | 把 agent 放在 episode 起点 | 决定初始位置 |
| agent rotation | quaternion 或 yaw 等姿态表示 | 让 agent 一开始朝向走廊或背对目标 | 决定初始视野 |
| agent height | scalar，单位通常是米 | 模拟矮机器人或人高相机 | 影响相机默认高度和碰撞体 |
| agent radius | scalar，单位通常是米 | 让机器人不能穿过窄缝 | 影响 navmesh 可通行性和碰撞 |
| sensor mount position | `(3,)` 相对 agent 的位置 | 把相机装在头部或低位 | 改变看到的视角和深度几何 |
| sensor orientation | 角度或四元数 | 前视、俯视、侧视相机 | 改变观测方向 |

效果：身体参数会直接影响“什么地方能走、会不会撞、相机从哪里看”。例如半径变大后，同一个门洞可能从可通行变成不可通行。

## Sim Sensor 渲染能力

| 选项 | 数据形态 | 非代码用例 | 效果 |
| --- | --- | --- | --- |
| RGB sensor | `(H, W, 3)`，通常是 `uint8` 图像 | ObjectNav agent 用图像识别沙发 | agent 获得外观和纹理信息 |
| Depth sensor | `(H, W, 1)` 或 `(H, W)`，通常是 `float32` | PointNav agent 根据深度避障 | agent 获得几何距离信息 |
| Semantic sensor | `(H, W)`，每个像素是语义 id | 调试语义导航或 oracle 实验 | agent 或评估侧能知道像素类别 |
| 多个同类 sensor | 多个图像矩阵，key 不同 | 前视 RGB 加侧视 RGB | agent 可以获得多视角观测 |
| 不同分辨率 | H、W 由 config 指定 | 低分辨率快速训练，高分辨率调试小物体 | 影响速度、显存、识别细节 |
| 不同 FOV | scalar 角度 | 宽视野减少盲区，窄视野更接近长焦 | 改变投影几何和可见范围 |
| depth 范围 | min / max depth scalar | 只关心近距离避障或远距离空间 | 影响 depth 有效值和建图距离 |

边界：RGB、Depth、Semantic 是 Habitat-Sim 常见原始渲染信号。ObjectGoal、GPS、Compass 不是 sim sensor 渲染出来的图像，它们属于 Habitat-Lab 的 task / lab sensor。

## 运动和碰撞能力

| 选项 | 数据形态 | 非代码用例 | 效果 |
| --- | --- | --- | --- |
| forward step size | scalar，单位通常是米 | 每次 MOVE_FORWARD 走 0.25 米或 0.1 米 | 影响路径粒度和碰撞风险 |
| turn angle | scalar，单位通常是度 | 每次 TURN_LEFT 转 30 度或 10 度 | 影响朝向控制精度和步数 |
| tilt angle | scalar，单位通常是度 | LOOK_UP / LOOK_DOWN 改变相机俯仰 | agent 能主动看高处或低处 |
| allow sliding | bool | 撞墙后是否沿墙滑动 | 开启会降低卡墙难度，但可能不真实 |
| collision check | bool / simulator state | agent 前进时碰到墙或家具 | 动作可能被阻挡，trajectory 改变 |
| physics simulation | structured simulator state | 交互任务中物体被推、抓、放 | Rearrange 类任务能模拟物理变化 |

边界：Simulator 执行动作的物理效果；Action 定义有哪些动作可选；Policy 决定什么时候选哪个动作。

## Navmesh 和路径查询能力

| 选项 | 数据形态 | 非代码用例 | 效果 |
| --- | --- | --- | --- |
| navmesh | 可通行表面数据 | 判断客厅到卧室是否连通 | 提供可通行区域 |
| navigable point | `(3,)` 位置 | 随机采样 episode 起点 | 起点落在可走区域 |
| is navigable | bool | 检查某个目标点能不能站 | 避免不可达位置 |
| geodesic distance | scalar 距离 | 评估起点到目标的最短可走距离 | 衡量 episode 难度和 SPL |
| shortest path | 位置序列或路径对象 | 生成 oracle 路径、验证任务可达 | 提供理想路径参考 |
| island / connectivity | bool 或连通区域 id | 检查起点和目标是否在同一可达区域 | 避免孤岛 episode |
| top-down map support | 2D 矩阵 / 图像 | 可视化 agent 轨迹 | 帮助调试和展示 |

边界：navmesh 查询可以用于生成数据集、评估、oracle 或调试。普通 agent 是否能看到这些真值信息，取决于 Sensor 和 Task 设计。

## 非代码用例库

| 目标 | Simulator 侧选择 | 效果 |
| --- | --- | --- |
| 训练低成本 PointNav | 低分辨率 Depth，较少 sim sensors | 训练吞吐更高，但视觉细节较少 |
| 做 ObjectNav 视觉评测 | RGB + Depth 前视相机 | agent 同时有外观和距离信息 |
| 做 oracle 语义实验 | 打开 Semantic sensor | 可以验证“如果语义感知完美，导航还有多难” |
| 模拟矮机器人 | 降低 agent height 和相机 mount height | 视野更低，家具遮挡更明显 |
| 模拟宽机器人 | 增大 agent radius | 可通行空间变少，窄门更难通过 |
| 分析相机 FOV 影响 | 比较 60、90、120 度 FOV | FOV 越大盲区越少，但图像畸变和几何处理更敏感 |
| 减少碰撞乐观性 | 关闭 allow sliding | agent 撞墙后不会自动沿墙滑开 |
| 做可达性筛选 | 用 geodesic distance 和 connectivity | 数据集里排除不可达目标 |

## 什么时候改 Simulator

优先改 Simulator，当你要改变：

- 场景资产和语义资产。
- agent 的尺寸、身体和碰撞行为。
- sim sensor 的类型、分辨率、FOV、位置、深度范围。
- 离散动作的物理步长和转角。
- navmesh、可通行性、最短路径查询。

如果只是想换目标类别，看 Dataset / Episode。

如果只是想新增 observation key，看 Sensor。

如果只是想改成功条件，看 Task / Measurement。

