# Dataset / Episode

Dataset / Episode 是 Habitat-Lab 里定义“测试题从哪里来”的边界。Dataset 负责装载一批 episode；episode 负责描述一集的场景、起点、目标和元数据。

一句话：Task 是规则，Dataset / Episode 是题目样本。

## Episode 常见字段

| 字段 | 数据形态 | 非代码用例 | 效果 |
| --- | --- | --- | --- |
| episode_id | string 或 int | 在日志中定位第 37 集 | 方便复现和统计 |
| scene_id | string 路径或场景标识 | 指向某个室内房屋场景 | 决定 agent 所在环境 |
| start_position | `(3,)` 浮点向量 | agent 从客厅门口出生 | 决定起点位置 |
| start_rotation | quaternion 或角度表示 | agent 一开始面向走廊 | 决定初始朝向 |
| goals | list / structured object | PointNav 的目标点或 ObjectNav 的目标实例 | 定义这一集要到哪里或找什么 |
| object_category | string 或类别 id | ObjectNav 中目标是 chair | 定义目标类别 |
| info | dict | 存 geodesic distance、difficulty 等 | 用于分析、筛选和评估 |
| shortest_paths | 路径点序列，可选 | 保存 oracle 路径 | 可用于教师或分析 |
| start_room / goal_room | string，可选 | 标注从卧室到厨房 | 可做房间级难度切片 |

不同任务的 episode 字段会不同。PointNav 更关注目标坐标；ObjectNav 更关注目标类别和目标实例；VLN 更关注 instruction；Rearrange 更关注物体状态。

## Dataset 级选项

| 选项 | 数据形态 | 非代码用例 | 效果 |
| --- | --- | --- | --- |
| split | string，例如 train / val / test | 训练用 train，报告用 val/test | 分离训练和评估 |
| data_path | string 路径 | 指向某个 episode 文件 | Habitat 知道从哪里加载题目 |
| scenes_dir | string 路径 | 指向 3D 场景资产目录 | episode 的 scene_id 能被解析 |
| episode list | list of episode objects | 一个 split 包含几千集 | 决定评测样本数量 |
| category mapping | dict，类别名到 id | ObjectNav 中 chair 对应类别 0 | ObjectGoal sensor 能输出稳定 id |
| content files | 按 scene 分片的 episode 文件 | 大数据集按场景懒加载 | 降低一次性加载成本 |
| dataset type | string / registry name | PointNav dataset、ObjectNav dataset | 决定如何解析 episode |

## 目标分布选项

| 目标分布 | 数据形态 | 非代码用例 | 效果 |
| --- | --- | --- | --- |
| 点目标 | `(3,)` 或 `(2,)` 位置 | PointNav 随机采样可通行点 | 测几何导航 |
| 物体类别 | string / int | ObjectNav 抽 chair、bed、sofa | 测语义搜索 |
| 目标实例列表 | list of object goals | 一个类别在场景里有多个实例 | 成功可对应任意有效实例 |
| 目标视点 | list of viewpoint positions | 站在哪些位置算看见目标 | 用于 ObjectNav 成功判定和距离 |
| 图像目标 | 图像路径或图像矩阵引用 | ImageNav 指定目标照片 | 测视觉匹配 |
| 指令目标 | string / token ids | VLN episode 带一段导航指令 | 测语言路线跟随 |
| 物体状态目标 | structured state | Rearrange 目标状态是物体在桌上 | 测交互操作 |

## 起点分布选项

| 起点设置 | 数据形态 | 非代码用例 | 效果 |
| --- | --- | --- | --- |
| 随机可通行起点 | `(3,)` 位置 + rotation | 每集从不同房间开始 | 提高泛化要求 |
| 固定起点 | 固定 start_position / rotation | 调试同一个失败 episode | 复现稳定 |
| 近距离起点 | info 中 geodesic distance 较小 | 目标就在同房间 | 主要测识别和 STOP |
| 长距离起点 | geodesic distance 较大 | 起点和目标隔多个房间 | 主要测探索和路径效率 |
| 初始可见目标 | episode 标注或筛选 | 开局就能看到目标 | 测识别，不测搜索 |
| 初始不可见目标 | episode 标注或筛选 | 需要转弯或跨房间 | 测搜索能力 |
| 跨楼层起点 | start 和 goal 高度 / 楼层不同 | 需要楼梯或连通路径 | 测复杂导航 |

## 难度和切片

| 切片方式 | 数据形态 | 非代码用例 | 效果 |
| --- | --- | --- | --- |
| 按 geodesic distance | scalar | 只看 5 米以上长程 episode | 分析长程导航能力 |
| 按 euclidean distance | scalar | 比较直线近但绕路远的样本 | 识别布局复杂性 |
| 按房间数量 | int / metadata | 起点和目标跨几个房间 | 分析探索难度 |
| 按目标类别 | category id / string | 只评估 chair 或 toilet | 分析类别差异 |
| 按场景 | scene_id | 某些场景失败多 | 定位场景资产或布局问题 |
| 按目标可见性 | bool / metadata | 初始可见 vs 不可见 | 区分识别和探索 |
| 按最短路径复杂度 | path length / turn count | 需要多次转弯的 episode | 分析路径规划困难 |

这些切片不一定都是 Habitat 标准字段，但 Dataset / Episode 的边界允许你在 info 或 metadata 里保存它们，用于分析和筛选。

## Dataset 与 Task 的配合例子

| 需求 | Task 侧 | Dataset / Episode 侧 | 效果 |
| --- | --- | --- | --- |
| 标准 PointNav | 规则是到达点目标 | episode 给 start 和 point goal | agent 学走到坐标 |
| 标准 ObjectNav | 规则是找类别并 STOP | episode 给 scene、start、object_category、goal instances | agent 学找目标类别 |
| 只测远距离 ObjectNav | ObjectNav 规则不变 | 只抽 geodesic distance 大的 episode | 暴露长程搜索能力 |
| 测初始可见识别 | ObjectNav 规则不变 | 只抽目标初始可见 episode | 主要看视觉识别和停止 |
| 测 ImageNav | ImageNav 规则 | episode 给目标图像和起点 | agent 找到图像对应位置 |
| 测 VLN | Instruction navigation 规则 | episode 给 instruction 和路线相关目标 | agent 按语言指令移动 |
| 测 Rearrange | Rearrange 规则 | episode 给物体初始状态和目标状态 | agent 需要操作物体 |

## 数据形态例子汇总

| 能力点 | 常见形态 |
| --- | --- |
| episode | structured object / dict |
| dataset | episode list + dataset metadata |
| scene_id | string |
| start_position | `(3,)` float |
| start_rotation | quaternion，常见是 `(4,)` float |
| object_category | string 或 int |
| goals | list of structured goals |
| geodesic_distance | scalar float |
| shortest path | list of `(3,)` positions |
| instruction | string 或 token sequence |
| image goal | image reference 或 `(H, W, 3)` |

## 边界

Dataset / Episode 可以控制：

- 在哪些场景中评测。
- agent 从哪里出生。
- 目标是什么。
- split 是 train、val 还是 test。
- episode 的难度和元数据。

Dataset / Episode 不应该负责：

- 渲染 RGB-D，那是 Simulator。
- 定义成功条件，那是 Task / Measurement。
- 让 policy 怎么决策，那是 Policy。
- 改 observation 内容，那是 Sensor。

## 什么时候改 Dataset / Episode

优先改 Dataset / Episode，当你要改变：

- 场景集合。
- 起点分布。
- 目标类别或目标实例分布。
- train / val / test split。
- 难度切片。
- episode 元数据。

